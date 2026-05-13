//! POST + `invoke`: один колбэк после **тишины** ([`INVOKE_DEBOUNCE_MS`]).
//! Финал, если набран целевой объём (**shares / USDC** из [`crate::account_order::PostOrderRequest::amount`]) как **max(HTTP-снимок, накопление WS `trade`)** — снимок с **POST** и при каждом успешном poll [`OpenOrderResponse`] (`size_matched`, для USDC-целей ещё `size_matched × price`) **или** наступил дедлайн
//! (**`expiration` → иначе `market_end_unix_ms` → короткий fallback**) **или** CLOB дал терминал / отмену.
//! Нулевое исполнение к дедлайну → `success=false`. Отмены — только HTTP. Агрегаты cancel-all / sell-all.
//!
//! Хаб живёт на [`crate::account::Account::order_invoke_hub`].

use crate::account::SharedAccount;
use crate::account_order::{OrderAmount, PostOrderRequest};
use polymarket_client_sdk::clob::types::response::OpenOrderResponse;
use polymarket_client_sdk::clob::types::{OrderStatusType, Side};
use polymarket_client_sdk::types::Decimal;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;

/// Пауза без новых событий (мс): после неё считается, что можно финализировать и вызвать invoke один раз.
const INVOKE_DEBOUNCE_MS: u64 = 450;
/// Запас по времени (сек), если ни `expiration`, ни `market_end_unix_ms` не заданы — верхняя граница опроса/финала.
pub(crate) const INVOKE_FALLBACK_POLL_DEADLINE_SEC: u64 = 30;
/// То же окно fallback в миллисекундах как `i64` для [`crate::util::current_timestamp_ms`].
pub(crate) const INVOKE_FALLBACK_DEADLINE_MS_I64: i64 =
    (INVOKE_FALLBACK_POLL_DEADLINE_SEC as i64).saturating_mul(1000);
const INVOKE_FALLBACK_POLL_MS: u64 = 500;
const ORDER_HTTP_POLL_TIMEOUT_SEC: u64 = 10;
/// Порог «достаточного» накопления outcome-shares при сравнении с целью из заявки.
const SHARE_EPS: f64 = 1e-7;
/// Порог «достаточной» набранной суммы в USDC при сравнении с целевым notional.
const USD_EPS: f64 = 1e-5;

fn zero_fill_matching_target_dimension(target_amount: OrderAmount) -> OrderAmount {
    match target_amount {
        OrderAmount::Shares(_) => OrderAmount::Shares(0.0),
        OrderAmount::UsdNotional(_) => OrderAmount::UsdNotional(0.0),
    }
}

/// Когда после POST ещё нет [`PostOrderRequest`] (SDK error / timeout), размерность «заполнения» неизвестна — отдаём 0 shares как конвенцию.
fn zero_fill_without_request_context() -> OrderAmount {
    OrderAmount::Shares(0.0)
}

#[derive(Debug, Clone)]
pub struct SingleOrderClobInvocationReport {
    pub order_id: String,
    pub filled_amount: OrderAmount,
    pub success: bool,
    pub partial: bool,
}

pub type SingleOrderInvokeCb =
    Box<dyn FnOnce(SingleOrderClobInvocationReport) + Send + 'static>;

/// Контекст после успешного HTTP POST для агрегатора invoke.
#[derive(Debug, Clone)]
pub struct PostOrderInvokeContext {
    pub request: PostOrderRequest,
    pub seed_making: f64,
    pub seed_taking: f64,
}

pub struct CompletionOnce<T: Send + 'static> {
    fired: AtomicBool,
    locked_callback: Mutex<Option<Box<dyn FnOnce(T) + Send + 'static>>>,
}

impl<T: Send + 'static> CompletionOnce<T> {
    pub fn new(initial_callback: Box<dyn FnOnce(T) + Send + 'static>) -> Self {
        Self {
            fired: AtomicBool::new(false),
            locked_callback: Mutex::new(Some(initial_callback)),
        }
    }

    pub fn fire(&self, value: T) {
        if self.fired.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut guard = match self.locked_callback.lock() {
            Ok(guard) => guard,
            Err(poison) => poison.into_inner(),
        };
        if let Some(callback_once) = guard.take() {
            callback_once(value);
        }
    }
}

#[derive(Debug)]
struct InvokeAggInner {
    /// Целевой объём из [`PostOrderRequest::amount`].
    target: OrderAmount,
    /// Накопленное исполнение только из user-WS `trade` (+=).
    filled_ws: OrderAmount,
    /// Последний HTTP-снимок исполнения: POST seed или GET `order()` при poll (**перезапись**).
    filled_http: OrderAmount,
    /// Unix время (ms): после этого момента finalize допускается даже без полного набора объёма.
    deadline_ms: i64,
    /// [`Side`] ордера (копия из [`crate::account_order::PostOrderRequest::side`]) — трактовка HTTP `making`/`taking` при seed.
    side: Side,
    /// От CLOB поступил успешный «исполнен»-терминал (WS MATCHED/FILLED, POST/poll/SDK `Matched`). Это входной сигнал, не то же что финальный [`SingleOrderClobInvocationReport::success`] — итог задаёт [`PostOrderInvokeAggregator::build_report`].
    success: bool,
    /// От CLOB поступила терминальная отмена (WS `CANCELED`, POST/poll `Canceled`). Входной сигнал, не финальный [`SingleOrderClobInvocationReport::partial`].
    partial: bool,
}

fn decimal_snap_f64(d: &Decimal) -> Option<f64> {
    let f = d.to_string().parse::<f64>().ok()?;
    f.is_finite().then_some(f)
}

/// Трактует `making`/`taking` как ответ POST — см. [`PostOrderInvokeAggregator::ingest_http_seed`].
fn invoke_write_filled_http_from_seed_pair(inner: &mut InvokeAggInner, making: f64, taking: f64) {
    if !making.is_finite() || !taking.is_finite() {
        return;
    }
    match (&inner.target, inner.side) {
        (OrderAmount::Shares(_), Side::Buy) => {
            inner.filled_http = OrderAmount::Shares(taking);
        }
        (OrderAmount::Shares(_), Side::Sell) => {
            inner.filled_http = OrderAmount::Shares(making);
        }
        (OrderAmount::UsdNotional(_), Side::Buy) => {
            inner.filled_http = OrderAmount::UsdNotional(making);
        }
        (OrderAmount::UsdNotional(_), Side::Sell) => {
            inner.filled_http = OrderAmount::UsdNotional(taking);
        }
        _ => {}
    }
}

/// GET `OpenOrderResponse`: перезаписывает [`InvokeAggInner::filled_http`] из `size_matched` (+ `price` для USDC-целей).
fn invoke_apply_open_order_rest_snapshot(inner: &mut InvokeAggInner, open: &OpenOrderResponse) {
    let Some(size_matched) = decimal_snap_f64(&open.size_matched) else {
        return;
    };
    let Some(price) = decimal_snap_f64(&open.price) else {
        return;
    };
    if !(size_matched >= 0.0 && price >= 0.0 && price.is_finite()) {
        return;
    }
    let (making, taking) = match (&inner.target, inner.side) {
        (OrderAmount::Shares(_), Side::Buy) => (0.0, size_matched),
        (OrderAmount::Shares(_), Side::Sell) => (size_matched, 0.0),
        (OrderAmount::UsdNotional(_), Side::Buy) => {
            let quote = size_matched * price;
            if !(quote >= 0.0 && quote.is_finite()) {
                return;
            }
            (quote, 0.0)
        }
        (OrderAmount::UsdNotional(_), Side::Sell) => {
            let quote = size_matched * price;
            if !(quote >= 0.0 && quote.is_finite()) {
                return;
            }
            (0.0, quote)
        }
        _ => return,
    };
    invoke_write_filled_http_from_seed_pair(inner, making, taking);
}

/// Агрегирует исполнение одного POST и один раз дергает `invoke` после тишины.
pub(crate) struct PostOrderInvokeAggregator {
    /// Колбэк пользователя один раз через [`CompletionOnce`].
    slot: Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
    /// Хаб аккаунта: `order_id` → наш трекер; запись удаляют после успешного `fire`.
    trackers: Arc<RwLock<HashMap<String, TrackerEntry>>>,
    /// `order_id` этого POST.
    order_id: String,
    /// Состояние накопления (объём, цели, терминалы): [`tokio::sync::RwLock`] — только асинхронные `.await` локи.
    inner: Arc<RwLock<InvokeAggInner>>,
    /// Номер «волны» debounce: после паузы финализируют, только если генерация не сменилась.
    debounce_generation: Arc<RwLock<u64>>,
    /// `true`, когда финальный путь уже взят или колбэк вызван — чтобы не дублировать и остановить poll.
    pub(crate) finished: Arc<RwLock<bool>>,
}

pub struct TrackerEntry {
    pub(crate) invoke_aggregator: Arc<PostOrderInvokeAggregator>,
}

impl fmt::Debug for TrackerEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrackerEntry").finish_non_exhaustive()
    }
}

impl PostOrderInvokeAggregator {
    fn new(
        slot: Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
        trackers: Arc<RwLock<HashMap<String, TrackerEntry>>>,
        order_id: String,
        post_request: PostOrderRequest,
    ) -> Arc<Self> {
        let timestamp_ms_started = crate::util::current_timestamp_ms();
        let deadline_ms = post_request
            .expiration
            .map(|expiration| expiration.timestamp_millis())
            .or(post_request.market_end_unix_ms)
            .unwrap_or_else(|| timestamp_ms_started)
            .max(timestamp_ms_started)
            .saturating_add(INVOKE_FALLBACK_DEADLINE_MS_I64);
        let target = post_request.amount;
        let zero_fill_placeholder = match target {
            OrderAmount::Shares(_) => OrderAmount::Shares(0.0),
            OrderAmount::UsdNotional(_) => OrderAmount::UsdNotional(0.0),
        };

        Arc::new(Self {
            slot,
            trackers,
            order_id,
            inner: Arc::new(RwLock::new(InvokeAggInner {
                target,
                filled_ws: zero_fill_placeholder,
                filled_http: zero_fill_placeholder,
                deadline_ms,
                side: post_request.side,
                success: false,
                partial: false,
            })),
            debounce_generation: Arc::new(RwLock::new(0)),
            finished: Arc::new(RwLock::new(false)),
        })
    }

    fn bump_debounce_finalize(aggregator: Arc<Self>) {
        tokio::spawn(async move {
            let debounce_wave = {
                let mut wave_counter = aggregator.debounce_generation.write().await;
                *wave_counter = (*wave_counter).saturating_add(1);
                *wave_counter
            };
            // Если эффективный объём уже покрывает цель заявки — не ждём тишину: finalize сразу.
            let effective_amount_matches_target_goal = {
                let state = aggregator.inner.read().await;
                Self::targets_met(&state)
            };
            if effective_amount_matches_target_goal {
                Self::try_finalize_locked(aggregator).await;
                return;
            }
            tokio::time::sleep(Duration::from_millis(INVOKE_DEBOUNCE_MS)).await;
            let current_generation = *aggregator.debounce_generation.read().await;
            if current_generation != debounce_wave {
                return;
            }
            Self::try_finalize_locked(aggregator).await;
        });
    }

    async fn ingest_http_seed(self: &Arc<Self>, making: f64, taking: f64) {
        {
            let mut state = self.inner.write().await;
            invoke_write_filled_http_from_seed_pair(&mut state, making, taking);
        }
        Self::bump_debounce_finalize(Arc::clone(self));
    }

    //[проверено]
    async fn record_ws_trade_fill(self: &Arc<Self>, outcome_size: f64, quote_usdc: f64) {
        if !outcome_size.is_finite()
            || outcome_size <= 0.0
            || !quote_usdc.is_finite()
            || quote_usdc < 0.0
        {
            return;
        }
        {
            let mut state = self.inner.write().await;
            match &mut state.filled_ws {
                OrderAmount::Shares(shares_filled_so_far) => *shares_filled_so_far += outcome_size,
                OrderAmount::UsdNotional(usdc_filled_so_far) => {
                    *usdc_filled_so_far += quote_usdc
                }
            }
        }
        Self::bump_debounce_finalize(Arc::clone(self));
    }

    async fn record_ws_order_status(self: &Arc<Self>, status_raw: &str) {
        let normalized_status = status_raw.to_ascii_uppercase();
        {
            let mut state = self.inner.write().await;
            if normalized_status == "CANCELED" {
                state.partial = true;
            }
            if matches!(normalized_status.as_str(), "MATCHED" | "FILLED") {
                state.success = true;
            }
        }
        Self::bump_debounce_finalize(Arc::clone(self));
    }

    async fn record_poll_http(self: &Arc<Self>, open_order: OpenOrderResponse) {
        {
            let mut state = self.inner.write().await;
            invoke_apply_open_order_rest_snapshot(&mut state, &open_order);
            if matches!(&open_order.status, OrderStatusType::Canceled) {
                state.partial = true;
            }
            if matches!(&open_order.status, OrderStatusType::Matched) {
                state.success = true;
            }
        }
        Self::bump_debounce_finalize(Arc::clone(self));
    }

    fn effective_fill(state: &InvokeAggInner) -> OrderAmount {
        match (&state.filled_ws, &state.filled_http) {
            (
                OrderAmount::Shares(websocket_accumulated_shares),
                OrderAmount::Shares(http_snapshot_shares),
            ) => OrderAmount::Shares((*websocket_accumulated_shares).max(*http_snapshot_shares)),
            (
                OrderAmount::UsdNotional(websocket_accumulated_usdc),
                OrderAmount::UsdNotional(http_snapshot_usdc),
            ) => OrderAmount::UsdNotional((*websocket_accumulated_usdc).max(*http_snapshot_usdc)),
            (_, _) => state.filled_ws,
        }
    }

    fn targets_met(state: &InvokeAggInner) -> bool {
        let effective_total = Self::effective_fill(state);
        match (&state.target, &effective_total) {
            (OrderAmount::Shares(target_shares), OrderAmount::Shares(effective_shares)) => {
                target_shares.is_finite()
                    && *target_shares > 0.0
                    && *effective_shares + SHARE_EPS >= *target_shares
            }
            (
                OrderAmount::UsdNotional(target_usdc),
                OrderAmount::UsdNotional(effective_usdc),
            ) => {
                target_usdc.is_finite()
                    && *target_usdc > 0.0
                    && *effective_usdc + USD_EPS >= *target_usdc
            }
            _ => false,
        }
    }

    fn should_invoke(state: &InvokeAggInner, timestamp_ms: i64) -> bool {
        if Self::targets_met(state) {
            return true;
        }
        if state.success || state.partial {
            return true;
        }
        timestamp_ms >= state.deadline_ms
    }

    fn build_report(state: &InvokeAggInner, timestamp_ms: i64) -> SingleOrderClobInvocationReport {
        let effective_fill_amount = Self::effective_fill(state);
        let has_nonzero_fill = match &effective_fill_amount {
            OrderAmount::Shares(effective_shares) => *effective_shares > SHARE_EPS,
            OrderAmount::UsdNotional(effective_usdc) => *effective_usdc > USD_EPS,
        };
        let target_reached = Self::targets_met(state);
        let deadline_hit = timestamp_ms >= state.deadline_ms;

        let report_success = target_reached
            || (state.success && has_nonzero_fill)
            || (deadline_hit && has_nonzero_fill);
        let report_partial = report_success
            && !target_reached
            && (has_nonzero_fill || state.success || state.partial);

        if !target_reached && !has_nonzero_fill && (deadline_hit || state.partial) {
            return SingleOrderClobInvocationReport {
                order_id: String::new(),
                filled_amount: effective_fill_amount,
                success: false,
                partial: false,
            };
        }

        SingleOrderClobInvocationReport {
            order_id: String::new(),
            filled_amount: effective_fill_amount,
            success: report_success,
            partial: report_partial,
        }
    }

    async fn try_finalize_locked(self: Arc<Self>) {
        if *self.finished.read().await {
            return;
        }

        let timestamp_ms = crate::util::current_timestamp_ms();
        let ready_to_invoke = {
            let state = self.inner.read().await;
            Self::should_invoke(&state, timestamp_ms)
        };
        if !ready_to_invoke {
            return;
        }

        let claimed_finish = {
            let mut finished_guard = self.finished.write().await;
            if *finished_guard {
                false
            } else {
                *finished_guard = true;
                true
            }
        };
        if !claimed_finish {
            return;
        }

        let (report, committed_order_id) = {
            let state = self.inner.read().await;
            let timestamp_ms_recheck = crate::util::current_timestamp_ms();
            if !Self::should_invoke(&state, timestamp_ms_recheck) {
                {
                    let mut finished_guard = self.finished.write().await;
                    *finished_guard = false;
                }
                Self::bump_debounce_finalize(Arc::clone(&self));
                return;
            }
            let cloned_order_id = self.order_id.clone();
            let mut invocation_report = Self::build_report(&state, timestamp_ms_recheck);
            invocation_report.order_id.clone_from(&cloned_order_id);
            (invocation_report, cloned_order_id)
        };

        let _ = self.trackers.write().await.remove(&committed_order_id);

        self.slot.fire(report);
    }
}

async fn take_tracker_entry(
    trackers: &Arc<RwLock<HashMap<String, TrackerEntry>>>,
    order_id: &str,
) -> Option<TrackerEntry> {
    trackers.write().await.remove(order_id)
}

/// Накапливает объём исполнения по **`order_id`** (WS `trade`): outcome `size`, USDC≈ `size * price`.
//[проверено]
pub(crate) async fn accumulate_invoke_from_ws_trade(
    trackers: &Arc<RwLock<HashMap<String, TrackerEntry>>>,
    order_id: &str,
    outcome_size: f64,
    price: f64,
) {
    if order_id.is_empty() {
        return;
    }
    let quote = outcome_size * price;
    if !quote.is_finite() {
        return;
    }
    let trackers_snapshot = trackers.read().await;
    let Some(tracker_entry) = trackers_snapshot.get(order_id) else {
        return;
    };
    let invoke_aggregator_arc = Arc::clone(&tracker_entry.invoke_aggregator);
    drop(trackers_snapshot);
    invoke_aggregator_arc
        .record_ws_trade_fill(outcome_size, quote)
        .await;
}

/// User-WS `order.status` → флаги терминала invoke-агрегатора.
pub(crate) async fn notify_terminal_ws_order_snapshot(
    trackers: &Arc<RwLock<HashMap<String, TrackerEntry>>>,
    order_id: &str,
    order_status: &str,
) {
    if order_id.is_empty() {
        return;
    }
    let trackers_snapshot = trackers.read().await;
    let Some(tracker_entry) = trackers_snapshot.get(order_id) else {
        return;
    };
    let invoke_aggregator_arc = Arc::clone(&tracker_entry.invoke_aggregator);
    drop(trackers_snapshot);
    invoke_aggregator_arc
        .record_ws_order_status(order_status)
        .await;
}

fn terminal_http_invoke_success(status: &OrderStatusType) -> Option<bool> {
    match *status {
        OrderStatusType::Matched => Some(true),
        OrderStatusType::Canceled => Some(false),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct PostOrderHttpOutcome {
    pub order_id: String,
    pub success: bool,
    pub status: OrderStatusType,
    pub detail: Value,
    pub invoke_ctx: Option<PostOrderInvokeContext>,
}

fn spawn_invoke_poll_fallback(
    account: SharedAccount,
    order_id: String,
    aggregator: Arc<PostOrderInvokeAggregator>,
) {
    tokio::spawn(async move {
        loop {
            if *aggregator.finished.read().await {
                return;
            }
            let timestamp_ms = crate::util::current_timestamp_ms();
            let deadline_ms = aggregator.inner.read().await.deadline_ms;
            if timestamp_ms >= deadline_ms {
                PostOrderInvokeAggregator::bump_debounce_finalize(Arc::clone(&aggregator));
                return;
            }

            tokio::time::sleep(Duration::from_millis(INVOKE_FALLBACK_POLL_MS)).await;

            if *aggregator.finished.read().await {
                return;
            }

            let auth_client = match (**account.clob_authed.load()).clone() {
                Some(client) => client,
                None => continue,
            };

            let polled_order = match tokio::time::timeout(
                Duration::from_secs(ORDER_HTTP_POLL_TIMEOUT_SEC),
                auth_client.order(&order_id),
            )
            .await
            {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => {
                    crate::tee_eprintln!(
                        "[order_invoke/poll] client.order({order_id}) упал: {error:#}"
                    );
                    continue;
                }
                Err(_) => {
                    crate::tee_eprintln!(
                        "[order_invoke/poll] client.order({order_id}) timeout"
                    );
                    continue;
                }
            };

            let status_is_known_terminal =
                terminal_http_invoke_success(&polled_order.status).is_some();
            aggregator.record_poll_http(polled_order).await;

            let invoke_finished = *aggregator.finished.read().await;
            if invoke_finished || status_is_known_terminal {
                return;
            }
        }
    });
}

/// После HTTP POST: ошибка — колбэк сразу; иначе агрегатор. Если ответ уже `Matched`/`Canceled`, только debounce финала; если статус недотерминальный — дополнительно poll до дедлайна ([`spawn_invoke_poll_fallback`]).
pub(crate) async fn after_post_order_maybe_track_invoke(
    account: &SharedAccount,
    trackers: Arc<RwLock<HashMap<String, TrackerEntry>>>,
    http_result: &PostOrderHttpOutcome,
    slot_opt: Option<Arc<CompletionOnce<SingleOrderClobInvocationReport>>>,
) {
    let Some(slot) = slot_opt else {
        return;
    };
    let cloned_order_id = http_result.order_id.clone();

    if !http_result.success {
        slot.fire(SingleOrderClobInvocationReport {
            order_id: cloned_order_id.clone(),
            filled_amount: zero_fill_without_request_context(),
            success: false,
            partial: false,
        });
        let _ = take_tracker_entry(&trackers, &cloned_order_id).await;
        return;
    }

    let Some(invoke_context) = http_result.invoke_ctx.clone() else {
        slot.fire(SingleOrderClobInvocationReport {
            order_id: cloned_order_id.clone(),
            filled_amount: zero_fill_without_request_context(),
            success: false,
            partial: false,
        });
        return;
    };
    let posted_order_request = invoke_context.request;
    let seed_making = invoke_context.seed_making;
    let seed_taking = invoke_context.seed_taking;
    if cloned_order_id.is_empty() {
        slot.fire(SingleOrderClobInvocationReport {
            order_id: String::new(),
            filled_amount: zero_fill_matching_target_dimension(posted_order_request.amount),
            success: false,
            partial: false,
        });
        return;
    }
    let invoke_aggregator = PostOrderInvokeAggregator::new(
        Arc::clone(&slot),
        Arc::clone(&trackers),
        cloned_order_id.clone(),
        posted_order_request,
    );
    // POST уже Matched: объёмы дублируют WS trades — суммируем только из живого ордера (Live/Delayed/…).
    if !matches!(http_result.status, OrderStatusType::Matched) {
        invoke_aggregator
            .ingest_http_seed(seed_making, seed_taking)
            .await;
    }

    {
        let mut trackers_write_guard = trackers.write().await;
        trackers_write_guard.insert(
            cloned_order_id.clone(),
            TrackerEntry {
                invoke_aggregator: Arc::clone(&invoke_aggregator),
            },
        );
    }

    let terminal_status_from_http_post =
        terminal_http_invoke_success(&http_result.status).is_some();

    // Matched / Canceled в теле POST: дальнейший REST-poll статуса бессмысленен —
    // финал после тишины ([`INVOKE_DEBOUNCE_MS`]), учитывая добегающие WS `trade`/`order`.
    if terminal_status_from_http_post {
        {
            let mut invoke_state = invoke_aggregator.inner.write().await;
            if matches!(http_result.status, OrderStatusType::Canceled) {
                invoke_state.partial = true;
            }
            if matches!(http_result.status, OrderStatusType::Matched) {
                invoke_state.success = true;
            }
        }
        PostOrderInvokeAggregator::bump_debounce_finalize(Arc::clone(&invoke_aggregator));
        return;
    }

    spawn_invoke_poll_fallback(Arc::clone(account), cloned_order_id, invoke_aggregator);
}

pub(crate) fn wrap_post_order_cb(
    optional_callback: Option<SingleOrderInvokeCb>,
) -> Option<Arc<CompletionOnce<SingleOrderClobInvocationReport>>> {
    optional_callback.map(|on_invoke| Arc::new(CompletionOnce::new(on_invoke)))
}
