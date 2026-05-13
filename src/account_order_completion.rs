//! POST + `invoke`: один колбэк после **тишины** ([`INVOKE_DEBOUNCE_MS`]), если это не **taker** и на этом bump уже `now < market_end`; иначе сразу без паузы и без «волны» debounce.
//! Финал, если набран целевой объём (**shares / USDC** из [`crate::account_order::PostOrderRequest::amount`]) как **max(HTTP-снимок, накопление user-WS `trade`)** по полям агрегатора [`LegAgg`]: **`making_amount`** (collateral/USDC) и **`taking_amount`** (условный объём матча от `PostOrder` / [`OpenOrderResponse.size_matched`] / [`TradeResponse.size`]).
//! — последний успешный poll **или** дедлайн (**`expiration` → иначе `market_end_unix_ms` → короткий fallback**) **или** терминал CLOB / отмена.
//! Нулевое исполнение к дедлайну → `success=false`. Отмены — только HTTP. Агрегаты cancel-all / sell-all.
//!
//! Колбёк [`SingleOrderClobInvocationReport`]: те же имена что в **`PostOrderResponse`** — **`making_amount`** (отдано) и **`taking_amount`** (получено), в типах всё равно [`OrderAmount`].

//! Хаб живёт на [`crate::account::Account::order_invoke_hub`].

use crate::account::SharedAccount;
use crate::account_order::{OrderAmount, OrderRole, PostOrderRequest};
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
/// Порог «достаточного» набранного условного объёма (`LegAgg.taking_amount`) при Shares-цели заявки.
const SHARE_EPS: f64 = 1e-7;
/// Порог «достаточной» набранной колонки `making_amount` (`LegAgg`) при USDC-цели.
const USD_EPS: f64 = 1e-5;

/// Нули в порядке (`making_amount`, `taking_amount`), как [`polymarket_client_sdk::clob::types::response::PostOrderResponse`].
/// Используется для отчёта-«пустышки» при любом отказе до накопления исполнения (BUY/SELL, Taker/Maker).
#[inline]
pub(crate) fn zero_making_taking_for_side(side: Side) -> (OrderAmount, OrderAmount) {
    match side {
        Side::Buy => (
            OrderAmount::UsdNotional(0.0),
            OrderAmount::Shares(0.0),
        ),
        Side::Sell => (
            OrderAmount::Shares(0.0),
            OrderAmount::UsdNotional(0.0),
        ),
        _ => (
            OrderAmount::UsdNotional(0.0),
            OrderAmount::Shares(0.0),
        ),
    }
}

/// Когда после POST ещё нет контекста: конвенция BUY (`making_amount`, `taking_amount`).
fn zero_fill_without_request_context() -> (OrderAmount, OrderAmount) {
    zero_making_taking_for_side(Side::Buy)
}

/// Однократно отправить отчёт-провал (`success=false`, `partial=false`, нулевые суммы по `side`) через [`CompletionOnce`].
/// Безопасно вызывать после любого live fire — [`CompletionOnce`] гарантирует не более одного срабатывания.
pub(crate) fn fire_failed_invocation_for_side(
    slot: &Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
    side: Side,
) {
    let (making_amount, taking_amount) = zero_making_taking_for_side(side);
    slot.fire(SingleOrderClobInvocationReport {
        order_id: None,
        making_amount,
        taking_amount,
        success: false,
        partial: false,
    });
}

#[inline]
fn sanitize_nonneg_f64(x: f64) -> f64 {
    if !x.is_finite() || x < 0.0 {
        return 0.0;
    }
    x
}

fn sanitize_order_amount(a: OrderAmount) -> OrderAmount {
    match a {
        OrderAmount::Shares(x) => OrderAmount::Shares(sanitize_nonneg_f64(x)),
        OrderAmount::UsdNotional(x) => OrderAmount::UsdNotional(sanitize_nonneg_f64(x)),
    }
}

#[inline]
fn order_amount_usd_scalar(a: OrderAmount) -> f64 {
    match a {
        OrderAmount::UsdNotional(x) => sanitize_nonneg_f64(x),
        OrderAmount::Shares(_) => 0.0,
    }
}

#[inline]
fn order_amount_shares_scalar(a: OrderAmount) -> f64 {
    match a {
        OrderAmount::Shares(x) => sanitize_nonneg_f64(x),
        OrderAmount::UsdNotional(_) => 0.0,
    }
}

#[derive(Debug, Clone, Copy)]
struct LegAgg {
    /// После seed OpenOrder/Poll/WS нормализовано: **`UsdNotional`** — collateral-quote.
    making_amount: OrderAmount,
    /// Условный объём (**`Shares`**).
    taking_amount: OrderAmount,
}

impl Default for LegAgg {
    fn default() -> Self {
        Self {
            making_amount: OrderAmount::UsdNotional(0.0),
            taking_amount: OrderAmount::Shares(0.0),
        }
    }
}

impl LegAgg {
    fn sanitize_mut(&mut self) {
        self.making_amount = sanitize_order_amount(self.making_amount);
        self.taking_amount = sanitize_order_amount(self.taking_amount);
    }
}

fn leg_agg_add_trade_fill(leg_agg: LegAgg, size: f64, quote: f64) -> LegAgg {
    if !size.is_finite()
        || size <= 0.0
        || !quote.is_finite()
        || quote < 0.0
    {
        return leg_agg;
    }
    let mut out_leg_agg = leg_agg;
    let shares = order_amount_shares_scalar(out_leg_agg.taking_amount) + size;
    let usd = order_amount_usd_scalar(out_leg_agg.making_amount) + quote;
    out_leg_agg.taking_amount = OrderAmount::Shares(shares);
    out_leg_agg.making_amount = OrderAmount::UsdNotional(usd);
    out_leg_agg.sanitize_mut();
    out_leg_agg
}

fn leg_agg_max_normalized(a: LegAgg, b: LegAgg) -> LegAgg {
    let sh_a = order_amount_shares_scalar(a.taking_amount);
    let sh_b = order_amount_shares_scalar(b.taking_amount);
    let usd_a = order_amount_usd_scalar(a.making_amount);
    let usd_b = order_amount_usd_scalar(b.making_amount);
    let mut leg = LegAgg {
        taking_amount: OrderAmount::Shares(sanitize_nonneg_f64(sh_a.max(sh_b))),
        making_amount: OrderAmount::UsdNotional(sanitize_nonneg_f64(usd_a.max(usd_b))),
    };
    leg.sanitize_mut();
    leg
}

/// По эффективной паре ног считает объём исполнения **в размерности заявки** ([`InvokeAggInner::target`]).
fn target_dimension_fill_from_leg(target: OrderAmount, eff: LegAgg) -> OrderAmount {
    match target {
        OrderAmount::Shares(_) => OrderAmount::Shares(order_amount_shares_scalar(eff.taking_amount)),
        OrderAmount::UsdNotional(_) => {
            OrderAmount::UsdNotional(order_amount_usd_scalar(eff.making_amount))
        }
    }
}

/// Колбёк как в REST: экономические **making**/**taking**, не столбцы [`LegAgg`].
fn report_making_and_taking_amounts(side: Side, eff: LegAgg) -> (OrderAmount, OrderAmount) {
    let mk_norm = sanitize_order_amount(eff.making_amount);
    let tk_norm = sanitize_order_amount(eff.taking_amount);
    match side {
        Side::Buy => (mk_norm, tk_norm),
        Side::Sell => (tk_norm, mk_norm),
        _ => (mk_norm, tk_norm),
    }
}

/// Финальный отчёт по одному CLOB-POST.
///
/// Колбэк [`SingleOrderInvokeCb`] вызывается ровно один раз для любого результата POST —
/// провал (валидация, auth, HTTP/SDK error, timeout, server `success=false`),
/// частичная сделка или полное достижение цели — независимо от [`OrderRole::Taker`]/[`OrderRole::Maker`].
///
/// Конвенция `making_amount`/`taking_amount` идентична `PostOrderResponse` и единая для Taker и Maker:
/// - **BUY** (любая роль): `making_amount` — отданный USDC ([`OrderAmount::UsdNotional`]),
///   `taking_amount` — полученные shares ([`OrderAmount::Shares`]).
/// - **SELL** (любая роль): `making_amount` — отданные shares ([`OrderAmount::Shares`]),
///   `taking_amount` — полученный USDC ([`OrderAmount::UsdNotional`]).
///
/// При провале и при нулевом исполнении возвращаются нули в той же типовой раскладке по `side`,
/// и `order_id = None`.
#[derive(Debug, Clone)]
pub struct SingleOrderClobInvocationReport {
    /// `Some` только если CLOB принял ордер и было ненулевое исполнение (см. [`Self::success`]).
    pub order_id: Option<String>,
    /// «Отдано»: BUY → USDC, SELL → shares. Эквивалент `PostOrderResponse.making_amount`.
    pub making_amount: OrderAmount,
    /// «Получено»: BUY → shares, SELL → USDC. Эквивалент `PostOrderResponse.taking_amount`.
    pub taking_amount: OrderAmount,
    /// `true`, если было хоть какое-то исполнение (полное или частичное).
    pub success: bool,
    /// `true`, только если `success=true` и цель [`PostOrderRequest::amount`] не достигнута полностью.
    pub partial: bool,
}

#[inline]
fn nonempty_order_id_str(s: &str) -> Option<String> {
    (!s.is_empty()).then(|| s.to_string())
}

pub type SingleOrderInvokeCb =
    Box<dyn FnOnce(SingleOrderClobInvocationReport) + Send + 'static>;

/// Контекст после успешного HTTP POST для агрегатора invoke.
#[derive(Debug, Clone)]
pub struct PostOrderInvokeContext {
    pub request: PostOrderRequest,
    /// Колонки wire-ответа POST (`making_amount` / `taking_amount` из тела как `OrderAmount`).
    pub making_amount: OrderAmount,
    pub taking_amount: OrderAmount,
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
    /// Накопление по user-WS `trade` (`LegAgg`: USDC + shares).
    filled_ws: LegAgg,
    /// Снимок POST seed или последний GET [`OpenOrderResponse`] (**перезапись**).
    filled_http: LegAgg,
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

/// GET [`OpenOrderResponse`]: [`OpenOrderResponse.size_matched`] → [`LegAgg.taking_amount`], `×price` → [`LegAgg.making_amount`].
fn apply_open_order_response_snapshot(inner: &mut InvokeAggInner, open: &OpenOrderResponse) {
    let Some(size_matched) = decimal_snap_f64(&open.size_matched) else {
        return;
    };
    let Some(price) = decimal_snap_f64(&open.price) else {
        return;
    };
    if !(size_matched >= 0.0 && price >= 0.0 && price.is_finite()) {
        return;
    }
    let making_quote = size_matched * price;
    if !(making_quote >= 0.0 && making_quote.is_finite()) {
        return;
    }
    inner.filled_http = LegAgg {
        making_amount: OrderAmount::UsdNotional(making_quote),
        taking_amount: OrderAmount::Shares(size_matched),
    };
    inner.filled_http.sanitize_mut();
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
    /// Номер «волны» debounce: после паузы финализируют, только если генерация не сменилась (ветка паузы — см. [`Self::bump_debounce_finalize`]).
    debounce_generation: Arc<RwLock<u64>>,
    role: OrderRole,
    market_end_unix_ms: Option<i64>,
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

        Arc::new(Self {
            slot,
            trackers,
            order_id,
            inner: Arc::new(RwLock::new(InvokeAggInner {
                target,
                filled_ws: LegAgg::default(),
                filled_http: LegAgg::default(),
                deadline_ms,
                side: post_request.side,
                success: false,
                partial: false,
            })),
            debounce_generation: Arc::new(RwLock::new(0)),
            role: post_request.role,
            market_end_unix_ms: post_request.market_end_unix_ms,
            finished: Arc::new(RwLock::new(false)),
        })
    }

    fn bump_debounce_finalize(aggregator: Arc<Self>) {
        tokio::spawn(async move {
            let effective_amount_matches_target_goal = {
                let state = aggregator.inner.read().await;
                Self::targets_met(&state)
            };
            // Если эффективный объём уже покрывает цель заявки — не ждём тишину: finalize сразу.
            if effective_amount_matches_target_goal {
                Self::try_finalize_locked(aggregator).await;
                return;
            }
            let timestamp_ms = crate::util::current_timestamp_ms();
            if matches!(aggregator.role, OrderRole::Taker)
                || aggregator
                    .market_end_unix_ms
                    .is_some_and(|end_ms| end_ms <= timestamp_ms)
            {
                Self::try_finalize_locked(aggregator).await;
                return;
            }
            let debounce_wave = {
                let mut wave_counter = aggregator.debounce_generation.write().await;
                *wave_counter = (*wave_counter).saturating_add(1);
                *wave_counter
            };
            tokio::time::sleep(Duration::from_millis(INVOKE_DEBOUNCE_MS)).await;
            let current_generation = *aggregator.debounce_generation.read().await;
            if current_generation != debounce_wave {
                return;
            }
            Self::try_finalize_locked(aggregator).await;
        });
    }

    async fn ingest_post_order_snapshot(
        self: &Arc<Self>,
        post_making_amount: OrderAmount,
        post_taking_amount: OrderAmount,
    ) {
        {
            let mut state = self.inner.write().await;
            let mut leg = match state.side {
                Side::Buy => LegAgg {
                    making_amount: post_making_amount,
                    taking_amount: post_taking_amount,
                },
                Side::Sell => LegAgg {
                    making_amount: post_taking_amount,
                    taking_amount: post_making_amount,
                },
                _ => LegAgg {
                    making_amount: post_making_amount,
                    taking_amount: post_taking_amount,
                },
            };
            leg.sanitize_mut();
            state.filled_http = leg;
        }
        Self::bump_debounce_finalize(Arc::clone(self));
    }

    //[проверено]
    async fn record_trade_aggregate_from_ws_event(
        self: &Arc<Self>,
        size: f64,
        quote: f64,
    ) {
        if !size.is_finite()
            || size <= 0.0
            || !quote.is_finite()
            || quote < 0.0
        {
            return;
        }
        {
            let mut state = self.inner.write().await;
            state.filled_ws = leg_agg_add_trade_fill(state.filled_ws, size, quote);
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
            apply_open_order_response_snapshot(&mut state, &open_order);
            if matches!(&open_order.status, OrderStatusType::Canceled) {
                state.partial = true;
            }
            if matches!(&open_order.status, OrderStatusType::Matched) {
                state.success = true;
            }
        }
        Self::bump_debounce_finalize(Arc::clone(self));
    }

    fn effective_leg(state: &InvokeAggInner) -> LegAgg {
        leg_agg_max_normalized(state.filled_ws, state.filled_http)
    }

    fn target_progress(state: &InvokeAggInner) -> OrderAmount {
        let effective_leg = Self::effective_leg(state);
        target_dimension_fill_from_leg(state.target, effective_leg)
    }

    fn targets_met(state: &InvokeAggInner) -> bool {
        let target_progress = Self::target_progress(state);
        match (&state.target, &target_progress) {
            (OrderAmount::Shares(target_shares), OrderAmount::Shares(effective_shares)) => {
                target_shares.is_finite() && *target_shares > 0.0 && *effective_shares + SHARE_EPS >= *target_shares
            }
            (OrderAmount::UsdNotional(target_usdc), OrderAmount::UsdNotional(effective_usdc)) => {
                target_usdc.is_finite() && *target_usdc > 0.0 && *effective_usdc + USD_EPS >= *target_usdc
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
        let effective_leg = Self::effective_leg(state);
        let (making_amount, taking_amount) = report_making_and_taking_amounts(state.side, effective_leg);

        let sheres: f64 = order_amount_shares_scalar(effective_leg.taking_amount);
        let usd = order_amount_usd_scalar(effective_leg.making_amount);
        let has_nonzero_fill = sheres > SHARE_EPS || usd > USD_EPS;
        let target_reached = Self::targets_met(state);
        let deadline_hit = timestamp_ms >= state.deadline_ms;

        // Любое ненулевое исполнение (taker/maker, частичное или полное целевое) ⇒ `success`.
        let report_success = target_reached || has_nonzero_fill;
        let report_partial = report_success
            && !target_reached
            && (has_nonzero_fill || state.success || state.partial);

        if !target_reached && !has_nonzero_fill && (deadline_hit || state.partial) {
            return SingleOrderClobInvocationReport {
                order_id: None,
                making_amount,
                taking_amount,
                success: false,
                partial: false,
            };
        }

        SingleOrderClobInvocationReport {
            order_id: None,
            making_amount,
            taking_amount,
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
            invocation_report.order_id = nonempty_order_id_str(&cloned_order_id);
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

/// Накапливает исполнение по **`order_id`** (WS [`TradeResponse`]: `size`×`price` → `quote` в коллатерале).
//[проверено]
pub(crate) async fn accumulate_invoke_from_ws_trade(
    trackers: &Arc<RwLock<HashMap<String, TrackerEntry>>>,
    order_id: &str,
    size: f64,
    price: f64,
) {
    if order_id.is_empty() {
        return;
    }
    let quote = size * price;
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
        .record_trade_aggregate_from_ws_event(size, quote)
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
    slot: Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
) {
    let cloned_order_id = http_result.order_id.clone();
    let side_for_zero_fill = http_result
        .invoke_ctx
        .as_ref()
        .map(|c| c.request.side)
        .unwrap_or(Side::Buy);

    if !http_result.success {
        let (making_z, taking_z) = zero_making_taking_for_side(side_for_zero_fill);
        slot.fire(SingleOrderClobInvocationReport {
            order_id: nonempty_order_id_str(&cloned_order_id),
            making_amount: making_z,
            taking_amount: taking_z,
            success: false,
            partial: false,
        });
        let _ = take_tracker_entry(&trackers, &cloned_order_id).await;
        return;
    }

    let Some(invoke_context) = http_result.invoke_ctx.clone() else {
        let (making_z, taking_z) = zero_fill_without_request_context();
        slot.fire(SingleOrderClobInvocationReport {
            order_id: nonempty_order_id_str(&cloned_order_id),
            making_amount: making_z,
            taking_amount: taking_z,
            success: false,
            partial: false,
        });
        return;
    };
    let posted_order_request = invoke_context.request;
    let making_amount = invoke_context.making_amount;
    let taking_amount = invoke_context.taking_amount;
    if cloned_order_id.is_empty() {
        let (making_z, taking_z) = zero_making_taking_for_side(posted_order_request.side);
        slot.fire(SingleOrderClobInvocationReport {
            order_id: None,
            making_amount: making_z,
            taking_amount: taking_z,
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
            .ingest_post_order_snapshot(making_amount, taking_amount)
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
    invoke: SingleOrderInvokeCb,
) -> Arc<CompletionOnce<SingleOrderClobInvocationReport>> {
    Arc::new(CompletionOnce::new(invoke))
}
