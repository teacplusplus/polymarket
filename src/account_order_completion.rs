//! POST + `invoke`: один колбэк один раз — но **только после on-chain settlement** (debit + credit средств).
//!
//! Источник истины — [`polymarket_client_sdk::clob::types::TradeStatusType`] лифсайкла трейда:
//! `Matched` (только в книге CLOB) → `Mined` (включён в блок) → `Confirmed` (после финализации).
//! Зачислением считаем `Mined|Confirmed` (см. [`PostOrderInvokeAggregator`]). Состояния
//! `Retrying|Failed` контрибьютят в book-match агрегат (`filled_*`), но **не** в settlement-агрегат
//! (`settled_*`); если они так и не перейдут в `Mined|Confirmed` к дедлайну —
//! [`SingleOrderClobInvocationReport::success`] = `false` с диагностикой `error_msg = "settlement_timeout: ..."`.
//!
//! Финал = `success=true` фаирится, как только выполнено хотя бы одно из:
//! 1. **`max(settled_ws, settled_http)`** покрывает целевой объём заявки
//!    [`crate::account_order::PostOrderRequest::amount`] (full target reached);
//! 2. достигнут book-level терминал ([`InvokeAggInner::book_terminal_reached`]) — для Taker
//!    это всегда сразу после `POST /order`, для Maker — статус `Matched|Canceled|Unmatched`
//!    из POST/REST/WS — **и** settlement догнал book-match по обеим осям (нечего больше ждать);
//! 3. дедлайн ([`INVOKE_FALLBACK_POLL_DEADLINE_SEC`]) — best-effort с диагностикой в `error_msg`.
//!
//! Поэтому Taker FAK с partial fill финализируется через settlement-задержку (~1–3s),
//! а не через 30s дедлайн.
//!
//! Источники сигнала:
//! - **HTTP**: REST-poll каждые [`INVOKE_FALLBACK_POLL_MS`] дёргает `client.order(order_id)` +
//!   (если `size_matched > 0`) `client.trades(order_id)` и партиционирует трейды на
//!   book-matched (всё) и settled (`status ∈ {Mined, Confirmed}`). Поллинг **не** прерывается на
//!   `OrderStatusType::Matched` — он живёт до finalize или дедлайна
//!   ([`INVOKE_FALLBACK_POLL_DEADLINE_SEC`]).
//! - **WS**: user-channel `trade`-события несут `status` по тому же лифсайклу. Дедуплицируется
//!   по `trade_id` ([`InvokeAggInner::seen_ws_trade_ids`]) — каждый трейд учитывается в
//!   `filled_ws` ровно один раз и в `settled_ws` ровно один раз (при первом `MINED|CONFIRMED`).
//!
//! **Учёт fee:** per-trade `fee_rate_bps` из [`TradeResponse`] / WS `trade.fee_rate_bps`
//! применяется к **taking-стороне** (BUY → меньше shares; SELL → меньше USDC), см.
//! [`apply_fee_to_taking_side`]. Settled-leg, из которой строится финальный отчёт, всегда NET.
//!
//! Колбёк [`SingleOrderClobInvocationReport`]: имена как в `PostOrderResponse` —
//! **`making_amount`** (отдано) и **`taking_amount`** (получено) в [`OrderAmount`]. Суммы в отчёте
//! берутся из **settled-leg** и уже NET of fee (правда о фактически зачисленных средствах).

//! Хаб живёт на [`crate::account::Account::order_invoke_hub`].

use crate::account::SharedAccount;
use crate::account_order::{OrderAmount, OrderRole, PostOrderRequest};
use polymarket_client_sdk::clob::types::request::TradesRequest;
use polymarket_client_sdk::clob::types::response::{OpenOrderResponse, TradeResponse};
use polymarket_client_sdk::clob::types::{OrderStatusType, Side, TradeStatusType};
use polymarket_client_sdk::types::Decimal;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;

/// Пауза без новых событий (мс): после неё считается, что можно финализировать и вызвать invoke один раз.
const INVOKE_DEBOUNCE_MS: u64 = 450;
/// Запас по времени (сек), если ни `expiration`, ни `market_end_unix_ms` не заданы — верхняя граница опроса/финала.
pub(crate) const INVOKE_FALLBACK_POLL_DEADLINE_SEC: u64 = 5;
/// То же окно fallback в миллисекундах как `i64` для [`crate::util::current_timestamp_ms`].
pub(crate) const INVOKE_FALLBACK_DEADLINE_MS_I64: i64 =
    (INVOKE_FALLBACK_POLL_DEADLINE_SEC as i64).saturating_mul(1000);
const INVOKE_FALLBACK_POLL_MS: u64 = 500;
const ORDER_HTTP_POLL_TIMEOUT_SEC: u64 = 10;
/// SDK-маркер «страниц больше нет» в [`polymarket_client_sdk::clob::types::response::Page::next_cursor`] (base64 `-1`).
const TRADES_PAGE_TERMINAL_CURSOR: &str = "LTE=";
/// Защита от деградации в бесконечный paginate в `client.trades(...)`.
const TRADES_MAX_PAGES_PER_POLL: usize = 8;
/// Порог «достаточного» набранного условного объёма (`LegAgg.taking_amount`) при Shares-цели заявки.
const SHARE_EPS: f64 = 1e-7;
/// Порог «достаточной» набранной колонки `making_amount` (`LegAgg`) при USDC-цели.
const USD_EPS: f64 = 1e-5;

/// Нули в порядке (`making_amount`, `taking_amount`), как [`polymarket_client_sdk::clob::types::response::PostOrderResponse`].
/// Используется для отчёта-«пустышки» при любом отказе до накопления исполнения (BUY/SELL, Taker/Maker).
#[inline]
pub(crate) fn zero_making_taking_for_side(side: Side) -> (OrderAmount, OrderAmount) {
    match side {
        Side::Buy => (OrderAmount::UsdNotional(0.0), OrderAmount::Shares(0.0)),
        Side::Sell => (OrderAmount::Shares(0.0), OrderAmount::UsdNotional(0.0)),
        _ => (OrderAmount::UsdNotional(0.0), OrderAmount::Shares(0.0)),
    }
}

/// Когда после POST ещё нет контекста: конвенция BUY (`making_amount`, `taking_amount`).
fn zero_fill_without_request_context() -> (OrderAmount, OrderAmount) {
    zero_making_taking_for_side(Side::Buy)
}

/// Запускает [`CompletionOnce::fire`] **на отдельной** [`tokio::spawn`]-таске, чтобы пользовательский
/// колбэк никогда не выполнялся синхронно в стэке вызывающего (в т.ч. внутри [`crate::account_order::post_order_on_clob`]).
/// [`CompletionOnce`] продолжает гарантировать «не более одного срабатывания» при гонке с агрегатором/poll/WS.
pub(crate) fn spawn_fire_invocation_report(
    slot: &Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
    report: SingleOrderClobInvocationReport,
) {
    let slot_arc = Arc::clone(slot);
    tokio::spawn(async move {
        slot_arc.fire(report);
    });
}

/// Однократно отправить отчёт-провал (`success=false`, `partial=false`, нулевые суммы по `side`,
/// `error_msg` для диагностики) через [`CompletionOnce`]. Колбэк уходит на отдельную
/// [`tokio::spawn`]-таску ([`spawn_fire_invocation_report`]) — синхронно в стэке вызывающего он **не**
/// исполнится. Безопасно вызывать после любого live fire — [`CompletionOnce`] гарантирует
/// не более одного срабатывания.
pub(crate) fn fire_failed_invocation_for_side(
    slot: &Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
    side: Side,
    error_msg: Option<String>,
) {
    let (making_amount, taking_amount) = zero_making_taking_for_side(side);
    spawn_fire_invocation_report(
        slot,
        SingleOrderClobInvocationReport {
            order_id: None,
            making_amount,
            taking_amount,
            success: false,
            partial: false,
            error_msg,
        },
    );
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
    if !size.is_finite() || size <= 0.0 || !quote.is_finite() || quote < 0.0 {
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
        OrderAmount::Shares(_) => {
            OrderAmount::Shares(order_amount_shares_scalar(eff.taking_amount))
        }
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
/// Срабатывание всегда происходит на отдельной [`tokio::spawn`]-таске
/// (`spawn_fire_invocation_report` для ранних провалов, [`PostOrderInvokeAggregator`] для finalize),
/// т.е. **никогда** синхронно в стэке вызывающего `post_order_on_clob`.
///
/// **Гарантии тайминга:** `success=true` фаирится сразу как только выполнено хотя бы одно из:
/// 1. settlement покрыл целевой объём (фул-филл с зачислением on-chain), либо
/// 2. достигнут book-level терминал заявки (POST `Matched|Canceled|Unmatched` для Taker всегда;
///    для Maker — Matched/Canceled/Unmatched по POST/REST/WS) **И** settlement догнал
///    book-match по обеим осям (нечего больше ждать), либо
/// 3. дедлайн (best-effort, см. [`SingleOrderClobInvocationReport::error_msg`]).
///
/// Поэтому для Taker FAK колбэк выстреливает в течение settlement-задержки релайера
/// (~1–3s в норме), а не по дедлайну.
///
/// **Гарантии чисел:** `making_amount`/`taking_amount` — **net of fee** (то, что реально
/// списано/зачислено на чейне), не gross-нотасчик подписанного ордера. Источник истины —
/// per-trade `TradeResponse.fee_rate_bps` (REST `client.trades(...)`) и WS user-channel
/// `trade.fee_rate_bps`. Fee удерживается с **taking-стороны**: BUY → меньше shares; SELL →
/// меньше USDC. Конвенция `making`/`taking` идентична `PostOrderResponse`:
/// - **BUY** (любая роль): `making_amount` — отданный USDC ([`OrderAmount::UsdNotional`]),
///   `taking_amount` — полученные shares **после fee** ([`OrderAmount::Shares`]).
/// - **SELL** (любая роль): `making_amount` — отданные shares ([`OrderAmount::Shares`]),
///   `taking_amount` — полученный USDC **после fee** ([`OrderAmount::UsdNotional`]).
///
/// При провале и при нулевом исполнении возвращаются нули в той же типовой раскладке по `side`,
/// и `order_id = None`.
#[derive(Debug, Clone)]
pub struct SingleOrderClobInvocationReport {
    /// `Some` только если CLOB принял ордер и было ненулевое исполнение (см. [`Self::success`]).
    pub order_id: Option<String>,
    /// «Отдано» **после fee**: BUY → USDC (gross, fee на taking-стороне), SELL → shares (gross).
    /// Эквивалент `PostOrderResponse.making_amount` по семантике, но с учётом удержанной fee.
    pub making_amount: OrderAmount,
    /// «Получено» **после fee**: BUY → shares (net), SELL → USDC (net).
    /// Эквивалент `PostOrderResponse.taking_amount` по семантике, но с учётом удержанной fee.
    pub taking_amount: OrderAmount,
    /// `true`, если было хоть какое-то исполнение (полное или частичное).
    pub success: bool,
    /// `true`, только если `success=true` и цель [`PostOrderRequest::amount`] не достигнута полностью.
    pub partial: bool,
    /// Текст ошибки для диагностики при `success=false` (HTTP/SDK error и его тело, server `error_msg`,
    /// валидация, отсутствие auth/signer, build/sign-сбой, дедлайн без исполнения, отмена до исполнения).
    /// `None` для всех «нормальных» исходов (полная или частичная сделка).
    pub error_msg: Option<String>,
}

#[inline]
fn nonempty_order_id_str(s: &str) -> Option<String> {
    (!s.is_empty()).then(|| s.to_string())
}

pub type SingleOrderInvokeCb = Box<dyn FnOnce(SingleOrderClobInvocationReport) + Send + 'static>;

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
    /// Book-matched накопление по user-WS `trade` (любой из `MATCHED|MINED|CONFIRMED`).
    /// Дедуплицируется по `trade_id` ([`Self::seen_ws_trade_ids`]).
    filled_ws: LegAgg,
    /// On-chain settled накопление по user-WS `trade` (только `MINED|CONFIRMED`).
    /// Дедуплицируется по `trade_id` ([`Self::settled_seen_ws_trade_ids`]).
    settled_ws: LegAgg,
    /// Book-matched HTTP-снимок (POST seed + max-merge с REST-poll агрегатом всех `trades`
    /// либо `size_matched × order.price`).
    filled_http: LegAgg,
    /// On-chain settled HTTP-снимок (max-merge с агрегатом REST-`trades` со статусом
    /// `Mined|Confirmed`). POST seed **не** контрибьютит сюда — POST-ответ говорит только о
    /// book-match, релайер ещё не отработал.
    settled_http: LegAgg,
    /// `trade_id` уже учтённые в `filled_ws` — защита от тройного счёта при WS-лифсайкле
    /// одного трейда (`MATCHED`→`MINED`→`CONFIRMED` могут прийти как 1..3 события).
    seen_ws_trade_ids: HashSet<String>,
    /// `trade_id` уже учтённые в `settled_ws` — отдельный сет, потому что settle-первая
    /// контрибьюция — это первый `MINED|CONFIRMED` после возможного предыдущего `MATCHED`.
    settled_seen_ws_trade_ids: HashSet<String>,
    /// Unix время (ms): после этого момента finalize допускается даже без полного набора объёма.
    deadline_ms: i64,
    /// [`Side`] ордера (копия из [`crate::account_order::PostOrderRequest::side`]) — управляет
    /// трактовкой HTTP `making`/`taking` при seed и применением fee к нужной стороне (taking).
    side: Side,
    /// Book-level терминал заявки: больше fill'ов не ожидается. Ставится:
    /// - **сразу** в [`after_post_order_maybe_track_invoke`] для любого Taker (taker не остаётся
    ///   в книге) — отсюда taker FAK с partial fill финализируется на settlement, а не по дедлайну;
    /// - в [`Self::record_poll_http`] / [`Self::record_ws_order_status`] для статусов CLOB
    ///   `Matched|Canceled|Unmatched` (POST/REST) либо `MATCHED|FILLED|CANCELED` (WS) у Maker.
    ///
    /// Главный гейт `success=true` в [`PostOrderInvokeAggregator::should_invoke`]: terminal +
    /// `settlement_caught_up_with_match` ⇒ финализируем (выстреливаем колбэк), потому что
    /// ждать больше нечего.
    book_terminal_reached: bool,
    /// От CLOB поступил успешный «исполнен»-терминал на уровне book (WS MATCHED/FILLED,
    /// POST/poll/SDK `Matched`). **Информационный**: гейт finalize строится на
    /// [`Self::book_terminal_reached`] + settlement; см. [`PostOrderInvokeAggregator::should_invoke`].
    success: bool,
    /// От CLOB поступила терминальная отмена (WS `CANCELED`, POST/poll `Canceled`). Информационный
    /// флаг для диагностики `error_msg`; book-level терминал отдельно — [`Self::book_terminal_reached`].
    partial: bool,
}

fn decimal_snap_f64(d: &Decimal) -> Option<f64> {
    let f = d.to_string().parse::<f64>().ok()?;
    f.is_finite().then_some(f)
}

/// Компактная строка для observability: shares/USDC по обеим ногам и флаги терминала.
/// Используется в `[order_invoke/ws]`, `[order_invoke/poll]`, `[order_invoke/final]`.
fn leg_summary_for_log(state: &InvokeAggInner) -> String {
    let book_shares = order_amount_shares_scalar(
        leg_agg_max_normalized(state.filled_ws, state.filled_http).taking_amount,
    );
    let book_usd = order_amount_usd_scalar(
        leg_agg_max_normalized(state.filled_ws, state.filled_http).making_amount,
    );
    let settled_shares = order_amount_shares_scalar(
        leg_agg_max_normalized(state.settled_ws, state.settled_http).taking_amount,
    );
    let settled_usd = order_amount_usd_scalar(
        leg_agg_max_normalized(state.settled_ws, state.settled_http).making_amount,
    );
    let f_ws_sh = order_amount_shares_scalar(state.filled_ws.taking_amount);
    let f_ws_us = order_amount_usd_scalar(state.filled_ws.making_amount);
    let f_ht_sh = order_amount_shares_scalar(state.filled_http.taking_amount);
    let f_ht_us = order_amount_usd_scalar(state.filled_http.making_amount);
    let s_ws_sh = order_amount_shares_scalar(state.settled_ws.taking_amount);
    let s_ws_us = order_amount_usd_scalar(state.settled_ws.making_amount);
    let s_ht_sh = order_amount_shares_scalar(state.settled_http.taking_amount);
    let s_ht_us = order_amount_usd_scalar(state.settled_http.making_amount);
    format!(
        "book={book_shares:.6}sh/{book_usd:.6}$ (ws={f_ws_sh:.6}/{f_ws_us:.6}, \
         http={f_ht_sh:.6}/{f_ht_us:.6}) settled={settled_shares:.6}sh/{settled_usd:.6}$ \
         (ws={s_ws_sh:.6}/{s_ws_us:.6}, http={s_ht_sh:.6}/{s_ht_us:.6}) \
         term={} part={} succ={}",
        state.book_terminal_reached, state.partial, state.success,
    )
}

/// `true` если трейд on-chain (включён в блок или подтверждён). См. [`TradeStatusType`].
/// `Matched` — только в книге CLOB (ещё не on-chain). `Retrying`/`Failed` — попытки, но пока
/// нет факта зачисления. `Unknown` — консервативно `false`.
#[inline]
pub(crate) fn trade_status_settled_on_chain(status: &TradeStatusType) -> bool {
    matches!(status, TradeStatusType::Mined | TradeStatusType::Confirmed)
}

/// То же по сырой строке статуса WS user-channel `trade.status`.
#[inline]
pub(crate) fn ws_trade_status_settled_on_chain(status_raw: &str) -> bool {
    matches!(
        status_raw.to_ascii_uppercase().as_str(),
        "MINED" | "CONFIRMED"
    )
}

/// Per-trade fee_factor: `(1 - fee_rate_bps/10_000)`, защищён от мусорных значений.
#[inline]
fn fee_factor_from_bps(fee_rate_bps: f64) -> f64 {
    let bps = if fee_rate_bps.is_finite() {
        fee_rate_bps.clamp(0.0, 10_000.0)
    } else {
        0.0
    };
    (1.0 - bps / 10_000.0).clamp(0.0, 1.0)
}

/// Применить fee к (size, quote) в зависимости от стороны ордера: Polymarket удерживает fee
/// с **taking-стороны** (то, что user получает). Возвращает NET значения, готовые для
/// аккумуляции в нормированный [`LegAgg`] (shares в `taking_amount`, USDC в `making_amount`).
/// - **BUY** taking=shares → `size_net = size × fee_factor`, `quote_net = size × price` (gross).
/// - **SELL** taking=USDC → `size_net = size` (gross), `quote_net = size × price × fee_factor`.
#[inline]
fn apply_fee_to_taking_side(side: Side, size: f64, quote: f64, fee_factor: f64) -> (f64, f64) {
    match side {
        Side::Buy => (size * fee_factor, quote),
        Side::Sell => (size, quote * fee_factor),
        _ => (size, quote),
    }
}

/// Сумма `Σ` по реальным фактам из [`TradeResponse`] с **учётом per-trade fee_rate_bps**.
/// Fee удерживается с taking-стороны (см. [`apply_fee_to_taking_side`]). Точно для обеих
/// сторон и обеих ролей: для taker нивелирует worst-acceptable cap, для maker эквивалентно
/// `size_matched × limit_price` минус fee.
fn aggregate_trades_into_leg<'a>(
    side: Side,
    trades: impl IntoIterator<Item = &'a TradeResponse>,
) -> LegAgg {
    let mut total_shares = 0.0_f64;
    let mut total_usdc = 0.0_f64;
    for trade in trades {
        let Some(size) = decimal_snap_f64(&trade.size) else {
            continue;
        };
        let Some(price) = decimal_snap_f64(&trade.price) else {
            continue;
        };
        if !(size >= 0.0 && price >= 0.0 && price.is_finite()) {
            continue;
        }
        let fee_factor = fee_factor_from_bps(decimal_snap_f64(&trade.fee_rate_bps).unwrap_or(0.0));
        let quote = size * price;
        if !quote.is_finite() {
            continue;
        }
        let (size_net, quote_net) = apply_fee_to_taking_side(side, size, quote, fee_factor);
        if !size_net.is_finite() || !quote_net.is_finite() {
            continue;
        }
        total_shares += size_net;
        total_usdc += quote_net;
    }
    let mut leg = LegAgg {
        making_amount: OrderAmount::UsdNotional(sanitize_nonneg_f64(total_usdc)),
        taking_amount: OrderAmount::Shares(sanitize_nonneg_f64(total_shares)),
    };
    leg.sanitize_mut();
    leg
}

/// REST-poll → обновить **обе** ноги:
/// - `filled_http` (book-matched): если REST-`trades` полные (`Σ size ≥ size_matched`) —
///   **перезаписываем** (authoritative net с учётом fee), иначе fallback на
///   `size_matched × order.price` (gross) через max-merge. Это важно: max-merge с gross-seed'ом
///   из POST вместо перезаписи маскировал бы NET-снижение через fee.
/// - `settled_http` (on-chain, `status ∈ {Mined, Confirmed}`): max-merge с агрегатом
///   отфильтрованных трейдов (монотонно растёт по определению — max-merge безопасен).
///
/// Также выставляет [`InvokeAggInner::book_terminal_reached`] для `OrderStatusType` ∈
/// {`Matched`, `Canceled`, `Unmatched`} — это «больше fill'ов не будет» сигнал, по которому
/// finalize может выстрелить сразу после catch-up settlement.
fn apply_polled_snapshot(
    inner: &mut InvokeAggInner,
    open: &OpenOrderResponse,
    trades: Option<&[TradeResponse]>,
) {
    let side = inner.side;
    let Some(size_matched) = decimal_snap_f64(&open.size_matched) else {
        return;
    };
    let Some(order_price) = decimal_snap_f64(&open.price) else {
        return;
    };
    if !(size_matched >= 0.0 && order_price >= 0.0 && order_price.is_finite()) {
        return;
    }

    match trades {
        Some(ts) => {
            let trades_leg = aggregate_trades_into_leg(side, ts.iter());
            // Полноту определяем по **gross** `Σ size` (до fee), потому что
            // `OpenOrderResponse::size_matched` тоже gross. fee может сжать `taking_amount`
            // в `trades_leg`, но не меняет shares-учёт самого матча.
            let trades_gross_size_sum: f64 = ts
                .iter()
                .filter_map(|t| decimal_snap_f64(&t.size))
                .filter(|v| v.is_finite() && *v >= 0.0)
                .sum();
            let trades_cover_matched =
                size_matched > 0.0 && trades_gross_size_sum + SHARE_EPS >= size_matched;
            if trades_cover_matched {
                // Trades полные — authoritative NET, **перезапись** (не max-merge), чтобы
                // gross-сид от POST не маскировал NET-снижение через fee.
                inner.filled_http = trades_leg;
            } else if size_matched > 0.0 {
                // Trades ещё не подтянулись до `size_matched` — gross fallback через max-merge.
                let making_quote = size_matched * order_price;
                if making_quote.is_finite() {
                    let fallback_leg = LegAgg {
                        making_amount: OrderAmount::UsdNotional(sanitize_nonneg_f64(making_quote)),
                        taking_amount: OrderAmount::Shares(sanitize_nonneg_f64(size_matched)),
                    };
                    inner.filled_http = leg_agg_max_normalized(inner.filled_http, fallback_leg);
                }
            }

            let settled_iter = ts
                .iter()
                .filter(|trade| trade_status_settled_on_chain(&trade.status));
            let settled_leg = aggregate_trades_into_leg(side, settled_iter);
            inner.settled_http = leg_agg_max_normalized(inner.settled_http, settled_leg);
        }
        None => {
            // Trades request упал/таймаут — пользуемся только `size_matched × order.price`
            // как gross-оценкой; settled_http не трогаем (нет данных).
            if size_matched > 0.0 {
                let making_quote = size_matched * order_price;
                if making_quote.is_finite() {
                    let fallback_leg = LegAgg {
                        making_amount: OrderAmount::UsdNotional(sanitize_nonneg_f64(making_quote)),
                        taking_amount: OrderAmount::Shares(sanitize_nonneg_f64(size_matched)),
                    };
                    inner.filled_http = leg_agg_max_normalized(inner.filled_http, fallback_leg);
                }
            }
        }
    }

    if matches!(
        &open.status,
        OrderStatusType::Matched | OrderStatusType::Canceled | OrderStatusType::Unmatched
    ) {
        inner.book_terminal_reached = true;
    }
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
    /// Целевой объём (копия из [`PostOrderRequest::amount`]) — для observability в логах.
    target: OrderAmount,
    /// Side ордера (копия) — для observability и в replay-cmd.
    side: Side,
    /// Asset (token_id строкой) — для replay/curl.
    asset_id: String,
    /// Unix-ms старта трекера (момент возврата POST), для подсчёта latency до колбэка.
    started_at_ms: i64,
    /// Сколько раз отработал HTTP-poll (`spawn_invoke_poll_fallback` iteration count).
    /// Под `RwLock` (а не `AtomicU64`) для консистентности с остальными `Arc<RwLock<…>>`
    /// полями структуры; инкремент идёт под `.write().await` и возвращает свежее значение
    /// в той же критической секции — никаких read-after-write race'ов.
    http_poll_count: Arc<RwLock<u64>>,
    /// Сколько раз пришло учтённое user-WS `trade`-событие (та же причина для `RwLock`).
    ws_trade_count: Arc<RwLock<u64>>,
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
        let side = post_request.side;
        let role = post_request.role;
        let asset_id = post_request.asset_id.clone();
        let market_end_unix_ms = post_request.market_end_unix_ms;
        let expiration_unix_ms = post_request.expiration.map(|e| e.timestamp_millis());

        let aggregator = Arc::new(Self {
            slot,
            trackers,
            order_id: order_id.clone(),
            inner: Arc::new(RwLock::new(InvokeAggInner {
                target,
                filled_ws: LegAgg::default(),
                settled_ws: LegAgg::default(),
                filled_http: LegAgg::default(),
                settled_http: LegAgg::default(),
                seen_ws_trade_ids: HashSet::new(),
                settled_seen_ws_trade_ids: HashSet::new(),
                deadline_ms,
                side,
                book_terminal_reached: false,
                success: false,
                partial: false,
            })),
            debounce_generation: Arc::new(RwLock::new(0)),
            role,
            market_end_unix_ms,
            target,
            side,
            asset_id: asset_id.clone(),
            started_at_ms: timestamp_ms_started,
            http_poll_count: Arc::new(RwLock::new(0)),
            ws_trade_count: Arc::new(RwLock::new(0)),
            finished: Arc::new(RwLock::new(false)),
        });

        // Observability: старт трекера со всеми параметрами заявки. Полезно для трассировки
        // «один POST → один колбэк», особенно при параллельных ордерах и реплее по логу.
        crate::test_tee_println!(
            "[order_invoke/start] order_id={order_id} side={side:?} role={role:?} \
             asset_id={asset_id} target={target:?} expiration_unix_ms={expiration_unix_ms:?} \
             market_end_unix_ms={market_end_unix_ms:?} deadline_ms={deadline_ms} \
             started_at_ms={timestamp_ms_started}",
        );

        aggregator
    }

    fn bump_debounce_finalize(aggregator: Arc<Self>) {
        tokio::spawn(async move {
            // Settlement-aware fast-path: фаирим без debounce только если условие гейта уже
            // выполнено (settled покрывает таргет, либо cancel + settled догнал book-match,
            // либо дедлайн). Никакого «быстрого» finalize по book-match без on-chain settlement —
            // см. [`Self::should_invoke`].
            let timestamp_ms_initial = crate::util::current_timestamp_ms();
            let ready_now = {
                let state = aggregator.inner.read().await;
                Self::should_invoke(&state, timestamp_ms_initial)
            };
            if ready_now {
                Self::try_finalize_locked(aggregator).await;
                return;
            }
            // Taker / market-end → пытаемся finalize, но `try_finalize_locked` всё равно не
            // выстрелит без settlement (или дедлайна). Это полезно как «опросная точка», когда
            // должен сработать deadline-branch.
            if matches!(aggregator.role, OrderRole::Taker)
                || aggregator
                    .market_end_unix_ms
                    .is_some_and(|end_ms| end_ms <= timestamp_ms_initial)
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

    /// Учёт одного user-WS `trade`-события. Дедуплицирует по `trade_id`:
    /// - первый раз: добавляет в `filled_ws` (book-match);
    /// - первый раз с `is_settled_on_chain=true`: добавляет в `settled_ws`.
    ///
    /// Значения **NET of fee** (см. [`apply_fee_to_taking_side`]): fee_rate_bps удерживается с
    /// taking-стороны (BUY → меньше shares; SELL → меньше USDC) — это то, что реально движется
    /// на чейне и попадает в [`SingleOrderClobInvocationReport`].
    ///
    /// Дедуп безопасен для лифсайкла одного трейда: `MATCHED → MINED → CONFIRMED` приходят как
    /// отдельные сообщения и без дедупа дали бы x2/x3 счёт.
    async fn record_trade_aggregate_from_ws_event(
        self: &Arc<Self>,
        trade_id: &str,
        size: f64,
        quote: f64,
        fee_rate_bps: f64,
        is_settled_on_chain: bool,
    ) {
        if !size.is_finite() || size <= 0.0 || !quote.is_finite() || quote < 0.0 {
            return;
        }
        let trade_id = trade_id.to_string();
        let mut state_changed = false;
        {
            let mut state = self.inner.write().await;
            let fee_factor = fee_factor_from_bps(fee_rate_bps);
            let (size_net, quote_net) =
                apply_fee_to_taking_side(state.side, size, quote, fee_factor);
            if !size_net.is_finite() || !quote_net.is_finite() {
                return;
            }
            if trade_id.is_empty() {
                // Без trade_id дедуплицировать не можем. Чтобы не плодить тройной счёт по
                // лифсайклу — игнорируем не-settled и считаем settled только если он есть.
                // Это безопасный no-op для аномальных событий.
                if is_settled_on_chain {
                    state.filled_ws = leg_agg_add_trade_fill(state.filled_ws, size_net, quote_net);
                    state.settled_ws =
                        leg_agg_add_trade_fill(state.settled_ws, size_net, quote_net);
                    state_changed = true;
                }
            } else {
                if state.seen_ws_trade_ids.insert(trade_id.clone()) {
                    state.filled_ws = leg_agg_add_trade_fill(state.filled_ws, size_net, quote_net);
                    state_changed = true;
                }
                if is_settled_on_chain && state.settled_seen_ws_trade_ids.insert(trade_id) {
                    state.settled_ws =
                        leg_agg_add_trade_fill(state.settled_ws, size_net, quote_net);
                    state_changed = true;
                }
            }
        }
        if state_changed {
            Self::bump_debounce_finalize(Arc::clone(self));
        }
    }

    async fn record_ws_order_status(self: &Arc<Self>, status_raw: &str) {
        let normalized_status = status_raw.to_ascii_uppercase();
        {
            let mut state = self.inner.write().await;
            // Любой book-level терминал — снимает «ждать ещё fill'ов» от gate'a finalize.
            // Maker `MATCHED` тут — это полностью съеденный лимит-ордер; `CANCELED` — отмена;
            // `FILLED` — синоним полностью сматченного на стороне CLOB.
            if matches!(
                normalized_status.as_str(),
                "MATCHED" | "FILLED" | "CANCELED"
            ) {
                state.book_terminal_reached = true;
            }
            if normalized_status == "CANCELED" {
                state.partial = true;
            }
            if matches!(normalized_status.as_str(), "MATCHED" | "FILLED") {
                state.success = true;
            }
        }
        Self::bump_debounce_finalize(Arc::clone(self));
    }

    async fn record_poll_http(
        self: &Arc<Self>,
        open_order: OpenOrderResponse,
        trades: Option<Vec<TradeResponse>>,
    ) {
        {
            let mut state: tokio::sync::RwLockWriteGuard<'_, InvokeAggInner> =
                self.inner.write().await;
            // `apply_polled_snapshot` сам выставит `book_terminal_reached` для
            // `Matched|Canceled|Unmatched` — дублируем здесь только информационные флаги.
            apply_polled_snapshot(&mut state, &open_order, trades.as_deref());
            if matches!(&open_order.status, OrderStatusType::Canceled) {
                state.partial = true;
            }
            if matches!(&open_order.status, OrderStatusType::Matched) {
                state.success = true;
            }
        }
        Self::bump_debounce_finalize(Arc::clone(self));
    }

    /// Book-matched эффективный leg — `max(filled_ws, filled_http)`. **Информационный**:
    /// в финальный отчёт идут settled-суммы, см. [`Self::effective_settled_leg`].
    fn effective_leg(state: &InvokeAggInner) -> LegAgg {
        leg_agg_max_normalized(state.filled_ws, state.filled_http)
    }

    /// On-chain settled эффективный leg — `max(settled_ws, settled_http)`. Это правда о
    /// зачисленных средствах: только эту цифру сообщает [`SingleOrderClobInvocationReport`].
    fn effective_settled_leg(state: &InvokeAggInner) -> LegAgg {
        leg_agg_max_normalized(state.settled_ws, state.settled_http)
    }

    fn settled_target_progress(state: &InvokeAggInner) -> OrderAmount {
        let settled_leg = Self::effective_settled_leg(state);
        target_dimension_fill_from_leg(state.target, settled_leg)
    }

    fn target_amount_meets(target: &OrderAmount, progress: &OrderAmount) -> bool {
        match (target, progress) {
            (OrderAmount::Shares(target_shares), OrderAmount::Shares(effective_shares)) => {
                target_shares.is_finite()
                    && *target_shares > 0.0
                    && *effective_shares + SHARE_EPS >= *target_shares
            }
            (OrderAmount::UsdNotional(target_usdc), OrderAmount::UsdNotional(effective_usdc)) => {
                target_usdc.is_finite()
                    && *target_usdc > 0.0
                    && *effective_usdc + USD_EPS >= *target_usdc
            }
            _ => false,
        }
    }

    fn settled_targets_met(state: &InvokeAggInner) -> bool {
        Self::target_amount_meets(&state.target, &Self::settled_target_progress(state))
    }

    /// `true` если settled-leg догнал book-matched leg (по обеим осям). Используется для
    /// finalize cancel-сценариев: после `Canceled` делать вид, что больше fills не будет, но
    /// ждать settlement уже состоявшегося book-match'а.
    fn settlement_caught_up_with_match(state: &InvokeAggInner) -> bool {
        let book_leg = Self::effective_leg(state);
        let settled_leg = Self::effective_settled_leg(state);
        let book_shares = order_amount_shares_scalar(book_leg.taking_amount);
        let settled_shares = order_amount_shares_scalar(settled_leg.taking_amount);
        let book_usd = order_amount_usd_scalar(book_leg.making_amount);
        let settled_usd = order_amount_usd_scalar(settled_leg.making_amount);
        settled_shares + SHARE_EPS >= book_shares && settled_usd + USD_EPS >= book_usd
    }

    /// Ready-to-finalize, если выполнено **строгое** settlement-условие:
    /// 1. settlement покрыл целевой объём (`success=true` в отчёте), **или**
    /// 2. book-level терминал заявки достигнут ([`InvokeAggInner::book_terminal_reached`])
    ///    **и** settlement догнал book-match по обеим осям — больше fill'ов не будет
    ///    (Taker FAK сразу после settlement, Maker — после `MATCHED|CANCELED|UNMATCHED`), **или**
    /// 3. дедлайн (best-effort: отчёт по факту settled, диагностика в `error_msg`).
    fn should_invoke(state: &InvokeAggInner, timestamp_ms: i64) -> bool {
        if Self::settled_targets_met(state) {
            return true;
        }
        if state.book_terminal_reached && Self::settlement_caught_up_with_match(state) {
            return true;
        }
        timestamp_ms >= state.deadline_ms
    }

    fn build_report(state: &InvokeAggInner, timestamp_ms: i64) -> SingleOrderClobInvocationReport {
        // Отчёт всегда по settled-leg — это правда о зачисленных средствах (NET of fee).
        // Book-matched (`effective_leg`) используем только для диагностики `settlement_timeout`.
        let settled_leg = Self::effective_settled_leg(state);
        let book_leg = Self::effective_leg(state);
        let (making_amount, taking_amount) =
            report_making_and_taking_amounts(state.side, settled_leg);

        let settled_shares = order_amount_shares_scalar(settled_leg.taking_amount);
        let settled_usd = order_amount_usd_scalar(settled_leg.making_amount);
        let book_shares = order_amount_shares_scalar(book_leg.taking_amount);
        let book_usd = order_amount_usd_scalar(book_leg.making_amount);

        let has_settled_fill = settled_shares > SHARE_EPS || settled_usd > USD_EPS;
        let has_book_fill = book_shares > SHARE_EPS || book_usd > USD_EPS;
        let settled_target_reached = Self::settled_targets_met(state);
        let deadline_hit = timestamp_ms >= state.deadline_ms;

        let report_success = has_settled_fill;
        let report_partial = report_success && !settled_target_reached;

        let error_msg = if has_settled_fill {
            // Что-то реально зачислено on-chain.
            if settled_target_reached {
                None
            } else if has_book_fill
                && (book_shares > settled_shares + SHARE_EPS || book_usd > settled_usd + USD_EPS)
                && deadline_hit
            {
                Some(format!(
                    "partial_settlement: book matched shares={book_shares:.6} usdc={book_usd:.6}, \
                     settled shares={settled_shares:.6} usdc={settled_usd:.6} within \
                     {INVOKE_FALLBACK_POLL_DEADLINE_SEC}s deadline"
                ))
            } else {
                None
            }
        } else if has_book_fill {
            // Сматчили в книге, но settlement не дошёл (Mined/Confirmed не пришли) —
            // либо дедлайн, либо Retrying/Failed на чейне.
            Some(format!(
                "settlement_timeout: matched book shares={book_shares:.6} usdc={book_usd:.6} \
                 but 0 settled on-chain (deadline_hit={deadline_hit}, \
                 book_terminal={}, canceled={}) within {INVOKE_FALLBACK_POLL_DEADLINE_SEC}s",
                state.book_terminal_reached, state.partial
            ))
        } else if state.book_terminal_reached {
            // Book-уровень закрыт (Unmatched/Canceled без fill'а) — ждать больше нечего.
            Some(format!(
                "book_terminal_no_fill: order reached terminal book status with 0 matches \
                 (canceled={})",
                state.partial
            ))
        } else if deadline_hit {
            Some(format!(
                "no_fill_no_settlement: order neither matched nor settled within \
                 {INVOKE_FALLBACK_POLL_DEADLINE_SEC}s deadline"
            ))
        } else {
            // Сюда теоретически не должны приходить: `should_invoke` бы вернул false.
            None
        };

        SingleOrderClobInvocationReport {
            order_id: None,
            making_amount,
            taking_amount,
            success: report_success,
            partial: report_partial,
            error_msg,
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

        let (report, committed_order_id, summary_at_fire, deadline_ms_for_log) = {
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
            let summary = leg_summary_for_log(&state);
            let deadline = state.deadline_ms;
            (invocation_report, cloned_order_id, summary, deadline)
        };

        let _ = self.trackers.write().await.remove(&committed_order_id);

        let elapsed_ms = crate::util::current_timestamp_ms() - self.started_at_ms;
        let http_polls = *self.http_poll_count.read().await;
        let ws_trades = *self.ws_trade_count.read().await;

        // Финальный лог: всё, что вызывающий и аудит увидят в `SingleOrderClobInvocationReport`,
        // плюс счётчики событий и итоговые агрегаты обеих ног.
        crate::test_tee_println!(
            "[order_invoke/final] order_id={committed_order_id} elapsed_ms={elapsed_ms} \
             http_polls={http_polls} ws_trades={ws_trades} side={side:?} role={role:?} \
             target={target:?} deadline_ms={deadline_ms_for_log} | \
             success={success} partial={partial} making={making:?} taking={taking:?} \
             error_msg={error_msg:?} | {summary_at_fire}",
            side = self.side,
            role = self.role,
            target = self.target,
            success = report.success,
            partial = report.partial,
            making = report.making_amount,
            taking = report.taking_amount,
            error_msg = report.error_msg,
        );

        // Replay-инструкция: набросок того, как вручную сверить трейды на CLOB V2 и on-chain.
        // CLOB `/trades` требует L2-auth (`POLY-API-KEY/PASSPHRASE/SIGNATURE/TIMESTAMP`),
        // поэтому простой curl без подписи не сработает — даём минимум для SDK-replay'a
        // и публичные ссылки на explorer для уже эмитнутых трейдов (`tx_hash` поля).
        crate::test_tee_println!(
            "[order_invoke/replay] order_id={committed_order_id} asset_id={asset_id} \
             side={side:?} | для ручной сверки повторите `auth_client.trades(\
             TradesRequest::builder().id(\"{committed_order_id}\").build(), None)` \
             (см. `polymarket_client_sdk::clob::Client::trades`); on-chain трейды по \
             `tx_hash` из `[order_invoke/poll/*]`",
            asset_id = self.asset_id,
            side = self.side,
        );

        self.slot.fire(report);
    }
}

async fn take_tracker_entry(
    trackers: &Arc<RwLock<HashMap<String, TrackerEntry>>>,
    order_id: &str,
) -> Option<TrackerEntry> {
    trackers.write().await.remove(order_id)
}

/// Накапливает исполнение по **`order_id`** из user-WS `trade`-события.
/// `trade_id` — уникальный id трейда (для дедупа лифсайкла `MATCHED → MINED → CONFIRMED`,
/// которые могут прийти как несколько событий с одним `id`). `fee_rate_bps` — per-trade
/// fee из user-WS (`trade.fee_rate_bps`); удерживается с **taking-стороны** заявки
/// (BUY → меньше shares; SELL → меньше USDC). `is_settled_on_chain` —
/// `true` для `MINED|CONFIRMED` (см. [`ws_trade_status_settled_on_chain`]).
pub(crate) async fn accumulate_invoke_from_ws_trade(
    trackers: &Arc<RwLock<HashMap<String, TrackerEntry>>>,
    order_id: &str,
    trade_id: &str,
    size: f64,
    price: f64,
    fee_rate_bps: f64,
    is_settled_on_chain: bool,
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
        // Observability: WS прилетел до регистрации трекера или после finalize — это норма для
        // чужих ордеров, но при разборе кейса полезно знать, что событие отброшено.
        crate::test_tee_println!(
            "[order_invoke/ws] dropped (no tracker): order_id={order_id} trade_id={trade_id} \
             size={size} price={price} fee_bps={fee_rate_bps} settled={is_settled_on_chain}",
        );
        return;
    };
    let invoke_aggregator_arc = Arc::clone(&tracker_entry.invoke_aggregator);
    drop(trackers_snapshot);

    // Текущие агрегаты ДО учёта события — нужно для лог-вывода «как агрегировались».
    let snapshot_before = {
        let state = invoke_aggregator_arc.inner.read().await;
        leg_summary_for_log(&state)
    };

    invoke_aggregator_arc
        .record_trade_aggregate_from_ws_event(
            trade_id,
            size,
            quote,
            fee_rate_bps,
            is_settled_on_chain,
        )
        .await;

    // Инкремент + чтение под одной write-секцией: счётчик публикуется в логе вместе с
    // монотонно-валидным значением (no read-after-write race).
    let ws_count_after = {
        let mut guard = invoke_aggregator_arc.ws_trade_count.write().await;
        *guard += 1;
        *guard
    };

    // Текущие агрегаты ПОСЛЕ учёта — показываем дельту в логе.
    let snapshot_after = {
        let state = invoke_aggregator_arc.inner.read().await;
        leg_summary_for_log(&state)
    };
    crate::test_tee_println!(
        "[order_invoke/ws] order_id={order_id} trade_id={trade_id} size={size:.6} \
         price={price:.6} fee_bps={fee_rate_bps:.3} settled={is_settled_on_chain} \
         → ws_count={ws_count_after} | {snapshot_before} → {snapshot_after}",
    );
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

#[derive(Debug, Clone)]
pub struct PostOrderHttpOutcome {
    pub order_id: String,
    pub success: bool,
    pub status: OrderStatusType,
    pub detail: Value,
    pub invoke_ctx: Option<PostOrderInvokeContext>,
    pub error_msg: Option<String>,
}

fn spawn_invoke_poll_fallback(
    account: SharedAccount,
    order_id: String,
    aggregator: Arc<PostOrderInvokeAggregator>,
) {
    tokio::spawn(async move {
        crate::test_tee_println!(
            "[order_invoke/poll/spawn] order_id={order_id} interval_ms={INVOKE_FALLBACK_POLL_MS} \
             trades_pages_max={TRADES_MAX_PAGES_PER_POLL} \
             http_timeout_s={ORDER_HTTP_POLL_TIMEOUT_SEC} \
             deadline_s={INVOKE_FALLBACK_POLL_DEADLINE_SEC}",
        );
        loop {
            if *aggregator.finished.read().await {
                return;
            }
            let timestamp_ms = crate::util::current_timestamp_ms();
            let deadline_ms = aggregator.inner.read().await.deadline_ms;
            if timestamp_ms >= deadline_ms {
                crate::test_tee_println!(
                    "[order_invoke/poll/deadline] order_id={order_id} \
                     ts={timestamp_ms} deadline_ms={deadline_ms} → bump finalize",
                );
                PostOrderInvokeAggregator::bump_debounce_finalize(Arc::clone(&aggregator));
                return;
            }

            tokio::time::sleep(Duration::from_millis(INVOKE_FALLBACK_POLL_MS)).await;

            if *aggregator.finished.read().await {
                return;
            }

            let auth_client = match (**account.clob_authed.load()).clone() {
                Some(client) => client,
                None => {
                    crate::test_tee_eprintln!(
                        "[order_invoke/poll/skip] order_id={order_id} clob_authed=None — \
                         пропуск итерации, ждём heartbeat re-auth",
                    );
                    continue;
                }
            };

            let iter_idx = {
                let mut guard = aggregator.http_poll_count.write().await;
                *guard += 1;
                *guard
            };

            let order_t0 = std::time::Instant::now();
            let polled_order = match tokio::time::timeout(
                Duration::from_secs(ORDER_HTTP_POLL_TIMEOUT_SEC),
                auth_client.order(&order_id),
            )
            .await
            {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => {
                    crate::test_tee_eprintln!(
                        "[order_invoke/poll/{iter_idx}] GET /order order_id={order_id} \
                         error: {error:#}",
                    );
                    continue;
                }
                Err(_) => {
                    crate::test_tee_eprintln!(
                        "[order_invoke/poll/{iter_idx}] GET /order order_id={order_id} \
                         timeout > {ORDER_HTTP_POLL_TIMEOUT_SEC}s",
                    );
                    continue;
                }
            };
            let order_elapsed_ms = order_t0.elapsed().as_millis();
            let order_status_str = format!("{:?}", &polled_order.status);
            let size_matched_log = decimal_snap_f64(&polled_order.size_matched).unwrap_or(0.0);
            let price_log = decimal_snap_f64(&polled_order.price).unwrap_or(0.0);

            // Дёргаем trades только когда есть что фетчить (`size_matched > 0`); до первого матча
            // taker-Delayed остаётся одним запросом на тик — не плодим лишний трафик.
            let size_matched_positive =
                decimal_snap_f64(&polled_order.size_matched).is_some_and(|s| s > 0.0);
            let trades_t0 = std::time::Instant::now();
            let polled_trades: Option<Vec<TradeResponse>> = if size_matched_positive {
                let trades_request = TradesRequest::builder().id(order_id.clone()).build();
                let mut collected: Vec<TradeResponse> = Vec::new();
                let mut cursor: Option<String> = None;
                let mut pages_ok = true;
                let mut pages_fetched = 0u32;
                for _ in 0..TRADES_MAX_PAGES_PER_POLL {
                    match tokio::time::timeout(
                        Duration::from_secs(ORDER_HTTP_POLL_TIMEOUT_SEC),
                        auth_client.trades(&trades_request, cursor.clone()),
                    )
                    .await
                    {
                        Ok(Ok(page)) => {
                            pages_fetched += 1;
                            collected.extend(page.data);
                            if page.next_cursor.is_empty()
                                || page.next_cursor == TRADES_PAGE_TERMINAL_CURSOR
                            {
                                break;
                            }
                            cursor = Some(page.next_cursor);
                        }
                        Ok(Err(error)) => {
                            crate::test_tee_eprintln!(
                                "[order_invoke/poll/{iter_idx}] GET /trades?id={order_id} \
                                 page={pages_fetched} error: {error:#}",
                            );
                            pages_ok = false;
                            break;
                        }
                        Err(_) => {
                            crate::test_tee_eprintln!(
                                "[order_invoke/poll/{iter_idx}] GET /trades?id={order_id} \
                                 page={pages_fetched} timeout > {ORDER_HTTP_POLL_TIMEOUT_SEC}s",
                            );
                            pages_ok = false;
                            break;
                        }
                    }
                }
                let _ = pages_fetched;
                pages_ok.then_some(collected)
            } else {
                None
            };
            let trades_elapsed_ms = trades_t0.elapsed().as_millis();

            // Лог per-trade полей: воспроизводит curl-результат `GET /trades?id=<order_id>`.
            // Содержит всё, что нужно для ручной сверки on-chain (tx_hash) и фискал-аудита.
            let trades_log = match polled_trades.as_deref() {
                Some([]) => "trades=[] (empty)".to_string(),
                Some(ts) => {
                    let mut buf = format!("trades=[{} items]:\n", ts.len());
                    for t in ts {
                        let size = decimal_snap_f64(&t.size).unwrap_or(0.0);
                        let price = decimal_snap_f64(&t.price).unwrap_or(0.0);
                        let fee_bps = decimal_snap_f64(&t.fee_rate_bps).unwrap_or(0.0);
                        let tx_hash = format!("{:#x}", t.transaction_hash);
                        buf.push_str(&format!(
                            "    - id={} status={:?} size={:.6} price={:.6} fee_bps={:.3} \
                             tx_hash={} match_time={} last_update={} maker={} taker_order_id={}\n",
                            t.id,
                            t.status,
                            size,
                            price,
                            fee_bps,
                            tx_hash,
                            t.match_time.timestamp(),
                            t.last_update.timestamp(),
                            t.maker_address,
                            t.taker_order_id,
                        ));
                    }
                    buf.trim_end().to_string()
                }
                None => {
                    if size_matched_positive {
                        "trades=ERROR (fetch failed, see prior log)".to_string()
                    } else {
                        "trades=SKIPPED (size_matched=0)".to_string()
                    }
                }
            };

            crate::test_tee_println!(
                "[order_invoke/poll/{iter_idx}] order_id={order_id} \
                 GET /order → status={order_status_str} size_matched={size_matched_log:.6} \
                 price={price_log:.6} ({order_elapsed_ms}ms); GET /trades → \
                 ({trades_elapsed_ms}ms) {trades_log}",
            );

            // Снимок ДО агрегации.
            let snapshot_before = {
                let state = aggregator.inner.read().await;
                leg_summary_for_log(&state)
            };
            aggregator
                .record_poll_http(polled_order, polled_trades)
                .await;
            let snapshot_after = {
                let state = aggregator.inner.read().await;
                leg_summary_for_log(&state)
            };
            crate::test_tee_println!(
                "[order_invoke/poll/{iter_idx}/agg] order_id={order_id} | {snapshot_before} → {snapshot_after}",
            );

            // Раньше тут был ранний `return` на `Matched|Canceled` (`OrderStatusType` book-level).
            // Теперь поллим до фактического finalize (`finished=true`) или дедлайна:
            // book-match без settlement — это не финал; settlement проявится в следующих poll'ах
            // как `TradeStatusType::Mined|Confirmed`.
            if *aggregator.finished.read().await {
                return;
            }
        }
    });
}

/// После HTTP POST:
/// - ошибка / `success=false` — колбэк-провал сразу (через [`spawn_fire_invocation_report`]);
/// - иначе **всегда** спауним [`spawn_invoke_poll_fallback`], чтобы дождаться on-chain settlement
///   (`TradeStatusType::Mined|Confirmed`). Book-level `Matched` от сервера НЕ короткозамыкает
///   finalize — релайер исполняет ERC-1155 трансфер асинхронно и наш контракт колбэка
///   ([`SingleOrderInvocation`]) гарантирует, что `success=true` фаирится **только** после факта
///   зачисления, см. [`PostOrderInvokeAggregator::should_invoke`].
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
        let server_error = http_result.error_msg.clone().or_else(|| {
            Some(format!(
                "server returned success=false, status={:?}",
                http_result.status
            ))
        });
        crate::test_tee_println!(
            "[order_invoke/early-fail] order_id={cloned_order_id} status={:?} side={:?} \
             server_success=false → fire `success=false` immediately (no tracker registered) | \
             error_msg={server_error:?}",
            http_result.status,
            side_for_zero_fill,
        );
        let (making_z, taking_z) = zero_making_taking_for_side(side_for_zero_fill);
        spawn_fire_invocation_report(
            &slot,
            SingleOrderClobInvocationReport {
                order_id: nonempty_order_id_str(&cloned_order_id),
                making_amount: making_z,
                taking_amount: taking_z,
                success: false,
                partial: false,
                error_msg: server_error,
            },
        );
        let _ = take_tracker_entry(&trackers, &cloned_order_id).await;
        return;
    }

    let Some(invoke_context) = http_result.invoke_ctx.clone() else {
        let (making_z, taking_z) = zero_fill_without_request_context();
        spawn_fire_invocation_report(
            &slot,
            SingleOrderClobInvocationReport {
                order_id: nonempty_order_id_str(&cloned_order_id),
                making_amount: making_z,
                taking_amount: taking_z,
                success: false,
                partial: false,
                error_msg: Some(
                    "after_post_order_maybe_track_invoke: invoke_ctx=None при success=true (defensive)"
                        .to_string(),
                ),
            },
        );
        return;
    };
    let posted_order_request = invoke_context.request;
    let making_amount = invoke_context.making_amount;
    let taking_amount = invoke_context.taking_amount;
    if cloned_order_id.is_empty() {
        let (making_z, taking_z) = zero_making_taking_for_side(posted_order_request.side);
        spawn_fire_invocation_report(
            &slot,
            SingleOrderClobInvocationReport {
                order_id: None,
                making_amount: making_z,
                taking_amount: taking_z,
                success: false,
                partial: false,
                error_msg: Some("CLOB вернул пустой order_id при success=true".to_string()),
            },
        );
        return;
    }
    let invoke_aggregator = PostOrderInvokeAggregator::new(
        Arc::clone(&slot),
        Arc::clone(&trackers),
        cloned_order_id.clone(),
        posted_order_request,
    );

    {
        let mut trackers_write_guard = trackers.write().await;
        trackers_write_guard.insert(
            cloned_order_id.clone(),
            TrackerEntry {
                invoke_aggregator: Arc::clone(&invoke_aggregator),
            },
        );
    }

    // Сидим [`InvokeAggInner::filled_http`] (book-matched, **gross** — POST не вычитает fee)
    // из тела `POST /order` для **любого** статуса. На `settled_http` это **не** влияет —
    // POST-ответ говорит только про book-match, релайер ещё не отработал. При первом успешном
    // REST-poll'е `apply_polled_snapshot` **перезапишет** `filled_http` на NET-агрегат фактических
    // `client.trades(...)` (см. `trades_cover_matched` ветку) — gross-сид «выместится» NET'ом.
    //
    // Также выставляем `book_terminal_reached`:
    // - **Taker** — всегда сразу: taker не остаётся в книге, любой POST-ответ — это финальное
    //   состояние book-уровня (Matched/Unmatched/Canceled — больше fill'ов не будет).
    // - **Maker** — только для терминальных POST-статусов (`Matched|Canceled|Unmatched`);
    //   `Live|Delayed` оставляем без флага — ордер живёт в книге, ждём fill'ы.
    let order_role = invoke_aggregator.role;
    {
        let mut invoke_state = invoke_aggregator.inner.write().await;
        let mut leg = match invoke_state.side {
            Side::Buy => LegAgg {
                making_amount,
                taking_amount,
            },
            Side::Sell => LegAgg {
                making_amount: taking_amount,
                taking_amount: making_amount,
            },
            _ => LegAgg {
                making_amount,
                taking_amount,
            },
        };
        leg.sanitize_mut();
        invoke_state.filled_http = leg_agg_max_normalized(invoke_state.filled_http, leg);
        let post_status_is_book_terminal = matches!(
            http_result.status,
            OrderStatusType::Matched | OrderStatusType::Canceled | OrderStatusType::Unmatched
        );
        if matches!(order_role, OrderRole::Taker) || post_status_is_book_terminal {
            invoke_state.book_terminal_reached = true;
        }
        if matches!(http_result.status, OrderStatusType::Canceled) {
            invoke_state.partial = true;
        }
        if matches!(http_result.status, OrderStatusType::Matched) {
            invoke_state.success = true;
        }
    }

    // Любой не-провальный исход → нужен on-chain settlement; для **всех** статусов спауним poll.
    // - `Matched`: ждём `TradeStatusType::Mined|Confirmed` в `client.trades(...)`.
    // - `Live` / `Delayed` / `Unmatched`: ждём как fill'ы, так и settlement.
    // - `Canceled`: ждём settlement уже произошедших до отмены fill'ов; на
    //   `book_terminal_reached + settled догнал book-match` finalize (см.
    //   [`PostOrderInvokeAggregator::should_invoke`]).
    PostOrderInvokeAggregator::bump_debounce_finalize(Arc::clone(&invoke_aggregator));
    spawn_invoke_poll_fallback(Arc::clone(account), cloned_order_id, invoke_aggregator);
}

pub(crate) fn wrap_post_order_cb(
    invoke: SingleOrderInvokeCb,
) -> Arc<CompletionOnce<SingleOrderClobInvocationReport>> {
    Arc::new(CompletionOnce::new(invoke))
}
