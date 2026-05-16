//! POST + `invoke`: один колбэк один раз — **только после явного терминала от CLOB** (HTTP/WS).
//! Никаких timestamp-фоллбэков: агрегатор ждёт сигналов столько, сколько нужно.
//!
//! Источник истины — [`polymarket_client_sdk::clob::types::TradeStatusType`] лифсайкла трейда:
//! `Matched` (только в книге CLOB) → `Mined` (включён в блок) → `Confirmed` (после финализации).
//! Зачислением считаем `Mined|Confirmed` (см. [`PostOrderInvokeAggregator`]). Состояния
//! `Retrying` контрибьютят в book-match агрегат (`filled_*`), но **не** в settlement-агрегат
//! (`settled_*`); ждём, пока CLOB пере-эмитит этот трейд как `Mined|Confirmed` либо `Failed`.
//! Терминальный `Failed` (релайер сдался) ведёт отдельный учёт (`failed_*`): он **не** прибавляется
//! к зачисленным средствам (on-chain ничего не произошло), но засчитывается как «терминальный
//! объём» в гейте finalize — иначе при race «book CANCELED + 1 трейд застрял на чейне как
//! Failed» агрегатор бы ждал вечно (см. [`PostOrderInvokeAggregator::settlement_caught_up_with_match`]).
//!
//! Финал = `success=true` фаирится, как только выполнено хотя бы одно из:
//! 1. **`max(settled_ws, settled_http)`** покрывает целевой объём заявки
//!    [`crate::account_order::PostOrderRequest::amount`] (full target reached);
//! 2. достигнут book-level терминал ([`InvokeAggInner::book_terminal_reached`]) — для Taker
//!    это всегда сразу после `POST /order`, для Maker — статус `Matched|Canceled|Unmatched`
//!    из POST/REST/WS — **и** settlement догнал book-match по обеим осям (нечего больше ждать).
//!
//! Никакого «timestamp-deadline» finalize нет: maker-`expiration` (GTD) обрабатывает сам CLOB и
//! пришлёт нам `Canceled|Unmatched` через WS/HTTP — этот сигнал и есть наш терминал. Если CLOB
//! не отвечает (сеть/баг), агрегатор ждёт; вызывающий код должен таймаутить сам, если нужно.
//!
//! Источники сигнала:
//! - **HTTP**: REST-poll каждые [`INVOKE_FALLBACK_POLL_MS`] дёргает `client.order(order_id)` +
//!   (если `size_matched > 0`) `client.trades(order_id)` и партиционирует трейды на
//!   book-matched (всё) и settled (`status ∈ {Mined, Confirmed}`). Поллинг крутится до тех пор,
//!   пока [`PostOrderInvokeAggregator::finished`] не станет `true`.
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
/// Дастр-толерантность по shares **только** для финального
/// [`SingleOrderClobInvocationReport::partial`] (и связанной `error_msg`-диагностики):
/// 0.01 = один CLOB-lot. На уровне CLOB Polymarket помечает ордер `OrderStatusType::Matched`
/// и снимает с книги остаток ниже лота (например, при `original_size=5.0`
/// и `size_matched=4.995078` оставшиеся 0.004922 sh уже не сматчатся — биржа их обнуляет).
/// Без этого допуска такие «полные» исполнения на уровне книги попадали бы в `partial=true`
/// из-за строгого [`SHARE_EPS`]; пользователь же видит «вся нога продана». На гейт finalize
/// ([`PostOrderInvokeAggregator::should_invoke`] / [`PostOrderInvokeAggregator::settled_targets_met`])
/// он не влияет — там по-прежнему [`SHARE_EPS`], чтобы taker FAK с настоящим partial-fill
/// дождался on-chain settlement и сообщил честную цифру через terminal-ветку.
const SHARES_REPORT_FULL_FILL_DUST_TOLERANCE: f64 = 0.01;
/// USDC-аналог [`SHARES_REPORT_FULL_FILL_DUST_TOLERANCE`]: 1 цент. Используется ТОЛЬКО для
/// `partial`/`error_msg` в финальном отчёте, не для гейта finalize.
const USDC_REPORT_FULL_FILL_DUST_TOLERANCE: f64 = 0.01;

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
///    book-match по обеим осям (нечего больше ждать).
///
/// Никаких timestamp-фоллбэков нет: агрегатор ждёт явных HTTP/WS сигналов сколько нужно. Для
/// Taker FAK это даёт колбэк в пределах settlement-задержки релайера (~1–3s в норме). Для
/// Maker GTD `expiration` обрабатывает CLOB и эмитит `Canceled` через WS/HTTP — это и есть
/// наш терминал. Если CLOB по какой-то причине не отвечает (сеть/баг), вызывающий код
/// должен таймаутить сам.
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
    /// `true`, только если `success=true` и цель [`PostOrderRequest::amount`] не достигнута
    /// полностью **с учётом дастр-допуска в один CLOB-lot**: при `target=Shares(N)` нужно
    /// `settled_shares + 0.01 < N`, при `target=UsdNotional(U)` — `settled_usdc + 0.01 < U`.
    /// Без этого допуска полностью исполнённый maker, у которого Polymarket снял sub-lot
    /// остаток ниже `min_order_size` и пометил книгу `OrderStatusType::Matched` (типичный кейс
    /// `Shares(5.0) → settled Shares(4.995078)`), ошибочно попадал бы в `partial=true`,
    /// тогда как пользователь видит «вся нога продана».
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
    /// Terminal-failed накопление по user-WS `trade` (только `FAILED` — релайер сдался,
    /// on-chain ничего не зачислилось). Дедуплицируется по `trade_id`
    /// ([`Self::failed_seen_ws_trade_ids`]). **Не** прибавляется к `success`-цифрам в отчёте,
    /// но засчитывается как «терминальный объём» в [`PostOrderInvokeAggregator::settlement_caught_up_with_match`],
    /// чтобы `filled = settled + failed` гарантированно покрывалось — иначе при race
    /// «book CANCELED + один трейд застрял в `Failed`» агрегатор зависнет.
    failed_ws: LegAgg,
    /// Terminal-failed HTTP-снимок (max-merge с агрегатом REST-`trades` со статусом `Failed`).
    /// Аналогично [`Self::failed_ws`]: «релайер сдался, on-chain ничего не зачислилось» —
    /// контрибьютит только в гейт finalize и в диагностику `error_msg`, не в `success`.
    failed_http: LegAgg,
    /// `trade_id` уже учтённые в `filled_ws` — защита от тройного счёта при WS-лифсайкле
    /// одного трейда (`MATCHED`→`MINED`→`CONFIRMED` могут прийти как 1..3 события).
    seen_ws_trade_ids: HashSet<String>,
    /// `trade_id` уже учтённые в `settled_ws` — отдельный сет, потому что settle-первая
    /// контрибьюция — это первый `MINED|CONFIRMED` после возможного предыдущего `MATCHED`.
    settled_seen_ws_trade_ids: HashSet<String>,
    /// `trade_id` уже учтённые в `failed_ws` — гарантирует ровно один учёт `FAILED`-перехода
    /// для одного трейда (лифсайкл `MATCHED → RETRYING → FAILED` присылает несколько событий).
    failed_seen_ws_trade_ids: HashSet<String>,
    /// [`Side`] ордера (копия из [`crate::account_order::PostOrderRequest::side`]) — управляет
    /// трактовкой HTTP `making`/`taking` при seed и применением fee к нужной стороне (taking).
    side: Side,
    /// Book-level терминал заявки: больше fill'ов не ожидается. Ставится:
    /// - **сразу** в [`after_post_order_maybe_track_invoke`] для любого Taker (taker не остаётся
    ///   в книге) — отсюда taker FAK с partial fill финализируется на settlement;
    /// - в [`Self::record_poll_http`] / [`Self::record_ws_order_status`] для статусов CLOB
    ///   `Canceled|Unmatched` (POST/REST) либо `CANCELED|UNMATCHED` (WS) — безусловно (ордер
    ///   ушёл из книги, новых матчей не будет; для maker GTD это ровно то, что приходит после
    ///   `expiration` — CLOB сам снимает ордер и эмитит `Canceled`);
    /// - для CLOB `Matched`/`MATCHED|FILLED` (Maker) — **только** когда
    ///   [`is_book_fully_matched_observed`] = `true`, т.е. наблюдаемый `size_matched` покрыл
    ///   `original_size` с дастр-допуском. Polymarket шлёт `MATCHED` после **каждого** трейда
    ///   maker'а, поэтому без этой проверки колбэк мог бы выстрелить прематурно (settlement
    ///   догнал book по первому трейду — а матч ещё мог продолжаться).
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
    /// Наблюдаемый `original_size` ордера на стороне CLOB. Сидится в [`PostOrderInvokeAggregator::new`]
    /// из `target` (`OrderAmount::Shares` — иначе `None`; такие заявки — только Taker, и для
    /// них book-terminal выставляется по другому каналу — Taker FAK не остаётся в книге).
    /// Дальше max-merge'ится из [`OpenOrderResponse::original_size`] (poll) и из
    /// `OrderMessage.original_size` (WS user-channel). Используется как делитель в
    /// [`is_book_fully_matched_observed`].
    original_size_observed: Option<f64>,
    /// Максимум наблюдаемого `size_matched` ордера на стороне CLOB: max-merge из
    /// [`OpenOrderResponse::size_matched`] (poll), `OrderMessage.size_matched` (WS user-channel)
    /// и making/taking-сидов POST-ответа (см. [`after_post_order_maybe_track_invoke`]).
    /// Монотонно растёт. В паре с [`Self::original_size_observed`] определяет
    /// [`is_book_fully_matched_observed`].
    size_matched_observed: f64,
}

fn decimal_snap_f64(d: &Decimal) -> Option<f64> {
    let f = d.to_string().parse::<f64>().ok()?;
    f.is_finite().then_some(f)
}

/// `true` если по наблюдаемым `(original_size, size_matched)` ордер сматчен на book-уровне
/// «целиком» (с дастр-допуском в один CLOB-lot — Polymarket сам снимает sub-lot остаток ниже
/// `min_order_size`). Используется как обязательный гейт перевода CLOB-статуса `MATCHED`/`FILLED`
/// в [`InvokeAggInner::book_terminal_reached`]: Polymarket шлёт `MATCHED` после каждого трейда
/// maker'а, и без этой проверки колбэк мог бы выстрелить ПРЕМАТУРНО — например, после первого
/// `MATCHED` event'а у maker'а 100 sh, когда сматчено лишь 30 sh, а оставшиеся 70 sh ещё могли
/// бы дальше матчиться при движении цены к нашему лимиту.
fn is_book_fully_matched_observed(
    original_size_observed: Option<f64>,
    size_matched_observed: f64,
) -> bool {
    match original_size_observed {
        Some(orig) if orig.is_finite() && orig > 0.0 => {
            size_matched_observed.is_finite()
                && size_matched_observed + SHARES_REPORT_FULL_FILL_DUST_TOLERANCE >= orig
        }
        _ => false,
    }
}

/// Max-merge для [`InvokeAggInner::original_size_observed`]. Защищает от случайных
/// нулей/мусора в `original_size` поле WS/poll-event'а и обеспечивает монотонность.
fn update_original_size_observed(state: &mut InvokeAggInner, observed: f64) {
    if !observed.is_finite() || observed <= 0.0 {
        return;
    }
    state.original_size_observed = Some(
        state
            .original_size_observed
            .map(|prev| prev.max(observed))
            .unwrap_or(observed),
    );
}

/// Max-merge для [`InvokeAggInner::size_matched_observed`]. Монотонно растёт.
fn update_size_matched_observed(state: &mut InvokeAggInner, observed: f64) {
    if !observed.is_finite() || observed < 0.0 {
        return;
    }
    if observed > state.size_matched_observed {
        state.size_matched_observed = observed;
    }
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
    let failed_shares = order_amount_shares_scalar(
        leg_agg_max_normalized(state.failed_ws, state.failed_http).taking_amount,
    );
    let failed_usd = order_amount_usd_scalar(
        leg_agg_max_normalized(state.failed_ws, state.failed_http).making_amount,
    );
    let f_ws_sh = order_amount_shares_scalar(state.filled_ws.taking_amount);
    let f_ws_us = order_amount_usd_scalar(state.filled_ws.making_amount);
    let f_ht_sh = order_amount_shares_scalar(state.filled_http.taking_amount);
    let f_ht_us = order_amount_usd_scalar(state.filled_http.making_amount);
    let s_ws_sh = order_amount_shares_scalar(state.settled_ws.taking_amount);
    let s_ws_us = order_amount_usd_scalar(state.settled_ws.making_amount);
    let s_ht_sh = order_amount_shares_scalar(state.settled_http.taking_amount);
    let s_ht_us = order_amount_usd_scalar(state.settled_http.making_amount);
    let fa_ws_sh = order_amount_shares_scalar(state.failed_ws.taking_amount);
    let fa_ws_us = order_amount_usd_scalar(state.failed_ws.making_amount);
    let fa_ht_sh = order_amount_shares_scalar(state.failed_http.taking_amount);
    let fa_ht_us = order_amount_usd_scalar(state.failed_http.making_amount);
    format!(
        "book={book_shares:.6}sh/{book_usd:.6}$ (ws={f_ws_sh:.6}/{f_ws_us:.6}, \
         http={f_ht_sh:.6}/{f_ht_us:.6}) settled={settled_shares:.6}sh/{settled_usd:.6}$ \
         (ws={s_ws_sh:.6}/{s_ws_us:.6}, http={s_ht_sh:.6}/{s_ht_us:.6}) \
         failed={failed_shares:.6}sh/{failed_usd:.6}$ \
         (ws={fa_ws_sh:.6}/{fa_ws_us:.6}, http={fa_ht_sh:.6}/{fa_ht_us:.6}) \
         orig={:?} matched={:.6} fully={} term={} part={} succ={}",
        state.original_size_observed,
        state.size_matched_observed,
        is_book_fully_matched_observed(
            state.original_size_observed,
            state.size_matched_observed,
        ),
        state.book_terminal_reached,
        state.partial,
        state.success,
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

/// Сырой WS `trade.status` — нужно ли прокидывать событие в invoke-агрегатор для учёта в
/// book-match ноге (`filled_*`): любой этап лифсайкла, по которому Polymarket шлёт trade-event
/// (`MATCHED` … `FAILED`). Дедуп по `trade_id` в [`PostOrderInvokeAggregator`] гарантирует один
/// счёт на трейд; см. [`ws_trade_status_settled_on_chain`], [`ws_trade_status_terminal_failed`].
#[inline]
pub(crate) fn ws_trade_status_for_invoke_book_match(status_raw: &str) -> bool {
    matches!(
        status_raw.to_ascii_uppercase().as_str(),
        "MATCHED" | "MINED" | "CONFIRMED" | "RETRYING" | "FAILED"
    )
}

/// `true` если трейд **терминально провалился** на чейне: релайер сдался, on-chain ничего не
/// зачислилось и больше попыток не будет. См. [`TradeStatusType::Failed`]. `Retrying` — это
/// **не** terminal: релайер ещё пробует, переход может быть в `Mined` либо `Failed`.
#[inline]
pub(crate) fn trade_status_terminal_failed(status: &TradeStatusType) -> bool {
    matches!(status, TradeStatusType::Failed)
}

/// То же по сырой строке статуса WS user-channel `trade.status`.
#[inline]
pub(crate) fn ws_trade_status_terminal_failed(status_raw: &str) -> bool {
    status_raw.eq_ignore_ascii_case("FAILED")
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
/// Также обновляет [`InvokeAggInner::original_size_observed`] и
/// [`InvokeAggInner::size_matched_observed`] (max-merge) и выставляет
/// [`InvokeAggInner::book_terminal_reached`] по правилу:
/// - `Canceled` / `Unmatched` — безусловно (ордер ушёл из книги, новых матчей не будет).
/// - `Matched` — **только** если [`is_book_fully_matched_observed`] = `true`, т.е. наблюдаемый
///   `size_matched` покрыл `original_size` с дастр-допуском. Polymarket помечает книгу
///   `Matched` после каждого трейда (см. `OrderStatusType::Matched` в SDK — это «order has
///   been matched», не «fully matched»), поэтому без этой проверки колбэк мог бы выстрелить
///   прематурно для partial maker'а, ещё способного матчиться дальше.
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

    // Observed-поля для book-fully-matched гейта. `original_size` — поле `OpenOrderResponse`,
    // должно быть стабильным; max-merge просто страхует от случайных нулей при сетевых сбоях.
    if let Some(orig) = decimal_snap_f64(&open.original_size) {
        update_original_size_observed(inner, orig);
    }
    update_size_matched_observed(inner, size_matched);

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

            // Failed-only ветка: трейды, по которым релайер сдался — on-chain пусто, но они
            // занимают «слот» в book-match'е, поэтому без их учёта `settled_caught_up_with_match`
            // навсегда `false` и агрегатор зависает. Fee применяется идентично (gross→net через
            // `apply_fee_to_taking_side`), чтобы сравнение `settled + failed >= filled` оставалось
            // консистентным в одной размерности с `filled_*` (BUY=NET shares, SELL=NET USDC).
            let failed_iter = ts
                .iter()
                .filter(|trade| trade_status_terminal_failed(&trade.status));
            let failed_leg = aggregate_trades_into_leg(side, failed_iter);
            inner.failed_http = leg_agg_max_normalized(inner.failed_http, failed_leg);
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

    match &open.status {
        // Ордер ушёл из книги — гарантированно больше не сматчится.
        OrderStatusType::Canceled | OrderStatusType::Unmatched => {
            inner.book_terminal_reached = true;
        }
        // `Matched` в Polymarket — «по ордеру был хотя бы один матч», НЕ «сматчен полностью».
        // Терминал ставим, только когда `size_matched` действительно покрыл `original_size`
        // (с дастр-допуском в один CLOB-lot — sub-lot остаток биржа снимает сама). Иначе ждём
        // дальнейших трейдов или явного `Canceled`/`Unmatched`.
        OrderStatusType::Matched => {
            if is_book_fully_matched_observed(
                inner.original_size_observed,
                inner.size_matched_observed,
            ) {
                inner.book_terminal_reached = true;
            }
        }
        _ => {}
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
    role: OrderRole,
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
        let target = post_request.amount;
        let side = post_request.side;
        let role = post_request.role;
        let asset_id = post_request.asset_id.clone();
        // Поля только для observability — никакой timestamp-фоллбэк finalize не использует:
        // maker GTD `expiration` обрабатывает CLOB и пришлёт `Canceled|Unmatched` через
        // WS/HTTP — это и есть наш терминал.
        let expiration_unix_ms = post_request.expiration.map(|e| e.timestamp_millis());
        let market_end_unix_ms = post_request.market_end_unix_ms;
        // Сид `original_size_observed` из target'a: для Shares-target (всегда у Maker и часто
        // у Taker SELL) знаем сразу. Для UsdNotional-target (Taker BUY) `original_size` в shares
        // на CLOB определяется post-factum через `taking_amount` POST-ответа — для таких заявок
        // book-terminal проставляется по другому каналу (Taker FAK не остаётся в книге).
        let original_size_observed = match target {
            OrderAmount::Shares(s) if s.is_finite() && s > 0.0 => Some(s),
            _ => None,
        };

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
                failed_ws: LegAgg::default(),
                failed_http: LegAgg::default(),
                seen_ws_trade_ids: HashSet::new(),
                settled_seen_ws_trade_ids: HashSet::new(),
                failed_seen_ws_trade_ids: HashSet::new(),
                side,
                book_terminal_reached: false,
                success: false,
                partial: false,
                original_size_observed,
                size_matched_observed: 0.0,
            })),
            role,
            target,
            side,
            asset_id: asset_id.clone(),
            started_at_ms: timestamp_ms_started,
            http_poll_count: Arc::new(RwLock::new(0)),
            ws_trade_count: Arc::new(RwLock::new(0)),
            finished: Arc::new(RwLock::new(false)),
        });

        crate::test_tee_println!(
            "[order_invoke/start] order_id={order_id} side={side:?} role={role:?} \
             asset_id={asset_id} target={target:?} expiration_unix_ms={expiration_unix_ms:?} \
             market_end_unix_ms={market_end_unix_ms:?} started_at_ms={timestamp_ms_started}",
        );

        aggregator
    }

    /// Спаунит попытку финализации на отдельной таске (чтобы не выполнять `try_finalize_locked` /
    /// и тем более callback в стэке вызывающего, в т.ч. внутри `post_order_on_clob`).
    /// `try_finalize_locked` сам проверит [`Self::should_invoke`] и фаирнет колбэк ровно один
    /// раз через `finished` flag — параллельные вызовы из разных событий безопасны.
    ///
    /// Раньше тут было debounce-окно `INVOKE_DEBOUNCE_MS` для накопления близких событий до
    /// единого finalize-attempt'а; теперь, когда `should_invoke` детерминирован и срабатывает
    /// только на явных монотонных терминалах от CLOB, окно бессмысленно — финал и так
    /// выстреливает на первом ready-событии.
    fn schedule_finalize_attempt(aggregator: Arc<Self>) {
        tokio::spawn(async move {
            Self::try_finalize_locked(aggregator).await;
        });
    }

    /// Учёт одного user-WS `trade`-события. Дедуплицирует по `trade_id`:
    /// - первый раз: добавляет в `filled_ws` (book-match);
    /// - первый раз с `is_settled_on_chain=true`: добавляет в `settled_ws`;
    /// - первый раз с `is_terminal_failed=true`: добавляет в `failed_ws`.
    ///
    /// `is_settled_on_chain` и `is_terminal_failed` — взаимоисключающие исходы (`Mined|Confirmed`
    /// vs `Failed`); защитимся проверкой xor, но даже при «обоих true» дедуп-сеты не пересекаются.
    ///
    /// Значения **NET of fee** (см. [`apply_fee_to_taking_side`]): fee_rate_bps удерживается с
    /// taking-стороны (BUY → меньше shares; SELL → меньше USDC) — это то, что реально движется
    /// на чейне и попадает в [`SingleOrderClobInvocationReport`]. Для `Failed` on-chain ничего
    /// не движется (fee нулевая де-факто), но мы применяем тот же fee_factor, чтобы failed_leg
    /// шёл в той же размерности, что filled_leg (важно для `settlement_caught_up_with_match`).
    ///
    /// Дедуп безопасен для лифсайкла одного трейда: `MATCHED → RETRYING → MINED → CONFIRMED`
    /// (или `… → FAILED`) приходят как отдельные сообщения и без дедупа дали бы x2/x3 счёт.
    async fn record_trade_aggregate_from_ws_event(
        self: &Arc<Self>,
        trade_id: &str,
        size: f64,
        quote: f64,
        fee_rate_bps: f64,
        is_settled_on_chain: bool,
        is_terminal_failed: bool,
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
                // лифсайклу — игнорируем не-терминальные статусы; для терминальных (settled или
                // failed) контрибьютим в book + соответствующую settlement-ось ровно один раз.
                // Это безопасный no-op для аномальных событий.
                if is_settled_on_chain {
                    state.filled_ws = leg_agg_add_trade_fill(state.filled_ws, size_net, quote_net);
                    state.settled_ws =
                        leg_agg_add_trade_fill(state.settled_ws, size_net, quote_net);
                    state_changed = true;
                } else if is_terminal_failed {
                    state.filled_ws = leg_agg_add_trade_fill(state.filled_ws, size_net, quote_net);
                    state.failed_ws = leg_agg_add_trade_fill(state.failed_ws, size_net, quote_net);
                    state_changed = true;
                }
            } else {
                if state.seen_ws_trade_ids.insert(trade_id.clone()) {
                    state.filled_ws = leg_agg_add_trade_fill(state.filled_ws, size_net, quote_net);
                    state_changed = true;
                }
                if is_settled_on_chain
                    && state.settled_seen_ws_trade_ids.insert(trade_id.clone())
                {
                    state.settled_ws =
                        leg_agg_add_trade_fill(state.settled_ws, size_net, quote_net);
                    state_changed = true;
                }
                if is_terminal_failed && state.failed_seen_ws_trade_ids.insert(trade_id) {
                    state.failed_ws = leg_agg_add_trade_fill(state.failed_ws, size_net, quote_net);
                    state_changed = true;
                }
            }
        }
        if state_changed {
            Self::schedule_finalize_attempt(Arc::clone(self));
        }
    }

    /// Применяет WS user-channel `order`-event: подтягивает `original_size`/`size_matched`
    /// (если есть) в observed-поля и выставляет terminal-флаги по правилу:
    /// - `CANCELED` — book-terminal безусловно, плюс `partial=true`.
    /// - `UNMATCHED` — book-terminal безусловно (FAK без ликвидности / expired GTD).
    /// - `MATCHED`/`FILLED` — `success=true` всегда (информационно), book-terminal **только**
    ///   когда [`is_book_fully_matched_observed`] = `true`, т.к. Polymarket шлёт `MATCHED` после
    ///   каждого трейда maker'а, и без этой проверки колбэк мог бы выстрелить прематурно для
    ///   partial maker'а, ещё способного матчиться дальше.
    /// - Прочие (`LIVE`, `INVALID`, etc) — только обновление observed-полей, terminal не ставим;
    ///   ждём дальнейших WS/HTTP-событий (timestamp-фоллбэка нет).
    async fn record_ws_order_status(
        self: &Arc<Self>,
        status_raw: &str,
        original_size_hint: Option<f64>,
        size_matched_hint: Option<f64>,
    ) {
        let normalized_status = status_raw.to_ascii_uppercase();
        {
            let mut state = self.inner.write().await;
            if let Some(orig) = original_size_hint {
                update_original_size_observed(&mut state, orig);
            }
            if let Some(matched) = size_matched_hint {
                update_size_matched_observed(&mut state, matched);
            }
            let book_fully_matched = is_book_fully_matched_observed(
                state.original_size_observed,
                state.size_matched_observed,
            );
            match normalized_status.as_str() {
                "CANCELED" => {
                    state.book_terminal_reached = true;
                    state.partial = true;
                }
                "UNMATCHED" => {
                    state.book_terminal_reached = true;
                }
                "MATCHED" | "FILLED" => {
                    state.success = true;
                    if book_fully_matched {
                        state.book_terminal_reached = true;
                    }
                }
                _ => {}
            }
        }
        Self::schedule_finalize_attempt(Arc::clone(self));
    }

    async fn record_poll_http(
        self: &Arc<Self>,
        open_order: OpenOrderResponse,
        trades: Option<Vec<TradeResponse>>,
    ) {
        {
            let mut state: tokio::sync::RwLockWriteGuard<'_, InvokeAggInner> =
                self.inner.write().await;
            // `apply_polled_snapshot` сам обновит observed-поля и выставит
            // `book_terminal_reached` (для `Canceled|Unmatched` — безусловно, для `Matched` —
            // только при `is_book_fully_matched_observed`); здесь дублируем информационные флаги.
            apply_polled_snapshot(&mut state, &open_order, trades.as_deref());
            if matches!(&open_order.status, OrderStatusType::Canceled) {
                state.partial = true;
            }
            if matches!(&open_order.status, OrderStatusType::Matched) {
                state.success = true;
            }
        }
        Self::schedule_finalize_attempt(Arc::clone(self));
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

    /// Terminal-failed эффективный leg — `max(failed_ws, failed_http)`. Это объём book-match'а,
    /// по которому релайер сдался: on-chain ничего не зачислилось и **не зачислится**. В
    /// [`Self::build_report`] **не** прибавляется к `success`-цифрам (`making_amount`/
    /// `taking_amount`), но засчитывается в [`Self::settlement_caught_up_with_match`] как
    /// «терминальный объём» — иначе при race «book CANCELED + один трейд застрял `Failed`»
    /// агрегатор бы ждал вечно (settled навсегда меньше filled).
    fn effective_failed_leg(state: &InvokeAggInner) -> LegAgg {
        leg_agg_max_normalized(state.failed_ws, state.failed_http)
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

    /// Тоже самое, что [`Self::target_amount_meets`], но с допуском «дастр» в один CLOB-lot
    /// ([`SHARES_REPORT_FULL_FILL_DUST_TOLERANCE`] / [`USDC_REPORT_FULL_FILL_DUST_TOLERANCE`]).
    /// Применяется **только** для решения `partial`/`error_msg` в [`Self::build_report`] —
    /// гейт finalize ([`Self::should_invoke`] / [`Self::settled_targets_met`]) остаётся строгим.
    /// Цель: не помечать ордер как `partial=true`, когда CLOB **сам** маркирует исполнение
    /// `Matched` и обнуляет sub-lot остаток (типичный случай: `target=Shares(5.0)`,
    /// `settled=Shares(4.995078)` — пользователь видит «вся нога продана»).
    fn settled_meets_target_with_report_dust(state: &InvokeAggInner) -> bool {
        let progress = Self::settled_target_progress(state);
        match (&state.target, &progress) {
            (OrderAmount::Shares(target_shares), OrderAmount::Shares(effective_shares)) => {
                target_shares.is_finite()
                    && *target_shares > 0.0
                    && *effective_shares + SHARES_REPORT_FULL_FILL_DUST_TOLERANCE
                        >= *target_shares
            }
            (OrderAmount::UsdNotional(target_usdc), OrderAmount::UsdNotional(effective_usdc)) => {
                target_usdc.is_finite()
                    && *target_usdc > 0.0
                    && *effective_usdc + USDC_REPORT_FULL_FILL_DUST_TOLERANCE >= *target_usdc
            }
            _ => false,
        }
    }

    /// `true` если **терминальный** объём (settled + failed) догнал book-matched leg
    /// (по обеим осям). Используется для finalize cancel-сценариев: после `Canceled` делать
    /// вид, что больше fills не будет, но ждать settlement уже состоявшегося book-match'а.
    ///
    /// **Failed-трейды** учитываются здесь как terminal-объём, потому что для них релайер
    /// сдался — больше изменений по этим `trade_id` не будет ни on-chain, ни в книге. Без этого
    /// при race «`Canceled` пришёл, последний из N трейдов застрял в `Failed`» условие
    /// `settled ≥ filled` оставалось бы навсегда `false`, и агрегатор зависал бы (мы убрали
    /// timestamp-дедлайны, см. модульный комментарий). `success` от учёта Failed не растёт:
    /// в [`Self::build_report`] он считается строго по `effective_settled_leg`.
    ///
    /// Дополнительно защищает от race «WS `order` с `size_matched` пришёл раньше
    /// соответствующих WS `trade` event'ов»: если CLOB утверждает, что сматчено больше, чем
    /// мы успели накопить в `book_leg` через trade-event'ы (или REST `trades`-агрегат), —
    /// settlement по определению ещё не догнал. Без этой проверки finalize выстрелил бы с
    /// пустыми сидами и `success=false`, хотя on-chain трейды на подходе. Допуск — один
    /// CLOB-lot ([`SHARES_REPORT_FULL_FILL_DUST_TOLERANCE`]): для SELL shares-сторона у нас
    /// gross (fee удерживается с USDC), сравнение прямое; для BUY fee на shares-стороне даёт
    /// малое shrinkage, но у Polymarket в большинстве маркетов 0 bps, и lot-допуск его
    /// покрывает.
    fn settlement_caught_up_with_match(state: &InvokeAggInner) -> bool {
        let book_leg = Self::effective_leg(state);
        let settled_leg = Self::effective_settled_leg(state);
        let failed_leg = Self::effective_failed_leg(state);
        let book_shares = order_amount_shares_scalar(book_leg.taking_amount);
        let settled_shares = order_amount_shares_scalar(settled_leg.taking_amount);
        let failed_shares = order_amount_shares_scalar(failed_leg.taking_amount);
        let book_usd = order_amount_usd_scalar(book_leg.making_amount);
        let settled_usd = order_amount_usd_scalar(settled_leg.making_amount);
        let failed_usd = order_amount_usd_scalar(failed_leg.making_amount);

        if state.size_matched_observed > book_shares + SHARES_REPORT_FULL_FILL_DUST_TOLERANCE
        {
            return false;
        }

        let terminal_shares = settled_shares + failed_shares;
        let terminal_usd = settled_usd + failed_usd;
        terminal_shares + SHARE_EPS >= book_shares && terminal_usd + USD_EPS >= book_usd
    }

    /// Ready-to-finalize **строго** по событиям от CLOB — никаких timestamp-фоллбэков:
    /// 1. settlement покрыл целевой объём (`success=true` в отчёте), **или**
    /// 2. book-level терминал заявки достигнут ([`InvokeAggInner::book_terminal_reached`])
    ///    **и** terminal-объём (`settled + failed`) догнал book-match по обеим осям — больше
    ///    изменений не будет:
    ///    - Taker — сразу после settlement (FAK не остаётся в книге).
    ///    - Maker `CANCELED`/`UNMATCHED` — безусловно (ордер ушёл из книги; для GTD это и есть
    ///      событие после `expiration` — CLOB сам снимает ордер).
    ///    - Maker `MATCHED`/`FILLED` — **только** если
    ///      [`is_book_fully_matched_observed`] = `true` (т.е. наблюдаемый `size_matched`
    ///      покрыл `original_size` с дастр-допуском). Это защищает partial maker'а от
    ///      прематурного finalize, когда Polymarket уже прислал `MATCHED` после первого
    ///      трейда, а матч ещё может продолжаться (см. [`InvokeAggInner::book_terminal_reached`]).
    ///
    /// `Failed`-трейды (релайер сдался) считаются терминальным объёмом наравне с `Mined|Confirmed`
    /// в гейте `(2)` — см. [`Self::settlement_caught_up_with_match`] — иначе при race
    /// «`CANCELED` пришёл, последний из N трейдов застрял в `Failed`» агрегатор бы зависал.
    /// При этом `Failed` **не** прибавляется к `success`-цифрам отчёта (там только реально
    /// зачисленное on-chain).
    fn should_invoke(state: &InvokeAggInner) -> bool {
        if Self::settled_targets_met(state) {
            return true;
        }
        state.book_terminal_reached && Self::settlement_caught_up_with_match(state)
    }

    fn build_report(state: &InvokeAggInner) -> SingleOrderClobInvocationReport {
        // Отчёт всегда по settled-leg — это правда о зачисленных средствах (NET of fee).
        // Book-matched (`effective_leg`) и failed-leg (`effective_failed_leg`) используем
        // только для диагностики (`partial_settlement` / `relayer_failed`); в numerical
        // `making_amount`/`taking_amount` они **не** входят: пользователь видит ровно то,
        // что реально упало в его кошелёк on-chain.
        let settled_leg = Self::effective_settled_leg(state);
        let book_leg = Self::effective_leg(state);
        let failed_leg = Self::effective_failed_leg(state);
        let (making_amount, taking_amount) =
            report_making_and_taking_amounts(state.side, settled_leg);

        let settled_shares = order_amount_shares_scalar(settled_leg.taking_amount);
        let settled_usd = order_amount_usd_scalar(settled_leg.making_amount);
        let book_shares = order_amount_shares_scalar(book_leg.taking_amount);
        let book_usd = order_amount_usd_scalar(book_leg.making_amount);
        let failed_shares = order_amount_shares_scalar(failed_leg.taking_amount);
        let failed_usd = order_amount_usd_scalar(failed_leg.making_amount);

        let has_settled_fill = settled_shares > SHARE_EPS || settled_usd > USD_EPS;
        let has_book_fill = book_shares > SHARE_EPS || book_usd > USD_EPS;
        let has_failed_terminal = failed_shares > SHARE_EPS || failed_usd > USD_EPS;
        // Для решения `partial` используем **дастр**-допуск: Polymarket снимает sub-lot остаток
        // при `OrderStatusType::Matched` (book-terminal), и пользовательски это «вся нога
        // продана». Гейт finalize этим не затрагиваем — там по-прежнему строгий
        // [`Self::settled_targets_met`] (см. [`Self::should_invoke`]).
        let settled_target_reached_with_dust = Self::settled_meets_target_with_report_dust(state);

        let report_success = has_settled_fill;
        let report_partial = report_success && !settled_target_reached_with_dust;

        let error_msg = if has_settled_fill {
            // Что-то реально зачислено on-chain. Сначала чекаем terminal-failed — это самый
            // громкий диагностический сигнал: были book-match'и, по которым релайер сдался.
            if has_failed_terminal {
                Some(format!(
                    "relayer_failed: settled shares={settled_shares:.6} usdc={settled_usd:.6}, \
                     failed_on_relayer shares={failed_shares:.6} usdc={failed_usd:.6} \
                     (book_matched shares={book_shares:.6} usdc={book_usd:.6})"
                ))
            } else if settled_target_reached_with_dust {
                None
            } else if has_book_fill
                && state.book_terminal_reached
                && (book_shares > settled_shares + SHARE_EPS || book_usd > settled_usd + USD_EPS)
            {
                // Книга закрыта, но on-chain зачислилось меньше, чем сматчилось, и failed
                // ничего не объясняет — значит часть трейдов ещё `Retrying` (не terminal).
                // Гейт `should_invoke` сюда привести не должен; диагностика на случай гонок.
                Some(format!(
                    "partial_settlement: book matched shares={book_shares:.6} usdc={book_usd:.6}, \
                     settled shares={settled_shares:.6} usdc={settled_usd:.6}"
                ))
            } else {
                None
            }
        } else if state.book_terminal_reached {
            // Book-уровень закрыт, on-chain settle = 0. Если has_failed_terminal — все трейды
            // (или единственные имевшиеся) провалились на релайере. Иначе — либо book-fill
            // ещё `Retrying|Matched`, либо `Unmatched|Canceled` вовсе без fill'ов.
            if has_failed_terminal {
                Some(format!(
                    "relayer_failed_all: book matched shares={book_shares:.6} usdc={book_usd:.6}, \
                     all failed_on_relayer shares={failed_shares:.6} usdc={failed_usd:.6} \
                     (canceled={})",
                    state.partial
                ))
            } else if has_book_fill {
                Some(format!(
                    "book_terminal_no_settle: book matched shares={book_shares:.6} \
                     usdc={book_usd:.6} but 0 settled on-chain (book_terminal=true, \
                     canceled={})",
                    state.partial
                ))
            } else {
                Some(format!(
                    "book_terminal_no_fill: order reached terminal book status with 0 matches \
                     (canceled={})",
                    state.partial
                ))
            }
        } else {
            // Сюда теоретически не должны приходить: `should_invoke` бы вернул false без
            // settled fill'а и без book-terminal'а.
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

        let ready_to_invoke = {
            let state = self.inner.read().await;
            Self::should_invoke(&state)
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

        let (report, committed_order_id, summary_at_fire) = {
            let state = self.inner.read().await;
            if !Self::should_invoke(&state) {
                {
                    let mut finished_guard = self.finished.write().await;
                    *finished_guard = false;
                }
                Self::schedule_finalize_attempt(Arc::clone(&self));
                return;
            }
            let cloned_order_id = self.order_id.clone();
            let mut invocation_report = Self::build_report(&state);
            invocation_report.order_id = nonempty_order_id_str(&cloned_order_id);
            let summary = leg_summary_for_log(&state);
            (invocation_report, cloned_order_id, summary)
        };

        let _ = self.trackers.write().await.remove(&committed_order_id);

        let elapsed_ms = crate::util::current_timestamp_ms() - self.started_at_ms;
        let http_polls = *self.http_poll_count.read().await;
        let ws_trades = *self.ws_trade_count.read().await;

        crate::test_tee_println!(
            "[order_invoke/final] order_id={committed_order_id} elapsed_ms={elapsed_ms} \
             http_polls={http_polls} ws_trades={ws_trades} side={side:?} role={role:?} \
             target={target:?} | success={success} partial={partial} making={making:?} \
             taking={taking:?} error_msg={error_msg:?} | {summary_at_fire}",
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
/// `trade_id` — уникальный id трейда (для дедупа лифсайкла
/// `MATCHED → RETRYING → MINED → CONFIRMED` или `… → FAILED`, которые могут прийти
/// как несколько событий с одним `id`). `fee_rate_bps` — per-trade fee из user-WS
/// (`trade.fee_rate_bps`); удерживается с **taking-стороны** заявки (BUY → меньше shares;
/// SELL → меньше USDC). `is_settled_on_chain` — `true` для `MINED|CONFIRMED`
/// (см. [`ws_trade_status_settled_on_chain`]). `is_terminal_failed` — `true` для `FAILED`
/// (см. [`ws_trade_status_terminal_failed`]): релайер сдался, on-chain ничего не зачислится.
pub(crate) async fn accumulate_invoke_from_ws_trade(
    trackers: &Arc<RwLock<HashMap<String, TrackerEntry>>>,
    order_id: &str,
    trade_id: &str,
    size: f64,
    price: f64,
    fee_rate_bps: f64,
    is_settled_on_chain: bool,
    is_terminal_failed: bool,
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
             size={size} price={price} fee_bps={fee_rate_bps} settled={is_settled_on_chain} \
             failed={is_terminal_failed}",
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
            is_terminal_failed,
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
         failed={is_terminal_failed} → ws_count={ws_count_after} | {snapshot_before} → {snapshot_after}",
    );
}

/// User-WS `order.status` → флаги терминала invoke-агрегатора.
pub(crate) async fn notify_terminal_ws_order_snapshot(
    trackers: &Arc<RwLock<HashMap<String, TrackerEntry>>>,
    order_id: &str,
    order_status: &str,
    original_size: Option<f64>,
    size_matched: Option<f64>,
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
        .record_ws_order_status(order_status, original_size, size_matched)
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
             (no timestamp deadline; loop until terminal CLOB event)",
        );
        loop {
            if *aggregator.finished.read().await {
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
    // - **Maker** `Canceled|Unmatched` — безусловный терминал (ордер ушёл из книги).
    // - **Maker** `Matched` — терминал **только** если book-match по POST покрыл `original_size`
    //   (с дастр-допуском в один CLOB-lot). Polymarket помечает книгу `Matched` после **каждого**
    //   матча, поэтому без этой проверки колбэк мог бы выстрелить прематурно у partial maker'а,
    //   ещё способного матчиться дальше. Если PostOrderResponse-сид показывает неполный матч,
    //   ждём дальнейших трейдов через poll/WS, либо явного `Canceled`/`Unmatched`.
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

        // Сидим `size_matched_observed` из gross-shares в POST-ответе: после `LegAgg`-свопа
        // `leg.taking_amount` всегда лежит в `OrderAmount::Shares` (`making`=USDC, `taking`=Shares
        // по нашей внутренней конвенции). Это позволяет диагностировать полный мгновенный
        // maker-матч уже на POST-ответе без ожидания первого poll'а (~500ms экономии latency).
        if let OrderAmount::Shares(s) = leg.taking_amount {
            update_size_matched_observed(&mut invoke_state, s);
        }

        let post_status_is_sure_terminal = matches!(
            http_result.status,
            OrderStatusType::Canceled | OrderStatusType::Unmatched
        );
        let post_status_matched_fully = matches!(http_result.status, OrderStatusType::Matched)
            && is_book_fully_matched_observed(
                invoke_state.original_size_observed,
                invoke_state.size_matched_observed,
            );
        if matches!(order_role, OrderRole::Taker)
            || post_status_is_sure_terminal
            || post_status_matched_fully
        {
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
    PostOrderInvokeAggregator::schedule_finalize_attempt(Arc::clone(&invoke_aggregator));
    spawn_invoke_poll_fallback(Arc::clone(account), cloned_order_id, invoke_aggregator);
}

pub(crate) fn wrap_post_order_cb(
    invoke: SingleOrderInvokeCb,
) -> Arc<CompletionOnce<SingleOrderClobInvocationReport>> {
    Arc::new(CompletionOnce::new(invoke))
}
