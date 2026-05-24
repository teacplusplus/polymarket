//! Mock-completion для [`crate::account_mock_order::post_order_on_clob`]: фейковая
//! симуляция исполнения ордера по WS-снапшоту
//! ([`crate::project_manager::ProjectManager::last_snapshot_by_asset_id`])
//! без выхода в сеть и без CLOB. Колбэк
//! [`crate::account_order_completion::SingleOrderInvokeCb`] фаирится ровно один раз через
//! [`crate::account_order_completion::CompletionOnce`] (как в реальном пути).
//!
//! Сценарии:
//! * **Taker (`OrderRole::Taker`)** — в моменте идёт прохождение книги (BUY ↦ asks, SELL ↦ bids).
//!   Условие срабатывания: всё или ничего в пределах cap-цены (`request.price`) или
//!   `max_slippage_pp` от L1. Применяется «крипто»-fee
//!   [`crate::history_sim::POLYMARKET_CRYPTO_TAKER_FEE_RATE`] с taking-стороны.
//! * **Maker (`OrderRole::Maker`)** — таска ждёт, пока best ask/bid не достигнет лимитной
//!   цены `request.price`; при достижении — фикс полного объёма по лимит-цене **без fee**.
//!   Таймаут — `request.expiration` (если задан). Cancel через
//!   [`crate::account_mock_order::cancel_order_on_clob`] фаирит фейл-репорт.

use crate::account_order::{OrderAmount, OrderRole, PostOrderRequest};
use crate::account_order_completion::{
    CompletionOnce, SingleOrderClobInvocationReport, spawn_fire_invocation_report,
    zero_making_taking_for_side,
};
use crate::history_sim::POLYMARKET_CRYPTO_TAKER_FEE_RATE;
use crate::market_snapshot::MarketSnapshot;
use crate::project_manager::ProjectManager;
use crate::xframe::BookLevel;
use polymarket_client_sdk::clob::types::Side;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, oneshot};

/// Период опроса WS-снапшота в режиме maker (ждём пересечения лимит-цены).
const MOCK_MAKER_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Максимальный возраст WS-снапшота, при котором считаем книгу актуальной для исполнения.
const MOCK_WS_BOOK_MAX_AGE_MS: i64 = 1_000;
/// Допуск для сравнения цен (`f64` к концу floor/clamp может терять биты в последнем знаке).
const PRICE_COMPARE_EPS: f64 = 1e-9;
/// Допуск, чтобы taker не отказывал на цифровом шуме при обходе книги до цели.
const FILL_QUANTITY_EPS: f64 = 1e-9;

/// Реестр cancel-каналов: один процессный singleton, ключ — `order_id`.
fn cancel_registry() -> &'static Mutex<HashMap<String, oneshot::Sender<()>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, oneshot::Sender<()>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) async fn register_mock_order_cancel_channel(
    order_id: &str,
    sender: oneshot::Sender<()>,
) {
    cancel_registry()
        .lock()
        .await
        .insert(order_id.to_string(), sender);
}

/// `true`, если для `order_id` был активный канал и сигнал ушёл.
pub(crate) async fn signal_mock_order_cancel(order_id: &str) -> bool {
    let Some(sender) = cancel_registry().lock().await.remove(order_id) else {
        return false;
    };
    sender.send(()).is_ok()
}

async fn forget_mock_order_cancel_channel(order_id: &str) {
    let _ = cancel_registry().lock().await.remove(order_id);
}

/// Свежий book-снимок из WS (`bids`/`asks` уже отсортированы лучшим уровнем первым).
struct MockBookSnapshot {
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

impl MockBookSnapshot {
    fn from_snapshot(snapshot: &MarketSnapshot) -> Option<Self> {
        let bids = snapshot.book_bids.clone().unwrap_or_default();
        let asks = snapshot.book_asks.clone().unwrap_or_default();
        if bids.is_empty() && asks.is_empty() {
            return None;
        }
        Some(Self { bids, asks })
    }

    fn best_ask_price(&self) -> Option<f64> {
        first_live_level(&self.asks).map(|level| level.price)
    }

    fn best_bid_price(&self) -> Option<f64> {
        first_live_level(&self.bids).map(|level| level.price)
    }
}

fn first_live_level(levels: &[BookLevel]) -> Option<&BookLevel> {
    levels
        .iter()
        .find(|level| level.price > 0.0 && level.size > 0.0)
}

async fn load_recent_mock_book(
    project_manager: &Arc<ProjectManager>,
    asset_id: &str,
    now_ms: i64,
) -> Option<MockBookSnapshot> {
    let guard = project_manager.last_snapshot_by_asset_id.read().await;
    let snapshot = guard.get(asset_id)?;
    if now_ms.saturating_sub(snapshot.timestamp_ms) > MOCK_WS_BOOK_MAX_AGE_MS {
        return None;
    }
    MockBookSnapshot::from_snapshot(snapshot)
}

/// Walk по уровням до набора `target_shares` — возвращает `(vwap, total_usd)` при полном фи.
fn walk_levels_for_shares_target(
    levels: &[BookLevel],
    target_shares: f64,
) -> Option<(f64, f64)> {
    if !target_shares.is_finite() || target_shares <= 0.0 {
        return None;
    }
    let mut remaining_shares = target_shares;
    let mut accumulated_usd = 0.0_f64;
    for level in levels {
        if level.price <= 0.0 || level.size <= 0.0 {
            continue;
        }
        if remaining_shares <= level.size {
            accumulated_usd += remaining_shares * level.price;
            remaining_shares = 0.0;
            break;
        }
        accumulated_usd += level.size * level.price;
        remaining_shares -= level.size;
    }
    if remaining_shares > FILL_QUANTITY_EPS || accumulated_usd <= 0.0 {
        return None;
    }
    let vwap = accumulated_usd / target_shares;
    Some((vwap, accumulated_usd))
}

/// Walk по asks до полного списания `target_usd` — возвращает `(vwap, gross_shares)`.
fn walk_asks_for_usd_target(asks: &[BookLevel], target_usd: f64) -> Option<(f64, f64)> {
    if !target_usd.is_finite() || target_usd <= 0.0 {
        return None;
    }
    let mut remaining_usd = target_usd;
    let mut gross_shares = 0.0_f64;
    for level in asks {
        if level.price <= 0.0 || level.size <= 0.0 {
            continue;
        }
        let level_usd_capacity = level.price * level.size;
        if remaining_usd <= level_usd_capacity {
            gross_shares += remaining_usd / level.price;
            remaining_usd = 0.0;
            break;
        }
        gross_shares += level.size;
        remaining_usd -= level_usd_capacity;
    }
    if remaining_usd > FILL_QUANTITY_EPS || gross_shares <= 0.0 {
        return None;
    }
    let vwap = target_usd / gross_shares;
    Some((vwap, gross_shares))
}

/// Polymarket-style crypto taker fee в USD: `shares × rate × price × (1 - price)`.
fn polymarket_taker_fee_usd(shares: f64, price: f64) -> f64 {
    let bounded_price = price.clamp(0.0, 1.0);
    let safe_shares = shares.max(0.0);
    safe_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * bounded_price * (1.0 - bounded_price)
}

/// `(making, taking, fee_usd)` для taker BUY от USDC-нотанала: `making=UsdNotional(gross)`, `taking=Shares(net)`.
fn taker_buy_legs_from_usd(
    usd_notional: f64,
    gross_shares: f64,
    vwap: f64,
) -> (OrderAmount, OrderAmount, f64) {
    let fee_usd = polymarket_taker_fee_usd(gross_shares, vwap);
    let fee_shares = if vwap > 0.0 { fee_usd / vwap } else { 0.0 };
    let net_shares = (gross_shares - fee_shares).max(0.0);
    (
        OrderAmount::UsdNotional(usd_notional),
        OrderAmount::Shares(net_shares),
        fee_usd,
    )
}

/// `(making, taking, fee_usd)` для taker BUY от целевого числа shares.
fn taker_buy_legs_from_shares(target_shares: f64, vwap: f64) -> (OrderAmount, OrderAmount, f64) {
    let gross_usd = target_shares * vwap;
    let fee_usd = polymarket_taker_fee_usd(target_shares, vwap);
    let net_shares = if vwap > 0.0 {
        (target_shares - fee_usd / vwap).max(0.0)
    } else {
        0.0
    };
    (
        OrderAmount::UsdNotional(gross_usd),
        OrderAmount::Shares(net_shares),
        fee_usd,
    )
}

/// `(making, taking, fee_usd)` для taker SELL: `making=Shares(gross)`, `taking=UsdNotional(net)`.
fn taker_sell_legs(shares: f64, gross_usd: f64, vwap: f64) -> (OrderAmount, OrderAmount, f64) {
    let fee_usd = polymarket_taker_fee_usd(shares, vwap);
    let net_usd = (gross_usd - fee_usd).max(0.0);
    (
        OrderAmount::Shares(shares),
        OrderAmount::UsdNotional(net_usd),
        fee_usd,
    )
}

/// Maker — без fee, ровно `shares × limit_price` USDC. `fee_usd = 0`.
fn maker_legs_no_fee(
    side: Side,
    shares: f64,
    limit_price: f64,
) -> (OrderAmount, OrderAmount, f64) {
    let usd = shares * limit_price;
    let (making, taking) = match side {
        Side::Buy => (OrderAmount::UsdNotional(usd), OrderAmount::Shares(shares)),
        Side::Sell => (OrderAmount::Shares(shares), OrderAmount::UsdNotional(usd)),
        _ => (OrderAmount::UsdNotional(usd), OrderAmount::Shares(shares)),
    };
    (making, taking, 0.0)
}

fn fire_mock_success_report(
    slot: &Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
    order_id: String,
    making_amount: OrderAmount,
    taking_amount: OrderAmount,
    fee_paid_usdc: f64,
) {
    spawn_fire_invocation_report(
        slot,
        SingleOrderClobInvocationReport {
            order_id: Some(order_id),
            making_amount,
            taking_amount,
            success: true,
            partial: false,
            error_msg: None,
            fee_paid_usdc,
        },
    );
}

fn fire_mock_failed_report(
    slot: &Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
    order_id: Option<String>,
    side: Side,
    error_msg: Option<String>,
) {
    let (making_amount, taking_amount) = zero_making_taking_for_side(side);
    spawn_fire_invocation_report(
        slot,
        SingleOrderClobInvocationReport {
            order_id,
            making_amount,
            taking_amount,
            success: false,
            partial: false,
            error_msg,
            fee_paid_usdc: 0.0,
        },
    );
}

/// Спаунит таску, которая «исполнит» mock-ордер и однократно фаирит колбэк.
pub(crate) fn spawn_mock_order_processor(
    project_manager: Arc<ProjectManager>,
    request: PostOrderRequest,
    order_id: String,
    slot: Arc<CompletionOnce<SingleOrderClobInvocationReport>>,
    cancel_rx: oneshot::Receiver<()>,
) {
    tokio::spawn(async move {
        let outcome = match request.role {
            OrderRole::Taker => run_taker_fill(&project_manager, &request, cancel_rx).await,
            OrderRole::Maker => {
                run_maker_wait_for_fill(&project_manager, &request, &order_id, cancel_rx).await
            }
        };

        match outcome {
            MockFillOutcome::Filled {
                making_amount,
                taking_amount,
                fee_paid_usdc,
            } => fire_mock_success_report(
                &slot,
                order_id.clone(),
                making_amount,
                taking_amount,
                fee_paid_usdc,
            ),
            MockFillOutcome::Failed { error_msg } => {
                fire_mock_failed_report(&slot, Some(order_id.clone()), request.side, Some(error_msg));
            }
            MockFillOutcome::Canceled => fire_mock_failed_report(
                &slot,
                Some(order_id.clone()),
                request.side,
                Some("mock_canceled: cancel_order_on_clob".to_string()),
            ),
        }

        forget_mock_order_cancel_channel(&order_id).await;
    });
}

enum MockFillOutcome {
    Filled {
        making_amount: OrderAmount,
        taking_amount: OrderAmount,
        fee_paid_usdc: f64,
    },
    Failed {
        error_msg: String,
    },
    Canceled,
}

async fn run_taker_fill(
    project_manager: &Arc<ProjectManager>,
    request: &PostOrderRequest,
    mut cancel_rx: oneshot::Receiver<()>,
) -> MockFillOutcome {
    if matches!(cancel_rx.try_recv(), Ok(())) {
        return MockFillOutcome::Canceled;
    }

    let now_ms = crate::util::current_timestamp_ms();
    let Some(book) = load_recent_mock_book(project_manager, &request.asset_id, now_ms).await else {
        return MockFillOutcome::Failed {
            error_msg: format!(
                "mock_taker_no_fresh_ws_book: asset_id={} max_age_ms={MOCK_WS_BOOK_MAX_AGE_MS}",
                request.asset_id,
            ),
        };
    };

    match (request.side, request.amount) {
        (Side::Buy, OrderAmount::UsdNotional(usd_notional)) => {
            let Some((vwap, gross_shares)) = walk_asks_for_usd_target(&book.asks, usd_notional)
            else {
                return MockFillOutcome::Failed {
                    error_msg: format!(
                        "mock_taker_buy_no_depth: usd={usd_notional:.4} asset_id={}",
                        request.asset_id,
                    ),
                };
            };
            if let Some(cap_violation) = taker_cap_violation_message(request, &book, vwap) {
                return MockFillOutcome::Failed {
                    error_msg: cap_violation,
                };
            }
            let (making_amount, taking_amount, fee_paid_usdc) =
                taker_buy_legs_from_usd(usd_notional, gross_shares, vwap);
            MockFillOutcome::Filled {
                making_amount,
                taking_amount,
                fee_paid_usdc,
            }
        }
        (Side::Buy, OrderAmount::Shares(target_shares)) => {
            let Some((vwap, _gross_usd)) =
                walk_levels_for_shares_target(&book.asks, target_shares)
            else {
                return MockFillOutcome::Failed {
                    error_msg: format!(
                        "mock_taker_buy_shares_no_depth: shares={target_shares:.4} asset_id={}",
                        request.asset_id,
                    ),
                };
            };
            if let Some(cap_violation) = taker_cap_violation_message(request, &book, vwap) {
                return MockFillOutcome::Failed {
                    error_msg: cap_violation,
                };
            }
            let (making_amount, taking_amount, fee_paid_usdc) =
                taker_buy_legs_from_shares(target_shares, vwap);
            MockFillOutcome::Filled {
                making_amount,
                taking_amount,
                fee_paid_usdc,
            }
        }
        (Side::Sell, OrderAmount::Shares(shares)) => {
            let Some((vwap, gross_usd)) = walk_levels_for_shares_target(&book.bids, shares)
            else {
                return MockFillOutcome::Failed {
                    error_msg: format!(
                        "mock_taker_sell_no_depth: shares={shares:.4} asset_id={}",
                        request.asset_id,
                    ),
                };
            };
            if let Some(cap_violation) = taker_cap_violation_message(request, &book, vwap) {
                return MockFillOutcome::Failed {
                    error_msg: cap_violation,
                };
            }
            let (making_amount, taking_amount, fee_paid_usdc) =
                taker_sell_legs(shares, gross_usd, vwap);
            MockFillOutcome::Filled {
                making_amount,
                taking_amount,
                fee_paid_usdc,
            }
        }
        (Side::Sell, OrderAmount::UsdNotional(_)) => MockFillOutcome::Failed {
            error_msg: "mock_taker_sell_unsupported_amount: SELL ожидает Shares".to_string(),
        },
        (other_side, _) => MockFillOutcome::Failed {
            error_msg: format!("mock_taker_side_unsupported: side={other_side:?}"),
        },
    }
}

/// Проверка cap для taker: либо явный `price` (worst-acceptable), либо `max_slippage_pp` от L1.
fn taker_cap_violation_message(
    request: &PostOrderRequest,
    book: &MockBookSnapshot,
    vwap: f64,
) -> Option<String> {
    if let Some(explicit_cap_price) = request.price {
        return match request.side {
            Side::Buy if vwap > explicit_cap_price + PRICE_COMPARE_EPS => Some(format!(
                "mock_taker_buy_above_cap: vwap={vwap:.6} cap={explicit_cap_price:.6}"
            )),
            Side::Sell if vwap + PRICE_COMPARE_EPS < explicit_cap_price => Some(format!(
                "mock_taker_sell_below_cap: vwap={vwap:.6} cap={explicit_cap_price:.6}"
            )),
            _ => None,
        };
    }
    let slippage_cap = request.max_slippage_pp?;
    match request.side {
        Side::Buy => {
            let best_ask = book.best_ask_price()?;
            (vwap - best_ask > slippage_cap + PRICE_COMPARE_EPS).then(|| {
                format!(
                    "mock_taker_buy_slip_exceeded: vwap={vwap:.6} best_ask={best_ask:.6} \
                     cap_pp={slippage_cap:.6}"
                )
            })
        }
        Side::Sell => {
            let best_bid = book.best_bid_price()?;
            (best_bid - vwap > slippage_cap + PRICE_COMPARE_EPS).then(|| {
                format!(
                    "mock_taker_sell_slip_exceeded: vwap={vwap:.6} best_bid={best_bid:.6} \
                     cap_pp={slippage_cap:.6}"
                )
            })
        }
        _ => None,
    }
}

async fn run_maker_wait_for_fill(
    project_manager: &Arc<ProjectManager>,
    request: &PostOrderRequest,
    order_id: &str,
    mut cancel_rx: oneshot::Receiver<()>,
) -> MockFillOutcome {
    let limit_price = request
        .price
        .expect("maker validated by validate_post_order_request");
    let shares = match request.amount {
        OrderAmount::Shares(s) => s,
        _ => {
            return MockFillOutcome::Failed {
                error_msg: "mock_maker_unsupported_amount: ожидается Shares".to_string(),
            };
        }
    };
    let maker_deadline = request.expiration.map(|expiration_dt| {
        let target_ms = expiration_dt.timestamp_millis();
        let now_ms = crate::util::current_timestamp_ms();
        let remaining_ms = target_ms.saturating_sub(now_ms).max(0) as u64;
        Instant::now() + Duration::from_millis(remaining_ms)
    });

    loop {
        if matches!(cancel_rx.try_recv(), Ok(())) {
            return MockFillOutcome::Canceled;
        }
        if let Some(deadline) = maker_deadline
            && Instant::now() >= deadline
        {
            return MockFillOutcome::Failed {
                error_msg: format!(
                    "mock_maker_expired: order_id={order_id} limit_price={limit_price:.6} \
                     shares={shares:.4}"
                ),
            };
        }

        let now_ms = crate::util::current_timestamp_ms();
        let condition_met = match load_recent_mock_book(
            project_manager,
            &request.asset_id,
            now_ms,
        )
        .await
        {
            Some(book) => match request.side {
                Side::Buy => book
                    .best_ask_price()
                    .is_some_and(|best_ask| best_ask <= limit_price + PRICE_COMPARE_EPS),
                Side::Sell => book
                    .best_bid_price()
                    .is_some_and(|best_bid| best_bid + PRICE_COMPARE_EPS >= limit_price),
                _ => false,
            },
            None => false,
        };
        if condition_met {
            let (making_amount, taking_amount, fee_paid_usdc) =
                maker_legs_no_fee(request.side, shares, limit_price);
            return MockFillOutcome::Filled {
                making_amount,
                taking_amount,
                fee_paid_usdc,
            };
        }

        tokio::time::sleep(MOCK_MAKER_POLL_INTERVAL).await;
    }
}
