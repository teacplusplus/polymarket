use super::*;
use crate::account::{
    Account, POLY_PRIVATE_KEY_ENV, SharedAccount, spawn_heartbeat,
    try_authenticate_clob_for_heartbeats,
};
use crate::account_order::{cancel_order_on_clob, CancelOrderRequest};
use crate::account_order_completion::SingleOrderClobInvocationReport;
use crate::account_ws::spawn_user_ws_listener;
use crate::constants::CurrencyUpDownOutcome;
use crate::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT;
use crate::util::{current_timestamp_ms, detect_country_and_ip, fetch_gamma_event_data_for_slug};
use anyhow::Context;
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::Side;
use polymarket_client_sdk::types::U256;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::{Notify, oneshot};

/// Период в секундах у slug `btc-updown-5m-{ts}`.
const BTC_UPDOWN_5M_PERIOD_SEC: i64 = 300;
/// Общий HTTP timeout в live-сценарии теста.
const LIVE_ORDER_HTTP_TIMEOUT_SEC: u64 = 20;
/// Окно прогрева user-WS: даём `spawn_user_ws_listener` законнектиться и подписаться
/// до размещения BUY, чтобы `filled_ws`/`settled_ws` участвовали наравне с HTTP-поллингом
/// (в финале берётся `max`-merge — итог корректен и без WS, но смысл теста — проверить и WS).
const LIVE_TEST_USER_WS_WARMUP_SECS: u64 = 3;

fn current_btc_updown_5m_slug(now_ms: i64) -> String {
    let poly_sec = now_ms / 1000;
    let window_start_sec = (poly_sec / BTC_UPDOWN_5M_PERIOD_SEC) * BTC_UPDOWN_5M_PERIOD_SEC;
    format!("btc-updown-5m-{window_start_sec}")
}

/// Конец 5m-окна в unix **ms** для slug `btc-updown-5m-{window_start_sec}` (стартует `window_start`).
fn btc_updown_5m_window_end_unix_ms_from_slug(slug: &str) -> Option<i64> {
    slug
        .strip_prefix("btc-updown-5m-")
        .and_then(|s| s.parse::<i64>().ok())
        .map(|window_start_sec| {
            window_start_sec
                .saturating_add(BTC_UPDOWN_5M_PERIOD_SEC)
                .saturating_mul(1000)
        })
}

fn decimal_to_f64(d: &polymarket_client_sdk::types::Decimal) -> anyhow::Result<f64> {
    d.to_string()
        .parse::<f64>()
        .map_err(|err| anyhow::anyhow!("Decimal {d} → f64: {err}"))
}

/// Минимально-достаточный notional для taker BUY в долларах (CLOB не пропустит меньше $1
/// и меньше `min_order_size × best_ask`).
fn min_taker_buy_usd_notional(min_order_size: f64, best_ask: f64) -> f64 {
    // CLOB marketable BUY: не меньше $1 (иначе 400 `min size: $1`).
    const CLOB_MIN_MARKETABLE_BUY_USD: f64 = 1.0;
    let raw = min_order_size * best_ask;
    let rounded = (raw * 100.0).ceil() / 100.0;
    rounded.max(CLOB_MIN_MARKETABLE_BUY_USD)
}

async fn live_btc_updown_book_buy_floor(
    account: &SharedAccount,
    asset_id: &str,
    slug: &str,
) -> anyhow::Result<(f64, f64, f64)> {
    let token_id = U256::from_str(asset_id)
        .with_context(|| format!("невалидный asset_id={asset_id} из Gamma slug={slug}"))?;
    let book_request = OrderBookSummaryRequest::builder()
        .token_id(token_id)
        .build();
    let book = account
        .clob
        .order_book(&book_request)
        .await
        .with_context(|| format!("order_book({asset_id}) для slug={slug}"))?;
    let min_order_size = decimal_to_f64(&book.min_order_size)?;
    let best_ask = best_ask_sdk(&book)
        .ok_or_else(|| anyhow::anyhow!("пустой asks book для asset_id={asset_id} slug={slug}"))?;
    let best_ask_f64 = decimal_to_f64(&best_ask)?;
    let market_floor_buy_usd = min_taker_buy_usd_notional(min_order_size, best_ask_f64);
    Ok((min_order_size, best_ask_f64, market_floor_buy_usd))
}

/// Комиссия уже в [`SingleOrderClobInvocationReport`]; цена BUY ≈ USD spent / NET shares.
fn implied_buy_px_per_share(rep: &SingleOrderClobInvocationReport) -> Option<f64> {
    let usd = match rep.making_amount {
        OrderAmount::UsdNotional(u) => u,
        OrderAmount::Shares(_) => return None,
    };
    let shares = match rep.taking_amount {
        OrderAmount::Shares(s) => s,
        OrderAmount::UsdNotional(_) => return None,
    };
    if !shares.is_finite() || shares <= 1e-12 || !usd.is_finite() || usd <= 0.0 {
        None
    } else {
        Some(usd / shares)
    }
}

#[derive(Clone, Debug)]
struct LegPrep {
    outcome: CurrencyUpDownOutcome,
    asset_id: String,
    min_order_size: f64,
    #[allow(dead_code)]
    best_ask: f64,
    buy_usd: f64,
    worst_buy: f64,
}

async fn duel_leg_prep_for_outcome(
    account: &SharedAccount,
    slug: &str,
    cu: &HashMap<String, CurrencyUpDownOutcome>,
    outcome: CurrencyUpDownOutcome,
) -> anyhow::Result<LegPrep> {
    let asset_id = cu
        .iter()
        .find(|(_, o)| **o == outcome)
        .map(|(aid, _)| aid.clone())
        .with_context(|| format!("нет outcome={outcome:?} в Gamma cu для slug={slug}"))?;
    let (min_order_size, best_ask_f64, floor_usd) =
        live_btc_updown_book_buy_floor(account, &asset_id, slug).await?;
    let worst_buy = (best_ask_f64 + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999);
    Ok(LegPrep {
        outcome,
        asset_id,
        min_order_size,
        best_ask: best_ask_f64,
        buy_usd: floor_usd,
        worst_buy,
    })
}

/// Ждём финала duel: один maker полностью исполнился (success, не partial),
/// противоположный maker сняли, противоположные shares — в taker.
const LIVE_DUAL_MAKER_RACE_DEADLINE_SEC: u64 = 180;
/// Лимит-продажа maker на +20% к средней цене taker BUY.
const LIVE_MAKER_TP_MULT: f64 = 1.2;

#[derive(Clone, Debug, Default)]
struct DuelState {
    /// Кто первым полностью реализовал maker-takeprofit.
    winner: Option<CurrencyUpDownOutcome>,
    /// Купленные shares (floor до 0.01), сохранённые после BUY-invoke перед maker POST.
    up_buy_floor: Option<f64>,
    down_buy_floor: Option<f64>,
    /// `order_id` resting maker после успешного HTTP POST maker (до settle invoke).
    maker_id_up: Option<String>,
    maker_id_down: Option<String>,
}

struct DuelHarness {
    state: Mutex<DuelState>,
    prep_up: LegPrep,
    prep_down: LegPrep,
    done: Notify,
}

impl DuelHarness {
    fn new(prep_up: LegPrep, prep_down: LegPrep) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(DuelState::default()),
            prep_up,
            prep_down,
            done: Notify::new(),
        })
    }

    fn prep_ref(&self, o: CurrencyUpDownOutcome) -> &LegPrep {
        match o {
            CurrencyUpDownOutcome::Up => &self.prep_up,
            CurrencyUpDownOutcome::Down => &self.prep_down,
        }
    }

    fn set_maker_order_id(&self, o: CurrencyUpDownOutcome, oid: Option<String>) {
        let mut g = self.state.lock().unwrap();
        match o {
            CurrencyUpDownOutcome::Up => g.maker_id_up = oid,
            CurrencyUpDownOutcome::Down => g.maker_id_down = oid,
        }
    }

    fn record_buy_floor(&self, o: CurrencyUpDownOutcome, shares: f64) {
        let mut g = self.state.lock().unwrap();
        match o {
            CurrencyUpDownOutcome::Up => g.up_buy_floor = Some(shares),
            CurrencyUpDownOutcome::Down => g.down_buy_floor = Some(shares),
        }
    }

    /// Если maker полностью набрал объём первым — противоположный maker id (если известен) и shares «лузера».
    fn claim_first_full_maker_hit(
        &self,
        winner: CurrencyUpDownOutcome,
    ) -> Option<(Option<String>, f64)> {
        let mut g = self.state.lock().unwrap();
        if g.winner.is_some() {
            return None;
        }
        let other = winner.opposite();
        let cancel_oid = match other {
            CurrencyUpDownOutcome::Up => g.maker_id_up.clone(),
            CurrencyUpDownOutcome::Down => g.maker_id_down.clone(),
        };
        let opp_sh = match other {
            CurrencyUpDownOutcome::Up => g.up_buy_floor?,
            CurrencyUpDownOutcome::Down => g.down_buy_floor?,
        };
        g.winner = Some(winner);
        Some((cancel_oid, opp_sh))
    }

    fn snapshot_state_unlocked_clone(&self) -> DuelState {
        self.state.lock().unwrap().clone()
    }
}

async fn duel_cancel_if_some(
    account: &SharedAccount,
    label: &str,
    oid: Option<&String>,
    wall_ms: u64,
) {
    let Some(id) = oid else {
        eprintln!(
            "[от старта {wall_ms} ms] duel: {label}: cancel skipped — maker order_id unknown"
        );
        return;
    };
    match cancel_order_on_clob(
        account,
        CancelOrderRequest {
            order_id: id.clone(),
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
        },
    )
    .await
    {
        Ok(r) => eprintln!(
            "[от старта {wall_ms} ms] duel: {label}: cancel order_id={id} canceled={}",
            r.canceled
        ),
        Err(err) => eprintln!(
            "[от старта {wall_ms} ms] duel: {label}: cancel order_id={id} err: {err:#}"
        ),
    }
}

async fn duel_taker_flatten_floor(
    account: &SharedAccount,
    slug: &str,
    outcome: CurrencyUpDownOutcome,
    aid: &str,
    shares_floor: f64,
    min_sz: f64,
    invoke_label: &str,
) -> anyhow::Result<SingleOrderClobInvocationReport> {
    if shares_floor < min_sz || !shares_floor.is_finite() {
        anyhow::bail!(
            "duel flatten {invoke_label}: shares_floor={shares_floor:.4} < min_order_size={min_sz:.6} slug={slug} outcome={outcome:?}",
        );
    }
    let (tx, rx) = oneshot::channel();
    post_order_on_clob(
        account,
        PostOrderRequest {
            asset_id: aid.to_string(),
            side: Side::Sell,
            role: OrderRole::Taker,
            amount: OrderAmount::Shares(shares_floor),
            price: None,
            max_slippage_pp: None,
            expiration: None,
            market_end_unix_ms: None,
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
            strict_book: None,
        },
        Box::new(move |rep| {
            let _ = tx.send(rep);
        }),
    )
    .await
    .with_context(|| format!("duel flatten POST taker slug={slug} invoke={invoke_label}"))?;
    rx.await
        .map_err(|_| anyhow::anyhow!("duel flatten: invoke-колбёк потерян ({invoke_label})"))
}

async fn duel_on_full_maker_winner_flow(
    harness: Arc<DuelHarness>,
    account: SharedAccount,
    slug: String,
    wall_ms: u64,
    maker_outcome: CurrencyUpDownOutcome,
    maker_rep: SingleOrderClobInvocationReport,
) {
    eprintln!(
        "[от старта {wall_ms} ms] duel: maker финал {:?}: success={}, partial={}, order_id={:?}, error_msg={:?}",
        maker_outcome,
        maker_rep.success,
        maker_rep.partial,
        maker_rep.order_id,
        maker_rep.error_msg,
    );

    if !maker_rep.success {
        return;
    }
    if maker_rep.partial {
        return;
    }

    let Some((cancel_other_oid, opp_sh_floor)) = harness.claim_first_full_maker_hit(maker_outcome)
    else {
        eprintln!(
            "[от старта {wall_ms} ms] duel: второй/full maker финал после победителя или гонка — {:?} игнор",
            maker_outcome
        );
        return;
    };

    let other_outcome = maker_outcome.opposite();
    duel_cancel_if_some(
        &account,
        &format!("отмен противоположного maker ({other_outcome:?})"),
        cancel_other_oid.as_ref(),
        wall_ms,
    )
    .await;

    let opp_prep = harness.prep_ref(other_outcome);
    let flatten_res = duel_taker_flatten_floor(
        &account,
        &slug,
        other_outcome,
        &opp_prep.asset_id,
        opp_sh_floor,
        opp_prep.min_order_size,
        &format!("unwind loser taker {:?}", other_outcome),
    )
    .await;

    match flatten_res {
        Ok(sell_rep) => {
            eprintln!(
                "[от старта {wall_ms} ms] duel: unwind противоположного {:?}: sold floor={:.4}, maker/taking {:?}/{:?}, order_id={:?}, success={}, partial={}, err={:?}",
                other_outcome,
                opp_sh_floor,
                sell_rep.making_amount,
                sell_rep.taking_amount,
                sell_rep.order_id,
                sell_rep.success,
                sell_rep.partial,
                sell_rep.error_msg,
            );
            if !(sell_rep.success && sell_rep.order_id.as_deref().is_some_and(|s| !s.is_empty()))
            {
                eprintln!(
                    "[от старта {wall_ms} ms] duel: WARNING unwind taker финал без полного успеха"
                );
            }
        }
        Err(err) => {
            eprintln!(
                "[от старта {wall_ms} ms] duel: unwind противоположного {:?} упал: {err:#}",
                other_outcome
            );
        }
    }

    harness.done.notify_waiters();
}

/// Сигнал в `rx`, когда BUY-колбэк выставил maker и завершился весь `spawn` цикл (ошибочный BUY тоже).
struct AckDrop(Option<oneshot::Sender<()>>);

impl Drop for AckDrop {
    fn drop(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}

async fn duel_post_buy_then_maker_in_callback(
    account: SharedAccount,
    duel: Arc<DuelHarness>,
    prep: LegPrep,
    slug: String,
    wall_anchor: Arc<std::time::Instant>,
) -> anyhow::Result<()> {
    let aid = prep.asset_id.clone();
    let buy_usd = prep.buy_usd;
    let worst = prep.worst_buy;
    let outcome_t = prep.outcome;
    let min_sz_buy = prep.min_order_size;

    let (chain_done_tx, chain_done_rx) = oneshot::channel();

    let wall_in_cb = Arc::clone(&wall_anchor);
    let buy_spawn_account = account.clone();
    post_order_on_clob(
        &account,
        PostOrderRequest {
            asset_id: aid.clone(),
            side: Side::Buy,
            role: OrderRole::Taker,
            amount: OrderAmount::UsdNotional(buy_usd),
            price: Some(worst),
            max_slippage_pp: None,
            expiration: None,
            market_end_unix_ms: None,
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
            strict_book: None,
        },
        Box::new(move |buy_rep| {
            let account = buy_spawn_account.clone();
            let duel = Arc::clone(&duel);
            let slug_buy = slug.clone();
            let wall_spawn = Arc::clone(&wall_in_cb);
            tokio::spawn(async move {
                let _ack = AckDrop(Some(chain_done_tx));
                let wall_ms = wall_spawn.elapsed().as_millis() as u64;
                eprintln!(
                    "[от старта {wall_ms} ms] duel: BUY taker финал {:?} asset={}: success={}, partial={}, order_id={:?}, making/taking {:?}/{:?}, err={:?}",
                    outcome_t,
                    aid,
                    buy_rep.success,
                    buy_rep.partial,
                    buy_rep.order_id,
                    buy_rep.making_amount,
                    buy_rep.taking_amount,
                    buy_rep.error_msg,
                );
                if !buy_rep.success {
                    return;
                }
                let shares_net = match buy_rep.taking_amount {
                    OrderAmount::Shares(s) => s,
                    OrderAmount::UsdNotional(_) => {
                        eprintln!("duel: BUY {:?}: ожидались Shares в taking_amount", outcome_t);
                        return;
                    }
                };
                if !(shares_net > 0.0 && shares_net.is_finite()) {
                    eprintln!("duel: BUY {:?}: плохой shares_net={}", outcome_t, shares_net);
                    return;
                }
                let shares_floor = (shares_net * 100.0).floor() / 100.0;
                if shares_floor < min_sz_buy {
                    eprintln!(
                        "duel: BUY {:?}: после floor до 0.01 shares {:.4} < min_order {:.4}",
                        outcome_t, shares_floor, min_sz_buy
                    );
                    return;
                }
                duel.record_buy_floor(outcome_t, shares_floor);

                let Some(implied_px) = implied_buy_px_per_share(&buy_rep) else {
                    eprintln!("duel: BUY {:?}: не удалось восстановить среднюю цену BUY", outcome_t);
                    return;
                };
                let maker_price = (implied_px * LIVE_MAKER_TP_MULT).clamp(0.001, 0.999);

                let market_end_unix_ms =
                    btc_updown_5m_window_end_unix_ms_from_slug(slug_buy.as_str()).or_else(|| {
                        let ms = current_timestamp_ms();
                        let poly_sec = ms / 1000;
                        let ws =
                            (poly_sec / BTC_UPDOWN_5M_PERIOD_SEC) * BTC_UPDOWN_5M_PERIOD_SEC;
                        Some((ws.saturating_add(BTC_UPDOWN_5M_PERIOD_SEC)).saturating_mul(1000))
                    });

                let (mk_invoke_tx, mk_invoke_rx) = oneshot::channel();
                let post_res = post_order_on_clob(
                    &account,
                    PostOrderRequest {
                        asset_id: aid.clone(),
                        side: Side::Sell,
                        role: OrderRole::Maker,
                        amount: OrderAmount::Shares(shares_floor),
                        price: Some(maker_price),
                        max_slippage_pp: None,
                        expiration: None,
                        market_end_unix_ms,
                        timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
                        strict_book: None,
                    },
                    Box::new(move |rep| {
                        let _ = mk_invoke_tx.send(rep);
                    }),
                )
                .await;

                match &post_res {
                    Ok(Some(oid)) => {
                        duel.set_maker_order_id(outcome_t, Some(oid.clone()));
                        let wall_oid = wall_ms;
                        eprintln!(
                            "[от старта {wall_oid} ms] duel: maker SELL resting POST {:?} order_id={oid} price={maker_price:.5} shares={shares_floor:.2} market_end_unix_ms={market_end_unix_ms:?}",
                            outcome_t
                        );
                    }
                    Ok(None) => eprintln!(
                        "duel: maker POST {:?} без order_id (success=false телом?)",
                        outcome_t
                    ),
                    Err(err) => {
                        eprintln!("duel: maker POST {:?} err: {err:#}", outcome_t);
                        return;
                    }
                }

                let maker_evt = mk_invoke_rx.await;
                match maker_evt {
                    Ok(maker_rep) => {
                        duel_on_full_maker_winner_flow(
                            duel,
                            account,
                            slug_buy.clone(),
                            wall_ms,
                            outcome_t,
                            maker_rep,
                        )
                        .await;
                    }
                    Err(_) => eprintln!(
                        "duel: {:?} maker invoke-колбёк потерян до финала агрегатора",
                        outcome_t
                    ),
                };
            });
        }),
    )
    .await
    .with_context(|| format!("duel BUY taker {:?}", outcome_t))?;
    chain_done_rx
        .await
        .map_err(|_| anyhow::anyhow!("duel BUY→maker spawn завершился без Ack ({outcome_t:?})"))?;
    Ok(())
}

async fn duel_emergency_flatten_and_cancel_all(
    account: &SharedAccount,
    duel: Arc<DuelHarness>,
    slug: &str,
    wall_ms: u64,
    snap: &DuelState,
) {
    eprintln!(
        "[от старта {wall_ms} ms] duel: EMERGENCY timeout/cleanup slug={slug} snapshot winner={:?} up_buy={:?} down_buy={:?}",
        snap.winner,
        snap.up_buy_floor,
        snap.down_buy_floor
    );

    duel_cancel_if_some(
        account,
        "cleanup maker UP",
        snap.maker_id_up.as_ref(),
        wall_ms,
    )
    .await;
    duel_cancel_if_some(
        account,
        "cleanup maker DOWN",
        snap.maker_id_down.as_ref(),
        wall_ms,
    )
    .await;

    if let Some(sf) = snap.up_buy_floor {
        let hp = duel.prep_ref(CurrencyUpDownOutcome::Up);
        if sf >= hp.min_order_size {
            match duel_taker_flatten_floor(
                account,
                slug,
                CurrencyUpDownOutcome::Up,
                &hp.asset_id,
                sf,
                hp.min_order_size,
                "emergency_flatten_up",
            )
            .await
            {
                Ok(r) => eprintln!(
                    "[от старта {wall_ms} ms] duel: emergency flatten UP ok success={}",
                    r.success
                ),
                Err(err) => eprintln!(
                    "[от старта {wall_ms} ms] duel: emergency flatten UP failed: {err:#}"
                ),
            }
        }
    }

    if let Some(sf) = snap.down_buy_floor {
        let hp = duel.prep_ref(CurrencyUpDownOutcome::Down);
        if sf >= hp.min_order_size {
            match duel_taker_flatten_floor(
                account,
                slug,
                CurrencyUpDownOutcome::Down,
                &hp.asset_id,
                sf,
                hp.min_order_size,
                "emergency_flatten_down",
            )
            .await
            {
                Ok(r) => eprintln!(
                    "[от старта {wall_ms} ms] duel: emergency flatten DOWN ok success={}",
                    r.success
                ),
                Err(err) => eprintln!(
                    "[от старта {wall_ms} ms] duel: emergency flatten DOWN failed: {err:#}"
                ),
            }
        }
    }

    duel.done.notify_waiters();
}

/// Live BUY→SELL taker по текущему 5m BTC up/down: берётся **минимальный допустимый**
/// CLOB notional (`min_order_size × best_ask`, не ниже $1); среди исходов Gamma выбирается
/// сторона с меньшим floor (часто обе дороже $1 — тогда тратится меньший из двух минимумов).
///
/// ```bash
/// POLY_PRIVATE_KEY=0x… \
///     cargo test --bin poly account_order::tests::live_taker_roundtrip_btc_updown_5m -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "live network: требует POLY_PRIVATE_KEY и pUSD на Safe; BUY на CLOB-min notional (может быть > $1)"]
async fn live_taker_roundtrip_btc_updown_5m() -> anyhow::Result<()> {
    use std::time::Instant;

    let _ = dotenvy::dotenv();
    let _ = rustls::crypto::ring::default_provider().install_default();

    let t0 = Instant::now();
    let mut last_evt = t0;
    macro_rules! evt_ms {
        ($last:ident, $t0:ident) => {{
            let now = Instant::now();
            let prev = std::mem::replace(&mut $last, now);
            let dt = now.saturating_duration_since(prev).as_millis() as u64;
            let wall = now.saturating_duration_since($t0).as_millis() as u64;
            (dt, wall)
        }};
    }

    // Открываем отдельный test-only tee-канал: подробные `[order_invoke/...]` логи
    // (HTTP-запросы, WS-события, агрегация, latency, replay-инструкция) идут **только**
    // в этот файл и не засоряют stdout/stderr. Файл — на каждый прогон уникальный
    // (timestamp в имени), кладём в `target/` рядом с артефактами cargo. Сбой
    // инициализации сам по себе не валит тест — макросы `test_tee_*` без открытого
    // файла становятся no-op.
    let log_path = std::path::PathBuf::from(format!(
        "target/live_taker_roundtrip_btc_updown_5m_{}.log",
        current_timestamp_ms()
    ));
    if let Err(err) =
        crate::tee_log::init_test_tee_log_file(&log_path, "live_taker_roundtrip_btc_updown_5m")
    {
        let (dt, wall) = evt_ms!(last_evt, t0);
        eprintln!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: init_test_tee_log_file({}) failed: {err:#} \
             — test продолжит, но детальный `[order_invoke/...]` лог в файл писаться не будет",
            log_path.display(),
        );
    } else {
        let (dt, wall) = evt_ms!(last_evt, t0);
        eprintln!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: detailed `[order_invoke/...]` log → {}",
            log_path.display(),
        );
    }

    let geo = detect_country_and_ip()
        .await
        .ok_or_else(|| {
            let (dt, wall) = evt_ms!(last_evt, t0);
            anyhow::anyhow!(
                "[от старта {wall} ms | с прошлого {dt} ms] Polymarket geoblock: не удалось GET https://polymarket.com/api/geoblock"
            )
        })?;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        !geo.blocked,
        "[от старта {wall} ms | с прошлого {dt} ms] Polymarket geoblock: торговля с этого региона заблокирована \
         (country={:?}, region={:?}, ip={:?})",
        geo.country,
        geo.region,
        geo.ip,
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: geo:{:?}",
        geo
    );

    let private_key_set = std::env::var(POLY_PRIVATE_KEY_ENV)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .is_some();
    if !private_key_set {
        let (dt, wall) = evt_ms!(last_evt, t0);
        eprintln!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: {POLY_PRIVATE_KEY_ENV} не задан, тест пропущен",
        );
        return Ok(());
    }
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC))
        .build()?;
    let slug = current_btc_updown_5m_slug(current_timestamp_ms());
    let gamma = fetch_gamma_event_data_for_slug(&http, &slug).await?;
    let cu = &gamma.currency_up_down_by_asset_id;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        !cu.is_empty(),
        "[от старта {wall} ms | с прошлого {dt} ms] Gamma не вернул clobTokenIds для slug={slug}",
    );

    let account = Account::new_shared();
    try_authenticate_clob_for_heartbeats(&account).await;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        account.clob_authed.load().is_some(),
        "[от старта {wall} ms | с прошлого {dt} ms] CLOB auth не поднялся — проверьте {POLY_PRIVATE_KEY_ENV} и логи [heartbeat]",
    );

    // Поднимаем heartbeat (продлевает CLOB auth каждые `CLOB_HEARTBEAT_INTERVAL_SEC`)
    // и user-WS listener (питает `filled_ws`/`settled_ws` в `PostOrderInvokeAggregator`
    // параллельно с REST-поллингом). Финал колбэка берётся через `max`-merge обоих
    // источников — тест поэтому проверяет именно совместную работу WS + HTTP.
    spawn_heartbeat(account.clone());
    spawn_user_ws_listener(account.clone());

    tokio::time::sleep(Duration::from_secs(LIVE_TEST_USER_WS_WARMUP_SECS)).await;

    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: дождался {LIVE_TEST_USER_WS_WARMUP_SECS}s на прогрев user-WS subscribe \
         (поищите в логе строки `[user_ws] подписан`/`[user_ws] trade`)",
    );

    let mut best: Option<(String, f64, f64, f64)> = None;
    for cand_id in cu.keys() {
        let row = live_btc_updown_book_buy_floor(&account, cand_id, &slug).await?;
        let take = match &best {
            None => true,
            Some((_, _, _, best_floor)) => row.2 < *best_floor - 1e-12,
        };
        if take {
            best = Some((cand_id.clone(), row.0, row.1, row.2));
        }
    }
    let (asset_id, min_order_size, best_ask_f64, market_floor_buy_usd) =
        best.unwrap_or_else(|| {
            let (dt, wall) = evt_ms!(last_evt, t0);
            panic!(
                "[от старта {wall} ms | с прошлого {dt} ms] cu не пустой — выше ensure: best=None"
            )
        });
    let buy_usd = market_floor_buy_usd;
    let worst_acceptable_buy = (best_ask_f64 + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999);

    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: slug={slug}, asset_id={asset_id}, \
         min_order_size={min_order_size:.4}, best_ask={best_ask_f64:.4}, \
         market_floor_buy_usd={market_floor_buy_usd:.4}, buy_usd={buy_usd:.4} \
         (CLOB-min notional), worst_acceptable_buy={worst_acceptable_buy:.4}",
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: покупка — цель ≈ {buy_usd:.4} USD notional (taker BUY)"
    );

    let (buy_invoke_tx, buy_invoke_rx) = tokio::sync::oneshot::channel();
    post_order_on_clob(
        &account,
        PostOrderRequest {
            asset_id: asset_id.clone(),                // Gamma outcome token
            side: Side::Buy,                           // вход long
            role: OrderRole::Taker,                    // FAK
            amount: OrderAmount::UsdNotional(buy_usd), // мин. допустимый notional
            price: Some(worst_acceptable_buy),         // явный worst-acceptable
            max_slippage_pp: None,                     // не используем slip от L1
            expiration: None,                          // taker
            market_end_unix_ms: None,
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC), // POST /order
            strict_book: None,                                         // GET book выше
        },
        Box::new(move |rep| {
            let _ = buy_invoke_tx.send(rep);
        }),
    )
    .await
    .with_context(|| {
        let (dt, wall) = evt_ms!(last_evt, t0);
        format!(
            "[от старта {wall} ms | с прошлого {dt} ms] BUY taker slug={slug} asset_id={asset_id}"
        )
    })?;
    let buy_result = buy_invoke_rx.await.map_err(|_| {
        let (dt, wall) = evt_ms!(last_evt, t0);
        anyhow::anyhow!("[от старта {wall} ms | с прошлого {dt} ms] BUY taker колбёк потерян")
    })?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    let paid = match buy_result.making_amount {
        OrderAmount::UsdNotional(u) => format!("{u:.4} USDC"),
        OrderAmount::Shares(s) => format!("{s:.6} shares (making)"),
    };
    let got = match buy_result.taking_amount {
        OrderAmount::Shares(s) => format!("{s:.6} shares"),
        OrderAmount::UsdNotional(u) => format!("{u:.4} USDC (taking)"),
    };
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: куплено — отдано {paid}, получено {got}, \
         order_id={:?}, partial={}, целевой notional до ордера ≈{buy_usd:.4} USDC",
        buy_result.order_id, buy_result.partial,
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        buy_result
            .order_id
            .as_deref()
            .is_some_and(|id| !id.is_empty()),
        "[от старта {wall} ms | с прошлого {dt} ms] пустой order_id после BUY"
    );
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        buy_result.success,
        "[от старта {wall} ms | с прошлого {dt} ms] BUY taker финал не успех: order_id={:?}, partial={}, error_msg={:?}",
        buy_result.order_id,
        buy_result.partial,
        buy_result.error_msg,
    );

    let shares_received_net = match buy_result.taking_amount {
        OrderAmount::Shares(s) => s,
        OrderAmount::UsdNotional(_) => {
            let (dt, wall) = evt_ms!(last_evt, t0);
            anyhow::bail!(
                "[от старта {wall} ms | с прошлого {dt} ms] BUY taker: ожидались Shares в taking_amount, получили USD notion"
            );
        }
    };
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        shares_received_net > 0.0 && shares_received_net.is_finite(),
        "[от старта {wall} ms | с прошлого {dt} ms] BUY taker не дал shares в taking_amount: {:?}, order_id={:?}",
        buy_result.taking_amount,
        buy_result.order_id,
    );

    // Polymarket V2 SDK валидирует `Amount::shares(...)` максимум на 2 десятичных знака
    // (лот-сайз = 0.01 shares), иначе `Validation: invalid: Unable to build Amount with N
    // decimal points, must be <= 2`. BUY-callback дал NET-shares с произвольной точностью
    // (например `33.333332` от `$1 / $0.03`). Округляем **вниз** до 0.01 — гарантия, что
    // мы не пытаемся продать больше, чем реально зачислено на чейне.
    let shares_to_sell = (shares_received_net * 100.0).floor() / 100.0;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        shares_to_sell >= min_order_size,
        "[от старта {wall} ms | с прошлого {dt} ms] после округления вниз до 0.01 shares_to_sell={shares_to_sell:.2} < \
         min_order_size={min_order_size:.4}; net_from_buy={shares_received_net:.6}",
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: продажа — цель {shares_to_sell:.2} shares \
         (net_from_buy={shares_received_net:.6}, rounded down to 0.01 lot) (taker SELL)"
    );

    let (sell_invoke_tx, sell_invoke_rx) = tokio::sync::oneshot::channel();
    post_order_on_clob(
        &account,
        PostOrderRequest {
            asset_id: asset_id.clone(),                  // тот же токен
            side: Side::Sell,                            // unwind
            role: OrderRole::Taker,                      // FAK
            amount: OrderAmount::Shares(shares_to_sell), // весь fill с BUY (округлён вниз до лот-сайза)
            price: None,                                 // маркет-продажа в bid
            max_slippage_pp: None,                       // без cap
            expiration: None,                            // taker
            market_end_unix_ms: None,
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC), // POST /order
            strict_book: None,                                         // нет локального book
        },
        Box::new(move |rep| {
            let _ = sell_invoke_tx.send(rep);
        }),
    )
    .await
    .with_context(|| {
        let (dt, wall) = evt_ms!(last_evt, t0);
        format!(
            "[от старта {wall} ms | с прошлого {dt} ms] SELL taker slug={slug} asset_id={asset_id}"
        )
    })?;
    let sell_result = sell_invoke_rx.await.map_err(|_| {
        let (dt, wall) = evt_ms!(last_evt, t0);
        anyhow::anyhow!("[от старта {wall} ms | с прошлого {dt} ms] SELL taker колбёк потерян")
    })?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    let sold = match sell_result.making_amount {
        OrderAmount::Shares(s) => format!("{s:.2} shares"),
        OrderAmount::UsdNotional(u) => format!("{u:.4} USDC (making)"),
    };
    let proceeds = match sell_result.taking_amount {
        OrderAmount::UsdNotional(u) => format!("{u:.4} USDC"),
        OrderAmount::Shares(s) => format!("{s:.6} shares (taking)"),
    };
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: продано — отдано {sold}, получено {proceeds}, \
         order_id={:?}, partial={}",
        sell_result.order_id, sell_result.partial,
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        sell_result.success,
        "[от старта {wall} ms | с прошлого {dt} ms] SELL taker финал не успех: order_id={:?}, partial={}, error_msg={:?}",
        sell_result.order_id,
        sell_result.partial,
        sell_result.error_msg,
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m OK: buy_order_id={:?}, sell_order_id={:?}, \
         buy_usd={buy_usd:.4}, shares_sold={shares_to_sell:.4}",
        buy_result.order_id, sell_result.order_id,
    );
    // Гарантируем, что хвост `[order_invoke/...]` ушёл на диск до возврата из теста.
    // На fail-путях (ранние `?`/`ensure!`) `BufWriter` всё равно сфлашится в Drop
    // статика при штатном завершении процесса.
    crate::tee_log::finish_test_tee_log();
    Ok(())
}

/// Duel: параллельные **taker BUY** по **Up** и **Down** (`tokio::try_join`), в каждом BUY-invoke —
/// `tokio::spawn` и из него **maker SELL** всей набранной позиции (+20% к среднему BUY `USD/shares`).
/// Первый maker с полным исполнением (`success && !partial`) бьёт `claim`: **cancel** maker другой стороны,
/// противоположные shares разгружаются **taker SELL**.

///
/// Если за [`LIVE_DUAL_MAKER_RACE_DEADLINE_SEC`] никто из maker не финализирует победную ветку — emergency:
/// отменить обоих maker (если известны `order_id`), затем два **taker SELL** всего сохранённого floor.
///
/// ```bash
/// POLY_PRIVATE_KEY=0x… \
///     cargo test --bin poly account_order::tests::live_duel_up_down_maker_race_tp20 -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "live duel: требует POLY_PRIVATE_KEY + pUSD; две покупки, maker TP 20%, гонка, cancel + taker flat"]
async fn live_duel_up_down_maker_race_tp20() -> anyhow::Result<()> {
    use std::time::Instant;

    let _ = dotenvy::dotenv();
    let _ = rustls::crypto::ring::default_provider().install_default();

    let t0 = Instant::now();
    let mut last_evt = t0;
    macro_rules! evt_ms {
        ($last:ident, $t0:ident) => {{
            let now = Instant::now();
            let prev = std::mem::replace(&mut $last, now);
            let dt = now.saturating_duration_since(prev).as_millis() as u64;
            let wall = now.saturating_duration_since($t0).as_millis() as u64;
            (dt, wall)
        }};
    }

    let wall_anchor = Arc::new(t0);

    let log_path = std::path::PathBuf::from(format!(
        "target/live_duel_up_down_maker_race_tp20_{}.log",
        current_timestamp_ms()
    ));
    if let Err(err) =
        crate::tee_log::init_test_tee_log_file(&log_path, "live_duel_up_down_maker_race_tp20")
    {
        let (dt, wall) = evt_ms!(last_evt, t0);
        eprintln!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel: init_test_tee_log_file({}) failed: {err:#}",
            log_path.display(),
        );
    } else {
        let (dt, wall) = evt_ms!(last_evt, t0);
        eprintln!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel: `[order_invoke/...]` tee → {}",
            log_path.display(),
        );
    }

    let geo = detect_country_and_ip()
        .await
        .ok_or_else(|| {
            let (dt, wall) = evt_ms!(last_evt, t0);
            anyhow::anyhow!(
                "[от старта {wall} ms | с прошлого {dt} ms] geoblock: не удалось GET polymarket.com/api/geoblock"
            )
        })?;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        !geo.blocked,
        "[от старта {wall} ms | с прошлого {dt} ms] geoblocked (country={:?}, region={:?}, ip={:?})",
        geo.country,
        geo.region,
        geo.ip,
    );

    let private_key_set = std::env::var(POLY_PRIVATE_KEY_ENV)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .is_some();
    if !private_key_set {
        let (dt, wall) = evt_ms!(last_evt, t0);
        eprintln!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel: {POLY_PRIVATE_KEY_ENV} не задан — skip",
        );
        return Ok(());
    }

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC))
        .build()?;
    let slug = current_btc_updown_5m_slug(current_timestamp_ms());
    let gamma = fetch_gamma_event_data_for_slug(&http, &slug).await?;
    let cu = &gamma.currency_up_down_by_asset_id;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        !cu.is_empty(),
        "[от старта {wall} ms | с прошлого {dt} ms] Gamma: пусто для slug={slug}",
    );

    let account = Account::new_shared();
    try_authenticate_clob_for_heartbeats(&account).await;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        account.clob_authed.load().is_some(),
        "[от старта {wall} ms | с прошлого {dt} ms] CLOB auth — проверьте {POLY_PRIVATE_KEY_ENV}",
    );

    spawn_heartbeat(account.clone());
    spawn_user_ws_listener(account.clone());
    tokio::time::sleep(Duration::from_secs(LIVE_TEST_USER_WS_WARMUP_SECS)).await;
    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel: slug={slug} прогрет user-WS {}s",
        LIVE_TEST_USER_WS_WARMUP_SECS,
    );

    let prep_up =
        duel_leg_prep_for_outcome(&account, &slug, cu, CurrencyUpDownOutcome::Up).await?;
    let prep_down =
        duel_leg_prep_for_outcome(&account, &slug, cu, CurrencyUpDownOutcome::Down).await?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel: UP buy_usd={:.4} asset={} ; DOWN buy_usd={:.4} asset={}",
        prep_up.buy_usd,
        prep_up.asset_id,
        prep_down.buy_usd,
        prep_down.asset_id,
    );

    let duel_h = DuelHarness::new(prep_up.clone(), prep_down.clone());

    let h_up = {
        let account = account.clone();
        let duel_h = Arc::clone(&duel_h);
        let slug = slug.clone();
        let wall_anchor = Arc::clone(&wall_anchor);
        tokio::spawn(async move {
            duel_post_buy_then_maker_in_callback(account, duel_h, prep_up, slug, wall_anchor).await
        })
    };
    let h_dn = {
        let account = account.clone();
        let duel_h = Arc::clone(&duel_h);
        let slug = slug.clone();
        let wall_anchor = Arc::clone(&wall_anchor);
        tokio::spawn(async move {
            duel_post_buy_then_maker_in_callback(account, duel_h, prep_down, slug, wall_anchor).await
        })
    };
    let (j_up, j_dn) = tokio::join!(h_up, h_dn);
    j_up.map_err(|e| anyhow::anyhow!("live_duel tokio::spawn UP JoinError: {e}"))?
        .with_context(|| format!("live_duel BUY→maker нога UP slug={slug}"))?;
    j_dn.map_err(|e| anyhow::anyhow!("live_duel tokio::spawn DOWN JoinError: {e}"))?
        .with_context(|| format!("live_duel BUY→maker нога DOWN slug={slug}"))?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    eprintln!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel: обе ноги BUY→maker ACK; жду гонку maker или дедлайн {}s…",
        LIVE_DUAL_MAKER_RACE_DEADLINE_SEC,
    );

    match tokio::time::timeout(
        Duration::from_secs(LIVE_DUAL_MAKER_RACE_DEADLINE_SEC),
        duel_h.done.notified(),
    )
    .await
    {
        Ok(_) => {
            let st = duel_h.snapshot_state_unlocked_clone();
            let (dt, wall) = evt_ms!(last_evt, t0);
            anyhow::ensure!(
                st.winner.is_some(),
                "[от старта {wall} ms | с прошлого {dt} ms] live_duel: notify без победителя state={:?} — вероятно emergency path",
                st,
            );
            eprintln!(
                "[от старта {wall} ms | с прошлого {dt} ms] live_duel OK: победила сторона {:?}; \
                 сохранённые floor up={:?} down={:?}",
                st.winner, st.up_buy_floor, st.down_buy_floor,
            );
        }
        Err(_elapsed) => {
            let wm = wall_anchor.elapsed().as_millis() as u64;
            let snap = duel_h.snapshot_state_unlocked_clone();
            eprintln!(
                "[от старта {wm} ms] live_duel: дедлайн {}s без победной нотификации winner={:?}",
                LIVE_DUAL_MAKER_RACE_DEADLINE_SEC,
                snap.winner,
            );
            duel_emergency_flatten_and_cancel_all(&account, Arc::clone(&duel_h), &slug, wm, &snap)
                .await;
            let (dt, wall) = evt_ms!(last_evt, t0);
            anyhow::bail!(
                "[от старта {wall} ms | с прошлого {dt} ms] live_duel: maker-гонка не завершилась за {}s — выполнен cleanup",
                LIVE_DUAL_MAKER_RACE_DEADLINE_SEC,
            );
        }
    }

    crate::tee_log::finish_test_tee_log();
    Ok(())
}
