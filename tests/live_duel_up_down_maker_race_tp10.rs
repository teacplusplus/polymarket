//! Live duel (dual taker BUY + maker TP race) integration test.

use anyhow::Context;
use poly::account::{
    Account, POLY_PRIVATE_KEY_ENV, SharedAccount, spawn_heartbeat,
    try_authenticate_clob_for_heartbeats,
};
use poly::account_order::{
    best_ask_sdk, cancel_order_on_clob, post_order_on_clob, CancelOrderRequest, OrderAmount,
    OrderRole, PostOrderRequest, SingleOrderClobInvocationReport,
};
use poly::account_ws::spawn_user_ws_listener;
use poly::constants::CurrencyUpDownOutcome;
use poly::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT;
use poly::util::{
    current_timestamp_ms, detect_country_and_ip, fetch_gamma_event_data_for_gamma_client,
};
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::Side;
use polymarket_client_sdk::types::U256;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
/// Лимит-продажа maker на **+10%** к средней цене taker BUY.
const LIVE_MAKER_TP_MULT: f64 = 1.1;

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
    /// On-chain settled shares **отданные** maker'ом по каждой ноге (NET, кумулятив).
    /// Пишется из финального `invoke` (single) на POST maker'a — источник правды о
    /// том, сколько из изначального `buy_floor` уже выкуплено через resting maker. Используется
    /// в [`duel_on_full_maker_winner_flow`] вместо обращения к Data API `/positions` для
    /// решения «нужно ли SELL'ить opp-инвентарь» при double-winner-гонке.
    maker_settled_shares_up: Mutex<f64>,
    maker_settled_shares_down: Mutex<f64>,
}

impl DuelHarness {
    fn new(prep_up: LegPrep, prep_down: LegPrep) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(DuelState::default()),
            prep_up,
            prep_down,
            done: Notify::new(),
            maker_settled_shares_up: Mutex::new(0.0),
            maker_settled_shares_down: Mutex::new(0.0),
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

    /// Multi-progress snapshot из агрегатора maker'a: `shares_settled_cum` — кумулятивные
    /// **отданные** shares on-chain (NET of fee) с момента POST maker'a. Сохраняем как
    /// `max(prev, новое)` для устойчивости к возможным регрессиям из аномальных событий
    /// (`max(settled_ws, settled_http)` на стороне агрегатора уже монотонен, но дополнительный
    /// `max` на нашей стороне — копеечная страховка).
    fn record_maker_settled_shares(&self, o: CurrencyUpDownOutcome, shares_settled_cum: f64) {
        if !shares_settled_cum.is_finite() || shares_settled_cum < 0.0 {
            return;
        }
        let cell = match o {
            CurrencyUpDownOutcome::Up => &self.maker_settled_shares_up,
            CurrencyUpDownOutcome::Down => &self.maker_settled_shares_down,
        };
        let mut guard = cell.lock().unwrap();
        if shares_settled_cum > *guard {
            *guard = shares_settled_cum;
        }
    }

    fn get_maker_settled_shares(&self, o: CurrencyUpDownOutcome) -> f64 {
        let cell = match o {
            CurrencyUpDownOutcome::Up => &self.maker_settled_shares_up,
            CurrencyUpDownOutcome::Down => &self.maker_settled_shares_down,
        };
        *cell.lock().unwrap()
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
        poly::test_tee_println!(
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
        Ok(r) => poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: {label}: cancel order_id={id} canceled={}",
            r.canceled
        ),
        Err(err) => poly::test_tee_println!(
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
    let invoke_label_static = invoke_label.to_string();
    let invoke_lost = invoke_label_static.clone();
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
            poly::test_tee_println!(
                "[flatten {invoke_label_static}] taker SELL finalize: outcome={:?} success={}, partial={}, \
                 order_id={:?}, making={:?}, taking={:?}, error_msg={:?}",
                outcome,
                rep.success,
                rep.partial,
                rep.order_id,
                rep.making_amount,
                rep.taking_amount,
                rep.error_msg,
            );
            let _ = tx.send(rep);
        }),
    )
    .await
    .with_context(|| format!("duel flatten POST taker slug={slug} invoke={invoke_label}"))?;
    rx.await
        .map_err(|_| anyhow::anyhow!("duel flatten: invoke-колбёк потерян ({invoke_lost})"))
}

/// Противоположная нога **не купила** (нет `buy_floor`), на этой — есть инвентарь и (возможно) maker.
/// Снимаем maker, продаём все купленные shares taker-ом, выходим из сценария без ожидания гонки.
async fn duel_unwind_inventory_when_other_leg_buy_failed(
    account: &SharedAccount,
    duel: Arc<DuelHarness>,
    slug: &str,
    wall_ms: u64,
    successful: CurrencyUpDownOutcome,
    failed_peer: CurrencyUpDownOutcome,
    bought_floor_shares: f64,
) -> anyhow::Result<()> {
    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel early-exit: нога {:?} не получила успешный taker BUY; \
         успешная нога {:?}: снимаем maker (если был) → taker SELL всех {:.2} shares (floor)",
        failed_peer, successful, bought_floor_shares,
    );

    let snap = duel.snapshot_state_unlocked_clone();
    let maker_oid_to_cancel = match successful {
        CurrencyUpDownOutcome::Up => snap.maker_id_up.clone(),
        CurrencyUpDownOutcome::Down => snap.maker_id_down.clone(),
    };

    duel_cancel_if_some(
        account,
        &format!("ранний выход: снять maker {:?} перед market-SELL всей позиции", successful),
        maker_oid_to_cancel.as_ref(),
        wall_ms,
    )
    .await;

    let prep = duel.prep_ref(successful);
    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel early-exit: taker SELL всего набранного {:?} slug={slug} asset_id={} shares={:.2}",
        successful, prep.asset_id, bought_floor_shares,
    );

    let sell_rep = duel_taker_flatten_floor(
        account,
        slug,
        successful,
        &prep.asset_id,
        bought_floor_shares,
        prep.min_order_size,
        "early_exit_other_leg_buy_failed_flatten_all",
    )
    .await?;

    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel early-exit: продажа успешной ноги {:?} завершена — \
         отдано (making) {:?}, получено (taking) {:?}, order_id={:?}, partial={}, ok={}",
        successful,
        sell_rep.making_amount,
        sell_rep.taking_amount,
        sell_rep.order_id,
        sell_rep.partial,
        sell_rep.success,
    );

    anyhow::ensure!(
        sell_rep.success,
        "duel early-exit: SELL успешной ноги {:?} финал без success slug={slug}, err={:?}",
        successful,
        sell_rep.error_msg,
    );
    anyhow::ensure!(
        sell_rep.order_id.as_deref().is_some_and(|id| !id.is_empty()),
        "duel early-exit: пустой order_id после SELL {:?}", successful,
    );

    duel.done.notify_one();
    Ok(())
}

async fn duel_on_full_maker_winner_flow(
    harness: Arc<DuelHarness>,
    account: SharedAccount,
    slug: String,
    wall_ms: u64,
    maker_outcome: CurrencyUpDownOutcome,
    maker_rep: SingleOrderClobInvocationReport,
) {
    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel: ВХОД maker finalize {:?} колбёк после settle: БЫЛИ поставлены продажи-maker; финал rep: success={}, partial={}, order_id={:?}, making={:?}, taking={:?}, err={:?}",
        maker_outcome,
        maker_rep.success,
        maker_rep.partial,
        maker_rep.order_id,
        maker_rep.making_amount,
        maker_rep.taking_amount,
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
        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: второй/full maker финал после победителя или гонка — {:?} игнор",
            maker_outcome
        );
        return;
    };

    let other_outcome = maker_outcome.opposite();
    let opp_prep = harness.prep_ref(other_outcome);

    // Double-winner защита: оба maker могли сматчиться почти одновременно. Источник правды
    // об «уже зачисленных on-chain shares opp maker'a» — промежуточные/финальные обновления
    // `harness.maker_settled_shares_*` из `invoke` maker'a (см. `DuelHarness::record_maker_settled_shares`).
    // Финальный `SingleOrderInvokeCb` для opp maker'a мог ещё не вызваться; для main maker'а
    // single-invoke несёт settled-итог после on-chain.
    //
    // Если opp maker'у успело уйти ≥ `opp_sh_floor - lot`, цели «продать остаток» нет:
    // купленные в BUY-ноге shares уже выкупил opp maker через свой собственный fill, и
    // SELL на эти же shares падает на CLOB-balance/allowance check (баланс уже сменился).
    let opp_maker_settled = harness.get_maker_settled_shares(other_outcome);
    let remaining_inventory = (opp_sh_floor - opp_maker_settled).max(0.0);
    // Floor до CLOB-lot (0.01) — sub-lot хвост биржа всё равно не примет в новый ордер.
    let sell_shares = (remaining_inventory * 100.0).floor() / 100.0;
    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel: opp {:?} maker settled (saved)={:.4} sh (saved_floor={:.4}); \
         remaining_inventory={:.4} → SELL={:.4}",
        other_outcome, opp_maker_settled, opp_sh_floor, remaining_inventory, sell_shares,
    );

    duel_cancel_if_some(
        &account,
        &format!("отмен противоположного maker ({other_outcome:?})"),
        cancel_other_oid.as_ref(),
        wall_ms,
    )
    .await;

    if sell_shares < opp_prep.min_order_size {
        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: opp {:?} SELL пропущен — sell_shares={:.4} < min_order_size={:.4} \
             (opp maker уже выкупил инвентарь через свой fill, либо остался sub-lot хвост); \
             считаем gracefully unwound — double-winner",
            other_outcome, sell_shares, opp_prep.min_order_size,
        );
        harness.done.notify_one();
        return;
    }

    let flatten_res = duel_taker_flatten_floor(
        &account,
        &slug,
        other_outcome,
        &opp_prep.asset_id,
        sell_shares,
        opp_prep.min_order_size,
        &format!("unwind loser taker {:?}", other_outcome),
    )
    .await;

    match flatten_res {
        Ok(sell_rep) => {
            poly::test_tee_println!(
                "[от старта {wall_ms} ms] duel: unwind противоположного {:?}: sold floor={:.4}, maker/taking {:?}/{:?}, order_id={:?}, success={}, partial={}, err={:?}",
                other_outcome,
                sell_shares,
                sell_rep.making_amount,
                sell_rep.taking_amount,
                sell_rep.order_id,
                sell_rep.success,
                sell_rep.partial,
                sell_rep.error_msg,
            );
            if !(sell_rep.success && sell_rep.order_id.as_deref().is_some_and(|s| !s.is_empty()))
            {
                poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: WARNING unwind taker финал без полного успеха"
                );
            }
        }
        Err(err) => {
            poly::test_tee_println!(
                "[от старта {wall_ms} ms] duel: unwind противоположного {:?} упал: {err:#}",
                other_outcome
            );
        }
    }

    harness.done.notify_one();
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
                poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: ВХОД в invoke-колбёк taker BUY {:?} slug={} asset_id={} (целевой notion ≈{buy_usd:.4} USDC worst={worst:.5})",
                    outcome_t, slug_buy, aid,
                );
                poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: taker BUY {:?} итог после settle: успех={} частичн.={} order_id={:?}; \
                     КУПИЛИ/потратили making={:?}, taking={:?}, err={:?}",
                    outcome_t,
                    buy_rep.success,
                    buy_rep.partial,
                    buy_rep.order_id,
                    buy_rep.making_amount,
                    buy_rep.taking_amount,
                    buy_rep.error_msg,
                );
                if !buy_rep.success {
                    poly::test_tee_println!(
                        "[от старта {wall_ms} ms] duel: taker BUY {:?} ПРОВАЛ — maker не выставляется, вторая нога может разгрузить сценарий",
                        outcome_t,
                    );
                    return;
                }
                let shares_net = match buy_rep.taking_amount {
                    OrderAmount::Shares(s) => s,
                    OrderAmount::UsdNotional(_) => {
                        poly::test_tee_println!(
                            "[от старта {wall_ms} ms] duel: BUY {:?}: ожидались Shares в taking_amount — без maker",
                            outcome_t,
                        );
                        return;
                    }
                };
                if !(shares_net > 0.0 && shares_net.is_finite()) {
                    poly::test_tee_println!(
                        "[от старта {wall_ms} ms] duel: BUY {:?}: невалидный shares_net={} — maker не ставится",
                        outcome_t, shares_net,
                    );
                    return;
                }
                let shares_floor = (shares_net * 100.0).floor() / 100.0;
                if shares_floor < min_sz_buy {
                    poly::test_tee_println!(
                        "[от старта {wall_ms} ms] duel: BUY {:?}: после floor до 0.01 shares {:.4} < min_order {:.4} — maker не ставится",
                        outcome_t, shares_floor, min_sz_buy
                    );
                    return;
                }
                let Some(implied_px) = implied_buy_px_per_share(&buy_rep) else {
                    poly::test_tee_println!(
                        "[от старта {wall_ms} ms] duel: BUY {:?}: не смогли восстановить USD/share — без maker",
                        outcome_t,
                    );
                    return;
                };
                let maker_price_raw = implied_px * LIVE_MAKER_TP_MULT;

                poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: BUY {:?} зачтено для maker: NET shares {:.6} → floor {:.2}; \
                     сырая TP-цена до тика на CLOB (`post_order_on_clob`) ≈ {:.6}",
                    outcome_t,
                    shares_net,
                    shares_floor,
                    maker_price_raw,
                );
                duel.record_buy_floor(outcome_t, shares_floor);

                let market_end_unix_ms =
                    btc_updown_5m_window_end_unix_ms_from_slug(slug_buy.as_str()).or_else(|| {
                        let ms = current_timestamp_ms();
                        let poly_sec = ms / 1000;
                        let ws =
                            (poly_sec / BTC_UPDOWN_5M_PERIOD_SEC) * BTC_UPDOWN_5M_PERIOD_SEC;
                        Some((ws.saturating_add(BTC_UPDOWN_5M_PERIOD_SEC)).saturating_mul(1000))
                    });

                let (mk_invoke_tx, mk_invoke_rx) = oneshot::channel();
                // `record_maker_settled_shares` из финального отчёта maker SELL (NET shares в
                // making_amount). Раньше дублировали из progress-колбэка; теперь только финал.
                let duel_for_invoke = Arc::clone(&duel);
                let outcome_for_invoke = outcome_t;
                let post_res = post_order_on_clob(
                    &account,
                    PostOrderRequest {
                        asset_id: aid.clone(),
                        side: Side::Sell,
                        role: OrderRole::Maker,
                        amount: OrderAmount::Shares(shares_floor),
                        price: Some(maker_price_raw),
                        max_slippage_pp: None,
                        expiration: None,
                        market_end_unix_ms,
                        timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
                        strict_book: None,
                    },
                    Box::new(move |rep| {
                        if let OrderAmount::Shares(s) = rep.making_amount {
                            duel_for_invoke.record_maker_settled_shares(outcome_for_invoke, s);
                        }
                        poly::test_tee_println!(
                            "[maker POST {:?}] ВХОД в invoke финала лимита: success={}, partial={}, order_id={:?}, making={:?}, taking={:?}, err={:?}",
                            outcome_t,
                            rep.success,
                            rep.partial,
                            rep.order_id,
                            rep.making_amount,
                            rep.taking_amount,
                            rep.error_msg,
                        );
                        let _ = mk_invoke_tx.send(rep);
                    }),
                )
                .await;

                let resting_oid = match &post_res {
                    Ok(Some(oid)) if !oid.trim().is_empty() => Some(oid.clone()),
                    Ok(Some(_)) => {
                        poly::test_tee_println!(
                            "[от старта {wall_ms} ms] duel: maker POST {:?} вернул пустой order_id",
                            outcome_t,
                        );
                        None
                    }
                    Ok(None) => {
                        poly::test_tee_println!(
                            "[от старта {wall_ms} ms] duel: maker POST {:?} Ok(None) — resting нет до invoke",
                            outcome_t,
                        );
                        None
                    }
                    Err(err) => {
                        poly::test_tee_println!(
                            "[от старта {wall_ms} ms] duel: maker POST {:?} упал до/на REST: {err:#}",
                            outcome_t,
                        );
                        None
                    }
                };

                if let Some(oid) = resting_oid.clone() {
                    duel.set_maker_order_id(outcome_t, Some(oid.clone()));
                    poly::test_tee_println!(
                        "[от старта {wall_ms} ms] duel: maker {:?} принят книгой order_id={oid} \
                         сырая лимит-цена ≈{maker_price_raw:.6} (нормализация тика 0.01 в `post_order_on_clob`) \
                         shares={shares_floor:.2} market_end_unix_ms={market_end_unix_ms:?}",
                        outcome_t,
                    );
                }

                let maker_fin = mk_invoke_rx.await;
                match maker_fin {
                    Ok(maker_rep) => {
                        if resting_oid.is_none() {
                            poly::test_tee_println!(
                                "[от старта {wall_ms} ms] duel: maker {:?} не на книге — получен только отчёт агрегатора (success={}, err={:?}); гонку не ждём с этой ноги",
                                outcome_t,
                                maker_rep.success,
                                maker_rep.error_msg,
                            );
                            return;
                        }
                        poly::test_tee_println!(
                            "[от старта {wall_ms} ms] duel: финальный invoke resting maker {:?} (гонка take-profit)",
                            outcome_t,
                        );
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
                    Err(_) => poly::test_tee_println!(
                        "duel: {:?} maker invoke-колбёк потерян до финала агрегатора",
                        outcome_t,
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

/// Обе BUY-ноги зафлоорены, но для сценария гонки по тексту нужны **два** resting maker.
/// Если на книге меньше двух (`order_id`): **немедленно** — cancel всех висимых maker (если есть), затем
/// два **taker SELL** всего floor (не ждём дедлайн гонки).
async fn duel_abort_race_cancel_any_makers_then_flatten_both_fills(
    account: &SharedAccount,
    duel: Arc<DuelHarness>,
    slug: &str,
    wall_ms: u64,
    snap: &DuelState,
    reason_human: &str,
) -> anyhow::Result<()> {
    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel early-exit: {reason_human} slug={slug} \
         resting maker_id up={:?} down={:?} snap полный {:?}",
        snap.maker_id_up, snap.maker_id_down, snap,
    );

    duel_cancel_if_some(
        account,
        "early-exit cancel maker UP",
        snap.maker_id_up.as_ref(),
        wall_ms,
    )
    .await;
    duel_cancel_if_some(
        account,
        "early-exit cancel maker DOWN",
        snap.maker_id_down.as_ref(),
        wall_ms,
    )
    .await;

    let u = snap
        .up_buy_floor
        .with_context(|| "duel early-exit internal: отсутствует up_buy_floor")?;
    let d = snap
        .down_buy_floor
        .with_context(|| "duel early-exit internal: отсутствует down_buy_floor")?;
    let up_prep = duel.prep_ref(CurrencyUpDownOutcome::Up);
    let dn_prep = duel.prep_ref(CurrencyUpDownOutcome::Down);

    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel early-exit: taker SELL всего набранного UP — {u:.2} shares",
    );
    let up_sell = duel_taker_flatten_floor(
        account,
        slug,
        CurrencyUpDownOutcome::Up,
        &up_prep.asset_id,
        u,
        up_prep.min_order_size,
        "abort_race_flatten_up",
    )
    .await?;
    anyhow::ensure!(
        up_sell.success && up_sell.order_id.as_deref().is_some_and(|s| !s.is_empty()),
        "abort-race UP unwind: {:?}", up_sell,
    );

    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel early-exit: taker SELL всего набранного DOWN — {d:.2} shares",
    );
    let dn_sell = duel_taker_flatten_floor(
        account,
        slug,
        CurrencyUpDownOutcome::Down,
        &dn_prep.asset_id,
        d,
        dn_prep.min_order_size,
        "abort_race_flatten_down",
    )
    .await?;
    anyhow::ensure!(
        dn_sell.success && dn_sell.order_id.as_deref().is_some_and(|s| !s.is_empty()),
        "abort-race DOWN unwind: {:?}", dn_sell,
    );

    duel.done.notify_one();
    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel early-exit завершён: cancel (если были) + оба taker SELL",
    );
    Ok(())
}

async fn duel_emergency_flatten_and_cancel_all(
    account: &SharedAccount,
    duel: Arc<DuelHarness>,
    slug: &str,
    wall_ms: u64,
    snap: &DuelState,
) {
    poly::test_tee_println!(
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
                Ok(r) => poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: emergency flatten UP ok success={}",
                    r.success
                ),
                Err(err) => poly::test_tee_println!(
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
                Ok(r) => poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: emergency flatten DOWN ok success={}",
                    r.success
                ),
                Err(err) => poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: emergency flatten DOWN failed: {err:#}"
                ),
            }
        }
    }

    duel.done.notify_one();
}

/// Duel: параллельные **taker BUY** (**Up** и **Down**) — две таски **`tokio::spawn`**, ожидание через **`tokio::join!`**.
/// В каждом **BUY-invoke** (`SingleOrderInvokeCb`) вызывается **`tokio::spawn`**: там же выставляется **maker SELL**
/// всей набранной позиции (**+10%** к среднему BUY `USD/shares`).
///
/// Если **ровно одна** нога **не получила успешный taker BUY** (`record_buy_floor` не вызван) — без ожидания гонки maker:
/// на **успешной** стороне **cancel maker** (если известен `order_id`) и **taker SELL всех** накопленных shares по floor → **`Ok`**.
///
/// Если **обе купили**, но **нет ровно двух** resting maker (0 или 1 `order_id`) — **немедленно**:
/// **cancel** всех висимых maker, затем два **taker SELL** всего floor (гонку не ждём; по смыслу нужны **оба** maker).
///
/// Если **обе купили** и **обоих maker** на книге: первый **полностью исполнившийся** maker (`success && !partial`) делает **claim** —
/// **cancel** maker другой стороны, противоположные shares снимаются **taker SELL**.
///
/// В лог (**`stderr`**) пишется **`ВХОД`** и **итог** на ключевых invoke: taker BUY, колбёк финала resting maker после POST,
/// колбёк финала maker из агрегатора (гонка/takeprofit), каждый taker flatten.
///
/// Если за [`LIVE_DUAL_MAKER_RACE_DEADLINE_SEC`] никто из maker не дал победную нотификацию — **emergency**:
/// отменить обоих maker при необходимости, затем два **taker SELL** сохранённого floor.
///
/// ```bash
/// POLY_PRIVATE_KEY=0x… \
///     cargo test --test live_duel_up_down_maker_race_tp10 -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "live duel: требует POLY_PRIVATE_KEY + pUSD; две покупки, maker TP 10%, гонка, cancel + taker flat"]
async fn live_duel_up_down_maker_race_tp10() -> anyhow::Result<()> {
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

    let log_path = std::path::Path::new("xframes/last_stream.txt");
    if let Err(err) =
        poly::tee_log::init_stream_tee_log_file(&log_path, "live_duel_up_down_maker_race_tp10")
    {
        let (dt, wall) = evt_ms!(last_evt, t0);
        poly::test_tee_println!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel: init_stream_tee_log_file({}) failed: {err:#}",
            log_path.display(),
        );
    } else {
        let (dt, wall) = evt_ms!(last_evt, t0);
        poly::test_tee_println!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel: `[order_invoke/...]` tee → {}",
            log_path.display(),
        );
    }

    let account = Account::new_shared();
    let geo = detect_country_and_ip(account.http.as_ref())
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
        poly::test_tee_println!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel: {POLY_PRIVATE_KEY_ENV} не задан — skip",
        );
        return Ok(());
    }

    let slug = current_btc_updown_5m_slug(current_timestamp_ms());
    let gamma = fetch_gamma_event_data_for_gamma_client(account.gamma.as_ref(), &slug).await?;
    let cu = &gamma.currency_up_down_by_asset_id;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        !cu.is_empty(),
        "[от старта {wall} ms | с прошлого {dt} ms] Gamma: пусто для slug={slug}",
    );

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
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel: slug={slug} прогрет user-WS {}s",
        LIVE_TEST_USER_WS_WARMUP_SECS,
    );

    let prep_up =
        duel_leg_prep_for_outcome(&account, &slug, cu, CurrencyUpDownOutcome::Up).await?;
    let prep_down =
        duel_leg_prep_for_outcome(&account, &slug, cu, CurrencyUpDownOutcome::Down).await?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
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

    let wall_ms_post_legs = wall_anchor.elapsed().as_millis() as u64;
    let snap_two = duel_h.snapshot_state_unlocked_clone();

    match (snap_two.up_buy_floor, snap_two.down_buy_floor) {
        (None, None) => {
            let (dt, wall) = evt_ms!(last_evt, t0);
            anyhow::bail!(
                "[от старта {wall} ms | с прошлого {dt} ms] live_duel: обе BUY ноги без fill — см. «ВХОД/итог» в invoke taker BUY; slug={slug}",
            );
        }
        (Some(up_sh), None) => {
            let (dt, wall) = evt_ms!(last_evt, t0);
            poly::test_tee_println!(
                "[от старта {wall} ms | с прошлого {dt} ms] live_duel: DOWN не купила — снимаем maker UP (если есть) и taker SELL всех {up_sh:.2} shares UP",
            );
            duel_unwind_inventory_when_other_leg_buy_failed(
                &account,
                Arc::clone(&duel_h),
                &slug,
                wall_ms_post_legs,
                CurrencyUpDownOutcome::Up,
                CurrencyUpDownOutcome::Down,
                up_sh,
            )
            .await?;
            poly::tee_log::finish_stream_tee_log();
            return Ok(());
        }
        (None, Some(dn_sh)) => {
            let (dt, wall) = evt_ms!(last_evt, t0);
            poly::test_tee_println!(
                "[от старта {wall} ms | с прошлого {dt} ms] live_duel: UP не купила — снимаем maker DOWN (если есть) и taker SELL всех {dn_sh:.2} shares DOWN",
            );
            duel_unwind_inventory_when_other_leg_buy_failed(
                &account,
                Arc::clone(&duel_h),
                &slug,
                wall_ms_post_legs,
                CurrencyUpDownOutcome::Down,
                CurrencyUpDownOutcome::Up,
                dn_sh,
            )
            .await?;
            poly::tee_log::finish_stream_tee_log();
            return Ok(());
        }
        (Some(up_sh), Some(dn_sh)) => {
            let snap_race = duel_h.snapshot_state_unlocked_clone();
            let up_has_maker = snap_race
                .maker_id_up
                .as_deref()
                .map_or(false, |s| !s.trim().is_empty());
            let dn_has_maker = snap_race
                .maker_id_down
                .as_deref()
                .map_or(false, |s| !s.trim().is_empty());

            if !(up_has_maker && dn_has_maker) {
                let (dt, wall) = evt_ms!(last_evt, t0);
                let reason = if !up_has_maker && !dn_has_maker {
                    "обе BUY ok, но **ни один** maker не на книге"
                } else {
                    "обе BUY ok, но на книге **только один** maker — по сценарию нужны **два**"
                };
                poly::test_tee_println!(
                    "[от старта {wall} ms | с прошлого {dt} ms] live_duel: {reason}; \
                     cancel висящих maker (если есть) → два taker SELL всего floor; без ожидания {}s гонки",
                    LIVE_DUAL_MAKER_RACE_DEADLINE_SEC,
                );
                duel_abort_race_cancel_any_makers_then_flatten_both_fills(
                    &account,
                    Arc::clone(&duel_h),
                    &slug,
                    wall_anchor.elapsed().as_millis() as u64,
                    &snap_race,
                    reason,
                )
                .await?;
                poly::tee_log::finish_stream_tee_log();
                return Ok(());
            }

            let (dt, wall) = evt_ms!(last_evt, t0);
            poly::test_tee_println!(
                "[от старта {wall} ms | с прошлого {dt} ms] live_duel: обе BUY ok (floor up={up_sh:.2}, down={dn_sh:.2}); \
                 **оба** maker на книге (up_id={:?} down_id={:?}) — жду полный maker на одной стороне или дедлайн {}s…",
                snap_race.maker_id_up,
                snap_race.maker_id_down,
                LIVE_DUAL_MAKER_RACE_DEADLINE_SEC,
            );
        }
    }

    // Защита от race с `notify_one`: maker-callback может вызвать `done.notify_one()` ДО того,
    // как мы дойдём до `notified().await` (kейс double-winner: оба maker matched в ранней WS-пачке
    // ещё до того, как `tokio::join!` BUY-ног отдал нам управление). С `notify_one` permit
    // накапливается, поэтому следующий `notified().await` всё равно получит сигнал моментально;
    // но если winner уже set'нут — пройдём OK без лишней асинхронной сериализации и быстрее.
    if let Some(winner) = duel_h.snapshot_state_unlocked_clone().winner {
        let (dt, wall) = evt_ms!(last_evt, t0);
        poly::test_tee_println!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel OK (winner-already): победила сторона {winner:?} \
             до того, как главный поток дошёл до `done.notified()` — нет смысла ждать таймаут",
        );
        poly::tee_log::finish_stream_tee_log();
        return Ok(());
    }

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
            poly::test_tee_println!(
                "[от старта {wall} ms | с прошлого {dt} ms] live_duel OK: победила сторона {:?}; \
                 сохранённые floor up={:?} down={:?}",
                st.winner, st.up_buy_floor, st.down_buy_floor,
            );
        }
        Err(_elapsed) => {
            let wm = wall_anchor.elapsed().as_millis() as u64;
            let snap = duel_h.snapshot_state_unlocked_clone();
            poly::test_tee_println!(
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

    poly::tee_log::finish_stream_tee_log();
    Ok(())
}
