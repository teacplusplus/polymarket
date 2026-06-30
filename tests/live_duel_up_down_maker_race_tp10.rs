//! Live duel (dual taker BUY + maker TP race) integration test.

use anyhow::Context;
use poly::account::{
    Account, POLY_PRIVATE_KEY_ENV, SharedAccount, spawn_heartbeat,
    try_authenticate_clob_for_heartbeats,
};
use poly::account_order::{
    CancelOrderRequest, OrderAmount, OrderRole, PostOrderRequest, SingleOrderClobInvocationReport,
    best_ask_sdk, cancel_order_on_clob, invoke_settlement_watch, post_order_on_clob,
    wait_invoke_settlement,
};
use poly::account_ws::spawn_user_ws_listener;
use poly::constants::CurrencyUpDownOutcome;
use poly::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT;
use poly::util::{
    current_timestamp_ms, detect_country_and_ip, fetch_gamma_event_data_for_gamma_client,
};
use polymarket_client_sdk::clob::types::Side;
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::types::U256;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Период в секундах у slug `btc-updown-5m-{ts}`.
const BTC_UPDOWN_5M_PERIOD_SEC: i64 = 300;
/// Общий HTTP timeout в live-сценарии теста.
const LIVE_ORDER_HTTP_TIMEOUT_SEC: u64 = 20;
/// Прогрев user-WS до BUY (HTTP vs WS в invoke; финал — max-merge).
const LIVE_TEST_USER_WS_WARMUP_SECS: u64 = 3;

fn current_btc_updown_5m_slug(now_ms: i64) -> String {
    let poly_sec = now_ms / 1000;
    let window_start_sec = (poly_sec / BTC_UPDOWN_5M_PERIOD_SEC) * BTC_UPDOWN_5M_PERIOD_SEC;
    format!("btc-updown-5m-{window_start_sec}")
}

/// Конец 5m-окна в unix **ms** для slug `btc-updown-5m-{window_start_sec}` (стартует `window_start`).
fn btc_updown_5m_window_end_unix_ms_from_slug(slug: &str) -> Option<i64> {
    slug.strip_prefix("btc-updown-5m-")
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

/// Возвращает `(best_ask, recommended_buy_usd, min_order_size_shares)`.
///
/// Nominal BUY в USDC выбираем так: при худшем случае **`amount / price_cap`**, где
/// `price_cap = (best_ask + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp( … )`, тот же самый,
/// что в [`duel_leg_prep_for_outcome`], не оказывается **ниже `min_order_size`**.
///
/// Раньше брали `min_order_size * best_ask` без slippage cap — из-за этого
/// гард «ожидаемые shares ниже min» триггерил ранний выход без BUY (≈ `4.79` vs `5.0`).
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
    let buy_price_cap = (best_ask_f64 + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999);
    const CLOB_MIN_MARKETABLE_BUY_USD: f64 = 1.0;
    let raw_usd_floor = min_order_size * buy_price_cap + LIVE_DUEL_BUY_USD_HEADROOM;
    let rounded = (raw_usd_floor * 100.0).ceil() / 100.0;
    let amount = rounded.max(CLOB_MIN_MARKETABLE_BUY_USD);
    Ok((best_ask_f64, amount, min_order_size))
}
/// Комиссия уже в [`SingleOrderClobInvocationReport`]; цена BUY ≈ USD spent / NET shares.
fn implied_buy_price_per_share(rep: &SingleOrderClobInvocationReport) -> Option<f64> {
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
    amount: f64,
    price: f64,
    min_order_size_shares: f64,
}

async fn duel_leg_prep_for_outcome(
    account: &SharedAccount,
    slug: &str,
    currency_up_down_by_asset_id: &HashMap<String, CurrencyUpDownOutcome>,
    outcome: CurrencyUpDownOutcome,
) -> anyhow::Result<LegPrep> {
    let asset_id = currency_up_down_by_asset_id
        .iter()
        .find(|(_, o)| **o == outcome)
        .map(|(aid, _)| aid.clone())
        .with_context(|| {
            format!("нет outcome={outcome:?} в Gamma currency_up_down_by_asset_id для slug={slug}")
        })?;
    let (best_ask, amount, min_order_size_shares) =
        live_btc_updown_book_buy_floor(account, &asset_id, slug).await?;
    let price = (best_ask + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999);
    Ok(LegPrep {
        outcome,
        asset_id,
        amount,
        price,
        min_order_size_shares,
    })
}

/// Лимит-продажа maker на **+10%** к средней цене taker BUY.
const LIVE_MAKER_TP_MULT: f64 = 1.1;
/// Дополнительный USDC после `ceil` к центам над `min_order_size * price_cap`, чтобы гард
/// `amount / price_cap` надёжно не падал ниже `min_order_size_shares`.
const LIVE_DUEL_BUY_USD_HEADROOM: f64 = 0.03;
/// Повторы taker SELL при unwind противоположной ноги (FAK без матча и т.п.), подряд без пауз.
const UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS: u32 = 3;

fn invoke_wait_until_market_end_plus(market_end_unix_ms: Option<i64>) -> Duration {
    let now_ms = current_timestamp_ms();
    let deadline_ms = market_end_unix_ms
        .map(|end_ms| end_ms.saturating_add((LIVE_ORDER_HTTP_TIMEOUT_SEC * 1000) as i64))
        .unwrap_or(now_ms.saturating_add((LIVE_ORDER_HTTP_TIMEOUT_SEC * 1000) as i64));
    let wait_ms = deadline_ms.saturating_sub(now_ms).max(1_000);
    Duration::from_millis(wait_ms as u64)
}

#[derive(Clone, Debug, Default)]
struct DuelHarness {
    up_market_order_id: Option<String>,
    down_market_order_id: Option<String>,
    /// После invoke maker: `true` если `success` и нет `partial` (лимит полностью продал shares).
    up_maker_full_sell_ok: bool,
    down_maker_full_sell_ok: bool,
    /// Фактический остаток outcome-shares после BUY (NET) минус settled maker SELL (`making_amount`).
    up_shares_remaining: f64,
    down_shares_remaining: f64,
}

impl DuelHarness {
    fn new_shared() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(DuelHarness::default()))
    }

    fn set_resting_market_order_id(&mut self, o: CurrencyUpDownOutcome, id: Option<String>) {
        match o {
            CurrencyUpDownOutcome::Up => self.up_market_order_id = id,
            CurrencyUpDownOutcome::Down => self.down_market_order_id = id,
        }
    }

    fn set_bought_shares_net(&mut self, o: CurrencyUpDownOutcome, shares_net: f64) {
        match o {
            CurrencyUpDownOutcome::Up => self.up_shares_remaining = shares_net,
            CurrencyUpDownOutcome::Down => self.down_shares_remaining = shares_net,
        }
    }

    fn apply_maker_sell_settled(&mut self, o: CurrencyUpDownOutcome, sold_shares: f64) {
        let slot = match o {
            CurrencyUpDownOutcome::Up => &mut self.up_shares_remaining,
            CurrencyUpDownOutcome::Down => &mut self.down_shares_remaining,
        };
        *slot = (*slot - sold_shares).max(0.0);
    }

    fn set_maker_full_sell_ok(&mut self, o: CurrencyUpDownOutcome, full_sell_ok: bool) {
        match o {
            CurrencyUpDownOutcome::Up => self.up_maker_full_sell_ok = full_sell_ok,
            CurrencyUpDownOutcome::Down => self.down_maker_full_sell_ok = full_sell_ok,
        }
    }

    /// Противоположная нога уже завершила maker с полным исполнением.
    fn opposite_maker_full_sell_succeeded(&self, this_outcome: CurrencyUpDownOutcome) -> bool {
        match this_outcome.opposite() {
            CurrencyUpDownOutcome::Up => self.up_maker_full_sell_ok,
            CurrencyUpDownOutcome::Down => self.down_maker_full_sell_ok,
        }
    }
}

/// Снять противоположный maker с книги и при необходимости вымыть остаток shares тейкер-SELL.
/// Ошибки SELL/cancel (кроме уже залогированных веток) пишутся в tee и не пробрасываются.
async fn duel_unwind_opposite_maker_and_taker_flush(
    account: &SharedAccount,
    duel: &Arc<RwLock<DuelHarness>>,
    this_outcome: CurrencyUpDownOutcome,
    opposite_prep: &LegPrep,
    wall_ms: u64,
    slug: &str,
    market_start_unix_ms: Option<i64>,
) {
    let opposite = this_outcome.opposite();
    let maybe_oid = {
        let g = duel.read().await;
        match opposite {
            CurrencyUpDownOutcome::Up => g.up_market_order_id.clone(),
            CurrencyUpDownOutcome::Down => g.down_market_order_id.clone(),
        }
    };

    if let Some(oid) = maybe_oid.filter(|s| !s.trim().is_empty()) {
        match cancel_order_on_clob(
            account,
            None,
            CancelOrderRequest {
                order_id: oid.clone(),
                timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
            },
        )
        .await
        {
            Ok(res) => {
                poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: нога {this_outcome:?}: unwind — cancel противоположного {:?} maker slug={slug} order_id={} canceled={} err={:?}",
                    opposite,
                    res.order_id,
                    res.canceled,
                    res.error_msg,
                );
                if res.canceled {
                    duel.write()
                        .await
                        .set_resting_market_order_id(opposite, None);
                }
            }
            Err(err) => poly::test_tee_println!(
                "[от старта {wall_ms} ms] duel: нога {this_outcome:?}: unwind — cancel противоположного {:?} slug={slug} order_id={oid}: {err:#}",
                opposite,
            ),
        }
    }

    for attempt in 1..=UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS {
        let opp_shares = {
            let g = duel.read().await;
            match opposite {
                CurrencyUpDownOutcome::Up => g.up_shares_remaining,
                CurrencyUpDownOutcome::Down => g.down_shares_remaining,
            }
        };
        // SDK `polymarket-client-sdk-v2 v0.6.0-canary.1` валидирует `Amount::shares(...)`
        // ровно на 2 знака после запятой (`Unable to build Amount with 6 decimal points,
        // must be <= 2`), хотя on-chain ERC-1155 у Polymarket — 6 знаков. Поэтому
        // округляем **вниз** к 0.01 sh — `round` к 2 знакам недопустим: результат
        // может оказаться больше реального остатка и контракт реверт-нет с
        // `insufficient balance`. ~0.006 sh теряется по этой причине, не по нашей.
        let shares_to_sell = (opp_shares * 100.0).floor() / 100.0;
        if !(shares_to_sell > 0.0 && shares_to_sell.is_finite()) {
            break;
        }

        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: нога {this_outcome:?}: unwind — taker SELL противоположного {:?} slug={slug} asset_id={} shares={shares_to_sell:.2} (raw remaining={opp_shares:.6}) попытка {attempt}/{}",
            opposite,
            opposite_prep.asset_id,
            UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS,
        );
        let (sell_invoke_tx, mut sell_invoke_rx) = invoke_settlement_watch();
        if let Err(err) = post_order_on_clob(
            account,
            None,
            PostOrderRequest {
                asset_id: opposite_prep.asset_id.clone(),
                disable_http_settlement_poll_during_market: false,
                side: Side::Sell,
                role: OrderRole::Taker,
                amount: OrderAmount::Shares(shares_to_sell),
                price: None,
                max_slippage_pp: None,
                market_start_unix_ms,
                market_end_unix_ms: None,
                expiration: None,
                timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
                strict_book: None,
            },
            Box::new(move |rep| {
                let _ = sell_invoke_tx.send(Some(rep));
            }),
        )
        .await
        {
            poly::test_tee_println!(
                "[от старта {wall_ms} ms] duel: нога {:?} — ошибка unwind opposite (taker SELL post) попытка {attempt}/{}: {err:#}",
                this_outcome,
                UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS,
            );
            continue;
        }

        let sell_rep = match wait_invoke_settlement(
            &mut sell_invoke_rx,
            Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC.saturating_mul(30)),
        )
        .await
        {
            Some(rep) => rep,
            None => {
                poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: нога {:?} — ошибка unwind opposite: invoke taker SELL timeout попытка {attempt}/{}",
                    this_outcome,
                    UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS,
                );
                continue;
            }
        };
        if sell_rep.success {
            let sold_shares = match &sell_rep.making_amount {
                OrderAmount::Shares(s) if s.is_finite() && *s >= 0.0 => *s,
                _ => 0.0,
            };
            duel.write()
                .await
                .apply_maker_sell_settled(opposite, sold_shares);
            poly::test_tee_println!(
                "[от старта {wall_ms} ms] duel: нога {this_outcome:?}: unwind — taker SELL противоположного {:?} итог попытка {attempt}/{}: success={} partial={} making={:?} taking={:?} err={:?}",
                opposite,
                UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS,
                sell_rep.success,
                sell_rep.partial,
                sell_rep.making_amount,
                sell_rep.taking_amount,
                sell_rep.error_msg,
            );
            break;
        }
    }
}

/// Продаёт свою ногу `outcome_t` через **taker FAK** (повтор до
/// [`UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS`] — как unwind противоположной ноги).
/// Остаток читается из `duel` на каждой итерации; SDK ≤2 знаков → floor к 0.01 sh.
async fn duel_self_taker_sell_flush(
    account: &SharedAccount,
    duel: &Arc<RwLock<DuelHarness>>,
    outcome_t: CurrencyUpDownOutcome,
    asset_id: &str,
    wall_ms: u64,
    slug: &str,
    market_start_unix_ms: Option<i64>,
) {
    for attempt in 1..=UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS {
        let my_shares = {
            let g = duel.read().await;
            match outcome_t {
                CurrencyUpDownOutcome::Up => g.up_shares_remaining,
                CurrencyUpDownOutcome::Down => g.down_shares_remaining,
            }
        };
        let shares_to_sell = (my_shares * 100.0).floor() / 100.0;
        if !(shares_to_sell > 0.0 && shares_to_sell.is_finite()) {
            break;
        }

        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: нога {outcome_t:?}: self-flush — taker FAK SELL slug={slug} asset_id={asset_id} shares={shares_to_sell:.2} (raw remaining={my_shares:.6}) попытка {attempt}/{}",
            UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS,
        );
        let (sell_invoke_tx, mut sell_invoke_rx) = invoke_settlement_watch();
        if let Err(err) = post_order_on_clob(
            account,
            None,
            PostOrderRequest {
                asset_id: asset_id.to_string(),
                disable_http_settlement_poll_during_market: false,
                side: Side::Sell,
                role: OrderRole::Taker,
                amount: OrderAmount::Shares(shares_to_sell),
                price: None,
                max_slippage_pp: None,
                market_start_unix_ms,
                market_end_unix_ms: None,
                expiration: None,
                timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
                strict_book: None,
            },
            Box::new(move |rep| {
                let _ = sell_invoke_tx.send(Some(rep));
            }),
        )
        .await
        {
            poly::test_tee_println!(
                "[от старта {wall_ms} ms] duel: нога {outcome_t:?}: self-flush — ошибка post taker SELL попытка {attempt}/{}: {err:#}",
                UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS,
            );
            continue;
        }
        let sell_rep = match wait_invoke_settlement(
            &mut sell_invoke_rx,
            Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC.saturating_mul(30)),
        )
        .await
        {
            Some(rep) => rep,
            None => {
                poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: нога {outcome_t:?}: self-flush — invoke taker SELL timeout попытка {attempt}/{}",
                    UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS,
                );
                continue;
            }
        };
        if sell_rep.success {
            let sold_shares = match &sell_rep.making_amount {
                OrderAmount::Shares(s) if s.is_finite() && *s >= 0.0 => *s,
                _ => 0.0,
            };
            duel.write()
                .await
                .apply_maker_sell_settled(outcome_t, sold_shares);
            poly::test_tee_println!(
                "[от старта {wall_ms} ms] duel: нога {outcome_t:?}: self-flush итог попытка {attempt}/{}: success={} partial={} making={:?} taking={:?} err={:?}",
                UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS,
                sell_rep.success,
                sell_rep.partial,
                sell_rep.making_amount,
                sell_rep.taking_amount,
                sell_rep.error_msg,
            );
            break;
        }
    }
}

async fn duel_post_buy_then_maker(
    account: SharedAccount,
    duel: Arc<RwLock<DuelHarness>>,
    prep: LegPrep,
    slug: String,
    market_start_unix_ms: Option<i64>,
    wall_anchor: Arc<std::time::Instant>,
    opposite_prep: LegPrep,
) -> anyhow::Result<()> {
    let aid = prep.asset_id.clone();
    let amount = prep.amount;
    let price = prep.price;
    let outcome_t = prep.outcome;

    let (buy_invoke_tx, mut buy_invoke_rx) = invoke_settlement_watch();
    post_order_on_clob(
        &account,
        None,
        PostOrderRequest {
            asset_id: aid.clone(),
            disable_http_settlement_poll_during_market: false,
            side: Side::Buy,
            role: OrderRole::Taker,
            amount: OrderAmount::UsdNotional(amount),
            price: Some(price),
            max_slippage_pp: None,
            market_start_unix_ms,
            market_end_unix_ms: None,
            expiration: None,
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
            strict_book: None,
        },
        Box::new(move |buy_rep| {
            let _ = buy_invoke_tx.send(Some(buy_rep));
        }),
    )
    .await
    .with_context(|| format!("duel BUY taker {:?}", outcome_t))?;

    let buy_rep = wait_invoke_settlement(
        &mut buy_invoke_rx,
        Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC.saturating_mul(30)),
    )
    .await
    .ok_or_else(|| anyhow::anyhow!("duel BUY taker {:?}: invoke timeout", outcome_t))?;

    let wall_ms = wall_anchor.elapsed().as_millis() as u64;
    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel: ВХОД в invoke taker BUY {:?} slug={} asset_id={} (целевой amount ≈{amount:.4} USDC price={price:.5})",
        outcome_t,
        slug,
        aid,
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
        duel_unwind_opposite_maker_and_taker_flush(
            &account,
            &duel,
            outcome_t,
            &opposite_prep,
            wall_ms,
            slug.as_str(),
            market_start_unix_ms,
        )
        .await;
        return Ok(());
    }
    let shares_net = match buy_rep.taking_amount {
        OrderAmount::Shares(s) => s,
        OrderAmount::UsdNotional(_) => {
            poly::test_tee_println!(
                "[от старта {wall_ms} ms] duel: BUY {:?}: ожидались Shares в taking_amount — без maker",
                outcome_t,
            );
            duel_unwind_opposite_maker_and_taker_flush(
                &account,
                &duel,
                outcome_t,
                &opposite_prep,
                wall_ms,
                slug.as_str(),
                market_start_unix_ms,
            )
            .await;
            return Ok(());
        }
    };
    if !(shares_net > 0.0 && shares_net.is_finite()) {
        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: BUY {:?}: невалидный shares_net={} — maker не ставится",
            outcome_t,
            shares_net,
        );
        duel_unwind_opposite_maker_and_taker_flush(
            &account,
            &duel,
            outcome_t,
            &opposite_prep,
            wall_ms,
            slug.as_str(),
            market_start_unix_ms,
        )
        .await;
        return Ok(());
    }
    duel.write()
        .await
        .set_bought_shares_net(outcome_t, shares_net);
    let shares_floor = (shares_net * 100.0).floor() / 100.0;
    let Some(implied_buy_price) = implied_buy_price_per_share(&buy_rep) else {
        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: BUY {:?}: не смогли восстановить USD/share — без maker",
            outcome_t,
        );
        duel_unwind_opposite_maker_and_taker_flush(
            &account,
            &duel,
            outcome_t,
            &opposite_prep,
            wall_ms,
            slug.as_str(),
            market_start_unix_ms,
        )
        .await;
        return Ok(());
    };
    let maker_price_raw = implied_buy_price * LIVE_MAKER_TP_MULT;

    poly::test_tee_println!(
        "[от старта {wall_ms} ms] duel: BUY {:?} зачтено для maker: NET shares {:.6} → floor {:.2}; \
         сырая TP-цена до тика на CLOB (`post_order_on_clob`) ≈ {:.6}",
        outcome_t,
        shares_net,
        shares_floor,
        maker_price_raw,
    );

    // Гард: shares_floor < min_order_size — резидентный лимит CLOB пройти не сможет
    // (`Size lower than the minimum: N`). Тут maker не выставляем; вместо этого
    // сразу же `taker FAK SELL` своей ноги (FAK не подчиняется этому валидатору)
    // и закрываем противоположную через unwind. Логика взята из наблюдения:
    // см. `last_live_duel_maker_race.txt` строка 13 (maker-реджект 4.8 < 5)
    // и успешный браузерный curl `orderType=FAK` на ту же 4.8 sh — FAK проходит.
    if shares_floor < prep.min_order_size_shares {
        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: нога {outcome_t:?}: shares_floor={shares_floor:.2} < min_order_size={:.2} — \
             maker-TP не выставляем (CLOB реджект); fallback: self-flush taker FAK SELL + unwind противоположной",
            prep.min_order_size_shares,
        );
        duel_self_taker_sell_flush(
            &account,
            &duel,
            outcome_t,
            aid.as_str(),
            wall_ms,
            slug.as_str(),
            market_start_unix_ms,
        )
        .await;
        duel_unwind_opposite_maker_and_taker_flush(
            &account,
            &duel,
            outcome_t,
            &opposite_prep,
            wall_ms,
            slug.as_str(),
            market_start_unix_ms,
        )
        .await;
        return Ok(());
    }

    if duel
        .read()
        .await
        .opposite_maker_full_sell_succeeded(outcome_t)
    {
        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: maker {:?} не выставляем — противоположная нога уже full_sell_ok=true",
            outcome_t,
        );
        return Ok(());
    }

    let market_end_unix_ms =
        btc_updown_5m_window_end_unix_ms_from_slug(slug.as_str()).or_else(|| {
            let ms = current_timestamp_ms();
            let poly_sec = ms / 1000;
            let ws = (poly_sec / BTC_UPDOWN_5M_PERIOD_SEC) * BTC_UPDOWN_5M_PERIOD_SEC;
            Some((ws.saturating_add(BTC_UPDOWN_5M_PERIOD_SEC)).saturating_mul(1000))
        });

    let (mk_invoke_tx, mut mk_invoke_rx) = invoke_settlement_watch();
    let post_res = post_order_on_clob(
        &account,
        None,
        PostOrderRequest {
            asset_id: aid.clone(),
            disable_http_settlement_poll_during_market: false,
            side: Side::Sell,
            role: OrderRole::Maker,
            amount: OrderAmount::Shares(shares_floor),
            price: Some(maker_price_raw),
            max_slippage_pp: None,
            market_start_unix_ms,
            market_end_unix_ms,
            expiration: None,
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
            strict_book: None,
        },
        Box::new(move |rep| {
            let _ = mk_invoke_tx.send(Some(rep));
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

    duel.write()
        .await
        .set_resting_market_order_id(outcome_t, resting_oid.clone());

    if let Some(oid) = resting_oid.as_ref() {
        poly::test_tee_println!(
            "[от старта {wall_ms} ms] duel: maker {:?} принят книгой order_id={oid} \
             сырая лимит-цена ≈{maker_price_raw:.6} (нормализация тика 0.01 в `post_order_on_clob`) \
             shares={shares_floor:.2} market_end_unix_ms={market_end_unix_ms:?}",
            outcome_t,
        );
    }

    let maker_invoke_wait = invoke_wait_until_market_end_plus(market_end_unix_ms);
    let maker_fin = wait_invoke_settlement(&mut mk_invoke_rx, maker_invoke_wait).await;

    match maker_fin {
        Some(maker_rep) => {
            let sold_shares = match &maker_rep.making_amount {
                OrderAmount::Shares(s) if s.is_finite() && *s >= 0.0 => *s,
                _ => 0.0,
            };
            let full_sell_ok = maker_rep.success && !maker_rep.partial;
            {
                let mut duel_guard = duel.write().await;
                duel_guard.apply_maker_sell_settled(outcome_t, sold_shares);
                duel_guard.set_maker_full_sell_ok(outcome_t, full_sell_ok);
            }
            poly::test_tee_println!(
                "[maker POST {:?}] invoke финала лимита: success={}, partial={}, order_id={:?}, making={:?}, taking={:?}, err={:?}",
                outcome_t,
                maker_rep.success,
                maker_rep.partial,
                maker_rep.order_id,
                maker_rep.making_amount,
                maker_rep.taking_amount,
                maker_rep.error_msg,
            );
            if resting_oid.is_none() {
                poly::test_tee_println!(
                    "[от старта {wall_ms} ms] duel: maker {:?} не на книге — получен только отчёт агрегатора (success={}, err={:?}); гонку не ждём с этой ноги",
                    outcome_t,
                    maker_rep.success,
                    maker_rep.error_msg,
                );
            }
        }
        None => poly::test_tee_println!(
            "duel: {:?} maker invoke timeout до финала агрегатора ({maker_invoke_wait:?})",
            outcome_t,
        ),
    }
    duel_unwind_opposite_maker_and_taker_flush(
        &account,
        &duel,
        outcome_t,
        &opposite_prep,
        wall_ms,
        slug.as_str(),
        market_start_unix_ms,
    )
    .await;
    Ok(())
}

/// Live: параллельный taker BUY Up+Down и maker лимит +10% к среднему BUY; при выходе ноги — cancel+flush противоположной.
/// Логи — `test_tee_*`, `[order_invoke/..]` — stream tee.
///
/// ```bash
/// POLY_PRIVATE_KEY=0x… \
///     cargo test --test live_duel_up_down_maker_race_tp10 -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "live duel: POLY_PRIVATE_KEY + pUSD; параллельные BUY Up/Down и maker TP 10%"]
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

    let test_log_path = poly::path_config::xframes_path("last_live_duel_maker_race.txt");
    poly::tee_log::init_test_tee_log_file(&test_log_path, "live_duel_up_down_maker_race_tp10")?;

    let stream_log_path = poly::path_config::xframes_path("last_stream.txt");
    poly::tee_log::init_stream_tee_log_file(&stream_log_path)?;
    let user_stream_log_path = poly::path_config::xframes_path("last_user_stream.txt");
    poly::tee_log::init_user_stream_tee_log_file(&user_stream_log_path)?;
    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel ноги Up+Down: `[order_invoke/...]` tee → {}; `[user_ws]` tee → {}",
        stream_log_path.display(),
        user_stream_log_path.display(),
    );

    let account = Account::new_shared();
    let country_and_ip = detect_country_and_ip(account.http.as_ref())
        .await
        .ok_or_else(|| {
            let (dt, wall) = evt_ms!(last_evt, t0);
            anyhow::anyhow!(
                "[от старта {wall} ms | с прошлого {dt} ms] Polymarket geoblock: не удалось GET https://polymarket.com/api/geoblock"
            )
        })?;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        !country_and_ip.blocked,
        "[от старта {wall} ms | с прошлого {dt} ms] Polymarket geoblock: торговля с этого региона заблокирована \
         (country={:?}, region={:?}, ip={:?})",
        country_and_ip.country,
        country_and_ip.region,
        country_and_ip.ip,
    );
    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel ноги Up+Down: country_and_ip={country_and_ip:?}",
    );

    let private_key_set = std::env::var(POLY_PRIVATE_KEY_ENV)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .is_some();
    if !private_key_set {
        let (dt, wall) = evt_ms!(last_evt, t0);
        poly::test_tee_println!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel ноги Up+Down: {POLY_PRIVATE_KEY_ENV} не задан, тест пропущен",
        );
        poly::tee_log::finish_test_tee_log();
        poly::tee_log::finish_stream_tee_log();
        poly::tee_log::finish_user_stream_tee_log();
        return Ok(());
    }

    let now_ms = current_timestamp_ms();
    let market_start_unix_ms =
        Some((now_ms / 1000 / BTC_UPDOWN_5M_PERIOD_SEC) * BTC_UPDOWN_5M_PERIOD_SEC * 1000);
    let slug = current_btc_updown_5m_slug(now_ms);
    let gamma = fetch_gamma_event_data_for_gamma_client(account.gamma.as_ref(), &slug).await?;
    let currency_up_down_by_asset_id = &gamma.currency_up_down_by_asset_id;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        !currency_up_down_by_asset_id.is_empty(),
        "[от старта {wall} ms | с прошлого {dt} ms] Gamma не вернул clobTokenIds для slug={slug}",
    );

    try_authenticate_clob_for_heartbeats(&account).await;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        account.clob_authed.load().is_some(),
        "[от старта {wall} ms | с прошлого {dt} ms] CLOB auth не поднялся — проверьте {POLY_PRIVATE_KEY_ENV} и логи [heartbeat]",
    );

    spawn_heartbeat(account.clone());
    spawn_user_ws_listener(account.clone());
    tokio::time::sleep(Duration::from_secs(LIVE_TEST_USER_WS_WARMUP_SECS)).await;
    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel ноги Up+Down: slug={slug} user-WS warmup {}s",
        LIVE_TEST_USER_WS_WARMUP_SECS,
    );

    let prep_cu_up = currency_up_down_by_asset_id.clone();
    let prep_cu_down = currency_up_down_by_asset_id.clone();
    let prep_account_up = account.clone();
    let prep_account_down = account.clone();
    let prep_slug_up = slug.clone();
    let prep_slug_down = slug.clone();
    let prep_up_h = tokio::spawn(async move {
        duel_leg_prep_for_outcome(
            &prep_account_up,
            &prep_slug_up,
            &prep_cu_up,
            CurrencyUpDownOutcome::Up,
        )
        .await
    });
    let prep_down_h = tokio::spawn(async move {
        duel_leg_prep_for_outcome(
            &prep_account_down,
            &prep_slug_down,
            &prep_cu_down,
            CurrencyUpDownOutcome::Down,
        )
        .await
    });
    let (prep_up, prep_down) = tokio::join!(prep_up_h, prep_down_h);
    let prep_up = prep_up
        .map_err(|e| anyhow::anyhow!("live_duel prep UP tokio::spawn JoinError: {e}"))?
        .with_context(|| format!("live_duel prep UP slug={slug}"))?;
    let prep_down = prep_down
        .map_err(|e| anyhow::anyhow!("live_duel prep DOWN tokio::spawn JoinError: {e}"))?
        .with_context(|| format!("live_duel prep DOWN slug={slug}"))?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel: UP amount={:.4} asset={} (min_order_size={:.2}) ; DOWN amount={:.4} asset={} (min_order_size={:.2})",
        prep_up.amount,
        prep_up.asset_id,
        prep_up.min_order_size_shares,
        prep_down.amount,
        prep_down.asset_id,
        prep_down.min_order_size_shares,
    );

    // Гард: оцениваем сверху худшее количество shares от taker BUY = `amount / price`
    // (worst case: всё купили по slippage cap'у `prep.price`). Если хоть одна нога не
    // дотянет до своего `min_order_size`, **обе** ноги не запускаем — иначе в
    // `duel_post_buy_then_maker` сработает реджект maker-TP (`Size lower than min`),
    // и мы будем выкручиваться через self-flush + unwind. Лучше не заходить вовсе.
    let expected_up = prep_up.amount / prep_up.price;
    let expected_dn = prep_down.amount / prep_down.price;
    if expected_up < prep_up.min_order_size_shares || expected_dn < prep_down.min_order_size_shares
    {
        let (dt, wall) = evt_ms!(last_evt, t0);
        poly::test_tee_println!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_duel: НЕ заходим — \
             ожидаемые shares в worst-case (amount/price) ниже min_order_size \
             (UP expected={expected_up:.4} vs min={:.2}; DOWN expected={expected_dn:.4} vs min={:.2}); \
             выходим без BUY",
            prep_up.min_order_size_shares,
            prep_down.min_order_size_shares,
        );
        poly::tee_log::finish_stream_tee_log();
        poly::tee_log::finish_user_stream_tee_log();
        poly::tee_log::finish_test_tee_log();
        return Ok(());
    }

    let duel_h = DuelHarness::new_shared();

    let opposite_prep_for_up_leg = prep_down.clone();
    let opposite_prep_for_down_leg = prep_up.clone();

    let h_up = {
        let account = account.clone();
        let duel_h = Arc::clone(&duel_h);
        let slug = slug.clone();
        let wall_anchor = Arc::clone(&wall_anchor);
        tokio::spawn(async move {
            duel_post_buy_then_maker(
                account,
                duel_h,
                prep_up,
                slug,
                market_start_unix_ms,
                wall_anchor,
                opposite_prep_for_up_leg,
            )
            .await
        })
    };
    let h_dn = {
        let account = account.clone();
        let duel_h = Arc::clone(&duel_h);
        let slug = slug.clone();
        let wall_anchor = Arc::clone(&wall_anchor);
        tokio::spawn(async move {
            duel_post_buy_then_maker(
                account,
                duel_h,
                prep_down,
                slug,
                market_start_unix_ms,
                wall_anchor,
                opposite_prep_for_down_leg,
            )
            .await
        })
    };
    let (j_up, j_dn) = tokio::join!(h_up, h_dn);
    j_up.map_err(|e| anyhow::anyhow!("live_duel tokio::spawn UP JoinError: {e}"))?
        .with_context(|| format!("live_duel BUY→maker нога UP slug={slug}"))?;
    j_dn.map_err(|e| anyhow::anyhow!("live_duel tokio::spawn DOWN JoinError: {e}"))?
        .with_context(|| format!("live_duel BUY→maker нога DOWN slug={slug}"))?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    let snap = duel_h.read().await.clone();
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_duel ноги Up+Down: после ног BUY+maker slug={slug} state={snap:?}",
    );

    poly::tee_log::finish_stream_tee_log();
    poly::tee_log::finish_user_stream_tee_log();
    poly::tee_log::finish_test_tee_log();
    Ok(())
}
