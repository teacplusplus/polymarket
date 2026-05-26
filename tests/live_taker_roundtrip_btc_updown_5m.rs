//! Live taker roundtrip integration test (BTC 5m up/down).

use anyhow::Context;
use poly::account::{
    Account, POLY_PRIVATE_KEY_ENV, SharedAccount, spawn_heartbeat,
    try_authenticate_clob_for_heartbeats,
};
use poly::account_order::{
    best_ask_sdk, invoke_settlement_watch, post_order_on_clob, wait_invoke_settlement, OrderAmount,
    OrderRole, PostOrderRequest,
};
use poly::account_ws::spawn_user_ws_listener;
use poly::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT;
use poly::util::{
    current_timestamp_ms, detect_country_and_ip, fetch_gamma_event_data_for_gamma_client,
};
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::Side;
use polymarket_client_sdk::types::U256;
use std::str::FromStr;
use std::time::Duration;

const BTC_UPDOWN_5M_PERIOD_SEC: i64 = 300;
const LIVE_ORDER_HTTP_TIMEOUT_SEC: u64 = 20;
const LIVE_TEST_USER_WS_WARMUP_SECS: u64 = 3;

fn current_btc_updown_5m_slug(now_ms: i64) -> String {
    let poly_sec = now_ms / 1000;
    let window_start_sec = (poly_sec / BTC_UPDOWN_5M_PERIOD_SEC) * BTC_UPDOWN_5M_PERIOD_SEC;
    format!("btc-updown-5m-{window_start_sec}")
}

fn decimal_to_f64(d: &polymarket_client_sdk::types::Decimal) -> anyhow::Result<f64> {
    d.to_string()
        .parse::<f64>()
        .map_err(|err| anyhow::anyhow!("Decimal {d} → f64: {err}"))
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
    const CLOB_MIN_MARKETABLE_BUY_USD: f64 = 1.0;
    let raw = min_order_size * best_ask_f64;
    let rounded = (raw * 100.0).ceil() / 100.0;
    let market_floor_buy_usd = rounded.max(CLOB_MIN_MARKETABLE_BUY_USD);
    Ok((min_order_size, best_ask_f64, market_floor_buy_usd))
}

/// Live BUY→SELL taker по текущему 5m BTC up/down (минимальный допустимый CLOB notional).
///
/// ```bash
/// POLY_PRIVATE_KEY=0x… \
///     cargo test --test live_taker_roundtrip_btc_updown_5m -- --ignored --nocapture
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

    let test_log_path = std::path::Path::new("xframes/last_live_taker_roundtrip.txt");
    poly::tee_log::init_test_tee_log_file(test_log_path, "live_taker_roundtrip_btc_updown_5m")?;

    let stream_log_path = std::path::Path::new("xframes/last_stream.txt");
    poly::tee_log::init_stream_tee_log_file(
        stream_log_path,
    )?;
    let user_stream_log_path = std::path::Path::new("xframes/last_user_stream.txt");
    poly::tee_log::init_user_stream_tee_log_file(
        user_stream_log_path,
    )?;

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
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: country_and_ip={country_and_ip:?}",
    );

    let private_key_set = std::env::var(POLY_PRIVATE_KEY_ENV)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .is_some();
    if !private_key_set {
        let (dt, wall) = evt_ms!(last_evt, t0);
        poly::test_tee_println!(
            "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: {POLY_PRIVATE_KEY_ENV} не задан, тест пропущен",
        );
        poly::tee_log::finish_test_tee_log();
        poly::tee_log::finish_stream_tee_log();
        poly::tee_log::finish_user_stream_tee_log();
        return Ok(());
    }
    let slug = current_btc_updown_5m_slug(current_timestamp_ms());
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
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: user-WS warmup {LIVE_TEST_USER_WS_WARMUP_SECS}s",
    );

    let mut best: Option<(String, f64, f64, f64)> = None;
    for asset_id in currency_up_down_by_asset_id.keys() {
        let (min_order_size, best_ask_f64, market_floor_buy_usd) =
            live_btc_updown_book_buy_floor(&account, asset_id, &slug).await?;
        let take = match &best {
            None => true,
            Some((_, _, _, prev_market_floor_buy_usd)) => {
                market_floor_buy_usd < *prev_market_floor_buy_usd - 1e-12
            }
        };
        if take {
            best = Some((
                asset_id.to_string(),
                min_order_size,
                best_ask_f64,
                market_floor_buy_usd,
            ));
        }
    }
    let (asset_id, min_order_size, best_ask_f64, market_floor_buy_usd) =
        best.unwrap_or_else(|| {
            let (dt, wall) = evt_ms!(last_evt, t0);
            panic!(
                "[от старта {wall} ms | с прошлого {dt} ms] currency_up_down_by_asset_id не пустой — выше ensure: best=None"
            )
        });
    let worst_acceptable_buy = (best_ask_f64 + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999);

    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: slug={slug}, asset_id={asset_id}, \
         min_order_size={min_order_size:.4}, best_ask={best_ask_f64:.4}, \
         market_floor_buy_usd={market_floor_buy_usd:.4} worst_acceptable_buy={worst_acceptable_buy:.4}",
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: taker BUY market_floor_buy_usd≈{market_floor_buy_usd:.4}",
    );

    let (buy_invoke_tx, mut buy_invoke_rx) = invoke_settlement_watch();
    post_order_on_clob(
        &account,
        None,
        PostOrderRequest {
            asset_id: asset_id.clone(),
            side: Side::Buy,
            role: OrderRole::Taker,
            amount: OrderAmount::UsdNotional(market_floor_buy_usd),
            price: Some(worst_acceptable_buy),
            max_slippage_pp: None,
            expiration: None,
            market_end_unix_ms: None,
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
            strict_book: None,
        },
        Box::new(move |rep| {
            let _ = buy_invoke_tx.send(Some(rep));
        }),
    )
    .await
    .with_context(|| {
        let (dt, wall) = evt_ms!(last_evt, t0);
        format!(
            "[от старта {wall} ms | с прошлого {dt} ms] BUY taker slug={slug} asset_id={asset_id}"
        )
    })?;
    let buy_single_order_clob_invocation_report = wait_invoke_settlement(
        &mut buy_invoke_rx,
        Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC.saturating_mul(30)),
    )
    .await
    .ok_or_else(|| {
        let (dt, wall) = evt_ms!(last_evt, t0);
        anyhow::anyhow!("[от старта {wall} ms | с прошлого {dt} ms] BUY taker invoke timeout")
    })?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: BUY making_amount={:?} taking_amount={:?} order_id={:?} partial={} market_floor_buy_usd≈{market_floor_buy_usd:.4}",
        buy_single_order_clob_invocation_report.making_amount,
        buy_single_order_clob_invocation_report.taking_amount,
        buy_single_order_clob_invocation_report.order_id,
        buy_single_order_clob_invocation_report.partial,
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        buy_single_order_clob_invocation_report
            .order_id
            .as_deref()
            .is_some_and(|id| !id.is_empty()),
        "[от старта {wall} ms | с прошлого {dt} ms] пустой order_id после BUY"
    );
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        buy_single_order_clob_invocation_report.success,
        "[от старта {wall} ms | с прошлого {dt} ms] BUY taker финал не успех: order_id={:?}, partial={}, error_msg={:?}",
        buy_single_order_clob_invocation_report.order_id,
        buy_single_order_clob_invocation_report.partial,
        buy_single_order_clob_invocation_report.error_msg,
    );

    let taking_amount_shares_net = match buy_single_order_clob_invocation_report.taking_amount {
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
        taking_amount_shares_net > 0.0 && taking_amount_shares_net.is_finite(),
        "[от старта {wall} ms | с прошлого {dt} ms] BUY taker не дал shares в taking_amount: {:?}, order_id={:?}",
        buy_single_order_clob_invocation_report.taking_amount,
        buy_single_order_clob_invocation_report.order_id,
    );

    let shares_to_sell = (taking_amount_shares_net * 100.0).floor() / 100.0;
    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        shares_to_sell >= min_order_size,
        "[от старта {wall} ms | с прошлого {dt} ms] после округления вниз до 0.01 shares_to_sell={shares_to_sell:.2} < \
         min_order_size={min_order_size:.4}; taking_amount_shares_net={taking_amount_shares_net:.6}",
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: taker SELL shares_to_sell={shares_to_sell:.2} taking_amount_shares_net={taking_amount_shares_net:.6}",
    );

    let (sell_invoke_tx, mut sell_invoke_rx) = invoke_settlement_watch();
    post_order_on_clob(
        &account,
        None,
        PostOrderRequest {
            asset_id: asset_id.clone(),
            side: Side::Sell,
            role: OrderRole::Taker,
            amount: OrderAmount::Shares(shares_to_sell),
            price: None,
            max_slippage_pp: None,
            expiration: None,
            market_end_unix_ms: None,
            timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
            strict_book: None,
        },
        Box::new(move |rep| {
            let _ = sell_invoke_tx.send(Some(rep));
        }),
    )
    .await
    .with_context(|| {
        let (dt, wall) = evt_ms!(last_evt, t0);
        format!(
            "[от старта {wall} ms | с прошлого {dt} ms] SELL taker slug={slug} asset_id={asset_id}"
        )
    })?;
    let sell_single_order_clob_invocation_report = wait_invoke_settlement(
        &mut sell_invoke_rx,
        Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC.saturating_mul(30)),
    )
    .await
    .ok_or_else(|| {
        let (dt, wall) = evt_ms!(last_evt, t0);
        anyhow::anyhow!("[от старта {wall} ms | с прошлого {dt} ms] SELL taker invoke timeout")
    })?;

    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m: SELL making_amount={:?} taking_amount={:?} order_id={:?} partial={}",
        sell_single_order_clob_invocation_report.making_amount,
        sell_single_order_clob_invocation_report.taking_amount,
        sell_single_order_clob_invocation_report.order_id,
        sell_single_order_clob_invocation_report.partial,
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    anyhow::ensure!(
        sell_single_order_clob_invocation_report.success,
        "[от старта {wall} ms | с прошлого {dt} ms] SELL taker финал не успех: order_id={:?}, partial={}, error_msg={:?}",
        sell_single_order_clob_invocation_report.order_id,
        sell_single_order_clob_invocation_report.partial,
        sell_single_order_clob_invocation_report.error_msg,
    );

    let (dt, wall) = evt_ms!(last_evt, t0);
    poly::test_tee_println!(
        "[от старта {wall} ms | с прошлого {dt} ms] live_taker_roundtrip_btc_updown_5m OK: buy order_id={:?} sell order_id={:?} \
         market_floor_buy_usd={market_floor_buy_usd:.4} shares_to_sell={shares_to_sell:.4}",
        buy_single_order_clob_invocation_report.order_id,
        sell_single_order_clob_invocation_report.order_id,
    );
    poly::tee_log::finish_stream_tee_log();
    poly::tee_log::finish_user_stream_tee_log();
    poly::tee_log::finish_test_tee_log();
    Ok(())
}
