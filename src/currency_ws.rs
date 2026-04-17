use crate::project_manager::ProjectManager;
use crate::run_log;
use crate::util::current_timestamp_ms;
use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::sync::Arc;
use tokio::time::{interval, Duration, MissedTickBehavior};
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Аналог market CLOB WS — endpoint RTDS.
const POLYMARKET_RTDS_WS_URL: &str = "wss://ws-live-data.polymarket.com";
const RTDS_RECONNECT_DELAY_SECS: u64 = 3;
const RTDS_PING_INTERVAL_SECS: u64 = 5;
const RTDS_PAYLOAD_TS_HISTORY_MS: i64 = 20 * 60 * 1000;
/// Как часто watchdog проверяет свежесть данных (не чаще раза в 10 секунд).
const RTDS_WATCHDOG_INTERVAL_SECS: u64 = 10;
/// Максимальный возраст последнего payload_ts_ms: если последние данные старше — форсируем реконнект.
const RTDS_STALE_PRICE_MAX_AGE_MS: i64 = 45_000;
/// Если последний ключ [`ProjectManager::rtds_currency_prices_by_ms`] старше стены более чем на это — [`crate::xframe::XFrame::stable`] = `false`.
pub const RTDS_MS_MAX_LAG_FOR_STABLE_FRAME: i64 = 45_000;

/// Символ подписки RTDS `crypto_prices`: `{currency}` в нижнем регистре + `usdt` (как у Binance spot), например `eth` → `ethusdt`.
pub fn rtds_spot_pair_symbol(currency: &str) -> String {
    format!("{}usdt", currency.to_lowercase())
}

pub fn spawn_rtds_currency_pipeline(project_manager: Arc<ProjectManager>) {
    tokio::spawn(run_rtds_spot_ws_loop(project_manager.clone()));
    tokio::spawn(run_spot_second_sampler(project_manager));
}

async fn run_spot_second_sampler(project_manager: Arc<ProjectManager>) {
    let mut tick = interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let bucket_sec = current_timestamp_ms() / 1000;
        let tail = {
            let rtds_currency_prices_by_ms_lock = project_manager
                .rtds_currency_prices_by_ms
                .read()
                .await;
            rtds_currency_prices_by_ms_lock.iter().next_back().map(|(&ts_ms, &price)| (ts_ms, price))
        };
        if let Some((price_ts_ms, price)) = tail {
            let mut map = project_manager.rtds_currency_prices_by_sec.write().await;
            map.insert(bucket_sec, price);
            let bars_in_history = map.len();
            let pair_symbol = rtds_spot_pair_symbol(project_manager.currency.as_str());
            let wall_ms = current_timestamp_ms();
            run_log::rtds_currency_sec_bar_inserted(
                &pair_symbol,
                bucket_sec,
                price,
                price_ts_ms,
                wall_ms,
                bars_in_history,
            );
            while map.len() > (RTDS_PAYLOAD_TS_HISTORY_MS / 1000) as usize {
                let Some(first_key) = map.keys().next().copied() else {
                    break;
                };
                map.remove(&first_key);
            }
        }
    }
}

async fn run_rtds_spot_ws_loop(project_manager: Arc<ProjectManager>) {
    loop {
        if let Err(err) = run_rtds_spot_session(&project_manager).await {
            let pair = rtds_spot_pair_symbol(project_manager.currency.as_str());
            eprintln!("rtds ({pair}): сессия завершилась: {err}");
        }
        tokio::time::sleep(Duration::from_secs(RTDS_RECONNECT_DELAY_SECS)).await;
    }
}

async fn run_rtds_spot_session(project_manager: &Arc<ProjectManager>) -> Result<()> {
    let (ws_stream, _http_response) = connect_async(POLYMARKET_RTDS_WS_URL)
        .await
        .context("rtds connect_async")?;
    let (mut write, mut read) = ws_stream.split();

    let pair_symbol = rtds_spot_pair_symbol(project_manager.currency.as_str());
    let filters = serde_json::json!({ "symbol": pair_symbol }).to_string();
    let subscribe = serde_json::json!({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices",
            "type": "update",
            "filters": filters
        }]
    });
    write
        .send(Message::Text(subscribe.to_string()))
        .await
        .context("rtds subscribe")?;
    eprintln!("rtds ({pair_symbol}): подключились");

    let mut ping = interval(Duration::from_secs(RTDS_PING_INTERVAL_SECS));
    let mut watchdog = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_secs(RTDS_WATCHDOG_INTERVAL_SECS),
        Duration::from_secs(RTDS_WATCHDOG_INTERVAL_SECS),
    );
    watchdog.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ping.tick() => {
                write.send(Message::Text("PING".into()))
                    .await
                    .context("rtds PING")?;
            }
            _ = watchdog.tick() => {
                let latest = {
                    let rtds_currency_prices_by_ms_lock = project_manager.rtds_currency_prices_by_ms.read().await;
                    rtds_currency_prices_by_ms_lock.iter().next_back().map(|(&ts_ms, &price)| (ts_ms, price))
                };
                let now_ms = current_timestamp_ms();
                match latest {
                    None => {
                        run_log::rtds_watchdog_reconnect(&pair_symbol, None, now_ms);
                        return Err(anyhow::anyhow!("rtds watchdog: нет данных за последние {RTDS_WATCHDOG_INTERVAL_SECS}s"));
                    }
                    Some((latest_ts_ms, _)) if now_ms - latest_ts_ms > RTDS_STALE_PRICE_MAX_AGE_MS => {
                        run_log::rtds_watchdog_reconnect(&pair_symbol, Some(latest_ts_ms), now_ms);
                        return Err(anyhow::anyhow!("rtds watchdog: данные устарели на {}ms", now_ms - latest_ts_ms));
                    }
                    _ => {}
                }
            }
            opt = read.next() => {
                let Some(ws_message) = opt else {
                    return Err(anyhow::anyhow!("rtds: stream ended"));
                };
                let ws_message = ws_message.context("rtds read")?;
                match ws_message {
                    Message::Text(text) => {
                        if text == "PONG" {
                            continue;
                        }
                        if let Ok(parsed) = serde_json::from_str::<Value>(&text) {
                            ingest_rtds_spot_update(project_manager, &parsed, pair_symbol.as_str()).await;
                        }
                    }
                    Message::Binary(binary) => {
                        if let Ok(text) = String::from_utf8(binary.to_vec())
                            && let Ok(parsed) = serde_json::from_str::<Value>(&text) {
                            ingest_rtds_spot_update(project_manager, &parsed, pair_symbol.as_str()).await;
                        }
                    }
                    Message::Ping(ping_payload) => {
                        write.send(Message::Pong(ping_payload)).await.context("rtds Pong")?;
                    }
                    Message::Close(_) => return Err(anyhow::anyhow!("rtds: close")),
                    _ => {}
                }
            }
        }
    }
}

async fn ingest_rtds_spot_update(
    project_manager: &Arc<ProjectManager>,
    rtds_message: &Value,
    subscribed_pair_symbol: &str,
) {
    if let Some(batch) = rtds_message.as_array() {
        for item in batch {
            ingest_rtds_spot_update_item(project_manager, item, subscribed_pair_symbol).await;
        }
        return;
    }
    ingest_rtds_spot_update_item(project_manager, rtds_message, subscribed_pair_symbol).await;
}

async fn ingest_rtds_spot_update_item(
    project_manager: &Arc<ProjectManager>,
    rtds_message: &Value,
    subscribed_pair_symbol: &str,
) {
    let status_bad = rtds_message
        .get("statusCode")
        .and_then(|code| code.as_u64().or_else(|| code.as_i64().map(|i| i as u64)))
        .is_some_and(|code| code >= 400);
    if status_bad {
        if let Some(body) = rtds_message.get("body") {
            eprintln!("rtds (crypto_prices): ошибка API: {}", body);
        }
        return;
    }

    let topic = rtds_message.get("topic").and_then(Value::as_str);
    let msg_type = rtds_message.get("type").and_then(Value::as_str);
    if topic != Some("crypto_prices") || msg_type != Some("update") {
        return;
    }

    let Some(payload) = rtds_message.get("payload") else {
        return;
    };
    let symbol = payload.get("symbol").and_then(Value::as_str);
    if !symbol.is_some_and(|s| s.eq_ignore_ascii_case(subscribed_pair_symbol)) {
        return;
    }

    let Some(price_value) = payload.get("value").and_then(json_as_f64) else {
        return;
    };

    let Some(payload_ts_ms) = payload.get("timestamp").and_then(Value::as_i64) else {
        return;
    };

    {
        let now_ms = current_timestamp_ms();
        let cutoff = now_ms.saturating_sub(RTDS_PAYLOAD_TS_HISTORY_MS);
        let mut rtds_currency_prices_by_ms_lock = project_manager
            .rtds_currency_prices_by_ms
            .write()
            .await;
            rtds_currency_prices_by_ms_lock.insert(payload_ts_ms, price_value);
        while let Some(&k) = rtds_currency_prices_by_ms_lock.keys().next() {
            if k < cutoff {
                rtds_currency_prices_by_ms_lock.remove(&k);
            } else {
                break;
            }
        }
    }
}

fn json_as_f64(json: &Value) -> Option<f64> {
    match json {
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}
