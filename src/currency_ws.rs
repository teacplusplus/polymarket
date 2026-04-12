use crate::project_manager::ProjectManager;
use crate::run_log;
use crate::util::current_timestamp_ms;
use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::sync::Arc;
use tokio::time::{interval, Duration, MissedTickBehavior};
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Последний тик спота (Binance) из RTDS; обновляется при каждом сообщении WS.
#[derive(Clone, Debug)]
pub struct RtdsCurrencyLatest {
    pub value: f64,
    pub price_timestamp_ms: i64,
    pub message_timestamp_ms: i64,
}

/// Аналог market CLOB WS — endpoint RTDS.
const POLYMARKET_RTDS_WS_URL: &str = "wss://ws-live-data.polymarket.com";
const RTDS_RECONNECT_DELAY_SECS: u64 = 3;
const RTDS_PING_INTERVAL_SECS: u64 = 5;
const MAX_CURRENCY_PRICE_BUCKETS: usize = 86_400;

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
        let latest = project_manager.rtds_currency_latest.read().await.clone();
        if let Some(sample) = latest {
            let mut map = project_manager.rtds_currency_prices_by_sec.write().await;
            map.insert(bucket_sec, sample.value);
            let bars_in_history = map.len();
            let pair_symbol = rtds_spot_pair_symbol(&project_manager.currency);
            run_log::rtds_currency_sec_bar_inserted(
                &pair_symbol,
                bucket_sec,
                sample.value,
                sample.price_timestamp_ms,
                sample.message_timestamp_ms,
                bars_in_history,
            );
            while map.len() > MAX_CURRENCY_PRICE_BUCKETS {
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
            let pair = rtds_spot_pair_symbol(&project_manager.currency);
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

    let pair_symbol = rtds_spot_pair_symbol(&project_manager.currency);
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

    let mut ping = interval(Duration::from_secs(RTDS_PING_INTERVAL_SECS));

    loop {
        tokio::select! {
            _ = ping.tick() => {
                write.send(Message::Text("PING".into()))
                    .await
                    .context("rtds PING")?;
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
                            ingest_rtds_spot_update(project_manager, &parsed, pair_symbol.as_str())
                                .await;
                        }
                    }
                    Message::Binary(binary) => {
                        if let Ok(text) = String::from_utf8(binary.to_vec())
                            && let Ok(parsed) = serde_json::from_str::<Value>(&text)
                        {
                            ingest_rtds_spot_update(project_manager, &parsed, pair_symbol.as_str())
                                .await;
                        }
                    }
                    Message::Ping(ping_payload) => {
                        write
                            .send(Message::Pong(ping_payload))
                            .await
                            .context("rtds Pong")?;
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

    let envelope_ts = rtds_message
        .get("timestamp")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let price_ts = payload
        .get("timestamp")
        .and_then(Value::as_i64)
        .unwrap_or(envelope_ts);

    let sample = RtdsCurrencyLatest {
        value: price_value,
        price_timestamp_ms: price_ts,
        message_timestamp_ms: envelope_ts,
    };

    let mut lock = project_manager.rtds_currency_latest.write().await;
    *lock = Some(sample);
}

fn json_as_f64(json: &Value) -> Option<f64> {
    match json {
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}
