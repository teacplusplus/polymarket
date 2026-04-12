use crate::currency_updown_sibling::update_currency_updown_sibling_slots;
use crate::project_manager::ProjectManager;
use crate::run_log;
use crate::util::current_timestamp_ms;
use anyhow::{anyhow, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, sleep, Duration, MissedTickBehavior};
use tokio_tungstenite::{connect_async, tungstenite::Message};

pub use crate::market_snapshot::{
    CurrencyUpDownDelayClass, CurrencyUpDownOutcome, MarketSnapshot, TradeSide, XFrameIntervalKind,
};

use crate::market_snapshot::aggregate_events;

const POLYMARKET_MARKET_WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const WS_RECONNECT_DELAY_SECS: u64 = 3;
/// Документация Polymarket: клиентский heartbeat для market channel.
const WS_PING_INTERVAL_SECS: u64 = 10;
const BUFFER: usize = 4096;

/// Команда на подписку для одного из двух долгоживущих market WS (5m / 15m).
#[derive(Debug, Clone)]
pub struct MarketWsSubscription {
    /// Сегмент горизонта в slug Polymarket: `"5m"` или `"15m"` (см. `format!("{{}}-updown-{{period}}-…")`).
    pub period: &'static str,
    pub slug: String,
    pub asset_ids: Vec<String>,
    pub market_ids: Vec<String>,
    pub period_sec: i64,
    pub window_start_sec: i64,
    pub gamma_question: Option<String>,
}

pub struct Ws {
    pub market_snapshot_sender: mpsc::Sender<Arc<MarketSnapshot>>,
}

pub type MarketSnapshotBuffer = HashMap<String, HashMap<String, Vec<MarketSnapshot>>>;

/// Мутации буфера — на `RwLockWriteGuard<MarketSnapshotBuffer>`.
pub trait MarketSnapshotBufferMut {
    fn push_snapshot(&mut self, snapshot: MarketSnapshot);
    fn drain_aggregated_snapshots(&mut self, timestamp_ms: i64) -> Vec<MarketSnapshot>;
}

impl MarketSnapshotBufferMut for MarketSnapshotBuffer {
    fn push_snapshot(&mut self, snapshot: MarketSnapshot) {
        let bucket = self
            .entry(snapshot.market_id.clone())
            .or_default()
            .entry(snapshot.asset_id.clone())
            .or_default();
        bucket.push(snapshot);
    }

    fn drain_aggregated_snapshots(&mut self, timestamp_ms: i64) -> Vec<MarketSnapshot> {
        let mut collected = Vec::new();
        for by_asset in self.values_mut() {
            for events in by_asset.values_mut() {
                if events.is_empty() {
                    continue;
                }
                let drained = std::mem::take(events);
                if let Some(aggregated_snapshot) = aggregate_events(drained, timestamp_ms) {
                    collected.push(aggregated_snapshot);
                }
            }
        }
        collected
    }
}

pub fn make_ws_channel() -> (Arc<Ws>, mpsc::Receiver<Arc<MarketSnapshot>>) {
    let (market_snapshot_sender, market_snapshot_receiver) =
        mpsc::channel::<Arc<MarketSnapshot>>(BUFFER);
    (
        Arc::new(Ws {
            market_snapshot_sender,
        }),
        market_snapshot_receiver,
    )
}

/// Один долгоживущий market WS на горизонт (5m или 15m): начальная подписка и смена рынка через
/// [update subscription](https://docs.polymarket.com/api-reference/wss/market) без разрыва TCP.
pub fn spawn_persistent_interval_market_ws(
    project_manager: Arc<ProjectManager>,
    cmd_rx: mpsc::Receiver<MarketWsSubscription>,
    xframe_interval_kind: XFrameIntervalKind,
) {
    tokio::spawn(async move {
        run_persistent_interval_market_ws_inner(project_manager, cmd_rx, xframe_interval_kind).await;
    });
}

async fn run_persistent_interval_market_ws_inner(
    project_manager: Arc<ProjectManager>,
    mut cmd_rx: mpsc::Receiver<MarketWsSubscription>,
    xframe_interval_kind: XFrameIntervalKind,
) {
    let mut active_asset_ids: Vec<String> = Vec::new();

    // Ждём первую непустую команду подписки (пока `asset_ids` пусты — пропускаем и ждём дальше).
    let initial_subscription = loop {
        match cmd_rx.recv().await {
            None => return,
            Some(initial_subscription) => {
                if !initial_subscription.asset_ids.is_empty() {
                    break initial_subscription;
                }
            }
        }
    };
    active_asset_ids.clone_from(&initial_subscription.asset_ids);
    let mut active_market_ws_command: Option<MarketWsSubscription> = Some(initial_subscription);

    loop {
        let (websocket_stream, _http_response) = match connect_async(POLYMARKET_MARKET_WS_URL).await {
            Ok(stream_and_response) => stream_and_response,
            Err(connect_err) => {
                run_log::market_ws_session_err(&connect_err);
                sleep(Duration::from_secs(WS_RECONNECT_DELAY_SECS)).await;
                continue;
            }
        };
        let (mut write, mut read) = websocket_stream.split();

        let subscribe = json!({
            "assets_ids": &active_asset_ids,
            "type": "market",
            "custom_feature_enabled": true
        });
        if let Err(e) = write.send(Message::Text(subscribe.to_string())).await {
            run_log::market_ws_session_err(&e);
            sleep(Duration::from_secs(WS_RECONNECT_DELAY_SECS)).await;
            continue;
        }

        if let Some(ref active_market_ws_command) = active_market_ws_command {
            update_currency_updown_sibling_slots(
                project_manager.currency_updown_sibling_ws_state.clone(),
                active_market_ws_command.period_sec,
                active_market_ws_command.window_start_sec,
                &active_market_ws_command.market_ids,
                active_market_ws_command.gamma_question.as_deref(),
            )
            .await;
            project_manager
                .record_market_ws_connect_wall_ms(&active_market_ws_command.market_ids)
                .await;
        }

        let mut ping = interval(Duration::from_secs(WS_PING_INTERVAL_SECS));
        ping.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let mut disconnect = false;
        while !disconnect {
            tokio::select! {
                biased;
                _ = ping.tick() => {
                    if write.send(Message::Text("PING".into())).await.is_err() {
                        run_log::market_ws_session_err("market ws PING send failed");
                        disconnect = true;
                    }
                }
                message = read.next() => {
                    match message {
                        None => {
                            run_log::market_ws_session_err("websocket stream ended");
                            disconnect = true;
                        }
                        Some(Err(read_err)) => {
                            run_log::market_ws_session_err(&format!("websocket read: {read_err}"));
                            disconnect = true;
                        }
                        Some(Ok(message)) => match message {
                            Message::Text(text) => {
                                if let Ok(json_value) = serde_json::from_str::<Value>(&text) {
                                    let _ = ingest_json_event(
                                        &project_manager,
                                        &json_value,
                                        xframe_interval_kind,
                                    )
                                    .await;
                                }
                            }
                            Message::Binary(binary) => {
                                if let Ok(text) = String::from_utf8(binary.to_vec())
                                    && let Ok(json_value) = serde_json::from_str::<Value>(&text)
                                {
                                    let _ = ingest_json_event(
                                        &project_manager,
                                        &json_value,
                                        xframe_interval_kind,
                                    )
                                    .await;
                                }
                            }
                            Message::Ping(payload) => {
                                let _ = write.send(Message::Pong(payload)).await;
                            }
                            Message::Close(_) => {
                                run_log::market_ws_session_err("websocket closed");
                                disconnect = true;
                            }
                            _ => {}
                        },
                    }
                }
                cmd_opt = cmd_rx.recv() => {
                    let Some(next_command) = cmd_opt else {
                        return;
                    };
                    if next_command.asset_ids.is_empty() {
                        continue;
                    }
                    if next_command.asset_ids == active_asset_ids {
                        active_market_ws_command = Some(next_command);
                        continue;
                    }

                    // Сначала подписка на новый рынок — без «дыры» без данных; затем отписка от старого.
                    let old_asset_ids = active_asset_ids.clone();

                    let sub = json!({
                        "operation": "subscribe",
                        "assets_ids": &next_command.asset_ids,
                        "custom_feature_enabled": true
                    });
                    if let Err(e) = write.send(Message::Text(sub.to_string())).await {
                        run_log::market_ws_session_err(&format!("subscribe update: {e}"));
                        disconnect = true;
                        continue;
                    }

                    let unsub = json!({
                        "operation": "unsubscribe",
                        "assets_ids": &old_asset_ids,
                    });
                    if let Err(e) = write.send(Message::Text(unsub.to_string())).await {
                        run_log::market_ws_session_err(&format!("unsubscribe: {e}"));
                        active_asset_ids.clone_from(&next_command.asset_ids);
                        active_market_ws_command = Some(next_command.clone());
                        update_currency_updown_sibling_slots(
                            project_manager.currency_updown_sibling_ws_state.clone(),
                            next_command.period_sec,
                            next_command.window_start_sec,
                            &next_command.market_ids,
                            next_command.gamma_question.as_deref(),
                        )
                        .await;
                        project_manager
                            .record_market_ws_connect_wall_ms(&next_command.market_ids)
                            .await;
                        disconnect = true;
                        continue;
                    }

                    active_asset_ids.clone_from(&next_command.asset_ids);
                    active_market_ws_command = Some(next_command.clone());

                    update_currency_updown_sibling_slots(
                        project_manager.currency_updown_sibling_ws_state.clone(),
                        next_command.period_sec,
                        next_command.window_start_sec,
                        &next_command.market_ids,
                        next_command.gamma_question.as_deref(),
                    )
                    .await;
                    project_manager
                        .record_market_ws_connect_wall_ms(&next_command.market_ids)
                        .await;
                }
            }
        }

        while let Ok(queued_command) = cmd_rx.try_recv() {
            if !queued_command.asset_ids.is_empty() {
                active_asset_ids.clone_from(&queued_command.asset_ids);
                active_market_ws_command = Some(queued_command);
            }
        }
        sleep(Duration::from_secs(WS_RECONNECT_DELAY_SECS)).await;
    }
}

async fn ingest_json_event(
    project_manager: &Arc<ProjectManager>,
    value: &Value,
    xframe_interval_kind: XFrameIntervalKind,
) -> Result<()> {
    if let Some(events) = value.as_array() {
        for event in events {
            ingest_single(project_manager, event, xframe_interval_kind).await?;
        }
        return Ok(());
    }
    ingest_single(project_manager, value, xframe_interval_kind).await
}

async fn ingest_single(
    project_manager: &Arc<ProjectManager>,
    value: &Value,
    xframe_interval_kind: XFrameIntervalKind,
) -> Result<()> {
    let Some(event_type) = value.get("event_type").and_then(Value::as_str) else {
        return Ok(());
    };
    let currency_up_down_by_asset_id = project_manager
        .currency_up_down_by_asset_id
        .read()
        .await;
    let snapshots = parse_snapshots_from_event(
        value,
        event_type,
        xframe_interval_kind,
        &currency_up_down_by_asset_id,
    );
    for snapshot in snapshots {
        project_manager
            .ws
            .market_snapshot_sender
            .send(Arc::new(snapshot))
            .await
            .map_err(|_| anyhow!("snapshot receiver task dropped"))?;
    }
    Ok(())
}

pub fn parse_snapshots_from_event(
    value: &Value,
    event_type: &str,
    xframe_interval_kind: XFrameIntervalKind,
    currency_up_down_by_asset_id: &HashMap<String, CurrencyUpDownOutcome>,
) -> Vec<MarketSnapshot> {
    match event_type {
        "book" | "last_trade_price" | "best_bid_ask" | "tick_size_change" | "market_resolved" => {
            parse_single_snapshot(
                value,
                event_type,
                xframe_interval_kind,
                currency_up_down_by_asset_id,
            )
            .into_iter()
            .collect()
        }
        "price_change" => {
            parse_price_change_snapshots(value, xframe_interval_kind, currency_up_down_by_asset_id)
        }
        "new_market" => {
            parse_new_market_snapshots(value, xframe_interval_kind, currency_up_down_by_asset_id)
        }
        _ => Vec::new(),
    }
}

fn parse_single_snapshot(
    value: &Value,
    event_type: &str,
    xframe_interval_kind: XFrameIntervalKind,
    currency_up_down_by_asset_id: &HashMap<String, CurrencyUpDownOutcome>,
) -> Option<MarketSnapshot> {
    let asset_id = value
        .get("asset_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let market_id = value
        .get("market")
        .or_else(|| value.get("condition_id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if asset_id.is_empty() || market_id.is_empty() {
        return None;
    }
    let currency_up_down_outcome = *currency_up_down_by_asset_id.get(&asset_id)?;

    let timestamp_ms = parse_i64(value.get("timestamp")).unwrap_or_else(current_timestamp_ms);
    let book = parse_book_top3(value);

    Some(MarketSnapshot {
        market_id,
        asset_id,
        xframe_interval_kind,
        currency_up_down_outcome,
        timestamp_ms,
        book_bid_l1_price: book.book_bid_l1_price,
        book_ask_l1_price: book.book_ask_l1_price,
        book_bid_l1_size: book.book_bid_l1_size,
        book_ask_l1_size: book.book_ask_l1_size,
        book_bid_l2_price: book.book_bid_l2_price,
        book_bid_l2_size: book.book_bid_l2_size,
        book_bid_l3_price: book.book_bid_l3_price,
        book_bid_l3_size: book.book_bid_l3_size,
        book_ask_l2_price: book.book_ask_l2_price,
        book_ask_l2_size: book.book_ask_l2_size,
        book_ask_l3_price: book.book_ask_l3_price,
        book_ask_l3_size: book.book_ask_l3_size,
        tick_size: parse_f64(value.get("new_tick_size")).or(parse_f64(value.get("tick_size"))),
        spread: parse_f64(value.get("spread")),
        // WS trade event uses `price` for last trade value.
        last_trade_price: parse_f64(value.get("price")).or(parse_f64(value.get("last_trade_price"))),
        last_trade_size: parse_f64(value.get("size")),
        trade_volume_bucket: None,
        trade_side: parse_trade_side(value.get("side")),
        market_resolved: event_type == "market_resolved",
    })
}

fn parse_price_change_snapshots(
    value: &Value,
    xframe_interval_kind: XFrameIntervalKind,
    currency_up_down_by_asset_id: &HashMap<String, CurrencyUpDownOutcome>,
) -> Vec<MarketSnapshot> {
    let market_id = value
        .get("market")
        .or_else(|| value.get("condition_id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if market_id.is_empty() {
        return Vec::new();
    }
    let timestamp_ms = parse_i64(value.get("timestamp")).unwrap_or_else(current_timestamp_ms);

    let Some(changes) = value.get("price_changes").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut snapshots = Vec::with_capacity(changes.len());
    for change in changes {
        let asset_id = change
            .get("asset_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if asset_id.is_empty() {
            continue;
        }
        let Some(&currency_up_down_outcome) = currency_up_down_by_asset_id.get(&asset_id) else {
            continue;
        };
        snapshots.push(MarketSnapshot {
            market_id: market_id.clone(),
            asset_id,
            xframe_interval_kind,
            currency_up_down_outcome,
            timestamp_ms,
            book_bid_l1_price: parse_f64(change.get("best_bid")),
            book_ask_l1_price: parse_f64(change.get("best_ask")),
            book_bid_l1_size: None,
            book_ask_l1_size: None,
            book_bid_l2_price: None,
            book_bid_l2_size: None,
            book_bid_l3_price: None,
            book_bid_l3_size: None,
            book_ask_l2_price: None,
            book_ask_l2_size: None,
            book_ask_l3_price: None,
            book_ask_l3_size: None,
            tick_size: None,
            spread: None,
            last_trade_price: None,
            last_trade_size: None,
            trade_volume_bucket: None,
            trade_side: parse_trade_side(change.get("side")),
            market_resolved: false,
        });
    }
    snapshots
}

fn parse_new_market_snapshots(
    value: &Value,
    xframe_interval_kind: XFrameIntervalKind,
    currency_up_down_by_asset_id: &HashMap<String, CurrencyUpDownOutcome>,
) -> Vec<MarketSnapshot> {
    let market_id = value
        .get("market")
        .or_else(|| value.get("condition_id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if market_id.is_empty() {
        return Vec::new();
    }
    let timestamp_ms = parse_i64(value.get("timestamp")).unwrap_or_else(current_timestamp_ms);
    let tick_size = parse_f64(value.get("order_price_min_tick_size"));
    let Some(asset_ids) = value.get("assets_ids").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut snapshots = Vec::with_capacity(asset_ids.len());
    for asset_id_json in asset_ids {
        let asset_id = asset_id_json.as_str().unwrap_or_default().to_string();
        if asset_id.is_empty() {
            continue;
        }
        let Some(&currency_up_down_outcome) = currency_up_down_by_asset_id.get(&asset_id) else {
            continue;
        };
        snapshots.push(MarketSnapshot {
            market_id: market_id.clone(),
            asset_id,
            xframe_interval_kind,
            currency_up_down_outcome,
            timestamp_ms,
            book_bid_l1_price: None,
            book_ask_l1_price: None,
            book_bid_l1_size: None,
            book_ask_l1_size: None,
            book_bid_l2_price: None,
            book_bid_l2_size: None,
            book_bid_l3_price: None,
            book_bid_l3_size: None,
            book_ask_l2_price: None,
            book_ask_l2_size: None,
            book_ask_l3_price: None,
            book_ask_l3_size: None,
            tick_size,
            spread: None,
            last_trade_price: None,
            last_trade_size: None,
            trade_volume_bucket: None,
            trade_side: None,
            market_resolved: false,
        });
    }
    snapshots
}

#[derive(Default)]
struct ParsedBookTop3 {
    book_bid_l1_price: Option<f64>,
    book_ask_l1_price: Option<f64>,
    book_bid_l1_size: Option<f64>,
    book_ask_l1_size: Option<f64>,
    book_bid_l2_price: Option<f64>,
    book_bid_l2_size: Option<f64>,
    book_bid_l3_price: Option<f64>,
    book_bid_l3_size: Option<f64>,
    book_ask_l2_price: Option<f64>,
    book_ask_l2_size: Option<f64>,
    book_ask_l3_price: Option<f64>,
    book_ask_l3_size: Option<f64>,
}

fn parse_side_levels_sorted(levels: Option<&Vec<Value>>, bids: bool) -> Vec<(f64, f64)> {
    let Some(orderbook_levels) = levels else {
        return Vec::new();
    };
    let mut out: Vec<(f64, f64)> = Vec::new();
    for level in orderbook_levels {
        let Some(price) = parse_f64(level.get("price")) else {
            continue;
        };
        let size = parse_f64(level.get("size")).unwrap_or(0.0);
        out.push((price, size));
    }
    if bids {
        out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    out
}

/// Топ-3 уровня bid/ask из `bids`/`asks` (`book`); при отсутствии массивов — только поля `best_bid`/`best_ask` в корне JSON.
fn parse_book_top3(value: &Value) -> ParsedBookTop3 {
    let bids = value.get("bids").and_then(Value::as_array);
    let asks = value.get("asks").and_then(Value::as_array);
    let bid_levels = parse_side_levels_sorted(bids, true);
    let ask_levels = parse_side_levels_sorted(asks, false);

    let mut out = ParsedBookTop3::default();

    if !bid_levels.is_empty() {
        out.book_bid_l1_price = Some(bid_levels[0].0);
        out.book_bid_l1_size = Some(bid_levels[0].1);
        if bid_levels.len() > 1 {
            out.book_bid_l2_price = Some(bid_levels[1].0);
            out.book_bid_l2_size = Some(bid_levels[1].1);
        }
        if bid_levels.len() > 2 {
            out.book_bid_l3_price = Some(bid_levels[2].0);
            out.book_bid_l3_size = Some(bid_levels[2].1);
        }
    } else {
        out.book_bid_l1_price = parse_f64(value.get("best_bid"));
    }

    if !ask_levels.is_empty() {
        out.book_ask_l1_price = Some(ask_levels[0].0);
        out.book_ask_l1_size = Some(ask_levels[0].1);
        if ask_levels.len() > 1 {
            out.book_ask_l2_price = Some(ask_levels[1].0);
            out.book_ask_l2_size = Some(ask_levels[1].1);
        }
        if ask_levels.len() > 2 {
            out.book_ask_l3_price = Some(ask_levels[2].0);
            out.book_ask_l3_size = Some(ask_levels[2].1);
        }
    } else {
        out.book_ask_l1_price = parse_f64(value.get("best_ask"));
    }

    out
}

fn parse_trade_side(json_field: Option<&Value>) -> Option<TradeSide> {
    json_field
        .and_then(Value::as_str)
        .and_then(|side| match side.to_ascii_uppercase().as_str() {
            "BUY" => Some(TradeSide::Buy),
            "SELL" => Some(TradeSide::Sell),
            _ => None,
        })
}

fn parse_f64(json_field: Option<&Value>) -> Option<f64> {
    match json_field {
        Some(Value::Number(number)) => number.as_f64(),
        Some(Value::String(text)) => text.parse::<f64>().ok(),
        _ => None,
    }
}

fn parse_i64(json_field: Option<&Value>) -> Option<i64> {
    match json_field {
        Some(Value::Number(number)) => number.as_i64(),
        Some(Value::String(text)) => text.parse::<i64>().ok(),
        _ => None,
    }
}
