use crate::project_manager::{ProjectManager, FRAME_BUILD_INTERVAL_SECS};
use crate::ws_parser::parse_snapshots_from_event;
use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::Message};

pub use crate::market_snapshot::{MarketSnapshot, TradeSide};

use crate::market_snapshot::aggregate_events;

const POLYMARKET_MARKET_WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const WS_RECONNECT_DELAY_SECS: u64 = 3;
const BUFFER: usize = 4096;

pub struct Ws {
    pub market_snapshot_sender: mpsc::Sender<Arc<MarketSnapshot>>,
}

pub type SnapshotsByAlignedTs = BTreeMap<i64, MarketSnapshot>;
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

pub fn new(
    project_manager: Arc<ProjectManager>,
    asset_ids: Vec<String>,
) -> Result<JoinHandle<()>> {
    if asset_ids.is_empty() {
        return Err(anyhow!("asset_ids cannot be empty"));
    }

    for index in 0..FRAME_BUILD_INTERVAL_SECS.len() {
        let frame_builder_manager = project_manager.clone();
        tokio::spawn(async move {
            frame_builder_manager.run_frame_builder_loop(index).await;
        });
    }

    let handle = tokio::spawn(async move {
        loop {
            let result = run_single_market_ws_session(project_manager.clone(), asset_ids.clone()).await;
            if let Err(err) = result {
                eprintln!("market ws session ended with error: {err}");
            } else {
                eprintln!("market ws session closed, reconnecting");
            }
            sleep(Duration::from_secs(WS_RECONNECT_DELAY_SECS)).await;
        }
    });

    Ok(handle)
}

async fn run_single_market_ws_session(
    project_manager: Arc<ProjectManager>,
    asset_ids: Vec<String>,
) -> Result<()> {
    let (websocket_stream, _) = connect_async(POLYMARKET_MARKET_WS_URL)
        .await
        .context("connect to polymarket market websocket")?;
    let (mut websocket_writer, mut websocket_reader) = websocket_stream.split();

    let subscribe = json!({
        "assets_ids": asset_ids,
        "type": "market",
        "custom_feature_enabled": true
    });
    websocket_writer
        .send(Message::Text(subscribe.to_string()))
        .await
        .context("send subscription message")?;

    while let Some(message) = websocket_reader.next().await {
        let message = message.context("read websocket message")?;
        match message {
            Message::Text(text) => {
                if let Ok(json_value) = serde_json::from_str::<Value>(&text) {
                    ingest_json_event(&project_manager, &json_value).await?;
                }
            }
            Message::Binary(binary) => {
                if let Ok(text) = String::from_utf8(binary.to_vec()) {
                    if let Ok(json_value) = serde_json::from_str::<Value>(&text) {
                        ingest_json_event(&project_manager, &json_value).await?;
                    }
                }
            }
            Message::Ping(payload) => {
                websocket_writer
                    .send(Message::Pong(payload))
                    .await?;
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    Err(anyhow!("websocket stream ended"))
}

async fn ingest_json_event(project_manager: &Arc<ProjectManager>, value: &Value) -> Result<()> {
    if let Some(events) = value.as_array() {
        for event in events {
            ingest_single(project_manager, event).await?;
        }
        return Ok(());
    }
    ingest_single(project_manager, value).await
}

async fn ingest_single(project_manager: &Arc<ProjectManager>, value: &Value) -> Result<()> {
    let Some(event_type) = value.get("event_type").and_then(Value::as_str) else {
        return Ok(());
    };
    let snapshots = parse_snapshots_from_event(value, event_type);
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
