use crate::market_snapshot::aggregate_events;
use anyhow::bail;
use crate::btcusdt_ws::RtdsBtcLatest;
use crate::run_log;
use crate::util::current_timestamp_ms;
use crate::data_ws::{
    make_ws_channel, BtcUpDownOutcome, MarketSnapshot, MarketSnapshotBuffer,
    MarketSnapshotBufferMut, Ws,
};
use crate::xframe::{
    btc_price_z_score_from_sec_history, find_opposite_asset_id, XFrame, SIZE,
};
use polymarket_client_sdk::clob;
use polymarket_client_sdk::gamma;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{self, Duration};

type MarketFrames = HashMap<String, HashMap<String, BTreeMap<i64, XFrame<SIZE>>>>;

/// Кадр, собранный в этом тике `build_frames_from_buffer_once`, до записи в `xframes_by_market`.
struct BuiltXframeEntry {
    market_id: String,
    asset_id: String,
    aligned_ts: i64,
    frame: XFrame<SIZE>,
}

const FRAME_BUILD_INTERVAL_SEC: u64 = 1;

#[derive(Debug, Clone, Default)]
pub struct MarketEventData {
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
    pub price_to_beat: Option<f64>,
    pub gamma_question: Option<String>,
}

pub struct ProjectManager {
    pub xframes_by_market: Arc<RwLock<MarketFrames>>,
    pub ws_buffer_by_market: Arc<RwLock<MarketSnapshotBuffer>>,
    pub event_data_by_market: Arc<RwLock<HashMap<String, MarketEventData>>>,
    pub btc_up_down_by_asset_id: Arc<RwLock<HashMap<String, BtcUpDownOutcome>>>,
    pub rtds_btc_latest: Arc<RwLock<Option<RtdsBtcLatest>>>,
    pub rtds_btc_prices_by_sec: Arc<RwLock<BTreeMap<i64, f64>>>,
    pub ws: Arc<Ws>,
    pub http: Arc<reqwest::Client>,
    pub gamma: Arc<gamma::Client>,
    pub clob: Arc<clob::Client>,
}

impl ProjectManager {
    pub fn new() -> Arc<Self> {
        let (ws, mut ws_snapshot_receiver) = make_ws_channel();

        let http = Arc::new(
            reqwest::Client::builder()
                .use_rustls_tls()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        );
        let gamma = Arc::new(gamma::Client::default());
        let clob = Arc::new(clob::Client::new("https://clob.polymarket.com", clob::Config::default()).expect("failed to create Polymarket CLOB client"));

        let project_manager = Arc::new(Self {
            xframes_by_market: Arc::new(RwLock::new(HashMap::new())),
            ws_buffer_by_market: Arc::new(RwLock::new(HashMap::new())),
            event_data_by_market: Arc::new(RwLock::new(HashMap::new())),
            btc_up_down_by_asset_id: Arc::new(RwLock::new(HashMap::<String, BtcUpDownOutcome>::new())),
            rtds_btc_latest: Arc::new(RwLock::new(None)),
            rtds_btc_prices_by_sec: Arc::new(RwLock::new(BTreeMap::new())),
            ws,
            http,
            gamma,
            clob,
        });

        crate::btcusdt_ws::spawn_rtds_btc_pipeline(project_manager.clone());
        let project_manager_cloned = project_manager.clone();
        tokio::spawn(async move {
            while let Some(snapshot_arc) = ws_snapshot_receiver.recv().await {
                if let Err(err) = project_manager_cloned
                    .ingest_snapshot((*snapshot_arc).clone())
                    .await
                {
                    eprintln!("ingest_snapshot: {err:#}");
                }
            }
        });

        let project_manager_cloned = project_manager.clone();
        tokio::spawn(async move {
            project_manager_cloned.run_frame_builder_loop().await;
        });

        project_manager
    }

    pub async fn merge_market_event_data(
        &self,
        starts: HashMap<String, Option<i64>>,
        ends: HashMap<String, Option<i64>>,
        price_to_beat: Option<f64>,
        gamma_question: Option<String>,
        btc_up_down_by_asset_id: HashMap<String, BtcUpDownOutcome>,
    ) {
        let mut lock = self.event_data_by_market.write().await;
        for (market_id, start_ms) in starts {
            let entry = lock.entry(market_id).or_default();
            entry.start_ms = start_ms;
            if let Some(p) = price_to_beat {
                entry.price_to_beat = Some(p);
            }
            if let Some(ref q) = gamma_question {
                entry.gamma_question = Some(q.clone());
            }
        }
        for (market_id, end_ms) in ends {
            let entry = lock.entry(market_id).or_default();
            entry.end_ms = end_ms;
            if let Some(p) = price_to_beat {
                entry.price_to_beat = Some(p);
            }
            if let Some(ref q) = gamma_question {
                entry.gamma_question = Some(q.clone());
            }
        }
        drop(lock);

        if !btc_up_down_by_asset_id.is_empty() {
            let mut map = self.btc_up_down_by_asset_id.write().await;
            for (asset_id, code) in btc_up_down_by_asset_id {
                map.insert(asset_id, code);
            }
        }
    }

    pub async fn ingest_snapshot(&self, mut snapshot: MarketSnapshot) -> anyhow::Result<()> {
        let Some(code) = self
            .btc_up_down_by_asset_id
            .read()
            .await
            .get(&snapshot.asset_id)
            .copied()
        else {
            bail!(
                "нет Up/Down для asset_id={} (нужен merge_market_event_data с outcomes из Gamma)",
                snapshot.asset_id
            );
        };
        snapshot.btc_up_down_outcome = code;
        let mut ws_buffer_by_market_write_lock = self.ws_buffer_by_market.write().await;
        ws_buffer_by_market_write_lock.push_snapshot(snapshot);
        Ok(())
    }

    pub async fn run_frame_builder_loop(self: Arc<Self>) {
        let mut interval = time::interval(Duration::from_secs(FRAME_BUILD_INTERVAL_SEC));
        loop {
            interval.tick().await;
            self.build_frames_from_buffer_once().await;
        }
    }

    pub async fn build_frames_from_buffer_once(&self) {
        let now_ms = current_timestamp_ms();
        let snapshots: Vec<MarketSnapshot> = {
            let mut ws_buffer_by_market_write_lock = self.ws_buffer_by_market.write().await;
            let drained = ws_buffer_by_market_write_lock.drain_aggregated_snapshots(now_ms);
            drop(ws_buffer_by_market_write_lock);
            drained
        };

        if snapshots.is_empty() {
            return;
        }

        let interval_secs = FRAME_BUILD_INTERVAL_SEC;

        let mut by_bucket: HashMap<(String, String, i64), Vec<MarketSnapshot>> = HashMap::new();
        for snapshot in snapshots {
            let aligned_ts = align_timestamp_ms_to_interval(snapshot.timestamp_ms, interval_secs);
            let key = (
                snapshot.market_id.clone(),
                snapshot.asset_id.clone(),
                aligned_ts,
            );
            by_bucket.entry(key).or_default().push(snapshot);
        }

        let btc_ref_sec = current_timestamp_ms() / 1000;
        let (btc_price_z_score, btc_spot_usd) = {
            let hist = self.rtds_btc_prices_by_sec.read().await;
            let btc_price_z_score =
                btc_price_z_score_from_sec_history(&hist, btc_ref_sec);
            let btc_spot_usd = hist
                .range(..=btc_ref_sec)
                .next_back()
                .map(|(_, price)| *price);
            (btc_price_z_score, btc_spot_usd)
        };

        let mut built_xframes: Vec<BuiltXframeEntry> = Vec::new();

        for ((market_id, asset_id, aligned_ts), group) in by_bucket {
            let Some(snapshot) = aggregate_events(group, aligned_ts) else {
                continue;
            };
            let frames_history = {
                let xframes_by_market_read_lock = self.xframes_by_market.read().await;
                let history = xframes_by_market_read_lock
                    .get(&market_id)
                    .and_then(|by_asset_id| by_asset_id.get(&asset_id))
                    .map(|aligned_ts_to_xframe| {
                        aligned_ts_to_xframe
                            .range(..aligned_ts)
                            .map(|(ts, xframe)| (*ts, xframe.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                drop(xframes_by_market_read_lock);
                history
            };

            let event_guard = self.event_data_by_market.read().await;
            let event_data = event_guard.get(&market_id);
            let event_end_ms = event_data.and_then(|t| t.end_ms);
            let price_to_beat = event_data.and_then(|t| t.price_to_beat);
            let gamma_question_owned = event_data.and_then(|t| t.gamma_question.clone());
            drop(event_guard);

            let btc_price_vs_beat_pct =
                btc_price_vs_price_to_beat_pct(price_to_beat, btc_spot_usd);

            let window_ms = FRAME_BUILD_INTERVAL_SEC as i64 * 1000;
            let frame = XFrame::<SIZE>::new(
                snapshot,
                &frames_history,
                event_end_ms,
                gamma_question_owned.as_deref(),
                btc_price_z_score,
                btc_price_vs_beat_pct,
                window_ms,
            );

            run_log::xframe_built(&market_id, &asset_id);

            built_xframes.push(BuiltXframeEntry {
                market_id,
                asset_id,
                aligned_ts,
                frame,
            });
        }


        let mut batch_assets_by_market: HashMap<String, HashSet<String>> = HashMap::new();
        for entry in &built_xframes {
            batch_assets_by_market
                .entry(entry.market_id.clone())
                .or_default()
                .insert(entry.asset_id.clone());
        }
        let batch_frame_by_bucket: HashMap<(String, String, i64), XFrame<SIZE>> = built_xframes
            .iter()
            .map(|entry| {
                (
                    (
                        entry.market_id.clone(),
                        entry.asset_id.clone(),
                        entry.aligned_ts,
                    ),
                    entry.frame.clone(),
                )
            })
            .collect();

        let btc_up_down_by_asset_id: HashMap<String, BtcUpDownOutcome> = {
            let guard = self.btc_up_down_by_asset_id.read().await;
            guard.clone()
        };

        for entry in &mut built_xframes {
            let candidate_asset_ids: HashSet<String> = batch_assets_by_market.get(&entry.market_id).cloned().unwrap_or_default();

            let other_asset_id = match find_opposite_asset_id(
                &entry.asset_id,
                &btc_up_down_by_asset_id,
                &candidate_asset_ids) {
                Ok(id) => id,
                Err(err) => {
                    eprintln!("find_opposite_asset_id: {err:#}");
                    continue;
                }
            };
            let Some(other_frame) = batch_frame_by_bucket.get(&(entry.market_id.clone(), other_asset_id.clone(), entry.aligned_ts)) else {
                continue;
            };
            entry.frame.copy_other_leg_features_from(other_frame);
        }

        let mut xframes_by_market_lock = self.xframes_by_market.write().await;
        for entry in built_xframes {
            xframes_by_market_lock
                .entry(entry.market_id)
                .or_insert_with(HashMap::new)
                .entry(entry.asset_id)
                .or_insert_with(BTreeMap::new)
                .insert(entry.aligned_ts, entry.frame);
        }
        drop(xframes_by_market_lock);
    }
}

/// `(price_to_beat - btc_spot) / price_to_beat * 100` — отклонение спота от уровня «beat» в процентах; знак «+», если спот ниже beat.
fn btc_price_vs_price_to_beat_pct(
    price_to_beat: Option<f64>,
    btc_spot_usd: Option<f64>,
) -> Option<f64> {
    const MIN_BEAT: f64 = 1e-6;
    let beat = price_to_beat?;
    if !beat.is_finite() || beat.abs() <= MIN_BEAT {
        return None;
    }
    let spot = btc_spot_usd?;
    if !spot.is_finite() {
        return None;
    }
    Some((beat - spot) / beat * 100.0)
}

/// Ключ бакета: `timestamp_ms` кратен `interval_secs * 1000` (начало интервала в мс).
fn align_timestamp_ms_to_interval(timestamp_ms: i64, interval_secs: u64) -> i64 {
    let bucket_ms = (interval_secs as i64).saturating_mul(1000);
    if bucket_ms <= 0 {
        return timestamp_ms;
    }
    timestamp_ms.div_euclid(bucket_ms).saturating_mul(bucket_ms)
}
