use crate::util::current_timestamp_ms;
use crate::api_data_manager::{
    event_end_unix_ms_from_rfc3339, fetch_asset_snapshot, fetch_order_filled_events_for_range,
    make_api_channel, price_at_or_before, ApiDataHub, ApiDataJob, ApiMarketDataStore,
    OrderFilledRow,
};
use crate::ws::{
    make_ws_channel, MarketSnapshot, MarketSnapshotBuffer, MarketSnapshotBufferMut,
    SnapshotsByAlignedTs, Ws,
};
use crate::market_snapshot::market_snapshot_from_historical_bucket;
use crate::xframe::{SIZE, XFrame};
use polymarket_client_sdk::clob;
use polymarket_client_sdk::gamma;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;
use std::sync::{Arc, Weak};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{self, Duration};

type MarketFrames = HashMap<String, HashMap<String, BTreeMap<i64, XFrame<SIZE>>>>;
type MarketSnapshotsByAlignedTs = HashMap<String, HashMap<String, SnapshotsByAlignedTs>>;
pub const FRAME_BUILD_INTERVAL_SECS: [u64; 3] = [60, 300, 900];

pub struct ProjectManager {
    pub xframes_by_market: Vec<Arc<RwLock<MarketFrames>>>,
    pub ws_snapshots_by_market: Vec<Arc<RwLock<MarketSnapshotsByAlignedTs>>>,
    pub ws_buffer_by_market: Vec<Arc<RwLock<MarketSnapshotBuffer>>>,
    pub api_context_by_market: Arc<RwLock<ApiMarketDataStore>>,
    pub ws: Arc<Ws>,
    pub api: Arc<ApiDataHub>,
    http: Arc<reqwest::Client>,
    gamma: Arc<gamma::Client>,
    clob: Arc<clob::Client>,
}

impl ProjectManager {
    pub fn new() -> Arc<Self> {
        let frame_interval_count = FRAME_BUILD_INTERVAL_SECS.len();
        let (ws, mut ws_snapshot_receiver) = make_ws_channel();
        let api_market_data_store = Arc::new(RwLock::new(ApiMarketDataStore::new()));
        let (api_data_hub, api_job_receiver) = make_api_channel();

        let http = Arc::new(
            reqwest::Client::builder()
                .use_rustls_tls()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        );
        let gamma = Arc::new(gamma::Client::default());
        let clob = Arc::new(
            clob::Client::new("https://clob.polymarket.com", clob::Config::default())
                .expect("failed to create Polymarket CLOB client"),
        );

        let project_manager = Arc::new_cyclic(move |weak_project_manager| {
            spawn_api_worker_loop(
                api_market_data_store.clone(),
                weak_project_manager.clone(),
                api_job_receiver,
            );
            let weak_for_ws_ingest = weak_project_manager.clone();
            tokio::spawn(async move {
                while let Some(snapshot_arc) = ws_snapshot_receiver.recv().await {
                    if let Some(project_manager) = weak_for_ws_ingest.upgrade() {
                        project_manager
                            .ingest_snapshot((*snapshot_arc).clone())
                            .await;
                    }
                }
            });
            Self {
                xframes_by_market: (0..frame_interval_count)
                    .map(|_| Arc::new(RwLock::new(HashMap::new())))
                    .collect(),
                ws_snapshots_by_market: (0..frame_interval_count)
                    .map(|_| Arc::new(RwLock::new(HashMap::new())))
                    .collect(),
                ws_buffer_by_market: (0..frame_interval_count)
                    .map(|_| Arc::new(RwLock::new(HashMap::new())))
                    .collect(),
                api_context_by_market: api_market_data_store,
                ws,
                api: api_data_hub,
                http,
                gamma,
                clob,
            }
        });
        project_manager
    }

    /// Все кадры с `aligned_ts > inserted_min`, кроме `inserted_this_batch`, пересобираются из сохранённых
    /// [MarketSnapshot] и актуальной истории `XFrame` (нужно после вставки прошлых бакетов перед уже живыми WS-кадрами).
    async fn recompute_xframes_after_historical_inserts(
        &self,
        index: usize,
        market_id: &str,
        asset_id: &str,
        inserted_min: i64,
        inserted_this_batch: &HashSet<i64>,
    ) {
        let aligned_timestamps_to_recompute: Vec<i64> = {
            let xframes_by_market_read_lock = self.xframes_by_market[index].read().await;
            let Some(aligned_ts_to_xframe_for_asset) =
                xframes_by_market_read_lock
                    .get(market_id)
                    .and_then(|by_asset_id| by_asset_id.get(asset_id))
            else {
                drop(xframes_by_market_read_lock);
                return;
            };
            let out = aligned_ts_to_xframe_for_asset
                .range((Bound::Excluded(inserted_min), Bound::Unbounded))
                .filter(|(aligned_timestamp_ms, _)| {
                    !inserted_this_batch.contains(aligned_timestamp_ms)
                })
                .map(|(aligned_timestamp_ms, _)| *aligned_timestamp_ms)
                .collect();
            drop(xframes_by_market_read_lock);
            out
        };

        for aligned_timestamp_ms in aligned_timestamps_to_recompute {
            let optional_stored_snapshot = {
                let ws_snapshots_by_market_read_lock =
                    self.ws_snapshots_by_market[index].read().await;
                let optional = ws_snapshots_by_market_read_lock
                    .get(market_id)
                    .and_then(|by_asset_id| by_asset_id.get(asset_id))
                    .and_then(|snapshots_by_aligned_ts| {
                        snapshots_by_aligned_ts.get(&aligned_timestamp_ms)
                    })
                    .cloned();
                drop(ws_snapshots_by_market_read_lock);
                optional
            };
            let Some(stored_market_snapshot) = optional_stored_snapshot else {
                eprintln!(
                    "recompute xframe: missing MarketSnapshot for {market_id} / {asset_id} @ {aligned_timestamp_ms}"
                );
                continue;
            };

            let event_end_ms = {
                let api_context_by_market_read_lock = self.api_context_by_market.read().await;
                let optional_end_ms = api_context_by_market_read_lock
                    .get(market_id)
                    .and_then(|by_asset_id| by_asset_id.get(asset_id))
                    .and_then(|api_asset_snapshot| {
                        event_end_unix_ms_from_rfc3339(
                            api_asset_snapshot.stable.end_date_rfc3339.as_deref(),
                        )
                    });
                drop(api_context_by_market_read_lock);
                optional_end_ms
            };

            let frames_history: BTreeMap<i64, XFrame<SIZE>> = {
                let xframes_by_market_read_lock = self.xframes_by_market[index].read().await;
                let Some(aligned_ts_to_xframe_for_asset) =
                    xframes_by_market_read_lock
                        .get(market_id)
                        .and_then(|by_asset_id| by_asset_id.get(asset_id))
                else {
                    drop(xframes_by_market_read_lock);
                    continue;
                };
                let history = aligned_ts_to_xframe_for_asset
                    .range(..aligned_timestamp_ms)
                    .map(|(aligned_ts, xframe)| (*aligned_ts, xframe.clone()))
                    .collect();
                drop(xframes_by_market_read_lock);
                history
            };

            let frame =
                XFrame::<SIZE>::new(stored_market_snapshot, &frames_history, event_end_ms);
            let mut xframes_by_market_write_lock = self.xframes_by_market[index].write().await;
            if let Some(aligned_ts_to_xframe) = xframes_by_market_write_lock
                .get_mut(market_id)
                .and_then(|by_asset_id| by_asset_id.get_mut(asset_id))
            {
                aligned_ts_to_xframe.insert(aligned_timestamp_ms, frame);
            }
            drop(xframes_by_market_write_lock);
        }
    }

    /// Загрузка Gamma + CLOB (tick, prices-history) + subgraph fills, сборка [XFrame] для каждой группировки [FRAME_BUILD_INTERVAL_SECS]
    /// без перезаписи уже существующих ключей `aligned_ts` (WS или прошлые вызовы).
    pub async fn build_historical_xframes(
        &self,
        asset_id: &str,
        past_bucket_count: usize,
    ) -> anyhow::Result<()> {
        if past_bucket_count == 0 {
            return Ok(());
        }

        let api_asset_snapshot =
            fetch_asset_snapshot(self.gamma.as_ref(), self.clob.as_ref(), asset_id).await?;
        let market_id = api_asset_snapshot
            .stable
            .condition_id
            .clone()
            .unwrap_or_else(|| format!("__orphan__/{asset_id}"));
        let inner_key = api_asset_snapshot.stable.asset_id.clone();
        {
            let mut api_context_by_market_write_lock = self.api_context_by_market.write().await;
            api_context_by_market_write_lock
                .entry(market_id.clone())
                .or_default()
                .insert(inner_key, api_asset_snapshot.clone());
            drop(api_context_by_market_write_lock);
        }

        let event_end_ms = event_end_unix_ms_from_rfc3339(
            api_asset_snapshot.stable.end_date_rfc3339.as_deref(),
        );
        let price_history = &api_asset_snapshot.live.price_history;
        let live_tick_size_str = api_asset_snapshot.live.tick_size.as_deref();

        let now_ms = current_timestamp_ms();
        let mut max_lookback_ms: i64 = 0;
        for &interval_secs in FRAME_BUILD_INTERVAL_SECS.iter() {
            let interval_lookback_span_ms =
                interval_secs as i64 * 1000 * past_bucket_count as i64;
            if interval_lookback_span_ms > max_lookback_ms {
                max_lookback_ms = interval_lookback_span_ms;
            }
        }
        let t_min_sec = (now_ms - max_lookback_ms - 120_000).max(0) / 1000;
        let t_max_sec = now_ms / 1000;

        let fills: Vec<OrderFilledRow> = fetch_order_filled_events_for_range(
            self.http.as_ref(),
            asset_id,
            t_min_sec,
            t_max_sec,
        )
        .await?;

        for (index, interval_secs) in FRAME_BUILD_INTERVAL_SECS.iter().copied().enumerate() {
            let mut inserted_this_batch: HashSet<i64> = HashSet::new();
            let bucket_ms = interval_secs as i64 * 1000;
            let now_aligned = align_timestamp_ms_to_interval(now_ms, interval_secs);
            let mut bucket_starts: Vec<i64> = (0..past_bucket_count)
                .map(|bucket_index| {
                    now_aligned - bucket_index as i64 * bucket_ms
                })
                .collect();
            bucket_starts.sort_unstable();

            for aligned_timestamp_ms in bucket_starts {
                let skip_existing_xframe = {
                    let xframes_by_market_read_lock = self.xframes_by_market[index].read().await;
                    let skip = xframes_by_market_read_lock
                        .get(&market_id)
                        .and_then(|by_asset_id| by_asset_id.get(asset_id))
                        .map(|aligned_ts_to_xframe| {
                            aligned_ts_to_xframe.contains_key(&aligned_timestamp_ms)
                        })
                        .unwrap_or(false);
                    drop(xframes_by_market_read_lock);
                    skip
                };
                if skip_existing_xframe {
                    continue;
                }

                let bucket_end_ms = aligned_timestamp_ms + bucket_ms - 1;
                let bucket_cutoff_sec = bucket_end_ms / 1000;
                let mid_price_for_bucket_cutoff =
                    price_at_or_before(price_history, bucket_cutoff_sec);

                let rows_in_bucket: Vec<OrderFilledRow> = fills
                    .iter()
                    .filter(|order_filled_row| {
                        let fill_timestamp_ms =
                            order_filled_row.timestamp_sec.saturating_mul(1000);
                        fill_timestamp_ms >= aligned_timestamp_ms
                            && fill_timestamp_ms <= bucket_end_ms
                    })
                    .cloned()
                    .collect();

                let market_snapshot = market_snapshot_from_historical_bucket(
                    market_id.clone(),
                    asset_id.to_string(),
                    aligned_timestamp_ms,
                    interval_secs,
                    &rows_in_bucket,
                    mid_price_for_bucket_cutoff,
                    live_tick_size_str,
                    api_asset_snapshot.live.resolved,
                );
                let snap_for_store = market_snapshot.clone();

                let frames_history = {
                    let mut xframes_by_market_write_lock =
                        self.xframes_by_market[index].write().await;
                    let aligned_ts_to_xframe = xframes_by_market_write_lock
                        .entry(market_id.clone())
                        .or_default()
                        .entry(asset_id.to_string())
                        .or_insert_with(BTreeMap::new);

                    if aligned_ts_to_xframe.contains_key(&aligned_timestamp_ms) {
                        drop(xframes_by_market_write_lock);
                        continue;
                    }
                    let history = aligned_ts_to_xframe.clone();
                    drop(xframes_by_market_write_lock);
                    history
                };

                let frame =
                    XFrame::<SIZE>::new(market_snapshot, &frames_history, event_end_ms);

                let mut did_insert = false;
                {
                    let mut xframes_by_market_write_lock =
                        self.xframes_by_market[index].write().await;
                    let aligned_ts_to_xframe = xframes_by_market_write_lock
                        .entry(market_id.clone())
                        .or_default()
                        .entry(asset_id.to_string())
                        .or_insert_with(BTreeMap::new);
                    if !aligned_ts_to_xframe.contains_key(&aligned_timestamp_ms) {
                        aligned_ts_to_xframe.insert(aligned_timestamp_ms, frame);
                        did_insert = true;
                    }
                    drop(xframes_by_market_write_lock);
                }
                if did_insert {
                    inserted_this_batch.insert(aligned_timestamp_ms);
                    let mut ws_snapshots_by_market_write_lock =
                        self.ws_snapshots_by_market[index].write().await;
                    ws_snapshots_by_market_write_lock
                        .entry(market_id.clone())
                        .or_default()
                        .entry(asset_id.to_string())
                        .or_default()
                        .insert(aligned_timestamp_ms, snap_for_store);
                    drop(ws_snapshots_by_market_write_lock);
                }
            }
            if let Some(min_ts) = inserted_this_batch.iter().copied().min() {
                self.recompute_xframes_after_historical_inserts(
                    index,
                    &market_id,
                    asset_id,
                    min_ts,
                    &inserted_this_batch,
                )
                .await;
            }
        }
        Ok(())
    }

    pub async fn ingest_snapshot(&self, snapshot: MarketSnapshot) {
        for ws_buffer_rwlock_arc in &self.ws_buffer_by_market {
            let mut ws_buffer_by_market_write_lock = ws_buffer_rwlock_arc.write().await;
            ws_buffer_by_market_write_lock.push_snapshot(snapshot.clone());
            drop(ws_buffer_by_market_write_lock);
        }
    }

    pub async fn run_frame_builder_loop(self: Arc<Self>, index: usize) {
        let interval_secs = FRAME_BUILD_INTERVAL_SECS[index];
        let mut interval = time::interval(Duration::from_secs(interval_secs));
        loop {
            interval.tick().await;
            self.build_frames_from_buffer_once(index).await;
        }
    }

    pub async fn build_frames_from_buffer_once(&self, index: usize) {
        let now_ms = current_timestamp_ms();
        let snapshots: Vec<MarketSnapshot> = {
            let mut ws_buffer_by_market_write_lock =
                self.ws_buffer_by_market[index].write().await;
            let drained = ws_buffer_by_market_write_lock.drain_aggregated_snapshots(now_ms);
            drop(ws_buffer_by_market_write_lock);
            drained
        };

        if snapshots.is_empty() {
            return;
        }

        let interval_secs = FRAME_BUILD_INTERVAL_SECS[index];

        let event_end_by_pair: HashMap<(String, String), Option<i64>> = {
            let end_date_rfc3339_by_market_asset_pair: HashMap<(String, String), Option<String>> = {
                let api_context_by_market_read_lock = self.api_context_by_market.read().await;
                let mut map = HashMap::new();
                for market_snapshot in &snapshots {
                    map.entry((
                        market_snapshot.market_id.clone(),
                        market_snapshot.asset_id.clone(),
                    ))
                    .or_insert_with(|| {
                        api_context_by_market_read_lock
                            .get(&market_snapshot.market_id)
                            .and_then(|by_asset_id| by_asset_id.get(&market_snapshot.asset_id))
                            .and_then(|api_asset_snapshot| {
                                api_asset_snapshot.stable.end_date_rfc3339.clone()
                            })
                    });
                }
                drop(api_context_by_market_read_lock);
                map
            };
            end_date_rfc3339_by_market_asset_pair
                .into_iter()
                .map(|(market_asset_pair, end_date_rfc3339_opt)| {
                    (
                        market_asset_pair,
                        event_end_unix_ms_from_rfc3339(end_date_rfc3339_opt.as_deref()),
                    )
                })
                .collect()
        };

        for mut snapshot in snapshots {
            let aligned_ts =
                align_timestamp_ms_to_interval(snapshot.timestamp_ms, interval_secs);
            snapshot.timestamp_ms = aligned_ts;

            let market_id = snapshot.market_id.clone();
            let asset_id = snapshot.asset_id.clone();
            let pair_key = (market_id.clone(), asset_id.clone());
            let event_end_ms = event_end_by_pair
                .get(&pair_key)
                .copied()
                .flatten();

            let frames_history = {
                let mut xframes_by_market_write_lock =
                    self.xframes_by_market[index].write().await;
                let aligned_ts_to_xframe = xframes_by_market_write_lock
                    .entry(market_id.clone())
                    .or_insert_with(HashMap::new)
                    .entry(asset_id.clone())
                    .or_insert_with(BTreeMap::new);

                if aligned_ts_to_xframe.contains_key(&aligned_ts) {
                    drop(xframes_by_market_write_lock);
                    continue;
                }

                let history = aligned_ts_to_xframe.clone();
                drop(xframes_by_market_write_lock);
                history
            };

            let snapshot_for_store = snapshot.clone();
            let frame =
                XFrame::<SIZE>::new(snapshot, &frames_history, event_end_ms);

            let mut did_insert = false;
            {
                let mut xframes_by_market_write_lock =
                    self.xframes_by_market[index].write().await;
                let aligned_ts_to_xframe = xframes_by_market_write_lock
                    .entry(market_id.clone())
                    .or_insert_with(HashMap::new)
                    .entry(asset_id.clone())
                    .or_insert_with(BTreeMap::new);
                if !aligned_ts_to_xframe.contains_key(&aligned_ts) {
                    aligned_ts_to_xframe.insert(aligned_ts, frame);
                    did_insert = true;
                }
                drop(xframes_by_market_write_lock);
            }
            if did_insert {
                let mut ws_snapshots_by_market_write_lock =
                    self.ws_snapshots_by_market[index].write().await;
                ws_snapshots_by_market_write_lock
                    .entry(market_id)
                    .or_default()
                    .entry(asset_id)
                    .or_default()
                    .insert(aligned_ts, snapshot_for_store);
                drop(ws_snapshots_by_market_write_lock);
            }
        }
    }
}

fn spawn_api_worker_loop(
    api_market_data_store: Arc<RwLock<ApiMarketDataStore>>,
    weak_project_manager: Weak<ProjectManager>,
    mut api_job_receiver: mpsc::Receiver<ApiDataJob>,
) {
    tokio::spawn(async move {
        while let Some(api_data_job) = api_job_receiver.recv().await {
            let Some(project_manager) = weak_project_manager.upgrade() else {
                continue;
            };
            match api_data_job {
                ApiDataJob::FetchAssetFull { asset_id } => {
                    let api_asset_snapshot = match fetch_asset_snapshot(
                        project_manager.gamma.as_ref(),
                        project_manager.clob.as_ref(),
                        &asset_id,
                    )
                    .await
                    {
                        Ok(snapshot) => snapshot,
                        Err(error) => {
                            eprintln!("api_data_worker: fetch {asset_id}: {error}");
                            continue;
                        }
                    };
                    let market_outer_key = api_asset_snapshot
                        .stable
                        .condition_id
                        .clone()
                        .unwrap_or_else(|| format!("__orphan__/{asset_id}"));
                    let asset_inner_key = api_asset_snapshot.stable.asset_id.clone();
                    let mut api_market_data_store_write_lock =
                        api_market_data_store.write().await;
                    api_market_data_store_write_lock
                        .entry(market_outer_key)
                        .or_default()
                        .insert(asset_inner_key, api_asset_snapshot);
                    drop(api_market_data_store_write_lock);
                }
                ApiDataJob::BuildHistoricalXFrames {
                    asset_id,
                    past_bucket_count,
                } => {
                    if let Err(error) = project_manager
                        .build_historical_xframes(&asset_id, past_bucket_count)
                        .await
                    {
                        eprintln!(
                            "api_data_worker: BuildHistoricalXFrames {asset_id}: {error}"
                        );
                    }
                }
            }
        }
    });
}

/// Ключ бакета: `timestamp_ms` кратен `interval_secs * 1000` (начало интервала в мс).
fn align_timestamp_ms_to_interval(timestamp_ms: i64, interval_secs: u64) -> i64 {
    let bucket_ms = (interval_secs as i64).saturating_mul(1000);
    if bucket_ms <= 0 {
        return timestamp_ms;
    }
    timestamp_ms.div_euclid(bucket_ms).saturating_mul(bucket_ms)
}
