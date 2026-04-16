use crate::market_snapshot::aggregate_events;
use anyhow::bail;
use crate::constants::XFrameIntervalKind;
use crate::run_log;
use crate::util::{
    current_timestamp_ms, fetch_gamma_event_data_for_slug,
    fetch_price_to_beat_from_polymarket_event_page, CurrencyEventSlugData,
};
use crate::xframe_dump;
pub use crate::currency_updown_sibling::{
    five_min_belongs_to_fifteen_window, CurrencyUpDownSiblingSlot, CurrencyUpDownSiblingState,
};
pub use crate::constants::{CurrencyUpDownInterval, FIFTEEN_MIN_SEC, FIVE_MIN_SEC};
use crate::data_ws::{
    make_ws_channel, spawn_persistent_interval_market_ws, CurrencyUpDownOutcome, MarketSnapshot,
    MarketSnapshotBuffer, MarketSnapshotBufferMut, MarketWsSubscription, Ws, WsCommand,
};
use crate::xframe::{
    currency_price_z_score_from_sec_history, compute_xframe_stable, find_opposite_asset_id,
    find_same_outcome_sibling_asset_id, XFrame, SIZE,
};
use polymarket_client_sdk::clob;
use polymarket_client_sdk::gamma;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
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
/// Размер очереди команд смены подписки на долгоживущих market WS (5m / 15m).
const MARKET_WS_SUBSCRIPTION_CHANNEL_CAP: usize = 8;

#[derive(Debug, Clone, Default)]
pub struct MarketEventData {
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
    pub gamma_question: Option<String>,
}

pub struct ProjectManager {
    pub currency: Arc<String>,
    pub xframes_by_market: Arc<RwLock<MarketFrames>>,
    pub ws_buffer_by_market: Arc<RwLock<MarketSnapshotBuffer>>,
    pub event_data_by_market: Arc<RwLock<HashMap<String, MarketEventData>>>,
    pub slug_to_market_id: Arc<RwLock<HashMap<String, String>>>,
    pub price_to_beat_by_market: Arc<RwLock<HashMap<String, f64>>>,
    pub currency_up_down_by_asset_id: Arc<RwLock<HashMap<String, CurrencyUpDownOutcome>>>,
    pub ws_connect_wall_ms_by_asset_id: Arc<RwLock<HashMap<String, i64>>>,
    pub currency_updown_sibling_state: Arc<RwLock<CurrencyUpDownSiblingState>>,
    pub rtds_currency_prices_by_ms: Arc<RwLock<BTreeMap<i64, f64>>>,
    pub rtds_currency_prices_by_sec: Arc<RwLock<BTreeMap<i64, f64>>>,
    pub market_asset_ids_by_market: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    pub ws: Arc<Ws>,
    pub http: Arc<reqwest::Client>,
    pub gamma: Arc<gamma::Client>,
    pub clob: Arc<clob::Client>,
    pub market_ws_tx_5m: mpsc::Sender<WsCommand>,
    pub market_ws_tx_15m: mpsc::Sender<WsCommand>,
}

impl ProjectManager {
    pub fn new(currency: String) -> Arc<Self> {
        let (ws, mut ws_snapshot_receiver) = make_ws_channel();

        let http = Arc::new(
            reqwest::Client::builder()
                .use_rustls_tls()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        );
        let gamma = Arc::new(gamma::Client::default());
        let clob = Arc::new(clob::Client::new("https://clob.polymarket.com", clob::Config::default()).expect("failed to create Polymarket CLOB client"));

        let (market_ws_tx_5m, market_ws_rx_5m) =
            mpsc::channel::<WsCommand>(MARKET_WS_SUBSCRIPTION_CHANNEL_CAP);
        let (market_ws_tx_15m, market_ws_rx_15m) =
            mpsc::channel::<WsCommand>(MARKET_WS_SUBSCRIPTION_CHANNEL_CAP);

        let project_manager = Arc::new(Self {
            currency: Arc::new(currency),
            xframes_by_market: Arc::new(RwLock::new(HashMap::new())),
            ws_buffer_by_market: Arc::new(RwLock::new(HashMap::new())),
            event_data_by_market: Arc::new(RwLock::new(HashMap::new())),
            slug_to_market_id: Arc::new(RwLock::new(HashMap::new())),
            price_to_beat_by_market: Arc::new(RwLock::new(HashMap::new())),
            currency_up_down_by_asset_id: Arc::new(RwLock::new(HashMap::<String, CurrencyUpDownOutcome>::new())),
            ws_connect_wall_ms_by_asset_id: Arc::new(RwLock::new(HashMap::new())),
            currency_updown_sibling_state: Arc::new(RwLock::new(CurrencyUpDownSiblingState::default())),
            rtds_currency_prices_by_ms: Arc::new(RwLock::new(BTreeMap::new())),
            rtds_currency_prices_by_sec: Arc::new(RwLock::new(BTreeMap::new())),
            market_asset_ids_by_market: Arc::new(RwLock::new(HashMap::new())),
            ws,
            http,
            gamma,
            clob,
            market_ws_tx_5m,
            market_ws_tx_15m,
        });

        spawn_persistent_interval_market_ws(
            project_manager.clone(),
            market_ws_rx_5m,
            XFrameIntervalKind::FiveMin,
        );
        spawn_persistent_interval_market_ws(
            project_manager.clone(),
            market_ws_rx_15m,
            XFrameIntervalKind::FifteenMin,
        );

        crate::currency_ws::spawn_rtds_currency_pipeline(project_manager.clone());
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

        let pm_5m = project_manager.clone();
        tokio::spawn(async move {
            pm_5m.run_currency_updown_interval(FIVE_MIN_SEC, "5m").await;
        });
        let pm_15m = project_manager.clone();
        tokio::spawn(async move {
            pm_15m
                .run_currency_updown_interval(FIFTEEN_MIN_SEC, "15m")
                .await;
        });

        project_manager
    }

    pub async fn merge_market_event_data(
        &self,
        starts: &HashMap<String, Option<i64>>,
        ends: &HashMap<String, Option<i64>>,
        gamma_question: Option<String>,
        currency_up_down_by_asset_id: &HashMap<String, CurrencyUpDownOutcome>,
        slug: Option<&str>,
    ) {
        let mut market_ids_touched: HashSet<String> = HashSet::new();
        market_ids_touched.extend(starts.keys().cloned());
        market_ids_touched.extend(ends.keys().cloned());

        let mut event_data_by_market_lock = self.event_data_by_market.write().await;
        for (market_id, start_ms) in starts {
            let entry = event_data_by_market_lock.entry(market_id.clone()).or_default();
            entry.start_ms = *start_ms;
            if let Some(ref q) = gamma_question {
                entry.gamma_question = Some(q.clone());
            }
        }

        for (market_id, end_ms) in ends {
            let entry = event_data_by_market_lock.entry(market_id.clone()).or_default();
            entry.end_ms = *end_ms;
            if let Some(ref q) = gamma_question {
                entry.gamma_question = Some(q.clone());
            }
        }
        drop(event_data_by_market_lock);

        if let Some(slug) = slug {
            if let Some(market_id) = market_ids_touched.iter().min().cloned() {
                let mut slug_to_market_id_lock = self.slug_to_market_id.write().await;
                slug_to_market_id_lock.insert(slug.to_string(), market_id);
            }
        }

        if !currency_up_down_by_asset_id.is_empty() {
            let mut currency_up_down_by_asset_id_lock = self.currency_up_down_by_asset_id.write().await;
            for (asset_id, code) in currency_up_down_by_asset_id.iter() {
                currency_up_down_by_asset_id_lock.insert(asset_id.clone(), *code);
            }
            drop(currency_up_down_by_asset_id_lock);
            let mut market_asset_ids_lock = self.market_asset_ids_by_market.write().await;
            for market_id_touched in market_ids_touched {
                market_asset_ids_lock
                    .entry(market_id_touched)
                    .or_default()
                    .extend(currency_up_down_by_asset_id.keys().cloned());
            }
        }
    }

    /// Восстанавливает те же поля, что даёт [`fetch_gamma_event_data_for_slug`], из кэшей
    /// [`Self::slug_to_market_id`], [`Self::event_data_by_market`], [`Self::market_asset_ids_by_market`], [`Self::currency_up_down_by_asset_id`].
    async fn try_restore_currency_event_from_slug_cache(
        &self,
        slug: &str,
    ) -> Option<(
        HashMap<String, Option<i64>>,
        HashMap<String, Option<i64>>,
        Option<String>,
        HashMap<String, CurrencyUpDownOutcome>,
    )> {
        let market_id = self.slug_to_market_id.read().await.get(slug).cloned()?;
        let event_data = self
            .event_data_by_market
            .read()
            .await
            .get(&market_id)
            .cloned()?;
        let asset_ids = self
            .market_asset_ids_by_market
            .read()
            .await
            .get(&market_id)
            .cloned()?;
        if asset_ids.is_empty() {
            return None;
        }
        let currency_up_down_by_asset_id_lock = self.currency_up_down_by_asset_id.read().await;
        let mut currency_up_down_by_asset_id = HashMap::with_capacity(asset_ids.len());
        for asset_id in &asset_ids {
            let code = currency_up_down_by_asset_id_lock.get(asset_id).copied()?;
            currency_up_down_by_asset_id.insert(asset_id.clone(), code);
        }
        drop(currency_up_down_by_asset_id_lock);

        let mut market_event_start_ms = HashMap::new();
        market_event_start_ms.insert(market_id.clone(), event_data.start_ms);
        let mut market_event_end_ms = HashMap::new();
        market_event_end_ms.insert(market_id.clone(), event_data.end_ms);

        Some((
            market_event_start_ms,
            market_event_end_ms,
            event_data.gamma_question,
            currency_up_down_by_asset_id,
        ))
    }

    /// Те же условия, что [`Self::try_restore_currency_event_from_slug_cache`], без сборки мап в память.
    async fn slug_currency_event_fully_cached(&self, slug: &str) -> bool {
        let Some(market_id) = self.slug_to_market_id.read().await.get(slug).cloned() else {
            return false;
        };
        let Some(_) = self.event_data_by_market.read().await.get(&market_id) else {
            return false;
        };
        let Some(asset_ids) = self
            .market_asset_ids_by_market
            .read()
            .await
            .get(&market_id)
            .cloned()
        else {
            return false;
        };
        if asset_ids.is_empty() {
            return false;
        }
        let cu = self.currency_up_down_by_asset_id.read().await;
        asset_ids.iter().all(|aid| cu.contains_key(aid))
    }

    /// Удаляет все данные, накопленные для завершённого маркета, из всех кешей `ProjectManager`.
    pub async fn cleanup_stale_market_data(&self, market_id: &str) {
        let asset_ids: HashSet<String> = self
            .market_asset_ids_by_market
            .write()
            .await
            .remove(market_id)
            .unwrap_or_default();

        self.xframes_by_market.write().await.remove(market_id);
        self.ws_buffer_by_market.write().await.remove(market_id);
        self.event_data_by_market.write().await.remove(market_id);
        self.price_to_beat_by_market.write().await.remove(market_id);

        {
            let mut currency_up_down_by_asset_id_lock = self.currency_up_down_by_asset_id.write().await;
            for asset_id in &asset_ids {
                currency_up_down_by_asset_id_lock.remove(asset_id);
            }
        }
        {
            let mut ws_connect_wall_ms_by_asset_id_lock = self.ws_connect_wall_ms_by_asset_id.write().await;
            for asset_id in &asset_ids {
                ws_connect_wall_ms_by_asset_id_lock.remove(asset_id);
            }
        }
        {
            let mut slugs = self.slug_to_market_id.write().await;
            slugs.retain(|_, v| v != market_id);
        }
    }

    /// Загружает данные окна из Gamma, вызывает [`Self::merge_market_event_data`] и возвращает те же поля, что [`Self::try_restore_currency_event_from_slug_cache`].
    async fn fetch_currency_event_from_gamma_and_merge(
        &self,
        slug: &str,
        period: &'static str,
    ) -> Option<(
        HashMap<String, Option<i64>>,
        HashMap<String, Option<i64>>,
        Option<String>,
        HashMap<String, CurrencyUpDownOutcome>,
    )> {
        let CurrencyEventSlugData {
            currency_up_down_by_asset_id,
            market_event_start_ms,
            market_event_end_ms,
            gamma_question,
        } = match fetch_gamma_event_data_for_slug(self.http.as_ref(), slug).await {
            Ok(d) => d,
            Err(e) => {
                run_log::gamma_fetch_err(period, slug, &e);
                return None;
            }
        };
        self.merge_market_event_data(
            &market_event_start_ms,
            &market_event_end_ms,
            gamma_question.clone(),
            &currency_up_down_by_asset_id,
            Some(slug),
        )
        .await;
        Some((
            market_event_start_ms,
            market_event_end_ms,
            gamma_question,
            currency_up_down_by_asset_id,
        ))
    }

    pub async fn merge_market_price_to_beat(
        &self,
        price_to_beat: f64,
        market_ids: &HashSet<String>,
    ) {
        let mut map = self.price_to_beat_by_market.write().await;
        for mid in market_ids {
            map.insert(mid.clone(), price_to_beat);
        }
    }

    /// Вызывать после успешной подписки на CLOB market WS: записывает `ws_connect_wall_ms` по каждому `asset_id` для [`crate::xframe::compute_xframe_stable`].
    pub async fn record_ws_connect_wall_ms_for_asset_ids(&self, asset_ids: &[String]) {
        let now_ms = current_timestamp_ms();
        let mut ws_connect_wall_ms_by_asset_id_lock = self.ws_connect_wall_ms_by_asset_id.write().await;
        for asset_id in asset_ids {
            ws_connect_wall_ms_by_asset_id_lock.insert(asset_id.clone(), now_ms);
        }
    }

    pub async fn ingest_snapshot(&self, mut snapshot: MarketSnapshot) -> anyhow::Result<()> {
        let Some(code) = self
            .currency_up_down_by_asset_id
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
        snapshot.currency_up_down_outcome = code;
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

        let currency_ref_sec = current_timestamp_ms() / 1000;
        let (currency_price_z_score, currency_spot_usd) = {
            let hist = self.rtds_currency_prices_by_sec.read().await;
            let currency_price_z_score =
                currency_price_z_score_from_sec_history(&hist, currency_ref_sec);
            let currency_spot_usd = hist
                .range(..=currency_ref_sec)
                .next_back()
                .map(|(_, price)| *price);
            (currency_price_z_score, currency_spot_usd)
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
            let gamma_question_owned = event_data.and_then(|t| t.gamma_question.clone());
            let event_start_ms = event_data.and_then(|t| t.start_ms);
            drop(event_guard);

            let price_to_beat = {
                let ptb = self.price_to_beat_by_market.read().await;
                ptb.get(&market_id).copied()
            };

            let ws_connect_wall_ms = {
                let ws_connect_wall_ms_by_asset_id_lock = self.ws_connect_wall_ms_by_asset_id.read().await;
                ws_connect_wall_ms_by_asset_id_lock.get(&asset_id).copied()
            };

            let currency_price_vs_beat_pct =
                currency_price_vs_price_to_beat_pct(price_to_beat, currency_spot_usd);

            let window_ms = FRAME_BUILD_INTERVAL_SEC as i64 * 1000;
            let stable = compute_xframe_stable(
                market_id.as_str(),
                snapshot.timestamp_ms,
                event_start_ms,
                ws_connect_wall_ms,
            );
            let frame = XFrame::<SIZE>::new(
                snapshot,
                &frames_history,
                event_end_ms,
                gamma_question_owned.as_deref(),
                currency_price_z_score,
                currency_price_vs_beat_pct,
                window_ms,
                stable,
            );

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

        let currency_up_down_by_asset_id: HashMap<String, CurrencyUpDownOutcome> = {
            let guard = self.currency_up_down_by_asset_id.read().await;
            guard.clone()
        };

        {
            let sibling_state = self.currency_updown_sibling_state.read().await;
            let market_asset_ids = self.market_asset_ids_by_market.read().await;
            let sibling_market_by_market: HashMap<String, String> = {
                let mut sibling_market_lookup = HashMap::new();
                if let Some((five_market_id, fifteen_market_id)) =
                    sibling_state.paired_five_and_fifteen_market_ids()
                {
                    sibling_market_lookup.insert(five_market_id.clone(), fifteen_market_id.clone());
                    sibling_market_lookup.insert(fifteen_market_id, five_market_id);
                }
                sibling_market_lookup
            };

            let xframes_stored = self.xframes_by_market.read().await;

            for entry in &mut built_xframes {
                let mut candidate_asset_ids: HashSet<String> = batch_assets_by_market
                    .get(&entry.market_id)
                    .cloned()
                    .unwrap_or_default();
                if let Some(by_asset) = xframes_stored.get(&entry.market_id) {
                    candidate_asset_ids.extend(by_asset.keys().cloned());
                }

                let other_asset_id = match find_opposite_asset_id(
                    &entry.asset_id,
                    &currency_up_down_by_asset_id,
                    &candidate_asset_ids,
                ) {
                    Ok(id) => id,
                    Err(err) => {
                        eprintln!(
                            "{} find_opposite_asset_id: {err:#}",
                            current_timestamp_ms()
                        );
                        continue;
                    }
                };
                let Some(other_frame) = lookup_frame_for_leg_merge(
                    &entry.market_id,
                    &other_asset_id,
                    entry.aligned_ts,
                    &batch_frame_by_bucket,
                    &xframes_stored,
                ) else {
                    continue;
                };
                entry.frame.merge_other_leg_features_from(other_frame);
            }

            for entry in &mut built_xframes {
                let Some(sibling_market_id) = sibling_market_by_market.get(&entry.market_id) else {
                    continue;
                };
                let mut sibling_candidates: HashSet<String> = batch_assets_by_market
                    .get(sibling_market_id)
                    .cloned()
                    .unwrap_or_default();
                if let Some(ids) = market_asset_ids.get(sibling_market_id) {
                    sibling_candidates.extend(ids.iter().cloned());
                }
                let sibling_asset_id = match find_same_outcome_sibling_asset_id(
                    &entry.asset_id,
                    sibling_market_id.as_str(),
                    &currency_up_down_by_asset_id,
                    &sibling_candidates,
                ) {
                    Ok(id) => id,
                    Err(err) => {
                        if entry.frame.stable {
                            eprintln!(
                                "{} find_same_outcome_sibling_asset_id: {err:#}",
                                current_timestamp_ms()
                            );
                        }
                        continue;
                    }
                };
                let Some(sibling_frame) = lookup_frame_for_leg_merge(
                    sibling_market_id.as_str(),
                    &sibling_asset_id,
                    entry.aligned_ts,
                    &batch_frame_by_bucket,
                    &xframes_stored,
                ) else {
                    continue;
                };
                entry.frame.merge_sibling_market_features_from(sibling_frame);
            }
        }

        let mut xframes_by_market_lock = self.xframes_by_market.write().await;
        for entry in built_xframes {
            if entry.frame.stable {
                run_log::xframe_stored(&entry.frame);
            }
            xframes_by_market_lock
                .entry(entry.market_id)
                .or_insert_with(HashMap::new)
                .entry(entry.asset_id)
                .or_insert_with(BTreeMap::new)
                .insert(entry.aligned_ts, entry.frame);
        }
        drop(xframes_by_market_lock);
    }

    pub async fn run_currency_updown_interval(self: Arc<Self>, period_sec: i64, period: &'static str) {
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

        let mut prev_market_id: Option<String> = None;
        let mut prev_gamma_question: Option<String> = None;
        let mut next_window_start_sec: Option<i64> = None;

        loop {
            tick.tick().await;
            let now_ms = current_timestamp_ms();
            let poly_sec = now_ms / 1000;
            let window_start_sec = (poly_sec / period_sec) * period_sec;
            let ws_end_sec = window_start_sec + period_sec;

            if now_ms >= ws_end_sec * 1000 {
                continue;
            }

            let mut subscribed: Vec<String> = Vec::new();
            let slug = format!("{}-updown-{period}-{window_start_sec}", self.currency.to_lowercase());

            let (market_event_start_ms, market_event_end_ms, gamma_question, currency_up_down_by_asset_id) =
                if let Some(restored) = self.try_restore_currency_event_from_slug_cache(&slug).await {
                    run_log::gamma_event_data_from_cache(period, &slug);
                    restored
                } else if let Some(fetched) = self.fetch_currency_event_from_gamma_and_merge(&slug, period).await {
                    fetched
                } else {
                    continue;
                };

            if next_window_start_sec != Some(window_start_sec) {
                next_window_start_sec = Some(window_start_sec);
                let project_manager_cloned = self.clone();
                let currency_lower = self.currency.to_lowercase();
                tokio::spawn(async move {
                    const PREFETCH_UPCOMING_WINDOW_SLUGS: i64 = 3;
                    for k in 1_i64..=PREFETCH_UPCOMING_WINDOW_SLUGS {
                        let next_window_start_sec = window_start_sec.saturating_add(period_sec.saturating_mul(k));
                        let prefetch_slug = format!("{currency_lower}-updown-{period}-{next_window_start_sec}");
                        if project_manager_cloned.slug_currency_event_fully_cached(&prefetch_slug).await {
                            continue;
                        }
                        if let Some((_, _, _, currency_up_down_by_asset_id)) = project_manager_cloned.fetch_currency_event_from_gamma_and_merge(&prefetch_slug, period).await
                        {
                            run_log::gamma_event_prefetch_fetched(period, &prefetch_slug);
                            let mut asset_ids: Vec<String> = currency_up_down_by_asset_id.keys().cloned().collect();
                            asset_ids.sort_unstable();
                            let tx = match period_sec {
                                FIVE_MIN_SEC => &project_manager_cloned.market_ws_tx_5m,
                                FIFTEEN_MIN_SEC => &project_manager_cloned.market_ws_tx_15m,
                                _ => &project_manager_cloned.market_ws_tx_15m,
                            };
                            match tx.send(WsCommand::PrefetchSubscribe { asset_ids }).await {
                                Err(_) => run_log::ws_spawn_err(period, &prefetch_slug, "market ws command channel closed"),
                                _ => {}
                            }
                        }
                    }
                });
            }

            let mut ids: Vec<String> = currency_up_down_by_asset_id.keys().cloned().collect();
            ids.sort_unstable();

            let market_end_ms = market_event_end_ms
                .values()
                .copied()
                .flatten()
                .max()
                .unwrap_or(ws_end_sec * 1000);

            let market_ids: Vec<String> = market_event_end_ms.keys().cloned().collect();

            let market_start_ms = market_event_start_ms
                .values()
                .copied()
                .flatten()
                .min();

            let project_manager_cloned = self.clone();
            let currency = self.currency.clone();

            let price_to_beat = match market_start_ms {
                Some(start_ms) => {           
                    let rtds_currency_prices_by_ms_lock = project_manager_cloned.rtds_currency_prices_by_ms.read().await;
                    if let Some(&price) = rtds_currency_prices_by_ms_lock.get(&start_ms) {
                        run_log::price_to_beat_from_rtds(
                            period,
                            &slug,
                            &market_ids,
                            start_ms,
                            price,
                        );
                        Some(price)
                    } else {
                        None
                    }
                }
                None => None,
            };

            let price_to_beat = if let Some(price_to_beat) = price_to_beat {
                let slug_cloned = slug.clone();
                let market_ids_cloned = market_ids.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    const MAX_ATTEMPTS: u32 = 15;
                    for attempt in 1..=MAX_ATTEMPTS {
                        match fetch_price_to_beat_from_polymarket_event_page(project_manager_cloned.http.as_ref(), &slug_cloned, currency.as_str()).await {
                            Ok(price_to_beat) => {
                                run_log::price_to_beat_from_event_page(
                                    period,
                                    &slug_cloned,
                                    price_to_beat,
                                    Some(10),
                                );
                                let market_ids_for_ptb: HashSet<String> = market_ids_cloned.iter().cloned().collect();
                                project_manager_cloned.merge_market_price_to_beat(price_to_beat, &market_ids_for_ptb).await;
                                break;
                            }
                            Err(err) => {
                                if attempt >= MAX_ATTEMPTS {
                                    run_log::gamma_fetch_err(period, &slug_cloned, &err);
                                } else {
                                    tokio::time::sleep(Duration::from_secs(5)).await;
                                }
                            }
                        }
                    }
                });

                Some(price_to_beat)
            } else {
                const MAX_ATTEMPTS: u32 = 15;
                let mut fetched: Option<f64> = None;
                for attempt in 1..=MAX_ATTEMPTS {
                    match fetch_price_to_beat_from_polymarket_event_page(self.http.as_ref(), &slug, currency.as_str()).await {
                        Ok(price_to_beat) => {
                            run_log::price_to_beat_from_event_page(period, &slug, price_to_beat, None);
                            fetched = Some(price_to_beat);
                            break;
                        }
                        Err(e) => {
                            if attempt >= MAX_ATTEMPTS {
                                run_log::gamma_fetch_err(period, &slug, &e);
                            } else {
                                tokio::time::sleep(Duration::from_secs(5)).await;
                            }
                        }
                    }
                }
                let Some(price_to_beat) = fetched else {
                    continue;
                };
                Some(price_to_beat)
            };


            if let Some(price_to_beat) = price_to_beat {
                let market_ids_for_ptb: HashSet<String> = market_ids.iter().cloned().collect();
                self.merge_market_price_to_beat(price_to_beat, &market_ids_for_ptb).await;
            }

            while current_timestamp_ms() < ws_end_sec * 1000 {
                let now_poly_ms = current_timestamp_ms();
                if now_poly_ms >= ws_end_sec * 1000 {
                    break;
                }

                let should_restart_ws = ids != subscribed;

                if should_restart_ws {
                    let remain_ms = (market_end_ms - current_timestamp_ms()).max(0) as u64;

                    run_log::ws_start(
                        period,
                        &slug,
                        price_to_beat,
                        &market_ids,
                        &ids,
                        remain_ms,
                        market_end_ms,
                    );

                    let cmd = WsCommand::ActivateWindow(MarketWsSubscription {
                        period,
                        slug: slug.clone(),
                        asset_ids: ids.clone(),
                        market_ids: market_ids.clone(),
                        period_sec,
                        window_start_sec,
                        gamma_question: gamma_question.clone(),
                    });
                    let tx = match period_sec {
                        FIVE_MIN_SEC => &self.market_ws_tx_5m,
                        FIFTEEN_MIN_SEC => &self.market_ws_tx_15m,
                        _ => &self.market_ws_tx_15m,
                    };
                    if tx.send(cmd).await.is_err() {
                        run_log::ws_spawn_err(period, &slug, "market ws command channel closed");
                        continue;
                    }

                    subscribed = ids.clone();
                    prev_market_id = market_ids.first().cloned();
                    prev_gamma_question = gamma_question.clone();
                }

                tick.tick().await;
            }

            run_log::ws_window_end_wait(period, &slug, subscribed.len());
            if !ids.is_empty() {
                let tx = match period_sec {
                    FIVE_MIN_SEC => &self.market_ws_tx_5m,
                    FIFTEEN_MIN_SEC => &self.market_ws_tx_15m,
                    _ => &self.market_ws_tx_15m,
                };
                match tx.send(WsCommand::PruneStaleIds { stale_ids: ids.clone() }).await {
                    Ok(()) => {}
                    Err(_) => run_log::ws_spawn_err(period, &slug, "market ws command channel closed"),
                }
            }
            if let Some(prev_market_id) = prev_market_id.as_ref() {
                let interval_kind = XFrameIntervalKind::from_period_sec(period_sec);
                xframe_dump::spawn_dump_market_xframes_binary(
                    self.clone(),
                    prev_market_id.clone(),
                    prev_gamma_question.clone(),
                    interval_kind,
                );
            }
        }
    }
}

/// Противоположная нога / sibling: кадр из текущего батча, иначе уже сохранённый с тем же `aligned_ts`, иначе последний с `aligned_ts` ≤ запрошенного.
fn lookup_frame_for_leg_merge<'a>(
    market_id: &str,
    asset_id: &str,
    aligned_ts: i64,
    batch: &'a HashMap<(String, String, i64), XFrame<SIZE>>,
    stored: &'a MarketFrames,
) -> Option<&'a XFrame<SIZE>> {
    if let Some(frame) = batch.get(&(market_id.to_string(), asset_id.to_string(), aligned_ts)) {
        return Some(frame);
    }
    let by_asset = stored.get(market_id)?;
    let by_ts = by_asset.get(asset_id)?;
    if let Some(frame) = by_ts.get(&aligned_ts) {
        return Some(frame);
    }
    by_ts.range(..=aligned_ts).next_back().map(|(_, frame)| frame)
}

/// `(price_to_beat - currency_spot) / price_to_beat * 100` — отклонение спота от уровня «beat» в процентах; знак «+», если спот ниже beat.
fn currency_price_vs_price_to_beat_pct(
    price_to_beat: Option<f64>,
    currency_spot_usd: Option<f64>,
) -> Option<f64> {
    const MIN_BEAT: f64 = 1e-6;
    let beat = price_to_beat?;
    if !beat.is_finite() || beat.abs() <= MIN_BEAT {
        return None;
    }
    let spot = currency_spot_usd?;
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
