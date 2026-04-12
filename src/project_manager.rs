use crate::market_snapshot::aggregate_events;
use anyhow::bail;
use crate::currency_ws::RtdsCurrencyLatest;
use crate::constants::XFrameIntervalKind;
use crate::data_ws::spawn_bounded_market_ws;
use crate::run_log;
use crate::util::{
    current_timestamp_ms, fetch_gamma_event_data_for_slug,
    fetch_price_to_beat_from_polymarket_event_page, CurrencyEventSlugData,
};
use crate::xframe_dump;
use std::time::Instant;

pub use crate::currency_updown_sibling::{
    five_min_belongs_to_fifteen_window, CurrencyUpDownSiblingSlot, CurrencyUpDownSiblingWsState,
};
pub use crate::constants::{CurrencyUpDownInterval, FIFTEEN_MIN_SEC, FIVE_MIN_SEC};
use crate::data_ws::{
    make_ws_channel, CurrencyUpDownOutcome, MarketSnapshot, MarketSnapshotBuffer,
    MarketSnapshotBufferMut, Ws,
};
use crate::xframe::{
    currency_price_z_score_from_sec_history, compute_xframe_stable, find_opposite_asset_id,
    find_same_outcome_sibling_asset_id, XFrame, SIZE,
};
use polymarket_client_sdk::clob;
use polymarket_client_sdk::gamma;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{self, Duration};
use crate::currency_updown_sibling::on_currency_updown_ws_connected;

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
    pub currency: String,
    pub xframes_by_market: Arc<RwLock<MarketFrames>>,
    pub ws_buffer_by_market: Arc<RwLock<MarketSnapshotBuffer>>,
    pub event_data_by_market: Arc<RwLock<HashMap<String, MarketEventData>>>,
    pub currency_up_down_by_asset_id: Arc<RwLock<HashMap<String, CurrencyUpDownOutcome>>>,
    pub ws_connect_wall_ms_by_market: Arc<RwLock<HashMap<String, i64>>>,
    pub currency_updown_sibling_ws_state: Arc<RwLock<CurrencyUpDownSiblingWsState>>,
    pub rtds_currency_latest: Arc<RwLock<Option<RtdsCurrencyLatest>>>,
    pub rtds_currency_prices_by_sec: Arc<RwLock<BTreeMap<i64, f64>>>,
    pub ws: Arc<Ws>,
    pub http: Arc<reqwest::Client>,
    pub gamma: Arc<gamma::Client>,
    pub clob: Arc<clob::Client>,
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

        let project_manager = Arc::new(Self {
            currency,
            xframes_by_market: Arc::new(RwLock::new(HashMap::new())),
            ws_buffer_by_market: Arc::new(RwLock::new(HashMap::new())),
            event_data_by_market: Arc::new(RwLock::new(HashMap::new())),
            currency_up_down_by_asset_id: Arc::new(RwLock::new(HashMap::<String, CurrencyUpDownOutcome>::new())),
            ws_connect_wall_ms_by_market: Arc::new(RwLock::new(HashMap::new())),
            currency_updown_sibling_ws_state: Arc::new(RwLock::new(
                CurrencyUpDownSiblingWsState::default(),
            )),
            rtds_currency_latest: Arc::new(RwLock::new(None)),
            rtds_currency_prices_by_sec: Arc::new(RwLock::new(BTreeMap::new())),
            ws,
            http,
            gamma,
            clob,
        });

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
        starts: HashMap<String, Option<i64>>,
        ends: HashMap<String, Option<i64>>,
        price_to_beat: Option<f64>,
        gamma_question: Option<String>,
        currency_up_down_by_asset_id: HashMap<String, CurrencyUpDownOutcome>,
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

        if !currency_up_down_by_asset_id.is_empty() {
            let mut map = self.currency_up_down_by_asset_id.write().await;
            for (asset_id, code) in currency_up_down_by_asset_id {
                map.insert(asset_id, code);
            }
        }
    }

    /// Вызывать при перезапуске подписки, сразу после лога старта WS и до `spawn_bounded_market_ws` (`btc-updown-5m-*` / `btc-updown-15m-*`).
    /// Обновляет [`Self::currency_updown_sibling_ws_state`]; пару для sibling-признаков читает [`CurrencyUpDownSiblingWsState::paired_five_and_fifteen_market_ids`].
    pub async fn on_currency_updown_ws_connected(
        &self,
        interval_sec: i64,
        slug_window_start_unix_sec: i64,
        market_ids: &[String],
        gamma_question: Option<&str>) {
        on_currency_updown_ws_connected(
            self.currency_updown_sibling_ws_state.clone(),
            interval_sec,
            slug_window_start_unix_sec,
            market_ids,
            gamma_question,
        ).await;
    }

    /// Вызывать сразу после успешного `spawn_bounded_market_ws`: фиксирует wall-time подключения WS по `market_id` для [`crate::xframe::XFrame::stable`].
    pub async fn record_currency_updown_ws_connect(&self, market_ids: &[String]) {
        let now_ms = current_timestamp_ms();
        let mut lock = self.ws_connect_wall_ms_by_market.write().await;
        for id in market_ids {
            lock.insert(id.clone(), now_ms);
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
            let price_to_beat = event_data.and_then(|t| t.price_to_beat);
            let gamma_question_owned = event_data.and_then(|t| t.gamma_question.clone());
            let event_start_ms = event_data.and_then(|t| t.start_ms);
            drop(event_guard);

            let ws_connect_wall_ms = {
                let ws_connect_wall_ms_map_guard = self.ws_connect_wall_ms_by_market.read().await;
                ws_connect_wall_ms_map_guard.get(&market_id).copied()
            };

            let currency_price_vs_beat_pct =
                currency_price_vs_price_to_beat_pct(price_to_beat, currency_spot_usd);

            let window_ms = FRAME_BUILD_INTERVAL_SEC as i64 * 1000;
            let stable = compute_xframe_stable(
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

        let sibling_market_by_market: HashMap<String, String> = {
            let sibling_ws_state_read_guard = self.currency_updown_sibling_ws_state.read().await;
            let mut sibling_market_lookup = HashMap::new();
            if let Some((five_market_id, fifteen_market_id)) = sibling_ws_state_read_guard.paired_five_and_fifteen_market_ids() {
                sibling_market_lookup.insert(five_market_id.clone(), fifteen_market_id.clone());
                sibling_market_lookup.insert(fifteen_market_id, five_market_id);
            }
            sibling_market_lookup
        };

        {
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
                if let Some(by_asset) = xframes_stored.get(sibling_market_id) {
                    sibling_candidates.extend(by_asset.keys().cloned());
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

    pub async fn run_currency_updown_interval(self: Arc<Self>, period_sec: i64, slug_mid: &'static str) {
        let xframe_interval_kind = match period_sec {
            ps if ps == FIVE_MIN_SEC => XFrameIntervalKind::FiveMin,
            ps if ps == FIFTEEN_MIN_SEC => XFrameIntervalKind::FifteenMin,
            _ => XFrameIntervalKind::FifteenMin,
        };

        let mut tick = tokio::time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tick.tick().await;
            let now_ms = current_timestamp_ms();
            let poly_sec = now_ms / 1000;
            let window_start_sec = (poly_sec / period_sec) * period_sec;
            let ws_end_sec = window_start_sec + period_sec;

            if now_ms >= ws_end_sec * 1000 {
                continue;
            }

            let mut ws_handle: Option<tokio::task::JoinHandle<()>> = None;
            let mut subscribed: Vec<String> = Vec::new();
            let mut ws_session_market_id: Option<String> = None;
            let mut ws_session_gamma_question: Option<String> = None;

            while current_timestamp_ms() < ws_end_sec * 1000 {
                let now_poly_ms = current_timestamp_ms();
                if now_poly_ms >= ws_end_sec * 1000 {
                    break;
                }

                let slug = format!(
                    "{}-updown-{slug_mid}-{window_start_sec}",
                    self.currency.to_lowercase(),
                );
                let http_gamma = self.http.clone();
                let http_price = self.http.clone();
                let slug_gamma = slug.clone();
                let slug_price = slug.clone();
                let currency = self.currency.clone();

                let (gamma_join_res, price_join_res) = tokio::join!(
                    tokio::spawn(async move {
                        fetch_gamma_event_data_for_slug(http_gamma.as_ref(), &slug_gamma).await
                    }),
                    tokio::spawn(async move {
                        fetch_price_to_beat_from_polymarket_event_page(
                            http_price.as_ref(),
                            &slug_price,
                            &currency,
                        )
                        .await
                    }),
                );

                let (
                    ids,
                    market_event_start_ms,
                    market_event_end_ms,
                    gamma_question,
                    currency_up_down_by_asset_id,
                ) = match gamma_join_res {
                    Ok(Ok(CurrencyEventSlugData {
                        clob_token_ids,
                        currency_up_down_by_asset_id,
                        market_event_start_ms,
                        market_event_end_ms,
                        gamma_question,
                    })) => (
                        clob_token_ids,
                        market_event_start_ms,
                        market_event_end_ms,
                        gamma_question,
                        currency_up_down_by_asset_id,
                    ),
                    Ok(Err(e)) => {
                        run_log::gamma_fetch_err(slug_mid, &slug, &e);
                        continue;
                    }
                    Err(e) => {
                        run_log::gamma_fetch_err(slug_mid, &slug, &format!("join gamma: {e}"));
                        continue;
                    }
                };

                let price_to_beat = match price_join_res {
                    Ok(Ok(price_to_beat)) => Some(price_to_beat),
                    Ok(Err(e)) => {
                        run_log::gamma_fetch_err(slug_mid, &slug, &e);
                        continue;
                    }
                    Err(e) => {
                        run_log::gamma_fetch_err(
                            slug_mid,
                            &slug,
                            &format!("join price_to_beat: {e}"),
                        );
                        continue;
                    }
                };

                let wall_end_ms = market_event_end_ms
                    .values()
                    .copied()
                    .flatten()
                    .max()
                    .unwrap_or(ws_end_sec * 1000);

                let market_ids: Vec<String> = market_event_end_ms.keys().cloned().collect();

                self.merge_market_event_data(
                    market_event_start_ms,
                    market_event_end_ms,
                    price_to_beat,
                    gamma_question.clone(),
                    currency_up_down_by_asset_id,
                )
                .await;

                let should_restart_ws = ids != subscribed;

                if should_restart_ws {
                    if let Some(h) = ws_handle.take() {
                        run_log::ws_stop_replace(slug_mid, &slug, subscribed.len());
                        h.abort();
                    }

                    let remain_ms = (wall_end_ms - current_timestamp_ms()).max(0) as u64;
                    let session_deadline = Instant::now() + Duration::from_millis(remain_ms);

                    run_log::ws_start(
                        slug_mid,
                        &slug,
                        price_to_beat,
                        &market_ids,
                        &ids,
                        remain_ms,
                        wall_end_ms,
                    );

                    self.on_currency_updown_ws_connected(
                        period_sec,
                        window_start_sec,
                        &market_ids,
                        gamma_question.as_deref(),
                    ).await;

                    match spawn_bounded_market_ws(
                        self.clone(),
                        ids.clone(),
                        session_deadline,
                        xframe_interval_kind,
                    ) {
                        Ok(h) => {

                            self.record_currency_updown_ws_connect(&market_ids).await;
                            ws_handle = Some(h);
                            subscribed = ids.clone();
                            ws_session_market_id = market_ids.first().cloned();
                            ws_session_gamma_question = gamma_question.clone();
                        }
                        Err(e) => {
                            run_log::ws_spawn_err(slug_mid, &slug, &e);
                            continue;
                        }
                    }
                }

                tick.tick().await;
            }

            if let Some(h) = ws_handle.take() {
                let slug = format!(
                    "{}-updown-{slug_mid}-{window_start_sec}",
                    self.currency.to_lowercase(),
                );
                run_log::ws_window_end_wait(slug_mid, &slug, subscribed.len());
                let _ = h.await;
                run_log::ws_task_joined(slug_mid, &slug);
                if let Some(mid) = ws_session_market_id {
                    xframe_dump::spawn_dump_market_xframes_binary(
                        self.clone(),
                        mid,
                        ws_session_gamma_question,
                    );
                }
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
