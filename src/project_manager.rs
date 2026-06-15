use crate::account::SharedAccount;
use crate::constants::XFrameIntervalKind;
pub use crate::constants::{CurrencyUpDownInterval, FIFTEEN_MIN_SEC, FIVE_MIN_SEC};
pub use crate::currency_updown_sibling::{
    CurrencyUpDownSiblingSlot, CurrencyUpDownSiblingState, five_min_belongs_to_fifteen_window,
};
use crate::currency_ws::{RTDS_MS_MAX_LAG_FOR_STABLE_FRAME, rtds_spot_pair_symbol};
use crate::data_ws::{
    CurrencyUpDownOutcome, MarketSnapshot, MarketSnapshotBuffer, MarketSnapshotBufferMut,
    MarketWsSubscription, Ws, WsCommand, make_ws_channel, spawn_persistent_interval_market_ws,
};
use crate::market_snapshot::aggregate_events;
use crate::run_log;
use crate::util::{
    CurrencyEventSlugData, current_timestamp_ms, fetch_gamma_event_data_for_gamma_client,
    fetch_price_to_beat_from_vatic_api,
};
use crate::xframe::{
    SIZE, XFrame, compute_xframe_stable, currency_price_z_score_from_sec_history,
    find_opposite_asset_id, find_same_outcome_sibling_asset_id,
};
use crate::xframe_dump;
use anyhow::bail;
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::gamma;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::time::{self, Duration};

type MarketFrames = HashMap<String, HashMap<String, BTreeMap<i64, XFrame<SIZE>>>>;

/// Prev-маркет для дампа: `final_price` = exact текущего окна, `price_to_beat` = exact из `exact_price_to_beat_rx` (без повторного HTTP).
struct PrevMarket {
    market_id: Option<String>,
    gamma_question: Option<String>,
    /// Для проверки `current_window_start_sec == prev + period` — иначе дамп prev пропускаем.
    window_start_sec: i64,
    /// Exact PTB от refine этой итерации; следующий refine ждёт для пары в дампе.
    exact_price_to_beat_rx: oneshot::Receiver<f64>,
    /// `event_end_ms` маркета от Gamma (`endDate` через
    event_end_ms: Option<i64>,
}

/// Стабильный кадр лейна 1s → [`real_sim`](crate::real_sim) по каналам (без поллинга `xframes_by_market`).
/// `market_id` / `asset_id` / интервал / сторона — в [`XFrame`]; окно Gamma — в
/// [`ProjectManager::event_data_by_market`] по `frame.market_id`.
#[derive(Clone, Debug)]
pub struct LaneFrame {
    /// Кэш PTB на момент фанаута → CSV; `None`, если страница ещё не дала значение.
    pub price_to_beat: Option<f64>,
    /// Текст вопроса Gamma для имени дампа и синтетического пути `.bin` в CSV ([`crate::xframe_dump::synthetic_xframes_dump_bin_path_for_csv_link`]).
    pub gamma_question: Option<String>,
    pub frame: XFrame<SIZE>,
}

#[derive(Clone, Debug)]
pub struct WsStreamEntry {
    pub market_id: String,
    pub asset_id: String,
    pub event_type: String,
    pub ingest_wall_ms: i64,
    pub event_timestamp_ms: i64,
    pub payload_raw: String,
}

/// Результат одного тика сборщика до записи в `xframes_by_market`.
struct BuiltXframeEntry {
    market_id: String,
    asset_id: String,
    aligned_ts: i64,
    frame: XFrame<SIZE>,
}

/// Интервал(s) тика сборщика по лейну; один лейн — все маркеты на этом шаге.
pub const FRAME_BUILD_INTERVALS_SEC: [u64; 1] = [1];
/// Индекс лейна 1s в [`FRAME_BUILD_INTERVALS_SEC`] и `xframes_by_market`.
pub const XFRAMES_LANE_1S: usize = 0;
/// Очередь команд единого market WS (5m+15m).
const MARKET_WS_SUBSCRIPTION_CHANNEL_CAP: usize = 8;

#[derive(Debug, Clone, Default)]
pub struct MarketEventData {
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
    pub min_order_size: Option<f64>,
    pub gamma_question: Option<String>,
}

/// Резолюционное состояние маркета: порог (`price_to_beat`, спот в начале окна,
/// фетчится из Vatic API) и финальная цена (`final_price`, спот в конце окна =
/// `price_to_beat` следующего окна). По правилу `final_price >= price_to_beat`
/// определяется победившая сторона UP/DOWN — нужно для post-market-end
/// финализации в [`crate::account_close_position::close_position_submit_resolution`]:
/// несённые residual-шеры payout'ятся `1.0` (наша сторона выиграла) или `0.0`.
///
/// `final_price = None` до тех пор, пока следующее окно не подтянет свой
/// `current_exact` — тогда [`ProjectManager::merge_market_final_price`] апдейтит
/// поле у prev-market. До этого момента residual-резолюция вынуждена скипать.
#[derive(Debug, Clone, Copy)]
pub struct MarketResolution {
    pub price_to_beat: f64,
    pub final_price: Option<f64>,
}

/// Capacity-cap для [`ProjectManager::market_resolution_by_market`]: при
/// `len > MARKET_RESOLUTION_RETENTION` дёргаем `BTreeMap::pop_first` (старейший
/// по `market_id` лексикографически — backstop поверх явного
/// [`ProjectManager::cleanup_stale_market_data`], чтобы кэш не рос неограниченно
/// даже если cleanup сломан/пропущен).
pub const MARKET_RESOLUTION_RETENTION: usize = 10;

pub struct ProjectManager {
    pub currency: Arc<String>,
    pub xframes_by_market: Vec<RwLock<MarketFrames>>,
    pub ws_buffer_by_market: Vec<RwLock<MarketSnapshotBuffer>>,
    pub ws_stream_by_asset_id: Arc<RwLock<HashMap<String, Vec<WsStreamEntry>>>>,
    pub event_data_by_market: Arc<RwLock<HashMap<String, MarketEventData>>>,
    pub slug_to_market_id: Arc<RwLock<HashMap<String, String>>>,
    pub market_resolution_by_market: Arc<RwLock<BTreeMap<String, MarketResolution>>>,
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
    pub market_ws_tx: mpsc::Sender<WsCommand>,
    pub xframe_interval_kind_by_asset_id: Arc<RwLock<HashMap<String, XFrameIntervalKind>>>,
    pub last_snapshot_by_asset_id: Arc<RwLock<HashMap<String, MarketSnapshot>>>,
    pub account: SharedAccount,
}

impl ProjectManager {
    pub fn clob_authed(&self) -> Option<clob::Client<Authenticated<Normal>>> {
        (**self.account.clob_authed.load()).clone()
    }

    /// WS, сборщик XFrame, циклы 5m/15m. Карта каналов [`LaneFrameChannels`](crate::real_sim::LaneFrameChannels) пуста до регистрации воркерами `real_sim`.
    ///
    /// `http` — [`SharedAccount::http`](crate::account::Account::http);
    /// `gamma` — [`SharedAccount::gamma`](crate::account::Account::gamma).
    ///
    /// CLOB-клиент создаётся ровно в [`crate::account::Account::new`] и
    /// переиспользуется: один пул соединений / DNS-кэш на все валюты плюс
    /// [`crate::authenticate::spawn_heartbeat`].
    pub fn new(currency: String, account: SharedAccount) -> Arc<Self> {
        let (ws, mut ws_snapshot_receiver) = make_ws_channel();

        let http = account.http.clone();
        let gamma = account.gamma.clone();
        // CLOB-клиент берём из `Account` (single source of truth, см.
        // `Account::clob` doc). `clone` ⇒ клон `Arc` без `await` под
        // локом — тот же `ClientInner` внутри SDK.
        let clob = account.clob.clone();

        let (market_ws_tx, market_ws_rx) =
            mpsc::channel::<WsCommand>(MARKET_WS_SUBSCRIPTION_CHANNEL_CAP);

        let project_manager = Arc::new(Self {
            currency: Arc::new(currency.clone()),
            xframes_by_market: (0..FRAME_BUILD_INTERVALS_SEC.len())
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
            ws_buffer_by_market: (0..FRAME_BUILD_INTERVALS_SEC.len())
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
            ws_stream_by_asset_id: Arc::new(RwLock::new(HashMap::new())),
            event_data_by_market: Arc::new(RwLock::new(HashMap::new())),
            slug_to_market_id: Arc::new(RwLock::new(HashMap::new())),
            market_resolution_by_market: Arc::new(RwLock::new(BTreeMap::new())),
            currency_up_down_by_asset_id: Arc::new(RwLock::new(HashMap::<
                String,
                CurrencyUpDownOutcome,
            >::new())),
            ws_connect_wall_ms_by_asset_id: Arc::new(RwLock::new(HashMap::new())),
            currency_updown_sibling_state: Arc::new(RwLock::new(
                CurrencyUpDownSiblingState::default(),
            )),
            rtds_currency_prices_by_ms: Arc::new(RwLock::new(BTreeMap::new())),
            rtds_currency_prices_by_sec: Arc::new(RwLock::new(BTreeMap::new())),
            market_asset_ids_by_market: Arc::new(RwLock::new(HashMap::new())),
            ws,
            http,
            gamma,
            clob,
            market_ws_tx,
            xframe_interval_kind_by_asset_id: Arc::new(RwLock::new(HashMap::new())),
            last_snapshot_by_asset_id: Arc::new(RwLock::new(HashMap::new())),
            account,
        });

        spawn_persistent_interval_market_ws(project_manager.clone(), market_ws_rx);

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

        project_manager.clone().run_frame_builder_loop();

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

    pub async fn merge_market_event_data(&self, data: &CurrencyEventSlugData, slug: &str) {
        if let Some(market_id) = &data.market_id {
            let mut event_data_by_market_lock = self.event_data_by_market.write().await;
            let entry = event_data_by_market_lock
                .entry(market_id.clone())
                .or_default();
            entry.start_ms = data.event_start_ms;
            entry.end_ms = data.event_end_ms;
            entry.min_order_size = data.min_order_size;
            if let Some(ref q) = data.gamma_question {
                entry.gamma_question = Some(q.clone());
            }
            drop(event_data_by_market_lock);

            let mut slug_to_market_id_lock = self.slug_to_market_id.write().await;
            slug_to_market_id_lock.insert(slug.to_string(), market_id.clone());

            if !data.currency_up_down_by_asset_id.is_empty() {
                let mut currency_up_down_by_asset_id_lock =
                    self.currency_up_down_by_asset_id.write().await;
                for (asset_id, code) in data.currency_up_down_by_asset_id.iter() {
                    currency_up_down_by_asset_id_lock.insert(asset_id.clone(), *code);
                }
                drop(currency_up_down_by_asset_id_lock);
                let mut market_asset_ids_lock = self.market_asset_ids_by_market.write().await;
                market_asset_ids_lock
                    .entry(market_id.clone())
                    .or_default()
                    .extend(data.currency_up_down_by_asset_id.keys().cloned());
            }
        } else if !data.currency_up_down_by_asset_id.is_empty() {
            let mut currency_up_down_by_asset_id_lock =
                self.currency_up_down_by_asset_id.write().await;
            for (asset_id, code) in data.currency_up_down_by_asset_id.iter() {
                currency_up_down_by_asset_id_lock.insert(asset_id.clone(), *code);
            }
        }
    }

    /// Как [`fetch_gamma_event_data_for_gamma_client`](crate::util::fetch_gamma_event_data_for_gamma_client), но из кэшей PM.
    async fn try_restore_currency_event_from_slug_cache(
        &self,
        slug: &str,
    ) -> Option<CurrencyEventSlugData> {
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

        Some(CurrencyEventSlugData {
            currency_up_down_by_asset_id,
            market_id: Some(market_id),
            event_start_ms: event_data.start_ms,
            event_end_ms: event_data.end_ms,
            min_order_size: event_data.min_order_size,
            gamma_question: event_data.gamma_question,
        })
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

    /// Снос всех кэшей по `market_id`.
    pub async fn cleanup_stale_market_data(&self, market_id: &str) {
        let asset_ids: HashSet<String> = self
            .market_asset_ids_by_market
            .write()
            .await
            .remove(market_id)
            .unwrap_or_default();

        for xframes_by_market in &self.xframes_by_market {
            xframes_by_market.write().await.remove(market_id);
        }
        for ws_buffer_by_market in &self.ws_buffer_by_market {
            ws_buffer_by_market.write().await.remove(market_id);
        }
        {
            let mut ws_stream_by_asset_id = self.ws_stream_by_asset_id.write().await;
            for asset_id in &asset_ids {
                ws_stream_by_asset_id.remove(asset_id);
            }
        }
        self.event_data_by_market.write().await.remove(market_id);
        // `market_resolution_by_market` намеренно НЕ дёргаем здесь: post-market-end
        // финализация submit-режима читает её через ~5s после `event_end_ms`, а
        // cleanup может прилететь раньше (xframe_dump завершается быстро при
        // успешном дампе). BTreeMap с capacity-cap [`MARKET_RESOLUTION_RETENTION`]
        // сам вытеснит старые записи на следующем `merge_*`.

        {
            let mut currency_up_down_by_asset_id_lock =
                self.currency_up_down_by_asset_id.write().await;
            for asset_id in &asset_ids {
                currency_up_down_by_asset_id_lock.remove(asset_id);
            }
        }
        {
            let mut ws_connect_wall_ms_by_asset_id_lock =
                self.ws_connect_wall_ms_by_asset_id.write().await;
            for asset_id in &asset_ids {
                ws_connect_wall_ms_by_asset_id_lock.remove(asset_id);
            }
        }
        {
            let mut interval_by_asset = self.xframe_interval_kind_by_asset_id.write().await;
            for asset_id in &asset_ids {
                interval_by_asset.remove(asset_id);
            }
        }
        {
            let mut last_snapshot_by_asset_id = self.last_snapshot_by_asset_id.write().await;
            for asset_id in &asset_ids {
                last_snapshot_by_asset_id.remove(asset_id);
            }
        }
        {
            let mut slugs = self.slug_to_market_id.write().await;
            slugs.retain(|_, v| v != market_id);
        }
    }

    /// Поддерживает «живой» агрегированный снимок WS-стрима по `asset_id` для
    /// быстрого StrictBook'а в [`crate::real_sim::tick_once`] (см.
    /// [`Self::last_snapshot_by_asset_id`]).
    ///
    /// Новый WS-event мерджится поверх предыдущего через
    /// [`aggregate_events`]: полные лестницы `book_bids/asks` сохраняются, если
    /// текущий event их не несёт (например, `price_change` со чисто L1
    /// `best_bid`/`best_ask`), `timestamp_ms` — `max` по входу. Без этого слепое
    /// перезаписывание оставило бы кэш бесполезным после первого `book` —
    /// последующие `price_change`-события не имеют bids/asks и StrictBook
    /// собрать было бы нельзя.
    pub async fn update_last_snapshot(&self, snapshot: &MarketSnapshot) {
        let mut snapshot = snapshot.clone();
        if let Some(event_data) = self
            .event_data_by_market
            .read()
            .await
            .get(&snapshot.market_id)
        {
            snapshot.min_order_size = event_data.min_order_size;
        }
        let mut last_snapshot_by_asset_id = self.last_snapshot_by_asset_id.write().await;
        let merged = match last_snapshot_by_asset_id.remove(&snapshot.asset_id) {
            Some(prev) => aggregate_events(vec![prev, snapshot.clone()], snapshot.timestamp_ms)
                .unwrap_or_else(|| snapshot.clone()),
            None => snapshot.clone(),
        };
        last_snapshot_by_asset_id.insert(snapshot.asset_id.clone(), merged);
    }

    pub async fn append_ws_stream_entries(&self, entries: Vec<WsStreamEntry>) {
        if entries.is_empty() {
            return;
        }
        let mut ws_stream_by_asset_id = self.ws_stream_by_asset_id.write().await;
        for entry in entries {
            ws_stream_by_asset_id
                .entry(entry.asset_id.clone())
                .or_default()
                .push(entry);
        }
    }

    /// Gamma + [`merge_market_event_data`]; возврат как у [`try_restore_currency_event_from_slug_cache`](Self::try_restore_currency_event_from_slug_cache).
    async fn fetch_currency_event_from_gamma_and_merge(
        &self,
        slug: &str,
        period: &'static str,
    ) -> Option<CurrencyEventSlugData> {
        let data = match fetch_gamma_event_data_for_gamma_client(self.gamma.as_ref(), slug).await
        {
            Ok(d) => d,
            Err(e) => {
                run_log::gamma_fetch_err(period, slug, &e);
                return None;
            }
        };
        self.merge_market_event_data(&data, slug).await;
        Some(data)
    }

    /// Один `market_id` на окно up/down — одно значение PTB в кэше. Сохраняет
    /// уже записанный `final_price` (если был выставлен через
    /// [`Self::merge_market_final_price`] — теоретически возможно при out-of-order
    /// колбэке). Evicts старейшие записи если `len > MARKET_RESOLUTION_RETENTION`.
    pub async fn merge_market_price_to_beat(&self, price_to_beat: f64, market_id: &str) {
        let mut map = self.market_resolution_by_market.write().await;
        map.entry(market_id.to_string())
            .and_modify(|entry| entry.price_to_beat = price_to_beat)
            .or_insert(MarketResolution {
                price_to_beat,
                final_price: None,
            });
        while map.len() > MARKET_RESOLUTION_RETENTION {
            map.pop_first();
        }
    }

    /// Выставляет `final_price` для завершившегося маркета (вызывается из
    /// [`spawn_bg_price_to_beat_refine`] следующего окна, чьё `current_exact` =
    /// `final_price` предыдущего окна). Если `price_to_beat` ещё не известен —
    /// пишем placeholder-запись с теми же значениями (final_price=spot,
    /// price_to_beat=spot), это лучше чем терять final_price: на сторону
    /// потребителя (`close_position_submit_resolution`) такая запись даст
    /// `up_won = (spot >= spot) = true` — приемлемая дефолтная политика для
    /// крайне редкого race. Evicts при `len > MARKET_RESOLUTION_RETENTION`.
    pub async fn merge_market_final_price(&self, final_price: f64, market_id: &str) {
        let mut map = self.market_resolution_by_market.write().await;
        map.entry(market_id.to_string())
            .and_modify(|entry| entry.final_price = Some(final_price))
            .or_insert(MarketResolution {
                price_to_beat: final_price,
                final_price: Some(final_price),
            });
        while map.len() > MARKET_RESOLUTION_RETENTION {
            map.pop_first();
        }
    }

    /// После подписки на market WS — wall time для [`compute_xframe_stable`](crate::xframe::compute_xframe_stable).
    pub async fn record_ws_connect_wall_ms_for_asset_ids(&self, asset_ids: &[String]) {
        let now_ms = current_timestamp_ms();
        let mut ws_connect_wall_ms_by_asset_id_lock =
            self.ws_connect_wall_ms_by_asset_id.write().await;
        for asset_id in asset_ids {
            ws_connect_wall_ms_by_asset_id_lock.insert(asset_id.clone(), now_ms);
        }
    }

    pub async fn ingest_snapshot(&self, mut snapshot: MarketSnapshot) -> anyhow::Result<()> {
        let Some(currency_up_down_outcome) = self
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
        snapshot.currency_up_down_outcome = currency_up_down_outcome;

        // Prefetched markets (окно ещё не началось, `event_start_ms > now_ms`):
        // WS-подписка для них активна заранее, но snapshot'ы из «будущего» нам
        // в buffer не нужны — иначе frame-builder построит для них xframes
        // без PTB (start окна не наступил, цена-якорь неизвестна) и забьёт
        // diag-лог. Дропаем на входе.
        let event_data_guard = self.event_data_by_market.read().await;
        if let Some(event_data) = event_data_guard.get(&snapshot.market_id) {
            snapshot.min_order_size = event_data.min_order_size;
            if let Some(start_ms) = event_data.start_ms
                && start_ms > current_timestamp_ms()
            {
                return Ok(());
            }
        }
        drop(event_data_guard);

        for ws_buffer_by_market in &self.ws_buffer_by_market {
            let mut ws_buffer_by_market_lock = ws_buffer_by_market.write().await;
            ws_buffer_by_market_lock.push_snapshot(snapshot.clone());
        }
        Ok(())
    }

    /// Спавнит цикл на каждый лейн: тик раз в `FRAME_BUILD_INTERVALS_SEC[i]` с.
    pub fn run_frame_builder_loop(self: Arc<Self>) {
        for lane in 0..FRAME_BUILD_INTERVALS_SEC.len() {
            let project_manager = self.clone();
            tokio::spawn(async move {
                let secs = FRAME_BUILD_INTERVALS_SEC[lane];
                let mut interval = time::interval(Duration::from_secs(secs));
                loop {
                    interval.tick().await;
                    project_manager
                        .build_frames_from_buffer_lane_once(lane)
                        .await;
                }
            });
        }
    }

    pub async fn build_frames_from_buffer_lane_once(&self, lane: usize) {
        let drained = {
            let mut buf = self.ws_buffer_by_market[lane].write().await;
            buf.drain_all()
        };

        if drained.is_empty() {
            return;
        }

        let interval_secs = FRAME_BUILD_INTERVALS_SEC[lane];

        let mut by_asset_group: HashMap<(String, String), Vec<MarketSnapshot>> = HashMap::new();
        for (market_id, by_asset) in drained {
            for (asset_id, events) in by_asset {
                if events.is_empty() {
                    continue;
                }
                by_asset_group
                    .entry((market_id.clone(), asset_id))
                    .or_default()
                    .extend(events);
            }
        }

        if by_asset_group.is_empty() {
            return;
        }

        let now_ms = current_timestamp_ms();
        let currency_ref_sec = now_ms / 1000;
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

        let (rtds_ms_fresh, rtds_last_key_ms) = {
            let g = self.rtds_currency_prices_by_ms.read().await;
            let last_key = g.iter().next_back().map(|(&ts, _)| ts);
            let fresh = match last_key {
                None => false,
                Some(ts) => now_ms.saturating_sub(ts) <= RTDS_MS_MAX_LAG_FOR_STABLE_FRAME,
            };
            (fresh, last_key)
        };
        if !rtds_ms_fresh {
            let rtds_symbol = rtds_spot_pair_symbol(self.currency.as_str());
            run_log::rtds_currency_prices_lagging_for_xframe(
                self.currency.as_str(),
                rtds_symbol.as_str(),
                now_ms,
                rtds_last_key_ms,
                RTDS_MS_MAX_LAG_FOR_STABLE_FRAME,
            );
        }

        let mut built_xframes: Vec<BuiltXframeEntry> = Vec::new();

        for ((market_id, asset_id), group) in by_asset_group {
            let Some(snapshot) = aggregate_events(group, 0) else {
                continue;
            };
            let aligned_ts =
                align_timestamp_ms_to_interval(snapshot.timestamp_ms, interval_secs);
            let frames_history = {
                let xframes_by_market_read_lock = self.xframes_by_market[lane].read().await;
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
                let map = self.market_resolution_by_market.read().await;
                map.get(&market_id).map(|entry| entry.price_to_beat)
            };

            let ws_connect_wall_ms = {
                let ws_connect_wall_ms_by_asset_id_lock =
                    self.ws_connect_wall_ms_by_asset_id.read().await;
                ws_connect_wall_ms_by_asset_id_lock.get(&asset_id).copied()
            };

            let currency_price_vs_beat_pct =
                currency_price_vs_price_to_beat_pct(price_to_beat, currency_spot_usd);

            let window_ms = interval_secs as i64 * 1000;
            let stable = compute_xframe_stable(
                market_id.as_str(),
                snapshot.timestamp_ms,
                event_start_ms,
                ws_connect_wall_ms,
            ) && rtds_ms_fresh;
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

            // let last_stored_aligned_ts_for_asset: Option<i64> = {
            //     let lock = self.xframes_by_market[lane].read().await;
            //     lock.get(&market_id)
            //         .and_then(|by_asset| by_asset.get(&asset_id))
            //         .and_then(|m| m.keys().next_back().copied())
            // };

            // if frame.stable {
            //     println!(
            //         "[diag][builder] xframe lane={} mkt={} asset={} aligned_ts={} hist_len={} last_stored_ts={:?} delta_to_last={:?} spot={:?} z={:?} ptb={:?} vs_beat={:?} stable={}",
            //         lane,
            //         market_id,
            //         asset_id,
            //         aligned_ts,
            //         frames_history.len(),
            //         last_stored_aligned_ts_for_asset,
            //         last_stored_aligned_ts_for_asset.map(|t| aligned_ts - t),
            //         currency_spot_usd,
            //         currency_price_z_score,
            //         price_to_beat,
            //         currency_price_vs_beat_pct,
            //         frame.stable,
            //     );
            // }


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
        let batch_frame_by_asset: HashMap<(String, String), XFrame<SIZE>> = built_xframes
            .iter()
            .map(|entry| {
                (
                    (entry.market_id.clone(), entry.asset_id.clone()),
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
                if let Some((five_market_id, fifteen_market_id)) = sibling_state.paired_five_and_fifteen_market_ids()
                {
                    sibling_market_lookup.insert(five_market_id.clone(), fifteen_market_id.clone());
                    sibling_market_lookup.insert(fifteen_market_id, five_market_id);
                }
                sibling_market_lookup
            };

            let xframes_stored_lane = self.xframes_by_market[lane].read().await;

            for entry in &mut built_xframes {
                let mut candidate_asset_ids: HashSet<String> = batch_assets_by_market
                    .get(&entry.market_id)
                    .cloned()
                    .unwrap_or_default();
                if let Some(by_asset) = xframes_stored_lane.get(&entry.market_id) {
                    candidate_asset_ids.extend(by_asset.keys().cloned());
                }

                match find_opposite_asset_id(
                    &entry.asset_id,
                    &currency_up_down_by_asset_id,
                    &candidate_asset_ids,
                ) {
                    Ok(other_asset_id) => match lookup_frame_for_leg_merge(
                        &entry.market_id,
                        &other_asset_id,
                        &batch_frame_by_asset,
                        &xframes_stored_lane,
                    ) {
                        Some(other_frame) => {
                            entry.frame.merge_other_leg_features_from(other_frame)
                        }
                        None => {
                            if entry.frame.stable {
                                eprintln!(
                                    "[diag][builder] other-leg frame missing market={} asset={} other_asset={}",
                                    entry.market_id, entry.asset_id, other_asset_id
                                );
                            }
                        }
                    },
                    Err(err) => {
                        eprintln!("{} find_opposite_asset_id: {err:#}", current_timestamp_ms());
                    }
                }
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
                match find_same_outcome_sibling_asset_id(
                    &entry.asset_id,
                    sibling_market_id.as_str(),
                    &currency_up_down_by_asset_id,
                    &sibling_candidates,
                ) {
                    Ok(sibling_asset_id) => match lookup_frame_for_leg_merge(
                        sibling_market_id.as_str(),
                        &sibling_asset_id,
                        &batch_frame_by_asset,
                        &xframes_stored_lane,
                    ) {
                        Some(sibling_frame) => entry
                            .frame
                            .merge_sibling_market_features_from(sibling_frame),
                        None => {
                            if entry.frame.stable {
                                eprintln!(
                                    "[diag][builder] sibling frame missing market={} sibling_market={} asset={} sibling_asset={}",
                                    entry.market_id,
                                    sibling_market_id,
                                    entry.asset_id,
                                    sibling_asset_id
                                );
                            }
                        }
                    },
                    Err(err) => {
                        if entry.frame.stable {
                            eprintln!(
                                "{} find_same_outcome_sibling_asset_id: {err:#}",
                                current_timestamp_ms()
                            );
                        }
                    }
                }
            }
        }

        let price_to_beat_by_market_snapshot: HashMap<String, Option<f64>> = {
            let guard = self.market_resolution_by_market.read().await;
            let mut map: HashMap<String, Option<f64>> = HashMap::new();
            for entry in &built_xframes {
                map.entry(entry.market_id.clone()).or_insert_with(|| {
                    guard
                        .get(&entry.market_id)
                        .map(|resolution| resolution.price_to_beat)
                });
            }
            map
        };
        let gamma_question_by_market: HashMap<String, Option<String>> = {
            let guard = self.event_data_by_market.read().await;
            let mut map: HashMap<String, Option<String>> = HashMap::new();
            for entry in &built_xframes {
                map.entry(entry.market_id.clone()).or_insert_with(|| {
                    guard
                        .get(&entry.market_id)
                        .and_then(|d| d.gamma_question.clone())
                });
            }
            map
        };

        for entry in built_xframes {
            if entry.frame.stable {
                run_log::xframe_stored(&entry.frame);
            }

            if lane == 0
                && let Some(kind) = XFrameIntervalKind::from_i32(entry.frame.xframe_interval_type)
                && let Some(side) = CurrencyUpDownOutcome::from_i32(entry.frame.currency_up_down_outcome)
                && let Some(state_arc) = self
                    .account
                    .real_sim_state_for_currency(self.currency.as_str())
                    .await
            {
                let channels_arc = state_arc.read().await.lane_frame_channels.channels.clone();
                let channels_guard = channels_arc.read().await;
                if let Some(tx) = channels_guard.get(&(kind, side)) {
                    let price_to_beat = price_to_beat_by_market_snapshot
                        .get(&entry.market_id)
                        .copied()
                        .flatten();
                    let gamma_question = gamma_question_by_market
                        .get(&entry.market_id)
                        .cloned()
                        .flatten();
                    let lane_frame = LaneFrame {
                        price_to_beat,
                        gamma_question,
                        frame: entry.frame.clone(),
                    };
                    let _ = tx.send(lane_frame).await;
                }
            }
            let mut xframes_by_market_lock = self.xframes_by_market[lane].write().await;
            xframes_by_market_lock
                .entry(entry.market_id)
                .or_insert_with(HashMap::new)
                .entry(entry.asset_id)
                .or_insert_with(BTreeMap::new)
                .insert(entry.aligned_ts, entry.frame);
            drop(xframes_by_market_lock);
        }
    }

    pub async fn run_currency_updown_interval(
        self: Arc<Self>,
        period_sec: i64,
        period: &'static str,
    ) {
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

        let mut prev_market: Option<PrevMarket> = None;
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

            let slug = format!(
                "{}-updown-{period}-{window_start_sec}",
                self.currency.to_lowercase()
            );

            let slug_data = if let Some(restored) =
                self.try_restore_currency_event_from_slug_cache(&slug).await
            {
                run_log::gamma_event_data_from_cache(period, &slug);
                restored
            } else if let Some(fetched) = self
                .fetch_currency_event_from_gamma_and_merge(&slug, period)
                .await
            {
                fetched
            } else {
                continue;
            };

            let currency_up_down_by_asset_id = &slug_data.currency_up_down_by_asset_id;
            let gamma_question = &slug_data.gamma_question;
            let market_end_ms = slug_data
                .event_end_ms
                .unwrap_or(ws_end_sec * 1000);
            let market_id = slug_data.market_id.clone();
            let market_start_ms = slug_data.event_start_ms;

            {
                let interval_kind = XFrameIntervalKind::from_period_sec(period_sec);
                let mut xframe_interval_kind_by_asset_id_lock =
                    self.xframe_interval_kind_by_asset_id.write().await;
                for asset_id in currency_up_down_by_asset_id.keys() {
                    xframe_interval_kind_by_asset_id_lock.insert(asset_id.clone(), interval_kind);
                }
            }

            if next_window_start_sec != Some(window_start_sec) {
                next_window_start_sec = Some(window_start_sec);
                let project_manager_cloned = self.clone();
                let currency_lower = self.currency.to_lowercase();
                let prefetch_period_sec = period_sec;
                tokio::spawn(async move {
                    let prefetch_interval_kind =
                        XFrameIntervalKind::from_period_sec(prefetch_period_sec);
                    const PREFETCH_UPCOMING_WINDOW_SLUGS: i64 = 3;
                    for k in 1_i64..=PREFETCH_UPCOMING_WINDOW_SLUGS {
                        let next_window_start_sec =
                            window_start_sec.saturating_add(prefetch_period_sec.saturating_mul(k));
                        let prefetch_slug =
                            format!("{currency_lower}-updown-{period}-{next_window_start_sec}");
                        if project_manager_cloned
                            .slug_currency_event_fully_cached(&prefetch_slug)
                            .await
                        {
                            continue;
                        }
                        if let Some(prefetched) = project_manager_cloned
                            .fetch_currency_event_from_gamma_and_merge(&prefetch_slug, period)
                            .await
                        {
                            run_log::gamma_event_prefetch_fetched(period, &prefetch_slug);

                            {
                                let mut xframe_interval_kind_by_asset_id_lock =
                                    project_manager_cloned
                                        .xframe_interval_kind_by_asset_id
                                        .write()
                                        .await;
                                for asset_id in prefetched.currency_up_down_by_asset_id.keys() {
                                    xframe_interval_kind_by_asset_id_lock
                                        .insert(asset_id.clone(), prefetch_interval_kind);
                                }
                            }
                            let mut asset_ids: Vec<String> = prefetched
                                .currency_up_down_by_asset_id
                                .keys()
                                .cloned()
                                .collect();
                            asset_ids.sort_unstable();
                            match project_manager_cloned
                                .market_ws_tx
                                .send(WsCommand::PrefetchSubscribe { asset_ids })
                                .await
                            {
                                Err(_) => run_log::ws_spawn_err(
                                    period,
                                    &prefetch_slug,
                                    "market ws command channel closed",
                                ),
                                _ => {}
                            }
                        }
                    }
                });
            }

            let mut ids: Vec<String> = currency_up_down_by_asset_id.keys().cloned().collect();
            ids.sort_unstable();

            let project_manager_cloned = self.clone();
            let currency = self.currency.clone();

            // Быстрый PTB в кэш (RTDS по start_ms или Vatic API target/timestamp); exact подтянет фон.
            let inline_ptb_opt: Option<f64> = match market_start_ms {
                Some(start_ms) => {
                    let rtds_currency_prices_by_ms_lock = project_manager_cloned
                        .rtds_currency_prices_by_ms
                        .read()
                        .await;
                    if let Some(&price) = rtds_currency_prices_by_ms_lock.get(&start_ms) {
                        run_log::price_to_beat_from_rtds(
                            period,
                            &slug,
                            market_id.as_deref(),
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
            let inline_ptb_opt = if let Some(price) = inline_ptb_opt {
                Some(price)
            } else {
                match fetch_price_to_beat_from_vatic_api(
                    self.http.as_ref(),
                    &slug,
                    currency.as_str(),
                )
                .await
                {
                    Ok(price) => {
                        run_log::price_to_beat_from_event_page(period, &slug, price);
                        Some(price)
                    }
                    Err(err) => {
                        run_log::gamma_fetch_err(period, &slug, &err);
                        None
                    }
                }
            };
            if let (Some(ptb), Some(mid)) = (inline_ptb_opt, market_id.as_deref()) {
                self.merge_market_price_to_beat(ptb, mid).await;
            }

            // Фон: exact PTB → кэш + oneshot для следующей итерации; дамп prev по паре exact.
            let (current_exact_tx, current_exact_rx) = oneshot::channel::<f64>();
            spawn_bg_price_to_beat_refine(
                self.clone(),
                slug.clone(),
                market_id.clone(),
                currency.clone(),
                period,
                prev_market.take(),
                period_sec,
                window_start_sec,
                current_exact_tx,
            );
            let price_to_beat = inline_ptb_opt;

            {
                let remain_ms = (market_end_ms - current_timestamp_ms()).max(0) as u64;

                run_log::ws_start(
                    period,
                    &slug,
                    price_to_beat,
                    market_id.as_deref(),
                    &ids,
                    remain_ms,
                    market_end_ms,
                );

                let cmd = WsCommand::ActivateWindow(MarketWsSubscription {
                    period,
                    slug: slug.clone(),
                    asset_ids: ids.clone(),
                    market_id: market_id.clone(),
                    period_sec,
                    window_start_sec,
                    gamma_question: gamma_question.clone(),
                });
                if self.market_ws_tx.send(cmd).await.is_err() {
                    run_log::ws_spawn_err(period, &slug, "market ws command channel closed");
                    continue;
                }

                prev_market = Some(PrevMarket {
                    market_id: market_id.clone(),
                    gamma_question: gamma_question.clone(),
                    window_start_sec,
                    exact_price_to_beat_rx: current_exact_rx,
                    event_end_ms: slug_data.event_end_ms,
                });
            }

            let sleep_until_ms = ws_end_sec * 1000;
            let now_ms = current_timestamp_ms();
            if now_ms < sleep_until_ms {
                tokio::time::sleep(Duration::from_millis((sleep_until_ms - now_ms) as u64)).await;
            }

            run_log::ws_window_end_wait(period, &slug, ids.len());
            if !ids.is_empty() {
                match self
                    .market_ws_tx
                    .send(WsCommand::PruneStaleIds {
                        asset_ids: ids.clone(),
                    })
                    .await
                {
                    Ok(()) => {}
                    Err(_) => {
                        run_log::ws_spawn_err(period, &slug, "market ws command channel closed")
                    }
                }
            }
        }
    }
}

/// Exact PTB через Vatic API (retry): обновляет кэш, шлёт в oneshot, дампит prev при непрерывности окон.
fn spawn_bg_price_to_beat_refine(
    project_manager: Arc<ProjectManager>,
    slug: String,
    market_id: Option<String>,
    currency: Arc<String>,
    period: &'static str,
    prev_market: Option<PrevMarket>,
    period_sec: i64,
    current_window_start_sec: i64,
    current_exact_tx: oneshot::Sender<f64>,
) {
    const MAX_REFINE_ATTEMPTS: u32 = 30;
    const REFINE_RETRY_DELAY: Duration = Duration::from_secs(5);

    tokio::spawn(async move {
        let current_exact = match retry_fetch_exact_price_to_beat(
            project_manager.http.as_ref(),
            &slug,
            currency.as_str(),
            MAX_REFINE_ATTEMPTS,
            REFINE_RETRY_DELAY,
        )
        .await
        {
            Some(exact_price) => {
                run_log::price_to_beat_from_event_page(period, &slug, exact_price);
                if let Some(market_id) = market_id.as_deref() {
                    project_manager
                        .merge_market_price_to_beat(exact_price, market_id)
                        .await;
                }
                let _ = current_exact_tx.send(exact_price);
                exact_price
            }
            None => {
                eprintln!(
                    "xframe_dump: slug={slug}: refine не получил exact priceToBeat за {MAX_REFINE_ATTEMPTS} попыток, дамп prev пропущен"
                );
                if let Some(prev) = prev_market.as_ref() {
                    if let Some(prev_market_id) = prev.market_id.as_ref() {
                        project_manager
                            .cleanup_stale_market_data(prev_market_id)
                            .await;
                    }
                }
                return;
            }
        };

        let Some(prev) = prev_market else {
            return;
        };
        let Some(prev_market_id) = prev.market_id.clone() else {
            return;
        };

        // `current_exact` для окна N = `final_price` окна N-1 (спот в момент
        // открытия нового окна = спот на закрытие предыдущего). Пишем в
        // `market_resolution_by_market` ДО guard'ов ниже (window-continuity /
        // exact_price_to_beat_rx), чтобы even при early-return из-за разрыва
        // окон prev-market получил `final_price` — submit-резолюция читает
        // именно её.
        project_manager
            .merge_market_final_price(current_exact, &prev_market_id)
            .await;

        let expected_current_window_start_sec = prev.window_start_sec.saturating_add(period_sec);
        if current_window_start_sec != expected_current_window_start_sec {
            eprintln!(
                "xframe_dump: market_id={prev_market_id}: разрыв непрерывности окон \
                 (prev.window_start_sec={prev_ws} + period_sec={period_sec} = {expected}, \
                 current.window_start_sec={current_ws}), дамп пропущен",
                prev_ws = prev.window_start_sec,
                expected = expected_current_window_start_sec,
                current_ws = current_window_start_sec,
            );
            project_manager
                .cleanup_stale_market_data(&prev_market_id)
                .await;
            return;
        }

        let prev_exact = match prev.exact_price_to_beat_rx.await {
            Ok(price) => price,
            Err(_) => {
                eprintln!(
                    "xframe_dump: market_id={prev_market_id}: refine prev-итерации не вернул exact priceToBeat (sender дропнут), дамп пропущен"
                );
                project_manager
                    .cleanup_stale_market_data(&prev_market_id)
                    .await;
                return;
            }
        };

        let prev_slug = format!(
            "{}-updown-{period}-{}",
            currency.to_lowercase(),
            prev.window_start_sec
        );
        let prev_event_end_ms = match prev.event_end_ms {
            Some(ms) => ms,
            None => {
                let fallback = prev
                    .window_start_sec
                    .saturating_add(period_sec)
                    .saturating_mul(1000);
                eprintln!(
                    "xframe_dump: market_id={prev_market_id}: у Gamma не было \
                     event_end_ms (endDate), использую fallback \
                     (window_start_sec + period_sec) * 1000 = {fallback}; \
                     имя файла дампа может разойтись с синтетическим CSV-путём \
                     и партиал-HTML по этому рынку"
                );
                fallback
            }
        };
        xframe_dump::spawn_dump_market_xframes_binary(
            project_manager.clone(),
            prev_market_id,
            prev.gamma_question,
            period_sec,
            prev_exact,
            current_exact,
            prev_slug,
            prev_event_end_ms,
        );
    });
}

/// Vatic API `targets/timestamp` с повторами; `None` после исчерпания попыток.
async fn retry_fetch_exact_price_to_beat(
    http: &reqwest::Client,
    slug: &str,
    currency: &str,
    max_attempts: u32,
    retry_delay: Duration,
) -> Option<f64> {
    for attempt in 1..=max_attempts {
        match fetch_price_to_beat_from_vatic_api(http, slug, currency).await {
            Ok(price) => return Some(price),
            Err(_) => {
                if attempt < max_attempts {
                    tokio::time::sleep(retry_delay).await;
                }
            }
        }
    }
    None
}

/// Кадр другой ноги: батч текущего тика → последний известный из хранилища.
/// После Scenario A в `batch` гарантированно ≤ 1 запись на `(market, asset)`,
/// а в `stored` берём самый свежий кадр без ограничения по `aligned_ts` —
/// это и есть «пустышка с прошлых итераций», если asset молчал > 1 сек.
fn lookup_frame_for_leg_merge<'a>(
    market_id: &str,
    asset_id: &str,
    batch: &'a HashMap<(String, String), XFrame<SIZE>>,
    stored: &'a MarketFrames,
) -> Option<&'a XFrame<SIZE>> {
    if let Some(frame) = batch.get(&(market_id.to_string(), asset_id.to_string())) {
        return Some(frame);
    }
    stored
        .get(market_id)?
        .get(asset_id)?
        .values()
        .next_back()
}

/// `(beat - spot) / beat * 100`; положительно, если spot ниже beat.
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

/// Округление `timestamp_ms` вниз к границе интервала (мс).
fn align_timestamp_ms_to_interval(timestamp_ms: i64, interval_secs: u64) -> i64 {
    let bucket_ms = (interval_secs as i64).saturating_mul(1000);
    if bucket_ms <= 0 {
        return timestamp_ms;
    }
    timestamp_ms.div_euclid(bucket_ms).saturating_mul(bucket_ms)
}
