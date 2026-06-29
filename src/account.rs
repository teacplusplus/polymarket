//! Капитал и MtM (`bankroll`, peak, max DD); per-lane позиции и CLOB-клиенты.
//! Один [`SharedAccount`] на процесс: поля под отдельными `RwLock`, auth в [`ArcSwapAny`] (read-mostly).
//! CLOB L2 authenticate и heartbeat — [`crate::authenticate`].
//! Порядок локов: `bankroll` → `peak_bankroll` → `max_drawdown_pct` → `last_prob` → `positions` → `pending_close_positions` → `future_positions` → `closing` → `recently_resolved_markets` → один inner на позицию.

use crate::account_order_completion::TrackerEntry;
use crate::account_proxy::PolyProxyEnvGuard;
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{CloseReason, INITIAL_BANKROLL, LanePositions, SharedOpenPosition};
use crate::real_sim::RealSimState;
use crate::sim_stats::SimStats;
use alloy::signers::local::PrivateKeySigner;
use arc_swap::ArcSwapAny;
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::data;
use polymarket_client_sdk::gamma;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Production CLOB V2 host (cutover 2026-04-28; см.
/// [docs.polymarket.com/v2-migration](https://docs.polymarket.com/v2-migration)).
///
/// Старый `clob-v2.polymarket.com` остаётся как pre-cutover testing host и
/// перестал маршрутизировать `POST /order` — edge отдаёт `405` с пустым телом
/// (auth/GET-ручки ещё отвечают, поэтому ошибка ловится только на отправке
/// ордера). Дефолт SDK (`clob::Client::default()`) указывает на тот старый
/// хост — поэтому мы конструируем клиент явно с production-URL.
pub const POLYMARKET_CLOB_HOST: &str = "https://clob.polymarket.com";

/// `Arc<Account>`; синхронизация только на полях [`Account`].
pub type SharedAccount = Arc<Account>;

/// Ключ лейна: `(currency, interval, side)` — маршрут в maps и `last_prob`.
pub type LaneKey = (String, XFrameIntervalKind, CurrencyUpDownOutcome);

/// Ключ redeem-01 tail режима: `(coin, period)`.
pub type Redeem01TailMarketRegimeKey = (String, XFrameIntervalKind);

/// Счёт процесса: cash, MtM, позиции, CLOB. См. модульный комментарий про порядок локов и один inner на позицию.
#[derive(Debug)]
pub struct Account {
    /// Реализованный USDC (cash).
    pub bankroll: Arc<RwLock<f64>>,
    /// Пик equity (MtM).
    pub peak_bankroll: Arc<RwLock<f64>>,
    /// Max drawdown, % от пика.
    pub max_drawdown_pct: Arc<RwLock<f64>>,
    /// Последняя implied prob по лейну (MtM, по `currency` в ключе).
    pub last_prob: Arc<RwLock<HashMap<LaneKey, f64>>>,
    /// Открытые позиции; тот же `Arc`, что в записи закрытия (`position`).
    /// Для одной лейны могут сосуществовать позиции разных `market_id` — старые
    /// (с чужим `asset_id` относительно текущего кадра) живут здесь как
    /// «припаркованные» до резолюции; см. [`Self::resolve_pending_market_sync`].
    pub positions: Arc<RwLock<HashMap<LaneKey, LanePositions>>>,
    /// Submit/Mock-only: позиции, на которых уже сработал
    /// [`crate::history_sim::sell_gate`] и спавнен
    /// [`crate::account_submit::spawn_sell_taker`], но
    /// [`crate::account_close_position::close_position_after_submit`] ещё не
    /// финализировал PnL (BUY/maker-TP/taker-FAK invoke'ы в полёте).
    pub pending_close_positions: Arc<RwLock<HashMap<LaneKey, LanePositions>>>,
    /// Позиции рынков, которые ещё не наступили: создаются
    /// [`crate::history_sim::future_open_positions`] (BUY-ордер выставляется сразу),
    /// но до первого WS-снимка их `asset_id` они «припаркованы» здесь и НЕ
    /// участвуют ни в `tick_once` MtM, ни в гейтах (`MAX_OPEN_POSITIONS` /
    /// same-asset), ни в [`Self::resolve_pending_market_sync`]. Когда рынок
    /// стартует по времени (`now >= event_end_ms − interval_ms`) переносятся в
    /// [`Self::positions`] через [`Self::promote_started_future_positions`].
    pub future_positions: Arc<RwLock<HashMap<LaneKey, LanePositions>>>,
    /// HTTP с rustls; тот же `Arc`, что у [`ProjectManager::http`](crate::project_manager::ProjectManager::http).
    pub http: Arc<reqwest::Client>,
    /// Общий unauth CLOB SDK-клиент (клоны в PM и др.).
    pub clob: Arc<clob::Client>,
    /// Gamma API SDK-клиент; тот же `Arc`, что у [`ProjectManager::gamma`](crate::project_manager::ProjectManager::gamma).
    pub gamma: Arc<gamma::Client>,
    /// Polymarket Data API (`/positions` и др.).
    pub data: Arc<data::Client>,
    /// Authed-сессия: heartbeat и ордеры ([`crate::account_order`]).
    pub clob_authed: ArcSwapAny<Arc<Option<clob::Client<Authenticated<Normal>>>>>,
    /// EOA-подписант под ордеры; задаётся вместе с `clob_authed`.
    pub clob_signer: ArcSwapAny<Arc<Option<PrivateKeySigner>>>,
    /// Трекер единоразовых колбэков POST /order (WS + REST fallback).
    pub order_invoke_hub: Arc<RwLock<HashMap<String, TrackerEntry>>>,
    /// `currency` → [`RealSimState`]; лок отдельно от цепочки `bankroll → …`.
    pub real_sim_state_by_currency: Arc<RwLock<HashMap<String, Arc<RwLock<RealSimState>>>>>,
}

impl Account {
    pub fn new() -> Self {
        // Если в `.env` заданы `POLY_PROXY_IP` + `POLY_PROXY_PORT` (и опционально пароль / логин), временно
        // выставляем `HTTP_PROXY`/`HTTPS_PROXY` для сборки клиентов (reqwest подхватывает их по умолчанию).
        let poly_proxy_env = PolyProxyEnvGuard::install_from_env();

        // Production CLOB V2 host (см. [`POLYMARKET_CLOB_HOST`]): `clob.polymarket.com`.
        // `clob::Client::default()` указывает на pre-cutover testing host
        // `clob-v2.polymarket.com`, где `POST /order` уже не работает.
        let clob = Arc::new(
            clob::Client::new(POLYMARKET_CLOB_HOST, clob::Config::default())
                .expect("CLOB client with production host should construct"),
        );
        let http = Arc::new(
            reqwest::Client::builder()
                .use_rustls_tls()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        );
        let gamma = Arc::new(gamma::Client::default());
        let data = Arc::new(data::Client::default());
        PolyProxyEnvGuard::uninstall_from_env(poly_proxy_env);

        Self {
            bankroll: Arc::new(RwLock::new(INITIAL_BANKROLL)),
            peak_bankroll: Arc::new(RwLock::new(INITIAL_BANKROLL)),
            max_drawdown_pct: Arc::new(RwLock::new(0.0)),
            last_prob: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            pending_close_positions: Arc::new(RwLock::new(HashMap::new())),
            future_positions: Arc::new(RwLock::new(HashMap::new())),
            http,
            clob,
            gamma,
            data,
            clob_authed: ArcSwapAny::new(Arc::new(None)),
            clob_signer: ArcSwapAny::new(Arc::new(None)),
            order_invoke_hub: Arc::new(RwLock::new(HashMap::new())),
            real_sim_state_by_currency: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Короткий read → clone Arc state; без state для валюты — [`None`].
    pub async fn real_sim_state_for_currency(
        &self,
        currency: &str,
    ) -> Option<Arc<RwLock<RealSimState>>> {
        self.real_sim_state_by_currency
            .read()
            .await
            .get(currency)
            .cloned()
    }

    /// Пустые buckets лейнов валюты ([`crate::real_sim::run_real_sim`]); `or_default` идемпотентен.
    /// Параллельно создаём bucket в [`Self::pending_close_positions`] под тем
    /// же [`LaneKey`], чтобы `tick_once` мог брать `.read()` без `or_default`.
    pub async fn register_currency_lanes(
        &self,
        currency: &str,
        lanes: &[(XFrameIntervalKind, CurrencyUpDownOutcome)],
    ) {
        let mut positions = self.positions.write().await;
        let mut pending = self.pending_close_positions.write().await;
        let mut future = self.future_positions.write().await;
        for (interval, side) in lanes {
            let key = (currency.to_string(), *interval, *side);
            positions.entry(key.clone()).or_default();
            pending.entry(key.clone()).or_default();
            future.entry(key).or_default();
        }
    }

    /// Переносит из [`Self::future_positions`] в [`Self::positions`] все позиции,
    /// чей рынок уже **стартанул** по времени: `now_ms >= market_start`, где
    /// `market_start = event_end_ms − interval_ms`. Если у позиции нет
    /// `event_end_ms` (старт неизвестен) — переносим сразу, чтобы она не застряла
    /// в `future_positions` навсегда. После переноса позиция становится обычной
    /// (MtM/гейты/резолюция). Вызывается периодически
    /// ([`crate::project_manager::ProjectManager::update_last_snapshot`]).
    ///
    /// Два фазы, чтобы НЕ брать write-лок `positions`, если переносить нечего:
    /// сначала под локом `future_positions` дренируем стартанувшие позиции в
    /// локальный буфер; `positions` лочим только если буфер не пуст. Так оба лока
    /// никогда не держатся одновременно (модульная цепочка `positions → … →
    /// future_positions` не нарушается). Идемпотентно: нет стартанувших — no-op без
    /// касания `positions`.
    pub async fn promote_started_future_positions(&self, now_ms: i64) {
        let mut started: Vec<(LaneKey, String, SharedOpenPosition)> = Vec::new();
        {
            let mut future = self.future_positions.write().await;
            for (lane_key, lane_positions) in future.iter_mut() {
                let pos_ids: Vec<String> = lane_positions.keys().cloned().collect();
                for pos_id in pos_ids {
                    let is_started = match lane_positions.get(&pos_id) {
                        Some(pos_arc) => {
                            let pos = pos_arc.read().await;
                            match pos.event_end_ms {
                                Some(end_ms) => {
                                    let interval_ms = XFrameIntervalKind::from_i32(
                                        pos.xframe_interval_type_at_open,
                                    )
                                    .map(|kind| kind.interval_ms())
                                    .unwrap_or(0);
                                    now_ms >= end_ms.saturating_sub(interval_ms)
                                }
                                None => true,
                            }
                        }
                        None => false,
                    };
                    if is_started && let Some(pos_arc) = lane_positions.shift_remove(&pos_id) {
                        started.push((lane_key.clone(), pos_id, pos_arc));
                    }
                }
            }
        }
        if started.is_empty() {
            return;
        }
        let mut positions = self.positions.write().await;
        for (lane_key, pos_id, pos_arc) in started {
            positions
                .entry(lane_key)
                .or_default()
                .insert(pos_id, pos_arc);
        }
    }

    /// Ядро: дренирует `positions[lane]` по `market_id` (включая «припаркованные» с чужим
    /// `asset_id` текущего тика), считает бинарный payout, обновляет `bankroll`/`SimStats`,
    /// пишет CSV через [`crate::trade_csv_log::write_trade_csv_row`].
    ///
    /// **Drain stale-маркетов:** позиции той же лейны (`currency`+`interval`), но с
    /// **другим** `market_id` (хвост от прошлого маркета, для которого resolution-событие
    /// не пришло), считаются полностью неактуальными — списываются как полная потеря
    /// (`pnl = -position_size`), `bankroll` корректируется, в `SimStats` они **не** попадают
    /// (это не легитимный trade outcome, а защита от утечки локнутого капитала), CSV-строка
    /// тоже не пишется. На каждый такой инцидент — `tee_eprintln!`-предупреждение.
    ///
    /// `final_price`: `None` → `pos.final_price`, иначе override для realtime.
    pub async fn resolve_pending_market_sync(
        account: &SharedAccount,
        sim_stats: &mut SimStats,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
        final_price: Option<f64>,
    ) {
        // bankroll → positions → recently_resolved

        let mut positions = account.positions.write().await;
        let mut to_close: Vec<(SharedOpenPosition, bool, CurrencyUpDownOutcome)> = Vec::new();

        for ((cur, int_kind, side), lane_positions) in positions.iter_mut() {
            if cur.as_str() != currency || *int_kind != interval {
                continue;
            }
            let token_won = match side {
                CurrencyUpDownOutcome::Up => up_won,
                CurrencyUpDownOutcome::Down => !up_won,
            };

            let pos_ids: Vec<String> = lane_positions.keys().cloned().collect();
            for pos_id in pos_ids {
                let Some(pos_arc) = lane_positions.get(&pos_id) else {
                    continue;
                };
                let matches_market = pos_arc.read().await.market_id == market_id;
                if !matches_market {
                    continue;
                }

                let Some(pos_arc) = lane_positions.shift_remove(&pos_id) else {
                    continue;
                };
                to_close.push((pos_arc, token_won, *side));
            }
        }
        drop(positions);

        let mut redeem_x_groups: HashMap<String, Vec<(SharedOpenPosition, bool)>> = HashMap::new();
        let mut regular_close: Vec<(SharedOpenPosition, bool, CurrencyUpDownOutcome)> = Vec::new();
        for (pos_arc, token_won, side) in to_close {
            let pos = pos_arc.read().await;
            if pos.redeem_x {
                let market_id = pos.market_id.clone();
                drop(pos);
                redeem_x_groups
                    .entry(market_id)
                    .or_default()
                    .push((pos_arc, token_won));
            } else {
                drop(pos);
                regular_close.push((pos_arc, token_won, side));
            }
        }

        for (pos_arc, token_won, side) in regular_close {
            let side_stats = match side {
                CurrencyUpDownOutcome::Up => &mut sim_stats.up,
                CurrencyUpDownOutcome::Down => &mut sim_stats.down,
            };
            // Записываем свежий `final_price` в саму позицию ДО `close_position`:
            // CSV-логгер внутри читает `pos.final_price` (override-параметра
            // больше нет). При None — оставляем то, что уже стояло (могло быть
            // выставлено в `open_position`).
            if let Some(fp) = final_price {
                pos_arc.write().await.final_price = Some(fp);
            }
            let reason = if token_won {
                CloseReason::ResolutionWin
            } else {
                CloseReason::ResolutionLoss
            };
            // `gross_usdc=None` → close_position сам выведет `shares_held` /
            // `0.0` из `reason` (см. дефолт внутри); `pos_arc`/`account`
            // разворачиваются под коротким локом ВНУТРИ функции — здесь не
            // держим ни pos.read, ни bankroll.write.
            crate::account_close_position::close_position(
                account, &pos_arc, side_stats, &reason, None, 0,
            )
            .await;
        }

        for group in redeem_x_groups.into_values() {
            crate::account_close_position::close_position_redeem(
                account,
                sim_stats,
                group,
                up_won,
                final_price,
            )
            .await;
        }
    }

    /// `Arc::new(Account::new())` для `main` и PM.
    pub fn new_shared() -> SharedAccount {
        Arc::new(Self::new())
    }

    /// Обновляет пик и max DD от переданной MtM equity (`peak_bankroll` → `max_drawdown_pct`).
    pub async fn update_drawdown(&self, equity: f64) {
        let mut peak = self.peak_bankroll.write().await;
        let mut max_dd = self.max_drawdown_pct.write().await;
        if equity > *peak {
            *peak = equity;
        }
        if *peak > 0.0 {
            let drawdown_pct = (*peak - equity) / *peak * 100.0;
            if drawdown_pct > *max_dd {
                *max_dd = drawdown_pct;
            }
        }
    }
}

impl Default for Account {
    fn default() -> Self {
        Self::new()
    }
}

pub use crate::authenticate::{
    POLY_PRIVATE_KEY_ENV, spawn_heartbeat, try_authenticate_clob_for_heartbeats,
};

// Ордерный API — [`crate::account_order`] (здесь только `clob_authed` / `clob_signer`).
