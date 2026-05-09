//! Live `real_sim`: торговая логика как в [`crate::history_sim`], кадры с [`ProjectManager`] (фанаут 1s → 4 × [`tick_once`]).
//!
//! Воркеры без таймера (`recv().await`). Общие [`RealSimState`] и портфельный [`Account`]. Стаканы — батч [`run_book_coordinator`].
//! WS расходится с HTTP (> ~`2×tick_size` по L1–L3) → без новых входов, закрытия как обычно. После buy/sell — [`print_sim_stats`].

use crate::account::SharedAccount;
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{
    BuyGate, any_position_would_sell, buy_gate, compute_p_win_now, compute_pnl_inference,
    load_booster, manage_positions, print_sim_stats, try_open_position, OpenPosition,
    SimStats, StrictBook, HOLD_TO_END_THRESHOLD_SEC,
};
/// Тот же cap, что в [`crate::history_sim::manage_positions`] / `book_fill_*` (на TP при выполненном пороге VWAP — cap может игнорироваться).
pub use crate::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT;
use crate::market_snapshot::MarketSnapshot;
use crate::xframe::BookLevel;
use crate::project_manager::{LaneFrame, ProjectManager};
use crate::train_mode::{load_calibration, Calibration};
use crate::util::current_timestamp_ms;
use crate::xframe::{XFrame, SIZE};

use anyhow::{anyhow, Result};
use futures_util::FutureExt;
use indexmap::IndexSet;
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::response::OrderBookSummaryResponse;
use polymarket_client_sdk::types::U256;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::time::MissedTickBehavior;
use xgb::Booster;

/// Очередь `LaneFrame` на `(interval, side)` (фанаут lane 0).
const LANE_FRAME_CHANNEL_CAP: usize = 64;

/// Очередь [`BookRequest`] в [`run_book_coordinator`].
const BOOK_REQUEST_CHANNEL_CAP: usize = 64;

/// Пауза без новых запросов → собираем батч ([`run_book_coordinator`]).
const BOOK_BATCH_IDLE_MS: u64 = 5;

/// Абсолютный потолок ожидания добора батча от первого запроса.
const BOOK_BATCH_MAX_MS: u64 = 50;

/// HTTP `order_books`; при таймауте — `None` всем ожидающим (WS-fallback).
const BOOK_HTTP_TIMEOUT_MS: u64 = 2000;

/// Максимальный возраст последнего WS-снимка по `asset_id`
/// ([`ProjectManager::last_snapshot_by_asset_id`]), при котором
/// [`tick_once`] собирает [`StrictBook`] прямо из него и пропускает HTTP
/// `order_books`. WS-канал шлёт `book` на subscribe и далее непрерывный
/// поток `price_change`/`last_trade_price` — за 1с обычно есть свежий
/// мердж, и HTTP-roundtrip в `BOOK_BATCH_*` (~5–50ms idle + 2s timeout)
/// можно избежать. Если за этот порог снимок не приходит — считаем поток
/// «остановившимся» и идём в HTTP как обычно.
pub(crate) const WS_STRICT_BOOK_MAX_AGE_MS: i64 = 1_000;

/// Ожидание ответа координатора в [`fetch_http_strict_book`] (`≈ 3×` HTTP).
const BOOK_REPLY_TIMEOUT_MS: u64 = BOOK_HTTP_TIMEOUT_MS * 3;

/// FIFO-кольцо [`RealSimState::seen_market_ids`] на интервал (`shift_remove_index(0)` сверх лимита).
const SEEN_MARKET_IDS_CAP: usize = 8;

/// Период локального snapshot'а (`Account` + per-interval [`SimStats`])
/// в [`spawn_stats_snapshot`]. `print_sim_stats` сам по себе дёргается
/// только при сделке (см. [`tick_once`]), а live-режим может час и
/// больше идти без сделок: без этого таска в логе тишина — нет ни
/// bankroll, ни max_dd, ни числа обработанных кадров. 5 минут —
/// компромисс между шумом в `tee` и видимостью «жив ли пайплайн».
const STATS_HEARTBEAT_INTERVAL_SEC: u64 = 5 * 60;

/// Полный набор 4 ключей фанаута 1s-кадров: `(interval, side)`.
const LANE_FRAME_ROUTES: [(XFrameIntervalKind, CurrencyUpDownOutcome); 4] = [
    (XFrameIntervalKind::FifteenMin, CurrencyUpDownOutcome::Down),
    (XFrameIntervalKind::FifteenMin, CurrencyUpDownOutcome::Up),
    (XFrameIntervalKind::FiveMin, CurrencyUpDownOutcome::Down),
    (XFrameIntervalKind::FiveMin, CurrencyUpDownOutcome::Up),
];

/// Таблица `Sender` для фанаута lane 0; реальный `rx` у воркера, в карте — dummy пара для типа.
pub struct LaneFrameChannels {
    pub channels: Arc<RwLock<HashMap<(XFrameIntervalKind, CurrencyUpDownOutcome), mpsc::Sender<LaneFrame>>>>,
}

impl LaneFrameChannels {
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// [`SimStats`] по интервалам, каналы кадров, dedupe `events` через `seen_market_ids`.
pub struct RealSimState {
    /// Агрегированная статистика по интервалам (per-interval счётчики).
    /// Карта инициализируется оба ключа сразу, воркеры делают
    /// `get_mut(&kind).unwrap()`.
    pub stats: HashMap<XFrameIntervalKind, SimStats>,
    pub lane_frame_channels: LaneFrameChannels,
    /// Новый `market_id` в интервале → один bump `stats[].events` ([`IndexSet`] + cap [`SEEN_MARKET_IDS_CAP`]).
    pub seen_market_ids: HashMap<XFrameIntervalKind, IndexSet<String>>,
}

impl RealSimState {
    pub fn new() -> Self {
        let mut stats = HashMap::with_capacity(2);
        stats.insert(XFrameIntervalKind::FiveMin, SimStats::new());
        stats.insert(XFrameIntervalKind::FifteenMin, SimStats::new());
        let mut seen_market_ids = HashMap::with_capacity(2);
        seen_market_ids.insert(XFrameIntervalKind::FiveMin, IndexSet::new());
        seen_market_ids.insert(XFrameIntervalKind::FifteenMin, IndexSet::new());
        Self {
            stats,
            lane_frame_channels: LaneFrameChannels::new(),
            seen_market_ids,
        }
    }
}

/// Модели одной стороны одного интервала (PnL обязательная, Resolution — опциональная).
struct SideModels {
    booster_pnl: Arc<Booster>,
    calibration_pnl: Option<Calibration>,
    booster_resolution: Option<Arc<Booster>>,
    calibration_resolution: Option<Calibration>,
}

pub(crate) fn interval_label(kind: XFrameIntervalKind) -> &'static str {
    match kind {
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    }
}

/// `https://polymarket.com/event/...` из [`LaneFrame::event_start_ms`] (Gamma); пустая строка если `None`.
fn polymarket_event_url_from_frame(
    currency: &str,
    interval_kind: XFrameIntervalKind,
    event_start_ms: Option<i64>,
) -> String {
    let Some(start_ms) = event_start_ms else {
        return String::new();
    };
    let window_start_sec = start_ms / 1_000;
    format!(
        "https://polymarket.com/event/{currency}-updown-{interval}-{window_start_sec}",
        currency = currency.to_lowercase(),
        interval = interval_label(interval_kind),
    )
}

pub(crate) fn side_label(side: CurrencyUpDownOutcome) -> &'static str {
    match side {
        CurrencyUpDownOutcome::Up => "up",
        CurrencyUpDownOutcome::Down => "down",
    }
}

/// Загрузка моделей из `xframes/{currency}/{version}/`, 4 воркера → [`tick_once`].
pub async fn run_real_sim(project_manager: Arc<ProjectManager>) -> Result<()> {
    let currency_arc = project_manager.currency.clone();
    let currency = currency_arc.as_str().to_string();
    let version_path = latest_version_path(&currency)
        .ok_or_else(|| anyhow!(
            "нет ни одной версии в xframes/{currency}/ — сначала соберите данные (STATUS=default) и обучите модели (STATUS=train)"
        ))?;
    let version = dir_name(&version_path);
    let tag_prefix = format!("{currency}/{version}");

    crate::tee_println!(
        "[real_sim] версия моделей: {tag_prefix} (из {})",
        version_path.display(),
    );

    let state = project_manager.real_sim_state.clone();
    let account = project_manager.account.clone();
    let last_snapshot_by_asset_id = project_manager.last_snapshot_by_asset_id.clone();
    let channels = state.read().await.lane_frame_channels.channels.clone();

    account
        .register_currency_lanes(&currency, &LANE_FRAME_ROUTES)
        .await;

    let (book_tx, book_rx) = mpsc::channel::<BookRequest>(BOOK_REQUEST_CHANNEL_CAP);
    {
        let project_manager = project_manager.clone();
        tokio::spawn(async move {
            run_book_coordinator(project_manager, book_rx).await;
        });
    }

    spawn_stats_snapshot(state.clone(), account.clone(), tag_prefix.clone());

    for (interval_kind, side) in LANE_FRAME_ROUTES {
        let label = interval_label(interval_kind);
        let side_lbl = side_label(side);
        let models = load_side_models(&version_path, label, side_lbl).ok_or_else(|| {
            anyhow!("не удалось загрузить pnl-модель {label}/{side_lbl}")
        })?;
        crate::tee_println!(
            "[real_sim] {tag_prefix}/{label}/{side_lbl}: pnl ✓  resolution={}",
            if models.booster_resolution.is_some() { "✓" } else { "✗" },
        );

        let (tx, rx) = mpsc::channel::<LaneFrame>(LANE_FRAME_CHANNEL_CAP);
        channels
            .write()
            .await
            .insert((interval_kind, side), tx);

        spawn_side_worker(
            book_tx.clone(),
            state.clone(),
            account.clone(),
            currency_arc.clone(),
            last_snapshot_by_asset_id.clone(),
            interval_kind,
            side,
            models,
            tag_prefix.clone(),
            rx,
        );
    }

    Ok(())
}

fn spawn_side_worker(
    book_tx: mpsc::Sender<BookRequest>,
    state: Arc<RwLock<RealSimState>>,
    account: SharedAccount,
    currency: Arc<String>,
    last_snapshot_by_asset_id: Arc<RwLock<HashMap<String, MarketSnapshot>>>,
    interval_kind: XFrameIntervalKind,
    side: CurrencyUpDownOutcome,
    models: SideModels,
    tag_prefix: String,
    mut rx: mpsc::Receiver<LaneFrame>,
) {
    tokio::spawn(async move {
        let tag = format!(
            "{tag_prefix}/{}/{}",
            interval_label(interval_kind),
            side_label(side),
        );
        let mut last_market_id: Option<String> = None;
        while let Some(lane_frame) = rx.recv().await {
            let result = AssertUnwindSafe(tick_once(
                &book_tx,
                &state,
                &account,
                currency.as_str(),
                &last_snapshot_by_asset_id,
                interval_kind,
                side,
                &models,
                &tag,
                &mut last_market_id,
                lane_frame,
            ))
            .catch_unwind()
            .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    crate::tee_eprintln!("[real_sim] {tag}: tick error: {err:#}");
                }
                Err(payload) => {
                    let msg = panic_payload_message(&payload);
                    crate::tee_eprintln!(
                        "[real_sim] {tag}: tick PANIC ({msg}) — кадр пропущен, воркер живой"
                    );
                }
            }
        }
        crate::tee_eprintln!("[real_sim] {tag}: канал закрыт — воркер завершён");
    });
}

/// Per-currency snapshot пайплайна: раз в [`STATS_HEARTBEAT_INTERVAL_SEC`]
/// печатает [`print_sim_stats`] по обоим интервалам
/// ([`XFrameIntervalKind::FiveMin`] / [`XFrameIntervalKind::FifteenMin`])
/// + банкролл/просадку текущего [`Account`]. Без этого таска live-режим
/// без сделок выглядит как «висящий процесс» — нет ни bankroll, ни
/// max_dd, ни сколько кадров обработано.
///
/// Привязан к `state` (`RealSimState` per-currency) и `tag_prefix`
/// (`"{currency}/{version}"`), поэтому живёт здесь, а не в [`crate::account`]
/// (где сидит глобальный CLOB heartbeat без привязки к валюте).
/// Спавнится один раз на валюту в [`run_real_sim`].
///
/// Локи `state`/`account` берутся read-only, чтобы не конкурировать с
/// торговыми воркерами; печать идёт через `tee_log` (свой mutex). Если
/// сделка случается ровно в момент snapshot'а, оба `print_sim_stats`
/// отработают подряд — это OK, один из них покажет состояние «до»,
/// другой «после», без блокировки.
fn spawn_stats_snapshot(
    state: Arc<RwLock<RealSimState>>,
    account: SharedAccount,
    tag_prefix: String,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(STATS_HEARTBEAT_INTERVAL_SEC));
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // Первый tick срабатывает мгновенно — пропускаем, чтобы snapshot
        // не печатался до первой реальной активности (на старте все
        // счётчики нулевые, такой снапшот бесполезен и шумит в логе).
        tick.tick().await;
        loop {
            tick.tick().await;
            let state_guard = state.read().await;
            // Снапшот `bankroll` / `max_drawdown_pct` под per-field read-локами:
            // оба short-lived, держатся только пока копируем `f64`. Не висят
            // через печать `print_sim_stats`, чтобы trade-воркеры могли
            // спокойно идти в `update_drawdown` / payout.
            let bankroll_now = *account.bankroll.read().await;
            let max_drawdown_pct_now = *account.max_drawdown_pct.read().await;
            for kind in [XFrameIntervalKind::FiveMin, XFrameIntervalKind::FifteenMin] {
                let Some(stats) = state_guard.stats.get(&kind) else {
                    continue;
                };
                let tag = format!("{tag_prefix}/{} [heartbeat]", interval_label(kind));
                print_sim_stats(&tag, stats, bankroll_now, max_drawdown_pct_now, true);
            }
        }
    });
}

/// Сообщение из `catch_unwind` для лога.
fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Один кадр: HTTP strict-book при необходимости, [`manage_positions`], опционально вход.
async fn tick_once(
    book_tx: &mpsc::Sender<BookRequest>,
    state: &Arc<RwLock<RealSimState>>,
    account: &SharedAccount,
    currency: &str,
    last_snapshot_by_asset_id: &Arc<RwLock<HashMap<String, MarketSnapshot>>>,
    interval_kind: XFrameIntervalKind,
    side: CurrencyUpDownOutcome,
    models: &SideModels,
    tag: &str,
    last_market_id: &mut Option<String>,
    lane_frame: LaneFrame,
) -> Result<()> {
    let LaneFrame {
        market_id,
        asset_id,
        event_start_ms,
        event_end_ms,
        price_to_beat,
        gamma_question,
        frame,
    } = lane_frame;

    // `market_changed` до early-return — для dedupe `events`; резолюция — в колбеке Account.
    let market_changed = last_market_id.as_deref() != Some(market_id.as_str());

    if market_changed {
        let mut state_guard = state.write().await;
        let RealSimState {
            seen_market_ids,
            stats,
            ..
        } = &mut *state_guard;
        let seen = seen_market_ids.entry(interval_kind).or_default();
        if seen.insert(market_id.clone()) {
            // FIFO-вытеснение: вышли за cap → выкидываем самый старый.
            // `shift_remove_index(0)` сохраняет порядок (O(n) копирование,
            // но n ≤ SEEN_MARKET_IDS_CAP, так что для 1024 элементов это
            // микросекунды и происходит максимум раз на тик).
            while seen.len() > SEEN_MARKET_IDS_CAP {
                seen.shift_remove_index(0);
            }
            stats
                .get_mut(&interval_kind)
                .expect("stats map initialized for both intervals")
                .events += 1;
        }
    }

    let Some(raw_prob) = frame.currency_implied_prob else {
        return Ok(());
    };
    if !raw_prob.is_finite() || raw_prob <= 0.0 || raw_prob >= 1.0 {
        crate::tee_eprintln!(
            "[real_sim] {tag}: bogus currency_implied_prob={raw_prob} \
             (market={market_id}) — кадр пропущен"
        );
        *last_market_id = Some(market_id);
        return Ok(());
    }
    let currency_implied_prob = raw_prob.clamp(0.001, 0.999);

    let lane_key = (currency.to_string(), interval_kind, side);

    // WS-prob в last_prob до HTTP; после fetch перепишем на effective_prob.
    {
        let mut last_prob = account.last_prob.write().await;
        last_prob.insert(lane_key.clone(), currency_implied_prob);
    }

    let (
        has_positions,
        needs_sell,
        available_bankroll_pre,
        dd_halt_active,
        account_max_dd_pct,
        market_already_resolved,
    ) = {
        // Snapshot фаза: берём read-локи только тех полей, которые читаем; в порядке
        // объявления полей `Account` (`bankroll → max_drawdown_pct → positions →
        // recently_resolved_markets`).
        let bankroll_guard = account.bankroll.read().await;
        let max_dd_guard = account.max_drawdown_pct.read().await;
        let positions_guard = account.positions.read().await;
        let recently_resolved_guard = account.recently_resolved_markets.read().await;

        let this_positions = positions_guard
            .get(&lane_key)
            .expect("Account.positions pre-populated by run_real_sim");
        let total_locked: f64 = positions_guard
            .values()
            .flat_map(|v| v.iter())
            .map(|p| p.entry_cost)
            .sum();
        let available = (*bankroll_guard - total_locked).max(0.0);
        let dd_halt = match crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT {
            Some(threshold) => *max_dd_guard >= threshold,
            None => false,
        };
        let market_resolved = recently_resolved_guard.contains(market_id.as_str());
        (
            !this_positions.is_empty(),
            any_position_would_sell(this_positions, &frame, None),
            available,
            dd_halt,
            *max_dd_guard,
            market_resolved,
        )
    };

    // Predict вне write-локов; gate до HTTP с `strict_book=None` (WS prob).
    let pnl_inference = compute_pnl_inference(
        &frame,
        &models.booster_pnl,
        models.calibration_pnl.as_ref(),
        true,
    );
    let p_win_now = compute_p_win_now(
        &frame,
        models.booster_resolution.as_deref(),
        models.calibration_resolution.as_ref(),
        true,
        HOLD_TO_END_THRESHOLD_SEC,
    );

    let buy_gate_proceed = matches!(
        buy_gate(&frame, pnl_inference, available_bankroll_pre, None, true),
        BuyGate::Proceed { .. }
    );
    let may_open = !dd_halt_active && !market_already_resolved && buy_gate_proceed;
    if buy_gate_proceed && dd_halt_active {
        crate::tee_eprintln!(
            "[real_sim] {tag}: halt by drawdown — новые позиции заблокированы (порог={:?}%, max_dd_pct={:.2}%), закрытия продолжаем",
            crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT,
            account_max_dd_pct
        );
    }
    if buy_gate_proceed && market_already_resolved {
        crate::tee_eprintln!(
            "[real_sim] {tag}: skip open — market={market_id} уже резолвнулся, кадр пришёл с задержкой"
        );
    }
    let needs_http = needs_sell || may_open;

    // Сначала пробуем «живой» WS-снимок: если в кэше
    // (`ProjectManager::last_snapshot_by_asset_id`) есть свежий мердж
    // (≤ `WS_STRICT_BOOK_MAX_AGE_MS`) с заполненными лестницами `book_bids/asks` —
    // собираем StrictBook напрямую и пропускаем HTTP `order_books` (экономим
    // батч-окно `BOOK_BATCH_*` + сетевой roundtrip). При отсутствии/протухании
    // кэша поведение прежнее: один request через `run_book_coordinator`.
    let strict_book: Option<StrictBook> = if needs_http {
        match try_fresh_ws_strict_book(last_snapshot_by_asset_id, &asset_id, current_timestamp_ms())
            .await
        {
            Some(book) => Some(book),
            None => fetch_http_strict_book(book_tx, &asset_id, tag).await,
        }
    } else {
        None
    };

    // Свежесть WS определяется по верхним 3 уровням HTTP-стакана относительно L1/L2/L3 WS-кадра.
    let ws_lagging = match strict_book.as_ref() {
        Some(book) => {
            let lagging = is_ws_lagging(book, &frame);
            if lagging {
                crate::tee_eprintln!(
                    "[real_sim] {tag}: WS отстаёт — ордербук по HTTP расходится с last XFrame (market={market_id} asset={asset_id}); новые позиции пропускаем, ведём только закрытия"
                );
            }
            lagging
        }
        None => false,
    };

    // SHAP-топ для CSV считаем **до** взятия trade write-лока: `predict_contributions`
    // — это XGBoost-инференс (~ms), под `state.write + account.write` он сериализовал бы
    // все 4 воркера. Гейтим тем же `may_open && !ws_lagging`, что и сам `try_open_position`;
    // если внутри лока `may_open` обнулится (dd_halt/resolve между snapshot и write) —
    // строка просто будет отброшена, корректность не страдает, теряется только этот CPU-расчёт.
    let pnl_top5_shap_at_open_precomputed: Option<String> = if may_open
        && !ws_lagging
        && !crate::history_sim::HISTORY_SIM_SKIP_TRADE_SHAP_CONTRIBUTIONS
    {
        Some(crate::history_sim::top_pnl_shap_features_csv_cell(
            &models.booster_pnl,
            &frame,
            crate::train_mode::PNL_MAX_LAG,
            5,
        ))
    } else {
        Some(String::new())
    };

    // Торговля + MtM: порядок state.write → bankroll → max_drawdown_pct →
    // last_prob → positions → pending_resolution → closing → recently_resolved_markets.
    // state дропаем после фазы торговли — MtM только по Account; печать — под read ниже.
    let mut sold = false;
    let mut bought = false;
    // effective_prob: как в history_sim (HTTP mid/last trade → fallback WS); шкала MtM / EV-exit.
    let effective_prob = crate::history_sim::effective_implied_prob(&frame, strict_book.as_ref())
        .unwrap_or(currency_implied_prob);
    {
        let mut state_guard = state.write().await;
        // Поля `Account` берём индивидуально в порядке объявления, чтобы избежать
        // deadlock'а с другими потребителями (см. doc у `Account`).
        // `peak_bankroll` нужен только в MtM-фазе, но захватываем его в правильном
        // порядке заранее, иначе получили бы инверсию с `update_drawdown`/`_blocking`,
        // которые берут peak → max_dd.
        let mut bankroll_guard = account.bankroll.write().await;
        let mut peak_guard = account.peak_bankroll.write().await;
        let mut max_dd_guard = account.max_drawdown_pct.write().await;
        let mut last_prob_guard = account.last_prob.write().await;
        let mut positions_guard = account.positions.write().await;
        let mut pending_guard = account.pending_resolution.write().await;
        let mut closing_guard = account.closing.write().await;
        let recently_resolved_guard = account.recently_resolved_markets.read().await;

        // Обновить last_prob после HTTP (первая запись была WS до fetch).
        last_prob_guard.insert(lane_key.clone(), effective_prob);

        // Повторная проверка halt после HTTP (snapshot мог устареть за время запроса).
        let dd_halt_now = match crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT {
            Some(threshold) => *max_dd_guard >= threshold,
            None => false,
        };
        if !dd_halt_active && dd_halt_now && may_open {
            crate::tee_eprintln!(
                "[real_sim] {tag}: halt by drawdown сработал между snapshot'ом и HTTP — \
                 новый вход отменяем (max_dd_pct={:.2}%)",
                *max_dd_guard
            );
        }
        // Та же логика для резолюции: колбек `Account::resolve_pending_market`
        // мог вписать `market_id` в `recently_resolved_markets` за время HTTP-fetch.
        // Без этой повторной сверки `try_open_position` может открыть позицию
        // в только что резолвнутом маркете (до следующего тика, когда
        // `manage_positions` вытолкнет её в pending как чужой `asset_id`).
        let market_resolved_now = recently_resolved_guard.contains(market_id.as_str());
        if !market_already_resolved && market_resolved_now && may_open {
            crate::tee_eprintln!(
                "[real_sim] {tag}: market={market_id} резолвнулся между snapshot'ом и HTTP — отмена входа"
            );
        }
        // recently_resolved-лок больше не нужен — отпускаем, чтобы не держать его
        // через торговую фазу.
        drop(recently_resolved_guard);
        let may_open = may_open && !dd_halt_now && !market_resolved_now;

        // Фаза 1: торговля (sold/bought из возвратов manage_positions / try_open_position).
        if has_positions || may_open {
            // Чужие лейны + их pending: entry_cost всё ещё занят до резолюции.
            let cross_lanes_locked: f64 = positions_guard
                .iter()
                .filter(|(k, _)| *k != &lane_key)
                .flat_map(|(_, v)| v.iter())
                .map(|p| p.entry_cost)
                .chain(
                    pending_guard
                        .iter()
                        .filter(|(k, _)| *k != &lane_key)
                        .flat_map(|(_, v)| v.iter())
                        .map(|p| p.entry_cost),
                )
                .sum();

            let stats: &mut SimStats = state_guard
                .stats
                .get_mut(&interval_kind)
                .expect("stats map initialized for both intervals");
            let side_stats = match side {
                CurrencyUpDownOutcome::Up => &mut stats.up,
                CurrencyUpDownOutcome::Down => &mut stats.down,
            };

            // Три разные HashMap — три get_mut на один lane_key без конфликта.
            let this_positions: &mut Vec<OpenPosition> = positions_guard
                .get_mut(&lane_key)
                .expect("Account.positions pre-populated by run_real_sim");
            let this_pending: &mut Vec<OpenPosition> = pending_guard
                .get_mut(&lane_key)
                .expect("Account.pending_resolution pre-populated by run_real_sim");
            let this_closing: &mut Vec<crate::history_sim::ClosingPosition> = closing_guard
                .get_mut(&lane_key)
                .expect("Account.closing pre-populated by run_real_sim");

            // Закрытия / carry / pending при смене маркета; payout — в Account::resolve_pending_market.
            if has_positions {
                sold = manage_positions(
                    this_positions,
                    this_pending,
                    this_closing,
                    &frame,
                    false, // history_sim only: last-frame fallback
                    p_win_now,
                    side_stats,
                    &mut bankroll_guard,
                    strict_book.as_ref(),
                    None, // MINPOSITION_FRAMES — выдержка нужна только в history_sim
                );
            }

            // BUY: без входа при ws_lagging; bankroll после возможного sell этого лейна.
            if may_open && !ws_lagging {
                let same_locked_post: f64 = this_positions
                    .iter()
                    .chain(this_pending.iter())
                    .map(|p| p.entry_cost)
                    .sum();
                let available_bankroll_post =
                    (*bankroll_guard - cross_lanes_locked - same_locked_post).max(0.0);
                let polymarket_url = polymarket_event_url_from_frame(
                    currency,
                    interval_kind,
                    event_start_ms,
                );
                let graph_dump_bin_path_str = gamma_question
                    .as_deref()
                    .map(|gq| {
                        let stem =
                            crate::util::sanitized_filename_from_gamma_question(Some(gq));
                        crate::xframe_dump::synthetic_xframes_dump_bin_path_for_csv_link(
                            currency,
                            interval_kind,
                            &stem,
                        )
                    })
                    .flatten()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                bought = try_open_position(
                    &frame,
                    pnl_inference,
                    Some(&models.booster_pnl),
                    this_positions,
                    side_stats,
                    available_bankroll_post,
                    strict_book.as_ref(),
                    currency,
                    true,
                    &polymarket_url,
                    price_to_beat,
                    None,
                    event_end_ms,
                    graph_dump_bin_path_str.as_str(),
                    gamma_question.as_deref(),
                    // SHAP уже посчитан вне локов; передаём строку как override.
                    pnl_top5_shap_at_open_precomputed,
                );
            }
        }

        drop(state_guard);
        // closing нужен был только для торговой фазы выше; отпускаем заранее
        // — MtM ниже его не трогает.
        drop(closing_guard);

        // Фаза 2: портфельный MtM каждый тик → update_drawdown. Активные: shares × prob (этот лейн — effective_prob,
        // остальные — last_prob, clamp). Pending: shares × buy_price (капитал заблокирован до резолюции).
        let total_value: f64 = {
            let active: f64 = positions_guard
                .iter()
                .map(|((c, i, s), pos_vec)| {
                    let prob_raw = if c.as_str() == currency && *i == interval_kind && *s == side {
                        effective_prob
                    } else {
                        last_prob_guard
                            .get(&(c.clone(), *i, *s))
                            .copied()
                            .unwrap_or(0.5)
                    };
                    let prob = if prob_raw.is_finite() {
                        prob_raw.clamp(0.001, 0.999)
                    } else {
                        0.5
                    };
                    pos_vec.iter().map(|p| p.shares_held * prob).sum::<f64>()
                })
                .sum();
            let pending: f64 = pending_guard
                .values()
                .flat_map(|v| v.iter())
                .map(|p| p.shares_held * p.buy_price)
                .sum();
            active + pending
        };
        let equity = *bankroll_guard + total_value;
        // Пишем `peak_bankroll` / `max_drawdown_pct` инлайном, не дёргая
        // `update_drawdown(_blocking)`: те хотят свои собственные локи на эти
        // поля, но мы уже держим `peak_guard` / `max_dd_guard` write-локами
        // (захвачены выше в каноническом порядке).
        if equity > *peak_guard {
            *peak_guard = equity;
        }
        if *peak_guard > 0.0 {
            let drawdown_pct = (*peak_guard - equity) / *peak_guard * 100.0;
            if drawdown_pct > *max_dd_guard {
                *max_dd_guard = drawdown_pct;
            }
        }
    }

    // Печать только при сделке; снимок bankroll/max_dd под short-lived read-локами
    // (consistency best-effort после дропа write выше).
    if bought || sold {
        let state_guard = state.read().await;
        let bankroll_now = *account.bankroll.read().await;
        let max_drawdown_pct_now = *account.max_drawdown_pct.read().await;
        let stats = state_guard
            .stats
            .get(&interval_kind)
            .expect("stats map initialized for both intervals");
        let action = if bought && sold {
            "buy+sell"
        } else if bought {
            "buy"
        } else {
            "sell"
        };
        crate::tee_println!(
            "[real_sim] {tag}: {action} @ t={} market={market_id} prob={currency_implied_prob:.4}",
            current_timestamp_ms(),
        );
        print_sim_stats(tag, stats, bankroll_now, max_drawdown_pct_now, true); // kelly
    }

    *last_market_id = Some(market_id);
    Ok(())
}

/// Запрос в [`run_book_coordinator`]: `asset_id` + oneshot; ответ `Some(StrictBook)` или `None`.
struct BookRequest {
    asset_id: String,
    reply: oneshot::Sender<Option<StrictBook>>,
}

/// Один [`StrictBook`] за тик через координатор (батч `order_books`). Ошибки → `None`, торговля на WS-fallback.
async fn fetch_http_strict_book(
    book_tx: &mpsc::Sender<BookRequest>,
    asset_id: &str,
    tag: &str,
) -> Option<StrictBook> {
    let (reply_tx, reply_rx) = oneshot::channel();
    let req = BookRequest {
        asset_id: asset_id.to_string(),
        reply: reply_tx,
    };
    if book_tx.send(req).await.is_err() {
        crate::tee_eprintln!(
            "[real_sim] {tag}: book-coord канал закрыт — strict-fill выключен на тик"
        );
        return None;
    }
    // Иначе воркер зависнет на oneshot и забьёт канал кадров.
    match tokio::time::timeout(Duration::from_millis(BOOK_REPLY_TIMEOUT_MS), reply_rx).await {
        Ok(Ok(book)) => book,
        Ok(Err(_)) => {
            crate::tee_eprintln!(
                "[real_sim] {tag}: book-coord уронил oneshot до ответа — strict-fill выключен на тик"
            );
            None
        }
        Err(_) => {
            crate::tee_eprintln!(
                "[real_sim] {tag}: ожидание ответа book-coord > {BOOK_REPLY_TIMEOUT_MS}ms — strict-fill выключен на тик"
            );
            None
        }
    }
}

/// Собрать [`StrictBook`] из последнего «живого» WS-снимка
/// ([`ProjectManager::last_snapshot_by_asset_id`]) — или `None`, если в нём
/// нет полных лестниц `book_bids/asks` (например, шёл только поток
/// `price_change` со чисто L1 `best_bid`/`best_ask`, а агрегата `book` ещё не
/// случилось). `min_order_size` в WS-сообщениях не приходит, оставляем `None`
/// — это соответствует «без min-фильтра» в `book_fill_*_strict` (см.
/// `if let Some(min) = book.min_order_size` в [`crate::history_sim`]).
fn strict_book_from_snapshot(snapshot: &MarketSnapshot) -> Option<StrictBook> {
    let bids = snapshot.book_bids.clone()?;
    let asks = snapshot.book_asks.clone()?;
    if bids.is_empty() || asks.is_empty() {
        return None;
    }
    Some(StrictBook {
        bids,
        asks,
        last_trade_price: snapshot.last_trade_price,
        min_order_size: None,
    })
}

/// Свежий WS-снимок по `asset_id` → [`StrictBook`]; иначе `None` и идём в HTTP.
///
/// Условия freshness: `now_ms - snapshot.timestamp_ms <= WS_STRICT_BOOK_MAX_AGE_MS`
/// **и** в снимке есть обе лестницы (`book_bids`+`book_asks` непусты). Read-лок
/// short-lived: только на `get` + клон — мерджит снимки писатель в
/// [`ProjectManager::update_last_snapshot`], тут только читаем.
async fn try_fresh_ws_strict_book(
    last_snapshot_by_asset_id: &Arc<RwLock<HashMap<String, MarketSnapshot>>>,
    asset_id: &str,
    now_ms: i64,
) -> Option<StrictBook> {
    let guard = last_snapshot_by_asset_id.read().await;
    let snapshot = guard.get(asset_id)?;
    if now_ms.saturating_sub(snapshot.timestamp_ms) > WS_STRICT_BOOK_MAX_AGE_MS {
        return None;
    }
    strict_book_from_snapshot(snapshot)
}

/// Один таск: собирает [`BookRequest`] в батч (idle + max wait + cap по числу лейнов), дедуп по `asset_id`, один `order_books`.
async fn run_book_coordinator(
    project_manager: Arc<ProjectManager>,
    mut rx: mpsc::Receiver<BookRequest>,
) {
    while let Some(first) = rx.recv().await {
        let mut batch: Vec<BookRequest> = vec![first];

        // Idle между recv + абсолютный max + не больше одного запроса на лейн.
        let absolute_deadline = tokio::time::Instant::now() + Duration::from_millis(BOOK_BATCH_MAX_MS);
        while batch.len() < LANE_FRAME_ROUTES.len() {
            let idle_deadline = tokio::time::Instant::now() + Duration::from_millis(BOOK_BATCH_IDLE_MS);
            let next_deadline = idle_deadline.min(absolute_deadline);
            match tokio::time::timeout_at(next_deadline, rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                Ok(None) | Err(_) => break, // канал закрыт ИЛИ idle/absolute истёк
            }
        }

        let mut by_asset: HashMap<String, Vec<oneshot::Sender<Option<StrictBook>>>> = HashMap::new();
        for req in batch {
            by_asset.entry(req.asset_id).or_default().push(req.reply);
        }

        let mut requests: Vec<OrderBookSummaryRequest> = Vec::with_capacity(by_asset.len());
        let mut valid_ids: Vec<String> = Vec::with_capacity(by_asset.len());
        let invalid_ids: Vec<String> = by_asset
            .keys()
            .filter(|aid| U256::from_str(aid).is_err())
            .cloned()
            .collect();
        for aid in invalid_ids {
            crate::tee_eprintln!("[real_sim/book-coord] невалидный asset_id={aid} — отвечаем None");
            if let Some(senders) = by_asset.remove(&aid) {
                for s in senders {
                    let _ = s.send(None);
                }
            }
        }
        for aid in by_asset.keys() {
            let token_id = U256::from_str(aid).expect("invalid asset_ids filtered above");
            requests.push(
                OrderBookSummaryRequest::builder()
                    .token_id(token_id)
                    .build(),
            );
            valid_ids.push(aid.clone());
        }

        if requests.is_empty() {
            continue;
        }

        let n = requests.len();
        let http_result = tokio::time::timeout(
            Duration::from_millis(BOOK_HTTP_TIMEOUT_MS),
            project_manager.clob.order_books(&requests),
        )
        .await;
        match http_result {
            Ok(Ok(responses)) if responses.len() == n => {
                for (aid, resp) in valid_ids.iter().zip(responses.iter()) {
                    let book = parse_book_levels(resp);
                    if let Some(senders) = by_asset.remove(aid) {
                        let mut iter = senders.into_iter();
                        if let Some(last) = iter.next_back() {
                            for s in iter {
                                let _ = s.send(Some(book.clone()));
                            }
                            let _ = last.send(Some(book));
                        }
                    }
                }
            }
            Ok(Ok(responses)) => {
                crate::tee_eprintln!(
                    "[real_sim/book-coord] order_books вернул {} ответов на {n} запросов — отбрасываем батч",
                    responses.len(),
                );
                for senders in by_asset.into_values() {
                    for s in senders {
                        let _ = s.send(None);
                    }
                }
            }
            Ok(Err(err)) => {
                crate::tee_eprintln!(
                    "[real_sim/book-coord] order_books({n} assets) failed: {err:#}"
                );
                for senders in by_asset.into_values() {
                    for s in senders {
                        let _ = s.send(None);
                    }
                }
            }
            Err(_) => {
                crate::tee_eprintln!(
                    "[real_sim/book-coord] order_books({n} assets) timed out > {BOOK_HTTP_TIMEOUT_MS}ms — отбрасываем батч"
                );
                for senders in by_asset.into_values() {
                    for s in senders {
                        let _ = s.send(None);
                    }
                }
            }
        }
    }
    crate::tee_eprintln!("[real_sim/book-coord] mpsc закрыт — координатор завершён");
}

/// CLOB `POST /books` → [`StrictBook`]: bids/asks лучший→худший (реверс API), фильтр мусора;
/// плюс `last_trade_price` и `min_order_size` для [`crate::history_sim::effective_implied_prob`] и strict-fill.
fn parse_book_levels(book: &OrderBookSummaryResponse) -> StrictBook {
    let to_level = |o: &polymarket_client_sdk::clob::types::response::OrderSummary| {
        let price = o.price.to_string().parse::<f64>().ok()?;
        let size = o.size.to_string().parse::<f64>().ok()?;
        if price <= 0.0 || size <= 0.0 {
            return None;
        }
        Some(BookLevel { price, size })
    };
    let bids: Vec<BookLevel> = book.bids.iter().rev().filter_map(to_level).collect();
    let asks: Vec<BookLevel> = book.asks.iter().rev().filter_map(to_level).collect();
    let last_trade_price = book
        .last_trade_price
        .and_then(|d| d.to_string().parse::<f64>().ok())
        .filter(|p| p.is_finite() && *p > 0.0);
    let min_order_size = book
        .min_order_size
        .to_string()
        .parse::<f64>()
        .ok()
        .filter(|s| s.is_finite() && *s > 0.0);
    StrictBook {
        bids,
        asks,
        last_trade_price,
        min_order_size,
    }
}

/// HTTP vs WS по L1–L3 bid/ask; порог `2×tick_size` (или 0.02 при дефолтном tick). Три уровня — ловим stale, когда L1 совпал случайно.
fn is_ws_lagging(book: &StrictBook, frame: &XFrame<SIZE>) -> bool {
    let tol = frame.tick_size.unwrap_or(0.01).max(1e-6) * 2.0;
    let diverges = |ws: Option<f64>, http: Option<f64>| -> bool {
        match (ws, http) {
            (Some(a), Some(b)) => (a - b).abs() > tol,
            (None, Some(_)) | (Some(_), None) => true,
            (None, None) => false,
        }
    };

    let http_level = |side: &[BookLevel], idx: usize| side.get(idx).map(|l| l.price);
    let http_bid = |i| http_level(&book.bids, i);
    let http_ask = |i| http_level(&book.asks, i);

    let ws_bid = [
        frame.book_bid_l1_price,
        frame.book_bid_l2_price,
        frame.book_bid_l3_price,
    ];
    let ws_ask = [
        frame.book_ask_l1_price,
        frame.book_ask_l2_price,
        frame.book_ask_l3_price,
    ];

    let bid_bad = (0..ws_bid.len()).any(|i| diverges(ws_bid[i], http_bid(i)));
    let ask_bad = (0..ws_ask.len()).any(|i| diverges(ws_ask[i], http_ask(i)));

    if bid_bad || ask_bad {
        crate::tee_eprintln!(
            "[real_sim] WS vs HTTP ордербук (tol={tol:.4}):\n  \
             bid WS  L1/L2/L3 = {:?}/{:?}/{:?}\n  \
             bid HTTP L1/L2/L3 = {:?}/{:?}/{:?}\n  \
             ask WS  L1/L2/L3 = {:?}/{:?}/{:?}\n  \
             ask HTTP L1/L2/L3 = {:?}/{:?}/{:?}",
            ws_bid[0], ws_bid[1], ws_bid[2],
            http_bid(0), http_bid(1), http_bid(2),
            ws_ask[0], ws_ask[1], ws_ask[2],
            http_ask(0), http_ask(1), http_ask(2),
        );
        true
    } else {
        false
    }
}

fn latest_version_path(currency: &str) -> Option<PathBuf> {
    let base = Path::new("xframes").join(currency);
    let mut versions: Vec<(usize, PathBuf)> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if !path.is_dir() {
                return None;
            }
            let name = path.file_name()?.to_string_lossy().to_string();
            let n = name.parse::<usize>().ok()?;
            Some((n, path))
        })
        .collect();
    versions.sort_by_key(|(n, _)| *n);
    versions.pop().map(|(_, p)| p)
}

fn load_side_models(version_path: &Path, interval: &str, side: &str) -> Option<SideModels> {
    let pnl_path = version_path.join(format!("model_{interval}_1s_pnl_{side}.ubj"));
    let resolution_path =
        version_path.join(format!("model_{interval}_1s_resolution_{side}.ubj"));

    let booster_pnl = load_booster(&pnl_path)?;
    let calibration_pnl = load_calibration(&pnl_path).ok();
    let booster_resolution = load_booster(&resolution_path).map(Arc::new);
    let calibration_resolution = load_calibration(&resolution_path).ok();

    Some(SideModels {
        booster_pnl: Arc::new(booster_pnl),
        calibration_pnl,
        booster_resolution,
        calibration_resolution,
    })
}

fn dir_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}
