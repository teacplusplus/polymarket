//! Live-сим: логика как [`crate::history_sim`], кадры из [`ProjectManager::xframes_by_market`], HTTP-батч стаканов [`run_book_coordinator`].

use crate::account::SharedAccount;
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
/// Реэкспорт cap slippage для TP/strict ([`crate::history_sim`]).
pub use crate::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT;
use crate::history_sim::{
    BuyGate, ENABLE_PNL, ENABLE_RESOLUTION, MIN_ENTRY_REMAINING_MS, REDEEM_01, StrictBook,
    any_position_would_sell, buy_gate, compute_pnl_inference, compute_resolution_inference,
    load_booster, manage_positions, try_open_position,
};
use crate::market_snapshot::MarketSnapshot;
use crate::project_manager::{LaneFrame, ProjectManager};
use crate::sim_stats::{SimStats};
use crate::train_mode::{Calibration, load_calibration};
use crate::util::current_timestamp_ms;
use crate::xframe::BookLevel;

use anyhow::{Result, anyhow};
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
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, oneshot};
use xgb::Booster;

/// Размер `mpsc` для [`LaneFrame`] на лейн.
const LANE_FRAME_CHANNEL_CAP: usize = 64;

/// Размер `mpsc` [`BookRequest`] → [`run_book_coordinator`].
const BOOK_REQUEST_CHANNEL_CAP: usize = 64;

/// Пауза добора батча запросов стакана (мс).
const BOOK_BATCH_IDLE_MS: u64 = 5;

/// Макс. ожидание батча от первого запроса (мс).
const BOOK_BATCH_MAX_MS: u64 = 50;

/// Таймаут HTTP `order_books` (мс); по истечении — `None` ожидающим.
const BOOK_HTTP_TIMEOUT_MS: u64 = 2000;

/// Макс. возраст WS-снимка (мс), чтобы собрать [`StrictBook`] без HTTP ([`ProjectManager::last_snapshot_by_asset_id`]).
pub(crate) const WS_STRICT_BOOK_MAX_AGE_MS: i64 = 45_000;

/// Таймаут ответа координатора в [`fetch_http_strict_book`] (~3× HTTP).
const BOOK_REPLY_TIMEOUT_MS: u64 = BOOK_HTTP_TIMEOUT_MS * 3;

/// Max время от старта [`tick_once`] до [`try_open_position`]; продажи не ограничиваются,
/// но lag на пути закрытия тоже логируем.
const TICK_OPEN_MAX_ELAPSED: Duration = Duration::from_millis(500);

/// Лимит FIFO [`RealSimState::seen_market_ids`] на интервал.
const SEEN_MARKET_IDS_CAP: usize = 8;

/// Четыре лейна фанаута 1s: `(interval, side)`.
const LANE_FRAME_ROUTES: [(XFrameIntervalKind, CurrencyUpDownOutcome); 4] = [
    (XFrameIntervalKind::FifteenMin, CurrencyUpDownOutcome::Down),
    (XFrameIntervalKind::FifteenMin, CurrencyUpDownOutcome::Up),
    (XFrameIntervalKind::FiveMin, CurrencyUpDownOutcome::Down),
    (XFrameIntervalKind::FiveMin, CurrencyUpDownOutcome::Up),
];

/// Фанаут lane 0 → side-воркеры.
#[derive(Debug)]
pub struct LaneFrameChannels {
    /// [`LaneFrame`] на `(interval, side)`.
    pub channels:
        Arc<RwLock<HashMap<(XFrameIntervalKind, CurrencyUpDownOutcome), mpsc::Sender<LaneFrame>>>>,
}

impl LaneFrameChannels {
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// Статистика по интервалам, фанаут кадров, dedupe `events`.
#[derive(Debug)]
pub struct RealSimState {
    /// [`SimStats`] для 5m и 15m.
    pub stats: HashMap<XFrameIntervalKind, SimStats>,
    /// Каналы на side-воркеры.
    pub lane_frame_channels: LaneFrameChannels,
    /// Виденные `market_id` на интервал (счётчик событий + cap [`SEEN_MARKET_IDS_CAP`]).
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

/// Booster(+cal) PnL и опционально resolution для одного лейна.
struct SideModels {
    /// Модель PnL.
    booster_pnl: Option<Booster>,
    /// Калибровка PnL.
    calibration_pnl: Option<Calibration>,
    /// Модель resolution.
    booster_resolution: Option<Booster>,
    /// Калибровка resolution.
    calibration_resolution: Option<Calibration>,
}

pub(crate) fn interval_label(kind: XFrameIntervalKind) -> &'static str {
    match kind {
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    }
}

/// `https://polymarket.com/event/...` из Gamma `start_ms` ([`ProjectManager::event_data_by_market`]); пустая строка если `None`.
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

/// Загрузка моделей, публикация [`RealSimState`] в [`crate::account::Account`], 4 воркера
/// [`tick_once`]. `submit_mode` — выбор CLOB-исполнителя (`Submit` боевой, `Mock` фейковый
/// через WS-стакан, `None` чисто виртуально); см. [`crate::account_submit::SubmitMode`].
pub async fn run_real_sim(
    project_manager: Arc<ProjectManager>,
    submit_mode: crate::account_submit::SubmitMode,
) -> Result<()> {
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

    let account = project_manager.account.clone();

    let state = Arc::new(RwLock::new(RealSimState::new()));
    {
        let mut map = account.real_sim_state_by_currency.write().await;
        map.insert(currency.to_string(), state.clone());
    }
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

    for (interval_kind, side) in LANE_FRAME_ROUTES {
        let label = interval_label(interval_kind);
        let side_lbl = side_label(side);
        let models = load_side_models(&version_path, label, side_lbl)
            .ok_or_else(|| anyhow!("не загружено ни одной модели {label}/{side_lbl}"))?;
        crate::tee_println!(
            "[real_sim] {tag_prefix}/{label}/{side_lbl}: pnl={}  resolution={}",
            if models.booster_pnl.is_some() { "✓" } else { "✗" },
            if models.booster_resolution.is_some() { "✓" } else { "✗" },
        );

        let (tx, rx) = mpsc::channel::<LaneFrame>(LANE_FRAME_CHANNEL_CAP);
        channels.write().await.insert((interval_kind, side), tx);

        spawn_side_worker(
            book_tx.clone(),
            state.clone(),
            account.clone(),
            project_manager.clone(),
            currency_arc.clone(),
            models,
            rx,
            submit_mode,
        );
    }

    Ok(())
}

fn spawn_side_worker(
    book_tx: mpsc::Sender<BookRequest>,
    state: Arc<RwLock<RealSimState>>,
    account: SharedAccount,
    project_manager: Arc<ProjectManager>,
    currency: Arc<String>,
    models: SideModels,
    mut rx: mpsc::Receiver<LaneFrame>,
    submit_mode: crate::account_submit::SubmitMode,
) {
    tokio::spawn(async move {
        let mut last_market_id: Option<String> = None;
        let mut last_tick_wall_ms_by_asset_id: HashMap<String, i64> = HashMap::new();
        while let Some(lane_frame) = rx.recv().await {
            let result = AssertUnwindSafe(tick_once(
                &book_tx,
                &state,
                &account,
                &project_manager,
                currency.as_str(),
                &models,
                &mut last_market_id,
                &mut last_tick_wall_ms_by_asset_id,
                lane_frame,
                submit_mode,
            ))
            .catch_unwind()
            .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    crate::tee_eprintln!("[real_sim] tick error: {err:#}");
                }
                Err(payload) => {
                    let msg = panic_payload_message(&payload);
                    crate::tee_eprintln!(
                        "[real_sim] tick PANIC ({msg}) — кадр пропущен, воркер живой"
                    );
                }
            }
        }
        crate::tee_eprintln!("[real_sim] канал закрыт — воркер завершён");
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

/// Один кадр: WS/HTTP [`StrictBook`], [`manage_positions`], опционально [`try_open_position`].
#[allow(clippy::too_many_arguments)]
async fn tick_once(
    book_tx: &mpsc::Sender<BookRequest>,
    state: &Arc<RwLock<RealSimState>>,
    account: &SharedAccount,
    project_manager: &Arc<ProjectManager>,
    currency: &str,
    models: &SideModels,
    last_market_id: &mut Option<String>,
    last_tick_wall_ms_by_asset_id: &mut HashMap<String, i64>,
    lane_frame: LaneFrame,
    submit_mode: crate::account_submit::SubmitMode,
) -> Result<()> {
    let LaneFrame {
        market_id,
        asset_id,
        price_to_beat,
        gamma_question,
    } = lane_frame;

    let Some((aligned_ts, xframe_cell)) = project_manager
        .latest_xframe_cell(market_id.as_str(), asset_id.as_str())
        .await
    else {
        return Ok(());
    };

    let mut frame = xframe_cell.read().await.clone();

    // Минимальная выдержка позиции до первой проверки SL/TP в [`sell_gate`] /
    // [`any_position_would_sell`]. Включена в Mock-режиме (как в history_sim),
    // чтобы убирать micro-noise exits сразу после открытия. В Submit-режиме
    // реальные ордера уже стоят на CLOB и затягивать выход искусственно нельзя
    // (особенно SL) — поэтому `None`.
    let min_position_frames: Option<usize> = match submit_mode {
        crate::account_submit::SubmitMode::Mock => Some(crate::history_sim::MINPOSITION_FRAMES),
        crate::account_submit::SubmitMode::Submit | crate::account_submit::SubmitMode::None => None,
    };

    let market_id = market_id.as_str();
    let asset_id = asset_id.as_str();
    let Some(interval_kind) = XFrameIntervalKind::from_i32(frame.xframe_interval_type) else {
        return Ok(());
    };
    let Some(side) = CurrencyUpDownOutcome::from_i32(frame.currency_up_down_outcome) else {
        return Ok(());
    };
    let (event_start_ms, event_end_ms) = {
        let guard = project_manager.event_data_by_market.read().await;
        guard
            .get(market_id)
            .map(|e| (e.start_ms, e.end_ms))
            .unwrap_or((None, None))
    };
    // let direction = match side {
    //     CurrencyUpDownOutcome::Up => "UP",
    //     CurrencyUpDownOutcome::Down => "DOWN",
    // };
    // let polymarket_url = polymarket_event_url_from_frame(currency, interval_kind, event_start_ms);
    let now_wall_ms = current_timestamp_ms();
    // let since_prev_ms_opt = last_tick_wall_ms_by_asset_id
    //     .get(asset_id)
    //     .map(|prev_ms| now_wall_ms.saturating_sub(*prev_ms));
    // let since_prev_s = since_prev_ms_opt.map(|d| d as f64 / 1000.0);
    last_tick_wall_ms_by_asset_id.insert(asset_id.to_string(), now_wall_ms);
    // 
    // let ws_event_lag_ms: Option<i64> = {
    //     let guard = project_manager.last_snapshot_by_asset_id.read().await;
    //     guard
    //         .get(asset_id)
    //         .map(|snapshot| now_wall_ms.saturating_sub(snapshot.timestamp_ms))
    // };
    // println!(
    //     "[diag][worker] recv {direction} market={market_id} asset={asset_id} \
    //      ws_event_lag_ms={ws_event_lag_ms:?} since_prev_s={since_prev_s:?} recv_wall_ms={now_wall_ms} url={polymarket_url}",
    // );

    let tick_started = Instant::now();

    let market_changed = last_market_id.as_deref() != Some(market_id);

    if market_changed {
        let mut state_guard = state.write().await;
        let RealSimState {
            seen_market_ids,
            stats,
            ..
        } = &mut *state_guard;
        let seen = seen_market_ids.entry(interval_kind).or_default();
        if seen.insert(market_id.to_string()) {
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
            "[real_sim] bogus currency_implied_prob={raw_prob} \
             (market={market_id}) — кадр пропущен"
        );
        *last_market_id = Some(market_id.to_string());
        return Ok(());
    }
    let currency_implied_prob = raw_prob.clamp(0.001, 0.999);

    let lane_key = (currency.to_string(), interval_kind, side);

    {
        let mut last_prob = account.last_prob.write().await;
        last_prob.insert(lane_key.clone(), currency_implied_prob);
    }

    let market_already_resolved = event_end_ms.is_some_and(|end_ms| now_wall_ms >= end_ms);

    let (
        has_positions,
        needs_sell,
        available_bankroll_pre,
        dd_halt_active,
        account_max_dd_pct,
    ) = {
        let bankroll_guard = account.bankroll.read().await;
        let max_dd_guard = account.max_drawdown_pct.read().await;
        let positions_guard = account.positions.read().await;
        let pending_close_guard = account.pending_close_positions.read().await;

        let this_positions = positions_guard
            .get(&lane_key)
            .expect("Account.positions pre-populated by run_real_sim");
        let mut total_locked = 0.0;
        for v in positions_guard.values() {
            for p in v.values() {
                total_locked += p.read().await.position_size;
            }
        }
        for v in pending_close_guard.values() {
            for p in v.values() {
                total_locked += p.read().await.position_size;
            }
        }
        let available = (*bankroll_guard - total_locked).max(0.0);
        let dd_halt = match crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT {
            Some(threshold) => *max_dd_guard >= threshold,
            None => false,
        };
        (
            !this_positions.is_empty(),
            any_position_would_sell(this_positions, &frame, min_position_frames).await,
            available,
            dd_halt,
            *max_dd_guard,
        )
    };

    let submit_market_window_open: bool = match (event_start_ms, event_end_ms) {
        (Some(start_ms), Some(end_ms)) => now_wall_ms >= start_ms && now_wall_ms < end_ms,
        _ => false,
    };
    let could_attempt_entry = !dd_halt_active
        && !market_already_resolved
        && frame.stable
        && frame.currency_implied_prob.is_some()
        && frame.event_remaining_ms >= MIN_ENTRY_REMAINING_MS
        && submit_market_window_open;
    let needs_http = needs_sell || has_positions || could_attempt_entry;

    let strict_book: Option<StrictBook> = if needs_http {
        match try_fresh_ws_strict_book(project_manager, asset_id, now_wall_ms).await {
            Some(book) => Some(book),
            None => fetch_http_strict_book(book_tx, asset_id).await,
        }
    } else {
        None
    };

    if let Some(ref book) = strict_book {
        project_manager
            .apply_strict_book_to_latest_xframe(
                market_id,
                asset_id,
                aligned_ts,
                xframe_cell.clone(),
                book,
            )
            .await;
        frame = xframe_cell.read().await.clone();
    }

    let needs_sell = if has_positions {
        let positions_guard = account.positions.read().await;
        let this_positions = positions_guard
            .get(&lane_key)
            .expect("Account.positions pre-populated by run_real_sim");
        any_position_would_sell(this_positions, &frame, min_position_frames).await
    } else {
        false
    };

    let currency_implied_prob = frame
        .currency_implied_prob
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(0.001, 0.999))
        .unwrap_or(currency_implied_prob);

    let pnl_inference = compute_pnl_inference(
        &frame,
        models.booster_pnl.as_ref(),
        models.calibration_pnl.as_ref(),
        true,
    );
    let resolution_inference = compute_resolution_inference(
        &frame,
        models.booster_resolution.as_ref(),
        models.calibration_resolution.as_ref(),
        true,
    );

    let buy_gate_proceed = matches!(
        buy_gate(
            &frame,
            pnl_inference,
            resolution_inference,
            available_bankroll_pre,
            strict_book.as_ref(),
            true,
            models.booster_pnl.as_ref(),
            currency,
            event_end_ms,
            submit_mode,
            Some(account),
        )
        .await,
        BuyGate::Proceed { .. }
    );
    let may_open =
        !dd_halt_active && !market_already_resolved && buy_gate_proceed && submit_market_window_open;

    if buy_gate_proceed && dd_halt_active {
        crate::tee_eprintln!(
            "[real_sim] halt by drawdown — новые позиции заблокированы (порог={:?}%, max_dd_pct={:.2}%), закрытия продолжаем",
            crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT,
            account_max_dd_pct
        );
    }

    let pnl_top5_shap_at_open_precomputed: Option<String> = if may_open {
        Some(
            models
                .booster_pnl
                .as_ref()
                .map(|b| crate::history_sim::pnl_top5_shap_csv_cell(b, &frame))
                .unwrap_or_default(),
        )
    } else {
        Some(String::new())
    };

    let effective_prob = crate::history_sim::effective_implied_prob(&frame, strict_book.as_ref())
        .unwrap_or(currency_implied_prob);
    {
        let mut state_guard = state.write().await;
        // `account.bankroll` write пред-захватывать НЕ нужно:
        // [`crate::account_close_position::close_position`] (внутри
        // `manage_positions`) берёт его сам коротким локом для apply'я PNL;
        // try_open_position bankroll не пишет (только positions). Чтения для
        // `available_bankroll_post` / `equity` ниже берём отдельным read'ом.
        let mut peak_guard = account.peak_bankroll.write().await;
        let mut max_dd_guard = account.max_drawdown_pct.write().await;
        let mut last_prob_guard = account.last_prob.write().await;
        let mut positions_guard = account.positions.write().await;

        last_prob_guard.insert(lane_key.clone(), effective_prob);

        let dd_halt_now = match crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT {
            Some(threshold) => *max_dd_guard >= threshold,
            None => false,
        };
        if !dd_halt_active && dd_halt_now && may_open {
            crate::tee_eprintln!(
                "[real_sim] halt by drawdown сработал между snapshot'ом и HTTP — \
                 новый вход отменяем (max_dd_pct={:.2}%)",
                *max_dd_guard
            );
        }

        let now_after_http_ms = current_timestamp_ms();
        let market_resolved_now = event_end_ms
            .is_some_and(|end_ms| now_after_http_ms >= end_ms);
        if !market_already_resolved && market_resolved_now && may_open {
            crate::tee_eprintln!(
                "[real_sim] market={market_id} прошёл event_end_ms между snapshot'ом и HTTP \
                 (now={now_after_http_ms} end={:?}) — отмена входа",
                event_end_ms,
            );
        }
        let may_open = may_open && !dd_halt_now && !market_resolved_now;

        if has_positions || may_open {
            let stats: &mut SimStats = state_guard
                .stats
                .get_mut(&interval_kind)
                .expect("stats map initialized for both intervals");
            let side_stats = match side {
                CurrencyUpDownOutcome::Up => &mut stats.up,
                CurrencyUpDownOutcome::Down => &mut stats.down,
            };

            if has_positions {
                let this_positions: &mut crate::history_sim::LanePositions = positions_guard
                    .get_mut(&lane_key)
                    .expect("Account.positions pre-populated by run_real_sim");
                manage_positions(
                    this_positions,
                    &frame,
                    false,
                    side_stats,
                    strict_book.as_ref(),
                    min_position_frames,
                    submit_mode,
                    Some(project_manager),
                    account,
                    &lane_key,
                )
                .await;
                if needs_sell && tick_started.elapsed() > TICK_OPEN_MAX_ELAPSED {
                    crate::tee_eprintln!(
                        "[real_sim] tick elapsed {:.0}ms > {}ms \
                         (market={market_id}) — закрытие позиции обработано с лагом",
                        tick_started.elapsed().as_secs_f64() * 1000.0,
                        TICK_OPEN_MAX_ELAPSED.as_millis(),
                    );
                }
            }

            if may_open {
                if tick_started.elapsed() > TICK_OPEN_MAX_ELAPSED {
                    crate::tee_eprintln!(
                        "[real_sim] tick elapsed {:.0}ms > {}ms \
                         (market={market_id}) — новый вход пропущен",
                        tick_started.elapsed().as_secs_f64() * 1000.0,
                        TICK_OPEN_MAX_ELAPSED.as_millis(),
                    );
                } else {
                    // `available_bankroll_post` = bankroll − **весь** locked
                    // капитал по всем lane'ам в обоих HashMap'ах (как в
                    // [`crate::history_sim::run_side_simulation`]). Один
                    // read-guard на `pending_close_positions` под этим scope'ом
                    // также пробрасывается в [`try_open_position`] для гейтов
                    // [`MAX_OPEN_POSITIONS`] / [`BLOCK_SAME_ASSET_OPEN`].
                    // Lock-order соблюдён: positions write → pending_close read.
                    let pending_close_guard = account.pending_close_positions.read().await;
                    let mut total_locked = 0.0;
                    for lane_positions in positions_guard.values() {
                        for p in lane_positions.values() {
                            total_locked += p.read().await.position_size;
                        }
                    }
                    for lane_pending in pending_close_guard.values() {
                        for p in lane_pending.values() {
                            total_locked += p.read().await.position_size;
                        }
                    }
                    let bankroll_post = *account.bankroll.read().await;
                    let available_bankroll_post = (bankroll_post - total_locked).max(0.0);
                    let polymarket_url =
                        polymarket_event_url_from_frame(currency, interval_kind, event_start_ms);
                    let graph_dump_bin_path_str = gamma_question
                        .as_deref()
                        .map(|gq| {
                            let stem = crate::util::sanitized_filename_from_gamma_question(Some(gq));
                            crate::xframe_dump::synthetic_xframes_dump_bin_path_for_csv_link(
                                currency,
                                interval_kind,
                                &stem,
                                event_end_ms,
                            )
                        })
                        .flatten()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    try_open_position(
                        &frame,
                        pnl_inference,
                        resolution_inference,
                        models.booster_pnl.as_ref(),
                        &mut *positions_guard,
                        &*pending_close_guard,
                        &lane_key,
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
                        pnl_top5_shap_at_open_precomputed,
                        submit_mode,
                        Some(project_manager),
                        account,
                    )
                    .await;
                }
            }
        }

        drop(state_guard);

        let total_value: f64 = {
            let mut active = 0.0;
            let pending_close_guard = account.pending_close_positions.read().await;
            for ((c, i, s), pos_vec) in positions_guard
                .iter()
                .chain(pending_close_guard.iter())
            {
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
                for p in pos_vec.values() {
                    active += p.read().await.shares_held * prob;
                }
            }
            active
        };
        let equity = *account.bankroll.read().await + total_value;
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

    *last_market_id = Some(market_id.to_string());
    Ok(())
}

/// Запрос батча стакана в [`run_book_coordinator`].
struct BookRequest {
    /// CLOB token id.
    asset_id: String,
    /// Ответ: [`StrictBook`] или `None`.
    reply: oneshot::Sender<Option<StrictBook>>,
}

/// Один [`StrictBook`] за тик через координатор (батч `order_books`). Ошибки → `None`, торговля на WS-fallback.
async fn fetch_http_strict_book(
    book_tx: &mpsc::Sender<BookRequest>,
    asset_id: &str,
) -> Option<StrictBook> {
    let (reply_tx, reply_rx) = oneshot::channel();
    let req = BookRequest {
        asset_id: asset_id.to_string(),
        reply: reply_tx,
    };
    if book_tx.send(req).await.is_err() {
        crate::tee_eprintln!(
            "[real_sim] book-coord канал закрыт — strict-fill выключен на тик"
        );
        return None;
    }
    match tokio::time::timeout(Duration::from_millis(BOOK_REPLY_TIMEOUT_MS), reply_rx).await {
        Ok(Ok(book)) => book,
        Ok(Err(_)) => {
            crate::tee_eprintln!(
                "[real_sim] book-coord уронил oneshot до ответа — strict-fill выключен на тик"
            );
            None
        }
        Err(_) => {
            crate::tee_eprintln!(
                "[real_sim] ожидание ответа book-coord > {BOOK_REPLY_TIMEOUT_MS}ms — strict-fill выключен на тик"
            );
            None
        }
    }
}

/// [`StrictBook`] из WS-снимка при непустых `book_bids`/`book_asks`.
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
        min_order_size: snapshot.min_order_size,
    })
}

/// Свежий снимок (`≤` [`WS_STRICT_BOOK_MAX_AGE_MS`]) + полные лестницы → book; иначе `None`.
async fn try_fresh_ws_strict_book(
    project_manager: &ProjectManager,
    asset_id: &str,
    now_ms: i64,
) -> Option<StrictBook> {
    let guard = project_manager.last_snapshot_by_asset_id.read().await;
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

        let absolute_deadline =
            tokio::time::Instant::now() + Duration::from_millis(BOOK_BATCH_MAX_MS);
        while batch.len() < LANE_FRAME_ROUTES.len() {
            let idle_deadline =
                tokio::time::Instant::now() + Duration::from_millis(BOOK_BATCH_IDLE_MS);
            let next_deadline = idle_deadline.min(absolute_deadline);
            match tokio::time::timeout_at(next_deadline, rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                Ok(None) | Err(_) => break, // канал закрыт ИЛИ idle/absolute истёк
            }
        }

        let mut by_asset: HashMap<String, Vec<oneshot::Sender<Option<StrictBook>>>> =
            HashMap::new();
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

/// Ответ `order_books` → [`StrictBook`] (уровни от худшего к лучшему в API → разворачиваем).
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

fn latest_version_path(currency: &str) -> Option<PathBuf> {
    let base = crate::path_config::xframes_path(currency);
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
    let resolution_path = version_path.join(format!("model_{interval}_1s_resolution_{side}.ubj"));

    // PnL-канал грузим только под общим тумблером [`ENABLE_PNL`];
    // иначе `None` → входы идут только через Resolution, если он включён.
    let (booster_pnl, calibration_pnl) = if ENABLE_PNL {
        (load_booster(&pnl_path), load_calibration(&pnl_path).ok())
    } else {
        (None, None)
    };
    // Resolution-канал грузим только под общим тумблером [`ENABLE_RESOLUTION`];
    // иначе `None` → hold-zone fallback не активируется (PnL-канал работает).
    let (booster_resolution, calibration_resolution) = if ENABLE_RESOLUTION {
        (
            load_booster(&resolution_path),
            load_calibration(&resolution_path).ok(),
        )
    } else {
        (None, None)
    };

    if booster_pnl.is_none() && booster_resolution.is_none() && !REDEEM_01 {
        return None;
    }
    Some(SideModels {
        booster_pnl,
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
