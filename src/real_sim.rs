//! Live-сим: логика как [`crate::history_sim`], кадры из [`ProjectManager`], HTTP-батч стаканов [`run_book_coordinator`], при расхождении WS↔HTTP — только закрытия.

use crate::account::SharedAccount;
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
/// Реэкспорт cap slippage для TP/strict ([`crate::history_sim`]).
pub use crate::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT;
use crate::history_sim::{
    BuyGate, HOLD_TO_END_THRESHOLD_SEC, INITIAL_BANKROLL, StrictBook, any_position_would_sell,
    buy_gate, compute_p_win_now, compute_pnl_inference, load_booster, manage_positions,
    try_open_position,
};
use crate::market_snapshot::MarketSnapshot;
use crate::project_manager::{LaneFrame, ProjectManager};
use crate::sim_stats::{SimStats, print_sim_stats};
use crate::train_mode::{Calibration, load_calibration};
use crate::util::current_timestamp_ms;
use crate::xframe::BookLevel;
use crate::xframe::{SIZE, XFrame};

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
use std::time::Duration;
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::time::MissedTickBehavior;
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
pub(crate) const WS_STRICT_BOOK_MAX_AGE_MS: i64 = 1_000;

/// Таймаут ответа координатора в [`fetch_http_strict_book`] (~3× HTTP).
const BOOK_REPLY_TIMEOUT_MS: u64 = BOOK_HTTP_TIMEOUT_MS * 3;

/// Лимит FIFO [`RealSimState::seen_market_ids`] на интервал.
const SEEN_MARKET_IDS_CAP: usize = 8;

/// Интервал heartbeat-печати [`print_sim_stats`] при отсутствии сделок (сек).
const STATS_HEARTBEAT_INTERVAL_SEC: u64 = 5 * 60;

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
    booster_pnl: Arc<Booster>,
    /// Калибровка PnL.
    calibration_pnl: Option<Calibration>,
    /// Модель resolution.
    booster_resolution: Option<Arc<Booster>>,
    /// Калибровка resolution.
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

/// Загрузка моделей, публикация [`RealSimState`] в [`crate::account::Account`], 4 воркера [`tick_once`]. `submit` — реальный CLOB vs виртуальный fill.
pub async fn run_real_sim(project_manager: Arc<ProjectManager>, submit: bool) -> Result<()> {
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
    let last_snapshot_by_asset_id = project_manager.last_snapshot_by_asset_id.clone();

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

    spawn_stats_snapshot(state.clone(), account.clone(), tag_prefix.clone());

    for (interval_kind, side) in LANE_FRAME_ROUTES {
        let label = interval_label(interval_kind);
        let side_lbl = side_label(side);
        let models = load_side_models(&version_path, label, side_lbl)
            .ok_or_else(|| anyhow!("не удалось загрузить pnl-модель {label}/{side_lbl}"))?;
        crate::tee_println!(
            "[real_sim] {tag_prefix}/{label}/{side_lbl}: pnl ✓  resolution={}",
            if models.booster_resolution.is_some() {
                "✓"
            } else {
                "✗"
            },
        );

        let (tx, rx) = mpsc::channel::<LaneFrame>(LANE_FRAME_CHANNEL_CAP);
        channels.write().await.insert((interval_kind, side), tx);

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
            submit,
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
    submit: bool,
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
                submit,
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

/// Периодическая печать [`print_sim_stats`] по интервалам + bankroll/dd ([`STATS_HEARTBEAT_INTERVAL_SEC`]).
fn spawn_stats_snapshot(
    state: Arc<RwLock<RealSimState>>,
    account: SharedAccount,
    tag_prefix: String,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(STATS_HEARTBEAT_INTERVAL_SEC));
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        tick.tick().await;
        loop {
            tick.tick().await;
            let state_guard = state.read().await;
            let bankroll_now = *account.bankroll.read().await;
            let max_drawdown_pct_now = *account.max_drawdown_pct.read().await;
            for kind in [XFrameIntervalKind::FiveMin, XFrameIntervalKind::FifteenMin] {
                let Some(stats) = state_guard.stats.get(&kind) else {
                    continue;
                };
                let tag = format!("{tag_prefix}/{} [heartbeat]", interval_label(kind));
                print_sim_stats(
                    &tag,
                    stats,
                    bankroll_now,
                    max_drawdown_pct_now,
                    true,
                    INITIAL_BANKROLL,
                );
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

/// Один кадр: WS/HTTP [`StrictBook`], [`manage_positions`], опционально [`try_open_position`].
#[allow(clippy::too_many_arguments)]
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
    submit: bool,
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
        let bankroll_guard = account.bankroll.read().await;
        let max_dd_guard = account.max_drawdown_pct.read().await;
        let positions_guard = account.positions.read().await;
        let recently_resolved_guard = account.recently_resolved_markets.read().await;

        let this_positions = positions_guard
            .get(&lane_key)
            .expect("Account.positions pre-populated by run_real_sim");
        let mut total_locked = 0.0;
        for v in positions_guard.values() {
            for p in v.iter() {
                total_locked += p.read().await.position_size;
            }
        }
        let available = (*bankroll_guard - total_locked).max(0.0);
        let dd_halt = match crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT {
            Some(threshold) => *max_dd_guard >= threshold,
            None => false,
        };
        let market_resolved = recently_resolved_guard.contains(market_id.as_str());
        (
            !this_positions.is_empty(),
            any_position_would_sell(this_positions, &frame, None).await,
            available,
            dd_halt,
            *max_dd_guard,
            market_resolved,
        )
    };

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
    // Submit: новые BUY только при wall-clock внутри `[event_start_ms, event_end_ms)` от Gamma.
    let now_wall_ms = current_timestamp_ms();
    let submit_market_window_open: bool = if submit {
        match (event_start_ms, event_end_ms) {
            (Some(start_ms), Some(end_ms)) => now_wall_ms >= start_ms && now_wall_ms < end_ms,
            _ => false,
        }
    } else {
        true
    };
    let may_open = !dd_halt_active && !market_already_resolved && buy_gate_proceed && submit_market_window_open;
    
    if buy_gate_proceed && dd_halt_active {
        crate::tee_eprintln!(
            "[real_sim] {tag}: halt by drawdown — новые позиции заблокированы (порог={:?}%, max_dd_pct={:.2}%), закрытия продолжаем",
            crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT,
            account_max_dd_pct
        );
    }
    
    let needs_http = needs_sell || may_open;

    let strict_book: Option<StrictBook> = if needs_http {
        match try_fresh_ws_strict_book(last_snapshot_by_asset_id, &asset_id, now_wall_ms).await {
            Some(book) => Some(book),
            None => fetch_http_strict_book(book_tx, &asset_id, tag).await,
        }
    } else {
        None
    };

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

    let mut sold = false;
    let mut bought = false;
    let effective_prob = crate::history_sim::effective_implied_prob(&frame, strict_book.as_ref())
        .unwrap_or(currency_implied_prob);
    {
        let mut state_guard = state.write().await;
        let mut bankroll_guard = account.bankroll.write().await;
        let mut peak_guard = account.peak_bankroll.write().await;
        let mut max_dd_guard = account.max_drawdown_pct.write().await;
        let mut last_prob_guard = account.last_prob.write().await;
        let mut positions_guard = account.positions.write().await;
        let mut pending_guard = account.pending_resolution.write().await;
        let recently_resolved_guard = account.recently_resolved_markets.read().await;

        last_prob_guard.insert(lane_key.clone(), effective_prob);

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
        let market_resolved_now = recently_resolved_guard.contains(market_id.as_str());
        if !market_already_resolved && market_resolved_now && may_open {
            crate::tee_eprintln!(
                "[real_sim] {tag}: market={market_id} резолвнулся между snapshot'ом и HTTP — отмена входа"
            );
        }
        drop(recently_resolved_guard);
        let may_open = may_open && !dd_halt_now && !market_resolved_now;

        if has_positions || may_open {
            let mut cross_lanes_locked = 0.0;
            for (k, v) in positions_guard.iter() {
                if k == &lane_key {
                    continue;
                }
                for p in v.iter() {
                    cross_lanes_locked += p.read().await.position_size;
                }
            }
            for (k, v) in pending_guard.iter() {
                if k == &lane_key {
                    continue;
                }
                for p in v.iter() {
                    cross_lanes_locked += p.read().await.position_size;
                }
            }

            let stats: &mut SimStats = state_guard
                .stats
                .get_mut(&interval_kind)
                .expect("stats map initialized for both intervals");
            let side_stats = match side {
                CurrencyUpDownOutcome::Up => &mut stats.up,
                CurrencyUpDownOutcome::Down => &mut stats.down,
            };

            let this_positions: &mut Vec<crate::history_sim::SharedOpenPosition> = positions_guard
                .get_mut(&lane_key)
                .expect("Account.positions pre-populated by run_real_sim");
            let this_pending: &mut Vec<crate::history_sim::SharedOpenPosition> = pending_guard
                .get_mut(&lane_key)
                .expect("Account.pending_resolution pre-populated by run_real_sim");

            if has_positions {
                sold = manage_positions(
                    this_positions,
                    this_pending,
                    &frame,
                    false,
                    p_win_now,
                    side_stats,
                    &mut bankroll_guard,
                    strict_book.as_ref(),
                    None,
                    submit,
                    account,
                )
                .await;
            }

            if may_open && !ws_lagging {
                let mut same_locked_post = 0.0;
                for p in this_positions.iter() {
                    same_locked_post += p.read().await.position_size;
                }
                for p in this_pending.iter() {
                    same_locked_post += p.read().await.position_size;
                }
                let available_bankroll_post =
                    (*bankroll_guard - cross_lanes_locked - same_locked_post).max(0.0);
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
                    pnl_top5_shap_at_open_precomputed,
                    submit,
                    account,
                )
                .await;
            }
        }

        drop(state_guard);

        let total_value: f64 = {
            let mut active = 0.0;
            for ((c, i, s), pos_vec) in positions_guard.iter() {
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
                for p in pos_vec.iter() {
                    active += p.read().await.shares_held * prob;
                }
            }
            let mut pending = 0.0;
            for v in pending_guard.values() {
                for p in v.iter() {
                    let g = p.read().await;
                    pending += g.shares_held * g.buy_price;
                }
            }
            active + pending
        };
        let equity = *bankroll_guard + total_value;
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
        print_sim_stats(
            tag,
            stats,
            bankroll_now,
            max_drawdown_pct_now,
            true,
            INITIAL_BANKROLL,
        );
    }

    *last_market_id = Some(market_id);
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

/// [`StrictBook`] из WS-снимка при непустых `book_bids`/`book_asks`; `min_order_size`: `None`.
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

/// Свежий снимок (`≤` [`WS_STRICT_BOOK_MAX_AGE_MS`]) + полные лестницы → book; иначе `None`.
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
            ws_bid[0],
            ws_bid[1],
            ws_bid[2],
            http_bid(0),
            http_bid(1),
            http_bid(2),
            ws_ask[0],
            ws_ask[1],
            ws_ask[2],
            http_ask(0),
            http_ask(1),
            http_ask(2),
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
    let resolution_path = version_path.join(format!("model_{interval}_1s_resolution_{side}.ubj"));

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
