//! Режим исторической симуляции: загружает дампы [`crate::xframe_dump::MarketXFramesDump`],
//! синхронно проходит по парным кадрам UP/DOWN и виртуально торгует обоими токенами.
//!
//! # Механика Polymarket
//!
//! Каждый бинарный рынок имеет два токена: UP и DOWN.
//! `price_up + price_down ≈ 1.0` (арбитражное равновесие CLOB).
//! Победивший токен погашается за $1.00/шер, проигравший — $0.00 (сгорает).
//!
//! # Комиссии (категория Crypto, BTC Up/Down)
//!
//! ```text
//! fee_usdc = C × 0.072 × p × (1 − p)   // пик 1.8% при p=0.5
//! ```
//! * **Покупка** — комиссия списывается из получаемых шерсов:
//!   `actual_shares = (cost/p) × (1 − 0.072 × p × (1−p))`
//! * **Продажа** — комиссия вычитается из USDC:
//!   `net_usdc = shares × p × (1 − 0.072 × (1−p))`
//! * **Погашение** победившего токена — комиссии нет.
//!
//! # Торговая логика
//!
//! Один синхронный цикл по парным кадрам (UP[i], DOWN[i]).
//! Если модель выдаёт `prediction >= SIM_BUY_THRESHOLD` для токена — открывается позиция.
//! Позиция закрывается по TP/SL (те же пороги что в `calc_y_train_pnl`) или при окончании события.

use crate::tee_log::TEE_LOG;
use crate::train_mode::{
    collect_bin_paths, load_calibration, split_counts,
    Calibration, PNL_MAX_LAG, RESOLUTION_MAX_LAG, TEST_FRACTION, VAL_FRACTION,
};
use crate::xframe::{apply_side_symmetry, XFrame, SIZE, Y_TRAIN_HORIZON_FRAMES, Y_TRAIN_TAKE_PROFIT_PP, Y_TRAIN_STOP_LOSS_PP};
use crate::xframe_dump::MarketXFramesDump;
use crate::{tee_eprintln, tee_println};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use xgb::{Booster, DMatrix};

/// Порог сырого предсказания модели (0.0–1.0) для рассмотрения входа в позицию.
/// Дальнейшую селекцию делает Kelly (`f* > 0`), поэтому порог служит только
/// грубым префильтром, отсекающим заведомо шумовые сигналы.
pub const SIM_BUY_THRESHOLD: f32 = 0.6;

/// Стартовый виртуальный банкролл (USDC).
pub const INITIAL_BANKROLL: f64 = 1000.0;
/// Множитель Келли: `1.0` = full-Kelly, `0.5` = half-Kelly (меньше размер — меньше volatility).
pub const KELLY_MULTIPLIER: f64 = 1.0;
/// Максимальная доля банкролла на одну сделку.
pub const MAX_BET_FRACTION: f64 = 0.10;
/// Минимальный размер позиции в USDC (меньше — не торгуем).
pub const MIN_POSITION_USD: f64 = 0.1;

/// Коэффициент taker-комиссии Polymarket для категории **Crypto** (CLOB):
/// `fee_usdc = C × POLYMARKET_CRYPTO_TAKER_FEE_RATE × p × (1 − p)`, где C — число шерсов, p — цена.
/// См. [Polymarket: Fees](https://docs.polymarket.com/trading/fees).
pub const POLYMARKET_CRYPTO_TAKER_FEE_RATE: f64 = 0.072;

/// Порог «удержания до конца» в секундах: когда `event_remaining_ms` ≤
/// `HOLD_TO_END_THRESHOLD_SEC × 1000`, активируется зона удержания — TP и
/// Timeout перестают закрывать позицию (ждём резолюцию ради выплаты $1 без
/// fee). Основания для выхода в зоне:
/// * **SL** по ценовой дельте (`current_prob − entry_prob ≤ Y_TRAIN_STOP_LOSS_PP`) —
///   жёсткий предохранитель от catastrophic loss при переуверенной
///   resolution-модели;
/// * **EV-правило** (см. [`EV_EXIT_MARGIN`]): `EV_sell · (1 − MARGIN) > EV_hold` —
///   мягкий выход, когда рыночная продажа выгоднее ожидаемой резолюции.
pub const HOLD_TO_END_THRESHOLD_SEC: i64 = 45;

/// Коэффициент EMA для сглаживания `p_win` от resolution-модели в зоне
/// удержания: `p_ema = α · p_now + (1 − α) · p_prev`. Чем меньше α, тем
/// плавнее и менее чувствителен EV-exit к одиночному выбросу модели.
pub const EV_EXIT_P_WIN_EMA_ALPHA: f64 = 0.3;

/// Защитный зазор EV-exit: продаём только если `EV_sell · (1 − MARGIN) >
/// EV_hold`. Сглаживает границу равенства EV и компенсирует неучтённые
/// эффекты (шум модели, неполнота bid-стака, погрешность калибровки).
pub const EV_EXIT_MARGIN: f64 = 0.01;

/// Минимальный остаток времени до конца маркета (мс) для открытия НОВОЙ
/// позиции: `Y_TRAIN_HORIZON_FRAMES × step_sec × 1000` при симуляции на
/// 1s-моделях = 15 с. Ближе к резолюции TP/SL за горизонт почти не
/// играют — вход превращается в лотерею и тратит комиссию.
pub const MIN_ENTRY_REMAINING_MS: i64 = (Y_TRAIN_HORIZON_FRAMES as i64) * 1000;

// ─── Типы ─────────────────────────────────────────────────────────────────────

/// Открытая виртуальная позиция в одном токене.
#[derive(Debug, Clone)]
struct OpenPosition {
    /// Количество шерсов после вычета комиссии при покупке.
    shares_held: f64,
    /// Цена входа (prob) — для TP/SL-слежения.
    entry_prob: f64,
    /// USDC потраченные на покупку (= POSITION_SIZE_USD).
    entry_cost: f64,
    /// Сколько кадров позиция уже удерживается (для таймаута).
    frames_held: usize,
    /// EMA вероятности выигрыша от resolution-модели, используется для
    /// EV-exit в зоне удержания (см. [`EV_EXIT_P_WIN_EMA_ALPHA`]). `None`,
    /// пока позиция ни разу не попадала в зону удержания / пока модель
    /// resolution не вернула валидного предсказания.
    p_win_ema: Option<f64>,
}

/// Причина закрытия позиции.
#[derive(Debug, Clone, PartialEq)]
enum CloseReason {
    TakeProfit,
    StopLoss,
    /// Событие закончилось, токен погашён по итогу (1.0 или 0.0).
    Resolution { won: bool },
    /// Позиция удерживалась больше [`crate::xframe::Y_TRAIN_HORIZON_FRAMES`] кадров без TP/SL — боковик, выход по рынку.
    Timeout,
    /// EV-правило сработало в hold zone **с прибылью** (`EV_sell > entry_cost`):
    /// рыночный выход даёт больше USDC, чем вложили на вход. Рыночный выход
    /// по бид-стаку.
    EvExitProfit,
    /// EV-правило сработало в hold zone **с убытком** (`EV_sell ≤ entry_cost`):
    /// продажа сейчас выгоднее ожидания резолюции, но ниже цены входа.
    /// Срабатывает, когда модель быстрее рынка увидела негативный исход.
    EvExitLoss,
}

/// Статистика торговли по одной стороне (UP или DOWN).
#[derive(Debug, Default)]
struct SideStats {
    /// Общее число закрытых сделок (каждая открытая позиция — одна сделка).
    trades: usize,
    /// Число сделок с P&L ≥ 0.
    wins: usize,
    /// Число сделок с P&L < 0.
    losses: usize,
    /// Суммарный P&L в USDC по всем сделкам (уже за вычетом комиссий).
    pnl_usd: f64,
    /// Суммарные комиссии taker, уплаченные за все открытия и рыночные закрытия.
    fees_paid: f64,
    /// Число закрытий по Take Profit (delta >= `Y_TRAIN_TAKE_PROFIT_PP`).
    tp_count: usize,
    /// Число закрытий по Stop Loss (delta <= `Y_TRAIN_STOP_LOSS_PP`).
    sl_count: usize,
    /// Число погашений победившего токена при резолюции события (exit = 1.0, без fee).
    resolution_win: usize,
    /// Число сгораний проигравшего токена при резолюции события (exit = 0.0).
    resolution_loss: usize,
    /// Число выходов по таймауту: позиция удерживалась >= [`crate::xframe::Y_TRAIN_HORIZON_FRAMES`] кадров без TP/SL.
    timeout_count: usize,
    /// Число прибыльных EV-exit-ов (см. [`CloseReason::EvExitProfit`]).
    ev_exit_profit_count: usize,
    /// Число убыточных EV-exit-ов (см. [`CloseReason::EvExitLoss`]).
    ev_exit_loss_count: usize,
    /// Число пропущенных входов из-за приближения к резолюции (`event_remaining_ms < MIN_ENTRY_REMAINING_MS`).
    late_entry_skips: usize,
    /// Число пропущенных входов из-за Kelly f* ≤ 0 (нет edge).
    kelly_skips: usize,
    /// Число кадров, где raw >= threshold (для диагностики воронки).
    raw_above_threshold: usize,
    /// Суммарные значения для расчёта средних (диагностика).
    diag_sum_raw: f64,
    diag_sum_calibrated: f64,
    diag_sum_entry_prob: f64,
    diag_sum_kelly_f: f64,
}

/// Накопленная статистика за версию.
#[derive(Debug)]
struct SimStats {
    /// Текущий виртуальный банкролл (USDC).
    bankroll: f64,
    /// Пиковый банкролл (для расчёта drawdown).
    peak_bankroll: f64,
    /// Максимальная просадка в процентах: `(peak - trough) / peak × 100`.
    max_drawdown_pct: f64,
    /// Число обработанных событий (файлов `.bin`) за версию.
    events: usize,
    up: SideStats,
    down: SideStats,
}

impl SimStats {
    fn new() -> Self {
        Self {
            bankroll: INITIAL_BANKROLL,
            peak_bankroll: INITIAL_BANKROLL,
            max_drawdown_pct: 0.0,
            events: 0,
            up: SideStats::default(),
            down: SideStats::default(),
        }
    }

    fn total_trades(&self) -> usize { self.up.trades + self.down.trades }
    fn total_wins(&self) -> usize { self.up.wins + self.down.wins }
    fn total_losses(&self) -> usize { self.up.losses + self.down.losses }
    fn total_pnl(&self) -> f64 { self.up.pnl_usd + self.down.pnl_usd }
    fn total_fees(&self) -> f64 { self.up.fees_paid + self.down.fees_paid }
    fn total_kelly_skips(&self) -> usize { self.up.kelly_skips + self.down.kelly_skips }

    fn update_drawdown(&mut self) {
        if self.bankroll > self.peak_bankroll {
            self.peak_bankroll = self.bankroll;
        }
        if self.peak_bankroll > 0.0 {
            let drawdown_pct = (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100.0;
            if drawdown_pct > self.max_drawdown_pct {
                self.max_drawdown_pct = drawdown_pct;
            }
        }
    }
}

// ─── Точка входа ──────────────────────────────────────────────────────────────

pub fn run_sim_mode() -> anyhow::Result<()> {
    let xframes_root = Path::new("xframes");
    if !xframes_root.exists() {
        anyhow::bail!("Папка xframes/ не найдена — сначала соберите данные (STATUS=default)");
    }

    let log_path = xframes_root.join("last_history_sim.txt");
    {
        let file = File::create(&log_path)?;
        let mut guard = TEE_LOG.lock().expect("TEE_LOG poisoned");
        *guard = Some(BufWriter::new(file));
    }
    tee_println!("[sim] лог пишется в {}", log_path.display());

    for currency_path in fs_sorted_dirs(xframes_root)? {
        let currency = dir_name(&currency_path);

        for version_path in fs_sorted_dirs(&currency_path)? {
            let version = dir_name(&version_path);
            if version.parse::<usize>().is_err() {
                continue;
            }

            for interval in ["5m", "15m"] {
                let interval_path = version_path.join(interval);
                if !interval_path.is_dir() {
                    continue;
                }

                let model_up_path   = version_path.join(format!("model_{interval}_1s_pnl_up.ubj"));
                let model_down_path = version_path.join(format!("model_{interval}_1s_pnl_down.ubj"));
                let model_resolution_up_path   = version_path.join(format!("model_{interval}_1s_resolution_up.ubj"));
                let model_resolution_down_path = version_path.join(format!("model_{interval}_1s_resolution_down.ubj"));

                let tag = format!("{currency}/{version}/{interval}");

                let booster_up = match load_booster(&model_up_path) {
                    Some(b) => b,
                    None => {
                        tee_println!("[sim] {tag}: model pnl_up не найдена, пропуск");
                        continue;
                    }
                };
                let booster_down = match load_booster(&model_down_path) {
                    Some(b) => b,
                    None => {
                        tee_println!("[sim] {tag}: model pnl_down не найдена, пропуск");
                        continue;
                    }
                };

                let calibration_up = load_calibration(&model_up_path).ok();
                let calibration_down = load_calibration(&model_down_path).ok();

                // Resolution-модели (1s) используются только для EV-exit в зоне удержания.
                // Их отсутствие не блокирует симуляцию — в зоне удержания без них позиция
                // просто ждёт резолюции (TP/SL/Timeout в зоне всё равно отключены).
                let booster_resolution_up   = load_booster(&model_resolution_up_path);
                let booster_resolution_down = load_booster(&model_resolution_down_path);
                let calibration_resolution_up   = load_calibration(&model_resolution_up_path).ok();
                let calibration_resolution_down = load_calibration(&model_resolution_down_path).ok();

                let cal_info = |cal: &Option<Calibration>, label: &str| -> String {
                    match cal {
                        Some(c) => format!(
                            "{label}=✓(breakpoints={} | 0.7→{:.3} 0.8→{:.3} 0.9→{:.3})",
                            c.xs.len(),
                            c.apply(0.7), c.apply(0.8), c.apply(0.9),
                        ),
                        None => format!("{label}=✗"),
                    }
                };

                tee_println!(
                    "[sim] {tag}: модели pnl загружены | {} | {} \
                     | resolution: up={} down={} \
                     | hold_zone≤{HOLD_TO_END_THRESHOLD_SEC}s ev_margin={EV_EXIT_MARGIN} ema_α={EV_EXIT_P_WIN_EMA_ALPHA} min_entry_ms={MIN_ENTRY_REMAINING_MS} \
                     | threshold={SIM_BUY_THRESHOLD} | kelly={KELLY_MULTIPLIER} | max_bet={MAX_BET_FRACTION} \
                     | bankroll={INITIAL_BANKROLL}$ | fee_rate={POLYMARKET_CRYPTO_TAKER_FEE_RATE}",
                    cal_info(&calibration_up, "cal_up"),
                    cal_info(&calibration_down, "cal_down"),
                    if booster_resolution_up.is_some()   { "✓" } else { "✗" },
                    if booster_resolution_down.is_some() { "✓" } else { "✗" },
                );

                let mut sim_stats = SimStats::new();

                let step_path = interval_path.join("1s");
                let all_paths = collect_bin_paths(&step_path)?;
                let (train_count, val_count, test_count) = split_counts(all_paths.len());
                let test_paths = &all_paths[train_count + val_count..];

                tee_println!(
                    "[sim] {tag}: маркетов всего={} → сплит {train_count}/{val_count}/{test_count} (train/val/test), TEST_FRACTION={TEST_FRACTION}, VAL_FRACTION={VAL_FRACTION}",
                    all_paths.len(),
                );

                for file_path in test_paths {
                    match load_market_xframes(file_path) {
                        Ok(market_xframes) => {
                            simulate_event(
                                &market_xframes,
                                &booster_up, &booster_down,
                                calibration_up.as_ref(), calibration_down.as_ref(),
                                booster_resolution_up.as_ref(), booster_resolution_down.as_ref(),
                                calibration_resolution_up.as_ref(), calibration_resolution_down.as_ref(),
                                &mut sim_stats,
                            );
                            sim_stats.events += 1;
                        }
                        Err(err) => tee_eprintln!("[sim] {}: {err}", file_path.display()),
                    }
                }

                print_sim_stats(&tag, &sim_stats);
            }
        }
    }

    {
        let mut guard = TEE_LOG.lock().expect("TEE_LOG poisoned");
        if let Some(mut w) = guard.take() {
            let _ = w.flush();
        }
    }

    Ok(())
}

// ─── Симуляция события ────────────────────────────────────────────────────────

/// Один синхронный цикл по парным кадрам (UP[i], DOWN[i]).
///
/// UP и DOWN — зеркальные токены одного события: при 50/50 шансах
/// `prob_up ≈ 1 − prob_down`. Итерируем по `min(len_up, len_down)` стабильным кадрам.
#[allow(clippy::too_many_arguments)]
fn simulate_event(
    market_xframes: &MarketXFramesDump,
    booster_up: &Booster,
    booster_down: &Booster,
    calibration_up: Option<&Calibration>,
    calibration_down: Option<&Calibration>,
    booster_resolution_up: Option<&Booster>,
    booster_resolution_down: Option<&Booster>,
    calibration_resolution_up: Option<&Calibration>,
    calibration_resolution_down: Option<&Calibration>,
    sim_stats: &mut SimStats,
) {
    let frames_up: Vec<&XFrame<SIZE>> = market_xframes.frames_up.iter().filter(|f| f.stable).collect();
    let frames_down: Vec<&XFrame<SIZE>> = market_xframes.frames_down.iter().filter(|f| f.stable).collect();

    let len = frames_up.len().min(frames_down.len());
    if len == 0 {
        return;
    }

    let last_idx = len - 1;
    let mut positions_up: Vec<OpenPosition>   = Vec::new();
    let mut positions_down: Vec<OpenPosition> = Vec::new();

    for idx in 0..len {
        let frame_up = frames_up[idx];
        let frame_down = frames_down[idx];

        let (Some(prob_up), Some(prob_down)) =
            (frame_up.currency_implied_prob, frame_down.currency_implied_prob)
        else {
            continue;
        };

        let is_last = idx == last_idx;

        // ── Определяем цены выхода для последнего кадра ───────────────────────
        // Если событие завершилось (event_remaining_ms ≤ 0), токены погашаются
        // по двоичному исходу: победитель → 1.0, проигравший → 0.0.
        let (exit_prob_up, exit_prob_down, is_resolution) = if is_last {
            (
                if market_xframes.up_won() { 1.0_f64 } else { 0.0_f64 },
                if market_xframes.up_won() { 0.0_f64 } else { 1.0_f64 },
                true,
            )
        } else {
            (prob_up, prob_down, false)
        };

        manage_positions(
            &mut positions_up,
            frame_up,
            prob_up,
            exit_prob_up,
            is_last,
            is_resolution,
            booster_resolution_up,
            calibration_resolution_up,
            &mut sim_stats.up,
            &mut sim_stats.bankroll,
        );
        manage_positions(
            &mut positions_down,
            frame_down,
            prob_down,
            exit_prob_down,
            is_last,
            is_resolution,
            booster_resolution_down,
            calibration_resolution_down,
            &mut sim_stats.down,
            &mut sim_stats.bankroll,
        );

        if !is_last { sim_stats.update_drawdown(); }

        // ── Открытие новых позиций ───────────────────────────────────────────
        // Не входим в позицию на последнем кадре (там только закрытие) и слишком
        // близко к резолюции — за оставшиеся < MIN_ENTRY_REMAINING_MS TP/SL
        // за горизонт не успеют сработать, вход превращается в лотерею.
        if !is_last {
            try_open_position(
                frame_up, prob_up,
                booster_up, calibration_up,
                &mut positions_up,
                &mut sim_stats.up,
                sim_stats.bankroll,
            );
            try_open_position(
                frame_down, prob_down,
                booster_down, calibration_down,
                &mut positions_down,
                &mut sim_stats.down,
                sim_stats.bankroll,
            );
        }
    }
}

/// Решение об открытии новой позиции для одной стороны на одном кадре.
///
/// Сигнал входа берётся у **pnl-модели** (большая обучающая выборка,
/// стабильный AUC), сайзинг — через классический TP/SL Kelly
/// ([`kelly_gain_ratio`] / [`kelly_loss_ratio`]).
///
/// В hold zone TP по ценовой дельте отключён, зато доступны SL как
/// catastrophic-предохранитель и мягкий EV-exit (см. `manage_positions`).
/// TP/SL-Kelly здесь даёт **более консервативный** сайз, чем
/// резолюционная формула — это осознанный выбор: в переобученных/
/// нестационарных окнах мы не хотим раздувать позиции под ожидание
/// полной выплаты резолюции.
///
/// При `event_remaining_ms < MIN_ENTRY_REMAINING_MS` вход пропускается
/// (`late_entry_skips++`): за оставшееся время TP/SL физически не
/// успевают сработать, вход вырождается в лотерею.
fn try_open_position(
    frame: &XFrame<SIZE>,
    entry_prob: f64,
    booster_pnl: &Booster,
    calibration_pnl: Option<&Calibration>,
    positions: &mut Vec<OpenPosition>,
    stats: &mut SideStats,
    bankroll: f64,
) {
    if frame.event_remaining_ms > 0 && frame.event_remaining_ms < MIN_ENTRY_REMAINING_MS {
        stats.late_entry_skips += 1;
        return;
    }

    let kelly_gain = kelly_gain_ratio(entry_prob);
    let kelly_loss = kelly_loss_ratio(entry_prob);

    let Some(raw) = predict_frame(booster_pnl, frame, PNL_MAX_LAG) else { return };
    if raw < SIM_BUY_THRESHOLD {
        return;
    }

    let pred = calibration_pnl.map_or(raw, |c| c.apply(raw));
    let kelly_f = kelly_fraction(pred as f64, kelly_gain, kelly_loss);

    stats.raw_above_threshold += 1;
    stats.diag_sum_raw += raw as f64;
    stats.diag_sum_calibrated += pred as f64;
    stats.diag_sum_entry_prob += entry_prob;
    stats.diag_sum_kelly_f += kelly_f;

    let kelly_f_adj = kelly_f * KELLY_MULTIPLIER;
    if kelly_f_adj <= 0.0 {
        stats.kelly_skips += 1;
        return;
    }
    let size = kelly_f_adj.min(MAX_BET_FRACTION) * bankroll;
    if size < MIN_POSITION_USD {
        stats.kelly_skips += 1;
        return;
    }

    positions.push(open_position(frame, size, stats));
}

/// Общий lifecycle позиций одной стороны за один кадр: инкремент `frames_held`,
/// обновление EMA `p_win` (в зоне удержания), проверка TP/SL/Timeout/
/// Resolution/EvExit и закрытие подходящих позиций в `stats`. P&L закрытий
/// добавляется к `bankroll`.
///
/// При `event_remaining_ms ≤ HOLD_TO_END_THRESHOLD_SEC × 1000` включается
/// «зона удержания до конца»: TP и Timeout НЕ закрывают позицию, зато
/// доступны два выхода:
///
/// 1. **Hard SL** по ценовой дельте (`current_prob − entry_prob ≤
///    Y_TRAIN_STOP_LOSS_PP`) — предохранитель от сценария, когда
///    resolution-модель переуверенно держит `p_ema` высоким, а цена
///    уходит вниз. Проверяется первым.
/// 2. **EV-правило** (soft exit): продаём, если рыночный exit чистым USDC
///    (после fee) выгоднее ожидаемого удержания до резолюции:
///    ```text
///    EV_sell = book_fill_sell(shares) − taker_fee(shares, sell_vwap)
///    EV_hold = EMA(p_win_resolution) · shares · 1.0
///    sell_now  ⇔  EV_sell · (1 − EV_EXIT_MARGIN) > EV_hold
///    ```
///
/// Если resolution-модель не загружена — EV-exit недоступен, но hard SL
/// всё равно работает; в остальных случаях позиция ждёт резолюции.
#[allow(clippy::too_many_arguments)]
fn manage_positions(
    positions: &mut Vec<OpenPosition>,
    frame: &XFrame<SIZE>,
    current_prob: f64,
    exit_prob_last: f64,
    is_last: bool,
    is_resolution: bool,
    booster_resolution: Option<&Booster>,
    calibration_resolution: Option<&Calibration>,
    stats: &mut SideStats,
    bankroll: &mut f64,
) {
    for pos in positions.iter_mut() { pos.frames_held += 1; }

    let in_hold_zone = !is_last
        && frame.event_remaining_ms > 0
        && frame.event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000;

    // p_win от resolution-модели считаем один раз на кадр — все позиции
    // одной стороны делят один и тот же кадр. EMA обновляется уже
    // индивидуально для каждой позиции (у каждой своя история).
    let p_win_now: Option<f64> = if in_hold_zone && !positions.is_empty() {
        booster_resolution.and_then(|b| {
            predict_frame(b, frame, RESOLUTION_MAX_LAG).map(|raw| {
                calibration_resolution.map_or(raw, |c| c.apply(raw)) as f64
            })
        })
    } else {
        None
    };

    let mut remaining: Vec<OpenPosition> = Vec::new();
    for mut pos in positions.drain(..) {
        if in_hold_zone {
            if let Some(p) = p_win_now {
                pos.p_win_ema = Some(match pos.p_win_ema {
                    Some(prev) => {
                        EV_EXIT_P_WIN_EMA_ALPHA * p + (1.0 - EV_EXIT_P_WIN_EMA_ALPHA) * prev
                    }
                    None => p,
                });
            }
        }

        let close = if is_last {
            let reason = if is_resolution {
                CloseReason::Resolution { won: exit_prob_last > 0.5 }
            } else {
                CloseReason::Timeout
            };
            Some((exit_prob_last, reason))
        } else if in_hold_zone {
            // В зоне удержания TP и Timeout отключены, но остаются два выхода:
            //
            // 1. Hard SL по ценовой дельте — предохранитель от переуверенной
            //    resolution-модели. Проверяется первым, чтобы в catastrophic
            //    сценариях ограничить потерю на уровне SL, а не всего entry_cost.
            // 2. EV-правило — мягкий выход, когда рыночная продажа даёт больше
            //    USDC, чем ожидаемая выплата при резолюции.
            let delta = current_prob - pos.entry_prob;
            if delta <= Y_TRAIN_STOP_LOSS_PP {
                Some((current_prob, CloseReason::StopLoss))
            } else if let Some(p_ema) = pos.p_win_ema {
                let gross_usdc = book_fill_sell(frame, pos.shares_held);
                let sell_vwap = if pos.shares_held > 0.0 {
                    (gross_usdc / pos.shares_held).clamp(0.001, 0.999)
                } else {
                    current_prob.clamp(0.001, 0.999)
                };
                let fee = pos.shares_held
                    * POLYMARKET_CRYPTO_TAKER_FEE_RATE
                    * sell_vwap
                    * (1.0 - sell_vwap);
                let ev_sell = gross_usdc - fee;
                let ev_hold = p_ema * pos.shares_held;
                if ev_sell * (1.0 - EV_EXIT_MARGIN) > ev_hold {
                    let reason = if ev_sell > pos.entry_cost {
                        CloseReason::EvExitProfit
                    } else {
                        CloseReason::EvExitLoss
                    };
                    Some((current_prob, reason))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            let delta = current_prob - pos.entry_prob;
            if delta >= Y_TRAIN_TAKE_PROFIT_PP {
                Some((current_prob, CloseReason::TakeProfit))
            } else if delta <= Y_TRAIN_STOP_LOSS_PP {
                Some((current_prob, CloseReason::StopLoss))
            } else if pos.frames_held >= Y_TRAIN_HORIZON_FRAMES {
                Some((current_prob, CloseReason::Timeout))
            } else {
                None
            }
        };
        if let Some((exit_price, reason)) = close {
            let pnl = close_position(&pos, exit_price, &reason, frame, stats);
            *bankroll += pnl;
        } else {
            remaining.push(pos);
        }
    }
    *positions = remaining;
}


// ─── Kelly criterion ──────────────────────────────────────────────────────────

/// Ожидаемая доля выигрыша при срабатывании Take Profit.
///
/// Моделирует покупку $1 токена по `entry_prob`, продажу по `entry_prob + TP`,
/// с учётом taker-fee на обоих концах.
fn kelly_gain_ratio(entry_prob: f64) -> f64 {
    let sell_price = (entry_prob + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999);
    let net = net_round_trip(entry_prob, sell_price);
    (net - 1.0).max(1e-9)
}

/// Ожидаемая доля убытка при срабатывании Stop Loss.
///
/// Моделирует покупку $1 токена по `entry_prob`, продажу по `entry_prob + SL`,
/// с учётом taker-fee на обоих концах.
fn kelly_loss_ratio(entry_prob: f64) -> f64 {
    let sell_price = (entry_prob + Y_TRAIN_STOP_LOSS_PP).clamp(0.001, 0.999);
    let net = net_round_trip(entry_prob, sell_price);
    (1.0 - net).max(1e-9)
}

/// Чистый возврат на $1 при покупке по `buy` и продаже по `sell` (с fee на обоих концах).
fn net_round_trip(buy: f64, sell: f64) -> f64 {
    let nominal_shares = 1.0 / buy;
    let buy_fee = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy * (1.0 - buy);
    let actual_shares = nominal_shares - buy_fee / buy;

    let gross = actual_shares * sell;
    let sell_fee = actual_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell * (1.0 - sell);
    gross - sell_fee
}

/// Оптимальная доля банкролла по формуле Келли: `f* = p/l − q/g`.
///
/// - `p_win` — вероятность прибыльной сделки (предсказание модели).
/// - `gain` — доля выигрыша при TP ([`kelly_gain_ratio`]).
/// - `loss` — доля убытка при SL ([`kelly_loss_ratio`]).
///
/// Возвращает «сырую» долю (может быть > 1 при высоком edge).
fn kelly_fraction(p_win: f64, gain: f64, loss: f64) -> f64 {
    if gain <= 0.0 || loss <= 0.0 {
        return 0.0;
    }
    let q = 1.0 - p_win;
    p_win / loss - q / gain
}

// ─── Торговые операции с учётом комиссий ──────────────────────────────────────

/// Открывает виртуальную позицию за `position_size` USDC.
///
/// Цена исполнения определяется обходом ask-стакана (L1→L2→L3): если L1 не хватает
/// ликвидности — добираем с L2, затем L3. VWAP покупки = `position_size / total_shares`.
/// Taker-комиссия вычитается из полученных шерсов:
/// `actual_shares = nominal_shares − nominal_shares × FEE_RATE × p × (1−p)`
fn open_position(frame: &XFrame<SIZE>, position_size: f64, stats: &mut SideStats) -> OpenPosition {
    let (buy_price, nominal_shares) = book_fill_buy(frame, position_size);
    let buy_price = buy_price.clamp(0.001, 0.999);

    let fee_usdc = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy_price * (1.0 - buy_price);
    let fee_shares = fee_usdc / buy_price;
    let actual_shares = nominal_shares - fee_shares;

    stats.fees_paid += fee_usdc;

    OpenPosition {
        shares_held: actual_shares,
        entry_prob: frame.currency_implied_prob.unwrap_or(buy_price),
        entry_cost: position_size,
        frames_held: 0,
        p_win_ema: None,
    }
}

/// Закрывает позицию и возвращает P&L в USDC (может быть отрицательным).
///
/// * **TP / SL / Timeout** — рыночная продажа по bid-стакану (L1→L2→L3);
///   VWAP продажи = `gross_usdc / shares_held`; taker-fee вычитается из USDC:
///   `net = gross − shares × FEE_RATE × p_sell × (1−p_sell)`
/// * **Resolution won** — погашение победителя без комиссии: `net = shares × 1.0`
/// * **Resolution lost** — токен сгорел: `net = 0`
fn close_position(
    pos: &OpenPosition,
    exit_price: f64,
    reason: &CloseReason,
    frame: &XFrame<SIZE>,
    stats: &mut SideStats,
) -> f64 {
    let net_usdc = match reason {
        CloseReason::Resolution { won: true } => pos.shares_held,
        CloseReason::Resolution { won: false } => 0.0,
        CloseReason::TakeProfit
        | CloseReason::StopLoss
        | CloseReason::Timeout
        | CloseReason::EvExitProfit
        | CloseReason::EvExitLoss => {
            let gross_usdc = book_fill_sell(frame, pos.shares_held);
            let sell_price = if pos.shares_held > 0.0 {
                (gross_usdc / pos.shares_held).clamp(0.001, 0.999)
            } else {
                exit_price.clamp(0.001, 0.999)
            };
            let fee_usdc = pos.shares_held * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell_price * (1.0 - sell_price);
            stats.fees_paid += fee_usdc;
            gross_usdc - fee_usdc
        }
    };

    let pnl = net_usdc - pos.entry_cost;
    stats.pnl_usd += pnl;

    stats.trades += 1;
    if pnl >= 0.0 { stats.wins += 1; } else { stats.losses += 1; }

    match reason {
        CloseReason::TakeProfit                => stats.tp_count += 1,
        CloseReason::StopLoss                  => stats.sl_count += 1,
        CloseReason::Resolution { won: true }  => stats.resolution_win += 1,
        CloseReason::Resolution { won: false } => stats.resolution_loss += 1,
        CloseReason::Timeout                   => stats.timeout_count += 1,
        CloseReason::EvExitProfit              => stats.ev_exit_profit_count += 1,
        CloseReason::EvExitLoss                => stats.ev_exit_loss_count += 1,
    }

    pnl
}

// ─── Обход стакана ────────────────────────────────────────────────────────────

/// Покупка `position_size` USDC по ask-стакану (L1→L2→L3).
///
/// Возвращает `(vwap_price, total_nominal_shares)`.
/// Если ликвидности на трёх уровнях не хватает — остаток добирается по `currency_implied_prob`.
fn book_fill_buy(frame: &XFrame<SIZE>, position_size: f64) -> (f64, f64) {
    let levels = [
        (frame.book_ask_l1_price, frame.book_ask_l1_size),
        (frame.book_ask_l2_price, frame.book_ask_l2_size),
        (frame.book_ask_l3_price, frame.book_ask_l3_size),
    ];

    let mut remaining_usdc = position_size;
    let mut total_shares = 0.0_f64;

    for (price_opt, size_opt) in levels {
        let (Some(price), Some(size)) = (price_opt, size_opt) else { break };
        if price <= 0.0 || size <= 0.0 { break }

        let affordable = remaining_usdc / price;
        if affordable <= size {
            total_shares += affordable;
            remaining_usdc = 0.0;
            break;
        } else {
            total_shares += size;
            remaining_usdc -= size * price;
        }
    }

    if remaining_usdc > 1e-9 {
        let fallback = frame.currency_implied_prob
            .unwrap_or(0.5)
            .clamp(0.001, 0.999);
        total_shares += remaining_usdc / fallback;
    }

    let vwap = if total_shares > 0.0 { position_size / total_shares } else { 0.5 };
    (vwap, total_shares)
}

/// Продажа `shares_to_sell` по bid-стакану (L1→L2→L3).
///
/// Возвращает валовый USDC до вычета fee.
/// Если ликвидности не хватает — остаток продаётся по `currency_implied_prob`.
fn book_fill_sell(frame: &XFrame<SIZE>, shares_to_sell: f64) -> f64 {
    let levels = [
        (frame.book_bid_l1_price, frame.book_bid_l1_size),
        (frame.book_bid_l2_price, frame.book_bid_l2_size),
        (frame.book_bid_l3_price, frame.book_bid_l3_size),
    ];

    let mut remaining = shares_to_sell;
    let mut total_usdc = 0.0_f64;

    for (price_opt, size_opt) in levels {
        let (Some(price), Some(size)) = (price_opt, size_opt) else { break };
        if price <= 0.0 || size <= 0.0 { break }

        if remaining <= size {
            total_usdc += remaining * price;
            remaining = 0.0;
            break;
        } else {
            total_usdc += size * price;
            remaining -= size;
        }
    }

    // Остаток при нехватке ликвидности — продаём по currency_implied_prob
    if remaining > 1e-9 {
        let fallback = frame.currency_implied_prob
            .unwrap_or(0.0)
            .clamp(0.0, 0.999);
        total_usdc += remaining * fallback;
    }

    total_usdc
}

// ─── Предсказание ─────────────────────────────────────────────────────────────

fn predict_frame(booster: &Booster, frame: &XFrame<SIZE>, max_lag: Option<usize>) -> Option<f32> {
    let features = match max_lag {
        Some(n) => frame.to_x_train_n_with(n, apply_side_symmetry),
        None => frame.to_x_train_with(apply_side_symmetry),
    };
    let expected = match max_lag {
        Some(n) => XFrame::<SIZE>::count_features_n(n),
        None => XFrame::<SIZE>::count_features(),
    };
    if features.len() != expected {
        return None;
    }
    let dmat = DMatrix::from_dense(&features, 1).ok()?;
    booster.predict(&dmat).ok()?.into_iter().next()
}

// ─── Вывод статистики ─────────────────────────────────────────────────────────

fn print_side_stats(tag: &str, side_label: &str, s: &SideStats) {
    let n = s.raw_above_threshold.max(1) as f64;
    let diag = format!(
        "raw≥thr={} avg_raw={:.3} avg_cal={:.3} avg_entry={:.3} avg_kelly_f={:.4} kelly_skips={}",
        s.raw_above_threshold,
        s.diag_sum_raw / n,
        s.diag_sum_calibrated / n,
        s.diag_sum_entry_prob / n,
        s.diag_sum_kelly_f / n,
        s.kelly_skips,
    );
    tee_println!("[sim] {tag} [{side_label}]   {diag}");

    if s.trades == 0 {
        tee_println!("[sim] {tag} [{side_label}]: нет сделок");
        return;
    }
    let win_rate = s.wins as f64 / s.trades as f64 * 100.0;
    let avg_pnl = s.pnl_usd / s.trades as f64;
    tee_println!(
        "[sim] {tag} [{side_label}] \
         | trades={} win={:.1}% \
         | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
         | TP={} SL={} Timeout={} EvExit✓={} EvExit✗={} Res✓={} Res✗={} late_skips={}",
        s.trades, win_rate, s.pnl_usd, avg_pnl, s.fees_paid,
        s.tp_count, s.sl_count, s.timeout_count,
        s.ev_exit_profit_count, s.ev_exit_loss_count,
        s.resolution_win, s.resolution_loss, s.late_entry_skips,
    );
}

fn print_sim_stats(tag: &str, sim_stats: &SimStats) {
    let total_trades = sim_stats.total_trades();
    if total_trades == 0 {
        tee_println!("[sim] {tag}: нет сделок ({} событий, kelly_skips={})", sim_stats.events, sim_stats.total_kelly_skips());
        print_side_stats(tag, "UP",   &sim_stats.up);
        print_side_stats(tag, "DOWN", &sim_stats.down);
        return;
    }

    let total_pnl = sim_stats.total_pnl();
    let total_wins = sim_stats.total_wins();
    let total_fees = sim_stats.total_fees();
    let win_rate = total_wins as f64 / total_trades as f64 * 100.0;
    let avg_pnl = total_pnl / total_trades as f64;
    let roi_pct = (sim_stats.bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100.0;

    let total_losses = sim_stats.total_losses();
    tee_println!(
        "[sim] {tag} \
         | events={} trades={} win={:.1}% \
         | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
         | wins={total_wins} losses={total_losses}",
        sim_stats.events, total_trades, win_rate, total_pnl, avg_pnl, total_fees,
    );
    tee_println!(
        "[sim]   bankroll: {:.2}$ (start={INITIAL_BANKROLL}$) ROI={:+.2}% max_drawdown={:.2}%",
        sim_stats.bankroll, roi_pct, sim_stats.max_drawdown_pct,
    );

    print_side_stats(tag, "UP",   &sim_stats.up);
    print_side_stats(tag, "DOWN", &sim_stats.down);
}

// ─── Утилиты ──────────────────────────────────────────────────────────────────

fn load_booster(path: &Path) -> Option<Booster> {
    if !path.exists() {
        return None;
    }
    match Booster::load(path) {
        Ok(b) => Some(b),
        Err(err) => {
            tee_eprintln!("[sim] не удалось загрузить модель {}: {err}", path.display());
            None
        }
    }
}

fn load_market_xframes(path: &Path) -> anyhow::Result<MarketXFramesDump> {
    let bytes = fs::read(path)?;
    Ok(bincode::deserialize(&bytes)?)
}

fn fs_sorted_dirs(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut entries: Vec<std::path::PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .map(|entry| entry.path())
        .collect();
    entries.sort();
    Ok(entries)
}

fn dir_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}
