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

use crate::train_mode::{load_calibration, Calibration, TEST_FRACTION, VAL_FRACTION};
use crate::xframe::{XFrame, SIZE, Y_TRAIN_HORIZON_FRAMES, Y_TRAIN_TAKE_PROFIT_PP, Y_TRAIN_STOP_LOSS_PP};
use crate::xframe_dump::MarketXFramesDump;
use std::fs;
use std::path::{Path, PathBuf};
use xgb::{Booster, DMatrix};

/// Порог предсказания модели (0.0–1.0) для входа в позицию.
pub const SIM_BUY_THRESHOLD: f32 = 0.70;

/// Стартовый виртуальный банкролл (USDC).
pub const INITIAL_BANKROLL: f64 = 1000.0;
/// Множитель Келли: 0.5 = half-Kelly (снижает волатильность при неточности модели).
pub const KELLY_MULTIPLIER: f64 = 0.5;
/// Максимальная доля банкролла на одну сделку.
pub const MAX_BET_FRACTION: f64 = 0.20;
/// Минимальный размер позиции в USDC (меньше — не торгуем).
pub const MIN_POSITION_USD: f64 = 1.0;

/// Коэффициент taker-комиссии Polymarket для категории **Crypto** (CLOB):
/// `fee_usdc = C × POLYMARKET_CRYPTO_TAKER_FEE_RATE × p × (1 − p)`, где C — число шерсов, p — цена.
/// См. [Polymarket: Fees](https://docs.polymarket.com/trading/fees).
pub const POLYMARKET_CRYPTO_TAKER_FEE_RATE: f64 = 0.072;

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

                let tag = format!("{currency}/{version}/{interval}");

                let booster_up = match load_booster(&model_up_path) {
                    Some(b) => b,
                    None => {
                        println!("[sim] {tag}: model pnl_up не найдена, пропуск");
                        continue;
                    }
                };
                let booster_down = match load_booster(&model_down_path) {
                    Some(b) => b,
                    None => {
                        println!("[sim] {tag}: model pnl_down не найдена, пропуск");
                        continue;
                    }
                };

                let calibration_up = load_calibration(&model_up_path).ok();
                let calibration_down = load_calibration(&model_down_path).ok();

                let cal_info = |cal: &Option<Calibration>, label: &str| -> String {
                    match cal {
                        Some(c) => format!(
                            "{label}=✓(w={:.3} b={:.3} | 0.7→{:.3} 0.8→{:.3} 0.9→{:.3})",
                            c.param_w, c.intercept,
                            c.apply(0.7), c.apply(0.8), c.apply(0.9),
                        ),
                        None => format!("{label}=✗"),
                    }
                };

                println!(
                    "[sim] {tag}: модели pnl загружены | {} | {} \
                     | threshold={SIM_BUY_THRESHOLD} | kelly={KELLY_MULTIPLIER} | max_bet={MAX_BET_FRACTION} \
                     | bankroll={INITIAL_BANKROLL}$ | fee_rate={POLYMARKET_CRYPTO_TAKER_FEE_RATE}",
                    cal_info(&calibration_up, "cal_up"),
                    cal_info(&calibration_down, "cal_down"),
                );

                let mut sim_stats = SimStats::new();

                let step_path = interval_path.join("1s");
                let all_paths = collect_bin_paths(&step_path)?;
                let test_paths = test_split_paths(&all_paths);

                println!(
                    "[sim] {tag}: маркетов всего={}, test={} (TEST_FRACTION={TEST_FRACTION}, VAL_FRACTION={VAL_FRACTION})",
                    all_paths.len(),
                    test_paths.len(),
                );

                for file_path in test_paths {
                    match load_market_xframes(file_path) {
                        Ok(market_xframes) => {
                            simulate_event(&market_xframes, &booster_up, &booster_down, calibration_up.as_ref(), calibration_down.as_ref(), &mut sim_stats);
                            sim_stats.events += 1;
                        }
                        Err(err) => eprintln!("[sim] {}: {err}", file_path.display()),
                    }
                }

                print_sim_stats(&tag, &sim_stats);
            }
        }
    }

    Ok(())
}

// ─── Симуляция события ────────────────────────────────────────────────────────

/// Один синхронный цикл по парным кадрам (UP[i], DOWN[i]).
///
/// UP и DOWN — зеркальные токены одного события: при 50/50 шансах
/// `prob_up ≈ 1 − prob_down`. Итерируем по `min(len_up, len_down)` стабильным кадрам.
fn simulate_event(
    market_xframes: &MarketXFramesDump,
    booster_up: &Booster,
    booster_down: &Booster,
    calibration_up: Option<&Calibration>,
    calibration_down: Option<&Calibration>,
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

        // ── Управление позициями UP ───────────────────────────────────────────
        for pos in positions_up.iter_mut() { pos.frames_held += 1; }
        let mut remaining_up: Vec<OpenPosition> = Vec::new();
        for pos in positions_up.drain(..) {
            let close = if is_last {
                let reason = if is_resolution {
                    CloseReason::Resolution { won: exit_prob_up > 0.5 }
                } else {
                    CloseReason::Timeout
                };
                Some((exit_prob_up, reason))
            } else {
                let delta = prob_up - pos.entry_prob;
                if delta >= Y_TRAIN_TAKE_PROFIT_PP {
                    Some((prob_up, CloseReason::TakeProfit))
                } else if delta <= Y_TRAIN_STOP_LOSS_PP {
                    Some((prob_up, CloseReason::StopLoss))
                } else if pos.frames_held >= Y_TRAIN_HORIZON_FRAMES {
                    Some((prob_up, CloseReason::Timeout))
                } else {
                    None
                }
            };
            if let Some((exit_price, reason)) = close {
                let pnl = close_position(&pos, exit_price, &reason, frame_up, &mut sim_stats.up);
                sim_stats.bankroll += pnl;
                sim_stats.update_drawdown();
            } else {
                remaining_up.push(pos);
            }
        }
        positions_up = remaining_up;

        // ── Управление позициями DOWN ─────────────────────────────────────────
        for pos in positions_down.iter_mut() { pos.frames_held += 1; }
        let mut remaining_down: Vec<OpenPosition> = Vec::new();
        for pos in positions_down.drain(..) {
            let close = if is_last {
                let reason = if is_resolution {
                    CloseReason::Resolution { won: exit_prob_down > 0.5 }
                } else {
                    CloseReason::Timeout
                };
                Some((exit_prob_down, reason))
            } else {
                let delta = prob_down - pos.entry_prob;
                if delta >= Y_TRAIN_TAKE_PROFIT_PP {
                    Some((prob_down, CloseReason::TakeProfit))
                } else if delta <= Y_TRAIN_STOP_LOSS_PP {
                    Some((prob_down, CloseReason::StopLoss))
                } else if pos.frames_held >= Y_TRAIN_HORIZON_FRAMES {
                    Some((prob_down, CloseReason::Timeout))
                } else {
                    None
                }
            };
            if let Some((exit_price, reason)) = close {
                let pnl = close_position(&pos, exit_price, &reason, frame_down, &mut sim_stats.down);
                sim_stats.bankroll += pnl;
                sim_stats.update_drawdown();
            } else {
                remaining_down.push(pos);
            }
        }
        positions_down = remaining_down;

        // ── Открытие новых позиций (не на последнем кадре) ───────────────────
        if !is_last {
            if let Some(raw) = predict_frame(booster_up, frame_up, None) {
                if raw >= SIM_BUY_THRESHOLD {
                    let pred = calibration_up.map_or(raw, |c| c.apply(raw));
                    let g = kelly_gain_ratio(prob_up);
                    let l = kelly_loss_ratio(prob_up);
                    let f = kelly_fraction(pred as f64, g, l);

                    sim_stats.up.raw_above_threshold += 1;
                    sim_stats.up.diag_sum_raw += raw as f64;
                    sim_stats.up.diag_sum_calibrated += pred as f64;
                    sim_stats.up.diag_sum_entry_prob += prob_up;
                    sim_stats.up.diag_sum_kelly_f += f;

                    let size = kelly_position_size(pred, prob_up, sim_stats.bankroll);
                    if size > 0.0 {
                        positions_up.push(open_position(frame_up, size, &mut sim_stats.up));
                    } else {
                        sim_stats.up.kelly_skips += 1;
                    }
                }
            }
            if let Some(raw) = predict_frame(booster_down, frame_down, None) {
                if raw >= SIM_BUY_THRESHOLD {
                    let pred = calibration_down.map_or(raw, |c| c.apply(raw));
                    let g = kelly_gain_ratio(prob_down);
                    let l = kelly_loss_ratio(prob_down);
                    let f = kelly_fraction(pred as f64, g, l);

                    sim_stats.down.raw_above_threshold += 1;
                    sim_stats.down.diag_sum_raw += raw as f64;
                    sim_stats.down.diag_sum_calibrated += pred as f64;
                    sim_stats.down.diag_sum_entry_prob += prob_down;
                    sim_stats.down.diag_sum_kelly_f += f;

                    let size = kelly_position_size(pred, prob_down, sim_stats.bankroll);
                    if size > 0.0 {
                        positions_down.push(open_position(frame_down, size, &mut sim_stats.down));
                    } else {
                        sim_stats.down.kelly_skips += 1;
                    }
                }
            }
        }
    }
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

/// Вычисляет размер позиции (USDC) по half-Kelly с учётом банкролла.
///
/// Возвращает `0.0` если edge отсутствует или позиция меньше [`MIN_POSITION_USD`].
fn kelly_position_size(prediction: f32, entry_prob: f64, bankroll: f64) -> f64 {
    let g = kelly_gain_ratio(entry_prob);
    let l = kelly_loss_ratio(entry_prob);
    let f = kelly_fraction(prediction as f64, g, l);
    let f_adj = f * KELLY_MULTIPLIER;
    if f_adj <= 0.0 {
        return 0.0;
    }
    let size = f_adj.min(MAX_BET_FRACTION) * bankroll;
    if size < MIN_POSITION_USD { 0.0 } else { size }
}

// ─── Торговые операции с учётом комиссий ──────────────────────────────────────

/// Открывает виртуальную позицию за `position_size` USDC.
///
/// Цена исполнения определяется обходом ask-стакана (L1→L2→L3): если L1 не хватает
/// ликвидности — добираем с L2, затем L3. VWAP покупки = `position_size / total_shares`.
/// Taker-комиссия вычитается из полученных шерсов:
/// `actual_shares = nominal_shares − nominal_shares × FEE_RATE × p × (1−p)`
fn open_position(frame: &XFrame<SIZE>, position_size: f64, side: &mut SideStats) -> OpenPosition {
    let (buy_price, nominal_shares) = book_fill_buy(frame, position_size);
    let buy_price = buy_price.clamp(0.001, 0.999);

    let fee_usdc = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy_price * (1.0 - buy_price);
    let fee_shares = fee_usdc / buy_price;
    let actual_shares = nominal_shares - fee_shares;

    side.fees_paid += fee_usdc;

    OpenPosition {
        shares_held: actual_shares,
        entry_prob: frame.currency_implied_prob.unwrap_or(buy_price),
        entry_cost: position_size,
        frames_held: 0,
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
    side: &mut SideStats,
) -> f64 {
    let net_usdc = match reason {
        CloseReason::Resolution { won: true } => pos.shares_held,
        CloseReason::Resolution { won: false } => 0.0,
        CloseReason::TakeProfit | CloseReason::StopLoss | CloseReason::Timeout => {
            let gross_usdc = book_fill_sell(frame, pos.shares_held);
            let sell_price = if pos.shares_held > 0.0 {
                (gross_usdc / pos.shares_held).clamp(0.001, 0.999)
            } else {
                exit_price.clamp(0.001, 0.999)
            };
            let fee_usdc = pos.shares_held * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell_price * (1.0 - sell_price);
            side.fees_paid += fee_usdc;
            gross_usdc - fee_usdc
        }
    };

    let pnl = net_usdc - pos.entry_cost;
    side.pnl_usd += pnl;

    side.trades += 1;
    if pnl >= 0.0 { side.wins += 1; } else { side.losses += 1; }

    match reason {
        CloseReason::TakeProfit                => side.tp_count += 1,
        CloseReason::StopLoss                  => side.sl_count += 1,
        CloseReason::Resolution { won: true }  => side.resolution_win += 1,
        CloseReason::Resolution { won: false } => side.resolution_loss += 1,
        CloseReason::Timeout                   => side.timeout_count += 1,
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
        Some(n) => frame.to_x_train_n(n),
        None => frame.to_x_train(),
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
        "raw≥thr={} avg_raw={:.3} avg_cal={:.3} avg_entry={:.3} avg_kelly_f={:.4}",
        s.raw_above_threshold,
        s.diag_sum_raw / n,
        s.diag_sum_calibrated / n,
        s.diag_sum_entry_prob / n,
        s.diag_sum_kelly_f / n,
    );

    if s.trades == 0 {
        println!("[sim] {tag} [{side_label}]: нет сделок (kelly_skips={}) | {diag}", s.kelly_skips);
        return;
    }
    let win_rate = s.wins as f64 / s.trades as f64 * 100.0;
    let avg_pnl = s.pnl_usd / s.trades as f64;
    println!(
        "[sim] {tag} [{side_label}] \
         | trades={} win={:.1}% \
         | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
         | TP={} SL={} Timeout={} Res✓={} Res✗={} kelly_skips={}",
        s.trades, win_rate, s.pnl_usd, avg_pnl, s.fees_paid,
        s.tp_count, s.sl_count, s.timeout_count,
        s.resolution_win, s.resolution_loss, s.kelly_skips,
    );
    println!("[sim] {tag} [{side_label}]   {diag}");
}

fn print_sim_stats(tag: &str, sim_stats: &SimStats) {
    let total_trades = sim_stats.total_trades();
    if total_trades == 0 {
        println!("[sim] {tag}: нет сделок ({} событий, kelly_skips={})", sim_stats.events, sim_stats.total_kelly_skips());
        return;
    }

    let total_pnl = sim_stats.total_pnl();
    let total_wins = sim_stats.total_wins();
    let total_fees = sim_stats.total_fees();
    let win_rate = total_wins as f64 / total_trades as f64 * 100.0;
    let avg_pnl = total_pnl / total_trades as f64;
    let roi_pct = (sim_stats.bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100.0;

    let total_losses = sim_stats.total_losses();
    println!(
        "[sim] {tag} \
         | events={} trades={} win={:.1}% \
         | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
         | wins={total_wins} losses={total_losses}",
        sim_stats.events, total_trades, win_rate, total_pnl, avg_pnl, total_fees,
    );
    println!(
        "[sim]   bankroll: {:.2}$ (start={INITIAL_BANKROLL}$) ROI={:+.2}% max_drawdown={:.2}%",
        sim_stats.bankroll, roi_pct, sim_stats.max_drawdown_pct,
    );
    print_side_stats(tag, "UP", &sim_stats.up);
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
            eprintln!("[sim] не удалось загрузить модель {}: {err}", path.display());
            None
        }
    }
}

fn load_market_xframes(path: &Path) -> anyhow::Result<MarketXFramesDump> {
    let bytes = fs::read(path)?;
    Ok(bincode::deserialize(&bytes)?)
}

/// Собирает все `.bin` файлы из `step_path/{date}/` в отсортированном порядке
/// (идентично `load_dumps` в `train_mode`).
fn collect_bin_paths(step_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if !step_path.is_dir() {
        return Ok(paths);
    }
    for date_path in fs_sorted_dirs(step_path)? {
        if !date_path.is_dir() {
            continue;
        }
        for file_path in fs_sorted_dirs(&date_path)? {
            if file_path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
                paths.push(file_path);
            }
        }
    }
    Ok(paths)
}

/// Возвращает test-срез путей — тот же сплит, что и в `train_and_save`:
/// последние `ceil(n * TEST_FRACTION)` маркетов.
fn test_split_paths(all_paths: &[PathBuf]) -> &[PathBuf] {
    let n = all_paths.len();
    let test_count = ((n as f64) * TEST_FRACTION).ceil() as usize;
    let val_count = ((n as f64) * VAL_FRACTION).ceil() as usize;
    let train_count = n.saturating_sub(test_count + val_count);
    let start = train_count + val_count.min(n.saturating_sub(train_count));
    &all_paths[start..]
}

fn fs_sorted_dirs(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
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
