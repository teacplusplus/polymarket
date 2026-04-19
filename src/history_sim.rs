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

use crate::xframe::{XFrame, SIZE, Y_TRAIN_HORIZON_FRAMES, Y_TRAIN_TAKE_PROFIT_PP, Y_TRAIN_STOP_LOSS_PP};
use crate::xframe_dump::MarketXFramesDump;
use std::fs;
use std::path::{Path, PathBuf};
use xgb::{Booster, DMatrix};

/// Порог предсказания модели (0.0–1.0) для входа в позицию.
pub const SIM_BUY_THRESHOLD: f32 = 0.85;
/// Размер виртуальной позиции в USDC.
pub const POSITION_SIZE_USD: f64 = 10.0;

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

/// Накопленная статистика за версию.
#[derive(Debug, Default)]
struct SimStats {
    /// Число обработанных событий (файлов `.bin`) за версию.
    events: usize,
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

                println!(
                    "[sim] {tag}: модели pnl загружены \
                     | threshold={SIM_BUY_THRESHOLD} | position={POSITION_SIZE_USD}$ | fee_rate={POLYMARKET_CRYPTO_TAKER_FEE_RATE}"
                );

                let mut stats = SimStats::default();

                // Дампы с шагом 1s — совпадает со step pnl-модели.
                let step_path = interval_path.join("1s");
                if step_path.is_dir() {
                    for date_path in fs_sorted_dirs(&step_path)? {
                        if !date_path.is_dir() {
                            continue;
                        }
                        for file_path in fs_sorted_dirs(&date_path)? {
                            if file_path.extension().and_then(|ext| ext.to_str()) != Some("bin") {
                                continue;
                            }
                            match load_dump(&file_path) {
                                Ok(dump) => {
                                    simulate_event(&dump, &booster_up, &booster_down, &mut stats);
                                    stats.events += 1;
                                }
                                Err(err) => eprintln!("[sim] {}: {err}", file_path.display()),
                            }
                        }
                    }
                }

                print_stats(&tag, &stats);
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
fn simulate_event(dump: &MarketXFramesDump, booster_up: &Booster, booster_down: &Booster, stats: &mut SimStats) {
    let frames_up: Vec<&XFrame<SIZE>> = dump.frames_up.iter().filter(|f| f.stable).collect();
    let frames_down: Vec<&XFrame<SIZE>> = dump.frames_down.iter().filter(|f| f.stable).collect();

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
                if dump.up_won() { 1.0_f64 } else { 0.0_f64 },
                if dump.up_won() { 0.0_f64 } else { 1.0_f64 },
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
                let pnl = close_position(&pos, exit_price, &reason, frame_up, stats);
                stats.pnl_usd += pnl;
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
                let pnl = close_position(&pos, exit_price, &reason, frame_down, stats);
                stats.pnl_usd += pnl;
            } else {
                remaining_down.push(pos);
            }
        }
        positions_down = remaining_down;

        // ── Открытие новых позиций (не на последнем кадре) ───────────────────
        if !is_last {
            if let Some(pred) = predict_frame(booster_up, frame_up) {
                if pred >= SIM_BUY_THRESHOLD {
                    positions_up.push(open_position(frame_up, stats));
                }
            }
            if let Some(pred) = predict_frame(booster_down, frame_down) {
                if pred >= SIM_BUY_THRESHOLD {
                    positions_down.push(open_position(frame_down, stats));
                }
            }
        }
    }
}

// ─── Торговые операции с учётом комиссий ──────────────────────────────────────

/// Открывает виртуальную позицию за `POSITION_SIZE_USD`.
///
/// Цена исполнения определяется обходом ask-стакана (L1→L2→L3): если L1 не хватает
/// ликвидности — добираем с L2, затем L3. VWAP покупки = `POSITION_SIZE_USD / total_shares`.
/// Taker-комиссия вычитается из полученных шерсов:
/// `actual_shares = nominal_shares − nominal_shares × FEE_RATE × p × (1−p)`
fn open_position(frame: &XFrame<SIZE>, stats: &mut SimStats) -> OpenPosition {
    let (buy_price, nominal_shares) = book_fill_buy(frame);
    let buy_price = buy_price.clamp(0.001, 0.999);

    let fee_usdc = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy_price * (1.0 - buy_price);
    let fee_shares = fee_usdc / buy_price;
    let actual_shares = nominal_shares - fee_shares;

    stats.fees_paid += fee_usdc;

    OpenPosition {
        shares_held: actual_shares,
        // entry_prob берём из currency_implied_prob для TP/SL-слежения
        entry_prob: frame.currency_implied_prob.unwrap_or(buy_price),
        entry_cost: POSITION_SIZE_USD,
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
    stats: &mut SimStats,
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
            stats.fees_paid += fee_usdc;
            gross_usdc - fee_usdc
        }
    };

    let pnl = net_usdc - pos.entry_cost;

    stats.trades += 1;
    if pnl >= 0.0 { stats.wins += 1; } else { stats.losses += 1; }

    match reason {
        CloseReason::TakeProfit                => stats.tp_count += 1,
        CloseReason::StopLoss                  => stats.sl_count += 1,
        CloseReason::Resolution { won: true }  => stats.resolution_win += 1,
        CloseReason::Resolution { won: false } => stats.resolution_loss += 1,
        CloseReason::Timeout                   => stats.timeout_count += 1,
    }

    pnl
}

// ─── Обход стакана ────────────────────────────────────────────────────────────

/// Покупка `POSITION_SIZE_USD` по ask-стакану (L1→L2→L3).
///
/// Возвращает `(vwap_price, total_nominal_shares)`.
/// Если ликвидности на трёх уровнях не хватает — остаток добирается по `currency_implied_prob`.
fn book_fill_buy(frame: &XFrame<SIZE>) -> (f64, f64) {
    let levels = [
        (frame.book_ask_l1_price, frame.book_ask_l1_size),
        (frame.book_ask_l2_price, frame.book_ask_l2_size),
        (frame.book_ask_l3_price, frame.book_ask_l3_size),
    ];

    let mut remaining_usdc = POSITION_SIZE_USD;
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

    // Остаток при нехватке ликвидности — добираем по currency_implied_prob
    if remaining_usdc > 1e-9 {
        let fallback = frame.currency_implied_prob
            .unwrap_or(0.5)
            .clamp(0.001, 0.999);
        total_shares += remaining_usdc / fallback;
    }

    let vwap = if total_shares > 0.0 { POSITION_SIZE_USD / total_shares } else { 0.5 };
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

fn predict_frame(booster: &Booster, frame: &XFrame<SIZE>) -> Option<f32> {
    let features = frame.to_x_train();
    if features.len() != XFrame::<SIZE>::count_features() {
        return None;
    }
    let dmat = DMatrix::from_dense(&features, 1).ok()?;
    booster.predict(&dmat).ok()?.into_iter().next()
}

// ─── Вывод статистики ─────────────────────────────────────────────────────────

fn print_stats(tag: &str, stats: &SimStats) {
    if stats.trades == 0 {
        println!("[sim] {tag}: нет сделок ({} событий)", stats.events);
        return;
    }

    let win_rate = stats.wins as f64 / stats.trades as f64 * 100.0;
    let avg_pnl = stats.pnl_usd / stats.trades as f64;

    println!(
        "[sim] {tag} \
         | events={} (событий) trades={} (сделок) win={:.1}% (винрейт) \
         | pnl={:+.4}$ (итог) avg={:+.4}$/trade (среднее) fees={:.4}$ (комиссии)",
        stats.events, stats.trades, win_rate, stats.pnl_usd, avg_pnl, stats.fees_paid,
    );
    println!(
        "[sim]   TP={} (тейкпрофит) SL={} (стоплосс) Timeout={} (таймаут) \
         Res✓={} (резолюция выиграл) Res✗={} (резолюция проиграл) \
         | wins={} (побед) losses={} (потерь)",
        stats.tp_count, stats.sl_count, stats.timeout_count,
        stats.resolution_win, stats.resolution_loss,
        stats.wins, stats.losses,
    );
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

fn load_dump(path: &Path) -> anyhow::Result<MarketXFramesDump> {
    let bytes = fs::read(path)?;
    Ok(bincode::deserialize(&bytes)?)
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
