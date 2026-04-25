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
/// 1s-моделях = 15 с. Ближе к резолюции TP/SL за горизонт физически не
/// успевают сработать — вход вырождается в лотерею с уплатой taker-fee.
/// Используется в [`buy_gate`] как ранний отказ под именем `LateEntry`.
pub const MIN_ENTRY_REMAINING_MS: i64 = 15 * 1000;

// ─── Типы ─────────────────────────────────────────────────────────────────────

/// Реальный срез стакана для **строгого** исполнения без fallback-ов.
///
/// В `history_sim` исполнение идёт по трём уровням WS-кадра (`frame.book_*_l{1..3}_*`)
/// с добивкой остатка через `currency_implied_prob` — это валидная идеализация
/// для backtest-а, но в живой торговле ([`crate::real_sim`]) она запрещена:
/// там надо опираться на фактический HTTP-стакан Polymarket CLOB без
/// предположений об отсутствующих уровнях.
///
/// Поэтому [`try_open_position`] / [`manage_positions`] принимают
/// `Option<&StrictBook>`:
/// * `None`  → поведение `history_sim` (WS-кадр + fallback).
/// * `Some(book)` → строгое исполнение:
///   - **buy**: если суммарной ликвидности на `asks` не хватает на
///     `position_size` — позиция не открывается ([`book_fill_buy_strict`]
///     возвращает `None`).
///   - **sell**: если суммарной ликвидности на `bids` не хватает на
///     `shares_held` — продажа откладывается, позиция остаётся открытой
///     до следующего тика ([`book_fill_sell_strict`] возвращает `None`).
///
/// Уровни внутри `Vec`-ов ожидаются **от лучшего к худшему** (для asks — по
/// возрастанию цены, для bids — по убыванию). Ответственность вызывающего.
///
/// `StrictBook` сам **владеет** двумя лестницами (`Vec<BookLevel>`), а не
/// одалживает их — это убирает лайфтайм из сигнатур `book_fill_*_strict` /
/// `manage_positions` / `try_open_position` и снимает нужду в промежуточном
/// кортеже `(Vec, Vec)` на вызывающей стороне. Аллокация та же (два `Vec`
/// раз на кадр), но форма данных совпадает с [`crate::real_sim`]-снимком
/// HTTP-стакана без лишней обёртки.
#[derive(Debug, Clone, Default)]
pub(crate) struct StrictBook {
    /// Уровни спроса (bids), лучший bid = первый.
    pub(crate) bids: Vec<BookLevel>,
    /// Уровни предложения (asks), лучший ask = первый.
    pub(crate) asks: Vec<BookLevel>,
}

/// Один уровень стакана: цена в probability-шкале `[0..1]` и размер в шерсах.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BookLevel {
    /// Цена уровня (probability, `0..1`).
    pub(crate) price: f64,
    /// Размер уровня в шерсах.
    pub(crate) size: f64,
}

/// Строгая покупка `position_size` USDC по `asks` [`StrictBook`] без fallback.
///
/// Возвращает `Some((vwap, total_shares))` если удалось полностью заполнить
/// `position_size` за счёт фактических уровней стакана. Если суммарной
/// ликвидности не хватает (или `asks` пуст) — возвращает `None`: позиция
/// не должна открываться.
pub(crate) fn book_fill_buy_strict(
    book: &StrictBook,
    position_size: f64,
) -> Option<(f64, f64)> {
    if position_size <= 0.0 {
        return None;
    }
    let mut remaining_usdc = position_size;
    let mut total_shares = 0.0_f64;
    for level in &book.asks {
        if level.price <= 0.0 || level.size <= 0.0 {
            continue;
        }
        let affordable = remaining_usdc / level.price;
        if affordable <= level.size {
            total_shares += affordable;
            remaining_usdc = 0.0;
            break;
        } else {
            total_shares += level.size;
            remaining_usdc -= level.size * level.price;
        }
    }
    if remaining_usdc > 1e-9 || total_shares <= 0.0 {
        return None;
    }
    let vwap = position_size / total_shares;
    Some((vwap, total_shares))
}

/// Строгая продажа `shares_to_sell` по `bids` [`StrictBook`] без fallback.
///
/// Возвращает `Some(gross_usdc)` (до вычета taker-fee), если ликвидности
/// достаточно, иначе `None`.
pub(crate) fn book_fill_sell_strict(book: &StrictBook, shares_to_sell: f64) -> Option<f64> {
    if shares_to_sell <= 0.0 {
        return Some(0.0);
    }
    let mut remaining = shares_to_sell;
    let mut total_usdc = 0.0_f64;
    for level in &book.bids {
        if level.price <= 0.0 || level.size <= 0.0 {
            continue;
        }
        if remaining <= level.size {
            total_usdc += remaining * level.price;
            remaining = 0.0;
            break;
        } else {
            total_usdc += level.size * level.price;
            remaining -= level.size;
        }
    }
    if remaining > 1e-9 {
        return None;
    }
    Some(total_usdc)
}

/// Открытая виртуальная позиция в одном токене.
#[derive(Debug, Clone)]
pub struct OpenPosition {
    /// Количество шерсов после вычета комиссии при покупке.
    pub(crate) shares_held: f64,
    /// Цена входа (prob) — для TP/SL-слежения.
    pub(crate) entry_prob: f64,
    /// USDC потраченные на покупку (= POSITION_SIZE_USD).
    pub(crate) entry_cost: f64,
    /// Сколько кадров позиция уже удерживается (для таймаута).
    pub(crate) frames_held: usize,
    /// EMA вероятности выигрыша от resolution-модели, используется для
    /// EV-exit в зоне удержания (см. [`EV_EXIT_P_WIN_EMA_ALPHA`]). `None`,
    /// пока позиция ни разу не попадала в зону удержания / пока модель
    /// resolution не вернула валидного предсказания.
    pub(crate) p_win_ema: Option<f64>,
}

/// Причина закрытия позиции.
#[derive(Debug, Clone, PartialEq)]
pub enum CloseReason {
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
pub struct SideStats {
    /// Общее число закрытых сделок (каждая открытая позиция — одна сделка).
    pub(crate) trades: usize,
    /// Число сделок с P&L ≥ 0.
    pub(crate) wins: usize,
    /// Число сделок с P&L < 0.
    pub(crate) losses: usize,
    /// Суммарный P&L в USDC по всем сделкам (уже за вычетом комиссий).
    pub(crate) pnl_usd: f64,
    /// Суммарные комиссии taker, уплаченные за все открытия и рыночные закрытия.
    pub(crate) fees_paid: f64,
    /// Число закрытий по Take Profit (delta >= `Y_TRAIN_TAKE_PROFIT_PP`).
    pub(crate) tp_count: usize,
    /// Число закрытий по Stop Loss (delta <= `Y_TRAIN_STOP_LOSS_PP`).
    pub(crate) sl_count: usize,
    /// Число погашений победившего токена при резолюции события (exit = 1.0, без fee).
    pub(crate) resolution_win: usize,
    /// Число сгораний проигравшего токена при резолюции события (exit = 0.0).
    pub(crate) resolution_loss: usize,
    /// Число выходов по таймауту: позиция удерживалась >= [`crate::xframe::Y_TRAIN_HORIZON_FRAMES`] кадров без TP/SL.
    pub(crate) timeout_count: usize,
    /// Число прибыльных EV-exit-ов (см. [`CloseReason::EvExitProfit`]).
    pub(crate) ev_exit_profit_count: usize,
    /// Число убыточных EV-exit-ов (см. [`CloseReason::EvExitLoss`]).
    pub(crate) ev_exit_loss_count: usize,
    /// Число пропущенных входов из-за приближения к резолюции
    /// (`event_remaining_ms < MIN_ENTRY_REMAINING_MS`, включая `≤ 0`).
    pub(crate) late_entry_skips: usize,
    /// Число пропущенных входов из-за «нестабильного» кадра
    /// (`!frame.stable` — поздний WS-коннект, нет `event_start_ms`).
    /// Закрытие уже открытых позиций такие кадры **не** блокируют —
    /// время идёт, и `manage_positions` отрабатывает TP/SL/Resolution
    /// как обычно. Только новые входы пропускаются.
    pub(crate) unstable_skips: usize,
    /// Число пропущенных входов из-за Kelly f* ≤ 0 (нет edge).
    pub(crate) kelly_skips: usize,
    /// Число пропущенных входов в **strict**-режиме ([`crate::real_sim`]):
    /// сигнал на вход был (raw ≥ threshold, Kelly f* > 0, `size ≥ MIN_POSITION_USD`),
    /// но фактической ликвидности в `asks` HTTP-стакана не хватило, чтобы
    /// полностью заполнить `size` USDC — покупку пропустили. В `history_sim`
    /// (без strict) остаётся `0`.
    pub(crate) kelly_strict_buy_skips: usize,
    /// Число отложенных закрытий в **strict**-режиме: решение закрыть позицию
    /// (TP/SL/Timeout/EV) принято, но ликвидности в `bids` HTTP-стакана не
    /// хватило на `shares_held` — позиция осталась открытой до следующего
    /// тика (или до `Resolution`, если не успеем продать). В `history_sim`
    /// (без strict) остаётся `0`.
    pub(crate) kelly_strict_sell_skips: usize,
    /// Число кадров, где raw >= threshold (для диагностики воронки).
    pub(crate) raw_above_threshold: usize,
    /// Сумма сырых (некалиброванных) предсказаний pnl-модели по кадрам,
    /// прошедшим `raw ≥ SIM_BUY_THRESHOLD`. Делением на [`Self::raw_above_threshold`]
    /// получаем средний raw-скор претендентов на вход (диагностика воронки).
    pub(crate) diag_sum_raw: f64,
    /// Сумма калиброванных предсказаний pnl-модели (`calibration.apply(raw)` или
    /// `raw`, если калибровка отсутствует) по тем же кадрам-претендентам. Среднее
    /// показывает, куда реально «сдвигает» raw-скор isotonic-калибровка.
    pub(crate) diag_sum_calibrated: f64,
    /// Сумма `entry_prob` (цена входа = ask L1 в probability-шкале) по кадрам-претендентам.
    /// Среднее — типичная цена, по которой срабатывает фильтр на покупку.
    pub(crate) diag_sum_entry_prob: f64,
    /// Сумма «сырого» Kelly f* (до применения [`KELLY_MULTIPLIER`]) по кадрам-претендентам,
    /// посчитанного как `kelly_fraction(pred, kelly_gain_ratio, kelly_loss_ratio)`.
    /// Среднее отражает, насколько «жирный» edge обычно видит модель.
    pub(crate) diag_sum_kelly_f: f64,
}

/// Накопленная статистика за версию.
#[derive(Debug)]
pub struct SimStats {
    /// Текущий виртуальный банкролл (USDC).
    pub(crate) bankroll: f64,
    /// Пиковый банкролл (для расчёта drawdown).
    pub(crate) peak_bankroll: f64,
    /// Максимальная просадка в процентах: `(peak - trough) / peak × 100`.
    pub(crate) max_drawdown_pct: f64,
    /// Число обработанных событий (файлов `.bin`) за версию.
    pub(crate) events: usize,
    /// Статистика по стороне UP (ставка на «цена вырастет»). Агрегируется
    /// по всем сделкам, открытым на UP-токене в рамках текущей версии.
    pub(crate) up: SideStats,
    /// Статистика по стороне DOWN (ставка на «цена упадёт»). Агрегируется
    /// по всем сделкам, открытым на DOWN-токене в рамках текущей версии.
    pub(crate) down: SideStats,
}

impl SimStats {
    pub(crate) fn new() -> Self {
        Self {
            bankroll: INITIAL_BANKROLL,
            peak_bankroll: INITIAL_BANKROLL,
            max_drawdown_pct: 0.0,
            events: 0,
            up: SideStats::default(),
            down: SideStats::default(),
        }
    }

    pub(crate) fn total_trades(&self) -> usize { self.up.trades + self.down.trades }
    pub(crate) fn total_wins(&self) -> usize { self.up.wins + self.down.wins }
    pub(crate) fn total_losses(&self) -> usize { self.up.losses + self.down.losses }
    pub(crate) fn total_pnl(&self) -> f64 { self.up.pnl_usd + self.down.pnl_usd }
    pub(crate) fn total_fees(&self) -> f64 { self.up.fees_paid + self.down.fees_paid }
    pub(crate) fn total_kelly_skips(&self) -> usize { self.up.kelly_skips + self.down.kelly_skips }
    pub(crate) fn total_kelly_strict_buy_skips(&self) -> usize {
        self.up.kelly_strict_buy_skips + self.down.kelly_strict_buy_skips
    }
    pub(crate) fn total_kelly_strict_sell_skips(&self) -> usize {
        self.up.kelly_strict_sell_skips + self.down.kelly_strict_sell_skips
    }

    /// Обновляет `peak_bankroll` и `max_drawdown_pct` по **equity**
    /// (mark-to-market): `equity = bankroll + Σ(shares_held × current_prob)`
    /// для всех открытых позиций. Считается на **каждом** тике (а не только
    /// на моменте сделки), иначе `max_drawdown_pct` системно занижен — между
    /// `open` и `close` реализованный bankroll не двигается, хотя цена
    /// открытой позиции может уйти в большой минус.
    ///
    /// Имена полей `peak_bankroll`/`max_drawdown_pct` остаются прежними для
    /// совместимости с логами и `print_sim_stats`, но **семантика теперь
    /// equity-based**, а не bankroll-based.
    pub(crate) fn update_drawdown(&mut self, equity: f64) {
        if equity > self.peak_bankroll {
            self.peak_bankroll = equity;
        }
        if self.peak_bankroll > 0.0 {
            let drawdown_pct = (self.peak_bankroll - equity) / self.peak_bankroll * 100.0;
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
                     | hold_zone≤{HOLD_TO_END_THRESHOLD_SEC}s ev_margin={EV_EXIT_MARGIN} ema_α={EV_EXIT_P_WIN_EMA_ALPHA} \
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
                                &booster_up, 
                                &booster_down,
                                calibration_up.as_ref(), 
                                calibration_down.as_ref(),
                                booster_resolution_up.as_ref(), booster_resolution_down.as_ref(),
                                calibration_resolution_up.as_ref(), calibration_resolution_down.as_ref(),
                                &mut sim_stats,
                            );
                            sim_stats.events += 1;
                        }
                        Err(err) => tee_eprintln!("[sim] {}: {err}", file_path.display()),
                    }
                }

                // Инвариант бухгалтерии: накопленный PnL по сторонам обязан
                // в точности совпадать с дельтой bankroll'а от стартового
                // капитала. Любое расхождение — тихий drift в close_position
                // (двойной учёт fee, потерянная сделка, невычтенный entry_cost
                // и т. п.). Допуск 1e-6 USDC покрывает округление f64 на
                // длинной серии сделок без маскирования реальных багов.
                debug_assert!(
                    {
                        let sides_pnl = sim_stats.up.pnl_usd + sim_stats.down.pnl_usd;
                        let bankroll_pnl = sim_stats.bankroll - INITIAL_BANKROLL;
                        (sides_pnl - bankroll_pnl).abs() < 1e-6
                    },
                    "[sim] {tag}: PnL invariant violated — up.pnl + down.pnl = {:.6}, bankroll - INITIAL = {:.6}",
                    sim_stats.up.pnl_usd + sim_stats.down.pnl_usd,
                    sim_stats.bankroll - INITIAL_BANKROLL,
                );

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

/// Симулирует один маркет: **два независимых прохода** по кадрам — сначала
/// сторона UP, затем сторона DOWN. Связи между кадрами UP/DOWN по индексу
/// нет (они хранятся в `MarketXFramesDump` как два независимых временных
/// ряда, и любая попытка пройти их «параллельно по `idx`» десинхронизуется
/// при первом же пропуске кадра у одной из сторон).
///
/// Bankroll шарится между двумя проходами: после прохода UP все его позиции
/// закрыты по Resolution (либо естественно при `event_remaining_ms ≤ 0`,
/// либо через `last_idx`-fallback на битом буфере), и DOWN стартует с
/// post-UP-bankroll. Это совпадает с фактической бухгалтерией live-режима
/// для ОДНОГО интервала (см. [`crate::real_sim`]).
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
    // Дамп `xframe_dump.rs` уже отфильтрован по `stable=true` (см.
    // `dump_market_xframes_binary_lane` → `if !frame.stable { continue; }`),
    // поэтому повторный filter здесь был бы no-op'ом и лишним аллоком.
    // Стабильность всё равно перепроверяется в `buy_gate` через
    // `BuyGate::Unstable` — это симметрично с `real_sim` (там кадры
    // приходят из live-фанаута без предварительного фильтра).
    let frames_up: Vec<&XFrame<SIZE>>   = market_xframes.frames_up.iter().collect();
    let frames_down: Vec<&XFrame<SIZE>> = market_xframes.frames_down.iter().collect();

    // Бинарная выплата на резолюции: победивший токен → $1, проигравший → $0.
    // Считаем один раз — `sell_gate` возьмёт её только при
    // `event_remaining_ms ≤ 0` (или на `is_last_idx`-fallback'е).
    let up_won = market_xframes.up_won();

    // Сначала UP: все его позиции закроются к концу прохода, после чего
    // bankroll корректно описывает доступный капитал для DOWN-прохода.
    {
        let SimStats { bankroll, peak_bankroll, max_drawdown_pct, up, .. } = &mut *sim_stats;
        let mut positions_up: Vec<OpenPosition> = Vec::new();
        run_side_simulation(
            &frames_up, up_won,
            booster_up, calibration_up,
            booster_resolution_up, calibration_resolution_up,
            &mut positions_up,
            bankroll, peak_bankroll, max_drawdown_pct,
            up,
        );
    }
    {
        let SimStats { bankroll, peak_bankroll, max_drawdown_pct, down, .. } = &mut *sim_stats;
        let mut positions_down: Vec<OpenPosition> = Vec::new();
        run_side_simulation(
            &frames_down, !up_won,
            booster_down, calibration_down,
            booster_resolution_down, calibration_resolution_down,
            &mut positions_down,
            bankroll, peak_bankroll, max_drawdown_pct,
            down,
        );
    }
}

/// Один проход одной стороны (UP или DOWN) по своему временно́му ряду
/// кадров. Внутри: `manage_positions` → `try_open_position` →
/// **mark-to-market equity** drawdown.
///
/// Equity на каждом тике считается как `bankroll + Σ(pos.shares_held ×
/// current_prob)`. Это критично: между `open` и `close` реализованный
/// `bankroll` не двигается, поэтому если считать drawdown только по
/// `bankroll`, метрика системно занижается на длинных удержаниях,
/// уходящих в красное и закрывающихся через Resolution.
///
/// Сайзинг новых позиций идёт от **available capital** =
/// `bankroll − Σ(open.entry_cost)` той же стороны (cross-side для
/// history_sim не нужен, т.к. UP и DOWN запускаются последовательно и
/// другая сторона на данном проходе всегда пуста).
#[allow(clippy::too_many_arguments)]
fn run_side_simulation(
    frames: &[&XFrame<SIZE>],
    won: bool,
    booster_pnl: &Booster,
    calibration_pnl: Option<&Calibration>,
    booster_resolution: Option<&Booster>,
    calibration_resolution: Option<&Calibration>,
    positions: &mut Vec<OpenPosition>,
    bankroll: &mut f64,
    peak_bankroll: &mut f64,
    max_drawdown_pct: &mut f64,
    side_stats: &mut SideStats,
) {
    let len = frames.len();
    if len == 0 {
        return;
    }
    // Fallback на «битый» буфер: дамп может оборваться раньше, чем
    // `event_remaining_ms ≤ 0`. В штатных прогонах последний кадр уже в
    // резолюции и `sell_gate` закрывает всё сам; `last_idx` нужен только
    // чтобы добить остатки в аномалии, не теряя PnL ещё открытых позиций.
    let last_idx = len - 1;
    let won_opt: Option<bool> = Some(won);

    for (idx, frame) in frames.iter().enumerate() {
        let is_last_idx = idx == last_idx;

        // 1) Закрытия по TP/SL/Timeout/EV/Resolution.
        manage_positions(
            positions,
            frame,
            is_last_idx,
            won_opt,
            booster_resolution,
            calibration_resolution,
            side_stats,
            bankroll,
            None,
            "",
        );

        // 2) Открытие новой позиции — от available_bankroll, иначе Kelly
        //    раздувает экспозицию: bankroll не уменьшается на open
        //    (всё списание идёт через PnL на close), и без этой коррекции
        //    последовательные сигналы открыли бы 5×10% = 50% bankroll
        //    параллельно вместо одного 10%.
        let same_side_locked: f64 = positions.iter().map(|p| p.entry_cost).sum();
        let available = (*bankroll - same_side_locked).max(0.0);
        try_open_position(
            frame,
            booster_pnl,
            calibration_pnl,
            positions,
            side_stats,
            available,
            None,
            "",
        );

        // 3) Mark-to-market equity на каждом тике (а не только на сделке).
        //    Не используем `currency_implied_prob.unwrap_or(0.0)` бездумно:
        //    `None` означает «mark-to-market неизвестен», и оценивать
        //    позицию в нуль — это занижение equity на ровном месте; в
        //    таком кадре пропускаем апдейт, метрика подождёт следующего.
        if let Some(prob) = frame.currency_implied_prob {
            let prob = prob.clamp(0.0, 1.0);
            let positions_value: f64 = positions.iter().map(|p| p.shares_held * prob).sum();
            let equity = *bankroll + positions_value;
            if equity > *peak_bankroll {
                *peak_bankroll = equity;
            }
            if *peak_bankroll > 0.0 {
                let dd = (*peak_bankroll - equity) / *peak_bankroll * 100.0;
                if dd > *max_drawdown_pct {
                    *max_drawdown_pct = dd;
                }
            }
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
///
/// При `!frame.stable` вход также пропускается (`unstable_skips++`):
/// фичи такого кадра не покрываются обучением (поздний WS-коннект),
/// pnl-модель нельзя считать применимой.
/// Результат «сухого» прогона всех ранних `return`-ов [`try_open_position`].
///
/// Нужен, чтобы одна и та же цепочка проверок (late-entry → predict →
/// threshold → Kelly → min size) жила в одном месте ([`buy_gate`]), а её
/// потребители — настоящий вход (`try_open_position`, мутирует `stats`) и
/// дешёвый предикат для `real_sim` (`matches!(buy_gate(..), Proceed)` до
/// HTTP-запроса стакана) — не копипастили одни и те же условия.
///
/// Варианты упорядочены от самого раннего отказа к успеху; каждый несёт
/// ровно те промежуточные значения, которые нужны дальше (для `diag_sum_*`
/// и вызова `open_position`), чтобы `try_open_position` не пересчитывал их
/// повторно.
pub enum BuyGate {
    /// До резолюции осталось меньше [`MIN_ENTRY_REMAINING_MS`]
    /// (≈ горизонт обучения, обычно 15 с) **или** событие уже завершилось
    /// (`event_remaining_ms ≤ 0`). Вход бессмысленен: TP/SL за оставшийся
    /// горизонт физически не успеют сработать, а на уже закрытом событии
    /// покупка — это лотерея по биркам `0/1`. В `try_open_position` это
    /// `stats.late_entry_skips += 1`.
    LateEntry,
    /// Кадр помечен `stable=false` — WS-коннект случился позже, чем
    /// `event_start_ms` или `ws_connect_wall_ms + SIZE`-секунд истории
    /// (см. [`crate::xframe::compute_xframe_stable`]). Pnl-модель обучалась
    /// только на стабильных кадрах, применять её к нестабильным некорректно.
    /// В `try_open_position` это `stats.unstable_skips += 1`.
    Unstable,
    /// `predict_frame` не вернул значение (нет свежих фич / лаг больше
    /// `PNL_MAX_LAG`) **или** сырой скор ниже `SIM_BUY_THRESHOLD`.
    /// Счётчики не мутируются (до `raw_above_threshold` мы не дошли).
    BelowThreshold,
    /// Порог прошли (обновляем `diag_sum_*` и `raw_above_threshold`), но
    /// Kelly не даёт edge: `kelly_f_adj ≤ 0` или итоговый размер меньше
    /// `MIN_POSITION_USD`. В обоих случаях `try_open_position` бьёт
    /// `kelly_skips`, поэтому различать их отдельно не нужно.
    KellySkip { raw: f32, pred: f32, kelly_f: f64 },
    /// Все проверки пройдены — есть смысл звать `open_position(size)`.
    Proceed { raw: f32, pred: f32, kelly_f: f64, size: f64 },
}

/// Прогоняет весь decision-tree входа **без** побочных эффектов и возвращает
/// стадию, на которой он остановился (см. [`BuyGate`]).
///
/// Вся «математика» входа (late-entry, порог сырой модели, калибрация,
/// Kelly, `KELLY_MULTIPLIER`, `MAX_BET_FRACTION`, `MIN_POSITION_USD`) живёт
/// только здесь. `try_open_position` оборачивает это в счётчики и реальный
/// `open_position`; `real_sim` использует `matches!(.., BuyGate::Proceed)`
/// как дешёвый gate до HTTP-запроса стакана.
pub(crate) fn buy_gate(
    frame: &XFrame<SIZE>,
    booster_pnl: &Booster,
    calibration_pnl: Option<&Calibration>,
    bankroll: f64,
) -> BuyGate {

    if frame.event_remaining_ms < MIN_ENTRY_REMAINING_MS {
        return BuyGate::LateEntry;
    }
    if !frame.stable {
        return BuyGate::Unstable;
    }
    // Без `currency_implied_prob` ни Kelly, ни размер посчитать не из
    // чего. Оба вызывающих (`run_history_sim`, `real_sim::tick_once`)
    // отбрасывают такие кадры до вызова `buy_gate`, но защитно отдаём
    // `BelowThreshold` — эта ветка не обновляет `diag_sum_*`, поэтому
    // статистика остаётся consistent с прежним поведением.
    let Some(entry_prob) = frame.currency_implied_prob else {
        return BuyGate::BelowThreshold;
    };
    let Some(raw) = predict_frame(booster_pnl, frame, PNL_MAX_LAG) else {
        return BuyGate::BelowThreshold;
    };
    if raw < SIM_BUY_THRESHOLD {
        return BuyGate::BelowThreshold;
    }
    let pred = calibration_pnl.map_or(raw, |c| c.apply(raw));
    let kelly_gain = kelly_gain_ratio(entry_prob);
    let kelly_loss = kelly_loss_ratio(entry_prob);
    let kelly_f = kelly_fraction(pred as f64, kelly_gain, kelly_loss);
    let kelly_f_adj = kelly_f * KELLY_MULTIPLIER;
    if kelly_f_adj <= 0.0 {
        return BuyGate::KellySkip { raw, pred, kelly_f };
    }
    let size = kelly_f_adj.min(MAX_BET_FRACTION) * bankroll;
    if size < MIN_POSITION_USD {
        return BuyGate::KellySkip { raw, pred, kelly_f };
    }
    BuyGate::Proceed { raw, pred, kelly_f, size }
}

/// Возвращает `true`, если на этом тике реально открыли позицию
/// (`positions.push(pos)` сработал). На любом skip-пути — `false`,
/// чтобы вызывающая сторона (см. `real_sim::tick_once`) могла
/// триггерить вывод/`update_drawdown` без локального трекинга
/// `positions.len()`/`stats.trades`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_open_position(
    frame: &XFrame<SIZE>,
    booster_pnl: &Booster,
    calibration_pnl: Option<&Calibration>,
    positions: &mut Vec<OpenPosition>,
    stats: &mut SideStats,
    bankroll: f64,
    strict_book: Option<&StrictBook>,
    log_tag: &str,
) -> bool {
    // Единая точка принятия решения о входе: ниже — только бухгалтерия
    // (счётчики пропусков + `open_position` при успехе). Вся логика
    // «брать/не брать» — в `buy_gate`, её же использует `real_sim` как
    // дешёвый gate до HTTP-запроса стакана (`matches!(.., Proceed)`).
    //
    // Нет `currency_implied_prob` — `buy_gate` всё равно вернул бы
    // `BelowThreshold` (no-op здесь, `diag_sum_*` не трогаются), так
    // что коротко замыкаем до booster-инференса и пропускаем тик.
    let Some(entry_prob) = frame.currency_implied_prob else {
        return false;
    };
    match buy_gate(frame, booster_pnl, calibration_pnl, bankroll) {
        BuyGate::LateEntry => {
            stats.late_entry_skips += 1;
            false
        }
        BuyGate::Unstable => {
            stats.unstable_skips += 1;
            false
        }
        BuyGate::BelowThreshold => {
            // До диагностических сумм мы не дошли — ни `raw_above_threshold`,
            // ни `diag_sum_*` не трогаем (совместимо с прежним поведением).
            false
        }
        BuyGate::KellySkip { raw, pred, kelly_f } => {
            stats.raw_above_threshold += 1;
            stats.diag_sum_raw += raw as f64;
            stats.diag_sum_calibrated += pred as f64;
            stats.diag_sum_entry_prob += entry_prob;
            stats.diag_sum_kelly_f += kelly_f;
            stats.kelly_skips += 1;
            false
        }
        BuyGate::Proceed { raw, pred, kelly_f, size } => {
            stats.raw_above_threshold += 1;
            stats.diag_sum_raw += raw as f64;
            stats.diag_sum_calibrated += pred as f64;
            stats.diag_sum_entry_prob += entry_prob;
            stats.diag_sum_kelly_f += kelly_f;
            match open_position(frame, size, stats, strict_book) {
                Some(pos) => {
                    positions.push(pos);
                    true
                }
                None => {
                    // Сюда попадаем только в strict-режиме ([`crate::real_sim`]):
                    // HTTP-стакана не хватило для покупки на `size` USDC — вход
                    // пропускаем, копим отдельный `kelly_strict_buy_skips`,
                    // чтобы не путать с «нет edge по Kelly».
                    stats.kelly_strict_buy_skips += 1;
                    let prefix = if log_tag.is_empty() {
                        String::new()
                    } else {
                        format!("[{log_tag}] ")
                    };
                    tee_eprintln!(
                        "{prefix}buy skip: HTTP-стакан не закрывает size={size:.4} USDC — пропускаем вход"
                    );
                    false
                }
            }
        }
    }
}

/// Результат «сухого» прогона decision-tree закрытия позиции.
///
/// Парный к [`BuyGate`] на sell-стороне: одна и та же цепочка условий
/// (Resolution на `event_remaining_ms ≤ 0` → hard SL → hold-zone EV-exit →
/// TP / SL / Timeout вне hold zone) живёт в одном месте ([`sell_gate`]), а
/// её потребители — настоящий цикл `manage_positions` (мутирует позиции/
/// статистику) и дешёвый WS-предикат [`any_position_would_sell`] для
/// `real_sim` — не копипастят эти ветви.
pub(crate) enum SellGate {
    /// Hold в **PnL-зоне** — обычный режим ведения позиции: TP/SL/Timeout
    /// по ценовой дельте на горизонте pnl-модели. `pos.p_win_ema` в этой
    /// зоне намеренно не трогается (EMA — это исключительно резолюционный
    /// концепт hold-zone), поэтому и возвращать из гейта нечего.
    HoldPnl,
    /// Hold в **hold-zone** (resolution-зона) — близко к концу события,
    /// TP/Timeout отключены, работают hard SL и EV-exit по резолюционной
    /// модели. `new_p_win_ema` — результат EMA-апдейта этим тиком; caller
    /// **обязан** записать его в `pos.p_win_ema`, иначе EMA регрессирует
    /// на одно состояние назад. Если `booster_resolution=None` (WS-предикат)
    /// или predict не удался — равен `pos.p_win_ema` без изменений.
    HoldResolution { new_p_win_ema: Option<f64> },
    /// Позицию закрываем с указанной причиной и ценой выхода. `exit_price`
    /// идёт в `close_position` для учёта реального fill (через `strict_book`
    /// или WS-fallback), а также в статистику/логи. EMA тут не возвращается:
    /// позиция всё равно уйдёт из `positions`.
    Close { exit_price: f64, reason: CloseReason },
}

/// Прогоняет весь decision-tree закрытия **одной** позиции без побочных
/// эффектов (парно с [`buy_gate`]). Включает и resolution-predict
/// (`booster_resolution` → `calibration_resolution` → EMA `p_win`), и
/// проверку причин закрытия — всё, что раньше было разбросано между
/// `manage_positions` и вручную копипастилось в WS-предикате.
///
/// Контракт параметров, которые могут отличаться у разных потребителей:
/// * `frames_held` — как бы **после** инкремента этим тиком. В
///   `manage_positions` это уже инкрементированное `pos.frames_held`; в
///   WS-предикате — `pos.frames_held + 1`, т.к. инкремент ещё не случился.
/// * `p_win_now` — свежая оценка `P(win)` от resolution-модели на этом
///   кадре (после калибровки), посчитанная **одной** инференцией в
///   вызывающем `manage_positions` и переиспользуемая для всех его
///   позиций. WS-предикат [`any_position_would_sell`] передаёт `None`,
///   чтобы намеренно **не дёргать predict_frame** на «тихих» тиках; в
///   этом случае EMA не обновляется (`new_p_win_ema = pos.p_win_ema`)
///   и EV-exit считается по последнему известному EMA.
/// * `strict_book` — `Some(book)` только в реальном исполнении через
///   `manage_positions` при наличии HTTP-стакана. В WS-предикате всегда
///   `None` — EV-exit оценивается через `book_fill_sell` (WS-fallback).
///
/// Если в hold-zone по strict-стакану не хватает bid-ликвидности на
/// `pos.shares_held`, возвращается [`SellGate::Hold`]: решение о выходе
/// зафиксировать честно нельзя (остаток пришлось бы досчитывать по
/// implied prob), ждём следующего тика.
///
/// Стоимость: `predict_frame` resolution-модели вызывается **один раз
/// на кадр** в `manage_positions` и переиспользуется для всех его
/// позиций. `sell_gate` сам инференса не делает; его доля каждой
/// позиции — только EMA-апдейт, проверки TP/SL/Timeout/EV.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sell_gate(
    pos: &OpenPosition,
    frames_held: usize,
    frame: &XFrame<SIZE>,
    is_last: bool,
    won: Option<bool>,
    p_win_now: Option<f64>,
    strict_book: Option<&StrictBook>,
) -> SellGate {
    // Событие завершилось (`event_remaining_ms ≤ 0`) или вызывающий
    // принудительно дожимает последний кадр (`is_last = true` — fallback
    // для битого буфера в `run_history_sim`) — обрабатываем первым:
    // любое другое правило (TP/SL/Timeout/EV) неактуально, позиция
    // гасится по бинарной выплате CTF ($1/шер победителю, $0 проигравшему
    // — без комиссии). `won` определяется вызывающим: в историческом
    // режиме — по `market.up_won()` (см. `XFrameMarket::up_won` в
    // `xframe_dump.rs`), в real_sim — по `currency_implied_prob ≥ 0.5`.
    // `close_position` в ветке Resolution не обращается к `strict_book`
    // — HTTP-запрос ордербука не нужен, а `exit_price` там же не
    // используется (`net_usdc` считается как `shares_held`/`0.0`), но
    // для логов/статистики берём последний known implied prob кадра: он
    // ближе всего к реальной рыночной цене на момент резолюции.
    //
    // `won = None` легален только у «пустого» вызова из
    // `any_position_would_sell`, который сам коротко замыкает на
    // `is_last || event_remaining_ms ≤ 0` до вызова `sell_gate`. Если
    // мы всё-таки оказались здесь с `None` — это баг вызывающего
    // (нарушен контракт «на резолюции `won` обязан быть `Some`»),
    // молча удерживать позицию и маскировать проблему нельзя.
    if is_last || frame.event_remaining_ms <= 0 {
        let won = won.expect(
            "sell_gate: is_last || event_remaining_ms <= 0 требует Some(won) \
             (up_won в истории / currency_implied_prob ≥ 0.5 в real_sim)",
        );
        return SellGate::Close {
            exit_price: frame
                .currency_implied_prob
                .unwrap_or(if won { 1.0 } else { 0.0 }),
            reason: CloseReason::Resolution { won },
        };
    }

    // Вне ветки Resolution без `currency_implied_prob` решение принять
    // нельзя — ни TP/SL/Timeout, ни EV-exit посчитать не из чего. Оба
    // вызывающих (`run_history_sim`, `real_sim::tick_once`) до этого
    // момента уже отбрасывают кадры без prob, но защитно возвращаем Hold,
    // а не panic — повторный тик с валидным кадром обработает позицию.
    let Some(current_prob) = frame.currency_implied_prob else {
        return SellGate::HoldPnl;
    };

    let in_hold_zone = frame.event_remaining_ms > 0
        && frame.event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000;
    let delta = current_prob - pos.entry_prob;

    if in_hold_zone {
        // EMA-апдейт делаем здесь же, чтобы вся логика sell-решения
        // была в одном месте. Если `p_win_now=None` (WS-предикат или
        // predict не удался — лаг больше `RESOLUTION_MAX_LAG`) — EMA
        // этим тиком не обновляется, EV-exit считаем по последнему
        // известному `pos.p_win_ema`. Сам инференс делает вызывающий
        // (`manage_positions`) **один раз на кадр**, см. контракт.
        let new_p_win_ema: Option<f64> = match (p_win_now, pos.p_win_ema) {
            (Some(p), Some(prev)) => Some(
                EV_EXIT_P_WIN_EMA_ALPHA * p + (1.0 - EV_EXIT_P_WIN_EMA_ALPHA) * prev,
            ),
            (Some(p), None) => Some(p),
            (None, existing) => existing,
        };

        // В зоне удержания TP и Timeout отключены (модель резолюции лучше
        // оценивает P(win) вблизи конца события), но остаются два выхода:
        // 1) hard SL — предохранитель от переуверенной resolution-модели
        //    (цена уходит вниз быстрее, чем EMA успевает на это среагировать);
        // 2) EV-exit — мягкий выход, если рыночная продажа после fee
        //    строго выгоднее ожидаемого удержания до резолюции.
        if delta <= Y_TRAIN_STOP_LOSS_PP {
            return SellGate::Close { exit_price: current_prob, reason: CloseReason::StopLoss };
        }
        let Some(p_ema) = new_p_win_ema else {
            return SellGate::HoldResolution { new_p_win_ema };
        };
        let gross_usdc_opt = match strict_book {
            Some(book) => book_fill_sell_strict(book, pos.shares_held),
            None => Some(book_fill_sell(frame, pos.shares_held)),
        };
        let Some(gross_usdc) = gross_usdc_opt else {
            return SellGate::HoldResolution { new_p_win_ema };
        };
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
            return SellGate::Close { exit_price: current_prob, reason };
        }
        return SellGate::HoldResolution { new_p_win_ema };
    }

    // PnL-зона: классический TP/SL/Timeout по ценовой дельте и горизонту
    // pnl-модели. EMA `p_win` здесь не трогается и в возврат не
    // включается — это исключительно резолюционный концепт hold-zone.
    if delta >= Y_TRAIN_TAKE_PROFIT_PP {
        return SellGate::Close { exit_price: current_prob, reason: CloseReason::TakeProfit };
    }
    if delta <= Y_TRAIN_STOP_LOSS_PP {
        return SellGate::Close { exit_price: current_prob, reason: CloseReason::StopLoss };
    }
    if frames_held >= Y_TRAIN_HORIZON_FRAMES {
        return SellGate::Close { exit_price: current_prob, reason: CloseReason::Timeout };
    }
    SellGate::HoldPnl
}

/// Дешёвый WS-предикат «нужен ли strict HTTP-стакан для продажи этим тиком».
///
/// Парный к `matches!(buy_gate(..), BuyGate::Proceed)` на sell-стороне:
/// используется в `real_sim` как gate до HTTP-запроса стакана, чтобы не
/// дёргать CLOB на «тихих» тиках, когда позиции просто стоят на хранении.
/// Возвращает `true`, если [`sell_gate`] на одной из позиций при
/// WS-fallback (без strict book, стандартное `pos.p_win_ema`,
/// `frames_held + 1`) вернул бы [`SellGate::Close`]:
///
/// * hard SL — работает и в обычной зоне, и в hold zone;
/// * TP / Timeout — только **вне** hold zone;
/// * EV-exit в hold zone — по `book_fill_sell(frame, shares)` и
///   **последнему известному** `p_win_ema`. На реальном ходу
///   `manage_positions` сперва обновит EMA, потом проверит по
///   strict-стакану — результат может чуть отличаться, но WS-оценка
///   обычно консистентна с HTTP (HTTP не хуже по глубине).
///
/// **На `event_remaining_ms ≤ 0` и на `is_last = true` возвращаем `false`**:
/// завершившееся событие (или принудительное закрытие на последнем кадре
/// битого буфера в историческом прогоне) закрывается через
/// `CloseReason::Resolution` (бинарная выплата `0`/`1`), `close_position`
/// в этой ветке **не обращается** к `strict_book` — HTTP не нужен.
/// `manage_positions` всё равно должен быть вызван (для Resolution),
/// поэтому в `real_sim` условие вызова `manage_positions` отделено от
/// HTTP-гейта: `has_positions → вызываем; needs_sell → заодно тянем HTTP`.
pub(crate) fn any_position_would_sell(
    positions: &[OpenPosition],
    frame: &XFrame<SIZE>,
    is_last: bool,
) -> bool {
    if positions.is_empty() || is_last || frame.event_remaining_ms <= 0 {
        return false;
    }
    positions.iter().any(|pos| {
        matches!(
            sell_gate(
                pos,
                // `frames_held` как бы после инкремента в `manage_positions`.
                pos.frames_held + 1,
                frame,
                // `is_last = false` и `won = None`: сюда мы попадаем только
                // при `!is_last && event_remaining_ms > 0` (короткое
                // замыкание выше), первая ветвь `sell_gate` заведомо не
                // срабатывает — `expect(..)` на `won` не взведётся.
                false,
                None,
                // `p_win_now = None`: EV-exit использует последнее
                // известное `pos.p_win_ema` без свежего predict_frame —
                // дешёвый gate, на «тихих» тиках мы намеренно не
                // запускаем resolution-инференс.
                None,
                // Без HTTP: EV считаем через WS-fallback `book_fill_sell`.
                None,
            ),
            SellGate::Close { .. }
        )
    })
}

/// Общий lifecycle позиций одной стороны за один кадр: инкремент `frames_held`
/// и делегация решения о закрытии в [`sell_gate`] (там же живут predict
/// resolution-модели, EMA `p_win`, hold-zone EV-exit и TP/SL/Timeout). На
/// `SellGate::Hold` персистим обновлённый EMA в `pos.p_win_ema`; на
/// `SellGate::Close` — зовём `close_position` (P&L → `bankroll`, счётчики
/// → `stats`). В strict-режиме при нехватке bid-ликвидности на
/// `shares_held` `close_position` возвращает `None`, позиция остаётся
/// открытой и ретраится на следующем тике (`kelly_strict_sell_skips++`).
///
/// Возвращает `true`, если хотя бы одна позиция была успешно закрыта на
/// этом тике (т.е. `bankroll` сдвинулся и `stats.trades` инкрементнулся).
/// На `Hold*` для всех позиций или на strict skip’ах закрытия — `false`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn manage_positions(
    positions: &mut Vec<OpenPosition>,
    frame: &XFrame<SIZE>,
    is_last: bool,
    won: Option<bool>,
    booster_resolution: Option<&Booster>,
    calibration_resolution: Option<&Calibration>,
    stats: &mut SideStats,
    bankroll: &mut f64,
    strict_book: Option<&StrictBook>,
    log_tag: &str,
) -> bool {
    for pos in positions.iter_mut() { pos.frames_held += 1; }

    // `predict_frame` resolution-модели стоит дёшево относительно HTTP,
    // но всё же это booster-инференс по полному вектору фич. Считаем
    // **один раз на кадр**, переиспользуем для всех позиций этой
    // стороны (на практике 0–1 одновременно, но если правила входа
    // станут более либеральными — экономия линейная по числу позиций).
    //
    // Считаем только если действительно нужно: hold-zone — единственная
    // ветка, где `sell_gate` смотрит на `p_win_now`. Иначе — `None`
    // (EV-exit и EMA-апдейт всё равно не сработают вне зоны).
    let in_hold_zone = frame.event_remaining_ms > 0
        && frame.event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000;
    let p_win_now: Option<f64> = if in_hold_zone && !positions.is_empty() {
        booster_resolution.and_then(|b| {
            predict_frame(b, frame, RESOLUTION_MAX_LAG).map(|raw| {
                calibration_resolution.map_or(raw, |c| c.apply(raw)) as f64
            })
        })
    } else {
        None
    };

    let mut sold = false;
    let mut remaining: Vec<OpenPosition> = Vec::new();
    for mut pos in positions.drain(..) {
        // Весь decision-tree (EMA-апдейт `p_win_ema`, проверки
        // TP/SL/Timeout/EV) живёт в `sell_gate`. Здесь подаём
        // `pos.frames_held` уже после инкремента и **готовый**
        // `p_win_now` (один инференс на кадр выше). Текущий
        // `current_prob` гейт достаёт из `frame.currency_implied_prob`.
        // `is_last` = true принудительно закрывает позицию через
        // Resolution (fallback для битого буфера в истории).
        let close = match sell_gate(
            &pos,
            pos.frames_held,
            frame,
            is_last,
            won,
            p_win_now,
            strict_book,
        ) {
            SellGate::Close { exit_price, reason } => Some((exit_price, reason)),
            SellGate::HoldResolution { new_p_win_ema } => {
                // Персистим обновлённый EMA на позицию — только в hold-zone.
                // На `Close` EMA возвращать не надо: позиция уйдёт из Vec.
                pos.p_win_ema = new_p_win_ema;
                None
            }
            // В PnL-зоне EMA не обновлялся — `pos.p_win_ema` остаётся как есть.
            SellGate::HoldPnl => None,
        };
        if let Some((exit_price, reason)) = close {
            match close_position(&pos, exit_price, &reason, frame, stats, strict_book) {
                Some(pnl) => {
                    *bankroll += pnl;
                    sold = true;
                }
                None => {
                    stats.kelly_strict_sell_skips += 1;
                    let prefix = if log_tag.is_empty() { String::new() } else { format!("[{log_tag}] ") };
                    tee_eprintln!(
                        "{prefix}sell skip ({reason:?}): HTTP-стакана не хватает на shares={:.4}; держим позицию и ретраим в след. тике",
                        pos.shares_held,
                    );
                    remaining.push(pos);
                }
            }
        } else {
            remaining.push(pos);
        }
    }
    *positions = remaining;
    sold
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
fn open_position(
    frame: &XFrame<SIZE>,
    position_size: f64,
    stats: &mut SideStats,
    strict_book: Option<&StrictBook>,
) -> Option<OpenPosition> {
    let (buy_price, nominal_shares) = match strict_book {
        Some(book) => book_fill_buy_strict(book, position_size)?,
        None => book_fill_buy(frame, position_size),
    };
    let buy_price = buy_price.clamp(0.001, 0.999);

    let fee_usdc = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy_price * (1.0 - buy_price);
    let fee_shares = fee_usdc / buy_price;
    let actual_shares = nominal_shares - fee_shares;

    stats.fees_paid += fee_usdc;

    Some(OpenPosition {
        shares_held: actual_shares,
        entry_prob: frame.currency_implied_prob.unwrap_or(buy_price),
        entry_cost: position_size,
        frames_held: 0,
        p_win_ema: None,
    })
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
    strict_book: Option<&StrictBook>,
) -> Option<f64> {
    let net_usdc = match reason {
        CloseReason::Resolution { won: true } => pos.shares_held,
        CloseReason::Resolution { won: false } => 0.0,
        CloseReason::TakeProfit
        | CloseReason::StopLoss
        | CloseReason::Timeout
        | CloseReason::EvExitProfit
        | CloseReason::EvExitLoss => {
            let gross_usdc = match strict_book {
                Some(book) => book_fill_sell_strict(book, pos.shares_held)?,
                None => book_fill_sell(frame, pos.shares_held),
            };
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

    Some(pnl)
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
///
/// Шеры, для которых не хватает bid-ликвидности на трёх верхних уровнях,
/// **сжигаются**: они вносят `0` USDC в выручку — то есть учитываются
/// как полная потеря. Это явная замена прежнего fallback'а по
/// `currency_implied_prob`, который **системно завышал** оценку модели:
/// он позволял допродать любой остаток «бесплатно по mid», тогда как
/// в живой торговле непокрытый ликвидностью объём — это либо рейс
/// по более худшим уровням стакана (скрытым в L4+), либо вообще
/// невозможность исполнения.
///
/// Для оценки обученной модели worst-case-лосс на хвосте — корректнее,
/// чем оптимистичный fallback. На стороне `real_sim` строгий вариант
/// [`book_fill_sell_strict`] вообще возвращает `None` (откладываем
/// продажу до следующего тика); здесь же позиция **обязана закрыться**
/// (Resolution на конце буфера, либо явный TP/SL/Timeout/EV), поэтому
/// «отложить» нельзя — выбираем консервативную оценку выручки.
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
            break;
        } else {
            total_usdc += size * price;
            remaining -= size;
        }
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

pub(crate) fn print_side_stats(tag: &str, side_label: &str, s: &SideStats) {
    let n = s.raw_above_threshold.max(1) as f64;
    let diag = format!(
        "raw≥thr={} avg_raw={:.3} avg_cal={:.3} avg_entry={:.3} avg_kelly_f={:.4} kelly_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={}",
        s.raw_above_threshold,
        s.diag_sum_raw / n,
        s.diag_sum_calibrated / n,
        s.diag_sum_entry_prob / n,
        s.diag_sum_kelly_f / n,
        s.kelly_skips,
        s.kelly_strict_buy_skips,
        s.kelly_strict_sell_skips,
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
         | TP={} SL={} Timeout={} EvExit✓={} EvExit✗={} Res✓={} Res✗={} late_skips={} unstable_skips={}",
        s.trades, win_rate, s.pnl_usd, avg_pnl, s.fees_paid,
        s.tp_count, s.sl_count, s.timeout_count,
        s.ev_exit_profit_count, s.ev_exit_loss_count,
        s.resolution_win, s.resolution_loss, s.late_entry_skips, s.unstable_skips,
    );
}

pub(crate) fn print_sim_stats(tag: &str, sim_stats: &SimStats) {
    let total_trades = sim_stats.total_trades();
    if total_trades == 0 {
        tee_println!(
            "[sim] {tag}: нет сделок ({} событий, kelly_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={})",
            sim_stats.events,
            sim_stats.total_kelly_skips(),
            sim_stats.total_kelly_strict_buy_skips(),
            sim_stats.total_kelly_strict_sell_skips(),
        );
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

pub(crate) fn load_booster(path: &Path) -> Option<Booster> {
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
