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
//! Синхронный цикл UP/DOWN по кадрам; вход при проходе Kelly/gates; выход TP/SL/timeout/EV или резолюция (`calc_y_train_pnl`-пороги).

use crate::account::Account;
use crate::constants::{
    CurrencyUpDownOutcome, XFrameIntervalKind,
};
use crate::real_sim::interval_label;
use crate::train_mode::{
    collect_bin_paths, load_calibration, split_counts,
    Calibration, PNL_MAX_LAG, RESOLUTION_MAX_LAG, TEST_FRACTION, VAL_FRACTION,
};
use crate::xframe::{
    apply_side_symmetry, BookLevel, XFrame, SIZE,
    Y_TRAIN_TAKE_PROFIT_PP, Y_TRAIN_STOP_LOSS_PP,
    Y_TRAIN_NO_TRADE_PROB_LOW, Y_TRAIN_NO_TRADE_PROB_HIGH,
};
use crate::xframe_dump::MarketXFramesDump;
use crate::{tee_eprintln, tee_println};
use std::fs;
use std::path::Path;
use xgb::{Booster, DMatrix};

/// Префильтр raw-предикта перед Kelly (`f* > 0`).
pub const SIM_BUY_THRESHOLD: f32 = 0.70;

/// Cap проскальзывания VWAP от L1 при `book_fill_*` (см. y_train в train_mode).
pub const SIM_MAX_SLIPPAGE_FROM_L1_PCT: f64 = 0.02;

/// Стартовый виртуальный банкролл (USDC).
pub const INITIAL_BANKROLL: f64 = 50.0;
/// Множитель Келли (<1 — fractional Kelly).
pub const KELLY_MULTIPLIER: f64 = 0.1;
/// Максимальная доля банкролла на одну сделку.
pub const MAX_BET_FRACTION: f64 = 0.10;
/// Минимальный размер позиции в USDC (меньше — не торгуем).
pub const MIN_POSITION_USD: f64 = 0.01;

/// Жёсткий cap номинала сделки (USDC) поверх `MAX_BET_FRACTION × bankroll`:
/// ограничивает число шеров на тонком стакане (слиппедж walk по bid/ask).
pub const MAX_POSITION_USD: f64 = 300.0;

/// Фиксированный размер входа в режиме `run_sim_mode(is_kelly=false)` (без Kelly/калибровки; raw pred).
pub const NO_KELLY_POSITION_SIZE_USD: f64 = 30.0;

/// Коэффициент taker-комиссии Polymarket для категории **Crypto** (CLOB):
/// `fee_usdc = C × POLYMARKET_CRYPTO_TAKER_FEE_RATE × p × (1 − p)`, где C — число шерсов, p — цена.
/// См. [Polymarket: Fees](https://docs.polymarket.com/trading/fees).
pub const POLYMARKET_CRYPTO_TAKER_FEE_RATE: f64 = 0.072;

/// Hold-zone: конец окна по времени; TP/timeout off; остаются hard SL и EV-exit по resolution-модели.
pub const HOLD_TO_END_THRESHOLD_SEC: i64 = 0;

/// EMA по `p_win` resolution-модели в hold-zone (`α` мало → плавнее EV-exit).
pub const EV_EXIT_P_WIN_EMA_ALPHA: f64 = 0.3;

/// Зазор EV-exit: `EV_sell × (1 − margin) > EV_hold`.
pub const EV_EXIT_MARGIN: f64 = 0.01;

/// Лимит кадров без TP/SL → [`CloseReason::Timeout`]. Должен совпадать с `Y_TRAIN_HORIZON_FRAMES` в xframe.
pub const POSITION_TIMEOUT_FRAMES: usize = 30;

/// Минимум `event_remaining_ms` для нового входа ([`buy_gate`] LateEntry).
pub const MIN_ENTRY_REMAINING_MS: i64 = 10 * 1000;

/// Halt новых входов при `max_drawdown_pct ≥ pct` (только `real_sim`; закрытие позиций не трогает).
pub const EMERGENCY_HALT_DRAWDOWN_PCT: Option<f64> = Some(30.0);

/// HTTP-снимок CLOB для `real_sim`: `Some` → `book_fill_*_strict` + slippage cap + `min_order_size`; `None` → WS `book_fill_*` в history_sim.
#[derive(Debug, Clone, Default)]
pub(crate) struct StrictBook {
    /// Уровни спроса (bids), лучший bid = первый.
    pub(crate) bids: Vec<BookLevel>,
    /// Уровни предложения (asks), лучший ask = первый.
    pub(crate) asks: Vec<BookLevel>,
    /// Last trade CLOB (wide spread → как в `currency_implied_prob_polymarket_style`).
    pub(crate) last_trade_price: Option<f64>,
    /// Мин. размер ордера в шерах (HTTP); strict-fill без этого не открывает/не закрывает.
    pub(crate) min_order_size: Option<f64>,
}

/// Polymarket-style implied prob: из [`StrictBook`] как `currency_implied_prob_polymarket_style`, иначе `frame.currency_implied_prob`.
pub(crate) fn effective_implied_prob(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
) -> Option<f64> {
    if let Some(book) = strict_book {
        let best_bid = book
            .bids
            .iter()
            .find(|l| l.price > 0.0 && l.size > 0.0)
            .map(|l| l.price);
        let best_ask = book
            .asks
            .iter()
            .find(|l| l.price > 0.0 && l.size > 0.0)
            .map(|l| l.price);
        let spread = match (best_bid, best_ask) {
            (Some(b), Some(a)) => Some((a - b).max(0.0)),
            _ => None,
        };
        if let Some(p) = crate::xframe::currency_implied_prob_polymarket_style(
            best_bid,
            best_ask,
            spread,
            book.last_trade_price,
        ) {
            return Some(p.clamp(0.001, 0.999));
        }
    }
    frame.currency_implied_prob
}

/// Покупка по HTTP asks: полный fill `position_size`, cap от L1 ask, опционально `min_order_size`.
pub(crate) fn book_fill_buy_strict(
    book: &StrictBook,
    position_size: f64,
) -> Option<(f64, f64)> {
    if position_size <= 0.0 {
        return None;
    }
    let best_ask = book
        .asks
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)?;
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
    if (vwap - best_ask) / best_ask > SIM_MAX_SLIPPAGE_FROM_L1_PCT {
        return None;
    }
    if let Some(min) = book.min_order_size {
        if total_shares < min {
            return None;
        }
    }
    Some((vwap, total_shares))
}

/// Продажа по HTTP bids: gross USDC до fee; `Some(cap)` — voluntary (TP/EvProfit), `None` — urgent (SL/timeout/…); проверка `min_order_size`.
pub(crate) fn book_fill_sell_strict(
    book: &StrictBook,
    shares_to_sell: f64,
    slippage_cap: Option<f64>,
) -> Option<f64> {
    if shares_to_sell <= 0.0 {
        return Some(0.0);
    }
    if let Some(min) = book.min_order_size {
        if shares_to_sell < min {
            return None;
        }
    }
    let best_bid = book
        .bids
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)?;
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
    let vwap = total_usdc / shares_to_sell;
    if let Some(cap) = slippage_cap {
        if (best_bid - vwap) / best_bid > cap {
            return None;
        }
    }
    Some(total_usdc)
}

/// Открытая позиция; в `real_sim` фильтр `asset_id == frame.asset_id` от чужого маркета.
#[derive(Debug, Clone)]
pub struct OpenPosition {
    pub(crate) asset_id: String,
    #[allow(dead_code)]
    pub(crate) market_id: String,
    /// Количество шерсов после вычета комиссии при покупке.
    pub(crate) shares_held: f64,
    /// Отображаемая prob на входе; пайплайн оценки использует только [`Self::buy_price`].
    #[allow(dead_code)]
    pub(crate) entry_prob: f64,
    /// VWAP покупки — база для TP/SL, pending MtM, CSV `buy_price`, гистограмма входов.
    pub(crate) buy_price: f64,
    /// USDC потраченные на покупку (= POSITION_SIZE_USD).
    pub(crate) entry_cost: f64,
    /// L1 bid на входе: maker vs taker для модельной TP-лимитки в [`close_position`].
    pub(crate) best_bid_at_entry: Option<f64>,
    /// Сколько кадров позиция уже удерживается (для таймаута).
    pub(crate) frames_held: usize,
    /// EMA `p_win` resolution-модели (hold-zone EV-exit).
    pub(crate) p_win_ema: Option<f64>,
    /// CSV only: raw/cal/kelly на открытии.
    pub(crate) raw_pred_at_open: f32,
    pub(crate) cal_pred_at_open: f32,
    pub(crate) kelly_f_at_open: f64,
    pub(crate) event_remaining_ms_at_open: i64,
    pub(crate) xframe_interval_type_at_open: i32,
    pub(crate) currency_up_down_outcome_at_open: i32,
    pub(crate) currency: String,
    pub(crate) polymarket_url: String,
    pub(crate) price_to_beat: Option<f64>,
    pub(crate) final_price: Option<f64>,
    /// Конец окна UTC (мс) для `open_unix_ms`/`close_unix_ms` в CSV.
    pub(crate) event_end_ms: Option<i64>,
}

/// Рыночный выход до резолюции; бинарное закрытие — в [`crate::account::Account::resolve_pending_market`].
#[derive(Debug, Clone, PartialEq)]
pub enum CloseReason {
    TakeProfit,
    StopLoss,
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

impl CloseReason {
    /// TP / EvExitProfit — можно отложить выход при слишком глубоком slippage (`SIM_MAX_SLIPPAGE_FROM_L1_PCT`).
    pub fn is_voluntary_exit(&self) -> bool {
        matches!(self, CloseReason::TakeProfit | CloseReason::EvExitProfit)
    }
}

/// Статистика по одной стороне (UP/DOWN).
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
    pub(crate) sl_count: usize,
    /// Число погашений победившего токена при резолюции события (exit = 1.0, без fee).
    /// Это **token-outcome** счётчик: ставка зашла, как мы и ставили.
    /// Знак P&L при этом может быть и отрицательным — если зашли
    /// слишком дорого (`entry_prob` близко к 1.0), entry-fee и
    /// зафиксированный `entry_cost = position_size` могут оставить
    /// итоговый pnl ниже нуля. Точная разбивка см. в
    /// [`Self::resolution_win_profit`] / [`Self::resolution_win_loss`].
    pub(crate) resolution_win: usize,
    /// Подмножество [`Self::resolution_win`], где сделка завершилась
    /// **прибыльно** (`pnl ≥ 0`). Делим, чтобы не путать
    /// «токен победил» (token-outcome) и «сделка в плюс» (pnl-sign):
    /// они расходятся при дорогих входах. По этому полю плюс
    /// `resolution_win_loss` всегда восстанавливается полный
    /// `resolution_win`.
    pub(crate) resolution_win_profit: usize,
    /// Подмножество [`Self::resolution_win`], где сделка завершилась
    /// **убытком** (`pnl < 0`) несмотря на правильный исход:
    /// `entry_cost` оказался выше выплаты `shares_held × 1.0` после
    /// учёта entry-fee. Сигнал того, что Kelly входит в позиции на
    /// слишком высоких `entry_prob`, где маржа выплаты съедается
    /// комиссией.
    pub(crate) resolution_win_loss: usize,
    /// Число сгораний проигравшего токена при резолюции события (exit = 0.0).
    /// Всегда `pnl < 0` (теряется весь `entry_cost`), отдельной
    /// разбивки по знаку pnl не нужно.
    pub(crate) resolution_loss: usize,
    /// Число выходов по таймауту: позиция удерживалась >= [`POSITION_TIMEOUT_FRAMES`] кадров без TP/SL.
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
    /// Попытка второй раз открыть позицию на **тот же** `asset_id`, пока
    /// первая ещё в `positions` (см. [`try_open_position`]: проверяется
    /// только в ветке `BuyGate::Proceed`, поэтому считает **сигналы
    /// Kelly**, а не каждый кадр удержания). Один токен = одна
    /// `OpenPosition` за раз — проще CLOB, TP/SL, бухгалтерия.
    pub(crate) same_asset_open_skips: usize,
    /// Число пропущенных входов из-за Kelly f* ≤ 0 (нет edge).
    pub(crate) kelly_skips: usize,
    /// Число пропущенных входов из-за `entry_prob` в no-trade-зоне
    /// `(Y_TRAIN_NO_TRADE_PROB_LOW..Y_TRAIN_NO_TRADE_PROB_HIGH)` (см.
    /// [`BuyGate::EntryProbOutOfRange`] и [`crate::xframe::calc_y_train_pnl`]).
    /// Считаются **отдельно** от `kelly_skips`: семантически это «рынок
    /// balanced, обе стороны равновероятны, y-метка туда не попадает —
    /// inference на distribution shift запрещён», в то время как
    /// `kelly_skips` — «модель сигналит, но edge недостаточен по f*».
    pub(crate) entry_prob_skips: usize,
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
    /// Гистограмма `entry_prob` в моменте успешного **открытия** позиции
    /// (`BuyGate::Proceed` + `open_position` отработали). 5 бакетов по 0.2:
    /// `[0.0..0.2)`, `[0.2..0.4)`, `[0.4..0.6)`, `[0.6..0.8)`, `[0.8..1.0]`.
    /// Без этой разбивки нельзя понять, в какую часть распределения
    /// «дешёвые / дорогие» входы перекошены — а это критично для оценки
    /// корректности Kelly-сайзинга.
    pub(crate) histogram_entry_prob: [usize; 5],
    /// Гистограмма калиброванного `pred` в моменте открытия позиции,
    /// та же сетка бакетов, что у [`Self::histogram_entry_prob`].
    /// Сравнение этих двух гистограмм показывает, на каком «edge'е»
    /// модели реально торгуем (в идеале `cal_pred > entry_prob`).
    pub(crate) histogram_cal_pred: [usize; 5],
    /// Сумма PnL по позициям, закрытым по [`CloseReason::TakeProfit`].
    /// Сейчас в [`Self::tp_count`] есть только число таких закрытий —
    /// без знания P&L нельзя видеть, что одно «удачное» TP не съедает
    /// 5 «неудачных» SL.
    pub(crate) pnl_tp: f64,
    /// Сумма PnL по позициям, закрытым по [`CloseReason::StopLoss`].
    pub(crate) pnl_sl: f64,
    /// Сумма PnL по позициям, закрытым по [`CloseReason::Timeout`].
    pub(crate) pnl_timeout: f64,
    /// Сумма PnL по позициям, закрытым по [`CloseReason::EvExitProfit`].
    pub(crate) pnl_ev_exit_profit: f64,
    /// Сумма PnL по позициям, закрытым по [`CloseReason::EvExitLoss`].
    pub(crate) pnl_ev_exit_loss: f64,
    /// Сумма PnL по позициям, закрытым через резолюцию маркета как
    /// **победившие** (`Account::resolve_pending_market_sync`,
    /// `token_won = true`). Может быть отрицательной при дорогих
    /// входах (см. doc у `Self::resolution_win_loss`).
    pub(crate) pnl_resolution_win: f64,
    /// Сумма PnL по позициям, закрытым через резолюцию маркета как
    /// **проигравшие** (`token_won = false`). Всегда `<= 0`
    /// (теряется весь `entry_cost`).
    pub(crate) pnl_resolution_loss: f64,
}

/// Статистика симуляции по версии; деньги/dd — в [`crate::account::Account`].
#[derive(Debug)]
pub struct SimStats {
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
    pub(crate) fn total_same_asset_open_skips(&self) -> usize {
        self.up.same_asset_open_skips + self.down.same_asset_open_skips
    }
    pub(crate) fn total_entry_prob_skips(&self) -> usize {
        self.up.entry_prob_skips + self.down.entry_prob_skips
    }
}

/// Два прогона подряд: `kelly` и `raw` ([`NO_KELLY_POSITION_SIZE_USD`]). Колонка CSV `regime`; свой [`Account`] на прогон.
pub fn run_sim_mode() -> anyhow::Result<()> {
    let xframes_root = Path::new("xframes");
    if !xframes_root.exists() {
        anyhow::bail!("Папка xframes/ не найдена — сначала соберите данные (STATUS=default)");
    }

    crate::tee_log::init_tee_log_file(&xframes_root.join("last_history_sim.txt"), "sim")?;
    crate::trade_csv_log::init_trade_csv_log_file(
        &xframes_root.join("last_history_sim_trades.csv"),
    )?;

    crate::trade_csv_log::set_current_regime("kelly");
    tee_println!("[sim] === regime=kelly (Kelly + calibration, min(MAX_BET_FRACTION × bankroll, MAX_POSITION_USD)) ===");
    run_sim_mode_inner(true)?;

    crate::trade_csv_log::set_current_regime("raw");
    tee_println!("[sim] === regime=raw (no Kelly, no calibration, ${NO_KELLY_POSITION_SIZE_USD} entry) ===");
    run_sim_mode_inner(false)?;

    crate::trade_csv_log::set_current_regime("");
    crate::trade_csv_log::finish_trade_csv_log();
    crate::tee_log::finish_tee_log();

    Ok(())
}

/// Один режим `is_kelly`; свой свежий [`Account::new()`].
fn run_sim_mode_inner(is_kelly: bool) -> anyhow::Result<()> {
    let xframes_root = Path::new("xframes");
    let regime_label = if is_kelly { "kelly" } else { "raw" };

    for currency_path in fs_sorted_dirs(xframes_root)? {
        let currency = dir_name(&currency_path);

        for version_path in fs_sorted_dirs(&currency_path)? {
            let version = dir_name(&version_path);
            if version.parse::<usize>().is_err() {
                continue;
            }

            let mut account = Account::new();

            for interval_kind in [XFrameIntervalKind::FiveMin, XFrameIntervalKind::FifteenMin] {
                let interval = interval_label(interval_kind);
                let interval_path = version_path.join(interval);
                if !interval_path.is_dir() {
                    continue;
                }

                let model_up_path   = version_path.join(format!("model_{interval}_1s_pnl_up.ubj"));
                let model_down_path = version_path.join(format!("model_{interval}_1s_pnl_down.ubj"));
                let model_resolution_up_path   = version_path.join(format!("model_{interval}_1s_resolution_up.ubj"));
                let model_resolution_down_path = version_path.join(format!("model_{interval}_1s_resolution_down.ubj"));

                let tag = format!("{currency}/{version}/{interval}/{regime_label}");

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

                if is_kelly {
                    tee_println!(
                        "[sim] {tag}: модели pnl загружены | {} | {} \
                         | resolution: up={} down={} \
                         | hold_zone≤{HOLD_TO_END_THRESHOLD_SEC}s ev_margin={EV_EXIT_MARGIN} ema_α={EV_EXIT_P_WIN_EMA_ALPHA} \
                         | threshold={SIM_BUY_THRESHOLD} | kelly={KELLY_MULTIPLIER} | max_bet={MAX_BET_FRACTION} | max_pos=${MAX_POSITION_USD} \
                         | no_trade_zone=({Y_TRAIN_NO_TRADE_PROB_LOW}..{Y_TRAIN_NO_TRADE_PROB_HIGH}) \
                         | bankroll={INITIAL_BANKROLL}$ | fee_rate={POLYMARKET_CRYPTO_TAKER_FEE_RATE}",
                        cal_info(&calibration_up, "cal_up"),
                        cal_info(&calibration_down, "cal_down"),
                        if booster_resolution_up.is_some()   { "✓" } else { "✗" },
                        if booster_resolution_down.is_some() { "✓" } else { "✗" },
                    );
                } else {
                    tee_println!(
                        "[sim] {tag}: модели pnl загружены | resolution: up={} down={} \
                         | hold_zone≤{HOLD_TO_END_THRESHOLD_SEC}s ev_margin={EV_EXIT_MARGIN} ema_α={EV_EXIT_P_WIN_EMA_ALPHA} \
                         | threshold={SIM_BUY_THRESHOLD} | entry=${NO_KELLY_POSITION_SIZE_USD} (fixed, no Kelly, no calibration) \
                         | no_trade_zone=({Y_TRAIN_NO_TRADE_PROB_LOW}..{Y_TRAIN_NO_TRADE_PROB_HIGH}) \
                         | bankroll={INITIAL_BANKROLL}$ | fee_rate={POLYMARKET_CRYPTO_TAKER_FEE_RATE}",
                        if booster_resolution_up.is_some()   { "✓" } else { "✗" },
                        if booster_resolution_down.is_some() { "✓" } else { "✗" },
                    );
                }

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
                            let polymarket_url =
                                polymarket_event_url_from_dump_path(file_path, &currency, interval_kind)
                                    .unwrap_or_default();
                            let event_end_ms = window_bounds_from_dump_path(file_path, interval_kind)
                                .map(|b| b.event_end_ms);
                            simulate_event(
                                &market_xframes,
                                &currency,
                                interval_kind,
                                &booster_up,
                                &booster_down,
                                calibration_up.as_ref(),
                                calibration_down.as_ref(),
                                booster_resolution_up.as_ref(), booster_resolution_down.as_ref(),
                                calibration_resolution_up.as_ref(), calibration_resolution_down.as_ref(),
                                &mut sim_stats,
                                &mut account,
                                is_kelly,
                                &polymarket_url,
                                event_end_ms,
                            );
                            sim_stats.events += 1;
                        }
                        Err(err) => tee_eprintln!("[sim] {}: {err}", file_path.display()),
                    }
                }

                print_sim_stats(&tag, &sim_stats, &account, is_kelly);
            }
        }
    }

    Ok(())
}

/// Один маркет: последовательные проходы UP и DOWN по двум независимым рядам кадров.
/// Общий банкролл (как в [`crate::real_sim`]).
#[allow(clippy::too_many_arguments)]
fn simulate_event(
    market_xframes: &MarketXFramesDump,
    currency: &str,
    interval_kind: XFrameIntervalKind,
    booster_up: &Booster,
    booster_down: &Booster,
    calibration_up: Option<&Calibration>,
    calibration_down: Option<&Calibration>,
    booster_resolution_up: Option<&Booster>,
    booster_resolution_down: Option<&Booster>,
    calibration_resolution_up: Option<&Calibration>,
    calibration_resolution_down: Option<&Calibration>,
    sim_stats: &mut SimStats,
    account: &mut Account,
    is_kelly: bool,
    polymarket_url: &str,
    event_end_ms: Option<i64>,
) {
    let price_to_beat = Some(market_xframes.price_to_beat);
    let final_price = Some(market_xframes.final_price);
    let lane_key_up = (currency.to_string(), interval_kind, CurrencyUpDownOutcome::Up);
    let lane_key_down = (currency.to_string(), interval_kind, CurrencyUpDownOutcome::Down);
    let frames_up: Vec<&XFrame<SIZE>>   = market_xframes.frames_up.iter().collect();
    let frames_down: Vec<&XFrame<SIZE>> = market_xframes.frames_down.iter().collect();

    let up_won = market_xframes.up_won();

    let market_id_opt: Option<String> = frames_up
        .first()
        .map(|f| f.market_id.clone())
        .or_else(|| frames_down.first().map(|f| f.market_id.clone()));

    {
        let mut positions_up: Vec<OpenPosition> = Vec::new();
        run_side_simulation(
            &frames_up,
            booster_up, calibration_up,
            booster_resolution_up, calibration_resolution_up,
            &mut positions_up,
            account,
            &lane_key_up,
            &mut sim_stats.up,
            currency,
            is_kelly,
            polymarket_url,
            price_to_beat,
            final_price,
            event_end_ms,
        );
    }
    {
        let mut positions_down: Vec<OpenPosition> = Vec::new();
        run_side_simulation(
            &frames_down,
            booster_down, calibration_down,
            booster_resolution_down, calibration_resolution_down,
            &mut positions_down,
            account,
            &lane_key_down,
            &mut sim_stats.down,
            currency,
            is_kelly,
            polymarket_url,
            price_to_beat,
            final_price,
            event_end_ms,
        );
    }

    if let Some(market_id) = market_id_opt {
        account.resolve_pending_market_sync(
            sim_stats,
            currency,
            interval_kind,
            &market_id,
            up_won,
        );
    }

    // После resolve pending по этому маркету должен быть пуст (иначе утечка между маркетами).
    assert!(
        account
            .pending_resolution
            .get(&lane_key_up)
            .map(|v| v.is_empty())
            .unwrap_or(true)
            && account
                .pending_resolution
                .get(&lane_key_down)
                .map(|v| v.is_empty())
                .unwrap_or(true),
        "history_sim: pending_resolution не опустошён после resolve_pending_market_sync \
         (lane_key_up={lane_key_up:?}, lane_key_down={lane_key_down:?}); \
         dump invariant violated",
    );
}

/// Один проход стороны (UP/DOWN) по ряду кадров: manage/open → MtM equity.
/// Живые позиции в локальной `Vec`; после цикла — в `pending_resolution`, финальный payout в `simulate_event`.
///
/// Equity: `bankroll + Σ(local×prob) + Σ(pending×buy_price)` (как `real_sim::tick_once`). Сайзинг от `bankroll − Σ(entry_cost)` на этой стороне.
#[allow(clippy::too_many_arguments)]
fn run_side_simulation(
    frames: &[&XFrame<SIZE>],
    booster_pnl: &Booster,
    calibration_pnl: Option<&Calibration>,
    booster_resolution: Option<&Booster>,
    calibration_resolution: Option<&Calibration>,
    positions: &mut Vec<OpenPosition>,
    account: &mut Account,
    lane_key: &(String, XFrameIntervalKind, CurrencyUpDownOutcome),
    side_stats: &mut SideStats,
    currency: &str,
    is_kelly: bool,
    polymarket_url: &str,
    price_to_beat: Option<f64>,
    final_price: Option<f64>,
    event_end_ms: Option<i64>,
) {
    if frames.is_empty() {
        return;
    }
    let last_idx = frames.len().saturating_sub(1);

    for (idx, frame) in frames.iter().enumerate() {
        let is_last_idx = idx == last_idx;
        let p_win_now = compute_p_win_now(
            frame,
            booster_resolution,
            calibration_resolution,
            is_kelly,
        );
        let pnl_inference = compute_pnl_inference(frame, booster_pnl, calibration_pnl, is_kelly);

        {
            let Account {
                bankroll,
                pending_resolution,
                ..
            } = &mut *account;
            let pending = pending_resolution
                .entry(lane_key.clone())
                .or_default();
            manage_positions(
                positions,
                pending,
                frame,
                is_last_idx,
                p_win_now,
                side_stats,
                bankroll,
                None,
                "",
                is_kelly,
            );
        }

        let same_side_locked: f64 = positions.iter().map(|p| p.entry_cost).sum();
        let available = (account.bankroll - same_side_locked).max(0.0);
        try_open_position(
            frame,
            pnl_inference,
            positions,
            side_stats,
            available,
            None,
            currency,
            is_kelly,
            polymarket_url,
            price_to_beat,
            final_price,
            event_end_ms,
        );

        // MtM equity (как real_sim): без prob на кадре тик пропускаем.
        if let Some(prob) = frame.currency_implied_prob {
            let prob = prob.clamp(0.0, 1.0);
            let positions_value: f64 = positions.iter().map(|p| p.shares_held * prob).sum();
            let pending_value: f64 = account
                .pending_resolution
                .values()
                .flat_map(|v| v.iter())
                .map(|p| p.shares_held * p.buy_price)
                .sum();
            let equity = account.bankroll + positions_value + pending_value;
            account.update_drawdown(equity);
        }
    }

    if !positions.is_empty() {
        let pending = account
            .pending_resolution
            .entry(lane_key.clone())
            .or_default();
        pending.append(positions);
    }
}

/// Сырой скор (`raw`) для порога и калиброванный (`pred`) для Kelly; считаются одним вызовом [`compute_pnl_inference`].
#[derive(Clone, Copy, Debug)]
pub struct PnlInference {
    pub raw: f32,
    pub pred: f32,
}

/// Booster + калибровка PnL на кадр. `None`: поздний вход / unstable / нет prob / лаг > [`PNL_MAX_LAG`].
/// Калибровка здесь, не в [`buy_gate`], чтобы real_sim считал инференс до write-локов.
pub(crate) fn compute_pnl_inference(
    frame: &XFrame<SIZE>,
    booster_pnl: &Booster,
    calibration_pnl: Option<&Calibration>,
    is_kelly: bool,
) -> Option<PnlInference> {
    if frame.event_remaining_ms < MIN_ENTRY_REMAINING_MS {
        return None;
    }
    if !frame.stable {
        return None;
    }
    if frame.currency_implied_prob.is_none() {
        return None;
    }
    let raw = predict_frame(booster_pnl, frame, PNL_MAX_LAG)?;
    let pred = if is_kelly {
        calibration_pnl.map_or(raw, |c| c.apply(raw))
    } else {
        raw
    };
    Some(PnlInference { raw, pred })
}

/// P(win) resolution-модели в hold-zone; `None` вне зоны / нет booster / лаг > [`RESOLUTION_MAX_LAG`].
/// Без гейта «есть позиции»: predict каждый тик в зоне — EMA не отстаёт на тик открытия.
pub(crate) fn compute_p_win_now(
    frame: &XFrame<SIZE>,
    booster_resolution: Option<&Booster>,
    calibration_resolution: Option<&Calibration>,
    is_kelly: bool,
) -> Option<f64> {
    let in_hold_zone = frame.event_remaining_ms > 0
        && frame.event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000;
    if !in_hold_zone {
        return None;
    }
    booster_resolution.and_then(|b| {
        predict_frame(b, frame, RESOLUTION_MAX_LAG).map(|raw| {
            if is_kelly {
                calibration_resolution.map_or(raw, |c| c.apply(raw)) as f64
            } else {
                raw as f64
            }
        })
    })
}

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
    /// `entry_prob` попал в no-trade-зону
    /// `(Y_TRAIN_NO_TRADE_PROB_LOW..Y_TRAIN_NO_TRADE_PROB_HIGH)` —
    /// центр распределения, где обе стороны равновероятны, и
    /// y-разметка ([`crate::xframe::calc_y_train_pnl`]) сюда не пишет
    /// меток. Inference на этом интервале — distribution shift, а
    /// edge модели в нём всё равно ≈0. Сигнал модели игнорируем
    /// независимо от его силы. В `try_open_position` это
    /// `stats.entry_prob_skips += 1` — диагностические суммы при этом
    /// **обновляются** (raw/cal/entry_prob/kelly_f), чтобы воронку было
    /// видно: «модель сигнальнула, отбросили по entry_prob».
    EntryProbOutOfRange { raw: f32, pred: f32, kelly_f: f64 },
    /// Порог прошли (обновляем `diag_sum_*` и `raw_above_threshold`), но
    /// Kelly не даёт edge: `kelly_f_adj ≤ 0` или итоговый размер меньше
    /// `MIN_POSITION_USD`. В обоих случаях `try_open_position` бьёт
    /// `kelly_skips`, поэтому различать их отдельно не нужно.
    /// Срезание сверху до `MAX_POSITION_USD` сюда **не** попадает —
    /// это не отказ, а сужение позиции (см. [`MAX_POSITION_USD`]).
    KellySkip { raw: f32, pred: f32, kelly_f: f64 },
    /// Успех → вызов `open_position(size)` ниже по модулю.
    Proceed { raw: f32, pred: f32, kelly_f: f64, size: f64 },
}

/// Дерево решений входа без побочных эффектов ([`BuyGate`]). Инференс снаружи — [`compute_pnl_inference`].
pub(crate) fn buy_gate(
    frame: &XFrame<SIZE>,
    pnl_inference: Option<PnlInference>,
    bankroll: f64,
    strict_book: Option<&StrictBook>,
    is_kelly: bool,
) -> BuyGate {

    if frame.event_remaining_ms < MIN_ENTRY_REMAINING_MS {
        return BuyGate::LateEntry;
    }
    if !frame.stable {
        return BuyGate::Unstable;
    }
    let Some(entry_prob) = effective_implied_prob(frame, strict_book) else {
        return BuyGate::BelowThreshold;
    };
    let Some(PnlInference { raw, pred }) = pnl_inference else {
        return BuyGate::BelowThreshold;
    };

    if raw < SIM_BUY_THRESHOLD {
        return BuyGate::BelowThreshold;
    }

    let best_bid_at_entry = match strict_book {
        Some(book) => book.bids.first().map(|lvl| lvl.price),
        None => frame.book_bid_l1_price,
    };
    let kelly_gain = kelly_gain_ratio(entry_prob, best_bid_at_entry);
    let kelly_loss = kelly_loss_ratio(entry_prob);
    let kelly_f = kelly_fraction(pred as f64, kelly_gain, kelly_loss);

    // Фильтр «только хвосты»: открываем позиции **только** при
    // `entry_prob ≤ Y_TRAIN_NO_TRADE_PROB_LOW` или
    // `entry_prob ≥ Y_TRAIN_NO_TRADE_PROB_HIGH`. В центре распределения
    // (`LOW..HIGH`) рынок balanced — обе стороны равновероятны,
    // edge модели размазан, и y-метка туда не попадает (см.
    // [`crate::xframe::calc_y_train_pnl`]). Если runtime сюда зайдёт,
    // модель будет inference'ить на распределении, на котором не
    // училась — distribution shift.
    if entry_prob > Y_TRAIN_NO_TRADE_PROB_LOW && entry_prob < Y_TRAIN_NO_TRADE_PROB_HIGH {
        return BuyGate::EntryProbOutOfRange { raw, pred, kelly_f };
    }

    if !is_kelly {
        let size = NO_KELLY_POSITION_SIZE_USD.min(bankroll).max(0.0);
        if size < MIN_POSITION_USD {
            // Bankroll исчерпан в ноль — открывать на пыль не имеет
            // смысла. Кодируем как `KellySkip`, чтобы caller увеличил
            // `kelly_skips` (печать в no-kelly режиме всё равно
            // переименует это поле в `bankroll_too_small_skips`,
            // см. `print_side_stats`).
            return BuyGate::KellySkip { raw, pred, kelly_f };
        }
        return BuyGate::Proceed { raw, pred, kelly_f, size };
    }

    let kelly_f_adj = kelly_f * KELLY_MULTIPLIER;
    if kelly_f_adj <= MIN_POSITION_USD {
        return BuyGate::KellySkip { raw, pred, kelly_f };
    }
    // Kelly-сайзинг с двумя cap'ами:
    // 1. `MAX_BET_FRACTION × bankroll` — масштабируется с банкроллом;
    // 2. `MAX_POSITION_USD` — абсолютный cap по числу шеров, чтобы
    //    walk_sell не вылезал за L1 на тонкой стороне стакана (см.
    //    doc у [`MAX_POSITION_USD`]).
    // Срезание до `MAX_POSITION_USD` НЕ переводит сделку в `KellySkip`
    // — логика «edge есть, но размер большой» это не отказ от входа,
    // а сужение позиции.
    let size = (kelly_f_adj.min(MAX_BET_FRACTION) * bankroll).min(MAX_POSITION_USD);
    if size < MIN_POSITION_USD {
        return BuyGate::KellySkip { raw, pred, kelly_f };
    }
    BuyGate::Proceed { raw, pred, kelly_f, size }
}

/// При успешном [`open_position`] — `true` и push в `positions`; иначе счётчики skip и `false`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_open_position(
    frame: &XFrame<SIZE>,
    pnl_inference: Option<PnlInference>,
    positions: &mut Vec<OpenPosition>,
    stats: &mut SideStats,
    bankroll: f64,
    strict_book: Option<&StrictBook>,
    currency: &str,
    is_kelly: bool,
    polymarket_url: &str,
    price_to_beat: Option<f64>,
    final_price: Option<f64>,
    event_end_ms: Option<i64>,
) -> bool {
    let Some(entry_prob) = effective_implied_prob(frame, strict_book) else {
        return false;
    };
    match buy_gate(frame, pnl_inference, bankroll, strict_book, is_kelly) {
        BuyGate::LateEntry => {
            stats.late_entry_skips += 1;
            false
        }
        BuyGate::Unstable => {
            stats.unstable_skips += 1;
            false
        }
        BuyGate::BelowThreshold => false,
        BuyGate::EntryProbOutOfRange { raw, pred, kelly_f } => {
            stats.raw_above_threshold += 1;
            stats.diag_sum_raw += raw as f64;
            stats.diag_sum_calibrated += pred as f64;
            stats.diag_sum_entry_prob += entry_prob;
            stats.diag_sum_kelly_f += kelly_f;
            stats.entry_prob_skips += 1;
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
            if positions
                .iter()
                .any(|p| p.asset_id == frame.asset_id)
            {
                stats.same_asset_open_skips += 1;
                return false;
            }
            stats.raw_above_threshold += 1;
            stats.diag_sum_raw += raw as f64;
            stats.diag_sum_calibrated += pred as f64;
            stats.diag_sum_entry_prob += entry_prob;
            stats.diag_sum_kelly_f += kelly_f;

            match open_position(
                frame, size, stats, strict_book, raw, pred, kelly_f, currency,
                polymarket_url, price_to_beat, final_price, event_end_ms,
            ) {
                Some(pos) => {
                    // Гистограммы заполняем только для **успешно открытых**
                    // позиций — нас интересует распределение реальных входов,
                    // а не «kelly_skip / thin_book_skip». Бакет берём по
                    // `pos.buy_price` (фактический VWAP заполнения, реальная
                    // цена покупки) и `pred` (калиброванный); это две точки,
                    // между которыми живёт edge модели. Используем buy_price,
                    // а не `entry_prob`, чтобы гистограмма отражала
                    // **фактические цены входа**, а не displayed-prob (mid/
                    // last_trade), который при широком спреде может далеко
                    // расходиться с реальной ценой fill'а.
                    let bucket_entry = prob_bucket_index(pos.buy_price);
                    let bucket_pred = prob_bucket_index(pred as f64);
                    stats.histogram_entry_prob[bucket_entry] += 1;
                    stats.histogram_cal_pred[bucket_pred] += 1;
                    positions.push(pos);
                    true
                }
                None => {
                    stats.kelly_strict_buy_skips += 1;
                    false
                }
            }
        }
    }
}

/// Парный к [`BuyGate`]: решение о закрытии без побочных эффектов.
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

/// Bid-walk с cap + fee + maker-флаг для TP/SL-дельты и EV-exit (один расчёт).
#[derive(Clone, Copy)]
struct CappedSellFill {
    gross_usdc: f64,
    /// VWAP gross по bid-walk (pp на шер).
    sell_vwap: f64,
    ev_sell_taker: f64,
    ev_is_maker: bool,
}

fn capped_sell_fill_for_gate(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    shares_held: f64,
    current_prob: f64,
) -> Option<CappedSellFill> {
    let gross_usdc = match strict_book {
        Some(book) => book_fill_sell_strict(book, shares_held, Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT)),
        None => book_fill_sell(frame, shares_held, Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT)),
    }?;
    let sell_vwap = if shares_held > 0.0 {
        (gross_usdc / shares_held).clamp(0.001, 0.999)
    } else {
        current_prob.clamp(0.001, 0.999)
    };
    let fee_usdc =
        shares_held * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell_vwap * (1.0 - sell_vwap);
    let current_best_bid = match strict_book {
        Some(book) => book.bids.first().map(|lvl| lvl.price),
        None => frame.book_bid_l1_price,
    };
    let ev_is_maker = match current_best_bid {
        Some(b) => current_prob > b,
        None => true,
    };
    let ev_sell_taker = gross_usdc - fee_usdc;
    Some(CappedSellFill {
        gross_usdc,
        sell_vwap,
        ev_sell_taker,
        ev_is_maker,
    })
}

/// `frames_held` — уже после инкремента тика (`manage_positions`) или `+1` в WS-предикате.
/// `p_win_now` — из одного predict на кадр; `None` в [`any_position_would_sell`] (EMA не двигается).
pub(crate) fn sell_gate(
    pos: &OpenPosition,
    frames_held: usize,
    frame: &XFrame<SIZE>,
    is_last: bool,
    p_win_now: Option<f64>,
    strict_book: Option<&StrictBook>,
) -> SellGate {
    if is_last || frame.event_remaining_ms <= 0 {
        return SellGate::HoldPnl;
    }

    let Some(current_prob) = effective_implied_prob(frame, strict_book) else {
        return SellGate::HoldPnl;
    };

    let in_hold_zone = frame.event_remaining_ms > 0 && frame.event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000;

    let Some(fill) = capped_sell_fill_for_gate(frame, strict_book, pos.shares_held, current_prob)
    else {
        return SellGate::HoldPnl;
    };
    let delta = fill.sell_vwap - pos.buy_price;

    if in_hold_zone {
        let new_p_win_ema: Option<f64> = match (p_win_now, pos.p_win_ema) {
            (Some(p), Some(prev)) => Some(EV_EXIT_P_WIN_EMA_ALPHA * p + (1.0 - EV_EXIT_P_WIN_EMA_ALPHA) * prev),
            (Some(p), None) => Some(p),
            (None, existing) => existing,
        };

        if delta <= Y_TRAIN_STOP_LOSS_PP {
            return SellGate::Close { exit_price: current_prob, reason: CloseReason::StopLoss };
        }
        let ev_close: Option<(f64, CloseReason)> = new_p_win_ema.and_then(|p_ema| {
            let ev_sell_maker = if fill.ev_is_maker {
                fill.gross_usdc
            } else {
                fill.ev_sell_taker
            };
            let ev_hold = p_ema * pos.shares_held;
            if fill.ev_sell_taker * (1.0 - EV_EXIT_MARGIN) > ev_hold {
                let reason = if ev_sell_maker > pos.entry_cost {
                    CloseReason::EvExitProfit
                } else {
                    CloseReason::EvExitLoss
                };
                Some((current_prob, reason))
            } else {
                None
            }
        });
        if let Some((exit_price, reason)) = ev_close {
            return SellGate::Close { exit_price, reason };
        }
        return SellGate::HoldResolution { new_p_win_ema };
    }

    if delta >= Y_TRAIN_TAKE_PROFIT_PP {
        return SellGate::Close { exit_price: current_prob, reason: CloseReason::TakeProfit };
    }
    if delta <= Y_TRAIN_STOP_LOSS_PP {
        return SellGate::Close { exit_price: current_prob, reason: CloseReason::StopLoss };
    }
    if frames_held >= POSITION_TIMEOUT_FRAMES {
        return SellGate::Close { exit_price: current_prob, reason: CloseReason::Timeout };
    }
    SellGate::HoldPnl
}

/// Gate до HTTP: был бы [`sell_gate`] в режиме WS (`Close`) на этом тике.
pub(crate) fn any_position_would_sell(
    positions: &[OpenPosition],
    frame: &XFrame<SIZE>,
) -> bool {
    if positions.is_empty() || frame.event_remaining_ms <= 0 {
        return false;
    }
    positions.iter().any(|pos| {
        if pos.asset_id != frame.asset_id {
            return false;
        }
        matches!(
            sell_gate(
                pos,
                pos.frames_held + 1,
                frame,
                false,
                None,
                None,
            ),
            SellGate::Close { .. }
        )
    })
}

/// Закрытия через [`sell_gate`] / `close_position`; чужой `asset_id` → [`pending_resolution`](crate::account::Account::pending_resolution).
/// `true`, если был хотя бы один успешный close (bankroll обновился).
#[allow(clippy::too_many_arguments)]
pub(crate) fn manage_positions(
    positions: &mut Vec<OpenPosition>,
    pending_resolution: &mut Vec<OpenPosition>,
    frame: &XFrame<SIZE>,
    is_last: bool,
    p_win_now: Option<f64>,
    stats: &mut SideStats,
    bankroll: &mut f64,
    strict_book: Option<&StrictBook>,
    _log_tag: &str,
    _is_kelly: bool,
) -> bool {
    for pos in positions.iter_mut() { pos.frames_held += 1; }

    let mut sold = false;
    let mut remaining: Vec<OpenPosition> = Vec::new();
    for mut pos in positions.drain(..) {
        if pos.asset_id != frame.asset_id {
            pending_resolution.push(pos);
            continue;
        }
        let close = match sell_gate(
            &pos,
            pos.frames_held,
            frame,
            is_last,
            p_win_now,
            strict_book,
        ) {
            SellGate::Close { exit_price, reason } => Some((exit_price, reason)),
            SellGate::HoldResolution { new_p_win_ema } => {
                pos.p_win_ema = new_p_win_ema;
                None
            }
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


/// Доля выигрыша при TP: вход taker по `entry_prob`, выход по TP; maker/taker выхода по `best_bid_at_entry` (как в `close_position`).
fn kelly_gain_ratio(entry_prob: f64, best_bid_at_entry: Option<f64>) -> f64 {
    let sell_price = (entry_prob + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999);
    let tp_is_maker = match best_bid_at_entry {
        Some(best_bid) => sell_price > best_bid,
        None => true,
    };
    let net = net_round_trip(entry_prob, sell_price, /*sell_is_taker=*/ !tp_is_maker);
    (net - 1.0).max(1e-9)
}

/// Доля убытка при SL: всегда taker на выходе (как `close_position` на SL).
fn kelly_loss_ratio(entry_prob: f64) -> f64 {
    let sell_price = (entry_prob + Y_TRAIN_STOP_LOSS_PP).clamp(0.001, 0.999);
    let net = net_round_trip(entry_prob, sell_price, /*sell_is_taker=*/ true);
    (1.0 - net).max(1e-9)
}

/// Чистый множитель на $1 notional: вход taker; `sell_is_taker` — urgent vs maker выход.
fn net_round_trip(buy: f64, sell: f64, sell_is_taker: bool) -> f64 {
    let nominal_shares = 1.0 / buy;
    let buy_fee = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy * (1.0 - buy);
    let actual_shares = nominal_shares - buy_fee / buy;

    let gross = actual_shares * sell;
    let sell_fee = if sell_is_taker {
        actual_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell * (1.0 - sell)
    } else {
        0.0
    };
    gross - sell_fee
}

/// Kelly `f* = p/l − q/g`; может быть >1 — caller режет по [`MAX_BET_FRACTION`] / USD-cap.
fn kelly_fraction(p_win: f64, gain: f64, loss: f64) -> f64 {
    if gain <= 0.0 || loss <= 0.0 {
        return 0.0;
    }
    let q = 1.0 - p_win;
    p_win / loss - q / gain
}

/// Индекс бакета 0..=4 для гистограмм (шаг 0.2); `<0`/NaN→0, `≥1`→4.
pub(crate) fn prob_bucket_index(p: f64) -> usize {
    if !p.is_finite() || p < 0.0 {
        return 0;
    }
    let idx = (p * 5.0).floor() as i64;
    idx.clamp(0, 4) as usize
}

/// Виртуальный вход на `position_size` USDC: ask-walk, fee из шеров, VWAP→[`OpenPosition::buy_price`].
/// `raw/cal/kelly` и `currency` — только для CSV; gate уже прошёл.
#[allow(clippy::too_many_arguments)]
fn open_position(
    frame: &XFrame<SIZE>,
    position_size: f64,
    stats: &mut SideStats,
    strict_book: Option<&StrictBook>,
    raw_pred_at_open: f32,
    cal_pred_at_open: f32,
    kelly_f_at_open: f64,
    currency: &str,
    polymarket_url: &str,
    price_to_beat: Option<f64>,
    final_price: Option<f64>,
    event_end_ms: Option<i64>,
) -> Option<OpenPosition> {
    let (buy_price, nominal_shares) = match strict_book {
        Some(book) => book_fill_buy_strict(book, position_size)?,
        None => book_fill_buy(frame, position_size, Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT))?,
    };
    if nominal_shares <= 0.0 {
        return None;
    }
    let buy_price = buy_price.clamp(0.001, 0.999);

    let fee_usdc = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy_price * (1.0 - buy_price);
    let fee_shares = fee_usdc / buy_price;
    let actual_shares = nominal_shares - fee_shares;

    stats.fees_paid += fee_usdc;

    // Отображаемая prob (CSV/MtM); TP/SL считаются от buy_price.
    let entry_prob = effective_implied_prob(frame, strict_book).unwrap_or(buy_price);

    let best_bid_at_entry = match strict_book {
        Some(book) => book.bids.first().map(|lvl| lvl.price),
        None => frame.book_bid_l1_price,
    };

    Some(OpenPosition {
        asset_id: frame.asset_id.clone(),
        market_id: frame.market_id.clone(),
        shares_held: actual_shares,
        entry_prob,
        buy_price,
        entry_cost: position_size,
        best_bid_at_entry,
        frames_held: 0,
        p_win_ema: None,
        raw_pred_at_open,
        cal_pred_at_open,
        kelly_f_at_open,
        event_remaining_ms_at_open: frame.event_remaining_ms,
        xframe_interval_type_at_open: frame.xframe_interval_type,
        currency_up_down_outcome_at_open: frame.currency_up_down_outcome,
        currency: currency.to_string(),
        polymarket_url: polymarket_url.to_string(),
        price_to_beat,
        final_price,
        event_end_ms,
    })
}

/// Рыночный выход (TP/SL/Timeout/EvExit): bid-walk, fee. Резолюция — в [`crate::account::Account`].
fn close_position(
    pos: &OpenPosition,
    exit_price: f64,
    reason: &CloseReason,
    frame: &XFrame<SIZE>,
    stats: &mut SideStats,
    strict_book: Option<&StrictBook>,
) -> Option<f64> {
    let slippage_cap = if reason.is_voluntary_exit() {
        Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT)
    } else {
        None
    };
    let gross_usdc = match strict_book {
        Some(book) => book_fill_sell_strict(book, pos.shares_held, slippage_cap)?,
        None => book_fill_sell(frame, pos.shares_held, slippage_cap)?,
    };
    let sell_price = if pos.shares_held > 0.0 {
        (gross_usdc / pos.shares_held).clamp(0.001, 0.999)
    } else {
        exit_price.clamp(0.001, 0.999)
    };
    // TP: maker по bid на входе; EvExitProfit: по текущему bid; иначе taker.
    let voluntary_is_maker = match reason {
        CloseReason::TakeProfit => {
            let tp_target = (pos.buy_price + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999);
            match pos.best_bid_at_entry {
                Some(b) => tp_target > b,
                None => true,
            }
        }
        CloseReason::EvExitProfit => {
            let exit_clamped = exit_price.clamp(0.001, 0.999);
            let current_best_bid = match strict_book {
                Some(book) => book.bids.first().map(|lvl| lvl.price),
                None => frame.book_bid_l1_price,
            };
            match current_best_bid {
                Some(b) => exit_clamped > b,
                None => true,
            }
        }
        _ => false,
    };
    let fee_usdc = if voluntary_is_maker {
        0.0
    } else {
        pos.shares_held * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell_price * (1.0 - sell_price)
    };
    stats.fees_paid += fee_usdc;
    let net_usdc = gross_usdc - fee_usdc;

    let pnl = net_usdc - pos.entry_cost;
    stats.pnl_usd += pnl;

    stats.trades += 1;
    if pnl >= 0.0 { stats.wins += 1; } else { stats.losses += 1; }

    match reason {
        CloseReason::TakeProfit   => { stats.tp_count += 1;              stats.pnl_tp += pnl; }
        CloseReason::StopLoss     => { stats.sl_count += 1;              stats.pnl_sl += pnl; }
        CloseReason::Timeout      => { stats.timeout_count += 1;         stats.pnl_timeout += pnl; }
        CloseReason::EvExitProfit => { stats.ev_exit_profit_count += 1;  stats.pnl_ev_exit_profit += pnl; }
        CloseReason::EvExitLoss   => { stats.ev_exit_loss_count += 1;    stats.pnl_ev_exit_loss += pnl; }
    }

    // Per-trade CSV-лог (если открыт через `init_trade_csv_log_file`).
    // Пишется ровно одной строкой на закрытие; resolution-закрытия
    // (бинарная выплата $1/$0) пишет `Account::resolve_pending_market_sync`.
    let interval_str = position_interval_label(pos);
    let side_str = position_side_label(pos);
    let open_unix_ms = pos.event_end_ms.map(|e| e - pos.event_remaining_ms_at_open);
    let close_unix_ms = pos.event_end_ms.map(|e| e - frame.event_remaining_ms);
    crate::trade_csv_log::write_trade_csv_row(crate::trade_csv_log::TradeCsvRow {
        polymarket_url: &pos.polymarket_url,
        price_to_beat: pos.price_to_beat,
        final_price: pos.final_price,
        market_id: &pos.market_id,
        asset_id: &pos.asset_id,
        side: side_str,
        interval: interval_str,
        currency: &pos.currency,
        exit_reason: trade_csv_close_reason_label(reason),
        buy_price: pos.buy_price,
        raw_pred: pos.raw_pred_at_open,
        cal_pred: pos.cal_pred_at_open,
        kelly_f: pos.kelly_f_at_open,
        entry_cost: pos.entry_cost,
        shares_held: pos.shares_held,
        exit_price: sell_price,
        fee_usdc,
        pnl,
        frames_held: pos.frames_held,
        p_win_ema_at_close: pos.p_win_ema,
        event_remaining_ms_at_open: pos.event_remaining_ms_at_open,
        event_remaining_ms_at_close: frame.event_remaining_ms,
        open_unix_ms,
        close_unix_ms,
    });

    Some(pnl)
}

/// CSV: `"5m"` / `"15m"` / `"unknown"` через [`crate::real_sim::interval_label`].
pub(crate) fn position_interval_label(pos: &OpenPosition) -> &'static str {
    match XFrameIntervalKind::from_i32(pos.xframe_interval_type_at_open) {
        Some(kind) => crate::real_sim::interval_label(kind),
        None => "unknown",
    }
}

/// Лейбл стороны позиции для CSV: `"up"` / `"down"` / `"unknown"`.
pub(crate) fn position_side_label(pos: &OpenPosition) -> &'static str {
    match CurrencyUpDownOutcome::from_i32(pos.currency_up_down_outcome_at_open) {
        Some(outcome) => crate::real_sim::side_label(outcome),
        None => "unknown",
    }
}

/// Стабильный `exit_reason` для CSV (не `Debug`).
pub(crate) fn trade_csv_close_reason_label(reason: &CloseReason) -> &'static str {
    match reason {
        CloseReason::TakeProfit => "TP",
        CloseReason::StopLoss => "SL",
        CloseReason::Timeout => "Timeout",
        CloseReason::EvExitProfit => "EvExitProfit",
        CloseReason::EvExitLoss => "EvExitLoss",
    }
}

/// Ask-walk до полного `position_size`; опционально cap VWAP к best ask ([`SIM_MAX_SLIPPAGE_FROM_L1_PCT`]) — как y_train / [`book_fill_buy_strict`].
/// Легаси: `book_asks` пуст → L1–L3 фичи.
fn book_fill_buy(
    frame: &XFrame<SIZE>,
    position_size: f64,
    slippage_cap: Option<f64>,
) -> Option<(f64, f64)> {
    if position_size <= 0.0 {
        return None;
    }
    let fallback_asks;
    let asks: &[BookLevel] = match frame.book_asks.as_deref() {
        Some(asks) => asks,
        None => {
            fallback_asks = book_levels_from_legacy_l123([
                (frame.book_ask_l1_price, frame.book_ask_l1_size),
                (frame.book_ask_l2_price, frame.book_ask_l2_size),
                (frame.book_ask_l3_price, frame.book_ask_l3_size),
            ]);
            &fallback_asks
        }
    };
    let best_ask = asks
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)?;

    let mut remaining_usdc = position_size;
    let mut total_shares = 0.0_f64;
    for level in asks {
        if level.price <= 0.0 || level.size <= 0.0 { continue }
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
    if let Some(cap) = slippage_cap {
        if (vwap - best_ask) / best_ask > cap {
            return None;
        }
    }
    Some((vwap, total_shares))
}

/// Bid-walk на полный объём; `slippage_cap`: voluntary — cap vs best bid, urgent — только полный fill.
/// Симметрично y_train (неполный fill → нет выхода на тике). Легаси: L1–L3.
fn book_fill_sell(
    frame: &XFrame<SIZE>,
    shares_to_sell: f64,
    slippage_cap: Option<f64>,
) -> Option<f64> {
    if shares_to_sell <= 0.0 {
        return Some(0.0);
    }
    let fallback_bids;
    let bids: &[BookLevel] = match frame.book_bids.as_deref() {
        Some(bids) => bids,
        None => {
            fallback_bids = book_levels_from_legacy_l123([
                (frame.book_bid_l1_price, frame.book_bid_l1_size),
                (frame.book_bid_l2_price, frame.book_bid_l2_size),
                (frame.book_bid_l3_price, frame.book_bid_l3_size),
            ]);
            &fallback_bids
        }
    };
    let best_bid = bids
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)?;

    let mut remaining = shares_to_sell;
    let mut total_usdc = 0.0_f64;
    for level in bids {
        if level.price <= 0.0 || level.size <= 0.0 { continue }
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
    let vwap = total_usdc / shares_to_sell;
    if let Some(cap) = slippage_cap {
        if (best_bid - vwap) / best_bid > cap {
            return None;
        }
    }
    Some(total_usdc)
}

/// До трёх уровней из L1–L3, если в дампе нет полной лестницы ([`book_fill_buy`]/[`book_fill_sell`]).
fn book_levels_from_legacy_l123(levels: [(Option<f64>, Option<f64>); 3]) -> Vec<BookLevel> {
    let mut out = Vec::with_capacity(3);
    for (price_opt, size_opt) in levels {
        if let (Some(price), Some(size)) = (price_opt, size_opt) {
            if price > 0.0 && size > 0.0 && price.is_finite() && size.is_finite() {
                out.push(BookLevel { price, size });
            }
        }
    }
    out
}

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

pub(crate) fn print_side_stats(tag: &str, side_label: &str, s: &SideStats, is_kelly: bool) {
    let n = s.raw_above_threshold.max(1) as f64;
    let diag = if is_kelly {
        format!(
            "raw≥thr={} avg_raw={:.3} avg_cal={:.3} avg_entry={:.3} avg_kelly_f={:.4} kelly_skips={} entry_prob_skips={} same_asset_open_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={}",
            s.raw_above_threshold,
            s.diag_sum_raw / n,
            s.diag_sum_calibrated / n,
            s.diag_sum_entry_prob / n,
            s.diag_sum_kelly_f / n,
            s.kelly_skips,
            s.entry_prob_skips,
            s.same_asset_open_skips,
            s.kelly_strict_buy_skips,
            s.kelly_strict_sell_skips,
        )
    } else {
        format!(
            "raw≥thr={} avg_raw={:.3} avg_entry={:.3} entry_prob_skips={} same_asset_open_skips={} bankroll_too_small_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={}",
            s.raw_above_threshold,
            s.diag_sum_raw / n,
            s.diag_sum_entry_prob / n,
            s.entry_prob_skips,
            s.same_asset_open_skips,
            s.kelly_skips,
            s.kelly_strict_buy_skips,
            s.kelly_strict_sell_skips,
        )
    };
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
         | TP={} SL={} Timeout={} EvExit✓={} EvExit✗={} Res✓={}(profit={}/loss={}) Res✗={} late_skips={} unstable_skips={} same_asset_open_skips={}",
        s.trades, win_rate, s.pnl_usd, avg_pnl, s.fees_paid,
        s.tp_count, s.sl_count, s.timeout_count,
        s.ev_exit_profit_count, s.ev_exit_loss_count,
        s.resolution_win, s.resolution_win_profit, s.resolution_win_loss,
        s.resolution_loss, s.late_entry_skips, s.unstable_skips, s.same_asset_open_skips,
    );

    tee_println!(
        "[sim] {tag} [{side_label}] entry_prob hist (0..0.2 / 0.2..0.4 / 0.4..0.6 / 0.6..0.8 / 0.8..1): {} / {} / {} / {} / {}",
        s.histogram_entry_prob[0], s.histogram_entry_prob[1], s.histogram_entry_prob[2],
        s.histogram_entry_prob[3], s.histogram_entry_prob[4],
    );
    if is_kelly {
        tee_println!(
            "[sim] {tag} [{side_label}] cal_pred  hist (0..0.2 / 0.2..0.4 / 0.4..0.6 / 0.6..0.8 / 0.8..1): {} / {} / {} / {} / {}",
            s.histogram_cal_pred[0], s.histogram_cal_pred[1], s.histogram_cal_pred[2],
            s.histogram_cal_pred[3], s.histogram_cal_pred[4],
        );
    }

    let avg = |sum: f64, cnt: usize| if cnt == 0 { 0.0 } else { sum / cnt as f64 };
    tee_println!(
        "[sim] {tag} [{side_label}] pnl_by_reason: \
         TP={tp_pnl:+.2}$(avg={tp_avg:+.4}) SL={sl_pnl:+.2}$(avg={sl_avg:+.4}) \
         Timeout={to_pnl:+.2}$(avg={to_avg:+.4}) \
         EvExit✓={evp_pnl:+.2}$(avg={evp_avg:+.4}) EvExit✗={evl_pnl:+.2}$(avg={evl_avg:+.4}) \
         Res✓={rw_pnl:+.2}$(avg={rw_avg:+.4}) Res✗={rl_pnl:+.2}$(avg={rl_avg:+.4})",
        tp_pnl = s.pnl_tp,                tp_avg = avg(s.pnl_tp, s.tp_count),
        sl_pnl = s.pnl_sl,                sl_avg = avg(s.pnl_sl, s.sl_count),
        to_pnl = s.pnl_timeout,           to_avg = avg(s.pnl_timeout, s.timeout_count),
        evp_pnl = s.pnl_ev_exit_profit,   evp_avg = avg(s.pnl_ev_exit_profit, s.ev_exit_profit_count),
        evl_pnl = s.pnl_ev_exit_loss,     evl_avg = avg(s.pnl_ev_exit_loss, s.ev_exit_loss_count),
        rw_pnl = s.pnl_resolution_win,    rw_avg = avg(s.pnl_resolution_win, s.resolution_win),
        rl_pnl = s.pnl_resolution_loss,   rl_avg = avg(s.pnl_resolution_loss, s.resolution_loss),
    );
}

pub(crate) fn print_sim_stats(tag: &str, sim_stats: &SimStats, account: &Account, is_kelly: bool) {
    let total_trades = sim_stats.total_trades();
    if total_trades == 0 {
        if is_kelly {
            tee_println!(
                "[sim] {tag}: нет сделок ({} событий, kelly_skips={} entry_prob_skips={} same_asset_open_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={})",
                sim_stats.events,
                sim_stats.total_kelly_skips(),
                sim_stats.total_entry_prob_skips(),
                sim_stats.total_same_asset_open_skips(),
                sim_stats.total_kelly_strict_buy_skips(),
                sim_stats.total_kelly_strict_sell_skips(),
            );
        } else {
            tee_println!(
                "[sim] {tag}: нет сделок ({} событий, entry_prob_skips={} same_asset_open_skips={} bankroll_too_small_skips={})",
                sim_stats.events,
                sim_stats.total_entry_prob_skips(),
                sim_stats.total_same_asset_open_skips(),
                sim_stats.total_kelly_skips(),
            );
        }
        print_side_stats(tag, "UP",   &sim_stats.up,   is_kelly);
        print_side_stats(tag, "DOWN", &sim_stats.down, is_kelly);
        return;
    }

    let total_pnl = sim_stats.total_pnl();
    let total_wins = sim_stats.total_wins();
    let total_fees = sim_stats.total_fees();
    let win_rate = total_wins as f64 / total_trades as f64 * 100.0;
    let avg_pnl = total_pnl / total_trades as f64;
    let roi_pct = (account.bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100.0;

    let total_losses = sim_stats.total_losses();
    if is_kelly {
        tee_println!(
            "[sim] {tag} \
             | events={} trades={} win={:.1}% \
             | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
             | wins={total_wins} losses={total_losses} \
             | kelly_skips={ks} entry_prob_skips={eps} same_asset_open_skips={sas} kelly_strict_buy_skips={ksb} kelly_strict_sell_skips={kss}",
            sim_stats.events, total_trades, win_rate, total_pnl, avg_pnl, total_fees,
            ks = sim_stats.total_kelly_skips(),
            eps = sim_stats.total_entry_prob_skips(),
            sas = sim_stats.total_same_asset_open_skips(),
            ksb = sim_stats.total_kelly_strict_buy_skips(),
            kss = sim_stats.total_kelly_strict_sell_skips(),
        );
    } else {
        tee_println!(
            "[sim] {tag} \
             | events={} trades={} win={:.1}% \
             | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
             | wins={total_wins} losses={total_losses} \
             | entry_prob_skips={eps} same_asset_open_skips={sas} bankroll_too_small_skips={bts}",
            sim_stats.events, total_trades, win_rate, total_pnl, avg_pnl, total_fees,
            eps = sim_stats.total_entry_prob_skips(),
            sas = sim_stats.total_same_asset_open_skips(),
            bts = sim_stats.total_kelly_skips(),
        );
    }
    tee_println!(
        "[sim]   bankroll: {:.2}$ (start={INITIAL_BANKROLL}$) ROI={:+.2}% max_drawdown={:.2}%",
        account.bankroll, roi_pct, account.max_drawdown_pct,
    );

    print_side_stats(tag, "UP",   &sim_stats.up,   is_kelly);
    print_side_stats(tag, "DOWN", &sim_stats.down, is_kelly);
}

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

/// URL события из имени дампа `{stem}__{dump_ts_ms}.bin`: `event_end_ms = floor(ts/interval)×interval`, slug `{currency}-updown-{5m|15m}-{window_start_sec}`.
/// `None`, если парсинг/лаг ≥ интервала (floor попадает в следующее окно).
fn polymarket_event_url_from_dump_path(
    dump_file_path: &Path,
    currency: &str,
    interval_kind: XFrameIntervalKind,
) -> Option<String> {
    let bounds = window_bounds_from_dump_path(dump_file_path, interval_kind)?;
    let interval_label_str = match interval_kind {
        XFrameIntervalKind::FiveMin    => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    };
    Some(format!(
        "https://polymarket.com/event/{currency}-updown-{interval_label_str}-{window_start_sec}",
        currency = currency.to_lowercase(),
        window_start_sec = bounds.window_start_sec,
    ))
}

/// Границы окна из `...__{dump_ts_ms}.bin`: `event_end_ms = floor(ts/interval)×interval`, лаг ∈ `[0, interval)`.
/// Общая логика для URL, CSV unix_ms и миграций (`price_to_beat`).
pub(crate) fn window_bounds_from_dump_path(
    dump_file_path: &Path,
    interval_kind: XFrameIntervalKind,
) -> Option<DumpWindowBounds> {
    let stem = dump_file_path.file_stem()?.to_str()?;
    let ts_part = stem.rsplit("__").next()?;
    let dump_ts_ms: i64 = ts_part.parse().ok()?;
    let interval_ms = interval_kind.interval_ms();
    let event_end_ms = (dump_ts_ms / interval_ms) * interval_ms;
    let lag_ms = dump_ts_ms - event_end_ms;
    if !(0..interval_ms).contains(&lag_ms) {
        return None;
    }
    Some(DumpWindowBounds {
        window_start_sec: (event_end_ms - interval_ms) / 1_000,
        event_end_ms,
    })
}

/// Из [`window_bounds_from_dump_path`]: левая граница (sec) и резолюция (UTC ms).
pub(crate) struct DumpWindowBounds {
    /// Левая граница окна (UTC, секунды). `Polymarket window_start_sec`.
    pub window_start_sec: i64,
    /// Правая граница окна (UTC, миллисекунды). Момент резолюции
    /// маркета — он же начало следующего окна.
    pub event_end_ms: i64,
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
