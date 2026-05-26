//! История: дампы [`crate::xframe_dump::MarketXFramesDump`], синхронный проход UP/DOWN, виртуальные сделки.
//! Бинарный рынок: UP+DOWN ≈ 1; победа токена → $1/шер. Crypto fee: `fee ∝ p(1−p)` ([Fees](https://docs.polymarket.com/trading/fees)).
//! Логика: Kelly/gates, выход TP/SL/timeout/EV или резолюция (`calc_y_train_pnl`).

use crate::account::{Account, SharedAccount};
use crate::project_manager::ProjectManager;
use crate::account_order::{
    InvokeSettlementWatch, OrderAmount, invoke_settlement_ready, invoke_settlement_report,
    wait_invoke_settlement,
};
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::real_sim::interval_label;
use crate::train_mode::{
    collect_bin_paths, load_calibration, split_counts,
    Calibration, PNL_MAX_LAG, RESOLUTION_MAX_LAG, TEST_FRACTION, VAL_FRACTION,
};
use crate::xframe::{
    BookLevel, SIZE, XFrame, Y_TRAIN_NO_TRADE_PROB_HIGH, Y_TRAIN_NO_TRADE_PROB_LOW,
    Y_TRAIN_SL_MIN_REF_SELL_REL_DROP, Y_TRAIN_STOP_LOSS_PP, Y_TRAIN_TAKE_PROFIT_PP,
    apply_side_symmetry,
};
use crate::xframe_dump::MarketXFramesDump;
use crate::{tee_eprintln, tee_println};

pub use crate::sim_stats::{
    print_side_stats, print_sim_stats, SideStats, SimStats, SimStatsLogSink,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use xgb::{Booster, DMatrix};

/// Нижний порог raw перед Kelly (`f* > 0`).
pub const SIM_BUY_THRESHOLD: f32 = 0.60;

/// Max отклонение VWAP от L1 при strict fill; voluntary TP может обойти cap
/// ([`sell_gate`], [`crate::account_close_position::gross_usdc_sell_take_profit`]).
pub const SIM_MAX_SLIPPAGE_FROM_L1_PCT: f64 = 0.02;

/// Стартовый банкролл (USDC).
pub const INITIAL_BANKROLL: f64 = 50.0;
/// Доля Kelly (<1 — fractional).
pub const KELLY_MULTIPLIER: f64 = 0.1;
/// Max доля банкролла на сделку.
pub const MAX_BET_FRACTION: f64 = 0.10;
/// Min размер позиции (USDC).
pub const MIN_POSITION_USD: f64 = 0.01;

/// Жёсткий cap USDC на сделку (поверх `MAX_BET_FRACTION × bankroll`).
pub const MAX_POSITION_USD: f64 = 300.0;

/// Размер входа в `run_sim_mode` при `is_kelly=false` (фикс, без калибровки).
pub const NO_KELLY_POSITION_SIZE_USD: f64 = 30.0;

/// `true` — не считать SHAP-топ5 для CSV (экономия CPU).
pub const HISTORY_SIM_SKIP_TRADE_SHAP_CONTRIBUTIONS: bool = false;

/// Множитель в crypto taker fee: `fee ∝ rate × p × (1−p)` ([Fees](https://docs.polymarket.com/trading/fees)).
pub const POLYMARKET_CRYPTO_TAKER_FEE_RATE: f64 = 0.07;

/// Порог секунд до конца окна = hold-zone (TP/timeout off; SL + EV-exit).
pub const HOLD_TO_END_THRESHOLD_SEC: i64 = 0;

/// α EMA для `p_win` в hold-zone.
pub const EV_EXIT_P_WIN_EMA_ALPHA: f64 = 0.3;

/// Зазор EV: `EV_sell × (1 − margin) > EV_hold`.
pub const EV_EXIT_MARGIN: f64 = 0.01;

/// Кадров без TP/SL → Timeout (как горизонт в xframe train).
pub const POSITION_TIMEOUT_FRAMES: usize = 30;

/// Мин. кадров удержания до проверки SL/TP/EV в history_sim; в [`crate::real_sim`] передают `None`.
pub const MINPOSITION_FRAMES: usize = 2;

/// Одна активная позиция на `asset_id`; синхронно с калибровкой [`crate::train_mode::first_entry_calibration_samples`].
pub const BLOCK_SAME_ASSET_OPEN: bool = false;

/// Max одновременных позиций в лейне (`positions.len()`); `None` — без лимита.
pub const MAX_OPEN_POSITIONS: Option<usize> = Some(1);

/// Запас к CLOB `min_order_size` при расчёте BUY notional: `min_shares × price_cap × (1 + pct)`.
pub const MIN_ORDER_SIZE_BUY_HEADROOM_PCT: f64 = 0.05;

/// Min `event_remaining_ms` для входа ([`BuyGate::LateEntry`]).
pub const MIN_ENTRY_REMAINING_MS: i64 = 10 * 1000;

/// Стоп новых входов при DD ≥ pct (`real_sim` только).
pub const EMERGENCY_HALT_DRAWDOWN_PCT: Option<f64> = Some(30.0);

/// HTTP стакан для strict fill в `real_sim`: `None` в history (WS [`book_fill_*`]).
#[derive(Debug, Clone, Default)]
pub struct StrictBook {
    /// Bids, лучший первый.
    pub(crate) bids: Vec<BookLevel>,
    /// Asks, лучший первый.
    pub(crate) asks: Vec<BookLevel>,
    /// Last trade (широкий спред → как polymarket-style mid).
    pub(crate) last_trade_price: Option<f64>,
    /// Min размер ордера в шерах (strict).
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

/// Min USDC notional для strict BUY: `min_order_size × (best_ask + slippage) × (1 + headroom)`.
pub(crate) fn min_order_size_buy_usd_floor(book: &StrictBook) -> Option<f64> {
    let min_shares = book.min_order_size.filter(|m| m.is_finite() && *m > 0.0)?;
    let best_ask = book
        .asks
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)?;
    let price_cap = (best_ask + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999);
    Some(min_shares * price_cap * (1.0 + MIN_ORDER_SIZE_BUY_HEADROOM_PCT))
}

/// Strict BUY notional: не ниже [`min_order_size_buy_usd_floor`] (как live-тесты duel/roundtrip).
pub(crate) fn effective_buy_usdc_strict(book: &StrictBook, position_size: f64) -> f64 {
    match min_order_size_buy_usd_floor(book) {
        Some(floor) => position_size.max(floor),
        None => position_size,
    }
}

/// Покупка по HTTP asks: полный fill `position_size`, cap от L1 ask, опционально `min_order_size`.
pub(crate) fn book_fill_buy_strict(book: &StrictBook, position_size: f64) -> Option<(f64, f64)> {
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

/// Продажа по HTTP bids: gross USDC до fee; `Some(cap)` — voluntary (TP/EvProfit), `None` — urgent (SL/timeout/…).
pub(crate) fn book_fill_sell_strict(
    book: &StrictBook,
    shares_to_sell: f64,
    slippage_cap: Option<f64>,
) -> Option<f64> {
    if shares_to_sell <= 0.0 {
        return Some(0.0);
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

/// Один `Arc<RwLock<OpenPosition>>` везде ([`crate::account::Account`]; max один inner-lock за операцию).
pub type SharedOpenPosition = std::sync::Arc<tokio::sync::RwLock<OpenPosition>>;

/// То же для записи закрытия.
///
/// `OpenPosition` держит **сильные** [`SharedClosingPosition`]
/// ([`OpenPosition::maker_tp_position`] / [`OpenPosition::taker_positions`]) —
/// `ClosingPosition` **не** хранит обратной ссылки на [`SharedOpenPosition`],
/// поэтому цикла strong↔strong нет; запись закрытия живёт ровно столько,
/// сколько жива сама `OpenPosition`.
pub type SharedClosingPosition = std::sync::Arc<tokio::sync::RwLock<ClosingPosition>>;

/// Открытые позиции одной лейны в [`crate::account::Account::positions`]; ключ —
/// [`OpenPosition::id`].
pub type LanePositions = HashMap<String, SharedOpenPosition>;

/// Открытая позиция; в real_sim фильтр `asset_id == frame.asset_id`.
#[derive(Debug, Clone)]
pub struct OpenPosition {
    /// Локальный uuid логов; не путать с CLOB order ids.
    pub(crate) id: String,
    /// Gamma outcome asset id.
    pub(crate) asset_id: String,
    /// Condition id маркета (Gamma).
    #[allow(dead_code)]
    pub(crate) market_id: String,
    /// Фактически купленные шеры (after fee). В backtest = `planned_shares_held`
    /// (виртуальный fill). В submit: при создании = `planned_shares_held`; после
    /// успешного `buy_rep` обновляется на `buy_rep.taking_amount`
    /// ([`crate::account_submit::spawn_open_buy_taker`]).
    pub(crate) shares_held: f64,
    /// План шер: расчётное от Kelly/SIM_*. Никогда не меняется после создания.
    pub(crate) planned_shares_held: f64,
    /// Prob на входе (legacy); решения по [`Self::buy_price`].
    #[allow(dead_code)]
    pub(crate) entry_prob: f64,
    /// Фактический VWAP входа. В backtest = `planned_buy_price`. В submit: при
    /// создании = `planned_buy_price`; после `buy_rep` — `buy_rep.making_amount /
    /// buy_rep.taking_amount`.
    pub(crate) buy_price: f64,
    /// План: best-ask на момент решения. Никогда не меняется после создания.
    pub(crate) planned_buy_price: f64,
    /// Ref voluntary sell VWAP на входе (SL vs [`crate::xframe::Y_TRAIN_SL_MIN_REF_SELL_REL_DROP`]).
    pub(crate) sell_vwap_entry: f64,
    /// Фактически потраченные USDC (entry cost). В backtest = `planned_entry_cost`.
    /// В submit: при создании = `planned_entry_cost`; после `buy_rep` —
    /// `buy_rep.making_amount` (UsdNotional).
    pub(crate) position_size: f64,
    /// План: целевые USDC от Kelly (входной `amount`). Никогда не меняется после создания.
    pub(crate) planned_entry_cost: f64,
    /// L1 bid на входе (maker TP в [`crate::account_close_position::close_position`]).
    pub(crate) best_bid_at_entry: Option<f64>,
    /// Кадров удержания ([`POSITION_TIMEOUT_FRAMES`]).
    pub(crate) frames_held: usize,
    /// EMA `p_win` resolution в hold-zone.
    pub(crate) p_win_ema: Option<f64>,
    /// CSV: raw pred на входе.
    pub(crate) raw_pred_at_open: f32,
    /// CSV: calibrated pred на входе.
    pub(crate) cal_pred_at_open: f32,
    /// CSV: Kelly f на входе.
    pub(crate) kelly_f_at_open: f64,
    /// Оставшееся время события на входе (мс).
    pub(crate) event_remaining_ms_at_open: i64,
    /// Интервал лейна на входе (discriminant).
    pub(crate) xframe_interval_type_at_open: i32,
    /// Сторона UP/DOWN на входе (discriminant).
    pub(crate) currency_up_down_outcome_at_open: i32,
    /// Тикер актива.
    pub(crate) currency: String,
    /// URL рынка PM.
    pub(crate) polymarket_url: String,
    /// Порог цены для CSV.
    pub(crate) price_to_beat: Option<f64>,
    /// Финальная цена окна для CSV.
    pub(crate) final_price: Option<f64>,
    /// Конец окна UTC (мс), unix-колонки CSV.
    pub(crate) event_end_ms: Option<i64>,
    /// Путь `.bin` для графика в CSV; в real_sim может быть синтетический.
    pub(crate) graph_dump_bin_path: String,
    /// Fallback stem из Gamma question если путь пуст.
    pub(crate) gamma_question_at_open: Option<String>,
    /// Текст SHAP топ-5 для CSV.
    pub(crate) pnl_top5_shap_at_open: String,
    /// Статус ордера на открытие ([`OpenPositionStatus`]); виртуально сразу `Open`.
    /// ID BUY-ордера CLOB из user-WS; `None` если виртуально.
    pub(crate) open_order_id: Option<String>,
    /// Taker BUY invoke: `None` до POST; затем [`InvokeSettlementWatch`] (`Some(report)` после колбэка).
    pub(crate) open_buy_invoke: Option<InvokeSettlementWatch>,
    pub(crate) maker_tp_position: Option<SharedClosingPosition>,
    /// Taker FAK SELL: по одной записи на каждый успешный invoke ([`crate::account_submit`]).
    pub(crate) taker_positions: Vec<SharedClosingPosition>,
    /// Идемпотентность [`crate::account_close_position::close_position_after_submit`]:
    /// взводится в `true` при первом входе (под `position.write().await`) и блокирует
    /// повторные вызовы (CSV/SideStats/bankroll/graph должны записаться ровно один раз).
    /// Maker-TP callback в [`crate::account_submit::spawn_open_buy_taker`] и taker-FAK
    /// callback в [`crate::account_submit::spawn_sell_taker`] могут гоняться за один и
    /// тот же fully-closed PNL — флаг гарантирует, что финализирует только победитель.
    pub(crate) close_after_submit_finalized: bool,
    /// Фактически удержанная BUY-fee в USDC, которая уже учтена в
    /// [`crate::sim_stats::SideStats::fees_paid`] лейна. В backtest (`SubmitMode::None`)
    /// = [`Self::planned_fee_usdc`] (виртуальный fill идентичен плану). При первом
    /// создании позиции (real_sim) — тоже план; после settle BUY
    /// [`crate::account_submit::spawn_open_buy_taker`] заменяет его на actual из
    /// mock/CLOB-fill (через
    /// [`crate::account_order::SingleOrderClobInvocationReport::fee_paid_usdc`]):
    /// `delta = actual − stored` → правка `stats.fees_paid`, новое значение
    /// перезаписывает поле. Гарантирует, что суммарный entry-fee в SideStats бьётся
    /// с реально удержанной CLOB'ом fee (а не с «плановым стаканом кадра», по
    /// которому делалось решение).
    pub(crate) entry_fee_usdc: f64,
    /// Плановая BUY-fee в USDC: посчитана [`crate::history_sim::open_position`] по
    pub(crate) planned_fee_usdc: f64,
}

/// Возврат [`OpenPosition::shares_remaining_to_sell`] при не-settled invoke-колбэке:
/// при `block_on_pending_invokes=false` любая нога без settled-колбэка немедленно
/// даёт эту ошибку; при `=true` — только если [`wait_invoke_settlement`] ушёл в
/// таймаут (по умолчанию `event_end_ms` + `ORDER_HTTP_TIMEOUT_SEC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvokePendingError {
    /// Какая нога не settled: `"open_buy"` / `"maker_tp"` / `"taker_sell"`.
    pub which: &'static str,
}

impl std::fmt::Display for InvokePendingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invoke not settled: {}", self.which)
    }
}

impl std::error::Error for InvokePendingError {}

impl OpenPosition {
    /// Сколько ещё нужно продать шеров (raw `f64`, **без округления** — caller
    /// сам `floor`'ит до CLOB-lot'а 0.01); читает [`Self::open_buy_invoke`] и
    /// invoke-каналы `maker_tp_position` + `taker_positions`.
    pub async fn shares_remaining_to_sell(
        &self,
        block_on_pending_invokes: bool,
    ) -> Result<Option<f64>, InvokePendingError> {
        let resolve_report = async |watch_opt: Option<InvokeSettlementWatch>, which: &'static str|-> Result<Option<crate::account_order::SingleOrderClobInvocationReport>, InvokePendingError> {
            let Some(mut watch) = watch_opt else {
                return Ok(None);
            };
            if invoke_settlement_ready(&watch) {
                return Ok(invoke_settlement_report(&watch));
            }
            if !block_on_pending_invokes {
                return Err(InvokePendingError { which });
            }
            let timeout = crate::account_submit::invoke_wait_until_market_end_plus(self.event_end_ms);
            match wait_invoke_settlement(&mut watch, timeout).await {
                Some(report) => Ok(Some(report)),
                None => Err(InvokePendingError { which }),
            }
        };

        // BUY invoke → shares_bought_net.
        let Some(buy_report) = resolve_report(self.open_buy_invoke.clone(), "open_buy").await? else {
            return Ok(None);
        };
        if !buy_report.success {
            return Ok(None);
        }
        let shares_bought_net = match buy_report.taking_amount {
            OrderAmount::Shares(s) if s.is_finite() && s > 0.0 => s,
            _ => return Ok(None),
        };

        let mut shares_sold = 0.0_f64;

        // maker TP (если есть): settled+success → making_amount как shares.
        if let Some(arc) = self.maker_tp_position.as_ref() {
            let watch_opt = {
                let closing = arc.read().await;
                closing.invoke_settle.clone()
            };
            if let Some(report) = resolve_report(watch_opt, "maker_tp").await? {
                if report.success {
                    if let OrderAmount::Shares(s) = report.making_amount {
                        if s.is_finite() && s > 0.0 {
                            shares_sold += s;
                        }
                    }
                }
            }
        }

        // taker FAK SELL'ы: сумма making_amount по settled+success invoke'ам.
        for arc in &self.taker_positions {
            let watch_opt = {
                let closing = arc.read().await;
                closing.invoke_settle.clone()
            };
            let Some(report) = resolve_report(watch_opt, "taker_sell").await? else {
                continue;
            };
            if !report.success {
                continue;
            }
            if let OrderAmount::Shares(s) = report.making_amount {
                if s.is_finite() && s >= 0.0 {
                    shares_sold += s;
                }
            }
        }

        Ok(Some((shares_bought_net - shares_sold).max(0.0)))
    }
}

/// Запись закрытия для WS/polling ([`manage_positions`], [`crate::account::apply_user_ws_event`]).
///
/// Не хранит ссылку на [`SharedOpenPosition`]: parent `OpenPosition` сам держит
/// сильные [`SharedClosingPosition`] в своих полях
/// ([`OpenPosition::maker_tp_position`] / [`OpenPosition::taker_positions`]) —
/// обратная ссылка дала бы strong-cycle.
#[derive(Debug, Clone)]
pub struct ClosingPosition {
    /// Причина (как в CSV).
    pub reason: CloseReason,
    /// ID SELL на CLOB; `None` в sim или пока не создан.
    pub order_id: Option<String>,
    /// SELL invoke: `None` до POST; затем [`InvokeSettlementWatch`] (`Some(report)` после колбэка).
    pub invoke_settle: Option<InvokeSettlementWatch>,
    /// Был вызван [`crate::account_order::cancel_order_on_clob`] для этой записи.
    pub canceled: bool,
    /// UTC ms создания записи (TTL/диагностика).
    pub created_unix_ms: i64,
}

/// Выход до резолюции; иначе см. [`crate::account::Account::resolve_pending_market_sync`].
#[derive(Debug, Clone, PartialEq)]
pub enum CloseReason {
    /// TP по [`crate::xframe::Y_TRAIN_TAKE_PROFIT_PP`].
    TakeProfit,
    /// SL по ref-VWAP правилу.
    StopLoss,
    /// Удержание ≥ [`POSITION_TIMEOUT_FRAMES`].
    Timeout,
    /// Hold-zone: EV продажи > вложения.
    EvExitProfit,
    /// Hold-zone: продать выгоднее ждать резолюцию, цена ниже входа.
    EvExitLoss,
    /// Резолюция рынка: наша сторона выиграла ($1/шер payout). Производит
    /// [`crate::account::Account::resolve_pending_market_sync`].
    ResolutionWin,
    /// Резолюция рынка: наша сторона проиграла ($0/шер payout). Производит
    /// [`crate::account::Account::resolve_pending_market_sync`].
    ResolutionLoss,
}

impl CloseReason {
    /// TP — допускает отложенный выход при глубоком slippage ([`SIM_MAX_SLIPPAGE_FROM_L1_PCT`]).
    pub fn is_voluntary_exit(&self) -> bool {
        matches!(self, CloseReason::TakeProfit)
    }
}

/// Два прогона: `kelly` и `raw` ([`NO_KELLY_POSITION_SIZE_USD`]); колонка CSV `regime`; отдельный [`Account`] на режим.
pub async fn run_sim_mode() -> anyhow::Result<()> {
    let xframes_root = Path::new("xframes");
    if !xframes_root.exists() {
        anyhow::bail!("Папка xframes/ не найдена — сначала соберите данные (STATUS=default)");
    }

    crate::tee_log::init_tee_log_file(&xframes_root.join("last_history_sim.txt"))?;
    crate::trade_csv_log::init_trade_csv_log_file(
        &xframes_root.join("last_history_sim_trades.csv"),
    )?;

    crate::trade_csv_log::set_current_regime("kelly");
    tee_println!("[sim] === regime=kelly (Kelly + calibration, min(MAX_BET_FRACTION × bankroll, MAX_POSITION_USD)) ===");
    run_sim_mode_inner(true).await?;

    crate::trade_csv_log::set_current_regime("raw");
    tee_println!("[sim] === regime=raw (no Kelly, no calibration, ${NO_KELLY_POSITION_SIZE_USD} entry) ===");
    run_sim_mode_inner(false).await?;

    crate::trade_csv_log::set_current_regime("");
    crate::trade_csv_log::finish_trade_csv_log();
    crate::tee_log::finish_tee_log();

    Ok(())
}

/// Один режим `is_kelly`; свой свежий [`Account::new()`].
async fn run_sim_mode_inner(is_kelly: bool) -> anyhow::Result<()> {
    let xframes_root = Path::new("xframes");
    let regime_label = if is_kelly { "kelly" } else { "raw" };

    for currency_path in fs_sorted_dirs(xframes_root)? {
        let currency = dir_name(&currency_path);

        for version_path in fs_sorted_dirs(&currency_path)? {
            let version = dir_name(&version_path);
            if version.parse::<usize>().is_err() {
                continue;
            }

            let account = Account::new_shared();

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

                let booster_resolution_up = load_booster(&model_resolution_up_path);
                let booster_resolution_down = load_booster(&model_resolution_down_path);
                let calibration_resolution_up   = load_calibration(&model_resolution_up_path).ok();
                let calibration_resolution_down = load_calibration(&model_resolution_down_path).ok();

                let cal_info = |cal: &Option<Calibration>, label: &str| -> String {
                    match cal {
                        Some(c) => format!(
                            "{label}=✓(breakpoints={} | 0.7→{:.3} 0.8→{:.3} 0.9→{:.3})",
                            c.xs.len(),
                            c.apply(0.7),
                            c.apply(0.8),
                            c.apply(0.9),
                        ),
                        None => format!("{label}=✗"),
                    }
                };

                let step_path = interval_path.join("1s");
                let all_paths = collect_bin_paths(&step_path)?;
                let (train_count, val_count, test_count) = split_counts(all_paths.len());
                let test_paths = &all_paths[train_count + val_count..];

                let test_period_str = test_period_label(test_paths, interval_kind);

                if is_kelly {
                    tee_println!(
                        "[sim] {tag}: модели pnl загружены | {} | {} \
                         | resolution: up={} down={} \
                         | hold_zone≤{HOLD_TO_END_THRESHOLD_SEC}s ev_margin={EV_EXIT_MARGIN} ema_α={EV_EXIT_P_WIN_EMA_ALPHA} \
                         | threshold={SIM_BUY_THRESHOLD} | kelly={KELLY_MULTIPLIER} | max_bet={MAX_BET_FRACTION} | max_pos=${MAX_POSITION_USD} \
                         | no_trade_zone=({Y_TRAIN_NO_TRADE_PROB_LOW}..{Y_TRAIN_NO_TRADE_PROB_HIGH}) \
                         | bankroll={INITIAL_BANKROLL}$ | fee_rate={POLYMARKET_CRYPTO_TAKER_FEE_RATE} \
                         | {test_period_str}",
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
                         | bankroll={INITIAL_BANKROLL}$ | fee_rate={POLYMARKET_CRYPTO_TAKER_FEE_RATE} \
                         | {test_period_str}",
                        if booster_resolution_up.is_some()   { "✓" } else { "✗" },
                        if booster_resolution_down.is_some() { "✓" } else { "✗" },
                    );
                }

                let mut sim_stats = SimStats::new();

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
                                booster_resolution_up.as_ref(),
                                booster_resolution_down.as_ref(),
                                calibration_resolution_up.as_ref(),
                                calibration_resolution_down.as_ref(),
                                &mut sim_stats,
                                &account,
                                is_kelly,
                                &polymarket_url,
                                event_end_ms,
                                file_path.as_path(),
                            )
                            .await;
                            sim_stats.events += 1;
                        }
                        Err(err) => tee_eprintln!("[sim] {}: {err}", file_path.display()),
                    }
                }

                let bankroll_now = *account.bankroll.read().await;
                let max_drawdown_pct_now = *account.max_drawdown_pct.read().await;
                print_sim_stats(
                    &tag,
                    &sim_stats,
                    bankroll_now,
                    max_drawdown_pct_now,
                    is_kelly,
                    INITIAL_BANKROLL,
                    SimStatsLogSink::Tee,
                );
            }
        }
    }

    Ok(())
}

/// Один маркет: UP и DOWN по отдельным рядам кадров; общий [`Account`] как в [`crate::real_sim`].
#[allow(clippy::too_many_arguments)]
async fn simulate_event(
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
    account: &SharedAccount,
    is_kelly: bool,
    polymarket_url: &str,
    event_end_ms: Option<i64>,
    bin_dump_path: &std::path::Path,
) {
    let graph_dump_bin_path = bin_dump_path.to_string_lossy().into_owned();
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

    run_side_simulation(
        &frames_up,
        booster_up,
        calibration_up,
        booster_resolution_up,
        calibration_resolution_up,
        account,
        &lane_key_up,
        &mut sim_stats.up,
        currency,
        is_kelly,
        polymarket_url,
        price_to_beat,
        final_price,
        event_end_ms,
        &graph_dump_bin_path,
        HOLD_TO_END_THRESHOLD_SEC,
    )
    .await;
    run_side_simulation(
        &frames_down,
        booster_down,
        calibration_down,
        booster_resolution_down,
        calibration_resolution_down,
        account,
        &lane_key_down,
        &mut sim_stats.down,
        currency,
        is_kelly,
        polymarket_url,
        price_to_beat,
        final_price,
        event_end_ms,
        &graph_dump_bin_path,
        HOLD_TO_END_THRESHOLD_SEC,
    )
    .await;

    if let Some(market_id) = market_id_opt {
        crate::account::Account::resolve_pending_market_sync(
            &account,
            sim_stats,
            currency,
            interval_kind,
            &market_id,
            up_won,
            None,
        )
        .await;
    }


}

/// Один проход стороны (UP/DOWN) по ряду кадров: manage/open → MtM equity.
/// Живые позиции — `account.positions[lane_key]` (source-of-truth, как в [`crate::real_sim`]);
/// финальный payout по ним делает caller через
/// [`crate::account::Account::resolve_pending_market_sync`] (см. [`simulate_event`]).
///
/// Equity: `bankroll + Σ(shares×prob)` по **всем** лейнам в `positions` и
/// `pending_close_positions` (для текущего `lane_key` — `prob` из кадра, для
/// остальных — fallback на `account.last_prob` / `0.5`), как в
/// [`crate::real_sim::tick_once`]. `pending_close_positions` в `SubmitMode::None`
/// всегда пуст, но читаем его «честно для симметрии».
/// Сайзинг от `bankroll − Σ(entry_cost across all lanes & pending_close)` на этой стороне.
///
/// `hold_to_end_threshold_sec` — окно, в котором применяется resolution-модель
/// и собираются точки для её калибровки (см. [`compute_p_win_now`] и
/// [`SideStats::hold_zone_resolution_predictions`]).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_side_simulation(
    frames: &[&XFrame<SIZE>],
    booster_pnl: &Booster,
    calibration_pnl: Option<&Calibration>,
    booster_resolution: Option<&Booster>,
    calibration_resolution: Option<&Calibration>,
    account: &SharedAccount,
    lane_key: &(String, XFrameIntervalKind, CurrencyUpDownOutcome),
    side_stats: &mut SideStats,
    currency: &str,
    is_kelly: bool,
    polymarket_url: &str,
    price_to_beat: Option<f64>,
    final_price: Option<f64>,
    event_end_ms: Option<i64>,
    graph_dump_bin_path: &str,
    hold_to_end_threshold_sec: i64,
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
            hold_to_end_threshold_sec,
        );
        // Sim-replay калибровки `ModelType::Resolution`: для каждого hold-zone
        // кадра, на котором есть raw-предсказание, копим его в side_stats.
        // Гейт `calibration_resolution.is_none()` гарантирует, что:
        //  (а) production sim сюда не зайдёт (там cal Some после успешной загрузки);
        //  (б) `p_win_now` уже **сырой** скор, а не калиброванный (см.
        //      `compute_p_win_now`: при cal=None возвращает `raw as f64`).
        if calibration_resolution.is_none() {
            if let Some(p) = p_win_now {
                side_stats.hold_zone_resolution_predictions.push(p as f32);
            }
        }
        let pnl_inference = compute_pnl_inference(frame, booster_pnl, calibration_pnl, is_kelly);

        // Фаза 1: manage_positions. `bankroll` пред-захватывать не нужно —
        // [`crate::account_close_position::close_position`] сам берёт
        // `account.bankroll.write().await` под коротким локом для apply'я PNL.
        // Держим только `account.positions.write()` (drain внутри iter'а).
        {
            let mut positions = account.positions.write().await;
            let positions_v = positions.entry(lane_key.clone()).or_default();
            manage_positions(
                positions_v,
                frame,
                is_last_idx,
                p_win_now,
                side_stats,
                None,
                Some(MINPOSITION_FRAMES),
                crate::account_submit::SubmitMode::None,
                None,
                account,
                lane_key,
            )
            .await;
        }

        // Фаза 2: try_open_position. `available` — bankroll минус **весь**
        // locked-капитал по всем lane'ам в `positions` и `pending_close_positions`
        // (как в [`crate::real_sim::tick_once`]). Bankroll один общий, поэтому
        // и кросс-lane (другие currency/interval/side в той же сессии), и
        // pending-close-bucket тоже тратят его. В backtest (`SubmitMode::None`)
        // `pending_close_positions` всегда пуст, но честно читаем для
        // симметрии. Lock-order: bankroll → positions → pending_close_positions.
        // Передаём оба HashMap'а под взятыми lock'ами — гейты
        // [`MAX_OPEN_POSITIONS`] / [`BLOCK_SAME_ASSET_OPEN`] считаются внутри
        // [`try_open_position`].
        {
            let bankroll = account.bankroll.read().await;
            let mut positions = account.positions.write().await;
            let pending_close = account.pending_close_positions.read().await;
            // Гарантируем существование bucket'а текущей lane до подсчёта,
            // чтобы [`try_open_position`] могла вставить позицию через
            // `entry(...).or_default()` без расхождения counts.
            positions.entry(lane_key.clone()).or_default();
            let total_locked = {
                let mut sum = 0.0;
                for lane_positions in positions.values() {
                    for p in lane_positions.values() {
                        sum += p.read().await.position_size;
                    }
                }
                for lane_pending in pending_close.values() {
                    for p in lane_pending.values() {
                        sum += p.read().await.position_size;
                    }
                }
                sum
            };
            let available = (*bankroll - total_locked).max(0.0);
            try_open_position(
                frame,
                pnl_inference,
                Some(booster_pnl),
                &mut *positions,
                &*pending_close,
                lane_key,
                side_stats,
                available,
                None,
                currency,
                is_kelly,
                polymarket_url,
                price_to_beat,
                final_price,
                event_end_ms,
                graph_dump_bin_path,
                None,
                None,
                crate::account_submit::SubmitMode::None,
                None,
                account,
            )
            .await;
        }

        // MtM equity (как [`crate::real_sim::tick_once`]): без prob на кадре
        // тик пропускаем. Суммируем **все** позиции в `positions` и
        // `pending_close_positions` (по всем lane'ам, кросс-currency/interval/side),
        // а не только текущую лейну — bankroll один общий и DD должен видеть
        // всю книгу. Для текущего `lane_key` используем `prob` из кадра, для
        // остальных — `account.last_prob` (или 0.5 при отсутствии записи, как
        // в real_sim). В backtest (`SubmitMode::None`) `pending_close_positions`
        // всегда пуст, но читаем его «честно для симметрии».
        // Lock-order: bankroll → last_prob → positions → pending_close_positions.
        if let Some(prob) = frame.currency_implied_prob {
            let prob = prob.clamp(0.0, 1.0);
            let equity = {
                let bankroll = account.bankroll.read().await;
                let last_prob_guard = account.last_prob.read().await;
                let positions = account.positions.read().await;
                let pending_close = account.pending_close_positions.read().await;
                let mut positions_value = 0.0;
                for (key, lane_positions) in positions.iter().chain(pending_close.iter()) {
                    let prob_raw = if key == lane_key {
                        prob
                    } else {
                        last_prob_guard.get(key).copied().unwrap_or(0.5)
                    };
                    let prob_use = if prob_raw.is_finite() {
                        prob_raw.clamp(0.001, 0.999)
                    } else {
                        0.5
                    };
                    for p in lane_positions.values() {
                        positions_value += p.read().await.shares_held * prob_use;
                    }
                }
                *bankroll + positions_value
            };
            account.update_drawdown(equity).await;
        }
    }

    // Хвост открытых позиций остаётся в `account.positions[lane_key]`; финальный
    // payout делает caller через `Account::resolve_pending_market_sync`, который
    // дренирует их по `market_id` напрямую из `positions`.
}

/// Сырой (`raw`) и калиброванный (`pred`) скор PnL; см. [`compute_pnl_inference`].
#[derive(Clone, Copy, Debug)]
pub struct PnlInference {
    /// Raw booster до порога [`SIM_BUY_THRESHOLD`].
    pub raw: f32,
    /// Для Kelly — после калибровки; иначе совпадает с `raw`.
    pub pred: f32,
}

/// Infеренс PnL на кадр; `None`: поздний вход / unstable / нет prob / лаг > [`PNL_MAX_LAG`]. Калибровка здесь, не в [`buy_gate`].
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

/// P(win) resolution в hold-zone; `None` вне неё / нет модели / лаг > [`RESOLUTION_MAX_LAG`].
/// `hold_to_end_threshold_sec`: prod — [`HOLD_TO_END_THRESHOLD_SEC`]; replay-калибровка может подставить своё ([`crate::train_mode`]).
pub(crate) fn compute_p_win_now(
    frame: &XFrame<SIZE>,
    booster_resolution: Option<&Booster>,
    calibration_resolution: Option<&Calibration>,
    is_kelly: bool,
    hold_to_end_threshold_sec: i64,
) -> Option<f64> {
    let in_hold_zone = frame.event_remaining_ms > 0
        && frame.event_remaining_ms <= hold_to_end_threshold_sec * 1000;
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
    /// Мало времени до резолюции или событие закончилось ([`MIN_ENTRY_REMAINING_MS`]).
    LateEntry,
    /// Кадр нестабилен ([`crate::xframe::compute_xframe_stable`]).
    Unstable,
    /// Нет инференса или raw < [`SIM_BUY_THRESHOLD`].
    BelowThreshold,
    /// Центральная no-trade зона по `entry_prob`; диагностика суммируется.
    EntryProbOutOfRange { raw: f32, pred: f32, kelly_f: f64 },
    /// После порога нет edge или размер < [`MIN_POSITION_USD`] (`kelly_skips`).
    KellySkip { raw: f32, pred: f32, kelly_f: f64 },
    /// Открыть на `size` USDC.
    Proceed { raw: f32, pred: f32, kelly_f: f64, size: f64 },
}

/// Дерево решений входа без побочных эффектов ([`BuyGate`]). Инференс снаружи — [`compute_pnl_inference`].
pub(crate) fn buy_gate(
    frame: &XFrame<SIZE>,
    pnl_inference: Option<PnlInference>,
    bankroll: f64,
    strict_book: Option<&StrictBook>,
    is_kelly: bool,
    _booster_pnl_for_shap: Option<&Booster>,
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

    // if frame.stable {
    //     println!("raw {}", raw);
    //     // if let Some(booster) = booster_pnl_for_shap {
    //     //     let shap = pnl_top5_shap_csv_cell(booster, frame);
    //     //     if !shap.is_empty() {
    //     //         println!("shap top5:\n{shap}");
    //     //     }
    //     // }
    //     // println!("frame {:?}", frame);
    //     if raw > 0.1 {
    //         if let Some(booster) = booster_pnl_for_shap {
    //             let shap = pnl_top5_shap_csv_cell(booster, frame);
    //             if !shap.is_empty() {
    //                 println!("shap top5:\n{shap}");
    //             }
    //         }
    //         println!("frame {:?}", frame);
    //     }
    // }

    if raw < 0.3 { //SIM_BUY_THRESHOLD
        return BuyGate::BelowThreshold;
    }

    let best_bid_at_entry = match strict_book {
        Some(book) => book.bids.first().map(|lvl| lvl.price),
        None => frame.book_bid_l1_price,
    };
    let kelly_gain = kelly_gain_ratio(entry_prob, best_bid_at_entry);
    let kelly_loss = kelly_loss_ratio(entry_prob);
    let kelly_f = kelly_fraction(pred as f64, kelly_gain, kelly_loss);

    // Хвосты распределения: вне центральной no-trade зоны ([`crate::xframe::calc_y_train_pnl`]).
    if entry_prob > Y_TRAIN_NO_TRADE_PROB_LOW && entry_prob < Y_TRAIN_NO_TRADE_PROB_HIGH {
        return BuyGate::EntryProbOutOfRange { raw, pred, kelly_f };
    }

    if !is_kelly {
        let size = NO_KELLY_POSITION_SIZE_USD.min(bankroll).max(0.0);
        if size < MIN_POSITION_USD {
            // KellySkip → в no-kelly печати как bankroll_too_small ([`print_side_stats`]).
            return BuyGate::KellySkip { raw, pred, kelly_f };
        }
        return BuyGate::Proceed { raw, pred, kelly_f, size };
    }

    let kelly_f_adj = kelly_f * KELLY_MULTIPLIER;
    if kelly_f_adj <= MIN_POSITION_USD {
        return BuyGate::KellySkip { raw, pred, kelly_f };
    }
    // Kelly size: cap по доле банка и [`MAX_POSITION_USD`] (срез ≠ KellySkip).
    let size = (kelly_f_adj.min(MAX_BET_FRACTION) * bankroll).min(MAX_POSITION_USD);
    if size < MIN_POSITION_USD {
        return BuyGate::KellySkip { raw, pred, kelly_f };
    }
    BuyGate::Proceed { raw, pred, kelly_f, size }
}

/// `true` если позиция открыта и вставлена в `positions_by_lane[lane_key]`;
/// иначе skip-счётчики ([`buy_gate`], same-asset, max-open).
///
/// Принимает write-guard на **весь** `account.positions` HashMap и read-guard
/// на **весь** `account.pending_close_positions` HashMap (оба caller'а держат
/// эти lock'и уже сейчас, lock-order `positions → pending_close_positions`
/// см. в [`crate::account`]). Это нужно, чтобы гейты учитывали суммарную
/// картину по обоим bucket'ам, а функция была без неявных lock-acquire'ов.
///
/// Гейты:
///   - [`BLOCK_SAME_ASSET_OPEN`] — проверяем live `positions_by_lane[lane_key]`
///     и `pending_close_positions[lane_key]` (та же lane). Иначе позиция,
///     перекочевавшая в pending-close на время async-SELL'а, не «защищала» бы
///     от повторного входа.
///   - [`MAX_OPEN_POSITIONS`] — суммарный count по **всем** lane'ам как в
///     live, так и в pending-close. Лимит интерпретируется как глобальный
///     потолок одновременно открытых позиций процесса.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn try_open_position(
    frame: &XFrame<SIZE>,
    pnl_inference: Option<PnlInference>,
    booster_pnl_for_shap: Option<&Booster>,
    positions_by_lane: &mut HashMap<crate::account::LaneKey, LanePositions>,
    pending_close_by_lane: &HashMap<crate::account::LaneKey, LanePositions>,
    lane_key: &crate::account::LaneKey,
    stats: &mut SideStats,
    bankroll: f64,
    strict_book: Option<&StrictBook>,
    currency: &str,
    is_kelly: bool,
    polymarket_url: &str,
    price_to_beat: Option<f64>,
    final_price: Option<f64>,
    event_end_ms: Option<i64>,
    graph_dump_bin_path: &str,
    gamma_question_at_open: Option<&str>,
    pnl_top5_shap_at_open_override: Option<String>,
    submit_mode: crate::account_submit::SubmitMode,
    project_manager: Option<&Arc<ProjectManager>>,
    account: &SharedAccount,
) -> bool {
    if crate::account_exit::is_halted() {
        stats.late_entry_skips += 1;
        return false;
    }
    let Some(entry_prob) = effective_implied_prob(frame, strict_book) else {
        return false;
    };
    match buy_gate(
        frame,
        pnl_inference,
        bankroll,
        strict_book,
        is_kelly,
        booster_pnl_for_shap,
    ) {
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
            if BLOCK_SAME_ASSET_OPEN {
                let mut same_asset_open = false;
                if let Some(lane_positions) = positions_by_lane.get(lane_key) {
                    for p in lane_positions.values() {
                        if p.read().await.asset_id == frame.asset_id {
                            same_asset_open = true;
                            break;
                        }
                    }
                }
                if !same_asset_open
                    && let Some(lane_pending) = pending_close_by_lane.get(lane_key)
                {
                    for p in lane_pending.values() {
                        if p.read().await.asset_id == frame.asset_id {
                            same_asset_open = true;
                            break;
                        }
                    }
                }
                if same_asset_open {
                    stats.same_asset_open_skips += 1;
                    return false;
                }
            }
            if let Some(max_open) = MAX_OPEN_POSITIONS {
                let live_total: usize = positions_by_lane.values().map(|m| m.len()).sum();
                let pending_total: usize =
                    pending_close_by_lane.values().map(|m| m.len()).sum();
                if live_total + pending_total >= max_open {
                    stats.max_open_positions_skips += 1;
                    return false;
                }
            }
            stats.raw_above_threshold += 1;
            stats.diag_sum_raw += raw as f64;
            stats.diag_sum_calibrated += pred as f64;
            stats.diag_sum_entry_prob += entry_prob;
            stats.diag_sum_kelly_f += kelly_f;

            let pnl_top5_shap_at_open = match pnl_top5_shap_at_open_override {
                Some(s) => s,
                None => {
                    if HISTORY_SIM_SKIP_TRADE_SHAP_CONTRIBUTIONS {
                        String::new()
                    } else {
                        booster_pnl_for_shap
                            .map(|b| top_pnl_shap_features_csv_cell(b, frame, PNL_MAX_LAG, 5))
                            .unwrap_or_default()
                    }
                }
            };

            match open_position(
                frame,
                size,
                stats,
                strict_book,
                raw,
                pred,
                kelly_f,
                currency,
                polymarket_url,
                price_to_beat,
                final_price,
                event_end_ms,
                graph_dump_bin_path,
                gamma_question_at_open,
                &pnl_top5_shap_at_open,
            ) {
                Some(pos) => {
                    // Бакеты по фактическому VWAP входа и cal pred (не mid displayed-prob).
                    let bucket_entry = prob_bucket_index(pos.buy_price);
                    let bucket_pred = prob_bucket_index(pred as f64);
                    stats.histogram_entry_prob[bucket_entry] += 1;
                    stats.histogram_cal_pred[bucket_pred] += 1;

                    // Submit: optimistic fill + spawn BUY taker; правки по WS ([`crate::account_ws`]).
                    let decision_price = strict_book.and_then(crate::account_order::best_ask_strict).map(|ask| (ask + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999));
                    let decision_book = strict_book.cloned();
                    let pos_id = pos.id.clone();
                    let pos_arc: SharedOpenPosition = std::sync::Arc::new(tokio::sync::RwLock::new(pos));
                    positions_by_lane
                        .entry(lane_key.clone())
                        .or_default()
                        .insert(pos_id, pos_arc.clone());
                    crate::account_submit::spawn_open_buy_taker(
                        account.clone(),
                        project_manager.cloned(),
                        pos_arc,
                        decision_price,
                        decision_book,
                        submit_mode,
                    );
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
    /// Обычная зона: TP/SL/timeout по PnL; `p_win_ema` не обновляется.
    HoldPnl,
    /// Hold-zone: SL + EV; caller записывает `new_p_win_ema` в позицию.
    HoldResolution { new_p_win_ema: Option<f64> },
    /// Закрыть по VWAP `exit_price` и причине (maker vs taker fee в [`crate::account_close_position::close_position`]).
    Close { exit_price: f64, reason: CloseReason },
}

/// Один bid-walk для гейта. `net_usdc`: при `exit_as_maker == true` — gross без комиссии на выход;
/// при `false` — после taker fee.
#[derive(Clone, Copy)]
struct CappedSellFill {
    /// VWAP цены продажи, доли вероятности на один share (0–1), после walk по книге.
    sell_vwap: f64,
    /// Чистая выручка USDC в выбранном режиме (maker vs taker).
    net_usdc: f64,
}

fn capped_sell_fill_for_gate(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    shares_held: f64,
    slippage_cap: Option<f64>,
    exit_as_maker: bool,
    current_prob: f64,
) -> Option<CappedSellFill> {
    let gross_usdc = match strict_book {
        Some(book) => book_fill_sell_strict(book, shares_held, slippage_cap),
        None => book_fill_sell(frame, shares_held, slippage_cap),
    }?;
    let sell_vwap = if shares_held > 0.0 {
        (gross_usdc / shares_held).clamp(0.001, 0.999)
    } else {
        current_prob.clamp(0.001, 0.999)
    };
    let net_usdc = if exit_as_maker {
        gross_usdc
    } else {
        let fee_usdc = shares_held * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell_vwap * (1.0 - sell_vwap);
        gross_usdc - fee_usdc
    };
    Some(CappedSellFill {
        sell_vwap,
        net_usdc,
    })
}

/// Urgent sell VWAP просел относительно входного ref (со slippage cap) не меньше чем на [`Y_TRAIN_SL_MIN_REF_SELL_REL_DROP`].
fn stop_loss_sell_deteriorated_vs_entry_ref(pos: &OpenPosition, urgent_sell_vwap: f64) -> bool {
    let sell_vwap_entry = pos.sell_vwap_entry;
    if !(sell_vwap_entry > 0.0) || !sell_vwap_entry.is_finite() {
        return true;
    }
    let threshold = sell_vwap_entry * (1.0 - Y_TRAIN_SL_MIN_REF_SELL_REL_DROP);
    urgent_sell_vwap <= threshold
}

/// `frames_held` — уже после инкремента тика (`manage_positions`) или `+1` в WS-предикате.
/// `p_win_now` — из одного predict на кадр; `None` в [`any_position_would_sell`] (EMA не двигается).
/// `min_position_frames` — минимальная выдержка позиции до первой проверки
/// SL/TP/EV-exit; `Some(MINPOSITION_FRAMES)` в history_sim, `None` в
/// [`crate::real_sim`] (см. [`MINPOSITION_FRAMES`]).
pub(crate) fn sell_gate(
    pos: &OpenPosition,
    frames_held: usize,
    frame: &XFrame<SIZE>,
    is_last: bool,
    p_win_now: Option<f64>,
    strict_book: Option<&StrictBook>,
    min_position_frames: Option<usize>,
) -> SellGate {
    if is_last || frame.event_remaining_ms <= 0 {
        return SellGate::HoldPnl;
    }

    let Some(current_prob) = effective_implied_prob(frame, strict_book) else {
        return SellGate::HoldPnl;
    };

    if let Some(min_frames) = min_position_frames {
        if frames_held < min_frames {
            return SellGate::HoldPnl;
        }
    }

    let in_hold_zone = frame.event_remaining_ms > 0 && frame.event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000;

    if in_hold_zone {
        let Some(fill) = capped_sell_fill_for_gate(
            frame,
            strict_book,
            pos.shares_held,
            None,
            true,
            current_prob,
        ) else {
            return SellGate::HoldPnl;
        };

        let delta = fill.sell_vwap - pos.buy_price;

        let new_p_win_ema: Option<f64> = match (p_win_now, pos.p_win_ema) {
            (Some(p), Some(prev)) => Some(EV_EXIT_P_WIN_EMA_ALPHA * p + (1.0 - EV_EXIT_P_WIN_EMA_ALPHA) * prev),
            (Some(p), None) => Some(p),
            (None, existing) => existing,
        };

        if delta <= Y_TRAIN_STOP_LOSS_PP && stop_loss_sell_deteriorated_vs_entry_ref(pos, fill.sell_vwap) {
            return SellGate::Close {
                exit_price: fill.sell_vwap,
                reason: CloseReason::StopLoss,
            };
        }
        let ev_close: Option<(f64, CloseReason)> = new_p_win_ema.and_then(|p_ema| {
            let ev_hold = p_ema * pos.shares_held;
            if fill.net_usdc * (1.0 - EV_EXIT_MARGIN) > ev_hold {
                if fill.net_usdc > pos.position_size {
                    Some((fill.sell_vwap, CloseReason::EvExitProfit))
                } else {
                    Some((fill.sell_vwap, CloseReason::EvExitLoss))
                }
            } else {
                None
            }
        });
        if let Some((exit_price, reason)) = ev_close {
            return SellGate::Close { exit_price, reason };
        }
        return SellGate::HoldResolution { new_p_win_ema };
    } else {
        let fill_v = capped_sell_fill_for_gate(
            frame,
            strict_book,
            pos.shares_held,
            Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT),
            true,
            current_prob,
        );
        let Some(fill_u) = capped_sell_fill_for_gate(
            frame,
            strict_book,
            pos.shares_held,
            None,
            false,
            current_prob,
        ) else {
            return SellGate::HoldPnl;
        };

        let delta_sl = fill_u.sell_vwap - pos.buy_price;

        // TP: сначала voluntary+cap (maker); если порог TP достигается только глубже по книге —
        // закрываем по полному walk, игнорируя cap slippage от L1.
        if let Some(fill_v) = fill_v {
            let delta_tp = fill_v.sell_vwap - pos.buy_price;
            if delta_tp >= Y_TRAIN_TAKE_PROFIT_PP {
                return SellGate::Close {
                    exit_price: fill_v.sell_vwap,
                    reason: CloseReason::TakeProfit,
                };
            }
        }
        if delta_sl >= Y_TRAIN_TAKE_PROFIT_PP {
            return SellGate::Close {
                exit_price: fill_u.sell_vwap,
                reason: CloseReason::TakeProfit,
            };
        }
        if delta_sl <= Y_TRAIN_STOP_LOSS_PP
            && stop_loss_sell_deteriorated_vs_entry_ref(pos, fill_u.sell_vwap)
        {
            return SellGate::Close {
                exit_price: fill_u.sell_vwap,
                reason: CloseReason::StopLoss,
            };
        }
        if frames_held >= POSITION_TIMEOUT_FRAMES {
            return SellGate::Close {
                exit_price: fill_u.sell_vwap,
                reason: CloseReason::Timeout,
            };
        }
    }

    SellGate::HoldPnl
}

/// Gate до HTTP: был бы [`sell_gate`] в режиме WS (`Close`) на этом тике.
/// `min_position_frames` — синхронно с одноимённым параметром
/// [`sell_gate`] / [`manage_positions`]; в [`crate::real_sim`] всегда `None`.
pub(crate) async fn any_position_would_sell(
    positions: &LanePositions,
    frame: &XFrame<SIZE>,
    min_position_frames: Option<usize>,
) -> bool {
    if positions.is_empty() || frame.event_remaining_ms <= 0 {
        return false;
    }
    for pos_arc in positions.values() {
        let pos = pos_arc.read().await;
        if pos.asset_id != frame.asset_id {
            continue;
        }
        if matches!(
            sell_gate(
                &pos,
                pos.frames_held + 1,
                frame,
                false,
                None,
                None,
                min_position_frames,
            ),
            SellGate::Close { .. }
        ) {
            return true;
        }
    }
    false
}

/// Закрытия через [`sell_gate`] / [`crate::account_close_position::close_position`]; чужой `asset_id` (позиция
/// другого маркета той же лейны) — позиция возвращается в `positions` как
/// «припаркованная» и ждёт [`crate::account::Account::resolve_pending_market_sync`].
/// `true`, если был хотя бы один успешный close (bankroll обновился).
/// `min_position_frames` пробрасывается в [`sell_gate`] (см. там).
///
/// Параметр `closing` — буфер записей о закрытиях для матчинга с user-WS
/// событиями ([`crate::account::Account::closing`]). На каждом тике мы
/// **сначала вытесняем** из него все терминальные записи
/// ([`ClosingPositionStatus::Closed`] / [`ClosingPositionStatus::CloseFailed`]) —
/// это и cleanup для history_sim/real_sim (где `Closed` ставится сразу),
/// и эвикция уже подтверждённых WS-колбеком закрытий в real-торговле.
/// Записи `PendingClose` остаются жить до прихода WS-подтверждения /
/// явного `apply_user_ws_event`.
///
/// **Hot-path для виртуальной торговли (history_sim/real_sim):** после
/// успешного [`crate::account_close_position::close_position`] (PnL уже учтён в `bankroll`/`stats`) сюда
/// пушится `ClosingPosition` со статусом [`ClosingPositionStatus::Closed`]
/// и заполненным `pnl`; `close_order_id` пуст. Это шаблон, который
/// real-торговля заменит на «push с `PendingClose` + `Some(order_id)`,
/// без правок `bankroll` — pnl и инкременты сделает WS-колбек».
///
/// `lane_key` нужен только для submit-веток ([`spawn_sell_taker`](crate::account_submit::spawn_sell_taker)):
/// перед спавном продавца мы перекладываем `pos_arc` из `positions` в
/// [`crate::account::Account::pending_close_positions`] (под тем же ключом),
/// чтобы `position_size` оставался залоченым в `available_bankroll` и MtM
/// equity до тех пор, пока [`crate::account_close_position::close_position_after_submit`]
/// не зачислит выручку в `bankroll`. В `SubmitMode::None` ветке этот ключ
/// не используется (закрытие синхронное, окна нет).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn manage_positions(
    positions: &mut LanePositions,
    frame: &XFrame<SIZE>,
    is_last: bool,
    p_win_now: Option<f64>,
    stats: &mut SideStats,
    strict_book: Option<&StrictBook>,
    min_position_frames: Option<usize>,
    submit_mode: crate::account_submit::SubmitMode,
    project_manager: Option<&Arc<ProjectManager>>,
    account: &SharedAccount,
    lane_key: &(String, XFrameIntervalKind, CurrencyUpDownOutcome),
) -> bool {
    for pos in positions.values_mut() {
        pos.write().await.frames_held += 1;
    }

    let mut sold = false;
    let mut remaining: LanePositions = HashMap::new();
    for (pos_id, pos_arc) in std::mem::take(positions) {
        // Snapshot позиции один раз — все sell_gate / submit-предикаты
        // ниже работают на этом snapshot'е. Нам не нужно держать pos-lock
        // через async-вызовы (CSV / spawn'ы).
        let snapshot = pos_arc.read().await.clone();

        if snapshot.asset_id != frame.asset_id {
            remaining.insert(pos_id, pos_arc);
            continue;
        }
        let close = match sell_gate(
            &snapshot,
            snapshot.frames_held,
            frame,
            is_last,
            p_win_now,
            strict_book,
            min_position_frames,
        ) {
            SellGate::Close { exit_price, reason } => Some((exit_price, reason)),
            SellGate::HoldResolution { new_p_win_ema } => {
                crate::account_submit::spawn_cancel_order(
                    account.clone(),
                    project_manager.cloned(),
                    pos_arc.clone(),
                    submit_mode,
                );
                if let Some(ema) = new_p_win_ema {
                    pos_arc.write().await.p_win_ema = Some(ema);
                }
                None
            }
            SellGate::HoldPnl => None,
        };
        if let Some((exit_price, reason)) = close {
            if submit_mode == crate::account_submit::SubmitMode::None {
                // Bid-walk (voluntary TP может ослабить L1-cap через
                // [`crate::account_close_position::gross_usdc_sell_take_profit`];
                // non-voluntary всегда полный
                // book-fill без cap). Если стакан не дал заполниться — позиция
                // возвращается в `remaining`, инкремент skip-счётчика.
                // `exit_price` от `sell_gate` тут не используется: в
                // [`crate::account_close_position::close_position`] sell-VWAP
                // вычисляется из `gross_usdc / shares_held` (а в резолюции —
                // бинарно через `reason`), `exit_price` нужен только
                // submit-ветке для taker-FAK price-hint'а ниже.
                let gross_usdc_opt = if reason.is_voluntary_exit() {
                    crate::account_close_position::gross_usdc_sell_take_profit(
                        frame,
                        &snapshot,
                        strict_book,
                    )
                } else {
                    match strict_book {
                        Some(book) => book_fill_sell_strict(book, snapshot.shares_held, None),
                        None => book_fill_sell(frame, snapshot.shares_held, None),
                    }
                };
                match gross_usdc_opt {
                    Some(gross_usdc) => {
                        crate::account_close_position::close_position(
                            account,
                            &pos_arc,
                            stats,
                            &reason,
                            Some(gross_usdc),
                            frame.event_remaining_ms,
                        )
                        .await;
                        sold = true;
                    }
                    None => {
                        stats.kelly_strict_sell_skips += 1;
                        remaining.insert(pos_id, pos_arc);
                    }
                }
            } else {
                // Перекладываем позицию из `positions` (откуда она уже выпала
                // через `std::mem::take`) в `pending_close_positions[lane_key]`
                // ДО спавна продавца. Так `available_bankroll` (см.
                // [`crate::real_sim::tick_once`]) и MtM equity продолжают
                // вычитать `position_size` / учитывать `shares_held × prob`,
                // пока async maker-TP / taker-FAK / post-market-end residual
                // не приведут к [`crate::account_close_position::close_position_after_submit`],
                // который вычистит позицию из обоих мапов одним блоком.
                {
                    let mut pending_guard =
                        account.pending_close_positions.write().await;
                    let lane_pending = pending_guard
                        .entry(lane_key.clone())
                        .or_default();
                    lane_pending.insert(pos_id.clone(), pos_arc.clone());
                }
                crate::account_submit::spawn_sell_taker(
                    account.clone(),
                    project_manager.cloned(),
                    pos_arc,
                    exit_price,
                    reason,
                    strict_book.cloned(),
                    submit_mode,
                );
            }         
        } else {
            remaining.insert(pos_id, pos_arc);
        }
    }
    *positions = remaining;
    sold
}

/// Доля выигрыша при TP: вход taker по `entry_prob`, выход по TP; maker/taker выхода по `best_bid_at_entry` (как в [`crate::account_close_position::close_position`]).
fn kelly_gain_ratio(entry_prob: f64, best_bid_at_entry: Option<f64>) -> f64 {
    let sell_price = (entry_prob + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999);
    let tp_is_maker = match best_bid_at_entry {
        Some(best_bid) => sell_price > best_bid,
        None => true,
    };
    let net = net_round_trip(entry_prob, sell_price, /*sell_is_taker=*/ !tp_is_maker);
    (net - 1.0).max(1e-9)
}

/// Доля убытка при SL: всегда taker на выходе (как [`crate::account_close_position::close_position`] на SL).
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

/// Виртуальный вход: ask-walk + fee → [`OpenPosition`]; CSV-поля из аргументов (gate снаружи).
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
    graph_dump_bin_path: &str,
    gamma_question_at_open: Option<&str>,
    pnl_top5_shap_at_open: &str,
) -> Option<OpenPosition> {
    let effective_size = match strict_book {
        Some(book) => effective_buy_usdc_strict(book, position_size),
        None => position_size,
    };
    let (buy_price, nominal_shares) = match strict_book {
        Some(book) => book_fill_buy_strict(book, effective_size)?,
        None => book_fill_buy(frame, position_size, Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT))?,
    };
    if nominal_shares <= 0.0 {
        return None;
    }
    let buy_price = buy_price.clamp(0.001, 0.999);

    let fee_usdc = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy_price * (1.0 - buy_price);
    let fee_shares = fee_usdc / buy_price;
    let shares_held = nominal_shares - fee_shares;

    stats.fees_paid += fee_usdc;

    let entry_prob = effective_implied_prob(frame, strict_book).unwrap_or(buy_price);

    let best_bid_at_entry = match strict_book {
        Some(book) => book.bids.first().map(|lvl| lvl.price),
        None => frame.book_bid_l1_price,
    };

    let gross_sell = match strict_book {
        Some(book) => book_fill_sell_strict(book, shares_held, None)?,
        None => book_fill_sell(frame, shares_held, None)?,
    };
    let sell_vwap_entry = (gross_sell / shares_held).clamp(0.001, 0.999);

    Some(OpenPosition {
        id: uuid::Uuid::new_v4().to_string(),
        asset_id: frame.asset_id.clone(),
        market_id: frame.market_id.clone(),
        shares_held,
        planned_shares_held: shares_held,
        entry_prob,
        buy_price,
        planned_buy_price: buy_price,
        sell_vwap_entry,
        position_size: effective_size,
        planned_entry_cost: effective_size,
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
        graph_dump_bin_path: graph_dump_bin_path.to_string(),
        gamma_question_at_open: gamma_question_at_open.map(|s| s.to_string()),
        pnl_top5_shap_at_open: pnl_top5_shap_at_open.to_string(),
        open_order_id: None,
        open_buy_invoke: None,
        maker_tp_position: None,
        taker_positions: Vec::new(),
        close_after_submit_finalized: false,
        entry_fee_usdc: fee_usdc,
        planned_fee_usdc: fee_usdc,
    })
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
        CloseReason::ResolutionWin => "ResolutionWin",
        CloseReason::ResolutionLoss => "ResolutionLoss",
    }
}

/// Ask-walk на USDC; опционально cap VWAP к L1 ([`SIM_MAX_SLIPPAGE_FROM_L1_PCT`]); без полной лестницы — L1–L3.
pub(crate) fn book_fill_buy(
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

/// Bid-walk на шеры; `slippage_cap` для voluntary (vs best bid); неполный fill → `None`; без лестницы — L1–L3.
pub(crate) fn book_fill_sell(
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
    let dmat = frame_to_prediction_dmatrix(frame, max_lag)?;
    booster.predict(&dmat).ok()?.into_iter().next()
}

fn frame_to_prediction_dmatrix(frame: &XFrame<SIZE>, max_lag: Option<usize>) -> Option<DMatrix> {
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
    DMatrix::from_dense(&features, 1).ok()
}

/// Обёртка над [`top_pnl_shap_features_csv_cell`] с глобальным skip-флагом
/// [`HISTORY_SIM_SKIP_TRADE_SHAP_CONTRIBUTIONS`] и стандартными параметрами
/// (`max_lag=PNL_MAX_LAG`, `top_n=5`). Используется из [`crate::real_sim::tick_once`]
/// (precompute до trade-lock) и из [`buy_gate`] для диагностического println.
pub(crate) fn pnl_top5_shap_csv_cell(booster: &Booster, frame: &XFrame<SIZE>) -> String {
    if HISTORY_SIM_SKIP_TRADE_SHAP_CONTRIBUTIONS {
        return String::new();
    }
    top_pnl_shap_features_csv_cell(booster, frame, crate::train_mode::PNL_MAX_LAG, 5)
}

/// Топ-|SHAP| признаков в одну CSV-ячейку (`\n`). `pub(crate)` для вызова из [`crate::real_sim::tick_once`] до trade-lock.
pub(crate) fn top_pnl_shap_features_csv_cell(
    booster: &Booster,
    frame: &XFrame<SIZE>,
    max_lag: Option<usize>,
    top_n: usize,
) -> String {
    let Some(dmat) = frame_to_prediction_dmatrix(frame, max_lag) else {
        return String::new();
    };
    let Ok((shap_values, (num_rows, num_cols))) = booster.predict_contributions(&dmat) else {
        return String::new();
    };
    if num_rows != 1 || num_cols < 2 {
        return String::new();
    }
    let n_features = num_cols - 1;
    let total_abs: f32 = (0..n_features).map(|i| shap_values[i].abs()).sum();

    let mut contributions: Vec<(String, f32, f32)> = (0..n_features)
        .filter_map(|feat_idx| {
            let shap = shap_values[feat_idx];
            let name = match max_lag {
                Some(n) => XFrame::<SIZE>::feature_name_n(feat_idx, n),
                None => XFrame::<SIZE>::feature_name(feat_idx),
            }?;
            let percent = if total_abs > 0.0 {
                shap.abs() / total_abs * 100.0
            } else {
                0.0
            };
            Some((name.to_string(), shap, percent))
        })
        .collect();
    contributions.sort_by(|(_, _, pct_a), (_, _, pct_b)| {
        pct_b.partial_cmp(pct_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    contributions
        .into_iter()
        .take(top_n)
        .map(|(name, shap, percent)| format!("   {shap:>8.4}   {percent:>6.2}%  {name}"))
        .collect::<Vec<_>>()
        .join("\n")
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

pub(crate) fn load_market_xframes(path: &Path) -> anyhow::Result<MarketXFramesDump> {
    let bytes = fs::read(path)?;
    Ok(bincode::deserialize(&bytes)?)
}

/// PM URL из `{stem}__{ts}.bin`; `None` если парсинг/`lag` вне окна.
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

/// `event_end_ms = floor(ts/interval)×interval`; лаг должен быть в `[0, interval)`.
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

/// Окно дампа (см. [`window_bounds_from_dump_path`]).
pub(crate) struct DumpWindowBounds {
    /// Начало окна, UTC сек (slug PM).
    pub window_start_sec: i64,
    /// Конец окна / резолюция, UTC ms.
    pub event_end_ms: i64,
}

/// Длительность тест-сплита: `n_paths × interval` (не span по датам файлов).
fn test_period_label(paths: &[std::path::PathBuf], interval_kind: XFrameIntervalKind) -> String {
    if paths.is_empty() {
        return "период=—".to_string();
    }
    let interval_min = interval_kind.interval_ms() / 60_000;
    let total_min = paths.len() as i64 * interval_min;
    let hours = total_min / 60;
    let minutes = total_min % 60;
    format!("период={hours}h {minutes}m")
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