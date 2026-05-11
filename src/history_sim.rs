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

use crate::account::{Account, SharedAccount};
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
    Y_TRAIN_SL_MIN_REF_SELL_REL_DROP,
    Y_TRAIN_TAKE_PROFIT_PP, Y_TRAIN_STOP_LOSS_PP,
    Y_TRAIN_NO_TRADE_PROB_LOW, Y_TRAIN_NO_TRADE_PROB_HIGH,
};
use crate::xframe_dump::MarketXFramesDump;
use crate::{tee_eprintln, tee_println};
use std::fs;
use std::path::Path;
use xgb::{Booster, DMatrix};

/// Префильтр raw-предикта перед Kelly (`f* > 0`).
pub const SIM_BUY_THRESHOLD: f32 = 0.60;

/// Cap проскальзывания VWAP от L1 при `book_fill_*` (см. y_train в train_mode).
/// Для **take profit**, если VWAP полного bid-walk даёт прибыль ≥
/// [`Y_TRAIN_TAKE_PROFIT_PP`], cap не применяется ([`sell_gate`], [`close_position`]).
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

/// Если `true` — не считаем SHAP-топ5 для PnL-модели на входе; колонка `pnl_top5_shap` в CSV пустая (экономия CPU).
pub const HISTORY_SIM_SKIP_TRADE_SHAP_CONTRIBUTIONS: bool = false;

/// Коэффициент taker-комиссии Polymarket для категории **Crypto** (CLOB):
/// `fee_usdc = C × POLYMARKET_CRYPTO_TAKER_FEE_RATE × p × (1 − p)`, где C — число шерсов, p — цена.
/// См. [Polymarket: Fees](https://docs.polymarket.com/trading/fees).
pub const POLYMARKET_CRYPTO_TAKER_FEE_RATE: f64 = 0.07;

/// Hold-zone: конец окна по времени; TP/timeout off; остаются hard SL и EV-exit по resolution-модели.
pub const HOLD_TO_END_THRESHOLD_SEC: i64 = 0;

/// EMA по `p_win` resolution-модели в hold-zone (`α` мало → плавнее EV-exit).
pub const EV_EXIT_P_WIN_EMA_ALPHA: f64 = 0.3;

/// Зазор EV-exit: `EV_sell × (1 − margin) > EV_hold`.
pub const EV_EXIT_MARGIN: f64 = 0.01;

/// Лимит кадров без TP/SL → [`CloseReason::Timeout`]. Должен совпадать с `Y_TRAIN_HORIZON_FRAMES` в xframe.
pub const POSITION_TIMEOUT_FRAMES: usize = 30;

/// Минимальная выдержка позиции (в кадрах) до того, как [`sell_gate`] вообще
/// начинает проверять SL/TP/EV-exit. В history_sim'e защищает от моментальных
/// «вход на тике t — выход на тике t» из-за всплеска WS-цены в hold-zone.
///
/// В режиме [`crate::real_sim`] этот гейт отключается передачей `None` в
/// параметре `min_position_frames` ([`sell_gate`] / [`manage_positions`] /
/// [`any_position_would_sell`]) — там кадры приходят раз в секунду и каждая
/// «выдержка» эквивалентна реальной задержке торговли.
pub const MINPOSITION_FRAMES: usize = 2;

/// Запрет одновременного открытия второй позиции на тот же `asset_id`,
/// пока первая ещё не закрылась.
///
/// * `true` (single-position-per-asset, режим «как в real_sim») — на каждый
///   asset одна активная позиция; повторные `raw≥thr` сигналы между
///   входом и закрытием идут в `same_asset_open_skips`. Калибровка в
///   [`crate::train_mode::first_entry_calibration_samples`] синхронно
///   фильтрует по cooldown'у [`POSITION_TIMEOUT_FRAMES`] — на каждый
///   маркет остаётся 1+ «entry-кадр», и `cal_pred` отражает win-rate
///   реального входа, а не per-frame сигнала.
///
/// * `false` (multi-position-per-asset) — позволяем каскад позиций на
///   одном asset (например, для усреднения / pyramid-входа). Калибровка
///   тогда вырождается в per-frame и сильно занижается, потому что
///   повторные кадры с `raw≥thr` после уже сработавшего TP помечаются
///   `y=0` (TP не повторится в горизонте). Kelly с такой калибровкой
///   почти не открывается.
///
/// Эта константа — единственный switch, синхронизирующий gate в
/// [`try_open_position`] и cooldown-фильтр в калибровке. Не разносить.
pub const BLOCK_SAME_ASSET_OPEN: bool = false;

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

/// Жизненный цикл live-ордера на открытие позиции (BUY) на Polymarket CLOB.
///
/// В history_sim/real_sim позиции открываются «виртуально» и сразу
/// существуют, поэтому дефолт — [`OpenPositionStatus::Open`]. Когда поверх
/// будет поднят реальный pipeline постановки ордеров (CLOB `post_order`),
/// позиция будет создаваться со статусом [`OpenPositionStatus::PendingOpen`]
/// и [`OpenPosition::open_order_id`] — `Some(...)`; колбек user-WS канала
/// (см. [`crate::account::spawn_user_ws_listener`]) переведёт её в
/// [`OpenPositionStatus::Open`] (по `MATCHED`) или в
/// [`OpenPositionStatus::OpenFailed`] (по `CANCELED`/`FAILED`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenPositionStatus {
    /// BUY-ордер поставлен на CLOB, ждём подтверждения через user-WS.
    /// В этом состоянии MtM позиции ведётся по `entry_cost` (как pending),
    /// поскольку шеры ещё не зачислены.
    PendingOpen,
    /// BUY подтверждён (WS прислал `MATCHED`/`CONFIRMED` либо это
    /// виртуальная позиция history_sim/real_sim). Доступна для
    /// `manage_positions` / закрытия.
    Open,
    /// BUY был отменён / упал на пути placement→matched (`CANCELED`,
    /// `FAILED`, `RETRYING` исчерпан). Позицию надо удалить и вернуть
    /// `entry_cost` в bankroll. Сейчас этот переход — TODO для real
    /// торговли.
    OpenFailed,
}

/// Shared-handle на [`OpenPosition`]: одна и та же запись живёт во многих
/// местах (`Account.positions` / `Account.pending_resolution` /
/// `ClosingPosition.position` / spawned-таски `account_submit`/`account_ws`),
/// и все они держат **один и тот же** `Arc<RwLock<...>>`. Это убирает
/// рассинхрон, когда `OpenPosition` копировался во вложенные контейнеры
/// (`ClosingPosition.position = pos.clone()`) и WS-fill писал только в
/// одну копию.
///
/// Лок-ордеринг: эти inner-локи лежат в самом конце канонического
/// порядка (см. [`crate::account::Account`]); внутри одной операции
/// допустимо удерживать **максимум один** pos-lock одновременно.
pub type SharedOpenPosition = std::sync::Arc<tokio::sync::RwLock<OpenPosition>>;

/// Shared-handle на [`ClosingPosition`]: см. [`SharedOpenPosition`].
pub type SharedClosingPosition = std::sync::Arc<tokio::sync::RwLock<ClosingPosition>>;

/// Weak-handle на [`ClosingPosition`]: используется в
/// [`OpenPosition::closing_position`] для разрыва циклической Arc-ссылки
/// (`OpenPosition → ClosingPosition.position → OpenPosition`). Через
/// `WeakClosingPosition::upgrade()` получаем `SharedClosingPosition`, если
/// `ClosingPosition` ещё жив (т.е. ещё лежит в `Account.closing` либо
/// держится spawn-таской `account_submit`); `None` означает, что
/// `ClosingPosition` уже выкинута `manage_positions` cleanup'ом — это
/// нормально, polling-fallback в этом случае идёт REST-веткой.
pub type WeakClosingPosition = std::sync::Weak<tokio::sync::RwLock<ClosingPosition>>;

/// Открытая позиция; в `real_sim` фильтр `asset_id == frame.asset_id` от чужого маркета.
#[derive(Debug, Clone)]
pub struct OpenPosition {
    /// Локальный uuid позиции (`Uuid::new_v4().to_string()`), генерируется в
    /// [`open_position`] синхронно с созданием структуры. Используется как
    /// корреляционный ключ во всех логах submit-флоу
    /// ([`crate::account_submit`] / [`crate::account_ws`]) — позволяет
    /// проследить весь жизненный цикл позиции (BUY → TP-place → trade-fills
    /// → SL/Timeout/EvExit → SELL → finalize) одним grep'ом по `id=…`. Для
    /// событий через [`ClosingPosition`] этот id берётся через
    /// `closing.position.read().await.id` (см. [`ClosingPosition::position`]).
    ///
    /// Не путать с [`Self::open_order_id`] / [`Self::tp_order_id`] /
    /// `ClosingPosition::close_order_id` — там CLOB-ные id ордеров, у одной
    /// позиции их может быть несколько (BUY + TP + SELL); наш `id` — один
    /// и стабильный на всю жизнь записи.
    pub(crate) id: String,
    pub(crate) asset_id: String,
    #[allow(dead_code)]
    pub(crate) market_id: String,
    /// Количество шерсов после вычета комиссии при покупке.
    ///
    /// **В submit-режиме это поле «живое»**: при первом WS BUY trade event'е
    /// [`crate::account_ws::apply_buy_fill`] обнуляет его и аккумулирует
    /// реальные fills (см. [`Self::optimistic_fill_replaced`]). Если нужны
    /// **исходные теоретические шеры** (то, что насчитал
    /// `book_fill_buy_strict` на кадре входа) — читай
    /// [`Self::planned_shares_held`].
    pub(crate) shares_held: f64,
    /// Отображаемая prob на входе; пайплайн оценки использует только [`Self::buy_price`].
    #[allow(dead_code)]
    pub(crate) entry_prob: f64,
    /// VWAP покупки — база для TP/SL, pending MtM, CSV `buy_price`, гистограмма входов.
    ///
    /// **В submit-режиме это поле «живое»**: в [`crate::account_ws::apply_buy_fill`]
    /// пересчитывается из реальных fills как `entry_cost / shares_held` после
    /// замержа всех partial-fills. Исходный теоретический VWAP, посчитанный
    /// `book_fill_buy_strict` на кадре входа, — в [`Self::planned_buy_price`].
    pub(crate) buy_price: f64,
    /// Sell VWAP на кадре входа для [`Self::shares_held`] с cap к L1 ([`SIM_MAX_SLIPPAGE_FROM_L1_PCT`]);
    /// gross walk / шеры, как у voluntary-ветки. Для SL: urgent VWAP должен просесть относительно
    /// этого уровня не меньше [`crate::xframe::Y_TRAIN_SL_MIN_REF_SELL_REL_DROP`].
    pub(crate) sell_vwap_entry: f64,
    /// USDC потраченные на покупку (= POSITION_SIZE_USD).
    ///
    /// **В submit-режиме это поле «живое»**: при первом WS BUY trade event'е
    /// [`crate::account_ws::apply_buy_fill`] обнуляет его и аккумулирует
    /// реальный `Σ size × price` от Polymarket (см.
    /// [`Self::optimistic_fill_replaced`]). Исходный плановый размер позиции
    /// (`POSITION_SIZE_USD` в submit / `book_fill_buy_strict` для виртуальных
    /// режимов) — в [`Self::planned_entry_cost`].
    pub(crate) entry_cost: f64,
    /// **Plan-snapshot** [`Self::shares_held`] на кадре входа — то, что вернул
    /// `book_fill_buy_strict` (или эквивалент для submit-flow) до того, как
    /// долетели реальные WS fills. Никогда не модифицируется после
    /// [`open_position`]. Используется как референс «сколько мы хотели
    /// купить» для аналитики (slippage по объёму = `1 - shares_held /
    /// planned_shares_held`), CSV-логов и сравнения «план vs факт». Для
    /// виртуальных режимов (history_sim / real_sim без submit)
    /// `shares_held == planned_shares_held` весь жизненный цикл позиции.
    pub(crate) planned_shares_held: f64,
    /// **Plan-snapshot** [`Self::buy_price`] на кадре входа — теоретический
    /// VWAP покупки из `book_fill_buy_strict`. Никогда не модифицируется
    /// после [`open_position`]. Slippage по цене входа =
    /// `buy_price - planned_buy_price`. Та же логика «план vs факт», что
    /// у [`Self::planned_shares_held`].
    pub(crate) planned_buy_price: f64,
    /// **Plan-snapshot** [`Self::entry_cost`] на кадре входа —
    /// `POSITION_SIZE_USD` для submit-flow (то, что мы готовы потратить /
    /// чем лочим bankroll) или фактический gross walk от
    /// `book_fill_buy_strict` для виртуальных режимов. Никогда не
    /// модифицируется после [`open_position`]. Slippage по сумме =
    /// `planned_entry_cost - entry_cost` (FAK taker мог взять меньше из-за
    /// тонкой книги). Та же логика «план vs факт», что у
    /// [`Self::planned_shares_held`].
    pub(crate) planned_entry_cost: f64,
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
    /// Путь к `.bin` дампу этого маркета (`xframes/…`) для колонки `graph_html_file_uri` в CSV.
    /// В [`crate::real_sim`] — синтетический путь по Gamma stem или пусто, если вопроса ещё нет.
    pub(crate) graph_dump_bin_path: String,
    /// Gamma `question` на входе ([`crate::real_sim::LaneFrame::gamma_question`]) — синтетический путь `.bin` для CSV, если явный путь пуст.
    pub(crate) gamma_question_at_open: Option<String>,
    /// Топ-5 SHAP-вкладов PnL-бустера на открытии (многострочная ячейка CSV); пусто если расчёт отключён или недоступен.
    pub(crate) pnl_top5_shap_at_open: String,
    /// Состояние live-ордера на открытие; см. [`OpenPositionStatus`]. Для
    /// виртуальных трейдов history_sim/real_sim сразу создаётся со
    /// значением `Open` ([`open_position`]).
    pub(crate) open_status: OpenPositionStatus,
    /// Идентификатор BUY-ордера на CLOB (`id` из user-WS события `order`).
    /// `None` — это виртуальная позиция (history_sim/real_sim) или ордер
    /// ещё не успел проставиться. Используется в
    /// [`crate::account::apply_user_ws_event`] для матчинга колбека.
    pub(crate) open_order_id: Option<String>,
    /// Идентификатор активной TP-лимитки (maker SELL) на CLOB. Заполняется в
    /// [`crate::account_submit`] после успешного `post_order_on_clob` сразу после
    /// подтверждения BUY-MATCHED. `None` — TP ещё не выставлен или закрытие уже
    /// в процессе через [`ClosingPosition`]. Используется
    /// [`crate::account::apply_user_ws_event`] для матчинга TP-fill'а
    /// (мы maker — order_id попадает в `maker_orders[].order_id`),
    /// и [`crate::account_submit::spawn_close_via_taker`] для отмены TP перед
    /// SELL taker по SL/Timeout/EvExit.
    pub(crate) tp_order_id: Option<String>,
    /// Дедуп / pre-suppress: `true` означает «попытка выставить TP-maker уже
    /// была — повторять не надо». Гейт в
    /// [`crate::account_submit::try_place_tp_maker`] на этом флаге сразу
    /// возвращается без HTTP. Два источника, где это поле уходит в `true`:
    ///
    /// 1. **Внутри `try_place_tp_maker`** — после взводящего snapshot'а ДО
    ///    сетевого вызова `post_order_on_clob`. Защищает от двойного TP при
    ///    гонке между WS-колбеком (`apply_user_ws_event_value` на BUY-MATCHED)
    ///    и delayed-verify-таской из polling-flow `account_submit`.
    /// 2. **На конструкции позиции в [`open_position`]**, если кадр входа уже
    ///    лежит в hold-zone (`event_remaining_ms > 0 && <= HOLD_TO_END_THRESHOLD_SEC * 1000`).
    ///    В hold-zone выходы должны идти только через resolution-модель
    ///    (EvExit*-taker) или hard SL — TP-maker по фиксированному
    ///    `Y_TRAIN_TAKE_PROFIT_PP` мешает поймать резолюционную выплату $1.
    ///    Поэтому WS/polling-колбек BUY-MATCHED для такой позиции даже не
    ///    попытается поставить TP — гейт в `try_place_tp_maker` отобьёт сразу.
    ///
    /// Атомарно проверяется + взводится под коротким inner-write `pos_arc` до
    /// сетевого вызова; HTTP идёт без лока. В виртуальных режимах
    /// (history_sim / real_sim без submit) попыток выставить TP вообще нет;
    /// flag `true` только если опен случился в hold-zone (pre-suppress; в
    /// virtual paths он всё равно никогда не читается).
    pub(crate) tp_placement_attempted: bool,
    /// Дедуп: `true` означает «cancel maker-TP для этой позиции уже
    /// инициирован — повторять не надо». Покрывает **оба** пути отмены
    /// TP-лимитки (hold-zone и SELL-taker SL/Timeout/EvExit*), чтобы
    /// не было двойного `DELETE /order` по одному `tp_order_id`:
    ///
    /// 1. **В `manage_positions`-ветке [`SellGate::HoldResolution`]**, когда
    ///    позиция первый раз попадает в hold-zone с живой TP-лимиткой:
    ///    атомарно (под inner-write `pos_arc`) проверяется
    ///    `tp_order_id.is_some() && !tp_cancel_attempted`,
    ///    флаг взводится в `true`, и спавнится
    ///    [`crate::account_submit::spawn_cancel_tp_for_hold_zone`] →
    ///    `cancel_order_on_clob`. Стратегия: в hold-zone выходы должны быть
    ///    только через resolution-модель (`EvExitProfit`/`EvExitLoss` taker'ом)
    ///    или hard SL, а не по фиксированному `Y_TRAIN_TAKE_PROFIT_PP`-таргету
    ///    maker-лимитки.
    /// 2. **В [`crate::account_submit::spawn_close_via_taker`]** перед
    ///    отправкой SELL-taker'а (SL/Timeout/EvExit*): атомарно
    ///    проверяется/взводится тот же флаг. Если кто-то уже инициировал
    ///    cancel (например, hold-zone-ветка) — SELL-taker аборится
    ///    (`CloseFailed`), следующий тик `manage_positions` повторит.
    /// 3. **На конструкции позиции в [`open_position`]**, если кадр входа уже
    ///    лежит в hold-zone. Парный pre-suppress со
    ///    [`Self::tp_placement_attempted`] (см. doc там): для позиций,
    ///    открытых уже внутри hold-zone, TP-maker не ставится ВООБЩЕ, и
    ///    отменять тоже нечего — но флаг ставим в `true` для внутренней
    ///    консистентности (никакая дальнейшая логика не должна интерпретировать
    ///    отсутствие `tp_order_id` как «TP ещё не успел встать»).
    ///
    /// **Не путать с** [`Self::tp_order_id`]: то поле обнуляется только на
    /// **подтверждённом** `canceled=true` из CLOB внутри cancel-таски, чтобы
    /// не потерять TP-fill в гонке «cancel HTTP в полёте ↔ TP уже сматчился».
    /// Этот флаг же — чисто локальный single-shot-маркер «cancel-таск
    /// уже спавнили / cancel не требуется».
    ///
    /// В виртуальных режимах (history_sim / real_sim без submit) TP-лимитки
    /// на CLOB не существует, флаг по большей части `false`, кроме случая
    /// hold-zone-входа (там pre-suppress тоже выставляется — harmless, никто
    /// не читает).
    pub(crate) tp_cancel_attempted: bool,
    /// `true` после того, как ПЕРВЫЙ WS `trade` fill (BUY) этой позиции был
    /// замержен в [`Self::shares_held`] / [`Self::entry_cost`] / [`Self::buy_price`]
    /// в submit-режиме. Используется в [`crate::account_ws::apply_buy_fill`]
    /// чтобы:
    /// - первый fill **сбросил** оптимистичные числа из `book_fill_buy_strict`
    ///   и записал реальные `size×price` (Polymarket-fee из `fee_rate_bps`),
    /// - последующие partial-fill'ы **аккумулировались** поверх первого
    ///   (мы можем получить N events на один FAK-ордер).
    ///
    /// В виртуальных режимах остаётся `false` — оптимистичный fill сразу
    /// «реальный», коррекция не требуется.
    pub(crate) optimistic_fill_replaced: bool,
    /// `true` после того, как PnL для этой позиции был финализирован (вычтен
    /// `entry_cost` из аккумулированных `c.pnl`-proceeds, обновлён `bankroll`,
    /// прошли stat-counter'ы) ровно один раз — см.
    /// [`crate::account_ws::finalize_close_pnl_in_place`] (taker SELL close
    /// path) и [`crate::account_ws::finalize_tp_close_after_creation`] (maker
    /// TP path). Идемпотентность всех путей финализации (WS-trade event и
    /// REST-fallback из [`crate::account_submit::apply_order_status_from_polling`])
    /// строится на проверке этого флага: если `true` — функция-финализатор
    /// делает no-op.
    ///
    /// Не путать с [`Self::frames_held`] (счётчик кадров удержания для
    /// `POSITION_TIMEOUT_FRAMES`-проверки в `sell_gate`); раньше эта роль
    /// была захардкожена в `frames_held == usize::MAX`-маркер, что мешало
    /// читать `frames_held` как обычный счётчик в логах/CSV.
    ///
    /// В виртуальных режимах (`history_sim`/`real_sim` без submit) PnL
    /// финализируется внутри [`close_position`] синхронно; это поле
    /// остаётся `false` (для них финализатор не дёргается).
    pub(crate) pnl_finalized: bool,
    /// **Weak**-ссылка на [`ClosingPosition`], созданную для этой позиции
    /// (taker SELL close из `manage_positions`, либо TP-fill из
    /// [`crate::account_ws::apply_sell_fill`], либо virtual close
    /// в [`close_position`]). Заполняется ровно в момент создания
    /// `ClosingPosition` — единственная point-of-truth, [`crate::account.closing`]
    /// HashMap дальше **не сканируется** для матчинга close-записи к этой
    /// позиции (см. [`crate::account_submit::drive_tp_pnl_finalization_via_polling`]).
    ///
    /// Тип `Weak`, а не `Arc`, нужен чтобы разорвать циклическую ссылку
    /// `OpenPosition.closing_position → ClosingPosition → ClosingPosition.position
    /// → OpenPosition`. Без этого retain'а из `Account.closing` и swap_remove'а
    /// из `Account.positions` было бы недостаточно — Arc-counts остались бы 1+1
    /// и обе записи навсегда висели в памяти.
    ///
    /// Через `closing_position.as_ref().and_then(Weak::upgrade)` получаем
    /// [`SharedClosingPosition`], если `ClosingPosition` ещё жив; `None` —
    /// нормальный случай (cleanup в `manage_positions` уже её выкинул, либо
    /// этой позиции вообще не было `ClosingPosition`).
    pub(crate) closing_position: Option<WeakClosingPosition>,
}

impl OpenPosition {
    pub(crate) fn set_closing_position(&mut self, weak: WeakClosingPosition) {
        if self.closing_position.is_some() {
            return;
        }
        self.closing_position = Some(weak);
    }

    /// Должна ли эта позиция учитываться в resolution-payout'е по своему
    /// маркету. Возвращает `true`, только если у нас **есть реальные**
    /// шеры на Polymarket Safe для этого `asset_id`:
    pub(crate) fn is_redeemable_at_resolution(&self) -> bool {
        match self.open_status {
            OpenPositionStatus::Open => true,
            OpenPositionStatus::PendingOpen => self.optimistic_fill_replaced,
            OpenPositionStatus::OpenFailed => false,
        }
    }
}

/// Жизненный цикл live-ордера на закрытие позиции (SELL) на Polymarket CLOB.
///
/// В history_sim/real_sim закрытия выполняются «виртуально» и сразу
/// финализируют PnL (см. [`close_position`]); такие записи создаются со
/// статусом [`ClosingPositionStatus::Closed`] и удаляются из
/// [`crate::account::Account::closing`] на следующем тике
/// [`manage_positions`] (cleanup pass в начале функции).
///
/// Для real-торговли flow такой:
/// 1. `manage_positions` принимает решение закрыться (TP/SL/Timeout/EV)
/// 2. SELL-ордер ставится на CLOB, в `closing` появляется запись
///    `ClosingPosition { close_status: PendingClose, pnl: None, .. }`
/// 3. user-WS event `MATCHED` → `apply_user_ws_event` переводит в
///    `Closed`, проставляет `pnl`, обновляет `bankroll`/`stats`
/// 4. Cleanup на следующем тике вытесняет запись.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosingPositionStatus {
    /// SELL-ордер поставлен, ждём `MATCHED`/`CONFIRMED` через user-WS.
    /// `entry_cost` всё ещё заблокирован, MtM ведётся по правилам
    /// активной позиции (через `last_prob`).
    PendingClose,
    /// SELL исполнен — PnL финализирован, `bankroll`/`stats` обновлены.
    /// В history_sim/real_sim это immediate-state после
    /// [`close_position`]; в real-торговле — после WS-колбека.
    Closed,
    /// SELL не прошёл (`CANCELED`/`FAILED`/timeout): позицию надо
    /// вернуть в `Account.positions` для следующей попытки. Сейчас
    /// этот переход — TODO для real торговли.
    CloseFailed,
}

/// Запись о закрытии позиции для матчинга real-time подтверждения через
/// user-WS (см. [`ClosingPositionStatus`]). Создаётся в [`manage_positions`];
/// читается / апдейтится в [`crate::account::apply_user_ws_event`].
#[derive(Debug, Clone)]
pub struct ClosingPosition {
    /// Сама позиция, которую закрываем — **shared-handle** на тот же
    /// `OpenPosition`, что лежит в `Account.positions` (см.
    /// [`SharedOpenPosition`]). Когда WS приходит partial BUY-fill, он
    /// пишет в эту же запись, и `entry_cost` в финализаторе close
    /// читается уже актуальный, без снимка-на-момент-решения.
    pub position: SharedOpenPosition,
    /// VWAP цена выхода, посчитанная в [`sell_gate`] на момент решения
    /// закрыться (для real-торговли это limit/market price ордера).
    pub exit_price: f64,
    /// Причина закрытия — синхронно с тем, что попадает в CSV-лог.
    pub reason: CloseReason,
    /// Реализованный PnL после fill'а; `None` пока не подтверждён
    /// колбеком (для виртуальных закрытий — заполняется сразу).
    pub pnl: Option<f64>,
    /// Текущий статус, см. [`ClosingPositionStatus`].
    pub close_status: ClosingPositionStatus,
    /// Идентификатор SELL-ордера на CLOB (`id` из user-WS события
    /// `order`). `None` — это виртуальное закрытие (sim) или ордер ещё
    /// не успел проставиться.
    pub close_order_id: Option<String>,
    /// Дедуп выставления SELL-ордера в submit-режиме: `true` после первой
    /// попытки `post_order_on_clob` SELL (успех или нет). Защищает от
    /// двойного выставления при гонке между [`manage_positions`]-тиком и
    /// delayed-verify-таском [`crate::account_submit::spawn_close_via_taker`].
    /// В виртуальных режимах (history_sim / real_sim без submit) всегда `false`.
    pub close_placement_attempted: bool,
    /// Wall-time момента создания записи (UTC ms) — для будущих TTL-чисток
    /// «застрявших» pending-ов и для CSV-лога диагностики.
    pub created_unix_ms: i64,
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
        matches!(self, CloseReason::TakeProfit)
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
    /// `(raw_pred_at_open, won)` для каждого закрытого трейда (TP/SL/Timeout/EvExit
    /// из [`close_position`] плюс ResolutionWin/ResolutionLoss из
    /// [`crate::account::Account::resolve_pending_market_sync`]). `won = pnl > 0.0`.
    ///
    /// Заполняется только ради
    /// [`crate::train_mode::fit_calibration_via_sim_replay`]: per-frame калибровка
    /// на val'е страдает distribution shift'ом (raw сигнал держится десятки
    /// кадров вокруг entry-момента, y-разметка маркирует только узкое окно
    /// TP-горизонта), поэтому калибруемся не на «кадрах с y», а на реальных
    /// трейдах симулятора (raw на открытии, факт закрытия в плюс/минус). Раз
    /// данные уже здесь — добавляем поле, не плодя параллельный сборщик.
    ///
    /// В обычных прогонах `run_sim_mode_inner` поле остаётся пустым (никто не
    /// читает) — стоимость памяти на закрытый трейд: `4 + 1 = 5` байт + Vec
    /// overhead.
    pub(crate) closed_trade_entries: Vec<(f32, bool)>,
    /// Сырое предсказание resolution-модели **на каждом кадре в hold-zone**
    /// (`event_remaining_ms <= hold_to_end_threshold_sec*1000` и `> 0`),
    /// для которого `compute_p_win_now` вернул `Some(_)`. Заполняется только
    /// когда `booster_resolution` передан и `calibration_resolution = None` —
    /// единственная конфигурация, в которой [`compute_p_win_now`] возвращает
    /// **сырой** скор, а не калиброванный.
    ///
    /// Используется в [`crate::train_mode::fit_calibration_via_sim_replay`]
    /// для калибровки `ModelType::Resolution`: каждое значение пары'ится с
    /// `token_won` маркета (`up_won` для UP-стороны, `!up_won` для DOWN);
    /// получаем `(raw_resolution, token_won)` точки для PAV.
    ///
    /// В обычных прогонах поле остаётся пустым: production sim передаёт
    /// `calibration_resolution = Some(...)`, гейт блокирует push (см.
    /// [`run_side_simulation`]).
    pub(crate) hold_zone_resolution_predictions: Vec<f32>,
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
///
/// `async fn`, потому что внутри идут `Account::*`-вызовы под `.write().await` /
/// `.read().await` per-field RwLock'ов; вызывается из `main` (`#[tokio::main]`)
/// через `.await`.
pub async fn run_sim_mode() -> anyhow::Result<()> {
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
                                booster_resolution_up.as_ref(), booster_resolution_down.as_ref(),
                                calibration_resolution_up.as_ref(), calibration_resolution_down.as_ref(),
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
                print_sim_stats(&tag, &sim_stats, bankroll_now, max_drawdown_pct_now, is_kelly);
            }
        }
    }

    Ok(())
}

/// Один маркет: последовательные проходы UP и DOWN по двум независимым рядам кадров.
/// Общий банкролл (как в [`crate::real_sim`]).
///
/// `async fn`, потому что [`run_side_simulation`] и [`Account::resolve_pending_market_sync`]
/// берут поля `Account` под `.write().await` per-field RwLock'ов.
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
        booster_up, calibration_up,
        booster_resolution_up, calibration_resolution_up,
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
        booster_down, calibration_down,
        booster_resolution_down, calibration_resolution_down,
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
        account
            .resolve_pending_market_sync(
                sim_stats,
                currency,
                interval_kind,
                &market_id,
                up_won,
                None,
            )
            .await;
    }

    // После resolve pending по этому маркету должен быть пуст (иначе утечка между маркетами).
    {
        let pending = account.pending_resolution.read().await;
        assert!(
            pending.get(&lane_key_up).map(|v| v.is_empty()).unwrap_or(true)
                && pending.get(&lane_key_down).map(|v| v.is_empty()).unwrap_or(true),
            "history_sim: pending_resolution не опустошён после resolve_pending_market_sync \
             (lane_key_up={lane_key_up:?}, lane_key_down={lane_key_down:?}); \
             dump invariant violated",
        );
    }
}

/// Один проход стороны (UP/DOWN) по ряду кадров: manage/open → MtM equity.
/// Живые позиции — `account.positions[lane_key]` (source-of-truth, как в [`crate::real_sim`]);
/// после цикла дренируются в `account.pending_resolution[lane_key]`, финальный payout —
/// в [`simulate_event`].
///
/// Equity: `bankroll + Σ(local×prob) + Σ(pending×buy_price)` (как `real_sim::tick_once`).
/// Сайзинг от `bankroll − Σ(entry_cost)` на этой стороне.
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

        // Фаза 1: manage_positions. Берём поля Account под отдельными write-локами
        // в каноническом порядке (`bankroll → positions → pending_resolution → closing`),
        // как в real_sim::tick_once — иначе deadlock с другими потребителями.
        {
            let mut bankroll = account.bankroll.write().await;
            let mut positions = account.positions.write().await;
            let mut pending_resolution = account.pending_resolution.write().await;
            let mut closing = account.closing.write().await;
            let positions_v = positions.entry(lane_key.clone()).or_default();
            let pending = pending_resolution.entry(lane_key.clone()).or_default();
            let closing_v = closing.entry(lane_key.clone()).or_default();
            manage_positions(
                positions_v,
                pending,
                closing_v,
                frame,
                is_last_idx,
                p_win_now,
                side_stats,
                &mut bankroll,
                None,
                Some(MINPOSITION_FRAMES),
                false, // history_sim: submit всегда выключен
                account,
            )
            .await;
        }

        // Фаза 2: try_open_position. available считается на тех же live-позициях
        // (в данном лейне same_side_locked). Порядок: bankroll → positions.
        {
            let bankroll = account.bankroll.read().await;
            let mut positions = account.positions.write().await;
            let positions_v = positions.entry(lane_key.clone()).or_default();
            let mut same_side_locked = 0.0;
            for p in positions_v.iter() {
                same_side_locked += p.read().await.entry_cost;
            }
            let available = (*bankroll - same_side_locked).max(0.0);
            try_open_position(
                frame,
                pnl_inference,
                Some(booster_pnl),
                positions_v,
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
                false, // history_sim: submit всегда выключен
                account,
            )
            .await;
        }

        // MtM equity (как real_sim): без prob на кадре тик пропускаем.
        if let Some(prob) = frame.currency_implied_prob {
            let prob = prob.clamp(0.0, 1.0);
            let equity = {
                let bankroll = account.bankroll.read().await;
                let positions = account.positions.read().await;
                let pending = account.pending_resolution.read().await;
                let mut positions_value = 0.0;
                if let Some(v) = positions.get(lane_key) {
                    for p in v {
                        positions_value += p.read().await.shares_held * prob;
                    }
                }
                let mut pending_value = 0.0;
                for v in pending.values() {
                    for p in v {
                        let g = p.read().await;
                        pending_value += g.shares_held * g.buy_price;
                    }
                }
                *bankroll + positions_value + pending_value
            };
            account.update_drawdown(equity).await;
        }
    }

    // Хвост открытых позиций уезжает в pending_resolution — финальный payout
    // делает caller через `Account::resolve_pending_market_sync`.
    {
        let mut positions = account.positions.write().await;
        let mut pending_resolution = account.pending_resolution.write().await;
        let positions_v = positions.entry(lane_key.clone()).or_default();
        if !positions_v.is_empty() {
            let pending = pending_resolution.entry(lane_key.clone()).or_default();
            pending.append(positions_v);
        }
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
///
/// `hold_to_end_threshold_sec` — параметр, а не константа: production-вызовы
/// (real_sim, simulate_event) передают [`HOLD_TO_END_THRESHOLD_SEC`]; sim-replay
/// калибровка ([`crate::train_mode::fit_calibration_via_sim_replay`]) подменяет
/// его на `RESOLUTION_CALIBRATION_HOLD_SEC` (см. train_mode), чтобы собирать
/// `(raw_resolution, token_won)` точки в реалистичном окне даже когда production
/// EvExit временно отключён константой `= 0`.
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
///
/// Async, потому что `positions: &mut Vec<SharedOpenPosition>` — для
/// `BLOCK_SAME_ASSET_OPEN`-проверки берём `.read().await` на каждый
/// inner-lock (см. [`crate::history_sim::SharedOpenPosition`]).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn try_open_position(
    frame: &XFrame<SIZE>,
    pnl_inference: Option<PnlInference>,
    booster_pnl_for_shap: Option<&Booster>,
    positions: &mut Vec<SharedOpenPosition>,
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
    submit: bool,
    account: &SharedAccount,
) -> bool {
    // Graceful-shutdown гейт: в submit-режиме после `account_exit::graceful_exit`
    // (SIGINT/SIGTERM) флаг `HALT_NEW_ORDERS` блокирует любые новые BUY-таски —
    // мы как раз закрываем процесс, новых позиций открывать нельзя. В
    // virtual-режимах `is_halted()` всегда `false` (флаг ставится только из
    // `account_exit::graceful_exit`), так что эта проверка для них no-op.
    if submit && crate::account_exit::is_halted() {
        stats.late_entry_skips += 1;
        return false;
    }
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
            if BLOCK_SAME_ASSET_OPEN {
                let mut same_asset_open = false;
                for p in positions.iter() {
                    if p.read().await.asset_id == frame.asset_id {
                        same_asset_open = true;
                        break;
                    }
                }
                if same_asset_open {
                    stats.same_asset_open_skips += 1;
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
                frame, size, stats, strict_book, raw, pred, kelly_f, currency,
                polymarket_url, price_to_beat, final_price, event_end_ms,
                graph_dump_bin_path,
                gamma_question_at_open,
                &pnl_top5_shap_at_open,
            ) {
                Some(mut pos) => {
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

                    // Submit-режим: переключаем виртуальный fill в pending +
                    // спавним отправку BUY-taker на CLOB (см. модуль
                    // [`crate::account_submit`]). `entry_cost` остаётся
                    // оптимистичным (= `size`), что лочит bankroll до
                    // подтверждения через WS / polling-verify (см.
                    // [`crate::account_ws::apply_user_ws_trade_fill`] для
                    // последующей коррекции по реальным fills).
                    //
                    // `shares_held` / `buy_price` тоже остаются оптимистичными
                    // (как просил пользователь — «оптимистичный virtual fill, потом
                    // коррекция по WS»). Реальные числа подмерджит [`crate::account_ws`].
                    if submit {
                        // In-flight идентификация — через сам Arc; без
                        // synthetic-id'шников. `open_status=PendingOpen` +
                        // `open_order_id=None` означает «отправили на CLOB,
                        // ждём real `order_id` через HTTP-ответ». Spawned-
                        // таска [`crate::account_submit::spawn_open_buy_taker`]
                        // получает Arc и пишет real id напрямую.
                        pos.open_status = OpenPositionStatus::PendingOpen;
                        let decision_price = strict_book
                            .and_then(crate::account_order::best_ask_strict)
                            .map(|ask| {
                                (ask + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999)
                            });
                        let decision_book = strict_book.cloned();
                        let pos_arc: SharedOpenPosition =
                            std::sync::Arc::new(tokio::sync::RwLock::new(pos));
                        positions.push(pos_arc.clone());
                        crate::account_submit::spawn_open_buy_taker(
                            account.clone(),
                            pos_arc,
                            decision_price,
                            decision_book,
                        );
                    } else {
                        positions.push(std::sync::Arc::new(tokio::sync::RwLock::new(pos)));
                    }
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
    /// Позицию закрываем с указанной причиной. `exit_price` — **фактическая**
    /// цена продажи (VWAP после walk по bid). TP / EvExitProfit — из voluntary fill (cap + maker cash);
    /// SL / Timeout / EvExitLoss — из urgent fill (без cap + taker fee). PnL в `close_position` по тем же причинам.
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
                if fill.net_usdc > pos.entry_cost {
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
    positions: &[SharedOpenPosition],
    frame: &XFrame<SIZE>,
    min_position_frames: Option<usize>,
) -> bool {
    if positions.is_empty() || frame.event_remaining_ms <= 0 {
        return false;
    }
    for pos_arc in positions.iter() {
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

/// Закрытия через [`sell_gate`] / `close_position`; чужой `asset_id` → [`pending_resolution`](crate::account::Account::pending_resolution).
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
/// успешного [`close_position`] (PnL уже учтён в `bankroll`/`stats`) сюда
/// пушится `ClosingPosition` со статусом [`ClosingPositionStatus::Closed`]
/// и заполненным `pnl`; `close_order_id` пуст. Это шаблон, который
/// real-торговля заменит на «push с `PendingClose` + `Some(order_id)`,
/// без правок `bankroll` — pnl и инкременты сделает WS-колбек».
#[allow(clippy::too_many_arguments)]
pub(crate) async fn manage_positions(
    positions: &mut Vec<SharedOpenPosition>,
    pending_resolution: &mut Vec<SharedOpenPosition>,
    closing: &mut Vec<SharedClosingPosition>,
    frame: &XFrame<SIZE>,
    is_last: bool,
    p_win_now: Option<f64>,
    stats: &mut SideStats,
    bankroll: &mut f64,
    strict_book: Option<&StrictBook>,
    min_position_frames: Option<usize>,
    submit: bool,
    account: &SharedAccount,
) -> bool {
    // Snapshot Arc'ов `OpenPosition` с уже терминальным `close_status=Closed` —
    // PnL финализирован WS-колбеком (см. [`crate::account_ws::finalize_close_pnl_in_place`])
    // и `bankroll` уже обновлён. Сами `OpenPosition` для taker-SELL веток
    // финализатор не трогает (только TP-ветка `apply_sell_fill` делает
    // `vec.swap_remove`), поэтому здесь нам нужно явно дропнуть их из
    // `positions`, иначе на следующем тике `sell_gate` увидит `open_status=Open`
    // и инициирует **второй** `spawn_close_via_taker` (дубль ордера).
    //
    // Snapshot снимаем ДО `closing.retain`, который выкидывает терминальные
    // записи. `CloseFailed` сюда не включаем — это сигнал «SELL отвергли», и
    // следующий тик должен честно перезайти в `sell_gate` для retry.
    // Сверка по `Arc::ptr_eq`, а не по `open_order_id`-строке — без
    // synthetic-id'шников.
    let mut closed_pos_arcs: Vec<SharedOpenPosition> = Vec::new();
    for c_arc in closing.iter() {
        let c = c_arc.read().await;
        if !matches!(c.close_status, ClosingPositionStatus::Closed) {
            continue;
        }
        closed_pos_arcs.push(c.position.clone());
    }

    // Cleanup: терминальные `Closed`/`CloseFailed` уже отработаны
    // (виртуально или WS-колбеком), их можно отпускать. `PendingClose`
    // оставляем — ждём подтверждения. Делаем «вручную», т.к. для retain
    // нужен async-предикат, а его не существует.
    {
        let mut keep: Vec<SharedClosingPosition> = Vec::with_capacity(closing.len());
        for c_arc in closing.drain(..) {
            let keep_it = matches!(
                c_arc.read().await.close_status,
                ClosingPositionStatus::PendingClose
            );
            if keep_it {
                keep.push(c_arc);
            }
        }
        *closing = keep;
    }

    for pos in positions.iter_mut() {
        pos.write().await.frames_held += 1;
    }

    // Snapshot Arc'ов `OpenPosition`, у которых сейчас активна `PendingClose`
    // запись в `closing` — для already_closing-проверки в основном цикле без
    // повторного N×M обхода inner-локов. Сверяем по `Arc::ptr_eq` (идентичность
    // shared-handle), а не по `open_order_id`: in-flight BUY-позиция может
    // иметь `open_order_id=None`, и в этом случае string-сравнение не работает.
    let mut pending_close_pos_arcs: Vec<SharedOpenPosition> = Vec::new();
    for c_arc in closing.iter() {
        let c = c_arc.read().await;
        if !matches!(c.close_status, ClosingPositionStatus::PendingClose) {
            continue;
        }
        pending_close_pos_arcs.push(c.position.clone());
    }

    let mut sold = false;
    let mut remaining: Vec<SharedOpenPosition> = Vec::new();
    for pos_arc in positions.drain(..) {
        // Snapshot позиции один раз — все sell_gate / submit-предикаты
        // ниже работают на этом snapshot'е. Нам не нужно держать pos-lock
        // через async-вызовы (CSV / spawn'ы).
        let snapshot = pos_arc.read().await.clone();

        // PnL уже финализирован WS-колбеком (см.
        // [`crate::account_ws::finalize_close_pnl_in_place`] /
        // [`crate::account_ws::finalize_tp_close_after_creation`]):
        // bankroll/stats обновлены, держать позицию дальше нельзя — иначе
        // её подберёт либо carry в `pending_resolution` (asset_id !=
        // frame.asset_id), либо повторный `sell_gate`. И то, и другое
        // приведёт к двойному учёту PnL (resolution payout поверх уже
        // зачисленного proceeds-entry_cost, либо второй SELL ордер на
        // уже проданные шеры). Дроп **первым делом**, до carry-ветки —
        // защита от race window между WS-finalize и сменой `asset_id`
        // на следующем тике. Проверка `pnl_finalized` инвариантнее
        // `closed_pos_arcs`/Arc::ptr_eq, т.к. покрывает и TP-fill путь
        // (там `apply_sell_fill` уже удалил позицию из `positions`, но
        // если ту же позицию занесли карри'ем из соседнего лейна — флаг
        // всё равно будет true).
        if snapshot.pnl_finalized {
            continue;
        }

        if snapshot.asset_id != frame.asset_id {
            if !snapshot.is_redeemable_at_resolution() {                
                continue;
            }
            pending_resolution.push(pos_arc);
            continue;
        }
        // В submit-режиме пропускаем pending позиции через sell_gate
        // (TP/SL/EvExit бессмысленны до подтверждения BUY-MATCHED), но всё
        // равно даём `frames_held++` идти. `OpenFailed` тоже пропускаем —
        // их уберёт следующий cleanup-pass (см. ниже).
        if submit {
            // Закрытие уже финализировано (Closed) WS-колбеком: bankroll/stats
            // обновлены, дубль `sell_gate` тут запрещён — выбрасываем позицию.
            // Сверка по `Arc::ptr_eq` (идентичность shared-handle).
            //
            // Этот путь покрывает редкий случай, когда `close_status=Closed`
            // выставлен, но `pnl_finalized` ещё не успел подняться (race
            // между `update_position_statuses` и `finalize_close_pnl_in_place`,
            // см. doc у `OpenPosition::pnl_finalized`). `pnl_finalized`-гейт
            // выше его не отловит, но `closed_pos_arcs` поймает.
            let already_finalized_closed = closed_pos_arcs
                .iter()
                .any(|p| std::sync::Arc::ptr_eq(p, &pos_arc));
            if already_finalized_closed {
                continue;
            }
            if !matches!(snapshot.open_status, OpenPositionStatus::Open) {
                // OpenFailed — позиция фактически не открылась, выбрасываем
                // (entry_cost тоже больше не лочим: ничего не списано).
                if matches!(snapshot.open_status, OpenPositionStatus::OpenFailed) {
                    continue;
                }
                // PendingOpen: оставляем в активных, ждём WS-подтверждения.
                remaining.push(pos_arc);
                continue;
            }
            // Если позиция уже в процессе закрытия (есть запись в `closing` с
            // тем же shared-handle и status=PendingClose), `sell_gate` снова
            // дёргать не надо — иначе мы бы попытались инициировать второй
            // SELL-ордер. Нашли совпадение — оставляем позицию в `remaining`
            // (физически она пока живёт в `positions`, по WS-MATCHED её перенесут
            // в `closing`-only).
            let already_closing = pending_close_pos_arcs
                .iter()
                .any(|p| std::sync::Arc::ptr_eq(p, &pos_arc));
            if already_closing {
                remaining.push(pos_arc);
                continue;
            }
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
                // Атомарно: обновить EMA + проверить/взвести single-shot
                // флаг cancel'а maker-TP в hold-zone. Спавн самой cancel-таски
                // делаем вне inner-write, чтобы HTTP не шёл под локом позиции.
                //
                // Условие spawn'а: submit-режим (TP-лимитка на CLOB существует
                // только тут), TP-maker ещё жив (`tp_order_id.is_some()`),
                // cancel ещё не пробовали (`!tp_cancel_attempted`).
                // Стратегия: в hold-zone выходы — только resolution-модель
                // (EvExit*-taker) или hard SL; фиксированный TP-таргет лимитки
                // мешает поймать резолюционную выплату $1, см. doc у
                // `OpenPosition::tp_cancel_attempted`.
                let needs_tp_cancel_in_hold_zone = {
                    let mut pw = pos_arc.write().await;
                    pw.p_win_ema = new_p_win_ema;
                    let needs = submit
                        && pw.tp_order_id.is_some()
                        && !pw.tp_cancel_attempted;
                    if needs {
                        pw.tp_cancel_attempted = true;
                    }
                    needs
                };
                if needs_tp_cancel_in_hold_zone {
                    crate::account_submit::spawn_cancel_tp_for_hold_zone(
                        account.clone(),
                        pos_arc.clone(),
                    );
                }
                None
            }
            SellGate::HoldPnl => None,
        };
        if let Some((exit_price, reason)) = close {
            if submit {
                // Submit-режим: НЕ финализируем PnL/stats и НЕ удаляем `entry_cost`
                // из bankroll-расчётов. Создаём `ClosingPosition` со статусом
                // `PendingClose` + provisional `close_order_id` и спавним
                // [`crate::account_submit::spawn_close_via_taker`]. WS-колбек
                // (или polling) переведёт `Closed` и обновит bankroll/stats
                // по реальным fills из `trade` events.
                //
                // `position` остаётся **физически в** `Account.positions` (через
                // `remaining.push(pos_arc)` ниже), потому что `entry_cost` всё
                // ещё должен лочиться для available bankroll до подтверждения SELL.
                // Запись в `closing` ссылается на **тот же** Arc<RwLock<...>>
                // (см. [`SharedOpenPosition`]), поэтому partial-fill'ы из WS
                // видны и здесь, и в `Account.positions` без дублирования.
                // Когда WS-колбек переведёт `close_status` в `Closed`, snapshot
                // `closed_open_ids` в начале следующего вызова `manage_positions`
                // дропнет `OpenPosition` из `positions`, одновременно
                // `closing.retain` выкинет терминальную запись.
                //
                // In-flight идентификация — через сам Arc; без synthetic id.
                // `close_status=PendingClose` + `close_order_id=None` означает
                // «отправили SELL, ждём real `order_id` через HTTP». Spawned-
                // таска [`crate::account_submit::spawn_close_via_taker`] получает
                // Arc и читает `asset_id`/`shares_held`/`tp_order_id` сама из
                // `closing_arc.position`, и пишет real id закрытия напрямую.
                //
                // Но `tp_placement_attempted=true` ставим **синхронно здесь**,
                // под inner-write локом позиции, ДО `tokio::spawn`. Это
                // закрывает окно между push'ем `ClosingPosition` и стартом
                // спавн-таски, в которое запоздавший WS/polling-колбек мог бы
                // вызвать `try_place_tp_maker` и поставить новый TP, о
                // существовании которого `spawn_close_via_taker` потом не
                // узнал бы. Сам `tp_order_id` НЕ берём здесь — его прочитает
                // и `take()`-нет спавн-таска, что даёт небольшое окно для
                // завершения in-flight HTTP TP-постановки (если она ещё
                // выполнялась) и попадания её результата в `pos.tp_order_id`.
                let closing_arc: SharedClosingPosition =
                    std::sync::Arc::new(tokio::sync::RwLock::new(ClosingPosition {
                        position: pos_arc.clone(),
                        exit_price,
                        reason: reason.clone(),
                        pnl: None,
                        close_status: ClosingPositionStatus::PendingClose,
                        close_order_id: None,
                        // Идемпотентность: ставим `attempted=true` ДО спавна задачи.
                        // Если manage_positions сюда вернётся повторно (та же позиция,
                        // следующий тик), already_closing-проверка выше отсечёт повтор,
                        // и эта ветка не выполнится снова.
                        close_placement_attempted: false,
                        created_unix_ms: crate::util::current_timestamp_ms(),
                    }));
                {
                    let mut pw = pos_arc.write().await;
                    pw.tp_placement_attempted = true;
                    // Прямая Weak-ссылка на ClosingPosition — единственный путь
                    // матчинга для polling-fallback (без скана `Account.closing`).
                    pw.set_closing_position(std::sync::Arc::downgrade(&closing_arc));
                }
                closing.push(closing_arc.clone());
                crate::account_submit::spawn_close_via_taker(account.clone(), closing_arc);
                // Counter-факт «решение принято»: считаем как намерение продать.
                // Реальные тип-счётчики (tp/sl/timeout/ev*) обновит WS-колбек по факту.
                sold = true;
                remaining.push(pos_arc);
            } else {
                match close_position(&snapshot, exit_price, &reason, frame, stats, strict_book) {
                    Some(pnl) => {
                        *bankroll += pnl;
                        sold = true;
                        // Виртуальное закрытие: PnL уже в `bankroll`/`stats`,
                        // теневая запись `Closed` нужна только для:
                        //   (а) симметрии с real-flow (в обоих случаях
                        //       завершённое закрытие проходит через `closing`),
                        //   (б) того, чтобы cleanup-шаг следующего тика её
                        //       отпустил — без cleanup'а Vec бы рос неограниченно.
                        let closing_arc: SharedClosingPosition =
                            std::sync::Arc::new(tokio::sync::RwLock::new(ClosingPosition {
                                position: pos_arc.clone(),
                                exit_price,
                                reason,
                                pnl: Some(pnl),
                                close_status: ClosingPositionStatus::Closed,
                                close_order_id: None,
                                close_placement_attempted: false,
                                created_unix_ms: crate::util::current_timestamp_ms(),
                            }));
                        // Прямая Weak-ссылка (см. real-flow выше). Для виртуального
                        // closure `pnl_finalized`-маркер не нужен (финализация
                        // синхронная, прямо тут), но поле всё равно заполняем для
                        // симметрии с submit-флоу и потенциальных будущих consumer'ов.
                        {
                            let mut pw = pos_arc.write().await;
                            pw.set_closing_position(std::sync::Arc::downgrade(&closing_arc));
                        }
                        closing.push(closing_arc);
                    }
                    None => {
                        stats.kelly_strict_sell_skips += 1;
                        remaining.push(pos_arc);
                    }
                }
            }            
        } else {
            remaining.push(pos_arc);
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
    graph_dump_bin_path: &str,
    gamma_question_at_open: Option<&str>,
    pnl_top5_shap_at_open: &str,
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

    let gross_sell = match strict_book {
        Some(book) => book_fill_sell_strict(book, actual_shares, None)?,
        None => book_fill_sell(frame, actual_shares, None)?,
    };
    let sell_vwap_entry = (gross_sell / actual_shares).clamp(0.001, 0.999);

    // Если вход случился уже внутри hold-zone (по `frame.event_remaining_ms`
    // относительно `HOLD_TO_END_THRESHOLD_SEC`), то TP-maker для этой позиции
    // создавать не надо — выходы должны идти только через resolution-модель
    // (`EvExit*`-taker) или hard SL, как у любой позиции после перехода в
    // hold-zone (см. doc у `OpenPosition::tp_cancel_attempted`
    // и ветку `SellGate::HoldResolution` в `manage_positions`). Поэтому
    // превентивно взводим оба `tp_*_attempted`-флага в `true`:
    //   * `tp_placement_attempted=true` — гейт `try_place_tp_maker`
    //     (`account_submit.rs:212-214` `if pos.tp_placement_attempted || pos.tp_order_id.is_some()`)
    //     отбьёт любую попытку поставить TP в WS/polling-колбеке BUY-MATCHED;
    //   * `tp_cancel_attempted=true` — single-shot-дедуп cancel'а
    //     (cancel сам по себе тут и не дёрнется — `spawn_cancel_tp_for_hold_zone`
    //     требует `tp_order_id.is_some()`, а оно `None`),
    //     но держим флаги внутренне-консистентно.
    //
    // Условие in-hold-zone идентично `sell_gate` и `compute_p_win_now`:
    // `event_remaining_ms > 0 && <= HOLD_TO_END_THRESHOLD_SEC * 1000`. При
    // текущем production `HOLD_TO_END_THRESHOLD_SEC=0` условие ложно всегда,
    // флаги остаются `false` — поведение не меняется. При повышении порога
    // (например до 30–60s) логика активируется автоматически.
    let entering_in_hold_zone: bool = frame.event_remaining_ms > 0
        && frame.event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000;

    Some(OpenPosition {
        // Локальный uuid позиции — корреляционный ключ для логов submit-флоу;
        // см. doc у поля.
        id: uuid::Uuid::new_v4().to_string(),
        asset_id: frame.asset_id.clone(),
        market_id: frame.market_id.clone(),
        shares_held: actual_shares,
        entry_prob,
        buy_price,
        sell_vwap_entry,
        entry_cost: position_size,
        // Plan-snapshot: фиксируем то, что насчитал `book_fill_buy_strict`
        // на кадре входа. После долёта реальных WS BUY fills'ов
        // `apply_buy_fill` затрёт «живые» `shares_held`/`entry_cost`/`buy_price`
        // реальными числами, а эти три останутся неизменными — референс
        // для slippage/«план vs факт». Для виртуальных режимов всегда
        // совпадают с «живыми» (apply_buy_fill не вызывается).
        planned_shares_held: actual_shares,
        planned_buy_price: buy_price,
        planned_entry_cost: position_size,
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
        // Виртуальный fill: книжный sweep уже выполнен, шеры зачислены — позиция сразу `Open`,
        // CLOB-ордера здесь нет, потому `open_order_id = None`. Реальная торговля заменит обе
        // строки: `PendingOpen` + `Some(order_id)`, переход в `Open` — через user-WS колбек.
        open_status: OpenPositionStatus::Open,
        open_order_id: None,
        // TP/SL/Timeout управляются полностью внутри `manage_positions` —
        // в виртуальной торговле TP-ордера на CLOB нет.
        tp_order_id: None,
        // Если вход случился в hold-zone — оба TP-флага предварительно
        // ставим в `true`, чтобы WS/polling-колбек BUY-MATCHED не дёрнул
        // `try_place_tp_maker` (см. блок выше с расчётом `entering_in_hold_zone`).
        tp_placement_attempted: entering_in_hold_zone,
        tp_cancel_attempted: entering_in_hold_zone,
        // В виртуальной торговле оптимистичный fill сразу «реальный» — нечего
        // перезаписывать. В submit-режиме `apply_buy_fill` поставит этот флаг
        // в `true` после первого WS BUY trade event'а (см. doc у поля).
        optimistic_fill_replaced: false,
        // PnL финализируется только в submit-режиме через
        // `finalize_close_pnl_in_place` / `finalize_tp_close_after_creation`;
        // в виртуальной торговле флаг остаётся `false` (см. doc у поля).
        pnl_finalized: false,
        // Заполняется в момент создания `ClosingPosition`
        // (см. `manage_positions` / `apply_sell_fill` TP-ветка / `close_position`).
        closing_position: None,
    })
}

/// Gross USDC при TP: если полный sell-walk даёт порог TP — можно обойти cap к L1 (см. [`sell_gate`]).
fn gross_usdc_sell_take_profit(
    frame: &XFrame<SIZE>,
    pos: &OpenPosition,
    strict_book: Option<&StrictBook>,
) -> Option<f64> {
    let cap = Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT);
    let meets_tp = |gross: f64| -> bool {
        if pos.shares_held <= 1e-18 {
            return false;
        }
        let vwap = gross / pos.shares_held;
        vwap - pos.buy_price >= Y_TRAIN_TAKE_PROFIT_PP
    };

    match strict_book {
        Some(book) => {
            let uncapped = book_fill_sell_strict(book, pos.shares_held, None)?;
            let capped = book_fill_sell_strict(book, pos.shares_held, cap);
            if meets_tp(uncapped) {
                match capped {
                    Some(g) if meets_tp(g) => Some(g),
                    Some(_) => Some(uncapped),
                    None => Some(uncapped),
                }
            } else {
                capped.or(Some(uncapped))
            }
        }
        None => {
            let uncapped = book_fill_sell(frame, pos.shares_held, None)?;
            let capped = book_fill_sell(frame, pos.shares_held, cap);
            if meets_tp(uncapped) {
                match capped {
                    Some(g) if meets_tp(g) => Some(g),
                    Some(_) => Some(uncapped),
                    None => Some(uncapped),
                }
            } else {
                capped.or(Some(uncapped))
            }
        }
    }
}

/// Бампит счётчики [`SideStats`] для одного закрытия позиции:
/// `pnl_usd`, `trades`, `wins`/`losses`, `closed_trade_entries`
/// и per-[`CloseReason`] пары `*_count` / `pnl_*`.
///
/// **Не трогает `fees_paid`** — комиссия в virtual-flow известна заранее (расчёт
/// `gross_usdc → fee_usdc → net_usdc`), а в submit-flow финализаторы получают
/// уже net'нутый `pnl` (Polymarket прислал fee как поле trade-event'а, оно
/// вошло в `c.pnl` через `apply_sell_fill`). Caller сам решает, нужно ли
/// дополнительно прибавлять `fees_paid`.
///
/// Используется как в виртуальной `close_position`, так и в submit-финализаторах
/// [`crate::account_ws::finalize_close_pnl_in_place`] и
/// [`crate::account_ws::finalize_tp_close_after_creation`] — единая точка
/// обновления per-side-счётчиков, нет дрейфа между путями.
pub(crate) fn apply_close_to_side_stats(
    stats: &mut SideStats,
    reason: &CloseReason,
    pnl: f64,
    raw_pred_at_open: f32,
) {
    stats.pnl_usd += pnl;
    stats.trades += 1;
    if pnl >= 0.0 {
        stats.wins += 1;
    } else {
        stats.losses += 1;
    }
    // См. doc у `SideStats::closed_trade_entries`. В обычных прогонах никто
    // не читает — но если sim запущен из `train_mode` ради калибровки,
    // именно эти пары (raw, won) идут в isotonic вместо per-frame y-меток.
    stats.closed_trade_entries.push((raw_pred_at_open, pnl > 0.0));
    match reason {
        CloseReason::TakeProfit => {
            stats.tp_count += 1;
            stats.pnl_tp += pnl;
        }
        CloseReason::StopLoss => {
            stats.sl_count += 1;
            stats.pnl_sl += pnl;
        }
        CloseReason::Timeout => {
            stats.timeout_count += 1;
            stats.pnl_timeout += pnl;
        }
        CloseReason::EvExitProfit => {
            stats.ev_exit_profit_count += 1;
            stats.pnl_ev_exit_profit += pnl;
        }
        CloseReason::EvExitLoss => {
            stats.ev_exit_loss_count += 1;
            stats.pnl_ev_exit_loss += pnl;
        }
    }
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
    let gross_usdc = if reason.is_voluntary_exit() {
        gross_usdc_sell_take_profit(frame, pos, strict_book)?
    } else {
        // SL / Timeout / EvExit*: как urgent в `sell_gate` — без cap от L1 (TakeProfit уже разобран выше).
        match strict_book {
            Some(book) => book_fill_sell_strict(book, pos.shares_held, None)?,
            None => book_fill_sell(frame, pos.shares_held, None)?,
        }
    };
    let sell_price = if pos.shares_held > 0.0 {
        (gross_usdc / pos.shares_held).clamp(0.001, 0.999)
    } else {
        exit_price.clamp(0.001, 0.999)
    };
    // TP: maker по bid на входе. EvExitProfit: maker при exit VWAP > L1 bid; иначе taker (как urgent в `sell_gate`).
    let voluntary_is_maker = match reason {
        CloseReason::TakeProfit => {
            let tp_target = (pos.buy_price + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999);
            match pos.best_bid_at_entry {
                Some(b) => tp_target > b,
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
    apply_close_to_side_stats(stats, reason, pnl, pos.raw_pred_at_open);

    // Per-trade CSV-лог (если открыт через `init_trade_csv_log_file`).
    // Пишется ровно одной строкой на закрытие; resolution-закрытия
    // (бинарная выплата $1/$0) пишет `Account::resolve_pending_market_sync`.
    let interval_str = position_interval_label(pos);
    let side_str = position_side_label(pos);
    let open_unix_ms = pos.event_end_ms.map(|e| e - pos.event_remaining_ms_at_open);
    let close_unix_ms = pos.event_end_ms.map(|e| e - frame.event_remaining_ms);
    let graph_html_file_uri = crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri(pos)
        .map(|p| crate::xframe_graph_dump::graph_html_trade_file_uri(&p, open_unix_ms, close_unix_ms, Some(side_str)))
        .unwrap_or_default();
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
        graph_html_file_uri: graph_html_file_uri.as_str(),
        pnl_top5_shap: pos.pnl_top5_shap_at_open.as_str(),
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

/// Bid-walk на полный объём; `slippage_cap`: voluntary — cap vs best bid, urgent — только полный fill.
/// Симметрично y_train (неполный fill → нет выхода на тике). Легаси: L1–L3.
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

/// Топ `top_n` признаков по |SHAP| для одной строки (как [`crate::train_mode::print_contributions`]),
/// без bias; строки — формат `   shap   pct%  name` для одной ячейки CSV (через `\n`).
///
/// `pub(crate)`, чтобы [`crate::real_sim::tick_once`] мог посчитать SHAP **до** взятия
/// trade write-лока и передать готовую строку в [`try_open_position`] через
/// `pnl_top5_shap_at_open_override` — иначе `predict_contributions` блокирует
/// `state.write + account.write` на длительность XGBoost-инференса (~ms),
/// что сериализует все 4 воркера real_sim между собой.
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
    let total_abs: f32 = (0..n_features)
        .map(|i| shap_values[i].abs())
        .sum();

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

/// Печать статистики прогона. `bankroll_now` / `max_drawdown_pct_now` передаются явно,
/// чтобы саму печать оставить sync (без `await`-точек посреди форматирования) —
/// вызыватели снимают значения короткими `account.bankroll.read().await` /
/// `account.max_drawdown_pct.read().await` непосредственно перед вызовом.
pub(crate) fn print_sim_stats(
    tag: &str,
    sim_stats: &SimStats,
    bankroll_now: f64,
    max_drawdown_pct_now: f64,
    is_kelly: bool,
) {
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
    let roi_pct = (bankroll_now - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100.0;

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
        bankroll_now, roi_pct, max_drawdown_pct_now,
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

pub(crate) fn load_market_xframes(path: &Path) -> anyhow::Result<MarketXFramesDump> {
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

/// Суммарная длительность маркетов тест-сплита: `период=Hh Mm`,
/// где `total_min = n_paths × interval_minutes`. Не зависит от порядка
/// `paths` (в отличие от span first..last) — на тест-сплите с разреженной
/// историей маркеты могут идти не подряд, span между крайними не совпадает
/// с реальным «временем работы стратегии». Возвращает `период=—` при пустом
/// списке.
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
