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

use crate::account::Account;
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::real_sim::interval_label;
use crate::train_mode::{
    collect_bin_paths, load_calibration, split_counts,
    Calibration, PNL_MAX_LAG, RESOLUTION_MAX_LAG, TEST_FRACTION, VAL_FRACTION,
};
use crate::xframe::{apply_side_symmetry, BookLevel, XFrame, SIZE, Y_TRAIN_HORIZON_FRAMES, Y_TRAIN_TAKE_PROFIT_PP, Y_TRAIN_STOP_LOSS_PP};
use crate::xframe_dump::MarketXFramesDump;
use crate::{tee_eprintln, tee_println};
use std::fs;
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
pub const MIN_POSITION_USD: f64 = 0.10;

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

/// Максимально допустимое отклонение VWAP **strict-fill'а** от лучшей цены
/// (L1) в долях. `0.02` = 2%: если для покупки за `position_size` USDC
/// VWAP `(position_size / total_shares)` уходит больше чем на 2% выше
/// лучшего ask — [`book_fill_buy_strict`] возвращает `None`, позиция не
/// открывается. Симметрично для [`book_fill_sell_strict`]: VWAP продажи
/// ниже best bid больше чем на 2% → `None`, продажа откладывается на
/// следующий тик.
///
/// **Зачем**: в `real_sim` `StrictBook.asks`/`bids` — это **полная**
/// лестница CLOB, не первые три уровня. Без cap'а Kelly мог бы
/// «доедать» позицию L4–L20 на тонком маркете, выкупая шерсы на 5–20¢
/// хуже mid и сжигая весь edge модели. Cap отсекает такие ситуации
/// и оставляет позицию закрытой/неоткрытой до восстановления глубины.
///
/// Значение 2% выбрано как «хуже типичного спреда (≤ 10¢ → mid),
/// но ещё в пределах сделок, где fee + slippage не съедают edge модели
/// (`SIM_BUY_THRESHOLD = 0.6`, после калибровки edge ~ 1–3%)». На
/// `history_sim`-пути cap не применяется (там `book_fill_buy`/`sell`
/// без strict-режима).
pub const MAX_SLIPPAGE_FROM_L1_PCT: f64 = 0.02;

/// Аварийный halt новых входов по mark-to-market drawdown (`Account::max_drawdown_pct`).
/// `Some(pct)` — при `max_drawdown_pct ≥ pct` `try_open_position` отклоняет
/// все новые позиции до конца жизни процесса (drawdown — историческая
/// величина, не сбрасывается). `None` — kill-switch выключен.
///
/// **Что не трогает**: ведение/закрытие уже открытых позиций
/// ([`manage_positions`], TP/SL/EV-exit/Timeout, резолюционный колбек).
/// Выйти из позиции важнее, чем дождаться улучшения equity.
///
/// **Где проверяется**: только в `real_sim::tick_once` перед
/// `try_open_position` — `history_sim` это offline-backtest, halt по
/// drawdown'у искажал бы итоговую метрику модели.
///
/// Значение по умолчанию (30%) — точка, после которой Kelly-сайзинг
/// на оставшемся капитале математически даёт ROI restoration ≥ 43%
/// от текущего bankroll (1 / (1 − 0.30) − 1), что заведомо вне
/// статпредсказуемости моделей такого типа: дальше «лезть в позиции»
/// — это азартное усреднение в просадке, не торговля.
pub const EMERGENCY_HALT_DRAWDOWN_PCT: Option<f64> = Some(30.0);

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
    /// `OrderBookSummaryResponse.last_trade_price` Polymarket CLOB —
    /// цена последней сделки по этому asset_id на момент HTTP-снимка.
    /// Нужна, чтобы воспроизвести логику
    /// [`crate::xframe::currency_implied_prob_polymarket_style`]:
    /// при широком спреде (`> POLYMARKET_WIDE_SPREAD_USD = 10¢`) UI
    /// Polymarket показывает не mid L1, а last trade. Без этого поля
    /// HTTP-выведенная implied_prob систематически расходилась бы
    /// с `XFrame.currency_implied_prob` (mid вместо last trade) на
    /// тонких маркетах — а именно она используется как фича модели
    /// и как oporная вероятность для Kelly-расчёта в `open_position`
    /// и `sell_gate`. `None` если CLOB не вернул `last_trade_price`
    /// (новый рынок без сделок).
    pub(crate) last_trade_price: Option<f64>,
    /// `OrderBookSummaryResponse.min_order_size` Polymarket CLOB —
    /// **минимальный размер ордера в шерсах**, который маркет принимает
    /// (статичный атрибут маркета). Используется в
    /// [`book_fill_buy_strict`] / [`book_fill_sell_strict`]: если число
    /// шерсов меньше `min_order_size` — strict-исполнение отказывает
    /// (`None`), потому что в реальной торговле такой ордер CLOB
    /// отклонил бы, а локальная бухгалтерия должна симметрично не
    /// открывать/не закрывать позицию.
    ///
    /// **Не приходит из WS**: market WS-канал Polymarket публикует
    /// только динамику (`book`, `price_change`, `tick_size_change`,
    /// `last_trade_price`, `market_resolved`), а статика маркета
    /// (`min_order_size`, `neg_risk`, фактический `tick_size` на
    /// старте) живёт в HTTP CLOB. Поэтому `XFrame` это поле не
    /// содержит — `history_sim` идёт через
    /// `book_fill_buy`/`book_fill_sell` без min-проверки. На
    /// `real_sim`-пути значение заполняется в [`crate::real_sim::parse_book_levels`]
    /// из ответа `clob.order_books(&[...])`. `None` — CLOB не вернул
    /// поле (теоретически не должно случаться, но защитно — пропускаем
    /// проверку, чтобы не зарубить торговлю на пустом значении).
    pub(crate) min_order_size: Option<f64>,
}

/// Эффективная `currency_implied_prob` для текущего тика.
///
/// Если в тике передан HTTP-стакан ([`StrictBook`], только real_sim) —
/// считаем «отображаемую на UI Polymarket» вероятность по тем же
/// правилам, что и WS-фича `XFrame.currency_implied_prob` (см.
/// [`crate::xframe::currency_implied_prob_polymarket_style`]):
/// при спреде `≤ 10¢` — mid L1, при широком — `last_trade_price`
/// (с фоллбэком на mid, если last trade неизвестен). Это критично:
/// `XFrame.currency_implied_prob` уходит в модель как фича и как
/// «опорная цена» для Kelly. Если бы тут было «всегда mid», на тонких
/// маркетах HTTP-prob и WS-prob систематически расходились бы — Kelly
/// в `open_position` и delta в `sell_gate` начали бы считаться в
/// другой шкале, чем та, на которой обучалась модель.
///
/// HTTP всё равно свежее WS-кадра (нет буфера/реконнекта), поэтому
/// при наличии strict_book берём именно его. Иначе — `frame.currency_implied_prob`
/// как было. На `history_sim`-пути `strict_book = None` всегда.
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

/// Строгая покупка `position_size` USDC по `asks` [`StrictBook`] без fallback.
///
/// Возвращает `Some((vwap, total_shares))`, если удалось полностью заполнить
/// `position_size` фактическими уровнями стакана **и** выполнены оба
/// инвариативных условия живой торговли:
///
/// 1. **Slippage cap** ([`MAX_SLIPPAGE_FROM_L1_PCT`]): VWAP не должен
///    отклоняться от лучшего ask больше, чем на `MAX_SLIPPAGE_FROM_L1_PCT`.
///    На полной лестнице CLOB без cap'а Kelly мог бы доедать L4–L20 на
///    тонком маркете и платить на 5–20¢ выше mid — это сжигает edge
///    модели и в `real_sim` представляет реальный финансовый риск.
/// 2. **`min_order_size`** (`StrictBook.min_order_size`, статичный атрибут
///    маркета из `OrderBookSummaryResponse.min_order_size`): итоговое
///    `total_shares` не должно быть меньше этого значения. Иначе CLOB
///    отклонил бы ордер, а локальная бухгалтерия зеркалом должна не
///    создавать «фантомную» позицию.
///
/// На любом из этих отказов возвращаем `None` — позиция не открывается,
/// `try_open_position` инкрементирует `kelly_strict_buy_skips` и
/// печатает skip-лог. Если `book.min_order_size = None` (CLOB не вернул
/// поле) — проверка `min_order_size` пропускается; защитный fallback,
/// чтобы случайно не зарубить торговлю при отсутствии данных.
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
    // Slippage cap: VWAP относительно best ask. Покупка тейкером
    // **выше** best ask — это и есть проскальзывание; считаем по `vwap`,
    // не по последнему сожранному уровню, потому что fee/EV-расчёты
    // ниже по стеку идут от VWAP.
    if (vwap - best_ask) / best_ask > MAX_SLIPPAGE_FROM_L1_PCT {
        return None;
    }
    if let Some(min) = book.min_order_size {
        if total_shares < min {
            return None;
        }
    }
    Some((vwap, total_shares))
}

/// Строгая продажа `shares_to_sell` по `bids` [`StrictBook`] без fallback.
///
/// Возвращает `Some(gross_usdc)` (до вычета taker-fee), если ликвидности
/// достаточно **и** выполнены инвариантные условия живой торговли,
/// что у [`book_fill_buy_strict`]:
///
/// 1. **Slippage cap** (`slippage_cap`): VWAP продажи
///    `(gross_usdc / shares_to_sell)` не должен быть ниже best bid
///    больше, чем на переданный процент. Иначе — `None`,
///    `manage_positions` оставит позицию открытой и попробует
///    повторно на следующем тике (`kelly_strict_sell_skips++`).
///
///    * `Some(pct)` — cap активен (типичное значение [`MAX_SLIPPAGE_FROM_L1_PCT`]).
///      Используем для **добровольных** выходов (TP / EvExitProfit),
///      где edge модели чувствителен к slippage и можно «передумать»,
///      подождав лучшего стакана. Решение принимает
///      [`CloseReason::is_voluntary_exit`].
///    * `None` — cap отключён, выходим по любому VWAP. Используем для
///      **обязательных** выходов (SL / Timeout / EvExitLoss): удерживать
///      позицию из-за широкого стакана опаснее, чем пройти slippage,
///      иначе позиция может уехать к резолюции проигравшего токена
///      за $0.
///
///    Параметр `Option<f64>` (а не `bool`) намеренно: даёт возможность
///    в будущем градуировать cap по reason (например, TP — 2%,
///    EvExitProfit — 3%) без изменения сигнатуры.
/// 2. **`min_order_size`**: `shares_to_sell` < `min_order_size` →
///    `None`, ровно по той же причине, что у buy: ордер CLOB бы
///    отклонил. Эта проверка независима от `slippage_cap` —
///    CLOB ничего не примет ниже минимума, даже если мы готовы
///    пройти любой slippage.
///
/// Если `book.min_order_size = None` — проверка `min` пропускается.
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

/// Открытая виртуальная позиция в одном токене.
///
/// `asset_id` / `market_id` — **обязательные якоря** на конкретный
/// CTF-токен. Без них в `real_sim`-сценарии длинная позиция могла
/// «осиротеть» при смене маркета внутри лейна (5m/15m раунды
/// сменяются), и `manage_positions` начал бы прогонять её через
/// `sell_gate` вместе с фреймом уже **другого** маркета — entry/exit
/// цены и hold-zone окно у нового маркета свои, к этой позиции
/// нерелевантны. Фильтр `pos.asset_id != frame.asset_id` в
/// `manage_positions` пропускает такие позиции на текущем тике (они
/// дождутся явной resolution-обработки через post-resolution
/// колбек final_price).
///
/// В `history_sim` практически любой проход — это один маркет, поэтому
/// все позиции одного `run_side_simulation` имеют одинаковый
/// `asset_id`, и фильтр на них тождественен. Поле всё равно несём,
/// чтобы `OpenPosition` имел один и тот же контракт во всех режимах.
#[derive(Debug, Clone)]
pub struct OpenPosition {
    /// Идентификатор CTF-токена, на котором открыта позиция (
    /// `frame.asset_id` в момент входа). Используется для фильтрации
    /// в `manage_positions`: при смене маркета в лейне (`real_sim`)
    /// мы НЕ применяем `sell_gate` нового маркета к старой позиции.
    pub(crate) asset_id: String,
    /// Идентификатор маркета (`condition_id`), на котором открыта
    /// позиция. Дублирует данные `asset_id` (1:1 mapping в Polymarket
    /// Up/Down рынках, см. `currency_up_down_by_asset_id`), но
    /// удобен для логов и для post-resolution callback'а: маркет
    /// резолвится «целиком», поэтому при поиске `final_price` для
    /// «осиротевших» позиций удобнее матчить по `market_id`.
    ///
    /// Чтение этого поля появится в #2 (resolution callback).
    #[allow(dead_code)]
    pub(crate) market_id: String,
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
    /// Сырое предсказание pnl-модели в момент открытия (`PnlInference::raw`).
    /// Используется только для per-trade CSV-лога; в торговой логике не участвует.
    pub(crate) raw_pred_at_open: f32,
    /// Калиброванное предсказание pnl-модели в момент открытия
    /// (`PnlInference::pred`). Только для per-trade CSV-лога.
    pub(crate) cal_pred_at_open: f32,
    /// «Сырой» Kelly f* (до `KELLY_MULTIPLIER` / `MAX_BET_FRACTION` /
    /// min-size cap) в момент открытия. Только для per-trade CSV-лога.
    pub(crate) kelly_f_at_open: f64,
    /// `event_remaining_ms` на тике открытия. Помогает в анализе CSV
    /// per-trade сопоставлять момент входа с фазой жизни маркета
    /// (далеко до резолюции / hold-zone / late-entry).
    pub(crate) event_remaining_ms_at_open: i64,
    /// Дискриминант [`XFrameIntervalKind`] (`5m` / `15m`). Берётся из
    /// `frame.xframe_interval_type` в момент открытия и переиспользуется
    /// в per-trade CSV (чтобы CSV-строка несла лейн без проброса лишних
    /// параметров вниз по `manage_positions`/`close_position`).
    pub(crate) xframe_interval_type_at_open: i32,
    /// Дискриминант [`CurrencyUpDownOutcome`] (`Up`/`Down`) — то же
    /// назначение, что у [`Self::xframe_interval_type_at_open`].
    pub(crate) currency_up_down_outcome_at_open: i32,
    /// Лейбл валюты лейна (`btc` / …). Используется только для
    /// per-trade CSV-лога; в торговой логике не участвует. Дублирует
    /// `lane_key.0`, но избавляет `close_position` /
    /// `resolve_pending_market_sync` от необходимости тащить
    /// `lane_key` через всю цепочку вызовов.
    pub(crate) currency: String,
}

/// Причина закрытия позиции.
///
/// Внимание: вариант `Resolution` намеренно отсутствует. Закрытие по
/// итогу события (бинарная выплата CTF $1/$0) происходит в отдельном
/// пути [`crate::account::Account::resolve_pending_market`] — там же,
/// где живёт обновление `bankroll` и `SideStats.resolution_win/loss`.
/// `sell_gate`/`manage_positions`/`close_position` отвечают только за
/// **рыночные** выходы (TP/SL/Timeout/EV) **внутри** жизни маркета,
/// до резолюции; на `event_remaining_ms <= 0` они просто перестают
/// действовать (см. doc у `sell_gate`), а позиция дожидается
/// колбека резолюции.
#[derive(Debug, Clone, PartialEq)]
pub enum CloseReason {
    TakeProfit,
    StopLoss,
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

impl CloseReason {
    /// Можно ли «передумать» этот выход, если стакан слишком тонкий?
    ///
    /// `true` — выход **добровольный**: TP / EvExitProfit. Тут разумно
    /// применять [`MAX_SLIPPAGE_FROM_L1_PCT`] и не выходить на VWAP, который
    /// глубже best bid'а допустимого процента — ждём лучшего стакана в
    /// следующем тике, edge сохраняется.
    ///
    /// `false` — **обязательный** выход: SL / Timeout / EvExitLoss. Здесь
    /// удерживать позицию из-за широкого стакана опаснее, чем пройти
    /// slippage: SL/Timeout срабатывают именно тогда, когда дальнейшее
    /// удержание ухудшает PnL, а EvExitLoss — когда модель уже увидела
    /// негативный исход и продажа сейчас лучше ожидания резолюции (где
    /// `final_price` может быть $0). Slippage cap здесь намеренно
    /// отключён: лучше выйти на 5–10¢ хуже mid, чем уехать к резолюции
    /// проигравшего токена за $0.
    pub fn is_voluntary_exit(&self) -> bool {
        matches!(self, CloseReason::TakeProfit | CloseReason::EvExitProfit)
    }
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
    /// Попытка второй раз открыть позицию на **тот же** `asset_id`, пока
    /// первая ещё в `positions` (см. [`try_open_position`]: проверяется
    /// только в ветке `BuyGate::Proceed`, поэтому считает **сигналы
    /// Kelly**, а не каждый кадр удержания). Один токен = одна
    /// `OpenPosition` за раз — проще CLOB, TP/SL, бухгалтерия.
    pub(crate) same_asset_open_skips: usize,
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
    /// Кадры, в которых [`buy_gate`] вернул [`BuyGate::Proceed`], но
    /// суммарной USDC-глубины на ask-стаке кадра (`book_asks` * price)
    /// не хватило бы на `Y_TRAIN_NOMINAL_USDC = $200` номинала. Подсчёт
    /// **независим** от `kelly_strict_buy_skips` (тот считает реальные
    /// отказы `book_fill_buy_strict` в strict-режиме / `book_fill_buy`
    /// с total_shares=0): тут просто диагностический сигнал «как часто
    /// рынок недостаточно ликвиден для сценария Y-разметки».
    pub(crate) thin_book_skips: usize,
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

/// Накопленная статистика за версию (per-interval счётчики, без денег).
///
/// **Денежные** поля (bankroll/peak_bankroll/max_drawdown_pct) и
/// `update_drawdown` вынесены в [`crate::account::Account`] — он один на
/// процесс/группу `ProjectManager`-ов, чтобы экспозиция и drawdown
/// считались по единому счёту, а не по «параллельным» псевдо-счетам
/// каждого интервала.
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
}

// ─── Точка входа ──────────────────────────────────────────────────────────────

pub fn run_sim_mode() -> anyhow::Result<()> {
    let xframes_root = Path::new("xframes");
    if !xframes_root.exists() {
        anyhow::bail!("Папка xframes/ не найдена — сначала соберите данные (STATUS=default)");
    }

    crate::tee_log::init_tee_log_file(&xframes_root.join("last_history_sim.txt"), "sim")?;
    // Per-trade CSV-лог: одна строка на каждое закрытие (рыночное и
    // резолюционное). См. `crate::trade_csv_log` — анализируется отдельно
    // от текстового лога (pandas/duckdb по `last_history_sim_trades.csv`).
    crate::trade_csv_log::init_trade_csv_log_file(
        &xframes_root.join("last_history_sim_trades.csv"),
    )?;

    for currency_path in fs_sorted_dirs(xframes_root)? {
        let currency = dir_name(&currency_path);

        for version_path in fs_sorted_dirs(&currency_path)? {
            let version = dir_name(&version_path);
            if version.parse::<usize>().is_err() {
                continue;
            }

            // Один `Account` на пару `(currency, version)`: общий
            // bankroll/peak/drawdown между 5m и 15m интервалами,
            // зеркало real_sim-инварианта «единый кошелёк на PM».
            // Без этого 5m и 15m получали независимые псевдо-счета с
            // одинаковым стартовым `INITIAL_BANKROLL`, и итоговые
            // PnL/ROI/DD по интервалам было нельзя суммировать в одну
            // картину (см. account.rs). Сейчас `print_sim_stats` в
            // конце каждого интервала печатает CUMULATIVE состояние
            // одного и того же счёта.
            let mut account = Account::new();

            // Итерируемся непосредственно по `XFrameIntervalKind`, а
            // строковую метку (`"5m"` / `"15m"`) для путей и логов
            // получаем через единый `interval_label` (см. `real_sim.rs`).
            // Раньше шёл обратный путь — литерал `"5m"`/`"15m"` →
            // `XFrameIntervalKind` через `match` с `unreachable!`-веткой:
            // регресс «добавили новый интервал в литералах, забыли в
            // match'е» снимался только в рантайме. Теперь источник
            // истины — сам `enum`, а строка генерится из него.
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
                            );
                            sim_stats.events += 1;
                        }
                        Err(err) => tee_eprintln!("[sim] {}: {err}", file_path.display()),
                    }
                }

            
                print_sim_stats(&tag, &sim_stats, &account);
            }
        }
    }

    crate::trade_csv_log::finish_trade_csv_log();
    crate::tee_log::finish_tee_log();

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
) {
    // Лейн-ключи для `Account.pending_resolution` (см. также real_sim).
    // history_sim всегда работает за один маркет, поэтому stale-позиций
    // на этих ключах быть не должно — это проверяет debug_assert! в
    // конце `run_side_simulation`.
    let lane_key_up = (currency.to_string(), interval_kind, CurrencyUpDownOutcome::Up);
    let lane_key_down = (currency.to_string(), interval_kind, CurrencyUpDownOutcome::Down);
    // Дамп `xframe_dump.rs` уже отфильтрован по `stable=true` (см.
    // `dump_market_xframes_binary_lane` → `if !frame.stable { continue; }`),
    // поэтому повторный filter здесь был бы no-op'ом и лишним аллоком.
    // Стабильность всё равно перепроверяется в `buy_gate` через
    // `BuyGate::Unstable` — это симметрично с `real_sim` (там кадры
    // приходят из live-фанаута без предварительного фильтра).
    let frames_up: Vec<&XFrame<SIZE>>   = market_xframes.frames_up.iter().collect();
    let frames_down: Vec<&XFrame<SIZE>> = market_xframes.frames_down.iter().collect();

    // Бинарная выплата на резолюции: победивший токен → $1, проигравший → $0.
    // Используется только в финальном `resolve_pending_market_sync`
    // ниже; сами `sell_gate`/`manage_positions` про резолюцию ничего
    // не знают (см. doc у `CloseReason`).
    let up_won = market_xframes.up_won();

    // `market_id` дампа: оба `frames_up` / `frames_down` относятся к
    // одному CTF-маркету, поэтому достаточно взять у первого
    // непустого ряда. Если оба пусты — резолюция не нужна (ниже —
    // ранний выход).
    let market_id_opt: Option<String> = frames_up
        .first()
        .map(|f| f.market_id.clone())
        .or_else(|| frames_down.first().map(|f| f.market_id.clone()));

    // Сначала UP: все его surviving-позиции уезжают в pending, потом
    // DOWN. На обоих сторонах резолюция выполняется одним вызовом
    // `resolve_pending_market_sync` ниже — атомарная картина:
    // bankroll/stats обновляются вместе, как реальный CTF-payout
    // обоих токенов одного маркета.
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
        );
    }

    // Финальная резолюция: на этом моменте `account.pending_resolution`
    // по обоим лейн-ключам содержит ВСЕ surviving-позиции (run_side_simulation
    // переносит их туда после своего цикла). Sync-ядро Account'а
    // закроет их по бинарной выплате `up_won` и обновит stats обеих
    // сторон одним проходом — точно так же, как async-обёртка делает
    // в real_sim после колбека `xframe_dump`.
    if let Some(market_id) = market_id_opt {
        // `resolve_pending_market_sync` всегда печатает
        // `[resolve]`-строки через `tee_println!` (в файл
        // `xframes/last_history_sim.txt`). Раньше тут стоял `log=false`,
        // чтобы не забивать stdout history_sim'а сотнями маркетов;
        // сейчас логи идут в файл, в stdout всё равно дублируются —
        // но трассировка per-market PnL критичнее, чем компактный
        // вывод (агрегаты `print_sim_stats` показывают ИТОГ, но не
        // отдельные win/loss-сделки, по которым он сложился).
        account.resolve_pending_market_sync(
            sim_stats,
            currency,
            interval_kind,
            &market_id,
            up_won,
        );
    }

    // После резолюции pending обоих лейн-ключей обязан опустеть:
    // history_sim — один симулируемый маркет за вызов, и
    // `resolve_pending_market_sync` фильтрует ровно по этому
    // `(currency, interval, market_id)`. Если что-то осталось — это
    // либо bug дампа (чужой `market_id` в `frames_*`), либо bug
    // sync-ядра. Ловим в release-сборках тоже (раньше был
    // `debug_assert!`): тихая утечка stale-позиций в `pending_resolution`
    // от маркета к маркету «капает» в bankroll следующих маркетов
    // через cross-lane Kelly-сайзинг, и пропустить такой регресс при
    // прогоне в `--release` — дороже, чем падение симулятора с
    // понятным сообщением.
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

/// Один проход одной стороны (UP или DOWN) по своему временно́му ряду
/// кадров. Внутри: `manage_positions` → `try_open_position` →
/// **mark-to-market equity** drawdown. После завершения цикла все
/// surviving-позиции переезжают в `account.pending_resolution[lane_key]`
/// — оттуда их закроет финальный `Account::resolve_pending_market_sync`
/// в `simulate_event` по бинарному CTF-payout.
///
/// Equity на каждом тике считается как
/// `bankroll + Σ(local pos × current_prob) + Σ(pending × entry_prob)`,
/// симметрично `real_sim::tick_once` (фаза 2). Это критично: между
/// `open` и `close` реализованный `bankroll` не двигается, поэтому
/// если считать drawdown только по `bankroll`, метрика системно
/// занижается на длинных удержаниях, уходящих в красное и
/// закрывающихся через резолюцию. Pending-слагаемое нужно, чтобы
/// surviving UP-позиции, переехавшие после UP-прохода в
/// `account.pending_resolution[lane_key_up]`, не «исчезали» из equity
/// на DOWN-проходе и не давали искусственный drawdown.
///
/// Сайзинг новых позиций идёт от **available capital** =
/// `bankroll − Σ(open.entry_cost)` той же стороны (cross-side для
/// history_sim не нужен, т.к. UP и DOWN запускаются последовательно и
/// другая сторона на данном проходе всегда пуста).
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
) {
    if frames.is_empty() {
        return;
    }
    // Fallback на «битый» буфер: в штатных прогонах последний кадр уже
    // в резолюции (`event_remaining_ms ≤ 0`), и условие `is_last` ничего
    // нового не даёт. На truncated-дампах (writer оборвался раньше
    // резолюции) `is_last_idx` гарантирует, что surviving-позиции уйдут
    // в pending без попытки рыночного выхода — финальная бинарная
    // выплата делается через `resolve_pending_market_sync` в
    // `simulate_event` по реальному `up_won` маркета.
    let last_idx = frames.len().saturating_sub(1);

    for (idx, frame) in frames.iter().enumerate() {
        let is_last_idx = idx == last_idx;
        // Booster-инференсы считаем ОДИН раз за кадр, перед передачей в
        // `manage_positions` / `try_open_position`. На однопоточном пути
        // history_sim это просто аккуратнее — общая API с real_sim, где
        // тот же предвычисленный `pnl_raw` / `p_win_now` живёт ВНЕ
        // write-локов (без него 4×N воркеров блокировали бы друг друга
        // на время `predict_frame`).
        let p_win_now = compute_p_win_now(
            frame,
            booster_resolution,
            calibration_resolution,
        );
        let pnl_inference = compute_pnl_inference(frame, booster_pnl, calibration_pnl);

        // 1) Рыночные закрытия (TP/SL/Timeout/EV). Split-borrow по
        //    `account`: одновременно нужен `&mut bankroll` (PnL
        //    закрытий) и `&mut pending_resolution[lane_key]` (sink
        //    stale-позиций — ловит асимметричные дампы, по контракту
        //    history_sim всегда пуст до финального переноса ниже).
        //    На `event_remaining_ms ≤ 0` `manage_positions` сама
        //    ничего не закрывает — `sell_gate` возвращает Hold,
        //    и позиция дождётся `resolve_pending_market_sync`.
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
            );
        }

        // 2) Открытие новой позиции — от available_bankroll, иначе Kelly
        //    раздувает экспозицию: bankroll не уменьшается на open
        //    (всё списание идёт через PnL на close), и без этой коррекции
        //    последовательные сигналы открыли бы 5×10% = 50% bankroll
        //    параллельно вместо одного 10%.
        let same_side_locked: f64 = positions.iter().map(|p| p.entry_cost).sum();
        let available = (account.bankroll - same_side_locked).max(0.0);
        try_open_position(
            frame,
            pnl_inference,
            positions,
            side_stats,
            available,
            None,
            "",
            currency,
        );

        // 3) Mark-to-market equity на каждом тике (а не только на сделке).
        //    Не используем `currency_implied_prob.unwrap_or(0.0)` бездумно:
        //    `None` означает «mark-to-market неизвестен», и оценивать
        //    позицию в нуль — это занижение equity на ровном месте; в
        //    таком кадре пропускаем апдейт, метрика подождёт следующего.
        //
        //    Equity-формула симметрична `real_sim::tick_once` (фаза 2):
        //      equity = bankroll
        //             + Σ(local positions × current_prob)
        //             + Σ(account.pending_resolution[*] × entry_prob)
        //
        //    Pending-учёт критичен на DOWN-проходе: к этому моменту
        //    surviving позиции UP-стороны уже переехали в
        //    `account.pending_resolution[lane_key_up]` (см. блок ниже,
        //    после цикла), и без них equity занижен на стоимость
        //    UP-pending → искусственный drawdown, которого реально нет
        //    (mark-to-market при переезде в pending не двигается:
        //    `entry_prob × shares ≈ entry_cost`, реальный PnL прилетит
        //    в `resolve_pending_market_sync` после обоих side-проходов).
        //
        //    `account.positions` в history_sim во время `run_side_simulation`
        //    всегда пуст (UP/DOWN запускаются последовательно, и каждый
        //    держит свои живые позиции в **локальной** Vec, не в
        //    `account.positions`; между `simulate_event`-ами
        //    `assert!`-инвариант ниже гарантирует пустое pending).
        //    Поэтому `account.positions`-слагаемое тут опускаем — оно
        //    тождественно ноль; в real_sim'е оно нужно потому, что там
        //    позиции 4×N лейнов реально живут в `account.positions`.
        if let Some(prob) = frame.currency_implied_prob {
            let prob = prob.clamp(0.0, 1.0);
            let positions_value: f64 = positions.iter().map(|p| p.shares_held * prob).sum();
            let pending_value: f64 = account
                .pending_resolution
                .values()
                .flat_map(|v| v.iter())
                .map(|p| p.shares_held * p.entry_prob)
                .sum();
            let equity = account.bankroll + positions_value + pending_value;
            account.update_drawdown(equity);
        }
    }

    // Перенос surviving-позиций в pending: их финал-резолюция —
    // ответственность `Account::resolve_pending_market_sync`, который
    // вызывает `simulate_event` сразу после возврата из обоих
    // `run_side_simulation` (UP и DOWN). На этом этапе `positions`
    // обнуляется — дальше им жить уже в `Account.pending_resolution`.
    if !positions.is_empty() {
        let pending = account
            .pending_resolution
            .entry(lane_key.clone())
            .or_default();
        pending.append(positions);
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
/// Результат предвычисленного booster + калибровка-прохода PnL-модели:
/// `raw` — сырой инференс (используется для проверки `SIM_BUY_THRESHOLD`),
/// `pred` — после применения [`Calibration`] (идёт в Kelly-формулу).
///
/// Оба значения нужны на одном тике: порог сравнивается с `raw` (как и было
/// до выноса), а Kelly использует калиброванное `pred`. Возвращаем парой,
/// чтобы caller считал инференс ОДИН раз и не повторял `Calibration::apply`
/// внутри [`buy_gate`].
#[derive(Clone, Copy, Debug)]
pub struct PnlInference {
    pub raw: f32,
    pub pred: f32,
}

/// Booster-инференс + калибровка PnL-модели для текущего кадра — основной
/// «дорогой» шаг buy-decision'а. Возвращает `None`, если входить заведомо
/// нельзя (`event_remaining_ms < MIN_ENTRY_REMAINING_MS`, `!frame.stable`,
/// нет `currency_implied_prob`) или если `predict_frame` отказался (лаг
/// превысил [`PNL_MAX_LAG`]).
///
/// Калибровку считаем **здесь же**, а не в [`buy_gate`]: она тоже
/// детерминирована от `raw` и не зависит от состояния (`Calibration` —
/// иммутабельная map / spline). Тогда `buy_gate` совсем не видит `&Booster`
/// / `Option<&Calibration>` и под write-локами в `real_sim::tick_once`
/// никаких CPU-инференсов не остаётся.
///
/// Вынесен из [`buy_gate`], чтобы [`crate::real_sim::tick_once`] мог делать
/// инференс **до** взятия write-локов (`state.write() + account.write()`):
/// один predict за кадр против двух раньше (один ради `may_open` снаружи,
/// второй внутри `try_open_position` под локами). На однопоточном пути
/// `history_sim` разницы нет — call-site вызывает `compute_pnl_inference`
/// сразу перед `buy_gate` / `try_open_position`.
pub(crate) fn compute_pnl_inference(
    frame: &XFrame<SIZE>,
    booster_pnl: &Booster,
    calibration_pnl: Option<&Calibration>,
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
    let pred = calibration_pnl.map_or(raw, |c| c.apply(raw));
    Some(PnlInference { raw, pred })
}

/// Резолюционная вероятность победы для активного hold-zone — booster +
/// калибровка одной командой. `None`, если вызывать модель не имеет смысла
/// (вне hold-zone, нет `booster_resolution`, predict отказался из-за
/// [`RESOLUTION_MAX_LAG`]).
///
/// Аналогично [`compute_pnl_raw`], вынесено для real_sim — чтобы
/// resolution-инференс уходил **до** write-локов. На history_sim это
/// один и тот же кадр × все позиции одной стороны, и `manage_positions`
/// раньше делал predict внутри себя; теперь предвычисление делает caller.
///
/// **Гейт `has_positions` убран намеренно**: раньше при `positions=[]`
/// resolution-инференс пропускался, и на тике, где позиция только что
/// открывалась в hold-zone, `pos.p_win_ema` сидел `None` весь этот тик.
/// EMA `p_win` могла сидеть «недосвезенной» лишний тик, EV-exit
/// откладывался. Теперь predict считается каждый тик в hold-zone
/// безусловно, и при `try_open_position` создающем позицию её
/// первый же `manage_positions` следующего тика стартует с уже
/// инициализированной EMA. Стоимость: ~1 inference resolution-модели
/// на кадр в hold-zone (до 45 тиков на маркет) даже без активных
/// позиций — пренебрежимо для real_sim/history_sim.
pub(crate) fn compute_p_win_now(
    frame: &XFrame<SIZE>,
    booster_resolution: Option<&Booster>,
    calibration_resolution: Option<&Calibration>,
) -> Option<f64> {
    let in_hold_zone = frame.event_remaining_ms > 0
        && frame.event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000;
    if !in_hold_zone {
        return None;
    }
    booster_resolution.and_then(|b| {
        predict_frame(b, frame, RESOLUTION_MAX_LAG).map(|raw| {
            calibration_resolution.map_or(raw, |c| c.apply(raw)) as f64
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
///
/// `pnl_inference` — **заранее посчитанный** booster + калибровка PnL-модели
/// (см. [`compute_pnl_inference`]). Гейт сам по себе не делает ни predict, ни
/// `Calibration::apply` — это позволяет `real_sim::tick_once` считать всю
/// «дорогую» часть decision-tree ОДИН раз за кадр **вне** write-локов и
/// переиспользовать в обоих местах (`may_open` и `try_open_position`), вместо
/// двух predict'ов под лок-критсекцией.
pub(crate) fn buy_gate(
    frame: &XFrame<SIZE>,
    pnl_inference: Option<PnlInference>,
    bankroll: f64,
    strict_book: Option<&StrictBook>,
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
    //
    // При наличии HTTP-стакана prob берём по нему (mid лучшего bid/ask,
    // см. `effective_implied_prob`): WS отстаёт от HTTP, и Kelly-расчёт
    // на свежем mid точнее. На history_sim-пути strict_book всегда `None`,
    // поведение совпадает с прежним.
    let Some(entry_prob) = effective_implied_prob(frame, strict_book) else {
        return BuyGate::BelowThreshold;
    };
    // `pnl_inference == None` означает либо «inference не звали из-за late_entry/
    // !stable/нет implied_prob» (мы уже выше вернулись бы), либо «predict
    // вернул None — лаг превысил `PNL_MAX_LAG`». В обоих случаях это эквивалент
    // прежнего `predict_frame(...) == None` → `BelowThreshold`, статистика
    // совпадает с прошлым поведением (диагностические суммы не трогаются).
    let Some(PnlInference { raw, pred }) = pnl_inference else {
        return BuyGate::BelowThreshold;
    };
    if raw < SIM_BUY_THRESHOLD {
        return BuyGate::BelowThreshold;
    }
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
///
/// `currency` — лейбл валюты лейна (`btc` и т.п.). Используется только
/// для записи в `OpenPosition.currency` (нужна для per-trade CSV); на
/// торговую логику не влияет.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_open_position(
    frame: &XFrame<SIZE>,
    pnl_inference: Option<PnlInference>,
    positions: &mut Vec<OpenPosition>,
    stats: &mut SideStats,
    bankroll: f64,
    strict_book: Option<&StrictBook>,
    log_tag: &str,
    currency: &str,
) -> bool {
    // Единая точка принятия решения о входе: ниже — только бухгалтерия
    // (счётчики пропусков + `open_position` при успехе). Вся логика
    // «брать/не брать» — в `buy_gate`, её же использует `real_sim` как
    // дешёвый gate до HTTP-запроса стакана (`matches!(.., Proceed)`).
    //
    // Нет `currency_implied_prob` — `buy_gate` всё равно вернул бы
    // `BelowThreshold` (no-op здесь, `diag_sum_*` не трогаются), так
    // что коротко замыкаем без обращения к gate и пропускаем тик.
    //
    // `effective_implied_prob` берёт prob из HTTP-стакана при его
    // наличии (mid лучшего bid/ask) — то же самое значение, что увидит
    // `buy_gate` ниже, чтобы `diag_sum_entry_prob` корректно отражал
    // именно ту цену, по которой принималось решение о входе.
    //
    // `pnl_inference` пробрасывается caller'ом (см. [`compute_pnl_inference`])
    // — это позволяет real_sim считать booster + калибровку ОДИН раз вне локов
    // и переиспользовать здесь, вместо повторного `predict_frame` /
    // `Calibration::apply` под write-локом.
    //
    // Второй вход на тот же `asset_id` запрещён: проверка в ветке
    // `BuyGate::Proceed` (см. `same_asset_open_skips`), не на каждом кадре,
    // чтобы не раздувать счётчик на всём удержании.
    let Some(entry_prob) = effective_implied_prob(frame, strict_book) else {
        return false;
    };
    match buy_gate(frame, pnl_inference, bankroll, strict_book) {
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

            // Диагностика «насколько тонкий стакан» — проверка независима
            // от kelly_strict_buy_skips (тот ловит реальный отказ
            // book_fill_buy/strict). Тут просто смотрим на `Y_TRAIN_NOMINAL_USDC`
            // — тот же $200, под который размечается y_train. Если на
            // ask'е суммарно меньше $200 USDC, исполнение по WS-лестнице
            // частично «сожжёт» остаток (в book_fill_buy non-strict он
            // не fallback'ится, см. doc у book_fill_buy).
            if let Some(asks) = frame.book_asks.as_deref() {
                let depth_usdc: f64 = asks
                    .iter()
                    .filter(|l| l.price > 0.0 && l.size > 0.0)
                    .map(|l| l.price * l.size)
                    .sum();
                if depth_usdc < crate::xframe::Y_TRAIN_NOMINAL_USDC {
                    stats.thin_book_skips += 1;
                }
            }
            match open_position(frame, size, stats, strict_book, raw, pred, kelly_f, currency) {
                Some(pos) => {
                    // Гистограммы заполняем только для **успешно открытых**
                    // позиций — нас интересует распределение реальных входов,
                    // а не «kelly_skip / thin_book_skip». Бакет берём по
                    // `pos.entry_prob` (фактическая цена входа после `effective_implied_prob`)
                    // и `pred` (калиброванный); это две точки, между которыми
                    // живёт edge модели.
                    let bucket_entry = prob_bucket_index(pos.entry_prob);
                    let bucket_pred = prob_bucket_index(pred as f64);
                    stats.histogram_entry_prob[bucket_entry] += 1;
                    stats.histogram_cal_pred[bucket_pred] += 1;
                    positions.push(pos);
                    true
                }
                None => {
                    // Сюда попадаем в двух случаях:
                    //   * strict-режим (real_sim): HTTP-стакана не хватило
                    //     для покупки на `size` USDC ([`book_fill_buy_strict`]
                    //     вернула `None`).
                    //   * non-strict (history_sim): WS-стакан совсем пуст —
                    //     ни одного валидного уровня в `book_ask_l{1,2,3}_*`,
                    //     `book_fill_buy` вернул `total_shares = 0`, и
                    //     `open_position` отказалась создавать позицию с
                    //     нулевой экспозицией (см. её doc).
                    // Считаем оба под `kelly_strict_buy_skips`: это
                    // «вход пропустили из-за стакана, не из-за Kelly».
                    stats.kelly_strict_buy_skips += 1;
                    let prefix = if log_tag.is_empty() {
                        String::new()
                    } else {
                        format!("[{log_tag}] ")
                    };
                    tee_eprintln!(
                        "{prefix}buy skip: ask-стакан не закрывает size={size:.4} USDC — пропускаем вход"
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
pub(crate) fn sell_gate(
    pos: &OpenPosition,
    frames_held: usize,
    frame: &XFrame<SIZE>,
    is_last: bool,
    p_win_now: Option<f64>,
    strict_book: Option<&StrictBook>,
) -> SellGate {
    // Событие закончилось ИЛИ это принудительно последний кадр прогона —
    // рыночные выходы (TP/SL/Timeout/EV) больше не применимы. Позиция
    // дождётся резолюционного колбека
    // [`crate::account::Account::resolve_pending_market`] и закроется
    // по бинарной выплате CTF ($1/шер победителю, $0 проигравшему —
    // без комиссии). На этом тике для неё — Hold.
    //
    // `is_last` — fallback для **битого буфера** в history_sim: дамп
    // мог оборваться раньше, чем `event_remaining_ms ≤ 0` (например,
    // из-за раннего kill / краша writer'а). На truncated-дампе TP/SL/
    // Timeout по последнему кадру дали бы рыночное закрытие по
    // неполным данным с фейковой комиссией, в то время как сам маркет
    // в реальности всё равно резолвнулся (`up_won` известен в
    // `simulate_event` через `MarketXFramesDump::up_won`). Hold +
    // путь через pending → `resolve_pending_market_sync` даёт
    // фактическую бинарную выплату.
    //
    // В real_sim понятия «последний кадр» не существует (live-поток),
    // поэтому caller всегда передаёт `is_last = false`.
    if is_last || frame.event_remaining_ms <= 0 {
        return SellGate::HoldPnl;
    }

    // Без `currency_implied_prob` решение принять нельзя — ни TP/SL/Timeout,
    // ни EV-exit посчитать не из чего. Защитно возвращаем Hold, а не
    // panic: повторный тик с валидным кадром обработает позицию.
    //
    // Если передан HTTP-стакан, берём `current_prob` по нему (mid лучшего
    // bid/ask) — HTTP всегда свежее WS, и решение о выходе должно
    // строиться на актуальной цене. Без strict_book — `frame.currency_implied_prob`.
    let Some(current_prob) = effective_implied_prob(frame, strict_book) else {
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
        // EV-оценка для решения «закрывать или ждать»: применяем cap
        // (`Some(MAX_SLIPPAGE_FROM_L1_PCT)`), чтобы решение было
        // **консервативным** — мы не хотим закрывать на слишком
        // плохой цене, если на следующем тике стакан может улучшиться.
        // Если cap зарезал — `gross_usdc_opt = None`, выход не считаем,
        // позиция уезжает в `HoldResolution`. Если EV всё-таки
        // сработает позже и закрытие пойдёт через `close_position` с
        // `CloseReason::EvExitLoss` — там cap уже отключён (must-exit),
        // см. doc у [`CloseReason::is_voluntary_exit`].
        let gross_usdc_opt = match strict_book {
            Some(book) => book_fill_sell_strict(book, pos.shares_held, Some(MAX_SLIPPAGE_FROM_L1_PCT)),
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
/// **На `event_remaining_ms ≤ 0` возвращаем `false`**: завершившееся
/// событие закрывается через резолюционный колбек
/// [`crate::account::Account::resolve_pending_market`] (бинарная
/// выплата `0`/`1` без HTTP), а `manage_positions` на таком кадре
/// для рыночных выходов уже неактивен — `sell_gate` сам возвращает
/// `Hold` при `event_remaining_ms <= 0`.
pub(crate) fn any_position_would_sell(
    positions: &[OpenPosition],
    frame: &XFrame<SIZE>,
) -> bool {
    if positions.is_empty() || frame.event_remaining_ms <= 0 {
        return false;
    }
    // Этот предикат — только для real_sim (history_sim не дёргает HTTP-стакан),
    // и в real_sim `is_last` не существует. Передаём `false`.
    positions.iter().any(|pos| {
        // Stale-позиции (asset_id ≠ текущему маркету лейна) на этом
        // тике скипаются в `manage_positions`, поэтому и HTTP-стакан
        // ради них тащить не нужно: `sell_gate` для них всё равно
        // не вызывается, EV/SL/TP по чужому маркету бессмысленны.
        if pos.asset_id != frame.asset_id {
            return false;
        }
        matches!(
            sell_gate(
                pos,
                // `frames_held` как бы после инкремента в `manage_positions`.
                pos.frames_held + 1,
                frame,
                // `is_last = false`: real_sim — это live-поток без понятия
                // последнего кадра, fallback на битый буфер тут не нужен.
                false,
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
/// `pending_resolution` — sink-корзина того же лейна, см.
/// [`crate::account::Account::pending_resolution`]. Сюда уезжают
/// stale-позиции (asset_id не совпал с фреймом), чтобы дождаться
/// post-resolution колбека и закрыться по реальному `final_price`
/// маркета, к которому они принадлежали. На активную книгу `positions`
/// этого лейна они больше не влияют — ни sell_gate, ни Kelly-сайзинг
/// (см. `try_open_position` → `available_bankroll`) их не учитывают;
/// учёт `entry_cost` для них переезжает на `pending_resolution`.
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
    log_tag: &str,
) -> bool {
    for pos in positions.iter_mut() { pos.frames_held += 1; }

    // `p_win_now` приходит уже посчитанным caller'ом (см. [`compute_p_win_now`]).
    // Это resolution-инференс booster + калибровка — единственный «дорогой»
    // шаг decision-tree продаж. Вынос наверх по стэку нужен в первую очередь
    // для real_sim: predict выполняется ВНЕ write-локов, не блокируя 4×N
    // параллельных воркеров. На history_sim семантика та же (вызов раз на
    // кадр, переиспользуется для всех позиций стороны), просто сама арифметика
    // живёт в caller'е, а не здесь.

    let mut sold = false;
    let mut remaining: Vec<OpenPosition> = Vec::new();
    for mut pos in positions.drain(..) {
        // Stale-позиции (asset_id ≠ текущему фрейму) — это позиции
        // ОТ предыдущего маркета лейна (5m/15m раунд сменился, токены
        // уже другие). К ним нельзя применять `sell_gate(frame, ...)`:
        // ни `currency_implied_prob`, ни `event_remaining_ms`, ни
        // hold-zone окно нового маркета не описывают старую позицию.
        //
        // **Перевозим их из активной книги в `pending_resolution`**
        // (см. doc-комментарий `Account::pending_resolution`). Там
        // они дождутся post-resolution колбека по реальному маркету
        // и закроются через [`Account::resolve_pending_market`] по
        // `final_price`, который придёт по тому же каналу, что
        // снабжает `xframe_dump::final_price`.
        if pos.asset_id != frame.asset_id {
            pending_resolution.push(pos);
            continue;
        }
        // Весь decision-tree (EMA-апдейт `p_win_ema`, проверки
        // TP/SL/Timeout/EV) живёт в `sell_gate`. Здесь подаём
        // `pos.frames_held` уже после инкремента и **готовый**
        // `p_win_now` (один инференс на кадр выше). Текущий
        // `current_prob` гейт достаёт из `frame.currency_implied_prob`.
        // На `event_remaining_ms ≤ 0` (и на `is_last` для truncated-дампа
        // в history_sim) `sell_gate` сам возвращает Hold, и резолюция
        // позиции уйдёт в колбек
        // [`crate::account::Account::resolve_pending_market`].
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

/// Бакет вероятности `[0..1]` в индекс `0..=4` для гистограмм
/// [`SideStats::histogram_entry_prob`] / [`SideStats::histogram_cal_pred`].
/// Сетка: `[0.0..0.2)`, `[0.2..0.4)`, `[0.4..0.6)`, `[0.6..0.8)`, `[0.8..1.0]`.
/// Значения `< 0` или `NaN` уходят в бакет `0`, `>= 1.0` — в `4`.
pub(crate) fn prob_bucket_index(p: f64) -> usize {
    if !p.is_finite() || p < 0.0 {
        return 0;
    }
    let idx = (p * 5.0).floor() as i64;
    idx.clamp(0, 4) as usize
}

// ─── Торговые операции с учётом комиссий ──────────────────────────────────────

/// Открывает виртуальную позицию за `position_size` USDC.
///
/// Цена исполнения определяется обходом ask-стакана (L1→L2→L3): если L1 не хватает
/// ликвидности — добираем с L2, затем L3. VWAP покупки = `position_size / total_shares`.
/// Taker-комиссия вычитается из полученных шерсов:
/// `actual_shares = nominal_shares − nominal_shares × FEE_RATE × p × (1−p)`
///
/// Параметры `raw_pred_at_open`/`cal_pred_at_open`/`kelly_f_at_open` —
/// диагностические значения decision-tree входа (`BuyGate::Proceed`),
/// сохраняются в [`OpenPosition`] и потом печатаются в per-trade CSV-логе.
/// На решение о торговле здесь не влияют (gate уже разрешил вход).
///
/// `currency` — лейбл валюты текущего лейна (`btc` и т.п.). Сохраняется
/// в [`OpenPosition::currency`] для per-trade CSV; на торговую логику
/// не влияет.
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
) -> Option<OpenPosition> {
    let (buy_price, nominal_shares) = match strict_book {
        Some(book) => book_fill_buy_strict(book, position_size)?,
        None => book_fill_buy(frame, position_size),
    };
    // Симметрично с strict-режимом: если на трёх верхних уровнях ask-стакана
    // ничего нет (`total_shares == 0`), позицию не открываем. Раньше
    // `book_fill_buy` молча добивал по `currency_implied_prob`, теперь —
    // пессимистичный режим без fallback'а (см. doc у `book_fill_buy`).
    if nominal_shares <= 0.0 {
        return None;
    }
    let buy_price = buy_price.clamp(0.001, 0.999);

    let fee_usdc = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * buy_price * (1.0 - buy_price);
    let fee_shares = fee_usdc / buy_price;
    let actual_shares = nominal_shares - fee_shares;

    stats.fees_paid += fee_usdc;

    // `entry_prob` для последующих TP/SL/EV-проверок: с HTTP-стаканом
    // берём mid лучшего bid/ask (живое значение, симметрично с
    // sell_gate/buy_gate), без него — `frame.currency_implied_prob`.
    // `unwrap_or(buy_price)` остаётся защитой — если ни strict, ни WS
    // не дали prob, но fill прошёл (теоретически только в strict при
    // частичном `asks`-стаке без bid'ов), `buy_price` лучше дефолтного 0.5.
    let entry_prob = effective_implied_prob(frame, strict_book).unwrap_or(buy_price);

    Some(OpenPosition {
        asset_id: frame.asset_id.clone(),
        market_id: frame.market_id.clone(),
        shares_held: actual_shares,
        entry_prob,
        entry_cost: position_size,
        frames_held: 0,
        p_win_ema: None,
        raw_pred_at_open,
        cal_pred_at_open,
        kelly_f_at_open,
        event_remaining_ms_at_open: frame.event_remaining_ms,
        xframe_interval_type_at_open: frame.xframe_interval_type,
        currency_up_down_outcome_at_open: frame.currency_up_down_outcome,
        currency: currency.to_string(),
    })
}

/// Закрывает позицию **рыночным** выходом и возвращает P&L в USDC
/// (может быть отрицательным).
///
/// Покрывает только `TP / SL / Timeout / EvExit*`: рыночная продажа
/// по bid-стакану (L1→L2→L3), VWAP продажи = `gross_usdc / shares_held`,
/// taker-fee вычитается из USDC:
/// `net = gross − shares × FEE_RATE × p_sell × (1−p_sell)`.
///
/// Резолюционные закрытия (бинарная выплата $1/$0 без комиссии)
/// здесь НЕ обрабатываются — для них есть отдельный путь
/// [`crate::account::Account::resolve_pending_market`], в т.ч. с
/// собственным обновлением `SideStats.resolution_win/loss`.
fn close_position(
    pos: &OpenPosition,
    exit_price: f64,
    reason: &CloseReason,
    frame: &XFrame<SIZE>,
    stats: &mut SideStats,
    strict_book: Option<&StrictBook>,
) -> Option<f64> {
    // Slippage cap включаем **только** для добровольных выходов
    // (TP / EvExitProfit) — там есть смысл «передумать» и подождать
    // лучшего стакана. Для SL / Timeout / EvExitLoss cap отключён
    // (`None`): удерживание позиции из-за тонкого стакана хуже, чем
    // выход с повышенным slippage, потому что эти причины именно
    // сигнализируют о том, что дальнейшее ожидание ухудшает PnL.
    let slippage_cap = if reason.is_voluntary_exit() {
        Some(MAX_SLIPPAGE_FROM_L1_PCT)
    } else {
        None
    };
    let gross_usdc = match strict_book {
        Some(book) => book_fill_sell_strict(book, pos.shares_held, slippage_cap)?,
        None => book_fill_sell(frame, pos.shares_held),
    };
    let sell_price = if pos.shares_held > 0.0 {
        (gross_usdc / pos.shares_held).clamp(0.001, 0.999)
    } else {
        exit_price.clamp(0.001, 0.999)
    };
    let fee_usdc = pos.shares_held * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell_price * (1.0 - sell_price);
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
    crate::trade_csv_log::write_trade_csv_row(crate::trade_csv_log::TradeCsvRow {
        market_id: &pos.market_id,
        asset_id: &pos.asset_id,
        side: side_str,
        interval: interval_str,
        currency: &pos.currency,
        exit_reason: trade_csv_close_reason_label(reason),
        entry_prob: pos.entry_prob,
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
    });

    Some(pnl)
}

/// Лейбл интервала позиции для CSV: `"5m"` / `"15m"` / `"unknown"`.
/// Использует [`XFrameIntervalKind::from_i32`] и общий
/// [`crate::real_sim::interval_label`].
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

/// Текстовый лейбл [`CloseReason`] для CSV-колонки `exit_reason`.
/// Стабильные значения (не меняются по локали / порядку enum'а) —
/// чтобы внешний анализ CSV не зависел от Debug-печати.
pub(crate) fn trade_csv_close_reason_label(reason: &CloseReason) -> &'static str {
    match reason {
        CloseReason::TakeProfit => "TP",
        CloseReason::StopLoss => "SL",
        CloseReason::Timeout => "Timeout",
        CloseReason::EvExitProfit => "EvExitProfit",
        CloseReason::EvExitLoss => "EvExitLoss",
    }
}

// ─── Обход стакана ────────────────────────────────────────────────────────────

/// Покупка `position_size` USDC по ask-стакану кадра ([`XFrame::book_asks`]).
///
/// Возвращает `(vwap_price, total_nominal_shares)`.
///
/// Идём по полному ask-стакану кадра (`book_asks`, от лучшего к худшему):
/// L1/L2/L3 фичи параллельно живут отдельно для XGBoost, исполнение же —
/// по единой лестнице, без ручного перечисления слотов. Если в будущем
/// WS начнёт отдавать L4+, этот код менять не придётся.
///
/// **Пессимистичный режим (симметрично [`book_fill_sell`])**: если
/// суммарной ликвидности на всех уровнях не хватает на `position_size`
/// USDC, остаток USDC **сжигается** — мы записываем
/// `entry_cost = position_size`, но шерсов получаем только столько,
/// сколько закрыли реальными уровнями. VWAP тогда автоматически
/// становится хуже (`position_size / total_shares`): деньги ушли,
/// экспозиция меньше, чем хотели.
///
/// Раньше остаток добивался по `currency_implied_prob`, что **системно
/// завышало** оценку модели: backtest как будто всегда исполнялся по
/// mid даже для тех тиков, где реальной L4+ глубины никто не наблюдал.
/// Это давало оптимистичный fill на хвосте и ассиметрию с
/// `book_fill_sell` (которая на нехватке bid'ов сжигает шерсы).
/// Теперь оба направления симметрично пессимистичные: на тонком стакане
/// и вход, и выход показывают худший результат, что лучше отражает
/// worst-case live-исполнения.
///
/// Если в `book_asks` нет ни одного валидного уровня (`total_shares = 0`),
/// `open_position` отказывается открывать (`None`) — вход пропускается
/// аналогично strict-режиму.
fn book_fill_buy(frame: &XFrame<SIZE>, position_size: f64) -> (f64, f64) {
    let mut remaining_usdc = position_size;
    let mut total_shares = 0.0_f64;

    // `book_asks: Option<Vec<...>>` — `None` встречается у легаси-дампов до
    // миграции на полную лестницу: тогда фоллбэчимся на L1/L2/L3 фичи кадра
    // (тот же набор, что использовался до перехода на векторы), чтобы
    // поведение `book_fill_buy` на старых кадрах не деградировало до пустого
    // стакана.
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

    if remaining_usdc > 1e-9 {
        let fallback = frame.currency_implied_prob
            .unwrap_or(0.5)
            .clamp(0.001, 0.999);
        total_shares += remaining_usdc / fallback;
    }

    let vwap = if total_shares > 0.0 { position_size / total_shares } else { 0.5 };
    (vwap, total_shares)
}

/// Продажа `shares_to_sell` по bid-стакану кадра ([`XFrame::book_bids`]).
///
/// Возвращает валовый USDC до вычета fee.
///
/// Идём по полному bid-стакану кадра (`book_bids`, от лучшего к худшему):
/// L1/L2/L3 фичи остаются для модели отдельно, исполнение работает по
/// единой лестнице.
///
/// Шеры, для которых не хватает bid-ликвидности, **сжигаются**: они
/// вносят `0` USDC в выручку — то есть учитываются как полная потеря.
/// Это явная замена прежнего fallback'а по `currency_implied_prob`,
/// который **системно завышал** оценку модели: он позволял допродать
/// любой остаток «бесплатно по mid», тогда как в живой торговле
/// непокрытый ликвидностью объём — это либо рейс по более худшим
/// уровням стакана (скрытым в L4+), либо вообще невозможность
/// исполнения.
///
/// Для оценки обученной модели worst-case-лосс на хвосте — корректнее,
/// чем оптимистичный fallback. На стороне `real_sim` строгий вариант
/// [`book_fill_sell_strict`] вообще возвращает `None` (откладываем
/// продажу до следующего тика); здесь же позиция **обязана закрыться**
/// (Resolution на конце буфера, либо явный TP/SL/Timeout/EV), поэтому
/// «отложить» нельзя — выбираем консервативную оценку выручки.
fn book_fill_sell(frame: &XFrame<SIZE>, shares_to_sell: f64) -> f64 {
    let mut remaining = shares_to_sell;
    let mut total_usdc = 0.0_f64;

    // `book_bids: Option<Vec<...>>` — `None` это легаси-дамп без сохранённой
    // лестницы. Фоллбэчимся на L1/L2/L3 фичи (как до миграции на векторы) —
    // иначе старые кадры детерминированно отдавали бы выручку 0 USDC и ломали
    // back-test на исторических данных.
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

    for level in bids {
        if level.price <= 0.0 || level.size <= 0.0 { continue }

        if remaining <= level.size {
            total_usdc += remaining * level.price;
            break;
        } else {
            total_usdc += level.size * level.price;
            remaining -= level.size;
        }
    }

    total_usdc
}

/// Пересобирает [`BookLevel`]-лестницу из трёх кандидатов `(price, size)` в
/// порядке «лучший → худший». Невалидные уровни (отсутствие цены/размера,
/// нулевые/отрицательные/нефинитные значения) пропускаются.
///
/// Используется как фоллбэк в [`book_fill_buy`]/[`book_fill_sell`] для
/// легаси-кадров без сохранённой полной лестницы (`book_bids`/`book_asks =
/// None` после миграции — см. [`crate::migration`]).
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
        "raw≥thr={} avg_raw={:.3} avg_cal={:.3} avg_entry={:.3} avg_kelly_f={:.4} kelly_skips={} same_asset_open_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={} thin_book_skips={}",
        s.raw_above_threshold,
        s.diag_sum_raw / n,
        s.diag_sum_calibrated / n,
        s.diag_sum_entry_prob / n,
        s.diag_sum_kelly_f / n,
        s.kelly_skips,
        s.same_asset_open_skips,
        s.kelly_strict_buy_skips,
        s.kelly_strict_sell_skips,
        s.thin_book_skips,
    );
    tee_println!("[sim] {tag} [{side_label}]   {diag}");

    if s.trades == 0 {
        tee_println!("[sim] {tag} [{side_label}]: нет сделок");
        return;
    }
    let win_rate = s.wins as f64 / s.trades as f64 * 100.0;
    let avg_pnl = s.pnl_usd / s.trades as f64;
    // `Res✓={n}(profit={p}/loss={l})`: первый счётчик — token-outcome
    // («токен резолвнулся как победивший»), скобки — разбивка по знаку
    // pnl. См. doc у `SideStats::resolution_win_profit/loss` — при дорогих
    // входах `resolution_win_loss` может быть существенно > 0 даже при
    // правильно угаданном исходе.
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

    // Гистограмма `entry_prob` / калиброванного `pred` на момент входа
    // — ровно на тех 5 бакетах по 0.2, что и [`prob_bucket_index`].
    // Полезно для проверки: если все входы в `[0.6..0.8)`, edge модели
    // мал, и любая ошибка калибровки немедленно «съедает» плюс.
    tee_println!(
        "[sim] {tag} [{side_label}] entry_prob hist (0..0.2 / 0.2..0.4 / 0.4..0.6 / 0.6..0.8 / 0.8..1): {} / {} / {} / {} / {}",
        s.histogram_entry_prob[0], s.histogram_entry_prob[1], s.histogram_entry_prob[2],
        s.histogram_entry_prob[3], s.histogram_entry_prob[4],
    );
    tee_println!(
        "[sim] {tag} [{side_label}] cal_pred  hist (0..0.2 / 0.2..0.4 / 0.4..0.6 / 0.6..0.8 / 0.8..1): {} / {} / {} / {} / {}",
        s.histogram_cal_pred[0], s.histogram_cal_pred[1], s.histogram_cal_pred[2],
        s.histogram_cal_pred[3], s.histogram_cal_pred[4],
    );

    // PnL по причине закрытия. Считаем **средний** PnL на сделку для
    // быстрой эвристики: «один TP компенсирует 5 SL?». Делим только на
    // непустые причины, чтобы не печатать nan.
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

pub(crate) fn print_sim_stats(tag: &str, sim_stats: &SimStats, account: &Account) {
    let total_trades = sim_stats.total_trades();
    if total_trades == 0 {
        tee_println!(
            "[sim] {tag}: нет сделок ({} событий, kelly_skips={} same_asset_open_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={})",
            sim_stats.events,
            sim_stats.total_kelly_skips(),
            sim_stats.total_same_asset_open_skips(),
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
    let roi_pct = (account.bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100.0;

    let total_losses = sim_stats.total_losses();
    // Симметрично с веткой `trades == 0` выше: `kelly_*_skips` нужны и
    // на «успешном» прогоне, чтобы видеть воронку (сколько сигналов
    // было, сколько отвалилось по Kelly, сколько по тонкому стакану).
    // Раньше эти поля печатались только при «нет сделок», и сравнить
    // одинаковую метрику между удачным и неудачным прогоном не получалось.
    tee_println!(
        "[sim] {tag} \
         | events={} trades={} win={:.1}% \
         | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
         | wins={total_wins} losses={total_losses} \
         | kelly_skips={ks} same_asset_open_skips={sas} kelly_strict_buy_skips={ksb} kelly_strict_sell_skips={kss}",
        sim_stats.events, total_trades, win_rate, total_pnl, avg_pnl, total_fees,
        ks = sim_stats.total_kelly_skips(),
        sas = sim_stats.total_same_asset_open_skips(),
        ksb = sim_stats.total_kelly_strict_buy_skips(),
        kss = sim_stats.total_kelly_strict_sell_skips(),
    );
    tee_println!(
        "[sim]   bankroll: {:.2}$ (start={INITIAL_BANKROLL}$) ROI={:+.2}% max_drawdown={:.2}%",
        account.bankroll, roi_pct, account.max_drawdown_pct,
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
