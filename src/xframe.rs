pub use crate::constants::{
    CurrencyUpDownDelayClass, CurrencyUpDownOutcome, TradeSide, XFrameIntervalKind,
};
use crate::gamma_question::currency_up_down_five_min_slot_from_gamma_question;
pub use crate::history_sim::POLYMARKET_CRYPTO_TAKER_FEE_RATE;
use crate::market_snapshot::MarketSnapshot;
use anyhow::bail;
use derivative::Derivative;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::{BTreeMap, HashMap, HashSet};
use xframe_features::FeatureLen;
use xframe_features_derive::XFeatures;

pub const SIZE: usize = 15;

const MIN_POSITIVE_ASK: f64 = 1e-12;

/// Один уровень стакана: цена в probability-шкале `[0..1]` и размер в шерсах.
///
/// Лежит в [`XFrame::book_bids`] / [`XFrame::book_asks`] (в порядке от лучшего
/// к худшему) и в [`StrictBook`]. Тип общий, поэтому
/// `XFrame`-side фолбэки в [`crate::history_sim::book_fill_buy`] /
/// [`crate::history_sim::book_fill_sell`] и strict-исполнение по HTTP-снимку
/// CLOB обходят одну и ту же лестницу.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BookLevel {
    /// Цена уровня (probability, `0..1`).
    pub price: f64,
    /// Размер уровня в шерсах.
    pub size: f64,
}

/// HTTP/WS стакан для strict fill в `real_sim`: `None` в history (WS [`crate::history_sim::book_fill_*`]).
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

/// Окно секундных цен спота для μ и σ в z-score (≈ 60 мин). Ключи `prices_by_sec` — Unix-секунды.
const CURRENCY_PRICE_ZSCORE_WINDOW_SEC: i64 = SIZE as i64;
/// Минимум точек в окне для выборочного СКО.
const CURRENCY_PRICE_ZSCORE_MIN_POINTS: usize = 2;

/// Кадр признаков по одному ассету: состояние стакана и сделок на момент снапшота плюс лаги по последним `N` предыдущим кадрам (от ближайшего по времени к более ранним). `tick_size`, `spread` и поля `book_*` приходят из WS; глубина — топ-3 уровня bid/ask из снимка `book` (L1 — объёмы на лучших ценах, L2/L3 — цена и объём).
///
/// Поля с атрибутом `#[xfeature]` попадают в вектор для обучения; `market_id`, `asset_id`, `bucket_flow_sign`, `stable` — без `#[xfeature]` (идентификаторы и служебные поля).
///
/// `xframe_interval_type`: дискриминант [`XFrameIntervalKind`] ([XFRAME_INTERVAL_TYPE_15M] / [XFRAME_INTERVAL_TYPE_5M]). `currency_up_down_outcome`: дискриминант [`CurrencyUpDownOutcome`] ([`CurrencyUpDownOutcome::Up`] / [`CurrencyUpDownOutcome::Down`] как `i32`).
/// `currency_implied_prob` — как отображаемая на Polymarket вероятность исхода **этого** токена: mid L1 при спреде ≤ 10¢, иначе last trade (см. `currency_implied_prob_polymarket_style`).
/// Поля `other_*` — микроструктура противоположной ноги на тот же бакет; подмешиваются через [XFrame::merge_other_leg_features_from] в `ProjectManager` после вставки пары кадров. Без `#[xfeature]` — в вектор XGBoost не входят (только дамп/merge).
///
/// Поля `sibling_*` — кадр токена **того же** исхода Up/Down на **парном** `market_id` (другой горизонт 5m↔15m); момент снапшота тот же, бакет — по сетке интервала sibling-лейна ([`crate::project_manager::FRAME_BUILD_INTERVALS_SEC`]); подмешиваются через [XFrame::merge_sibling_market_features_from]. Без валидной пары в [crate::project_manager::ProjectManager::currency_updown_sibling_state] (см. [crate::currency_updown_sibling::CurrencyUpDownSiblingState::paired_five_and_fifteen_market_ids]) остаются значения по умолчанию.
#[serde_as]
#[derive(Debug, Serialize, Deserialize, Derivative, Clone, XFeatures)]
#[derivative(Default)]
pub struct XFrame<const N: usize> {
    /// Идентификатор условия рынка (`condition_id`), как в поле `market` WS.
    pub market_id: String,
    /// Идентификатор токена в CLOB (`asset_id` / token id).
    pub asset_id: String,
    /// `true`, если WS у начала интервала по Gamma `start_ms`, либо кадр не раньше чем через [`SIZE`] с после `ws_connect`; см. [`compute_xframe_stable`].
    #[derivative(Default(value = "false"))]
    #[serde(default)]
    pub stable: bool,
    /// Тип окна up/down по валюте: `0` — 15 мин ([XFRAME_INTERVAL_TYPE_15M]), `1` — 5 мин ([XFRAME_INTERVAL_TYPE_5M]).
    // #[xfeature]
    #[derivative(Default(value = "0"))]
    pub xframe_interval_type: i32,
    /// Исход токена по Gamma (`outcomes` + `clobTokenIds`): [CurrencyUpDownOutcome] → [XFRAME_BTC_OUTCOME_UP] / [XFRAME_BTC_OUTCOME_DOWN].
    //#[xfeature]
    #[derivative(Default(value = "0"))]
    pub currency_up_down_outcome: i32,
    /// Дискриминант [`CurrencyUpDownDelayClass`] по Gamma `question` (см. [crate::gamma_question::currency_up_down_five_min_slot_from_gamma_question]): [XFRAME_BTC_5M_DELAY_CLASS_DELAY_5MIN] / [XFRAME_BTC_5M_DELAY_CLASS_DELAY_10MIN]. Для 15m — [XFRAME_BTC_15M_DELAY_CLASS_ALIGNED].
    //#[xfeature]
    #[derivative(Default(value = "0"))]
    pub currency_up_down_delay_class: i32,
    /// Рыночная оценка вероятности исхода для **этого** токена (см. [`Self::currency_up_down_outcome`]): mid L1 при спреде ≤ 10¢, при широком спреде — last trade, как в UI Polymarket.
    #[xfeature]
    pub currency_implied_prob: Option<f64>,
    /// Сколько миллисекунд осталось до конца события рынка: `event_end_ms - timestamp_ms` снапшота; при `event_end_ms <= timestamp` — `0`. Если конец события неизвестен — `0`. Для live-кадров после [`crate::market_snapshot::aggregate_events`] `timestamp_ms` — момент последнего события в бакете (не начало интервала).
    #[xfeature]
    #[derivative(Default(value = "-1"))]
    pub event_remaining_ms: i64,
    /// Лучшая цена bid на конец интервала / бакета снапшота.
    #[xfeature]
    pub book_bid_l1_price: Option<f64>,
    /// Разность текущего `book_bid_l1_price` и значения у `i`-го предыдущего кадра: индекс `0` — непосредственный предшественник по времени, далее глубже в прошлое; `None`, если такого кадра ещё не было.
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l1_price: [Option<f64>; N],
    /// Лучший ask: цена L1 на конец интервала.
    #[xfeature]
    pub book_ask_l1_price: Option<f64>,
    /// Разность текущего `book_ask_l1_price` и значения у `i`-го предыдущего кадра (индексация как у `delta_n_book_bid_l1_price`).
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l1_price: [Option<f64>; N],
    /// Минимальный шаг цены из WS (`tick_size` / `new_tick_size`); прокидывается с предыдущего кадра.
    #[xfeature]
    pub tick_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_tick_size: [Option<f64>; N],
    /// Спред из WS, если есть в сообщении.
    #[xfeature]
    pub spread: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_spread: [Option<f64>; N],
    /// Объём на лучшем bid (из снимка `book`).
    #[xfeature]
    pub book_bid_l1_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l1_size: [Option<f64>; N],
    #[xfeature]
    pub book_ask_l1_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l1_size: [Option<f64>; N],
    #[xfeature]
    pub book_bid_l2_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l2_price: [Option<f64>; N],
    #[xfeature]
    pub book_bid_l2_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l2_size: [Option<f64>; N],
    #[xfeature]
    pub book_bid_l3_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l3_price: [Option<f64>; N],
    #[xfeature]
    pub book_bid_l3_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l3_size: [Option<f64>; N],
    #[xfeature]
    pub book_ask_l2_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l2_price: [Option<f64>; N],
    #[xfeature]
    pub book_ask_l2_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l2_size: [Option<f64>; N],
    #[xfeature]
    pub book_ask_l3_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l3_price: [Option<f64>; N],
    #[xfeature]
    pub book_ask_l3_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l3_size: [Option<f64>; N],
    /// Полный bid-стакан кадра (от лучшего к худшему). Источник истины для
    /// исполнения **buy/sell** в [`crate::history_sim::book_fill_buy`] и
    /// [`crate::history_sim::book_fill_sell`]; парные L1/L2/L3 поля выше
    /// сохраняются как фичи модели (XGBoost обучается на них) и из них же
    /// заполняется этот вектор в [`XFrame::new`]. Без атрибута `#[xfeature]` —
    /// в обучающий вектор не идёт.
    ///
    /// `None` — стакан недоступен (например, пустой кадр-плейсхолдер или
    /// дамп до миграции, в котором уровней не было); `Some(vec)` — известный
    /// стакан, возможно пустой, если все L1/L2/L3 пришли невалидными.
    #[serde(default)]
    pub book_bids: Option<Vec<BookLevel>>,
    /// Полный ask-стакан кадра (от лучшего к худшему). Семантика как у
    /// [`Self::book_bids`].
    #[serde(default)]
    pub book_asks: Option<Vec<BookLevel>>,
    /// Цена последней известной сделки на бакете; прокидывается с предыдущего кадра, если в текущем нет обновления.
    #[xfeature]
    pub last_trade_price: Option<f64>,
    /// Разность `last_trade_price` между текущим и `i`-м предыдущим кадром; `None`, если в одном из кадров цены нет.
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_last_trade_price: [Option<f64>; N],
    /// Размер последней сделки в бакете (поле last trade из WS), 0 если сделки не было.
    #[xfeature]
    pub trade_size: f64,
    /// Разность `trade_size` с `i`-м предыдущим кадром.
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_trade_size: [Option<f64>; N],
    /// Накопленный объём сделок в бакете/бакетное поле из WS (агрегат за интервал; если нет — согласован с `trade_size`).
    #[xfeature]
    pub trade_volume_bucket: f64,
    /// Разность `trade_volume_bucket` с `i`-м предыдущим кадром.
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_trade_volume_bucket: [Option<f64>; N],
    /// Знак агрегированной сделки в бакете: `+1` buy, `−1` sell, `0` нет; только для `buy_count_window` / `sell_count_window`, не признак для XGBoost.
    #[derivative(Default(value = "0"))]
    #[serde(default)]
    pub bucket_flow_sign: i8,
    /// Число buy-сделок за скользящее окно длины бакета (мс) по временным меткам кадров, включая текущий бакет при наличии сделки.
    #[xfeature]
    pub buy_count_window: u64,
    /// Разность `buy_count_window` с `i`-м предыдущим кадром (в штуках сделок).
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_buy_count_window: [Option<i64>; N],
    /// Число sell-сделок за то же окно, что и `buy_count_window`.
    #[xfeature]
    pub sell_count_window: u64,
    /// Разность `sell_count_window` с `i`-м предыдущим кадром.
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_sell_count_window: [Option<i64>; N],
    // --- Противоположный токен в том же `market_id` (Up ↔ Down), те же поля, что выше до `currency_price_z_score`. ---
    #[xfeature]
    pub other_book_bid_l1_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l1_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l1_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l1_price: [Option<f64>; N],
    pub other_tick_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_tick_size: [Option<f64>; N],
    pub other_spread: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_spread: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l1_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l1_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l1_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l1_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l2_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l2_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l2_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l2_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l3_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l3_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l3_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l3_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l2_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l2_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l2_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l2_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l3_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l3_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l3_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l3_size: [Option<f64>; N],
    #[xfeature]
    pub other_last_trade_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_last_trade_price: [Option<f64>; N],
    #[derivative(Default(value = "0.0"))]
    pub other_trade_size: f64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_trade_size: [Option<f64>; N],
    #[derivative(Default(value = "0.0"))]
    pub other_trade_volume_bucket: f64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_trade_volume_bucket: [Option<f64>; N],
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub other_buy_count_window: u64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_buy_count_window: [Option<i64>; N],
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub other_sell_count_window: u64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_sell_count_window: [Option<i64>; N],
    pub other_burstiness_transactions_count: Option<f64>,
    /// Как [`Self::currency_implied_prob`], для противоположной ноги (`other` кадр).
    pub other_currency_implied_prob: Option<f64>,
    /// Z-score цены спота: `(p - mu) / sigma`; история — `ProjectManager::rtds_currency_prices_by_sec` (ключ Unix-секунда); `p` — последняя точка в окне, `mu`/`sigma` — по всем ценам окна.
    #[xfeature]
    pub currency_price_z_score: Option<f64>,
    /// Относительное отклонение спота от Gamma «price to beat»: `(price_to_beat - spot) / price_to_beat * 100` (%); спот — последняя секундная цена из `rtds_currency_prices_by_sec`.
    #[xfeature]
    pub currency_price_vs_beat_pct: Option<f64>,
    // --- Парный маркет (5m ↔ 15m), тот же Up или Down; микроструктура токена с `sibling` `market_id`. ---
    /// Остаток времени до конца события на парном маркете, мс (логика как у `event_remaining_ms`).
    #[derivative(Default(value = "-1"))]
    pub sibling_event_remaining_ms: i64,
    pub sibling_book_bid_l1_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l1_price: [Option<f64>; N],
    pub sibling_book_ask_l1_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l1_price: [Option<f64>; N],
    pub sibling_tick_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_tick_size: [Option<f64>; N],
    pub sibling_spread: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_spread: [Option<f64>; N],
    pub sibling_book_bid_l1_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l1_size: [Option<f64>; N],
    pub sibling_book_ask_l1_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l1_size: [Option<f64>; N],
    pub sibling_book_bid_l2_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l2_price: [Option<f64>; N],
    pub sibling_book_bid_l2_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l2_size: [Option<f64>; N],
    pub sibling_book_bid_l3_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l3_price: [Option<f64>; N],
    pub sibling_book_bid_l3_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l3_size: [Option<f64>; N],
    pub sibling_book_ask_l2_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l2_price: [Option<f64>; N],
    pub sibling_book_ask_l2_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l2_size: [Option<f64>; N],
    pub sibling_book_ask_l3_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l3_price: [Option<f64>; N],
    pub sibling_book_ask_l3_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l3_size: [Option<f64>; N],
    pub sibling_last_trade_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_last_trade_price: [Option<f64>; N],
    #[derivative(Default(value = "0.0"))]
    pub sibling_trade_size: f64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_trade_size: [Option<f64>; N],
    #[derivative(Default(value = "0.0"))]
    pub sibling_trade_volume_bucket: f64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_trade_volume_bucket: [Option<f64>; N],
    #[derivative(Default(value = "0"))]
    pub sibling_buy_count_window: u64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_buy_count_window: [Option<i64>; N],
    #[derivative(Default(value = "0"))]
    pub sibling_sell_count_window: u64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_sell_count_window: [Option<i64>; N],
    /// Как [`Self::currency_implied_prob`], для токена парного маркета (`sibling` кадр).
    pub sibling_currency_implied_prob: Option<f64>,
    /// `(price_to_beat_sibling - spot) / price_to_beat_sibling * 100` на парном маркете; тот же спот, другой beat из Gamma.
    pub sibling_currency_price_vs_beat_pct: Option<f64>,
}

/// Хук для side-симметричных коррекций `XFrame` перед сериализацией в feature
/// vector. Вызывается из `to_x_train_with` / `to_x_train_n_with` в `train_mode`
/// и `history_sim` — модификации видны только в векторе признаков, сам `XFrame`
/// в памяти/на диске не меняется (клон перед мутацией).
///
/// ### Сейчас делает
/// Нормализует «время до конца события» на длину интервала **своего** окна,
/// чтобы одна и та же модель одинаково работала на 5m и 15m:
///
/// * [`XFrame::event_remaining_ms`] — нормируется на `interval_ms` текущего
///   маркета (по `xframe_interval_type`).
/// * [`XFrame::sibling_event_remaining_ms`] — токены ходят парами 5m ↔ 15m,
///   поэтому sibling живёт в **противоположном** таймфрейме: нормируем на
///   `kind.sibling().interval_ms()`.
///
/// Шкала — ppm (`remaining_ms * 1_000_000 / interval_ms`), диапазон
/// `0..=1_000_000` (`1_000_000` = самое начало окна). Если дискриминант
/// `xframe_interval_type` невалиден — поля не трогаем.
pub fn apply_side_symmetry<const N: usize>(frame: &mut XFrame<N>) {
    const NORMALIZED_SCALE: i64 = 1_000_000;

    if let Some(kind) = XFrameIntervalKind::from_i32(frame.xframe_interval_type) {
        let self_interval_ms = kind.interval_ms();
        if self_interval_ms > 0 {
            frame.event_remaining_ms =
                frame.event_remaining_ms.saturating_mul(NORMALIZED_SCALE) / self_interval_ms;
        }

        let sibling_interval_ms = kind.sibling().interval_ms();
        if sibling_interval_ms > 0 {
            frame.sibling_event_remaining_ms = frame
                .sibling_event_remaining_ms
                .saturating_mul(NORMALIZED_SCALE)
                / sibling_interval_ms;
        }
    }
}

/// См. [`XFrame::stable`]. `market_id` — `condition_id` рынка (для логов). `event_start_ms` — Gamma `start_ms` в [`crate::project_manager::MarketEventData`]; `ws_connect_wall_ms` — [`crate::project_manager::ProjectManager::record_market_ws_connect_wall_ms`].
pub fn compute_xframe_stable(
    market_id: &str,
    snapshot_timestamp_ms: i64,
    event_start_ms: Option<i64>,
    ws_connect_wall_ms: Option<i64>,
) -> bool {
    use crate::run_log::XFRAME_LOG_ENABLED;

    let ws_connect_wall_ms = match ws_connect_wall_ms {
        Some(ws_connect_wall_ms) => ws_connect_wall_ms,
        None => {
            if XFRAME_LOG_ENABLED {
                eprintln!(
                    "compute_xframe_stable: stable=false — market_id={market_id} — нет ws_connect_wall_ms (record_market_ws_connect_wall_ms не вызывался)",
                );
            }
            return false;
        }
    };

    const JOIN_START_MAX_DELAY_MS: i64 = 2000;

    if let Some(event_start) = event_start_ms {
        let d_gamma = ws_connect_wall_ms - event_start;
        if (-JOIN_START_MAX_DELAY_MS..=JOIN_START_MAX_DELAY_MS).contains(&d_gamma) {
            return true;
        }
        // Prefetch: подписались задолго до старта окна — к event_start уже прошло SIZE секунд буфера.
        if ws_connect_wall_ms + (SIZE as i64) * 1000 <= event_start {
            return true;
        }
    }

    let threshold_ms = ws_connect_wall_ms + (SIZE as i64) * 1000;
    if snapshot_timestamp_ms >= threshold_ms {
        return true;
    }

    if XFRAME_LOG_ENABLED {
        let event_start_part = event_start_ms
            .map(|v| v.to_string())
            .unwrap_or_else(|| "нет".to_string());
        eprintln!(
            "compute_xframe_stable: stable=false — market_id={market_id} — поздний WS: event_start_ms={event_start_part}; ws_connect_wall_ms={ws_connect_wall_ms}; snapshot_ms={snapshot_timestamp_ms} < ws_connect+{SIZE}s={threshold_ms} ms",
        );
    }
    false
}

pub fn compute_currency_up_down_delay_class(
    interval_kind: XFrameIntervalKind,
    gamma_question: Option<&str>,
) -> CurrencyUpDownDelayClass {
    match interval_kind {
        XFrameIntervalKind::FifteenMin => CurrencyUpDownDelayClass::Aligned,
        XFrameIntervalKind::FiveMin => {
            let Some(q) = gamma_question else {
                return CurrencyUpDownDelayClass::Aligned;
            };
            currency_up_down_five_min_slot_from_gamma_question(q)
                .unwrap_or(CurrencyUpDownDelayClass::Aligned)
        }
    }
}

impl<const N: usize> XFrame<N> {
    pub fn new(
        snapshot: MarketSnapshot,
        frames: &BTreeMap<i64, XFrame<N>>,
        event_end_ms: Option<i64>,
        gamma_question: Option<&str>,
        currency_price_z_score: Option<f64>,
        currency_price_vs_beat_pct: Option<f64>,
        window_ms: i64,
        stable: bool,
    ) -> XFrame<N> {
        let previous = frames.values().next_back();

        let wall_ts_ms = snapshot.timestamp_ms;

        let book_bid_l1_price = snapshot
            .book_bid_l1_price
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_bid_l1_price));
        let book_ask_l1_price = snapshot
            .book_ask_l1_price
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_ask_l1_price));
        let tick_size = snapshot
            .tick_size
            .or_else(|| previous.and_then(|prior_frame| prior_frame.tick_size));
        let spread = snapshot
            .spread
            .or_else(|| previous.and_then(|prior_frame| prior_frame.spread));
        let book_bid_l1_size = snapshot
            .book_bid_l1_size
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_bid_l1_size));
        let book_ask_l1_size = snapshot
            .book_ask_l1_size
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_ask_l1_size));
        let book_bid_l2_price = snapshot
            .book_bid_l2_price
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_bid_l2_price));
        let book_bid_l2_size = snapshot
            .book_bid_l2_size
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_bid_l2_size));
        let book_bid_l3_price = snapshot
            .book_bid_l3_price
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_bid_l3_price));
        let book_bid_l3_size = snapshot
            .book_bid_l3_size
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_bid_l3_size));
        let book_ask_l2_price = snapshot
            .book_ask_l2_price
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_ask_l2_price));
        let book_ask_l2_size = snapshot
            .book_ask_l2_size
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_ask_l2_size));
        let book_ask_l3_price = snapshot
            .book_ask_l3_price
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_ask_l3_price));
        let book_ask_l3_size = snapshot
            .book_ask_l3_size
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_ask_l3_size));
        // Полный стакан кадра — берётся целиком из снапшота, как пришёл из
        // WS (`book`-сообщение содержит весь видимый CLOB-снимок). Если в
        // текущем тике лестницы не было (например, `price_change` без
        // глубины), наследуем последнюю известную с предыдущего кадра —
        // та же логика «прокидывания», что и для скалярных `book_*_l*_*`
        // полей выше. Параллельно живут L1/L2/L3 фичи для XGBoost; они
        // выводятся из `book_*_l*` полей снапшота, а не из этого вектора.
        let book_bids = snapshot
            .book_bids
            .clone()
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_bids.clone()));
        let book_asks = snapshot
            .book_asks
            .clone()
            .or_else(|| previous.and_then(|prior_frame| prior_frame.book_asks.clone()));
        let last_trade_price = snapshot
            .last_trade_price
            .or(previous.and_then(|prior_frame| prior_frame.last_trade_price));
        let trade_size = snapshot.last_trade_size.unwrap_or(0.0);
        let trade_volume_bucket = snapshot.trade_volume_bucket.unwrap_or(trade_size.max(0.0));
        let bucket_flow_sign = match snapshot.trade_side {
            Some(TradeSide::Buy) => 1,
            Some(TradeSide::Sell) => -1,
            None => 0,
        };

        let event_remaining_ms = match event_end_ms {
            Some(end_ms) => end_ms.saturating_sub(wall_ts_ms),
            None => 0,
        };

        let currency_up_down_delay_class =
            compute_currency_up_down_delay_class(snapshot.xframe_interval_kind, gamma_question)
                .as_i32();

        let currency_implied_prob = currency_implied_prob_polymarket_style(
            book_bid_l1_price,
            book_ask_l1_price,
            spread,
            last_trade_price,
        );

        let mut frame = XFrame::<N> {
            market_id: snapshot.market_id,
            asset_id: snapshot.asset_id,
            stable,
            xframe_interval_type: snapshot.xframe_interval_kind.as_i32(),
            currency_up_down_outcome: snapshot.currency_up_down_outcome.as_i32(),
            currency_up_down_delay_class,
            currency_implied_prob,
            event_remaining_ms,
            book_bid_l1_price,
            book_ask_l1_price,
            tick_size,
            spread,
            book_bid_l1_size,
            book_ask_l1_size,
            book_bid_l2_price,
            book_bid_l2_size,
            book_bid_l3_price,
            book_bid_l3_size,
            book_ask_l2_price,
            book_ask_l2_size,
            book_ask_l3_price,
            book_ask_l3_size,
            book_bids,
            book_asks,
            currency_price_z_score,
            currency_price_vs_beat_pct,
            last_trade_price,
            trade_size,
            trade_volume_bucket,
            bucket_flow_sign,
            ..Default::default()
        };
        frame.populate_window_metrics(frames, wall_ts_ms, window_ms);
        frame.populate_deltas(frames);
        frame
    }

    /// Подмешивает в `other_*` признаки стакана/сделок с кадра противоположной ноги (`Up`/`Down`) на тот же `aligned_ts` (тот же `market_id`, другой `asset_id`).
    pub fn merge_other_leg_features_from(&mut self, other: &XFrame<N>) {
        self.other_book_bid_l1_price = other.book_bid_l1_price;
        self.other_delta_n_book_bid_l1_price = other.delta_n_book_bid_l1_price;
        self.other_book_ask_l1_price = other.book_ask_l1_price;
        self.other_delta_n_book_ask_l1_price = other.delta_n_book_ask_l1_price;
        self.other_tick_size = other.tick_size;
        self.other_delta_n_tick_size = other.delta_n_tick_size;
        self.other_spread = other.spread;
        self.other_delta_n_spread = other.delta_n_spread;
        self.other_book_bid_l1_size = other.book_bid_l1_size;
        self.other_delta_n_book_bid_l1_size = other.delta_n_book_bid_l1_size;
        self.other_book_ask_l1_size = other.book_ask_l1_size;
        self.other_delta_n_book_ask_l1_size = other.delta_n_book_ask_l1_size;
        self.other_book_bid_l2_price = other.book_bid_l2_price;
        self.other_delta_n_book_bid_l2_price = other.delta_n_book_bid_l2_price;
        self.other_book_bid_l2_size = other.book_bid_l2_size;
        self.other_delta_n_book_bid_l2_size = other.delta_n_book_bid_l2_size;
        self.other_book_bid_l3_price = other.book_bid_l3_price;
        self.other_delta_n_book_bid_l3_price = other.delta_n_book_bid_l3_price;
        self.other_book_bid_l3_size = other.book_bid_l3_size;
        self.other_delta_n_book_bid_l3_size = other.delta_n_book_bid_l3_size;
        self.other_book_ask_l2_price = other.book_ask_l2_price;
        self.other_delta_n_book_ask_l2_price = other.delta_n_book_ask_l2_price;
        self.other_book_ask_l2_size = other.book_ask_l2_size;
        self.other_delta_n_book_ask_l2_size = other.delta_n_book_ask_l2_size;
        self.other_book_ask_l3_price = other.book_ask_l3_price;
        self.other_delta_n_book_ask_l3_price = other.delta_n_book_ask_l3_price;
        self.other_book_ask_l3_size = other.book_ask_l3_size;
        self.other_delta_n_book_ask_l3_size = other.delta_n_book_ask_l3_size;
        self.other_last_trade_price = other.last_trade_price;
        self.other_delta_n_last_trade_price = other.delta_n_last_trade_price;
        self.other_trade_size = other.trade_size;
        self.other_delta_n_trade_size = other.delta_n_trade_size;
        self.other_trade_volume_bucket = other.trade_volume_bucket;
        self.other_delta_n_trade_volume_bucket = other.delta_n_trade_volume_bucket;
        self.other_buy_count_window = other.buy_count_window;
        self.other_delta_n_buy_count_window = other.delta_n_buy_count_window;
        self.other_sell_count_window = other.sell_count_window;
        self.other_delta_n_sell_count_window = other.delta_n_sell_count_window;
        self.other_currency_implied_prob = currency_implied_prob_polymarket_style(
            other.book_bid_l1_price,
            other.book_ask_l1_price,
            other.spread,
            other.last_trade_price,
        );
    }

    /// Копирует в `sibling_*` поля «своей» ноги с кадра парного рынка (тот же Up/Down, тот же бакет), см. [find_same_outcome_sibling_asset_id].
    pub fn merge_sibling_market_features_from(&mut self, sibling: &XFrame<N>) {
        self.sibling_event_remaining_ms = sibling.event_remaining_ms;
        self.sibling_book_bid_l1_price = sibling.book_bid_l1_price;
        self.sibling_delta_n_book_bid_l1_price = sibling.delta_n_book_bid_l1_price;
        self.sibling_book_ask_l1_price = sibling.book_ask_l1_price;
        self.sibling_delta_n_book_ask_l1_price = sibling.delta_n_book_ask_l1_price;
        self.sibling_tick_size = sibling.tick_size;
        self.sibling_delta_n_tick_size = sibling.delta_n_tick_size;
        self.sibling_spread = sibling.spread;
        self.sibling_delta_n_spread = sibling.delta_n_spread;
        self.sibling_book_bid_l1_size = sibling.book_bid_l1_size;
        self.sibling_delta_n_book_bid_l1_size = sibling.delta_n_book_bid_l1_size;
        self.sibling_book_ask_l1_size = sibling.book_ask_l1_size;
        self.sibling_delta_n_book_ask_l1_size = sibling.delta_n_book_ask_l1_size;
        self.sibling_book_bid_l2_price = sibling.book_bid_l2_price;
        self.sibling_delta_n_book_bid_l2_price = sibling.delta_n_book_bid_l2_price;
        self.sibling_book_bid_l2_size = sibling.book_bid_l2_size;
        self.sibling_delta_n_book_bid_l2_size = sibling.delta_n_book_bid_l2_size;
        self.sibling_book_bid_l3_price = sibling.book_bid_l3_price;
        self.sibling_delta_n_book_bid_l3_price = sibling.delta_n_book_bid_l3_price;
        self.sibling_book_bid_l3_size = sibling.book_bid_l3_size;
        self.sibling_delta_n_book_bid_l3_size = sibling.delta_n_book_bid_l3_size;
        self.sibling_book_ask_l2_price = sibling.book_ask_l2_price;
        self.sibling_delta_n_book_ask_l2_price = sibling.delta_n_book_ask_l2_price;
        self.sibling_book_ask_l2_size = sibling.book_ask_l2_size;
        self.sibling_delta_n_book_ask_l2_size = sibling.delta_n_book_ask_l2_size;
        self.sibling_book_ask_l3_price = sibling.book_ask_l3_price;
        self.sibling_delta_n_book_ask_l3_price = sibling.delta_n_book_ask_l3_price;
        self.sibling_book_ask_l3_size = sibling.book_ask_l3_size;
        self.sibling_delta_n_book_ask_l3_size = sibling.delta_n_book_ask_l3_size;
        self.sibling_last_trade_price = sibling.last_trade_price;
        self.sibling_delta_n_last_trade_price = sibling.delta_n_last_trade_price;
        self.sibling_trade_size = sibling.trade_size;
        self.sibling_delta_n_trade_size = sibling.delta_n_trade_size;
        self.sibling_trade_volume_bucket = sibling.trade_volume_bucket;
        self.sibling_delta_n_trade_volume_bucket = sibling.delta_n_trade_volume_bucket;
        self.sibling_buy_count_window = sibling.buy_count_window;
        self.sibling_delta_n_buy_count_window = sibling.delta_n_buy_count_window;
        self.sibling_sell_count_window = sibling.sell_count_window;
        self.sibling_delta_n_sell_count_window = sibling.delta_n_sell_count_window;
        self.sibling_currency_implied_prob = currency_implied_prob_polymarket_style(
            sibling.book_bid_l1_price,
            sibling.book_ask_l1_price,
            sibling.spread,
            sibling.last_trade_price,
        );
        self.sibling_currency_price_vs_beat_pct = sibling.currency_price_vs_beat_pct;
    }

    fn populate_window_metrics(
        &mut self,
        frames: &BTreeMap<i64, XFrame<N>>,
        wall_ts_ms: i64,
        window_ms: i64,
    ) {
        let window_start = wall_ts_ms.saturating_sub(window_ms.max(0));
        let mut buy_count_window = if self.bucket_flow_sign > 0 { 1 } else { 0 };
        let mut sell_count_window = if self.bucket_flow_sign < 0 { 1 } else { 0 };

        for (&_aligned_timestamp_ms, prior_xframe) in frames.range(window_start..=wall_ts_ms) {
            if Self::frame_has_trade(prior_xframe) {
                if prior_xframe.bucket_flow_sign > 0 {
                    buy_count_window += 1;
                } else if prior_xframe.bucket_flow_sign < 0 {
                    sell_count_window += 1;
                }
            }
        }

        self.buy_count_window = buy_count_window;
        self.sell_count_window = sell_count_window;
    }

    fn frame_has_trade(frame: &XFrame<N>) -> bool {
        frame.trade_size > 0.0 || frame.trade_volume_bucket > 0.0
    }

    fn apply_book_field_deltas(&mut self, lag_index: usize, prior_frame: &XFrame<N>) {
        self.delta_n_book_bid_l1_price[lag_index] =
            match (self.book_bid_l1_price, prior_frame.book_bid_l1_price) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_ask_l1_price[lag_index] =
            match (self.book_ask_l1_price, prior_frame.book_ask_l1_price) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_spread[lag_index] = match (self.spread, prior_frame.spread) {
            (Some(current), Some(prior)) => Some(current - prior),
            _ => None,
        };
        self.delta_n_book_bid_l1_size[lag_index] =
            match (self.book_bid_l1_size, prior_frame.book_bid_l1_size) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_ask_l1_size[lag_index] =
            match (self.book_ask_l1_size, prior_frame.book_ask_l1_size) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_bid_l2_price[lag_index] =
            match (self.book_bid_l2_price, prior_frame.book_bid_l2_price) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_bid_l2_size[lag_index] =
            match (self.book_bid_l2_size, prior_frame.book_bid_l2_size) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_bid_l3_price[lag_index] =
            match (self.book_bid_l3_price, prior_frame.book_bid_l3_price) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_bid_l3_size[lag_index] =
            match (self.book_bid_l3_size, prior_frame.book_bid_l3_size) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_ask_l2_price[lag_index] =
            match (self.book_ask_l2_price, prior_frame.book_ask_l2_price) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_ask_l2_size[lag_index] =
            match (self.book_ask_l2_size, prior_frame.book_ask_l2_size) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_ask_l3_price[lag_index] =
            match (self.book_ask_l3_price, prior_frame.book_ask_l3_price) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_book_ask_l3_size[lag_index] =
            match (self.book_ask_l3_size, prior_frame.book_ask_l3_size) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
        self.delta_n_last_trade_price[lag_index] =
            match (self.last_trade_price, prior_frame.last_trade_price) {
                (Some(current_price), Some(prior_price)) => Some(current_price - prior_price),
                _ => None,
            };
    }

    /// Подменяет ордербук strict book и пересчитывает связанные `delta_n_*` по prior кадрам
    /// (как в [`Self::populate_deltas`], индекс `0` — ближайший предшественник).
    pub fn apply_strict_book(&mut self, book: &StrictBook, prior_frames: &[XFrame<N>]) {
        fn level_at(levels: &[BookLevel], idx: usize) -> (Option<f64>, Option<f64>) {
            levels
                .get(idx)
                .map(|l| (Some(l.price), Some(l.size)))
                .unwrap_or((None, None))
        }

        let (book_bid_l1_price, book_bid_l1_size) = level_at(&book.bids, 0);
        let (book_bid_l2_price, book_bid_l2_size) = level_at(&book.bids, 1);
        let (book_bid_l3_price, book_bid_l3_size) = level_at(&book.bids, 2);
        let (book_ask_l1_price, book_ask_l1_size) = level_at(&book.asks, 0);
        let (book_ask_l2_price, book_ask_l2_size) = level_at(&book.asks, 1);
        let (book_ask_l3_price, book_ask_l3_size) = level_at(&book.asks, 2);

        self.book_bid_l1_price = book_bid_l1_price;
        self.book_bid_l1_size = book_bid_l1_size;
        self.book_bid_l2_price = book_bid_l2_price;
        self.book_bid_l2_size = book_bid_l2_size;
        self.book_bid_l3_price = book_bid_l3_price;
        self.book_bid_l3_size = book_bid_l3_size;
        self.book_ask_l1_price = book_ask_l1_price;
        self.book_ask_l1_size = book_ask_l1_size;
        self.book_ask_l2_price = book_ask_l2_price;
        self.book_ask_l2_size = book_ask_l2_size;
        self.book_ask_l3_price = book_ask_l3_price;
        self.book_ask_l3_size = book_ask_l3_size;
        self.book_bids = Some(book.bids.clone());
        self.book_asks = Some(book.asks.clone());

        if let Some(ltp) = book.last_trade_price {
            self.last_trade_price = Some(ltp);
        }

        let spread = match (book_bid_l1_price, book_ask_l1_price) {
            (Some(b), Some(a)) if b.is_finite() && a.is_finite() => Some((a - b).max(0.0)),
            _ => None,
        };
        self.spread = spread;

        if let Some(prob) = currency_implied_prob_polymarket_style(
            book_bid_l1_price,
            book_ask_l1_price,
            spread,
            book.last_trade_price.or(self.last_trade_price),
        ) {
            self.currency_implied_prob = Some(prob.clamp(0.001, 0.999));
        }

        for (lag_index, prior_frame) in prior_frames.iter().take(N).enumerate() {
            self.apply_book_field_deltas(lag_index, prior_frame);
        }
    }

    fn populate_deltas(&mut self, frames: &BTreeMap<i64, XFrame<N>>) {
        for (lag_index, prior_frame) in frames.values().rev().take(N).enumerate() {
            self.apply_book_field_deltas(lag_index, prior_frame);
            self.delta_n_tick_size[lag_index] = match (self.tick_size, prior_frame.tick_size) {
                (Some(current), Some(prior)) => Some(current - prior),
                _ => None,
            };
            self.delta_n_trade_size[lag_index] = Some(self.trade_size - prior_frame.trade_size);
            self.delta_n_trade_volume_bucket[lag_index] =
                Some(self.trade_volume_bucket - prior_frame.trade_volume_bucket);
            self.delta_n_buy_count_window[lag_index] =
                Some(self.buy_count_window as i64 - prior_frame.buy_count_window as i64);
            self.delta_n_sell_count_window[lag_index] =
                Some(self.sell_count_window as i64 - prior_frame.sell_count_window as i64);
        }
    }
}

/// Вторая нога BTC up/down: другой `asset_id` с противоположным кодом в `currency_up_down_by_asset_id` среди кандидатов (батч + уже сохранённые кадры по тому же `market_id`).
pub fn find_opposite_asset_id(
    asset_id: &str,
    currency_up_down_by_asset_id: &HashMap<String, CurrencyUpDownOutcome>,
    candidate_asset_ids: &HashSet<String>,
) -> anyhow::Result<String> {
    let Some(&my_outcome) = currency_up_down_by_asset_id.get(asset_id) else {
        bail!(
            "неизвестный код currency up/down для asset_id={asset_id} в currency_up_down_by_asset_id"
        );
    };
    let other_outcome = my_outcome.opposite();
    for candidate_id in candidate_asset_ids {
        if candidate_id == asset_id {
            continue;
        }
        if currency_up_down_by_asset_id.get(candidate_id).copied() == Some(other_outcome) {
            return Ok(candidate_id.clone());
        }
    }
    bail!(
        "неизвестный кандидат currency up/down ({my_outcome:?}) для asset_id={asset_id} в currency_up_down_by_asset_id"
    );
}

/// Токен на **парном** `market_id` с тем же [`CurrencyUpDownOutcome`], что у `asset_id` (Up–Up или Down–Down).
/// `sibling_candidate_asset_ids` — множество `asset_id` только с sibling-маркета (тот же батч/тик).
pub fn find_same_outcome_sibling_asset_id(
    asset_id: &str,
    market_id: &str,
    currency_up_down_by_asset_id: &HashMap<String, CurrencyUpDownOutcome>,
    candidate_asset_ids: &HashSet<String>,
) -> anyhow::Result<String> {
    let Some(&my_outcome) = currency_up_down_by_asset_id.get(asset_id) else {
        bail!(
            "same_outcome_sibling: неизвестный currency up/down для asset_id={asset_id} (sibling market_id={market_id})"
        );
    };
    for candidate_id in candidate_asset_ids {
        if currency_up_down_by_asset_id.get(candidate_id).copied() == Some(my_outcome) {
            return Ok(candidate_id.clone());
        }
    }
    bail!(
        "same_outcome_sibling: нет токена с исходом {my_outcome:?} среди кандидатов sibling market_id={market_id}"
    );
}

/// z = (p - mu) / sigma: только секундный ряд `prices_by_sec` (ключ — Unix-секунда); `p` — цена последней точки в окне (самый поздний ключ ≤ `reference_sec`), `mu` и `sigma` — по всем ценам в том же окне.
pub fn currency_price_z_score_from_sec_history(
    prices_by_sec: &BTreeMap<i64, f64>,
    reference_sec: i64,
) -> Option<f64> {
    let window_start_sec = reference_sec.saturating_sub(CURRENCY_PRICE_ZSCORE_WINDOW_SEC);
    let window: Vec<f64> = prices_by_sec
        .range(window_start_sec..=reference_sec)
        .map(|(_, price)| *price)
        .collect();
    let n = window.len();
    if n < CURRENCY_PRICE_ZSCORE_MIN_POINTS {
        return None;
    }
    let current_price = *window.last()?;
    let n_f = n as f64;
    let mu = window.iter().sum::<f64>() / n_f;
    let sum_sq_dev: f64 = window
        .iter()
        .map(|price| {
            let d = price - mu;
            d * d
        })
        .sum();
    let sigma = (sum_sq_dev / (n_f - 1.0)).sqrt();
    if !sigma.is_finite() || sigma <= MIN_POSITIVE_ASK {
        return None;
    }
    Some((current_price - mu) / sigma)
}

/// Минимальная чистая доходность (после комиссий) для метки y=1 (Take Profit).
/// Например, 0.05 означает: вложенный 1 USDC должен принести ≥ 1.05 USDC нетто.
pub const Y_TRAIN_TAKE_PROFIT_PP: f64 = 0.05;
/// Максимальная чистая доходность для метки y=0 (Stop Loss) PnL-модели.
/// Используется [`calc_y_train_pnl`] и runtime PnL-каналом
/// ([`crate::history_sim::sell_gate`] вне hold-zone).
pub const Y_TRAIN_PNL_STOP_LOSS_PP: f64 = -0.03;
/// Максимальная чистая доходность для метки y=0 (Stop Loss) Resolution-модели.
/// Используется [`calc_y_train_resolution`] и runtime Resolution-каналом
/// ([`crate::history_sim::sell_gate`] для `opened_in_hold_zone`).
/// Численно совпадает с [`Y_TRAIN_PNL_STOP_LOSS_PP`], но семантически независим:
/// пороги двух моделей не обязаны быть равны.
pub const Y_TRAIN_RESOLUTION_STOP_LOSS_PP: f64 = -0.03;
/// Минимальная **относительная** просадка urgent sell VWAP относительно **фактического**
/// sell VWAP на входе (полный bid-walk в [`walk_sell_xfeatures`]). Та же доля, что
/// в runtime SL от ref с cap ([`crate::history_sim::sell_gate`]); в y-разметке ref без cap.
pub const Y_TRAIN_SL_MIN_REF_SELL_REL_DROP: f64 = 0.03;
/// Горизонт [`calc_y_train_pnl`] / [`calc_y_train_resolution`]: сколько
/// следующих кадров смотреть на этапе y-разметки.
///
/// Та же численная величина дублируется в
/// [`crate::history_sim::POSITION_TIMEOUT_FRAMES`] — это **намеренно**:
/// runtime-симулятор ограничивает время удержания позиции тем же
/// горизонтом, на котором модель размечена. За пределами горизонта
/// предсказание модели некалибровано, поэтому позиция принудительно
/// закрывается по рынку (`CloseReason::Timeout`).
///
/// Если эта связка нарушится (значения разойдутся), runtime будет
/// торговать на тиках, которые модель не видела при обучении, — это
/// прямой источник bias'а между симуляцией и live-режимом.
pub const Y_TRAIN_HORIZON_FRAMES: usize = 30;

/// Нижняя граница **исключённого** центра распределения `currency_implied_prob`
/// для [`calc_y_train_pnl`]: модель учится **только на хвостах** —
/// `p_buy ≤ Y_TRAIN_NO_TRADE_PROB_LOW` или `p_buy ≥ Y_TRAIN_NO_TRADE_PROB_HIGH`.
/// Кадры с `p_buy ∈ (LOW, HIGH)` (центр) возвращают `None` и в train/val
/// не попадают.
///
/// Семантика: «торгуем только направленные рынки». В центре `[0.3..0.7]`
/// рынок balanced — обе стороны примерно равновероятны, edge модели
/// размазан, шум резолюции преобладает. На хвостах же доминирующая
/// сторона уже выявлена ценой (`p_buy ≥ 0.7` ⇒ market priced YES win),
/// и y-метка более чисто отражает «оправдалась ли market expectation».
///
/// Runtime-фильтр в [`crate::history_sim::buy_gate`] синхронизирован
/// с этой константой: при `entry_prob ∈ (LOW, HIGH)` возвращается
/// `BuyGate::EntryProbOutOfRange` и инкрементируется `entry_prob_skips`,
/// поэтому inference происходит только на хвостах — на том же
/// распределении, на котором училась модель.
pub const Y_TRAIN_NO_TRADE_PROB_LOW: f64 = 0.5;

/// Верхняя граница исключённого центра для y-разметки.
/// См. [`Y_TRAIN_NO_TRADE_PROB_LOW`] для обоснования.
pub const Y_TRAIN_NO_TRADE_PROB_HIGH: f64 = 0.5;

/// Верхний потолок цены входа **только для Resolution-канала**
/// (held-to-resolution): при `entry_prob ≥` этого значения у позиции нет
/// извлекаемого edge — апсайд `≈ 1/p − 1` (доли процента при `p → 1`), а
/// даунсайд при проигрыше токена = вся ставка. Чтобы окупить такой payoff,
/// нужен realized win-rate ≥ `p`, но в зоне 0.99+ даже мелкая переуверенность
/// калибровки на 1–2 п.п. переворачивает реальный EV в минус (см. разбор в
/// истории: realized 0.91–0.98 при цене 0.99+). Поэтому кадры с
/// `entry_prob ≥ MAX` исключаются из Resolution-разметки ([`calc_y_train_resolution`]
/// → `None`) и из runtime-входа ([`crate::history_sim::buy_gate`] →
/// `EntryProbOutOfRange`). На PnL-канал не распространяется.
pub const Y_TRAIN_RESOLUTION_MAX_ENTRY_PROB: f64 = 0.90;
pub const Y_TRAIN_PNL_MAX_ENTRY_PROB: f64 = 0.90;

/// Целевой нотионал позиции (gross USDC), под который размечаются Y-метки
/// [`calc_y_train_pnl`] / [`calc_y_train_resolution`]. Совпадает с типичным
/// размером, которым реально торгует `real_sim` (Kelly-fraction × bankroll
/// в коридоре `MIN_POSITION_USD..MAX_POSITION_USD`); $200 — медиана этого
/// коридора. Размечать Y под фиксированный нотионал, а не «1 nominal USDC
/// по mid», нужно, чтобы разметка отражала **реально достижимый** PnL: на
/// тонком маркете $200 могут не пройти L1+L2+L3 ask и съесть VWAP за пределами
/// порога из [`crate::train_mode::Y_TRAIN_MAX_SLIPPAGE_FROM_L1_PCT`] (передаётся
/// в [`calc_y_train_pnl`] / [`calc_y_train_resolution`]) — такие кадры отфильтровываются.
pub const Y_TRAIN_NOMINAL_USDC: f64 = 30.0;

/// Результат walk'а через L1/L2/L3 ask из xframe для покупки `target_usdc`.
///
/// Заполняется [`walk_buy_xfeatures`]; используется только внутри Y-разметки
/// ([`calc_y_train_pnl`] / [`calc_y_train_resolution`]). Полная семантика
/// «купили `target_usdc` USDC gross, fee списывается в шерсах»
/// **поверх** gross — точно как в `try_open_position` и
/// [`crate::history_sim::book_fill_buy_strict`] (gross тратится по уровням,
/// fee per-level считается отдельно и вычитается из получаемых шерсов).
struct WalkBuyResult {
    /// Шеров, реально полученных в кошелёк (gross_shares − fee_shares).
    actual_shares: f64,
    /// VWAP на gross-шерсах: `target_usdc / gross_shares`. Для slippage-чека.
    vwap: f64,
    /// Лучший ask из L1 (best_ask), для сравнения с VWAP в slippage-чеке.
    best_ask: f64,
}

/// Результат walk'а через L1/L2/L3 bid из xframe для продажи `shares`.
///
/// Заполняется [`walk_sell_xfeatures`]; используется только внутри Y-разметки.
/// Семантика fee «в шерсах» симметричная: на каждом уровне списывается
/// `level_shares × fee × p × (1 − p)` USDC, итог вычитается из выручки.
struct WalkSellResult {
    /// Брутто-выручка: `Σ level_shares × level_price` без вычета fee.
    /// Эквивалент того, что получил бы **maker**, чья resting-лимитка
    /// была бы хитнута таker-ом: maker fee = 0, поэтому брутто = нетто.
    /// Используется в TP-детекции / max-progress y-разметки и в
    /// [`crate::account_close_position::close_position`] для voluntary-exit-ов (TP / EvExitProfit).
    gross_usdc: f64,
    /// Чистая выручка **taker-варианта**: `gross_usdc − fee_usdc_taker`,
    /// где `fee_usdc_taker = Σ level_shares × rate × p × (1 − p)`
    /// (per-level, не на VWAP'е). Эквивалент urgent-выхода через
    /// bid-walk: пересекаем спред, реально платим taker-fee. Используется
    /// в SL-детекции / drawdown-проверке y-разметки и в принудительных
    /// закрытиях [`crate::account_close_position::close_position`] (SL / Timeout / EvExitLoss).
    net_usdc: f64,
    /// VWAP на проданных шерсах: `gross_usdc / shares`. Для slippage-чека.
    #[allow(dead_code)]
    vwap: f64,
    /// Лучший bid из L1 (best_bid); сохраняется для отладки / будущих cap-чеков.
    #[allow(dead_code)]
    best_bid: f64,
}

fn collect_walk_ask_levels<const N: usize>(frame: &XFrame<N>) -> Vec<(f64, f64)> {
    if let Some(book) = &frame.book_asks {
        book.iter()
            .filter_map(|l| (l.price > 0.0 && l.size > 0.0).then_some((l.price, l.size)))
            .collect()
    } else {
        [
            (frame.book_ask_l1_price, frame.book_ask_l1_size),
            (frame.book_ask_l2_price, frame.book_ask_l2_size),
            (frame.book_ask_l3_price, frame.book_ask_l3_size),
        ]
        .into_iter()
        .filter_map(|(p, s)| match (p, s) {
            (Some(price), Some(size)) if price > 0.0 && size > 0.0 => Some((price, size)),
            _ => None,
        })
        .collect()
    }
}

/// Walk по ask из `frame`, тратя `target_usdc` gross.
///
/// Источник уровней — [`XFrame::book_asks`], если оно `Some` (как
/// [`crate::history_sim::book_fill_buy`]: полный стакан от лучшего к
/// худшему). Если `book_asks == None` (старый дамп / плейсхолдер без
/// книги), фолбэк на `book_ask_l{1,2,3}_{price,size}`.
///
/// Возвращает `None`, если:
/// 1. нет ни одного валидного (`price > 0 && size > 0`) уровня;
/// 2. суммарной глубины не хватило, чтобы потратить весь `target_usdc`
///    (`remaining_usdc > 1e-9` после прохода).
///
/// Slippage cap здесь **не** применяется — caller (Y-функция) проверяет
/// его сам относительно нужного для конкретного reason cap'а.
///
/// Fee в результат **не** включён: caller получает `actual_shares` уже
/// после fee. Это удобно, потому что в Y-метке `entry_cost` нормирован
/// на `target_usdc` (= `Y_TRAIN_NOMINAL_USDC`), а fee «съедает шеры», как
/// в реальном Polymarket-исполнении.
fn walk_buy_xfeatures<const N: usize>(
    frame: &XFrame<N>,
    target_usdc: f64,
) -> Option<WalkBuyResult> {
    if !target_usdc.is_finite() || target_usdc <= 0.0 {
        return None;
    }
    let levels = collect_walk_ask_levels(frame);
    let best_ask = levels.first().map(|(p, _)| *p)?;

    let mut remaining_usdc = target_usdc;
    let mut gross_shares = 0.0_f64;
    let mut fee_usdc_total = 0.0_f64;
    for &(price, size) in &levels {
        let usdc_at_level = (size * price).min(remaining_usdc);
        let shares_at_level = usdc_at_level / price;
        gross_shares += shares_at_level;
        // Polymarket fee per share = rate × p × (1 − p); честно считаем
        // **по уровню**, а не на VWAP'е — на разнопрайсной книге это
        // даёт другую сумму fee.
        fee_usdc_total +=
            shares_at_level * POLYMARKET_CRYPTO_TAKER_FEE_RATE * price * (1.0 - price);
        remaining_usdc -= usdc_at_level;
        if remaining_usdc <= 1e-9 {
            break;
        }
    }
    if remaining_usdc > 1e-9 || gross_shares <= 0.0 {
        return None;
    }
    let vwap = target_usdc / gross_shares;
    // Fee «в шерсах»: эквивалент того, как `try_open_position` в
    // history_sim считает `fee_buy_shares = fee_buy_usdc / vwap`. На
    // разнопрайсной книге `fee_usdc_total` — точная сумма, а делим на
    // VWAP, чтобы пересчитать в шеры (одинаковая шкала с `gross_shares`).
    let fee_shares = fee_usdc_total / vwap;
    let actual_shares = gross_shares - fee_shares;
    if !actual_shares.is_finite() || actual_shares <= 0.0 {
        return None;
    }
    Some(WalkBuyResult {
        actual_shares,
        vwap,
        best_ask,
    })
}

fn collect_walk_bid_levels<const N: usize>(frame: &XFrame<N>) -> Vec<(f64, f64)> {
    if let Some(book) = &frame.book_bids {
        book.iter()
            .filter_map(|l| (l.price > 0.0 && l.size > 0.0).then_some((l.price, l.size)))
            .collect()
    } else {
        [
            (frame.book_bid_l1_price, frame.book_bid_l1_size),
            (frame.book_bid_l2_price, frame.book_bid_l2_size),
            (frame.book_bid_l3_price, frame.book_bid_l3_size),
        ]
        .into_iter()
        .filter_map(|(p, s)| match (p, s) {
            (Some(price), Some(size)) if price > 0.0 && size > 0.0 => Some((price, size)),
            _ => None,
        })
        .collect()
    }
}

/// Walk по bid из `frame`, продавая `shares` штук.
///
/// Симметричен [`walk_buy_xfeatures`]: если [`XFrame::book_bids`] — `Some`,
/// полный bid-стакан; иначе фолбэк на `book_bid_l{1,2,3}_{price,size}`.
/// Fee per-level через `level_shares × rate × p × (1 − p)`;
/// возвращает `None`, если глубины не хватило на весь `shares`.
fn walk_sell_xfeatures<const N: usize>(frame: &XFrame<N>, shares: f64) -> Option<WalkSellResult> {
    if !shares.is_finite() || shares <= 0.0 {
        return None;
    }
    let levels = collect_walk_bid_levels(frame);
    let best_bid = levels.first().map(|(p, _)| *p)?;

    let mut remaining = shares;
    let mut gross_usdc = 0.0_f64;
    let mut fee_usdc_total = 0.0_f64;
    for &(price, size) in &levels {
        let take = remaining.min(size);
        gross_usdc += take * price;
        fee_usdc_total += take * POLYMARKET_CRYPTO_TAKER_FEE_RATE * price * (1.0 - price);
        remaining -= take;
        if remaining <= 1e-9 {
            break;
        }
    }
    if remaining > 1e-9 || gross_usdc <= 0.0 {
        return None;
    }
    let vwap = gross_usdc / shares;
    let net_usdc = gross_usdc - fee_usdc_total;
    if !net_usdc.is_finite() || !gross_usdc.is_finite() {
        return None;
    }
    Some(WalkSellResult {
        gross_usdc,
        net_usdc,
        vwap,
        best_bid,
    })
}

/// Метка y для PnL-модели — `«успеет ли позиция $200 нотиналом отбить TP до
/// конца горизонта или попадёт в SL»`.
///
/// # Модель торговли (как в `real_sim` / `history_sim`)
///
/// 1. **Вход**: купить ровно [`Y_TRAIN_NOMINAL_USDC`] gross-USDC walk'ом
///    ([`walk_buy_xfeatures`]): при `Some` — [`XFrame::book_asks`], иначе
///    фолбэк на `book_ask_l{1,2,3}` (старые дампы без полного стакана).
///    Fee per-level по точной формуле Polymarket
///    `level_shares × rate × p × (1 − p)`. Полученные шеры:
///    `actual_shares = gross_shares − fee_usdc / vwap`.
///
///    Кадр размечается как `None`, если:
///    - в выбранных уровнях ask нет валидного уровня, либо суммарной глубины не
///      хватило, чтобы потратить $200 (тонкий маркет);
///    - VWAP покупки уехал от best ask больше, чем на переданный
///      `max_slippage_from_l1_pct` (реальный исполнитель такой ордер
///      бы зарубил → семантически это «вход не открылся»).
///
/// 2. **Шаги горизонта** (1..=n): на каждом будущем кадре —
///    `walk_sell_xfeatures(actual_shares)`: при [`XFrame::book_bids`] `Some` —
///    полный bid, иначе L1–L3. Если глубины bid не хватает на
///    `actual_shares` — выйти нельзя ни в каком виде, ни TP, ни SL не
///    считаются на этом тике, идём дальше.
///
///    `net_ret = (net_usdc − Y_TRAIN_NOMINAL_USDC) / Y_TRAIN_NOMINAL_USDC`
///    нормирован относительно gross-нотионала, поэтому пороги
///    `Y_TRAIN_TAKE_PROFIT_PP` / `Y_TRAIN_PNL_STOP_LOSS_PP` (как % от
///    нотионала) переиспользуются 1:1.
///
///    * **SL** (mandatory exit, без cap'а): если `net_ret ≤ SL_PP` **и**
///      urgent sell VWAP просел относительно **фактического** sell VWAP на входе
///      (полный bid-walk) не меньше чем на [`Y_TRAIN_SL_MIN_REF_SELL_REL_DROP`] →
///      `Some(0.0)` (симметрично runtime [`crate::history_sim::sell_gate`]). Иначе SL на этом
///      тике не считается. На SL-выходе реальный исполнитель не применяет
///      cap к bid-walk, иначе позиция могла бы доехать до $0 при тонком стакане.
///    * **TP** (voluntary exit, **с** cap'ом): если slippage от best bid
///      ≤ `max_slippage_from_l1_pct` **и** `net_ret ≥ TP_PP` →
///      `Some(1.0)`. Если cap не соблюдён — TP «не сработал на этом
///      тике» (как `manage_positions` в реальности — ждём следующего).
///
/// 3. **Резолюция** (`event_remaining_ms ≤ 0` или нет следующего кадра):
///    выплата `actual_shares × payout_per_share − Y_TRAIN_NOMINAL_USDC`,
///    где `payout = 1.0` при победе токена (без fee, как в CLOB) и
///    `0.0` при проигрыше. Нормировка та же.
///
/// `up_won` определяется по [`MarketXFramesDump::up_won`]:
/// `final_price >= price_to_beat`. Какой токен победил — определяет
/// [`y_train_resolution_token_won`] по `currency_up_down_outcome`
/// текущего кадра (он константен по маркету).
///
/// # Возврат
///
/// * `Some(1.0)` — позиция $200 успела бы добежать до TP (с реальным
///   исполнителем) или выиграть резолюцию.
/// * `Some(0.0)` — позиция упала бы в SL раньше, проиграла резолюцию,
///   либо не смогла открыться/выйти из-за тонкого стакана
///   (`walk_buy_xfeatures`/`walk_sell_xfeatures` вернули `None`,
///   slippage-cap на входе) — это всё «реальный исполнитель не дал бы
///   позиции жить», поэтому метим как полный лосс.
/// * `Some(0.5 + value)`, где `value ∈ [0, 0.5]` — за весь горизонт `n`
///   `net_ret` **ни разу не уходил в минус** (никакой просадки, даже
///   мелкой, не до SL), и при этом ни SL, ни TP не сработали. `value`
///   линейно интерполирует максимально достигнутый `net_ret` между
///   `0` (стояли около нуля → лейбл `0.5`) и `Y_TRAIN_TAKE_PROFIT_PP`
///   (почти дотянули до TP → лейбл `1.0`). Если хоть на одном тике
///   `net_ret < 0` — лейбл `Some(0.0)` (любая просадка убивает
///   мягкий исход). XGBoost при `binary:logistic` принимает мягкие
///   цели в `[0,1]` (это становится cross-entropy с soft target) и
///   учится предсказывать вероятность «исход — спокойное восходящее
///   движение».
/// * `None` — кадр не размечается: не определена `currency_implied_prob`
///   (ранее использовалась как fallback при отсутствии стакана; больше
///   не нужно, оставлено как формальный инвариант).
pub fn calc_y_train_pnl(
    n: usize,
    x_frames: &[XFrame<SIZE>],
    index: usize,
    price_to_beat: f64,
    final_price: f64,
    max_slippage_from_l1_pct: f64,
) -> Option<f32> {
    let up_won = final_price >= price_to_beat;
    let current = x_frames.get(index)?;
    // currency_implied_prob больше не участвует в расчёте PnL, но его
    // отсутствие — индикатор «кадр без книги», такие кадры мы и без того
    // отвалим в `walk_buy_xfeatures`. Оставлено намеренно как ранний
    // skip, чтобы поведение совпадало с прежним «нет цены — нет y».
    let p_buy = current.currency_implied_prob?;

    // Потолок цены входа: в зоне `entry_prob ≥ MAX` нет извлекаемого edge
    // (апсайд → 0 при `p → 1`, даунсайд = вся ставка). Исключаем такие кадры
    // из обучения, чтобы модель не училась «угадывать» уже заложенный в цену
    // near-certain исход. Синхронизировано с runtime-гейтом в
    // [`crate::history_sim::buy_gate`]. См. [`Y_TRAIN_PNL_MAX_ENTRY_PROB`].
    if p_buy >= Y_TRAIN_PNL_MAX_ENTRY_PROB {
        return Some(0.0);
    }

    // Фильтр «только хвосты»: учим модель на направленных рынках,
    // где доминирующая сторона уже выявлена ценой
    // (`p_buy ≤ NO_TRADE_LOW` или `p_buy ≥ NO_TRADE_HIGH`).
    // В центре `(LOW..HIGH)` рынок balanced, шум резолюции преобладает
    // над edge'ем модели — кадры центра возвращают `None` и в train/val
    // не попадают. См. doc у `Y_TRAIN_NO_TRADE_PROB_LOW`.
    if p_buy > Y_TRAIN_NO_TRADE_PROB_LOW && p_buy < Y_TRAIN_NO_TRADE_PROB_HIGH {
        return Some(0.0);
    }

    let buy = match walk_buy_xfeatures(current, Y_TRAIN_NOMINAL_USDC) {
        None => return Some(0.0),
        Some(buy) => buy,
    };
    // Slippage cap на входе: реальный `book_fill_buy_strict` зарубил бы
    // такой ордер. Семантически это «позиция не открылась» → сэмпл не
    // учим (а не размечаем как лосс — лосс это уже про *открытую*
    // позицию, которая выбила SL).
    if (buy.vwap - buy.best_ask) / buy.best_ask > max_slippage_from_l1_pct {
        return Some(0.0);
    }
    let actual_shares = buy.actual_shares;
    // Фактический sell VWAP на входе — полный bid-walk ([`walk_sell_xfeatures`]), без cap к L1.
    // let sell_vwap_at_entry = match walk_sell_xfeatures(current, actual_shares) {
    //     None => return Some(0.0),
    //     Some(s) => s.vwap,
    // };
    // Maker-проверка решается **в момент входа**: сразу после buy-walk-а
    // мы выставляем resting-лимитку на TP-таргет, и в этот же момент
    // фиксируется её maker-/taker-статус. Если `tp_target > best_bid_at_entry`
    // — лимитка ложится в книгу как resting ask (maker, fee = 0). Если
    // `tp_target ≤ best_bid_at_entry` — лимитка пересекает спред при
    // постановке (taker, fee применяется). Состояние стакана в будущих
    // кадрах (когда лимитку реально хитнут) уже не меняет maker-статус
    // ордера. Без best_bid в `current` (одностронний стакан без bid'ов) —
    // нашему ask не на что было бы кросснуться, лимитка резинг-ит как maker.
    let tp_target = Y_TRAIN_NOMINAL_USDC * (1.0 + Y_TRAIN_TAKE_PROFIT_PP) / actual_shares;
    let tp_is_maker = match current.book_bid_l1_price {
        Some(best_bid_at_entry) => tp_target > best_bid_at_entry,
        None => true,
    };

    for i in 1..=n {
        // Отсутствие следующего кадра трактуется как конец маркета
        // (дампы обрезаются по реальному завершению события).
        let future_opt = x_frames.get(index + i);
        let next_future_opt = x_frames.get(index + i + 1);
        let reached_end = match future_opt {
            None => true,
            Some(f) => f.event_remaining_ms <= 0 && next_future_opt.is_none(),
        };

        if reached_end {
            // Резолюция: победитель получает $1/шер без fee.
            // `currency_up_down_outcome` константен на протяжении маркета.
            let won = y_train_resolution_token_won(current, up_won);
            let payout = if won { actual_shares } else { 0.0 };
            return Some(if (payout - Y_TRAIN_NOMINAL_USDC) > 0.0 {
                1.0
            } else {
                0.0
            });
        }

        let future = future_opt.expect("reached_end == false implies future_opt.is_some()");
        // Книга на этом тике может не пропускать `actual_shares` — тогда
        // выйти нельзя ни добровольно, ни принудительно; держим до
        // следующего кадра. Это **не** SL (SL — это про цену, а не про
        // ликвидность); в реальности `manage_positions` тоже бы
        // продолжил держать (`book_fill_sell_strict` вернул бы None при нехватке
        // глубины bid, но не из-за min_order_size).
        let sell = match walk_sell_xfeatures(future, actual_shares) {
            Some(s) => s,
            None => return Some(0.0),
        };
        // SL / drawdown — это «urgent-выход через bid-walk», т.е.
        // **taker-сценарий**: мы пересекаем спред, реально платим fee.
        // TP / max-progress — это «resting-лимитка над текущим mid»,
        // т.е. **maker-сценарий**: fee = 0, в карман падает gross_usdc.
        // См. `CloseReason::is_voluntary_exit` в `history_sim.rs` для
        // той же дихотомии в runtime-симуляторе.нщ
        let net_ret_taker = (sell.net_usdc - Y_TRAIN_NOMINAL_USDC) / Y_TRAIN_NOMINAL_USDC;
        let net_ret_maker = (sell.gross_usdc - Y_TRAIN_NOMINAL_USDC) / Y_TRAIN_NOMINAL_USDC;

        // SL — mandatory exit, slippage cap отключён, taker-fee; дополнительно —
        // urgent [`WalkSellResult::vwap`] должен просесть относительно входного ref
        // не меньше чем на [`Y_TRAIN_SL_MIN_REF_SELL_REL_DROP`] (симметрично runtime SL).
        if net_ret_taker <= Y_TRAIN_PNL_STOP_LOSS_PP
        // && y_train_stop_loss_sell_deteriorated_vs_entry_ref(sell_vwap_at_entry, sell.vwap)
        {
            return Some(0.0);
        }
        // TP — voluntary exit. Maker-/taker-статус ордера зафиксирован
        // в момент входа (`tp_is_maker`, см. выше): мы выставляем
        // resting-лимитку сразу после buy-walk-а, и её роль решается
        // тогда же. В будущих кадрах смотрим только, дотянул ли
        // gross-/net-ret до порога TP_PP.
        //
        // Slippage cap включён в обеих ветках: симметрично с
        // [`crate::history_sim::SellGate`], где voluntary-exit по
        // слишком тонкому стакану ждёт следующего тика.

        if tp_is_maker {
            if net_ret_maker >= Y_TRAIN_TAKE_PROFIT_PP {
                return Some(1.0);
            }
        } else if net_ret_taker >= Y_TRAIN_TAKE_PROFIT_PP {
            return Some(1.0);
        }
    }
    Some(0.0)
}

// pub fn calc_y_train_pnl(n: usize, x_frames: &[XFrame<SIZE>], index: usize, price_to_beat: f64, final_price: f64, _: f64) -> Option<f32> {
//     let up_won = final_price >= price_to_beat;
//     let current = x_frames.get(index)?;
//     let p_buy = current.currency_implied_prob?.clamp(0.001, 0.999);
//
//     // Размечаем PnL «$1 USDC gross» — после деления получаем долю в [-1.0, +∞),
//     // так что `Y_TRAIN_TAKE_PROFIT_PP` / `Y_TRAIN_STOP_LOSS_PP` — это сразу
//     // проценты доходности от потраченного.
//     let entry_cost_usdc: f64 = 1.0;
//
//     let nominal_shares = entry_cost_usdc / p_buy;
//     let fee_buy_usdc   = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * p_buy * (1.0 - p_buy);
//     let fee_buy_shares = fee_buy_usdc / p_buy;
//     let actual_shares  = nominal_shares - fee_buy_shares;
//
//     for i in 1..=n {
//         // Отсутствие следующего кадра трактуется как конец маркета
//         // (дампы обрезаются по реальному завершению события).
//         let future_opt = x_frames.get(index + i);
//         let reached_end = match future_opt {
//             None => true,
//             Some(f) => f.event_remaining_ms <= 0,
//         };
//
//         // Считаем сколько USDC реально получим на выходе на этом тике
//         // (или на резолюции). `net_ret` — доля изменения относительно
//         // потраченного на входе: `(received - entry) / entry`. Сравнение с
//         // `Y_TRAIN_TAKE_PROFIT_PP` / `Y_TRAIN_STOP_LOSS_PP` — в тех же
//         // долях.
//         let received_usdc = if reached_end {
//             // Резолюция: победитель получает $1/шер без fee
//             // (Polymarket CTF redemption бескомиссионный).
//             // `currency_up_down_outcome` константен на протяжении маркета,
//             // поэтому для определения токена берём текущий кадр.
//             let won = y_train_resolution_token_won(current, up_won);
//             if won { actual_shares * 1.0 } else { 0.0 }
//         } else {
//             let future = future_opt.expect("reached_end == false implies future_opt.is_some()");
//             let p_sell = future.currency_implied_prob?.clamp(0.001, 0.999);
//             let gross_usdc    = actual_shares * p_sell;
//             let fee_sell_usdc = actual_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * p_sell * (1.0 - p_sell);
//             gross_usdc - fee_sell_usdc
//         };
//         let net_ret = (received_usdc - entry_cost_usdc) / entry_cost_usdc;
//
//         if net_ret >= Y_TRAIN_TAKE_PROFIT_PP {
//             return Some(1.0);
//         } else if net_ret <= Y_TRAIN_STOP_LOSS_PP {
//             return Some(0.0);
//         }
//
//         // Кадры закончились — дальше смотреть некуда, ни TP/SL не сработал.
//         if future_opt.is_none() {
//             break;
//         }
//     }
//    Some(0.0)
// }

// pub fn calc_y_train_pnl(n: usize, x_frames: &[XFrame<SIZE>], index: usize, price_to_beat: f64, final_price: f64, _: f64) -> Option<f32> {
//     let up_won = final_price >= price_to_beat;
//     let current = x_frames.get(index)?;
//     let p_buy = current.currency_implied_prob?.clamp(0.001, 0.999);

//     // Фильтр «только хвосты»: учим модель на направленных рынках,
//     // где доминирующая сторона уже выявлена ценой
//     // (`p_buy ≤ NO_TRADE_LOW` или `p_buy ≥ NO_TRADE_HIGH`).
//     // В центре `(LOW..HIGH)` рынок balanced, шум резолюции преобладает
//     // над edge'ем модели — кадры центра возвращают `None` и в train/val
//     // не попадают. См. doc у `Y_TRAIN_NO_TRADE_PROB_LOW`.
//     if p_buy > Y_TRAIN_NO_TRADE_PROB_LOW && p_buy < Y_TRAIN_NO_TRADE_PROB_HIGH {
//         return Some(0.0);
//     }

//     // Размечаем PnL «$1 USDC gross» — после деления получаем долю в [-1.0, +∞),
//     // так что `Y_TRAIN_TAKE_PROFIT_PP` / `Y_TRAIN_STOP_LOSS_PP` — это сразу
//     // проценты доходности от потраченного.
//     let entry_cost_usdc: f64 = 1.0;

//     let actual_shares = entry_cost_usdc / p_buy;

//     for i in 1..=n {
//         // Отсутствие следующего кадра трактуется как конец маркета
//         // (дампы обрезаются по реальному завершению события).
//         let future_opt = x_frames.get(index + i);
//         let reached_end = match future_opt {
//             None => true,
//             Some(f) => f.event_remaining_ms <= 0,
//         };

//         // Считаем сколько USDC реально получим на выходе на этом тике
//         // (или на резолюции). `net_ret` — доля изменения относительно
//         // потраченного на входе: `(received - entry) / entry`. Сравнение с
//         // `Y_TRAIN_TAKE_PROFIT_PP` / `Y_TRAIN_STOP_LOSS_PP` — в тех же
//         // долях.
//         let received_usdc = if reached_end {
//             // Резолюция: победитель получает $1/шер без fee
//             // (Polymarket CTF redemption бескомиссионный).
//             // `currency_up_down_outcome` константен на протяжении маркета,
//             // поэтому для определения токена берём текущий кадр.
//             let won = y_train_resolution_token_won(current, up_won);
//             if won { actual_shares * 1.0 } else { 0.0 }
//         } else {
//             let future = future_opt.expect("reached_end == false implies future_opt.is_some()");
//             let p_sell = future.currency_implied_prob?.clamp(0.001, 0.999);
//             let gross_usdc = actual_shares * p_sell;
//             gross_usdc
//         };
//         let net_ret = (received_usdc - entry_cost_usdc) / entry_cost_usdc;

//         if net_ret >= Y_TRAIN_TAKE_PROFIT_PP {
//             return Some(1.0);
//         } else if net_ret <= Y_TRAIN_STOP_LOSS_PP {
//             return Some(0.0);
//         }

//     }
//     Some(0.0)
// }

/// Победил ли **этот** токен по итогу рынка.
///
/// Победил ли **этот** токен по итогу рынка.
///
/// Для Up-токена `won == up_won`, для Down-токена `won == !up_won`.
fn y_train_resolution_token_won(frame: &XFrame<SIZE>, up_won: bool) -> bool {
    match CurrencyUpDownOutcome::from_i32(frame.currency_up_down_outcome) {
        Some(CurrencyUpDownOutcome::Up) => up_won,
        Some(CurrencyUpDownOutcome::Down) => !up_won,
        None => up_won,
    }
}

/// Метка y для resolution-модели — `«доживёт ли позиция $200 нотиналом
/// до резолюции как победитель, не выбив SL по пути»`.
///
/// Зеркало [`calc_y_train_pnl`] с одной разницей: целевое событие —
/// резолюция (без отдельной TP-проверки). Поэтому slippage cap при
/// продаже здесь не нужен (TP не проверяется), а SL остаётся
/// mandatory-exit без cap'а.
///
/// # Модель торговли
///
/// 1. **Вход**: тот же [`walk_buy_xfeatures`] на $200 нотионала с теми
///    же отказами (`None`): тонкий ask, slippage cap превышен.
/// 2. **Шаги горизонта**: на каждом будущем кадре —
///    `walk_sell_xfeatures(actual_shares)`. Если глубины bid не хватает —
///    выйти нельзя, идём дальше (как в `calc_y_train_pnl`). Если хватает,
///    `net_ret = (net_usdc − Y_TRAIN_NOMINAL_USDC) / Y_TRAIN_NOMINAL_USDC`
///    и проверяется только SL: `net_ret ≤ Y_TRAIN_RESOLUTION_STOP_LOSS_PP` →
///    `Some(0.0)` (исполнитель закрыл бы по mandatory-exit).
/// 3. **Резолюция** (`event_remaining_ms ≤ 0` или нет следующего кадра):
///    `Some(1.0)` если токен выиграл, `Some(0.0)` если проиграл.
///
/// # Возврат
///
/// * `Some(1.0)` — позиция доехала до резолюции победителем без SL,
///   **или** прошла весь горизонт `n` кадров без просадки в минус
///   (зеркало [`calc_y_train_pnl`]: «спокойный восходящий исход»).
/// * `Some(0.0)` — позиция выбила SL, проиграла резолюцию, не смогла
///   открыться/выйти из-за тонкого стакана, **или** прошла весь
///   горизонт без SL/резолюции, но с хотя бы одной просадкой
///   `net_ret < 0`.
/// * `None` — `currency_implied_prob` не определена (формальный
///   инвариант, фактически отсекается раньше через `walk_buy_xfeatures`).
pub fn calc_y_train_resolution(
    n: usize,
    x_frames: &[XFrame<SIZE>],
    index: usize,
    price_to_beat: f64,
    final_price: f64,
    max_slippage_from_l1_pct: f64,
) -> Option<f32> {
    let up_won = final_price >= price_to_beat;
    let current = x_frames.get(index)?;
    let p_buy = current.currency_implied_prob?;

    // Потолок цены входа: в зоне `entry_prob ≥ MAX` нет извлекаемого edge
    // (апсайд → 0 при `p → 1`, даунсайд = вся ставка). Исключаем такие кадры
    // из обучения, чтобы модель не училась «угадывать» уже заложенный в цену
    // near-certain исход. Синхронизировано с runtime-гейтом в
    // [`crate::history_sim::buy_gate`]. См. [`Y_TRAIN_RESOLUTION_MAX_ENTRY_PROB`].
    if p_buy >= Y_TRAIN_RESOLUTION_MAX_ENTRY_PROB {
        return Some(0.0);
    }

    let buy = match walk_buy_xfeatures(current, Y_TRAIN_NOMINAL_USDC) {
        None => return Some(0.0),
        Some(buy) => buy,
    };

    if (buy.vwap - buy.best_ask) / buy.best_ask > max_slippage_from_l1_pct {
        return Some(0.0);
    }

    let actual_shares = buy.actual_shares;
    // Фактический sell VWAP на входе — полный bid-walk ([`walk_sell_xfeatures`]), без cap к L1.
    // let sell_vwap_at_entry = match walk_sell_xfeatures(current, actual_shares) {
    //     None => return Some(0.0),
    //     Some(s) => s.vwap,
    // };

    for i in 1..=n {
        // Отсутствие следующего кадра трактуется как конец маркета
        // (дампы обрезаются по реальному завершению события).
        let future_opt = x_frames.get(index + i);
        let next_future_opt = x_frames.get(index + i + 1);
        let reached_end = match future_opt {
            None => true,
            Some(f) => f.event_remaining_ms <= 0 && next_future_opt.is_none(),
        };

        if reached_end {
            // Резолюция: победитель получает $1/шер без fee.
            // `currency_up_down_outcome` константен на протяжении маркета.
            let won = y_train_resolution_token_won(current, up_won);
            let payout = if won { actual_shares } else { 0.0 };
            return Some(if (payout - Y_TRAIN_NOMINAL_USDC) > 0.0 {
                1.0
            } else {
                0.0
            });
        }

        let future = future_opt.expect("reached_end == false implies future_opt.is_some()");
        // Книга не пропускает `actual_shares` → ни SL, ни выход; держим
        // до следующего кадра.
        let sell = match walk_sell_xfeatures(future, actual_shares) {
            Some(s) => s,
            None => return Some(0.0),
        };
        // SL — urgent-выход через bid-walk: пересекаем спред,
        // применяется taker-fee. Maker-альтернативы у SL нет —
        // невозможно «спокойно ждать» в просадке, в реальной торговле
        // это всегда market-sell. Поэтому используем
        // `sell.net_usdc` (= gross − taker-fee) без maker-ветки.
        // Симметрично с обработкой SL в [`calc_y_train_pnl`].
        let net_ret_taker = (sell.net_usdc - Y_TRAIN_NOMINAL_USDC) / Y_TRAIN_NOMINAL_USDC;
        if net_ret_taker <= Y_TRAIN_RESOLUTION_STOP_LOSS_PP
        // && y_train_stop_loss_sell_deteriorated_vs_entry_ref(sell_vwap_at_entry, sell.vwap)
        {
            return Some(0.0);
        }
    }

    // Горизонт прошёл, ни SL, ни резолюция не сработали. Зеркало
    // концовки [`calc_y_train_pnl`]: любая просадка в минус ⇒
    // `Some(0.0)`, иначе ⇒ `Some(1.0)`.
    Some(1.0)
}

// pub fn calc_y_train_resolution(
//     n: usize,
//     x_frames: &[XFrame<SIZE>],
//     index: usize,
//     price_to_beat: f64,
//     final_price: f64,
//     _: f64
// ) -> Option<f32> {
//     let up_won = final_price >= price_to_beat;
//     let current = x_frames.get(index)?;
//     let p_buy = current.currency_implied_prob?.clamp(0.001, 0.999);
//
//     let nominal_shares = 1.0 / p_buy;
//     let fee_buy_usdc   = nominal_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * p_buy * (1.0 - p_buy);
//     let fee_buy_shares = fee_buy_usdc / p_buy;
//     let actual_shares  = nominal_shares - fee_buy_shares;
//
//     for i in 1..=n {
//         // Отсутствие следующего кадра трактуется как конец маркета
//         // (дампы обрезаются по реальному завершению события).
//         let future_opt = x_frames.get(index + i);
//         let reached_end = match future_opt {
//             None => true,
//             Some(f) => f.event_remaining_ms <= 0,
//         };
//
//         // Резолюция: комиссии нет, победитель получает $1/шер.
//         // `currency_up_down_outcome` константен на протяжении маркета.
//         if reached_end {
//             let won = y_train_resolution_token_won(current, up_won);
//             return Some(if won { 1.0 } else { 0.0 });
//         }
//
//         // Промежуточный кадр — taker-продажа с fee, проверяем досрочный стоп.
//         let future = future_opt.expect("reached_end == false implies future_opt.is_some()");
//         let p_sell = future.currency_implied_prob?.clamp(0.001, 0.999);
//         let gross_usdc    = actual_shares * p_sell;
//         let fee_sell_usdc = actual_shares * POLYMARKET_CRYPTO_TAKER_FEE_RATE * p_sell * (1.0 - p_sell);
//         let net_ret = (gross_usdc - fee_sell_usdc) - 1.0;
//         if net_ret <= Y_TRAIN_STOP_LOSS_PP {
//             return Some(0.0);
//         }
//     }
//
//     Some(1.0)
// }

/// Mid L1: (лучший bid + лучший ask) / 2.
pub(crate) fn book_l1_mid_price(best_bid: Option<f64>, best_ask: Option<f64>) -> Option<f64> {
    match (best_bid, best_ask) {
        (Some(b), Some(a)) if b.is_finite() && a.is_finite() => Some((b + a) * 0.5),
        _ => None,
    }
}

/// Порог как в UI Polymarket: при спреде **> 10¢** показывают last trade, иначе mid.
pub(crate) const POLYMARKET_WIDE_SPREAD_USD: f64 = 0.10;

/// Оценка «отображаемой» цены/вероятности: при узком спреде — mid L1; при широком (`> 10¢`) — [`last_trade_price`], если есть, иначе mid.
///
/// Используется и при сборке `XFrame.currency_implied_prob` из WS-кадра
/// (см. `XFrame::from_market_snapshot`), и в `history_sim::effective_implied_prob`
/// поверх HTTP-стакана ([`StrictBook`]) — один источник
/// истины для «отображаемой» Polymarket-вероятности, чтобы фичи модели и
/// решения о входе/выходе работали в одной шкале.
pub(crate) fn currency_implied_prob_polymarket_style(
    best_bid: Option<f64>,
    best_ask: Option<f64>,
    spread_reported: Option<f64>,
    last_trade_price: Option<f64>,
) -> Option<f64> {
    let mid = book_l1_mid_price(best_bid, best_ask);
    let spread_effective =
        spread_reported
            .filter(|s| s.is_finite())
            .or_else(|| match (best_bid, best_ask) {
                (Some(b), Some(a)) if b.is_finite() && a.is_finite() => Some((a - b).max(0.0)),
                _ => None,
            });

    if spread_effective.map(|s| s > POLYMARKET_WIDE_SPREAD_USD) == Some(true) {
        last_trade_price.filter(|p| p.is_finite()).or(mid)
    } else {
        mid
    }
}

/// Размер bincode-сериализации [`XFrame::<SIZE>::default()`]: сегмент каталога `xframes/{currency}/{байты}/…`, зеркально `graph/…`, `streams/…`.
/// Считается один раз при первом вызове [`xframe_bincode_schema_size_bytes`].
static XFRAME_BINCODE_SCHEMA_SIZE_BYTES: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| {
        bincode::serialized_size(&XFrame::<SIZE>::default())
            .expect("XFrame::<SIZE>::default() must be bincode-serializable") as usize
    });

/// Размер сериализованного по bincode дефолтного [`XFrame::<SIZE>`] — «fingerprint» раскладки полей для подкаталога дампов.
#[inline]
pub fn xframe_bincode_schema_size_bytes() -> usize {
    *XFRAME_BINCODE_SCHEMA_SIZE_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(ts: i64) -> MarketSnapshot {
        MarketSnapshot {
            market_id: "m1".to_string(),
            asset_id: "a1".to_string(),
            xframe_interval_kind: XFrameIntervalKind::FiveMin,
            currency_up_down_outcome: CurrencyUpDownOutcome::Up,
            timestamp_ms: ts,
            book_bid_l1_price: None,
            book_bid_l1_size: None,
            book_ask_l1_price: None,
            book_ask_l1_size: None,
            book_bid_l2_price: None,
            book_bid_l2_size: None,
            book_bid_l3_price: None,
            book_bid_l3_size: None,
            book_ask_l2_price: None,
            book_ask_l2_size: None,
            book_ask_l3_price: None,
            book_ask_l3_size: None,
            book_bids: None,
            book_asks: None,
            tick_size: None,
            min_order_size: None,
            spread: None,
            last_trade_price: None,
            last_trade_size: None,
            trade_volume_bucket: None,
            trade_side: None,
            market_resolved: false,
        }
    }

    #[test]
    fn xframe_new_forward_fills_and_computes_delta() {
        let mut frames = BTreeMap::new();
        let mut prev = XFrame::<2>::default();
        prev.book_bid_l1_price = Some(0.40);
        prev.book_ask_l1_price = Some(0.60);
        prev.spread = Some(0.20);
        prev.tick_size = Some(0.01);
        prev.trade_size = 2.0;
        prev.bucket_flow_sign = 1;
        frames.insert(1_000, prev);

        let mut s = snapshot(1_500);
        s.book_ask_l1_price = Some(0.58);
        s.last_trade_size = Some(3.0);
        s.trade_side = Some(TradeSide::Sell);

        let frame = XFrame::<2>::new(
            s,
            &frames,
            Some(2_000),
            None,
            Some(0.5),
            Some(-1.2),
            1_000,
            true,
        );
        assert_eq!(frame.book_bid_l1_price, Some(0.40));
        assert_eq!(frame.book_ask_l1_price, Some(0.58));
        let d = frame.delta_n_book_ask_l1_price[0].expect("delta should exist");
        assert!((d - (-0.02)).abs() < 1e-9);
        assert_eq!(frame.trade_size, 3.0);
        assert_eq!(frame.bucket_flow_sign, -1);
        assert_eq!(frame.event_remaining_ms, 500);
        assert_eq!(frame.buy_count_window, 1);
        assert_eq!(frame.sell_count_window, 1);
    }

    #[test]
    fn xframe_feature_count_matches_serialization_length() {
        let frame = XFrame::<SIZE>::default();
        let all = frame.to_x_train_with(apply_side_symmetry);
        assert_eq!(all.len(), XFrame::<SIZE>::count_features());

        let n = 7usize;
        let cropped = frame.to_x_train_n_with(n, apply_side_symmetry);
        assert_eq!(cropped.len(), XFrame::<SIZE>::count_features_n(n));
    }

    #[test]
    fn compute_xframe_stable_returns_true_after_warmup() {
        let stable = compute_xframe_stable("m1", 15_000, None, Some(0));
        assert!(stable);
    }
}
