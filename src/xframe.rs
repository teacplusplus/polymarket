use derivative::Derivative;
use anyhow::bail;
use crate::market_snapshot::MarketSnapshot;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::{BTreeMap, HashMap, HashSet};
use xframe_features::FeatureLen;
use xframe_features_derive::XFeatures;
pub use crate::constants::{
    CurrencyUpDownDelayClass, CurrencyUpDownOutcome, TradeSide, XFrameIntervalKind,
};
use crate::gamma_question::currency_up_down_five_min_slot_from_gamma_question;

pub const SIZE: usize = 13;

const MIN_POSITIVE_ASK: f64 = 1e-12;
/// Окно секундных цен спота для μ и σ в z-score (≈ 60 мин). Ключи `prices_by_sec` — Unix-секунды.
const CURRENCY_PRICE_ZSCORE_WINDOW_SEC: i64 = SIZE as i64;
/// Минимум точек в окне для выборочного СКО.
const CURRENCY_PRICE_ZSCORE_MIN_POINTS: usize = 2;

/// Кадр признаков по одному ассету: состояние стакана и сделок на момент снапшота плюс лаги по последним `N` предыдущим кадрам (от ближайшего по времени к более ранним). `tick_size`, `spread` и поля `book_*` приходят из WS; глубина — топ-3 уровня bid/ask из снимка `book` (L1 — объёмы на лучших ценах, L2/L3 — цена и объём).
///
/// Поля с атрибутом `#[xfeature]` попадают в вектор для обучения; `market_id`, `asset_id`, `bucket_flow_sign`, `stable` — без `#[xfeature]` (идентификаторы и служебные поля).
///
/// `xframe_interval_type`: дискриминант [`XFrameIntervalKind`] ([XFRAME_INTERVAL_TYPE_15M] / [XFRAME_INTERVAL_TYPE_5M]). `currency_up_down_outcome`: дискриминант [`CurrencyUpDownOutcome`] ([XFRAME_BTC_OUTCOME_UP] / [XFRAME_BTC_OUTCOME_DOWN]).
/// Поля `other_*` — микроструктура противоположной ноги на тот же бакет; подмешиваются через [XFrame::merge_other_leg_features_from] в `ProjectManager` после вставки пары кадров.
///
/// Поля `sibling_*` — кадр токена **того же** исхода Up/Down на **парном** `market_id` (другой горизонт 5m↔15m), тот же `aligned_ts`; подмешиваются через [XFrame::merge_sibling_market_features_from]. Без валидной пары в [crate::project_manager::ProjectManager::currency_updown_sibling_state] (см. [crate::currency_updown_sibling::CurrencyUpDownSiblingState::paired_five_and_fifteen_market_ids]) остаются значения по умолчанию.
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
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub xframe_interval_type: i32,
    /// Исход токена по Gamma (`outcomes` + `clobTokenIds`): [CurrencyUpDownOutcome] → [XFRAME_BTC_OUTCOME_UP] / [XFRAME_BTC_OUTCOME_DOWN].
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub currency_up_down_outcome: i32,
    /// Дискриминант [`CurrencyUpDownDelayClass`] по Gamma `question` (см. [crate::gamma_question::currency_up_down_five_min_slot_from_gamma_question]): [XFRAME_BTC_5M_DELAY_CLASS_DELAY_5MIN] / [XFRAME_BTC_5M_DELAY_CLASS_DELAY_10MIN]. Для 15m — [XFRAME_BTC_15M_DELAY_CLASS_ALIGNED].
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub currency_up_down_delay_class: i32,
    /// Сколько миллисекунд осталось до конца события рынка: `event_end_ms - timestamp_ms` снапшота; при `event_end_ms <= timestamp` — `0`. Если конец события неизвестен — `0`.
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
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l1_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l1_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l1_price: [Option<f64>; N],
    #[xfeature]
    pub other_tick_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_tick_size: [Option<f64>; N],
    #[xfeature]
    pub other_spread: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_spread: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l1_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l1_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l1_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l1_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l2_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l2_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l2_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l2_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l3_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l3_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_bid_l3_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l3_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l2_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l2_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l2_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l2_size: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l3_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l3_price: [Option<f64>; N],
    #[xfeature]
    pub other_book_ask_l3_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l3_size: [Option<f64>; N],
    #[xfeature]
    pub other_last_trade_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_last_trade_price: [Option<f64>; N],
    #[xfeature]
    #[derivative(Default(value = "0.0"))]
    pub other_trade_size: f64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_trade_size: [Option<f64>; N],
    #[xfeature]
    #[derivative(Default(value = "0.0"))]
    pub other_trade_volume_bucket: f64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_trade_volume_bucket: [Option<f64>; N],
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub other_buy_count_window: u64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_buy_count_window: [Option<i64>; N],
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub other_sell_count_window: u64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_sell_count_window: [Option<i64>; N],
    #[xfeature]
    pub other_burstiness_transactions_count: Option<f64>,
    /// Z-score цены спота: `(p - mu) / sigma`; история — `ProjectManager::rtds_currency_prices_by_sec` (ключ Unix-секунда); `p` — последняя точка в окне, `mu`/`sigma` — по всем ценам окна.
    #[xfeature]
    pub currency_price_z_score: Option<f64>,
    /// Относительное отклонение спота от Gamma «price to beat»: `(price_to_beat - spot) / price_to_beat * 100` (%); спот — последняя секундная цена из `rtds_currency_prices_by_sec`.
    #[xfeature]
    pub currency_price_vs_beat_pct: Option<f64>,
    // --- Парный маркет (5m ↔ 15m), тот же Up или Down; микроструктура токена с `sibling` `market_id`. ---
    /// Остаток времени до конца события на парном маркете, мс (логика как у `event_remaining_ms`).
    #[xfeature]
    #[derivative(Default(value = "-1"))]
    pub sibling_event_remaining_ms: i64,
    #[xfeature]
    pub sibling_book_bid_l1_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l1_price: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_ask_l1_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l1_price: [Option<f64>; N],
    #[xfeature]
    pub sibling_tick_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_tick_size: [Option<f64>; N],
    #[xfeature]
    pub sibling_spread: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_spread: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_bid_l1_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l1_size: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_ask_l1_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l1_size: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_bid_l2_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l2_price: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_bid_l2_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l2_size: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_bid_l3_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l3_price: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_bid_l3_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_bid_l3_size: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_ask_l2_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l2_price: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_ask_l2_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l2_size: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_ask_l3_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l3_price: [Option<f64>; N],
    #[xfeature]
    pub sibling_book_ask_l3_size: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_book_ask_l3_size: [Option<f64>; N],
    #[xfeature]
    pub sibling_last_trade_price: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_last_trade_price: [Option<f64>; N],
    #[xfeature]
    #[derivative(Default(value = "0.0"))]
    pub sibling_trade_size: f64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_trade_size: [Option<f64>; N],
    #[xfeature]
    #[derivative(Default(value = "0.0"))]
    pub sibling_trade_volume_bucket: f64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_trade_volume_bucket: [Option<f64>; N],
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub sibling_buy_count_window: u64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_buy_count_window: [Option<i64>; N],
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub sibling_sell_count_window: u64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub sibling_delta_n_sell_count_window: [Option<i64>; N],
    /// `(price_to_beat_sibling - spot) / price_to_beat_sibling * 100` на парном маркете; тот же спот, другой beat из Gamma.
    #[xfeature]
    pub sibling_currency_price_vs_beat_pct: Option<f64>,
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

        let currency_up_down_delay_class = compute_currency_up_down_delay_class(
            snapshot.xframe_interval_kind,
            gamma_question,
        )
        .as_i32();

        let mut frame = XFrame::<N> {
            market_id: snapshot.market_id,
            asset_id: snapshot.asset_id,
            stable,
            xframe_interval_type: snapshot.xframe_interval_kind.as_i32(),
            currency_up_down_outcome: snapshot.currency_up_down_outcome.as_i32(),
            currency_up_down_delay_class,
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
        self.sibling_currency_price_vs_beat_pct = sibling.currency_price_vs_beat_pct;
    }

    fn populate_window_metrics(
        &mut self,
        frames: &BTreeMap<i64, XFrame<N>>,
        wall_ts_ms: i64,
        window_ms: i64,
    ) {
        let window_start = wall_ts_ms.saturating_sub(window_ms.max(0));
        let mut buy_count_window = if self.bucket_flow_sign > 0 {
            1
        } else {
            0
        };
        let mut sell_count_window = if self.bucket_flow_sign < 0 {
            1
        } else {
            0
        };

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

    fn populate_deltas(&mut self, frames: &BTreeMap<i64, XFrame<N>>) {
        for (lag_index, prior_frame) in frames.values().rev().take(N).enumerate() {
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
            self.delta_n_tick_size[lag_index] = match (self.tick_size, prior_frame.tick_size) {
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
            self.delta_n_last_trade_price[lag_index] = match (
                self.last_trade_price,
                prior_frame.last_trade_price,
            ) {
                (Some(current_price), Some(prior_price)) => Some(current_price - prior_price),
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