//! Одноразовая миграция дампов [`crate::xframe_dump::MarketXFramesDump`] под
//! новую раскладку [`crate::xframe::XFrame`]: добавление полей
//! [`crate::xframe::XFrame::book_bids`] и [`crate::xframe::XFrame::book_asks`]
//! изменило бинарный формат, и старые `.bin`-файлы под `xframes/{cur}/<size>/`
//! больше не открываются текущим кодом без преобразования.
//!
//! # Стратегия
//!
//! Дампы хранятся в виде `xframes/{currency}/{schema_size}/{interval}/{step}/{date}/{name}.bin`,
//! где `schema_size = bincode::serialized_size(&XFrame::<SIZE>::default())` — стабильный
//! «fingerprint» раскладки структуры (см. [`crate::xframe_dump`]). Добавление двух
//! `Option<Vec<BookLevel>>` (по умолчанию `None` — 1 байт-тег на каждый,
//! без полезной нагрузки) увеличивает `schema_size` на 2 байта, поэтому новые
//! дампы лягут в **другой** под-каталог.
//!
//! Миграция:
//! 1. Считает `schema_size` текущего [`crate::xframe::XFrame`] через bincode.
//! 2. Обходит все `xframes/{currency}/<old_size>/...` каталоги, где
//!    `<old_size> != current_size`.
//! 3. Каждый `.bin` в `<interval>/<step>/<date>/` десериализует как
//!    [`LegacyMarketXFramesDump`] (старая раскладка кадра). При неудачной
//!    десериализации файл пропускается (не наша версия).
//! 4. Каждый кадр конвертируется в [`crate::xframe::XFrame`]: новые векторы
//!    выставляются в `None` — глубина стакана у старых дампов не сохранена,
//!    реконструировать её из L1/L2/L3 без размеров на остальных уровнях нельзя.
//! 5. Кадры пересериализуются **поверх исходного файла**.
//! 6. После успешного обхода каталог переименовывается:
//!    `xframes/{currency}/<old_size>/` → `xframes/{currency}/<current_size>/`.
//!    Это оставляет рядом обученные модели (`*.ubj` / `*.calibration.bin`,
//!    лежат на уровне `<size>/`) под актуальным `schema_size`, чтобы
//!    [`crate::history_sim::run_sim_mode`] и
//!    [`crate::real_sim::latest_version_path`] нашли их без отдельного шага.
//!
//! Команда идемпотентна: после успешного прогона устаревших каталогов
//! `<old_size>` не остаётся, повторный запуск ничего не делает.

use crate::xframe::{BookLevel, XFrame, SIZE};
use crate::xframe_dump::MarketXFramesDump;
use anyhow::{Context, Result};
use derivative::Derivative;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::fs;
use std::path::{Path, PathBuf};

const CURRENCIES: &[&str] = &["btc"];

/// Старая раскладка [`crate::xframe::XFrame`] **до** добавления полей
/// `book_bids`/`book_asks`. Используется только для десериализации
/// исторических `.bin`-дампов внутри миграции — bincode позиционен, поэтому
/// важна 1-в-1 соответствующая последовательность полей и их типы. Атрибуты
/// `#[xfeature]` и `derivative(Default)` намеренно опущены: миграции нужно
/// только `Deserialize`, а исходный кадр через `Default::default()` мы не
/// конструируем.
#[serde_as]
#[derive(Debug, Serialize, Deserialize, Derivative, Clone)]
#[derivative(Default)]
#[allow(dead_code)]
struct LegacyXFrame<const N: usize> {
    pub market_id: String,
    pub asset_id: String,
    #[serde(default)]
    pub stable: bool,
    #[derivative(Default(value = "0"))]
    pub xframe_interval_type: i32,
    #[derivative(Default(value = "0"))]
    pub currency_up_down_outcome: i32,
    #[derivative(Default(value = "0"))]
    pub currency_up_down_delay_class: i32,
    pub currency_implied_prob: Option<f64>,
    #[derivative(Default(value = "-1"))]
    pub event_remaining_ms: i64,
    pub book_bid_l1_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l1_price: [Option<f64>; N],
    pub book_ask_l1_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l1_price: [Option<f64>; N],
    pub tick_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_tick_size: [Option<f64>; N],
    pub spread: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_spread: [Option<f64>; N],
    pub book_bid_l1_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l1_size: [Option<f64>; N],
    pub book_ask_l1_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l1_size: [Option<f64>; N],
    pub book_bid_l2_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l2_price: [Option<f64>; N],
    pub book_bid_l2_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l2_size: [Option<f64>; N],
    pub book_bid_l3_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l3_price: [Option<f64>; N],
    pub book_bid_l3_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_bid_l3_size: [Option<f64>; N],
    pub book_ask_l2_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l2_price: [Option<f64>; N],
    pub book_ask_l2_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l2_size: [Option<f64>; N],
    pub book_ask_l3_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l3_price: [Option<f64>; N],
    pub book_ask_l3_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_book_ask_l3_size: [Option<f64>; N],
    pub last_trade_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_last_trade_price: [Option<f64>; N],
    pub trade_size: f64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_trade_size: [Option<f64>; N],
    pub trade_volume_bucket: f64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_trade_volume_bucket: [Option<f64>; N],
    #[derivative(Default(value = "0"))]
    #[serde(default)]
    pub bucket_flow_sign: i8,
    pub buy_count_window: u64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_buy_count_window: [Option<i64>; N],
    pub sell_count_window: u64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_sell_count_window: [Option<i64>; N],
    pub other_book_bid_l1_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l1_price: [Option<f64>; N],
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
    pub other_book_bid_l1_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l1_size: [Option<f64>; N],
    pub other_book_ask_l1_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l1_size: [Option<f64>; N],
    pub other_book_bid_l2_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l2_price: [Option<f64>; N],
    pub other_book_bid_l2_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l2_size: [Option<f64>; N],
    pub other_book_bid_l3_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l3_price: [Option<f64>; N],
    pub other_book_bid_l3_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_bid_l3_size: [Option<f64>; N],
    pub other_book_ask_l2_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l2_price: [Option<f64>; N],
    pub other_book_ask_l2_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l2_size: [Option<f64>; N],
    pub other_book_ask_l3_price: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l3_price: [Option<f64>; N],
    pub other_book_ask_l3_size: Option<f64>,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_book_ask_l3_size: [Option<f64>; N],
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
    #[derivative(Default(value = "0"))]
    pub other_buy_count_window: u64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_buy_count_window: [Option<i64>; N],
    #[derivative(Default(value = "0"))]
    pub other_sell_count_window: u64,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_sell_count_window: [Option<i64>; N],
    pub other_burstiness_transactions_count: Option<f64>,
    pub other_currency_implied_prob: Option<f64>,
    pub currency_price_z_score: Option<f64>,
    pub currency_price_vs_beat_pct: Option<f64>,
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
    pub sibling_currency_implied_prob: Option<f64>,
    pub sibling_currency_price_vs_beat_pct: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LegacyMarketXFramesDump {
    pub frames_up: Vec<LegacyXFrame<SIZE>>,
    pub frames_down: Vec<LegacyXFrame<SIZE>>,
    #[serde(default)]
    pub price_to_beat: f64,
    #[serde(default)]
    pub final_price: f64,
}

fn legacy_to_current(legacy: LegacyXFrame<SIZE>) -> XFrame<SIZE> {
    // Полные лестницы у легаси-кадров не сохранялись (в WS-событие на момент
    // дампа писались только L1/L2/L3 поля). Восстановить «честный» стакан
    // из верхних трёх уровней нельзя — это была бы фальсификация глубины,
    // поэтому оставляем `None`: потребители (`book_fill_buy/sell`) уже
    // умеют обрабатывать отсутствие лестницы и возвращают 0.0.
    let book_bids: Option<Vec<BookLevel>> = None;
    let book_asks: Option<Vec<BookLevel>> = None;

    XFrame::<SIZE> {
        market_id: legacy.market_id,
        asset_id: legacy.asset_id,
        stable: legacy.stable,
        xframe_interval_type: legacy.xframe_interval_type,
        currency_up_down_outcome: legacy.currency_up_down_outcome,
        currency_up_down_delay_class: legacy.currency_up_down_delay_class,
        currency_implied_prob: legacy.currency_implied_prob,
        event_remaining_ms: legacy.event_remaining_ms,
        book_bid_l1_price: legacy.book_bid_l1_price,
        delta_n_book_bid_l1_price: legacy.delta_n_book_bid_l1_price,
        book_ask_l1_price: legacy.book_ask_l1_price,
        delta_n_book_ask_l1_price: legacy.delta_n_book_ask_l1_price,
        tick_size: legacy.tick_size,
        delta_n_tick_size: legacy.delta_n_tick_size,
        spread: legacy.spread,
        delta_n_spread: legacy.delta_n_spread,
        book_bid_l1_size: legacy.book_bid_l1_size,
        delta_n_book_bid_l1_size: legacy.delta_n_book_bid_l1_size,
        book_ask_l1_size: legacy.book_ask_l1_size,
        delta_n_book_ask_l1_size: legacy.delta_n_book_ask_l1_size,
        book_bid_l2_price: legacy.book_bid_l2_price,
        delta_n_book_bid_l2_price: legacy.delta_n_book_bid_l2_price,
        book_bid_l2_size: legacy.book_bid_l2_size,
        delta_n_book_bid_l2_size: legacy.delta_n_book_bid_l2_size,
        book_bid_l3_price: legacy.book_bid_l3_price,
        delta_n_book_bid_l3_price: legacy.delta_n_book_bid_l3_price,
        book_bid_l3_size: legacy.book_bid_l3_size,
        delta_n_book_bid_l3_size: legacy.delta_n_book_bid_l3_size,
        book_ask_l2_price: legacy.book_ask_l2_price,
        delta_n_book_ask_l2_price: legacy.delta_n_book_ask_l2_price,
        book_ask_l2_size: legacy.book_ask_l2_size,
        delta_n_book_ask_l2_size: legacy.delta_n_book_ask_l2_size,
        book_ask_l3_price: legacy.book_ask_l3_price,
        delta_n_book_ask_l3_price: legacy.delta_n_book_ask_l3_price,
        book_ask_l3_size: legacy.book_ask_l3_size,
        delta_n_book_ask_l3_size: legacy.delta_n_book_ask_l3_size,
        book_bids,
        book_asks,
        last_trade_price: legacy.last_trade_price,
        delta_n_last_trade_price: legacy.delta_n_last_trade_price,
        trade_size: legacy.trade_size,
        delta_n_trade_size: legacy.delta_n_trade_size,
        trade_volume_bucket: legacy.trade_volume_bucket,
        delta_n_trade_volume_bucket: legacy.delta_n_trade_volume_bucket,
        bucket_flow_sign: legacy.bucket_flow_sign,
        buy_count_window: legacy.buy_count_window,
        delta_n_buy_count_window: legacy.delta_n_buy_count_window,
        sell_count_window: legacy.sell_count_window,
        delta_n_sell_count_window: legacy.delta_n_sell_count_window,
        other_book_bid_l1_price: legacy.other_book_bid_l1_price,
        other_delta_n_book_bid_l1_price: legacy.other_delta_n_book_bid_l1_price,
        other_book_ask_l1_price: legacy.other_book_ask_l1_price,
        other_delta_n_book_ask_l1_price: legacy.other_delta_n_book_ask_l1_price,
        other_tick_size: legacy.other_tick_size,
        other_delta_n_tick_size: legacy.other_delta_n_tick_size,
        other_spread: legacy.other_spread,
        other_delta_n_spread: legacy.other_delta_n_spread,
        other_book_bid_l1_size: legacy.other_book_bid_l1_size,
        other_delta_n_book_bid_l1_size: legacy.other_delta_n_book_bid_l1_size,
        other_book_ask_l1_size: legacy.other_book_ask_l1_size,
        other_delta_n_book_ask_l1_size: legacy.other_delta_n_book_ask_l1_size,
        other_book_bid_l2_price: legacy.other_book_bid_l2_price,
        other_delta_n_book_bid_l2_price: legacy.other_delta_n_book_bid_l2_price,
        other_book_bid_l2_size: legacy.other_book_bid_l2_size,
        other_delta_n_book_bid_l2_size: legacy.other_delta_n_book_bid_l2_size,
        other_book_bid_l3_price: legacy.other_book_bid_l3_price,
        other_delta_n_book_bid_l3_price: legacy.other_delta_n_book_bid_l3_price,
        other_book_bid_l3_size: legacy.other_book_bid_l3_size,
        other_delta_n_book_bid_l3_size: legacy.other_delta_n_book_bid_l3_size,
        other_book_ask_l2_price: legacy.other_book_ask_l2_price,
        other_delta_n_book_ask_l2_price: legacy.other_delta_n_book_ask_l2_price,
        other_book_ask_l2_size: legacy.other_book_ask_l2_size,
        other_delta_n_book_ask_l2_size: legacy.other_delta_n_book_ask_l2_size,
        other_book_ask_l3_price: legacy.other_book_ask_l3_price,
        other_delta_n_book_ask_l3_price: legacy.other_delta_n_book_ask_l3_price,
        other_book_ask_l3_size: legacy.other_book_ask_l3_size,
        other_delta_n_book_ask_l3_size: legacy.other_delta_n_book_ask_l3_size,
        other_last_trade_price: legacy.other_last_trade_price,
        other_delta_n_last_trade_price: legacy.other_delta_n_last_trade_price,
        other_trade_size: legacy.other_trade_size,
        other_delta_n_trade_size: legacy.other_delta_n_trade_size,
        other_trade_volume_bucket: legacy.other_trade_volume_bucket,
        other_delta_n_trade_volume_bucket: legacy.other_delta_n_trade_volume_bucket,
        other_buy_count_window: legacy.other_buy_count_window,
        other_delta_n_buy_count_window: legacy.other_delta_n_buy_count_window,
        other_sell_count_window: legacy.other_sell_count_window,
        other_delta_n_sell_count_window: legacy.other_delta_n_sell_count_window,
        other_burstiness_transactions_count: legacy.other_burstiness_transactions_count,
        other_currency_implied_prob: legacy.other_currency_implied_prob,
        currency_price_z_score: legacy.currency_price_z_score,
        currency_price_vs_beat_pct: legacy.currency_price_vs_beat_pct,
        sibling_event_remaining_ms: legacy.sibling_event_remaining_ms,
        sibling_book_bid_l1_price: legacy.sibling_book_bid_l1_price,
        sibling_delta_n_book_bid_l1_price: legacy.sibling_delta_n_book_bid_l1_price,
        sibling_book_ask_l1_price: legacy.sibling_book_ask_l1_price,
        sibling_delta_n_book_ask_l1_price: legacy.sibling_delta_n_book_ask_l1_price,
        sibling_tick_size: legacy.sibling_tick_size,
        sibling_delta_n_tick_size: legacy.sibling_delta_n_tick_size,
        sibling_spread: legacy.sibling_spread,
        sibling_delta_n_spread: legacy.sibling_delta_n_spread,
        sibling_book_bid_l1_size: legacy.sibling_book_bid_l1_size,
        sibling_delta_n_book_bid_l1_size: legacy.sibling_delta_n_book_bid_l1_size,
        sibling_book_ask_l1_size: legacy.sibling_book_ask_l1_size,
        sibling_delta_n_book_ask_l1_size: legacy.sibling_delta_n_book_ask_l1_size,
        sibling_book_bid_l2_price: legacy.sibling_book_bid_l2_price,
        sibling_delta_n_book_bid_l2_price: legacy.sibling_delta_n_book_bid_l2_price,
        sibling_book_bid_l2_size: legacy.sibling_book_bid_l2_size,
        sibling_delta_n_book_bid_l2_size: legacy.sibling_delta_n_book_bid_l2_size,
        sibling_book_bid_l3_price: legacy.sibling_book_bid_l3_price,
        sibling_delta_n_book_bid_l3_price: legacy.sibling_delta_n_book_bid_l3_price,
        sibling_book_bid_l3_size: legacy.sibling_book_bid_l3_size,
        sibling_delta_n_book_bid_l3_size: legacy.sibling_delta_n_book_bid_l3_size,
        sibling_book_ask_l2_price: legacy.sibling_book_ask_l2_price,
        sibling_delta_n_book_ask_l2_price: legacy.sibling_delta_n_book_ask_l2_price,
        sibling_book_ask_l2_size: legacy.sibling_book_ask_l2_size,
        sibling_delta_n_book_ask_l2_size: legacy.sibling_delta_n_book_ask_l2_size,
        sibling_book_ask_l3_price: legacy.sibling_book_ask_l3_price,
        sibling_delta_n_book_ask_l3_price: legacy.sibling_delta_n_book_ask_l3_price,
        sibling_book_ask_l3_size: legacy.sibling_book_ask_l3_size,
        sibling_delta_n_book_ask_l3_size: legacy.sibling_delta_n_book_ask_l3_size,
        sibling_last_trade_price: legacy.sibling_last_trade_price,
        sibling_delta_n_last_trade_price: legacy.sibling_delta_n_last_trade_price,
        sibling_trade_size: legacy.sibling_trade_size,
        sibling_delta_n_trade_size: legacy.sibling_delta_n_trade_size,
        sibling_trade_volume_bucket: legacy.sibling_trade_volume_bucket,
        sibling_delta_n_trade_volume_bucket: legacy.sibling_delta_n_trade_volume_bucket,
        sibling_buy_count_window: legacy.sibling_buy_count_window,
        sibling_delta_n_buy_count_window: legacy.sibling_delta_n_buy_count_window,
        sibling_sell_count_window: legacy.sibling_sell_count_window,
        sibling_delta_n_sell_count_window: legacy.sibling_delta_n_sell_count_window,
        sibling_currency_implied_prob: legacy.sibling_currency_implied_prob,
        sibling_currency_price_vs_beat_pct: legacy.sibling_currency_price_vs_beat_pct,
    }
}

fn current_schema_size() -> usize {
    bincode::serialized_size(&XFrame::<SIZE>::default())
        .expect("XFrame::<SIZE>::default() must be bincode-serializable") as usize
}

fn legacy_schema_size() -> usize {
    bincode::serialized_size(&LegacyXFrame::<SIZE>::default())
        .expect("LegacyXFrame::<SIZE>::default() must be bincode-serializable") as usize
}

/// Глубина каталога с дампами от `xframes/{currency}/<size>/`:
/// `<interval>/<step>/<date>/<file>.bin` — три уровня под `<size>`.
fn collect_dump_files(size_root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let read_dir = match fs::read_dir(size_root) {
        Ok(rd) => rd,
        Err(_) => return Ok(out),
    };
    for interval_entry in read_dir.flatten() {
        let interval_path = interval_entry.path();
        if !interval_path.is_dir() {
            continue;
        }
        for step_entry in fs::read_dir(&interval_path)?.flatten() {
            let step_path = step_entry.path();
            if !step_path.is_dir() {
                continue;
            }
            for date_entry in fs::read_dir(&step_path)?.flatten() {
                let date_path = date_entry.path();
                if !date_path.is_dir() {
                    continue;
                }
                for file_entry in fs::read_dir(&date_path)?.flatten() {
                    let file_path = file_entry.path();
                    if file_path.is_file()
                        && file_path.extension().and_then(|s| s.to_str()) == Some("bin")
                    {
                        out.push(file_path);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Возвращает все каталоги `<size>` внутри `xframes/{currency}/`, исключая
/// `<current_size>`. Не-числовые имена пропускаются — модели и калибровки
/// лежат не в `<size>/`-подкаталоге.
fn legacy_size_dirs(currency_root: &Path, current_size: usize) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let read_dir = match fs::read_dir(currency_root) {
        Ok(rd) => rd,
        Err(_) => return Ok(out),
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(size) = name.parse::<usize>() else {
            continue;
        };
        if size != current_size {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn migrate_file_in_place(src: &Path) -> Result<bool> {
    let bytes = fs::read(src)
        .with_context(|| format!("read {}", src.display()))?;
    let legacy: LegacyMarketXFramesDump = match bincode::deserialize(&bytes) {
        Ok(d) => d,
        Err(_) => return Ok(false),
    };

    let frames_up: Vec<XFrame<SIZE>> =
        legacy.frames_up.into_iter().map(legacy_to_current).collect();
    let frames_down: Vec<XFrame<SIZE>> =
        legacy.frames_down.into_iter().map(legacy_to_current).collect();
    let dump = MarketXFramesDump {
        frames_up,
        frames_down,
        price_to_beat: legacy.price_to_beat,
        final_price: legacy.final_price,
    };

    let serialized = bincode::serialize(&dump)
        .with_context(|| format!("serialize migrated dump for {}", src.display()))?;
    fs::write(src, serialized)
        .with_context(|| format!("write {}", src.display()))?;
    Ok(true)
}

/// Точка входа миграции (`STATUS=migrate`).
///
/// Идём по [`CURRENCIES`], для каждой валюты вычисляем текущий `schema_size`,
/// пересериализуем все `.bin`-дампы устаревшего каталога `<old_size>/` под
/// актуальную раскладку и переименовываем сам каталог в `<current_size>/`.
/// Печатает прогресс в stdout — никаких файлов кроме `xframes/...` не пишем.
///
/// Безопасные проверки:
/// * если `<current_size>/` уже существует одновременно с `<old_size>/`,
///   миграция отказывает — слияние пользователь должен сделать руками
///   (никогда не должно случиться при единственной активной версии);
/// * если в одной валюте найдено несколько устаревших версий — миграция
///   тоже отказывает (требуем единственный исходный каталог).
pub fn run_migration() -> Result<()> {
    let current_size = current_schema_size();
    let legacy_size = legacy_schema_size();
    println!(
        "[migration] schema_size: current={current_size} legacy(expected_old)={legacy_size}"
    );

    for currency in CURRENCIES {
        let currency_root = Path::new("xframes").join(currency);
        if !currency_root.exists() {
            println!(
                "[migration] {currency}: каталог {} отсутствует, пропуск",
                currency_root.display()
            );
            continue;
        }
        let dst_root = currency_root.join(format!("{current_size}"));
        let legacy_dirs = legacy_size_dirs(&currency_root, current_size)?;
        if legacy_dirs.is_empty() {
            println!(
                "[migration] {currency}: устаревших схем не найдено (актуальная — {})",
                dst_root.display()
            );
            continue;
        }
        if legacy_dirs.len() > 1 {
            anyhow::bail!(
                "[migration] {currency}: найдено несколько устаревших версий ({:?}) — оставьте одну и повторите",
                legacy_dirs
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
            );
        }
        let legacy_root = &legacy_dirs[0];
        if dst_root.exists() {
            anyhow::bail!(
                "[migration] {currency}: целевой каталог {} уже существует одновременно со старым {} — слияние не поддерживается",
                dst_root.display(),
                legacy_root.display()
            );
        }

        let files = collect_dump_files(legacy_root)?;
        println!(
            "[migration] {currency}: пересохраняем {} файлов в {}",
            files.len(),
            legacy_root.display()
        );
        let mut migrated = 0usize;
        let mut skipped = 0usize;
        for src in &files {
            match migrate_file_in_place(src) {
                Ok(true) => migrated += 1,
                Ok(false) => skipped += 1,
                Err(err) => {
                    eprintln!("[migration] {}: {err:#}", src.display());
                    skipped += 1;
                }
            }
        }
        println!("[migration]   migrated={migrated} skipped={skipped}");

        fs::rename(legacy_root, &dst_root).with_context(|| {
            format!(
                "rename {} → {}",
                legacy_root.display(),
                dst_root.display()
            )
        })?;
        println!(
            "[migration]   {} → {}",
            legacy_root.display(),
            dst_root.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: размер `LegacyXFrame::<SIZE>::default()` ровно на 2 байта
    /// меньше текущего `XFrame::<SIZE>::default()` (`Option<Vec<BookLevel>>`
    /// в варианте `None` сериализуется как 1-байтный тег; добавили два поля).
    #[test]
    fn legacy_schema_is_2_bytes_smaller_than_current() {
        let current = current_schema_size();
        let legacy = legacy_schema_size();
        assert_eq!(
            current,
            legacy + 2,
            "current({current}) != legacy({legacy}) + 2 — раскладка LegacyXFrame разошлась с XFrame до правки"
        );
    }

    /// Проверяет, что `LegacyXFrame` десериализует то, что **сериализовал
    /// бы** старый `XFrame` (имитируем — сериализуем `LegacyXFrame::default()`
    /// и десериализуем обратно).
    #[test]
    fn legacy_default_round_trip() {
        let legacy = LegacyXFrame::<SIZE>::default();
        let bytes = bincode::serialize(&legacy).expect("serialize legacy");
        let _: LegacyXFrame<SIZE> =
            bincode::deserialize(&bytes).expect("deserialize legacy back");
    }

    /// У легаси-дампов полной лестницы стакана не было — миграция оставляет
    /// `None`, а не пытается сфабриковать её из L1/L2/L3 (см. комментарий в
    /// [`super::legacy_to_current`]).
    #[test]
    fn legacy_to_current_leaves_book_vecs_none() {
        let cur = legacy_to_current(LegacyXFrame::<SIZE>::default());
        assert!(cur.book_bids.is_none());
        assert!(cur.book_asks.is_none());
    }

    /// Даже если в легаси-кадре заполнены L1/L2/L3 ask-поля, миграция
    /// **не** реконструирует `book_asks` — глубина была неизвестна.
    #[test]
    fn legacy_to_current_ignores_l1_l2_l3_for_book_asks() {
        let mut legacy = LegacyXFrame::<SIZE>::default();
        legacy.book_ask_l1_price = Some(0.42);
        legacy.book_ask_l1_size = Some(100.0);
        legacy.book_ask_l2_price = Some(0.43);
        legacy.book_ask_l2_size = Some(50.0);

        let cur = legacy_to_current(legacy);
        assert!(cur.book_asks.is_none());
        // L1/L2/L3 фичи (`#[xfeature]`) при этом мигрируются как есть —
        // обученные XGBoost-модели остаются совместимы.
        assert_eq!(cur.book_ask_l1_price, Some(0.42));
        assert_eq!(cur.book_ask_l1_size, Some(100.0));
        assert_eq!(cur.book_ask_l2_price, Some(0.43));
        assert_eq!(cur.book_ask_l2_size, Some(50.0));
    }
}
