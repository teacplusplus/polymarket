use derivative::Derivative;
use anyhow::bail;
use crate::market_snapshot::{MarketSnapshot, TradeSide};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::{BTreeMap, HashMap, HashSet};
use xframe_features::FeatureLen;
use xframe_features_derive::XFeatures;
pub use crate::market_snapshot::{BtcUpDownDelayClass, BtcUpDownOutcome, XFrameIntervalKind};
use crate::util::btc_up_down_five_min_slot_from_gamma_question;

pub const SIZE: usize = 13;

const MIN_POSITIVE_ASK: f64 = 1e-12;
/// Окно секундных цен BTC для μ и σ в z-score (≈ 60 мин). Ключи `prices_by_sec` — Unix-секунды.
const BTC_PRICE_ZSCORE_WINDOW_SEC: i64 = SIZE as i64;
/// Минимум точек в окне для выборочного СКО.
const BTC_PRICE_ZSCORE_MIN_POINTS: usize = 2;

/// Кадр признаков по одному ассету: состояние стакана и сделок на момент снапшота плюс лаги по последним `N` предыдущим кадрам (от ближайшего по времени к более ранним).
///
/// Поля с атрибутом `#[xfeature]` попадают в вектор для обучения; `market_id`, `asset_id`, `trade_side` и `delta_n_trade_side` — без `#[xfeature]` (идентификаторы и сторона сделки для логики/отладки).
///
/// `xframe_interval_type`: дискриминант [`XFrameIntervalKind`] ([XFRAME_INTERVAL_TYPE_15M] / [XFRAME_INTERVAL_TYPE_5M]). `btc_up_down_outcome`: дискриминант [`BtcUpDownOutcome`] ([XFRAME_BTC_OUTCOME_UP] / [XFRAME_BTC_OUTCOME_DOWN]).
/// Поля `other_*` — копия микроструктуры противоположной ноги на тот же бакет; заполняются через [XFrame::copy_other_leg_features_from] в `ProjectManager` после вставки пары кадров.
#[serde_as]
#[derive(Debug, Serialize, Deserialize, Derivative, Clone, XFeatures)]
#[derivative(Default)]
pub struct XFrame<const N: usize> {
    /// Идентификатор условия рынка (`condition_id`), как в поле `market` WS.
    pub market_id: String,
    /// Идентификатор токена в CLOB (`asset_id` / token id).
    pub asset_id: String,
    /// Тип окна BTC up/down: `0` — 15 мин ([XFRAME_INTERVAL_TYPE_15M]), `1` — 5 мин ([XFRAME_INTERVAL_TYPE_5M]).
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub xframe_interval_type: i32,
    /// Исход токена по Gamma (`outcomes` + `clobTokenIds`): [BtcUpDownOutcome] → [XFRAME_BTC_OUTCOME_UP] / [XFRAME_BTC_OUTCOME_DOWN].
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub btc_up_down_outcome: i32,
    /// Дискриминант [`BtcUpDownDelayClass`] по Gamma `question` (см. [crate::util::btc_up_down_five_min_slot_from_gamma_question]): [XFRAME_BTC_5M_DELAY_CLASS_DELAY_5MIN] / [XFRAME_BTC_5M_DELAY_CLASS_DELAY_10MIN]. Для 15m — [XFRAME_BTC_15M_DELAY_CLASS_ALIGNED].
    #[xfeature]
    #[derivative(Default(value = "0"))]
    pub btc_up_down_delay_class: i32,
    /// Сколько миллисекунд осталось до конца события рынка: `event_end_ms - timestamp_ms` снапшота; при `event_end_ms <= timestamp` — `0`. Если конец события неизвестен — `0`.
    #[xfeature]
    #[derivative(Default(value = "-1"))]
    pub event_remaining_ms: i64,
    /// Лучшая цена bid на конец интервала / бакета снапшота.
    #[xfeature]
    pub best_bid: f64,
    /// Разность текущего `best_bid` и `best_bid` у `i`-го предыдущего кадра: индекс `0` — непосредственный предшественник по времени, далее глубже в прошлое; `None`, если такого кадра ещё не было.
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_best_bid: [Option<f64>; N],
    /// Лучшая цена ask на конец интервала.
    #[xfeature]
    pub best_ask: f64,
    /// Разность текущего `best_ask` и `best_ask` у `i`-го предыдущего кадра (индексация как у `delta_n_best_bid`).
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_best_ask: [Option<f64>; N],
    /// Отношение `best_bid / best_ask` по этому токену; `None`, если ask ≈ 0 (как «yes относительно ask» в виде одного числа).
    #[xfeature]
    pub best_bid_over_best_ask: Option<f64>,
    /// Разность отношения bid/ask с `i`-м предыдущим кадром; `None`, если сравнивать нельзя.
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_best_bid_over_best_ask: [Option<f64>; N],
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
    /// Направление сделки в бакете: `+1` покупка, `−1` продажа, `0` — сделки не было (для окон и логики заполнения).
    pub trade_side: i8,
    /// Разность `trade_side` с `i`-м предыдущим кадром; не входит в признаки обучения.
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_trade_side: [Option<i8>; N],
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
    /// «Burstiness» интервалов между сделками в окне: `(std − mean) / (std + mean)` по мс-зазорам между временными метками сделок; `None`, если сделок меньше двух или знаменатель невалиден.
    #[xfeature]
    pub burstiness_transactions_count: Option<f64>,
    // --- Противоположный токен в том же `market_id` (Up ↔ Down), те же поля, что выше до burstiness. ---
    #[xfeature]
    #[derivative(Default(value = "0.0"))]
    pub other_best_bid: f64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_best_bid: [Option<f64>; N],
    #[xfeature]
    #[derivative(Default(value = "0.0"))]
    pub other_best_ask: f64,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_best_ask: [Option<f64>; N],
    #[xfeature]
    pub other_best_bid_over_best_ask: Option<f64>,
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_best_bid_over_best_ask: [Option<f64>; N],
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
    pub other_trade_side: i8,
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub other_delta_n_trade_side: [Option<i8>; N],
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
    /// Z-score цены BTC/USDT: `(p - mu) / sigma`; история — `ProjectManager::rtds_btc_prices_by_sec` (ключ Unix-секунда); `p` — последняя точка в окне, `mu`/`sigma` — по всем ценам окна.
    #[xfeature]
    pub btc_price_z_score: Option<f64>,
    /// Относительное отклонение спота BTC от Gamma «price to beat»: `(price_to_beat - btc_spot) / price_to_beat * 100` (%); спот — последняя секундная цена из `rtds_btc_prices_by_sec`.
    #[xfeature]
    pub btc_price_vs_beat_pct: Option<f64>,
}

pub fn compute_btc_up_down_delay_class(
    interval_kind: XFrameIntervalKind,
    gamma_question: Option<&str>,
) -> BtcUpDownDelayClass {
    match interval_kind {
        XFrameIntervalKind::FifteenMin => BtcUpDownDelayClass::Aligned,
        XFrameIntervalKind::FiveMin => {
            let Some(q) = gamma_question else {
                return BtcUpDownDelayClass::Aligned;
            };
            btc_up_down_five_min_slot_from_gamma_question(q).unwrap_or(BtcUpDownDelayClass::Aligned)
        }
    }
}

impl<const N: usize> XFrame<N> {
    pub fn new(
        snapshot: MarketSnapshot,
        frames: &BTreeMap<i64, XFrame<N>>,
        event_end_ms: Option<i64>,
        gamma_question: Option<&str>,
        btc_price_z_score: Option<f64>,
        btc_price_vs_beat_pct: Option<f64>,
        window_ms: i64,
    ) -> XFrame<N> {
        let previous = frames.values().next_back();

        let wall_ts_ms = snapshot.timestamp_ms;

        let best_bid = snapshot
            .best_bid
            .or(previous.map(|prior_frame| prior_frame.best_bid))
            .unwrap_or(0.0);
        let best_ask = snapshot
            .best_ask
            .or(previous.map(|prior_frame| prior_frame.best_ask))
            .unwrap_or(0.0);
        let last_trade_price = snapshot
            .last_trade_price
            .or(previous.and_then(|prior_frame| prior_frame.last_trade_price));
        let trade_size = snapshot.last_trade_size.unwrap_or(0.0);
        let trade_volume_bucket = snapshot.trade_volume_bucket.unwrap_or(trade_size.max(0.0));
        let trade_side = match snapshot.trade_side {
            Some(TradeSide::Buy) => 1,
            Some(TradeSide::Sell) => -1,
            None => 0,
        };

        let best_bid_over_best_ask = (best_ask > MIN_POSITIVE_ASK).then_some(best_bid / best_ask);

        let event_remaining_ms = match event_end_ms {
            Some(end_ms) => end_ms.saturating_sub(wall_ts_ms),
            None => 0,
        };

        let btc_up_down_delay_class = compute_btc_up_down_delay_class(
            snapshot.xframe_interval_kind,
            gamma_question,
        )
        .as_i32();

        let mut frame = XFrame::<N> {
            market_id: snapshot.market_id,
            asset_id: snapshot.asset_id,
            xframe_interval_type: snapshot.xframe_interval_kind.as_i32(),
            btc_up_down_outcome: snapshot.btc_up_down_outcome.as_i32(),
            btc_up_down_delay_class,
            event_remaining_ms,
            best_bid,
            best_ask,
            best_bid_over_best_ask,
            btc_price_z_score,
            btc_price_vs_beat_pct,
            last_trade_price,
            trade_size,
            trade_volume_bucket,
            trade_side,
            ..Default::default()
        };
        frame.populate_window_metrics(frames, wall_ts_ms, window_ms);
        frame.populate_deltas(frames);
        frame
    }

    /// Заполняет `other_*` копией признаков стакана/сделок с кадра противоположной ноги (`Up`/`Down`) на тот же `aligned_ts` (тот же `market_id`, другой `asset_id`).
    pub fn copy_other_leg_features_from(&mut self, other: &XFrame<N>) {
        self.other_best_bid = other.best_bid;
        self.other_delta_n_best_bid = other.delta_n_best_bid;
        self.other_best_ask = other.best_ask;
        self.other_delta_n_best_ask = other.delta_n_best_ask;
        self.other_best_bid_over_best_ask = other.best_bid_over_best_ask;
        self.other_delta_n_best_bid_over_best_ask = other.delta_n_best_bid_over_best_ask;
        self.other_last_trade_price = other.last_trade_price;
        self.other_delta_n_last_trade_price = other.delta_n_last_trade_price;
        self.other_trade_size = other.trade_size;
        self.other_delta_n_trade_size = other.delta_n_trade_size;
        self.other_trade_volume_bucket = other.trade_volume_bucket;
        self.other_delta_n_trade_volume_bucket = other.delta_n_trade_volume_bucket;
        self.other_trade_side = other.trade_side;
        self.other_delta_n_trade_side = other.delta_n_trade_side;
        self.other_buy_count_window = other.buy_count_window;
        self.other_delta_n_buy_count_window = other.delta_n_buy_count_window;
        self.other_sell_count_window = other.sell_count_window;
        self.other_delta_n_sell_count_window = other.delta_n_sell_count_window;
        self.other_burstiness_transactions_count = other.burstiness_transactions_count;
    }

    fn populate_window_metrics(
        &mut self,
        frames: &BTreeMap<i64, XFrame<N>>,
        wall_ts_ms: i64,
        window_ms: i64,
    ) {
        let window_start = wall_ts_ms.saturating_sub(window_ms.max(0));
        let mut trade_timestamps = Vec::new();
        let mut buy_count_window = if self.trade_side > 0 {
            1
        } else {
            0
        };
        let mut sell_count_window = if self.trade_side < 0 {
            1
        } else {
            0
        };

        if Self::frame_has_trade(self) {
            trade_timestamps.push(wall_ts_ms);
        }

        for (&aligned_timestamp_ms, prior_xframe) in frames.range(window_start..=wall_ts_ms) {
            if Self::frame_has_trade(prior_xframe) {
                if prior_xframe.trade_side > 0 {
                    buy_count_window += 1;
                } else if prior_xframe.trade_side < 0 {
                    sell_count_window += 1;
                }
                trade_timestamps.push(aligned_timestamp_ms);
            }
        }

        trade_timestamps.sort_unstable();
        let inter_trade_gaps: Vec<f64> = trade_timestamps
            .windows(2)
            .map(|adjacent_pair| (adjacent_pair[1] - adjacent_pair[0]) as f64)
            .collect();

        self.buy_count_window = buy_count_window;
        self.sell_count_window = sell_count_window;
        self.burstiness_transactions_count = Self::burstiness_from_gaps(&inter_trade_gaps);
    }

    fn frame_has_trade(frame: &XFrame<N>) -> bool {
        frame.trade_size > 0.0 || frame.trade_volume_bucket > 0.0
    }

    fn populate_deltas(&mut self, frames: &BTreeMap<i64, XFrame<N>>) {
        for (lag_index, prior_frame) in frames.values().rev().take(N).enumerate() {
            self.delta_n_best_bid[lag_index] = Some(self.best_bid - prior_frame.best_bid);
            self.delta_n_best_ask[lag_index] = Some(self.best_ask - prior_frame.best_ask);
            self.delta_n_best_bid_over_best_ask[lag_index] = match (
                self.best_bid_over_best_ask,
                prior_frame.best_bid_over_best_ask,
            ) {
                (Some(current_ratio), Some(prior_ratio)) => Some(current_ratio - prior_ratio),
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
            self.delta_n_trade_side[lag_index] = Some(self.trade_side - prior_frame.trade_side);
            self.delta_n_buy_count_window[lag_index] =
                Some(self.buy_count_window as i64 - prior_frame.buy_count_window as i64);
            self.delta_n_sell_count_window[lag_index] =
                Some(self.sell_count_window as i64 - prior_frame.sell_count_window as i64);
        }
    }

    fn burstiness_from_gaps(gaps: &[f64]) -> Option<f64> {
        if gaps.len() < 2 {
            return None;
        }
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        if mean <= 0.0 {
            return None;
        }
        let variance = gaps
            .iter()
            .map(|gap_seconds| {
                let deviation_from_mean = gap_seconds - mean;
                deviation_from_mean * deviation_from_mean
            })
            .sum::<f64>()
            / gaps.len() as f64;
        let std_dev = variance.sqrt();
        let denom = std_dev + mean;
        if denom > 0.0 {
            Some((std_dev - mean) / denom)
        } else {
            None
        }
    }
}

/// Вторая нога BTC up/down: другой `asset_id` с противоположным кодом в `btc_up_down_by_asset_id` среди кандидатов (батч + уже сохранённые кадры по тому же `market_id`).
pub fn find_opposite_asset_id(
    asset_id: &str,
    btc_up_down_by_asset_id: &HashMap<String, BtcUpDownOutcome>,
    candidate_asset_ids: &HashSet<String>,
) -> anyhow::Result<String> {
    let Some(&my_outcome) = btc_up_down_by_asset_id.get(asset_id) else {
        bail!(
            "неизвестный код btc up/down для asset_id={asset_id} в btc_up_down_by_asset_id"
        );
    };
    let other_outcome = my_outcome.opposite();
    for candidate_id in candidate_asset_ids {
        if candidate_id == asset_id {
            continue;
        }
        if btc_up_down_by_asset_id.get(candidate_id).copied() == Some(other_outcome) {
            return Ok(candidate_id.clone());
        }
    }
    bail!(
        "неизвестный кандидат btc up/down ({my_outcome:?}) для asset_id={asset_id} в btc_up_down_by_asset_id"
    );
}

/// z = (p - mu) / sigma: только секундный ряд `prices_by_sec` (ключ — Unix-секунда); `p` — цена последней точки в окне (самый поздний ключ ≤ `reference_sec`), `mu` и `sigma` — по всем ценам в том же окне.
pub fn btc_price_z_score_from_sec_history(
    prices_by_sec: &BTreeMap<i64, f64>,
    reference_sec: i64,
) -> Option<f64> {
    let window_start_sec = reference_sec.saturating_sub(BTC_PRICE_ZSCORE_WINDOW_SEC);
    let window: Vec<f64> = prices_by_sec
        .range(window_start_sec..=reference_sec)
        .map(|(_, price)| *price)
        .collect();
    let n = window.len();
    if n < BTC_PRICE_ZSCORE_MIN_POINTS {
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