use derivative::Derivative;
use crate::ws::{MarketSnapshot, TradeSide};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::BTreeMap;
use xframe_features::FeatureLen;
use xframe_features_derive::XFeatures;

pub const SIZE: usize = 13;
const WINDOW_MS: i64 = 60_000;

/// Кадр признаков по одному ассету: состояние стакана и сделок на момент снапшота плюс лаги по последним `N` предыдущим кадрам (от ближайшего по времени к более ранним).
///
/// Поля с атрибутом `#[xfeature]` попадают в вектор для обучения; `market_id`, `asset_id`, `trade_side` и `delta_n_trade_side` — без `#[xfeature]` (идентификаторы и сторона сделки для логики/отладки).
#[serde_as]
#[derive(Debug, Serialize, Deserialize, Derivative, Clone, XFeatures)]
#[derivative(Default)]
pub struct XFrame<const N: usize> {
    /// Идентификатор условия рынка (`condition_id`), как в поле `market` WS.
    pub market_id: String,
    /// Идентификатор токена в CLOB (`asset_id` / token id).
    pub asset_id: String,
    /// Плановое время окончания / resolution события (Unix UTC, мс), из Gamma `end_date_rfc3339`; `None`, если дата ещё не подгружена или строка не распарсилась.
    #[xfeature]
    pub event_end_ms: Option<i64>,
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
    /// Число buy-сделок за скользящее окно `WINDOW_MS` мс по ключам кадров (wall time), включая текущий бакет при наличии сделки.
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
    /// Разность `burstiness_transactions_count` с `i`-м предыдущим кадром; `None`, если в одном из кадров метрики не было.
    #[xfeature]
    #[derivative(Default(value = "[None; N]"))]
    #[serde_as(as = "[_; N]")]
    pub delta_n_burstiness_transactions_count: [Option<f64>; N],
}

impl<const N: usize> XFrame<N> {
    pub fn new(
        snapshot: MarketSnapshot,
        frames: &BTreeMap<i64, XFrame<N>>,
        event_end_ms: Option<i64>,
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

        let mut frame = XFrame::<N> {
            market_id: snapshot.market_id,
            asset_id: snapshot.asset_id,
            event_end_ms,
            best_bid,
            best_ask,
            last_trade_price,
            trade_size,
            trade_volume_bucket,
            trade_side,
            ..Default::default()
        };
        frame.populate_window_metrics(frames, wall_ts_ms);
        frame.populate_deltas(frames);
        frame
    }

    fn populate_window_metrics(&mut self, frames: &BTreeMap<i64, XFrame<N>>, wall_ts_ms: i64) {
        let window_start = wall_ts_ms - WINDOW_MS;
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
            self.delta_n_burstiness_transactions_count[lag_index] =
                match (
                    self.burstiness_transactions_count,
                    prior_frame.burstiness_transactions_count,
                ) {
                    (Some(current_burst), Some(prior_burst)) => Some(current_burst - prior_burst),
                    _ => None,
                };
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