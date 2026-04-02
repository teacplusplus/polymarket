use crate::api_data_manager::{OrderFilledRow, SUBGRAPH_COLLATERAL_ASSET_ID};

#[derive(Debug, Clone, Copy)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub market_id: String,
    pub asset_id: String,
    pub timestamp_ms: i64,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub tick_size: Option<f64>,
    pub spread: Option<f64>,
    pub last_trade_price: Option<f64>,
    pub last_trade_size: Option<f64>,
    pub trade_volume_bucket: Option<f64>,
    pub trade_side: Option<TradeSide>,
    pub market_resolved: bool,
}

pub(crate) fn aggregate_events(
    events: Vec<MarketSnapshot>,
    timestamp_ms: i64,
) -> Option<MarketSnapshot> {
    let first_event_snapshot = events.first()?;
    let mut aggregated_market_snapshot = MarketSnapshot {
        market_id: first_event_snapshot.market_id.clone(),
        asset_id: first_event_snapshot.asset_id.clone(),
        timestamp_ms,
        best_bid: None,
        best_ask: None,
        tick_size: None,
        spread: None,
        last_trade_price: None,
        last_trade_size: None,
        trade_volume_bucket: None,
        trade_side: None,
        market_resolved: false,
    };
    let mut bucket_trade_volume = 0.0_f64;
    let mut last_trade_side = None;

    for event_market_snapshot in events {
        if event_market_snapshot.best_bid.is_some() {
            aggregated_market_snapshot.best_bid = event_market_snapshot.best_bid;
        }
        if event_market_snapshot.best_ask.is_some() {
            aggregated_market_snapshot.best_ask = event_market_snapshot.best_ask;
        }
        if event_market_snapshot.tick_size.is_some() {
            aggregated_market_snapshot.tick_size = event_market_snapshot.tick_size;
        }
        if event_market_snapshot.spread.is_some() {
            aggregated_market_snapshot.spread = event_market_snapshot.spread;
        }
        if event_market_snapshot.last_trade_price.is_some() {
            aggregated_market_snapshot.last_trade_price = event_market_snapshot.last_trade_price;
        }
        if let Some(size) = event_market_snapshot.last_trade_size
            && size > 0.0
        {
            bucket_trade_volume += size;
            aggregated_market_snapshot.last_trade_size = Some(size);
            if let Some(side) = event_market_snapshot.trade_side {
                aggregated_market_snapshot.trade_side = Some(side);
                last_trade_side = Some(side);
            }
        }
        aggregated_market_snapshot.market_resolved |= event_market_snapshot.market_resolved;
    }

    if bucket_trade_volume > 0.0 {
        aggregated_market_snapshot.trade_volume_bucket = Some(bucket_trade_volume);
    }
    if aggregated_market_snapshot.trade_side.is_none() {
        aggregated_market_snapshot.trade_side = last_trade_side;
    }
    Some(aggregated_market_snapshot)
}

/// Сливает новый агрегат (события после предыдущего drain) с уже сохранённым состоянием того же бакета `aligned_ts`.
/// Котировки — последнее непустое значение; объёмы сделок суммируются; последняя сделка берётся из новой порции, если в ней есть сделка.
pub(crate) fn merge_incremental_bucket_snapshots(
    prior: MarketSnapshot,
    newer: MarketSnapshot,
    aligned_ts_ms: i64,
) -> MarketSnapshot {
    let volume_sum =
        prior.trade_volume_bucket.unwrap_or(0.0) + newer.trade_volume_bucket.unwrap_or(0.0);
    let trade_volume_bucket = if volume_sum > 0.0 {
        Some(volume_sum)
    } else {
        None
    };

    let newer_has_trade = newer.last_trade_size.is_some_and(|size| size > 0.0);
    let (last_trade_price, last_trade_size, trade_side) = if newer_has_trade {
        (
            newer.last_trade_price.or(prior.last_trade_price),
            newer.last_trade_size,
            newer.trade_side.or(prior.trade_side),
        )
    } else {
        (
            prior.last_trade_price.or(newer.last_trade_price),
            prior.last_trade_size.or(newer.last_trade_size),
            prior.trade_side.or(newer.trade_side),
        )
    };

    MarketSnapshot {
        market_id: prior.market_id,
        asset_id: prior.asset_id,
        timestamp_ms: aligned_ts_ms,
        best_bid: newer.best_bid.or(prior.best_bid),
        best_ask: newer.best_ask.or(prior.best_ask),
        tick_size: newer.tick_size.or(prior.tick_size),
        spread: newer.spread.or(prior.spread),
        last_trade_price,
        last_trade_size,
        trade_volume_bucket,
        trade_side,
        market_resolved: prior.market_resolved | newer.market_resolved,
    }
}

struct ParsedOutcomeFill {
    price: f64,
    side: TradeSide,
    token_size: f64,
}

fn parse_outcome_fill(
    asset_id: &str,
    order_filled_row: &OrderFilledRow,
) -> Option<ParsedOutcomeFill> {
    let maker_amt: f64 = order_filled_row.maker_amount_raw.parse().ok()?;
    let taker_amt: f64 = order_filled_row.taker_amount_raw.parse().ok()?;
    if maker_amt <= 0.0 || taker_amt <= 0.0 {
        return None;
    }
    let scale = 1e6_f64;
    if order_filled_row.maker_asset_id == SUBGRAPH_COLLATERAL_ASSET_ID
        && order_filled_row.taker_asset_id == asset_id
    {
        let price = (maker_amt / taker_amt).clamp(0.0, 1.0);
        let token_size = taker_amt / scale;
        Some(ParsedOutcomeFill {
            price,
            side: TradeSide::Buy,
            token_size,
        })
    } else if order_filled_row.taker_asset_id == SUBGRAPH_COLLATERAL_ASSET_ID
        && order_filled_row.maker_asset_id == asset_id
    {
        let price = (taker_amt / maker_amt).clamp(0.0, 1.0);
        let token_size = maker_amt / scale;
        Some(ParsedOutcomeFill {
            price,
            side: TradeSide::Sell,
            token_size,
        })
    } else {
        None
    }
}

/// Котировки — из CLOB [prices-history](https://docs.polymarket.com/trading/orderbook#price-history) (mid ± половина тика).
///
/// `live_resolved` — [`ApiAssetLive::resolved`](crate::api_data_manager::ApiAssetLive) на момент `fetch_asset_snapshot`; `None` → `market_resolved: false`.
pub fn market_snapshot_from_historical_bucket(
    market_id: String,
    asset_id: String,
    aligned_ts_ms: i64,
    _interval_secs: u64,
    rows_in_bucket: &[OrderFilledRow],
    mid_price: Option<f64>,
    tick_size: Option<&str>,
    live_resolved: Option<bool>,
) -> MarketSnapshot {
    let tick_half = tick_size
        .and_then(|tick_size_str| tick_size_str.parse::<f64>().ok())
        .map(|parsed_tick| parsed_tick * 0.5);

    let mut parsed: Vec<(i64, ParsedOutcomeFill)> = Vec::new();
    for order_filled_row in rows_in_bucket {
        let fill_timestamp_ms = order_filled_row.timestamp_sec.saturating_mul(1000);
        if let Some(parsed_fill) = parse_outcome_fill(asset_id.as_str(), order_filled_row) {
            parsed.push((fill_timestamp_ms, parsed_fill));
        }
    }

    let mut bucket_trade_volume = 0.0_f64;
    for (_fill_timestamp_ms, parsed_fill) in &parsed {
        bucket_trade_volume += parsed_fill.token_size;
    }

    let last_fill = parsed
        .iter()
        .max_by_key(|(fill_timestamp_ms, _parsed_fill)| *fill_timestamp_ms);

    let (best_bid, best_ask) = match mid_price {
        Some(mid_price_value) => {
            let tick_half_width = tick_half.unwrap_or(0.0);
            (
                Some((mid_price_value - tick_half_width).max(0.0)),
                Some((mid_price_value + tick_half_width).min(1.0)),
            )
        }
        None => (None, None),
    };

    let (last_trade_price, last_trade_size, trade_side) = match last_fill {
        Some((_fill_timestamp_ms, parsed_fill)) => (
            Some(parsed_fill.price),
            Some(parsed_fill.token_size),
            Some(parsed_fill.side),
        ),
        None => (None, None, None),
    };

    MarketSnapshot {
        market_id,
        asset_id,
        timestamp_ms: aligned_ts_ms,
        best_bid,
        best_ask,
        tick_size: tick_size.and_then(|tick_size_str| tick_size_str.parse().ok()),
        spread: match (best_bid, best_ask) {
            (Some(best_bid_price), Some(best_ask_price)) => {
                Some((best_ask_price - best_bid_price).max(0.0))
            }
            _ => None,
        },
        last_trade_price,
        last_trade_size,
        trade_volume_bucket: if bucket_trade_volume > 0.0 {
            Some(bucket_trade_volume)
        } else {
            None
        },
        trade_side,
        market_resolved: live_resolved.unwrap_or(false),
    }
}
