//! Разбор JSON-событий Polymarket market WebSocket в [crate::market_snapshot::MarketSnapshot].

use crate::market_snapshot::{MarketSnapshot, TradeSide};
use crate::util::current_timestamp_ms;
use serde_json::Value;

pub(crate) fn parse_snapshots_from_event(value: &Value, event_type: &str) -> Vec<MarketSnapshot> {
    match event_type {
        "book" | "last_trade_price" | "best_bid_ask" | "tick_size_change" | "market_resolved" => {
            parse_single_snapshot(value, event_type).into_iter().collect()
        }
        "price_change" => parse_price_change_snapshots(value),
        "new_market" => parse_new_market_snapshots(value),
        _ => Vec::new(),
    }
}

fn parse_single_snapshot(value: &Value, event_type: &str) -> Option<MarketSnapshot> {
    let asset_id = value
        .get("asset_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let market_id = value
        .get("market")
        .or_else(|| value.get("condition_id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if asset_id.is_empty() || market_id.is_empty() {
        return None;
    }

    let timestamp_ms = parse_i64(value.get("timestamp")).unwrap_or_else(current_timestamp_ms);
    let (best_bid, best_ask) = parse_book_best_bid_ask(value);

    Some(MarketSnapshot {
        market_id,
        asset_id,
        timestamp_ms,
        best_bid,
        best_ask,
        tick_size: parse_f64(value.get("new_tick_size")).or(parse_f64(value.get("tick_size"))),
        spread: parse_f64(value.get("spread")),
        // WS trade event uses `price` for last trade value.
        last_trade_price: parse_f64(value.get("price")).or(parse_f64(value.get("last_trade_price"))),
        last_trade_size: parse_f64(value.get("size")),
        trade_volume_bucket: None,
        trade_side: parse_trade_side(value.get("side")),
        market_resolved: event_type == "market_resolved",
    })
}

fn parse_price_change_snapshots(value: &Value) -> Vec<MarketSnapshot> {
    let market_id = value
        .get("market")
        .or_else(|| value.get("condition_id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if market_id.is_empty() {
        return Vec::new();
    }
    let timestamp_ms = parse_i64(value.get("timestamp")).unwrap_or_else(current_timestamp_ms);

    let Some(changes) = value.get("price_changes").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut snapshots = Vec::with_capacity(changes.len());
    for change in changes {
        let asset_id = change
            .get("asset_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if asset_id.is_empty() {
            continue;
        }
        snapshots.push(MarketSnapshot {
            market_id: market_id.clone(),
            asset_id,
            timestamp_ms,
            best_bid: parse_f64(change.get("best_bid")),
            best_ask: parse_f64(change.get("best_ask")),
            tick_size: None,
            spread: None,
            last_trade_price: None,
            last_trade_size: None,
            trade_volume_bucket: None,
            trade_side: parse_trade_side(change.get("side")),
            market_resolved: false,
        });
    }
    snapshots
}

fn parse_new_market_snapshots(value: &Value) -> Vec<MarketSnapshot> {
    let market_id = value
        .get("market")
        .or_else(|| value.get("condition_id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if market_id.is_empty() {
        return Vec::new();
    }
    let timestamp_ms = parse_i64(value.get("timestamp")).unwrap_or_else(current_timestamp_ms);
    let tick_size = parse_f64(value.get("order_price_min_tick_size"));
    let Some(asset_ids) = value.get("assets_ids").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut snapshots = Vec::with_capacity(asset_ids.len());
    for asset_id_json in asset_ids {
        let asset_id = asset_id_json.as_str().unwrap_or_default().to_string();
        if asset_id.is_empty() {
            continue;
        }
        snapshots.push(MarketSnapshot {
            market_id: market_id.clone(),
            asset_id,
            timestamp_ms,
            best_bid: None,
            best_ask: None,
            tick_size,
            spread: None,
            last_trade_price: None,
            last_trade_size: None,
            trade_volume_bucket: None,
            trade_side: None,
            market_resolved: false,
        });
    }
    snapshots
}

fn parse_best_price_from_side(levels: Option<&Vec<Value>>, is_bid: bool) -> Option<f64> {
    let Some(orderbook_levels) = levels else {
        return None;
    };
    let mut best_price_so_far: Option<f64> = None;
    for level in orderbook_levels {
        let Some(price) = parse_f64(level.get("price")) else {
            continue;
        };
        let is_better = match best_price_so_far {
            None => true,
            Some(previous_best) if is_bid => price > previous_best,
            Some(previous_best) => price < previous_best,
        };
        if is_better {
            best_price_so_far = Some(price);
        }
    }
    best_price_so_far
}

fn parse_book_best_bid_ask(value: &Value) -> (Option<f64>, Option<f64>) {
    let bids = value.get("bids").and_then(Value::as_array);
    let asks = value.get("asks").and_then(Value::as_array);
    let best_bid = parse_best_price_from_side(bids, true).or(parse_f64(value.get("best_bid")));
    let best_ask = parse_best_price_from_side(asks, false).or(parse_f64(value.get("best_ask")));
    (best_bid, best_ask)
}

fn parse_trade_side(json_field: Option<&Value>) -> Option<TradeSide> {
    json_field
        .and_then(Value::as_str)
        .and_then(|side| match side.to_ascii_uppercase().as_str() {
            "BUY" => Some(TradeSide::Buy),
            "SELL" => Some(TradeSide::Sell),
            _ => None,
        })
}

fn parse_f64(json_field: Option<&Value>) -> Option<f64> {
    match json_field {
        Some(Value::Number(number)) => number.as_f64(),
        Some(Value::String(text)) => text.parse::<f64>().ok(),
        _ => None,
    }
}

fn parse_i64(json_field: Option<&Value>) -> Option<i64> {
    match json_field {
        Some(Value::Number(number)) => number.as_i64(),
        Some(Value::String(text)) => text.parse::<i64>().ok(),
        _ => None,
    }
}
