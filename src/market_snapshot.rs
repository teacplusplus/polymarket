pub use crate::constants::{
    BtcUpDownDelayClass, BtcUpDownOutcome, TradeSide, XFrameIntervalKind,
};

#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub market_id: String,
    pub asset_id: String,
    pub xframe_interval_kind: XFrameIntervalKind,
    pub btc_up_down_outcome: BtcUpDownOutcome,
    pub timestamp_ms: i64,
    pub book_bid_l1_price: Option<f64>,
    pub book_bid_l1_size: Option<f64>,
    pub book_ask_l1_price: Option<f64>,
    pub book_ask_l1_size: Option<f64>,
    pub book_bid_l2_price: Option<f64>,
    pub book_bid_l2_size: Option<f64>,
    pub book_bid_l3_price: Option<f64>,
    pub book_bid_l3_size: Option<f64>,
    pub book_ask_l2_price: Option<f64>,
    pub book_ask_l2_size: Option<f64>,
    pub book_ask_l3_price: Option<f64>,
    pub book_ask_l3_size: Option<f64>,
    pub tick_size: Option<f64>,
    pub spread: Option<f64>,
    pub last_trade_price: Option<f64>,
    pub last_trade_size: Option<f64>,
    pub trade_volume_bucket: Option<f64>,
    pub trade_side: Option<TradeSide>,
    pub market_resolved: bool,
}

pub fn aggregate_events(events: Vec<MarketSnapshot>, timestamp_ms: i64) -> Option<MarketSnapshot> {
    let first_event_snapshot = events.first()?;
    let mut aggregated_market_snapshot = MarketSnapshot {
        market_id: first_event_snapshot.market_id.clone(),
        asset_id: first_event_snapshot.asset_id.clone(),
        xframe_interval_kind: first_event_snapshot.xframe_interval_kind,
        btc_up_down_outcome: first_event_snapshot.btc_up_down_outcome,
        timestamp_ms,
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
        aggregated_market_snapshot.btc_up_down_outcome = event_market_snapshot.btc_up_down_outcome;
        if event_market_snapshot.book_bid_l1_price.is_some() {
            aggregated_market_snapshot.book_bid_l1_price = event_market_snapshot.book_bid_l1_price;
        }
        if event_market_snapshot.book_ask_l1_price.is_some() {
            aggregated_market_snapshot.book_ask_l1_price = event_market_snapshot.book_ask_l1_price;
        }
        if event_market_snapshot.book_bid_l1_size.is_some() {
            aggregated_market_snapshot.book_bid_l1_size = event_market_snapshot.book_bid_l1_size;
        }
        if event_market_snapshot.book_ask_l1_size.is_some() {
            aggregated_market_snapshot.book_ask_l1_size = event_market_snapshot.book_ask_l1_size;
        }
        if event_market_snapshot.book_bid_l2_price.is_some() {
            aggregated_market_snapshot.book_bid_l2_price = event_market_snapshot.book_bid_l2_price;
        }
        if event_market_snapshot.book_bid_l2_size.is_some() {
            aggregated_market_snapshot.book_bid_l2_size = event_market_snapshot.book_bid_l2_size;
        }
        if event_market_snapshot.book_bid_l3_price.is_some() {
            aggregated_market_snapshot.book_bid_l3_price = event_market_snapshot.book_bid_l3_price;
        }
        if event_market_snapshot.book_bid_l3_size.is_some() {
            aggregated_market_snapshot.book_bid_l3_size = event_market_snapshot.book_bid_l3_size;
        }
        if event_market_snapshot.book_ask_l2_price.is_some() {
            aggregated_market_snapshot.book_ask_l2_price = event_market_snapshot.book_ask_l2_price;
        }
        if event_market_snapshot.book_ask_l2_size.is_some() {
            aggregated_market_snapshot.book_ask_l2_size = event_market_snapshot.book_ask_l2_size;
        }
        if event_market_snapshot.book_ask_l3_price.is_some() {
            aggregated_market_snapshot.book_ask_l3_price = event_market_snapshot.book_ask_l3_price;
        }
        if event_market_snapshot.book_ask_l3_size.is_some() {
            aggregated_market_snapshot.book_ask_l3_size = event_market_snapshot.book_ask_l3_size;
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
        if let Some(size) = event_market_snapshot.last_trade_size && size > 0.0 {
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
