#[derive(Debug, Clone, Copy)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Тип окна BTC up/down (в [`crate::xframe::XFrame`] хранится как `i32`-дискриминант).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
#[repr(i32)]
pub enum XFrameIntervalKind {
    #[default]
    FifteenMin = 0,
    FiveMin = 1,
}

impl XFrameIntervalKind {
    #[inline]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    #[inline]
    pub const fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::FifteenMin),
            1 => Some(Self::FiveMin),
            _ => None,
        }
    }
}

/// Исход токена BTC up/down (в [`crate::xframe::XFrame`] хранится как `i32`-дискриминант).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
#[repr(i32)]
pub enum BtcUpDownOutcome {
    #[default]
    Down = 0,
    Up = 1,
}

impl BtcUpDownOutcome {
    #[inline]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    #[inline]
    pub const fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Down),
            1 => Some(Self::Up),
            _ => None,
        }
    }

    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Down => Self::Up,
            Self::Up => Self::Down,
        }
    }
}

/// Класс сдвига / номер пятиминутки в 15m-блоке (в [`crate::xframe::XFrame`] хранится как `i32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
#[repr(i32)]
pub enum BtcUpDownDelayClass {
    /// 15m-рынок или первая пятиминутка в блоке.
    #[default]
    Aligned = 0,
    /// Вторая пятиминутка (+5 мин).
    Delay5Min = 1,
    /// Третья пятиминутка (+10 мин).
    Delay10Min = 2,
}

impl BtcUpDownDelayClass {
    #[inline]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    #[inline]
    pub const fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Aligned),
            1 => Some(Self::Delay5Min),
            2 => Some(Self::Delay10Min),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub market_id: String,
    pub asset_id: String,
    pub xframe_interval_kind: XFrameIntervalKind,
    pub btc_up_down_outcome: BtcUpDownOutcome,
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

pub fn aggregate_events(events: Vec<MarketSnapshot>, timestamp_ms: i64) -> Option<MarketSnapshot> {
    let first_event_snapshot = events.first()?;
    let mut aggregated_market_snapshot = MarketSnapshot {
        market_id: first_event_snapshot.market_id.clone(),
        asset_id: first_event_snapshot.asset_id.clone(),
        xframe_interval_kind: first_event_snapshot.xframe_interval_kind,
        btc_up_down_outcome: first_event_snapshot.btc_up_down_outcome,
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
        aggregated_market_snapshot.btc_up_down_outcome = event_market_snapshot.btc_up_down_outcome;
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
