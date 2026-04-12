//! Общие константы времени Polymarket up/down по валюте и дискриминанты для снапшотов / XFrame.

/// Длительность пятиминутного окна в секундах (slug `btc-updown-5m-*`).
pub const FIVE_MIN_SEC: i64 = 300;
/// Длительность пятнадцатиминутного окна в секундах (slug `btc-updown-15m-*`).
pub const FIFTEEN_MIN_SEC: i64 = 900;

/// Горизонт up/down по валюте на Polymarket (`btc-updown-5m-*` / `btc-updown-15m-*` в slug).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrencyUpDownInterval {
    FiveMin,
    FifteenMin,
}

impl CurrencyUpDownInterval {
    pub const fn interval_sec(self) -> i64 {
        match self {
            Self::FiveMin => FIVE_MIN_SEC,
            Self::FifteenMin => FIFTEEN_MIN_SEC,
        }
    }

    /// Длительность окна в минутах (как в Gamma `question`).
    pub const fn duration_minutes(self) -> i64 {
        match self {
            Self::FiveMin => FIVE_MIN_SEC / 60,
            Self::FifteenMin => FIFTEEN_MIN_SEC / 60,
        }
    }

    pub fn try_from_interval_sec(sec: i64) -> Option<Self> {
        match sec {
            FIVE_MIN_SEC => Some(Self::FiveMin),
            FIFTEEN_MIN_SEC => Some(Self::FifteenMin),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Тип окна up/down по валюте (в [`crate::xframe::XFrame`] хранится как `i32`-дискриминант).
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

/// Исход токена up/down по валюте (в [`crate::xframe::XFrame`] хранится как `i32`-дискриминант).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
#[repr(i32)]
pub enum CurrencyUpDownOutcome {
    #[default]
    Down = 0,
    Up = 1,
}

impl CurrencyUpDownOutcome {
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

/// Класс сдвига / номер пятиминутки в 15m-блоке для up/down по валюте (в [`crate::xframe::XFrame`] хранится как `i32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
#[repr(i32)]
pub enum CurrencyUpDownDelayClass {
    /// 15m-рынок или первая пятиминутка в блоке.
    #[default]
    Aligned = 0,
    /// Вторая пятиминутка (+5 мин).
    Delay5Min = 1,
    /// Третья пятиминутка (+10 мин).
    Delay10Min = 2,
}

impl CurrencyUpDownDelayClass {
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
