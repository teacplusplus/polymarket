//! Redeem-X reconstructed buy+redeem rule.
//!
//! This is an explicit reconstruction of the observed public-profile pattern,
//! not a proof of the trader's private implementation. It uses the current-leg
//! book, opposite-leg `other_*` features, event timing, and already-open
//! redeem_x inventory to size the next buy as a leg-balance action.

use crate::account::LaneKey;
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{
    LanePositions, MAX_BET_FRACTION, MAX_POSITION_USD, MIN_POSITION_USD,
    POLYMARKET_CRYPTO_TAKER_FEE_RATE, StrictBook,
};
use crate::xframe::{BookLevel, SIZE, XFrame};
use std::collections::HashMap;

struct PositionBalance {
    own_shares: f64,
    own_entry_cost: f64,
    own_positions: usize,
    opposite_shares: f64,
    opposite_entry_cost: f64,
    opposite_positions: usize,
}

impl PositionBalance {
    fn new() -> Self {
        Self {
            own_shares: 0.0,
            own_entry_cost: 0.0,
            own_positions: 0,
            opposite_shares: 0.0,
            opposite_entry_cost: 0.0,
            opposite_positions: 0,
        }
    }

    fn has_inventory(&self) -> bool {
        self.own_positions > 0 || self.opposite_positions > 0
    }

    fn opposite_cost_per_share(&self) -> Option<f64> {
        if self.opposite_shares > 0.0 && self.opposite_entry_cost.is_finite() {
            Some(self.opposite_entry_cost / self.opposite_shares)
        } else {
            None
        }
    }
}

/// The public sample had a row-weighted median buy notional around $21.
const REDEEM_X_TARGET_USDC: f64 = 21.0;
/// Max all-in cost per matched UP+DOWN share. Below 1.0 is true redeem edge;
/// the small buffer covers book/fee approximation in historical frames.
const REDEEM_X_MAX_PAIR_COST_WITH_FEE: f64 = 1.0025;
/// Residual leg entry when the opposite leg is not currently cheap enough.
const REDEEM_X_RESIDUAL_MAX_ASK: f64 = 0.10;
/// Guard against chasing resolved/near-certain legs unless they pair cheaply.
const REDEEM_X_MAX_SINGLE_ASK: f64 = 0.90;
/// Minimum visible cheap notional relative to the intended slice.
const REDEEM_X_MIN_VISIBLE_DEPTH_RATIO: f64 = 0.75;
const REDEEM_X_BALANCE_EPS_SHARES: f64 = 1e-6;

pub(crate) async fn redeem_x_entry_size(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    bankroll: f64,
    event_end_ms: Option<i64>,
    positions_by_lane: &HashMap<LaneKey, LanePositions>,
    pending_close_by_lane: &HashMap<LaneKey, LanePositions>,
) -> Option<f64> {
    event_end_ms?;
    let interval = XFrameIntervalKind::from_i32(frame.xframe_interval_type)?;
    if frame.event_remaining_ms <= 0 {
        return None;
    }
    if frame.event_remaining_ms > redeem_x_max_entry_remaining_ms(interval) {
        return None;
    }

    let current_side = CurrencyUpDownOutcome::from_i32(frame.currency_up_down_outcome)?;
    let balance = redeem_x_position_balance_for_market(
        frame,
        current_side,
        positions_by_lane,
        pending_close_by_lane,
    )
    .await;

    let own_ask = best_ask_price(frame, strict_book)?;
    if own_ask <= 0.0 || own_ask > REDEEM_X_MAX_SINGLE_ASK {
        return None;
    }

    let opposite_ask = frame
        .other_book_ask_l1_price
        .filter(|p| p.is_finite() && *p > 0.0);
    let pair_allowed = opposite_ask.is_some_and(|other| {
        redeem_x_pair_cost_with_fee(own_ask, other) <= REDEEM_X_MAX_PAIR_COST_WITH_FEE
    });
    let residual_allowed = own_ask <= REDEEM_X_RESIDUAL_MAX_ASK;

    let target = redeem_x_target_usdc(bankroll)?;
    let visible_notional = ask_notional_up_to_price(frame, strict_book, own_ask + 0.02)?;

    let own_overweight = balance.own_shares > balance.opposite_shares + REDEEM_X_BALANCE_EPS_SHARES;
    if own_overweight {
        return None;
    }

    let shortfall_shares = (balance.opposite_shares - balance.own_shares).max(0.0);
    let desired_notional = if shortfall_shares > REDEEM_X_BALANCE_EPS_SHARES {
        if !redeem_x_balance_buy_allowed(own_ask, &balance) && !residual_allowed {
            return None;
        }
        notional_for_actual_shares(shortfall_shares, own_ask).min(target)
    } else {
        // Balanced book, or no book yet: start a new pair slice only when the
        // current pair is cheap enough, or when this is a tiny residual-tail leg.
        if !pair_allowed && !(residual_allowed && !balance.has_inventory()) {
            return None;
        }
        target
    };

    let intended = desired_notional
        .min(bankroll)
        .min(MAX_POSITION_USD)
        .max(0.0);
    if intended < MIN_POSITION_USD {
        return None;
    }
    if visible_notional + 1e-9 < intended * REDEEM_X_MIN_VISIBLE_DEPTH_RATIO {
        return None;
    }

    let size = intended
        .min(visible_notional)
        .min(bankroll)
        .min(MAX_POSITION_USD);
    (size >= MIN_POSITION_USD).then_some(size)
}

async fn redeem_x_position_balance_for_market(
    frame: &XFrame<SIZE>,
    current_side: CurrencyUpDownOutcome,
    positions_by_lane: &HashMap<LaneKey, LanePositions>,
    pending_close_by_lane: &HashMap<LaneKey, LanePositions>,
) -> PositionBalance {
    let mut balance = PositionBalance::new();
    add_redeem_x_position_balance(frame, current_side, positions_by_lane, &mut balance).await;
    add_redeem_x_position_balance(frame, current_side, pending_close_by_lane, &mut balance).await;
    balance
}

async fn add_redeem_x_position_balance(
    frame: &XFrame<SIZE>,
    current_side: CurrencyUpDownOutcome,
    lanes: &HashMap<LaneKey, LanePositions>,
    balance: &mut PositionBalance,
) {
    for lane_positions in lanes.values() {
        for pos_arc in lane_positions.values() {
            let pos = pos_arc.read().await;
            if !pos.redeem_x {
                continue;
            }
            if pos.market_id.as_str() != frame.market_id.as_str() {
                continue;
            }
            if !(pos.shares_held > 0.0 && pos.position_size > 0.0) {
                continue;
            }
            match CurrencyUpDownOutcome::from_i32(pos.currency_up_down_outcome_at_open) {
                Some(side) if side == current_side => {
                    balance.own_shares += pos.shares_held;
                    balance.own_entry_cost += pos.position_size;
                    balance.own_positions += 1;
                }
                Some(_) => {
                    balance.opposite_shares += pos.shares_held;
                    balance.opposite_entry_cost += pos.position_size;
                    balance.opposite_positions += 1;
                }
                None => {}
            }
        }
    }
}

fn redeem_x_balance_buy_allowed(own_ask: f64, balance: &PositionBalance) -> bool {
    let Some(opposite_cost) = balance.opposite_cost_per_share() else {
        return false;
    };
    opposite_cost + taker_cost_per_actual_share(own_ask) <= REDEEM_X_MAX_PAIR_COST_WITH_FEE
}

fn redeem_x_max_entry_remaining_ms(interval: XFrameIntervalKind) -> i64 {
    match interval {
        XFrameIntervalKind::FiveMin => 5 * 60 * 1_000,
        XFrameIntervalKind::FifteenMin => 15 * 60 * 1_000,
    }
}

fn redeem_x_pair_cost_with_fee(own_ask: f64, opposite_ask: f64) -> f64 {
    taker_cost_per_actual_share(own_ask) + taker_cost_per_actual_share(opposite_ask)
}

fn taker_share_factor(price: f64) -> f64 {
    (1.0 - POLYMARKET_CRYPTO_TAKER_FEE_RATE * (1.0 - price)).max(1e-9)
}

fn taker_cost_per_actual_share(price: f64) -> f64 {
    price / taker_share_factor(price)
}

fn notional_for_actual_shares(shares: f64, price: f64) -> f64 {
    shares * taker_cost_per_actual_share(price)
}

fn redeem_x_target_usdc(bankroll: f64) -> Option<f64> {
    if !(bankroll > 0.0 && bankroll.is_finite()) {
        return None;
    }
    let target = REDEEM_X_TARGET_USDC
        .min(bankroll * MAX_BET_FRACTION)
        .min(MAX_POSITION_USD)
        .max(MIN_POSITION_USD);
    (target <= bankroll + 1e-9).then_some(target)
}

fn best_ask_price(frame: &XFrame<SIZE>, strict_book: Option<&StrictBook>) -> Option<f64> {
    if let Some(book) = strict_book {
        return book
            .asks
            .iter()
            .find(|level| level.price > 0.0 && level.size > 0.0)
            .map(|level| level.price);
    }
    ask_levels(frame, strict_book)
        .into_iter()
        .find(|level| level.price > 0.0 && level.size > 0.0)
        .map(|level| level.price)
}

fn ask_notional_up_to_price(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    max_price: f64,
) -> Option<f64> {
    let notional: f64 = ask_levels(frame, strict_book)
        .into_iter()
        .filter(|level| level.price > 0.0 && level.price <= max_price && level.size > 0.0)
        .map(|level| level.price * level.size)
        .sum();
    (notional > 0.0).then_some(notional)
}

fn ask_levels(frame: &XFrame<SIZE>, strict_book: Option<&StrictBook>) -> Vec<BookLevel> {
    if let Some(book) = strict_book {
        return book.asks.clone();
    }
    if let Some(asks) = frame.book_asks.as_ref() {
        return asks.clone();
    }
    [
        (frame.book_ask_l1_price, frame.book_ask_l1_size),
        (frame.book_ask_l2_price, frame.book_ask_l2_size),
        (frame.book_ask_l3_price, frame.book_ask_l3_size),
    ]
    .into_iter()
    .filter_map(|(price, size)| {
        Some(BookLevel {
            price: price?,
            size: size?,
        })
    })
    .collect()
}
