//! Redeem-X — реконструкция публичного momentum-maker'а по tail-отчёту
//! (held-to-resolution `PnL = winning_shares·$1 − bought`, совпал с реальным до цента).
//!
//!   * Не redeem-арбитраж: пара UP+DOWN в медиане >$1 → на самих парах он **в минусе**.
//!   * Прибыль — с непарного направленного излишка на «тяжёлой» (лидирующей) ноге;
//!     тяжёлая нога = победитель в ~75% (BTC 5m) / ~90% (BTC 15m, ETH 5m).
//!   * Исполняется пассивным maker'ом **фиксированным клипом по `coin+period`**
//!     (ETH 5m≈5, BTC 5m≈100, BTC 15m≈20 шер). Размер ОДНОГО ордера инвариантен к цене
//!     и ко второй ноге (~91% ордеров — полный клип); плечо набирается числом клипов
//!     лесенкой, а перекос ног возникает из выбора ноги, не из сжатия ордера.
//!
//! Поэтому сайзинг здесь — **полный клип** `coin+period` (см. [`redeem_x_clip_shares`]),
//! гейты цены/времени и асимметричный потолок инвентаря lead/lag (мягкий крен к лидеру).

use crate::account::SharedAccount;
use crate::constants::XFrameIntervalKind;
use crate::history_sim::{
    LanePositions, MAX_POSITION_USD, MIN_POSITION_USD, StrictBook,
};
use crate::xframe::{SIZE, XFrame};
use std::collections::HashMap;
use XFrameIntervalKind::FiveMin;

// --- Параметры входа (направленный momentum-maker) ------------------------------------

/// Ценовая полоса ноги (maker встаёт на best_bid): не котируем пыль / уже разрешённый исход.
const REDEEM_X_MIN_PRICE: f64 = 0.02;
const REDEEM_X_MAX_PRICE: f64 = 0.98;
/// Порог implied prob, выше которого нога — лидирующая (фаворит).
const REDEEM_X_LEAD_PROB: f64 = 0.50;
/// Абсолютные потолки inventory (shares) для BTC 5m из профиля бота:
/// лидер держим шире, отстающую ногу уже.
const REDEEM_X_MAX_LEAD_SHARES_BTC_5M: f64 = 8_000.0;
const REDEEM_X_MAX_LAG_SHARES_BTC_5M: f64 = 6_000.0;
/// Минимальный интервал между покупками в одном рынке: не чаще раза в N мс.
const REDEEM_X_MIN_REBUY_INTERVAL_MS: Option<i64> = None;
/// Минимальная глубина best_bid на входе относительно текущего клипа.
const REDEEM_X_MIN_BID_SIZE_CLIPS: f64 = 2.0;
/// Абсолютная минимальная глубина best_bid для BTC 5m (tail: p50 ≈ 308, p40≈200).
const REDEEM_X_MIN_BID_SIZE_SHARES_BTC_5M: f64 = 200.0;
/// Fallback-«время сделки» maker-входа REDEEM_X, когда `open_buy_invoke.report.landed_at
/// == None` (см. [`redeem_x_leg_scan`]). Сам ордер теперь GTC — истечение делается явным
/// cancel'ом, а не GTD-`expiration`, поэтому TTL-константы для постановки больше нет.
pub(crate) const REDEEM_X_MAKER_1_EXPIRATION_MS: i64 = 1_000;
// --- Решение о входе ------------------------------------------------------------------

/// Правило входа REDEEM_X: **полный клип** `coin+period` без исторического regime-гейта.
///
/// `None` — не заходим: не BTC 5m, нет цены/вне полосы, нет нужной глубины best bid,
/// нога уперлась в потолок инвентаря, либо размер ниже минимума по банку/позиции.
/// Иначе — нотинал USDC для полного клипа.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn redeem_x_entry_size(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    available_bankroll: f64,
    currency: &str,
    _event_end_ms: Option<i64>,
    positions_by_lane: &HashMap<crate::account::LaneKey, LanePositions>,
    pending_close_by_lane: &HashMap<crate::account::LaneKey, LanePositions>,
    _account: Option<&SharedAccount>,
) -> Option<f64> {
    let interval = XFrameIntervalKind::from_i32(frame.xframe_interval_type)?;
    // REDEEM_X оставляем только на BTC 5m.
    if !currency.eq_ignore_ascii_case("btc") || interval != XFrameIntervalKind::FiveMin {
        return None;
    }
    // Один проход по позициям рынка: шеры текущей ноги + мс с последней приземлившейся покупки.
    let (own_shares, ms_since_last_buy) =
        redeem_x_leg_scan([positions_by_lane, pending_close_by_lane], frame).await;
    // Троттлинг по времени: не чаще раза в N мс с последней покупки в этом рынке.
    if let (Some(since_ms), Some(min_rebuy_interval_ms)) =
        (ms_since_last_buy, REDEEM_X_MIN_REBUY_INTERVAL_MS)
        && since_ms < min_rebuy_interval_ms
    {
        return None;
    }

    // (1) Фиксированный клип coin+period; maker встаёт на best_bid (= цена shares↔USDC).
    let clip = redeem_x_clip_shares(currency, interval)?;
    let maker_price = strict_book
        .and_then(crate::account_order::best_bid_strict)
        .or(frame.book_bid_l1_price)
        .filter(|p| p.is_finite() && *p > 0.0)?;
    if !(REDEEM_X_MIN_PRICE..=REDEEM_X_MAX_PRICE).contains(&maker_price) {
        return None;
    }

    // (1a) Гейт глубины best_bid по профилю бота BTC 5m: требуем абсолютный floor и
    // клип-относительный минимум одновременно.
    let best_bid_size = strict_book
        .and_then(crate::account_order::best_bid_size_strict)
        .or(frame.book_bid_l1_size)
        .filter(|s| s.is_finite() && *s > 0.0)?;
    let min_bid_size = (REDEEM_X_MIN_BID_SIZE_CLIPS * clip).max(REDEEM_X_MIN_BID_SIZE_SHARES_BTC_5M);
    if best_bid_size < min_bid_size {
        return None;
    }

    // (2) Асимметричный потолок инвентаря ноги в абсолютных shares (ботоподобный профиль).
    let leg_prob = frame.currency_implied_prob.unwrap_or(maker_price);
    let leg_cap = if leg_prob >= REDEEM_X_LEAD_PROB {
        REDEEM_X_MAX_LEAD_SHARES_BTC_5M
    } else {
        REDEEM_X_MAX_LAG_SHARES_BTC_5M
    };
    if own_shares + clip > leg_cap {
        return None;
    }

    // (3) Полный клип → нотинал USDC с потолками банка/позиции.
    let size = (clip * maker_price).min(MAX_POSITION_USD);
    if size < MIN_POSITION_USD {
        return None;
    }
    // (3a) Учёт доступного банкролла: `available_bankroll` = bankroll − весь залоченный
    // капитал (см. вызов из [`crate::history_sim::buy_gate`]). Не открываем клип, если на
    // него не хватает свободных средств — иначе на реальном сабмите ловим отказы CLOB, а
    // в mock оборот бесконтрольно превышает банк. Стоп набора при исчерпании банка.
    if size > available_bankroll {
        return None;
    }
    Some(size)
}

/// Фиксированный клип ОДНОГО лимитного ордера по `coin+period` (медиана размера ордера из
/// tail-отчёта, инвариантна к цене). Неизвестная комбинация → `panic!`.
fn redeem_x_clip_shares(currency: &str, interval: XFrameIntervalKind) -> Option<f64> {  
    Some(match (currency.to_ascii_lowercase().as_str(), interval) {
        ("btc", FiveMin) => 5.0,
        (coin, interval) => panic!(
            "redeem_x_clip_shares: unsupported coin+period: coin={coin}, interval={interval:?}"
        ),
    })
}

/// Один проход по обоим bucket'ам для рынка `frame.market_id`: возвращает
/// `(own_shares, ms_since_last_buy)`:
///   * `own_shares` — суммарные **фактически удержанные** шеры ТЕКУЩЕЙ ноги
///     (`shares_held` обновляется после fill'а → корректно и для частичного исполнения);
///   * `ms_since_last_buy` — мс с последней **приземлившейся** покупки по `landed_at`
///     settled-отчёта `open_buy_invoke` (Some ⇔ success, включая partial; мок ставит
///     `landed_at = current_timestamp_ms()` — поэтому единый wall-clock и для моков).
///     `None`, если ни одна покупка ещё не приземлилась.
async fn redeem_x_leg_scan(
    buckets: [&HashMap<crate::account::LaneKey, LanePositions>; 2],
    frame: &XFrame<SIZE>,
) -> (f64, Option<i64>) {
    let now_ms = crate::util::current_timestamp_ms();
    let mut own_shares = 0.0;
    let mut ms_since_last_buy: Option<i64> = None;
    for by_lane in buckets {
        for lane_positions in by_lane.values() {
            for position in lane_positions.values() {
                let p = position.read().await;
                if p.market_id != frame.market_id {
                    continue;
                }
                if p.asset_id == frame.asset_id {
                    own_shares += p.shares_held;
                }
                let landed_at = p
                    .open_buy_invoke
                    .as_ref()
                    .and_then(crate::account_order::invoke_settlement_report)
                    .and_then(|report| report.landed_at);
                // Если landed_at ещё нет, считаем «время сделки» как момент входа +
                // [`REDEEM_X_MAKER_1_EXPIRATION_MS`] (грубая оценка задержки до fill'а).
                let fallback_landed_at = now_ms
                    .saturating_sub(
                        p.event_remaining_ms_at_open
                            .saturating_sub(frame.event_remaining_ms),
                    )
                    .saturating_add(REDEEM_X_MAKER_1_EXPIRATION_MS);
                let effective_landed_at = landed_at.unwrap_or(fallback_landed_at);
                let since = now_ms - effective_landed_at;
                if since >= 0 {
                    ms_since_last_buy = Some(ms_since_last_buy.map_or(since, |m| m.min(since)));
                }
            }
        }
    }
    (own_shares, ms_since_last_buy)
}
