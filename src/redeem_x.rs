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
use crate::account_order::{OrderAmount, invoke_settlement_report};
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
/// Минимальный интервал между покупками в одном рынке: не чаще раза в N мс.
const REDEEM_X_MIN_REBUY_INTERVAL_MS: Option<i64> = None;
/// Минимальная глубина best_bid на входе относительно текущего клипа.
const REDEEM_X_MIN_BID_SIZE_CLIPS: f64 = 2.0;
/// Абсолютная минимальная глубина best_bid для BTC 5m (tail: p50 ≈ 308, p40≈200).
const REDEEM_X_MIN_BID_SIZE_SHARES_BTC_5M: f64 = 200.0;
/// Пропорция сторон: ОТСТАЮЩУЮ (проигрывающую сейчас) ногу держим не больше `ratio ×`
/// инвентаря ВЫИГРЫВАЮЩЕЙ (sibling) ноги того же рынка → у бота тяжёлая нога = будущий
/// победитель (BTC 5m lead:lag ≈ 8000:6000 = 0.75). Выигрывающая нога свободна (её потолок
/// — только банкролл), поэтому она набирается в большей доле. При нулевом лидере отстающая
/// нога заблокирована (сначала набираем текущего фаворита).
const REDEEM_X_LAG_TO_LEAD_RATIO: f64 = 0.75;
/// Потолок КОМБИНИРОВАННОЙ цены пары `own_avg + sibling_avg` (VWAP обеих ног рынка) ПОСЛЕ
/// добавления нового клипа. Пара UP+DOWN на резолюции платит ровно $1, поэтому набранная
/// дороже этого порога сматченная пара — структурный убыток на каждой шере (в 8h-прогоне
/// средняя цена пары была 1.12, до 1.34 → −155 USDC на парах). Гейт запрещает докупать
/// ногу, если это поднимет средневзвешенную цену пары выше порога; при пустой sibling-ноге
/// (пары ещё нет) не применяется. Небольшой допуск >1.0 оставляет место направленному
/// излишку на тяжёлой ноге, ради которого бот и торгует. `None` — гейт выключен (у самого
/// бота такого потолка нет: он набирает пары до ~1.57, медиана ~1.04).
const REDEEM_X_MAX_PAIR_PRICE: Option<f64> = None;
/// Fallback-«время сделки» maker-входа REDEEM_X, когда `open_buy_invoke.report.landed_at
/// == None` (см. [`redeem_x_leg_scan`]). Сам ордер теперь GTC — истечение делается явным
/// cancel'ом, а не GTD-`expiration`, поэтому TTL-константы для постановки больше нет.
pub(crate) const REDEEM_X_MAKER_1_EXPIRATION_MS: i64 = 1_000;
// --- Решение о входе ------------------------------------------------------------------

/// Правило входа REDEEM_X: **полный клип** `coin+period` без исторического regime-гейта.
///
/// `None` — не заходим: не BTC 5m, нет цены/вне полосы, нет нужной глубины best bid,
/// отстающая нога упёрлась в пропорцию к лидеру ([`REDEEM_X_LAG_TO_LEAD_RATIO`]), либо
/// размер ниже минимума по банку/позиции. Иначе — нотинал USDC для полного клипа.
/// Двухсторонний набор с креном к текущему победителю — повторяем пассивного MM-бота.
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
    // Один проход по позициям: шеры текущей ноги, шеры sibling-ноги того же рынка (для
    // пропорции сторон) и мс с последней приземлившейся покупки (троттлинг).
    let (own_shares, own_cost, sibling_shares, sibling_cost, ms_since_last_buy) =
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

    // (2) Пропорция сторон с креном к ТЕКУЩЕМУ победителю. Нога считается выигрывающей,
    // если её implied prob не ниже prob второй ноги (fallback — по цене best_bid: prob ноги
    // ≈ maker_price, prob второй ≈ 1 − maker_price). Выигрывающую не ограничиваем (её
    // потолок — только банкролл), а отстающую держим не больше `ratio ×` инвентаря sibling —
    // так тяжёлая нога (будущий победитель) набирается в большей доле, как у бота.
    let leg_prob = frame.currency_implied_prob.unwrap_or(maker_price);
    let other_prob = frame
        .other_currency_implied_prob
        .unwrap_or(1.0 - maker_price);
    let is_leading = leg_prob >= other_prob;
    if !is_leading && own_shares + clip > sibling_shares * REDEEM_X_LAG_TO_LEAD_RATIO {
        return None;
    }

    // (2a) Гейт комбинированной цены пары. Пара UP+DOWN гасится в $1 на резолюции, поэтому
    // сматченная часть, набранная дороже $1, — структурный убыток. Считаем VWAP пары ПОСЛЕ
    // добавления клипа: own_avg' = (own_cost + clip·price)/(own_shares + clip), sibling_avg =
    // sibling_cost/sibling_shares. Применяем только когда гейт включён и sibling-нога непустая
    // (иначе пары ещё нет). Если проекция цены пары выше потолка — не докупаем ногу.
    if let Some(max_pair_price) = REDEEM_X_MAX_PAIR_PRICE
        && sibling_shares > 0.0
    {
        let own_avg_after = (own_cost + clip * maker_price) / (own_shares + clip);
        let sibling_avg = sibling_cost / sibling_shares;
        let projected_pair_price = own_avg_after + sibling_avg;
        if projected_pair_price > max_pair_price {
            crate::tee_println!(
                "[redeem_x] skip pair-price gate market_id={} asset_id={} own_avg_after={:.4} sibling_avg={:.4} pair={:.4} > cap={:.4} (own_sh={:.2} sib_sh={:.2} clip={:.2} price={:.4})",
                frame.market_id,
                frame.asset_id,
                own_avg_after,
                sibling_avg,
                projected_pair_price,
                max_pair_price,
                own_shares,
                sibling_shares,
                clip,
                maker_price,
            );
            return None;
        }
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

/// Один проход по обоим bucket'ам по позициям ТЕКУЩЕГО рынка. Возвращает
/// `(own_shares, own_cost, sibling_shares, sibling_cost, ms_since_last_buy)`:
///   * `own_shares` / `own_cost` — фактически удержанные шеры и потраченные USDC
///     (`position_size`) ТЕКУЩЕЙ ноги (тот же `asset_id`);
///   * `sibling_shares` / `sibling_cost` — то же для ПРОТИВОПОЛОЖНОЙ ноги (другой
///     `asset_id`, тот же `market_id`) — база для пропорции сторон lag↔lead и для гейта
///     комбинированной цены пары (см. [`REDEEM_X_MAX_PAIR_PRICE`]);
///   * `ms_since_last_buy` — мс с последней **приземлившейся** покупки по `landed_at`
///     settled-отчёта `open_buy_invoke` (Some ⇔ success, включая partial; мок ставит
///     `landed_at = current_timestamp_ms()`). `None`, если ни одна ещё не приземлилась.
async fn redeem_x_leg_scan(
    buckets: [&HashMap<crate::account::LaneKey, LanePositions>; 2],
    frame: &XFrame<SIZE>,
) -> (f64, f64, f64, f64, Option<i64>) {
    let now_ms = crate::util::current_timestamp_ms();
    let mut own_shares = 0.0;
    let mut own_cost = 0.0;
    let mut sibling_shares = 0.0;
    let mut sibling_cost = 0.0;
    let mut ms_since_last_buy: Option<i64> = None;
    for by_lane in buckets {
        for lane_positions in by_lane.values() {
            for position in lane_positions.values() {
                let p = position.read().await;
                if p.market_id != frame.market_id {
                    continue;
                }
                // Инвентарь и стоимость берём из ФАКТИЧЕСКОГО отчёта об исполнении
                // открывающего BUY (`open_buy_invoke`), только при `success` и ненулевом
                // filled'е: до/без успешного исполнения `shares_held`/`position_size` — лишь
                // план (виртуальные значения на момент создания), в пропорцию сторон и в VWAP
                // пары их учитывать нельзя. BUY-отчёт: `taking_amount` = net shares (после
                // fee), `making_amount` = потраченные USDC.
                let filled = p
                    .open_buy_invoke
                    .as_ref()
                    .and_then(invoke_settlement_report)
                    .filter(|report| report.success)
                    .and_then(|report| match (report.taking_amount, report.making_amount) {
                        (OrderAmount::Shares(shares), OrderAmount::UsdNotional(usd))
                            if shares.is_finite()
                                && shares > 0.0
                                && usd.is_finite()
                                && usd >= 0.0 =>
                        {
                            Some((shares, usd))
                        }
                        _ => None,
                    });
                if let Some((shares, usd)) = filled {
                    if p.asset_id == frame.asset_id {
                        own_shares += shares;
                        own_cost += usd;
                    } else {
                        sibling_shares += shares;
                        sibling_cost += usd;
                    }
                }
                let landed_at = p
                    .open_buy_invoke
                    .as_ref()
                    .and_then(invoke_settlement_report)
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
    (
        own_shares,
        own_cost,
        sibling_shares,
        sibling_cost,
        ms_since_last_buy,
    )
}
