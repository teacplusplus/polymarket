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
use crate::history_sim::{LanePositions, MAX_POSITION_USD, MIN_POSITION_USD, StrictBook};
use crate::xframe::{SIZE, XFrame};
use XFrameIntervalKind::FiveMin;
use std::collections::HashMap;

// --- Параметры входа (направленный momentum-maker) ------------------------------------

/// Ценовая полоса ноги (maker встаёт на best_bid): не котируем пыль / уже разрешённый исход.
const REDEEM_X_MIN_PRICE: f64 = 0.02;
const REDEEM_X_MAX_PRICE: f64 = 0.98;
/// Минимальный интервал между покупками в одном рынке: не чаще раза в N мс.
const REDEEM_X_MIN_REBUY_INTERVAL_MS: Option<i64> = None;
/// Минимальная глубина best_bid на входе относительно текущего клипа.
const REDEEM_X_MIN_BID_SIZE_CLIPS: f64 = 2.0;
/// Абсолютная минимальная глубина best_bid для BTC 5m (tail: p50 ≈ 308, p40≈200).
const REDEEM_X_MIN_BID_SIZE_SHARES_BTC_5M: f64 = 50.0;
/// Ступеньки потолка цены пары по вероятности ФАВОРИТА `fav = max(prob, other_prob)` —
/// «Вариант А», откалиброван под p90-envelope принятых ботом 2 пар (BTC 5m): чем увереннее
/// фаворит, тем выше допускается пара, т.к. направленный излишок на тяжёлой ноге окупает
/// переплату. Формат: `(верхняя граница fav [исключительно], потолок пары)`, отсортировано
/// по возрастанию. Последний бакет ловит `fav → 1.0`. См. [`redeem_x_max_pair_price`].
const REDEEM_X_PAIR_CAP_BY_FAV_PROB: &[(f64, f64)] = &[
    (0.60, 1.01),
    (0.70, 1.02),
    (0.80, 1.03),
    (0.90, 1.04),
    (1.00, 1.15),
];
/// Абсолютный потолок суммарной экспозиции (потраченных USDC по ОБЕИМ ногам) в ОДНОМ рынке.
/// Считается по фактическому филлу (`own_cost + other_cost`); если добавление клипа выведет
/// сумму за порог — новый вход в этот рынок блокируется. Ограничивает максимальный убыток
/// одного 5m-окна независимо от того, угадали ли мы сторону: у бота такого кэпа нет, и на
/// whipsaw-часах он грузил до ~$12.7k в одно окно с разворотом тяжёлой ноги → −$2.8k за окно
/// (BTC 5m 2026-07-03 02:45 UTC). `None` — кэп выключен. Значение подобрано под
/// `INITIAL_BANKROLL≈500` (в healthy-прогоне медиана оборота/рынок ≈ $75, max ≈ $200).
const REDEEM_X_MAX_MARKET_EXPOSURE_USD: Option<f64> = Some(300.0);
/// Потолок вложенных USDC в ОДИНОКУЮ ногу (партнёр ещё пуст, `other_shares == 0`). Пока
/// `own_cost < X` — андердогу разрешено набирать дешёвую базу в ожидании партнёра; как только
/// вложено ≥ X, докорм соло-ноги блокируется (см. solo-leg gate). Ограничивает максимальный
/// директональный убыток окна, где партнёр так и не встал (цена убежала, пара > cap): вместо
/// слива всей ноги (−$64 на 0x321edc92) теряем не больше ~X. `0.0` ⇒ открывается ровно один клип.
const REDEEM_X_SOLO_LEG_MAX_USD: f64 = 10.0;
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
    // Один проход по позициям: шеры текущей ноги, шеры other-ноги того же рынка (для
    // пропорции сторон) и мс с последней приземлившейся покупки (троттлинг).
    let (own_shares, own_cost, other_shares, other_cost, ms_since_last_buy) =
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
    let min_bid_size =
        (REDEEM_X_MIN_BID_SIZE_CLIPS * clip).max(REDEEM_X_MIN_BID_SIZE_SHARES_BTC_5M);
    if best_bid_size < min_bid_size {
        return None;
    }

    // (2) Балансировка ног по ЧИСЛУ ШЕРОВ + потолок цены пары. Прибыль у бота 2 строится на
    // ПАРЕ, купленной maker'ом ниже $1 и погашенной в $1 на резолюции; чтобы это работало
    // рыночно-нейтрально, обе ноги держим примерно равными (бот 2: медиана перекоса ≈ 1.22,
    // ≤2.0x у ~90% рынков, доля шеров на выигравшей ноге ≈ 50.8%). Правила:
    //   * первый клип рынка (ОБЕ ноги пусты) открывает ТОЛЬКО андердог (prob < other_prob):
    //     воркеры up/down спавнятся параллельно (real_sim::LANE_FRAME_ROUTES), иначе на старте
    //     открылись бы СРАЗУ обе; по prob квалифицируется ровно одна нога (устойчиво к гонке);
    //   * пока ПАРТНЁР ПУСТ (other=0), андердог набирает соло-ногу лишь до `REDEEM_X_SOLO_LEG_MAX_USD`
    //     по вложенным USDC; дальше докорм соло-ноги блокируется (см. solo-leg gate ниже) — без
    //     партнёра это направленная ставка, которая при разбегании цены сливала ногу целиком,
    //     поэтому директональный риск окна ограничен ~$X, а не всей ногой; ждём вторую ногу;
    //   * при непустой паре разрешаем усреднение вниз только для лёгкой/равной ноги, либо клип,
    //     который улучшает worst-case redemption и не пробивает потолок цены пары; тяжёлую ногу
    //     без такого edge не докармливаем;
    //   * при непустой паре — ещё и потолок цены пары [`redeem_x_max_pair_price`]: не
    //     переплачиваем за матч сверх кэпа, кроме клипов, что усредняют пару ВНИЗ.
    // Вероятности ног: implied prob из фрейма, fallback — по цене best_bid (prob своей ноги
    // ≈ maker_price, второй ≈ 1 − maker_price).
    let prob = frame
        .currency_implied_prob
        .filter(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0)
        .unwrap_or(maker_price.clamp(0.0, 1.0));
    let other_prob = frame
        .other_currency_implied_prob
        .filter(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0)
        .unwrap_or((1.0 - prob).clamp(0.0, 1.0));

    let clip_cost = clip * maker_price;
    if other_shares > 0.0 {
        let max_pair_price = redeem_x_max_pair_price(prob, other_prob);
        let other_avg = other_cost / other_shares;
        let own_avg_after = (own_cost + clip_cost) / (own_shares + clip);
        let projected_pair_price = own_avg_after + other_avg;
        // Старая цена пары до клипа — только если у своей ноги уже есть шеры (иначе пары нет,
        // это первый клип ноги, и «усреднения вниз» быть не может).
        let old_pair_price = (own_shares > 0.0).then(|| own_cost / own_shares + other_avg);
        let lowers_pair = old_pair_price.is_some_and(|old| projected_pair_price < old);

        // Баланс ног ВЕСЬ период + правильное НАПРАВЛЕНИЕ перекоса (как у бота 2: тяжёлая нога =
        // фаворит в 63%, побеждает в 62%). Клип проходит без проверки max-cap, если ЛИБО:
        //   1) `lowers_pair && own_shares <= other_shares` — усреднение вниз, но ТОЛЬКО пока нога
        //      не тяжелее другой (до паритета). Так андердог не может разогнаться за паритет
        //      (в v8 без этого ограничения проигравший андердог раздулся до 118:32, ratio 3.62);
        //   2) `projected_pair_price < 1.0 && prob >= other_prob` — ТЕКУЩИЙ ФАВОРИТ добирает, пока
        //      пара остаётся ниже $1 (матч-часть гарантированно в плюсе). Это доп. путь набора,
        //      которого у андердога нет, → тяжёлой становится нога вероятного победителя.
        // Всё остальное — только если клип держит пару под потолком И это нога ФАВОРИТА
        // (prob >= other_prob). Это единственный путь добора выше $1
        // (cap растёт с уверенностью фаворита до 1.15 — так же, как бот 2: он уводил пару за 1.0
        // тем охотнее, чем выше prob фаворита, median 0.795 vs 0.665; от времени до конца это НЕ
        // зависит). Догон ОТСТАЮЩЕГО андердога выше $1 здесь режется — андердогу остаётся только
        // усреднение вниз (ветка 1), что и есть его поведение у бота 2 при паре>1.0 (prob≈0.245).
        let blocked_reason = if lowers_pair && own_shares <= other_shares {
            None
        } else if projected_pair_price < max_pair_price && prob >= other_prob {
            None
        } else {
            Some("pair over cap")
        };
        if let Some(reason) = blocked_reason {
            crate::tee_println!(
                "[redeem_x] skip pair gate ({}) market_id={} asset_id={} prob={:.3} other_prob={:.3} own_avg_after={:.4} other_avg={:.4} pair={:.4} old_pair={:?} cap={:.4} (own_sh={:.2} other_sh={:.2} clip={:.2} price={:.4})",
                reason,
                frame.market_id,
                frame.asset_id,
                prob,
                other_prob,
                own_avg_after,
                other_avg,
                projected_pair_price,
                old_pair_price,
                max_pair_price,
                own_shares,
                other_shares,
                clip,
                maker_price,
            );
            return None;
        }
    } else if own_cost < REDEEM_X_SOLO_LEG_MAX_USD {
        // Партнёр ещё пуст (other_shares == 0), а в свою ногу вложено < REDEEM_X_SOLO_LEG_MAX_USD:
        // фаза открытия/набора дешёвой базы. Воркеры up/down спавнятся параллельно на все лейны
        // (см. real_sim::LANE_FRAME_ROUTES), поэтому без гейта на старте открылись бы СРАЗУ обе
        // ноги. Открываем/добираем рынок только ДЕШЁВОЙ стороной — андердогом (prob < other_prob):
        // квалифицируется ровно одна нога, устойчиво к гонке (фаворит режется по prob, а не по
        // other_shares). Бот так и делает: первый клип в ~90% — дешёвая/underdog-нога (в 15m-примере
        // Down@47c). Набор соло-ноги ограничен $X по вложенным USDC — дальше ждём партнёра
        // (solo-leg gate ниже); партнёр откроется через логику пары (ветка выше), как только пара
        // влезет под cap. Так директональный риск окна без партнёра ограничен ~$X, а не всей ногой.
        if prob >= other_prob {
            crate::tee_println!(
                "[redeem_x] skip first-clip gate (favorite, wait for underdog) market_id={} asset_id={} prob={:.3} other_prob={:.3} (own_sh={:.2} other_sh={:.2} clip={:.2} price={:.4})",
                frame.market_id,
                frame.asset_id,
                prob,
                other_prob,
                own_shares,
                other_shares,
                clip,
                maker_price,
            );
            return None;
        }
        // андердог — разрешаем клип (открытие рынка или добор базы до $X; далее к гейтам ниже).
    } else {
        // Партнёр ПУСТ (other_shares == 0), а в свою ногу уже вложено ≥ REDEEM_X_SOLO_LEG_MAX_USD.
        // НЕ доливаем соло-ногу дальше: именно безпартнёрный разгон одной стороны слил всю ногу
        // (-64.73 на 0x321edc92 — андердог открылся у ~0.46, цена убежала, партнёр так и не встал,
        // т.к. пара пробивала cap, а мы догнали ногу до ~298 шт и потеряли её целиком). pair-логика
        // выше авторизует докорм только при НЕПУСТОМ партнёре (улучшение worst-case или усреднение
        // пары вниз) — до его появления держим набранную базу (≤ $X). Кэп экспозиции ($300) тут не
        // спасает: дешёвую ногу можно набрать на сотни штук в лимите, поэтому лимит именно по соло-USD.
        crate::tee_println!(
            "[redeem_x] skip solo-leg gate (own open, partner empty — wait for other leg) market_id={} asset_id={} prob={:.3} other_prob={:.3} (own_sh={:.2} other_sh={:.2} clip={:.2} price={:.4})",
            frame.market_id,
            frame.asset_id,
            prob,
            other_prob,
            own_shares,
            other_shares,
            clip,
            maker_price,
        );
        return None;
    }

    // (2b) Абсолютный кэп экспозиции на рынок. Суммарно потрачено по обеим ногам
    // (`own_cost + other_cost`, по фактическому филлу); если добавление клипа выведет за
    // порог — новый вход в этот рынок блокируем. Ограничивает максимальный убыток одного
    // окна при развороте тяжёлой ноги (см. BTC 5m 02:45 UTC у бота: $12.7k → −$2.8k).
    if let Some(max_market_exposure) = REDEEM_X_MAX_MARKET_EXPOSURE_USD {
        let current_exposure = own_cost + other_cost;
        let projected_exposure = current_exposure + clip_cost;
        if projected_exposure > max_market_exposure {
            crate::tee_println!(
                "[redeem_x] skip market-exposure gate market_id={} asset_id={} current={:.2} +clip={:.2} projected={:.2} > cap={:.2} (own_cost={:.2} sib_cost={:.2} clip={:.2} price={:.4})",
                frame.market_id,
                frame.asset_id,
                current_exposure,
                clip_cost,
                projected_exposure,
                max_market_exposure,
                own_cost,
                other_cost,
                clip,
                maker_price,
            );
            return None;
        }
    }

    // (3) Полный клип → нотинал USDC с потолками банка/позиции.
    let size = (clip_cost).min(MAX_POSITION_USD);
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

/// Потолок КОМБИНИРОВАННОЙ цены пары `own_avg + other_avg` в зависимости от вероятности
/// ФАВОРИТА `fav = max(prob, other_prob)` («Вариант А», см. [`REDEEM_X_PAIR_CAP_BY_FAV_PROB`]).
/// Пара UP+DOWN гасится в $1 на резолюции, поэтому сматченная часть дороже потолка —
/// структурный убыток; но чем увереннее фаворит, тем больше направленного излишка на тяжёлой
/// ноге окупает переплату, поэтому порог растёт с `fav`. Аргументы `prob` / `other_prob` —
/// implied prob текущей и противоположной ноги в [0..1].
fn redeem_x_max_pair_price(prob: f64, other_prob: f64) -> f64 {
    let fav = prob.max(other_prob).clamp(0.0, 1.0);
    REDEEM_X_PAIR_CAP_BY_FAV_PROB
        .iter()
        .find(|(hi, _)| fav < *hi)
        .map(|(_, cap)| *cap)
        .unwrap_or(1.0)
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
/// `(own_shares, own_cost, other_shares, other_cost, ms_since_last_buy)`:
///   * `own_shares` / `own_cost` — фактически удержанные шеры и потраченные USDC
///     (`position_size`) ТЕКУЩЕЙ ноги (тот же `asset_id`);
///   * `other_shares` / `other_cost` — то же для ПРОТИВОПОЛОЖНОЙ ноги (другой
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
    let mut other_shares = 0.0;
    let mut other_cost = 0.0;
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
                    .and_then(
                        |report| match (report.taking_amount, report.making_amount) {
                            (OrderAmount::Shares(shares), OrderAmount::UsdNotional(usd))
                                if shares.is_finite()
                                    && shares > 0.0
                                    && usd.is_finite()
                                    && usd >= 0.0 =>
                            {
                                Some((shares, usd))
                            }
                            _ => None,
                        },
                    );
                if let Some((shares, usd)) = filled {
                    if p.asset_id == frame.asset_id {
                        own_shares += shares;
                        own_cost += usd;
                    } else {
                        other_shares += shares;
                        other_cost += usd;
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
        other_shares,
        other_cost,
        ms_since_last_buy,
    )
}
