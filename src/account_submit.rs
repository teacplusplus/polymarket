//! Submit-режим [`crate::main::AppMode::RealSimWithSubmit`]: реальная отправка
//! BUY/SELL/cancel ордеров на Polymarket CLOB через [`crate::account_order`]
//! и асинхронная верификация через `client.order(...)` (fallback к WS, см.
//! [`crate::account_ws`]).
//!
//! # Архитектура
//!
//! Все async-таски модуля идут через `tokio::spawn` без удержания глобальных
//! локов `Account.positions`/`closing` через сетевые вызовы. Дедуп
//! постановки/отмены — через локальные флаги
//! [`crate::history_sim::OpenPosition::tp_placement_attempted`] /
//! [`crate::history_sim::ClosingPosition::close_placement_attempted`], которые
//! атомарно выставляются под коротким write-локом ДО HTTP-вызова.
//!
//! # In-flight идентификация позиций
//!
//! Никаких synthetic-id'шников. Когда [`crate::history_sim::try_open_position`]
//! пушит позицию в `Account.positions` (для лочки `entry_cost` в расчёте
//! available bankroll), `OpenPosition.open_order_id` остаётся `None`, а
//! `open_status = PendingOpen` — это и есть индикатор «отправили, ждём
//! подтверждения CLOB». [`spawn_open_buy_taker`] получает в параметрах сам
//! Arc на эту запись (см. [`crate::history_sim::SharedOpenPosition`]) и
//! пишет real `order_id` напрямую через inner-RwLock. Аналогично с
//! [`crate::history_sim::ClosingPosition`] / [`spawn_close_via_taker`].
//!
//! # Polling-verify
//!
//! WS-канал может отстать или упасть (см. watchdog в [`crate::account_ws`]).
//! Чтобы не зависнуть в `PendingOpen`/`PendingClose`, после получения real
//! `order_id` спавним polling-таску: каждые [`POLL_INTERVAL_SEC`] секунд
//! `client.order(order_id)` до терминального статуса
//! ([`OrderStatusType::Matched`] / `Canceled` / etc.) или таймаута
//! [`POLL_TIMEOUT_SEC`]. На терминальном статусе таска вызывает те же
//! апдейты, что и WS-колбек (через
//! [`apply_order_status_from_polling`]).

use crate::account::SharedAccount;
use crate::account_order::{
    cancel_order_on_clob, post_order_on_clob, CancelOrderRequest, OrderAmount, OrderRole,
    PostOrderRequest,
};
use crate::history_sim::{
    ClosingPositionStatus, OpenPositionStatus, SharedClosingPosition, SharedOpenPosition,
    StrictBook, SIM_MAX_SLIPPAGE_FROM_L1_PCT,
};
use crate::xframe::Y_TRAIN_TAKE_PROFIT_PP;
use polymarket_client_sdk::clob::types::request::TradesRequest;
use polymarket_client_sdk::clob::types::{OrderStatusType, Side};
use std::sync::Arc;
use std::time::Duration;

/// HTTP-таймаут одного `POST /order` / `DELETE /order` в submit-флоу.
/// Сетевые вызовы идут в `tokio::spawn`, так что блокировки лейн-воркера нет;
/// но per-call таймаут нужен, чтобы умершие сокеты не съедали task'и часами.
const ORDER_HTTP_TIMEOUT_SEC: u64 = 10;

/// Период polling'а статуса ордера в [`spawn_polling_verify`] до терминального
/// статуса. WS — основной канал, polling — fallback при тишине.
const POLL_INTERVAL_SEC: u64 = 3;

/// Хард-таймаут polling-верификации: после N секунд без терминального статуса
/// бросаем таску и оставляем позицию в pending — следующий тик `manage_positions`
/// или ручная диагностика разрулит. Не делаем `OpenFailed` автоматически:
/// возможно, ордер реально лежит live и заполнится позже.
const POLL_TIMEOUT_SEC: u64 = 30;

/// Максимум попыток отправки SELL taker в [`spawn_close_via_taker`] (включая
/// первую). На исчерпании всех попыток `ClosingPosition.close_status` ставим
/// в `CloseFailed`, и следующий тик `manage_positions` после cleanup'а вновь
/// зайдёт в `sell_gate` и попробует — то есть retry-уровней два:
/// in-task (быстрый, exp-backoff) и tick-based (медленный, fallback).
const SELL_TAKER_MAX_ATTEMPTS: u32 = 3;

/// Базовая задержка перед 1-м retry SELL-taker'а; следующие удваиваются:
/// `500ms → 1s → 2s → …` (exp-backoff). С [`SELL_TAKER_MAX_ATTEMPTS`]=3
/// получаем ожидания 500ms и 1s между попытками.
const SELL_TAKER_RETRY_INITIAL_MS: u64 = 500;

/// Спавнит таск отправки **BUY taker** на CLOB. Размер — в USDC
/// (`amount=UsdNotional`).
///
/// # Worst-acceptable price (`price` / `max_slippage_pp`)
///
/// Поведение зависит от наличия `decision_price`:
/// - `Some(p)` — explicit worst-acceptable, **зафиксированный на момент
///   decision-time** в [`crate::history_sim::try_open_position`] из L1 ask
///   tick'ового [`StrictBook`] + [`SIM_MAX_SLIPPAGE_FROM_L1_PCT`]. Передаётся
///   в `PostOrderRequest::price`; SDK использует его как worst-acceptable
///   и игнорирует `max_slippage_pp` (см. контракт в
///   [`crate::account_order::PostOrderRequest::price`]). Этот вариант
///   предпочтителен: cap считается от той же L1, на которой `buy_gate` принял
///   решение, без дополнительного `GET /book` внутри SDK.
/// - `None` — fallback: tick'овый snapshot был недоступен (HTTP-fail или
///   ws-lag в [`crate::real_sim`]). Тогда `price=None`,
///   `max_slippage_pp=Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT)` — SDK сам сделает
///   `GET /book` и посчитает cap от свежей L1.
///
/// Поле `strict_book` в `PostOrderRequest` тоже пробрасываем
/// (`decision_book`) — оно используется SDK только в fallback-ветке (price=None)
/// для расчёта slippage cap'а **без HTTP** ([`crate::account_order`]:
/// `compute_taker_cap_price`); если есть и `decision_price` и `decision_book`,
/// SDK видит первый и игнорирует второй (harmless).
///
/// # Идентификация in-flight позиции
///
/// Через переданный `pos_arc` (см. [`SharedOpenPosition`]); `open_order_id`
/// пишется напрямую через inner-RwLock после получения real `order_id` от
/// CLOB.
///
/// # Семантика терминальных статусов
///
/// На успехе: записывает real `order_id` в `pos_arc.open_order_id`. Дальше
/// real-time апдейты летят через user-WS
/// ([`crate::account_ws::apply_user_ws_event_value`]).
///
/// На ошибке (HTTP падение или CLOB rejection): позиция помечается
/// `OpenPositionStatus::OpenFailed`. `manage_positions` (или cleanup в
/// submit-режиме) уберёт её и вернёт `entry_cost` в свободный bankroll
/// (фактически ничего не списано).
///
/// **Не блокирует** вызывающего: вся работа в `tokio::spawn`. После
/// HTTP-ответа спавнит [`spawn_polling_verify_open`] для fallback'а на
/// случай, если WS прокатится и не доедет.
/// [проверено]
pub(crate) fn spawn_open_buy_taker(
    account: SharedAccount,
    pos_arc: SharedOpenPosition,
    price: Option<f64>,
    strict_book: Option<StrictBook>,
) {
    tokio::spawn(async move {
        // Snapshot из позиции — `asset_id` и `entry_cost` (он же
        // `position_size_usd` для `UsdNotional`) уже лежат в `pos_arc`,
        // отдельные параметры передавать не нужно. Берём под коротким
        // read-локом, дальше HTTP идёт без локов.
        let (pos_id, asset_id, position_size_usd) = {
            let pos = pos_arc.read().await;
            (pos.id.clone(), pos.asset_id.clone(), pos.entry_cost)
        };
        // explicit price → max_slippage_pp игнорируется SDK; иначе — slippage-cap-флоу.
        let max_slippage_pp = if price.is_some() {
            None
        } else {
            Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT)
        };
        let request = PostOrderRequest {
            asset_id: asset_id.clone(),
            side: Side::Buy,
            role: OrderRole::Taker,
            amount: OrderAmount::UsdNotional(position_size_usd),
            price,
            max_slippage_pp,
            expiration: None,
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            strict_book,
        };
        match post_order_on_clob(&account, request).await {
            Ok(result) => {
                if !result.success {
                    crate::tee_eprintln!(
                        "[account_submit] BUY taker отвергнут CLOB: pos_id={pos_id}, error_msg={:?}, status={:?}, order_id={}",
                        result.error_msg, result.status, result.order_id,
                    );
                    pos_arc.write().await.open_status = OpenPositionStatus::OpenFailed;
                    return;
                }
                let real_order_id = result.order_id.clone();
                pos_arc.write().await.open_order_id = Some(real_order_id.clone());
                crate::tee_println!(
                    "[account_submit] BUY taker принят: pos_id={pos_id}, order_id={real_order_id}, status={:?}",
                    result.status,
                );
                spawn_polling_verify_open(account.clone(), pos_arc.clone());
            }
            Err(err) => {
                crate::tee_eprintln!(
                    "[account_submit] BUY taker упал: pos_id={pos_id}, asset={asset_id}: {err:#}"
                );
                pos_arc.write().await.open_status = OpenPositionStatus::OpenFailed;
            }
        }
    });
}


/// Идемпотентная постановка **maker SELL TP-лимитки** для уже открытой позиции.
/// Цена — точно `pos.buy_price + Y_TRAIN_TAKE_PROFIT_PP` (без slippage cap'а;
/// см. ответ пользователя в дискуссии настройки модуля).
///
/// Идентификация позиции — через переданный `pos_arc` (см.
/// [`SharedOpenPosition`]); никаких поисков по `open_order_id`. Caller (WS-колбек
/// в [`crate::account_ws::apply_user_ws_event_value`] / polling-verify в
/// [`apply_order_status_from_polling`]) уже знает, какая
/// именно позиция перешла `PendingOpen → Open`, и передаёт её Arc напрямую.
///
/// Дедуп: атомарно проверяет + взводит
/// [`crate::history_sim::OpenPosition::tp_placement_attempted`] под inner-write
/// `pos_arc`; если флаг уже `true` или `tp_order_id` уже `Some(_)` —
/// выходит без HTTP. Гонка между WS-колбеком и [`spawn_polling_verify_open`]
/// поэтому безопасна.
///
/// **Локи и сеть:** под inner-write одной позиции только проверка/взвод флага
/// и снятие snapshot'а параметров; HTTP-вызов идёт **без** локов; запись
/// `tp_order_id` — снова под коротким inner-локом.
/// [проверено]
pub async fn try_place_tp_maker(account: SharedAccount, pos_arc: SharedOpenPosition) {
    // Этап 1: snapshot + взвод флага под inner-write одной позиции.
    // Кортеж: `(pos_id, asset_id, shares, tp_price, open_order_id)`. Поле
    // `open_order_id` — только для логов (трассировка BUY → TP); на матчинг
    // не влияет (позиция идентифицируется через переданный `pos_arc`).
    let (pos_id, asset_id, shares, tp_price, open_order_id) = {
        let mut pos = pos_arc.write().await;
        if pos.tp_placement_attempted || pos.tp_order_id.is_some() {
            return;
        }
        if !matches!(pos.open_status, OpenPositionStatus::Open) {
            return;
        }
        if pos.shares_held <= 0.0 || !pos.shares_held.is_finite() {
            return;
        }
        pos.tp_placement_attempted = true;
        (
            pos.id.clone(),
            pos.asset_id.clone(),
            pos.shares_held,
            (pos.buy_price + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999),
            pos.open_order_id.clone(),
        )
    };

    // Этап 2: HTTP без локов.
    let request = PostOrderRequest {
        asset_id: asset_id.clone(),
        side: Side::Sell,
        role: OrderRole::Maker,
        amount: OrderAmount::Shares(shares),
        price: Some(tp_price),
        max_slippage_pp: None,
        expiration: None,
        timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
        strict_book: None,
    };
    let result = match post_order_on_clob(&account, request).await {
        Ok(r) => r,
        Err(err) => {
            crate::tee_eprintln!(
                "[account_submit] TP maker упал: pos_id={pos_id}, open_order_id={open_order_id:?}, asset={asset_id}: {err:#}",
            );
            return;
        }
    };
    if !result.success {
        crate::tee_eprintln!(
            "[account_submit] TP maker отвергнут CLOB: pos_id={pos_id}, error_msg={:?}, status={:?}, order_id={}",
            result.error_msg, result.status, result.order_id,
        );
        return;
    }
    let tp_order_id = result.order_id.clone();

    // Этап 3: запись `tp_order_id` напрямую через inner-write `pos_arc`.
    pos_arc.write().await.tp_order_id = Some(tp_order_id.clone());

    crate::tee_println!(
        "[account_submit] TP maker размещён: pos_id={pos_id}, tp_order_id={tp_order_id}, open_order_id={open_order_id:?}, price={tp_price:.4}, shares={shares:.4}",
    );
    // Polling-fallback симметрично BUY-taker'у: WS-канал может пропустить
    // `trade` event на TP-fill, поэтому отдельная таска опрашивает
    // `client.order(tp_order_id)` до Matched/Canceled или до `event_end_ms`
    // маркета (см. [`PollingPositionKind::TpMaker`]). На Matched, если
    // финализация PnL ещё не прошла через WS, polling фетчит
    // `client.trades(...)` и сам прогонит TP-ветку `apply_sell_fill` →
    // `finalize_tp_close_after_creation`.
    spawn_polling_verify_tp(account.clone(), pos_arc.clone());
}

/// Спавнит таск **закрытия позиции через taker SELL** (SL / Timeout / EvExit*):
/// 1) если у `closing_arc.position` есть активный `tp_order_id` —
///    `cancel_order_on_clob` (поле `take()`-нится под write-локом, чтобы
///    повторных попыток отменить ту же лимитку не было);
/// 2) `post_order_on_clob` SELL taker без slippage cap'а
///    (`max_slippage_pp=None` → CLOB зальёт сколько успеет с `Amount::shares`);
/// 3) обновляет `closing_arc.close_order_id` на real `order_id` напрямую через
///    inner-write;
/// 4) спавнит [`spawn_polling_verify_close`] для fallback'а к WS-колбеку.
///
/// Идентификация in-flight записи о закрытии — через переданный `closing_arc`
/// (см. [`SharedClosingPosition`]); `close_order_id` пишется напрямую через
/// inner-RwLock после получения real `order_id` от CLOB. `asset_id`,
/// `shares_held` и `tp_order_id` snapshot'ятся из `closing_arc.position` под
/// коротким write-локом в начале таски — отдельным параметрам в сигнатуре места нет.
///
/// Дедуп: caller (`manage_positions` в submit-режиме) уже взводит
/// `OpenPosition.tp_placement_attempted = true` под write-локом самой позиции
/// **синхронно ДО** вызова этой функции (защита от запоздавшего WS/polling
/// `try_place_tp_maker` в окне между push'ем `ClosingPosition` и стартом
/// этой `tokio::spawn`-таски). И `ClosingPosition.close_placement_attempted = true`
/// — защита от повторного входа в `manage_positions`-сценарий для той же позиции.
pub fn spawn_close_via_taker(account: SharedAccount, closing_arc: SharedClosingPosition) {
    tokio::spawn(async move {
        // Гейт по времени жизни маркета: если wall-clock уже за `event_end_ms`,
        // SELL-taker на CLOB НЕ отправляем (и TP-maker не отменяем — пусть он
        // имеет шанс ещё залиться до резолюции). Позиция доедет до
        // [`crate::account::Account::resolve_pending_market`]: auto-redeem
        // $1/$0, PnL/bankroll/SimStats и submit-CSV обновятся payout-колбеком
        // (см. `record_market_outcome` / `record_submit_close_to_csv_and_stats`).
        //
        // Зачем нужно: `manage_positions` может дёрнуть SL/Timeout/EvExit на
        // stale-кадре (когда `frame.event_remaining_ms` ещё > 0, потому что
        // `snapshot.timestamp_ms` отстал, а wall-clock уже за `event_end_ms`).
        // Без этого гейта мы бы тратили реальные taker-fee на SELL ровно в
        // момент резолюции, при том что auto-redeem всё равно поднимет шеры
        // до $1/$0 и закроет позицию. Зеркальный гейт по BUY стоит в
        // [`crate::real_sim::tick_once`].
        //
        // Терминальное состояние записи: `close_status=CloseFailed`.
        // `manage_positions` на следующем тике через `closing.retain` выкинет
        // эту запись (см. doc-комментарий перед циклом cleanup), и `sell_gate`
        // там же вернёт `HoldPnl` как только `frame.event_remaining_ms`
        // догонит wall-clock (≤ 1с при здоровом WS). Если кадр ещё не догнал
        // и `sell_gate` снова потребует close — спавн повторится и опять
        // мгновенно отвалится по этому же гейту без HTTP. Когда маркет
        // действительно резолвнется, `Account::resolve_pending_market`
        // вытащит [`OpenPosition`] из `positions` в `pending_resolution`
        // и проведёт бинарную выплату.
        {
            let now_wall_ms = crate::util::current_timestamp_ms();
            let (event_end_ms_opt, pos_id_log, asset_id_log) = {
                let pos_arc = closing_arc.read().await.position.clone();
                let pos = pos_arc.read().await;
                (pos.event_end_ms, pos.id.clone(), pos.asset_id.clone())
            };
            let past_event_end = match event_end_ms_opt {
                Some(end_ms) => now_wall_ms >= end_ms,
                None => false,
            };
            if past_event_end {
                closing_arc.write().await.close_status = ClosingPositionStatus::CloseFailed;
                crate::tee_println!(
                    "[account_submit] SELL taker пропущен — wall-clock за event_end_ms: \
                     pos_id={pos_id_log}, asset={asset_id_log}, now_wall_ms={now_wall_ms}, \
                     event_end_ms={event_end_ms_opt:?} — ждём резолюцию через \
                     Account::resolve_pending_market (PnL/bankroll/stats придут payout-колбеком)"
                );
                return;
            }
        }

        // Snapshot из позиции под коротким write-локом. Под этим же локом
        // делаем `take()` для `tp_order_id`, чтобы любой будущий код, заглянувший
        // в `pos.tp_order_id`, видел `None` (мы как раз его сейчас отменяем).
        // `tp_placement_attempted` уже `true` (выставлено в `manage_positions`).
        let (pos_id, asset_id, shares_to_sell, tp_order_id_to_cancel) = {
            let pos_arc = closing_arc.read().await.position.clone();
            let mut pos = pos_arc.write().await;
            (
                pos.id.clone(),
                pos.asset_id.clone(),
                pos.shares_held,
                pos.tp_order_id.take(),
            )
        };

        // Шаг 1: отмена TP (если есть). Игнорируем ошибки — после этого всё
        // равно идём в SELL taker. CLOB при `not_canceled` (TP уже сматчен/отменён)
        // вернёт причину; реальное состояние подтвердится через user-WS.
        if let Some(tp_id) = tp_order_id_to_cancel.as_deref() {
            let cancel_req = CancelOrderRequest {
                order_id: tp_id.to_string(),
                timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            };
            match cancel_order_on_clob(&account, cancel_req).await {
                Ok(res) => {
                    crate::tee_println!(
                        "[account_submit] TP cancel: pos_id={pos_id}, order_id={tp_id}, canceled={}, error_msg={:?}",
                        res.canceled, res.error_msg,
                    );
                }
                Err(err) => {
                    crate::tee_eprintln!(
                        "[account_submit] TP cancel упал: pos_id={pos_id}, tp_order_id={tp_id}: {err:#} — продолжаем SELL taker"
                    );
                }
            }
        }

        // Шаг 2: SELL taker без slippage cap'а.
        // Retry-loop с exp-backoff (500ms → 1s → 2s → …) до
        // [`SELL_TAKER_MAX_ATTEMPTS`] попыток. Retry'им и HTTP-падения, и
        // CLOB-rejection'ы (transient: rate limits, internal errors, network).
        // На исчерпании всех попыток ставим `CloseFailed`; следующий
        // `manage_positions`-тик после cleanup'а зайдёт в `sell_gate` снова
        // (tick-based retry на fallback'е).
        let request_template = PostOrderRequest {
            asset_id: asset_id.clone(),
            side: Side::Sell,
            role: OrderRole::Taker,
            amount: OrderAmount::Shares(shares_to_sell),
            price: None,
            max_slippage_pp: None,
            expiration: None,
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            strict_book: None,
        };
        let mut accepted: Option<crate::account_order::PostOrderResult> = None;
        for attempt in 1..=SELL_TAKER_MAX_ATTEMPTS {
            match post_order_on_clob(&account, request_template.clone()).await {
                Ok(r) if r.success => {
                    crate::tee_println!(
                        "[account_submit] SELL taker принят (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, order_id={}, status={:?}",
                        r.order_id, r.status,
                    );
                    accepted = Some(r);
                    break;
                }
                Ok(r) => {
                    crate::tee_eprintln!(
                        "[account_submit] SELL taker отвергнут CLOB (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, asset={asset_id}, error_msg={:?}, status={:?}, order_id={}",
                        r.error_msg, r.status, r.order_id,
                    );
                }
                Err(err) => {
                    crate::tee_eprintln!(
                        "[account_submit] SELL taker HTTP-ошибка (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, asset={asset_id}: {err:#}"
                    );
                }
            }
            if attempt < SELL_TAKER_MAX_ATTEMPTS {
                // exp-backoff: 500ms, 1s, 2s, 4s, …
                let delay_ms = SELL_TAKER_RETRY_INITIAL_MS << (attempt - 1);
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
        let Some(result) = accepted else {
            crate::tee_eprintln!(
                "[account_submit] SELL taker — все {SELL_TAKER_MAX_ATTEMPTS} попыток исчерпаны, CloseFailed: pos_id={pos_id}, asset={asset_id}; следующий manage_positions-тик попытается снова"
            );
            closing_arc.write().await.close_status = ClosingPositionStatus::CloseFailed;
            return;
        };
        let real_sell_id = result.order_id.clone();

        // Шаг 3: записать real `close_order_id` напрямую через inner-write.
        closing_arc.write().await.close_order_id = Some(real_sell_id.clone());
        spawn_polling_verify_close(account.clone(), closing_arc.clone());
    });
}

/// Polling-fallback: каждые [`POLL_INTERVAL_SEC`] дёргает `client.order(order_id)`
/// до терминального статуса (`MATCHED`/`CANCELED`/etc.) или таймаута
/// [`POLL_TIMEOUT_SEC`]. На терминальном статусе вызывает
/// [`apply_order_status_from_polling`] — те же транзишены,
/// что и WS-колбек, идемпотентно. Полезно для случая, когда WS-канал
/// прилёг и пропустил наше событие.
///
/// **Идентификация ордера — через Arc внутри [`PollingPositionKind`]**:
/// `order_id` снимается snapshot'ом из `pos.open_order_id` или
/// `c.close_order_id` в начале таски; отдельным параметром не передаётся.
/// [проверено]
fn spawn_polling_verify_open(account: SharedAccount, pos_arc: SharedOpenPosition) {
    spawn_polling_verify(account, PollingPositionKind::Open(pos_arc));
}
/// [проверено]
fn spawn_polling_verify_close(account: SharedAccount, c_arc: SharedClosingPosition) {
    spawn_polling_verify(account, PollingPositionKind::Close(c_arc));
}
/// Polling-fallback для maker TP (`pos.tp_order_id`): симметрично
/// [`spawn_polling_verify_open`] / [`spawn_polling_verify_close`]; на Matched
/// фетчим `client.trades(...)` и финализируем PnL (TP-ветка) если WS
/// не успел; на Canceled — TP отменён нами / CLOB протух, no-op.
/// [проверено]
fn spawn_polling_verify_tp(account: SharedAccount, pos_arc: SharedOpenPosition) {
    spawn_polling_verify(account, PollingPositionKind::TpMaker(pos_arc));
}

/// Дискриминатор + ссылка на конкретную in-flight запись для polling-флоу.
/// Каждый variant держит `Arc` на ту структуру, которую `apply_order_status_from_polling`
/// будет напрямую (без поиска по id) обновлять на терминальном статусе.
/// `order_id` для `client.order(...)` читается из соответствующего поля
/// (`open_order_id` / `close_order_id` / `tp_order_id`) этой же структуры.
#[derive(Clone)]
pub(crate) enum PollingPositionKind {
    /// BUY taker: poll `pos.open_order_id`; на `Matched` → `OpenPositionStatus::Open`
    /// (+ trigger `try_place_tp_maker`); на `Canceled`/etc. → `OpenFailed`.
    Open(SharedOpenPosition),
    /// SELL taker (SL/Timeout/EvExit-taker): poll `c.close_order_id`;
    /// на `Matched` → `ClosingPositionStatus::Closed` + финализация PnL
    /// (fallback к WS: если `c.pnl=None` — фетчим `client.trades(...)` и
    /// прогоняем `apply_sell_fill` на каждый trade; иначе только
    /// `finalize_close_pnl_in_place`); на `Canceled`/etc. → `CloseFailed`.
    Close(SharedClosingPosition),
    /// Maker TP: poll `pos.tp_order_id` (наша лимитка SELL, выставленная
    /// в [`try_place_tp_maker`]); на `Matched` — TP сам залился, идёт тот же
    /// PnL-fallback что и у `Close` (см. ниже), но через TP-ветку
    /// `apply_sell_fill` (создаёт `ClosingPosition { reason=TakeProfit,
    /// close_status=Closed }` и вызывает `finalize_tp_close_after_creation`);
    /// на `Canceled` — TP отменён нами в [`spawn_close_via_taker`] перед SELL
    /// taker'ом или CLOB протух — no-op, дальнейшую обработку драйвит
    /// тот SELL-taker.
    TpMaker(SharedOpenPosition),
}

/// Результат одного цикла polling'а: терминальность + опционально follow-up
/// действие (TP-spawn / PnL-финализация). Все side-effects'ы, требующие HTTP
/// (REST-fallback fetch trades, TP-постановка), выполняет **caller**
/// [`spawn_polling_verify`] — внутри [`apply_order_status_from_polling`] идут
/// только локальные мутации state'ов под inner-локами (никаких сетевых
/// вызовов, симметрично WS-колбеку [`crate::account_ws::apply_user_ws_event_value`]).
pub(crate) enum PollingApplyOutcome {
    /// Не-терминальный статус (`Live`/`Delayed`) — polling продолжается.
    Continue,
    /// Терминальный (`Matched`/`Canceled`), без follow-up действия.
    Terminal,
    /// Терминальный + позиция перешла `PendingOpen → Open` — caller обязан
    /// поставить TP-лимитку для переданного Arc (идемпотентно через
    /// `tp_placement_attempted`, гонка с WS-колбеком безопасна).
    TerminalTriggerTp(SharedOpenPosition),
    /// Терминальный `Matched` для SELL-taker close — caller обязан довести
    /// PnL-финализацию через [`drive_close_pnl_finalization_via_polling`]
    /// (REST-fallback на `client.trades(...)` + `finalize_close_pnl_in_place`).
    /// Идемпотентно через `OpenPosition.pnl_finalized`.
    TerminalFinalizeClose(SharedClosingPosition),
    /// Терминальный `Matched` для maker TP — caller обязан довести
    /// PnL-финализацию через [`drive_tp_pnl_finalization_via_polling`]
    /// (REST-fallback + `finalize_tp_close_after_creation`).
    /// Идемпотентно через `OpenPosition.pnl_finalized`.
    TerminalFinalizeTp(SharedOpenPosition),
}

impl PollingPositionKind {
    /// Имя варианта для логов (вместо `Debug`-формата `Arc<RwLock<…>>`,
    /// который был бы шумным и потенциально брал бы лок при печати).
    fn variant_name(&self) -> &'static str {
        match self {
            Self::Open(_) => "Open",
            Self::Close(_) => "Close",
            Self::TpMaker(_) => "TpMaker",
        }
    }

    /// Snapshot real `order_id` из соответствующего Arc'а под коротким
    /// read-локом. `None` означает «id ещё не получили от CLOB» — caller
    /// в этом случае не должен запускать polling, см. защитный return ниже.
    async fn snapshot_order_id(&self) -> Option<String> {
        match self {
            Self::Open(pos_arc) => pos_arc.read().await.open_order_id.clone(),
            Self::Close(c_arc) => c_arc.read().await.close_order_id.clone(),
            Self::TpMaker(pos_arc) => pos_arc.read().await.tp_order_id.clone(),
        }
    }

    /// Snapshot локального uuid позиции
    /// (см. [`crate::history_sim::OpenPosition::id`]) — для корреляции
    /// логов polling-таски с остальными submit-флоу логами. Lock-ordering:
    /// для `Close` сначала `c.read()` чтобы клонировать `position` Arc, дроп,
    /// потом `pos.read()` (max один inner-lock одновременно).
    async fn pos_id(&self) -> String {
        match self {
            Self::Open(pos_arc) | Self::TpMaker(pos_arc) => pos_arc.read().await.id.clone(),
            Self::Close(c_arc) => {
                let pos_arc = {
                    let c = c_arc.read().await;
                    c.position.clone()
                };
                let id = pos_arc.read().await.id.clone();
                id
            }
        }
    }

    /// Snapshot `event_end_ms` (UTC мс конца окна маркета) — дедлайн polling-таски.
    /// Для `Open` / `TpMaker` — `pos.event_end_ms` напрямую; для `Close` — через
    /// `c.position` (ссылается на тот же `OpenPosition`). `None` означает, что
    /// дедлайн неизвестен — caller использует [`POLL_TIMEOUT_SEC`] как fallback.
    /// Lock-ordering: внешний `c.read()` отпускаем до взятия `pos.read()` — max
    /// один inner-lock одновременно.
    async fn event_end_ms(&self) -> Option<i64> {
        match self {
            Self::Open(pos_arc) | Self::TpMaker(pos_arc) => pos_arc.read().await.event_end_ms,
            Self::Close(c_arc) => {
                let pos_arc = {
                    let c = c_arc.read().await;
                    c.position.clone()
                };
                pos_arc.read().await.event_end_ms
            }
        }
    }
}

/// Polling-fallback из [`spawn_polling_verify`]: применяет статус из
/// `client.order(...)` к локальному состоянию **напрямую через `Arc`**,
/// зашитый в [`PollingPositionKind`] (без поиска по `order_id`). Возвращает
/// [`PollingApplyOutcome`], сигнализирующий caller'у, что делать дальше.
///
/// **HTTP-вызовы здесь не делаются** — функция симметрична WS-колбеку
/// [`crate::account_ws::apply_user_ws_event_value`] и работает только с
/// локальным state'ом под inner-локами. REST-fallback PnL-финализации (для
/// `Close::Matched` / `TpMaker::Matched`) и постановка TP-лимитки (для
/// `Open::Matched` после `PendingOpen → Open`) выполняются **caller'ом**
/// [`spawn_polling_verify`] через `tokio::spawn` после возврата —
/// см. матч на `PollingApplyOutcome` ниже по файлу.
///
/// Идемпотентность с WS гарантируется маркерами:
/// - [`crate::history_sim::OpenPosition::pnl_finalized`] — для PnL.
/// - [`crate::history_sim::OpenPosition::tp_placement_attempted`] — для TP.
/// [проверено]
pub(crate) async fn apply_order_status_from_polling(
    status: &OrderStatusType,
    kind: PollingPositionKind,
) -> PollingApplyOutcome {
    use OrderStatusType::*;
    match kind {
        PollingPositionKind::Open(pos_arc) => {
            let new_status = match status {
                Matched => OpenPositionStatus::Open,
                Canceled => OpenPositionStatus::OpenFailed,
                _ => return PollingApplyOutcome::Continue,
            };
            // Транзишн PendingOpen → Open (по polling'у) триггерит TP, как и
            // в WS-колбеке. Идемпотентность через `tp_placement_attempted`
            // гарантирует, что повтор от WS не задвоит ордер. Сам spawn TP
            // делает caller (`spawn_polling_verify`) — здесь только сигнализируем
            // о необходимости.
            let trigger_tp = {
                let mut pos = pos_arc.write().await;
                let was_pending = matches!(pos.open_status, OpenPositionStatus::PendingOpen);
                pos.open_status = new_status;
                let oid = pos.open_order_id.clone();
                let pos_id = pos.id.clone();
                drop(pos);
                crate::tee_println!(
                    "[account_submit/poll] open_status({oid:?}) → {new_status:?} (pos_id={pos_id})",
                );
                was_pending && matches!(new_status, OpenPositionStatus::Open)
            };
            if trigger_tp {
                PollingApplyOutcome::TerminalTriggerTp(pos_arc)
            } else {
                PollingApplyOutcome::Terminal
            }
        }
        PollingPositionKind::Close(c_arc) => match status {
            Matched => {
                // Никаких HTTP здесь — финализацию доводит caller через
                // `drive_close_pnl_finalization_via_polling` (REST-fallback +
                // finalize). `close_status` тоже ставит он, **после**
                // прогона REST-fills (иначе `apply_sell_fill` дёрнет
                // finalize преждевременно — см. doc у драйвера).
                let (oid, pos_id) = {
                    let c = c_arc.read().await;
                    let pos_arc_inner = c.position.clone();
                    let oid = c.close_order_id.clone();
                    drop(c);
                    let pos_id = pos_arc_inner.read().await.id.clone();
                    (oid, pos_id)
                };
                crate::tee_println!(
                    "[account_submit/poll] close_status({oid:?}) → Matched (PnL-финализация в caller'е) (pos_id={pos_id})",
                );
                PollingApplyOutcome::TerminalFinalizeClose(c_arc)
            }
            Canceled => {
                let (oid, pos_arc_inner) = {
                    let mut c = c_arc.write().await;
                    c.close_status = ClosingPositionStatus::CloseFailed;
                    (c.close_order_id.clone(), c.position.clone())
                };
                let pos_id = pos_arc_inner.read().await.id.clone();
                crate::tee_println!(
                    "[account_submit/poll] close_status({oid:?}) → CloseFailed (pos_id={pos_id})",
                );
                PollingApplyOutcome::Terminal
            }
            _ => PollingApplyOutcome::Continue,
        },
        PollingPositionKind::TpMaker(pos_arc) => match status {
            Matched => {
                // Никаких HTTP здесь — финализацию доводит caller через
                // `drive_tp_pnl_finalization_via_polling` (REST-fallback +
                // finalize_tp_close_after_creation).
                let (tp_id, pos_id) = {
                    let p = pos_arc.read().await;
                    (p.tp_order_id.clone(), p.id.clone())
                };
                crate::tee_println!(
                    "[account_submit/poll] tp_order_id({tp_id:?}) → Matched (PnL-финализация в caller'е) (pos_id={pos_id})",
                );
                PollingApplyOutcome::TerminalFinalizeTp(pos_arc)
            }
            Canceled => {
                // TP отменён нами в `spawn_close_via_taker` перед SELL taker'ом
                // или CLOB протух (например, после market resolution через
                // `spawn_cancel_tp_orders_after_resolution`). В обоих случаях
                // соответствующий close-flow обработает позицию дальше — здесь
                // только лог.
                let (tp_id, pos_id) = {
                    let p = pos_arc.read().await;
                    (p.tp_order_id.clone(), p.id.clone())
                };
                crate::tee_println!(
                    "[account_submit/poll] tp_order_id({tp_id:?}) → Canceled (no-op, close-flow продолжит) (pos_id={pos_id})",
                );
                PollingApplyOutcome::Terminal
            }
            _ => PollingApplyOutcome::Continue,
        },
    }
}

/// Шаги PnL-финализации для SELL-taker close при polling-fallback'е
/// (см. [`apply_order_status_from_polling`]):
/// 1. Snapshot `pnl_finalized` (маркер «уже финализировано»), `c.pnl`,
///    `close_order_id`, `asset_id`.
/// 2. Если `pos.pnl_finalized == true` — выходим, finalize уже отработал.
/// 3. Если `c.pnl.is_none()` — WS ничего не дал; тащим fills из REST
///    (`client.trades(...)`) и прогоняем через `apply_sell_fill` с
///    `close_status=PendingClose` (без auto-finalize).
/// 4. Атомарно ставим `close_status=Closed` и зовём
///    [`crate::account_ws::finalize_close_pnl_in_place`] — он вычтет
///    `entry_cost`, проставит `pnl_finalized=true`, обновит bankroll.
async fn drive_close_pnl_finalization_via_polling(
    account: &SharedAccount,
    c_arc: &SharedClosingPosition,
) {
    // Snapshot: pos_arc + (pnl_finalized, pos_id) под коротким read'ом.
    let pos_arc = {
        let c = c_arc.read().await;
        c.position.clone()
    };
    let (pnl_finalized, pos_id) = {
        let p = pos_arc.read().await;
        (p.pnl_finalized, p.id.clone())
    };
    let (pnl_already_some, oid) = {
        let c = c_arc.read().await;
        (c.pnl.is_some(), c.close_order_id.clone())
    };

    if pnl_finalized {
        crate::tee_println!(
            "[account_submit/poll] close_status({oid:?}) → Closed (PnL уже финализирован WS, no-op) (pos_id={pos_id})",
        );
        return;
    }

    if !pnl_already_some {
        if let Some(order_id) = oid.as_deref() {
            // REST-fallback. close_status пока остаётся `PendingClose`, чтобы
            // `apply_sell_fill` не дёрнул finalize_close_pnl_in_place раньше
            // времени (мы хотим аккумулировать ВСЕ fills, прежде чем вычесть
            // entry_cost ровно один раз).
            fetch_and_apply_trades_for_order(account, &pos_id, order_id, OrderRole::Taker).await;
        }
    } else {
        crate::tee_println!(
            "[account_submit/poll] close_status({oid:?}) → Closed (WS уже накопил c.pnl, REST-fallback не нужен) (pos_id={pos_id})",
        );
    }

    // Только теперь — атомарно ставим Closed и финализируем.
    {
        let mut c = c_arc.write().await;
        c.close_status = ClosingPositionStatus::Closed;
    }
    crate::tee_println!(
        "[account_submit/poll] close_status({oid:?}) → Closed (pos_id={pos_id})",
    );
    crate::account_ws::finalize_close_pnl_in_place(account, c_arc.clone(), "Polling").await;
}

/// PnL-финализация для maker TP при polling-fallback'е:
/// 1. Snapshot из [`crate::history_sim::OpenPosition`]: `tp_order_id`,
///    `asset_id`, `pnl_finalized`, и Weak-ссылка `closing_position`
///    (point-of-truth, проставленная в момент создания `ClosingPosition` в
///    [`crate::account_ws::apply_sell_fill`] TP-ветка / `manage_positions`).
/// 2. Если `pnl_finalized=true` — выходим (WS уже отработал).
/// 3. Если `closing_position.upgrade()` отдаёт `Some(_)` — `ClosingPosition`
///    уже создана WS-колбеком; зовём
///    [`crate::account_ws::finalize_tp_close_after_creation`] (идемпотентно).
/// 4. Если `None` — WS ничего не дал; тащим fills из REST и прогоняем
///    через [`crate::account_ws::apply_sell_fill`]. TP-ветка `apply_sell_fill`
///    сама создаст `ClosingPosition { reason=TakeProfit, close_status=Closed,
///    pnl=Some(net-entry_cost) }` и зовёт `finalize_tp_close_after_creation`.
async fn drive_tp_pnl_finalization_via_polling(
    account: &SharedAccount,
    pos_arc: &SharedOpenPosition,
) {
    let (tp_order_id, pnl_finalized, pos_id, existing_close) = {
        let p = pos_arc.read().await;
        (
            p.tp_order_id.clone(),
            p.pnl_finalized,
            p.id.clone(),
            p.closing_position.as_ref().and_then(std::sync::Weak::upgrade),
        )
    };
    let Some(tp_id) = tp_order_id else {
        return;
    };

    if pnl_finalized {
        crate::tee_println!(
            "[account_submit/poll] tp_order_id({tp_id}) → Matched (PnL уже финализирован WS, no-op) (pos_id={pos_id})",
        );
        return;
    }

    if existing_close.is_some() {
        // WS уже создал ClosingPosition; финализируем (идемпотентно через
        // `pnl_finalized`-маркер).
        crate::tee_println!(
            "[account_submit/poll] tp_order_id({tp_id}) → Matched (ClosingPosition уже создана WS — финализируем) (pos_id={pos_id})",
        );
        crate::account_ws::finalize_tp_close_after_creation(account, &tp_id, "Polling").await;
    } else {
        // WS не успел; REST-fallback. `apply_sell_fill` (TP-ветка) сам создаст
        // `ClosingPosition` (и проставит `pos.closing_position`!) и
        // финализирует bankroll. Если придёт несколько partial-fills — первый
        // создаст, остальные пойдут close-веткой и финализируются (no-op после
        // первого) через `finalize_close_pnl_in_place`.
        crate::tee_println!(
            "[account_submit/poll] tp_order_id({tp_id}) → Matched (REST-fallback: тащим trades и финализируем) (pos_id={pos_id})",
        );
        fetch_and_apply_trades_for_order(account, &pos_id, &tp_id, OrderRole::Maker).await;
    }
}

/// REST-fallback для PnL-финализации: тащит `client.trades(...)` (страничный
/// фетч; SDK сам фильтрует по нашему юзеру) для заданного `asset_id`,
/// постфильтрует по `order_id` и применяет каждый fill через
/// [`crate::account_ws::apply_sell_fill`] — тот же путь, что у WS.
///
/// `role` определяет, в каком поле trade'а искать наш `order_id`:
/// - [`OrderRole::Taker`] — `trade.taker_order_id`.
/// - [`OrderRole::Maker`] — `trade.maker_orders[i].order_id` (наш TP).
///
/// Конверсия `Decimal → f64` через `to_string().parse::<f64>()` — тот же приём,
/// что в `account_order::f64_to_decimal` (predictable scale без IEEE-754 шума).
async fn fetch_and_apply_trades_for_order(
    account: &SharedAccount,
    pos_id: &str,
    order_id: &str,
    role: OrderRole,
) {
    let auth_client = match (**account.clob_authed.load()).clone() {
        Some(c) => c,
        None => {
            crate::tee_eprintln!(
                "[account_submit/poll-rest] auth-клиент пуст — REST-fallback пропускаем: pos_id={pos_id}, order_id={order_id}"
            );
            return;
        }
    };
    // Шаг 1: `client.order(order_id)` → `associate_trades: Vec<String>` —
    // готовый список trade-id'ов, относящихся именно к нашему ордеру. Это
    // делает `/data/trades?asset_id=…` + постфильтр по `taker_order_id` /
    // `maker_orders[i].order_id` ненужным: вместо страничного walk'а тащим
    // только нужные трейды точечно по `id`. Бонусом `client.order` уже
    // используется в polling-loop'е (см. `auth_client.order(...)` в
    // `spawn_polling_verify`), а тут — отдельный «свежий» снимок: между
    // последним polling-tick'ом и этим вызовом могли долететь partial fills.
    let order_resp = match tokio::time::timeout(
        Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
        auth_client.order(order_id),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            crate::tee_eprintln!(
                "[account_submit/poll-rest] client.order({order_id}) упал: {err:#} (pos_id={pos_id})"
            );
            return;
        }
        Err(_) => {
            crate::tee_eprintln!(
                "[account_submit/poll-rest] client.order({order_id}) timeout (pos_id={pos_id})"
            );
            return;
        }
    };
    let trade_ids: Vec<String> = order_resp.associate_trades;
    if trade_ids.is_empty() {
        crate::tee_println!(
            "[account_submit/poll-rest] order_id={order_id} role={role:?}: associate_trades пуст — нечего применять (pos_id={pos_id})",
        );
        return;
    }

    // Шаг 2: для каждого `trade_id` точечно `client.trades(TradesRequest {
    // id: Some(trade_id), … })` — у /data/trades есть фильтр по `id`
    // (см. `polymarket_client_sdk::clob::types::request::TradesRequest`).
    // Возвращается `Page<TradeResponse>` с одной записью; пагинации не нужно
    // (точечный запрос). Постфильтр по `order_id` всё равно делаем (паранойя:
    // у trade'а может быть несколько `maker_orders[]` — нас интересует
    // конкретно наш).
    let mut applied_count: usize = 0;
    for trade_id in trade_ids {
        let request = TradesRequest::builder().id(trade_id.clone()).build();
        let page = match tokio::time::timeout(
            Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            auth_client.trades(&request, None),
        )
        .await
        {
            Ok(Ok(p)) => p,
            Ok(Err(err)) => {
                crate::tee_eprintln!(
                    "[account_submit/poll-rest] client.trades(id={trade_id}) упал: {err:#} (pos_id={pos_id}, order_id={order_id})"
                );
                continue;
            }
            Err(_) => {
                crate::tee_eprintln!(
                    "[account_submit/poll-rest] client.trades(id={trade_id}) timeout (pos_id={pos_id}, order_id={order_id})"
                );
                continue;
            }
        };
        for trade in page.data.iter() {
            match role {
                OrderRole::Taker => {
                    if trade.taker_order_id != order_id {
                        continue;
                    }
                    let size = decimal_to_f64(&trade.size);
                    let price = decimal_to_f64(&trade.price);
                    let fee_rate_bps = decimal_to_f64(&trade.fee_rate_bps);
                    if !(size > 0.0 && size.is_finite())
                        || !(price > 0.0 && price.is_finite())
                    {
                        continue;
                    }
                    crate::account_ws::apply_sell_fill(
                        account,
                        order_id,
                        size,
                        price,
                        fee_rate_bps,
                        OrderRole::Taker,
                    )
                    .await;
                    applied_count += 1;
                }
                OrderRole::Maker => {
                    for m in trade.maker_orders.iter() {
                        if m.order_id != order_id {
                            continue;
                        }
                        let size = decimal_to_f64(&m.matched_amount);
                        let price = decimal_to_f64(&m.price);
                        let fee_rate_bps = decimal_to_f64(&m.fee_rate_bps);
                        if !(size > 0.0 && size.is_finite())
                            || !(price > 0.0 && price.is_finite())
                        {
                            continue;
                        }
                        crate::account_ws::apply_sell_fill(
                            account,
                            order_id,
                            size,
                            price,
                            fee_rate_bps,
                            OrderRole::Maker,
                        )
                        .await;
                        applied_count += 1;
                    }
                }
            }
        }
    }
    crate::tee_println!(
        "[account_submit/poll-rest] order_id={order_id} role={role:?}: applied {applied_count} fill(s) (pos_id={pos_id})",
    );
}

/// Конверсия `polymarket_client_sdk::types::Decimal → f64` через строковый
/// roundtrip (тот же приём, что у `account_order::f64_to_decimal` в обратную
/// сторону: предсказуемая точность, нет IEEE-754 шума).
fn decimal_to_f64(d: &polymarket_client_sdk::types::Decimal) -> f64 {
    d.to_string().parse::<f64>().unwrap_or(0.0)
}

/// [проверено]
fn spawn_polling_verify(account: SharedAccount, kind: PollingPositionKind) {
    tokio::spawn(async move {
        let kind_label = kind.variant_name();
        // Snapshot pos_id (корреляция логов) — ровно один раз, в начале таски.
        let pos_id = kind.pos_id().await;
        // Snapshot real `order_id` один раз в начале — после получения HTTP-
        // ответа caller успел его записать в Arc, дальше он неизменен (CLOB
        // id присваивается один раз).
        let order_id = match kind.snapshot_order_id().await {
            Some(id) => id,
            None => {
                crate::tee_eprintln!(
                    "[account_submit/poll] {kind_label}: real order_id ещё не получен — polling не запускаем (pos_id={pos_id})"
                );
                return;
            }
        };
        // Дедлайн polling'а — `OpenPosition.event_end_ms` (UTC мс конца окна
        // маркета). После него Polymarket уже резолвит маркет, и активный
        // ордер либо заматчился (TP), либо отменён системой (наш taker SELL —
        // unlikely, но возможно). Если `event_end_ms=None` (не должно быть для
        // submit-флоу: real_sim проставляет его в `try_open_position`) —
        // fallback на короткий [`POLL_TIMEOUT_SEC`].
        let now_ms = crate::util::current_timestamp_ms();
        let deadline_ms: i64 = match kind.event_end_ms().await {
            Some(end) if end > now_ms => end,
            Some(end) => {
                crate::tee_eprintln!(
                    "[account_submit/poll] {kind_label} order_id={order_id}: event_end_ms={end} уже в прошлом (now={now_ms}) — polling не запускаем (pos_id={pos_id})"
                );
                return;
            }
            None => now_ms.saturating_add((POLL_TIMEOUT_SEC as i64) * 1_000),
        };
        let mut tick = tokio::time::interval(Duration::from_secs(POLL_INTERVAL_SEC));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // первый tick мгновенный — пропускаем, чтобы не гонять API сразу
        // после успешного post_order (у CLOB может быть мини-задержка
        // согласованности «POST принят, GET ещё не показывает»).
        tick.tick().await;
        loop {
            tick.tick().await;
            let now_ms = crate::util::current_timestamp_ms();
            if now_ms >= deadline_ms {
                crate::tee_eprintln!(
                    "[account_submit/poll] {kind_label} order_id={order_id} — дедлайн event_end_ms={deadline_ms} достигнут, бросаем polling (pos_id={pos_id})"
                );
                return;
            }
            let auth_client = match (**account.clob_authed.load()).clone() {
                Some(c) => c,
                None => {
                    // Auth исчез/не поднялся — идём дальше, может авто-восстановится.
                    continue;
                }
            };
            let resp = match tokio::time::timeout(
                Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
                auth_client.order(&order_id),
            )
            .await
            {
                Ok(Ok(r)) => r,
                Ok(Err(err)) => {
                    crate::tee_eprintln!(
                        "[account_submit/poll] {kind_label} client.order({order_id}) упал: {err:#} (pos_id={pos_id})",
                    );
                    continue;
                }
                Err(_) => {
                    crate::tee_eprintln!(
                        "[account_submit/poll] {kind_label} client.order({order_id}) таймаут (pos_id={pos_id})"
                    );
                    continue;
                }
            };
            // Передаём в общий обработчик — апдейт идёт напрямую через Arc
            // внутри `kind`, без повторного поиска по `order_id`. **HTTP-
            // вызовы (REST-fallback / TP-постановка) тут не делаются —
            // только локальные мутации**; follow-up действия выполняем ниже,
            // после возврата, чтобы `apply_order_status_from_polling` оставался
            // симметричным WS-колбеку (`apply_user_ws_event_value`).
            let outcome = apply_order_status_from_polling(&resp.status, kind.clone()).await;
            match outcome {
                PollingApplyOutcome::Continue => {
                    // Не-терминальный — продолжаем polling.
                }
                PollingApplyOutcome::Terminal => {
                    crate::tee_println!(
                        "[account_submit/poll] {kind_label} order_id={order_id} терминальный статус {:?}, polling завершён (pos_id={pos_id})",
                        resp.status,
                    );
                    return;
                }
                PollingApplyOutcome::TerminalTriggerTp(pos_arc) => {
                    let acc = account.clone();
                    tokio::spawn(async move {
                        try_place_tp_maker(acc, pos_arc).await;
                    });
                    crate::tee_println!(
                        "[account_submit/poll] {kind_label} order_id={order_id} терминальный статус {:?}, polling завершён, TP-задача запущена (pos_id={pos_id})",
                        resp.status,
                    );
                    return;
                }
                PollingApplyOutcome::TerminalFinalizeClose(c_arc) => {
                    // PnL-финализация для SELL-taker close: REST-fallback на
                    // `client.trades(...)` + `finalize_close_pnl_in_place`.
                    // Запускаем в отдельной spawn-таске, чтобы не блокировать
                    // exit polling-таски (она сама и так заканчивается, но
                    // драйвер делает HTTP, который может занять секунды; мы
                    // не хотим, чтобы тег «polling завершён» в логах ждал
                    // окончания HTTP).
                    let acc = account.clone();
                    tokio::spawn(async move {
                        drive_close_pnl_finalization_via_polling(&acc, &c_arc).await;
                    });
                    crate::tee_println!(
                        "[account_submit/poll] {kind_label} order_id={order_id} терминальный статус {:?}, polling завершён, PnL-финализация (close) запущена (pos_id={pos_id})",
                        resp.status,
                    );
                    return;
                }
                PollingApplyOutcome::TerminalFinalizeTp(pos_arc) => {
                    // PnL-финализация для maker TP: REST-fallback +
                    // `finalize_tp_close_after_creation`. Аналогично выше.
                    let acc = account.clone();
                    tokio::spawn(async move {
                        drive_tp_pnl_finalization_via_polling(&acc, &pos_arc).await;
                    });
                    crate::tee_println!(
                        "[account_submit/poll] {kind_label} order_id={order_id} терминальный статус {:?}, polling завершён, PnL-финализация (TP) запущена (pos_id={pos_id})",
                        resp.status,
                    );
                    return;
                }
            }
        }
    });
}

/// Отменяет TP-ордера на резолвнутых маркетах — вызывается из
/// [`crate::account::Account::resolve_pending_market`] после
/// payout'а. Auto-redeem забирает шеры в USDC, но висящие maker-лимитки
/// CLOB сам не снимает: на следующем раунде маркета они либо протухают
/// (asset_id больше не торгуется), либо CLOB отдаст ошибку — в любом
/// случае мы их уберём явно. На failures просто логируем — не критично.
pub fn spawn_cancel_tp_orders_after_resolution(
    account: SharedAccount,
    positions: Vec<crate::history_sim::SharedOpenPosition>,
) {
    if positions.is_empty() {
        return;
    }
    tokio::spawn(async move {
        for pos_arc in positions {
            // Snapshot + `take()` под одним коротким write-lock'ом: после
            // резолюции маркета TP-лимитка больше не имеет смысла, обнуляем
            // её в позиции (любой будущий код увидит `None`). Если кто-то
            // успел дёрнуть `tp_order_id.take()` раньше — пропускаем.
            let (pos_id, tp_id) = {
                let mut pos_w = pos_arc.write().await;
                let pid = pos_w.id.clone();
                match pos_w.tp_order_id.take() {
                    Some(t) => (pid, t),
                    None => continue,
                }
            };
            let request = CancelOrderRequest {
                order_id: tp_id.clone(),
                timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            };
            match cancel_order_on_clob(&account, request).await {
                Ok(res) => {
                    crate::tee_println!(
                        "[account_submit] TP cancel after resolution: pos_id={pos_id}, order_id={tp_id}, canceled={}, error_msg={:?}",
                        res.canceled, res.error_msg,
                    );
                }
                Err(err) => {
                    crate::tee_eprintln!(
                        "[account_submit] TP cancel after resolution упал: pos_id={pos_id}, tp_order_id={tp_id}: {err:#}"
                    );
                }
            }
        }
    });
}

// Тихонько подсказываем компилятору, что `Arc` нам нужен (поля `Account.*`
// — это `Arc<RwLock<…>>`, мы их не клонируем напрямую, но они подразумеваются).
const _: fn(SharedAccount) -> Arc<crate::account::Account> = |a| a;
