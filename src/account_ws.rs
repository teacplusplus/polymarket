//! User-WebSocket CLOB (`POLYMARKET_USER_WS_URL`): `order`/`trade` → статусы позиций и fills.
//! Креды из `Account.clob_authed` ([`crate::account::try_authenticate_clob_for_heartbeats`] перед спавном).
//! Каркас как у interval/market WS: PING ([`USER_WS_PING_INTERVAL_SECS`]), watchdog ([`USER_WS_STALE_MAX_AGE_MS`]), reconnect ([`USER_WS_RECONNECT_DELAY_SECS`]).
use crate::account::SharedAccount;
use crate::account_order_completion::{
    accumulate_invoke_from_ws_trade, notify_terminal_ws_order_snapshot,
    ws_trade_status_for_invoke_book_match, ws_trade_status_settled_on_chain,
    ws_trade_status_terminal_failed,
};
use crate::util::current_timestamp_ms;
use futures_util::{SinkExt, StreamExt};
use polymarket_client_sdk::auth::Credentials;
use polymarket_client_sdk::auth::ExposeSecret as _;
use serde_json::{Value, json};
use std::time::Duration;
use tokio::time::{MissedTickBehavior, interval, sleep};
use tokio_tungstenite::tungstenite::Message;

use crate::ws_connect::{connect_async_maybe_proxy, ws_proxy_from_env};

/// URL user-канала (<https://docs.polymarket.com/api-reference/wss/user>).
const POLYMARKET_USER_WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/user";

/// Интервал текстового PING (сервер отвечает PONG).
const USER_WS_PING_INTERVAL_SECS: u64 = 10;

/// Нет входящих сообщений дольше этого → реконнект (~2× PING + запас).
const USER_WS_STALE_MAX_AGE_MS: i64 = 25_000;

/// Период проверки «стейла» последнего сообщения.
const USER_WS_WATCHDOG_INTERVAL_SECS: u64 = 5;

/// Пауза перед следующим connect после обрыва.
const USER_WS_RECONNECT_DELAY_SECS: u64 = 3;

/// Poll `clob_authed` при ожидании ключей.
const USER_WS_WAIT_AUTH_INTERVAL_MS: u64 = 250;

/// Лимит ожидания auth на первой итерации (`spawn_user_ws_listener` иначе panic).
const USER_WS_WAIT_AUTH_MAX_SECS: u64 = 30;

/// Спавнит единственный user-WS таск: `connect` [`POLYMARKET_USER_WS_URL`], subscribe с [`Account::clob_authed`], цикл через [`run_user_ws_session`] и паузу [`USER_WS_RECONNECT_DELAY_SECS`].
pub fn spawn_user_ws_listener(account: SharedAccount) {
    tokio::spawn(async move {
        // Первая итерация: долго ждём `clob_authed`, иначе panic (мисконфиг); дальше — короткий poll + re-auth из heartbeat без panic.
        crate::tee_println!("[user_ws] стартую: connect {POLYMARKET_USER_WS_URL}",);
        let mut first_iteration = true;
        loop {
            // На каждом reconnect заново берём ключи из ArcSwap (force-reauth).
            let credentials = match wait_for_clob_credentials(&account).await {
                Some(creds) => creds,
                None => {
                    if first_iteration {
                        crate::tee_eprintln!(
                            "[user_ws] auth не появился за {USER_WS_WAIT_AUTH_MAX_SECS}s — паникую: проверьте POLY_PRIVATE_KEY и [heartbeat] CLOB authenticate в логах",
                        );
                        panic!(
                            "[user_ws] CLOB auth не поднялся за {USER_WS_WAIT_AUTH_MAX_SECS}s — user-канал не запускается"
                        );
                    } else {
                        crate::tee_eprintln!(
                            "[user_ws] reconnect-итерация: clob_authed=None в момент wait_for_clob_credentials \
                             — ждём re-auth (sleep {USER_WS_RECONNECT_DELAY_SECS}s)"
                        );
                        sleep(Duration::from_secs(USER_WS_RECONNECT_DELAY_SECS)).await;
                        continue;
                    }
                }
            };
            first_iteration = false;
            if let Err(err) = run_user_ws_session(&account, &credentials).await {
                crate::tee_eprintln!("[user_ws] сессия упала: {err:#}");
            } else {
                crate::tee_eprintln!("[user_ws] сессия закрылась — реконнектимся");
            }
            sleep(Duration::from_secs(USER_WS_RECONNECT_DELAY_SECS)).await;
        }
    });
}

/// Ждём `clob_authed` до [`USER_WS_WAIT_AUTH_MAX_SECS`]; возвращает клонированные [`Credentials`] (не держим Guard через сессию).
async fn wait_for_clob_credentials(account: &SharedAccount) -> Option<Credentials> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(USER_WS_WAIT_AUTH_MAX_SECS);
    loop {
        if let Some(authed) = (**account.clob_authed.load()).as_ref() {
            return Some(authed.credentials().clone());
        }
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        sleep(Duration::from_millis(USER_WS_WAIT_AUTH_INTERVAL_MS)).await;
    }
}

/// Одна жизнь сокета: connect → subscribe → read + PING [`USER_WS_PING_INTERVAL_SECS`] + watchdog [`USER_WS_STALE_MAX_AGE_MS`]. `Ok` — порвать и реконнектить; `Err` — ошибка connect или отправки subscribe.
async fn run_user_ws_session(
    account: &SharedAccount,
    credentials: &Credentials,
) -> anyhow::Result<()> {
    let proxy = ws_proxy_from_env();
    let (ws_stream, _http_response) =
        connect_async_maybe_proxy(POLYMARKET_USER_WS_URL, proxy.as_ref()).await?;
    let (mut write, mut read) = ws_stream.split();

    // Без фильтра маркетов; отбор по нашим order_id в apply.
    let subscribe_payload = json!({
        "auth": {
            "apiKey": credentials.key().to_string(),
            "secret": credentials.secret().expose_secret(),
            "passphrase": credentials.passphrase().expose_secret(),
        },
        "type": "user",
    });
    write
        .send(Message::Text(subscribe_payload.to_string()))
        .await?;

    crate::tee_println!(
        "[user_ws] подписан (apiKey={}); ждём order/trade events",
        credentials.key(),
    );

    let mut ping_tick = interval(Duration::from_secs(USER_WS_PING_INTERVAL_SECS));
    ping_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);

    let mut watchdog = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_secs(USER_WS_WATCHDOG_INTERVAL_SECS),
        Duration::from_secs(USER_WS_WATCHDOG_INTERVAL_SECS),
    );
    watchdog.set_missed_tick_behavior(MissedTickBehavior::Delay);

    let mut last_message_wall_ms = current_timestamp_ms();

    loop {
        tokio::select! {
            biased;
            _ = ping_tick.tick() => {
                // Текстовый PING (не WS frame ping).
                if write.send(Message::Text("PING".into())).await.is_err() {
                    crate::tee_eprintln!("[user_ws] PING send упал — реконнект");
                    return Ok(());
                }
            }
            _ = watchdog.tick() => {
                let age_ms = current_timestamp_ms() - last_message_wall_ms;
                if age_ms > USER_WS_STALE_MAX_AGE_MS {
                    crate::tee_eprintln!(
                        "[user_ws] watchdog: тишина {age_ms}ms — форсирую реконнект",
                    );
                    return Ok(());
                }
            }
            msg = read.next() => {
                match msg {
                    None => {
                        crate::tee_eprintln!("[user_ws] стрим завершён — реконнект");
                        return Ok(());
                    }
                    Some(Err(err)) => {
                        crate::tee_eprintln!("[user_ws] read error: {err} — реконнект");
                        return Ok(());
                    }
                    Some(Ok(Message::Text(text))) => {
                        last_message_wall_ms = current_timestamp_ms();
                        // PONG — не JSON.
                        if text.trim() == "PONG" {
                            continue;
                        }
                        ingest_user_ws_payload(account, &text).await;
                    }
                    Some(Ok(Message::Binary(bin))) => {
                        last_message_wall_ms = current_timestamp_ms();
                        if let Ok(text) = String::from_utf8(bin.to_vec()) {
                            if text.trim() == "PONG" {
                                continue;
                            }
                            ingest_user_ws_payload(account, &text).await;
                        }
                    }
                    Some(Ok(Message::Ping(payload))) => {
                        last_message_wall_ms = current_timestamp_ms();
                        let _ = write.send(Message::Pong(payload)).await;
                    }
                    Some(Ok(Message::Pong(_))) => {
                        last_message_wall_ms = current_timestamp_ms();
                    }
                    Some(Ok(Message::Close(_))) => {
                        crate::tee_eprintln!("[user_ws] сервер закрыл соединение");
                        return Ok(());
                    }
                    Some(Ok(_)) => {
                        last_message_wall_ms = current_timestamp_ms();
                    }
                }
            }
        }
    }
}

/// Один объект или массив JSON из user-WS → каждое событие в [`apply_user_ws_event_value`].
async fn ingest_user_ws_payload(account: &SharedAccount, raw: &str) {
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        crate::tee_eprintln!("[user_ws] не-JSON payload: {raw}");
        return;
    };
    if let Some(arr) = value.as_array() {
        for event in arr {
            apply_user_ws_event_value(account, event).await;
        }
    } else {
        apply_user_ws_event_value(account, &value).await;
    }
}

/// Матч `order.id` или `trade.taker`/`maker_orders[*].order_id` к нашим ids; апдейт статусов и fills при submit.
///
/// **`order`:** MATCHED BUY → [`OpenPositionStatus::Open`], SELL → [`ClosingPositionStatus::Closed`]; CANCELED → failed; при `PendingOpen → Open` спавним [`crate::account_submit::try_place_tp_maker`].
///
/// **`trade`:** те же переходы статусов; на терминальном статусе — [`apply_user_ws_trade_fill`] (накопление, PnL при закрытии).
async fn apply_user_ws_event_value(account: &SharedAccount, value: &Value) {
    let event_type = value
        .get("event_type")
        .and_then(Value::as_str)
        .unwrap_or("?");

    match event_type {
        "order" => {
            let order_id = value.get("id").and_then(Value::as_str).unwrap_or("");
            let side = value.get("side").and_then(Value::as_str).unwrap_or("?");
            let order_status = value.get("status").and_then(Value::as_str).unwrap_or("?");
            let order_kind = value.get("type").and_then(Value::as_str).unwrap_or("?");
            // `original_size`/`size_matched` приходят строками-десятичными в `OrderMessage`
            // user-канала. Парсим их и пробрасываем в invoke-агрегатор: его гейт
            // book-fully-matched (см. `is_book_fully_matched_observed` в
            // `account_order_completion.rs`) использует именно их, чтобы `MATCHED` от
            // Polymarket не выстрелил колбэк прематурно для partial maker'а.
            let original_size = parse_decimal_str(value.get("original_size"));
            let size_matched = parse_decimal_str(value.get("size_matched"));
            crate::tee_println!(
                "[user_ws] order: id={order_id} side={side} type={order_kind} status={order_status} \
                 original_size={original_size:?} size_matched={size_matched:?}",
            );
            if order_id.is_empty() {
                return;
            }
            notify_terminal_ws_order_snapshot(
                &account.order_invoke_hub,
                order_id,
                order_status,
                original_size,
                size_matched,
            )
            .await;
        }
        "trade" => {
            let trade_id = value.get("id").and_then(Value::as_str).unwrap_or("");
            let taker_order_id = value
                .get("taker_order_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let trade_status = value.get("status").and_then(Value::as_str).unwrap_or("?");
            let trader_side = value
                .get("trader_side")
                .and_then(Value::as_str)
                .unwrap_or("?");
            let side = value.get("side").and_then(Value::as_str).unwrap_or("?");
            crate::tee_println!(
                "[user_ws] trade: id={trade_id} taker={taker_order_id} side={side} status={trade_status} trader_side={trader_side}",
            );
            // Сначала taker-статусы (чтобы SELL уже `Closed` к fill); затем partial/full fill; maker_orders — возможный наш TP.
            let trade_size = parse_decimal_str(value.get("size"));
            let trade_price = parse_decimal_str(value.get("price"));
            // `fee_rate_bps` per-trade — пробрасываем в invoke-агрегатор, чтобы
            // `making_amount`/`taking_amount` в финальном колбэке были **net of fee**.
            // Отсутствует → 0 bps (для большинства маркетов Polymarket V2 сейчас так).
            let trade_fee_rate_bps = parse_decimal_str(value.get("fee_rate_bps")).unwrap_or(0.0);
            // `is_book_terminal`: трейд достоин учёта в book-match агрегате — это любой статус
            // его жизненного цикла, включая terminal-failed (`MATCHED|RETRYING|MINED|
            // CONFIRMED|FAILED`). Дедуп по `trade_id` в `record_trade_aggregate_from_ws_event`
            // гарантирует, что повторные события одного `trade_id` не дают двойного счёта.
            // `is_settled_on_chain`: настоящий on-chain факт (`MINED|CONFIRMED`) —
            // только этот сигнал гейтит финальный `success=true` колбэка.
            // `is_terminal_failed`: релайер сдался (`FAILED`) — on-chain ничего не зачислится;
            // учитывается как terminal-объём в `settlement_caught_up_with_match`, иначе при
            // race «CANCELED + один трейд завис на чейне как Failed» агрегатор бы зависал.
            let is_book_terminal = ws_trade_status_for_invoke_book_match(trade_status);
            let is_settled_on_chain = ws_trade_status_settled_on_chain(trade_status);
            let is_terminal_failed = ws_trade_status_terminal_failed(trade_status);
            if !taker_order_id.is_empty() {
                if is_book_terminal && let (Some(size), Some(price)) = (trade_size, trade_price) {
                    accumulate_invoke_from_ws_trade(
                        &account.order_invoke_hub,
                        taker_order_id,
                        trade_id,
                        size,
                        price,
                        trade_fee_rate_bps,
                        is_settled_on_chain,
                        is_terminal_failed,
                    )
                    .await;
                }
            }
            if let Some(makers) = value.get("maker_orders").and_then(Value::as_array) {
                for maker in makers {
                    let maker_order_id =
                        maker.get("order_id").and_then(Value::as_str).unwrap_or("");
                    if maker_order_id.is_empty() {
                        continue;
                    }
                    let maker_size = parse_decimal_str(maker.get("matched_amount"));
                    let maker_price = parse_decimal_str(maker.get("price"));
                    // Maker fee_rate_bps может отличаться от taker'a — берём maker'ское, иначе общий fallback.
                    let maker_fee_rate_bps =
                        parse_decimal_str(maker.get("fee_rate_bps")).unwrap_or(trade_fee_rate_bps);
                    // TP по maker: open-ветку не триггерим; см. debug_assert ниже.
                    if is_book_terminal && let (Some(size), Some(price)) = (maker_size, maker_price)
                    {
                        accumulate_invoke_from_ws_trade(
                            &account.order_invoke_hub,
                            maker_order_id,
                            trade_id,
                            size,
                            price,
                            maker_fee_rate_bps,
                            is_settled_on_chain,
                            is_terminal_failed,
                        )
                        .await;
                    }
                }
            }
        }
        _ => {
            crate::tee_eprintln!("[user_ws] unknown event_type={event_type}");
        }
    }
}

/// `f64` из строки WS; `None` если пусто или не парсится.
fn parse_decimal_str(v: Option<&Value>) -> Option<f64> {
    let s = v.and_then(Value::as_str)?;
    if s.is_empty() {
        return None;
    }
    s.parse::<f64>().ok().filter(|x| x.is_finite())
}

