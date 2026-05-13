//! User-WebSocket CLOB (`POLYMARKET_USER_WS_URL`): `order`/`trade` → статусы позиций и fills.
//! Креды из `Account.clob_authed` ([`crate::account::try_authenticate_clob_for_heartbeats`] перед спавном).
//! Каркас как у interval/market WS: PING ([`USER_WS_PING_INTERVAL_SECS`]), watchdog ([`USER_WS_STALE_MAX_AGE_MS`]), reconnect ([`USER_WS_RECONNECT_DELAY_SECS`]).
use crate::account::SharedAccount;
use crate::account_order_completion::{
    accumulate_invoke_from_ws_trade, notify_terminal_ws_order_snapshot,
};
use crate::account_order::OrderRole;
use crate::history_sim::{CloseReason, ClosingPosition, ClosingPositionStatus, OpenPositionStatus};
use crate::util::current_timestamp_ms;
use futures_util::{SinkExt, StreamExt};
use polymarket_client_sdk::auth::Credentials;
use polymarket_client_sdk::auth::ExposeSecret as _;
use serde_json::{Value, json};
use std::time::Duration;
use tokio::time::{MissedTickBehavior, interval, sleep};
use tokio_tungstenite::{connect_async, tungstenite::Message};

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
    let (ws_stream, _http_response) = connect_async(POLYMARKET_USER_WS_URL).await?;
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
            crate::tee_println!(
                "[user_ws] order: id={order_id} side={side} type={order_kind} status={order_status}",
            );
            if order_id.is_empty() {
                return;
            }
            notify_terminal_ws_order_snapshot(
                &account.order_invoke_hub,
                order_id,
                order_status,
            )
            .await;
            let new_open = order_status_to_open_position_status(order_kind, order_status);
            let new_close = order_status_to_closing_position_status(order_kind, order_status);
            // После перехода в Open триггерим TP (идемпотентно внутри позиции).
            let trigger_tp_arcs =
                update_position_statuses(account, order_id, new_open, new_close).await;
            for pos_arc in trigger_tp_arcs {
                let acc = account.clone();
                tokio::spawn(async move {
                    crate::account_submit::try_place_tp_maker(acc, pos_arc).await;
                });
            }
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
            let fee_rate_bps = parse_decimal_str(value.get("fee_rate_bps")).unwrap_or(0.0);
            let new_open = trade_status_to_open_position_status(trade_status);
            let new_close = trade_status_to_closing_position_status(trade_status);
            let is_terminal = matches!(trade_status, "MATCHED" | "MINED" | "CONFIRMED");
            let mut trigger_tp_arcs_for_taker: Vec<crate::history_sim::SharedOpenPosition> =
                Vec::new();
            if !taker_order_id.is_empty() {
                trigger_tp_arcs_for_taker =
                    update_position_statuses(account, taker_order_id, new_open, new_close).await;
                if is_terminal && let (Some(size), Some(price)) = (trade_size, trade_price) {
                    accumulate_invoke_from_ws_trade(
                        &account.order_invoke_hub,
                        taker_order_id,
                        size,
                        price,
                    )
                    .await;
                    apply_user_ws_trade_fill(
                        account,
                        taker_order_id,
                        side,
                        size,
                        price,
                        fee_rate_bps,
                        OrderRole::Taker,
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
                    let maker_fee_bps =
                        parse_decimal_str(maker.get("fee_rate_bps")).unwrap_or(fee_rate_bps);
                    let maker_side = maker.get("side").and_then(Value::as_str).unwrap_or(side);
                    // TP по maker: open-ветку не триггерим; см. debug_assert ниже.
                    let ret =
                        update_position_statuses(account, maker_order_id, new_open, new_close)
                            .await;
                    debug_assert!(
                        ret.is_empty(),
                        "maker_order_id не должен матчиться по open_order_id"
                    );
                    if is_terminal && let (Some(size), Some(price)) = (maker_size, maker_price) {
                        accumulate_invoke_from_ws_trade(
                            &account.order_invoke_hub,
                            maker_order_id,
                            size,
                            price,
                        )
                        .await;
                        apply_user_ws_trade_fill(
                            account,
                            maker_order_id,
                            maker_side,
                            size,
                            price,
                            maker_fee_bps,
                            OrderRole::Maker,
                        )
                        .await;
                    }
                }
            }
            for pos_arc in trigger_tp_arcs_for_taker {
                let acc = account.clone();
                tokio::spawn(async move {
                    crate::account_submit::try_place_tp_maker(acc, pos_arc).await;
                });
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

/// `(order.type, order.status)` → новый BUY-статус; `None` = не трогаем.
fn order_status_to_open_position_status(
    order_type: &str,
    order_status: &str,
) -> Option<OpenPositionStatus> {
    match (order_type, order_status) {
        ("UPDATE" | "PLACEMENT", "MATCHED") => Some(OpenPositionStatus::Open),
        ("CANCELLATION", _) | (_, "CANCELED") => Some(OpenPositionStatus::OpenFailed),
        _ => None,
    }
}

/// Аналог для SELL ([`ClosingPositionStatus`]).
fn order_status_to_closing_position_status(
    order_type: &str,
    order_status: &str,
) -> Option<ClosingPositionStatus> {
    match (order_type, order_status) {
        ("UPDATE" | "PLACEMENT", "MATCHED") => Some(ClosingPositionStatus::Closed),
        ("CANCELLATION", _) | (_, "CANCELED") => Some(ClosingPositionStatus::CloseFailed),
        _ => None,
    }
}

/// Идемпотентно к паре с `order`-событием.
fn trade_status_to_open_position_status(trade_status: &str) -> Option<OpenPositionStatus> {
    match trade_status {
        "MATCHED" | "MINED" | "CONFIRMED" => Some(OpenPositionStatus::Open),
        "FAILED" => Some(OpenPositionStatus::OpenFailed),
        _ => None,
    }
}

/// Аналог закрытия по `trade.status`.
fn trade_status_to_closing_position_status(trade_status: &str) -> Option<ClosingPositionStatus> {
    match trade_status {
        "MATCHED" | "MINED" | "CONFIRMED" => Some(ClosingPositionStatus::Closed),
        "FAILED" => Some(ClosingPositionStatus::CloseFailed),
        _ => None,
    }
}

/// Обновляет `open_order_id` / `close_order_id`; возвращает позиции с только что произошедшим `PendingOpen → Open` для TP.
///
/// См. локи: не держать два inner-lock на одной позиции одновременно.
async fn update_position_statuses(
    account: &SharedAccount,
    order_id: &str,
    new_open: Option<OpenPositionStatus>,
    new_close: Option<ClosingPositionStatus>,
) -> Vec<crate::history_sim::SharedOpenPosition> {
    let mut to_trigger_tp: Vec<crate::history_sim::SharedOpenPosition> = Vec::new();
    if new_open.is_none() && new_close.is_none() {
        return to_trigger_tp;
    }

    if let Some(status) = new_open {
        let positions = account.positions.read().await;
        let mut hit_pos_ids: Vec<String> = Vec::new();
        for vec in positions.values() {
            for pos_arc in vec.iter() {
                let mut pos = pos_arc.write().await;
                if pos.open_order_id.as_deref() == Some(order_id) {
                    let was_pending = matches!(pos.open_status, OpenPositionStatus::PendingOpen);
                    pos.open_status = status;
                    if was_pending && matches!(status, OpenPositionStatus::Open) {
                        to_trigger_tp.push(pos_arc.clone());
                    }
                    hit_pos_ids.push(pos.id.clone());
                }
            }
        }
        if !hit_pos_ids.is_empty() {
            crate::tee_println!(
                "[user_ws] open_status({order_id}) → {status:?} (pos_id={hit_pos_ids:?})",
            );
        }
    }

    if let Some(status) = new_close {
        let closing = account.closing.read().await;
        let mut hit_pairs: Vec<(String, String)> = Vec::new();
        for vec in closing.values() {
            for c_arc in vec.iter() {
                let mut c = c_arc.write().await;
                if c.close_order_id.as_deref() == Some(order_id) {
                    c.close_status = status;
                    let oid_clone = c.close_order_id.clone().unwrap_or_default();
                    let pos_id = c.position.read().await.id.clone();
                    hit_pairs.push((oid_clone, pos_id));
                }
            }
        }
        // Только лог: inner read позиции — после отпускания c.write.
        for (oid, pos_id) in hit_pairs {
            crate::tee_println!("[user_ws] close_status({oid}) → {status:?} (pos_id={pos_id})",);
        }
    }

    to_trigger_tp
}

/// Один BUY/SELL fill по WS: аккумулирует позицию или закрытие; терминально — bankroll через finalize.
async fn apply_user_ws_trade_fill(
    account: &SharedAccount,
    order_id: &str,
    side: &str,
    size: f64,
    price: f64,
    fee_rate_bps: f64,
    role: OrderRole,
) {
    if size <= 0.0 || !size.is_finite() || price <= 0.0 || !price.is_finite() {
        return;
    }
    match side {
        "BUY" => {
            apply_buy_fill(account, order_id, size, price, fee_rate_bps, role).await;
        }
        "SELL" => {
            apply_sell_fill(account, order_id, size, price, fee_rate_bps, role).await;
        }
        _ => {
            crate::tee_eprintln!(
                "[user_ws] неизвестный side={side} в trade fill (order_id={order_id})"
            );
        }
    }
}

/// BUY: аккумулирует `shares_held`/`entry_cost`/`buy_price` по `fee_rate_bps`; первый fill сбрасывает оптимист (`optimistic_fill_replaced`).
async fn apply_buy_fill(
    account: &SharedAccount,
    order_id: &str,
    size: f64,
    price: f64,
    fee_rate_bps: f64,
    _role: OrderRole,
) {
    let positions = account.positions.read().await;
    let mut hit = false;
    for vec in positions.values() {
        for pos_arc in vec.iter() {
            let mut pos = pos_arc.write().await;
            if pos.open_order_id.as_deref() != Some(order_id) {
                continue;
            }
            if !pos.optimistic_fill_replaced {
                pos.shares_held = 0.0;
                pos.entry_cost = 0.0;
                pos.optimistic_fill_replaced = true;
            }
            let usd_paid = size * price;
            let fee_rate = fee_rate_bps / 10_000.0;
            let net_shares = size * (1.0 - fee_rate);
            pos.shares_held += net_shares;
            pos.entry_cost += usd_paid;
            if pos.shares_held > 0.0 {
                pos.buy_price = (pos.entry_cost / pos.shares_held).clamp(0.001, 0.999);
            }
            hit = true;
            crate::tee_println!(
                "[user_ws] BUY fill: pos_id={}, order_id={order_id}, size={size:.4}, price={price:.4}, fee_rate_bps={fee_rate_bps:.2} → shares_held={:.4} (plan {:.4}), entry_cost={:.4} (plan {:.4}), buy_price={:.4} (plan {:.4})",
                pos.id,
                pos.shares_held,
                pos.planned_shares_held,
                pos.entry_cost,
                pos.planned_entry_cost,
                pos.buy_price,
                pos.planned_buy_price,
            );
        }
    }
    if !hit {
        crate::tee_eprintln!(
            "[user_ws] BUY fill: order_id={order_id} не найден ни в одной OpenPosition"
        );
    }
}

/// SELL: `close_order_id` (taker) или TP по `tp_order_id` (maker); при отсутствии записи — создаём [`ClosingPosition`] TP и финализируем.
pub(crate) async fn apply_sell_fill(
    account: &SharedAccount,
    order_id: &str,
    size: f64,
    price: f64,
    fee_rate_bps: f64,
    _role: OrderRole,
) {
    let usd_received = size * price;
    let fee_rate = fee_rate_bps / 10_000.0;
    let net_usdc = usd_received * (1.0 - fee_rate);

    let mut to_finalize: Option<crate::history_sim::SharedClosingPosition> = None;
    let mut hit_pos_arcs: Vec<(crate::history_sim::SharedOpenPosition, f64)> = Vec::new();
    {
        let closing = account.closing.read().await;
        let mut hit = false;
        for vec in closing.values() {
            for c_arc in vec.iter() {
                let mut c = c_arc.write().await;
                if c.close_order_id.as_deref() != Some(order_id) {
                    continue;
                }
                let prev = c.pnl.unwrap_or(0.0);
                // Аккумулируем net USDC; вычитаем вход на finalize при `Closed`.
                c.pnl = Some(prev + net_usdc);
                hit = true;
                hit_pos_arcs.push((c.position.clone(), prev + net_usdc));
                if matches!(c.close_status, ClosingPositionStatus::Closed) {
                    to_finalize = Some(c_arc.clone());
                }
            }
        }
        if hit {
            drop(closing);
            for (pos_arc, acc_proceeds) in hit_pos_arcs {
                let pid = pos_arc.read().await.id.clone();
                crate::tee_println!(
                    "[user_ws] SELL fill (close): pos_id={pid}, order_id={order_id}, size={size:.4}, price={price:.4} → accumulated_proceeds={acc_proceeds:.4}",
                );
            }
            if let Some(c_arc) = to_finalize {
                finalize_close_pnl_in_place(account, c_arc, "Ws").await;
            }
            return;
        }
    }

    // TP maker: ищем позицию по `tp_order_id`, убираем из `positions`, пушим в `closing`.
    let mut maybe_pos: Option<(
        crate::account::LaneKey,
        crate::history_sim::SharedOpenPosition,
    )> = None;
    {
        let mut positions = account.positions.write().await;
        for (key, vec) in positions.iter_mut() {
            let mut idx = 0;
            while idx < vec.len() {
                let is_tp = vec[idx].read().await.tp_order_id.as_deref() == Some(order_id);
                if is_tp {
                    let pos_arc = vec.swap_remove(idx);
                    maybe_pos = Some((key.clone(), pos_arc));
                    break;
                }
                idx += 1;
            }
            if maybe_pos.is_some() {
                break;
            }
        }
    }
    let Some((lane_key, pos_arc)) = maybe_pos else {
        crate::tee_eprintln!(
            "[user_ws] SELL fill: order_id={order_id} не найден ни в ClosingPosition, ни как tp_order_id (возможно, дубль / уже обработан)"
        );
        return;
    };
    let (pos_id, entry_cost, shares_held, existing_closing) = {
        let p = pos_arc.read().await;
        (
            p.id.clone(),
            p.entry_cost,
            p.shares_held,
            p.closing_position
                .as_ref()
                .and_then(std::sync::Weak::upgrade),
        )
    };
    let exit_price = if shares_held > 0.0 {
        net_usdc / shares_held
    } else {
        price
    };
    let pnl = net_usdc - entry_cost;

    if let Some(c_arc) = existing_closing {
        // Гонка с taker-close: подменяем id и reason на TP, finalize найдёт ту же запись.
        {
            let mut c = c_arc.write().await;
            let prev_close_order_id = c.close_order_id.clone();
            let prev_reason = c.reason.clone();
            let prev_status = c.close_status;
            c.close_status = ClosingPositionStatus::Closed;
            c.reason = CloseReason::TakeProfit;
            c.pnl = Some(pnl);
            c.close_order_id = Some(order_id.to_string());
            c.exit_price = exit_price;
            crate::tee_println!(
                "[user_ws] TP maker fill (raced SELL-taker close): pos_id={pos_id}, order_id={order_id}, size={size:.4}, price={price:.4}, net_usdc={net_usdc:.4}, entry_cost={entry_cost:.4}, pnl={pnl:.4} \
                 — morphed existing ClosingPosition: status {prev_status:?} → Closed, reason {prev_reason:?} → TakeProfit, close_order_id {prev_close_order_id:?} → Some({order_id})"
            );
        }
        let _ = lane_key; // уже в `closing`
        finalize_tp_close_after_creation(account, order_id, "Ws").await;
        return;
    }

    crate::tee_println!(
        "[user_ws] TP maker fill: pos_id={pos_id}, order_id={order_id}, size={size:.4}, price={price:.4}, net_usdc={net_usdc:.4}, entry_cost={entry_cost:.4}, pnl={pnl:.4}"
    );
    let c_arc: crate::history_sim::SharedClosingPosition =
        std::sync::Arc::new(tokio::sync::RwLock::new(ClosingPosition {
            position: pos_arc.clone(),       // тот же Arc, что был в positions
            exit_price,                      // VWAP выхода из net/shares
            reason: CloseReason::TakeProfit, // причина — TP
            pnl: Some(pnl),                  // net_usdc − entry_cost (один fill)
            close_status: ClosingPositionStatus::Closed, // сразу финал
            close_order_id: Some(order_id.to_string()), // id ордера TP в CLOB
            close_placement_attempted: true, // submit: не ставить второй SELL
            created_unix_ms: current_timestamp_ms(), // wall ms создания записи
        }));
    // Weak в OpenPosition для polling в submit.
    {
        let mut p = pos_arc.write().await;
        p.set_closing_position(std::sync::Arc::downgrade(&c_arc));
    }
    {
        let mut closing = account.closing.write().await;
        closing.entry(lane_key).or_default().push(c_arc);
    }
    finalize_tp_close_after_creation(account, order_id, "Ws").await;
}

/// `pnl` − `entry_cost`, затем bankroll, drawdown, stats и CSV; идемпотентно через `pnl_finalized` на [`crate::history_sim::OpenPosition`]. Не смешивать два inner-write на одной позиции.
pub(crate) async fn finalize_close_pnl_in_place(
    account: &SharedAccount,
    c_arc: crate::history_sim::SharedClosingPosition,
    finalized_via: &'static str,
) {
    let (pos_arc, raw_pnl) = {
        let c = c_arc.read().await;
        (c.position.clone(), c.pnl.unwrap_or(0.0))
    };
    let (entry_cost, already_finalized, pos_id) = {
        let p = pos_arc.read().await;
        (p.entry_cost, p.pnl_finalized, p.id.clone())
    };
    if already_finalized {
        return;
    }
    let pnl = raw_pnl - entry_cost;
    {
        let mut c = c_arc.write().await;
        c.pnl = Some(pnl);
    }
    {
        let mut p = pos_arc.write().await;
        p.pnl_finalized = true;
    }

    let new_bankroll = {
        let mut bankroll = account.bankroll.write().await;
        *bankroll += pnl;
        *bankroll
    };
    crate::tee_println!(
        "[user_ws] finalize SELL: pos_id={pos_id}, pnl={pnl:.4} → bankroll={new_bankroll:.4}",
    );
    account.update_drawdown(new_bankroll).await;
    record_submit_close_to_csv_and_stats(account, &pos_arc, &c_arc, pnl, finalized_via).await;
}

/// После TP-ветки [`apply_sell_fill`]: bankroll, drawdown, stats, CSV; идемпотентно тем же маркером.
pub(crate) async fn finalize_tp_close_after_creation(
    account: &SharedAccount,
    order_id: &str,
    finalized_via: &'static str,
) {
    let target: Option<(
        crate::history_sim::SharedOpenPosition,
        crate::history_sim::SharedClosingPosition,
        f64,
    )> = {
        let closing = account.closing.read().await;
        let mut found: Option<(
            crate::history_sim::SharedOpenPosition,
            crate::history_sim::SharedClosingPosition,
            f64,
        )> = None;
        'outer: for vec in closing.values() {
            for c_arc in vec.iter() {
                let c = c_arc.read().await;
                if c.close_order_id.as_deref() == Some(order_id) {
                    found = Some((c.position.clone(), c_arc.clone(), c.pnl.unwrap_or(0.0)));
                    break 'outer;
                }
            }
        }
        found
    };
    let Some((pos_arc, c_arc, pnl)) = target else {
        return;
    };
    let pos_id = {
        let mut p = pos_arc.write().await;
        if p.pnl_finalized {
            return;
        }
        p.pnl_finalized = true;
        p.id.clone()
    };
    let new_bankroll = {
        let mut bankroll = account.bankroll.write().await;
        *bankroll += pnl;
        *bankroll
    };
    crate::tee_println!(
        "[user_ws] finalize TP: pos_id={pos_id}, pnl={pnl:.4} → bankroll={new_bankroll:.4}",
    );
    account.update_drawdown(new_bankroll).await;
    record_submit_close_to_csv_and_stats(account, &pos_arc, &c_arc, pnl, finalized_via).await;
}

/// Бамп stats по сторонам и одна строка submit-CSV; короткие read, без write-локов `Account`. No-op без `real_sim_state` или при `unknown` interval/side.
async fn record_submit_close_to_csv_and_stats(
    account: &SharedAccount,
    pos_arc: &crate::history_sim::SharedOpenPosition,
    c_arc: &crate::history_sim::SharedClosingPosition,
    pnl: f64,
    finalized_via: &'static str,
) {
    use crate::xframe::CurrencyUpDownOutcome;
    use crate::xframe::XFrameIntervalKind;

    let (
        pos_id,
        asset_id,
        market_id,
        currency,
        polymarket_url,
        side_idx,
        interval_type,
        raw_pred,
        cal_pred,
        kelly_f,
        planned_buy_price,
        buy_price,
        planned_shares_held,
        shares_held,
        planned_entry_cost,
        entry_cost,
        p_win_ema,
        frames_held,
        event_end_ms,
        event_remaining_ms_at_open,
        open_order_id,
        tp_order_id,
        price_to_beat,
        final_price,
        pnl_top5_shap,
        graph_html_file_uri,
    ) = {
        let p = pos_arc.read().await;
        let open_unix_ms_for_uri = p.event_end_ms.map(|e| e - p.event_remaining_ms_at_open);
        let close_unix_ms_for_uri = Some(crate::util::current_timestamp_ms());
        let side_str_for_uri = crate::history_sim::position_side_label(&p);
        let graph_html_file_uri =
            crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri(&p)
                .map(|bin_path| {
                    crate::xframe_graph_dump::graph_html_trade_file_uri(
                        &bin_path,
                        open_unix_ms_for_uri,
                        close_unix_ms_for_uri,
                        Some(side_str_for_uri),
                    )
                })
                .unwrap_or_default();
        (
            p.id.clone(),
            p.asset_id.clone(),
            p.market_id.clone(),
            p.currency.clone(),
            p.polymarket_url.clone(),
            p.currency_up_down_outcome_at_open,
            p.xframe_interval_type_at_open,
            p.raw_pred_at_open,
            p.cal_pred_at_open,
            p.kelly_f_at_open,
            p.planned_buy_price,
            p.buy_price,
            p.planned_shares_held,
            p.shares_held,
            p.planned_entry_cost,
            p.entry_cost,
            p.p_win_ema,
            p.frames_held,
            p.event_end_ms,
            p.event_remaining_ms_at_open,
            p.open_order_id.clone(),
            p.tp_order_id.clone(),
            p.price_to_beat,
            p.final_price,
            p.pnl_top5_shap_at_open.clone(),
            graph_html_file_uri,
        )
    };

    let (reason, exit_price, close_order_id) = {
        let c = c_arc.read().await;
        (c.reason.clone(), c.exit_price, c.close_order_id.clone())
    };

    let interval_kind = XFrameIntervalKind::from_i32(interval_type);
    let side_outcome = CurrencyUpDownOutcome::from_i32(side_idx);
    let interval_str = interval_kind
        .map(crate::real_sim::interval_label)
        .unwrap_or("unknown");
    let side_str = side_outcome
        .map(crate::real_sim::side_label)
        .unwrap_or("unknown");

    let fill_role: &'static str = if matches!(reason, crate::history_sim::CloseReason::TakeProfit) {
        "Maker"
    } else {
        "Taker"
    };

    if let (Some(interval), Some(side_kind)) = (interval_kind, side_outcome)
        && let Some(state_arc) = account.real_sim_state_for_currency(&currency).await
    {
        let mut state = state_arc.write().await;
        if let Some(stats) = state.stats.get_mut(&interval) {
            let side_stats = match side_kind {
                CurrencyUpDownOutcome::Up => &mut stats.up,
                CurrencyUpDownOutcome::Down => &mut stats.down,
            };
            crate::history_sim::apply_close_to_side_stats(side_stats, &reason, pnl, raw_pred);
        }
    }

    let now_ms = crate::util::current_timestamp_ms();
    let open_unix_ms = event_end_ms.map(|e| e - event_remaining_ms_at_open);
    let close_unix_ms = Some(now_ms);
    let event_remaining_ms_at_close = event_end_ms.map(|e| e - now_ms).unwrap_or(0);
    crate::trade_csv_log::write_submit_trade_csv_row(crate::trade_csv_log::SubmitTradeCsvRow {
        pos_id: &pos_id,                                                        // ид позиции
        polymarket_url: &polymarket_url,                                        // ссылка на рынок
        price_to_beat,                                                          // порог UP/DOWN
        final_price,                                                            // финал для лога
        currency: &currency,                                                    // актив
        interval: interval_str,                                                 // горизонт (строка)
        side: side_str,                                                         // UP/DOWN
        market_id: &market_id,                                                  // Gamma id рынка
        asset_id: &asset_id,                                                    // Gamma id outcome
        exit_reason: crate::history_sim::trade_csv_close_reason_label(&reason), // текст причины
        fill_role,           // Maker (TP) vs Taker
        finalized_via,       // источник финала (Ws, …)
        planned_buy_price,   // план входа
        buy_price,           // факт входа
        planned_shares_held, // план шэров
        shares_held,         // шэры после входа
        planned_entry_cost,  // план USDC входа
        entry_cost,          // факт USDC входа
        exit_price,          // VWAP выхода
        fee_usdc: 0.0,       // колонка: fee уже внутри pnl
        pnl,
        open_order_id: open_order_id.as_deref(), // входной ордер
        tp_order_id: tp_order_id.as_deref(),     // TP лимит
        close_order_id: close_order_id.as_deref(), // ордер закрытия
        raw_pred,                                // сырой прогноз на входе
        cal_pred,                                // откалиброванный
        kelly_f,                                 // kelly на входе
        p_win_ema_at_close: p_win_ema,           // модель перед закрытием
        frames_held,                             // кадров в позиции
        event_remaining_ms_at_open,              // TTL на входе
        event_remaining_ms_at_close,             // TTL при выходе
        open_unix_ms,                            // wall open (если известно)
        close_unix_ms,                           // wall close
        graph_html_file_uri: graph_html_file_uri.as_str(), // URI графа в CSV
        pnl_top5_shap: pnl_top5_shap.as_str(),   // shap топ-5 текстом
    });
}
