//! User-WS листенер CLOB-канала Polymarket
//! (`wss://ws-subscriptions-clob.polymarket.com/ws/user`,
//! см. <https://docs.polymarket.com/api-reference/wss/user>).
//!
//! Один таск на процесс: подписывается на real-time `order`/`trade` события
//! аутентифицированной сессии и переводит статусы
//! [`crate::history_sim::OpenPosition::open_status`] /
//! [`crate::history_sim::ClosingPosition::close_status`]. Креды берутся из
//! [`crate::account::Account::clob_authed`], поэтому в RealSim вызывают
//! [`crate::account::try_authenticate_clob_for_heartbeats`] в `main` до
//! [`crate::account::spawn_heartbeat`] и этого листенера. [`spawn_user_ws_listener`]
//! всё равно poll'ит появление auth'а на первых секундах (тесты, нестандартный
//! порядок спавна).
//!
//! Архитектурно копирует `data_ws::run_persistent_interval_market_ws_inner`:
//! PING каждые [`USER_WS_PING_INTERVAL_SECS`], watchdog по тишине,
//! авто-реконнект с задержкой [`USER_WS_RECONNECT_DELAY_SECS`].

use crate::account::SharedAccount;
use crate::history_sim::{ClosingPositionStatus, OpenPositionStatus};
use crate::util::current_timestamp_ms;
use futures_util::{SinkExt, StreamExt};
use polymarket_client_sdk::auth::Credentials;
use polymarket_client_sdk::auth::ExposeSecret as _;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::{interval, sleep, MissedTickBehavior};
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Endpoint user-канала Polymarket CLOB для real-time подтверждения
/// постановок/исполнений ордеров аутентифицированной сессии.
/// См. <https://docs.polymarket.com/api-reference/wss/user>.
const POLYMARKET_USER_WS_URL: &str =
    "wss://ws-subscriptions-clob.polymarket.com/ws/user";

/// Период PING-сообщения в user-канал. По доке Polymarket — каждые 10s,
/// сервер отвечает `PONG`. Без heartbeat'а сервер закроет соединение.
const USER_WS_PING_INTERVAL_SECS: u64 = 10;

/// Watchdog: если от сервера не было сообщений (включая PONG) дольше
/// этого порога — форсируем реконнект. 25s = два PING-цикла + запас на
/// сетевую задержку.
const USER_WS_STALE_MAX_AGE_MS: i64 = 25_000;

/// Тик watchdog'а — раз в столько секунд проверяем `last_message_wall_ms`.
const USER_WS_WATCHDOG_INTERVAL_SECS: u64 = 5;

/// Пауза перед реконнектом (после disconnect / connect-error). Та же
/// константа, что в `data_ws::WS_RECONNECT_DELAY_SECS`, специально не
/// разделяем — каналы независимые.
const USER_WS_RECONNECT_DELAY_SECS: u64 = 3;

/// Сколько ждём появления `Account.clob_authed` на старте таска
/// (в RealSim auth уже сделан в `main` до спавна; poll остаётся для
/// редких кейсов).
/// Без auth'а user-WS поднять нельзя — CLOB API key/secret/passphrase
/// нужны в самом subscribe-сообщении.
const USER_WS_WAIT_AUTH_INTERVAL_MS: u64 = 250;

/// Хард-кап ожидания auth'а; если за это время `clob_authed` остался
/// `None` (нет `POLY_PRIVATE_KEY` или `authenticate()` упал уже в `main`),
/// ниже таск [`panic!`]'ует — см. [`spawn_user_ws_listener`].
const USER_WS_WAIT_AUTH_MAX_SECS: u64 = 30;

/// Глобальный user-WS таск на процесс (один на аутентифицированную
/// CLOB-сессию): подписывается на канал
/// `wss://ws-subscriptions-clob.polymarket.com/ws/user` с CLOB API
/// credentials из [`Account::clob_authed`] и читает real-time
/// `order`/`trade` события для апдейта статусов
/// [`crate::history_sim::OpenPosition::open_status`] /
/// [`crate::history_sim::ClosingPosition::close_status`] через
/// [`apply_user_ws_event_value`].
///
/// Сейчас в `history_sim`/`real_sim` ордера не ставятся, поэтому
/// `open_order_id`/`close_order_id` всегда `None` — листенер просто
/// логирует входящие events. Когда поверх будет real CLOB
/// `post_order` — каждый поставленный ордер запишет свой `id` в
/// `OpenPosition.open_order_id` (BUY) или `ClosingPosition.close_order_id`
/// (SELL), и тогда колбек начнёт переводить статусы (см.
/// [`apply_user_ws_event_value`]).
pub fn spawn_user_ws_listener(account: SharedAccount) {
    tokio::spawn(async move {
        // Polymarket CLOB user-WS требует API key/secret/passphrase в самом
        // subscribe-сообщении. Auth в RealSim — в `main` до спавна (см.
        // [`crate::account::try_authenticate_clob_for_heartbeats`]). Здесь
        // ждём `Account::clob_authed`, чтобы забрать креды под subscribe
        // (поллинг остаётся для нестандартных entrypoints / тестов).
        // По таймауту `USER_WS_WAIT_AUTH_MAX_SECS` ниже — паника: нет ключей /
        // `authenticate()` упал уже в `main`, а канал статусов без кредитов
        // только замаскирует мисконфиг.
        let credentials = match wait_for_clob_credentials(&account).await {
            Some(creds) => creds,
            None => {
                // Без auth'а user-WS бесполезен (нет ордеров на матчинг),
                // но молча выйти нельзя: в `RealSim` пайплайн рассчитывает
                // на live-канал статусов, и молчащий таск маскирует
                // мисконфигурацию (отсутствие `POLY_PRIVATE_KEY`, упавший
                // `authenticate()`) — паникуем, чтобы это бросалось в
                // глаза в логах. `tokio::spawn` пропустит панику в
                // task-handler, но `tee_eprintln` ниже + tokio runtime
                // logs покажут причину.
                crate::tee_eprintln!(
                    "[user_ws] auth не появился за {USER_WS_WAIT_AUTH_MAX_SECS}s — паникую: проверьте POLY_PRIVATE_KEY и [heartbeat] CLOB authenticate в логах",
                );
                panic!(
                    "[user_ws] CLOB auth не поднялся за {USER_WS_WAIT_AUTH_MAX_SECS}s — user-канал не запускается"
                );
            }
        };

        crate::tee_println!(
            "[user_ws] стартую: connect {POLYMARKET_USER_WS_URL}",
        );

        loop {
            if let Err(err) = run_user_ws_session(&account, &credentials).await {
                crate::tee_eprintln!("[user_ws] сессия упала: {err:#}");
            } else {
                crate::tee_eprintln!("[user_ws] сессия закрылась — реконнектимся");
            }
            sleep(Duration::from_secs(USER_WS_RECONNECT_DELAY_SECS)).await;
        }
    });
}

/// Поллим `Account.clob_authed` (ArcSwap) до появления `Some(_)` либо до таймаута.
/// `Credentials` клонируем (это `Clone`-структура с `SecretString` внутри),
/// чтобы не держать `Guard<Arc<…>>` через всю жизнь WS-сессии.
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

/// Одна WS-сессия user-канала: connect → subscribe → loop(read/ping/watchdog).
/// Возвращает `Ok(())` если сессия штатно закрылась (надо реконнектиться),
/// `Err` — на ошибке connect'а или send'а subscribe.
async fn run_user_ws_session(
    account: &SharedAccount,
    credentials: &Credentials,
) -> anyhow::Result<()> {
    let (ws_stream, _http_response) = connect_async(POLYMARKET_USER_WS_URL).await?;
    let (mut write, mut read) = ws_stream.split();

    // Subscribe-сообщение: формат описан в asyncapi
    // (см. <https://docs.polymarket.com/api-reference/wss/user>).
    // `markets: []` (или отсутствие поля) — получаем events по всем маркетам
    // этой API-key, без фильтра. Фильтровать по `condition_id` пока нечем
    // (в `Account` нет per-market подписки), да и не нужно: лишний шум
    // отсекается на уровне `apply_user_ws_event_value` по `open_order_id`/
    // `close_order_id`.
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
    // Первый PING — мгновенно после первого тика; не пропускаем, чтобы
    // быстро закрепить «I'm alive» на стороне сервера.

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
                // По доке Polymarket клиент шлёт литеральный текст "PING",
                // сервер отвечает "PONG". WebSocket-уровневый ping не
                // используется — сервер ждёт именно строку.
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
                        // Сервер шлёт либо литеральный "PONG" (text), либо
                        // JSON с event'ом (одиночный объект или массив).
                        // PONG отбрасываем; всё остальное — в parser.
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

/// Парсит JSON-payload user-канала и применяет каждое событие к [`Account`]
/// через [`apply_user_ws_event_value`]. Polymarket иногда оборачивает
/// несколько событий в массив (как `data_ws`), поэтому поддерживаем и
/// одиночный объект, и массив.
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

/// Применяет одно событие user-канала к состоянию [`Account`]: матчит
/// `id` (для `order` events) или `taker_order_id` + maker order ids
/// (для `trade` events) против `open_order_id`/`close_order_id` в
/// [`Account::positions`] / [`Account::pending_resolution`] /
/// [`Account::closing`] и переводит статусы по таблице ниже.
///
/// Order events (`event_type = "order"`):
/// - `type = PLACEMENT`, `status = LIVE` — ордер вышел в книгу. Для
///   BUY оставляем `PendingOpen`; для SELL — `PendingClose` (статус не
///   меняется, лишь подтверждаем что ордер существует).
/// - `type = UPDATE`, `status = MATCHED` — полное исполнение. BUY →
///   `Open`, SELL → `Closed` (PnL должен прийти параллельно через
///   `trade` event либо посчитаться из локальной книги).
/// - `type = CANCELLATION`, `status = CANCELED` — ордер отменён.
///   BUY → `OpenFailed`, SELL → `CloseFailed`.
///
/// Trade events (`event_type = "trade"`, `status` ∈ MATCHED/MINED/
/// CONFIRMED/RETRYING/FAILED): сейчас используем как доп. сигнал
/// «ордер действительно сел на цепочку» — статус `Open`/`Closed` уже
/// мог быть выставлен по `order` event'у с `MATCHED`, дублирующее
/// присвоение идемпотентно.
///
/// **Сейчас это no-op** для всех событий: real CLOB `post_order` ещё не
/// подключён, `open_order_id`/`close_order_id` везде `None`, ни одно
/// событие не найдёт совпадения. Логирование сохраняем, чтобы видеть
/// поток событий в `tee_log` ради отладки сетевого слоя.
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
            let new_open = order_status_to_open_position_status(order_kind, order_status);
            let new_close = order_status_to_closing_position_status(order_kind, order_status);
            update_position_statuses(account, order_id, new_open, new_close).await;
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
            crate::tee_println!(
                "[user_ws] trade: id={trade_id} taker={taker_order_id} status={trade_status} trader_side={trader_side}",
            );
            // На trade event обновляем статусы по `taker_order_id` И по всем
            // `maker_orders[].order_id` — кто-то из них окажется нашим
            // (либо мы taker, либо maker — sdk не различает).
            let new_open = trade_status_to_open_position_status(trade_status);
            let new_close = trade_status_to_closing_position_status(trade_status);
            if !taker_order_id.is_empty() {
                update_position_statuses(account, taker_order_id, new_open, new_close).await;
            }
            if let Some(makers) = value.get("maker_orders").and_then(Value::as_array) {
                for maker in makers {
                    let maker_order_id = maker
                        .get("order_id")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if maker_order_id.is_empty() {
                        continue;
                    }
                    update_position_statuses(account, maker_order_id, new_open, new_close).await;
                }
            }
        }
        _ => {
            // Прочие event_types (PING/PONG не сюда — они отрезаны выше).
            // Логируем кратко на случай добавления новых типов сервером.
            crate::tee_eprintln!("[user_ws] unknown event_type={event_type}");
        }
    }
}

/// Маппинг `(order.type, order.status)` → новый [`OpenPositionStatus`]
/// (если события касаются BUY-ордера). `None` — статус не меняется.
fn order_status_to_open_position_status(
    order_type: &str,
    order_status: &str,
) -> Option<OpenPositionStatus> {
    match (order_type, order_status) {
        // Полное исполнение BUY-ордера: позиция реально открылась.
        ("UPDATE" | "PLACEMENT", "MATCHED") => Some(OpenPositionStatus::Open),
        // Отмена BUY-ордера до fill'а — позиция не открылась.
        ("CANCELLATION", _) | (_, "CANCELED") => Some(OpenPositionStatus::OpenFailed),
        _ => None,
    }
}

/// Маппинг `(order.type, order.status)` → новый [`ClosingPositionStatus`]
/// (если события касаются SELL-ордера).
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

/// Маппинг `trade.status` → [`OpenPositionStatus`] (доп. сигнал сверх
/// `order` event'а; идемпотентен с ним).
fn trade_status_to_open_position_status(trade_status: &str) -> Option<OpenPositionStatus> {
    match trade_status {
        "MATCHED" | "MINED" | "CONFIRMED" => Some(OpenPositionStatus::Open),
        "FAILED" => Some(OpenPositionStatus::OpenFailed),
        _ => None,
    }
}

/// Маппинг `trade.status` → [`ClosingPositionStatus`].
fn trade_status_to_closing_position_status(
    trade_status: &str,
) -> Option<ClosingPositionStatus> {
    match trade_status {
        "MATCHED" | "MINED" | "CONFIRMED" => Some(ClosingPositionStatus::Closed),
        "FAILED" => Some(ClosingPositionStatus::CloseFailed),
        _ => None,
    }
}

/// Сканирует [`Account::positions`] / [`Account::pending_resolution`] /
/// [`Account::closing`] на предмет совпадения `order_id` и переводит
/// статусы, если переданный `new_open`/`new_close` не `None`. Работает
/// под `account.write()`; вызывается **после** парсинга и логирования,
/// чтобы лок брался максимально кратко.
///
/// Сейчас всегда no-match (real-ордера не ставятся), но pipeline готов:
/// как только `OpenPosition.open_order_id` начнёт заполняться, эта
/// функция начнёт переводить статусы.
async fn update_position_statuses(
    account: &SharedAccount,
    order_id: &str,
    new_open: Option<OpenPositionStatus>,
    new_close: Option<ClosingPositionStatus>,
) {
    if new_open.is_none() && new_close.is_none() {
        return;
    }

    if let Some(status) = new_open {
        // Lock order: positions → pending_resolution (как в `Account` doc).
        let mut positions = account.positions.write().await;
        let mut pending_resolution = account.pending_resolution.write().await;
        let mut hit = false;
        for vec in positions.values_mut().chain(pending_resolution.values_mut()) {
            for pos in vec.iter_mut() {
                if pos.open_order_id.as_deref() == Some(order_id) {
                    pos.open_status = status;
                    hit = true;
                }
            }
        }
        if hit {
            crate::tee_println!(
                "[user_ws] open_status({order_id}) → {status:?}",
            );
        }
    }

    if let Some(status) = new_close {
        let mut closing = account.closing.write().await;
        let mut hit = false;
        for vec in closing.values_mut() {
            for c in vec.iter_mut() {
                if c.close_order_id.as_deref() == Some(order_id) {
                    c.close_status = status;
                    hit = true;
                }
            }
        }
        if hit {
            crate::tee_println!(
                "[user_ws] close_status({order_id}) → {status:?}",
            );
        }
    }
}
