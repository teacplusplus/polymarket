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
use crate::account_order::OrderRole;
use crate::history_sim::{CloseReason, ClosingPosition, ClosingPositionStatus, OpenPositionStatus};
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
        // [`crate::account::try_authenticate_clob_for_heartbeats`]). На
        // **первой** итерации ждём `Account::clob_authed` с длинным
        // таймаутом и паникуем при тишине (single-shot мисконфиг-detector);
        // на последующих итерациях reconnect'а просто перечитываем
        // ArcSwap (короткий wait, без паники), чтобы heartbeat-таск с
        // force-reauth (см. [`crate::account::HEARTBEAT_FAILS_BEFORE_REAUTH`])
        // мог подкинуть свежие креды, и subscribe ушёл уже с ними.
        crate::tee_println!(
            "[user_ws] стартую: connect {POLYMARKET_USER_WS_URL}",
        );
        let mut first_iteration = true;
        loop {
            // Перечитываем `clob_authed` на КАЖДОЙ итерации reconnect-loop'а.
            // Это критично для долгих сессий: force-reauth в heartbeat-таске
            // меняет `clob_authed` через ArcSwap; без перечитывания мы бы
            // подключались со старыми (отозванными сервером) ключами.
            let credentials = match wait_for_clob_credentials(&account).await {
                Some(creds) => creds,
                None => {
                    if first_iteration {
                        // Без auth'а user-WS бесполезен — паникуем (см.
                        // подробный комментарий выше).
                        crate::tee_eprintln!(
                            "[user_ws] auth не появился за {USER_WS_WAIT_AUTH_MAX_SECS}s — паникую: проверьте POLY_PRIVATE_KEY и [heartbeat] CLOB authenticate в логах",
                        );
                        panic!(
                            "[user_ws] CLOB auth не поднялся за {USER_WS_WAIT_AUTH_MAX_SECS}s — user-канал не запускается"
                        );
                    } else {
                        // На reconnect'е auth может быть в процессе
                        // force-reauth (heartbeat-таск работает асинхронно
                        // с нами). Не паникуем, просто ждём и пробуем
                        // снова на следующей итерации.
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
            // Трекаем PendingOpen → Open: при таком переходе спавним постановку
            // TP-лимитки (идемпотентно через `tp_placement_attempted`).
            // `update_position_statuses` возвращает Arc'и позиций, которые
            // именно сейчас стали Open, — передаём их в TP-таску напрямую,
            // без повторного поиска по `open_order_id`.
            let trigger_tp_arcs = update_position_statuses(account, order_id, new_open, new_close).await;
            for pos_arc in trigger_tp_arcs {
                let acc = account.clone();
                tokio::spawn(async move {
                    crate::account_submit::try_place_tp_maker(acc, pos_arc).await;
                });
            }
        }
        "trade" => {
            let trade_id = value.get("id").and_then(Value::as_str).unwrap_or("");
            let taker_order_id = value.get("taker_order_id").and_then(Value::as_str).unwrap_or("");
            let trade_status = value.get("status").and_then(Value::as_str).unwrap_or("?");
            let trader_side = value.get("trader_side").and_then(Value::as_str).unwrap_or("?");
            let side = value.get("side").and_then(Value::as_str).unwrap_or("?");
            crate::tee_println!(
                "[user_ws] trade: id={trade_id} taker={taker_order_id} side={side} status={trade_status} trader_side={trader_side}",
            );
            // На trade event:
            // 1) **сначала** обновляем статусы по `taker_order_id` (наш taker
            //    BUY/SELL), чтобы `apply_user_ws_trade_fill` ниже видел
            //    `close_status=Closed` и тут же финализировал PnL без лишнего
            //    раунд-трипа в bankroll.
            // 2) аккумулируем реальные fills (`size`/`price`/`fee_rate_bps`)
            //    в `OpenPosition` (BUY → shares/cost/buy_price, см.
            //    `optimistic_fill_replaced`-флаг) и `ClosingPosition`
            //    (SELL → proceeds → pnl); финализация PnL → bankroll/stats
            //    делается там же на терминальном fill'е.
            // 3) maker_orders[].order_id — там может быть наш TP-ордер.
            let trade_size = parse_decimal_str(value.get("size"));
            let trade_price = parse_decimal_str(value.get("price"));
            let fee_rate_bps = parse_decimal_str(value.get("fee_rate_bps")).unwrap_or(0.0);
            let new_open = trade_status_to_open_position_status(trade_status);
            let new_close = trade_status_to_closing_position_status(trade_status);
            let is_terminal = matches!(trade_status, "MATCHED" | "MINED" | "CONFIRMED");
            let mut trigger_tp_arcs_for_taker: Vec<crate::history_sim::SharedOpenPosition> = Vec::new();
            if !taker_order_id.is_empty() {
                trigger_tp_arcs_for_taker = update_position_statuses(account, taker_order_id, new_open, new_close).await;
                if is_terminal && let (Some(size), Some(price)) = (trade_size, trade_price) {
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
                    let maker_order_id = maker.get("order_id").and_then(Value::as_str).unwrap_or("");
                    if maker_order_id.is_empty() {
                        continue;
                    }
                    let maker_size = parse_decimal_str(maker.get("matched_amount"));
                    let maker_price = parse_decimal_str(maker.get("price"));
                    let maker_fee_bps = parse_decimal_str(maker.get("fee_rate_bps")).unwrap_or(fee_rate_bps);
                    let maker_side = maker.get("side").and_then(Value::as_str).unwrap_or(side);
                    // Возврат `Vec<SharedOpenPosition>` тут заведомо пуст
                    // (maker_order_id — это TP-ордер; он живёт в `tp_order_id`
                    // OpenPosition, а не в `open_order_id`, поэтому
                    // `update_position_statuses` не матчит по нему open-ветку
                    // и Vec остаётся пустым). Закрытие на TP-fill'е финализирует
                    // `apply_user_ws_trade_fill` ниже.
                    // `debug_assert!` фиксирует инвариант: любой будущий
                    // рефакторинг `update_position_statuses`, начавший пушить
                    // что-то в Vec по maker_order_id, упадёт на тестах.
                    let ret = update_position_statuses(account, maker_order_id, new_open, new_close).await;
                    debug_assert!(ret.is_empty(), "maker_order_id не должен матчиться по open_order_id");
                    if is_terminal && let (Some(size), Some(price)) = (maker_size, maker_price) {
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
            // Прочие event_types (PING/PONG не сюда — они отрезаны выше).
            // Логируем кратко на случай добавления новых типов сервером.
            crate::tee_eprintln!("[user_ws] unknown event_type={event_type}");
        }
    }
}

/// Парсит строковое десятичное (Polymarket в WS отдаёт всё как `String`)
/// в `f64`. Возвращает `None` для пустой/невалидной строки.
fn parse_decimal_str(v: Option<&Value>) -> Option<f64> {
    let s = v.and_then(Value::as_str)?;
    if s.is_empty() {
        return None;
    }
    s.parse::<f64>().ok().filter(|x| x.is_finite())
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
/// под write-локами; вызывается **после** парсинга и логирования,
/// чтобы лок брался максимально кратко.
///
/// Возвращает `true`, если был переход `PendingOpen → Open` для
/// какого-нибудь BUY-ордера — caller должен дёрнуть
/// [`crate::account_submit::try_place_tp_maker`] для постановки TP.
///
/// Если SELL/TP MATCHED но `ClosingPosition.pnl` ещё не финализирован
/// (нет `trade` event'а с size/price ИЛИ это лёг TP до того, как
/// прозвонило `manage_positions`) — здесь же финализируем: либо
/// напрямую (если у нас есть аккумулированный `realized_exit_usdc`),
/// либо переносим TP-fill через [`finalize_tp_close_in_place`].
/// Применяет статусы (`new_open`/`new_close`) к позициям, у которых
/// `open_order_id == Some(order_id)` (для new_open) или
/// `close_order_id == Some(order_id)` (для new_close).
///
/// Возвращает `Vec<SharedOpenPosition>` тех позиций, которые в результате
/// перехода именно сейчас стали `PendingOpen → Open` — caller спавнит для
/// каждой `try_place_tp_maker(pos_arc)` без повторного поиска по id.
/// В штатном режиме `order_id` уникален в CLOB, и Vec будет содержать ноль
/// или одну запись; форма Vec оставлена под маловероятный edge-case
/// «несколько локальных записей сослались на один real id» и для прямой
/// итерации в caller'ах.
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
        // Lock order: positions → pending_resolution → individual_pos_lock.
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
        // Lock-ordering: c.write дропнут выше; берём `pos.read()` без других
        // inner-локов одновременно (max один inner-lock invariant).
        for (oid, pos_id) in hit_pairs {
            crate::tee_println!(
                "[user_ws] close_status({oid}) → {status:?} (pos_id={pos_id})",
            );
        }
    }

    to_trigger_tp
}

/// Применяет один partial/full **fill** к позиции, на которую он указывает:
/// - `BUY` fill (наш taker BUY) → накладывается на `OpenPosition`:
///   `shares_held = Σ size_filled - taker_fee_in_shares`,
///   `entry_cost = Σ size × price` (USDC реально потраченные),
///   `buy_price = entry_cost / shares_held` (real VWAP).
/// - `SELL` fill (наш taker SELL для close или maker TP fill) → накладывается
///   на матчащую `ClosingPosition` (по `close_order_id` или `tp_order_id` →
///   соответствующая запись в `closing`/`positions`); по полному fill'у
///   финализирует `pnl = realized_sell_usdc - actual_entry_cost`,
///   обновляет `bankroll`/stats, проставляет `close_status=Closed`.
///
/// Если SELL fill приходит на TP-ордер (мы maker), а `ClosingPosition` для
/// этой позиции ещё не существует (TP «сам залился» до того, как
/// `manage_positions` решил закрыться) — тут же создаём её со
/// `reason=TakeProfit, close_status=Closed, pnl=Some(_)` и удаляем
/// `OpenPosition` из `Account.positions`.
///
/// Все апдейты идут под write-локами в каноническом порядке
/// (`bankroll → positions → pending_resolution → closing`).
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

/// BUY fill (наш taker BUY): корректирует `OpenPosition` по реальным числам.
/// Polymarket в категории Crypto заряжает taker-fee из получаемых shares:
/// `actual_shares = nominal × (1 − fee_rate)`, где `fee_rate ≈ 0.072 × p × (1−p)`,
/// но WS отдаёт `fee_rate_bps` напрямую — берём оттуда (если 0, считаем как
/// без fee — обычно так и есть для нашей категории, fee уже зашит в `size`
/// который CLOB вернул как «после fee»).
///
/// **Идемпотентность:** разные partial fills для одного `order_id` приходят
/// несколькими событиями. Мы аккумулируем их в `entry_cost`/`shares_held`
/// **поверх** оптимистичных значений из `book_fill_buy_strict` —
/// первый fill сбрасывает их в `0`, последующие add'ятся. Маркер «уже сбросили»
/// — это переход `open_status=PendingOpen → Open` (см. `update_position_statuses`),
/// который произошёл до этого fill'а на этом же `order_id`. Не очень надёжно
/// при гонке (если order MATCHED доходит до WS раньше, чем trade), поэтому
/// добавляем явный маркер `pos.frames_held == 0` + `tp_order_id.is_none()`
/// в качестве «накопителя ещё не активирован» — но это хрупко. Простое и
/// надёжное решение: при первом trade fill'е перезаписываем
/// `entry_cost=size×price`, `shares_held=size_after_fee`; последующие fills
/// add'ят. Достигается через флаг — ниже используем `bool` локально.
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
            // Первый WS-fill для этой позиции? Сбрасываем оптимистичные числа
            // из `book_fill_buy_strict` и накапливаем реальные. Последующие
            // partial fills уже идут поверх (FAK таker может расщепиться на
            // несколько trades).
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
                pos.shares_held, pos.planned_shares_held,
                pos.entry_cost, pos.planned_entry_cost,
                pos.buy_price, pos.planned_buy_price,
            );
        }
    }
    if !hit {
        crate::tee_eprintln!(
            "[user_ws] BUY fill: order_id={order_id} не найден ни в одной OpenPosition"
        );
    }
}

/// SELL fill: ищем матч по
/// 1) `ClosingPosition.close_order_id` (наш taker SELL после SL/Timeout/EvExit) —
///    аккумулируем `pnl` в `ClosingPosition.pnl`, при `close_status=Closed` финализируем
///    в bankroll/stats;
/// 2) `OpenPosition.tp_order_id` (наш maker TP залился) — создаём
///    `ClosingPosition { reason=TakeProfit, close_status=Closed, pnl: Some(_) }`,
///    удаляем `OpenPosition` из `positions`.
pub(crate) async fn apply_sell_fill(
    account: &SharedAccount,
    order_id: &str,
    size: f64,
    price: f64,
    fee_rate_bps: f64,
    role: OrderRole,
) {
    let usd_received = size * price;
    let fee_rate = fee_rate_bps / 10_000.0;
    // Для SELL fee вычитается из USDC: net_usdc = gross × (1 − fee_rate)
    // Maker TP в категории Crypto обычно с fee=0 (taker оплачивает обе стороны),
    // но раз WS отдал fee_rate_bps — учитываем его явно. Подход согласован
    // с тем, как [`crate::history_sim::close_position`] считает урджентные
    // выходы.
    let _ = role; // ниже разделяем по тому, где нашли order_id
    let net_usdc = usd_received * (1.0 - fee_rate);

    // Сначала пробуем `ClosingPosition` (наш taker SELL).
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
                // pnl = аккумулированные net USDC − entry_cost (entry_cost уже
                // лежит в `c.position.entry_cost`, скорректированный по BUY fills).
                // Так как partial fills могут приходить инкрементально, мы добавляем
                // `net_usdc` к `pnl` (а первоначально `pnl=None` → берём 0); финализация
                // (вычесть entry_cost) — на терминальном trade-status'е (см. ниже).
                c.pnl = Some(prev + net_usdc);
                hit = true;
                // Для лога pos_id'а: клонируем `position` Arc, прочитаем под
                // pos.read() позже (max один inner-lock одновременно — не
                // можем читать pos под уже-удержанным c.write).
                hit_pos_arcs.push((c.position.clone(), prev + net_usdc));
                // Если уже Closed (терминальный fill), финализируем после
                // отпускания c-write-лока (нужен будет pos.read() — два inner-лока
                // одновременно держать нельзя по lock-ordering invariant'у).
                if matches!(c.close_status, ClosingPositionStatus::Closed) {
                    to_finalize = Some(c_arc.clone());
                }
            }
        }
        if hit {
            // Финализация — снаружи и closing-HashMap-лока, и c-write-лока.
            drop(closing);
            // Лог с pos_id'ами — теперь pos.read() безопасен (никаких других inner-локов).
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

    // Fallback: maker TP fill (мы maker; пришёл trade с order_id в `maker_orders[].order_id`).
    // Здесь надо найти `OpenPosition` по `tp_order_id == order_id`, перенести её в `closing`
    // как `Closed/TakeProfit` и финализировать pnl.
    let mut maybe_pos: Option<(crate::account::LaneKey, crate::history_sim::SharedOpenPosition)> =
        None;
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
        // Гонка с SELL-taker close-flow: морфируем существующую PendingClose-запись.
        // Подменяем `close_order_id` на TP-id, чтобы finalize_tp_close_after_creation
        // нашёл её. `close_placement_attempted` оставляем `true` (уже взведено
        // manage_positions'ом). `spawn_close_via_taker` через post-cancel re-check
        // увидит `close_status==Closed` и не пойдёт в SELL-taker retry-loop.
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
        // lane_key уже извлечён из positions выше; в `closing` запись уже лежит
        // (manage_positions её туда добавила синхронно), новой push'и не нужно.
        let _ = lane_key;
        finalize_tp_close_after_creation(account, order_id, "Ws").await;
        return;
    }

    crate::tee_println!(
        "[user_ws] TP maker fill: pos_id={pos_id}, order_id={order_id}, size={size:.4}, price={price:.4}, net_usdc={net_usdc:.4}, entry_cost={entry_cost:.4}, pnl={pnl:.4}"
    );
    // Создаём `ClosingPosition` со статусом `Closed` сразу — pnl уже известен.
    let c_arc: crate::history_sim::SharedClosingPosition =
        std::sync::Arc::new(tokio::sync::RwLock::new(ClosingPosition {
            position: pos_arc.clone(),
            exit_price,
            reason: CloseReason::TakeProfit,
            pnl: Some(pnl),
            close_status: ClosingPositionStatus::Closed,
            close_order_id: Some(order_id.to_string()),
            close_placement_attempted: true,
            created_unix_ms: current_timestamp_ms(),
        }));
    // Прямая Weak-ссылка на `ClosingPosition` в `OpenPosition.closing_position`
    // — point-of-truth для polling-fallback в `account_submit`. Берём
    // pos.write() **до** closing-HashMap-лока (max один inner-lock одновременно;
    // closing-HashMap-лок и pos-inner-write — разные уровни canonical
    // порядка, конфликта нет, но всё равно держим максимально кратко).
    {
        let mut p = pos_arc.write().await;
        p.set_closing_position(std::sync::Arc::downgrade(&c_arc));
    }
    {
        let mut closing = account.closing.write().await;
        closing.entry(lane_key).or_default().push(c_arc);
    }
    // Финализация под собственными локами в каноническом порядке.
    finalize_tp_close_after_creation(account, order_id, "Ws").await;
}

/// Финализирует `pnl` (вычитает `entry_cost`) для уже-в-`Closed`-состоянии
/// `ClosingPosition`, обновляет `bankroll` и стат-счётчики (`pnl_tp` /
/// `pnl_sl` / etc., `trades`/`wins`/`losses`). Идемпотентен через маркер
/// [`crate::history_sim::OpenPosition::pnl_finalized`].
///
/// **Lock-ordering:** не держим одновременно ClosingPosition.write и
/// OpenPosition.write/read (canonical: max один inner-lock). Поэтому
/// сначала `pos.read()` для `entry_cost`, дроп; потом `c.write()` для
/// pnl + маркер; потом `pos.write()` для маркера на OpenPosition.
pub(crate) async fn finalize_close_pnl_in_place(
    account: &SharedAccount,
    c_arc: crate::history_sim::SharedClosingPosition,
    finalized_via: &'static str,
) {
    // Шаг 1: snapshot из ClosingPosition (clone Arc'а на position и pnl).
    let (pos_arc, raw_pnl) = {
        let c = c_arc.read().await;
        (c.position.clone(), c.pnl.unwrap_or(0.0))
    };
    // Шаг 2: read entry_cost / маркер / pos_id из OpenPosition. Если pnl_finalized=true
    // — финализатор уже отработал, no-op.
    let (entry_cost, already_finalized, pos_id) = {
        let p = pos_arc.read().await;
        (p.entry_cost, p.pnl_finalized, p.id.clone())
    };
    if already_finalized {
        return;
    }
    let pnl = raw_pnl - entry_cost;
    // Шаг 3: c.write — обновить pnl до финального значения.
    {
        let mut c = c_arc.write().await;
        c.pnl = Some(pnl);
    }
    // Шаг 4: pos.write — поставить маркер `pnl_finalized=true` (идемпотентность).
    {
        let mut p = pos_arc.write().await;
        p.pnl_finalized = true;
    }

    // Шаг 5: bankroll апдейт inline. Прежний `tokio::spawn` тут был
    // workaround'ом против case'а «caller всё ещё держит closing-HashMap-лок»,
    // но сейчас оба call-site'а (`apply_sell_fill` после `drop(closing)` и
    // `drive_close_pnl_finalization_via_polling`) приходят без удержанных
    // HashMap-локов, поэтому никакой инверсии canonical order'а
    // (`bankroll → … → closing`) нет — пишем напрямую.
    let new_bankroll = {
        let mut bankroll = account.bankroll.write().await;
        *bankroll += pnl;
        *bankroll
    };
    crate::tee_println!(
        "[user_ws] finalize SELL: pos_id={pos_id}, pnl={pnl:.4} → bankroll={new_bankroll:.4}",
    );
    // Шаг 6: peak/dd inline. Equity тут = cash-bankroll без MtM остальных
    // открытых позиций — это нижняя граница (MtM прибавит ≥0 к equity,
    // если у других OpenPosition'ов положительный MtM). На следующем
    // real_sim MtM-тике peak/dd пересчитается с учётом MtM-добавок;
    // здесь — «поспешный» update, чтобы emergency-halt сработал как
    // можно раньше после фиксации убытка.
    account.update_drawdown(new_bankroll).await;
    // Шаг 7: stats-каунтеры (per-side `pnl_tp` / `pnl_sl` / etc.) + строка
    // submit-orders CSV. Идёт после bankroll/peak — никаких Account-локов
    // тут уже не удерживается, lock-ordering чистый. Helper сам no-op'нет,
    // если real_sim_state не зарегистрирован (тесты / history_sim).
    record_submit_close_to_csv_and_stats(account, &pos_arc, &c_arc, pnl, finalized_via).await;
}

/// Финализирует TP-fill после того, как `apply_sell_fill` (TP ветка) создала
/// `ClosingPosition` с уже корректным `pnl` (т.е. proceeds − entry_cost).
/// Здесь только апдейт bankroll/stats — идемпотентно через тот же
/// [`crate::history_sim::OpenPosition::pnl_finalized`] маркер.
pub(crate) async fn finalize_tp_close_after_creation(
    account: &SharedAccount,
    order_id: &str,
    finalized_via: &'static str,
) {
    // Сначала снимаем snapshot {pos_arc, c_arc, pnl} под closing-HashMap-локом +
    // c-write inner. Чтобы не держать оба inner'а одновременно, маркер на
    // OpenPosition ставим уже после отпускания c-write.
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
    // Идемпотентность: если маркер уже стоит — выходим. Заодно snapshot'им pos_id для лога.
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
    // Inline peak/dd update — см. `finalize_close_pnl_in_place` Шаг 6.
    account.update_drawdown(new_bankroll).await;
    // Stats + submit-orders CSV — см. `finalize_close_pnl_in_place` Шаг 7.
    record_submit_close_to_csv_and_stats(account, &pos_arc, &c_arc, pnl, finalized_via).await;
}

/// Бампит per-side stats-каунтеры (через [`crate::history_sim::apply_close_to_side_stats`])
/// и пишет одну строку в submit-orders CSV
/// (через [`crate::trade_csv_log::write_submit_trade_csv_row`]) для одной
/// финализованной submit-сделки.
///
/// **Lock-ordering:** не удерживает никакие Account-локи. Берёт под коротким
/// read'ом `pos_arc` (для snapshot всех полей включая `id`/`currency`/`interval`
/// /…/`graph_dump_bin_path_for_trade_csv_uri`), потом отпускает, потом `c_arc.read()`
/// для `reason`/`exit_price`/`close_order_id`, отпускает, потом
/// `account.real_sim_state_by_currency.read()` миллисекундно (clone Arc'а), и
/// уже отдельно `state.write()` (Arc<RwLock<RealSimState>>) для bump'а stats — это
/// отдельный лок, к canonical Account-order'у не привязан.
///
/// No-op если:
/// - `currency` не зарегистрирован в [`crate::account::Account::real_sim_state_by_currency`]
///   (history_sim / тесты / ранний crash без `run_real_sim`);
/// - `interval` или `side` пары не парсятся (`unknown` — старые позиции/баги).
async fn record_submit_close_to_csv_and_stats(
    account: &SharedAccount,
    pos_arc: &crate::history_sim::SharedOpenPosition,
    c_arc: &crate::history_sim::SharedClosingPosition,
    pnl: f64,
    finalized_via: &'static str,
) {
    use crate::xframe::CurrencyUpDownOutcome;
    use crate::xframe::XFrameIntervalKind;

    // -------- snapshot pos --------
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
        let graph_html_file_uri = crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri(&p)
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

    // -------- snapshot c --------
    let (reason, exit_price, close_order_id) = {
        let c = c_arc.read().await;
        (c.reason.clone(), c.exit_price, c.close_order_id.clone())
    };

    // -------- маппинг interval / side --------
    let interval_kind = XFrameIntervalKind::from_i32(interval_type);
    let side_outcome = CurrencyUpDownOutcome::from_i32(side_idx);
    let interval_str = interval_kind
        .map(crate::real_sim::interval_label)
        .unwrap_or("unknown");
    let side_str = side_outcome
        .map(crate::real_sim::side_label)
        .unwrap_or("unknown");

    // -------- fill_role: TP = Maker, всё остальное (SL/Timeout/EvExit*) = Taker --------
    let fill_role: &'static str =
        if matches!(reason, crate::history_sim::CloseReason::TakeProfit) {
            "Maker"
        } else {
            "Taker"
        };

    // -------- bump per-side stats --------
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

    // -------- submit-orders CSV --------
    let now_ms = crate::util::current_timestamp_ms();
    let open_unix_ms = event_end_ms.map(|e| e - event_remaining_ms_at_open);
    let close_unix_ms = Some(now_ms);
    let event_remaining_ms_at_close = event_end_ms.map(|e| e - now_ms).unwrap_or(0);
    crate::trade_csv_log::write_submit_trade_csv_row(crate::trade_csv_log::SubmitTradeCsvRow {
        pos_id: &pos_id,
        polymarket_url: &polymarket_url,
        price_to_beat,
        final_price,
        currency: &currency,
        interval: interval_str,
        side: side_str,
        market_id: &market_id,
        asset_id: &asset_id,
        exit_reason: crate::history_sim::trade_csv_close_reason_label(&reason),
        fill_role,
        finalized_via,
        planned_buy_price,
        buy_price,
        planned_shares_held,
        shares_held,
        planned_entry_cost,
        entry_cost,
        exit_price,
        // Polymarket fee уже учтена в `c.pnl` (см. `apply_sell_fill`:
        // `net_usdc = usd_received × (1 − fee_rate)`); отдельно gross/fee
        // не аккумулируем — отдельная колонка стоила бы поля в
        // `ClosingPosition`. Если нужно — добавить `fee_usdc_accumulated`.
        fee_usdc: 0.0,
        pnl,
        open_order_id: open_order_id.as_deref(),
        tp_order_id: tp_order_id.as_deref(),
        close_order_id: close_order_id.as_deref(),
        raw_pred,
        cal_pred,
        kelly_f,
        p_win_ema_at_close: p_win_ema,
        frames_held,
        event_remaining_ms_at_open,
        event_remaining_ms_at_close,
        open_unix_ms,
        close_unix_ms,
        graph_html_file_uri: graph_html_file_uri.as_str(),
        pnl_top5_shap: pnl_top5_shap.as_str(),
    });
}
