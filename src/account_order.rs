//! CLOB: [`post_order_on_clob`] ([POST /order](https://docs.polymarket.com/api-reference/trade/post-a-new-order)),
//! [`cancel_order_on_clob`](https://docs.polymarket.com/api-reference/trade/cancel-single-order).
//! `clob_authed` / `clob_signer` — из [`crate::account::Account`] ([`crate::authenticate::try_authenticate_clob_for_heartbeats`]).
//! Шаги shutdown: [`cancel_all_orders_on_clob`], [`sell_all_positions_on_clob`] ([`crate::account_exit::graceful_exit`]).

use crate::account::{POLY_PRIVATE_KEY_ENV, SharedAccount};
use crate::account_order_completion::{
    PostOrderHttpOutcome, PostOrderInvokeContext, after_post_order_maybe_track_invoke,
    fire_failed_invocation_for_side, wrap_post_order_cb,
};
use crate::history_sim::StrictBook;
use crate::project_manager::ProjectManager;
use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::response::PostOrderResponse;
use polymarket_client_sdk::clob::types::{Amount, OrderType, Side, SignableOrder};
use polymarket_client_sdk::data::types::request::PositionsRequest;
use polymarket_client_sdk::types::{Decimal, U256};
use serde_json::json;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

/// Маркет (FAK) или лимит post-only (GTC/GTD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderRole {
    /// `market_order`, съём встречной ликвидности.
    Taker,
    /// `limit_order` + post-only; нужны цена и `Shares`.
    Maker,
}

/// Объём: USDC только у taker BUY; shares у SELL и у maker.
#[derive(Debug, Clone, Copy)]
pub enum OrderAmount {
    /// Спендить N USDC (taker BUY).
    UsdNotional(f64),
    /// Кол-во outcome-shares.
    Shares(f64),
}

/// Вход для [`post_order_on_clob`]: taker BUY (UsdNotional±slippage/worst-price), TP maker (Shares+price+[expiration]), taker SELL (Shares).
#[derive(Debug, Clone)]
pub struct PostOrderRequest {
    /// Десятичный `tokenId` (= `OpenPosition.asset_id`).
    pub asset_id: String,
    /// SDK `Side::Buy` / `Side::Sell`.
    pub side: Side,
    /// Taker vs maker (см. enum).
    pub role: OrderRole,
    /// USDC notional (taker BUY) или shares.
    pub amount: OrderAmount,
    /// Prob. [0,1], кратно tick; maker обязателен; для taker — worst-acceptable (выше slip-cap).
    pub price: Option<f64>,
    /// Cap от best L1 (prob.), только taker без явного `price`.
    pub max_slippage_pp: Option<f64>,
    /// GTD на CLOB только у maker; у taker при `Some` задаёт локальный дедлайн финального колбэка POST (не передаётся в `market_order`).
    pub expiration: Option<DateTime<Utc>>,
    /// Если `Some` — верхний предел времени unix ms для fallback poll после «живого» POST; если `None` — фиксированный запас секунд с момента POST.
    pub market_end_unix_ms: Option<i64>,
    /// Таймаут HTTP только на `POST /order`.
    pub timeout: Duration,
    /// При slip-cap без `price`: L1 без лишнего GET /book.
    pub strict_book: Option<StrictBook>,
}

pub use crate::account_order_completion::{
    InvokeSettlementWatch, InvokeSettlementWatchTx, SingleOrderClobInvocationReport,
    SingleOrderInvokeCb, invoke_settlement_ready, invoke_settlement_report,
    invoke_settlement_watch, wait_invoke_settlement,
};

/// `POST /order`: колбэк [`SingleOrderInvokeCb`] вызывается **ровно один раз** при любом исходе
/// (валидация, отсутствие auth/signer, ошибка билда/подписи, HTTP/SDK error, timeout,
/// `success=false`, частичная сделка, полная сделка). Одноразовость обеспечивает
/// [`crate::account_order_completion::CompletionOnce`] вокруг `invoke`.
///
/// Колбэк **никогда не выполняется синхронно в стэке этой функции**: все пути firing'а проходят через
/// [`crate::account_order_completion::spawn_fire_invocation_report`] (или через
/// [`crate::account_order_completion::PostOrderInvokeAggregator`], который уже работает на собственных
/// [`tokio::spawn`]-тасках — finalize-attempt/poll/WS). Это значит, что любое тело `invoke` выполняется
/// строго **после** возврата `post_order_on_clob` вызывающему — момент планирования таски
/// определяется тоkio-рантаймом, но не текущим стэком вызовов.
///
/// **Гарантия on-chain settlement:** [`crate::account_order_completion::SingleOrderClobInvocationReport::success`]
/// = `true` фаирится **только после факта зачисления средств on-chain** (как для maker, так и для
/// taker; для полного и для частичного исполнения). Источник истины — лифсайкл
/// [`polymarket_client_sdk::clob::types::TradeStatusType`]: трейды учитываются как «settled»
/// только при `Mined|Confirmed`, а book-level `Matched` короткозамыкания не делает. Сигналы
/// собираются и из user-WS (`trade.status`), и из REST-poll (`client.order(...)` +
/// `client.trades(...)`) и комбинируются через `max`-merge — поэтому работает даже при
/// недоступном WS. См. модуль [`crate::account_order_completion`].
///
/// Никаких timestamp-фоллбэков нет: агрегатор ждёт явных HTTP/WS терминалов от CLOB сколько
/// нужно. Если book-match состоялся, а on-chain settlement зависает (`Retrying|Failed`),
/// CLOB-сторона рано или поздно эмитит `Canceled|Unmatched` или сам ордер дойдёт до book-terminal
/// при очередном poll — и колбэк выстрелит с `success=false`/`partial=true` и диагностикой
/// `error_msg = "book_terminal_no_settle: ..."`. Если CLOB не отвечает (сеть/баг), вызывающий
/// код сам должен таймаутить — этот контракт не таймаутит ничего.
///
/// При ошибке до HTTP-ответа отчёт несёт `success=false, partial=false, order_id=None` и нули в
/// корректной типовой раскладке по `side` ([`crate::account_order_completion::SingleOrderClobInvocationReport`]).
///
/// `project_manager` — заглушка ради единой сигнатуры с
/// [`crate::account_mock_order::post_order_on_clob`]: реальный CLOB-путь
/// никаких полей из [`ProjectManager`] не читает.
pub async fn post_order_on_clob(
    account: &SharedAccount,
    project_manager: Option<&Arc<ProjectManager>>,
    request: PostOrderRequest,
    invoke: SingleOrderInvokeCb,
) -> Result<Option<String>> {
    let _ = project_manager;
    let invoke_slot = wrap_post_order_cb(invoke);

    if let Err(err) = validate_post_order_request(&request) {
        fire_failed_invocation_for_side(
            &invoke_slot,
            request.side,
            Some(format!("validate_post_order_request: {err:#}")),
        );
        return Err(err);
    }

    let auth_client: clob::Client<Authenticated<Normal>> = match (**account.clob_authed.load())
        .clone()
    {
        Some(c) => c,
        None => {
            let msg = format!(
                "clob_authed=None — CLOB не аутентифицирован, проверьте {POLY_PRIVATE_KEY_ENV} и [heartbeat] CLOB authenticate"
            );
            fire_failed_invocation_for_side(&invoke_slot, request.side, Some(msg.clone()));
            return Err(anyhow!("post_order_on_clob: {msg}"));
        }
    };
    let signer = match (**account.clob_signer.load()).clone() {
        Some(s) => s,
        None => {
            let msg = "clob_signer=None — auth-цикл не запускался?".to_string();
            fire_failed_invocation_for_side(&invoke_slot, request.side, Some(msg.clone()));
            return Err(anyhow!("post_order_on_clob: {msg}"));
        }
    };

    let token_id = match U256::from_str(&request.asset_id) {
        Ok(t) => t,
        Err(parse_err) => {
            let msg = format!(
                "невалидный asset_id={:?} (ожидается десятичный U256): {parse_err}",
                request.asset_id,
            );
            fire_failed_invocation_for_side(&invoke_slot, request.side, Some(msg.clone()));
            return Err(anyhow!("post_order_on_clob: {msg}"));
        }
    };

    let signable = match request.role {
        OrderRole::Maker => match build_maker_signable(&auth_client, token_id, &request).await {
            Ok(s) => s,
            Err(err) => {
                fire_failed_invocation_for_side(
                    &invoke_slot,
                    request.side,
                    Some(format!("build_maker_signable: {err:#}")),
                );
                return Err(err);
            }
        },
        OrderRole::Taker => match build_taker_signable(&auth_client, token_id, &request).await {
            Ok(s) => s,
            Err(err) => {
                fire_failed_invocation_for_side(
                    &invoke_slot,
                    request.side,
                    Some(format!("build_taker_signable: {err:#}")),
                );
                return Err(err);
            }
        },
    };

    let signed = match auth_client.sign(&signer, signable).await {
        Ok(s) => s,
        Err(err) => {
            let msg = format!("подпись ордера упала: {err:#}");
            fire_failed_invocation_for_side(&invoke_slot, request.side, Some(msg.clone()));
            return Err(anyhow!("post_order_on_clob: {msg}"));
        }
    };

    let resp = match tokio::time::timeout(request.timeout, auth_client.post_order(signed)).await {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            let msg = format!("POST /order SDK error: {err:#}");
            crate::tee_eprintln!("post_order_on_clob: {msg} (request may have hit network)");
            fire_failed_invocation_for_side(&invoke_slot, request.side, Some(msg));
            return Ok(None);
        }
        Err(_elapsed) => {
            let msg = format!(
                "POST /order timed out after {:?} (request may have been accepted)",
                request.timeout
            );
            crate::tee_eprintln!("post_order_on_clob: {msg}");
            fire_failed_invocation_for_side(&invoke_slot, request.side, Some(msg));
            return Ok(None);
        }
    };

    let PostOrderResponse {
        success,
        order_id,
        status,
        making_amount,
        taking_amount,
        error_msg,
        transaction_hashes,
        trade_ids,
        ..
    } = resp;

    let post_error_msg_sdk = error_msg.filter(|s| !s.is_empty());
    let http_detail = json!({
        "order_id": order_id.clone(),
        "success": success,
        "status": format!("{:?}", status),
        "error_msg": post_error_msg_sdk,
        "trade_ids": trade_ids,
        "making_amount": making_amount.to_string(),
        "taking_amount": taking_amount.to_string(),
        "transaction_hashes": transaction_hashes.iter().map(|h| format!("{h:#x}")).collect::<Vec<String>>(),
    });
    let making_f64 = making_amount.to_string().parse::<f64>().unwrap_or(0.0);
    let taking_f64 = taking_amount.to_string().parse::<f64>().unwrap_or(0.0);

    let (making_amount, taking_amount) = match request.side {
        Side::Buy => (
            OrderAmount::UsdNotional(making_f64),
            OrderAmount::Shares(taking_f64),
        ),
        Side::Sell => (
            OrderAmount::Shares(making_f64),
            OrderAmount::UsdNotional(taking_f64),
        ),
        _ => panic!(
            "post_order_on_clob: side={:?} не поддерживается (ожидается Buy/Sell)",
            request.side
        ),
    };

    let http_snap = PostOrderHttpOutcome {
        order_id: order_id.clone(),
        success,
        status: status.clone(),
        detail: http_detail,
        invoke_ctx: Some(PostOrderInvokeContext {
            request,
            making_amount,
            taking_amount,
        }),
        error_msg: post_error_msg_sdk.clone(),
    };

    after_post_order_maybe_track_invoke(
        account,
        Arc::clone(&account.order_invoke_hub),
        &http_snap,
        invoke_slot,
    )
    .await;
    Ok((success && !order_id.is_empty()).then_some(order_id))
}

/// Ошибки комбинаций полей до сети/SDK `build`.
pub(crate) fn validate_post_order_request(req: &PostOrderRequest) -> Result<()> {
    if req.timeout.is_zero() {
        bail!("post_order_on_clob: timeout=0 — POST /order не дождётся ответа");
    }
    match req.side {
        Side::Buy | Side::Sell => {}
        _ => bail!(
            "post_order_on_clob: side={:?} не поддерживается (ожидается Buy/Sell)",
            req.side
        ),
    }
    match req.role {
        OrderRole::Maker => {
            if req.price.is_none() {
                bail!("post_order_on_clob: maker требует явный `price` (limit-ордер)");
            }
            if !matches!(req.amount, OrderAmount::Shares(_)) {
                bail!(
                    "post_order_on_clob: maker amount должен быть Shares, получили {:?}",
                    req.amount
                );
            }
        }
        OrderRole::Taker => {
            if matches!(req.side, Side::Sell) && !matches!(req.amount, OrderAmount::Shares(_)) {
                bail!(
                    "post_order_on_clob: taker SELL требует Shares amount, получили {:?}",
                    req.amount
                );
            }
        }
    }
    if let Some(p) = req.price
        && (!p.is_finite() || !(0.0..=1.0).contains(&p))
    {
        bail!("post_order_on_clob: price={p} вне [0,1] либо не finite");
    }
    if let Some(s) = req.max_slippage_pp
        && (!s.is_finite() || !(0.0..=1.0).contains(&s))
    {
        bail!("post_order_on_clob: max_slippage_pp={s} вне [0,1] либо не finite");
    }
    if let OrderAmount::UsdNotional(usd) = req.amount
        && (!usd.is_finite() || usd <= 0.0)
    {
        bail!("post_order_on_clob: usd amount={usd} должен быть > 0 и finite");
    }
    if let OrderAmount::Shares(s) = req.amount
        && (!s.is_finite() || s <= 0.0)
    {
        bail!("post_order_on_clob: shares amount={s} должен быть > 0 и finite");
    }
    Ok(())
}

/// `f64` → `Decimal` через строку (стабильнее двоичного float).
fn f64_to_decimal(f: f64, ctx: &str) -> Result<Decimal> {
    if !f.is_finite() {
        bail!("post_order_on_clob: {ctx}: значение {f} не finite");
    }
    f.to_string()
        .parse::<Decimal>()
        .with_context(|| format!("post_order_on_clob: {ctx}: f64 {f} → Decimal не сконвертился"))
}

/// Минимальный тик Polymarket по цене исхода — **0.01** ⇒ в `f64` не более двух знаков после запятой.
/// Принимаем любой конечный `price` уже прошедший [`validate_post_order_request`] и округляем к «центу»,
/// затем режем диапазон, чтобы/SDK не падали на лишней точности `f64`.
fn normalize_probability_price_to_cent_tick(price: f64, ctx: &str) -> Result<f64> {
    if !price.is_finite() {
        bail!("post_order_on_clob: {ctx}: цена не finite ({price})");
    }
    if !(0.0..=1.0).contains(&price) {
        bail!("post_order_on_clob: {ctx}: цена {price} вне допустимого [0,1]");
    }
    let rounded = (price * 100.0).round() / 100.0;
    // Согласовано с фильтром лимита в `validate_post_order_request` после округления.
    Ok(rounded.clamp(0.001, 0.999))
}

/// `limit_order` post_only, GTC или GTD если есть `expiration`.
async fn build_maker_signable(
    client: &clob::Client<Authenticated<Normal>>,
    token_id: U256,
    req: &PostOrderRequest,
) -> Result<SignableOrder> {
    let price_raw = req.price.expect("validated in validate_post_order_request");
    let price = normalize_probability_price_to_cent_tick(price_raw, "maker price input")?;
    let shares = match req.amount {
        OrderAmount::Shares(s) => s,
        OrderAmount::UsdNotional(_) => unreachable!("validated"),
    };
    let price_dec = f64_to_decimal(price, "maker price (tick-normalized)")?;
    let size_dec = f64_to_decimal(shares, "maker shares")?;

    let order_type = if req.expiration.is_some() {
        OrderType::GTD
    } else {
        OrderType::GTC
    };

    let mut builder = client
        .limit_order()
        .token_id(token_id)
        .side(req.side)
        .price(price_dec)
        .size(size_dec)
        .order_type(order_type)
        .post_only(true);

    if let Some(exp) = req.expiration {
        builder = builder.expiration(exp);
    }

    builder
        .build()
        .await
        .map_err(|err| anyhow!("post_order_on_clob: limit_order().build() упал: {err:#}"))
}

/// `market_order` FAK; cap из `price` или L1±`max_slippage_pp`, иначе SDK сам режет книгу.
async fn build_taker_signable(
    client: &clob::Client<Authenticated<Normal>>,
    token_id: U256,
    req: &PostOrderRequest,
) -> Result<SignableOrder> {
    let cap_price = compute_taker_cap_price(client, token_id, req).await?;
    let amount = match req.amount {
        OrderAmount::UsdNotional(usd) => {
            let dec = f64_to_decimal(usd, "taker usd amount")?;
            Amount::usdc(dec)
                .map_err(|e| anyhow!("post_order_on_clob: Amount::usdc({usd}) упал: {e:#}"))?
        }
        OrderAmount::Shares(s) => {
            let dec = f64_to_decimal(s, "taker shares amount")?;
            Amount::shares(dec)
                .map_err(|e| anyhow!("post_order_on_clob: Amount::shares({s}) упал: {e:#}"))?
        }
    };

    let mut builder = client
        .market_order()
        .token_id(token_id)
        .side(req.side)
        .amount(amount)
        .order_type(OrderType::FAK);

    if let Some(p) = cap_price {
        builder = builder.price(p);
    }

    builder
        .build()
        .await
        .map_err(|err| anyhow!("post_order_on_clob: market_order().build() упал: {err:#}"))
}

/// Worst допустимая цена для taker: явный `price`, иначе L1±slip или `None` (режет SDK).
async fn compute_taker_cap_price(
    client: &clob::Client<Authenticated<Normal>>,
    token_id: U256,
    req: &PostOrderRequest,
) -> Result<Option<Decimal>> {
    if let Some(p) = req.price {
        let p_norm = normalize_probability_price_to_cent_tick(p, "taker explicit cap price")?;
        return Ok(Some(f64_to_decimal(
            p_norm,
            "taker price (tick-normalized)",
        )?));
    }
    let Some(slip) = req.max_slippage_pp else {
        return Ok(None);
    };
    let slip_dec = f64_to_decimal(slip, "taker slippage")?;

    let cap = match req.side {
        Side::Buy => {
            let best_ask_dec = match &req.strict_book {
                Some(sb) => {
                    let px = best_ask_strict(sb).ok_or_else(|| {
                        anyhow!(
                            "post_order_on_clob: strict_book без валидного ask для slippage cap \
                             (asset_id={}, token_id={token_id:#x})",
                            req.asset_id,
                        )
                    })?;
                    f64_to_decimal(px, "strict_book best ask")?
                }
                None => {
                    let book_request = OrderBookSummaryRequest::builder()
                        .token_id(token_id)
                        .build();
                    let book = client.order_book(&book_request).await.map_err(|err| {
                        anyhow!(
                            "post_order_on_clob: order_book({token_id:#x}) для slippage cap упал: \
                             {err:#}"
                        )
                    })?;
                    best_ask_sdk(&book).ok_or_else(|| {
                        anyhow!(
                            "post_order_on_clob: пустой asks book для slippage cap \
                             (token_id={token_id:#x})"
                        )
                    })?
                }
            };
            (best_ask_dec + slip_dec)
                .min(Decimal::ONE)
                .max(Decimal::ZERO)
        }
        Side::Sell => {
            let best_bid_dec = match &req.strict_book {
                Some(sb) => {
                    let px = best_bid_strict(sb).ok_or_else(|| {
                        anyhow!(
                            "post_order_on_clob: strict_book без валидного bid для slippage cap \
                             (asset_id={}, token_id={token_id:#x})",
                            req.asset_id,
                        )
                    })?;
                    f64_to_decimal(px, "strict_book best bid")?
                }
                None => {
                    let book_request = OrderBookSummaryRequest::builder()
                        .token_id(token_id)
                        .build();
                    let book = client.order_book(&book_request).await.map_err(|err| {
                        anyhow!(
                            "post_order_on_clob: order_book({token_id:#x}) для slippage cap упал: \
                             {err:#}"
                        )
                    })?;
                    best_bid_sdk(&book).ok_or_else(|| {
                        anyhow!(
                            "post_order_on_clob: пустой bids book для slippage cap \
                             (token_id={token_id:#x})"
                        )
                    })?
                }
            };
            (best_bid_dec - slip_dec)
                .max(Decimal::ZERO)
                .min(Decimal::ONE)
        }
        _ => bail!(
            "post_order_on_clob: side={:?} не поддерживается (ожидается Buy/Sell)",
            req.side
        ),
    };

    Ok(Some(cap))
}

/// Лучший ask из локального книжного снимка (первая ненулевая строка).
pub(crate) fn best_ask_strict(book: &StrictBook) -> Option<f64> {
    book.asks
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)
}

/// Лучший bid из локального книжного снимка (первая ненулевая строка).
pub(crate) fn best_bid_strict(book: &StrictBook) -> Option<f64> {
    book.bids
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)
}

pub fn best_ask_sdk(
    book: &polymarket_client_sdk::clob::types::response::OrderBookSummaryResponse,
) -> Option<Decimal> {
    book.asks.iter().map(|l| l.price).min()
}

fn best_bid_sdk(
    book: &polymarket_client_sdk::clob::types::response::OrderBookSummaryResponse,
) -> Option<Decimal> {
    book.bids.iter().map(|l| l.price).max()
}

/// Вход в [`cancel_order_on_clob`].
#[derive(Debug, Clone)]
pub struct CancelOrderRequest {
    /// CLOB `orderID`.
    pub order_id: String,
    /// Таймаут HTTP на `DELETE /order`.
    pub timeout: Duration,
}

/// Одна запись из ответа cancel: смотреть [`Self::canceled`] и [`Self::error_msg`].
#[derive(Debug, Clone)]
pub struct CancelOrderResult {
    /// Эхо из запроса.
    pub order_id: String,
    /// Попали в массив `canceled`.
    pub canceled: bool,
    /// Текст из `not_canceled` при `canceled == false`.
    pub error_msg: Option<String>,
}

/// `DELETE /order` под API-key; нужен только `clob_authed`. При сетевой/SDK-ошибке, таймауте или пустом теле ответа — **`Err`**; если ответ есть, **`Ok`** поля задают результат снятия (в т.ч. `not_canceled` от CLOB).
///
/// `project_manager` — заглушка ради единой сигнатуры с
/// [`crate::account_mock_order::cancel_order_on_clob`].
pub async fn cancel_order_on_clob(
    account: &SharedAccount,
    project_manager: Option<&Arc<ProjectManager>>,
    request: CancelOrderRequest,
) -> Result<CancelOrderResult> {
    let _ = project_manager;
    let oid = request.order_id.clone();

    if request.timeout.is_zero() {
        bail!("cancel_order_on_clob: timeout=0 — DELETE /order не дождётся ответа");
    }
    if request.order_id.is_empty() {
        bail!("cancel_order_on_clob: пустой order_id");
    }

    let auth_client = (**account.clob_authed.load()).clone().ok_or_else(|| {
        anyhow!(
            "cancel_order_on_clob: clob_authed=None — CLOB не аутентифицирован, проверьте {POLY_PRIVATE_KEY_ENV} и [heartbeat] CLOB authenticate"
        )
    })?;

    let resp = match tokio::time::timeout(
        request.timeout,
        auth_client.cancel_order(&request.order_id),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            crate::tee_eprintln!(
                "cancel_order_on_clob: DELETE /order ошибка после возможной отправки: {err:#}"
            );
            bail!(
                "cancel_order_on_clob: DELETE /order SDK error после возможной отправки, order_id={oid}: {err:#}"
            );
        }
        Err(_elapsed) => {
            let msg = format!("HTTP timeout {:?}", request.timeout);

            bail!(
                "cancel_order_on_clob: DELETE /order {}, order_id={oid} — запрос мог уйти в сеть",
                msg,
            );
        }
    };

    let (canceled, error_msg) = if !resp.not_canceled.is_empty() {
        let msg = resp
            .not_canceled
            .into_values()
            .next()
            .filter(|s| !s.is_empty());
        (false, msg)
    } else if !resp.canceled.is_empty() {
        (true, None)
    } else {
        crate::tee_eprintln!(
            "cancel_order_on_clob: DELETE /order пустой ответ (canceled=[], not_canceled={{}}), order_id={}",
            request.order_id
        );
        bail!(
            "cancel_order_on_clob: DELETE /order пустой ответ от CLOB (canceled=[], not_canceled={{}}), order_id={}",
            request.order_id
        );
    };

    let out = CancelOrderResult {
        order_id: request.order_id,
        canceled,
        error_msg: error_msg.clone(),
    };

    Ok(out)
}

/// HTTP timeout: `DELETE /cancel-all`, Data API, `POST /order` в graceful shutdown.
const EXIT_HTTP_TIMEOUT_SEC: u64 = 60;
/// Ниже этого размера в shares exit-SELL не шлём.
const SHARES_DUST_THRESHOLD: f64 = 0.0001;
/// Пауза между exit-SELL по позициям (rate limit).
const PER_POSITION_PAUSE_MS: u64 = 200;

/// `DELETE /cancel-all` для текущей CLOB-сессии; лог с префиксом `[account_exit]`.
pub(crate) async fn cancel_all_orders_on_clob(account: &SharedAccount) {
    let auth_client = match (**account.clob_authed.load()).clone() {
        Some(c) => c,
        None => {
            crate::tee_eprintln!(
                "[account_exit] clob_authed=None — cancel-all пропускаем (auth не поднялся)"
            );
            return;
        }
    };
    match tokio::time::timeout(
        Duration::from_secs(EXIT_HTTP_TIMEOUT_SEC),
        auth_client.cancel_all_orders(),
    )
    .await
    {
        Ok(Ok(resp)) => {
            crate::tee_println!(
                "[account_exit] cancel-all OK: canceled={}, not_canceled={}",
                resp.canceled.len(),
                resp.not_canceled.len(),
            );
            for (oid, reason) in &resp.not_canceled {
                crate::tee_eprintln!(
                    "[account_exit] cancel-all not_canceled: order_id={oid}, reason={reason}"
                );
            }
        }
        Ok(Err(err)) => {
            crate::tee_eprintln!("[account_exit] cancel-all упал: {err:#}");
        }
        Err(_) => {
            crate::tee_eprintln!("[account_exit] cancel-all timeout > {EXIT_HTTP_TIMEOUT_SEC}s");
        }
    }
}

/// Позиции `user = derive_safe_address(EOA)` → SELL taker без slippage cap; лог `[account_exit]`.
pub(crate) async fn sell_all_positions_on_clob(account: &SharedAccount) {
    let signer = match (**account.clob_signer.load()).as_ref().cloned() {
        Some(s) => s,
        None => {
            crate::tee_eprintln!(
                "[account_exit] clob_signer=None — не знаем EOA, sell-all пропускаем"
            );
            return;
        }
    };
    let eoa = signer.address();
    let safe = crate::poly_chain::derive_safe_address(eoa);
    crate::tee_println!(
        "[account_exit] data/positions: user=safe={safe:#x} (derived from eoa={eoa:#x})"
    );

    let positions_req = PositionsRequest::builder().user(safe).build();
    let positions = match tokio::time::timeout(
        Duration::from_secs(EXIT_HTTP_TIMEOUT_SEC),
        account.data.as_ref().positions(&positions_req),
    )
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(err)) => {
            crate::tee_eprintln!("[account_exit] data/positions упал: {err:#}");
            return;
        }
        Err(_) => {
            crate::tee_eprintln!(
                "[account_exit] data/positions timeout > {EXIT_HTTP_TIMEOUT_SEC}s"
            );
            return;
        }
    };

    crate::tee_println!(
        "[account_exit] позиций к продаже: {} (без фильтра по dust)",
        positions.len()
    );

    let mut sold = 0_usize;
    let mut skipped_dust = 0_usize;
    let mut failed = 0_usize;

    struct PendingExitSell {
        invoke_rx: tokio::sync::oneshot::Receiver<SingleOrderClobInvocationReport>,
        asset_id_str: String,
        shares: f64,
    }
    let mut pending_exit_sells: Vec<PendingExitSell> = Vec::new();

    for pos in positions {
        let shares = pos.size.to_string().parse::<f64>().unwrap_or(0.0);
        if !shares.is_finite() || shares < SHARES_DUST_THRESHOLD {
            skipped_dust += 1;
            continue;
        }
        let asset_id_str = pos.asset.to_string();
        let request = PostOrderRequest {
            asset_id: asset_id_str.clone(),
            side: Side::Sell,
            role: OrderRole::Taker,
            amount: OrderAmount::Shares(shares),
            price: None,
            max_slippage_pp: None,
            expiration: None,
            market_end_unix_ms: None,
            timeout: Duration::from_secs(EXIT_HTTP_TIMEOUT_SEC),
            strict_book: None,
        };
        let (invoke_tx, invoke_rx) = tokio::sync::oneshot::channel();
        match post_order_on_clob(
            account,
            None,
            request,
            Box::new(move |rep| {
                let _ = invoke_tx.send(rep);
            }),
        )
        .await
        {
            Err(err) => {
                crate::tee_eprintln!(
                    "[account_exit] SELL ошибка до колбека: asset={asset_id_str}, \
                     shares={shares:.4}: {err:#}"
                );
                failed += 1;
            }
            Ok(_) => pending_exit_sells.push(PendingExitSell {
                invoke_rx,
                asset_id_str,
                shares,
            }),
        }
        tokio::time::sleep(Duration::from_millis(PER_POSITION_PAUSE_MS)).await;
    }

    for PendingExitSell {
        invoke_rx,
        asset_id_str,
        shares,
    } in pending_exit_sells
    {
        match invoke_rx.await {
            Ok(r) if r.success => {
                crate::tee_println!(
                    "[account_exit] SELL ok: asset={asset_id_str}, shares={shares:.4}, \
                     order_id={:?}, success={}, partial={}",
                    r.order_id,
                    r.success,
                    r.partial,
                );
                sold += 1;
            }
            Ok(r) => {
                crate::tee_eprintln!(
                    "[account_exit] SELL неуспешен (invoke): asset={asset_id_str}, \
                     shares={shares:.4}, order_id={:?}, success={}, partial={}, error_msg={:?}",
                    r.order_id,
                    r.success,
                    r.partial,
                    r.error_msg,
                );
                failed += 1;
            }
            Err(_) => {
                crate::tee_eprintln!(
                    "[account_exit] SELL колбёк потерян: asset={asset_id_str}, shares={shares:.4}"
                );
                failed += 1;
            }
        }
    }
    crate::tee_println!(
        "[account_exit] sell-all итог: sold={sold}, failed={failed}, skipped_dust={skipped_dust}"
    );
}
