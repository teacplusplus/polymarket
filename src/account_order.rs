//! Постановка/отмена ордеров на Polymarket CLOB поверх
//! [`crate::account::Account`]: примитивы [`post_order_on_clob`] и
//! [`cancel_order_on_clob`] плюс публичные типы запроса/ответа.
//! Аутентифицированный `clob::Client` и EOA-подписант берутся из
//! [`crate::account::Account::clob_authed`] /
//! [`crate::account::Account::clob_signer`] (заполняются
//! [`crate::account::try_authenticate_clob_for_heartbeats`] на старте
//! `RealSim`); сам модуль состояние счёта не трогает.
//!
//! Документация эндпоинтов:
//! - <https://docs.polymarket.com/api-reference/trade/post-a-new-order>
//! - <https://docs.polymarket.com/api-reference/trade/cancel-single-order>

use crate::account::{POLY_PRIVATE_KEY_ENV, SharedAccount};
use crate::history_sim::StrictBook;
use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::response::PostOrderResponse;
use polymarket_client_sdk::clob::types::{
    Amount, OrderStatusType, OrderType, Side, SignableOrder,
};
use polymarket_client_sdk::types::{Decimal, U256};
use std::str::FromStr;
use std::time::Duration;

/// Роль ордера на CLOB:
/// - [`OrderRole::Taker`] — съедает встречную ликвидность (`market_order` →
///   `OrderType::FAK` по умолчанию: что не зальётся — отменяется);
///   опционально лимит-цена / cap слиппеджа от L1 (см. [`PostOrderRequest`]).
/// - [`OrderRole::Maker`] — лежит лимиткой (`limit_order` + `post_only=true`,
///   `OrderType::GTC` или `GTD` если задан `expiration`); цена обязательна.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderRole {
    Taker,
    Maker,
}

/// `amount` для [`PostOrderRequest`]: единицы зависят от role × side.
/// SDK сам валидирует (`Amount::usdc` / `Amount::shares`).
///
/// - `UsdNotional(usd)` — только для taker BUY (тратим N USDC, получаем
///   столько shares, сколько даст cutoff).
/// - `Shares(n)` — taker SELL / любой maker (мы знаем размер позиции в shares).
#[derive(Debug, Clone, Copy)]
pub enum OrderAmount {
    UsdNotional(f64),
    Shares(f64),
}

/// Параметры запроса к [`post_order_on_clob`]. Обязательные комбинации:
/// - **Taker BUY с USDC-нотионалом и слиппеджем** → `role=Taker`,
///   `side=Buy`, `amount=UsdNotional`, `price=None`,
///   `max_slippage_pp=Some`. SDK сам выведет cutoff из book; cap = best ask + slip.
/// - **Maker TP (limit SELL на ближайшее будущее)** → `role=Maker`,
///   `side=Sell`, `amount=Shares`, `price=Some`,
///   `expiration=Some(now + T)` (`OrderType::GTD` + `post_only`).
/// - **Taker SELL всех shares без слиппеджа** → `role=Taker`,
///   `side=Sell`, `amount=Shares`, `price=None`,
///   `max_slippage_pp=None` (SDK выведет cutoff из bids book; FAK
///   зальёт сколько успеет).
///
/// **Снимок стакана:** при `max_slippage_pp = Some(..)` без явного `price`
/// нужен лучший L1 для расчёта cap'а. Если задан [`StrictBook`], его
/// уровни используются без лишнего `GET …/book`; иначе делается
/// [`clob::Client::order_book`].
#[derive(Debug, Clone)]
pub struct PostOrderRequest {
    /// `tokenId` (десятичная строка U256), совпадает с `OpenPosition.asset_id`.
    pub asset_id: String,
    /// `BUY` (открытие YES/NO) или `SELL` (закрытие/TP).
    pub side: Side,
    pub role: OrderRole,
    pub amount: OrderAmount,
    /// Лимит-цена в probability `[tick_size, 1 - tick_size]`. Для maker —
    /// обязательна; для taker, если задана, выступает worst-acceptable
    /// (если задан и `max_slippage_pp` — `price` имеет приоритет).
    /// Округление до tick_size — на стороне вызывающего (SDK иначе вернёт
    /// validation error).
    pub price: Option<f64>,
    /// Cap слиппеджа в probability-units от best L1 (e.g. `0.02` ≈ 2pp).
    /// Применяется только для taker без `price`. `None` — без cap'а.
    pub max_slippage_pp: Option<f64>,
    /// Истечение для maker GTD. Для taker должно быть `None`.
    pub expiration: Option<DateTime<Utc>>,
    /// HTTP-таймаут на сам `POST /order` (без учёта `order_book` для
    /// slippage cap'а). Большие значения держат вызывающего под локом.
    pub timeout: Duration,
    /// Уже имеющийся HTTP-снимок стакана (`real_sim` / батч `order_books`).
    /// Учитывается только если нужен L1 для `max_slippage_pp` при отсутствии
    /// `price`; иначе игнорируется. Видимость `pub(crate)`, т.к. тип
    /// [`StrictBook`] — `pub(crate)` внутри этого крейта.
    pub(crate) strict_book: Option<StrictBook>,
}

/// Ответ [`post_order_on_clob`] — упакованный [`PostOrderResponse`] SDK.
/// `success=false + error_msg=Some` означает «сервер принял запрос, но
/// отверг ордер» (см. 400-кейсы в OpenAPI). `success=true + status=Live` —
/// лежит в book'е; `Matched` — частично/полностью залился (см.
/// `transaction_hashes`/`trade_ids`); `Delayed` — попал в risk-delay.
#[derive(Debug, Clone)]
pub struct PostOrderResult {
    /// Идентификатор ордера на CLOB (хэш ордера, поле API `orderID`). Им
    /// матчится user-WS и локальные `open_order_id` / `close_order_id`.
    pub order_id: String,
    /// Статус после обработки матчинг-движком: например **`Live`** —
    /// ордер в стакане; **`Matched`** — было исполнение (в т.ч. частичное);
    /// **`Delayed`** — задержка (например риск-слой); иные значения —
    /// см. [`OrderStatusType`].
    pub status: OrderStatusType,
    /// `true`, если сервер **принял и успешно обработал** заявку (ордер мог
    /// лечь live или быть сматчен). При `false` смотреть [`Self::error_msg`]
    /// и см. ошибки OpenAPI `/order`.
    pub success: bool,
    /// Сколько **maker** даёт контрагента по факту операции (`makingAmount`).
    /// Fixed-point число как в ответе CLOB (до **6** знаков дробной части /
    /// «микропорции», интерпретация BUY/SELL см. модель контрактов маркета).
    pub making_amount: Decimal,
    /// Сколько **taker** даёт поперечно (`takingAmount`). Та же fixed-point
    /// семантика, что у [`Self::making_amount`].
    pub taking_amount: Decimal,
    /// Сообщение об ошибке от API при отклонении ордера (`errorMsg`).
    /// Пустые строки превращаем в `None`.
    pub error_msg: Option<String>,
    /// Хэши on-chain транзакций после исполнения (`transactionsHashes`);
    /// строки вида `0x…` после сериализации из SDK.
    pub transaction_hashes: Vec<String>,
    /// Внутренние идентификаторы сделок Polymarket (`tradeIDs`), когда ордер
    /// дал matche(es).
    pub trade_ids: Vec<String>,
}

/// Универсальный примитив постановки ордера на CLOB Polymarket
/// ([POST /order](https://docs.polymarket.com/api-reference/trade/post-a-new-order)).
/// На этом примитиве дальше можно собрать pnl-логику:
/// - **buy USD как taker со слиппеджем** — `role=Taker, side=Buy, amount=UsdNotional, max_slippage_pp=Some`.
/// - **TP-мейкер на ближайшее будущее** — `role=Maker, side=Sell, amount=Shares, price=Some, expiration=Some`.
/// - **sell all shares как taker без слиппеджа** — `role=Taker, side=Sell, amount=Shares, max_slippage_pp=None`.
///
/// **Состояние Account** функция не трогает: вызывающий отвечает за
/// создание `OpenPosition`/`ClosingPosition` со статусом `Pending*` и
/// сохранение `order_id` из [`PostOrderResult`]; финальное подтверждение
/// (Open/Closed) приходит асинхронно через [`crate::account_ws`].
///
/// **Требования к Account**: `clob_authed = Some(_)` и
/// `clob_signer = Some(_)` (оба ставятся
/// [`crate::account::try_authenticate_clob_for_heartbeats`] на старте
/// `RealSim`). Иначе — `Err`.
pub async fn post_order_on_clob(
    account: &SharedAccount,
    request: PostOrderRequest,
) -> Result<PostOrderResult> {
    validate_post_order_request(&request)?;

    // Снимок auth-стейта через ArcSwap: hot-path без локов, оба `load()`
    // консистентны на момент вызова. `clob::Client` — обёртка над
    // `Arc<ClientInner>` (дешёвый clone), `PrivateKeySigner` тоже Clone;
    // клонируем под snapshot Arc, чтобы не держать guard через сетевые вызовы.
    let auth_client = (**account.clob_authed.load()).clone().ok_or_else(|| {
        anyhow!(
            "post_order_on_clob: clob_authed=None — CLOB не аутентифицирован, проверьте {POLY_PRIVATE_KEY_ENV} и [heartbeat] CLOB authenticate"
        )
    })?;
    let signer = (**account.clob_signer.load()).clone().ok_or_else(|| {
        anyhow!("post_order_on_clob: clob_signer=None — auth-цикл не запускался?")
    })?;

    let token_id = U256::from_str(&request.asset_id).with_context(|| {
        format!(
            "post_order_on_clob: невалидный asset_id={:?} (ожидается десятичный U256)",
            request.asset_id
        )
    })?;

    let signable = match request.role {
        OrderRole::Maker => build_maker_signable(&auth_client, token_id, &request).await?,
        OrderRole::Taker => build_taker_signable(&auth_client, token_id, &request).await?,
    };

    let signed = auth_client
        .sign(&signer, signable)
        .await
        .map_err(|err| anyhow!("post_order_on_clob: подпись ордера упала: {err:#}"))?;

    let resp = match tokio::time::timeout(request.timeout, auth_client.post_order(signed)).await {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => bail!("post_order_on_clob: POST /order упал: {err:#}"),
        Err(_elapsed) => bail!(
            "post_order_on_clob: POST /order не уложился в {:?}",
            request.timeout
        ),
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

    Ok(PostOrderResult {
        order_id,
        status,
        success,
        making_amount,
        taking_amount,
        error_msg: error_msg.filter(|s| !s.is_empty()),
        transaction_hashes: transaction_hashes
            .into_iter()
            .map(|h| format!("{h:#x}"))
            .collect(),
        trade_ids,
    })
}

/// Структурная валидация [`PostOrderRequest`] — ловит ошибки до сетевых
/// вызовов и до `OrderBuilder::build()`, чтобы вызывающий получал
/// человекочитаемые сообщения, а не `Error::validation` из SDK.
fn validate_post_order_request(req: &PostOrderRequest) -> Result<()> {
    if req.timeout.is_zero() {
        bail!("post_order_on_clob: timeout=0 — POST /order не дождётся ответа");
    }
    match req.side {
        Side::Buy | Side::Sell => {}
        // `Side` помечен `#[non_exhaustive]` SDK'ом — `Unknown` + любые
        // будущие варианты в нашем коде не имеют семантики, отбрасываем.
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
            if req.expiration.is_some() {
                bail!("post_order_on_clob: taker не поддерживает expiration (FAK/FOK)");
            }
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

/// Конверсия `f64` → `Decimal` через строковый roundtrip (тот же приём,
/// что в `real_sim::http_level`: избавляется от шума IEEE-754, точное
/// представление `0.1` и т.п.). `Decimal::try_from(f64)` тоже бы сработал,
/// но текстовый roundtrip даёт предсказуемый scale (ровно как в литерале).
fn f64_to_decimal(f: f64, ctx: &str) -> Result<Decimal> {
    if !f.is_finite() {
        bail!("post_order_on_clob: {ctx}: значение {f} не finite");
    }
    f.to_string()
        .parse::<Decimal>()
        .with_context(|| format!("post_order_on_clob: {ctx}: f64 {f} → Decimal не сконвертился"))
}

/// Maker = `limit_order().post_only(true)`, `OrderType::GTC` (или `GTD`,
/// если задан `expiration`). Цена и size обязательны и пройдут SDK-
/// валидацию по tick_size / lot_size / `fee_rate_bps` (последний берётся
/// SDK'ом из `markets/{condition_id}` при `build()`).
async fn build_maker_signable(
    client: &clob::Client<Authenticated<Normal>>,
    token_id: U256,
    req: &PostOrderRequest,
) -> Result<SignableOrder> {
    let price = req
        .price
        .expect("validated in validate_post_order_request");
    let shares = match req.amount {
        OrderAmount::Shares(s) => s,
        OrderAmount::UsdNotional(_) => unreachable!("validated"),
    };
    let price_dec = f64_to_decimal(price, "maker price")?;
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

/// Taker = `market_order()` с `OrderType::FAK` (заливаем сколько можем,
/// остаток отменяется). Если задан `price` — это явный worst-acceptable;
/// если задан только `max_slippage_pp` — cap от L1 из [`PostOrderRequest::strict_book`]
/// или HTTP [`clob::Client::order_book`] ([`compute_taker_cap_price`]); если оба
/// `None` — отдаём SDK'у самому вывести cutoff из books.
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

/// Считает worst-acceptable price для taker:
/// - `req.price.is_some()` → возвращаем его (приоритет над slippage).
/// - `req.max_slippage_pp.is_some()` → L1 + slip: при [`PostOrderRequest::strict_book`] =
///   `Some` берём лучший bid/ask из снимка (`history_sim::StrictBook`),
///   иначе — HTTP [`clob::Client::order_book`].
/// - оба `None` (`price` и `max_slippage_pp`) → `Ok(None)`, SDK сам режет cutoff.
async fn compute_taker_cap_price(
    client: &clob::Client<Authenticated<Normal>>,
    token_id: U256,
    req: &PostOrderRequest,
) -> Result<Option<Decimal>> {
    if let Some(p) = req.price {
        return Ok(Some(f64_to_decimal(p, "taker price")?));
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
                    let book_request =
                        OrderBookSummaryRequest::builder().token_id(token_id).build();
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
            (best_ask_dec + slip_dec).min(Decimal::ONE).max(Decimal::ZERO)
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
                    let book_request =
                        OrderBookSummaryRequest::builder().token_id(token_id).build();
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
            (best_bid_dec - slip_dec).max(Decimal::ZERO).min(Decimal::ONE)
        }
        _ => bail!(
            "post_order_on_clob: side={:?} не поддерживается (ожидается Buy/Sell)",
            req.side
        ),
    };

    Ok(Some(cap))
}

/// Лучший ask в [`StrictBook`]: первый уровень с положительной ценой и размером
/// (как [`crate::history_sim::book_fill_buy_strict`] / [`crate::history_sim::effective_implied_prob`]).
pub(crate) fn best_ask_strict(book: &StrictBook) -> Option<f64> {
    book.asks
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)
}

pub(crate) fn best_bid_strict(book: &StrictBook) -> Option<f64> {
    book.bids
        .iter()
        .find(|l| l.price > 0.0 && l.size > 0.0)
        .map(|l| l.price)
}

fn best_ask_sdk(book: &polymarket_client_sdk::clob::types::response::OrderBookSummaryResponse) -> Option<Decimal> {
    book.asks.iter().map(|l| l.price).min()
}

fn best_bid_sdk(book: &polymarket_client_sdk::clob::types::response::OrderBookSummaryResponse) -> Option<Decimal> {
    book.bids.iter().map(|l| l.price).max()
}

/// Параметры запроса к [`cancel_order_on_clob`].
#[derive(Debug, Clone)]
pub struct CancelOrderRequest {
    /// CLOB `orderID` (хэш ордера, формат `0x…`). Тот же `order_id`,
    /// что вернул [`PostOrderResult`] и которым матчатся события
    /// user-WS / локальные `open_order_id` / `close_order_id`.
    pub order_id: String,
    /// HTTP-таймаут на сам `DELETE /order`. Большие значения держат
    /// вызывающего под локом auth-снимка.
    pub timeout: Duration,
}

/// Ответ [`cancel_order_on_clob`] — распакованный
/// [`polymarket_client_sdk::clob::types::response::CancelOrdersResponse`]
/// под одиночный orderID.
///
/// HTTP 200 на `DELETE /order` сам по себе **не означает**, что ордер
/// реально отменён: CLOB может вернуть его в `not_canceled` (например,
/// уже сматчен/уже отменён/не найден — см. OpenAPI). Различай по
/// [`Self::canceled`] и [`Self::error_msg`].
#[derive(Debug, Clone)]
pub struct CancelOrderResult {
    /// Эхо `order_id` из запроса (упрощает логирование на стороне
    /// вызывающего).
    pub order_id: String,
    /// `true`, если `order_id` пришёл в массиве `canceled` ответа.
    /// `false` — если CLOB вернул его в `not_canceled` map'е (см.
    /// [`Self::error_msg`] для причины).
    pub canceled: bool,
    /// Сообщение-причина из `not_canceled[order_id]` при
    /// `canceled=false` (например `"Order not found or already canceled"`).
    /// Пустые строки превращаем в `None`.
    pub error_msg: Option<String>,
}

/// Отмена одиночного ордера на CLOB Polymarket
/// ([DELETE /order](https://docs.polymarket.com/api-reference/trade/cancel-single-order)).
/// Работает даже в cancel-only mode (см. 503 в OpenAPI: «cancels still
/// work in cancel-only mode»).
///
/// **Семантика результата:** успешный сетевой ответ (`Ok(_)`) ещё не
/// значит, что ордер действительно снят с книги — нужно проверить
/// [`CancelOrderResult::canceled`]:
/// - `canceled=true`  → orderID попал в `canceled[]` ответа CLOB.
/// - `canceled=false` → orderID был в `not_canceled` map'е, причина в
///   [`CancelOrderResult::error_msg`] (типичные причины: ордер уже
///   исполнен, уже отменён, не принадлежит этому API-ключу, не найден).
///
/// **Состояние Account** функция не трогает: вызывающий сам обновляет
/// локальные `OpenPosition`/`ClosingPosition`. Если ставка делается
/// на финальное подтверждение через user-WS — снимок состояния обновится
/// по событию из [`crate::account_ws`].
///
/// **Требования к Account**: `clob_authed = Some(_)` (signer не нужен —
/// эндпоинт использует только API-key аутентификацию, без EOA-подписи).
/// Иначе — `Err`.
pub async fn cancel_order_on_clob(
    account: &SharedAccount,
    request: CancelOrderRequest,
) -> Result<CancelOrderResult> {
    if request.timeout.is_zero() {
        bail!("cancel_order_on_clob: timeout=0 — DELETE /order не дождётся ответа");
    }
    if request.order_id.is_empty() {
        bail!("cancel_order_on_clob: пустой order_id");
    }

    // Снимок auth-клиента через ArcSwap.load() — без локов. `clob::Client` —
    // обёртка над `Arc<ClientInner>`, clone дешёвый. Signer для отмены не
    // требуется (в отличие от `post_order_on_clob`): SDK подписывает запрос
    // HMAC'ом по API-key creds.
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
        Ok(Err(err)) => bail!("cancel_order_on_clob: DELETE /order упал: {err:#}"),
        Err(_elapsed) => bail!(
            "cancel_order_on_clob: DELETE /order не уложился в {:?}",
            request.timeout
        ),
    };

    // Запросили один orderID — ожидаем, что ровно один из массивов
    // `canceled` / `not_canceled` непустой. Регистр / `0x`-префикс
    // CLOB'а сравнивать с нашим вводом не пытаемся (на практике
    // совпадает, но защищаемся проверкой по структуре, а не по
    // ID): если есть запись в `not_canceled` — берём её причину,
    // иначе — считаем отменённым по факту наличия в `canceled`.
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
        bail!(
            "cancel_order_on_clob: DELETE /order вернул пустой ответ \
             (canceled=[], not_canceled={{}}), order_id={}",
            request.order_id
        );
    };

    Ok(CancelOrderResult {
        order_id: request.order_id,
        canceled,
        error_msg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::{Account, POLY_PRIVATE_KEY_ENV, try_authenticate_clob_for_heartbeats};
    use crate::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT;
    use crate::util::{current_timestamp_ms, fetch_gamma_event_data_for_slug};
    use polymarket_client_sdk::clob::types::OrderStatusType;
    use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
    use polymarket_client_sdk::types::U256;
    const BTC_UPDOWN_5M_PERIOD_SEC: i64 = 300;
    const LIVE_ORDER_HTTP_TIMEOUT_SEC: u64 = 20;

    fn current_btc_updown_5m_slug(now_ms: i64) -> String {
        let poly_sec = now_ms / 1000;
        let window_start_sec = (poly_sec / BTC_UPDOWN_5M_PERIOD_SEC) * BTC_UPDOWN_5M_PERIOD_SEC;
        format!("btc-updown-5m-{window_start_sec}")
    }

    fn decimal_to_f64(d: &polymarket_client_sdk::types::Decimal) -> anyhow::Result<f64> {
        d.to_string()
            .parse::<f64>()
            .map_err(|err| anyhow::anyhow!("Decimal {d} → f64: {err}"))
    }

    fn min_taker_buy_usd_notional(min_order_size: f64, best_ask: f64) -> f64 {
        let raw = min_order_size * best_ask;
        let rounded = (raw * 100.0).ceil() / 100.0;
        rounded.max(0.01)
    }

    /// Live round-trip: taker BUY на минимальный notional в текущем 5m BTC
    /// up/down маркете, затем taker SELL всех полученных shares.
    ///
    /// ```bash
    /// POLY_PRIVATE_KEY=0x… \
    ///     cargo test --bin poly account_order::tests::live_taker_roundtrip_btc_updown_5m -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "live network: требует POLY_PRIVATE_KEY и USDC на Polymarket Safe; делает реальные CLOB-ордера"]
    async fn live_taker_roundtrip_btc_updown_5m() -> anyhow::Result<()> {
        let _ = dotenvy::dotenv();
        let _ = rustls::crypto::ring::default_provider().install_default();

        let private_key_set = std::env::var(POLY_PRIVATE_KEY_ENV)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .is_some();
        if !private_key_set {
            eprintln!(
                "live_taker_roundtrip_btc_updown_5m: {POLY_PRIVATE_KEY_ENV} не задан, тест пропущен",
            );
            return Ok(());
        }

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC))
            .build()?;
        let slug = current_btc_updown_5m_slug(current_timestamp_ms());
        let gamma = fetch_gamma_event_data_for_slug(&http, &slug).await?;
        let asset_id = gamma
            .currency_up_down_by_asset_id
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Gamma не вернул clobTokenIds для slug={slug}"))?;

        let account = Account::new_shared();
        try_authenticate_clob_for_heartbeats(&account).await;
        anyhow::ensure!(
            account.clob_authed.load().is_some(),
            "CLOB auth не поднялся — проверьте {POLY_PRIVATE_KEY_ENV} и логи [heartbeat]",
        );

        let token_id = U256::from_str(&asset_id)
            .with_context(|| format!("невалидный asset_id={asset_id} из Gamma slug={slug}"))?;
        let book_request = OrderBookSummaryRequest::builder()
            .token_id(token_id)
            .build();
        let book = account
            .clob
            .order_book(&book_request)
            .await
            .with_context(|| format!("order_book({asset_id}) для slug={slug}"))?;
        let min_order_size = decimal_to_f64(&book.min_order_size)?;
        let best_ask = best_ask_sdk(&book)
            .ok_or_else(|| anyhow::anyhow!("пустой asks book для asset_id={asset_id} slug={slug}"))?;
        let best_ask_f64 = decimal_to_f64(&best_ask)?;
        let buy_usd = min_taker_buy_usd_notional(min_order_size, best_ask_f64);
        let worst_acceptable_buy = (best_ask_f64 + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999);

        eprintln!(
            "live_taker_roundtrip_btc_updown_5m: slug={slug}, asset_id={asset_id}, \
             min_order_size={min_order_size:.4}, best_ask={best_ask_f64:.4}, buy_usd={buy_usd:.4}, \
             worst_acceptable_buy={worst_acceptable_buy:.4}",
        );

        let buy_result = post_order_on_clob(
            &account,
            PostOrderRequest {
                asset_id: asset_id.clone(),
                side: Side::Buy,
                role: OrderRole::Taker,
                amount: OrderAmount::UsdNotional(buy_usd),
                price: Some(worst_acceptable_buy),
                max_slippage_pp: None,
                expiration: None,
                timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
                strict_book: None,
            },
        )
        .await
        .with_context(|| format!("BUY taker slug={slug} asset_id={asset_id}"))?;
        anyhow::ensure!(
            buy_result.success,
            "BUY taker отвергнут: status={:?}, error_msg={:?}, order_id={}",
            buy_result.status,
            buy_result.error_msg,
            buy_result.order_id,
        );
        anyhow::ensure!(
            matches!(
                buy_result.status,
                OrderStatusType::Matched | OrderStatusType::Delayed
            ),
            "BUY taker не исполнен: status={:?}, order_id={}",
            buy_result.status,
            buy_result.order_id,
        );

        let shares_to_sell = decimal_to_f64(&buy_result.taking_amount)?;
        anyhow::ensure!(
            shares_to_sell > 0.0 && shares_to_sell.is_finite(),
            "BUY taker не дал shares: taking_amount={}, order_id={}",
            buy_result.taking_amount,
            buy_result.order_id,
        );

        let sell_result = post_order_on_clob(
            &account,
            PostOrderRequest {
                asset_id: asset_id.clone(),
                side: Side::Sell,
                role: OrderRole::Taker,
                amount: OrderAmount::Shares(shares_to_sell),
                price: None,
                max_slippage_pp: None,
                expiration: None,
                timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC),
                strict_book: None,
            },
        )
        .await
        .with_context(|| format!("SELL taker slug={slug} asset_id={asset_id}"))?;
        anyhow::ensure!(
            sell_result.success,
            "SELL taker отвергнут: status={:?}, error_msg={:?}, order_id={}",
            sell_result.status,
            sell_result.error_msg,
            sell_result.order_id,
        );
        anyhow::ensure!(
            matches!(
                sell_result.status,
                OrderStatusType::Matched | OrderStatusType::Delayed
            ),
            "SELL taker не исполнен: status={:?}, order_id={}",
            sell_result.status,
            sell_result.order_id,
        );

        eprintln!(
            "live_taker_roundtrip_btc_updown_5m OK: buy_order_id={}, sell_order_id={}, \
             buy_usd={buy_usd:.4}, shares_sold={shares_to_sell:.4}",
            buy_result.order_id,
            sell_result.order_id,
        );
        Ok(())
    }
}
