//! CLOB: [`post_order_on_clob`] ([POST /order](https://docs.polymarket.com/api-reference/trade/post-a-new-order)),
//! [`cancel_order_on_clob`](https://docs.polymarket.com/api-reference/trade/cancel-single-order).
//! `clob_authed` / `clob_signer` — из [`crate::account::Account`] ([`crate::account::try_authenticate_clob_for_heartbeats`]).

use crate::account::{POLY_PRIVATE_KEY_ENV, SharedAccount};
use crate::history_sim::StrictBook;
use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::response::PostOrderResponse;
use polymarket_client_sdk::clob::types::{Amount, OrderStatusType, OrderType, Side, SignableOrder};
use polymarket_client_sdk::types::{Decimal, U256};
use std::str::FromStr;
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
    /// GTD для maker; у taker `None`.
    pub expiration: Option<DateTime<Utc>>,
    /// Таймаут HTTP только на `POST /order`.
    pub timeout: Duration,
    /// При slip-cap без `price`: L1 без лишнего GET /book.
    pub(crate) strict_book: Option<StrictBook>,
}

/// Распакованный ответ POST /order (поля как у SDK).
#[derive(Debug, Clone)]
pub struct PostOrderResult {
    /// `orderID` для user-WS и `*_order_id`.
    pub order_id: String,
    /// Live / Matched / Delayed / …
    pub status: OrderStatusType,
    /// HTTP-ответ успешно обработан (ордер мог остаться live).
    pub success: bool,
    /// `makingAmount`.
    pub making_amount: Decimal,
    /// `takingAmount`.
    pub taking_amount: Decimal,
    /// `errorMsg` если success=false или отклонено.
    pub error_msg: Option<String>,
    /// `transactionsHashes`, строки `0x…`.
    pub transaction_hashes: Vec<String>,
    /// `tradeIDs`.
    pub trade_ids: Vec<String>,
}

/// Подписать ордер EOA и отправить на CLOB. Нужны `clob_authed` + `clob_signer`. Account не изменяется.
pub async fn post_order_on_clob(
    account: &SharedAccount,
    request: PostOrderRequest,
) -> Result<PostOrderResult> {
    validate_post_order_request(&request)?;

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

/// Ошибки комбинаций полей до сети/SDK `build`.
fn validate_post_order_request(req: &PostOrderRequest) -> Result<()> {
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

/// `f64` → `Decimal` через строку (стабильнее двоичного float).
fn f64_to_decimal(f: f64, ctx: &str) -> Result<Decimal> {
    if !f.is_finite() {
        bail!("post_order_on_clob: {ctx}: значение {f} не finite");
    }
    f.to_string()
        .parse::<Decimal>()
        .with_context(|| format!("post_order_on_clob: {ctx}: f64 {f} → Decimal не сконвертился"))
}

/// `limit_order` post_only, GTC или GTD если есть `expiration`.
async fn build_maker_signable(
    client: &clob::Client<Authenticated<Normal>>,
    token_id: U256,
    req: &PostOrderRequest,
) -> Result<SignableOrder> {
    let price = req.price.expect("validated in validate_post_order_request");
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

fn best_ask_sdk(
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
    /// CLOB `orderID` (совпадает с [`PostOrderResult::order_id`]).
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

/// `DELETE /order` под API-key; нужен только `clob_authed`. `Ok` не гарантирует снятие — проверьте поля результата.
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

    let auth_client = (**account.clob_authed.load()).clone().ok_or_else(|| {
        anyhow!(
            "cancel_order_on_clob: clob_authed=None — CLOB не аутентифицирован, проверьте {POLY_PRIVATE_KEY_ENV} и [heartbeat] CLOB authenticate"
        )
    })?;

    let resp =
        match tokio::time::timeout(request.timeout, auth_client.cancel_order(&request.order_id))
            .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(err)) => bail!("cancel_order_on_clob: DELETE /order упал: {err:#}"),
            Err(_elapsed) => bail!(
                "cancel_order_on_clob: DELETE /order не уложился в {:?}",
                request.timeout
            ),
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

    /// Период в секундах у slug `btc-updown-5m-{ts}`.
    const BTC_UPDOWN_5M_PERIOD_SEC: i64 = 300;
    /// Общий HTTP timeout в live-сценарии теста.
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

    /// Live BUY→SELL taker минимального notional по текущему 5m BTC up/down рынку.
    ///
    /// ```bash
    /// POLY_PRIVATE_KEY=0x… \
    ///     cargo test --bin poly account_order::tests::live_taker_roundtrip_btc_updown_5m -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "live network: требует POLY_PRIVATE_KEY и pUSD на Polymarket Safe; делает реальные CLOB-ордера"]
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
        let best_ask = best_ask_sdk(&book).ok_or_else(|| {
            anyhow::anyhow!("пустой asks book для asset_id={asset_id} slug={slug}")
        })?;
        let best_ask_f64 = decimal_to_f64(&best_ask)?;
        let buy_usd = min_taker_buy_usd_notional(min_order_size, best_ask_f64);
        let worst_acceptable_buy =
            (best_ask_f64 + SIM_MAX_SLIPPAGE_FROM_L1_PCT).clamp(0.001, 0.999);

        eprintln!(
            "live_taker_roundtrip_btc_updown_5m: slug={slug}, asset_id={asset_id}, \
             min_order_size={min_order_size:.4}, best_ask={best_ask_f64:.4}, buy_usd={buy_usd:.4}, \
             worst_acceptable_buy={worst_acceptable_buy:.4}",
        );

        let buy_result = post_order_on_clob(
            &account,
            PostOrderRequest {
                asset_id: asset_id.clone(),                // Gamma outcome token
                side: Side::Buy,                           // вход long
                role: OrderRole::Taker,                    // FAK
                amount: OrderAmount::UsdNotional(buy_usd), // мин. допустимый notional
                price: Some(worst_acceptable_buy),         // явный worst-acceptable
                max_slippage_pp: None,                     // не используем slip от L1
                expiration: None,                          // taker
                timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC), // POST /order
                strict_book: None,                         // GET book выше
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
                asset_id: asset_id.clone(),                                // тот же токен
                side: Side::Sell,                                          // unwind
                role: OrderRole::Taker,                                    // FAK
                amount: OrderAmount::Shares(shares_to_sell),               // весь fill с BUY
                price: None,                                               // маркет-продажа в bid
                max_slippage_pp: None,                                     // без cap
                expiration: None,                                          // taker
                timeout: Duration::from_secs(LIVE_ORDER_HTTP_TIMEOUT_SEC), // POST /order
                strict_book: None,                                         // нет локального book
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
            buy_result.order_id, sell_result.order_id,
        );
        Ok(())
    }
}
