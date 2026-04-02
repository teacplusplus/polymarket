
use std::collections::HashMap;
use std::sync::Arc;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::clob::types::request::PriceHistoryRequest;
use polymarket_client_sdk::clob::types::{Interval, TimeRange};
use polymarket_client_sdk::gamma;
use polymarket_client_sdk::gamma::types::request::MarketsRequest;
use polymarket_client_sdk::gamma::types::response::Market as GammaMarket;
use polymarket_client_sdk::types::{B256, U256, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;

const API_JOB_QUEUE: usize = 1024;

/// Публичный HTTP GraphQL orderbook subgraph (Goldsky).
pub const ORDERBOOK_SUBGRAPH_URL: &str =
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn";

/// В subgraph `makerAssetId == "0"` для коллатерала (USDC) в типичных филлах CLOB.
pub const SUBGRAPH_COLLATERAL_ASSET_ID: &str = "0";

const SUBGRAPH_PAGE: usize = 1000;

/// Unix UTC (мс) для строки `end_date_rfc3339` из Gamma (`ApiAssetStable`).
pub fn event_end_unix_ms_from_rfc3339(end_date_rfc3339: Option<&str>) -> Option<i64> {
    end_date_rfc3339.and_then(|rfc3339_str| {
        chrono::DateTime::parse_from_rfc3339(rfc3339_str.trim())
            .ok()
            .map(|dt| dt.timestamp_millis())
    })
}

/// Хранилище по рынку (condition_id) → по токену (clob asset_id).
pub type ApiMarketDataStore = HashMap<String, HashMap<String, ApiAssetSnapshot>>;

/// Сводка по токену: [ApiAssetStable] (каталог) + динамический блок [ApiAssetLive].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAssetSnapshot {
    /// Метаданные и идентификаторы (обычно обновляют реже, чем `live`).
    pub stable: ApiAssetStable,
    /// Текущие котировки, объёмы и флаги состояния после последнего опроса API.
    pub live: ApiAssetLive,
}

/// Идентичность и описание рынка из Gamma (и прочие поля, которые для модели «медленные»).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAssetStable {
    /// CLOB token id (`POLYMARKET_ASSET_IDS` / WebSocket).
    pub asset_id: String,
    /// Condition id: общий ключ Yes/No в одном рынке.
    pub condition_id: Option<String>,
    /// Внутренний id маркета в Gamma (строка в их API), удобен для повторных запросов и отладки.
    pub gamma_market_id: Option<String>,
    /// Id родительского события в Gamma.
    pub event_id: Option<String>,
    /// Формулировка рынка (вопрос) из Gamma.
    pub question: Option<String>,
    /// Список исходов (например «Yes», «No» или имена кандидатов), из Gamma.
    pub outcomes: Vec<String>,
    /// Поле категории маркета в Gamma (широкая тема: политика, спорт и т.д.).
    pub category: Option<String>,
    /// Теги маркета (подписи/slug), когда запрос к Gamma с `include_tag=true`.
    pub tags: Vec<String>,
    /// Плановое окончание (RFC 3339): `end_date` или конец дня по `end_date_iso`.
    pub end_date_rfc3339: Option<String>,
    /// Плановое начало / старт окна события (RFC 3339): `start_date` или полночь UTC по `start_date_iso`.
    pub start_date_rfc3339: Option<String>,
    /// Neg-risk у события Gamma (`events[0].neg_risk`): особая схема расчёта/клиринга.
    pub neg_risk_gamma: Option<bool>,
    /// Флаг `neg_risk_other` у маркета в Gamma (связанные neg-risk / мульти-исходные рынки).
    pub neg_risk_other: Option<bool>,
    /// `order_min_size` в Gamma; стакан — из WS.
    pub min_order_size: Option<String>,
}

/// Величины и состояние, которые имеет смысл обновлять при каждом опросе API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAssetLive {
    /// UTC сборки этого блока (мс), для стыковки с WS-кадрами.
    pub fetched_at_ms: i64,
    /// Секунды до `stable.end_date_rfc3339` от текущего UTC на момент сборки; отрицательное — срок уже прошёл.
    pub time_to_resolution_sec: Option<i64>,
    /// Маркет помечен как активный в Gamma (`active`).
    pub active: Option<bool>,
    /// Маркет закрыт (`closed`): обычно нет новой торговли в штатном режиме.
    pub closed: Option<bool>,
    /// Эвристика «исход зафиксирован» (`uma_resolution_status` / `automatically_resolved`).
    pub resolved: Option<bool>,
    /// Принимает ли CLOB заявки сейчас (`accepting_orders` в Gamma).
    pub accepting_orders: Option<bool>,
    /// Минимальный шаг цены, CLOB `GET /tick-size`, строка с десятичным значением.
    pub tick_size: Option<String>,
    /// Ряд точек истории цен CLOB (`prices-history`: время в сек, цена строкой).
    pub price_history: Vec<PriceHistoryPoint>,
    /// Сообщения об ошибках отдельных HTTP-вызовов; остальные поля `live` могут быть частично пустыми.
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceHistoryPoint {
    pub t_sec: i64,
    pub price: String,
}

#[derive(Debug, Clone)]
pub enum ApiDataJob {
    /// Полная выгрузка по одному clob token id.
    FetchAssetFull { asset_id: String },
    /// Для каждой группировки [crate::project_manager::FRAME_BUILD_INTERVAL_SECS] — до `past_bucket_count`
    /// последовательных бакетов назад от текущего выровненного времени; слияние с уже существующими кадрами (WS/повтор).
    BuildHistoricalXFrames {
        asset_id: String,
        past_bucket_count: usize,
    },
}

pub struct ApiDataHub {
    pub api_job_sender: mpsc::Sender<ApiDataJob>,
}

pub fn make_api_channel() -> (Arc<ApiDataHub>, mpsc::Receiver<ApiDataJob>) {
    let (api_job_sender, api_job_receiver) = mpsc::channel(API_JOB_QUEUE);
    (
        Arc::new(ApiDataHub { api_job_sender }),
        api_job_receiver,
    )
}

/// Строка `orderFilledEvents` из orderbook subgraph.
#[derive(Debug, Clone)]
pub struct OrderFilledRow {
    pub timestamp_sec: i64,
    pub maker_asset_id: String,
    pub taker_asset_id: String,
    pub maker_amount_raw: String,
    pub taker_amount_raw: String,
}

#[derive(Debug, Deserialize)]
struct GqlOrderFilledEventsData {
    #[serde(rename = "orderFilledEvents")]
    order_filled_events: Vec<GqlOrderFilledEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GqlOrderFilledEvent {
    timestamp: String,
    maker_asset_id: String,
    taker_asset_id: String,
    maker_amount_filled: String,
    taker_amount_filled: String,
}

/// Пагинация по `timestamp` (asc): все события с `makerAssetId` или `takerAssetId` = `asset_id`.
pub async fn fetch_order_filled_events_for_range(
    http: &reqwest::Client,
    asset_id: &str,
    t_min_sec: i64,
    t_max_sec: i64,
) -> anyhow::Result<Vec<OrderFilledRow>> {
    crate::poly_trace::api_line(
        "GraphQL (Goldsky orderbook subgraph)",
        &format!(
            "POST {ORDERBOOK_SUBGRAPH_URL} orderFilledEvents where asset={asset_id} timestamp in [{t_min_sec}, {t_max_sec}] sec (paginated)"
        ),
    );
    let mut out = Vec::new();
    let mut cursor_ts = t_min_sec;
    loop {
        crate::poly_trace::api_line(
            "GraphQL page",
            &format!(
                "orderFilledEvents first={SUBGRAPH_PAGE} orderBy=timestamp asc cursor_ts={cursor_ts} tmax={t_max_sec}"
            ),
        );
        let query = r#"
            query Fills($first: Int!, $tmin: BigInt!, $tmax: BigInt!, $asset: String!) {
              orderFilledEvents(
                first: $first
                orderBy: timestamp
                orderDirection: asc
                where: {
                  or: [
                    {
                      timestamp_gte: $tmin
                      timestamp_lte: $tmax
                      makerAssetId: $asset
                    }
                    {
                      timestamp_gte: $tmin
                      timestamp_lte: $tmax
                      takerAssetId: $asset
                    }
                  ]
                }
              ) {
                timestamp
                makerAssetId
                takerAssetId
                makerAmountFilled
                takerAmountFilled
              }
            }"#;
        let body = json!({
            "query": query,
            "variables": {
                "first": SUBGRAPH_PAGE,
                "tmin": cursor_ts.to_string(),
                "tmax": t_max_sec.to_string(),
                "asset": asset_id,
            }
        });
        let resp = http
            .post(ORDERBOOK_SUBGRAPH_URL)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            anyhow::bail!("subgraph HTTP {status}: {text}");
        }
        let response_json: serde_json::Value = serde_json::from_str(&text)?;
        if let Some(err) = response_json.get("errors") {
            anyhow::bail!("subgraph GraphQL errors: {err}");
        }
        let data: GqlOrderFilledEventsData = serde_json::from_value(
            response_json
                .get("data")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("subgraph: missing data"))?,
        )?;
        if data.order_filled_events.is_empty() {
            break;
        }
        let batch_len = data.order_filled_events.len();
        let mut max_ts = cursor_ts;
        for order_filled_event in data.order_filled_events {
            let fill_timestamp_sec = order_filled_event.timestamp.parse::<i64>()?;
            max_ts = max_ts.max(fill_timestamp_sec);
            out.push(OrderFilledRow {
                timestamp_sec: fill_timestamp_sec,
                maker_asset_id: order_filled_event.maker_asset_id,
                taker_asset_id: order_filled_event.taker_asset_id,
                maker_amount_raw: order_filled_event.maker_amount_filled,
                taker_amount_raw: order_filled_event.taker_amount_filled,
            });
        }
        if batch_len < SUBGRAPH_PAGE {
            break;
        }
        cursor_ts = max_ts.saturating_add(1);
        if cursor_ts > t_max_sec {
            break;
        }
    }
    Ok(out)
}

/// Последняя точка [PriceHistoryPoint] с `t_sec <= t_sec_cutoff` (CLOB fidelity, обычно секунды).
pub fn price_at_or_before(history: &[PriceHistoryPoint], t_sec_cutoff: i64) -> Option<f64> {
    history
        .iter()
        .filter(|price_history_point| price_history_point.t_sec <= t_sec_cutoff)
        .max_by_key(|price_history_point| price_history_point.t_sec)
        .and_then(|price_history_point| price_history_point.price.parse::<f64>().ok())
}

pub async fn fetch_asset_snapshot(
    gamma: &gamma::Client,
    clob: &clob::Client,
    asset_id: &str,
) -> anyhow::Result<ApiAssetSnapshot> {
    crate::poly_trace::api_line(
        "Gamma (SDK gamma::Client)",
        &format!(
            "markets(MarketsRequest {{ clob_token_ids: [{asset_id}], include_tag: true, limit: 10 }}) — базовый URL из polymarket-client-sdk"
        ),
    );
    crate::poly_trace::api_line(
        "CLOB (SDK clob::Client)",
        &format!(
            "tick_size({asset_id}) + price_history(market={asset_id}, Interval::Max, fidelity=60) — базовый URL из polymarket-client-sdk"
        ),
    );
    let mut errors = Vec::new();
    let token_id: U256 = asset_id
        .trim()
        .parse()
        .map_err(|parse_error| {
            anyhow::anyhow!("invalid POLYMARKET token id {asset_id}: {parse_error}")
        })?;

    let markets_req = MarketsRequest::builder()
        .clob_token_ids(vec![token_id])
        .include_tag(true)
        .limit(10)
        .build();

    let gamma_markets = match gamma.markets(&markets_req).await {
        Ok(markets) => markets,
        Err(gamma_error) => {
            errors.push(format!("gamma.markets: {gamma_error}"));
            Vec::new()
        }
    };

    let market_opt = gamma_markets.into_iter().next();
    let condition_id: Option<B256> = market_opt
        .as_ref()
        .and_then(|gamma_market| gamma_market.condition_id);
    let condition_hex = condition_id.map(|condition_b256| format!("{condition_b256:#x}"));

    let mut question = None;
    let mut outcomes = Vec::new();
    let mut category = None;
    let mut tags = Vec::new();
    let mut end_date_rfc3339 = None;
    let mut start_date_rfc3339 = None;
    let mut neg_risk_gamma = None;
    let mut neg_risk_other = None;
    let mut gamma_market_id = None;
    let mut event_id = None;
    let mut min_order_size_gamma = None;
    let mut time_to_resolution_sec = None;
    let mut active = None;
    let mut closed = None;
    let mut resolved = None;
    let mut accepting_orders = None;

    if let Some(ref gamma_market) = market_opt {
        tags = gamma_market
            .tags
            .as_ref()
            .map(|tag_list| {
                tag_list
                    .iter()
                    .filter_map(|gamma_tag| {
                        gamma_tag.label.clone().or_else(|| gamma_tag.slug.clone())
                    })
                    .collect()
            })
            .unwrap_or_default();
        outcomes = gamma_market.outcomes.clone().unwrap_or_default();
        let end_dt = gamma_market.end_date.or_else(|| {
            gamma_market
                .end_date_iso
                .and_then(|date| date.and_hms_opt(23, 59, 59).map(|dt| dt.and_utc()))
        });
        end_date_rfc3339 = end_dt.map(|end_datetime| end_datetime.to_rfc3339());
        let start_dt = gamma_market.start_date.or_else(|| {
            gamma_market
                .start_date_iso
                .and_then(|date| date.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc()))
        });
        start_date_rfc3339 = start_dt.map(|start_datetime| start_datetime.to_rfc3339());
        time_to_resolution_sec = end_dt.map(|end_datetime| (end_datetime - Utc::now()).num_seconds());
        event_id = gamma_market
            .events
            .as_ref()
            .and_then(|events| events.first())
            .map(|gamma_event| gamma_event.id.clone());

        question = gamma_market.question.clone();
        category = gamma_market.category.clone();
        neg_risk_gamma = gamma_market
            .events
            .as_ref()
            .and_then(|events| events.first())
            .and_then(|gamma_event| gamma_event.neg_risk);
        neg_risk_other = gamma_market.neg_risk_other;
        gamma_market_id = Some(gamma_market.id.clone());
        min_order_size_gamma = gamma_market
            .order_min_size
            .map(|decimal| decimal.normalize().to_string());
        active = gamma_market.active;
        closed = gamma_market.closed;
        resolved = infer_resolved(gamma_market);
        accepting_orders = gamma_market.accepting_orders;
    }

    let hist_req = PriceHistoryRequest::builder()
        .market(token_id)
        .time_range(TimeRange::from_interval(Interval::Max))
        .fidelity(60)
        .build();

    let tick_fut = clob.tick_size(token_id);
    let hist_fut = clob.price_history(&hist_req);

    let (tick_res, hist_res) = tokio::join!(tick_fut, hist_fut);

    let tick_size = match tick_res {
        Ok(tick_size_response) => Some(
            tick_size_response
                .minimum_tick_size
                .as_decimal()
                .normalize()
                .to_string(),
        ),
        Err(clob_error) => {
            errors.push(format!("clob.tick_size: {clob_error}"));
            None
        }
    };

    let price_history: Vec<PriceHistoryPoint> = match hist_res {
        Ok(price_history_response) => price_history_response
            .history
            .into_iter()
            .map(|history_point| PriceHistoryPoint {
                t_sec: history_point.t,
                price: history_point.p.normalize().to_string(),
            })
            .collect(),
        Err(clob_error) => {
            errors.push(format!("clob.price_history: {clob_error}"));
            Vec::new()
        }
    };

    let fetched_at_ms = Utc::now().timestamp_millis();

    Ok(ApiAssetSnapshot {
        stable: ApiAssetStable {
            asset_id: asset_id.trim().to_string(),
            condition_id: condition_hex,
            gamma_market_id,
            event_id,
            question,
            outcomes,
            category,
            tags,
            end_date_rfc3339,
            start_date_rfc3339,
            neg_risk_gamma,
            neg_risk_other,
            min_order_size: min_order_size_gamma,
        },
        live: ApiAssetLive {
            fetched_at_ms,
            time_to_resolution_sec,
            active,
            closed,
            resolved,
            accepting_orders,
            tick_size,
            price_history,
            errors,
        },
    })
}

fn infer_resolved(gamma_market: &GammaMarket) -> Option<bool> {
    if gamma_market.automatically_resolved == Some(true) {
        return Some(true);
    }
    if let Some(ref status) = gamma_market.uma_resolution_status {
        let lower = status.to_lowercase();
        if lower.contains("resolved") {
            return Some(true);
        }
    }
    None
}
