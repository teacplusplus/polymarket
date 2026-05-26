use anyhow::Context;
use serde_json::Value;
use std::collections::HashMap;

use crate::constants::CurrencyUpDownOutcome;
use polymarket_client_sdk::gamma;
use polymarket_client_sdk::gamma::types::request::MarketBySlugRequest;
use polymarket_client_sdk::gamma::types::response::Market;

/// Макс. длина имени в [`sanitized_filename_from_gamma_question`] до обрезки с `...`.
const MAX_SANITIZED_FILENAME_LEN: usize = 180;

pub fn current_timestamp_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis() as i64
}

/// Имя файла из Gamma `question`: недопустимые символы → `_`, обрезка по длине.
pub fn sanitized_filename_from_gamma_question(q: Option<&str>) -> String {
    let raw = q.unwrap_or("no_question");
    let s: String = raw
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    if s.len() > MAX_SANITIZED_FILENAME_LEN {
        format!("{}...", &s[..MAX_SANITIZED_FILENAME_LEN])
    } else {
        s
    }
}

/// Абсолютный путь → `file://` URI безопасными символами, остальное `%XX`.
pub fn encode_path_as_file_uri(abs_path: &str) -> String {
    let mut out = String::from("file://");
    for ch in abs_path.chars() {
        match ch {
            '/' | 'A'..='Z' | 'a'..='z' | '0'..='9' | '.' | '_' | '-' | '(' | ')' => out.push(ch),
            _ => {
                for b in ch.encode_utf8(&mut [0u8; 4]).bytes() {
                    use std::fmt::Write as _;
                    let _ = write!(&mut out, "%{b:02X}");
                }
            }
        }
    }
    out
}

pub struct CurrencyEventSlugData {
    /// Исход токена по `asset_id`: Up / Down.
    pub currency_up_down_by_asset_id: HashMap<String, CurrencyUpDownOutcome>,
    /// `conditionId` маркета (`0x…`).
    pub market_id: Option<String>,
    /// Старт окна (`eventStartTime` / fallback из Gamma).
    pub event_start_ms: Option<i64>,
    /// Конец окна (`endDate`, не `umaEndDate`).
    pub event_end_ms: Option<i64>,
    /// CLOB `orderMinSize` из Gamma (`order_min_size` в SDK).
    pub min_order_size: Option<f64>,
    /// Поле `question` маркета Gamma.
    pub gamma_question: Option<String>,
}

/// Price-to-beat окна через Vatic [`targets/timestamp`](https://docs.vatic.trading/api-reference/targets/timestamp.md).
/// Slug `{currency}-updown-{5m|15m}-{window_start_sec}`, `currency` в URL — lower-case.
pub async fn fetch_price_to_beat_from_vatic_api(
    http: &reqwest::Client,
    slug: &str,
    currency: &str,
) -> anyhow::Result<f64> {
    let (window_sec, market_type) = vatic_slug_window_sec_and_market_type(currency, slug)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "нет priceToBeat: slug {slug:?} не имеет формата {{currency}}-updown-{{5m|15m}}-{{ts}} (currency={currency})"
            )
        })?;
    let asset = currency.to_lowercase();
    let url = format!(
        "https://api.vatic.trading/api/v1/targets/timestamp?asset={asset}&type={market_type}&timestamp={window_sec}"
    );
    let response = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("vatic GET {url}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "нет priceToBeat: vatic вернул HTTP {status} для slug={slug:?} url={url} body={body}"
        );
    }
    let body: Value = response
        .json()
        .await
        .with_context(|| format!("vatic JSON {url}"))?;
    let price = body
        .get("price")
        .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "нет priceToBeat: в ответе vatic нет числового поля `price` для slug={slug:?} body={body}"
            )
        })?;
    if !price.is_finite() || price <= 0.0 {
        anyhow::bail!(
            "нет priceToBeat: некорректное значение {price} в ответе vatic для slug={slug:?}"
        );
    }
    Ok(price)
}

/// Slug Polymarket → `(window_start_sec, type)` для Vatic (`5min` / `15min`).
fn vatic_slug_window_sec_and_market_type(
    currency: &str,
    slug: &str,
) -> Option<(i64, &'static str)> {
    let prefix = format!("{}-updown-", currency.to_lowercase());
    let rest = slug.strip_prefix(prefix.as_str())?;
    let (mid, sec_str) = rest.rsplit_once('-')?;
    let window_sec: i64 = sec_str.parse().ok()?;
    let market_type = match mid {
        "5m" => "5min",
        "15m" => "15min",
        _ => return None,
    };
    Some((window_sec, market_type))
}

#[inline]
fn gamma_datetime_to_epoch_ms(dt: Option<chrono::DateTime<chrono::Utc>>) -> Option<i64> {
    dt.map(|d| d.timestamp_millis())
}

#[inline]
fn gamma_decimal_to_f64(d: polymarket_client_sdk::types::Decimal) -> Option<f64> {
    d.to_string()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// Собирает [`CurrencyEventSlugData`] из ответа Gamma SDK [`Market`].
/// Цепочка fallback для старта окна близка к прежнему разбору JSON:
/// `event_start_time` маркета → `events[0].start_time` → `start_date` маркета → `events[0].start_date`;
/// конец: `end_date` маркета → `events[0].end_date`.
pub fn currency_event_slug_data_from_gamma_market(
    m: &Market,
) -> anyhow::Result<CurrencyEventSlugData> {
    let outcomes = m.outcomes.clone().unwrap_or_default();
    if outcomes.is_empty() {
        anyhow::bail!("пустой outcomes в ответе Gamma для маркета");
    }

    let clob_token_ids: Vec<String> = m
        .clob_token_ids
        .as_ref()
        .map(|ids| ids.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    if clob_token_ids.is_empty() {
        anyhow::bail!("ни одного clobTokenId в ответе Gamma для маркета");
    }

    let currency_up_down_by_asset_id = zip_outcomes_clob_to_up_code(&outcomes, &clob_token_ids)
        .context("outcomes vs clobTokenIds")?;

    let gamma_question = m.question.clone();
    let min_order_size = m.order_min_size.and_then(gamma_decimal_to_f64);

    let (market_id, event_start_ms, event_end_ms) = if let Some(cid_b256) = m.condition_id {
        let cid = format!("{cid_b256:#x}");
        let event0 = m.events.as_ref().and_then(|ev| ev.first());

        let start_ms = gamma_datetime_to_epoch_ms(m.event_start_time)
            .or_else(|| event0.and_then(|e| gamma_datetime_to_epoch_ms(e.start_time)))
            .or_else(|| gamma_datetime_to_epoch_ms(m.start_date))
            .or_else(|| event0.and_then(|e| gamma_datetime_to_epoch_ms(e.start_date)));

        let end_ms = gamma_datetime_to_epoch_ms(m.end_date)
            .or_else(|| event0.and_then(|e| gamma_datetime_to_epoch_ms(e.end_date)));

        (Some(cid), start_ms, end_ms)
    } else {
        (None, None, None)
    };

    Ok(CurrencyEventSlugData {
        currency_up_down_by_asset_id,
        market_id,
        event_start_ms,
        event_end_ms,
        min_order_size,
        gamma_question,
    })
}

/// `GET /markets/slug/{slug}` через [`gamma::Client`] и разбор в [`CurrencyEventSlugData`].
pub async fn fetch_gamma_event_data_for_gamma_client(
    client: &gamma::Client,
    slug: &str,
) -> anyhow::Result<CurrencyEventSlugData> {
    let request = MarketBySlugRequest::builder()
        .slug(slug.to_string())
        .build();
    let market = client
        .market_by_slug(&request)
        .await
        .with_context(|| format!("Gamma market_by_slug slug={slug:?}"))?;
    currency_event_slug_data_from_gamma_market(&market)
        .with_context(|| format!("разбор ответа Gamma slug={slug:?}"))
}

fn gamma_outcome_label_to_currency_kind(label: &str) -> Option<CurrencyUpDownOutcome> {
    match label.trim().to_ascii_lowercase().as_str() {
        "up" => Some(CurrencyUpDownOutcome::Up),
        "down" => Some(CurrencyUpDownOutcome::Down),
        _ => None,
    }
}

fn zip_outcomes_clob_to_up_code(
    outcomes: &[String],
    clob_ids: &[String],
) -> anyhow::Result<HashMap<String, CurrencyUpDownOutcome>> {
    if outcomes.len() != clob_ids.len() {
        anyhow::bail!(
            "Gamma: len(outcomes)={} != len(clobTokenIds)={}",
            outcomes.len(),
            clob_ids.len()
        );
    }
    let mut map = HashMap::new();
    for (label, token_id) in outcomes.iter().zip(clob_ids.iter()) {
        if let Some(code) = gamma_outcome_label_to_currency_kind(label) {
            map.insert(token_id.clone(), code);
        }
    }
    Ok(map)
}

#[derive(Debug)]
pub struct CountryAndIp {
    /// `blocked` из [`detect_country_and_ip`] (`GET /api/geoblock` Polymarket).
    pub blocked: bool,
    /// Код страны (если есть).
    pub country: Option<String>,
    /// Регион/штат (если есть).
    pub region: Option<String>,
    /// Внешний IP (если есть).
    pub ip: Option<String>,
}

/// Страна, IP и флаг геоблока Polymarket (`GET https://polymarket.com/api/geoblock`); сбой → `None`.
/// `http` — обычно [`Account::http`](crate::account::Account::http).
pub async fn detect_country_and_ip(http: &reqwest::Client) -> Option<CountryAndIp> {
    let resp = http
        .get("https://polymarket.com/api/geoblock")
        .timeout(std::time::Duration::from_secs(5))
        .header(reqwest::header::USER_AGENT, "curl/8.9.1")
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: Value = resp.json().await.ok()?;
    let blocked = body.get("blocked")?.as_bool()?;
    let pick = |k: &str| -> Option<String> {
        body.get(k)
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    Some(CountryAndIp {
        blocked,
        country: pick("country"),
        region: pick("region"),
        ip: pick("ip"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Лайв: Vatic priceToBeat для `btc-updown-5m-1778267400` ≈ $80,061.62. `cargo test … -- --ignored`.
    #[tokio::test]
    #[ignore = "live network: GET https://api.vatic.trading/api/v1/targets/timestamp"]
    async fn live_fetch_price_to_beat_from_vatic_btc_updown_5m_1778267400() -> anyhow::Result<()> {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let account = crate::account::Account::new_shared();

        let slug = "btc-updown-5m-1778267400";
        let price = fetch_price_to_beat_from_vatic_api(account.http.as_ref(), slug, "btc").await?;

        let price_2dp = (price * 100.0).round() / 100.0;
        anyhow::ensure!(
            (price_2dp - 80061.62).abs() < 1e-9,
            "ожидался priceToBeat $80,061.62 для slug={slug}, получено {price} (rounded 2dp = {price_2dp})"
        );
        Ok(())
    }
}
