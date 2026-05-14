use anyhow::Context;
use chrono::DateTime;
use serde_json::Value;
use std::collections::HashMap;

use crate::constants::CurrencyUpDownOutcome;

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
    /// Старт окна по `conditionId` (`eventStartTime` / fallback из Gamma).
    pub market_event_start_ms: HashMap<String, Option<i64>>,
    /// Конец окна по `conditionId` (`endDate`, не `umaEndDate`).
    pub market_event_end_ms: HashMap<String, Option<i64>>,
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

pub async fn fetch_gamma_event_data_for_slug(
    http: &reqwest::Client,
    slug: &str,
) -> anyhow::Result<CurrencyEventSlugData> {
    let url = format!("https://gamma-api.polymarket.com/markets/slug/{slug}");
    let response = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("Gamma GET {url}"))?;
    let response = response
        .error_for_status()
        .with_context(|| format!("Gamma HTTP error slug={slug}"))?;
    let v: Value = response
        .json()
        .await
        .with_context(|| format!("Gamma JSON slug={slug}"))?;

    let clob_token_ids = parse_clob_token_ids_from_gamma_market(&v)
        .with_context(|| format!("clobTokenIds slug={slug}"))?;
    let outcomes =
        parse_outcomes_from_gamma_market(&v).with_context(|| format!("outcomes slug={slug}"))?;
    if outcomes.is_empty() {
        anyhow::bail!("пустой outcomes в ответе Gamma для slug={slug:?}");
    }
    let currency_up_down_by_asset_id = zip_outcomes_clob_to_up_code(&outcomes, &clob_token_ids)
        .with_context(|| format!("outcomes vs clobTokenIds slug={slug}"))?;

    let gamma_question = v
        .get("question")
        .and_then(|x| x.as_str())
        .map(str::to_string);

    let mut market_event_start_ms = HashMap::new();
    let mut market_event_end_ms = HashMap::new();

    if let Some(cid) = v
        .get("conditionId")
        .and_then(|x| x.as_str())
        .map(str::to_string)
    {
        let event0 = v
            .get("events")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        let start_ms = gamma_json_date_ms(v.get("eventStartTime"))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("eventStartTime"))))
            .or_else(|| gamma_json_date_ms(v.get("startTime")))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("startTime"))))
            .or_else(|| gamma_json_date_ms(v.get("startDate")))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("startDate"))));
        market_event_start_ms.insert(cid.clone(), start_ms);
        let end_ms = gamma_json_date_ms(v.get("endDate"))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("endDate"))));
        market_event_end_ms.insert(cid, end_ms);
    }

    if clob_token_ids.is_empty() {
        anyhow::bail!("ни одного clobTokenId в ответе Gamma для slug={slug:?}");
    }

    Ok(CurrencyEventSlugData {
        currency_up_down_by_asset_id,
        market_event_start_ms,
        market_event_end_ms,
        gamma_question,
    })
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

fn parse_outcomes_from_gamma_market(v: &Value) -> anyhow::Result<Vec<String>> {
    match v.get("outcomes") {
        Some(Value::String(encoded)) => Ok(serde_json::from_str(encoded)?),
        Some(Value::Array(items)) => Ok(items
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect()),
        _ => Ok(Vec::new()),
    }
}

/// RFC3339 из Gamma JSON → Unix ms UTC.
fn gamma_json_date_ms(v: Option<&Value>) -> Option<i64> {
    let s = v?.as_str()?;
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

fn parse_clob_token_ids_from_gamma_market(v: &Value) -> anyhow::Result<Vec<String>> {
    match v.get("clobTokenIds") {
        Some(Value::String(encoded)) => Ok(serde_json::from_str(encoded)?),
        Some(Value::Array(items)) => Ok(items
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect()),
        _ => Ok(Vec::new()),
    }
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
pub async fn detect_country_and_ip() -> Option<CountryAndIp> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;
    let resp = client
        .get("https://polymarket.com/api/geoblock")
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

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()?;

        let slug = "btc-updown-5m-1778267400";
        let price = fetch_price_to_beat_from_vatic_api(&http, slug, "btc").await?;

        let price_2dp = (price * 100.0).round() / 100.0;
        anyhow::ensure!(
            (price_2dp - 80061.62).abs() < 1e-9,
            "ожидался priceToBeat $80,061.62 для slug={slug}, получено {price} (rounded 2dp = {price_2dp})"
        );
        Ok(())
    }
}
