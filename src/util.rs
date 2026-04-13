use anyhow::Context;
use chrono::{DateTime, TimeZone, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

use crate::constants::CurrencyUpDownOutcome;

pub fn current_timestamp_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis() as i64
}

pub struct CurrencyEventSlugData {
    pub currency_up_down_by_asset_id: HashMap<String, CurrencyUpDownOutcome>,
    pub market_event_start_ms: HashMap<String, Option<i64>>,
    pub market_event_end_ms: HashMap<String, Option<i64>>,
    pub gamma_question: Option<String>,
}

/// `priceToBeat` из JSON скрипта `#__NEXT_DATA__` на [`https://polymarket.com/event/{slug}`](https://polymarket.com/event/).
///
/// `currency` — тикер как в [`crate::project_manager::ProjectManager::currency`] (например `eth`); в кэшах React Query сравнивается в **верхнем** регистре.
///
/// В `props.pageProps.dehydratedState.queries` (только данные **этого** окна из `slug`, иначе [`None`]):
/// 1. `["crypto-prices", …, <ISO начала>, <variant>, <ISO конца свечи>]` — `openPrice`;
///    `ISO конца` = начало окна + 300 с (5m) или + 900 с (15m); при дублях — запись с max `dataUpdatedAt`.
/// 2. `["past-results", …, <ISO начала окна>]` — в `results` строка с `endTime` = начало окна (RFC3339),
///    берём `closePrice` (не «последнюю» строку массива — там другие свечи).
pub async fn fetch_price_to_beat_from_polymarket_event_page(
    http: &reqwest::Client,
    slug: &str,
    currency: &str,
) -> anyhow::Result<f64> {
    let url = format!("https://polymarket.com/event/{slug}");
    let response = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("polymarket GET event page {url}"))?;
    if response.status() != reqwest::StatusCode::OK {
        anyhow::bail!(
            "нет priceToBeat: polymarket.com/event вернул HTTP {} для slug={slug:?}",
            response.status()
        );
    }
    let html = response
        .text()
        .await
        .with_context(|| format!("polymarket event page body {url}"))?;

    let price_result: anyhow::Result<f64> = (|| -> anyhow::Result<f64> {
        let json_str = extract_next_data_json(&html).ok_or_else(|| {
            anyhow::anyhow!(
                "нет priceToBeat: не найден скрипт __NEXT_DATA__ на polymarket.com/event для slug={slug:?}"
            )
        })?;
        let v: Value = serde_json::from_str(json_str).with_context(|| {
            format!("нет priceToBeat: JSON __NEXT_DATA__ не разобрать для slug={slug:?}")
        })?;
        let price = price_to_beat_from_next_data(&v, slug, currency).ok_or_else(|| {
            anyhow::anyhow!(
                "нет priceToBeat: в __NEXT_DATA__ для slug={slug:?} нет подходящего crypto-prices (openPrice + конец свечи) ни строки past-results с endTime=start окна"
            )
        })?;
        if !price.is_finite() || price <= 0.0 {
            anyhow::bail!(
                "нет priceToBeat: некорректное значение {price} в __NEXT_DATA__ для slug={slug:?}"
            );
        }
        Ok(price)
    })();

    if price_result.is_err() {
        dump_event_page_html_missing_price_to_beat(slug, &html);
    }

    price_result
}

fn dump_event_page_html_missing_price_to_beat(slug: &str, html: &str) {
    let path = Path::new("errors/html").join(format!("{slug}.html"));
    if let Err(e) = std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))
        .and_then(|_| std::fs::write(&path, html))
    {
        eprintln!("errors/html dump {path:?}: {e}");
    }
}

fn extract_next_data_json(html: &str) -> Option<&str> {
    const ID_MARK: &str = r#"id="__NEXT_DATA__""#;
    let i = html.find(ID_MARK)?;
    let tail = html.get(i..)?;
    let after_open = tail.find('>').map(|j| i + j + 1)?;
    let rest = html.get(after_open..)?;
    let end_rel = rest.find("</script>")?;
    Some(rest.get(..end_rel)?.trim())
}

fn json_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn price_to_beat_from_next_data(next_data: &Value, slug: &str, currency: &str) -> Option<f64> {
    price_to_beat_from_currency_open_price_crypto_prices(next_data, slug, currency)
        .or_else(|| price_to_beat_from_currency_past_results_close(next_data, slug, currency))
}

fn price_to_beat_from_currency_past_results_close(
    next_data: &Value,
    slug: &str,
    currency: &str,
) -> Option<f64> {
    let (window_sec, variant) = currency_updown_slug_window_sec_and_variant(currency, slug)?;
    let iso = window_start_iso_utc_z(window_sec)?;
    let queries = next_data
        .get("props")?
        .get("pageProps")?
        .get("dehydratedState")?
        .get("queries")?
        .as_array()?;
    let currency_upper = currency.to_uppercase();
    for q in queries {
        let key = match q.get("queryKey").and_then(|k| k.as_array()) {
            Some(k) if k.len() == 4 => k,
            _ => continue,
        };
        if key[0].as_str() != Some("past-results") {
            continue;
        }
        if key[1].as_str() != Some(currency_upper.as_str()) {
            continue;
        }
        if key[2].as_str() != Some(variant) {
            continue;
        }
        if key[3].as_str() != Some(iso.as_str()) {
            continue;
        }
        let Some(state_data) = q.get("state").and_then(|s| s.get("data")) else {
            continue;
        };
        let Some(results) = state_data
            .get("data")
            .and_then(|d| d.get("results"))
            .and_then(|r| r.as_array())
            .or_else(|| state_data.get("results").and_then(|r| r.as_array()))
        else {
            continue;
        };
        if let Some(p) = price_from_past_results_for_window_start(results, window_sec) {
            return Some(p);
        }
    }
    None
}

/// Свеча с `endTime` = началу окна: `closePrice` = цена на открытии окна (price to beat). Иначе строка с `startTime` = началу → `openPrice`.
fn price_from_past_results_for_window_start(results: &[Value], window_sec: i64) -> Option<f64> {
    for row in results {
        let Some(end) = row.get("endTime").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(end_sec) = parse_iso_time_to_unix_sec(end) else {
            continue;
        };
        if end_sec == window_sec {
            if let Some(p) = row.get("closePrice").and_then(json_f64) {
                return Some(p);
            }
        }
    }
    for row in results {
        let Some(start) = row.get("startTime").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(start_sec) = parse_iso_time_to_unix_sec(start) else {
            continue;
        };
        if start_sec == window_sec {
            if let Some(p) = row.get("openPrice").and_then(json_f64) {
                return Some(p);
            }
        }
    }
    None
}

fn parse_iso_time_to_unix_sec(s: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp())
}

fn currency_updown_slug_window_sec_and_variant(
    currency: &str,
    slug: &str,
) -> Option<(i64, &'static str)> {
    let prefix = format!("{}-updown-", currency.to_lowercase());
    let rest = slug.strip_prefix(prefix.as_str())?;
    let (mid, sec_str) = rest.rsplit_once('-')?;
    let window_sec: i64 = sec_str.parse().ok()?;
    let variant = match mid {
        "5m" => "fiveminute",
        "15m" => "fifteen",
        _ => return None,
    };
    Some((window_sec, variant))
}

fn window_start_iso_utc_z(window_sec: i64) -> Option<String> {
    let dt = Utc.timestamp_opt(window_sec, 0).single()?;
    Some(dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
}

fn updown_interval_sec_for_variant(variant: &str) -> Option<i64> {
    match variant {
        "fiveminute" => Some(300),
        "fifteen" => Some(900),
        _ => None,
    }
}

/// Кэш React Query `crypto-prices` — `openPrice` только для свечи `[window, window+interval)`; `queryKey[5]` = конец интервала.
fn price_to_beat_from_currency_open_price_crypto_prices(
    next_data: &Value,
    slug: &str,
    currency: &str,
) -> Option<f64> {
    let (window_sec, variant_expected) = currency_updown_slug_window_sec_and_variant(currency, slug)?;
    let interval = updown_interval_sec_for_variant(variant_expected)?;
    let candle_end_sec = window_sec.saturating_add(interval);
    let iso_start = window_start_iso_utc_z(window_sec)?;
    let queries = next_data
        .get("props")?
        .get("pageProps")?
        .get("dehydratedState")?
        .get("queries")?
        .as_array()?;
    let currency_upper = currency.to_uppercase();
    let mut best: Option<(i64, f64)> = None;
    for q in queries {
        let key = match q.get("queryKey").and_then(|k| k.as_array()) {
            Some(k) if k.len() >= 5 => k,
            _ => continue,
        };
        if key[0].as_str() != Some("crypto-prices") {
            continue;
        }
        if key[1].as_str() != Some("price") {
            continue;
        }
        if key[2].as_str() != Some(currency_upper.as_str()) {
            continue;
        }
        if key[3].as_str() != Some(iso_start.as_str()) {
            continue;
        }
        if key[4].as_str() != Some(variant_expected) {
            continue;
        }
        if let Some(key_end_sec) = key
            .get(5)
            .and_then(|v| v.as_str())
            .and_then(parse_iso_time_to_unix_sec)
        {
            if key_end_sec != candle_end_sec {
                continue;
            }
        }
        let Some(state) = q.get("state") else {
            continue;
        };
        let Some(data) = state.get("data") else {
            continue;
        };
        let Some(open) = data.get("openPrice") else {
            continue;
        };
        let Some(price) = json_f64(open) else {
            continue;
        };
        let updated = state
            .get("dataUpdatedAt")
            .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))
            .unwrap_or(0);
        best = match best {
            None => Some((updated, price)),
            Some((u, _)) if updated > u => Some((updated, price)),
            Some(bp) => Some(bp),
        };
    }
    best.map(|(_, p)| p)
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
    let outcomes = parse_outcomes_from_gamma_market(&v)
        .with_context(|| format!("outcomes slug={slug}"))?;
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

    if let Some(cid) = v.get("conditionId").and_then(|x| x.as_str()).map(str::to_string) {
        let event0 = v
            .get("events")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        // Старт окна: в первую очередь `eventStartTime` (маркет, затем `events[0]`), далее `startTime` / `startDate`.
        let start_ms = gamma_json_date_ms(v.get("eventStartTime"))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("eventStartTime"))))
            .or_else(|| gamma_json_date_ms(v.get("startTime")))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("startTime"))))
            .or_else(|| gamma_json_date_ms(v.get("startDate")))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("startDate"))));
        market_event_start_ms.insert(cid.clone(), start_ms);
        // Конец окна: в Gamma нет `eventEndTime`; `endDate` (UTC RFC3339) — граница окна, не путать с `umaEndDate` (UMA).
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

/// RFC3339 с `Z` или оффсетом — в миллисекунды UTC ([`DateTime::timestamp_millis`]).
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
