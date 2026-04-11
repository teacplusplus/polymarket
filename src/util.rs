use anyhow::Context;
use chrono::DateTime;
use serde_json::Value;
use std::collections::HashMap;

use crate::constants::BtcUpDownOutcome;

pub fn current_timestamp_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis() as i64
}

pub struct GammaEventSlugData {
    pub clob_token_ids: Vec<String>,
    pub btc_up_down_by_asset_id: HashMap<String, BtcUpDownOutcome>,
    pub market_event_start_ms: HashMap<String, Option<i64>>,
    pub market_event_end_ms: HashMap<String, Option<i64>>,
    pub price_to_beat: Option<f64>,
    pub gamma_question: Option<String>,
}

pub async fn fetch_gamma_event_data_for_slug(
    http: &reqwest::Client,
    slug: &str,
) -> anyhow::Result<GammaEventSlugData> {
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
    let btc_up_down_by_asset_id = if outcomes.is_empty() {
        HashMap::new()
    } else {
        zip_outcomes_clob_to_up_code(&outcomes, &clob_token_ids)
            .with_context(|| format!("outcomes vs clobTokenIds slug={slug}"))?
    };

    let price_to_beat = v
        .get("events")
        .and_then(Value::as_array)
        .and_then(|events| events.first())
        .and_then(|ev| ev.get("eventMetadata"))
        .and_then(|meta| meta.get("priceToBeat"))
        .and_then(|n| n.as_f64().or_else(|| n.as_str()?.parse().ok()));

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
        let start_ms = gamma_json_date_ms(v.get("startDate"))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("startDate"))));
        market_event_start_ms.insert(cid.clone(), start_ms);
        let end_ms = gamma_json_date_ms(v.get("endDate"))
            .or_else(|| event0.and_then(|e| gamma_json_date_ms(e.get("endDate"))));
        market_event_end_ms.insert(cid, end_ms);
    }

    if clob_token_ids.is_empty() {
        anyhow::bail!("ни одного clobTokenId в ответе Gamma для slug={slug:?}");
    }
    if price_to_beat.is_none() {
        anyhow::bail!("нет priceToBeat в events[0].eventMetadata в ответе Gamma для slug={slug:?}");
    }

    Ok(GammaEventSlugData {
        clob_token_ids,
        btc_up_down_by_asset_id,
        market_event_start_ms,
        market_event_end_ms,
        price_to_beat,
        gamma_question,
    })
}

fn gamma_outcome_label_to_btc_kind(label: &str) -> Option<BtcUpDownOutcome> {
    match label.trim().to_ascii_lowercase().as_str() {
        "up" => Some(BtcUpDownOutcome::Up),
        "down" => Some(BtcUpDownOutcome::Down),
        _ => None,
    }
}

fn zip_outcomes_clob_to_up_code(
    outcomes: &[String],
    clob_ids: &[String],
) -> anyhow::Result<HashMap<String, BtcUpDownOutcome>> {
    if outcomes.len() != clob_ids.len() {
        anyhow::bail!(
            "Gamma: len(outcomes)={} != len(clobTokenIds)={}",
            outcomes.len(),
            clob_ids.len()
        );
    }
    let mut map = HashMap::new();
    for (label, token_id) in outcomes.iter().zip(clob_ids.iter()) {
        if let Some(code) = gamma_outcome_label_to_btc_kind(label) {
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
