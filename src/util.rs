use anyhow::Context;
use chrono::DateTime;
use serde_json::Value;
use std::collections::HashMap;

use crate::constants::CurrencyUpDownOutcome;

pub fn current_timestamp_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis() as i64
}

/// Имя файла из текста Gamma `question`: безопасные символы и ограничение длины.
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
    const MAX: usize = 180;
    if s.len() > MAX {
        format!("{}...", &s[..MAX])
    } else {
        s
    }
}

/// Локальный абсолютный путь в `file://` URI: оставляем `A-Za-z0-9/._-()`, остальное — `%XX`.
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
    pub currency_up_down_by_asset_id: HashMap<String, CurrencyUpDownOutcome>,
    pub market_event_start_ms: HashMap<String, Option<i64>>,
    pub market_event_end_ms: HashMap<String, Option<i64>>,
    pub gamma_question: Option<String>,
}

/// `priceToBeat` (target/opening price) окна Polymarket из публичного Vatic API.
///
/// Заменяет старую логику чтения `__NEXT_DATA__` со страницы
/// `polymarket.com/event/{slug}` на единый GET к
/// `https://api.vatic.trading/api/v1/targets/timestamp?asset={currency}&type={5min|15min}&timestamp={window_start_sec}`
/// (см. <https://docs.vatic.trading/api-reference/targets/timestamp.md>).
///
/// `currency` — тикер как в [`crate::project_manager::ProjectManager::currency`]
/// (`btc`/`eth`/...), в URL уходит **в нижнем** регистре.
///
/// `slug` ожидается формата `{currency}-updown-{5m|15m}-{window_start_sec}`
/// (тот же, что лежит в `polymarket.com/event/{slug}`); из него извлекается
/// тип интервала и Unix-секунды окна.
///
/// Vatic возвращает точную опеновую цену окна (Chainlink Data Streams для
/// 5min/15min с retention ~14 дней), внутри уже делает 4 повтора с задержкой
/// 1с против publish-лага на границе окна — fallback-логика с предыдущим
/// `closePrice` больше не нужна.
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

/// Парсит `{currency}-updown-{5m|15m}-{ts}` → `(window_sec, market_type)`,
/// где `market_type` — значение параметра `type` в Vatic API
/// (`5min`/`15min`, см. <https://docs.vatic.trading/concepts/market-types.md>).
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

/// Результат [`detect_country_and_ip`] — страна и внешний IP исходящего
/// соединения, оба поля независимо опциональны (если в ответе отсутствует).
pub struct CountryAndIp {
    pub country: Option<String>,
    pub ip:      Option<String>,
}

/// Узнаёт страну и внешний IP через `https://ifconfig.co/json`
/// (тот же сервис, что `curl -s https://ifconfig.co/json`). Используется
/// только для печати в самом начале запуска — чтобы по логу было сразу
/// видно, через какую гео-точку (VPN/прокси) сейчас стучимся к биржам.
/// Любая ошибка (нет сети, таймаут, не-200, кривой JSON) не должна валить
/// запуск, поэтому возвращаем [`Option<CountryAndIp>`] и обрабатываем `None`
/// как «определить не удалось».
pub async fn detect_country_and_ip() -> Option<CountryAndIp> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;
    let resp = client
        .get("https://ifconfig.co/json")
        // Без `User-Agent: curl/...` сервис отдаёт HTML по этому URL,
        // а не JSON; ставим явный «curl-подобный» UA, чтобы получить JSON.
        .header(reqwest::header::USER_AGENT, "curl/8.9.1")
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: Value = resp.json().await.ok()?;
    let pick = |k: &str| -> Option<String> {
        body.get(k)
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    Some(CountryAndIp {
        country: pick("country"),
        ip:      pick("ip"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Лайв-тест [`fetch_price_to_beat_from_vatic_api`]: реальный GET к
    /// `api.vatic.trading` за target/opening price конкретного 5-минутного
    /// окна Polymarket'а
    /// [`btc-updown-5m-1778267400`](https://polymarket.com/event/btc-updown-5m-1778267400)
    /// (May 8 2026, 3:10–3:15PM ET, 19:10:00 UTC) и сверка с `$80,061.62` —
    /// точное значение Chainlink BTC/USD на открытии этого окна.
    ///
    /// Запуск:
    ///
    /// ```bash
    /// cargo test --bin poly util::tests::live_fetch_price_to_beat_from_vatic_btc_updown_5m_1778267400 -- --ignored --nocapture
    /// ```
    ///
    /// `#[ignore]` — не хотим бить по живому API в обычном `cargo test`.
    /// Помимо самого окна тест зависит от Chainlink retention (~14 дней
    /// для 5min), поэтому при запуске позже середины мая 2026 Vatic может
    /// вернуть 410 — это уже не баг функции.
    #[tokio::test]
    #[ignore = "live network: GET https://api.vatic.trading/api/v1/targets/timestamp"]
    async fn live_fetch_price_to_beat_from_vatic_btc_updown_5m_1778267400() -> anyhow::Result<()> {
        // rustls 0.23 требует CryptoProvider до первого TLS-запроса; в
        // обычном бинарнике это делает `main`, а в `tokio::test` — мы
        // сами. `install_default()` идемпотентен — повтор молча даст Err.
        let _ = rustls::crypto::ring::default_provider().install_default();

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()?;

        let slug = "btc-updown-5m-1778267400";
        let price = fetch_price_to_beat_from_vatic_api(&http, slug, "btc").await?;

        // Округлённое до 2 знаков должно дать ровно 80061.62 (на странице
        // Polymarket "Price to Beat" показан как `$80,061.62`). Реальное
        // значение Chainlink: ~80061.61963627425.
        let price_2dp = (price * 100.0).round() / 100.0;
        anyhow::ensure!(
            (price_2dp - 80061.62).abs() < 1e-9,
            "ожидался priceToBeat $80,061.62 для slug={slug}, получено {price} (rounded 2dp = {price_2dp})"
        );
        Ok(())
    }
}
