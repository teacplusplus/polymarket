pub mod poly_trace;
pub mod util;
pub mod api_data_manager;
pub mod xframe;
pub mod project_manager;
pub mod market_snapshot;
pub mod ws_parser;
pub mod ws;

use anyhow::{anyhow, Context, Result};
use project_manager::ProjectManager;

use crate::api_data_manager::ApiDataJob;

/// Сколько последовательных бакетов (на каждую группировку [project_manager::FRAME_BUILD_INTERVAL_SECS]) загрузить назад.
fn historical_bucket_count_from_env() -> usize {
    std::env::var("POLYMARKET_HISTORICAL_BUCKETS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|&parsed| parsed > 0)
        .unwrap_or(24)
}

/// Slug события из URL вида `https://polymarket.com/event/<slug>` (например `btc-updown-5m-1775171700`).
const DEFAULT_EVENT_SLUG: &str = "btc-updown-5m-1775171700";

/// Получить `clobTokenIds` из публичного Gamma API по slug (строка JSON внутри поля `clobTokenIds`).
async fn fetch_clob_token_ids_for_event_slug(
    http: &reqwest::Client,
    slug: &str,
) -> Result<Vec<String>> {
    let url = format!("https://gamma-api.polymarket.com/events?slug={slug}");
    crate::poly_trace::api_line(
        "Gamma public API (HTTP)",
        &format!("GET {url} (разрешение slug → clobTokenIds)"),
    );
    let response = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = response.status();
    let text = response.text().await?;
    if !status.is_success() {
        anyhow::bail!("Gamma API HTTP {status}: {text}");
    }
    let events: serde_json::Value =
        serde_json::from_str(&text).context("Gamma API: не JSON")?;
    let event = events
        .as_array()
        .and_then(|array| array.first())
        .context("Gamma API: пустой массив events")?;
    let markets = event
        .get("markets")
        .and_then(|markets| markets.as_array())
        .context("Gamma API: нет markets")?;
    let mut out = Vec::new();
    for market in markets {
        let Some(raw) = market.get("clobTokenIds").and_then(|value| value.as_str()) else {
            continue;
        };
        let tokens: Vec<String> =
            serde_json::from_str(raw).with_context(|| format!("parse clobTokenIds: {raw}"))?;
        out.extend(tokens);
    }
    if out.is_empty() {
        anyhow::bail!("ни одного clobTokenId в ответе Gamma для slug={slug:?}");
    }
    Ok(out)
}

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("rustls: install ring CryptoProvider (needed for WebSocket TLS)");
    run_debug_event_pipeline().await
}

/// Режим отладки: slug события → `clobTokenIds`, WebSocket, `FetchAssetFull` + `BuildHistoricalXFrames`, логи `POLY_TRACE`.
async fn run_debug_event_pipeline() -> Result<()> {
    let slug = std::env::var("POLYMARKET_EVENT_SLUG").unwrap_or_else(|_| DEFAULT_EVENT_SLUG.to_string());
    eprintln!(
        "=== poly debug: event slug={slug} (override: POLYMARKET_EVENT_SLUG) | отключить лог: POLY_TRACE=0 или POLYMARKET_TRACE=0 ==="
    );

    let http = reqwest::Client::builder()
        .use_rustls_tls()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let asset_ids = fetch_clob_token_ids_for_event_slug(&http, &slug).await?;
    eprintln!("=== clobTokenIds ({}) = {:?}", asset_ids.len(), asset_ids);

    let project_manager = ProjectManager::new();
    ws::new(project_manager.clone(), asset_ids.clone())?;
    let past_bucket_count = historical_bucket_count_from_env();
    for asset_id in &asset_ids {
        project_manager
            .api
            .api_job_sender
            .send(ApiDataJob::FetchAssetFull {
                asset_id: asset_id.clone(),
            })
            .await
            .map_err(|_| anyhow!("api data worker dropped before startup"))?;
        project_manager
            .api
            .api_job_sender
            .send(ApiDataJob::BuildHistoricalXFrames {
                asset_id: asset_id.clone(),
                past_bucket_count,
            })
            .await
            .map_err(|_| anyhow!("api data worker dropped before startup"))?;
    }
    eprintln!("=== WS + frame builders запущены; история загружается асинхронно. Остановка: Ctrl+C ===");
    std::future::pending::<()>().await;
    Ok(())
}

/*
 * Прежний вход: только `POLYMARKET_ASSET_IDS` (без разрешения slug).
 *
 * #[tokio::main]
 * async fn main() -> Result<()> {
 *     let asset_ids = read_asset_ids_from_env()?;
 *     let project_manager = ProjectManager::new();
 *     ws::new(project_manager.clone(), asset_ids.clone())?;
 *     let past_bucket_count = historical_bucket_count_from_env();
 *     for asset_id in &asset_ids {
 *         project_manager.api.api_job_sender.send(ApiDataJob::FetchAssetFull { asset_id: asset_id.clone() }).await?;
 *         project_manager.api.api_job_sender.send(ApiDataJob::BuildHistoricalXFrames { asset_id: asset_id.clone(), past_bucket_count }).await?;
 *     }
 *     std::future::pending::<()>().await;
 *     Ok(())
 * }
 *
 * fn read_asset_ids_from_env() -> Result<Vec<String>> { ... }
 */

#[allow(dead_code)]
fn read_asset_ids_from_env() -> Result<Vec<String>> {
    let raw = std::env::var("POLYMARKET_ASSET_IDS")
        .map_err(|_| anyhow!("set POLYMARKET_ASSET_IDS=asset1,asset2,..."))?;
    let ids: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if ids.is_empty() {
        return Err(anyhow!("POLYMARKET_ASSET_IDS has no valid asset ids"));
    }
    Ok(ids)
}
