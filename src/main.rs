pub mod util;
pub mod api_data_manager;
pub mod xframe;
pub mod project_manager;
pub mod market_snapshot;
pub mod ws_parser;
pub mod ws;

use anyhow::{anyhow, Result};
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

#[tokio::main]
async fn main() -> Result<()> {
    let asset_ids = read_asset_ids_from_env()?;
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
    std::future::pending::<()>().await;
    Ok(())
}

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
