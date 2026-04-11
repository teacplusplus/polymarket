//! Сохранение накопленных [`crate::xframe::XFrame`] в бинарный файл при пересоздании WS.

use crate::project_manager::ProjectManager;
use crate::run_log;
use crate::util::current_timestamp_ms;
use crate::xframe::{XFrame, SIZE};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketXFramesDump {
    pub frames: Vec<XFrame<SIZE>>,
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

/// Асинхронно пишет дамп в `xframes/{count_features}/{YYYY-MM-DD}/{name}.bin` (не блокирует остановку WS).
pub fn spawn_dump_market_xframes_binary(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
) {
    tokio::spawn(async move {
        if let Err(err) =
            dump_market_xframes_binary_inner(project_manager, market_id, gamma_question).await {
            eprintln!("xframe_dump: {err:#}");
        }
    });
}

async fn dump_market_xframes_binary_inner(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
) -> anyhow::Result<()> {
    let by_asset = {
        let guard = project_manager.xframes_by_market.read().await;
        guard.get(&market_id).cloned()
    };
    let Some(by_asset) = by_asset else {
        return Ok(());
    };

    let mut flat: Vec<(String, i64, XFrame<SIZE>)> = Vec::new();
    for (asset_id, by_ts) in by_asset.iter() {
        for (aligned_ts, frame) in by_ts.iter() {
            flat.push((asset_id.clone(), *aligned_ts, frame.clone()));
        }
    }
    flat.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let frames: Vec<XFrame<SIZE>> = flat
        .into_iter()
        .map(|(_, _, f)| f)
        .filter(|f| f.stable)
        .collect();

    if frames.is_empty() {
        return Ok(());
    }

    let dump = MarketXFramesDump { frames };

    let feature_count = XFrame::<SIZE>::count_features();
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let base: PathBuf = Path::new("xframes")
        .join(format!("{feature_count}"))
        .join(&date);
    tokio::fs::create_dir_all(&base).await?;

    let stem = sanitized_filename_from_gamma_question(gamma_question.as_deref());
    let fname = format!("{stem}__{}.bin", current_timestamp_ms());
    let path = base.join(&fname);
    let frame_count = dump.frames.len();
    let bytes = bincode::serialize(&dump)?;
    let byte_len = bytes.len();
    tokio::fs::write(&path, bytes).await?;
    run_log::xframe_dump_written(&path, &market_id, frame_count, byte_len);
    Ok(())
}
