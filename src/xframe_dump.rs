//! Сохранение накопленных [`crate::xframe::XFrame`] в бинарный файл при пересоздании WS.

use crate::constants::XFrameIntervalKind;
use crate::project_manager::ProjectManager;
use crate::run_log;
use crate::util::current_timestamp_ms;
use crate::xframe::{CurrencyUpDownOutcome, XFrame, SIZE};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketXFramesDump {
    /// Кадры токена с исходом Up, упорядоченные по `aligned_ts`.
    pub frames_up: Vec<XFrame<SIZE>>,
    /// Кадры токена с исходом Down, упорядоченные по `aligned_ts`.
    pub frames_down: Vec<XFrame<SIZE>>,
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

/// Асинхронно пишет дамп в `xframes/{currency}/{count_features}/{interval}/{YYYY-MM-DD}/{name}.bin`,
/// а после завершения (успех или ошибка) вызывает `cleanup_stale_market_data`
/// чтобы освободить память, занятую данными завершённого маркета.
pub fn spawn_dump_market_xframes_binary(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
    interval_kind: XFrameIntervalKind,
) {
    tokio::spawn(async move {
        if let Err(err) =
            dump_market_xframes_binary_inner(project_manager.clone(), market_id.clone(), gamma_question, interval_kind).await {
            eprintln!("xframe_dump: {err:#}");
        }
        project_manager.cleanup_stale_market_data(&market_id).await;
    });
}

async fn dump_market_xframes_binary_inner(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
    interval_kind: XFrameIntervalKind,
) -> anyhow::Result<()> {
    let by_asset = {
        let xframes_by_market_lock = project_manager.xframes_by_market.read().await;
        xframes_by_market_lock.get(&market_id).cloned()
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
    flat.sort_by_key(|(_, aligned_ts, _)| *aligned_ts);

    let mut frames_up: Vec<XFrame<SIZE>> = Vec::new();
    let mut frames_down: Vec<XFrame<SIZE>> = Vec::new();
    for (_, _, frame) in flat {
        if !frame.stable {
            continue;
        }
        match CurrencyUpDownOutcome::from_i32(frame.currency_up_down_outcome) {
            Some(CurrencyUpDownOutcome::Up) => frames_up.push(frame),
            Some(CurrencyUpDownOutcome::Down) => frames_down.push(frame),
            None => {}
        }
    }

    if frames_up.is_empty() && frames_down.is_empty() {
        return Ok(());
    }

    let interval_label = match interval_kind {
        XFrameIntervalKind::FiveMin    => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    };

    let frame_count = frames_up.len() + frames_down.len();
    let dump = MarketXFramesDump { frames_up, frames_down };

    let feature_count = XFrame::<SIZE>::count_features();

    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let base: PathBuf = Path::new("xframes")
        .join(project_manager.currency.as_str())
        .join(format!("{feature_count}"))
        .join(interval_label)
        .join(&date);
    tokio::fs::create_dir_all(&base).await?;

    let stem = sanitized_filename_from_gamma_question(gamma_question.as_deref());
    let fname = format!("{stem}__{}.bin", current_timestamp_ms());
    let path = base.join(&fname);
    let bytes = bincode::serialize(&dump)?;
    let byte_len = bytes.len();
    tokio::fs::write(&path, bytes).await?;
    run_log::xframe_dump_written(&path, &market_id, frame_count, byte_len);
    Ok(())
}
