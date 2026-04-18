//! Сохранение накопленных [`crate::xframe::XFrame`] в бинарный файл при пересоздании WS.

use crate::constants::XFrameIntervalKind;
use crate::project_manager::{ProjectManager, FRAME_BUILD_INTERVALS_SEC};
use crate::run_log;
use crate::util::{current_timestamp_ms, fetch_market_resolution_up_won};
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
    /// Фактический исход рынка из Gamma API после резолюции: `true` — победил Up, `false` — Down.
    /// Старые дампы без этого поля десериализуются как `false`.
    #[serde(default)]
    pub up_won: bool,
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

/// Асинхронно пишет дамп **каждого лейна** в `xframes/{currency}/{count_features}/{interval}/{step}s/{YYYY-MM-DD}/{name}.bin`,
/// а после завершения (успех или ошибка) вызывает `cleanup_stale_market_data`
/// чтобы освободить память, занятую данными завершённого маркета.
pub fn spawn_dump_market_xframes_binary(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
    interval_kind: XFrameIntervalKind,
) {
    tokio::spawn(async move {
        let max_step = *FRAME_BUILD_INTERVALS_SEC.iter().max().unwrap_or(&1);
        tokio::time::sleep(std::time::Duration::from_secs(max_step)).await;

        let up_won = match fetch_market_resolution_up_won(project_manager.gamma.as_ref(), &market_id).await {
            Ok(v) => v,
            Err(e) => {
                eprintln!("xframe_dump: market_id={market_id} resolution: {e:#}, дамп пропущен");
                project_manager.cleanup_stale_market_data(&market_id).await;
                return;
            }
        };
        eprintln!("xframe_dump: market_id={market_id} resolution: up_won={up_won}");

        for lane in 0..FRAME_BUILD_INTERVALS_SEC.len() {
            if let Err(err) =
                dump_market_xframes_binary_lane(
                    project_manager.clone(),
                    market_id.clone(),
                    gamma_question.clone(),
                    interval_kind,
                    lane,
                    up_won,
                ).await
            {
                eprintln!("xframe_dump lane={lane}: {err:#}");
            }
        }
        project_manager.cleanup_stale_market_data(&market_id).await;
    });
}

pub async fn dump_market_xframes_binary_lane(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
    interval_kind: XFrameIntervalKind,
    lane: usize,
    up_won: bool,
) -> anyhow::Result<()> {
    let by_asset = {
        let xframes_by_market_lock = project_manager.xframes_by_market[lane].read().await;
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

    let step_secs = FRAME_BUILD_INTERVALS_SEC[lane];

    let frame_count = frames_up.len() + frames_down.len();
    let dump = MarketXFramesDump { frames_up, frames_down, up_won };

    let feature_count = XFrame::<SIZE>::count_features();

    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let base: PathBuf = Path::new("xframes")
        .join(project_manager.currency.as_str())
        .join(format!("{feature_count}"))
        .join(interval_label)
        .join(format!("{step_secs}s"))
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
