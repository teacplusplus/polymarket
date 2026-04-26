//! Сохранение накопленных [`crate::xframe::XFrame`] в бинарный файл при пересоздании WS.

use crate::constants::XFrameIntervalKind;
use crate::project_manager::{ProjectManager, FRAME_BUILD_INTERVALS_SEC};
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
    /// Цена `price_to_beat` (открытие окна).
    #[serde(default)]
    pub price_to_beat: f64,
    /// Финальная цена (закрытие окна / открытие следующего).
    #[serde(default)]
    pub final_price: f64,
}

impl MarketXFramesDump {
    pub fn up_won(&self) -> bool {
        self.final_price >= self.price_to_beat
    }
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
    period_sec: i64,
    price_to_beat: f64,
    final_price: f64,
) {
    tokio::spawn(async move {
        let interval_kind = XFrameIntervalKind::from_period_sec(period_sec);
        let max_step = *FRAME_BUILD_INTERVALS_SEC.iter().max().unwrap_or(&1);
        tokio::time::sleep(std::time::Duration::from_secs(max_step)).await;

        let up_won = final_price >= price_to_beat;
        eprintln!("xframe_dump: market_id={market_id} price_to_beat={price_to_beat} final_price={final_price} up_won={up_won}");

        // Резолюционный колбек по `final_price`: закрывает все
        // pending-позиции этого маркета (Up/Down обоих лейнов, что
        // соответствует `interval_kind`) по бинарной выплате CTF.
        // Должен дёргаться **до** дампа, чтобы `bankroll`/`SideStats`
        // обновились до того, как `cleanup_stale_market_data`
        // снесёт буфер кадров. Sleep выше (`max_step` сек) — это
        // запас на доезд последних кадров до буфера; за это время
        // `tick_once` успевает довести stale-позиции в
        // `pending_resolution` через `manage_positions`, плюс
        // сама резолюция-обёртка дополнительно вытащит из
        // active book позиции с `pos.market_id == market_id`
        // (см. `Account::resolve_pending_market`).
        crate::account::Account::resolve_pending_market(
            &project_manager.account,
            &project_manager.real_sim_state,
            project_manager.currency.as_str(),
            interval_kind,
            &market_id,
            up_won,
        )
        .await;

        for lane in 0..FRAME_BUILD_INTERVALS_SEC.len() {
            if let Err(err) =
                dump_market_xframes_binary_lane(
                    project_manager.clone(),
                    market_id.clone(),
                    gamma_question.clone(),
                    interval_kind,
                    lane,
                    price_to_beat,
                    final_price,
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
    price_to_beat: f64,
    final_price: f64,
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
    let dump = MarketXFramesDump { frames_up, frames_down, price_to_beat, final_price };

    // Версия схемы дампа — размер сериализованного `XFrame<SIZE>` по умолчанию (bincode).
    // Меняется при любом изменении раскладки полей/констант структуры, поэтому
    // подходит в качестве стабильного «fingerprint» для разбиения по версиям.
    let schema_size = bincode::serialized_size(&XFrame::<SIZE>::default())
        .expect("XFrame::<SIZE>::default() must be bincode-serializable") as usize;

    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let base: PathBuf = Path::new("xframes")
        .join(project_manager.currency.as_str())
        .join(format!("{schema_size}"))
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
