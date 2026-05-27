//! Сохранение накопленных [`crate::xframe::XFrame`] в бинарный файл при пересоздании WS.

use crate::constants::XFrameIntervalKind;
use crate::project_manager::{FRAME_BUILD_INTERVALS_SEC, ProjectManager};
use crate::run_log;
use crate::util::{current_timestamp_ms, sanitized_filename_from_gamma_question};
use crate::xframe::{CurrencyUpDownOutcome, SIZE, XFrame};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Писать WS-стрим в `streams/...` после дампа xframes ([`dump_market_ws_stream_txt`]).
const DUMP_MARKET_WS_STREAM_TXT: bool = false;

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

/// Асинхронно пишет дамп **каждого лейна** в `xframes/{currency}/{count_features}/{interval}/{step}s/{YYYY-MM-DD}/{name}.bin`,
/// а после завершения (успех или ошибка) вызывает `cleanup_stale_market_data`
/// чтобы освободить память, занятую данными завершённого маркета.
// `slug` — slug дампируемого окна (формат `{currency}-updown-{period}-{window_start_sec}`).
// Нужен только для диагностического `eprintln!`: собираем `polymarket_event_url`,
// чтобы по логу можно было кликнуть на маркет и сравнить `price_to_beat` /
// `final_price` с тем, что показывает Polymarket.
pub fn spawn_dump_market_xframes_binary(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
    period_sec: i64,
    price_to_beat: f64,
    final_price: f64,
    slug: String,
    event_end_ms: i64,
) {
    tokio::spawn(async move {
        let interval_kind = XFrameIntervalKind::from_period_sec(period_sec);
        let max_step = *FRAME_BUILD_INTERVALS_SEC.iter().max().unwrap_or(&1);
        tokio::time::sleep(std::time::Duration::from_secs(max_step)).await;

        let up_won = final_price >= price_to_beat;
        let polymarket_event_url = format!("https://polymarket.com/event/{slug}");
        eprintln!(
            "xframe_dump: market_id={market_id} polymarket={polymarket_event_url} price_to_beat={price_to_beat} final_price={final_price} up_won={up_won}"
        );


        for lane in 0..FRAME_BUILD_INTERVALS_SEC.len() {
            if let Err(err) = dump_market_xframes_binary_lane(
                project_manager.clone(),
                market_id.clone(),
                gamma_question.clone(),
                interval_kind,
                lane,
                price_to_beat,
                final_price,
                event_end_ms,
            )
            .await
            {
                eprintln!("xframe_dump lane={lane}: {err:#}");
            }
            if let Err(err) = crate::xframe_graph_dump::dump_market_graph_html_lane(
                project_manager.clone(),
                market_id.clone(),
                gamma_question.clone(),
                interval_kind,
                lane,
                price_to_beat,
                final_price,
                event_end_ms,
            )
            .await
            {
                eprintln!("graph_dump lane={lane}: {err:#}");
            }
        }
        if DUMP_MARKET_WS_STREAM_TXT {
            if let Err(err) = dump_market_ws_stream_txt(
                project_manager.clone(),
                market_id.clone(),
                gamma_question.clone(),
                interval_kind,
            )
            .await
            {
                eprintln!("stream_dump: {err:#}");
            }
        }
        project_manager.cleanup_stale_market_data(&market_id).await;
    });
}

/// `event_end_ms` — Unix-мс конца окна Polymarket; используется как
/// детерминированный суффикс `__{ms}.bin` в имени файла, чтобы партиал-HTML
/// ([`crate::xframe_graph_dump::spawn_partial_market_graph_html_for_close`])
/// и финальный `.bin` / HTML по одному рынку лежали по совпадающим путям, и
/// финальный дамп перезаписывал партиал.
pub async fn dump_market_xframes_binary_lane(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
    interval_kind: XFrameIntervalKind,
    lane: usize,
    price_to_beat: f64,
    final_price: f64,
    event_end_ms: i64,
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
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    };

    let step_secs = FRAME_BUILD_INTERVALS_SEC[lane];

    let frame_count = frames_up.len() + frames_down.len();
    let dump = MarketXFramesDump {
        frames_up,
        frames_down,
        price_to_beat,
        final_price,
    };

    // Версия схемы дампа — см. [`crate::xframe::xframe_bincode_schema_size_bytes`].
    let schema_size = crate::xframe::xframe_bincode_schema_size_bytes();

    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let base: PathBuf = Path::new("xframes")
        .join(project_manager.currency.as_str())
        .join(format!("{schema_size}"))
        .join(interval_label)
        .join(format!("{step_secs}s"))
        .join(&date);
    tokio::fs::create_dir_all(&base).await?;

    let stem = sanitized_filename_from_gamma_question(gamma_question.as_deref());
    let fname = format!("{stem}__{event_end_ms}.bin");
    let path = base.join(&fname);
    let bytes = bincode::serialize(&dump)?;
    let byte_len = bytes.len();
    tokio::fs::write(&path, bytes).await?;
    run_log::xframe_dump_written(&path, &market_id, frame_count, byte_len);
    Ok(())
}

/// Собирает ожидаемый относительный путь `.bin` под `xframes/…` в том же виде, что [`dump_market_xframes_binary_lane`],
/// **без обращения к диску** — файл может появиться позже у дампера.
///
/// Суффикс `__{ts}` детерминирован: это `event_end_ms` (Unix-мс конца окна
/// Polymarket). Тот же суффикс используют [`dump_market_xframes_binary_lane`]
/// (финальный `.bin`) и [`crate::xframe_graph_dump::dump_market_graph_html_lane`]
/// (финальный HTML), а также [`crate::xframe_graph_dump::spawn_partial_market_graph_html_for_close`]
/// (партиал-HTML, путь которого выводится из этого `.bin`-пути через
/// [`crate::xframe_graph_dump::graph_html_path_from_bin`]). Благодаря общему
/// суффиксу финальный дамп при резолюции рынка перезаписывает партиал, а
/// несколько закрытий по одному рынку обновляют один и тот же HTML.
///
/// `event_end_ms == None` — fallback на `current_timestamp_ms()` (старое
/// поведение); файл тогда будет уникальным и не перезатрётся финалом, но
/// CSV-ссылка по крайней мере не пустая. Нужен только для колонки графика в
/// CSV ([`crate::real_sim`], fallback в [`crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri`]).
pub(crate) fn synthetic_xframes_dump_bin_path_for_csv_link(
    currency: &str,
    interval_kind: XFrameIntervalKind,
    stem: &str,
    event_end_ms: Option<i64>,
) -> Option<PathBuf> {
    if stem.is_empty() {
        return None;
    }
    let interval_label = match interval_kind {
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    };
    let schema_size = crate::xframe::xframe_bincode_schema_size_bytes();
    let step_secs = *FRAME_BUILD_INTERVALS_SEC.first().unwrap_or(&1);
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let ts_suffix = event_end_ms.unwrap_or_else(current_timestamp_ms);
    let fname = format!("{stem}__{ts_suffix}.bin");
    Some(
        Path::new("xframes")
            .join(currency)
            .join(format!("{schema_size}"))
            .join(interval_label)
            .join(format!("{step_secs}s"))
            .join(date)
            .join(fname),
    )
}

pub async fn dump_market_ws_stream_txt(
    project_manager: Arc<ProjectManager>,
    market_id: String,
    gamma_question: Option<String>,
    interval_kind: XFrameIntervalKind,
) -> anyhow::Result<()> {
    let entries = {
        let asset_ids = project_manager
            .market_asset_ids_by_market
            .read()
            .await
            .get(&market_id)
            .cloned()
            .unwrap_or_default();
        let ws_stream_by_asset_id = project_manager.ws_stream_by_asset_id.read().await;
        let mut out = Vec::new();
        for asset_id in asset_ids {
            if let Some(list) = ws_stream_by_asset_id.get(&asset_id) {
                out.extend(list.iter().cloned());
            }
        }
        out.sort_by_key(|e| (e.ingest_wall_ms, e.event_timestamp_ms));
        out
    };
    if entries.is_empty() {
        return Ok(());
    }
    let mut entries_by_asset_id: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for entry in entries {
        entries_by_asset_id
            .entry(entry.asset_id.clone())
            .or_default()
            .push(entry);
    }

    let interval_label = match interval_kind {
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    };

    let schema_size = crate::xframe::xframe_bincode_schema_size_bytes();
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let base: PathBuf = Path::new("streams")
        .join(project_manager.currency.as_str())
        .join(format!("{schema_size}"))
        .join(interval_label)
        .join(&date);

    let stem = sanitized_filename_from_gamma_question(gamma_question.as_deref());
    let ts = current_timestamp_ms();

    for (asset_id, asset_entries) in entries_by_asset_id {
        let asset_dir = base.join(asset_id.as_str());
        tokio::fs::create_dir_all(&asset_dir).await?;

        let fname = format!("{stem}__{ts}.txt");
        let path = asset_dir.join(&fname);

        let mut out = String::new();
        writeln!(&mut out, "# ws_stream_dump").ok();
        writeln!(&mut out, "market_id={market_id}").ok();
        writeln!(&mut out, "asset_id={asset_id}").ok();
        writeln!(&mut out, "currency={}", project_manager.currency).ok();
        writeln!(&mut out, "interval={interval_label}").ok();
        writeln!(
            &mut out,
            "format=ingest_wall_ms|event_ts_ms|asset_id|event_type|payload_raw"
        )
        .ok();
        for e in asset_entries {
            let payload_one_line = e.payload_raw.replace('\n', "\\n");
            writeln!(
                &mut out,
                "{}|{}|{}|{}|{}",
                e.ingest_wall_ms, e.event_timestamp_ms, e.asset_id, e.event_type, payload_one_line
            )
            .ok();
        }
        tokio::fs::write(&path, out).await?;
    }
    Ok(())
}
