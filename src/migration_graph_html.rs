//! Одноразовая генерация статических HTML-графиков для всех уже сохранённых дампов
//! `xframes/{currency}/<schema_size>/.../*.bin`: для каждого файла пишется зеркальный
//! `graph/.../*.html` (см. [`crate::xframe_graph_dump::try_write_graph_html_from_bin_dump`]).
//!
//! Запуск: `STATUS=migrate_graph_html`. Идемпотентна по смыслу перегенерации: существующие
//! `.html` перезаписываются.

use crate::constants::XFrameIntervalKind;
use crate::xframe_dump::MarketXFramesDump;
use crate::xframe_graph_dump::{GraphHtmlFromBinOutcome, try_write_graph_html_from_bin_dump};
use anyhow::{Context as _, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Точка входа (`STATUS=migrate_graph_html`).
pub fn run_graph_html_migration() -> Result<()> {
    let root = Path::new("xframes");
    if !root.exists() {
        println!(
            "[migration_graph_html] каталог {} отсутствует — нечего делать",
            root.display()
        );
        return Ok(());
    }

    let mut wrote = 0usize;
    let mut skip_bounds = 0usize;
    let mut skip_empty = 0usize;
    let mut err_bin = 0usize;
    let mut err_write = 0usize;

    for currency in crate::CURRENCIES {
        let cur_root = root.join(currency);
        if !cur_root.is_dir() {
            println!(
                "[migration_graph_html] {currency}: {} пропуск (нет каталога)",
                cur_root.display()
            );
            continue;
        }

        let bins = collect_bin_files_under_currency(&cur_root)
            .with_context(|| format!("обход {}", cur_root.display()))?;
        let total = bins.len();
        println!("[migration_graph_html] {currency}: найдено {total} .bin");

        // Шаг прогресса: ~каждые 1% от объёма, но не реже чем раз в 100 файлов
        // и не чаще чем раз в 10 файлов (для маленьких наборов).
        let progress_step = (total / 100).clamp(10, 100).max(1);
        let started = std::time::Instant::now();
        let cur_wrote_start = wrote;
        let cur_skip_bounds_start = skip_bounds;
        let cur_skip_empty_start = skip_empty;
        let cur_err_bin_start = err_bin;
        let cur_err_write_start = err_write;

        for (idx, (path, interval_kind)) in bins.into_iter().enumerate() {
            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    err_bin += 1;
                    eprintln!("[migration_graph_html] read {}: {e:#}", path.display());
                    continue;
                }
            };
            let dump: MarketXFramesDump = match bincode::deserialize(&bytes) {
                Ok(d) => d,
                Err(e) => {
                    err_bin += 1;
                    eprintln!(
                        "[migration_graph_html] deserialize {}: {e:#}",
                        path.display()
                    );
                    continue;
                }
            };

            match try_write_graph_html_from_bin_dump(&path, &dump, interval_kind) {
                Ok(GraphHtmlFromBinOutcome::Wrote(_)) => {
                    wrote += 1;
                }
                Ok(GraphHtmlFromBinOutcome::SkippedNoWindowBounds) => skip_bounds += 1,
                Ok(GraphHtmlFromBinOutcome::SkippedNoStableFrames) => skip_empty += 1,
                Err(e) => {
                    err_write += 1;
                    eprintln!("[migration_graph_html] {}: {e:#}", path.display());
                }
            }

            let done = idx + 1;
            if done % progress_step == 0 || done == total {
                let elapsed = started.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 { done as f64 / elapsed } else { 0.0 };
                let pct = (done as f64 / total.max(1) as f64) * 100.0;
                println!(
                    "[migration_graph_html] {currency}: {done}/{total} ({pct:.1}%), {rate:.1} файл/с, wrote={} skip_no_bounds={} skip_no_stable={} err_bin={} err_write={}",
                    wrote - cur_wrote_start,
                    skip_bounds - cur_skip_bounds_start,
                    skip_empty - cur_skip_empty_start,
                    err_bin - cur_err_bin_start,
                    err_write - cur_err_write_start,
                );
            }
        }
    }

    println!(
        "[migration_graph_html] итого: wrote={wrote} skip_no_bounds={skip_bounds} skip_no_stable={skip_empty} err_bin={err_bin} err_write={err_write}"
    );
    Ok(())
}

fn collect_bin_files_under_currency(cur_root: &Path) -> Result<Vec<(PathBuf, XFrameIntervalKind)>> {
    let mut out = Vec::new();
    // cur_root = xframes/{currency} — под ним любые каталоги schema_size (любая версия раскладки).
    for size_entry in fs::read_dir(cur_root)
        .with_context(|| format!("read_dir {}", cur_root.display()))?
        .flatten()
    {
        let size_path = size_entry.path();
        if !size_path.is_dir() {
            continue;
        }
        for interval_entry in fs::read_dir(&size_path)
            .with_context(|| size_path.display().to_string())?
            .flatten()
        {
            let interval_path = interval_entry.path();
            if !interval_path.is_dir() {
                continue;
            }
            let interval_name = interval_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let interval_kind = match interval_name {
                "5m" => XFrameIntervalKind::FiveMin,
                "15m" => XFrameIntervalKind::FifteenMin,
                _ => continue,
            };
            for step_entry in fs::read_dir(&interval_path)
                .with_context(|| interval_path.display().to_string())?
                .flatten()
            {
                let step_path = step_entry.path();
                if !step_path.is_dir() {
                    continue;
                }
                for date_entry in fs::read_dir(&step_path)
                    .with_context(|| step_path.display().to_string())?
                    .flatten()
                {
                    let date_path = date_entry.path();
                    if !date_path.is_dir() {
                        continue;
                    }
                    for file_entry in fs::read_dir(&date_path)
                        .with_context(|| date_path.display().to_string())?
                        .flatten()
                    {
                        let file_path = file_entry.path();
                        if !file_path.is_file() {
                            continue;
                        }
                        if file_path.extension().and_then(|s| s.to_str()) != Some("bin") {
                            continue;
                        }
                        out.push((file_path, interval_kind));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}
