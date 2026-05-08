//! Одноразовая миграция полей [`crate::xframe_dump::MarketXFramesDump::price_to_beat`]
//! и зависимых от него полей внутри [`crate::xframe::XFrame`] для уже сохранённых
//! дампов под `xframes/{currency}/<schema_size>/...`.
//!
//! # Зачем
//!
//! До правки цикла [`crate::project_manager::ProjectManager::run_currency_updown_interval`]
//! в дампы попадал `price_to_beat` со страницы polymarket.com со включённым
//! `fallback_to_latest=true`: если в момент старта окна `past-results` ещё не
//! содержал ряда `endTime == window_start`, fallback возвращал `closePrice`
//! предпоследнего окна — то есть сдвинутую на 1 окно цену. Это значение через
//! `prev_market.price_to_beat` записывалось в дамп **следующего** маркета и
//! одновременно использовалось при расчёте `XFrame::currency_price_vs_beat_pct`
//! (а через sibling-кадр — `XFrame::sibling_currency_price_vs_beat_pct`).
//!
//! Чтобы не выкидывать накопленные кадры, миграция:
//! 1. Заходит в `xframes/{currency}/<current_schema_size>/{5m,15m}/<step>/<date>/*.bin`.
//! 2. По имени файла восстанавливает `(currency, interval_kind, window_start_sec)`
//!    (как это делает [`crate::history_sim::polymarket_event_url_from_dump_path`])
//!    и формирует slug `{currency}-updown-{label}-{window_start_sec}`.
//! 3. Через [`crate::util::fetch_price_to_beat_from_vatic_api`] забирает
//!    истинный `priceToBeat` (target/opening price окна) из Vatic API
//!    `targets/timestamp` — единственный источник правды теперь, когда
//!    скрейп `__NEXT_DATA__` со страницы маркета удалён.
//! 4. Перезаписывает `dump.price_to_beat` и пересчитывает у каждого кадра
//!    `currency_price_vs_beat_pct` (использует тот же спот, восстановленный из
//!    старых полей: `spot = old_ptb * (1 - old_pct/100)`) и
//!    `sibling_currency_price_vs_beat_pct` (использует тот же спот и
//!    исправленный sibling `priceToBeat`, если он у нас уже выкачан в этой же
//!    миграции — иначе sibling-поле остаётся как было).
//!
//! # Что НЕ трогаем
//!
//! * `dump.final_price` — этот канал правильный: он берётся как
//!   `price_to_beat` СЛЕДУЮЩЕГО окна, и в момент его записи `prev_market.final
//!   = currentIter.price_to_beat`. Эмпирически проверено на пяти сохранённых
//!   подряд окнах: `final_price[N]` совпадает с `closePrice` past-results, а
//!   значит, и со следующим `priceToBeat[N+1]` (см. discussion в чате).
//! * Поля кадра, не зависящие от `price_to_beat` напрямую — стакан, объёмы,
//!   `currency_price_z_score` (использует историю спота, но не beat).
//!
//! # Идемпотентность
//!
//! Если `|new_ptb - old_ptb| < 1e-6` (значение и так точное) — файл
//! пропускается. Повторный запуск миграции на уже обновлённых дампах ничего
//! не меняет.

use crate::constants::XFrameIntervalKind;
use crate::util::fetch_price_to_beat_from_vatic_api;
use crate::xframe::{XFrame, SIZE};
use crate::xframe_dump::MarketXFramesDump;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Сколько раз дёргать Vatic API за `priceToBeat`, прежде чем сдаться и
/// пропустить окно. Между попытками — фиксированная пауза [`HTTP_RETRY_DELAY`].
/// На слишком старые окна (Chainlink retention ~14 дней) Vatic возвращает 410 —
/// дамп такого окна тоже не трогаем.
const HTTP_MAX_ATTEMPTS: u32 = 5;
const HTTP_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(2);

/// Точка входа миграции (`STATUS=migrate_price_to_beat`).
///
/// Async — потому что HTTP-запросы за `priceToBeat` идут через тот же
/// [`fetch_price_to_beat_from_vatic_api`], что и в основном цикле,
/// и оперируют `reqwest::Client` (`tokio`-async).
pub async fn run_price_to_beat_migration() -> Result<()> {
    let http = reqwest::Client::builder()
        .use_rustls_tls()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let current_size = crate::migration::current_schema_size();
    println!("[migration_ptb] current schema_size={current_size}");

    for currency in crate::CURRENCIES {
        let dump_root = Path::new("xframes")
            .join(currency)
            .join(format!("{current_size}"));
        if !dump_root.exists() {
            println!(
                "[migration_ptb] {currency}: каталог {} отсутствует, пропуск",
                dump_root.display()
            );
            continue;
        }

        // Pass 1: список всех дампов с восстановленным `(interval_kind, window_start_sec)`.
        let dumps = collect_dump_files_with_window(&dump_root)?;
        println!(
            "[migration_ptb] {currency}: найдено {} .bin файлов",
            dumps.len()
        );

        // Pass 2: для каждого уникального `(interval_kind, window_start_sec)` тянем
        // exact `priceToBeat`. Дедуп по ключу — нужен один HTTP-запрос на окно.
        let mut correct_ptb: HashMap<(XFrameIntervalKind, i64), f64> = HashMap::new();
        let mut unique_keys: Vec<(XFrameIntervalKind, i64)> = Vec::new();
        for (_, interval_kind, window_start_sec) in &dumps {
            let key = (*interval_kind, *window_start_sec);
            if !correct_ptb.contains_key(&key) {
                unique_keys.push(key);
                correct_ptb.insert(key, f64::NAN);
            }
        }
        println!(
            "[migration_ptb] {currency}: уникальных окон {}",
            unique_keys.len()
        );

        let mut fetched = 0usize;
        let mut failed = 0usize;
        for (idx, (interval_kind, window_start_sec)) in unique_keys.iter().enumerate() {
            let interval_label = interval_label(*interval_kind);
            let slug = format!("{currency}-updown-{interval_label}-{window_start_sec}");
            match fetch_exact_with_retries(&http, &slug, currency).await {
                Some(ptb) => {
                    correct_ptb.insert((*interval_kind, *window_start_sec), ptb);
                    fetched += 1;
                    println!(
                        "[migration_ptb] {currency}: HTTP {}/{} (last slug={slug} ptb={ptb})",
                        idx + 1,
                        unique_keys.len()
                    );
                }
                None => {
                    correct_ptb.remove(&(*interval_kind, *window_start_sec));
                    failed += 1;
                    eprintln!(
                        "[migration_ptb] {currency}: {slug} exact priceToBeat не получен — окно пропущено"
                    );
                }
            }
        }
        println!(
            "[migration_ptb] {currency}: HTTP fetched={fetched} failed={failed}"
        );

        // Pass 3: переписываем дампы. Если для окна нет exact ptb (HTTP упал) —
        // дамп этого окна тоже не трогаем. Sibling в зависимом поле кадра
        // обновляется ТОЛЬКО когда у нас есть exact ptb sibling-окна.
        let mut rewritten = 0usize;
        let mut unchanged = 0usize;
        let mut skipped_no_ptb = 0usize;
        let mut errored = 0usize;
        for (path, interval_kind, window_start_sec) in &dumps {
            let Some(&new_ptb) = correct_ptb.get(&(*interval_kind, *window_start_sec)) else {
                skipped_no_ptb += 1;
                continue;
            };
            match rewrite_dump(path, *interval_kind, *window_start_sec, new_ptb, &correct_ptb) {
                Ok(RewriteResult::Rewritten) => rewritten += 1,
                Ok(RewriteResult::Unchanged) => unchanged += 1,
                Err(err) => {
                    errored += 1;
                    eprintln!("[migration_ptb] {}: {err:#}", path.display());
                }
            }
        }
        println!(
            "[migration_ptb] {currency}: rewritten={rewritten} unchanged={unchanged} skipped_no_ptb={skipped_no_ptb} errored={errored}"
        );
    }

    Ok(())
}

enum RewriteResult {
    Rewritten,
    Unchanged,
}

/// Десериализует дамп, считает разницу `new_ptb - old_ptb`. Если пренебрежимо
/// мала — выходит без записи. Иначе пересчитывает у каждого кадра
/// `currency_price_vs_beat_pct` и `sibling_currency_price_vs_beat_pct` через
/// восстановленный спот и записывает дамп обратно.
fn rewrite_dump(
    path: &Path,
    interval_kind: XFrameIntervalKind,
    window_start_sec: i64,
    new_ptb: f64,
    correct_ptb: &HashMap<(XFrameIntervalKind, i64), f64>,
) -> Result<RewriteResult> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut dump: MarketXFramesDump = bincode::deserialize(&bytes)
        .with_context(|| format!("deserialize {}", path.display()))?;
    let old_ptb = dump.price_to_beat;

    if !old_ptb.is_finite() || !new_ptb.is_finite() {
        // Пишем только если оба значения корректны: иначе восстановить спот не
        // получится, а sibling-перерасчёт пойдёт мусором.
        anyhow::bail!(
            "не-finite priceToBeat: old={old_ptb} new={new_ptb} для {}",
            path.display()
        );
    }
    if (new_ptb - old_ptb).abs() < 1e-6 {
        return Ok(RewriteResult::Unchanged);
    }

    let interval_ms = interval_kind.interval_ms();
    let window_start_ms = window_start_sec.saturating_mul(1000);
    let window_end_ms = window_start_ms.saturating_add(interval_ms);
    let sibling_kind = interval_kind.sibling();
    let sibling_period_sec = sibling_kind.interval_ms() / 1000;

    for frame in dump.frames_up.iter_mut() {
        recompute_frame_pct_fields(
            frame,
            old_ptb,
            new_ptb,
            sibling_kind,
            sibling_period_sec,
            window_end_ms,
            correct_ptb,
        );
    }
    for frame in dump.frames_down.iter_mut() {
        recompute_frame_pct_fields(
            frame,
            old_ptb,
            new_ptb,
            sibling_kind,
            sibling_period_sec,
            window_end_ms,
            correct_ptb,
        );
    }

    dump.price_to_beat = new_ptb;
    let serialized = bincode::serialize(&dump)
        .with_context(|| format!("serialize {}", path.display()))?;
    fs::write(path, serialized).with_context(|| format!("write {}", path.display()))?;
    Ok(RewriteResult::Rewritten)
}

/// Пересчитывает у одного кадра поля, зависящие от `price_to_beat`.
///
/// Идея: если в кадре уже есть `currency_price_vs_beat_pct` (его записал live
/// pipeline по формуле `(old_ptb - spot) / old_ptb * 100`), мы можем
/// **восстановить** спот этого кадра без обращения к историческим RTDS:
///
/// ```text
/// pct  = (ptb - spot) / ptb * 100
///   ⇒  spot = ptb * (1 - pct/100)
/// ```
///
/// А имея спот, легко получить `new_pct = (new_ptb - spot) / new_ptb * 100`
/// и аналогично пересчитать `sibling_currency_price_vs_beat_pct` (тот же
/// спот, другой beat — sibling-окна).
fn recompute_frame_pct_fields(
    frame: &mut XFrame<SIZE>,
    old_ptb: f64,
    new_ptb: f64,
    sibling_kind: XFrameIntervalKind,
    sibling_period_sec: i64,
    window_end_ms: i64,
    correct_ptb: &HashMap<(XFrameIntervalKind, i64), f64>,
) {
    let Some(old_pct) = frame.currency_price_vs_beat_pct else {
        return;
    };
    if !old_pct.is_finite() {
        return;
    }
    // Восстанавливаем спот валюты на момент кадра из старых полей.
    let spot = old_ptb * (1.0 - old_pct / 100.0);
    if !spot.is_finite() {
        return;
    }

    if new_ptb.abs() > 1e-9 {
        let new_pct = (new_ptb - spot) / new_ptb * 100.0;
        if new_pct.is_finite() {
            frame.currency_price_vs_beat_pct = Some(new_pct);
        }
    }

    // Sibling: ищем sibling-окно по wall_time кадра и подставляем его exact ptb,
    // если оно у нас уже выкачано в текущей миграции. Если нет — оставляем
    // старое значение поля (лучше иметь приближённое, чем потерять признак).
    if frame.sibling_currency_price_vs_beat_pct.is_some() {
        let event_remaining_ms = frame.event_remaining_ms;
        if event_remaining_ms < 0 {
            return;
        }
        let frame_wall_time_ms = window_end_ms - event_remaining_ms;
        if frame_wall_time_ms < 0 {
            return;
        }
        let frame_wall_time_sec = frame_wall_time_ms / 1000;
        if sibling_period_sec <= 0 {
            return;
        }
        let sibling_ws = (frame_wall_time_sec / sibling_period_sec) * sibling_period_sec;
        if let Some(&sibling_new_ptb) = correct_ptb.get(&(sibling_kind, sibling_ws)) {
            if sibling_new_ptb.is_finite() && sibling_new_ptb.abs() > 1e-9 {
                let sibling_new_pct = (sibling_new_ptb - spot) / sibling_new_ptb * 100.0;
                if sibling_new_pct.is_finite() {
                    frame.sibling_currency_price_vs_beat_pct = Some(sibling_new_pct);
                }
            }
        }
    }
}

async fn fetch_exact_with_retries(
    http: &reqwest::Client,
    slug: &str,
    currency: &str,
) -> Option<f64> {
    for attempt in 1..=HTTP_MAX_ATTEMPTS {
        match fetch_price_to_beat_from_vatic_api(http, slug, currency).await {
            Ok(p) => return Some(p),
            Err(e) => {
                if attempt < HTTP_MAX_ATTEMPTS {
                    tokio::time::sleep(HTTP_RETRY_DELAY).await;
                } else {
                    eprintln!("[migration_ptb] {slug}: финальная ошибка HTTP — {e:#}");
                }
            }
        }
    }
    None
}

fn collect_dump_files_with_window(
    size_root: &Path,
) -> Result<Vec<(PathBuf, XFrameIntervalKind, i64)>> {
    let mut out = Vec::new();
    for interval_entry in fs::read_dir(size_root)
        .with_context(|| format!("read_dir {}", size_root.display()))?
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
        for step_entry in fs::read_dir(&interval_path)?.flatten() {
            let step_path = step_entry.path();
            if !step_path.is_dir() {
                continue;
            }
            for date_entry in fs::read_dir(&step_path)?.flatten() {
                let date_path = date_entry.path();
                if !date_path.is_dir() {
                    continue;
                }
                for file_entry in fs::read_dir(&date_path)?.flatten() {
                    let file_path = file_entry.path();
                    if !file_path.is_file() {
                        continue;
                    }
                    if file_path.extension().and_then(|s| s.to_str()) != Some("bin") {
                        continue;
                    }
                    let Some(bounds) = crate::history_sim::window_bounds_from_dump_path(
                        &file_path,
                        interval_kind,
                    ) else {
                        continue;
                    };
                    out.push((file_path, interval_kind, bounds.window_start_sec));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn interval_label(kind: XFrameIntervalKind) -> &'static str {
    match kind {
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    }
}
