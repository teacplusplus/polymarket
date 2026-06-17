//! Redeem-01 tail rule.
//!
//! This module contains the live/project decision rule only. It does not replay
//! bot activity and does not depend on precomputed market/asset tables.

use crate::constants::XFrameIntervalKind;
use crate::history_sim::{MIN_ENTRY_REMAINING_MS, StrictBook};
use crate::project_manager::ProjectManager;
use crate::xframe::{BookLevel, SIZE, XFrame};
use crate::xframe_dump::MarketXFramesDump;
use anyhow::Context as _;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::time::{self, Duration};

#[derive(Debug, Clone, Copy, Default)]
pub struct Redeem01TailMarketRegime {
    pub markets: usize,
    pub reversals: usize,
    pub reversal_rate: f64,
    pub can_buy: bool,
    pub updated_at_ms: i64,
}

const BTC_5M_CURRENCY: &str = "btc";
const BTC_5M_TAIL_MAX_PRICE: f64 = 0.03;
const BTC_5M_TARGET_SHARES: f64 = 5.0;
const MARKET_REGIME_WINDOW_MS: i64 = 24 * 60 * 60 * 1_000;
const MARKET_REGIME_REFRESH_SECS: u64 = 60 * 60;

pub fn spawn_redeem_01_tail_market_regime_refresh(project_manager: Arc<ProjectManager>) {
    tokio::spawn(async move {
        loop {
            if let Err(err) = refresh_redeem_01_tail_market_regime(project_manager.clone()).await {
                crate::tee_eprintln!(
                    "[redeem_01_tail] market regime refresh failed currency={}: {err:#}",
                    project_manager.currency.as_str(),
                );
            }
            time::sleep(Duration::from_secs(MARKET_REGIME_REFRESH_SECS)).await;
        }
    });
}

async fn refresh_redeem_01_tail_market_regime(
    project_manager: Arc<ProjectManager>,
) -> anyhow::Result<()> {
    let currency = project_manager.currency.to_ascii_lowercase();
    let xframes_root = crate::path_config::xframes_root();
    let now_ms = crate::util::current_timestamp_ms();
    let regimes = scan_redeem_01_tail_market_regime(&xframes_root, &currency, now_ms)?;

    let mut guard = project_manager.redeem_01_tail_market_regime.write().await;
    for (key, regime) in regimes {
        guard.insert(key, regime);
    }
    Ok(())
}

fn scan_redeem_01_tail_market_regime(
    xframes_root: &Path,
    currency: &str,
    now_ms: i64,
) -> anyhow::Result<HashMap<XFrameIntervalKind, Redeem01TailMarketRegime>> {
    let currency_root = xframes_root.join(currency);
    if !currency_root.exists() {
        return Ok(HashMap::new());
    }

    let cutoff_ms = now_ms - MARKET_REGIME_WINDOW_MS;
    let mut stats: HashMap<XFrameIntervalKind, (usize, usize)> = HashMap::new();
    let files = collect_bin_files(&currency_root)?;

    for path in files {
        let Some((interval, event_end_ms)) = dump_path_interval_and_end_ms(&currency_root, &path)
        else {
            continue;
        };
        if event_end_ms < cutoff_ms || event_end_ms > now_ms {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else { continue };
        let Ok(dump) = bincode::deserialize::<MarketXFramesDump>(&bytes) else {
            continue;
        };
        let winning_frames = if dump.up_won() {
            &dump.frames_up
        } else {
            &dump.frames_down
        };
        let reversed = winning_frames.iter().any(is_tail_frame);
        let entry = stats.entry(interval).or_insert((0, 0));
        entry.0 += 1;
        if reversed {
            entry.1 += 1;
        }
    }

    let mut regimes = HashMap::with_capacity(stats.len());
    for (key, (markets, reversals)) in stats {
        let reversal_rate = if markets > 0 {
            reversals as f64 / markets as f64
        } else {
            0.0
        };
        regimes.insert(
            key,
            Redeem01TailMarketRegime {
                markets,
                reversals,
                reversal_rate,
                can_buy: reversal_rate > BTC_5M_TAIL_MAX_PRICE,
                updated_at_ms: now_ms,
            },
        );
    }
    Ok(regimes)
}

fn collect_bin_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            out.extend(collect_bin_files(&path)?);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
            out.push(path);
        }
    }
    Ok(out)
}

fn dump_path_interval_and_end_ms(
    currency_root: &Path,
    path: &Path,
) -> Option<(XFrameIntervalKind, i64)> {
    let rel = path.strip_prefix(currency_root).ok()?;
    let mut comps = rel.components();
    let _version = comps.next()?;
    let period = comps.next()?.as_os_str().to_str()?;
    let interval = match period {
        "5m" => XFrameIntervalKind::FiveMin,
        "15m" => XFrameIntervalKind::FifteenMin,
        _ => return None,
    };
    let stem = path.file_stem()?.to_str()?;
    let end_ms = stem.rsplit("__").next()?.parse().ok()?;
    Some((interval, end_ms))
}

fn is_tail_frame(frame: &XFrame<SIZE>) -> bool {
    if !frame.stable || frame.event_remaining_ms < MIN_ENTRY_REMAINING_MS {
        return false;
    }
    tail_prob(frame).is_some_and(|p| p > 0.0 && p <= BTC_5M_TAIL_MAX_PRICE)
}

fn tail_prob(frame: &XFrame<SIZE>) -> Option<f64> {
    frame
        .currency_implied_prob
        .or(frame.book_ask_l1_price)
        .or(frame.last_trade_price)
}

pub(crate) async fn redeem_01_tail_entry_size(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    _entry_prob: f64,
    _bankroll: f64,
    currency: &str,
    project_manager: Option<&Arc<ProjectManager>>,
) -> Option<f64> {
    if !currency.eq_ignore_ascii_case(BTC_5M_CURRENCY) {
        return None;
    }
    let interval = XFrameIntervalKind::from_i32(frame.xframe_interval_type)?;
    if interval != XFrameIntervalKind::FiveMin {
        return None;
    }
    if frame.event_remaining_ms < 50_000
    {
        return None;
    }
    if !redeem_01_tail_market_regime_allows_buy(interval, project_manager).await {
        return None;
    }
    if cheap_tail_ask_notional(frame, strict_book, BTC_5M_TAIL_MAX_PRICE).is_none() {
        return None;
    }
    // if !btc_5m_tail_frame_allows_entry(frame, entry_prob) {
    //     return None;
    // }

    Some(BTC_5M_TARGET_SHARES)
}

async fn redeem_01_tail_market_regime_allows_buy(
    interval: XFrameIntervalKind,
    project_manager: Option<&Arc<ProjectManager>>,
) -> bool {
    let Some(project_manager) = project_manager else {
        return true;
    };
    let guard = project_manager.redeem_01_tail_market_regime.read().await;
    guard
        .get(&interval)
        .map(|regime| regime.can_buy)
        .unwrap_or(true)
}

fn cheap_tail_ask_notional(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    max_price: f64,
) -> Option<f64> {
    let notional: f64 = ask_levels(frame, strict_book)
        .into_iter()
        .filter(|level| level.price > 0.0 && level.price <= max_price && level.size > 0.0)
        .map(|level| level.price * level.size)
        .sum();
    (notional > 0.0).then_some(notional)
}

fn ask_levels(frame: &XFrame<SIZE>, strict_book: Option<&StrictBook>) -> Vec<BookLevel> {
    if let Some(book) = strict_book {
        return book.asks.clone();
    }
    if let Some(asks) = frame.book_asks.as_ref() {
        return asks.clone();
    }
    [
        (frame.book_ask_l1_price, frame.book_ask_l1_size),
        (frame.book_ask_l2_price, frame.book_ask_l2_size),
        (frame.book_ask_l3_price, frame.book_ask_l3_size),
    ]
    .into_iter()
    .filter_map(|(price, size)| {
        Some(BookLevel {
            price: price?,
            size: size?,
        })
    })
    .collect()
}
