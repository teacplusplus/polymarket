//! Redeem-01 tail rule.
//!
//! This module contains the live/project decision rule only. It does not replay
//! bot activity and does not depend on precomputed market/asset tables.

use crate::account::{Redeem01TailMarketRegimeKey, SharedAccount};
use crate::constants::XFrameIntervalKind;
use crate::history_sim::{
    KELLY_MULTIPLIER, MAX_BET_FRACTION, MAX_POSITION_USD, MIN_POSITION_USD,
    StrictBook,
};
use crate::xframe::{BookLevel, SIZE, XFrame};
use crate::xframe_dump::MarketXFramesDump;
use anyhow::Context as _;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

const BTC_5M_CURRENCY: &str = "btc";
const BTC_5M_MAX_TARGET_USDC: f64 = 1.0;
const DEFAULT_REDEEM_01_TAIL_ENTRY_REMAINING_MS: i64 = 50_000;
const REDEEM_01_TAIL_ENTRY_PRICE_COUNT: usize = 3;
const REDEEM_01_TAIL_ENTRY_PRICES: [f64; REDEEM_01_TAIL_ENTRY_PRICE_COUNT] = [0.01, 0.02, 0.03];
const MARKET_REGIME_WINDOW_MS: i64 = 24 * 60 * 60 * 1_000;

#[derive(Debug, Clone, Copy, Default)]
pub struct Redeem01TailMarketRegime {
    pub markets: usize,
    pub price_stats: [Redeem01TailPriceRegime; REDEEM_01_TAIL_ENTRY_PRICE_COUNT],
    pub recommendation: Option<Redeem01TailEntryRecommendation>,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Redeem01TailPriceRegime {
    pub entry_price: f64,
    pub reversals: usize,
    pub reversal_rate: f64,
    pub deadline_event_remaining_ms: Option<i64>,
    pub expected_roi: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Redeem01TailEntryRecommendation {
    pub entry_price: f64,
    pub reversal_rate: f64,
    pub expected_roi: f64,
    pub min_event_remaining_ms: i64,
}

#[derive(Debug, Clone, Copy)]
struct Redeem01TailEntryPlan {
    entry_price: f64,
    target_usdc: f64,
    min_event_remaining_ms: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Redeem01TailMarketRegimeLoadCommand {
    pub account: SharedAccount,
    pub currency: String,
    pub interval: XFrameIntervalKind,
}

pub(crate) async fn run_redeem_01_tail_market_regime_loader(
    mut rx: mpsc::Receiver<Redeem01TailMarketRegimeLoadCommand>,
) {
    while let Some(command) = rx.recv().await {
        if let Err(err) = load_redeem_01_tail_market_regime_key(command).await {
            crate::tee_eprintln!("[redeem_01_tail] market regime load failed: {err:#}");
        }
    }
}

async fn load_redeem_01_tail_market_regime_key(
    command: Redeem01TailMarketRegimeLoadCommand,
) -> anyhow::Result<()> {
    let key: Redeem01TailMarketRegimeKey = (command.currency.clone(), command.interval);
    {
        let guard = command.account.redeem_01_tail_market_regime.read().await;
        if guard.contains_key(&key) {
            return Ok(());
        }
    }

    let xframes_root = crate::path_config::xframes_root();
    let now_ms = crate::util::current_timestamp_ms();
    let mut regimes = scan_redeem_01_tail_market_regime(&xframes_root, &command.currency, now_ms)?;
    let regime = regimes
        .remove(&command.interval)
        .unwrap_or_else(|| Redeem01TailMarketRegime {
            updated_at_ms: now_ms,
            ..Redeem01TailMarketRegime::default()
        });

    let mut guard = command.account.redeem_01_tail_market_regime.write().await;
    guard.entry(key).or_insert(regime);
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
    let mut stats: HashMap<XFrameIntervalKind, MarketRegimeAccumulator> = HashMap::new();
    let files = collect_bin_files(&currency_root)?;

    for path in files {
        let Some((interval, event_end_ms)) = dump_path_interval_and_end_ms(&currency_root, &path)
        else {
            continue;
        };
        if event_end_ms < cutoff_ms || event_end_ms > now_ms {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(dump) = bincode::deserialize::<MarketXFramesDump>(&bytes) else {
            continue;
        };
        let winning_frames = if dump.up_won() {
            &dump.frames_up
        } else {
            &dump.frames_down
        };

        let entry = stats.entry(interval).or_default();
        entry.markets += 1;
        for (idx, entry_price) in REDEEM_01_TAIL_ENTRY_PRICES.iter().copied().enumerate() {
            if let Some(deadline_ms) = latest_tail_entry_remaining_ms(winning_frames, entry_price) {
                let price_entry = &mut entry.price_entries[idx];
                price_entry.reversals += 1;
                price_entry.latest_entry_remaining_ms.push(deadline_ms);
            }
        }
    }

    let mut regimes = HashMap::with_capacity(stats.len());
    for (key, acc) in stats {
        let price_stats = std::array::from_fn(|idx| {
            let entry_price = REDEEM_01_TAIL_ENTRY_PRICES[idx];
            let price_acc = &acc.price_entries[idx];
            let reversal_rate = if acc.markets > 0 {
                price_acc.reversals as f64 / acc.markets as f64
            } else {
                0.0
            };
            let expected_roi = if entry_price > 0.0 {
                reversal_rate / entry_price - 1.0
            } else {
                0.0
            };
            Redeem01TailPriceRegime {
                entry_price,
                reversals: price_acc.reversals,
                reversal_rate,
                deadline_event_remaining_ms: median_i64(&price_acc.latest_entry_remaining_ms),
                expected_roi,
            }
        });
        let recommendation = choose_redeem_01_tail_recommendation(&price_stats);
        regimes.insert(
            key,
            Redeem01TailMarketRegime {
                markets: acc.markets,
                price_stats,
                recommendation,
                updated_at_ms: now_ms,
            },
        );
    }
    Ok(regimes)
}

#[derive(Default)]
struct MarketPriceAccumulator {
    reversals: usize,
    latest_entry_remaining_ms: Vec<i64>,
}

struct MarketRegimeAccumulator {
    markets: usize,
    price_entries: [MarketPriceAccumulator; REDEEM_01_TAIL_ENTRY_PRICE_COUNT],
}

impl Default for MarketRegimeAccumulator {
    fn default() -> Self {
        Self {
            markets: 0,
            price_entries: std::array::from_fn(|_| MarketPriceAccumulator::default()),
        }
    }
}

fn median_i64(values: &[i64]) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    Some(sorted[sorted.len() / 2])
}

fn choose_redeem_01_tail_recommendation(
    price_stats: &[Redeem01TailPriceRegime; REDEEM_01_TAIL_ENTRY_PRICE_COUNT],
) -> Option<Redeem01TailEntryRecommendation> {
    price_stats
        .iter()
        .filter(|stat| stat.reversals > 0)
        .filter(|stat| stat.reversal_rate > stat.entry_price)
        .filter(|stat| stat.deadline_event_remaining_ms.is_some())
        .max_by(|a, b| {
            let by_roi = a
                .expected_roi
                .partial_cmp(&b.expected_roi)
                .unwrap_or(std::cmp::Ordering::Equal);
            if by_roi != std::cmp::Ordering::Equal {
                return by_roi;
            }
            let by_reversals = a.reversals.cmp(&b.reversals);
            if by_reversals != std::cmp::Ordering::Equal {
                return by_reversals;
            }
            b.entry_price
                .partial_cmp(&a.entry_price)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .and_then(|stat| {
            Some(Redeem01TailEntryRecommendation {
                entry_price: stat.entry_price,
                reversal_rate: stat.reversal_rate,
                expected_roi: stat.expected_roi,
                min_event_remaining_ms: stat.deadline_event_remaining_ms?,
            })
        })
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

fn latest_tail_entry_remaining_ms(frames: &[XFrame<SIZE>], max_price: f64) -> Option<i64> {
    frames
        .iter()
        .filter(|frame| frame.stable && frame.event_remaining_ms >= DEFAULT_REDEEM_01_TAIL_ENTRY_REMAINING_MS)
        .filter(|frame| cheap_tail_ask_notional(frame, None, max_price).is_some())
        .map(|frame| frame.event_remaining_ms)
        .min()
}

pub(crate) async fn redeem_01_tail_entry_size(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    _entry_prob: f64,
    bankroll: f64,
    currency: &str,
    account: Option<&SharedAccount>,
) -> Option<f64> {
    if !currency.eq_ignore_ascii_case(BTC_5M_CURRENCY) {
        return None;
    }
    let interval = XFrameIntervalKind::from_i32(frame.xframe_interval_type)?;
    if interval != XFrameIntervalKind::FiveMin {
        return None;
    }

    let plan = redeem_01_tail_market_regime_entry_plan(currency, interval, bankroll, account).await?;
    if frame.event_remaining_ms < plan.min_event_remaining_ms {
        return None;
    }
    let cheap_notional = cheap_tail_ask_notional(frame, strict_book, plan.entry_price)?;
    if cheap_notional + 1e-9 < plan.target_usdc {
        return None;
    }

    Some(plan.target_usdc)
}

async fn redeem_01_tail_market_regime_entry_plan(
    currency: &str,
    interval: XFrameIntervalKind,
    bankroll: f64,
    account: Option<&SharedAccount>,
) -> Option<Redeem01TailEntryPlan> {
    let Some(account) = account else {
        return None;
    };
    let key: Redeem01TailMarketRegimeKey = (currency.to_ascii_lowercase(), interval);
    let guard = account.redeem_01_tail_market_regime.read().await;
    match guard.get(&key) {
        Some(regime) => regime.recommendation.and_then(|recommendation| {
            redeem_01_tail_entry_plan_from_recommendation(recommendation, bankroll)
        }),
        None => {
            let _ = account
                .redeem_01_tail_market_regime_tx
                .try_send(Redeem01TailMarketRegimeLoadCommand {
                    account: account.clone(),
                    currency: currency.to_ascii_lowercase(),
                    interval,
                });
            None
        }
    }
}

fn redeem_01_tail_entry_plan_from_recommendation(
    recommendation: Redeem01TailEntryRecommendation,
    bankroll: f64,
) -> Option<Redeem01TailEntryPlan> {
    let target_usdc = redeem_01_tail_recommended_usdc(
        bankroll,
        recommendation.entry_price,
        recommendation.reversal_rate,
    )?;
    Some(Redeem01TailEntryPlan {
        entry_price: recommendation.entry_price,
        target_usdc,
        min_event_remaining_ms: recommendation.min_event_remaining_ms,
    })
}

fn redeem_01_tail_recommended_usdc(
    bankroll: f64,
    entry_price: f64,
    reversal_rate: f64,
) -> Option<f64> {
    if !(bankroll > 0.0 && bankroll.is_finite()) {
        return None;
    }
    if !(entry_price > 0.0 && entry_price < 1.0 && entry_price.is_finite()) {
        return None;
    }
    let p_win = reversal_rate.clamp(0.0, 1.0);
    if p_win <= entry_price {
        return None;
    }
    let gain = (1.0 / entry_price - 1.0).max(1e-9);
    let kelly_f = p_win - (1.0 - p_win) / gain;
    let size = (kelly_f * KELLY_MULTIPLIER).min(MAX_BET_FRACTION).max(0.0) * bankroll;
    let size = size.min(MAX_POSITION_USD).min(BTC_5M_MAX_TARGET_USDC);
    (size >= MIN_POSITION_USD).then_some(size)
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
