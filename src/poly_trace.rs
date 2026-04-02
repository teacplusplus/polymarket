//! Подробный stderr-лог для отладки буферов WS, снапшотов, кадров и порядка API.

use crate::market_snapshot::MarketSnapshot;

pub fn enabled() -> bool {
    for key in ["POLYMARKET_TRACE", "POLY_TRACE"] {
        if let Ok(value) = std::env::var(key) {
            if matches!(value.as_str(), "0" | "false" | "no") {
                return false;
            }
        }
    }
    true
}

pub fn api_line(label: &str, detail: &str) {
    if enabled() {
        eprintln!("[poly api] {label}: {detail}");
    }
}

pub fn buffer_ingest(frame_group: usize, market_id: &str, asset_id: &str, queue_len: usize) {
    if enabled() {
        eprintln!(
            "[poly ws_buffer[{frame_group}]] push → market_id={market_id} asset_id={asset_id} queue_len={queue_len}"
        );
    }
}

pub fn buffer_drain(
    frame_group: usize,
    drain_now_ms: i64,
    aggregated_count: usize,
    snapshots: &[MarketSnapshot],
) {
    if !enabled() {
        return;
    }
    eprintln!(
        "[poly ws_buffer[{frame_group}]] drain now_ms={drain_now_ms} → {aggregated_count} aggregated row(s)"
    );
    for (idx, snap) in snapshots.iter().enumerate() {
        eprintln!(
            "  [{idx}] market_id={} asset_id={} ts_ms={} best_bid={:?} best_ask={:?} last_px={:?} last_sz={:?} vol_bucket={:?} side={:?}",
            snap.market_id,
            snap.asset_id,
            snap.timestamp_ms,
            snap.best_bid,
            snap.best_ask,
            snap.last_trade_price,
            snap.last_trade_size,
            snap.trade_volume_bucket,
            snap.trade_side
        );
    }
}

pub fn ws_snapshots_insert(
    frame_group: usize,
    market_id: &str,
    asset_id: &str,
    aligned_ts: i64,
    merged: bool,
    total_keys_for_asset: usize,
) {
    if enabled() {
        eprintln!(
            "[poly ws_snapshots[{frame_group}]] insert aligned_ts={aligned_ts} market_id={market_id} asset_id={asset_id} merged_with_prior={merged} total_keys_this_asset={total_keys_for_asset}"
        );
    }
}

pub fn xframes_insert(
    frame_group: usize,
    market_id: &str,
    asset_id: &str,
    aligned_ts: i64,
    had_prior_merge: bool,
    total_keys_for_asset: usize,
) {
    if enabled() {
        eprintln!(
            "[poly xframes[{frame_group}]] insert aligned_ts={aligned_ts} market_id={market_id} asset_id={asset_id} after_merge_recompute_followups={had_prior_merge} total_keys_this_asset={total_keys_for_asset}"
        );
    }
}

pub fn historical_xframe_insert(
    frame_group: usize,
    interval_secs: u64,
    market_id: &str,
    asset_id: &str,
    aligned_ts: i64,
) {
    if enabled() {
        eprintln!(
            "[poly xframes[{frame_group}] hist] interval={interval_secs}s aligned_ts={aligned_ts} market_id={market_id} asset_id={asset_id}"
        );
    }
}

pub fn api_context_stored(market_id: &str, asset_id: &str, end_date: Option<&str>) {
    if enabled() {
        eprintln!(
            "[poly api_context_by_market] stored market_id={market_id} asset_id={asset_id} end_date_rfc3339={end_date:?}"
        );
    }
}

pub fn job_done(job: &str) {
    if enabled() {
        eprintln!("[poly api job] done: {job}");
    }
}
