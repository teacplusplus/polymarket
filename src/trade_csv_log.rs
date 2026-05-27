//! Per-trade CSV-лог симуляции: одна строка на каждую закрытую позицию
//! (рыночное закрытие через [`crate::account_close_position::close_position`] **и**
//! резолюционное закрытие через [`crate::account::Account::resolve_pending_market_sync`]).
//!
//! Строка сериализуется в [`write_trade_csv_row`] и сразу уходит в
//! [`crate::tee_log::trade_csv_log_write`]. Запись на диск — через общий
//! mpsc-writer в `tee_log`.

use std::path::Path;
use std::sync::Mutex;

/// Текущий режим симуляции для CSV-колонки `regime`.
static CURRENT_REGIME: Mutex<&'static str> = Mutex::new("");

pub fn set_current_regime(regime: &'static str) {
    if let Ok(mut guard) = CURRENT_REGIME.lock() {
        *guard = regime;
    }
}

const TRADE_CSV_HEADER: &str = "regime,polymarket_url,price_to_beat,final_price,\
currency,interval,side,market_id,asset_id,exit_reason,\
buy_price,raw_pred,cal_pred,kelly_f,entry_cost,shares_held,exit_price,fee_usdc,pnl,\
frames_held,p_win_ema_at_close,event_remaining_ms_at_open,event_remaining_ms_at_close,\
open_unix_ms,close_unix_ms,graph_html_file_uri,pnl_top5_shap,final_outcome";

pub fn init_trade_csv_log_file(path: &Path) -> std::io::Result<()> {
    crate::tee_log::init_trade_csv_log_file(path)?;
    crate::tee_log::trade_csv_log_write(TRADE_CSV_HEADER);
    Ok(())
}

pub fn finish_trade_csv_log() {
    crate::tee_log::finish_trade_csv_log();
}

#[derive(Debug, Clone, Copy)]
pub struct TradeCsvRow<'a> {
    pub polymarket_url: &'a str,
    pub price_to_beat: Option<f64>,
    pub final_price: Option<f64>,
    pub currency: &'a str,
    pub interval: &'a str,
    pub side: &'a str,
    pub market_id: &'a str,
    pub asset_id: &'a str,
    pub exit_reason: &'static str,
    pub buy_price: f64,
    pub raw_pred: f32,
    pub cal_pred: f32,
    pub kelly_f: f64,
    pub position_size: f64,
    pub shares_held: f64,
    pub exit_price: f64,
    pub fee_usdc: f64,
    pub pnl: f64,
    pub frames_held: usize,
    pub p_win_ema_at_close: Option<f64>,
    pub event_remaining_ms_at_open: i64,
    pub event_remaining_ms_at_close: i64,
    pub open_unix_ms: Option<i64>,
    pub close_unix_ms: Option<i64>,
    pub graph_html_file_uri: &'a str,
    pub pnl_top5_shap: &'a str,
    pub pos_id: &'a str,
    pub finalized_via: &'static str,
    pub planned_buy_price: Option<f64>,
    pub planned_shares_held: Option<f64>,
    pub planned_entry_cost: Option<f64>,
    pub planned_fee_usdc: Option<f64>,
    pub entry_fee_usdc: Option<f64>,
    pub open_order_id: Option<&'a str>,
    pub tp_order_id: Option<&'a str>,
    pub close_order_ids: &'a [&'a str],
}

/// Сериализует строку и ставит её в очередь trade-CSV (если канал открыт).
pub fn write_trade_csv_row(row: TradeCsvRow<'_>) {
    let regime: &'static str = CURRENT_REGIME.lock().map(|g| *g).unwrap_or("");
    let final_outcome = final_outcome_from_exit_reason(row.exit_reason);
    crate::tee_log::trade_csv_log_write(&format_trade_csv_row(regime, row, final_outcome));
}

fn final_outcome_from_exit_reason(exit_reason: &str) -> &'static str {
    match exit_reason {
        "ResolutionWin" => "win",
        "ResolutionLoss" => "loss",
        _ => "unknown",
    }
}

fn format_trade_csv_row(regime: &str, row: TradeCsvRow<'_>, final_outcome: &str) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        regime,
        csv_escape(row.polymarket_url),
        row.price_to_beat.map(fmt_f64).unwrap_or_default(),
        row.final_price.map(fmt_f64).unwrap_or_default(),
        csv_escape(row.currency),
        csv_escape(row.interval),
        csv_escape(row.side),
        csv_escape(row.market_id),
        csv_escape(row.asset_id),
        row.exit_reason,
        fmt_f64(row.buy_price),
        fmt_f32(row.raw_pred),
        fmt_f32(row.cal_pred),
        fmt_f64(row.kelly_f),
        fmt_f64(row.position_size),
        fmt_f64(row.shares_held),
        fmt_f64(row.exit_price),
        fmt_f64(row.fee_usdc),
        fmt_f64(row.pnl),
        row.frames_held,
        row.p_win_ema_at_close.map(fmt_f64).unwrap_or_default(),
        row.event_remaining_ms_at_open,
        row.event_remaining_ms_at_close,
        row.open_unix_ms.map(|v| v.to_string()).unwrap_or_default(),
        row.close_unix_ms.map(|v| v.to_string()).unwrap_or_default(),
        csv_escape(row.graph_html_file_uri),
        csv_escape(row.pnl_top5_shap),
        csv_escape(final_outcome),
    )
}

// ---------------------------------------------------------------------------
// Submit-orders CSV
// ---------------------------------------------------------------------------

const SUBMIT_TRADE_CSV_HEADER: &str = "regime,pos_id,polymarket_url,price_to_beat,final_price,\
currency,interval,side,market_id,asset_id,exit_reason,finalized_via,\
planned_buy_price,buy_price,planned_shares_held,shares_held,planned_entry_cost,entry_cost,\
planned_fee_usdc,entry_fee_usdc,exit_price,fee_usdc,pnl,\
open_order_id,tp_order_id,close_order_ids,\
raw_pred,cal_pred,kelly_f,p_win_ema_at_close,frames_held,\
event_remaining_ms_at_open,event_remaining_ms_at_close,open_unix_ms,close_unix_ms,\
graph_html_file_uri,pnl_top5_shap";

pub fn init_submit_trade_csv_log_file(path: &Path) -> std::io::Result<()> {
    crate::tee_log::init_submit_trade_csv_log_file(path)?;
    crate::tee_log::submit_trade_csv_log_write(SUBMIT_TRADE_CSV_HEADER);
    Ok(())
}

pub fn finish_submit_trade_csv_log() {
    crate::tee_log::finish_submit_trade_csv_log();
}

pub fn write_submit_trade_csv_row(row: TradeCsvRow<'_>) {
    let regime: &'static str = CURRENT_REGIME.lock().map(|g| *g).unwrap_or("");
    let line = format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        regime,
        csv_escape(row.pos_id),
        csv_escape(row.polymarket_url),
        row.price_to_beat.map(fmt_f64).unwrap_or_default(),
        row.final_price.map(fmt_f64).unwrap_or_default(),
        csv_escape(row.currency),
        csv_escape(row.interval),
        csv_escape(row.side),
        csv_escape(row.market_id),
        csv_escape(row.asset_id),
        row.exit_reason,
        row.finalized_via,
        row.planned_buy_price.map(fmt_f64).unwrap_or_default(),
        fmt_f64(row.buy_price),
        row.planned_shares_held.map(fmt_f64).unwrap_or_default(),
        fmt_f64(row.shares_held),
        row.planned_entry_cost.map(fmt_f64).unwrap_or_default(),
        fmt_f64(row.position_size),
        row.planned_fee_usdc.map(fmt_f64).unwrap_or_default(),
        row.entry_fee_usdc.map(fmt_f64).unwrap_or_default(),
        fmt_f64(row.exit_price),
        fmt_f64(row.fee_usdc),
        fmt_f64(row.pnl),
        row.open_order_id.map(csv_escape).unwrap_or_default(),
        row.tp_order_id.map(csv_escape).unwrap_or_default(),
        if row.close_order_ids.is_empty() {
            String::new()
        } else {
            csv_escape(&row.close_order_ids.join("\n"))
        },
        fmt_f32(row.raw_pred),
        fmt_f32(row.cal_pred),
        fmt_f64(row.kelly_f),
        row.p_win_ema_at_close.map(fmt_f64).unwrap_or_default(),
        row.frames_held,
        row.event_remaining_ms_at_open,
        row.event_remaining_ms_at_close,
        row.open_unix_ms.map(|v| v.to_string()).unwrap_or_default(),
        row.close_unix_ms.map(|v| v.to_string()).unwrap_or_default(),
        csv_escape(row.graph_html_file_uri),
        csv_escape(row.pnl_top5_shap),
    );
    crate::tee_log::submit_trade_csv_log_write(&line);
}

fn fmt_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        String::new()
    }
}

fn fmt_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        String::new()
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}
