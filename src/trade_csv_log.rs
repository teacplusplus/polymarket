//! Per-trade CSV-лог симуляции: одна строка на каждую закрытую позицию
//! (рыночное закрытие через [`crate::history_sim::close_position`] **и**
//! резолюционное закрытие через [`crate::account::Account::resolve_pending_market_sync`]).
//!
//! # Зачем отдельный CSV рядом с `last_history_sim.txt`?
//!
//! Текстовый лог `last_history_sim.txt` агрегированный (`SideStats` по
//! сторонам), а понять, какие именно сделки породили `−631$ ROI`, по нему
//! невозможно: `EvExit✗=15` на 5m UP — это 15 сделок, средняя из них на
//! `−12$`, но без `entry_prob`/`raw_pred`/`frames_held` неясно, какая часть
//! pipeline проседает. CSV даёт построчную трассировку, которую можно
//! загрузить в pandas/duckdb для bucket-анализа.
//!
//! # Когда пишется
//!
//! * **Рыночные закрытия** (TP / SL / Timeout / EvExit*): из
//!   [`crate::history_sim::close_position`] — после успешного `book_fill_sell*`,
//!   когда `pnl` уже посчитан и записан в `SideStats`.
//! * **Резолюционные закрытия** (бинарная выплата $1/$0): из
//!   [`crate::account::Account::resolve_pending_market_sync`] — после
//!   фактического обновления `bankroll` и `SideStats`.
//!
//! Оба пути пишут одинаковый набор колонок — это значит, можно брать
//! полный CSV и группировать по `exit_reason` без перекосов.
//!
//! # Lifecycle
//!
//! Файл инициализируется один раз на процесс через
//! [`init_trade_csv_log_file`] (обычно — рядом с `init_tee_log_file` в
//! `run_sim_mode`); первая строка — CSV-заголовок. После завершения
//! симуляции [`finish_trade_csv_log`] флашит и закрывает писатель.
//!
//! Если файл не инициализирован, [`write_trade_csv_row`] — no-op (та же
//! идея, что и у `tee_*` макросов: безопасно вызывать из путей, где
//! CSV-логи не нужны, например, из `real_sim`-режима).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// Глобальный buffered writer per-trade CSV. `None` — лог не открыт,
/// `write_trade_csv_row` молча пропускает. См. модульный комментарий.
pub static TRADE_CSV_LOG: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// CSV-заголовок: порядок колонок зафиксирован тут и должен совпадать
/// с `write_trade_csv_row`.
const TRADE_CSV_HEADER: &str = "currency,interval,side,market_id,asset_id,exit_reason,\
entry_prob,raw_pred,cal_pred,kelly_f,entry_cost,shares_held,exit_price,fee_usdc,pnl,\
frames_held,p_win_ema_at_close,event_remaining_ms_at_open,event_remaining_ms_at_close";

/// Открывает / перезаписывает файл `path` и записывает CSV-заголовок.
/// Идемпотентен в смысле «последний победил»: повторный вызов закроет
/// предыдущий писатель и откроет новый. На практике вызывается один
/// раз на процесс — в начале `run_sim_mode`.
pub fn init_trade_csv_log_file(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", TRADE_CSV_HEADER)?;
    {
        let mut guard = TRADE_CSV_LOG.lock().expect("TRADE_CSV_LOG poisoned");
        *guard = Some(writer);
    }
    Ok(())
}

/// Флашит и закрывает писатель в [`TRADE_CSV_LOG`], если он был открыт.
/// Симметрично `tee_log::finish_tee_log` — для контролируемого закрытия
/// в финале однократного режима.
pub fn finish_trade_csv_log() {
    if let Ok(mut guard) = TRADE_CSV_LOG.lock() {
        if let Some(mut w) = guard.take() {
            let _ = w.flush();
        }
    }
}

/// Все поля CSV-строки одной закрытой сделки. Структура нужна, чтобы
/// caller'ы из разных модулей (`close_position`, `resolve_pending_market_sync`)
/// собирали одинаковый набор колонок без копипасты `format!` и риска
/// разойтись по порядку столбцов.
///
/// Поля без значения (например, `p_win_ema_at_close` для резолюционного
/// закрытия — там EMA не считается) кодируются как пустая строка в CSV
/// (стандартное поведение для NULL).
#[derive(Debug, Clone, Copy)]
pub struct TradeCsvRow<'a> {
    /// Валюта (`btc` / …). Берётся из `lane_key.0` или `frame.asset_id`-mapping.
    pub currency: &'a str,
    /// Лейбл интервала (`5m` / `15m`).
    pub interval: &'a str,
    /// Лейбл стороны (`up` / `down`).
    pub side: &'a str,
    pub market_id: &'a str,
    pub asset_id: &'a str,
    /// `TP` / `SL` / `Timeout` / `EvExitProfit` / `EvExitLoss` /
    /// `ResolutionWin` / `ResolutionLoss`.
    pub exit_reason: &'static str,
    pub entry_prob: f64,
    pub raw_pred: f32,
    pub cal_pred: f32,
    pub kelly_f: f64,
    pub entry_cost: f64,
    pub shares_held: f64,
    /// VWAP продажи (рыночный выход) или `1.0` / `0.0` для резолюции.
    pub exit_price: f64,
    /// Фактически уплаченная taker-fee по продаже. На резолюции `0.0`.
    pub fee_usdc: f64,
    pub pnl: f64,
    pub frames_held: usize,
    /// EMA `p_win` resolution-модели на момент закрытия. `None` для
    /// рыночных выходов вне hold-zone и для резолюционных выходов.
    pub p_win_ema_at_close: Option<f64>,
    pub event_remaining_ms_at_open: i64,
    /// Текущий `event_remaining_ms` (на момент закрытия). `0` если
    /// резолюция уже состоялась.
    pub event_remaining_ms_at_close: i64,
}

/// Пишет одну строку в [`TRADE_CSV_LOG`] (если открыт). На любых
/// nan/inf числовых значениях пишет пустую ячейку — анализаторы CSV
/// не любят `NaN`/`Inf`.
pub fn write_trade_csv_row(row: TradeCsvRow<'_>) {
    let Ok(mut guard) = TRADE_CSV_LOG.lock() else {
        return;
    };
    let Some(w) = guard.as_mut() else {
        return;
    };
    let _ = writeln!(
        w,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        csv_escape(row.currency),
        csv_escape(row.interval),
        csv_escape(row.side),
        csv_escape(row.market_id),
        csv_escape(row.asset_id),
        row.exit_reason,
        fmt_f64(row.entry_prob),
        fmt_f32(row.raw_pred),
        fmt_f32(row.cal_pred),
        fmt_f64(row.kelly_f),
        fmt_f64(row.entry_cost),
        fmt_f64(row.shares_held),
        fmt_f64(row.exit_price),
        fmt_f64(row.fee_usdc),
        fmt_f64(row.pnl),
        row.frames_held,
        row.p_win_ema_at_close.map(fmt_f64).unwrap_or_default(),
        row.event_remaining_ms_at_open,
        row.event_remaining_ms_at_close,
    );
    let _ = w.flush();
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

/// CSV-эскейп: оборачивает значение в двойные кавычки, если внутри
/// есть `,`, `"` или `\n`; внутренние `"` удваиваются. Достаточно для
/// `market_id`/`asset_id` Polymarket (hex) и кратких лейблов.
fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}
