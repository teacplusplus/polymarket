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
//! # Lifecycle и буферизация
//!
//! Файл инициализируется один раз на процесс через
//! [`init_trade_csv_log_file`] (обычно — рядом с `init_tee_log_file` в
//! `run_sim_mode`); первая строка — CSV-заголовок. После завершения
//! симуляции [`finish_trade_csv_log`] флашит и закрывает писатель.
//!
//! [`write_trade_csv_row`] **не пишет на диск немедленно**: строка
//! помещается в in-memory буфер до тех пор, пока не вызовут
//! [`record_market_outcome`] для соответствующего `market_id`. На этом
//! шаге все буферизованные строки этого маркета (как рыночные, так и
//! резолюционные) выписываются на диск с заполненной колонкой
//! `final_outcome` (`win`/`loss`). Так гарантируем: финальный исход
//! маркета доступен на каждой строке трейда без external-join'ов.
//!
//! Если файл не инициализирован, [`write_trade_csv_row`] — no-op (та же
//! идея, что и у `tee_*` макросов: безопасно вызывать из путей, где
//! CSV-логи не нужны, например, из ранних crash'ев). При вызове
//! [`finish_trade_csv_log`] оставшиеся в буфере строки (для маркетов,
//! не дождавшихся резолюции — например, truncated dump в history_sim)
//! записываются с `final_outcome=unknown`.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// Глобальный buffered writer per-trade CSV. `None` — лог не открыт,
/// `write_trade_csv_row` молча пропускает. См. модульный комментарий.
pub static TRADE_CSV_LOG: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// In-memory очередь строк, ждущих финального исхода маркета.
///
/// Каждая строка [`write_trade_csv_row`] кладётся сюда. Когда
/// [`record_market_outcome`] вызывается для `market_id`, все строки
/// этого маркета извлекаются, дополняются `final_outcome` и пишутся
/// в файл. См. модульный комментарий о lifecycle и mode-операторах.
static TRADE_CSV_PENDING: Mutex<Vec<PendingTradeRow>> = Mutex::new(Vec::new());

/// Текущий режим симуляции для CSV-колонки `regime`.
///
/// Используется в `run_sim_mode(is_kelly)` (см. `history_sim.rs`):
/// один и тот же CSV-файл (`xframes/last_history_sim_trades.csv`)
/// заполняется обоими прогонами back-to-back, и `regime` —
/// единственный способ их различить при анализе. Установка идёт
/// через [`set_current_regime`] перед каждым прогоном; запись —
/// в [`write_trade_csv_row`]. Значение по умолчанию (`""`) пишется,
/// если режим не выставлен (например, в `real_sim`).
static CURRENT_REGIME: Mutex<&'static str> = Mutex::new("");

/// Устанавливает текущее значение колонки `regime`. Допустимые
/// значения: `"kelly"` / `"raw"` / `""`. `&'static str` — потому что
/// меняется буквально дважды за прогон и хранить динамические `String`
/// в Mutex смысла нет.
pub fn set_current_regime(regime: &'static str) {
    if let Ok(mut guard) = CURRENT_REGIME.lock() {
        *guard = regime;
    }
}

/// CSV-заголовок: порядок колонок зафиксирован тут и должен совпадать
/// с `write_pending_row_to_file`. Первая колонка `regime` — `kelly` /
/// `raw` / пусто (см. [`set_current_regime`]). Последняя — `final_outcome`
/// (`win` / `loss` / `unknown`), см. [`record_market_outcome`].
///
/// `polymarket_url`, `price_to_beat` и `final_price` идут блоком после
/// `regime` — это **per-market** контекст (одинаков для всех трейдов
/// одного дампа). Удобно для группировки в pandas / визуального чтения.
///
/// `open_unix_ms` / `close_unix_ms` — wall-clock метки входа и выхода
/// (UTC, ms). В history_sim рассчитываются из имени `.bin`-дампа
/// (см. [`crate::history_sim::window_bounds_from_dump_path`]) и
/// `event_remaining_ms_at_*`; в real_sim выходят пустыми.
///
/// `graph_html_file_uri` — `file:///.../graph/...html?ts1=...&ts2=...` для
/// локального просмотра с вертикалями открытия/закрытия (пусто без пути дампа).
///
/// `pnl_top5_shap` — топ-5 SHAP PnL-модели на входе (многострочная ячейка в кавычках).
const TRADE_CSV_HEADER: &str = "regime,polymarket_url,price_to_beat,final_price,\
currency,interval,side,market_id,asset_id,exit_reason,\
buy_price,raw_pred,cal_pred,kelly_f,entry_cost,shares_held,exit_price,fee_usdc,pnl,\
frames_held,p_win_ema_at_close,event_remaining_ms_at_open,event_remaining_ms_at_close,\
open_unix_ms,close_unix_ms,graph_html_file_uri,pnl_top5_shap,final_outcome";

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
    // На повторной инициализации сбрасываем и буфер: остатки от
    // предыдущей сессии не должны утечь в новый файл.
    if let Ok(mut pending) = TRADE_CSV_PENDING.lock() {
        pending.clear();
    }
    Ok(())
}

/// Очищает [`TRADE_CSV_PENDING`] **без** записи в файл.
///
/// Нужно для побочных запусков sim'а, которые делают `write_trade_csv_row`
/// (через `close_position` / `Account::resolve_pending_market_sync`), но
/// не должны попасть в финальный CSV — например, sim-replay калибровка
/// в [`crate::train_mode::fit_calibration_via_sim_replay`]. После каждого
/// маркета `record_market_outcome` дренирует свои строки из буфера, но
/// если writer не открыт (а в train phase он закрыт), они уходят в drop.
/// Этот метод — defensive sweep на случай маркетов, для которых
/// `record_market_outcome` не дёрнулся (пустой dump / отсутствующий
/// `market_id` / panic'нувший воркер): чистим буфер до того, как
/// `init_trade_csv_log_file` следующей фазы откроет writer и любая
/// нечаянная запись попадёт в финальный CSV.
pub fn clear_pending_buffer() {
    if let Ok(mut pending) = TRADE_CSV_PENDING.lock() {
        pending.clear();
    }
}

/// Флашит и закрывает писатель в [`TRADE_CSV_LOG`], если он был открыт.
/// Симметрично `tee_log::finish_tee_log` — для контролируемого закрытия
/// в финале однократного режима.
///
/// Перед закрытием файла оставшиеся в буфере строки (для маркетов, по
/// которым не пришёл [`record_market_outcome`] — например, в history_sim
/// дамп оборвался до резолюции) выписываются с `final_outcome=unknown`.
pub fn finish_trade_csv_log() {
    if let Ok(mut pending) = TRADE_CSV_PENDING.lock() {
        if let Ok(mut guard) = TRADE_CSV_LOG.lock() {
            if let Some(w) = guard.as_mut() {
                for row in pending.drain(..) {
                    write_pending_row_to_file(w, &row, "unknown");
                }
                let _ = w.flush();
            }
        }
    }
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
    /// Polymarket-URL события (`https://polymarket.com/event/<slug>`),
    /// см. [`crate::history_sim::OpenPosition::polymarket_url`]. Пустая
    /// строка — URL не известен (real_sim / распарсить имя дампа не
    /// удалось); в этом случае колонка пишется пустой.
    pub polymarket_url: &'a str,
    /// `priceToBeat` маркета (см. [`crate::history_sim::OpenPosition::price_to_beat`]).
    /// `None` пишется пустой ячейкой.
    pub price_to_beat: Option<f64>,
    /// `finalPrice` маркета (см. [`crate::history_sim::OpenPosition::final_price`]).
    /// `None` пишется пустой ячейкой.
    pub final_price: Option<f64>,
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
    /// Фактический VWAP заполнения buy-ордера (`book_fill_buy` /
    /// `book_fill_buy_strict`) — реальная цена, по которой купили шеры.
    /// Берётся из [`crate::history_sim::OpenPosition::buy_price`]. Это
    /// «честная» цена входа для оценки сделки: при широком спреде
    /// `currency_implied_prob` (mid/last_trade) может далеко расходиться
    /// с фактическим ask, по которому проходил fill.
    pub buy_price: f64,
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
    /// Wall-clock UTC ms открытия позиции (см.
    /// [`crate::history_sim::OpenPosition::event_end_ms`]). `None` —
    /// wall-clock не известен (real_sim или имя дампа не распарсилось);
    /// в этом случае колонка пишется пустой.
    pub open_unix_ms: Option<i64>,
    /// Wall-clock UTC ms закрытия позиции. Семантика `None` —
    /// та же, что у [`Self::open_unix_ms`].
    pub close_unix_ms: Option<i64>,
    /// Локальный `file://` URL HTML-графика с `ts1`/`ts2` и опционально `side=up|down` (см. [`crate::xframe_graph_dump::graph_html_trade_file_uri`]).
    /// Пустая строка, если нет привязки к дампу (`real_sim`) или URI не собрать.
    pub graph_html_file_uri: &'a str,
    /// Топ-5 SHAP вкладов PnL-модели в момент открытия (переводы строк `\n`); пусто если расчёт отключён.
    pub pnl_top5_shap: &'a str,
}

/// Owned-копия [`TradeCsvRow`] для буферизации до момента, когда
/// станет известен исход маркета (`record_market_outcome`). Хранит
/// `regime` снапшот на момент записи — это важно для дифференциации
/// kelly/raw прогонов в одном файле.
#[derive(Debug, Clone)]
struct PendingTradeRow {
    regime: &'static str,
    polymarket_url: String,
    price_to_beat: Option<f64>,
    final_price: Option<f64>,
    currency: String,
    interval: String,
    side: String,
    market_id: String,
    asset_id: String,
    exit_reason: &'static str,
    buy_price: f64,
    raw_pred: f32,
    cal_pred: f32,
    kelly_f: f64,
    entry_cost: f64,
    shares_held: f64,
    exit_price: f64,
    fee_usdc: f64,
    pnl: f64,
    frames_held: usize,
    p_win_ema_at_close: Option<f64>,
    event_remaining_ms_at_open: i64,
    event_remaining_ms_at_close: i64,
    open_unix_ms: Option<i64>,
    close_unix_ms: Option<i64>,
    graph_html_file_uri: String,
    pnl_top5_shap: String,
}

/// Кладёт одну строку в in-memory буфер [`TRADE_CSV_PENDING`]. Не пишет
/// на диск — это сделает [`record_market_outcome`] (когда придёт исход
/// маркета) или [`finish_trade_csv_log`] (для маркетов без резолюции).
///
/// Если файл не инициализирован, всё равно кладёт в буфер — это безвредно:
/// `finish_trade_csv_log` (без открытого писателя) очистит буфер без записи.
/// На любых nan/inf числовых значениях позже будет записана пустая ячейка
/// (см. [`fmt_f64`]/[`fmt_f32`]) — анализаторы CSV не любят `NaN`/`Inf`.
pub fn write_trade_csv_row(row: TradeCsvRow<'_>) {
    let regime: &'static str = CURRENT_REGIME
        .lock()
        .map(|g| *g)
        .unwrap_or("");
    let owned = PendingTradeRow {
        regime,
        polymarket_url: row.polymarket_url.to_string(),
        price_to_beat: row.price_to_beat,
        final_price: row.final_price,
        currency: row.currency.to_string(),
        interval: row.interval.to_string(),
        side: row.side.to_string(),
        market_id: row.market_id.to_string(),
        asset_id: row.asset_id.to_string(),
        exit_reason: row.exit_reason,
        buy_price: row.buy_price,
        raw_pred: row.raw_pred,
        cal_pred: row.cal_pred,
        kelly_f: row.kelly_f,
        entry_cost: row.entry_cost,
        shares_held: row.shares_held,
        exit_price: row.exit_price,
        fee_usdc: row.fee_usdc,
        pnl: row.pnl,
        frames_held: row.frames_held,
        p_win_ema_at_close: row.p_win_ema_at_close,
        event_remaining_ms_at_open: row.event_remaining_ms_at_open,
        event_remaining_ms_at_close: row.event_remaining_ms_at_close,
        open_unix_ms: row.open_unix_ms,
        close_unix_ms: row.close_unix_ms,
        graph_html_file_uri: row.graph_html_file_uri.to_string(),
        pnl_top5_shap: row.pnl_top5_shap.to_string(),
    };
    if let Ok(mut pending) = TRADE_CSV_PENDING.lock() {
        pending.push(owned);
    }
}

/// Извлекает из буфера все строки `market_id` и пишет их в файл с
/// заполненной `final_outcome` (зависит от `side` строки и `up_won`):
/// для UP-стороны `up_won → win`, для DOWN-стороны — наоборот.
///
/// Строки, чей `side` не совпадает с `up`/`down` (нештатный лейбл —
/// `unknown`), помечаются `final_outcome=unknown`.
///
/// Вызывается из [`crate::account::Account::resolve_pending_market_sync`]
/// один раз на маркет, после того как все позиции по нему обработаны
/// (как минимум все строки `ResolutionWin` / `ResolutionLoss` уже в
/// буфере). Идемпотентен: повторный вызов с тем же `market_id` ничего
/// не сделает (буфер уже пуст для этого id).
pub fn record_market_outcome(market_id: &str, up_won: bool) {
    let to_flush: Vec<PendingTradeRow> = {
        let Ok(mut pending) = TRADE_CSV_PENDING.lock() else {
            return;
        };
        // Извлекаем строки этого market_id, оставляя остальные в буфере.
        let mut keep: Vec<PendingTradeRow> = Vec::with_capacity(pending.len());
        let mut take: Vec<PendingTradeRow> = Vec::new();
        for row in pending.drain(..) {
            if row.market_id == market_id {
                take.push(row);
            } else {
                keep.push(row);
            }
        }
        *pending = keep;
        take
    };
    if to_flush.is_empty() {
        return;
    }

    let Ok(mut guard) = TRADE_CSV_LOG.lock() else {
        return;
    };
    let Some(w) = guard.as_mut() else {
        return;
    };
    for row in to_flush {
        let outcome = outcome_for_side(&row.side, up_won);
        write_pending_row_to_file(w, &row, outcome);
    }
    let _ = w.flush();
}

/// Вспомогательная: маппинг `side` → `win`/`loss` по флагу `up_won`.
fn outcome_for_side(side: &str, up_won: bool) -> &'static str {
    match side {
        "up" => if up_won { "win" } else { "loss" },
        "down" => if up_won { "loss" } else { "win" },
        _ => "unknown",
    }
}

/// Сериализует одну [`PendingTradeRow`] в CSV-файл. `final_outcome` —
/// внешний параметр, потому что одна и та же буферная строка может
/// быть выписана и с известным исходом ([`record_market_outcome`]),
/// и с `unknown` ([`finish_trade_csv_log`]).
fn write_pending_row_to_file(
    w: &mut BufWriter<File>,
    row: &PendingTradeRow,
    final_outcome: &str,
) {
    let _ = writeln!(
        w,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        row.regime,
        csv_escape(&row.polymarket_url),
        row.price_to_beat.map(fmt_f64).unwrap_or_default(),
        row.final_price.map(fmt_f64).unwrap_or_default(),
        csv_escape(&row.currency),
        csv_escape(&row.interval),
        csv_escape(&row.side),
        csv_escape(&row.market_id),
        csv_escape(&row.asset_id),
        row.exit_reason,
        fmt_f64(row.buy_price),
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
        row.open_unix_ms.map(|v| v.to_string()).unwrap_or_default(),
        row.close_unix_ms.map(|v| v.to_string()).unwrap_or_default(),
        csv_escape(&row.graph_html_file_uri),
        csv_escape(&row.pnl_top5_shap),
        csv_escape(final_outcome),
    );
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
