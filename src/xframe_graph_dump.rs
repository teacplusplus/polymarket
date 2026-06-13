//! Статический HTML с графиком UP/DOWN по [`crate::xframe::XFrame`] в `graph/...` (рядом с деревом `xframes/`).

use crate::constants::XFrameIntervalKind;
use crate::history_sim::{
    NO_KELLY_POSITION_SIZE_USD, OpenPosition, SIM_MAX_SLIPPAGE_FROM_L1_PCT, book_fill_buy,
    book_fill_sell,
};
use crate::project_manager::{FRAME_BUILD_INTERVALS_SEC, ProjectManager, XFRAMES_LANE_1S};
use crate::util::sanitized_filename_from_gamma_question;
use crate::xframe::{CurrencyUpDownOutcome, SIZE, XFrame};
use crate::xframe_dump::MarketXFramesDump;
use anyhow::Context as _;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Единственный `.bin` для ссылки на график в CSV: явный путь из history-симуляции или синтетический по Gamma stem ([`crate::xframe_dump::synthetic_xframes_dump_bin_path_for_csv_link`]).
pub(crate) fn graph_dump_bin_path_for_trade_csv_uri(pos: &OpenPosition) -> Option<PathBuf> {
    if !pos.graph_dump_bin_path.is_empty() {
        return Some(PathBuf::from(&pos.graph_dump_bin_path));
    }
    let ik = XFrameIntervalKind::from_i32(pos.xframe_interval_type_at_open)?;
    let gq = pos.gamma_question_at_open.as_deref()?;
    let stem = sanitized_filename_from_gamma_question(Some(gq));
    crate::xframe_dump::synthetic_xframes_dump_bin_path_for_csv_link(
        &pos.currency,
        ik,
        &stem,
        pos.event_end_ms,
    )
}

/// Результат [`try_write_graph_html_from_bin_dump`].
#[must_use]
pub enum GraphHtmlFromBinOutcome {
    /// Имя файла не `...__{ts}.bin` в пределах окна или иной сдвиг — как в [`crate::history_sim::window_bounds_from_dump_path`].
    SkippedNoWindowBounds,
    /// Нет стабильных кадрoв Up/Down.
    SkippedNoStableFrames,
    /// HTML записан по пути, зеркальному к `.bin` (корень `graph/` вместо `xframes/`).
    Wrote(PathBuf),
}

/// Одна точка графика: сжатые ключи JSON для встраивания в HTML (см. [`XFrame`]).
#[derive(Debug, Clone, Serialize)]
struct GraphHtmlRow {
    /// Время кадра: `aligned_ts` из буфера кадров (мс, Unix).
    t: i64,
    /// Сколько мс осталось до конца окна рынка: [`XFrame::event_remaining_ms`].
    er: i64,
    /// Рыночная оценка вероятности исхода токена: [`XFrame::currency_implied_prob`]; в JSON `0`, если в кадре `None`.
    ip: f64,
    /// VWAP покупки по стакану (ask-walk) на номинал [`crate::history_sim::NO_KELLY_POSITION_SIZE_USD`] USDC
    /// с cap к L1 [`crate::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT`]; `null` (разрыв линии в Plotly), если
    /// `book_fill_buy(...)` вернул `None` — недостаточно depth'a, либо VWAP пробил cap.
    bb: Option<f64>,
    /// VWAP продажи того же числа шеров (bid-walk) с тем же cap к best bid; `null` (разрыв), если bid-walk
    /// не дал заполниться или пробил cap. Считается **независимо** от cap-результата `bb`: размер шеров
    /// берётся из uncapped buy-walk'а ([`book_fill_buy`] без slip-cap'а), поэтому sell-метрика остаётся
    /// видимой даже когда buy-VWAP пробил cap.
    bs: Option<f64>,
    /// VWAP продажи того же числа шеров без cap (полный bid-walk); `null` (разрыв), если bid-walk не дал
    /// заполниться. Тот же decoupled `shares_estimate`, что и у [`Self::bs`].
    bu: Option<f64>,
    /// `(price_to_beat - spot) / price_to_beat * 100` (%): [`XFrame::currency_price_vs_beat_pct`]; `0`, если `None`.
    vb: f64,
    /// Z-score спота в окне: [`XFrame::currency_price_z_score`]; `0`, если `None`.
    zs: f64,
}

#[derive(Debug, Serialize)]
struct GraphHtmlPayload {
    /// Цена открытия окна — как в [`crate::xframe_dump::MarketXFramesDump::price_to_beat`].
    price_to_beat: f64,
    /// Цена закрытия / следующего открытия — как в [`crate::xframe_dump::MarketXFramesDump::final_price`].
    final_price: f64,
    /// Левый край окна рынка (Unix мс): из имени `.bin` ([`crate::history_sim::window_bounds_from_dump_path`])
    /// или из `median(t+er) − interval` для lane-графика ([`lane_window_start_ms_from_rows`]).
    window_start_ms: i64,
    /// Длительность окна в секундах (300 для 5m, 900 для 15m).
    window_duration_sec: i64,
    up: Vec<GraphHtmlRow>,
    down: Vec<GraphHtmlRow>,
}

const MARKET_GRAPH_HTML_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>XFrame market graph</title>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.3/plotly.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 12px; background: #fafafa; }
    #chart { width: 100%; height: min(85vh, 900px); background: #fff; }
    label { margin-right: 8px; }
    .hint { color: #444; font-size: 13px; max-width: 900px; margin: 8px 0 12px; }
  </style>
</head>
<body>
  <!-- market_id=__MARKET_ID__ -->
  <div id="resolutionMeta" class="hint"></div>
  <div class="hint">
    В JSON время <code>t</code> — Unix-мс, <code>er</code> — мс до конца окна. Начало окна на оси X: <code>median(t + er) − window_duration</code> по точкам с <code>er ≥ 0</code> (как у восстановления времени из <code>.bin</code>); затем секунды <code>(t − origin) / 1000</code> ограничиваются 0…window_duration_sec (5m → 300 с). Поле <code>window_start_ms</code> в JSON — то же начало окна.
    Ось Y — выбранное поле кадра, в т.ч. <code>currency_price_vs_beat_pct</code>, <code>currency_price_z_score</code> (нет значения → <code>0</code>).
    Для sim book VWAP'ов (<code>sim_book_buy_vwap</code>, <code>sim_book_sell_vwap_slip</code>, <code>sim_book_sell_vwap</code>) недостающие значения отображаются <strong>разрывами линии</strong> (Plotly <code>connectgaps:false</code>), а не нулями. Sell-метрики считаются на размере шеров из uncapped buy-walk и видны независимо от того, пробил ли buy slip-cap.
    Параметры URL: <code>y</code> — метрика по вертикали; устаревший <code>x</code> (≠ <code>time</code>) = как <code>y</code>.
    <code>side</code> — <code>up</code> или <code>down</code>.
    <code>ts1</code> — одна вертикальная линия открытия; <code>ts2</code> может повторяться (одна линия на каждое закрытие). **Unix-мс**, как <code>t</code> в данных; переводятся в секунды от начала окна.
    Клик по легенде — скрыть/показать ряд (Plotly).
  </div>
  <div>
    <label for="yField">Ось Y</label>
    <select id="yField"></select>
  </div>
  <div id="chart"></div>
  <script>
  const DATA = __PAYLOAD__;
  (function () {
    const el = document.getElementById('resolutionMeta');
    if (!el) return;
    const ptb = DATA.price_to_beat;
    const fp = DATA.final_price;
    const upWon = fp >= ptb;
    el.textContent =
      'price_to_beat=' + ptb +
      '  final_price=' + fp +
      '  up_won=' + upWon +
      ' (как в MarketXFramesDump / resolve)';
  })();
  const Y_METRICS = [
    { id: 'currency_implied_prob', label: 'currency_implied_prob', key: 'ip' },
    { id: 'currency_price_vs_beat_pct', label: 'currency_price_vs_beat_pct (%)', key: 'vb' },
    { id: 'currency_price_z_score', label: 'currency_price_z_score', key: 'zs' },
    { id: 'sim_book_buy_vwap', label: 'sim book buy VWAP ($30, slip cap)', key: 'bb' },
    { id: 'sim_book_sell_vwap_slip', label: 'sim book sell VWAP (same shares, slip cap)', key: 'bs' },
    { id: 'sim_book_sell_vwap', label: 'sim book sell VWAP (same shares, no slip cap)', key: 'bu' },
  ];
  const params = new URLSearchParams(window.location.search);
  function readSide() {
    const s = (params.get('side') || '').trim().toLowerCase();
    if (s === 'up' || s === 'down') return s;
    return null;
  }
  function readYMetricId() {
    const yRaw = (params.get('y') || '').trim().toLowerCase();
    if (yRaw) {
      const m = Y_METRICS.find(o => o.id === yRaw);
      if (m) return m.id;
    }
    const xLegacy = (params.get('x') || '').trim().toLowerCase();
    if (xLegacy && xLegacy !== 'time') {
      const m = Y_METRICS.find(o => o.id === xLegacy);
      if (m) return m.id;
    }
    return 'currency_implied_prob';
  }
  function readTs(name) {
    if (!params.has(name)) return null;
    const raw = params.get(name);
    if (raw === null || raw === '') return null;
    const v = Number(raw);
    return Number.isFinite(v) ? v : null;
  }
  function readTsAll(name) {
    if (!params.has(name)) return [];
    const out = [];
    for (const raw of params.getAll(name)) {
      if (raw === null || raw === '') continue;
      const v = Number(raw);
      if (Number.isFinite(v)) out.push(v);
    }
    return out;
  }
  const sel = document.getElementById('yField');
  for (const m of Y_METRICS) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    sel.appendChild(opt);
  }
  sel.value = readYMetricId();
  if (!Y_METRICS.some(o => o.id === sel.value)) sel.value = 'currency_implied_prob';
  sel.addEventListener('change', () => {
    params.set('y', sel.value);
    params.delete('x');
    history.replaceState(null, '', '?' + params.toString());
    render();
  });
  function windowDurationSec() {
    return Number.isFinite(DATA.window_duration_sec) ? DATA.window_duration_sec : 300;
  }
  /** Медиана конца окна по кадрам (er≥0): совпадает с логикой lane/.bin в Rust. */
  function inferredWindowStartMsFromEr() {
    const durMs = windowDurationSec() * 1000;
    const ends = [];
    for (const r of DATA.up) {
      if (typeof r.er === 'number' && r.er >= 0 && Number.isFinite(r.t)) ends.push(r.t + r.er);
    }
    for (const r of DATA.down) {
      if (typeof r.er === 'number' && r.er >= 0 && Number.isFinite(r.t)) ends.push(r.t + r.er);
    }
    if (!ends.length) return null;
    ends.sort((a, b) => a - b);
    const mid = ends[Math.floor(ends.length / 2)];
    return mid - durMs;
  }
  function dataMinTsMs() {
    let m = Infinity;
    for (const r of DATA.up) m = Math.min(m, r.t);
    for (const r of DATA.down) m = Math.min(m, r.t);
    return m === Infinity ? 0 : m;
  }
  function windowStartMs() {
    const fromEr = inferredWindowStartMsFromEr();
    if (fromEr !== null && Number.isFinite(fromEr)) return fromEr;
    const w = DATA.window_start_ms;
    if (w != null && Number.isFinite(w)) return w;
    return dataMinTsMs();
  }
  /** Секунды внутри окна рынка [0 … window_duration_sec]. */
  function msFromWindowStart(msWall) {
    const dur = windowDurationSec();
    const s = (msWall - windowStartMs()) / 1000;
    if (!Number.isFinite(s)) return 0;
    return Math.min(Math.max(s, 0), dur);
  }
  function rowY(row, key) {
    const v = row[key];
    return typeof v === 'number' && Number.isFinite(v) ? v : null;
  }
  function buildTraces(yModeDef) {
    const yKey = yModeDef.key;
    const side = readSide();
    const traces = [];
    if (side !== 'down' && DATA.up.length) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'UP',
        x: DATA.up.map(r => msFromWindowStart(r.t)),
        y: DATA.up.map(r => rowY(r, yKey)),
        line: { width: 2 },
        connectgaps: false,
      });
    }
    if (side !== 'up' && DATA.down.length) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'DOWN',
        x: DATA.down.map(r => msFromWindowStart(r.t)),
        y: DATA.down.map(r => rowY(r, yKey)),
        line: { width: 2 },
        connectgaps: false,
      });
    }
    return traces;
  }
  function render() {
    const yModeDef = Y_METRICS.find(o => o.id === sel.value) || Y_METRICS[0];
    const traces = buildTraces(yModeDef);
    const ts1 = readTs('ts1');
    const ts2List = readTsAll('ts2');
    const shapes = [];
    if (ts1 !== null) {
      shapes.push({
        type: 'line',
        x0: msFromWindowStart(ts1), x1: msFromWindowStart(ts1),
        yref: 'paper', y0: 0, y1: 1,
        line: { color: 'rgba(220,20,60,0.75)', width: 2, dash: 'dash' },
      });
    }
    for (const ts of ts2List) {
      shapes.push({
        type: 'line',
        x0: msFromWindowStart(ts), x1: msFromWindowStart(ts),
        yref: 'paper', y0: 0, y1: 1,
        line: { color: 'rgba(25,25,112,0.75)', width: 2, dash: 'dot' },
      });
    }
    const durSec = windowDurationSec();
    const layout = {
      title: 'UP / DOWN vs time — Y: ' + yModeDef.label,
      xaxis: {
        title: 'time in window (s)',
        range: [0, Math.max(durSec, 1)],
        tickformat: '.0f',
        exponentformat: 'none',
        showexponent: 'none',
        separatethousands: false,
      },
      yaxis: { title: yModeDef.label },
      hovermode: 'closest',
      shapes,
      legend: { orientation: 'h' },
      margin: { t: 48, r: 24, b: 56, l: 72 },
    };
    Plotly.react('chart', traces, layout, { responsive: true, displaylogo: false });
  }
  render();
  </script>
</body>
</html>
"#;

#[inline]
fn graph_html_f64_or_zero(o: Option<f64>) -> f64 {
    o.filter(|x| x.is_finite()).unwrap_or(0.0)
}

/// Три VWAP для графика HTML.
///
/// * `buy_vwap`: покупка на [`crate::history_sim::NO_KELLY_POSITION_SIZE_USD`] USDC через
///   [`crate::history_sim::book_fill_buy`] с cap к L1 ([`crate::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT`]).
///   `None` — недостаточно depth'a или VWAP пробил cap.
/// * `sell_slip`, `sell_uncapped`: продажа того же числа шеров через [`crate::history_sim::book_fill_sell`]
///   (с cap'ом и без). Размер шеров — из **uncapped buy-walk'а** (`book_fill_buy(..., None)`), поэтому
///   sell-метрики **развязаны** от того, пробил ли buy slip-cap: даже если `buy_vwap = None` из-за cap'а,
///   sell-стороны видны (и наоборот — если bid-стакан пуст, buy всё равно может остаться).
/// * Все три отдельно `Option<f64>` — на графике каждая ветка обрывается своими пропусками независимо.
fn graph_sim_book_roundtrip_vwaps(
    frame: &XFrame<SIZE>,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    let buy_vwap = book_fill_buy(
        frame,
        NO_KELLY_POSITION_SIZE_USD,
        Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT),
    )
    .map(|(vwap, _shares)| vwap)
    .filter(|v| v.is_finite() && *v > 0.0);

    // Decoupled baseline для sell-стороны: full depth-walk без cap'а. Возвращает Some только если
    // ask-side вообще способна впитать $30 (без оглядки на slip-cap). В этом случае мы знаем число
    // шеров, доступных на $30 USDC, и считаем sell-VWAP для них независимо от cap-исхода `buy_vwap`.
    let shares_estimate = book_fill_buy(frame, NO_KELLY_POSITION_SIZE_USD, None)
        .map(|(_vwap, shares)| shares)
        .filter(|s| s.is_finite() && *s > 0.0);

    let sell_slip = shares_estimate.and_then(|sh| {
        book_fill_sell(frame, sh, Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT))
            .map(|gross_usdc| gross_usdc / sh)
            .filter(|v| v.is_finite() && *v > 0.0)
    });
    let sell_uncapped = shares_estimate.and_then(|sh| {
        book_fill_sell(frame, sh, None)
            .map(|gross_usdc| gross_usdc / sh)
            .filter(|v| v.is_finite() && *v > 0.0)
    });

    (buy_vwap, sell_slip, sell_uncapped)
}

fn graph_html_row(frame: &XFrame<SIZE>, aligned_ts: i64) -> GraphHtmlRow {
    let (bb, bs, bu) = graph_sim_book_roundtrip_vwaps(frame);
    GraphHtmlRow {
        t: aligned_ts,
        er: frame.event_remaining_ms,
        ip: graph_html_f64_or_zero(frame.currency_implied_prob),
        bb,
        bs,
        bu,
        vb: graph_html_f64_or_zero(frame.currency_price_vs_beat_pct),
        zs: graph_html_f64_or_zero(frame.currency_price_z_score),
    }
}

/// Левый край окна Polymarket по точкам графика: `median(t + er) − interval_ms` среди строк с `er ≥ 0`
/// (та же идея, что `event_end_ms` в [`graph_html_rows_from_dump_frames`]).
fn lane_window_start_ms_from_rows<'a>(
    rows: impl Iterator<Item = &'a GraphHtmlRow>,
    interval_ms: i64,
) -> Option<i64> {
    let mut ends: Vec<i64> = rows
        .filter(|r| r.er >= 0)
        .map(|r| r.t.saturating_add(r.er))
        .collect();
    if ends.is_empty() {
        return None;
    }
    ends.sort_unstable();
    let median_end = ends[ends.len() / 2];
    Some(median_end.saturating_sub(interval_ms))
}

/// Время на оси X для кадра из `.bin`: `event_end_ms - event_remaining_ms`
/// (та же логика wall-time, что при пересчёте полей от `price_to_beat`).
fn graph_html_rows_from_dump_frames(
    frames: &[XFrame<SIZE>],
    event_end_ms: i64,
) -> Vec<GraphHtmlRow> {
    frames
        .iter()
        .filter(|f| f.stable)
        .map(|frame| {
            let er = frame.event_remaining_ms;
            let t = if er < 0 {
                event_end_ms
            } else {
                event_end_ms.saturating_sub(er)
            };
            graph_html_row(frame, t)
        })
        .collect()
}

/// Заменяет первый сегмент пути `xframes` на `graph` и расширение на `.html`.
pub fn graph_html_path_from_bin(bin_path: &Path) -> Option<PathBuf> {
    crate::path_config::graph_html_path_from_xframes_bin(bin_path)
}

/// `file:///.../graph/...html?ts1=<open_ms>&ts2=<close_ms>[&ts2=<close_ms>...][&side=up|down]`
/// для колонки CSV; пустая строка, если путь не под `xframes/` или не удалось собрать
/// абсолютный URI. `ts_*` — Unix **миллисекунды**, как на графике.
///
/// * `ts_open_ms` — **одна** точка открытия (`ts1`, красный пунктир).
/// * `ts_close_ms` — **массив** точек закрытия (`ts2`, синий пунктир): один и тот же
///   параметр повторяется в query-string по числу точек; в HTML читается через
///   `URLSearchParams.getAll('ts2')`. Невалидные/нефинитные значения пропускаются.
///
/// `trade_side` — `Some("up")` / `Some("down")`: на странице остаётся только выбранный
/// токен ([`MARKET_GRAPH_HTML_TEMPLATE`] → `side`).
pub fn graph_html_trade_file_uri(
    bin_dump_path: &Path,
    ts_open_ms: Option<i64>,
    ts_close_ms: &[i64],
    trade_side: Option<&str>,
) -> String {
    let Some(html_rel) = graph_html_path_from_bin(bin_dump_path) else {
        return String::new();
    };
    let Ok(cwd) = std::env::current_dir() else {
        return String::new();
    };
    let abs = cwd.join(html_rel);
    let Some(path_str) = abs.to_str() else {
        return String::new();
    };
    let mut uri = crate::util::encode_path_as_file_uri(path_str);
    let mut qs: Vec<String> = Vec::new();
    if let Some(a) = ts_open_ms {
        qs.push(format!("ts1={a}"));
    }
    for b in ts_close_ms {
        qs.push(format!("ts2={b}"));
    }
    if let Some(raw) = trade_side {
        let s = raw.trim().to_lowercase();
        if s == "up" || s == "down" {
            qs.push(format!("side={s}"));
        }
    }
    if !qs.is_empty() {
        use std::fmt::Write as _;
        let _ = write!(&mut uri, "?{}", qs.join("&"));
    }
    uri
}

fn render_graph_html(market_id: &str, payload: &GraphHtmlPayload) -> anyhow::Result<String> {
    let json = serde_json::to_string(payload)?;
    Ok(MARKET_GRAPH_HTML_TEMPLATE
        .replace("__PAYLOAD__", &json)
        .replace("__MARKET_ID__", market_id))
}

/// Строит HTML из уже загруженного [`MarketXFramesDump`] и пишет файл рядом с `.bin` в дереве `graph/`.
///
/// Ось времени точки восстанавливается из имени дампа и [`XFrame::event_remaining_ms`], т.к. в бинарнике нет `aligned_ts`.
pub fn try_write_graph_html_from_bin_dump(
    bin_path: &Path,
    dump: &MarketXFramesDump,
    interval_kind: XFrameIntervalKind,
) -> anyhow::Result<GraphHtmlFromBinOutcome> {
    let Some(bounds) = crate::history_sim::window_bounds_from_dump_path(bin_path, interval_kind)
    else {
        return Ok(GraphHtmlFromBinOutcome::SkippedNoWindowBounds);
    };
    let event_end_ms = bounds.event_end_ms;
    let up = graph_html_rows_from_dump_frames(&dump.frames_up, event_end_ms);
    let down = graph_html_rows_from_dump_frames(&dump.frames_down, event_end_ms);
    if up.is_empty() && down.is_empty() {
        return Ok(GraphHtmlFromBinOutcome::SkippedNoStableFrames);
    }
    let Some(out) = graph_html_path_from_bin(bin_path) else {
        anyhow::bail!(
            "graph HTML: путь не находится в дереве xframes: {}",
            bin_path.display()
        );
    };
    let market_id = dump
        .frames_up
        .first()
        .or_else(|| dump.frames_down.first())
        .map(|f| f.market_id.as_str())
        .unwrap_or("unknown");
    let window_start_ms = bounds.window_start_sec.saturating_mul(1000);
    let window_duration_sec = interval_kind.interval_ms() / 1000;
    let payload = GraphHtmlPayload {
        price_to_beat: dump.price_to_beat,
        final_price: dump.final_price,
        window_start_ms,
        window_duration_sec,
        up,
        down,
    };
    let html = render_graph_html(market_id, &payload)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create_dir_all {}", parent.display()))?;
    }
    std::fs::write(&out, html.as_bytes()).with_context(|| format!("write {}", out.display()))?;
    Ok(GraphHtmlFromBinOutcome::Wrote(out))
}

pub async fn dump_market_graph_html_lane(
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

    let mut up: Vec<GraphHtmlRow> = Vec::new();
    let mut down: Vec<GraphHtmlRow> = Vec::new();
    for (_, aligned_ts, frame) in flat {
        if !frame.stable {
            continue;
        }
        let row = graph_html_row(&frame, aligned_ts);
        match CurrencyUpDownOutcome::from_i32(frame.currency_up_down_outcome) {
            Some(CurrencyUpDownOutcome::Up) => up.push(row),
            Some(CurrencyUpDownOutcome::Down) => down.push(row),
            None => {}
        }
    }

    if up.is_empty() && down.is_empty() {
        return Ok(());
    }

    let interval_label = match interval_kind {
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    };
    let step_secs = FRAME_BUILD_INTERVALS_SEC[lane];
    let schema_size = crate::xframe::xframe_bincode_schema_size_bytes();
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let base: PathBuf = crate::path_config::graph_root()
        .join(project_manager.currency.as_str())
        .join(format!("{schema_size}"))
        .join(interval_label)
        .join(format!("{step_secs}s"))
        .join(&date);
    tokio::fs::create_dir_all(&base).await?;

    let stem = sanitized_filename_from_gamma_question(gamma_question.as_deref());
    let fname = format!("{stem}__{event_end_ms}.html");
    let path = base.join(&fname);

    let interval_ms = interval_kind.interval_ms();
    let window_start_ms = lane_window_start_ms_from_rows(up.iter().chain(down.iter()), interval_ms)
        .unwrap_or_else(|| up.iter().chain(down.iter()).map(|r| r.t).min().unwrap_or(0));

    let we = window_start_ms.saturating_add(interval_ms);
    let up_f: Vec<GraphHtmlRow> = up
        .iter()
        .filter(|r| r.t >= window_start_ms && r.t <= we)
        .cloned()
        .collect();
    let down_f: Vec<GraphHtmlRow> = down
        .iter()
        .filter(|r| r.t >= window_start_ms && r.t <= we)
        .cloned()
        .collect();
    let (up, down) = if up_f.is_empty() && down_f.is_empty() {
        (up, down)
    } else {
        (up_f, down_f)
    };
    let window_duration_sec = interval_kind.interval_ms() / 1000;

    let payload = GraphHtmlPayload {
        price_to_beat,
        final_price,
        window_start_ms,
        window_duration_sec,
        up,
        down,
    };
    let html = render_graph_html(&market_id, &payload)?;
    tokio::fs::write(&path, html.as_bytes()).await?;
    Ok(())
}

/// Фоновый партиальный HTML по кадрам PM → `graph_html_path_from_bin(pos.graph_dump_bin_path)` (ссылка из CSV).
/// Только real_sim при `spawn_partial_graph_html_on_close`; `final_price` — последний спот из `rtds_currency_prices_by_sec`.
pub fn spawn_partial_market_graph_html_for_close(
    project_manager: Arc<ProjectManager>,
    pos: &OpenPosition,
) {
    let market_id = pos.market_id.clone();
    let bin_path_str = pos.graph_dump_bin_path.clone();
    let price_to_beat = pos.price_to_beat.unwrap_or(0.0);
    let interval_kind_i32 = pos.xframe_interval_type_at_open;

    tokio::spawn(async move {
        if bin_path_str.is_empty() {
            return;
        }
        let Some(interval_kind) = XFrameIntervalKind::from_i32(interval_kind_i32) else {
            return;
        };
        let bin_path = PathBuf::from(&bin_path_str);
        let Some(target_html_path) = graph_html_path_from_bin(&bin_path) else {
            return;
        };

        let pm = project_manager;

        // final_price = «цена последнего xframe»: последний секундный
        // спот валюты, который PM кладёт в `rtds_currency_prices_by_sec`
        // (там же, откуда берётся `currency_spot_usd` в
        // `build_frames_from_buffer_lane_once`).
        let final_price = pm
            .rtds_currency_prices_by_sec
            .read()
            .await
            .iter()
            .next_back()
            .map(|(_, p)| *p)
            .unwrap_or(0.0);

        if pm.xframes_by_market.is_empty() {
            return;
        }
        let by_asset = {
            let lock = pm.xframes_by_market[XFRAMES_LANE_1S].read().await;
            lock.get(&market_id).cloned()
        };
        let Some(by_asset) = by_asset else {
            return;
        };

        let mut flat: Vec<(i64, XFrame<SIZE>)> = Vec::new();
        for (_asset_id, by_ts) in by_asset.iter() {
            for (aligned_ts, frame) in by_ts.iter() {
                flat.push((*aligned_ts, frame.clone()));
            }
        }
        flat.sort_by_key(|(aligned_ts, _)| *aligned_ts);

        let mut up: Vec<GraphHtmlRow> = Vec::new();
        let mut down: Vec<GraphHtmlRow> = Vec::new();
        for (aligned_ts, frame) in flat {
            if !frame.stable {
                continue;
            }
            let row = graph_html_row(&frame, aligned_ts);
            match CurrencyUpDownOutcome::from_i32(frame.currency_up_down_outcome) {
                Some(CurrencyUpDownOutcome::Up) => up.push(row),
                Some(CurrencyUpDownOutcome::Down) => down.push(row),
                None => {}
            }
        }
        if up.is_empty() && down.is_empty() {
            return;
        }

        let interval_ms = interval_kind.interval_ms();
        let window_start_ms = lane_window_start_ms_from_rows(up.iter().chain(down.iter()), interval_ms)
            .unwrap_or_else(|| {
                up.iter()
                    .chain(down.iter())
                    .map(|r| r.t)
                    .min()
                    .unwrap_or(0)
            });
        let window_end_ms = window_start_ms.saturating_add(interval_ms);
        let up_f: Vec<GraphHtmlRow> = up
            .iter()
            .filter(|r| r.t >= window_start_ms && r.t <= window_end_ms)
            .cloned()
            .collect();
        let down_f: Vec<GraphHtmlRow> = down
            .iter()
            .filter(|r| r.t >= window_start_ms && r.t <= window_end_ms)
            .cloned()
            .collect();
        let (up, down) = if up_f.is_empty() && down_f.is_empty() {
            (up, down)
        } else {
            (up_f, down_f)
        };
        let window_duration_sec = interval_ms / 1000;

        let payload = GraphHtmlPayload {
            price_to_beat,
            final_price,
            window_start_ms,
            window_duration_sec,
            up,
            down,
        };
        let html = match render_graph_html(&market_id, &payload) {
            Ok(s) => s,
            Err(err) => {
                crate::tee_eprintln!(
                    "[partial_graph_html] {market_id}: render failed: {err:#}"
                );
                return;
            }
        };
        if let Some(parent) = target_html_path.parent() {
            if let Err(err) = tokio::fs::create_dir_all(parent).await {
                crate::tee_eprintln!(
                    "[partial_graph_html] {market_id}: create_dir_all {} failed: {err:#}",
                    parent.display()
                );
                return;
            }
        }
        if let Err(err) = tokio::fs::write(&target_html_path, html.as_bytes()).await {
            crate::tee_eprintln!(
                "[partial_graph_html] {market_id}: write {} failed: {err:#}",
                target_html_path.display()
            );
        }
    });
}
