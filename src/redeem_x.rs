//! Redeem-X — реконструкция публичного momentum-maker'а по tail-отчёту
//! (held-to-resolution `PnL = winning_shares·$1 − bought`, совпал с реальным до цента).
//!
//!   * Не redeem-арбитраж: пара UP+DOWN в медиане >$1 → на самих парах он **в минусе**.
//!   * Прибыль — с непарного направленного излишка на «тяжёлой» (лидирующей) ноге;
//!     тяжёлая нога = победитель в ~75% (BTC 5m) / ~90% (BTC 15m, ETH 5m).
//!   * Исполняется пассивным maker'ом **фиксированным клипом по `coin+period`**
//!     (ETH 5m≈5, BTC 5m≈100, BTC 15m≈20 шер). Размер ОДНОГО ордера инвариантен к цене
//!     и ко второй ноге (~91% ордеров — полный клип); плечо набирается числом клипов
//!     лесенкой, а перекос ног возникает из выбора ноги, не из сжатия ордера.
//!
//! Поэтому сайзинг здесь — **полный клип** `coin+period` (см. [`redeem_x_clip_shares`]),
//! гейты цены/времени и асимметричный потолок инвентаря lead/lag (мягкий крен к лидеру).
//!
//! Вход разрешён только если исторический режим за последний час подтверждает винрейт:
//! по дампам стримов того же `coin+interval` восстанавливаем лидирующую ногу и её цену и
//! допускаем вход, если `lead_hold_rate > avg_lead_price` (EV покупки лидера > 0). Итог
//! (`allow: bool`) кэшируется в [`crate::account::Account`] по `(coin, interval)` и грузится
//! лениво фоновым загрузчиком: cache-miss → `None` (кадр пропускаем).

use crate::account::{RedeemXMarketRegimeKey, SharedAccount};
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::data_ws::WsStreamPriceChange;
use crate::history_sim::{
    LanePositions, MAX_BET_FRACTION, MAX_POSITION_USD, MIN_POSITION_USD, StrictBook,
};
use crate::xframe::{SIZE, XFrame};
use crate::xframe_dump::{MarketWsStreamDumpEntry, MarketWsStreamDumpMarket};
use anyhow::Context as _;
use flate2::read::GzDecoder;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tokio::sync::mpsc;

// --- Параметры входа (направленный momentum-maker) ------------------------------------

/// Ценовая полоса ноги (maker встаёт на best_bid): не котируем пыль / уже разрешённый исход.
const REDEEM_X_MIN_PRICE: f64 = 0.02;
const REDEEM_X_MAX_PRICE: f64 = 0.98;
/// Порог implied prob, выше которого нога — лидирующая (фаворит).
const REDEEM_X_LEAD_PROB: f64 = 0.50;
/// Мягкий крен задаётся ТОЛЬКО потолком инвентаря (в клипах): лидеру разрешаем накопить
/// больше клипов, чем отстающей ноге. Размер ОДНОГО ордера — всегда полный клип.
const REDEEM_X_MAX_LEAD_CLIPS: f64 = 80.0;
const REDEEM_X_MAX_LAG_CLIPS: f64 = 60.0;
/// Минимальный остаток времени на входе.
const REDEEM_X_ENTRY_MIN_REMAINING_MS: i64 = 10_000;
/// Минимальный интервал между покупками в одном рынке: не чаще раза в N мс.
const REDEEM_X_MIN_REBUY_INTERVAL_MS: i64 = 3_000;

// --- Параметры исторического режима ---------------------------------------------------

/// Окно «последнего часа» для оценки винрейта momentum-maker'а.
const REDEEM_X_REGIME_WINDOW_MS: i64 = 60 * 60 * 1_000;
/// Минимум разрешённых рынков в окне, чтобы доверять оценке.
const REDEEM_X_REGIME_MIN_MARKETS: usize = 4;
/// Снимки ближе этого остатка до конца игнорируем при оценке цены лидера
/// (там цена уже схлопнута к 0/1 и не отражает цену накопления).
const REDEEM_X_REGIME_SIGNAL_FLOOR_MS: i64 = 30_000;
/// LRU-капасити кэша распарсенных дампов.
const REDEEM_X_DUMP_CACHE_CAP: usize = 8_000;

// --- Публичные типы режима (кэшируются в Account) -------------------------------------

/// Снимок исторического режима для `(coin, interval)`: можно ли заходить.
#[derive(Debug, Clone, Copy, Default)]
pub struct RedeemXMarketRegime {
    /// Конец окна (Unix-мс) — ключ инвалидизации кэша.
    pub event_end_ms: i64,
    /// Формула давала винрейт за последний час (EV покупки лидера > 0).
    pub allow: bool,
}

/// Команда ленивой загрузки режима (как redeem-01 tail).
#[derive(Debug, Clone)]
pub(crate) struct RedeemXMarketRegimeLoadCommand {
    pub account: SharedAccount,
    pub currency: String,
    pub interval: XFrameIntervalKind,
    pub event_end_ms: i64,
}

// --- Async loader / lazy fill ---------------------------------------------------------

pub(crate) async fn run_redeem_x_market_regime_loader(
    mut rx: mpsc::Receiver<RedeemXMarketRegimeLoadCommand>,
) {
    while let Some(command) = rx.recv().await {
        if let Err(err) = load_redeem_x_market_regime_key(command).await {
            crate::tee_eprintln!("[redeem_x] market regime load failed: {err:#}");
        }
    }
}

async fn load_redeem_x_market_regime_key(
    command: RedeemXMarketRegimeLoadCommand,
) -> anyhow::Result<()> {
    let currency = command.currency.to_ascii_lowercase();
    let key: RedeemXMarketRegimeKey = (currency.clone(), command.interval);
    {
        let guard = command.account.redeem_x_market_regime.read().await;
        if guard
            .get(&key)
            .is_some_and(|regime| regime.event_end_ms == command.event_end_ms)
        {
            return Ok(());
        }
    }

    let streams_root = crate::path_config::streams_root();
    let regime = scan_redeem_x_market_regime(
        &streams_root,
        &currency,
        command.interval,
        command.event_end_ms,
    )?;

    let mut guard = command.account.redeem_x_market_regime.write().await;
    if guard
        .get(&key)
        .is_some_and(|existing| existing.event_end_ms != command.event_end_ms)
    {
        guard.remove(&key);
    }
    guard.entry(key).or_insert(regime);
    Ok(())
}

// --- Сканер исторического режима по дампам стримов ------------------------------------

fn scan_redeem_x_market_regime(
    streams_root: &Path,
    currency: &str,
    target_interval: XFrameIntervalKind,
    regime_event_end_ms: i64,
) -> anyhow::Result<RedeemXMarketRegime> {
    let currency_root = streams_root.join(currency);
    let mut markets = 0usize;
    let mut lead_holds = 0usize;
    let mut lead_price_sum = 0.0;
    if currency_root.exists() {
        let cutoff_ms = regime_event_end_ms - REDEEM_X_REGIME_WINDOW_MS;
        for path in collect_stream_dump_files(&currency_root)? {
            let Some((interval, event_end_ms)) =
                stream_dump_path_interval_and_end_ms(&currency_root, &path)
            else {
                continue;
            };
            // Только разрешённые рынки строго ВНУТРИ последнего часа перед текущим окном.
            if interval != target_interval
                || event_end_ms < cutoff_ms
                || event_end_ms >= regime_event_end_ms
            {
                continue;
            }
            let Some(cached) = cached_redeem_x_dump(&path) else {
                continue;
            };
            markets += 1;
            if cached.lead_won {
                lead_holds += 1;
            }
            lead_price_sum += cached.lead_price;
        }
    }

    // Вход разрешён: достаточно рынков и винрейт лидера бьёт его цену (EV покупки > 0).
    let allow = markets >= REDEEM_X_REGIME_MIN_MARKETS && {
        let lead_hold_rate = lead_holds as f64 / markets as f64;
        let avg_lead_price = lead_price_sum / markets as f64;
        avg_lead_price > 0.0 && lead_hold_rate > avg_lead_price
    };
    Ok(RedeemXMarketRegime {
        event_end_ms: regime_event_end_ms,
        allow,
    })
}

// --- Кэш распарсенных дампов ----------------------------------------------------------

/// Итог по одному рынку: выиграла ли лидирующая (фаворитная) нога и её средняя цена.
#[derive(Debug, Clone, Copy)]
struct CachedRedeemXDump {
    lead_won: bool,
    lead_price: f64,
}

static REDEEM_X_DUMP_CACHE: OnceLock<Mutex<IndexMap<PathBuf, CachedRedeemXDump>>> = OnceLock::new();

fn cached_redeem_x_dump(path: &Path) -> Option<CachedRedeemXDump> {
    let key = path.to_path_buf();
    let cache = REDEEM_X_DUMP_CACHE.get_or_init(|| Mutex::new(IndexMap::new()));
    {
        let mut guard = cache.lock().ok()?;
        if let Some(cached) = guard.shift_remove(&key) {
            guard.insert(key.clone(), cached);
            return Some(cached);
        }
    }

    let cached = load_redeem_x_dump(path)?;
    let mut guard = cache.lock().ok()?;
    if guard.len() >= REDEEM_X_DUMP_CACHE_CAP {
        guard.shift_remove_index(0);
    }
    guard.insert(key, cached);
    Some(cached)
}

fn load_redeem_x_dump(path: &Path) -> Option<CachedRedeemXDump> {
    let event_end_ms = stream_dump_event_end_ms(path)?;
    let bytes = fs::read(path).ok()?;
    let mut decoder = GzDecoder::new(bytes.as_slice());
    let mut decoded = Vec::new();
    decoder.read_to_end(&mut decoded).ok()?;
    let dump = bincode::deserialize::<MarketWsStreamDumpMarket>(&decoded).ok()?;
    momentum_market_outcome(&dump, event_end_ms)
}

/// По стрим-дампу одного рынка восстанавливает лидирующую (фаворитную) ногу и
/// проверяет, выиграла ли она. Лидер на момент = сторона с более высоким mid.
fn momentum_market_outcome(
    dump: &MarketWsStreamDumpMarket,
    event_end_ms: i64,
) -> Option<CachedRedeemXDump> {
    let winner_is_up = matches!(dump.winner, CurrencyUpDownOutcome::Up);

    // Совмещённая по времени лента (ts, is_up, mid).
    let mut events: Vec<(i64, bool, f64)> = Vec::new();
    for (ts, mid) in side_mid_series(&dump.up) {
        events.push((ts, true, mid));
    }
    for (ts, mid) in side_mid_series(&dump.down) {
        events.push((ts, false, mid));
    }
    if events.is_empty() {
        return None;
    }
    events.sort_by(|a, b| a.0.cmp(&b.0));

    let mut last_up: Option<f64> = None;
    let mut last_down: Option<f64> = None;
    // Снимки, где известны обе ноги: (remaining_ms, up_mid, down_mid).
    let mut snapshots: Vec<(i64, f64, f64)> = Vec::new();
    for (ts, is_up, mid) in events {
        if is_up {
            last_up = Some(mid);
        } else {
            last_down = Some(mid);
        }
        if let (Some(up), Some(down)) = (last_up, last_down) {
            let remaining_ms = event_end_ms.saturating_sub(ts);
            snapshots.push((remaining_ms, up, down));
        }
    }
    if snapshots.is_empty() {
        return None;
    }

    // Лидер на последнем (ближайшем к концу) снимке — решающий фаворит.
    let (_, last_up_mid, last_down_mid) = *snapshots.last()?;
    if (last_up_mid - last_down_mid).abs() < f64::EPSILON {
        return None;
    }
    let lead_is_up = last_up_mid > last_down_mid;
    let lead_won = lead_is_up == winner_is_up;

    // Средняя цена лидера по снимкам с остатком ≥ floor (цена накопления, не схлоп у конца).
    let mut sum = 0.0;
    let mut count = 0usize;
    for (remaining_ms, up, down) in &snapshots {
        if *remaining_ms < REDEEM_X_REGIME_SIGNAL_FLOOR_MS {
            continue;
        }
        sum += if lead_is_up { *up } else { *down };
        count += 1;
    }
    if count == 0 {
        // Фоллбэк: усредняем по всем снимкам.
        for (_, up, down) in &snapshots {
            sum += if lead_is_up { *up } else { *down };
            count += 1;
        }
    }
    let lead_price = sum / count as f64;
    if !(lead_price > 0.0 && lead_price < 1.0 && lead_price.is_finite()) {
        return None;
    }
    Some(CachedRedeemXDump {
        lead_won,
        lead_price,
    })
}

/// Восстанавливает ленту `(ts, mid)` одной ноги: best_bid/best_ask с переносом
/// последнего известного уровня, mid = середина (или одна из сторон, если есть только она).
fn side_mid_series(entries: &[MarketWsStreamDumpEntry]) -> Vec<(i64, f64)> {
    let mut sorted: Vec<&MarketWsStreamDumpEntry> = entries.iter().collect();
    sorted.sort_by_key(|entry| (stream_entry_ts(entry), entry.ingest_wall_ms));

    let mut last_bid: Option<f64> = None;
    let mut last_ask: Option<f64> = None;
    let mut out: Vec<(i64, f64)> = Vec::with_capacity(sorted.len());
    for entry in sorted {
        let ts = stream_entry_ts(entry);
        if let Some(bid) = payload_best_bid(&entry.payload) {
            last_bid = Some(bid);
        }
        if let Some(ask) = payload_best_ask(&entry.payload) {
            last_ask = Some(ask);
        }
        let mid = match (last_bid, last_ask) {
            (Some(bid), Some(ask)) if bid > 0.0 && ask > 0.0 => Some((bid + ask) / 2.0),
            (Some(bid), _) if bid > 0.0 => Some(bid),
            (_, Some(ask)) if ask > 0.0 => Some(ask),
            _ => entry.payload.price.filter(|p| *p > 0.0),
        };
        if let Some(mid) = mid.filter(|m| m.is_finite() && *m > 0.0 && *m < 1.0) {
            out.push((ts, mid));
        }
    }
    out
}

fn payload_best_bid(payload: &crate::data_ws::WsStreamPayload) -> Option<f64> {
    let from_book = payload
        .bids
        .iter()
        .filter(|lvl| lvl.price > 0.0 && lvl.size > 0.0 && lvl.price.is_finite())
        .map(|lvl| lvl.price)
        .fold(None, |acc: Option<f64>, p| Some(acc.map_or(p, |m| m.max(p))));
    from_book
        .or(payload.best_bid)
        .or_else(|| price_change_best_bid(&payload.price_changes))
}

fn payload_best_ask(payload: &crate::data_ws::WsStreamPayload) -> Option<f64> {
    let from_book = payload
        .asks
        .iter()
        .filter(|lvl| lvl.price > 0.0 && lvl.size > 0.0 && lvl.price.is_finite())
        .map(|lvl| lvl.price)
        .fold(None, |acc: Option<f64>, p| Some(acc.map_or(p, |m| m.min(p))));
    from_book
        .or(payload.best_ask)
        .or_else(|| price_change_best_ask(&payload.price_changes))
}

fn price_change_best_bid(changes: &[WsStreamPriceChange]) -> Option<f64> {
    changes.iter().rev().find_map(|change| change.best_bid)
}

fn price_change_best_ask(changes: &[WsStreamPriceChange]) -> Option<f64> {
    changes.iter().rev().find_map(|change| change.best_ask)
}

fn stream_entry_ts(entry: &MarketWsStreamDumpEntry) -> i64 {
    entry.payload.timestamp_ms.unwrap_or(entry.ingest_wall_ms)
}

// --- Перечисление файлов дампов -------------------------------------------------------

fn collect_stream_dump_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            out.extend(collect_stream_dump_files(&path)?);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".bin.gz"))
        {
            out.push(path);
        }
    }
    Ok(out)
}

fn stream_dump_path_interval_and_end_ms(
    currency_root: &Path,
    path: &Path,
) -> Option<(XFrameIntervalKind, i64)> {
    let rel = path.strip_prefix(currency_root).ok()?;
    let mut comps = rel.components();
    let _schema = comps.next()?;
    let period = comps.next()?.as_os_str().to_str()?;
    let interval = match period {
        "5m" => XFrameIntervalKind::FiveMin,
        "15m" => XFrameIntervalKind::FifteenMin,
        _ => return None,
    };
    Some((interval, stream_dump_event_end_ms(path)?))
}

fn stream_dump_event_end_ms(path: &Path) -> Option<i64> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".bin.gz")?;
    stem.rsplit("__").next()?.parse().ok()
}

// --- Решение о входе ------------------------------------------------------------------

/// Правило входа REDEEM_X: **полный клип** `coin+period` под гейтом исторического режима.
///
/// `None` — не заходим: режим ещё не готов или запретил вход (cache-miss → фоновая
/// загрузка), мало времени, нет цены/вне полосы, нога уперлась в потолок инвентаря, либо
/// размер ниже минимума по банку/позиции. Иначе — нотинал USDC для полного клипа.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn redeem_x_entry_size(
    frame: &XFrame<SIZE>,
    strict_book: Option<&StrictBook>,
    _bankroll: f64,
    currency: &str,
    event_end_ms: Option<i64>,
    positions_by_lane: &HashMap<crate::account::LaneKey, LanePositions>,
    pending_close_by_lane: &HashMap<crate::account::LaneKey, LanePositions>,
    account: Option<&SharedAccount>,
) -> Option<f64> {
    let interval = XFrameIntervalKind::from_i32(frame.xframe_interval_type)?;
    let event_end_ms = event_end_ms?;

    // (0) Исторический гейт: формула давала винрейт за последний час? (None = не готово.)
    if !redeem_x_market_regime_recommendation(currency, interval, event_end_ms, account?).await? {
        return None;
    }
    if frame.event_remaining_ms < REDEEM_X_ENTRY_MIN_REMAINING_MS {
        return None;
    }
    // Один проход по позициям рынка: шеры текущей ноги + мс с последней приземлившейся покупки.
    let (own_shares, ms_since_last_buy) =
        redeem_x_leg_scan([positions_by_lane, pending_close_by_lane], frame).await;
    // Троттлинг по времени: не чаще раза в N мс с последней покупки в этом рынке.
    if let Some(since_ms) = ms_since_last_buy
        && since_ms < REDEEM_X_MIN_REBUY_INTERVAL_MS
    {
        return None;
    }

    // (1) Фиксированный клип coin+period; maker встаёт на best_bid (= цена shares↔USDC).
    let clip = redeem_x_clip_shares(currency, interval)?;
    let maker_price = strict_book
        .and_then(crate::account_order::best_bid_strict)
        .or(frame.book_bid_l1_price)
        .or(strict_book.and_then(crate::account_order::best_ask_strict))
        .or(frame.book_ask_l1_price)
        .filter(|p| p.is_finite() && *p > 0.0)?;
    if !(REDEEM_X_MIN_PRICE..=REDEEM_X_MAX_PRICE).contains(&maker_price) {
        return None;
    }

    // (2) Асимметричный потолок инвентаря ноги — единственный направленный крен к лидеру.
    let leg_prob = frame.currency_implied_prob.unwrap_or(maker_price);
    let leg_cap = clip
        * if leg_prob >= REDEEM_X_LEAD_PROB {
            REDEEM_X_MAX_LEAD_CLIPS
        } else {
            REDEEM_X_MAX_LAG_CLIPS
        };
    if own_shares + clip > leg_cap {
        return None;
    }

    // (3) Полный клип → нотинал USDC с потолками банка/позиции.
    let size = (clip * maker_price)
        .min(MAX_POSITION_USD);
    (size >= MIN_POSITION_USD).then_some(size)
}

/// Фиксированный клип ОДНОГО лимитного ордера по `coin+period` (медиана размера ордера из
/// tail-отчёта, инвариантна к цене). Неизвестная комбинация → `panic!`.
fn redeem_x_clip_shares(currency: &str, interval: XFrameIntervalKind) -> Option<f64> {
    use XFrameIntervalKind::{FifteenMin, FiveMin};
    Some(match (currency.to_ascii_lowercase().as_str(), interval) {
        ("eth", FiveMin) => 5.0,
        ("btc", FiveMin) => 100.0,
        ("btc", FifteenMin) => 20.0,
        (coin, interval) => panic!(
            "redeem_x_clip_shares: unsupported coin+period: coin={coin}, interval={interval:?}"
        ),
    })
}

/// `None` — режим ещё не готов (поставили задачу фоновому загрузчику, кадр пропускаем);
/// `Some(allow)` — из кэша: можно ли заходить (формула давала винрейт за последний час).
async fn redeem_x_market_regime_recommendation(
    currency: &str,
    interval: XFrameIntervalKind,
    event_end_ms: i64,
    account: &SharedAccount,
) -> Option<bool> {
    let key: RedeemXMarketRegimeKey = (currency.to_ascii_lowercase(), interval);
    if let Some(regime) = account.redeem_x_market_regime.read().await.get(&key)
        && regime.event_end_ms == event_end_ms
    {
        return Some(regime.allow);
    }
    // cache-miss / устарело → фоновая загрузка, текущий кадр пропускаем.
    let _ = account
        .redeem_x_market_regime_tx
        .try_send(RedeemXMarketRegimeLoadCommand {
            account: account.clone(),
            currency: key.0,
            interval,
            event_end_ms,
        });
    None
}

/// Один проход по обоим bucket'ам для рынка `frame.market_id`: возвращает
/// `(own_shares, ms_since_last_buy)`:
///   * `own_shares` — суммарные **фактически удержанные** шеры ТЕКУЩЕЙ ноги
///     (`shares_held` обновляется после fill'а → корректно и для частичного исполнения);
///   * `ms_since_last_buy` — мс с последней **приземлившейся** покупки по `landed_at`
///     settled-отчёта `open_buy_invoke` (Some ⇔ success, включая partial; мок ставит
///     `landed_at = current_timestamp_ms()` — поэтому единый wall-clock и для моков).
///     `None`, если ни одна покупка ещё не приземлилась.
async fn redeem_x_leg_scan(
    buckets: [&HashMap<crate::account::LaneKey, LanePositions>; 2],
    frame: &XFrame<SIZE>,
) -> (f64, Option<i64>) {
    let now_ms = crate::util::current_timestamp_ms();
    let mut own_shares = 0.0;
    let mut ms_since_last_buy: Option<i64> = None;
    for by_lane in buckets {
        for lane_positions in by_lane.values() {
            for position in lane_positions.values() {
                let p = position.read().await;
                if p.market_id != frame.market_id {
                    continue;
                }
                if p.asset_id == frame.asset_id {
                    own_shares += p.shares_held;
                }
                if let Some(landed_at) = p
                    .open_buy_invoke
                    .as_ref()
                    .and_then(crate::account_order::invoke_settlement_report)
                    .and_then(|report| report.landed_at)
                {
                    let since = now_ms - landed_at;
                    if since >= 0 {
                        ms_since_last_buy =
                            Some(ms_since_last_buy.map_or(since, |m| m.min(since)));
                    }
                }
            }
        }
    }
    (own_shares, ms_since_last_buy)
}
