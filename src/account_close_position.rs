//! Все ветки закрытия позиций в одном месте; ранее жили по `history_sim` /
//! `account` / `account_submit`. Каждая ветка обновляет `bankroll` (для submit /
//! resolution) и [`crate::sim_stats::SideStats`] (`pnl_usd` / `trades` / `wins` /
//! `losses` / `closed_trade_entries` + специфичные счётчики), пишет
//! [`crate::trade_csv_log::TradeCsvRow`]; submit-ветка дополнительно пишет
//! `SUBMIT_TRADE_CSV_LOG` и спавнит partial-graph HTML.
//!
//! * [`close_position_market_exit`] — backtest / real_sim bid-walk выход
//!   (TP / SL / Timeout / EvExit), вызывается из [`crate::history_sim::manage_positions`].
//! * [`close_position_resolution`] — бинарный payout $1/$0, вызывается из
//!   [`crate::account::Account::resolve_pending_market_sync`].
//! * [`close_position_after_sell`] — успешный SELL-fill в submit/mock (maker-TP
//!   `resting → fill` или taker-FAK SELL, который выбрал остаток до ~0). Вызывается
//!   из [`crate::account_submit::spawn_open_buy_taker`] (после settle maker invoke)
//!   и [`crate::account_submit::spawn_sell_taker`] (после settle taker invoke, если
//!   `shares_remaining_to_sell ≈ 0`). Гард на `sell_rep.success / !partial` — у
//!   caller'а (см. соответствующие `if !*_rep.success || *_rep.partial { return; }`
//!   перед вызовом). Resolution submit — отдельным промтом, сейчас ветки нет.

use crate::account::SharedAccount;
use crate::account_order::{
    OrderAmount, SingleOrderClobInvocationReport, invoke_settlement_ready,
    invoke_settlement_report,
};
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{
    CloseReason, LanePositions, OpenPosition, POLYMARKET_CRYPTO_TAKER_FEE_RATE,
    SIM_MAX_SLIPPAGE_FROM_L1_PCT, SharedOpenPosition, StrictBook, book_fill_sell,
    book_fill_sell_strict, position_interval_label, position_side_label,
    trade_csv_close_reason_label,
};
use crate::project_manager::ProjectManager;
use crate::sim_stats::SideStats;
use crate::xframe::{SIZE, XFrame, Y_TRAIN_TAKE_PROFIT_PP};
use std::sync::Arc;

/// Gross USDC при TP: если полный sell-walk даёт порог TP — можно обойти cap к L1 (см. [`crate::history_sim::sell_gate`]).
fn gross_usdc_sell_take_profit(
    frame: &XFrame<SIZE>,
    pos: &OpenPosition,
    strict_book: Option<&StrictBook>,
) -> Option<f64> {
    let cap = Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT);
    let meets_tp = |gross: f64| -> bool {
        if pos.shares_held <= 1e-18 {
            return false;
        }
        let vwap = gross / pos.shares_held;
        vwap - pos.buy_price >= Y_TRAIN_TAKE_PROFIT_PP
    };

    match strict_book {
        Some(book) => {
            let uncapped = book_fill_sell_strict(book, pos.shares_held, None)?;
            let capped = book_fill_sell_strict(book, pos.shares_held, cap);
            if meets_tp(uncapped) {
                match capped {
                    Some(g) if meets_tp(g) => Some(g),
                    Some(_) => Some(uncapped),
                    None => Some(uncapped),
                }
            } else {
                capped.or(Some(uncapped))
            }
        }
        None => {
            let uncapped = book_fill_sell(frame, pos.shares_held, None)?;
            let capped = book_fill_sell(frame, pos.shares_held, cap);
            if meets_tp(uncapped) {
                match capped {
                    Some(g) if meets_tp(g) => Some(g),
                    Some(_) => Some(uncapped),
                    None => Some(uncapped),
                }
            } else {
                capped.or(Some(uncapped))
            }
        }
    }
}

/// Рыночный выход backtest / real_sim (TP / SL / Timeout / EvExit): bid-walk по
/// `frame` / `strict_book`, taker fee если TP не дотягивается до maker.
///
/// При успешном close: `*bankroll += pnl`, инкремент `stats.{tp_count,sl_count,...}`,
/// CSV-строка, возврат `true` (caller использует как `sold = true`). При отказе
/// (стакан не дал заполнить shares): `stats.kelly_strict_sell_skips += 1`,
/// позиция возвращается в `remaining` через `(pos_id, pos_arc)` и возврат `false`.
/// Резолюция и maker-TP-fill — в [`close_position_resolution`] /
/// [`close_position_maker_tp`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn close_position_market_exit(
    bankroll: &mut f64,
    remaining: &mut LanePositions,
    pos_id: String,
    pos_arc: SharedOpenPosition,
    pos: &OpenPosition,
    exit_price: f64,
    reason: &CloseReason,
    frame: &XFrame<SIZE>,
    stats: &mut SideStats,
    strict_book: Option<&StrictBook>,
) -> bool {
    let gross_usdc_opt = if reason.is_voluntary_exit() {
        gross_usdc_sell_take_profit(frame, pos, strict_book)
    } else {
        match strict_book {
            Some(book) => book_fill_sell_strict(book, pos.shares_held, None),
            None => book_fill_sell(frame, pos.shares_held, None),
        }
    };
    let Some(gross_usdc) = gross_usdc_opt else {
        stats.kelly_strict_sell_skips += 1;
        remaining.insert(pos_id, pos_arc);
        return false;
    };
    let sell_price = if pos.shares_held > 0.0 {
        (gross_usdc / pos.shares_held).clamp(0.001, 0.999)
    } else {
        exit_price.clamp(0.001, 0.999)
    };
    // Без taker fee на выходе только если TP исполняется как maker (таргет выше bid на входе).
    let voluntary_is_maker = match reason {
        CloseReason::TakeProfit => {
            let tp_target = (pos.buy_price + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999);
            match pos.best_bid_at_entry {
                Some(b) => tp_target > b,
                None => true,
            }
        }
        _ => false,
    };
    let fee_usdc = if voluntary_is_maker {
        0.0
    } else {
        pos.shares_held * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell_price * (1.0 - sell_price)
    };
    stats.fees_paid += fee_usdc;
    let net_usdc = gross_usdc - fee_usdc;

    let pnl = net_usdc - pos.position_size;
    stats.pnl_usd += pnl;
    stats.trades += 1;
    if pnl >= 0.0 {
        stats.wins += 1;
    } else {
        stats.losses += 1;
    }
    stats
        .closed_trade_entries
        .push((pos.raw_pred_at_open, pnl > 0.0));
    match reason {
        CloseReason::TakeProfit => {
            stats.tp_count += 1;
            stats.pnl_tp += pnl;
        }
        CloseReason::StopLoss => {
            stats.sl_count += 1;
            stats.pnl_sl += pnl;
        }
        CloseReason::Timeout => {
            stats.timeout_count += 1;
            stats.pnl_timeout += pnl;
        }
        CloseReason::EvExitProfit => {
            stats.ev_exit_profit_count += 1;
            stats.pnl_ev_exit_profit += pnl;
        }
        CloseReason::EvExitLoss => {
            stats.ev_exit_loss_count += 1;
            stats.pnl_ev_exit_loss += pnl;
        }
    }

    let close_unix_ms = pos.event_end_ms.map(|end_ms| end_ms - frame.event_remaining_ms);
    let interval_str = position_interval_label(pos);
    let side_str = position_side_label(pos);
    let open_unix_ms = pos
        .event_end_ms
        .map(|end_ms| end_ms - pos.event_remaining_ms_at_open);
    let graph_html_file_uri = crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri(pos)
        .map(|bin_path| {
            crate::xframe_graph_dump::graph_html_trade_file_uri(
                &bin_path,
                open_unix_ms,
                close_unix_ms,
                Some(side_str),
            )
        })
        .unwrap_or_default();
    crate::trade_csv_log::write_trade_csv_row(crate::trade_csv_log::TradeCsvRow {
        polymarket_url: &pos.polymarket_url,
        price_to_beat: pos.price_to_beat,
        final_price: pos.final_price,
        market_id: &pos.market_id,
        asset_id: &pos.asset_id,
        side: side_str,
        interval: interval_str,
        currency: &pos.currency,
        exit_reason: trade_csv_close_reason_label(reason),
        buy_price: pos.buy_price,
        raw_pred: pos.raw_pred_at_open,
        cal_pred: pos.cal_pred_at_open,
        kelly_f: pos.kelly_f_at_open,
        position_size: pos.position_size,
        shares_held: pos.shares_held,
        exit_price: sell_price,
        fee_usdc,
        pnl,
        frames_held: pos.frames_held,
        p_win_ema_at_close: pos.p_win_ema,
        event_remaining_ms_at_open: pos.event_remaining_ms_at_open,
        event_remaining_ms_at_close: frame.event_remaining_ms,
        open_unix_ms,
        close_unix_ms,
        graph_html_file_uri: graph_html_file_uri.as_str(),
        pnl_top5_shap: pos.pnl_top5_shap_at_open.as_str(),
        pos_id: pos.id.as_str(),
        fill_role: "",
        finalized_via: "",
        planned_buy_price: None,
        planned_shares_held: None,
        planned_entry_cost: None,
        open_order_id: None,
        tp_order_id: None,
        close_order_ids: &[],
    });

    *bankroll += pnl;
    // pos_arc намеренно не вставляем обратно в `remaining` — позиция закрыта.
    let _ = pos_arc;
    true
}

/// Резолюционное закрытие одной позиции внутри
/// [`crate::account::Account::resolve_pending_market_sync`]: бинарный payout
/// ($1/$0), обновление `bankroll`, [`SideStats`] (`trades` / `wins` / `losses` /
/// `closed_trade_entries` / `resolution_*` / `pnl_resolution_*`), CSV-строка
/// `ResolutionWin` / `ResolutionLoss`. Комиссия на резолюции `= 0.0`.
pub(crate) async fn close_position_resolution(
    account: &SharedAccount,
    pos_arc: SharedOpenPosition,
    token_won: bool,
    currency: &str,
    market_id: &str,
    final_price: Option<f64>,
    side_stats: &mut SideStats,
) {
    let pos = pos_arc.read().await;
    let pnl = if token_won {
        pos.shares_held - pos.position_size
    } else {
        -pos.position_size
    };
    *account.bankroll.write().await += pnl;
    side_stats.pnl_usd += pnl;
    side_stats.trades += 1;
    if pnl >= 0.0 {
        side_stats.wins += 1;
    } else {
        side_stats.losses += 1;
    }
    // Resolution не через рыночное закрытие — дублируем в closed_trade_entries (replay калибровки).
    side_stats
        .closed_trade_entries
        .push((pos.raw_pred_at_open, pnl > 0.0));
    if token_won {
        side_stats.resolution_win += 1;
        side_stats.pnl_resolution_win += pnl;
        if pnl >= 0.0 {
            side_stats.resolution_win_profit += 1;
        } else {
            side_stats.resolution_win_loss += 1;
        }
    } else {
        side_stats.resolution_loss += 1;
        side_stats.pnl_resolution_loss += pnl;
    }

    let exit_reason = if token_won {
        "ResolutionWin"
    } else {
        "ResolutionLoss"
    };
    let close_unix_ms = pos.event_end_ms;
    let interval_str = position_interval_label(&pos);
    let side_str = position_side_label(&pos);
    let open_unix_ms = pos
        .event_end_ms
        .map(|end_ms| end_ms - pos.event_remaining_ms_at_open);
    let graph_html_file_uri =
        crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri(&pos)
        .map(|bin_path| {
            crate::xframe_graph_dump::graph_html_trade_file_uri(
                &bin_path,
                open_unix_ms,
                close_unix_ms,
                Some(side_str),
            )
        })
        .unwrap_or_default();
    crate::trade_csv_log::write_trade_csv_row(crate::trade_csv_log::TradeCsvRow {
        polymarket_url: &pos.polymarket_url,
        price_to_beat: pos.price_to_beat,
        final_price: final_price.or(pos.final_price),
        currency,
        interval: interval_str,
        side: side_str,
        market_id,
        asset_id: &pos.asset_id,
        exit_reason,
        buy_price: pos.buy_price,
        raw_pred: pos.raw_pred_at_open,
        cal_pred: pos.cal_pred_at_open,
        kelly_f: pos.kelly_f_at_open,
        position_size: pos.position_size,
        shares_held: pos.shares_held,
        exit_price: if token_won { 1.0 } else { 0.0 },
        fee_usdc: 0.0,
        pnl,
        frames_held: pos.frames_held,
        p_win_ema_at_close: pos.p_win_ema,
        event_remaining_ms_at_open: pos.event_remaining_ms_at_open,
        event_remaining_ms_at_close: 0,
        open_unix_ms,
        close_unix_ms,
        graph_html_file_uri: graph_html_file_uri.as_str(),
        pnl_top5_shap: pos.pnl_top5_shap_at_open.as_str(),
        pos_id: pos.id.as_str(),
        fill_role: "",
        finalized_via: "",
        planned_buy_price: None,
        planned_shares_held: None,
        planned_entry_cost: None,
        open_order_id: None,
        tp_order_id: None,
        close_order_ids: &[],
    });
}

/// Закрытие позиции после SELL-fill'ов в submit/mock-режиме — общая ветка для
/// двух caller'ов:
/// * [`crate::account_submit::spawn_open_buy_taker`] — после settle maker-TP
///   invoke с `success && !partial` (целиком закрылись на maker'е);
///   `reason = CloseReason::TakeProfit`, `fill_role = "Maker"`,
///   `finalized_via = "maker_tp_fill"`.
/// * [`crate::account_submit::spawn_sell_taker`] — после settle taker-FAK invoke,
///   когда `shares_remaining_to_sell ≤
///   [`crate::account_submit::CLOSE_AFTER_SELL_REMAINING_SHARES_TOLERANCE`]`
///   (taker дошёл до ~0, возможно вместе с partial-maker'ом); `reason` —
///   фактический (TP / SL / Timeout / EvExit), `fill_role = "Taker"`,
///   `finalized_via = "taker_sell_fill"`.
///
/// Сам метод вытаскивает все SELL-fills и order-id'шники из
/// [`crate::history_sim::OpenPosition`]:
/// * `open_order_id` → `TradeCsvRow::open_order_id`;
/// * `maker_tp_position` (если есть) → `tp_order_id` + вклад в сумму SELL
///   (settled `success` invoke, partial допустим — например, maker частично выбился
///   до того, как taker FAK добил остаток);
/// * каждый элемент `taker_positions` → запись в `close_order_ids` + вклад в сумму
///   SELL (settled `success`, partial допустим). Это безопасно, потому что
///   аналог [`crate::history_sim::OpenPosition::shares_remaining_to_sell`] так же
///   суммирует `making_amount` по settled+success invoke'ам всех веток.
///
/// Из аргументов остаются только то, что НЕ выводится из позиции: `buy_rep`
/// (для sanity-check vs `position.position_size`) и метаданные ветки
/// (`reason / fill_role / finalized_via`).
///
/// Удаляет позицию из [`crate::account::Account::positions`], считает PnL/fee
/// (`fee_usdc = 0` — maker всегда `0`, taker-fee CLOB уже вычел из
/// `taking_amount`), обновляет `bankroll` и [`SideStats`] лейна (через
/// [`crate::account::Account::real_sim_state_for_currency`]: `trades` / `wins` /
/// `losses` / `pnl_usd` / `closed_trade_entries` + специфичный
/// `{tp,sl,timeout,ev_exit_profit,ev_exit_loss}_count` / `pnl_*` — `match reason`
/// как в [`close_position_market_exit`]). Пишет regular- и submit-CSV-строку,
/// спавнит partial-graph HTML.
///
/// Порядок локов `state → bankroll` совпадает с [`crate::real_sim::tick_once`]
/// (иначе deadlock при пересечении с tick'ом). Если [`crate::real_sim::RealSimState`]
/// ещё не зарегистрирован (run_real_sim не дошёл) — stats тихо пропускаем;
/// `bankroll` и CSV всё равно обновляются.
///
/// Решение «пора закрывать» — у caller'а (`maker_rep.success && !maker_rep.partial`
/// для maker-сайта; `shares_remaining_to_sell ≤ TOL` для taker-сайта; либо
/// `event_end_ms + POST_MARKET_END_RESOLUTION_DELAY_MS` для post-market
/// residual).
///
/// **Path detection (внутри функции — НЕ по `reason.is_none()`):**
///
/// «Post-market residual» путь триггерится если **рынок завершился**
/// (`now > event_end_ms`) **И** на счёте остался residual для оценки
/// (`actual_shares_net - shares_sold > CLOSE_AFTER_SELL_REMAINING_SHARES_TOLERANCE`).
/// В этом режиме residual оценивается бинарно через
/// [`crate::project_manager::MarketResolution`]: `$1`/шер если наша сторона
/// выиграла (`final_price >= price_to_beat` для UP, обратное для DOWN), иначе
/// `$0`. PNL = `usd_received + residual_payout - actual_entry_cost`; SideStats
/// апдейтит `resolution_win` / `resolution_loss` / `pnl_resolution_*`;
/// `exit_reason` в CSV — `"ResolutionWin"` / `"ResolutionLoss"`. Если
/// `MarketResolution` отсутствует или `final_price = None` — выходим БЕЗ
/// взвода `close_after_submit_finalized` (флаг сохраняем для будущего триггера).
///
/// Иначе («after-sell» путь): позиция полностью распродана через maker TP /
/// taker FAK. PNL = `usd_received - actual_entry_cost`; SideStats апдейтит
/// ветку по `reason` (`tp_count` / `sl_count` / `timeout_count` / `ev_exit_*`);
/// `exit_reason` в CSV — [`trade_csv_close_reason_label`]. Bail если SELL-fills
/// нулевые. Post-market caller передаёт `reason = None`; если до flag-set'а
/// другой callback успел добить позицию (residual ≈ 0) — выходим тихо (никаких
/// данных для логирования нет).
///
/// **Идемпотентность.** Maker-TP callback в `spawn_open_buy_taker`, taker-FAK
/// callback в `spawn_sell_taker` и post-market-end task могут параллельно
/// посчитать позицию закрытой — флаг
/// [`crate::history_sim::OpenPosition::close_after_submit_finalized`]
/// взводится в `true` под `position.write().await` при первом входе; второй
/// конкурент видит `true` и тихо выходит. `buy_rep` для sanity-check'а
/// (`making_amount` vs `position.position_size`) берётся из
/// `position.open_buy_invoke` (тот же watch, который писал actual в позицию через
/// [`crate::account_submit::spawn_open_buy_taker`]) — отдельным аргументом не
/// нужен.
pub(crate) async fn close_position_after_submit(
    account: &SharedAccount,
    position: &SharedOpenPosition,
    project_manager: Option<&Arc<ProjectManager>>,
    reason: Option<&CloseReason>,
    fill_role: &'static str,
    finalized_via: &'static str,
) {
    // Pre-flight snapshot: все reads + aggregation БЕЗ установки idempotency
    // флага. Path detection ниже требует `residual_shares` (а значит и
    // `actual_shares_net` + sums), поэтому делаем агрегацию заранее.
    // Параллельные callers могут продублировать это — безопасно, нет write'ов.
    let position_snapshot = position.read().await.clone();
    let pos_id = position_snapshot.id.clone();
    let market_id = position_snapshot.market_id.clone();
    let event_end_ms = position_snapshot.event_end_ms;
    let asset_id = position_snapshot.asset_id.as_str();
    let currency = position_snapshot.currency.as_str();
    let planned_buy_price = position_snapshot.planned_buy_price;
    let planned_shares_held = position_snapshot.planned_shares_held;
    let planned_entry_cost = position_snapshot.planned_entry_cost;
    let actual_buy_price = position_snapshot.buy_price;
    let actual_entry_cost = position_snapshot.position_size;

    // `actual_shares_net` берём напрямую из BUY-invoke watch'а
    // ([`crate::history_sim::OpenPosition::open_buy_invoke`]) — authoritative от
    // CLOB через WS-fill report. `position.shares_held` — derived field,
    // выставленный в [`crate::account_submit::spawn_open_buy_taker`] из того же
    // report'а, но при late WS-update'ах может расходиться. Если BUY-invoke
    // отсутствует / не settled / не success / taking_amount не Shares — `0.0`
    // (defensive: дальше aggregator-guard для after-sell поймает; для resolution
    // residual просто будет 0).
    let actual_shares_net = position_snapshot
        .open_buy_invoke
        .as_ref()
        .and_then(invoke_settlement_report)
        .filter(|report| report.success)
        .and_then(|report| match report.taking_amount {
            OrderAmount::Shares(shares) if shares.is_finite() && shares > 0.0 => Some(shares),
            _ => None,
        })
        .unwrap_or(0.0);

    // Агрегируем SELL fills по позиции: maker TP (если есть, settled+success;
    // partial допустим) + все taker SELL'ы (settled+success; partial допустим).
    // Зеркало логики [`crate::history_sim::OpenPosition::shares_remaining_to_sell`]
    // (там тоже суммируем `making_amount` по success-invoke'ам обеих веток). Это
    // гарантирует консистентность `shares_sold + residual ≈ actual_shares_net`.
    let mut shares_sold = 0.0_f64;
    let mut usd_received = 0.0_f64;
    let mut tp_order_id: Option<String> = None;
    let mut close_order_ids: Vec<String> = Vec::new();

    let accumulate = |report: &SingleOrderClobInvocationReport,
                      shares_acc: &mut f64,
                      usd_acc: &mut f64| {
        if !report.success {
            return;
        }
        if let (OrderAmount::Shares(s), OrderAmount::UsdNotional(u)) =
            (report.making_amount, report.taking_amount)
            && s.is_finite()
            && s > 0.0
            && u.is_finite()
            && u > 0.0
        {
            *shares_acc += s;
            *usd_acc += u;
        }
    };

    if let Some(weak) = position_snapshot.maker_tp_position.as_ref()
        && let Some(arc) = weak.upgrade()
    {
        let closing = arc.read().await;
        tp_order_id = closing
            .order_id
            .clone()
            .filter(|order_id| !order_id.trim().is_empty());
        if let Some(watch) = closing.invoke_settle.as_ref()
            && invoke_settlement_ready(watch)
            && let Some(report) = invoke_settlement_report(watch)
        {
            accumulate(&report, &mut shares_sold, &mut usd_received);
        }
    }

    for weak in &position_snapshot.taker_positions {
        let Some(arc) = weak.upgrade() else {
            continue;
        };
        let closing = arc.read().await;
        if let Some(order_id) = closing
            .order_id
            .as_ref()
            .filter(|order_id| !order_id.trim().is_empty())
        {
            close_order_ids.push(order_id.clone());
        }
        if let Some(watch) = closing.invoke_settle.as_ref()
            && invoke_settlement_ready(watch)
            && let Some(report) = invoke_settlement_report(watch)
        {
            accumulate(&report, &mut shares_sold, &mut usd_received);
        }
    }

    let residual_shares = (actual_shares_net - shares_sold).max(0.0);

    // Path detection: post-market-residual ИФФ (1) рынок завершился
    // (`now > event_end_ms`) И (2) остался непроданный residual выше
    // tolerance. НЕ полагаемся на `reason.is_none()`: post-market caller может
    // передать `None`, но если до нашего входа другой callback успел добить
    // позицию (residual ≈ 0) — этот вызов уйдёт в after-sell ветку и тихо
    // выйдет ниже (нет `reason` и нет residual для resolution).
    let now_ms = crate::util::current_timestamp_ms();
    let market_ended = event_end_ms.is_some_and(|end_ms| now_ms > end_ms);
    let is_post_market_residual = market_ended
        && residual_shares > crate::account_submit::CLOSE_AFTER_SELL_REMAINING_SHARES_TOLERANCE;

    // Resolution-path: lookup MarketResolution ДО взвода флага idempotency
    // (если данных нет — выходим, флаг сохраняем для будущего триггера).
    let market_price_to_beat_and_final_price: Option<(f64, f64)> = if is_post_market_residual {
        let Some(pm) = project_manager else {
            crate::tee_eprintln!(
                "[submit] post-market resolution pos_id={pos_id}: \
                 project_manager=None — финализация невозможна, пропуск",
            );
            return;
        };
        let lookup: Option<crate::project_manager::MarketResolution> = pm
            .market_resolution_by_market
            .read()
            .await
            .get(&market_id)
            .copied();
        let Some(market_resolution) = lookup else {
            crate::tee_eprintln!(
                "[submit] post-market resolution pos_id={pos_id} \
                 market_id={market_id}: MarketResolution отсутствует в кэше \
                 — финализация отложена (флаг НЕ взводим)",
            );
            return;
        };
        let Some(final_price) = market_resolution.final_price else {
            crate::tee_eprintln!(
                "[submit] post-market resolution pos_id={pos_id} \
                 market_id={market_id}: final_price=None (refine следующего \
                 окна ещё не пришёл) — финализация отложена (флаг НЕ взводим)",
            );
            return;
        };
        Some((market_resolution.price_to_beat, final_price))
    } else {
        None
    };

    // After-sell path требует ненулевых SELL-fills (`pnl = usd_received - entry`
    // даст шум если 0); resolution path допускает 0 (residual подберёт всё).
    // Бэйлим ДО взвода флага idempotency, чтобы не блокировать future-триггеры.
    let has_valid_sell_fills = shares_sold.is_finite()
        && shares_sold > 0.0
        && usd_received.is_finite()
        && usd_received > 0.0;
    if !(is_post_market_residual || has_valid_sell_fills) {
        crate::tee_eprintln!(
            "[submit] pnl pos_id={pos_id} reason={reason:?} role={fill_role}: \
             сумма SELL-fills нулевая/невалидная (shares={shares_sold:.6} USD={usd_received:.6}) — \
             лог пропускаем (флаг НЕ взводим)",
        );
        return;
    }

    // Edge case: caller — post-market spawn (`reason = None`), но между его
    // pre-check (residual > TOL) и нашим pre-flight другой callback успел
    // добить позицию (residual ≤ TOL → `is_post_market_residual = false`).
    // SideStats-ветка для `(None, None)` не определена; флаг не трогаем —
    // финализирует тот callback, что захватит флаг первым.
    if !is_post_market_residual && reason.is_none() {
        crate::tee_println!(
            "[submit] pnl pos_id={pos_id} role={fill_role}: reason=None и \
             residual≤TOL ⇒ финализация уже идёт через другой callback, пропуск",
        );
        return;
    }

    // Атомарный check-and-set флага идемпотентности (под write-lock без await
    // внутри — гонка с другими caller'ами разрешена однозначно).
    {
        let mut open_position = position.write().await;
        if open_position.close_after_submit_finalized {
            crate::tee_println!(
                "[submit] pnl pos_id={pos_id} reason={reason:?} role={fill_role}: \
                 close_after_submit_finalized=true — повторный вызов пропускаем",
            );
            return;
        }
        open_position.close_after_submit_finalized = true;
    }
    {
        let mut positions_guard = account.positions.write().await;
        for lane_positions in positions_guard.values_mut() {
            lane_positions.remove(&pos_id);
        }
    }

    // Resolution-path: `token_won` из `(price_to_beat, final_price)` + нашей
    // стороны UP/DOWN. Неизвестная сторона → loss (consistency с
    // `close_position_resolution`).
    let our_side =
        CurrencyUpDownOutcome::from_i32(position_snapshot.currency_up_down_outcome_at_open);
    let token_won_resolution: Option<bool> =
        market_price_to_beat_and_final_price.map(|(ptb, final_price)| {
            let up_won = final_price >= ptb;
            match our_side {
                Some(CurrencyUpDownOutcome::Up) => up_won,
                Some(CurrencyUpDownOutcome::Down) => !up_won,
                None => {
                    crate::tee_eprintln!(
                        "[submit] post-market resolution pos_id={pos_id}: неизвестный \
                         currency_up_down_outcome_at_open={} — финализируем как loss",
                        position_snapshot.currency_up_down_outcome_at_open,
                    );
                    false
                }
            }
        });
    // Sanity-check: `buy_rep.making_amount` (authoritative от CLOB) должен биться с
    // `position.position_size`, который был применён в `spawn_open_buy_taker`.
    // `buy_rep` берём из того же `position.open_buy_invoke` watch'а, который писал
    // actual в позицию — отдельным аргументом не передаём.
    if let Some(buy_watch) = position_snapshot.open_buy_invoke.as_ref()
        && let Some(buy_rep) = invoke_settlement_report(buy_watch)
        && let OrderAmount::UsdNotional(spent_usd) = buy_rep.making_amount
        && spent_usd.is_finite()
        && spent_usd > 0.0
        && (spent_usd - actual_entry_cost).abs() > 1e-6
    {
        crate::tee_eprintln!(
            "[submit] pnl pos_id={pos_id} reason={reason:?} role={fill_role}: \
             buy_rep.making_amount={spent_usd:.6} != position.position_size={actual_entry_cost:.6} \
             — possible WS-fill drift",
        );
    }

    // Payout от residual: только для resolution path. `residual_shares` уже
    // посчитан pre-flight'ом выше (`actual_shares_net - shares_sold`). Для
    // after-sell caller гарантирует residual ≤ TOL → пэйаут считаем как 0
    // (минимальный шум от <TOL шеров игнорируем, чтобы PNL = сугубо sell-fills).
    let residual_payout = match token_won_resolution {
        Some(true) => residual_shares,
        Some(false) | None => 0.0,
    };

    // Maker (TP) fee = 0; taker (FAK SELL) fee CLOB уже вычел из `taking_amount`
    // (`making_amount` остаётся валовыми shares). Resolution-payout — комиссия 0.
    // Поэтому здесь всегда 0, то же поведение, что и в [`close_position_market_exit`].
    let fee_usdc: f64 = 0.0;
    let pnl = usd_received + residual_payout - actual_entry_cost;
    // exit_price: для after-sell — sell-VWAP; для resolution — blended exit
    // (продано + payout residual) / total shares. После-sell guard выше
    // гарантирует `shares_sold > 0`.
    let exit_price = if token_won_resolution.is_some() {
        let total_exit_usd = usd_received + residual_payout;
        if actual_shares_net > 1e-18 {
            (total_exit_usd / actual_shares_net).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        (usd_received / shares_sold).clamp(0.001, 0.999)
    };

    let interval_kind =
        XFrameIntervalKind::from_i32(position_snapshot.xframe_interval_type_at_open);
    let real_sim_state = account.real_sim_state_for_currency(currency).await;
    match (real_sim_state, interval_kind, our_side) {
        (Some(real_sim_state), Some(interval_kind), Some(side)) => {
            let mut state_guard = real_sim_state.write().await;
            *account.bankroll.write().await += pnl;
            if let Some(sim_stats) = state_guard.stats.get_mut(&interval_kind) {
                let side_stats = match side {
                    CurrencyUpDownOutcome::Up => &mut sim_stats.up,
                    CurrencyUpDownOutcome::Down => &mut sim_stats.down,
                };
                side_stats.pnl_usd += pnl;
                side_stats.trades += 1;
                if pnl >= 0.0 {
                    side_stats.wins += 1;
                } else {
                    side_stats.losses += 1;
                }
                side_stats.fees_paid += fee_usdc;
                side_stats
                    .closed_trade_entries
                    .push((position_snapshot.raw_pred_at_open, pnl > 0.0));
                // Резолюционные счётчики (если path = resolution) либо reason-based
                // ветка (если path = after-sell). Path однозначно определяется тем,
                // какая из веток в `(token_won_resolution, reason)` пришла как Some
                // (выше есть mutually-exclusive проверки).
                match (token_won_resolution, reason) {
                    (Some(true), _) => {
                        side_stats.resolution_win += 1;
                        side_stats.pnl_resolution_win += pnl;
                        if pnl >= 0.0 {
                            side_stats.resolution_win_profit += 1;
                        } else {
                            side_stats.resolution_win_loss += 1;
                        }
                    }
                    (Some(false), _) => {
                        side_stats.resolution_loss += 1;
                        side_stats.pnl_resolution_loss += pnl;
                    }
                    (None, Some(CloseReason::TakeProfit)) => {
                        side_stats.tp_count += 1;
                        side_stats.pnl_tp += pnl;
                    }
                    (None, Some(CloseReason::StopLoss)) => {
                        side_stats.sl_count += 1;
                        side_stats.pnl_sl += pnl;
                    }
                    (None, Some(CloseReason::Timeout)) => {
                        side_stats.timeout_count += 1;
                        side_stats.pnl_timeout += pnl;
                    }
                    (None, Some(CloseReason::EvExitProfit)) => {
                        side_stats.ev_exit_profit_count += 1;
                        side_stats.pnl_ev_exit_profit += pnl;
                    }
                    (None, Some(CloseReason::EvExitLoss)) => {
                        side_stats.ev_exit_loss_count += 1;
                        side_stats.pnl_ev_exit_loss += pnl;
                    }
                    (None, None) => {
                        // unreachable: edge-case guard `!is_post_market_residual
                        // && reason.is_none()` выше выходит до flag-set'а.
                    }
                }
            }
        }
        _ => {
            *account.bankroll.write().await += pnl;
        }
    }

    let close_unix_ms = Some(crate::util::current_timestamp_ms());
    let interval_label = position_interval_label(&position_snapshot);
    let side_label = position_side_label(&position_snapshot);
    let open_unix_ms = position_snapshot
        .event_end_ms
        .map(|end_ms| end_ms - position_snapshot.event_remaining_ms_at_open);
    let graph_html_file_uri =
        crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri(&position_snapshot)
            .map(|bin_path| {
                crate::xframe_graph_dump::graph_html_trade_file_uri(
                    &bin_path,
                    open_unix_ms,
                    close_unix_ms,
                    Some(side_label),
                )
            })
            .unwrap_or_default();
    let market_id_str = position_snapshot.market_id.as_str();
    let event_remaining_ms_at_open = position_snapshot.event_remaining_ms_at_open;
    let event_remaining_ms_at_close = position_snapshot
        .event_end_ms
        .map(|end_ms| (end_ms - crate::util::current_timestamp_ms()).max(0))
        .unwrap_or(0);

    // `exit_reason` для CSV/лога: resolution → бинарный лейбл, after-sell →
    // [`trade_csv_close_reason_label`] от `reason`.
    let exit_reason_label: &str = match (token_won_resolution, reason) {
        (Some(true), _) => "ResolutionWin",
        (Some(false), _) => "ResolutionLoss",
        (None, Some(reason)) => trade_csv_close_reason_label(reason),
        (None, None) => "Unknown",
    };
    // `price_to_beat` / `final_price` в CSV: для resolution берём свежий снимок
    // из `MarketResolution` (authoritative, refine следующего окна уже пришёл);
    // для after-sell — старые значения из `OpenPosition`, выставленные в
    // `open_position` (могли быть None если на момент открытия PTB не было).
    let (csv_price_to_beat, csv_final_price) = match market_price_to_beat_and_final_price {
        Some((ptb, final_price)) => (Some(ptb), Some(final_price)),
        None => (
            position_snapshot.price_to_beat,
            position_snapshot.final_price,
        ),
    };

    match token_won_resolution {
        Some(token_won) => {
            let (ptb, final_price) = market_price_to_beat_and_final_price.expect(
                "token_won_resolution=Some ⇒ market_price_to_beat_and_final_price=Some",
            );
            crate::tee_println!(
                "[submit] pnl pos_id={pos_id} asset_id={asset_id} market_id={market_id_str} \
                 interval={interval_label} side={side_label} reason={exit_reason_label} \
                 role={fill_role} finalized_via={finalized_via} \
                 resolution(price_to_beat={ptb:.6} final_price={final_price:.6} token_won={token_won}) \
                 planned(USD={planned_entry_cost:.6} shares={planned_shares_held:.6} price={planned_buy_price:.6}) \
                 actual_buy(USD={actual_entry_cost:.6} shares={actual_shares_net:.6} price={actual_buy_price:.6}) \
                 sold(shares={shares_sold:.6} USD={usd_received:.6}) \
                 residual(shares={residual_shares:.6} payout={residual_payout:.6}) \
                 exit_price={exit_price:.6} fee_usdc={fee_usdc:.6} pnl={pnl:+.6}",
            );
        }
        None => {
            crate::tee_println!(
                "[submit] pnl pos_id={pos_id} asset_id={asset_id} market_id={market_id_str} \
                 interval={interval_label} side={side_label} reason={exit_reason_label} \
                 role={fill_role} finalized_via={finalized_via} \
                 planned(USD={planned_entry_cost:.6} shares={planned_shares_held:.6} price={planned_buy_price:.6}) \
                 actual_buy(USD={actual_entry_cost:.6} shares={actual_shares_net:.6} price={actual_buy_price:.6}) \
                 sell(price={exit_price:.6} shares={shares_sold:.6} USD={usd_received:.6}) \
                 residual_shares={residual_shares:.6} fee_usdc={fee_usdc:.6} pnl={pnl:+.6}",
            );
        }
    }

    let close_order_id_refs: Vec<&str> =
        close_order_ids.iter().map(|s| s.as_str()).collect();
    let trade_row = crate::trade_csv_log::TradeCsvRow {
        polymarket_url: &position_snapshot.polymarket_url,
        price_to_beat: csv_price_to_beat,
        final_price: csv_final_price,
        currency,
        interval: interval_label,
        side: side_label,
        market_id: market_id_str,
        asset_id,
        exit_reason: exit_reason_label,
        buy_price: actual_buy_price,
        raw_pred: position_snapshot.raw_pred_at_open,
        cal_pred: position_snapshot.cal_pred_at_open,
        kelly_f: position_snapshot.kelly_f_at_open,
        position_size: actual_entry_cost,
        shares_held: actual_shares_net,
        exit_price,
        fee_usdc,
        pnl,
        frames_held: position_snapshot.frames_held,
        p_win_ema_at_close: position_snapshot.p_win_ema,
        event_remaining_ms_at_open,
        event_remaining_ms_at_close,
        open_unix_ms,
        close_unix_ms,
        graph_html_file_uri: graph_html_file_uri.as_str(),
        pnl_top5_shap: position_snapshot.pnl_top5_shap_at_open.as_str(),
        pos_id: position_snapshot.id.as_str(),
        fill_role,
        finalized_via,
        planned_buy_price: Some(planned_buy_price),
        planned_shares_held: Some(planned_shares_held),
        planned_entry_cost: Some(planned_entry_cost),
        open_order_id: position_snapshot.open_order_id.as_deref(),
        tp_order_id: tp_order_id.as_deref(),
        close_order_ids: &close_order_id_refs,
    };
    crate::trade_csv_log::write_trade_csv_row(trade_row);
    crate::trade_csv_log::write_submit_trade_csv_row(trade_row);

    if let Some(project_manager) = project_manager {
        crate::xframe_graph_dump::spawn_partial_market_graph_html_for_close(
            project_manager.clone(),
            &position_snapshot,
        );
    }
}

