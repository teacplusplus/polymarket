//! Все ветки закрытия позиций в одном месте; ранее жили по `history_sim` /
//! `account` / `account_submit`. Каждая ветка обновляет `bankroll` и
//! [`crate::sim_stats::SideStats`] (`pnl_usd` / `trades` / `wins` / `losses` /
//! `closed_trade_entries` + специфичные счётчики), пишет
//! [`crate::trade_csv_log::TradeCsvRow`]; submit-ветка дополнительно пишет
//! `SUBMIT_TRADE_CSV_LOG` и спавнит partial-graph HTML.
//!
//! * [`close_position`] — единая ветка для **non-submit** закрытий:
//!     * backtest / real_sim рыночный выход (TP / SL / Timeout / EvExit) через
//!       bid-walk — вызывается из [`crate::history_sim::manage_positions`] после
//!       того как caller подготовил `gross_usdc` (либо
//!       [`gross_usdc_sell_take_profit`] для voluntary, либо
//!       прямого `book_fill_sell*`);
//!     * резолюция рынка ($1/$0 бинарный payout) — из
//!       [`crate::account::Account::resolve_pending_market_sync`]; caller
//!       подставляет `reason = CloseReason::Resolution{Win,Loss}` и
//!       `gross_usdc = shares_held` (win) / `0` (loss).
//! * [`close_position_after_submit`] — успешный SELL-fill в submit/mock
//!   (maker-TP `resting → fill` или taker-FAK SELL, который выбрал остаток до ~0),
//!   плюс post-market-end residual ($0/$1 через
//!   [`crate::project_manager::MarketResolution`]). Вызывается из
//!   [`crate::account_submit::spawn_open_buy_taker`] (после settle maker invoke,
//!   а также из заспавненного post-market-end таска) и
//!   [`crate::account_submit::spawn_sell_taker`] (после settle taker invoke, если
//!   `shares_remaining_to_sell ≈ 0`).

use crate::account::SharedAccount;
use crate::account_order::{
    OrderAmount, SingleOrderClobInvocationReport, invoke_settlement_ready,
    invoke_settlement_report,
};
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{
    CloseReason, OpenPosition, POLYMARKET_CRYPTO_TAKER_FEE_RATE, SharedOpenPosition,
    SIM_MAX_SLIPPAGE_FROM_L1_PCT, StrictBook, book_fill_sell, book_fill_sell_strict,
    position_interval_label, position_side_label, trade_csv_close_reason_label,
};
use crate::xframe::{SIZE, XFrame};
use crate::project_manager::ProjectManager;
use crate::sim_stats::SideStats;
use crate::xframe::Y_TRAIN_TAKE_PROFIT_PP;
use std::sync::Arc;

/// Единая ветка закрытия для backtest / real_sim (`SubmitMode::None`) и для
/// резолюции рынка: PnL/bankroll/SideStats/CSV в одном месте. Сама не лезет в
/// стакан и не дренит [`crate::account::Account::positions`] — caller обязан
/// (1) посчитать `gross_usdc` (bid-walk для рыночного выхода либо
/// `shares_held`/`0` для резолюции), (2) убрать позицию из своей структуры
/// (`remaining` в [`crate::history_sim::manage_positions`] или
/// `lane_positions.remove(...)` в
/// [`crate::account::Account::resolve_pending_market_sync`]).
///
/// Производные величины (`sell_price`, `fee_usdc`) вычисляются здесь по
/// `reason`, чтобы caller'у не приходилось дублировать формулы:
///
/// * `sell_price`:
///     * `ResolutionWin` → `1.0`, `ResolutionLoss` → `0.0`;
///     * иначе `(gross_usdc / shares_held).clamp(0.001, 0.999)`
///       (sell-VWAP по факту bid-walk'а).
/// * `fee_usdc`:
///     * `Resolution{Win,Loss}` → `0.0`;
///     * `TakeProfit` если `tp_target > best_bid_at_entry` (maker-сценарий) → `0.0`;
///     * иначе taker:
///       `shares_held * POLYMARKET_CRYPTO_TAKER_FEE_RATE * sell_price * (1 - sell_price)`.
///
/// `event_remaining_ms_at_close` для CSV: для backtest/real_sim `manage_positions`
/// — `frame.event_remaining_ms` (текущий шаг симуляции); для резолюции — `0`
/// (рынок уже завершился, `close_unix_ms == event_end_ms`).
///
/// `SideStats`-ветки совпадают с `match reason` (см. ниже). `pos.final_price`
/// в CSV — то, что caller записал в позицию ДО вызова (для резолюции из
/// [`crate::account::Account::resolve_pending_market_sync`] это означает
/// `pos_arc.write().await.final_price = Some(...)` перед `close_position`).
///
/// **Локи (async).** Внутри читаем `pos_arc.read().await` (snapshot для CSV /
/// формул), потом берём `account.bankroll.write().await` для applies — caller
/// НЕ должен пред-захватывать ни тот, ни другой (иначе deadlock).
pub(crate) async fn close_position(
    account: &SharedAccount,
    pos_arc: &SharedOpenPosition,
    stats: &mut SideStats,
    reason: &CloseReason,
    gross_usdc: Option<f64>,
    event_remaining_ms_at_close: i64,
) {
    let pos = pos_arc.read().await;
    // `gross_usdc`:
    //   * `Some(g)` — рыночный выход (`manage_positions` уже посчитал bid-walk);
    //   * `None`   — резолюция: gross механически выводится из `reason`
    //     (`ResolutionWin → shares_held`, `ResolutionLoss → 0`), чтобы
    //     resolve_pending_market_sync не дублировал чтение `pos.shares_held`.
    let gross_usdc = gross_usdc.unwrap_or_else(|| match reason {
        CloseReason::ResolutionWin => pos.shares_held,
        CloseReason::ResolutionLoss => 0.0,
        _ => {
            debug_assert!(
                false,
                "gross_usdc=None требует resolution-reason, получили {reason:?}"
            );
            0.0
        }
    });
    let sell_price = match reason {
        CloseReason::ResolutionWin => 1.0,
        CloseReason::ResolutionLoss => 0.0,
        _ if pos.shares_held > 0.0 => {
            (gross_usdc / pos.shares_held).clamp(0.001, 0.999)
        }
        _ => 0.0,
    };
    // Maker fee = 0 только если TP-таргет лежит выше bid на входе (выставляемся
    // как `resting → fill`); таргета нет (best_bid_at_entry=None) — оптимистично
    // считаем maker'ом (то же поведение, что и раньше в `close_position_market_exit`).
    let voluntary_is_maker = match reason {
        CloseReason::TakeProfit => {
            let tp_target = (pos.buy_price + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999);
            match pos.best_bid_at_entry {
                Some(best_bid) => tp_target > best_bid,
                None => true,
            }
        }
        _ => false,
    };
    let fee_usdc = match reason {
        CloseReason::ResolutionWin | CloseReason::ResolutionLoss => 0.0,
        _ if voluntary_is_maker => 0.0,
        _ => pos.shares_held
            * POLYMARKET_CRYPTO_TAKER_FEE_RATE
            * sell_price
            * (1.0 - sell_price),
    };

    let net_usdc = gross_usdc - fee_usdc;
    let pnl = net_usdc - pos.position_size;
    // Bankroll write берём ВНУТРИ функции под коротким локом — caller не должен
    // держать `account.bankroll.write().await` (deadlock). Лок-порядок в обоих
    // call-site (manage_positions / resolve_pending_market_sync) совпадает с
    // [`crate::real_sim::tick_once`]: `pos → bankroll`.
    *account.bankroll.write().await += pnl;
    stats.pnl_usd += pnl;
    stats.trades += 1;
    if pnl >= 0.0 {
        stats.wins += 1;
    } else {
        stats.losses += 1;
    }
    stats.fees_paid += fee_usdc;
    // Резолюция тоже попадает в `closed_trade_entries` — replay'ю калибровки
    // нужен и pnl-знак на резолюционных закрытиях.
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
        CloseReason::ResolutionWin => {
            stats.resolution_win += 1;
            stats.pnl_resolution_win += pnl;
            if pnl >= 0.0 {
                stats.resolution_win_profit += 1;
            } else {
                stats.resolution_win_loss += 1;
            }
        }
        CloseReason::ResolutionLoss => {
            stats.resolution_loss += 1;
            stats.pnl_resolution_loss += pnl;
        }
    }

    let close_unix_ms = pos
        .event_end_ms
        .map(|end_ms| end_ms - event_remaining_ms_at_close);
    let interval_label = position_interval_label(&pos);
    let side_label = position_side_label(&pos);
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
                    Some(side_label),
                )
            })
            .unwrap_or_default();
    crate::trade_csv_log::write_trade_csv_row(crate::trade_csv_log::TradeCsvRow {
        polymarket_url: &pos.polymarket_url,
        price_to_beat: pos.price_to_beat,
        final_price: pos.final_price,
        currency: &pos.currency,
        interval: interval_label,
        side: side_label,
        market_id: &pos.market_id,
        asset_id: &pos.asset_id,
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
        event_remaining_ms_at_close,
        open_unix_ms,
        close_unix_ms,
        graph_html_file_uri: graph_html_file_uri.as_str(),
        pnl_top5_shap: pos.pnl_top5_shap_at_open.as_str(),
        pos_id: pos.id.as_str(),
        finalized_via: "",
        planned_buy_price: None,
        planned_shares_held: None,
        planned_entry_cost: None,
        planned_fee_usdc: None,
        entry_fee_usdc: None,
        open_order_id: None,
        tp_order_id: None,
        close_order_ids: &[],
    });
}

/// Закрытие позиции после SELL-fill'ов в submit/mock-режиме — общая ветка для
/// двух caller'ов:
/// * [`crate::account_submit::spawn_open_buy_taker`] — после settle maker-TP
///   invoke с `success && !partial` (целиком закрылись на maker'е);
///   `reason = CloseReason::TakeProfit`,
///   `finalized_via = "maker_tp_fill"`.
/// * [`crate::account_submit::spawn_sell_taker`] — после settle taker-FAK invoke,
///   когда `shares_remaining_to_sell ≤
///   [`crate::account_submit::CLOSE_AFTER_SELL_REMAINING_SHARES_TOLERANCE`]`
///   (taker дошёл до ~0, возможно вместе с partial-maker'ом); `reason` —
///   фактический (TP / SL / Timeout / EvExit),
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
/// (`reason / finalized_via`).
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
/// residual). Caller post-market-residual'а сам подымает
/// [`crate::project_manager::MarketResolution`], определяет UP/DOWN-победителя
/// и нашу сторону, прописывает свежие `price_to_beat` / `final_price` в позицию
/// и передаёт сюда уже готовый [`CloseReason::ResolutionWin`] /
/// [`CloseReason::ResolutionLoss`].
///
/// **Path detection — по `reason`:**
///
/// * [`CloseReason::ResolutionWin`] / [`CloseReason::ResolutionLoss`] —
///   post-market-end резолюция. Residual = `actual_shares_net - shares_sold`
///   оценивается бинарно: `$1`/шер при `ResolutionWin`, `$0` при
///   `ResolutionLoss`. PNL = `usd_received + residual_payout -
///   actual_entry_cost`; SideStats апдейтит `resolution_win` /
///   `resolution_loss` / `pnl_resolution_*`. `exit_reason` в CSV — лейбл
///   `"ResolutionWin"` / `"ResolutionLoss"` через
///   [`trade_csv_close_reason_label`].
/// * Остальные ([`CloseReason::TakeProfit`] / [`CloseReason::StopLoss`] /
///   [`CloseReason::Timeout`] / [`CloseReason::EvExitProfit`] /
///   [`CloseReason::EvExitLoss`]) — after-sell: позиция полностью распродана
///   через maker TP / taker FAK. PNL = `usd_received - actual_entry_cost`;
///   SideStats апдейтит соответствующую ветку (`tp_count` / `sl_count` /
///   `timeout_count` / `ev_exit_*`). Bail если SELL-fills нулевые
///   (флаг НЕ взводим — пусть future-callback попробует).
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
    reason: &CloseReason,
    finalized_via: &'static str,
) {
    // Pre-flight snapshot: все reads + aggregation БЕЗ установки idempotency
    // флага. Параллельные callers могут продублировать read+агрегацию — это
    // безопасно (нет write'ов); flag-set ниже гарантирует, что финализирует
    // только победитель гонки.
    let position_snapshot = position.read().await.clone();
    let pos_id = position_snapshot.id.clone();
    let asset_id = position_snapshot.asset_id.as_str();
    let currency = position_snapshot.currency.as_str();
    let planned_buy_price = position_snapshot.planned_buy_price;
    let planned_shares_held = position_snapshot.planned_shares_held;
    let planned_entry_cost = position_snapshot.planned_entry_cost;
    let actual_buy_price = position_snapshot.buy_price;
    let actual_entry_cost = position_snapshot.position_size;

    // Path detection — по `reason`: Resolution-варианты приходят только от
    // post-market-end caller'а, который уже подтянул [`MarketResolution`] и
    // вычислил UP/DOWN-победителя. `token_won_resolution.is_some()`
    // эквивалентно «residual считаем как $1 / $0».
    let token_won_resolution: Option<bool> = match reason {
        CloseReason::ResolutionWin => Some(true),
        CloseReason::ResolutionLoss => Some(false),
        CloseReason::TakeProfit
        | CloseReason::StopLoss
        | CloseReason::Timeout
        | CloseReason::EvExitProfit
        | CloseReason::EvExitLoss => None,
    };

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
    let mut sell_fee_usdc = 0.0_f64;
    let mut tp_order_id: Option<String> = None;
    let mut close_order_ids: Vec<String> = Vec::new();

    // Maker (TP) у Polymarket всегда 0; taker (FAK SELL) — `report.fee_paid_usdc`
    // от mock'а (`polymarket_taker_fee_usd(gross, vwap)`) или real CLOB-агрегатора
    // (`Σ trade.size × trade.price × trade.fee_rate_bps / 10_000` по on-chain
    // settled trades). В обоих случаях это уже корректный USD-значок, складываем
    // напрямую в `sell_fee_usdc` для общего `side_stats.fees_paid += fee_usdc` ниже.
    let accumulate = |report: &SingleOrderClobInvocationReport,
                      shares_acc: &mut f64,
                      usd_acc: &mut f64,
                      fee_acc: &mut f64| {
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
            let fee = report.fee_paid_usdc;
            if fee.is_finite() && fee >= 0.0 {
                *fee_acc += fee;
            }
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
            accumulate(
                &report,
                &mut shares_sold,
                &mut usd_received,
                &mut sell_fee_usdc,
            );
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
            accumulate(
                &report,
                &mut shares_sold,
                &mut usd_received,
                &mut sell_fee_usdc,
            );
        }
    }

    let residual_shares = (actual_shares_net - shares_sold).max(0.0);

    // After-sell path требует ненулевых SELL-fills (`pnl = usd_received - entry`
    // даст шум если 0); resolution-path допускает 0 (residual подберёт всё).
    // Бэйлим ДО взвода флага idempotency, чтобы не блокировать future-триггеры.
    let has_valid_sell_fills = shares_sold.is_finite()
        && shares_sold > 0.0
        && usd_received.is_finite()
        && usd_received > 0.0;
    if token_won_resolution.is_none() && !has_valid_sell_fills {
        crate::tee_eprintln!(
            "[submit] pnl pos_id={pos_id} reason={reason:?}: \
             сумма SELL-fills нулевая/невалидная (shares={shares_sold:.6} USD={usd_received:.6}) — \
             лог пропускаем (флаг НЕ взводим)",
        );
        return;
    }

    // Атомарный check-and-set флага идемпотентности (под write-lock без await
    // внутри — гонка с другими caller'ами разрешена однозначно).
    {
        let mut open_position = position.write().await;
        if open_position.close_after_submit_finalized {
            crate::tee_println!(
                "[submit] pnl pos_id={pos_id} reason={reason:?}: \
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
    {
        let mut pending_guard = account.pending_close_positions.write().await;
        for lane_pending in pending_guard.values_mut() {
            lane_pending.remove(&pos_id);
        }
    }

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
            "[submit] pnl pos_id={pos_id} reason={reason:?}: \
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

    // Сумма SELL-fee для `SideStats.fees_paid` лейна. Maker (TP) — fee = 0;
    // taker (FAK SELL) — Polymarket вычитает fee из `taking_amount` (USDC),
    // оставляя `making_amount` (gross shares) нетронутым. Источник —
    // [`SingleOrderClobInvocationReport::fee_paid_usdc`] (mock считает явно через
    // `polymarket_taker_fee_usd`; real CLOB — `Σ trade.size × trade.price ×
    // trade.fee_rate_bps / 10_000` по on-chain settled trades).
    //
    // Для resolution-пути занулять fee нельзя: `sell_fee_usdc` накапливается
    // **только** по успешно settled SELL invoke'ам (см. `accumulate` выше), а
    // residual-payout ($1/$0 за непроданные shares) идёт мимо CLOB — за него и
    // так ничего не считается. Если до маркет-энда часть позиции уже
    // распродалась через maker TP / taker FAK — комиссия с тех fill'ов
    // реальная и должна остаться в `fees_paid`.
    let fee_usdc: f64 = sell_fee_usdc;
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
    let our_side =
        CurrencyUpDownOutcome::from_i32(position_snapshot.currency_up_down_outcome_at_open);
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
                // Reason carries path: ResolutionWin/Loss → резолюционные
                // счётчики; остальные → reason-based ветка (после-sell).
                match reason {
                    CloseReason::TakeProfit => {
                        side_stats.tp_count += 1;
                        side_stats.pnl_tp += pnl;
                    }
                    CloseReason::StopLoss => {
                        side_stats.sl_count += 1;
                        side_stats.pnl_sl += pnl;
                    }
                    CloseReason::Timeout => {
                        side_stats.timeout_count += 1;
                        side_stats.pnl_timeout += pnl;
                    }
                    CloseReason::EvExitProfit => {
                        side_stats.ev_exit_profit_count += 1;
                        side_stats.pnl_ev_exit_profit += pnl;
                    }
                    CloseReason::EvExitLoss => {
                        side_stats.ev_exit_loss_count += 1;
                        side_stats.pnl_ev_exit_loss += pnl;
                    }
                    CloseReason::ResolutionWin => {
                        side_stats.resolution_win += 1;
                        side_stats.pnl_resolution_win += pnl;
                        if pnl >= 0.0 {
                            side_stats.resolution_win_profit += 1;
                        } else {
                            side_stats.resolution_win_loss += 1;
                        }
                    }
                    CloseReason::ResolutionLoss => {
                        side_stats.resolution_loss += 1;
                        side_stats.pnl_resolution_loss += pnl;
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

    let exit_reason_label = trade_csv_close_reason_label(reason);
    // `price_to_beat` / `final_price` берём напрямую из позиции: post-market
    // caller прописал свежие значения из [`crate::project_manager::MarketResolution`]
    // через `position.write()` ДО вызова этой функции; для after-sell — те,
    // что выставил `crate::history_sim::open_position` (могут быть None если
    // на момент открытия PTB не было).
    match token_won_resolution {
        Some(token_won) => {
            crate::tee_println!(
                "[submit] pnl pos_id={pos_id} asset_id={asset_id} market_id={market_id_str} \
                 interval={interval_label} side={side_label} reason={exit_reason_label} \
                 finalized_via={finalized_via} \
                 resolution(price_to_beat={:?} final_price={:?} token_won={token_won}) \
                 planned(USD={planned_entry_cost:.6} shares={planned_shares_held:.6} price={planned_buy_price:.6}) \
                 actual_buy(USD={actual_entry_cost:.6} shares={actual_shares_net:.6} price={actual_buy_price:.6}) \
                 sold(shares={shares_sold:.6} USD={usd_received:.6}) \
                 residual(shares={residual_shares:.6} payout={residual_payout:.6}) \
                 exit_price={exit_price:.6} fee_usdc={fee_usdc:.6} pnl={pnl:+.6}",
                position_snapshot.price_to_beat,
                position_snapshot.final_price,
            );
        }
        None => {
            crate::tee_println!(
                "[submit] pnl pos_id={pos_id} asset_id={asset_id} market_id={market_id_str} \
                 interval={interval_label} side={side_label} reason={exit_reason_label} \
                 finalized_via={finalized_via} \
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
        price_to_beat: position_snapshot.price_to_beat,
        final_price: position_snapshot.final_price,
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
        finalized_via,
        planned_buy_price: Some(planned_buy_price),
        planned_shares_held: Some(planned_shares_held),
        planned_entry_cost: Some(planned_entry_cost),
        planned_fee_usdc: Some(position_snapshot.planned_fee_usdc),
        entry_fee_usdc: Some(position_snapshot.entry_fee_usdc),
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

/// Gross USDC при TP: если полный sell-walk даёт порог TP — можно обойти cap к
/// L1 (см. [`crate::history_sim::sell_gate`]). Используется только
/// [`crate::history_sim::manage_positions`] → [`close_position`] при
/// voluntary-exit (`reason.is_voluntary_exit()`) с целью разрешить более глубокий
/// уход за L1, если итоговый VWAP всё ещё ≥ TP-таргета.
pub(crate) fn gross_usdc_sell_take_profit(
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
