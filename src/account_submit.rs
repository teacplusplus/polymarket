//! `RealSimWithSubmit`: CLOB ордеры через [`crate::account_order`], подтверждение прежде всего WS
//! ([`crate::account_ws`]), дополнительно polling `client.order` ([`spawn_polling_verify`]).
//! Таски через `spawn` без долгих локов на `positions`/`closing`; дедуп TP/cancel/closing —
//! атомики/флаги на позиции до HTTP. После BUY/close/TP — poll до терминального статуса или
//! `event_end_ms`/`POLL_TIMEOUT_SEC`, затем [`apply_order_status_from_polling`] (как WS).
//!
//! `event_end_ms` из [`crate::history_sim::OpenPosition`] всегда пробрасывается в
//! [`crate::account_order::PostOrderRequest::market_end_unix_ms`] для POST здесь (дедлайн invoke/poll).
//!
//! Способ исполнения CLOB-ордеров выбирается параметром [`SubmitMode`]:
//! * [`SubmitMode::Submit`] — реальный CLOB ([`crate::account_order`]);
//! * [`SubmitMode::Mock`] — фейковая симуляция по WS-стакану
//!   ([`crate::account_mock_order`]);
//! * [`SubmitMode::None`] — ничего не вызываем (чистое виртуальное исполнение,
//!   как в `history_sim` backtest).
use crate::account::SharedAccount;
use crate::account_order::{
    CancelOrderRequest, CancelOrderResult, OrderAmount, OrderRole, PostOrderRequest,
    SingleOrderClobInvocationReport, invoke_settlement_ready, invoke_settlement_report,
    invoke_settlement_watch, wait_invoke_settlement,
};
use crate::history_sim::{
    CloseReason, ClosingPosition, SIM_MAX_SLIPPAGE_FROM_L1_PCT, SharedClosingPosition,
    SharedOpenPosition, StrictBook,
};
use crate::project_manager::ProjectManager;
use crate::xframe::Y_TRAIN_TAKE_PROFIT_PP;
use polymarket_client_sdk::clob::types::Side;
use std::sync::Arc;
use std::time::Duration;

/// Способ исполнения CLOB-ордеров [`spawn_open_buy_taker`] / [`spawn_sell_taker`] /
/// [`spawn_cancel_order`]. При `None` `spawn_*` выходит ранним return; в
/// [`crate::real_sim::tick_once`] также отключает submit-window-гейт.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitMode {
    /// Боевые методы [`crate::account_order::post_order_on_clob`] /
    /// [`crate::account_order::cancel_order_on_clob`].
    Submit,
    /// Мок [`crate::account_mock_order::post_order_on_clob`] /
    /// [`crate::account_mock_order::cancel_order_on_clob`] — данные из WS-стакана
    /// [`crate::project_manager::ProjectManager::last_snapshot_by_asset_id`].
    Mock,
    /// CLOB не дёргаем — чистое виртуальное исполнение (`spawn_*` ранний return).
    None,
}
/// Один REST/SUBMIT timeout — также для [`crate::account_order_completion`] и invoke-poll (через дубль константы там).
pub(crate) const ORDER_HTTP_TIMEOUT_SEC: u64 = 10;
/// Повторы taker SELL при SL/timeout/ev-exit (FAK без матча и т.п.), как
/// [`UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS`] в live duel test.
pub(crate) const TAKER_SELL_ATTEMPTS: u32 = 10;

fn is_past_market_end(event_end_unix_ms: Option<i64>) -> bool {
    event_end_unix_ms.is_some_and(|end_ms| crate::util::current_timestamp_ms() >= end_ms)
}

async fn taker_sell_attempt_backoff(attempt: u32) {
    if attempt > 3 {
        tokio::time::sleep(Duration::from_secs(u64::from(attempt - 3))).await;
    }
}

/// Диспатч `post_order_on_clob` по [`SubmitMode`]: единая точка ветвления real↦mock, чтобы
/// тело `spawn_open_buy_taker` / `spawn_sell_taker` не дублировало `match`. `None` не ожидается —
/// `spawn_*` уже выходит ранним return при `submit_mode == SubmitMode::None`.
async fn post_order_on_clob(
    account: &SharedAccount,
    project_manager: Option<&Arc<ProjectManager>>,
    submit_mode: SubmitMode,
    request: PostOrderRequest,
    invoke: crate::account_order::SingleOrderInvokeCb,
) -> anyhow::Result<Option<String>> {
    match submit_mode {
        SubmitMode::Submit => {
            crate::account_order::post_order_on_clob(account, None, request, invoke).await
        }
        SubmitMode::Mock => {
            crate::account_mock_order::post_order_on_clob(
                account,
                project_manager,
                request,
                invoke,
            )
            .await
        }
        SubmitMode::None => unreachable!(
            "post_order_on_clob (account_submit) вызывается из spawn_* только при \
             submit_mode != SubmitMode::None"
        ),
    }
}

/// Диспатч `cancel_order_on_clob` по [`SubmitMode`]: симметричный двойник
/// [`post_order_on_clob`]. `None` не ожидается — [`spawn_cancel_order`] выходит ранним return
/// при `submit_mode == SubmitMode::None`.
async fn cancel_order_on_clob(
    account: &SharedAccount,
    project_manager: Option<&Arc<ProjectManager>>,
    submit_mode: SubmitMode,
    request: CancelOrderRequest,
) -> anyhow::Result<CancelOrderResult> {
    match submit_mode {
        SubmitMode::Submit => {
            crate::account_order::cancel_order_on_clob(account, None, request).await
        }
        SubmitMode::Mock => {
            crate::account_mock_order::cancel_order_on_clob(account, project_manager, request)
                .await
        }
        SubmitMode::None => unreachable!(
            "cancel_order_on_clob (account_submit) вызывается из spawn_* только при \
             submit_mode != SubmitMode::None"
        ),
    }
}

pub(crate) fn spawn_cancel_order(
    account: SharedAccount,
    project_manager: Option<Arc<ProjectManager>>,
    position: SharedOpenPosition,
    submit_mode: SubmitMode,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if submit_mode == SubmitMode::None {
            return;
        }

        let (position_id, maker_tp_position) = {
            let open_position = position.read().await;
            (
                open_position.id.clone(),
                open_position.maker_tp_position.clone(),
            )
        };

        let Some(maker_tp_position) = maker_tp_position.and_then(|weak| weak.upgrade()) else {
            return;
        };
        let (maker_tp_order_id, maker_already_canceled) = {
            let maker_closing_read = maker_tp_position.read().await;
            (
                maker_closing_read
                    .order_id
                    .clone()
                    .filter(|order_id| !order_id.trim().is_empty()),
                maker_closing_read.canceled,
            )
        };
        let Some(order_id) = maker_tp_order_id else { return };
        if maker_already_canceled {
            return;
        }
        {
            let mut maker_closing_write = maker_tp_position.write().await;
            maker_closing_write.canceled = true;
        }

        let cancel_request = CancelOrderRequest {
            order_id: order_id.clone(),
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
        };
        let cancel_outcome = cancel_order_on_clob(
            &account,
            project_manager.as_ref(),
            submit_mode,
            cancel_request,
        )
        .await;

        match cancel_outcome {
            Ok(cancel_result) => {
                crate::tee_println!(
                    "[submit/{submit_mode:?}] cancel order pos_id={position_id}: cancel maker TP \
                     order_id={} canceled={} err={:?}",
                    cancel_result.order_id,
                    cancel_result.canceled,
                    cancel_result.error_msg,
                );
                if cancel_result.canceled {
                    let mut maker_closing_write = maker_tp_position.write().await;
                    maker_closing_write.order_id = None;
                }
            }
            Err(cancel_err) => {
                crate::tee_eprintln!(
                    "[submit/{submit_mode:?}] cancel order pos_id={position_id}: cancel maker TP \
                     order_id={order_id}: {cancel_err:#}",
                );
            }
        }
    })
}

pub(crate) fn spawn_sell_taker(
    account: SharedAccount,
    project_manager: Option<Arc<ProjectManager>>,
    position: SharedOpenPosition,
    exit_price: f64,
    reason: CloseReason,
    strict_book: Option<StrictBook>,
    submit_mode: SubmitMode,
) {
    if submit_mode == SubmitMode::None {
        return;
    }
    tokio::spawn(async move {
        let open_position = position.read().await;
        let position_id = open_position.id.clone();
        let asset_id = open_position.asset_id.clone();
        let event_end_unix_ms = open_position.event_end_ms;
        let maker_tp_position = open_position.maker_tp_position.clone();
        let mut open_buy_invoke = open_position.open_buy_invoke.clone();
        drop(open_position);

        let buy_invoke_report = match open_buy_invoke.as_mut() {
            Some(watch) => {
                if !invoke_settlement_ready(watch) {
                    let buy_invoke_wait =
                        invoke_wait_until_market_end_plus(event_end_unix_ms);
                    wait_invoke_settlement(watch, buy_invoke_wait).await
                } else {
                    invoke_settlement_report(watch)
                }
            }
            None => None,
        };
        let shares_bought_net = match buy_invoke_report.as_ref() {
            Some(invoke_report) if invoke_report.success => {
                match invoke_report.taking_amount {
                    OrderAmount::Shares(shares) => {
                        if shares.is_finite() && shares > 0.0 {
                            Some(shares)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(shares_bought_net) = shares_bought_net else {
            crate::tee_eprintln!(
                "[submit] sell taker pos_id={position_id}: нет NET shares в open_buy_invoke — пропуск",
            );
            return;
        };

        let mut shares_sold_by_maker = 0.0_f64;

        if reason == CloseReason::TakeProfit || reason == CloseReason::EvExitProfit {
            if let Some(maker_tp_weak) = maker_tp_position.as_ref() {
                if let Some(maker_tp_arc) = maker_tp_weak.upgrade() {
                    let maker_closing = maker_tp_arc.read().await;
                    let active_maker_tp = maker_closing
                        .order_id
                        .as_ref()
                        .is_some_and(|id| !id.trim().is_empty())
                        && !maker_closing.canceled
                        && maker_closing.invoke_settle.as_ref().is_some_and(|watch| {
                            !invoke_settlement_ready(watch)
                        });
                    if active_maker_tp {
                        crate::tee_println!(
                            "[submit] sell taker pos_id={position_id}: TakeProfit — maker TP на CLOB \
                             (order_id есть, invoke не settled) — пропуск cancel/taker",
                        );
                        return;
                    }
                }
            }
        }

        spawn_cancel_order(
            account.clone(),
            project_manager.clone(),
            position.clone(),
            submit_mode,
        )
        .await
        .ok();

        if let Some(maker_tp_position) = maker_tp_position {
            if let Some(maker_tp_position) = maker_tp_position.upgrade() {
                let mut maker_invoke_watch = {
                    let maker_closing_read = maker_tp_position.read().await;
                    maker_closing_read.invoke_settle.clone()
                };

                if let Some(watch) = maker_invoke_watch.as_mut() {
                    if !invoke_settlement_ready(watch) {
                        let maker_invoke_wait =
                            invoke_wait_until_market_end_plus(event_end_unix_ms);
                        let _ =
                            wait_invoke_settlement(watch, maker_invoke_wait).await;
                    }
                    if let Some(invoke_report) = invoke_settlement_report(watch) {
                        if invoke_report.success {
                            match invoke_report.making_amount {
                                OrderAmount::Shares(shares)
                                    if shares.is_finite() && shares > 0.0 =>
                                {
                                    shares_sold_by_maker = shares;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        let shares_remaining = (shares_bought_net - shares_sold_by_maker).max(0.0);
        if !(shares_remaining.is_finite() && shares_remaining > 0.0) {
            crate::tee_eprintln!(
                "[submit] sell taker pos_id={position_id}: shares_remaining={shares_remaining:.6} \
                 после BUY {shares_bought_net:.6} − maker {shares_sold_by_maker:.6} — пропуск",
            );
            return;
        }

        let sell_invoke_wait = invoke_wait_until_market_end_plus(event_end_unix_ms);

        for attempt in 1..=TAKER_SELL_ATTEMPTS {
            if is_past_market_end(event_end_unix_ms) {
                crate::tee_println!(
                    "[submit] sell taker pos_id={position_id}: event_end_ms достигнут — прекращаем taker SELL",
                );
                break;
            }

            // `block_on_pending_invokes=true`: метод сам дождётся pending taker
            // SELL invoke'ов (и maker TP / BUY если они в полёте), иначе мы
            // рискуем повторно отправить SELL на уже проданные шеры.
            //
            // Клонируем `OpenPosition` перед вызовом, чтобы не держать
            // `position` read-lock на время ожидания (внутри одного `attempt`
            // ниже идёт `position.write().await` для записи нового
            // `taker_closing` — read-lock через await его бы заблокировал).
            let position_snapshot = position.read().await.clone();
            let shares_remaining = match position_snapshot
                .shares_remaining_to_sell(true)
                .await
            {
                Ok(Some(n)) => n,
                Ok(None) => {
                    crate::tee_eprintln!(
                        "[submit] sell taker pos_id={position_id}: BUY-invoke не settled \
                         с NET shares — нечего продавать, прекращаем retry",
                    );
                    break;
                }
                Err(err) => {
                    crate::tee_eprintln!(
                        "[submit] sell taker pos_id={position_id}: \
                         shares_remaining_to_sell ошибка — invoke '{}' не settled даже после \
                         ожидания до event_end+{ORDER_HTTP_TIMEOUT_SEC}s, прекращаем retry",
                        err.which,
                    );
                    break;
                }
            };
            // CLOB-lot Polymarket = 0.01; `floor` чтобы не превысить остаток и
            // не словить `OrderStatusType::InsufficientBalance`.
            let shares_to_sell = (shares_remaining * 100.0).floor() / 100.0;
            if !(shares_to_sell > 0.0 && shares_to_sell.is_finite()) {
                break;
            }

            crate::tee_println!(
                "[submit] sell taker pos_id={position_id} asset_id={asset_id} reason={reason:?}: \
                 taker FAK SELL shares={shares_to_sell:.2} \
                 exit_price≈{exit_price:.6} попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
            );



            let (sell_invoke_tx, mut sell_invoke_rx) = invoke_settlement_watch();  
            
            let taker_closing = {
                let mut open_position = position.write().await;
                let taker_closing: SharedClosingPosition =
                    Arc::new(tokio::sync::RwLock::new(ClosingPosition {
                        position: position.clone(),
                        reason: reason.clone(),
                        order_id: None,
                        invoke_settle: Some(sell_invoke_rx.clone()),
                        canceled: false,
                        created_unix_ms: crate::util::current_timestamp_ms(),
                    }));

                open_position
                    .taker_positions
                    .push(Arc::downgrade(&taker_closing));
                taker_closing
            };

            let sell_post_request = PostOrderRequest {
                asset_id: asset_id.clone(),
                side: Side::Sell,
                role: OrderRole::Taker,
                amount: OrderAmount::Shares(shares_to_sell),
                price: None,
                max_slippage_pp: Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT),
                expiration: None,
                market_end_unix_ms: event_end_unix_ms,
                timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
                strict_book: strict_book.clone(),
            };
            let sell_invoke_cb: crate::account_order::SingleOrderInvokeCb =
                Box::new(move |sell_invoke_report| {
                    let _ = sell_invoke_tx.send(Some(sell_invoke_report));
                });
            let sell_post_result = post_order_on_clob(
                &account,
                project_manager.as_ref(),
                submit_mode,
                sell_post_request,
                sell_invoke_cb,
            )
            .await;

            let sell_order_id = match sell_post_result {
                Ok(Some(order_id)) if !order_id.trim().is_empty() => Some(order_id),
                Ok(Some(_)) => {
                    crate::tee_eprintln!(
                        "[submit] sell taker pos_id={position_id}: POST вернул пустой order_id \
                         попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                    );
                    taker_sell_attempt_backoff(attempt).await;
                    continue;
                }
                Ok(None) => {
                    crate::tee_eprintln!(
                        "[submit] sell taker pos_id={position_id}: POST Ok(None) \
                         попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                    );
                    taker_sell_attempt_backoff(attempt).await;
                    continue;
                }
                Err(post_err) => {
                    crate::tee_eprintln!(
                        "[submit] sell taker pos_id={position_id}: POST err={post_err:#} \
                         попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                    );
                    taker_sell_attempt_backoff(attempt).await;
                    continue;
                }
            };
            {
                let mut closing_write = taker_closing.write().await;
                closing_write.order_id = sell_order_id.clone();
            }
            let sell_invoke_report = match wait_invoke_settlement(
                &mut sell_invoke_rx,
                sell_invoke_wait,
            )
            .await
            {
                Some(invoke_report) => invoke_report,
                None => {
                    crate::tee_eprintln!(
                        "[submit] sell taker pos_id={position_id} order_id={sell_order_id:?}: \
                         invoke timeout {sell_invoke_wait:?} попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                    );
                    taker_sell_attempt_backoff(attempt).await;
                    continue;
                }
            };

            crate::tee_println!(
                "[submit] sell taker pos_id={position_id} asset_id={asset_id}: invoke settle \
                 попытка {attempt}/{TAKER_SELL_ATTEMPTS} success={} partial={} \
                 order_id={:?}; making={:?}, taking={:?}, err={:?}",
                sell_invoke_report.success,
                sell_invoke_report.partial,
                sell_invoke_report.order_id,
                sell_invoke_report.making_amount,
                sell_invoke_report.taking_amount,
                sell_invoke_report.error_msg,
            );

            taker_sell_attempt_backoff(attempt).await;
        }
    });
}


pub(crate) fn spawn_open_buy_taker(
    account: SharedAccount,
    project_manager: Option<Arc<ProjectManager>>,
    position: SharedOpenPosition,
    price: Option<f64>,
    strict_book: Option<StrictBook>,
    submit_mode: SubmitMode,
) {
    if submit_mode == SubmitMode::None {
        return;
    }
    tokio::spawn(async move {
        let (asset_id, amount, event_end_ms, pos_id) = {
            let p = position.read().await;
            (
                p.asset_id.clone(),
                p.position_size,
                p.event_end_ms,
                p.id.clone(),
            )
        };

        if !(amount > 0.0 && amount.is_finite()) {
            crate::tee_eprintln!(
                "[submit] open BUY taker pos_id={pos_id}: невалидный amount={amount} — OpenFailed",
            );
            return;
        }
        // Как в duel: USDC к центам вверх (`ceil`), иначе CLOB видит лишние знаки.
        let amount = (amount * 100.0).ceil() / 100.0;

        let min_order_size_shares = strict_book
            .as_ref()
            .and_then(|b| b.min_order_size)
            .filter(|m| m.is_finite() && *m > 0.0);

        let (invoke_tx, mut invoke_rx) = invoke_settlement_watch();
        {
            let mut open_position = position.write().await;
            open_position.open_buy_invoke = Some(invoke_rx.clone());
        }
        let buy_post_request = PostOrderRequest {
            asset_id: asset_id.clone(),
            side: Side::Buy,
            role: OrderRole::Taker,
            amount: OrderAmount::UsdNotional(amount),
            price,
            max_slippage_pp: None,
            expiration: None,
            market_end_unix_ms: event_end_ms,
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            strict_book,
        };
        let buy_invoke_cb: crate::account_order::SingleOrderInvokeCb =
            Box::new(move |buy_rep| {
                let _ = invoke_tx.send(Some(buy_rep));
            });
        let post_result = post_order_on_clob(
            &account,
            project_manager.as_ref(),
            submit_mode,
            buy_post_request,
            buy_invoke_cb,
        )
        .await;

        let http_order_id = match post_result {
            Ok(Some(order_id)) => {
                {
                    let mut p = position.write().await;
                    p.open_order_id = Some(order_id.clone());
                }
                crate::tee_println!(
                    "[submit] open BUY taker pos_id={pos_id} asset_id={asset_id}: HTTP POST ok order_id={order_id}",
                );
                Some(order_id)
            }
            Ok(None) => {
                crate::tee_eprintln!(
                    "[submit] open BUY taker pos_id={pos_id} asset_id={asset_id}: HTTP POST отклонён — OpenFailed",
                );
                return;
            }
            Err(err) => {
                crate::tee_eprintln!(
                    "[submit] open BUY taker pos_id={pos_id} asset_id={asset_id}: post_order_on_clob err={err:#} — OpenFailed",
                );
                return;
            }
        };

        let buy_rep = match wait_invoke_settlement(
            &mut invoke_rx,
            Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
        )
        .await
        {
            Some(rep) => rep,
            None => {
                crate::tee_eprintln!(
                    "[submit] open BUY taker pos_id={pos_id} order_id={http_order_id:?}: invoke timeout {ORDER_HTTP_TIMEOUT_SEC}s — OpenFailed",
                );
                return;
            }
        };

        crate::tee_println!(
            "[submit] open BUY taker pos_id={pos_id} asset_id={asset_id}: invoke settle success={} partial={} \
             order_id={:?}; making={:?}, taking={:?}, err={:?}",
            buy_rep.success,
            buy_rep.partial,
            buy_rep.order_id,
            buy_rep.making_amount,
            buy_rep.taking_amount,
            buy_rep.error_msg,
        );

        if !buy_rep.success {
            return;
        }

        let shares_net = match buy_rep.taking_amount {
            OrderAmount::Shares(s) if s.is_finite() && s > 0.0 => s,
            _ => {
                crate::tee_eprintln!(
                    "[submit] maker TP pos_id={pos_id}: BUY taking_amount не Shares — пропуск maker",
                );
                return;
            }
        };
   
        let shares_floor = (shares_net * 100.0).floor() / 100.0;
        let Some(implied_buy_price) = implied_buy_price_per_share(&buy_rep) else {
            crate::tee_eprintln!(
                "[submit] maker TP pos_id={pos_id}: не восстановили USD/share из BUY invoke — пропуск maker",
            );
            return;
        };
        let Some(min_order_size) = min_order_size_shares else {
            crate::tee_eprintln!(
                "[submit] maker TP pos_id={pos_id}: нет min_order_size в strict_book — пропуск maker",
            );
            return;
        };
        if shares_floor + 1e-9 < min_order_size {
            crate::tee_eprintln!(
                "[submit] maker TP pos_id={pos_id}: shares_floor={shares_floor:.4} < min_order_size={min_order_size:.4} — \
                 maker не выставляем (CLOB Size lower than the minimum)",
            );
            return;
        }
        let maker_price = (implied_buy_price + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999);
        crate::tee_println!(
            "[submit] maker TP pos_id={pos_id} asset_id={asset_id}: NET shares {shares_net:.6} → floor {shares_floor:.2}; \
             buy≈{implied_buy_price:.6} TP price={maker_price:.6} (+{Y_TRAIN_TAKE_PROFIT_PP} pp)",
        );

        let (mk_invoke_tx, mut mk_invoke_rx) = invoke_settlement_watch();
        let closing_arc: SharedClosingPosition = Arc::new(tokio::sync::RwLock::new(ClosingPosition {
            position: position.clone(),
            reason: CloseReason::TakeProfit,
            order_id: None,
            invoke_settle: Some(mk_invoke_rx.clone()),
            canceled: false,
            created_unix_ms: crate::util::current_timestamp_ms(),
        }));
        {
            let mut p = position.write().await;
            p.maker_tp_position = Some(Arc::downgrade(&closing_arc));
        }

        let maker_post_request = PostOrderRequest {
            asset_id: asset_id.clone(),
            side: Side::Sell,
            role: OrderRole::Maker,
            amount: OrderAmount::Shares(shares_floor),
            price: Some(maker_price),
            max_slippage_pp: None,
            expiration: None,
            market_end_unix_ms: event_end_ms,
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            strict_book: None,
        };
        let maker_invoke_cb: crate::account_order::SingleOrderInvokeCb =
            Box::new(move |rep| {
                let _ = mk_invoke_tx.send(Some(rep));
            });
        let post_res = post_order_on_clob(
            &account,
            project_manager.as_ref(),
            submit_mode,
            maker_post_request,
            maker_invoke_cb,
        )
        .await;

        let resting_oid = match &post_res {
            Ok(Some(oid)) if !oid.trim().is_empty() => Some(oid.clone()),
            Ok(Some(_)) => {
                crate::tee_eprintln!(
                    "[submit] maker TP pos_id={pos_id}: POST вернул пустой order_id",
                );
                None
            }
            Ok(None) => {
                crate::tee_eprintln!(
                    "[submit] maker TP pos_id={pos_id}: POST Ok(None) — resting нет до invoke",
                );
                None
            }
            Err(err) => {
                crate::tee_eprintln!(
                    "[submit] maker TP pos_id={pos_id}: POST err={err:#}",
                );
                None
            }
        };
        if let Some(oid) = resting_oid.as_ref() {
            let mut c = closing_arc.write().await;
            c.order_id = Some(oid.clone());
            crate::tee_println!(
                "[submit] maker TP pos_id={pos_id} asset_id={asset_id}: HTTP POST ok order_id={oid} \
                 price={maker_price:.6} shares={shares_floor:.2}",
            );
        }

        let maker_invoke_wait = invoke_wait_until_market_end_plus(event_end_ms);
        let maker_rep = match wait_invoke_settlement(&mut mk_invoke_rx, maker_invoke_wait).await
        {
            Some(rep) => rep,
            None => {
                crate::tee_eprintln!(
                    "[submit] maker TP pos_id={pos_id} order_id={resting_oid:?}: invoke timeout {:?} \
                     (до event_end + {ORDER_HTTP_TIMEOUT_SEC}s)",
                    maker_invoke_wait,
                );
                return;
            }
        };

        crate::tee_println!(
            "[submit] maker TP pos_id={pos_id} asset_id={asset_id}: invoke settle success={} partial={} \
             order_id={:?}; making={:?}, taking={:?}, err={:?}",
            maker_rep.success,
            maker_rep.partial,
            maker_rep.order_id,
            maker_rep.making_amount,
            maker_rep.taking_amount,
            maker_rep.error_msg,
        );
    });
}

/// Сколько ждать invoke-колбэк: до `market_end_unix_ms` + [`ORDER_HTTP_TIMEOUT_SEC`] с текущего момента.
pub(crate) fn invoke_wait_until_market_end_plus(market_end_unix_ms: Option<i64>) -> Duration {
    let now_ms = crate::util::current_timestamp_ms();
    let deadline_ms = market_end_unix_ms
        .map(|end_ms| end_ms.saturating_add((ORDER_HTTP_TIMEOUT_SEC * 1000) as i64))
        .unwrap_or(now_ms.saturating_add((ORDER_HTTP_TIMEOUT_SEC * 1000) as i64));
    let wait_ms = deadline_ms.saturating_sub(now_ms).max(1_000);
    Duration::from_millis(wait_ms as u64)
}

fn implied_buy_price_per_share(rep: &SingleOrderClobInvocationReport) -> Option<f64> {
    let usd = match rep.making_amount {
        OrderAmount::UsdNotional(u) if u.is_finite() && u > 0.0 => u,
        _ => return None,
    };
    let shares = match rep.taking_amount {
        OrderAmount::Shares(s) if s.is_finite() && s > 0.0 => s,
        _ => return None,
    };
    Some((usd / shares).clamp(0.001, 0.999))
}
