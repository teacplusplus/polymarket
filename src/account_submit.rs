//! `RealSimWithSubmit`: CLOB ордеры через [`crate::account_order`], подтверждение прежде всего WS
//! ([`crate::account_ws`]), дополнительно polling `client.order` ([`spawn_polling_verify`]).
//! Таски через `spawn` без долгих локов на `positions`/`closing`; дедуп TP/cancel/closing —
//! атомики/флаги на позиции до HTTP. После BUY/close/TP — poll до терминального статуса или
//! `event_end_ms`/`POLL_TIMEOUT_SEC`, затем [`apply_order_status_from_polling`] (как WS).
//!
//! `event_end_ms` из [`crate::history_sim::OpenPosition`] всегда пробрасывается в
//! [`crate::account_order::PostOrderRequest::market_end_unix_ms`] для POST здесь (дедлайн invoke/poll).
use crate::account::SharedAccount;
use crate::account_order::{
    CancelOrderRequest, OrderAmount, OrderRole, PostOrderRequest,
    SingleOrderClobInvocationReport, cancel_order_on_clob, post_order_on_clob,
};
use crate::history_sim::{
    ClosingPositionStatus, OpenPositionStatus, SIM_MAX_SLIPPAGE_FROM_L1_PCT, SharedClosingPosition,
    SharedOpenPosition, StrictBook,
};
use crate::xframe::Y_TRAIN_TAKE_PROFIT_PP;
use polymarket_client_sdk::clob::types::{Side};
use std::time::Duration;

/// Один REST/SUBMIT timeout — также для [`crate::account_order_completion`] и invoke-poll (через дубль константы там).
pub(crate) const ORDER_HTTP_TIMEOUT_SEC: u64 = 10;

/// Попытки SELL taker подряд (exp-backoff), иначе `CloseFailed`.
const SELL_TAKER_MAX_ATTEMPTS: u32 = 3;

/// Первая пауза между retry SELL taker («<< attempt» даёт 500ms, 1s, …).
const SELL_TAKER_RETRY_INITIAL_MS: u64 = 500;

/// Попытки cancel TP из hold-zone при сетевых ошибках.
const TP_HOLD_ZONE_CANCEL_MAX_ATTEMPTS: u32 = 3;

/// Первая пауза между retry cancel TP (hold-zone).
const TP_HOLD_ZONE_CANCEL_RETRY_INITIAL_MS: u64 = 500;

/// BUY taker, `UsdNotional = entry_cost`. `price`: worst с decision-L1 (предпочтительно);
/// если `None` — slip от свежей книги/SDK и опционально `strict_book` без GET.
///
/// После успешного HTTP POST → `PendingOpen` + `open_order_id`; при отказе POST — `OpenFailed`.
/// Колбэк invoke: успех → `Open` и poll; без успеха → `OpenFailed`.
pub(crate) fn spawn_open_buy_taker(
    account: SharedAccount,
    pos_arc: SharedOpenPosition,
    price: Option<f64>,
    strict_book: Option<StrictBook>,
) {
    tokio::spawn(async move {
        let (pos_id, asset_id, position_size_usd, market_end_unix_ms) = {
            let pos = pos_arc.read().await;
            (
                pos.id.clone(),
                pos.asset_id.clone(),
                pos.entry_cost,
                pos.event_end_ms,
            )
        };
        let max_slippage_pp = if price.is_some() {
            None
        } else {
            Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT)
        };
        let request = PostOrderRequest {
            asset_id: asset_id.clone(),                           // CLOB tokenId
            side: Side::Buy,                                      // вход
            role: OrderRole::Taker,                               // FAK BUY
            amount: OrderAmount::UsdNotional(position_size_usd),  // notional
            price,                                                // worst или None → slip
            max_slippage_pp,                                      // только если price None
            expiration: None,                                     // taker
            market_end_unix_ms,
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC), // post_order timeout
            strict_book,                                          // L1 для slip без GET
        };
        let pos_id_fail_log = pos_id.clone();
        let asset_fail_log = asset_id.clone();
        let (invoke_tx, invoke_rx) = tokio::sync::oneshot::channel::<SingleOrderClobInvocationReport>();
        let post_result = post_order_on_clob(
            &account,
            request,
            Box::new(move |result| {
                let _ = invoke_tx.send(result);
            }),
        )
        .await;
        match post_result {
            Err(err) => {
                crate::tee_eprintln!(
                    "[account_submit] BUY taker упал: pos_id={pos_id}, asset={asset_fail_log}: {err:#}"
                );
                pos_arc.write().await.open_status = OpenPositionStatus::OpenFailed;
            }
            Ok(None) => {
                crate::tee_eprintln!(
                    "[account_submit] BUY taker без принятого order_id после POST: pos_id={pos_id}, asset={asset_fail_log}"
                );
                pos_arc.write().await.open_status = OpenPositionStatus::OpenFailed;
            }
            Ok(Some(oid)) => {
                let mut pw = pos_arc.write().await;
                pw.open_order_id = Some(oid);
                pw.open_status = OpenPositionStatus::PendingOpen;
            }
        }
        match invoke_rx.await {
            Ok(report) => {
                if !report.success {
                    crate::tee_eprintln!(
                        "[account_submit] BUY taker без успеха (invoke): pos_id={pos_id_fail_log}, asset={asset_fail_log}, order_id={:?}, partial={}, error_msg={:?}",
                        report.order_id,
                        report.partial,
                        report.error_msg,
                    );
                    pos_arc.write().await.open_status = OpenPositionStatus::OpenFailed;
                    return;
                }
                let Some(real_order_id) = report.order_id.clone() else {
                    crate::tee_eprintln!(
                        "[account_submit] BUY taker без order_id CLOB при success invoke: pos_id={pos_id_fail_log}, asset={asset_fail_log}"
                    );
                    pos_arc.write().await.open_status = OpenPositionStatus::OpenFailed;
                    return;
                };
                {
                    let mut pw = pos_arc.write().await;
                    pw.open_status = OpenPositionStatus::Open;
                }
                crate::tee_println!(
                    "[account_submit] BUY размещён (invoke): pos_id={pos_id_fail_log}, order_id={real_order_id}, partial={}",
                    report.partial,
                );
            }
            Err(_) => {
                crate::tee_eprintln!(
                    "[account_submit] BUY taker invoke channel closed: pos_id={pos_id_fail_log}, asset={asset_fail_log}"
                );
            }
        }
    });
}

/// Maker TP по цене `buy_price + Y_TRAIN_TAKE_PROFIT_PP`. Идемпотентно через
/// `tp_placement_attempted` / существующий `tp_order_id`.
///
/// Успешный HTTP POST → `tp_order_id` сразу. Колбэк invoke без успеха сбрасывает `tp_order_id`
/// и `tp_placement_attempted`; успех → [`spawn_polling_verify_tp`]. Ошибки POST без id → сброс `tp_placement_attempted`.
pub async fn try_place_tp_maker(account: SharedAccount, pos_arc: SharedOpenPosition) {
    let (pos_id, asset_id, shares, tp_price, open_order_id, market_end_unix_ms) = {
        let mut pos = pos_arc.write().await;
        if pos.tp_placement_attempted || pos.tp_order_id.is_some() {
            return;
        }
        if !matches!(pos.open_status, OpenPositionStatus::Open) {
            return;
        }
        if pos.shares_held <= 0.0 || !pos.shares_held.is_finite() {
            return;
        }
        pos.tp_placement_attempted = true;
        (
            pos.id.clone(),
            pos.asset_id.clone(),
            pos.shares_held,
            (pos.buy_price + Y_TRAIN_TAKE_PROFIT_PP).clamp(0.001, 0.999),
            pos.open_order_id.clone(),
            pos.event_end_ms,
        )
    };

    let request = PostOrderRequest {
        asset_id: asset_id.clone(),                           // outcome token
        side: Side::Sell,                                     // TP short
        role: OrderRole::Maker,                               // post-only в SDK
        amount: OrderAmount::Shares(shares),                  // размер TP
        price: Some(tp_price),                                // limit
        max_slippage_pp: None,                                // не для maker
        expiration: None,                                     // GTC
        market_end_unix_ms,
        timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC), // post_order timeout
        strict_book: None,                                    // книга не нужна
    };
    let pos_id_fail_log = pos_id.clone();
    let open_order_id_fail_log = open_order_id.clone();
    let asset_fail_log = asset_id.clone();
    let pos_id_cb = pos_id.clone();
    let open_order_id_cb = open_order_id.clone();

    match post_order_on_clob(
        &account,
        request,
        Box::new(move |result| {
            let pos_id_log = pos_id_cb.clone();
            let open_oid_log = open_order_id_cb.clone();
            let tp_px = tp_price;
            let shr = shares;
            tokio::spawn(async move {
                
                if !result.success {
                    crate::tee_eprintln!(
                        "[account_submit] TP maker без успеха (invoke): pos_id={pos_id_log}, open_order_id={open_oid_log:?}, order_id={:?}, partial={}, error_msg={:?}",
                        result.order_id,
                        result.partial,
                        result.error_msg,
                    );
                    return;
                }
                let Some(tp_order_id) = result.order_id.clone() else {
                    crate::tee_eprintln!(
                        "[account_submit] TP maker без order_id при success invoke: pos_id={pos_id_log}, open_order_id={open_oid_log:?}",
                    );
                    return;
                };
                crate::tee_println!(
                    "[account_submit] TP maker размещён: pos_id={pos_id_log}, tp_order_id={tp_order_id}, open_order_id={open_oid_log:?}, price={tp_px:.4}, shares={shr:.4}",
                );
            });
        }),
    )
    .await
    {
        Err(err) => {
            crate::tee_eprintln!(
                "[account_submit] TP maker упал: pos_id={pos_id_fail_log}, open_order_id={open_order_id_fail_log:?}, asset={asset_fail_log}: {err:#}",
            );            
        }
        Ok(None) => {
            crate::tee_eprintln!(
                "[account_submit] TP maker без принятого order_id после POST: pos_id={pos_id_fail_log}, open_order_id={open_order_id_fail_log:?}, asset={asset_fail_log}",
            );            
        }
        Ok(Some(oid)) => {
            pos_arc.write().await.tp_order_id = Some(oid);
        }
    }
}

/// Снять TP при `HoldResolution`: caller уже взвёл `tp_cancel_attempted`.
/// `tp_order_id` только клонируем; обнуляем при `canceled=true`; иначе ждём WS/резолюцию.
pub fn spawn_cancel_tp_for_hold_zone(account: SharedAccount, pos_arc: SharedOpenPosition) {
    tokio::spawn(async move {
        let (pos_id, asset_id, tp_order_id_opt) = {
            let pos = pos_arc.read().await;
            (
                pos.id.clone(),
                pos.asset_id.clone(),
                pos.tp_order_id.clone(),
            )
        };
        let Some(tp_id) = tp_order_id_opt else {
            crate::tee_println!(
                "[account_submit] TP cancel (hold-zone) skipped — tp_order_id=None: pos_id={pos_id}, asset={asset_id} (вероятно, гонка с WS-MATCHED)"
            );
            return;
        };

        let cancel_req = CancelOrderRequest {
            order_id: tp_id.clone(),                              // maker TP id
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC), // cancel HTTP timeout
        };
        let mut last_result: Option<crate::account_order::CancelOrderResult> = None;
        for attempt in 1..=TP_HOLD_ZONE_CANCEL_MAX_ATTEMPTS {
            match cancel_order_on_clob(&account, cancel_req.clone()).await {
                Ok(res) => {
                    crate::tee_println!(
                        "[account_submit] TP cancel (hold-zone) attempt {attempt}/{TP_HOLD_ZONE_CANCEL_MAX_ATTEMPTS}: pos_id={pos_id}, order_id={tp_id}, canceled={}, error_msg={:?}",
                        res.canceled,
                        res.error_msg,
                    );
                    last_result = Some(res);
                    break;
                }
                Err(err) => {
                    crate::tee_eprintln!(
                        "[account_submit] TP cancel (hold-zone) HTTP-ошибка (attempt {attempt}/{TP_HOLD_ZONE_CANCEL_MAX_ATTEMPTS}): pos_id={pos_id}, asset={asset_id}, tp_order_id={tp_id}: {err:#}"
                    );
                }
            }
            if attempt < TP_HOLD_ZONE_CANCEL_MAX_ATTEMPTS {
                let delay_ms = TP_HOLD_ZONE_CANCEL_RETRY_INITIAL_MS << (attempt - 1);
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }

        match last_result {
            Some(res) if res.canceled => {
                let mut pw = pos_arc.write().await;
                if pw.tp_order_id.as_deref() == Some(tp_id.as_str()) {
                    pw.tp_order_id = None;
                }
                crate::tee_println!(
                    "[account_submit] TP cancel (hold-zone) confirmed: pos_id={pos_id}, order_id={tp_id} — tp_order_id обнулён"
                );
            }
            Some(_) => {
                crate::tee_println!(
                    "[account_submit] TP cancel (hold-zone) не подтверждён CLOB: pos_id={pos_id}, order_id={tp_id} — оставляем tp_order_id живым, ждём WS-MATCHED / резолюцию"
                );
            }
            None => {
                crate::tee_eprintln!(
                    "[account_submit] TP cancel (hold-zone) — все {TP_HOLD_ZONE_CANCEL_MAX_ATTEMPTS} попыток HTTP упали: pos_id={pos_id}, asset={asset_id}, tp_order_id={tp_id} — TP остаётся живым, будет снят резолюционным cleanup'ом"
                );
            }
        }
    });
}

/// SL/timeout/EvExit: cancel TP если есть (`tp_order_id` не take до HTTP — гонка с TP-fill),
/// иначе SELL taker с retry; `close_order_id` + polling. Caller взвёл `close_placement_attempted`.
pub fn spawn_close_via_taker(account: SharedAccount, closing_arc: SharedClosingPosition) {
    tokio::spawn(async move {
        let (pos_id, asset_id, tp_order_id_to_cancel, market_end_unix_ms) = {
            let pos_arc = closing_arc.read().await.position.clone();
            let pos = pos_arc.read().await;
            (
                pos.id.clone(),
                pos.asset_id.clone(),
                pos.tp_order_id.clone(),
                pos.event_end_ms,
            )
        };

        let bail_if_superseded = || {
            let closing_arc = closing_arc.clone();
            async move {
                let close_placement_attempted = closing_arc.read().await.close_placement_attempted;
                close_placement_attempted
            }
        };

        if let Some(tp_id) = tp_order_id_to_cancel.as_deref() {
            let cancel_req = CancelOrderRequest {
                order_id: tp_id.to_string(),                          // TP перед SELL
                timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC), // DELETE
            };
            match cancel_order_on_clob(&account, cancel_req).await {
                Ok(res) => {
                    crate::tee_println!(
                        "[account_submit] TP cancel: pos_id={pos_id}, order_id={tp_id}, canceled={}, error_msg={:?}",
                        res.canceled,
                        res.error_msg,
                    );
                    if res.canceled {
                        let pos_arc = closing_arc.read().await.position.clone();
                        let mut pw = pos_arc.write().await;
                        if pw.tp_order_id.as_deref() == Some(tp_id) {
                            pw.tp_order_id = None;
                        }
                    }
                }
                Err(err) => {
                    crate::tee_eprintln!(
                        "[account_submit] TP cancel упал: pos_id={pos_id}, tp_order_id={tp_id}: {err:#} — \
                     оставляем tp_order_id живым (WS-фоллбэк подберёт fill при гонке), продолжаем SELL taker"
                    );
                }
            }
        }

        if bail_if_superseded().await {
            return;
        }

        let shares_to_sell = {
            let pos_arc = closing_arc.read().await.position.clone();
            let p = pos_arc.read().await;
            p.shares_held
        };

        let request_template = PostOrderRequest {
            asset_id: asset_id.clone(), // outcome token
            side: Side::Sell,           // close long
            role: OrderRole::Taker,
            amount: OrderAmount::Shares(shares_to_sell), // после cancel TP
            price: None,                                 // worst из книги
            max_slippage_pp: None,                       // без cap
            expiration: None,                            // taker FAK
            market_end_unix_ms,
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC), // post_order timeout
            strict_book: None,                           // HTTP book внутри
        };
        for attempt in 1..=SELL_TAKER_MAX_ATTEMPTS {
            if bail_if_superseded().await {
                return;
            }
            {
                let mut cw = closing_arc.write().await;
                cw.close_placement_attempted = true;
            }
            let (invoke_tx, invoke_rx) = tokio::sync::oneshot::channel();
            match post_order_on_clob(
                &account,
                request_template.clone(),
                Box::new(move |rep| {
                    let _ = invoke_tx.send(rep);
                }),
            )
            .await
            {
                Err(err) => {
                    {
                        let mut cw = closing_arc.write().await;
                        cw.close_status = ClosingPositionStatus::CloseFailed;
                    }
                    crate::tee_eprintln!(
                        "[account_submit] SELL taker HTTP-ошибка (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, asset={asset_id}: {err:#}"
                    );
                }
                Ok(clob_order_id) => {
                    if let Some(ref oid) = clob_order_id {
                        crate::tee_println!(
                            "[account_submit] SELL POST принят CLOB (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, order_id={oid}",
                        );
                        {
                            let mut cw = closing_arc.write().await;
                            cw.close_order_id = Some(oid.clone());
                            cw.close_status = ClosingPositionStatus::PendingClose;
                        }
                    } else {
                        crate::tee_eprintln!(
                            "[account_submit] SELL POST без принятого order_id (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, asset={asset_id}",
                        );
                    }
                    match invoke_rx.await {
                        Ok(r) => {
                            let oid_accepted = r.order_id.clone().filter(|s| !s.is_empty());
                            match (r.success, oid_accepted) {
                                (true, Some(oid)) => {
                                    crate::tee_println!(
                                        "[account_submit] SELL размещён (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, order_id={oid}, partial={}",
                                        r.partial,
                                    );
                                    {
                                        let mut cw = closing_arc.write().await;
                                        cw.close_status = ClosingPositionStatus::Closed;
                                    }
                                    return;
                                }
                                _ => {
                                    crate::tee_eprintln!(
                                        "[account_submit] SELL не принят (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, asset={asset_id}, order_id={:?}, success={}, partial={}, error_msg={:?}",
                                        r.order_id,
                                        r.success,
                                        r.partial,
                                        r.error_msg,
                                    );                               
                                }
                            }
                        }
                        Err(_) => {
                            crate::tee_eprintln!(
                                "[account_submit] SELL taker колбёк потерян (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}"
                            );                    
                        }
                    }
                }
            }
            if attempt < SELL_TAKER_MAX_ATTEMPTS {
                let delay_ms = SELL_TAKER_RETRY_INITIAL_MS << (attempt - 1);
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
        crate::tee_eprintln!(
            "[account_submit] SELL taker — все {SELL_TAKER_MAX_ATTEMPTS} попыток исчерпаны, CloseFailed: pos_id={pos_id}, asset={asset_id}; следующий manage_positions-тик попытается снова"
        );
        closing_arc.write().await.close_status = ClosingPositionStatus::CloseFailed;
    });
}

/// Снять висящие TP после резолва маркета ([`crate::account::Account::resolve_pending_market`]).
pub fn spawn_cancel_tp_orders_after_resolution(
    account: SharedAccount,
    positions: Vec<crate::history_sim::SharedOpenPosition>,
) {
    if positions.is_empty() {
        return;
    }
    tokio::spawn(async move {
        for pos_arc in positions {
            let (pos_id, tp_id) = {
                let pos_g = pos_arc.read().await;
                let pid = pos_g.id.clone();
                match pos_g.tp_order_id.clone() {
                    Some(t) => (pid, t),
                    None => continue,
                }
            };
            let request = CancelOrderRequest {
                order_id: tp_id.clone(),                              // maker TP
                timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC), // cancel HTTP timeout
            };
            match cancel_order_on_clob(&account, request).await {
                Ok(res) => {
                    crate::tee_println!(
                        "[account_submit] TP cancel after resolution: pos_id={pos_id}, order_id={tp_id}, canceled={}, error_msg={:?}",
                        res.canceled,
                        res.error_msg,
                    );
                    if res.canceled {
                        let mut pw = pos_arc.write().await;
                        if pw.tp_order_id.as_deref() == Some(tp_id.as_str()) {
                            pw.tp_order_id = None;
                        }
                    }
                }
                Err(err) => {
                    crate::tee_eprintln!(
                        "[account_submit] TP cancel after resolution упал: pos_id={pos_id}, tp_order_id={tp_id}: {err:#}"
                    );
                }
            }
        }
    });
}
