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
    cancel_order_on_clob, post_order_on_clob, CancelOrderRequest, OrderAmount, OrderRole,
    PostOrderRequest, SingleOrderClobInvocationReport,
};
use crate::history_sim::{
    CloseReason, ClosingPosition, SharedClosingPosition, SharedOpenPosition, StrictBook,
    SIM_MAX_SLIPPAGE_FROM_L1_PCT,
};
use crate::xframe::Y_TRAIN_TAKE_PROFIT_PP;
use polymarket_client_sdk::clob::types::Side;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;


/// Один REST/SUBMIT timeout — также для [`crate::account_order_completion`] и invoke-poll (через дубль константы там).
pub(crate) const ORDER_HTTP_TIMEOUT_SEC: u64 = 10;
/// Повторы taker SELL при SL/timeout/ev-exit (FAK без матча и т.п.), как
/// [`UNWIND_OPPOSITE_TAKER_SELL_ATTEMPTS`] в live duel test.
pub(crate) const TAKER_SELL_ATTEMPTS: u32 = 3;

pub(crate) fn spawn_sell_taker(
    account: SharedAccount,
    position: SharedOpenPosition,
    exit_price: f64,
    reason: CloseReason,
    strict_book: Option<StrictBook>,
) {
    tokio::spawn(async move {
        match reason {
            CloseReason::TakeProfit | CloseReason::EvExitProfit => {
                let _ = (account, position, exit_price, strict_book);
            }
            CloseReason::StopLoss | CloseReason::Timeout | CloseReason::EvExitLoss => {
                let open_position = position.read().await;
                let position_id = open_position.id.clone();
                let asset_id = open_position.asset_id.clone();
                let event_end_unix_ms = open_position.event_end_ms;

                let shares_bought_net = match open_position.open_buy_invoke_report.as_ref() {
                    Some(invoke_report) => {
                        if !invoke_report.success {
                            None
                        } else {
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
                    }
                    None => None,
                };
                let Some(shares_bought_net) = shares_bought_net else {
                    crate::tee_eprintln!(
                        "[submit] sell taker pos_id={position_id}: нет NET shares в open_buy_invoke_report — пропуск",
                    );
                    return;
                };

                let mut shares_sold_by_maker = 0.0_f64;

                let maker_tp_position = open_position.maker_tp_position.clone();
                drop(open_position);

                if let Some(maker_tp_position) = maker_tp_position {
                    if let Some(maker_tp_position) = maker_tp_position.upgrade() {
                        let maker_closing_read = maker_tp_position.read().await;
                        let maker_tp_order_id = maker_closing_read
                            .order_id
                            .clone()
                            .filter(|order_id| !order_id.trim().is_empty());
                        let maker_tp_invoke_pending = maker_tp_order_id.is_some()
                            && maker_closing_read.invoke_report.is_none();
                        drop(maker_closing_read);

                        if maker_tp_invoke_pending {                            
                            if let Some(order_id) = maker_tp_order_id {
                                match cancel_order_on_clob(
                                    &account,
                                    CancelOrderRequest {
                                        order_id: order_id.clone(),
                                        timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
                                    },
                                )
                                .await
                                {
                                    Ok(cancel_result) => {
                                        crate::tee_println!(
                                            "[submit] sell taker pos_id={position_id}: cancel maker TP order_id={} canceled={} err={:?}",
                                            cancel_result.order_id,
                                            cancel_result.canceled,
                                            cancel_result.error_msg,
                                        );
                                        if cancel_result.canceled {
                                            let mut maker_closing_write =
                                                maker_tp_position.write().await;
                                            maker_closing_write.order_id = None;
                                        }
                                    }
                                    Err(cancel_err) => {
                                        crate::tee_eprintln!(
                                            "[submit] sell taker pos_id={position_id}: cancel maker TP order_id={order_id}: {cancel_err:#}",
                                        );
                                    }
                                }
                            }                        
                        }

                        todo!("добавить ожидание invoke-колбэка maker TP");

                        let maker_closing_read = maker_tp_position.read().await;
                        match maker_closing_read.invoke_report.as_ref() {
                            Some(invoke_report) if invoke_report.success => {
                                match invoke_report.making_amount {
                                    OrderAmount::Shares(shares)
                                        if shares.is_finite() && shares > 0.0 =>
                                    {
                                        shares_sold_by_maker = shares;
                                    }
                                    _ => {}
                                }
                            }
                            _ => {}
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

                let mut shares_left = shares_remaining;

                let sell_invoke_wait = invoke_wait_until_market_end_plus(event_end_unix_ms);

                for attempt in 1..=TAKER_SELL_ATTEMPTS {
                    let shares_to_sell = (shares_left * 100.0).floor() / 100.0;
                    if !(shares_to_sell > 0.0 && shares_to_sell.is_finite()) {
                        break;
                    }

                    crate::tee_println!(
                        "[submit] sell taker pos_id={position_id} asset_id={asset_id} reason={reason:?}: \
                         taker FAK SELL shares={shares_to_sell:.2} (shares_left {shares_left:.6}) \
                         exit_price≈{exit_price:.6} попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                    );

                    let (sell_invoke_tx, sell_invoke_rx) = oneshot::channel();
                    let sell_post_result = post_order_on_clob(
                        &account,
                        PostOrderRequest {
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
                        },
                        Box::new(move |sell_invoke_report| {
                            let _ = sell_invoke_tx.send(sell_invoke_report);
                        }),
                    )
                    .await;

                    let sell_order_id = match sell_post_result {
                        Ok(Some(order_id)) if !order_id.trim().is_empty() => Some(order_id),
                        Ok(Some(_)) => {
                            crate::tee_eprintln!(
                                "[submit] sell taker pos_id={position_id}: POST вернул пустой order_id \
                                 попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                            );
                            continue;
                        }
                        Ok(None) => {
                            crate::tee_eprintln!(
                                "[submit] sell taker pos_id={position_id}: POST Ok(None) \
                                 попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                            );
                            continue;
                        }
                        Err(post_err) => {
                            crate::tee_eprintln!(
                                "[submit] sell taker pos_id={position_id}: POST err={post_err:#} \
                                 попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                            );
                            continue;
                        }
                    };
                    let sell_invoke_report = match tokio::time::timeout(
                        sell_invoke_wait,
                        sell_invoke_rx,
                    )
                    .await
                    {
                        Ok(Ok(invoke_report)) => invoke_report,
                        Ok(Err(_)) => {
                            crate::tee_eprintln!(
                                "[submit] sell taker pos_id={position_id} order_id={sell_order_id:?}: \
                                 invoke-колбёк потерян попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                            );
                            continue;
                        }
                        Err(_) => {
                            crate::tee_eprintln!(
                                "[submit] sell taker pos_id={position_id} order_id={sell_order_id:?}: \
                                 invoke timeout {sell_invoke_wait:?} попытка {attempt}/{TAKER_SELL_ATTEMPTS}",
                            );
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

                    if sell_invoke_report.success {
                        let shares_sold = match sell_invoke_report.making_amount {
                            OrderAmount::Shares(shares) if shares.is_finite() && shares >= 0.0 => {
                                shares
                            }
                            _ => 0.0,
                        };
                        let sl_position: SharedClosingPosition =
                            Arc::new(tokio::sync::RwLock::new(ClosingPosition {
                                position: position.clone(),
                                reason: reason.clone(),
                                order_id: sell_order_id.clone(),
                                invoke_report: Some(sell_invoke_report.clone()),
                                created_unix_ms: crate::util::current_timestamp_ms(),
                            }));
                        {
                            let mut open_position = position.write().await;
                            open_position
                                .sl_positions
                                .push(Arc::downgrade(&sl_position));
                        }
                        shares_left = (shares_left - shares_sold).max(0.0);
                        let shares_left_to_sell = (shares_left * 100.0).floor() / 100.0;
                        if !(shares_left_to_sell > 0.0 && shares_left_to_sell.is_finite()) {
                            break;
                        }
                    }
                }
            }
        }
    });
}


pub(crate) fn spawn_open_buy_taker(
    account: SharedAccount,
    position: SharedOpenPosition,
    price: Option<f64>,
    strict_book: Option<StrictBook>,
) {
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

        if let Err(reason) = open_buy_min_size_guard(amount, price, strict_book.as_ref()) {
            crate::tee_eprintln!(
                "[submit] open BUY taker pos_id={pos_id} asset_id={asset_id}: min_order_size guard: {reason} — пропуск POST",
            );
            return;
        }

        let min_order_size_shares = strict_book
            .as_ref()
            .and_then(|b| b.min_order_size)
            .filter(|m| m.is_finite() && *m > 0.0);

        let (invoke_tx, invoke_rx) = oneshot::channel();
        let post_result = post_order_on_clob(
            &account,
            PostOrderRequest {
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
            },
            Box::new(move |buy_rep| {
                let _ = invoke_tx.send(buy_rep);
            }),
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

        let buy_rep = match tokio::time::timeout(
            Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            invoke_rx,
        )
        .await
        {
            Ok(Ok(rep)) => {
                {
                    let mut p = position.write().await;
                    p.open_buy_invoke_report = Some(rep.clone());
                }
                rep
            }
            Ok(Err(_)) => {
                crate::tee_eprintln!(
                    "[submit] open BUY taker pos_id={pos_id} order_id={http_order_id:?}: invoke-колбёк потерян — OpenFailed",
                );
                return;
            }
            Err(_) => {
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

        let closing_arc: SharedClosingPosition = Arc::new(tokio::sync::RwLock::new(ClosingPosition {
            position: position.clone(),
            reason: CloseReason::TakeProfit,
            order_id: None,
            invoke_report: None,
            created_unix_ms: crate::util::current_timestamp_ms(),
        }));
        {
            let mut p = position.write().await;
            p.maker_tp_position = Some(Arc::downgrade(&closing_arc));
        }

        let (mk_invoke_tx, mk_invoke_rx) = oneshot::channel();
        let post_res = post_order_on_clob(
            &account,
            PostOrderRequest {
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
            },
            Box::new(move |rep| {
                let _ = mk_invoke_tx.send(rep);
            }),
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
        let maker_rep = match tokio::time::timeout(maker_invoke_wait, mk_invoke_rx).await
        {
            Ok(Ok(rep)) => {
                let mut c = closing_arc.write().await;
                c.invoke_report = Some(rep.clone());
                rep
            }
            Ok(Err(_)) => {
                crate::tee_eprintln!(
                    "[submit] maker TP pos_id={pos_id} order_id={resting_oid:?}: invoke-колбёк потерян",
                );
                return;
            }
            Err(_) => {
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
fn invoke_wait_until_market_end_plus(market_end_unix_ms: Option<i64>) -> Duration {
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


fn open_buy_price_cap(price: Option<f64>) -> Result<f64, String> {
    price
        .filter(|p| p.is_finite() && *p > 0.0)
        .map(|p| p.clamp(0.001, 0.999))
        .ok_or_else(|| "нет price (worst-price cap для min_order_size guard)".to_string())
}

/// Worst-case shares при `amount / price_cap` должны быть ≥ CLOB `min_order_size` (как [`duel_leg_prep`] / post-BUY floor в duel).
fn open_buy_min_size_guard(
    amount_usd: f64,
    price: Option<f64>,
    strict_book: Option<&StrictBook>,
) -> Result<(), String> {
    let book = strict_book.ok_or_else(|| "нет strict_book".to_string())?;
    let min_shares = book
        .min_order_size
        .filter(|m| m.is_finite() && *m > 0.0)
        .ok_or_else(|| "strict_book.min_order_size отсутствует".to_string())?;
    let price_cap = open_buy_price_cap(price)?;
    let worst_case_shares = amount_usd / price_cap;
    let shares_floor = (worst_case_shares * 100.0).floor() / 100.0;
    if shares_floor + 1e-9 < min_shares {
        return Err(format!(
            "shares_floor={shares_floor:.4} < min_order_size={min_shares:.4} \
             (amount={amount_usd:.4} USDC, price_cap={price_cap:.5})"
        ));
    }
    Ok(())
}
