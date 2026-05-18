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
    post_order_on_clob, OrderAmount, OrderRole, PostOrderRequest, SingleOrderClobInvocationReport,
};
use crate::history_sim::{
    CloseReason, ClosingPosition, SharedClosingPosition, SharedOpenPosition, StrictBook,
};
use crate::xframe::Y_TRAIN_TAKE_PROFIT_PP;
use polymarket_client_sdk::clob::types::Side;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;


/// Один REST/SUBMIT timeout — также для [`crate::account_order_completion`] и invoke-poll (через дубль константы там).
pub(crate) const ORDER_HTTP_TIMEOUT_SEC: u64 = 10;

pub(crate) fn spawn_sell_taker(
    account: SharedAccount,
    pos_arc: SharedOpenPosition,
    exit_price: f64,
    reason: CloseReason,
    strict_book: Option<StrictBook>,
) {
    tokio::spawn(async move {
        let _ = (account, pos_arc, exit_price, reason, strict_book);
    });
}


pub(crate) fn spawn_open_buy_taker(
    account: SharedAccount,
    pos_arc: SharedOpenPosition,
    price: Option<f64>,
    strict_book: Option<StrictBook>,
) {
    tokio::spawn(async move {
        let (asset_id, amount, event_end_ms, pos_id) = {
            let p = pos_arc.read().await;
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
                    let mut p = pos_arc.write().await;
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
                    let mut p = pos_arc.write().await;
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
            position: pos_arc.clone(),
            reason: CloseReason::TakeProfit,
            pnl: None,
            order_id: None,
            invoke_report: None,
            created_unix_ms: crate::util::current_timestamp_ms(),
        }));
        {
            let mut p = pos_arc.write().await;
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
