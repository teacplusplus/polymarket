//! `RealSimWithSubmit`: CLOB ордеры через [`crate::account_order`], подтверждение прежде всего WS
//! ([`crate::account_ws`]), дополнительно polling `client.order` ([`spawn_polling_verify`]).
//! Таски через `spawn` без долгих локов на `positions`/`closing`; дедуп TP/cancel/closing —
//! атомики/флаги на позиции до HTTP. После BUY/close/TP — poll до терминального статуса или
//! `event_end_ms`/`POLL_TIMEOUT_SEC`, затем [`apply_order_status_from_polling`] (как WS).
//!
//! `event_end_ms` из [`crate::history_sim::OpenPosition`] всегда пробрасывается в
//! [`crate::account_order::PostOrderRequest::market_end_unix_ms`] для POST здесь (дедлайн invoke/poll).
use crate::account::SharedAccount;
use crate::account_order::{OrderAmount, OrderRole, PostOrderRequest, SingleOrderClobInvocationReport, post_order_on_clob};
use crate::history_sim::{SIM_MAX_SLIPPAGE_FROM_L1_PCT, SharedOpenPosition, StrictBook};
use polymarket_client_sdk::clob::types::Side;
use std::time::Duration;

/// Один REST/SUBMIT timeout — также для [`crate::account_order_completion`] и invoke-poll (через дубль константы там).
pub(crate) const ORDER_HTTP_TIMEOUT_SEC: u64 = 10;

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
                pos.position_size,
                pos.event_end_ms,
            )
        };
        let max_slippage_pp = if price.is_some() {
            None
        } else {
            Some(SIM_MAX_SLIPPAGE_FROM_L1_PCT)
        };
        let request = PostOrderRequest {
            asset_id: asset_id.clone(),                          // CLOB tokenId
            side: Side::Buy,                                     // вход
            role: OrderRole::Taker,                              // FAK BUY
            amount: OrderAmount::UsdNotional(position_size_usd), // notional
            price,                                               // worst или None → slip
            max_slippage_pp,                                     // только если price None
            expiration: None,                                    // taker
            market_end_unix_ms,
            timeout: Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC), // post_order timeout
            strict_book,                                          // L1 для slip без GET
        };
        let pos_id_fail_log = pos_id.clone();
        let asset_fail_log = asset_id.clone();
        let (invoke_tx, invoke_rx) =
            tokio::sync::oneshot::channel::<SingleOrderClobInvocationReport>();
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
            }
            Ok(None) => {
                crate::tee_eprintln!(
                    "[account_submit] BUY taker без принятого order_id после POST: pos_id={pos_id}, asset={asset_fail_log}"
                );
            }
            Ok(Some(oid)) => {
                let mut pw = pos_arc.write().await;
                pw.open_order_id = Some(oid);
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
                    return;
                }
                let Some(real_order_id) = report.order_id.clone() else {
                    crate::tee_eprintln!(
                        "[account_submit] BUY taker без order_id CLOB при success invoke: pos_id={pos_id_fail_log}, asset={asset_fail_log}"
                    );
                    return;
                };
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

