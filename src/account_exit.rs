//! Graceful shutdown для `RealSimWithSubmit`: SIGINT/SIGTERM → halt → `DELETE /cancel-all` →
//! Data API позиции по derived Safe → SELL taker. В виртуальных режимах не вызывается.

use crate::account::SharedAccount;
use crate::account_order::{OrderAmount, OrderRole, PostOrderRequest, post_order_on_clob};
use polymarket_client_sdk::clob::types::Side;
use polymarket_client_sdk::data;
use polymarket_client_sdk::data::types::request::PositionsRequest;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Стратегия не открывает новых ордеров после `graceful_exit` (`Acquire`/`Release`).
static HALT_NEW_ORDERS: AtomicBool = AtomicBool::new(false);

/// Таймаут HTTP на шагах shutdown (cancel-all, data, post_order).
const EXIT_HTTP_TIMEOUT_SEC: u64 = 60;

/// Ниже этого размера в shares SELL не шлём (dust / min size).
const SHARES_DUST_THRESHOLD: f64 = 0.0001;

/// Пауза между SELL по позициям, чтобы не упираться в rate limit.
const PER_POSITION_PAUSE_MS: u64 = 200;

/// `true`, если shutdown уже начался — проверять перед новыми BUY/TP.
pub fn is_halted() -> bool {
    HALT_NEW_ORDERS.load(Ordering::Acquire)
}

/// Halt → cancel-all → продать все позиции из Data API; ошибки только в лог.
pub async fn graceful_exit(account: SharedAccount) {
    crate::tee_println!("[account_exit] Старт graceful shutdown");
    HALT_NEW_ORDERS.store(true, Ordering::Release);
    crate::tee_println!(
        "[account_exit] HALT_NEW_ORDERS=true — strategy-driven пути больше не \
         спавнят новые BUY/TP-maker (см. is_halted)"
    );

    cancel_all_orders(&account).await;
    sell_all_positions(&account).await;

    crate::tee_println!("[account_exit] Graceful shutdown завершён");
}

/// `DELETE /cancel-all` для текущей CLOB-сессии.
async fn cancel_all_orders(account: &SharedAccount) {
    let auth_client = match (**account.clob_authed.load()).clone() {
        Some(c) => c,
        None => {
            crate::tee_eprintln!(
                "[account_exit] clob_authed=None — cancel-all пропускаем (auth не поднялся)"
            );
            return;
        }
    };
    match tokio::time::timeout(
        Duration::from_secs(EXIT_HTTP_TIMEOUT_SEC),
        auth_client.cancel_all_orders(),
    )
    .await
    {
        Ok(Ok(resp)) => {
            crate::tee_println!(
                "[account_exit] cancel-all OK: canceled={}, not_canceled={}",
                resp.canceled.len(),
                resp.not_canceled.len(),
            );
            for (oid, reason) in &resp.not_canceled {
                crate::tee_eprintln!(
                    "[account_exit] cancel-all not_canceled: order_id={oid}, reason={reason}"
                );
            }
        }
        Ok(Err(err)) => {
            crate::tee_eprintln!("[account_exit] cancel-all упал: {err:#}");
        }
        Err(_) => {
            crate::tee_eprintln!("[account_exit] cancel-all timeout > {EXIT_HTTP_TIMEOUT_SEC}s");
        }
    }
}

/// Позиции с `user = derive_safe_address(EOA)`, каждую — SELL taker без cap.
async fn sell_all_positions(account: &SharedAccount) {
    let signer = match (**account.clob_signer.load()).as_ref().cloned() {
        Some(s) => s,
        None => {
            crate::tee_eprintln!(
                "[account_exit] clob_signer=None — не знаем EOA, sell-all пропускаем"
            );
            return;
        }
    };
    let eoa = signer.address();
    let safe = crate::poly_chain::derive_safe_address(eoa);
    crate::tee_println!(
        "[account_exit] data/positions: user=safe={safe:#x} (derived from eoa={eoa:#x})"
    );

    let data_client = data::Client::default();
    let positions_req = PositionsRequest::builder().user(safe).build();
    let positions = match tokio::time::timeout(
        Duration::from_secs(EXIT_HTTP_TIMEOUT_SEC),
        data_client.positions(&positions_req),
    )
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(err)) => {
            crate::tee_eprintln!("[account_exit] data/positions упал: {err:#}");
            return;
        }
        Err(_) => {
            crate::tee_eprintln!(
                "[account_exit] data/positions timeout > {EXIT_HTTP_TIMEOUT_SEC}s"
            );
            return;
        }
    };

    crate::tee_println!(
        "[account_exit] позиций к продаже: {} (без фильтра по dust)",
        positions.len()
    );

    let mut sold = 0_usize;
    let mut skipped_dust = 0_usize;
    let mut failed = 0_usize;
    for pos in positions {
        let shares = pos.size.to_string().parse::<f64>().unwrap_or(0.0);
        if !shares.is_finite() || shares < SHARES_DUST_THRESHOLD {
            skipped_dust += 1;
            continue;
        }
        let asset_id_str = pos.asset.to_string();
        let request = PostOrderRequest {
            asset_id: asset_id_str.clone(),      // tokenId (decimal)
            side: Side::Sell,                    // unwind long
            role: OrderRole::Taker,              // есть ликвидность на bid
            amount: OrderAmount::Shares(shares), // size из Data API
            price: None,                         // worst — из стакана
            max_slippage_pp: None,               // без cap на выходе
            expiration: None,                    // maker-only
            timeout: Duration::from_secs(EXIT_HTTP_TIMEOUT_SEC), // POST /order
            strict_book: None,                   // снимок книги из HTTP при необходимости
        };
        match post_order_on_clob(account, request).await {
            Ok(r) if r.success => {
                crate::tee_println!(
                    "[account_exit] SELL ok: asset={asset_id_str}, shares={shares:.4}, \
                     order_id={}, status={:?}",
                    r.order_id,
                    r.status,
                );
                sold += 1;
            }
            Ok(r) => {
                crate::tee_eprintln!(
                    "[account_exit] SELL отвергнут CLOB: asset={asset_id_str}, \
                     shares={shares:.4}, error_msg={:?}, status={:?}",
                    r.error_msg,
                    r.status,
                );
                failed += 1;
            }
            Err(err) => {
                crate::tee_eprintln!(
                    "[account_exit] SELL HTTP-ошибка: asset={asset_id_str}, \
                     shares={shares:.4}: {err:#}"
                );
                failed += 1;
            }
        }
        tokio::time::sleep(Duration::from_millis(PER_POSITION_PAUSE_MS)).await;
    }
    crate::tee_println!(
        "[account_exit] sell-all итог: sold={sold}, failed={failed}, skipped_dust={skipped_dust}"
    );
}
