//! Graceful shutdown для `RealSimWithSubmit`: SIGINT/SIGTERM → halt → `DELETE /cancel-all` →
//! Data API позиции по derived Safe → SELL taker. В виртуальных режимах не вызывается.

use crate::account::SharedAccount;
use crate::account_order::{cancel_all_orders_on_clob, sell_all_positions_on_clob};
use std::sync::atomic::{AtomicBool, Ordering};

/// Стратегия не открывает новых ордеров после `graceful_exit` (`Acquire`/`Release`).
static HALT_NEW_ORDERS: AtomicBool = AtomicBool::new(false);

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

    cancel_all_orders_on_clob(&account).await;
    sell_all_positions_on_clob(&account).await;

    crate::tee_println!("[account_exit] Graceful shutdown завершён");
}
