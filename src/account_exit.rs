//! Graceful shutdown для [`crate::main::AppMode::RealSimWithSubmit`].
//!
//! Триггерится из `main` по `SIGINT`/`SIGTERM` (см. вызов
//! [`graceful_exit`] в `tokio::select!` после спавна воркеров). Делает
//! ровно три вещи в порядке безопасности «не открыть → отменить → продать»:
//!
//! 1. Атомарно ставит [`HALT_NEW_ORDERS`] в `true`. Все strategy-driven
//!    кодопути (BUY-taker / TP-maker / SELL-taker через
//!    [`crate::history_sim::manage_positions`] и [`crate::real_sim::tick_once`])
//!    должны заглядывать в [`is_halted`] и no-op'ать. Это останавливает
//!    приток новых ордеров на CLOB.
//! 2. Дёргает CLOB `DELETE /cancel-all` — один HTTP-вызов снимает **все**
//!    наши open-ордера, включая maker TP-лимитки и pending taker'ы,
//!    которые могут быть в полёте.
//! 3. Тянет позиции через Polymarket Data API (`/positions`) для
//!    derived Safe-адреса (см. [`crate::poly_chain::derive_safe_address`])
//!    и для каждой с `size > SHARES_DUST_THRESHOLD` постит SELL-taker
//!    на «по рынку» через [`crate::account_order::post_order_on_clob`]
//!    (без slippage cap'а — на shutdown'е приоритет «свернуть позицию»).
//!
//! После этого функция возвращает управление; `main` должен сделать
//! `std::process::exit(...)` или просто упасть в конец.
//!
//! **Не вызывается** в виртуальных режимах (`History` / `RealSim`-без-submit):
//! им graceful-shutdown не нужен (CLOB-ордеров нет).

use crate::account::SharedAccount;
use crate::account_order::{post_order_on_clob, OrderAmount, OrderRole, PostOrderRequest};
use polymarket_client_sdk::clob::types::Side;
use polymarket_client_sdk::data;
use polymarket_client_sdk::data::types::request::PositionsRequest;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Глобальный single-process флаг «не открывать новые ордера».
/// Set'ится в [`graceful_exit`]; читается в [`is_halted`] из всех
/// strategy-driven путей (`try_open_position`, `try_place_tp_maker`,
/// `tick_once`-loop).
///
/// Acquire/Release семантика достаточна: запись и чтения происходят
/// между разными tokio-задачами, но без необходимости в строгом
/// SeqCst-порядке относительно других атомиков — у нас их нет рядом.
static HALT_NEW_ORDERS: AtomicBool = AtomicBool::new(false);

/// Хард-таймаут одного HTTP-вызова в shutdown-пайплайне. Берём щедро
/// (60s), чтобы не пропустить cancel/sell на лагающей сети — на shutdown
/// лучше подождать, чем оставить живой ордер.
const EXIT_HTTP_TIMEOUT_SEC: u64 = 60;

/// Минимальный размер позиции (в shares), ниже которого SELL не
/// делается — Polymarket всё равно отвергнет ордер по min-size,
/// а dust-остаток после auto-redeem'а будет $0.
const SHARES_DUST_THRESHOLD: f64 = 0.0001;

/// Пауза между SELL-таскам'и: предотвращает rate-limit на массовом
/// shutdown'е (десятки позиций × 200ms ≈ секунды, приемлемо).
const PER_POSITION_PAUSE_MS: u64 = 200;

/// Возвращает `true`, если в этом процессе уже инициирован graceful
/// shutdown (т.е. [`graceful_exit`] выставила флаг). Все strategy-driven
/// места, спавнящие новые BUY / TP-maker ордера, обязаны проверять
/// этот флаг перед HTTP и no-op'ать.
///
/// **Не блокирует**: `Acquire`-load атомика.
pub fn is_halted() -> bool {
    HALT_NEW_ORDERS.load(Ordering::Acquire)
}

/// Точка входа graceful shutdown'а — вызывается из `main` после
/// сигнала. Шаги: halt-flag → cancel-all → sell-all-shares. Каждый
/// шаг защищён собственным таймаутом, ошибки логируются (не возвращаем
/// `Result` наружу — на shutdown'е возвращать некому, всё что можно
/// сделать — задокументировать через лог).
pub async fn graceful_exit(account: SharedAccount) {
    crate::tee_println!("[account_exit] Старт graceful shutdown");
    HALT_NEW_ORDERS.store(true, Ordering::Release);
    crate::tee_println!(
        "[account_exit] HALT_NEW_ORDERS=true — strategy-driven пути больше не \
         спавнят новые BUY/TP-maker (см. is_halted)"
    );

    // Шаг 1: одним HTTP'ом снять ВСЕ open-ордера.
    cancel_all_orders(&account).await;

    // Шаг 2: вытащить позиции из data-API и продать каждую SELL-taker'ом.
    sell_all_positions(&account).await;

    crate::tee_println!("[account_exit] Graceful shutdown завершён");
}

/// `DELETE /cancel-all` — снять все open-ордера для аутентифицированного
/// пользователя. Один HTTP-вызов вместо N последовательных `cancel_order`'ов.
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
            // Если CLOB вернул not_canceled с непустым словарём — это
            // ордера, которые он не смог отменить (уже сматчены/протухли);
            // нам важно, но не критично, просто логируем причины.
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
            crate::tee_eprintln!(
                "[account_exit] cancel-all timeout > {EXIT_HTTP_TIMEOUT_SEC}s"
            );
        }
    }
}

/// Тянет [`data::types::response::Position`] для derived Safe-адреса
/// EOA (см. [`crate::poly_chain::derive_safe_address`]) и продаёт
/// каждую SELL-taker'ом без slippage cap'а.
///
/// **Адрес для запроса**: Polymarket Safe (proxy) деривится по `CREATE2`
/// от EOA — это адрес, на котором CTF реально хранит shares; именно
/// он попадает в `Position.proxy_wallet`. Если в .env есть явный
/// `POLY_FUNDER` override — он бы шёл сюда, но в текущем коде такого
/// нет, поэтому используем deterministic deriviation.
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

    // Default endpoint Data API (`https://data-api.polymarket.com`) —
    // тот же, что использует SDK по умолчанию.
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
        // `Position.asset` — `U256` token id (`outcome token id`). CLOB
        // ожидает asset_id как десятичную строку — `to_string()` на
        // `U256` даёт именно это.
        let asset_id_str = pos.asset.to_string();
        // SELL-taker без slippage cap'а: на shutdown'е приоритет
        // «выйти любой ценой» (через секунды истечёт graceful-окно и
        // мы просто упадём; лучше продаться на лежащем bid'е, чем
        // оставить shares висеть до резолюции, тратя capital).
        let request = PostOrderRequest {
            asset_id: asset_id_str.clone(),
            side: Side::Sell,
            role: OrderRole::Taker,
            amount: OrderAmount::Shares(shares),
            price: None,
            max_slippage_pp: None,
            expiration: None,
            timeout: Duration::from_secs(EXIT_HTTP_TIMEOUT_SEC),
            strict_book: None,
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
