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
    CancelOrderRequest, OrderAmount, OrderRole, PostOrderRequest, cancel_order_on_clob,
    post_order_on_clob,
};
use crate::history_sim::{
    ClosingPositionStatus, OpenPositionStatus, SIM_MAX_SLIPPAGE_FROM_L1_PCT, SharedClosingPosition,
    SharedOpenPosition, StrictBook,
};
use crate::xframe::Y_TRAIN_TAKE_PROFIT_PP;
use polymarket_client_sdk::clob::types::request::TradesRequest;
use polymarket_client_sdk::clob::types::{OrderStatusType, Side};
use std::time::Duration;

/// Один REST/SUBMIT timeout — также для [`crate::account_order_completion`] и invoke-poll (через дубль константы там).
pub(crate) const ORDER_HTTP_TIMEOUT_SEC: u64 = 10;

/// Интервал `client.order` в [`spawn_polling_verify`].
const POLL_INTERVAL_SEC: u64 = 3;

/// Запас по времени polling, если нет «жёсткого» дедлайна маркета.
const POLL_TIMEOUT_SEC: u64 = 30;

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
/// Успех → `open_order_id` + [`spawn_polling_verify_open`]; ошибка → `OpenFailed`.
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
        let account_invoke = account.clone();
        let pos_arc_invoke = pos_arc.clone();
        match post_order_on_clob(
            &account,
            request,
            Box::new(move |result| {
                let account_i = account_invoke.clone();
                let pos_arc_i = pos_arc_invoke.clone();
                let pos_id_log = pos_id.clone();
                let asset_log = asset_id.clone();
                tokio::spawn(async move {
                    if !result.success {
                        crate::tee_eprintln!(
                            "[account_submit] BUY taker без успеха (invoke): pos_id={pos_id_log}, asset={asset_log}, order_id={:?}, partial={}",
                            result.order_id,
                            result.partial,
                        );
                        pos_arc_i.write().await.open_status = OpenPositionStatus::OpenFailed;
                        return;
                    }
                    let Some(real_order_id) = result.order_id.clone() else {
                        crate::tee_eprintln!(
                            "[account_submit] BUY taker без order_id CLOB при success invoke: pos_id={pos_id_log}, asset={asset_log}"
                        );
                        pos_arc_i.write().await.open_status = OpenPositionStatus::OpenFailed;
                        return;
                    };
                    {
                        let mut pw = pos_arc_i.write().await;
                        pw.open_order_id = Some(real_order_id.clone());
                        pw.open_status = OpenPositionStatus::Open;
                    }
                    crate::tee_println!(
                        "[account_submit] BUY размещён (invoke): pos_id={pos_id_log}, order_id={real_order_id}, partial={}",
                        result.partial,
                    );
                    spawn_polling_verify_open(account_i, pos_arc_i);
                });
            }),
        )
        .await
        {
            Err(err) => {
                crate::tee_eprintln!(
                    "[account_submit] BUY taker упал: pos_id={pos_id_fail_log}, asset={asset_fail_log}: {err:#}"
                );
                pos_arc.write().await.open_status = OpenPositionStatus::OpenFailed;
            }
            Ok(None) => {
                crate::tee_eprintln!(
                    "[account_submit] BUY taker без принятого order_id после POST: pos_id={pos_id_fail_log}, asset={asset_fail_log}"
                );
                pos_arc.write().await.open_status = OpenPositionStatus::OpenFailed;
            }
            Ok(Some(oid)) => {
                let mut pw = pos_arc.write().await;
                pw.open_order_id = Some(oid);
                pw.open_status = OpenPositionStatus::PendingOpen;
            }
        }
    });
}

/// Maker TP по цене `buy_price + Y_TRAIN_TAKE_PROFIT_PP`. Идемпотентно через
/// `tp_placement_attempted` / существующий `tp_order_id`. Успех → `spawn_polling_verify_tp`.
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
    let account_invoke = account.clone();
    let pos_arc_invoke = pos_arc.clone();
    let pos_id_cb = pos_id.clone();
    let open_order_id_cb = open_order_id.clone();

    if let Err(err) = post_order_on_clob(
        &account,
        request,
        Box::new(move |result| {
            let account_i = account_invoke.clone();
            let pos_arc_i = pos_arc_invoke.clone();
            let pos_id_log = pos_id_cb.clone();
            let open_oid_log = open_order_id_cb.clone();
            let tp_px = tp_price;
            let shr = shares;
            tokio::spawn(async move {
                if !result.success {
                    crate::tee_eprintln!(
                        "[account_submit] TP maker без успеха (invoke): pos_id={pos_id_log}, open_order_id={open_oid_log:?}, order_id={:?}, partial={}",
                        result.order_id,
                        result.partial,
                    );
                    return;
                }
                let Some(tp_order_id) = result.order_id.clone() else {
                    crate::tee_eprintln!(
                        "[account_submit] TP maker без order_id при success invoke: pos_id={pos_id_log}, open_order_id={open_oid_log:?}",
                    );
                    return;
                };
                pos_arc_i.write().await.tp_order_id = Some(tp_order_id.clone());
                crate::tee_println!(
                    "[account_submit] TP maker размещён: pos_id={pos_id_log}, tp_order_id={tp_order_id}, open_order_id={open_oid_log:?}, price={tp_px:.4}, shares={shr:.4}",
                );
                spawn_polling_verify_tp(account_i, pos_arc_i);
            });
        }),
    )
    .await
    {
        crate::tee_eprintln!(
            "[account_submit] TP maker упал: pos_id={pos_id_fail_log}, open_order_id={open_order_id_fail_log:?}, asset={asset_fail_log}: {err:#}",
        );
        return;
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
                    crate::tee_eprintln!(
                        "[account_submit] SELL taker HTTP-ошибка (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, asset={asset_id}: {err:#}"
                    );
                }
                Ok(clob_order_id) => {
                    if let Some(ref oid) = clob_order_id {
                        crate::tee_println!(
                            "[account_submit] SELL POST принят CLOB (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, order_id={oid}",
                        );
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
                                        cw.close_order_id = Some(oid);
                                    }
                                    spawn_polling_verify_close(account.clone(), closing_arc.clone());
                                    return;
                                }
                                _ => {
                                    crate::tee_eprintln!(
                                        "[account_submit] SELL не принят (attempt {attempt}/{SELL_TAKER_MAX_ATTEMPTS}): pos_id={pos_id}, asset={asset_id}, order_id={:?}, success={}, partial={}",
                                        r.order_id,
                                        r.success,
                                        r.partial,
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

/// Poll `client.order` для BUY после post (см. [`PollingPositionKind::Open`]).
fn spawn_polling_verify_open(account: SharedAccount, pos_arc: SharedOpenPosition) {
    spawn_polling_verify(account, PollingPositionKind::Open(pos_arc));
}

/// Poll для SELL close (`close_order_id`).
fn spawn_polling_verify_close(account: SharedAccount, c_arc: SharedClosingPosition) {
    spawn_polling_verify(account, PollingPositionKind::Close(c_arc));
}

/// Poll для maker TP (`tp_order_id`).
fn spawn_polling_verify_tp(account: SharedAccount, pos_arc: SharedOpenPosition) {
    spawn_polling_verify(account, PollingPositionKind::TpMaker(pos_arc));
}

/// Какую запись дергать при poll; `order_id` из соответствующего поля структуры.
#[derive(Clone)]
pub(crate) enum PollingPositionKind {
    /// `open_order_id`, Matched → Open + TP.
    Open(SharedOpenPosition),
    /// `close_order_id`, Matched → финализация close PnL.
    Close(SharedClosingPosition),
    /// `tp_order_id`, Matched → финализация TP PnL.
    TpMaker(SharedOpenPosition),
}

/// Следующий шаг после [`apply_order_status_from_polling`] (HTTP только в caller).
pub(crate) enum PollingApplyOutcome {
    /// Live/Delayed — ещё poll.
    Continue,
    /// Конец без spawn PnL/TP.
    Terminal,
    /// Matched open → поставить TP.
    TerminalTriggerTp(SharedOpenPosition),
    /// Matched SELL close → REST/trades + finalize.
    TerminalFinalizeClose(SharedClosingPosition),
    /// Matched TP maker → REST/trades + finalize TP.
    TerminalFinalizeTp(SharedOpenPosition),
}

impl PollingPositionKind {
    /// Короткое имя варианта для логов.
    fn variant_name(&self) -> &'static str {
        match self {
            Self::Open(_) => "Open",
            Self::Close(_) => "Close",
            Self::TpMaker(_) => "TpMaker",
        }
    }

    /// `open_order_id` / `close_order_id` / `tp_order_id`; `None` — poll не стартуем.
    async fn snapshot_order_id(&self) -> Option<String> {
        match self {
            Self::Open(pos_arc) => pos_arc.read().await.open_order_id.clone(),
            Self::Close(c_arc) => c_arc.read().await.close_order_id.clone(),
            Self::TpMaker(pos_arc) => pos_arc.read().await.tp_order_id.clone(),
        }
    }

    /// `OpenPosition.id` для логов (`Close` через `c.position`).
    async fn pos_id(&self) -> String {
        match self {
            Self::Open(pos_arc) | Self::TpMaker(pos_arc) => pos_arc.read().await.id.clone(),
            Self::Close(c_arc) => {
                let pos_arc = {
                    let c = c_arc.read().await;
                    c.position.clone()
                };
                let id = pos_arc.read().await.id.clone();
                id
            }
        }
    }

    /// Дедлайн poll (`event_end_ms` или fallback `POLL_TIMEOUT_SEC`).
    async fn event_end_ms(&self) -> Option<i64> {
        match self {
            Self::Open(pos_arc) | Self::TpMaker(pos_arc) => pos_arc.read().await.event_end_ms,
            Self::Close(c_arc) => {
                let pos_arc = {
                    let c = c_arc.read().await;
                    c.position.clone()
                };
                pos_arc.read().await.event_end_ms
            }
        }
    }
}

/// Локальные переходы как у WS; HTTP (trades, TP) — только в [`spawn_polling_verify`] по [`PollingApplyOutcome`].
pub(crate) async fn apply_order_status_from_polling(
    status: &OrderStatusType,
    kind: PollingPositionKind,
) -> PollingApplyOutcome {
    use OrderStatusType::*;
    match kind {
        PollingPositionKind::Open(pos_arc) => {
            let new_status = match status {
                Matched => OpenPositionStatus::Open,
                Canceled => OpenPositionStatus::OpenFailed,
                _ => return PollingApplyOutcome::Continue,
            };
            let trigger_tp = {
                let mut pos = pos_arc.write().await;
                let was_pending = matches!(pos.open_status, OpenPositionStatus::PendingOpen);
                pos.open_status = new_status;
                let oid = pos.open_order_id.clone();
                let pos_id = pos.id.clone();
                drop(pos);
                crate::tee_println!(
                    "[account_submit/poll] open_status({oid:?}) → {new_status:?} (pos_id={pos_id})",
                );
                was_pending && matches!(new_status, OpenPositionStatus::Open)
            };
            if trigger_tp {
                PollingApplyOutcome::TerminalTriggerTp(pos_arc)
            } else {
                PollingApplyOutcome::Terminal
            }
        }
        PollingPositionKind::Close(c_arc) => match status {
            Matched => {
                let (oid, pos_id) = {
                    let c = c_arc.read().await;
                    let pos_arc_inner = c.position.clone();
                    let oid = c.close_order_id.clone();
                    drop(c);
                    let pos_id = pos_arc_inner.read().await.id.clone();
                    (oid, pos_id)
                };
                crate::tee_println!(
                    "[account_submit/poll] close_status({oid:?}) → Matched (PnL-финализация в caller'е) (pos_id={pos_id})",
                );
                PollingApplyOutcome::TerminalFinalizeClose(c_arc)
            }
            Canceled => {
                let (oid, pos_arc_inner) = {
                    let mut c = c_arc.write().await;
                    c.close_status = ClosingPositionStatus::CloseFailed;
                    (c.close_order_id.clone(), c.position.clone())
                };
                let pos_id = pos_arc_inner.read().await.id.clone();
                crate::tee_println!(
                    "[account_submit/poll] close_status({oid:?}) → CloseFailed (pos_id={pos_id})",
                );
                PollingApplyOutcome::Terminal
            }
            _ => PollingApplyOutcome::Continue,
        },
        PollingPositionKind::TpMaker(pos_arc) => match status {
            Matched => {
                let (tp_id, pos_id) = {
                    let p = pos_arc.read().await;
                    (p.tp_order_id.clone(), p.id.clone())
                };
                crate::tee_println!(
                    "[account_submit/poll] tp_order_id({tp_id:?}) → Matched (PnL-финализация в caller'е) (pos_id={pos_id})",
                );
                PollingApplyOutcome::TerminalFinalizeTp(pos_arc)
            }
            Canceled => {
                let (tp_id, pos_id) = {
                    let p = pos_arc.read().await;
                    (p.tp_order_id.clone(), p.id.clone())
                };
                crate::tee_println!(
                    "[account_submit/poll] tp_order_id({tp_id:?}) → Canceled (no-op, close-flow продолжит) (pos_id={pos_id})",
                );
                PollingApplyOutcome::Terminal
            }
            _ => PollingApplyOutcome::Continue,
        },
    }
}

/// После poll Matched на SELL close: опционально REST fills, затем `finalize_close_pnl_in_place`.
async fn drive_close_pnl_finalization_via_polling(
    account: &SharedAccount,
    c_arc: &SharedClosingPosition,
) {
    let pos_arc = {
        let c = c_arc.read().await;
        c.position.clone()
    };
    let (pnl_finalized, pos_id) = {
        let p = pos_arc.read().await;
        (p.pnl_finalized, p.id.clone())
    };
    let (pnl_already_some, oid) = {
        let c = c_arc.read().await;
        (c.pnl.is_some(), c.close_order_id.clone())
    };

    if pnl_finalized {
        crate::tee_println!(
            "[account_submit/poll] close_status({oid:?}) → Closed (PnL уже финализирован WS, no-op) (pos_id={pos_id})",
        );
        return;
    }

    if !pnl_already_some {
        if let Some(order_id) = oid.as_deref() {
            fetch_and_apply_trades_for_order(account, &pos_id, order_id, OrderRole::Taker).await;
        }
    } else {
        crate::tee_println!(
            "[account_submit/poll] close_status({oid:?}) → Closed (WS уже накопил c.pnl, REST-fallback не нужен) (pos_id={pos_id})",
        );
    }

    {
        let mut c = c_arc.write().await;
        c.close_status = ClosingPositionStatus::Closed;
    }
    crate::tee_println!("[account_submit/poll] close_status({oid:?}) → Closed (pos_id={pos_id})",);
    crate::account_ws::finalize_close_pnl_in_place(account, c_arc.clone(), "Polling").await;
}

/// После poll Matched на TP: финал если уже есть ClosingPosition; иначе trades + `apply_sell_fill`.
async fn drive_tp_pnl_finalization_via_polling(
    account: &SharedAccount,
    pos_arc: &SharedOpenPosition,
) {
    let (tp_order_id, pnl_finalized, pos_id, existing_close) = {
        let p = pos_arc.read().await;
        (
            p.tp_order_id.clone(),
            p.pnl_finalized,
            p.id.clone(),
            p.closing_position
                .as_ref()
                .and_then(std::sync::Weak::upgrade),
        )
    };
    let Some(tp_id) = tp_order_id else {
        return;
    };

    if pnl_finalized {
        crate::tee_println!(
            "[account_submit/poll] tp_order_id({tp_id}) → Matched (PnL уже финализирован WS, no-op) (pos_id={pos_id})",
        );
        return;
    }

    if existing_close.is_some() {
        crate::tee_println!(
            "[account_submit/poll] tp_order_id({tp_id}) → Matched (ClosingPosition уже создана WS — финализируем) (pos_id={pos_id})",
        );
        crate::account_ws::finalize_tp_close_after_creation(account, &tp_id, "Polling").await;
    } else {
        crate::tee_println!(
            "[account_submit/poll] tp_order_id({tp_id}) → Matched (REST-fallback: тащим trades и финализируем) (pos_id={pos_id})",
        );
        fetch_and_apply_trades_for_order(account, &pos_id, &tp_id, OrderRole::Maker).await;
    }
}

/// `client.order` → `associate_trades`, затем `client.trades` по id; fills в `apply_sell_fill`.
/// `role`: taker — `taker_order_id`, maker — `maker_orders[].order_id`.
async fn fetch_and_apply_trades_for_order(
    account: &SharedAccount,
    pos_id: &str,
    order_id: &str,
    role: OrderRole,
) {
    let auth_client = match (**account.clob_authed.load()).clone() {
        Some(c) => c,
        None => {
            crate::tee_eprintln!(
                "[account_submit/poll-rest] auth-клиент пуст — REST-fallback пропускаем: pos_id={pos_id}, order_id={order_id}"
            );
            return;
        }
    };
    let order_resp = match tokio::time::timeout(
        Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
        auth_client.order(order_id),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            crate::tee_eprintln!(
                "[account_submit/poll-rest] client.order({order_id}) упал: {err:#} (pos_id={pos_id})"
            );
            return;
        }
        Err(_) => {
            crate::tee_eprintln!(
                "[account_submit/poll-rest] client.order({order_id}) timeout (pos_id={pos_id})"
            );
            return;
        }
    };
    let trade_ids: Vec<String> = order_resp.associate_trades;
    if trade_ids.is_empty() {
        crate::tee_println!(
            "[account_submit/poll-rest] order_id={order_id} role={role:?}: associate_trades пуст — нечего применять (pos_id={pos_id})",
        );
        return;
    }

    let mut applied_count: usize = 0;
    for trade_id in trade_ids {
        let request = TradesRequest::builder().id(trade_id.clone()).build(); // фильтр по trade id
        let page = match tokio::time::timeout(
            Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
            auth_client.trades(&request, None),
        )
        .await
        {
            Ok(Ok(p)) => p,
            Ok(Err(err)) => {
                crate::tee_eprintln!(
                    "[account_submit/poll-rest] client.trades(id={trade_id}) упал: {err:#} (pos_id={pos_id}, order_id={order_id})"
                );
                continue;
            }
            Err(_) => {
                crate::tee_eprintln!(
                    "[account_submit/poll-rest] client.trades(id={trade_id}) timeout (pos_id={pos_id}, order_id={order_id})"
                );
                continue;
            }
        };
        for trade in page.data.iter() {
            match role {
                OrderRole::Taker => {
                    if trade.taker_order_id != order_id {
                        continue;
                    }
                    let size = decimal_to_f64(&trade.size);
                    let price = decimal_to_f64(&trade.price);
                    let fee_rate_bps = decimal_to_f64(&trade.fee_rate_bps);
                    if !(size > 0.0 && size.is_finite()) || !(price > 0.0 && price.is_finite()) {
                        continue;
                    }
                    crate::account_ws::apply_sell_fill(
                        account,
                        order_id,
                        size,
                        price,
                        fee_rate_bps,
                        OrderRole::Taker,
                    )
                    .await;
                    applied_count += 1;
                }
                OrderRole::Maker => {
                    for m in trade.maker_orders.iter() {
                        if m.order_id != order_id {
                            continue;
                        }
                        let size = decimal_to_f64(&m.matched_amount);
                        let price = decimal_to_f64(&m.price);
                        let fee_rate_bps = decimal_to_f64(&m.fee_rate_bps);
                        if !(size > 0.0 && size.is_finite()) || !(price > 0.0 && price.is_finite())
                        {
                            continue;
                        }
                        crate::account_ws::apply_sell_fill(
                            account,
                            order_id,
                            size,
                            price,
                            fee_rate_bps,
                            OrderRole::Maker,
                        )
                        .await;
                        applied_count += 1;
                    }
                }
            }
        }
    }
    crate::tee_println!(
        "[account_submit/poll-rest] order_id={order_id} role={role:?}: applied {applied_count} fill(s) (pos_id={pos_id})",
    );
}

/// `Decimal` → `f64` через строку (без двоичного шума).
fn decimal_to_f64(d: &polymarket_client_sdk::types::Decimal) -> f64 {
    d.to_string().parse::<f64>().unwrap_or(0.0)
}

/// Цикл `client.order` до дедлайна (`event_end_ms` или now+[`POLL_TIMEOUT_SEC`]); исход → spawn TP/PnL.
fn spawn_polling_verify(account: SharedAccount, kind: PollingPositionKind) {
    tokio::spawn(async move {
        let kind_label = kind.variant_name();
        let pos_id = kind.pos_id().await;
        let order_id = match kind.snapshot_order_id().await {
            Some(id) => id,
            None => {
                crate::tee_eprintln!(
                    "[account_submit/poll] {kind_label}: real order_id ещё не получен — polling не запускаем (pos_id={pos_id})"
                );
                return;
            }
        };
        let now_ms = crate::util::current_timestamp_ms();
        let deadline_ms: i64 = match kind.event_end_ms().await {
            Some(end) if end > now_ms => end,
            Some(end) => {
                crate::tee_eprintln!(
                    "[account_submit/poll] {kind_label} order_id={order_id}: event_end_ms={end} уже в прошлом (now={now_ms}) — polling не запускаем (pos_id={pos_id})"
                );
                return;
            }
            None => now_ms.saturating_add((POLL_TIMEOUT_SEC as i64) * 1_000),
        };
        let mut tick = tokio::time::interval(Duration::from_secs(POLL_INTERVAL_SEC));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await;
        loop {
            tick.tick().await;
            let now_ms = crate::util::current_timestamp_ms();
            if now_ms >= deadline_ms {
                crate::tee_eprintln!(
                    "[account_submit/poll] {kind_label} order_id={order_id} — дедлайн event_end_ms={deadline_ms} достигнут, бросаем polling (pos_id={pos_id})"
                );
                return;
            }
            let auth_client = match (**account.clob_authed.load()).clone() {
                Some(c) => c,
                None => {
                    continue;
                }
            };
            let resp = match tokio::time::timeout(
                Duration::from_secs(ORDER_HTTP_TIMEOUT_SEC),
                auth_client.order(&order_id),
            )
            .await
            {
                Ok(Ok(r)) => r,
                Ok(Err(err)) => {
                    crate::tee_eprintln!(
                        "[account_submit/poll] {kind_label} client.order({order_id}) упал: {err:#} (pos_id={pos_id})",
                    );
                    continue;
                }
                Err(_) => {
                    crate::tee_eprintln!(
                        "[account_submit/poll] {kind_label} client.order({order_id}) таймаут (pos_id={pos_id})"
                    );
                    continue;
                }
            };
            let outcome = apply_order_status_from_polling(&resp.status, kind.clone()).await;
            match outcome {
                PollingApplyOutcome::Continue => {}
                PollingApplyOutcome::Terminal => {
                    crate::tee_println!(
                        "[account_submit/poll] {kind_label} order_id={order_id} терминальный статус {:?}, polling завершён (pos_id={pos_id})",
                        resp.status,
                    );
                    return;
                }
                PollingApplyOutcome::TerminalTriggerTp(pos_arc) => {
                    let acc = account.clone();
                    tokio::spawn(async move {
                        try_place_tp_maker(acc, pos_arc).await;
                    });
                    crate::tee_println!(
                        "[account_submit/poll] {kind_label} order_id={order_id} терминальный статус {:?}, polling завершён, TP-задача запущена (pos_id={pos_id})",
                        resp.status,
                    );
                    return;
                }
                PollingApplyOutcome::TerminalFinalizeClose(c_arc) => {
                    let acc = account.clone();
                    tokio::spawn(async move {
                        drive_close_pnl_finalization_via_polling(&acc, &c_arc).await;
                    });
                    crate::tee_println!(
                        "[account_submit/poll] {kind_label} order_id={order_id} терминальный статус {:?}, polling завершён, PnL-финализация (close) запущена (pos_id={pos_id})",
                        resp.status,
                    );
                    return;
                }
                PollingApplyOutcome::TerminalFinalizeTp(pos_arc) => {
                    let acc = account.clone();
                    tokio::spawn(async move {
                        drive_tp_pnl_finalization_via_polling(&acc, &pos_arc).await;
                    });
                    crate::tee_println!(
                        "[account_submit/poll] {kind_label} order_id={order_id} терминальный статус {:?}, polling завершён, PnL-финализация (TP) запущена (pos_id={pos_id})",
                        resp.status,
                    );
                    return;
                }
            }
        }
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
