//! Mock-вариант [`crate::account_order::post_order_on_clob`] / [`crate::account_order::cancel_order_on_clob`]
//! для real-time тестовой симуляции: ничего на CLOB не уходит, исполнение и `invoke`-колбэк
//! имитируются по WS-снапшоту
//! ([`crate::project_manager::ProjectManager::last_snapshot_by_asset_id`]).
//!
//! Сигнатуры **в точности** повторяют реальный [`crate::account_order`]; параметр
//! `project_manager` обязателен (без него нет источника WS-book'а).
//!
//! Поведение исполнения см. в модуле [`crate::account_mock_order_completion`].
//! Cancel здесь чисто фиктивный: всегда `canceled=true`, плюс по возможности сигналим
//! бегущей maker-таске остановиться (через
//! [`crate::account_mock_order_completion::signal_mock_order_cancel`]).

use crate::account::SharedAccount;
use crate::account_mock_order_completion::{
    register_mock_order_cancel_channel, signal_mock_order_cancel, spawn_mock_order_processor,
};
use crate::account_order::{
    CancelOrderRequest, CancelOrderResult, PostOrderRequest, validate_post_order_request,
};
use crate::account_order_completion::{
    SingleOrderInvokeCb, fire_failed_invocation_for_side, wrap_post_order_cb,
};
use crate::project_manager::ProjectManager;
use anyhow::{Result, anyhow, bail};
use std::sync::Arc;
use tokio::sync::oneshot;
use uuid::Uuid;

/// Префикс для фейковых `order_id` — отличаем mock от реального CLOB при дебаге логов.
const MOCK_ORDER_ID_PREFIX: &str = "mock-";

/// Mock POST /order:
///
/// * валидация — та же
///   ([`crate::account_order::validate_post_order_request`], реэкспорт как `pub(crate)`);
/// * генерируется фейковый `order_id` (`mock-<uuid>`);
/// * запускается отдельная таска
///   ([`crate::account_mock_order_completion::spawn_mock_order_processor`]):
///   - **taker** — мгновенный walk WS-стакана (всё или ничего, с учётом
///     `request.price`/`max_slippage_pp` и крипто-fee);
///   - **maker** — крутится до пересечения лимит-цены или `request.expiration`
///     (полный fill без fee, либо timeout/cancel — фейл);
/// * `invoke` фаирится **ровно один раз** в любом случае
///   ([`crate::account_order_completion::CompletionOnce`]).
///
/// Возвращает `Ok(Some(order_id))` для всех валидных заявок; провалы валидации/контекста
/// дополнительно фаирят failed-репорт и возвращают `Err`/`Ok(None)`.
pub async fn post_order_on_clob(
    account: &SharedAccount,
    project_manager: Option<&Arc<ProjectManager>>,
    request: PostOrderRequest,
    invoke: SingleOrderInvokeCb,
) -> Result<Option<String>> {
    let _ = account;
    let invoke_slot = wrap_post_order_cb(invoke);

    if let Err(validation_err) = validate_post_order_request(&request) {
        fire_failed_invocation_for_side(
            &invoke_slot,
            request.side,
            Some(format!(
                "mock_validate_post_order_request: {validation_err:#}"
            )),
        );
        return Err(validation_err);
    }

    let project_manager_arc = match project_manager {
        Some(pm) => Arc::clone(pm),
        None => {
            let msg = "mock_post_order_on_clob: project_manager=None — нет источника WS-book"
                .to_string();
            fire_failed_invocation_for_side(&invoke_slot, request.side, Some(msg.clone()));
            return Err(anyhow!(msg));
        }
    };

    let mock_order_id = format!("{MOCK_ORDER_ID_PREFIX}{}", Uuid::new_v4());
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    register_mock_order_cancel_channel(&mock_order_id, cancel_tx).await;

    spawn_mock_order_processor(
        project_manager_arc,
        request,
        mock_order_id.clone(),
        invoke_slot,
        cancel_rx,
    );

    Ok(Some(mock_order_id))
}

/// Mock DELETE /order: считаем отменённым **только** если для `order_id` нашлась активная
/// мок-таска и сигнал отмены был ей доставлен (тогда `canceled=true`, `error_msg=None`, а сама
/// таска однократно фаирит fail-репорт через `invoke`). Если в реестре пусто — ордер уже
/// завершился (taker отстрелял мгновенно, maker дождался цены либо `expiration`), отменять
/// нечего: `canceled=false` с диагностикой.
pub async fn cancel_order_on_clob(
    account: &SharedAccount,
    project_manager: Option<&Arc<ProjectManager>>,
    request: CancelOrderRequest,
) -> Result<CancelOrderResult> {
    let _ = account;
    let _ = project_manager;

    if request.timeout.is_zero() {
        bail!("mock_cancel_order_on_clob: timeout=0");
    }
    if request.order_id.is_empty() {
        bail!("mock_cancel_order_on_clob: пустой order_id");
    }

    let cancel_signal_delivered = signal_mock_order_cancel(&request.order_id).await;
    let (canceled, error_msg) = if cancel_signal_delivered {
        (true, None)
    } else {
        (
            false,
            Some(format!(
                "mock_cancel_no_pending_order: order_id={} уже завершён или не зарегистрирован",
                request.order_id,
            )),
        )
    };

    Ok(CancelOrderResult {
        order_id: request.order_id,
        canceled,
        error_msg,
    })
}
