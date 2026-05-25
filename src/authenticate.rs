//! CLOB L2 authenticate и периодический heartbeat ([`spawn_heartbeat`]).
//!
//! Креды из env (`POLY_PRIVATE_KEY`, опционально `POLY_DEPOSIT_WALLET`), результат в
//! [`crate::account::Account::clob_authed`] / [`crate::account::Account::clob_signer`].
//! Логи `[heartbeat]` — только в [`USER_STREAM_TEE_LOG`] (`xframes/last_user_stream.txt`).

use crate::account::{POLYMARKET_CLOB_HOST, SharedAccount};
use crate::account_proxy::PolyProxyEnvGuard;
use alloy::signers::Signer as _;
use alloy::signers::local::PrivateKeySigner;
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::Uuid as ClobUuid;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::clob::types::request::UpdateBalanceAllowanceRequest;
use polymarket_client_sdk::clob::types::{AssetType, SignatureType};
use polymarket_client_sdk::types::Address;
use polymarket_client_sdk::{POLYGON, derive_proxy_wallet};
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::MissedTickBehavior;

/// Интервал heartbeat CLOB (~5s; см. [доку heartbeat](https://docs.polymarket.com/developers/CLOB/orders/orders#heartbeat)). Без него сессия снимает ордера ~10s.
const CLOB_HEARTBEAT_INTERVAL_SEC: u64 = 5;

/// Env: EOA hex для CLOB-auth и split; пусто → [`try_authenticate_clob_for_heartbeats`] noop.
pub const POLY_PRIVATE_KEY_ENV: &str = "POLY_PRIVATE_KEY";

/// Env: funder deposit для `POLY_1271`; совпадение с Safe или proxy → тот профиль, не deposit.
const POLY_DEPOSIT_WALLET_ENV: &str = "POLY_DEPOSIT_WALLET";

/// Профиль EIP-712 для `authentication_builder` (Safe / proxy / deposit).
#[derive(Debug, Clone, Copy)]
enum ClobAuthProfile {
    /// CREATE2 Safe от EOA (как на сайте PM).
    GnosisSafe { safe: Address },
    /// `derive_proxy_wallet` (Magic/email).
    Proxy { proxy: Address },
    /// Отдельный on-chain funder, `signature_type=POLY_1271`.
    Poly1271 { funder: Address },
}

fn resolve_clob_auth_profile(eoa: Address) -> Option<ClobAuthProfile> {
    let safe = crate::poly_chain::derive_safe_address(eoa);
    let proxy = derive_proxy_wallet(eoa, POLYGON);

    if let Ok(raw) = std::env::var(POLY_DEPOSIT_WALLET_ENV) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let configured = match Address::from_str(trimmed) {
                Ok(addr) => addr,
                Err(err) => {
                    crate::user_stream_tee_eprintln!(
                        "[heartbeat] парсинг {POLY_DEPOSIT_WALLET_ENV}={trimmed:?} провалился: {err:#}; \
                         CLOB heartbeat отключён",
                    );
                    return None;
                }
            };
            if configured == safe {
                crate::user_stream_tee_eprintln!(
                    "[heartbeat] {POLY_DEPOSIT_WALLET_ENV}={configured:#x} совпадает с Polymarket Safe — \
                     используем GnosisSafe, не Poly1271 deposit",
                );
                return Some(ClobAuthProfile::GnosisSafe { safe });
            }
            if proxy.is_some_and(|proxy_addr| configured == proxy_addr) {
                crate::user_stream_tee_eprintln!(
                    "[heartbeat] {POLY_DEPOSIT_WALLET_ENV}={configured:#x} совпадает с Polymarket Proxy — \
                     используем Proxy, не Poly1271 deposit",
                );
                return Some(ClobAuthProfile::Proxy { proxy: configured });
            }
            return Some(ClobAuthProfile::Poly1271 { funder: configured });
        }
    }

    Some(ClobAuthProfile::GnosisSafe { safe })
}

/// Подряд ошибок heartbeat до принудительного re-auth ([`try_authenticate_clob_for_heartbeats_with_force`], `force=true`).
const HEARTBEAT_FAILS_BEFORE_REAUTH: u32 = 2;

/// Таск: каждые [`CLOB_HEARTBEAT_INTERVAL_SEC`]s `post_heartbeat`; без auth — noop; `MissedTickBehavior::Delay`; см. [heartbeat](https://docs.polymarket.com/developers/CLOB/orders/orders#heartbeat).
pub fn spawn_heartbeat(account: SharedAccount) {
    tokio::spawn(async move {
        let mut clob_tick = tokio::time::interval(Duration::from_secs(CLOB_HEARTBEAT_INTERVAL_SEC));
        clob_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        clob_tick.tick().await;

        let mut heartbeat_id: Option<ClobUuid> = None;

        let mut last_was_success = false;
        let mut had_first_log = false;
        let mut consecutive_errors: u32 = 0;

        loop {
            clob_tick.tick().await;
            let auth_client: Option<clob::Client<Authenticated<Normal>>> =
                (**account.clob_authed.load()).clone();
            let Some(client) = auth_client.as_ref() else {
                continue;
            };
            match client.post_heartbeat(heartbeat_id).await {
                Ok(resp) => {
                    heartbeat_id = Some(resp.heartbeat_id);
                    if !had_first_log {
                        crate::user_stream_tee_println!(
                            "[heartbeat] CLOB heartbeat OK (heartbeat_id={})",
                            resp.heartbeat_id,
                        );
                        had_first_log = true;
                    } else if !last_was_success {
                        crate::user_stream_tee_println!(
                            "[heartbeat] CLOB heartbeat восстановлен после {consecutive_errors} ошибок (heartbeat_id={})",
                            resp.heartbeat_id,
                        );
                    }
                    last_was_success = true;
                    consecutive_errors = 0;
                }
                Err(err) => {
                    consecutive_errors = consecutive_errors.saturating_add(1);
                    if last_was_success || !had_first_log {
                        crate::user_stream_tee_eprintln!(
                            "[heartbeat] CLOB heartbeat ошибка #{consecutive_errors}: {err:#} \
                             (открытые ордера могут быть отменены при тишине > 10s)",
                        );
                        had_first_log = true;
                    }
                    last_was_success = false;
                    heartbeat_id = None;

                    // Несколько фейлов подряд — force re-auth.
                    if consecutive_errors >= HEARTBEAT_FAILS_BEFORE_REAUTH {
                        crate::user_stream_tee_eprintln!(
                            "[heartbeat] {consecutive_errors} подряд ошибок — пробуем форсированный re-auth (force=true)"
                        );
                        try_authenticate_clob_for_heartbeats_with_force(&account, true).await;
                        consecutive_errors = 0;
                    }
                }
            }
        }
    });
}

/// CLOB-auth по env EOA (`POLY_PRIVATE_KEY`); кладёт authed клиент и signer в [`Account`]. Идемпотентно; при ошибках клиент `None`, heartbeat без POST.
pub async fn try_authenticate_clob_for_heartbeats(account: &SharedAccount) {
    try_authenticate_clob_for_heartbeats_with_force(account, false).await
}

/// Полный цикл authenticate; [`HEARTBEAT_FAILS_BEFORE_REAUTH`] задаёт когда heartbeat зовёт с `force=true` (повторная выдача L2, ArcSwap без локов).
async fn try_authenticate_clob_for_heartbeats_with_force(account: &SharedAccount, force: bool) {
    if !force && account.clob_authed.load().is_some() {
        return;
    }
    let private_key = match std::env::var(POLY_PRIVATE_KEY_ENV) {
        Ok(s) if !s.trim().is_empty() => s,
        Ok(_) | Err(_) => return,
    };
    let signer: PrivateKeySigner = match private_key.trim().parse() {
        Ok(s) => s,
        Err(err) => {
            crate::user_stream_tee_eprintln!(
                "[heartbeat] парсинг {POLY_PRIVATE_KEY_ENV} провалился: {err:#}; CLOB heartbeat отключён",
            );
            return;
        }
    };
    let signer = signer.with_chain_id(Some(POLYGON));
    let eoa = signer.address();
    let Some(profile) = resolve_clob_auth_profile(eoa) else {
        return;
    };
    if matches!(&profile, ClobAuthProfile::Poly1271 { .. }) {
        if let Err(err) =
            crate::poly_chain::ensure_deposit_wallet_deployed(account.http.as_ref(), eoa).await
        {
            crate::user_stream_tee_eprintln!(
                "[heartbeat] deposit wallet WALLET-CREATE провалился: {err:#}",
            );
        }
    }
    // Свежий unauth-клиент: SDK `authenticate()` делает `Arc::into_inner(self.client.inner)` и требует
    // refcount == 1. Если бы мы хранили один экземпляр и клонировали его, inner-Arc был бы у двух владельцев,
    // и SDK возвращал бы `Synchronization: multiple threads are attempting to log in or log out`.
    let proxy_env = PolyProxyEnvGuard::install_from_env();
    let unauth = match clob::Client::new(POLYMARKET_CLOB_HOST, clob::Config::default()) {
        Ok(c) => c,
        Err(err) => {
            PolyProxyEnvGuard::uninstall_from_env(proxy_env);
            crate::user_stream_tee_eprintln!(
                "[heartbeat] построить unauth CLOB-клиент не удалось: {err:#}; CLOB heartbeat отключён",
            );
            return;
        }
    };
    PolyProxyEnvGuard::uninstall_from_env(proxy_env);

    let auth_result = match profile {
        ClobAuthProfile::GnosisSafe { safe } => {
            unauth
                .authentication_builder(&signer)
                .signature_type(SignatureType::GnosisSafe)
                .funder(safe)
                .authenticate()
                .await
        }
        ClobAuthProfile::Proxy { proxy } => {
            unauth
                .authentication_builder(&signer)
                .signature_type(SignatureType::Proxy)
                .funder(proxy)
                .authenticate()
                .await
        }
        ClobAuthProfile::Poly1271 { funder } => {
            unauth
                .authentication_builder(&signer)
                .signature_type(SignatureType::Poly1271)
                .funder(funder)
                .authenticate()
                .await
        }
    };

    match auth_result {
        Ok(authed) => {
            if matches!(profile, ClobAuthProfile::Poly1271 { .. }) {
                let balance_sync = authed
                    .update_balance_allowance(
                        UpdateBalanceAllowanceRequest::builder()
                            .asset_type(AssetType::Collateral)
                            .build(),
                    )
                    .await;
                if let Err(err) = balance_sync {
                    crate::user_stream_tee_eprintln!(
                        "[heartbeat] CLOB balance-allowance/update (collateral) провалился: {err:#}",
                    );
                }
            }
            account.clob_authed.store(Arc::new(Some(authed)));
            account.clob_signer.store(Arc::new(Some(signer)));
            let mode = if force {
                "FORCE re-auth"
            } else {
                "authenticate"
            };
            let (funder, sig_label): (Address, &'static str) = match profile {
                ClobAuthProfile::GnosisSafe { safe } => (safe, "GnosisSafe"),
                ClobAuthProfile::Proxy { proxy } => (proxy, "Proxy"),
                ClobAuthProfile::Poly1271 { funder } => (funder, "Poly1271"),
            };
            crate::user_stream_tee_println!(
                "[heartbeat] CLOB {mode} OK (eoa={eoa:#x}, funder={funder:#x}, signature_type={sig_label}); \
                 heartbeat каждые {CLOB_HEARTBEAT_INTERVAL_SEC}s",
            );
        }
        Err(err) => {
            let mode = if force {
                "FORCE re-auth"
            } else {
                "authenticate"
            };
            crate::user_stream_tee_eprintln!(
                "[heartbeat] CLOB {mode} провалился: {err:#}; CLOB heartbeat отключён (для re-auth — следующая попытка через {HEARTBEAT_FAILS_BEFORE_REAUTH} ошибок)",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::Account;

    /// Интеграционный smoke: полный auth + два heartbeat; игнорируется в CI без ключа (см. `#[ignore]`).
    #[tokio::test]
    #[ignore = "live network: требует POLY_PRIVATE_KEY; делает HTTP к clob.polymarket.com/auth/api-key"]
    async fn live_try_authenticate_clob_for_heartbeats() -> anyhow::Result<()> {
        let _ = dotenvy::dotenv();

        // CryptoProvider нужен tls до первого запроса (в bin делает main).
        let _ = rustls::crypto::ring::default_provider().install_default();

        let private_key_set = std::env::var(POLY_PRIVATE_KEY_ENV)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .is_some();
        if !private_key_set {
            eprintln!(
                "live_try_authenticate_clob_for_heartbeats: {POLY_PRIVATE_KEY_ENV} не задан, тест пропущен",
            );
            return Ok(());
        }

        let account = Account::new_shared();

        anyhow::ensure!(
            account.clob_authed.load().is_none(),
            "новый Account должен идти с clob_authed=Arc::new(None)",
        );

        try_authenticate_clob_for_heartbeats(&account).await;
        anyhow::ensure!(
            account.clob_authed.load().is_some(),
            "после try_authenticate_clob_for_heartbeats clob_authed обязан быть Some \
             (если упало — смотри `xframes/last_user_stream.txt`: `[heartbeat] CLOB authenticate провалился: …`)",
        );

        let before = account.clob_authed.load_full();
        try_authenticate_clob_for_heartbeats(&account).await;
        let after = account.clob_authed.load_full();
        anyhow::ensure!(
            Arc::ptr_eq(&before, &after),
            "идемпотентность нарушена: clob_authed Arc был пересоздан повторным auth-вызовом",
        );

        let client = (**account.clob_authed.load())
            .clone()
            .expect("clob_authed обязан быть Some после успешного auth-цикла выше");

        let first = client
            .post_heartbeat(None)
            .await
            .map_err(|err| anyhow::anyhow!("первый POST /v1/heartbeats упал: {err:#}"))?;
        eprintln!(
            "live_try_authenticate_clob_for_heartbeats: первый heartbeat OK, heartbeat_id={}",
            first.heartbeat_id,
        );

        let second = client
            .post_heartbeat(Some(first.heartbeat_id))
            .await
            .map_err(|err| {
                anyhow::anyhow!("повторный POST /v1/heartbeats с chained id упал: {err:#}")
            })?;
        eprintln!(
            "live_try_authenticate_clob_for_heartbeats: chained heartbeat OK, heartbeat_id={}",
            second.heartbeat_id,
        );

        Ok(())
    }
}
