//! Капитал и MtM (`bankroll`, peak, max DD); per-lane позиции и CLOB-клиенты.
//! Один [`SharedAccount`] на процесс: поля под отдельными `RwLock`, auth в [`ArcSwapAny`] (read-mostly).
//! Порядок локов: `bankroll` → `peak_bankroll` → `max_drawdown_pct` → `last_prob` → `positions` → `pending_close_positions` → `closing` → `recently_resolved_markets` → один inner на позицию.

use crate::account_order_completion::TrackerEntry;
use crate::account_proxy::PolyProxyEnvGuard;
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{CloseReason, INITIAL_BANKROLL, LanePositions, SharedOpenPosition};
use crate::real_sim::RealSimState;
use crate::sim_stats::SimStats;
use alloy::signers::Signer as _;
use alloy::signers::local::PrivateKeySigner;
use arc_swap::ArcSwapAny;
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::Uuid as ClobUuid;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::clob::types::request::UpdateBalanceAllowanceRequest;
use polymarket_client_sdk::clob::types::{AssetType, SignatureType};
use polymarket_client_sdk::data;
use polymarket_client_sdk::gamma;
use polymarket_client_sdk::types::Address;
use polymarket_client_sdk::{POLYGON, derive_proxy_wallet};
use std::collections::HashMap;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::MissedTickBehavior;

/// Production CLOB V2 host (cutover 2026-04-28; см.
/// [docs.polymarket.com/v2-migration](https://docs.polymarket.com/v2-migration)).
///
/// Старый `clob-v2.polymarket.com` остаётся как pre-cutover testing host и
/// перестал маршрутизировать `POST /order` — edge отдаёт `405` с пустым телом
/// (auth/GET-ручки ещё отвечают, поэтому ошибка ловится только на отправке
/// ордера). Дефолт SDK (`clob::Client::default()`) указывает на тот старый
/// хост — поэтому мы конструируем клиент явно с production-URL.
pub const POLYMARKET_CLOB_HOST: &str = "https://clob.polymarket.com";

/// `Arc<Account>`; синхронизация только на полях [`Account`].
pub type SharedAccount = Arc<Account>;

/// Ключ лейна: `(currency, interval, side)` — маршрут в maps и `last_prob`.
pub type LaneKey = (String, XFrameIntervalKind, CurrencyUpDownOutcome);

/// Счёт процесса: cash, MtM, позиции, CLOB. См. модульный комментарий про порядок локов и один inner на позицию.
#[derive(Debug)]
pub struct Account {
    /// Реализованный USDC (cash).
    pub bankroll: Arc<RwLock<f64>>,
    /// Пик equity (MtM).
    pub peak_bankroll: Arc<RwLock<f64>>,
    /// Max drawdown, % от пика.
    pub max_drawdown_pct: Arc<RwLock<f64>>,
    /// Последняя implied prob по лейну (MtM, по `currency` в ключе).
    pub last_prob: Arc<RwLock<HashMap<LaneKey, f64>>>,
    /// Открытые позиции; тот же `Arc`, что в записи закрытия (`position`).
    /// Для одной лейны могут сосуществовать позиции разных `market_id` — старые
    /// (с чужим `asset_id` относительно текущего кадра) живут здесь как
    /// «припаркованные» до резолюции; см. [`Self::resolve_pending_market_sync`].
    pub positions: Arc<RwLock<HashMap<LaneKey, LanePositions>>>,
    /// Submit/Mock-only: позиции, на которых уже сработал
    /// [`crate::history_sim::sell_gate`] и спавнен
    /// [`crate::account_submit::spawn_sell_taker`], но
    /// [`crate::account_close_position::close_position_after_submit`] ещё не
    /// финализировал PnL (BUY/maker-TP/taker-FAK invoke'ы в полёте).
    pub pending_close_positions: Arc<RwLock<HashMap<LaneKey, LanePositions>>>,
    /// HTTP с rustls; тот же `Arc`, что у [`ProjectManager::http`](crate::project_manager::ProjectManager::http).
    pub http: Arc<reqwest::Client>,
    /// Общий unauth CLOB SDK-клиент (клоны в PM и др.).
    pub clob: Arc<clob::Client>,
    /// Gamma API SDK-клиент; тот же `Arc`, что у [`ProjectManager::gamma`](crate::project_manager::ProjectManager::gamma).
    pub gamma: Arc<gamma::Client>,
    /// Polymarket Data API (`/positions` и др.).
    pub data: Arc<data::Client>,
    /// Authed-сессия: heartbeat и ордеры ([`crate::account_order`]).
    pub clob_authed: ArcSwapAny<Arc<Option<clob::Client<Authenticated<Normal>>>>>,
    /// EOA-подписант под ордеры; задаётся вместе с `clob_authed`.
    pub clob_signer: ArcSwapAny<Arc<Option<PrivateKeySigner>>>,
    /// Трекер единоразовых колбэков POST /order (WS + REST fallback).
    pub order_invoke_hub: Arc<RwLock<HashMap<String, TrackerEntry>>>,
    /// `currency` → [`RealSimState`]; лок отдельно от цепочки `bankroll → …`.
    pub real_sim_state_by_currency: Arc<RwLock<HashMap<String, Arc<RwLock<RealSimState>>>>>,
}

impl Account {
    pub fn new() -> Self {
        // Если в `.env` заданы `POLY_PROXY_IP` + `POLY_PROXY_PORT` (и опционально пароль / логин), временно
        // выставляем `HTTP_PROXY`/`HTTPS_PROXY` для сборки клиентов (reqwest подхватывает их по умолчанию).
        let poly_proxy_env = PolyProxyEnvGuard::install_from_env();

        // Production CLOB V2 host (см. [`POLYMARKET_CLOB_HOST`]): `clob.polymarket.com`.
        // `clob::Client::default()` указывает на pre-cutover testing host
        // `clob-v2.polymarket.com`, где `POST /order` уже не работает.
        let clob = Arc::new(
            clob::Client::new(POLYMARKET_CLOB_HOST, clob::Config::default())
                .expect("CLOB client with production host should construct"),
        );
        let http = Arc::new(
            reqwest::Client::builder()
                .use_rustls_tls()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        );
        let gamma = Arc::new(gamma::Client::default());
        let data = Arc::new(data::Client::default());
        PolyProxyEnvGuard::uninstall_from_env(poly_proxy_env);
        Self {
            bankroll: Arc::new(RwLock::new(INITIAL_BANKROLL)),
            peak_bankroll: Arc::new(RwLock::new(INITIAL_BANKROLL)),
            max_drawdown_pct: Arc::new(RwLock::new(0.0)),
            last_prob: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            pending_close_positions: Arc::new(RwLock::new(HashMap::new())),
            http,
            clob,
            gamma,
            data,
            clob_authed: ArcSwapAny::new(Arc::new(None)),
            clob_signer: ArcSwapAny::new(Arc::new(None)),
            order_invoke_hub: Arc::new(RwLock::new(HashMap::new())),
            real_sim_state_by_currency: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Короткий read → clone Arc state; без state для валюты — [`None`].
    pub async fn real_sim_state_for_currency(
        &self,
        currency: &str,
    ) -> Option<Arc<RwLock<RealSimState>>> {
        self.real_sim_state_by_currency
            .read()
            .await
            .get(currency)
            .cloned()
    }

    /// Пустые buckets лейнов валюты ([`crate::real_sim::run_real_sim`]); `or_default` идемпотентен.
    /// Параллельно создаём bucket в [`Self::pending_close_positions`] под тем
    /// же [`LaneKey`], чтобы `tick_once` мог брать `.read()` без `or_default`.
    pub async fn register_currency_lanes(
        &self,
        currency: &str,
        lanes: &[(XFrameIntervalKind, CurrencyUpDownOutcome)],
    ) {
        let mut positions = self.positions.write().await;
        let mut pending = self.pending_close_positions.write().await;
        for (interval, side) in lanes {
            let key = (currency.to_string(), *interval, *side);
            positions.entry(key.clone()).or_default();
            pending.entry(key).or_default();
        }
    }

    /// Ядро: дренирует `positions[lane]` по `market_id` (включая «припаркованные» с чужим
    /// `asset_id` текущего тика), считает бинарный payout, обновляет `bankroll`/`SimStats`,
    /// пишет CSV + [`crate::trade_csv_log::record_market_outcome`].
    ///
    /// **Drain stale-маркетов:** позиции той же лейны (`currency`+`interval`), но с
    /// **другим** `market_id` (хвост от прошлого маркета, для которого resolution-событие
    /// не пришло), считаются полностью неактуальными — списываются как полная потеря
    /// (`pnl = -position_size`), `bankroll` корректируется, в `SimStats` они **не** попадают
    /// (это не легитимный trade outcome, а защита от утечки локнутого капитала), CSV-строка
    /// тоже не пишется. На каждый такой инцидент — `tee_eprintln!`-предупреждение.
    ///
    /// `final_price`: `None` → `pos.final_price`, иначе override для realtime.
    pub async fn resolve_pending_market_sync(
        account: &SharedAccount,
        sim_stats: &mut SimStats,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
        final_price: Option<f64>,
    ) {
        // bankroll → positions → recently_resolved

        let mut positions = account.positions.write().await;
        let mut to_close: Vec<(SharedOpenPosition, bool, CurrencyUpDownOutcome)> = Vec::new();

        for ((cur, int_kind, side), lane_positions) in positions.iter_mut() {
            if cur.as_str() != currency || *int_kind != interval {
                continue;
            }
            let token_won = match side {
                CurrencyUpDownOutcome::Up => up_won,
                CurrencyUpDownOutcome::Down => !up_won,
            };

            let pos_ids: Vec<String> = lane_positions.keys().cloned().collect();
            for pos_id in pos_ids {
                let Some(pos_arc) = lane_positions.get(&pos_id) else {
                    continue;
                };
                let matches_market = pos_arc.read().await.market_id == market_id;
                if !matches_market {
                    continue;
                }

                let Some(pos_arc) = lane_positions.remove(&pos_id) else {
                    continue;
                };
                to_close.push((pos_arc, token_won, *side));
            }
        }
        drop(positions);

        for (pos_arc, token_won, side) in to_close {
            let side_stats = match side {
                CurrencyUpDownOutcome::Up => &mut sim_stats.up,
                CurrencyUpDownOutcome::Down => &mut sim_stats.down,
            };
            // Записываем свежий `final_price` в саму позицию ДО `close_position`:
            // CSV-логгер внутри читает `pos.final_price` (override-параметра
            // больше нет). При None — оставляем то, что уже стояло (могло быть
            // выставлено в `open_position`).
            if let Some(fp) = final_price {
                pos_arc.write().await.final_price = Some(fp);
            }
            let reason = if token_won {
                CloseReason::ResolutionWin
            } else {
                CloseReason::ResolutionLoss
            };
            // `gross_usdc=None` → close_position сам выведет `shares_held` /
            // `0.0` из `reason` (см. дефолт внутри); `pos_arc`/`account`
            // разворачиваются под коротким локом ВНУТРИ функции — здесь не
            // держим ни pos.read, ни bankroll.write.
            crate::account_close_position::close_position(
                account,
                &pos_arc,
                side_stats,
                &reason,
                None,
                0,
            )
            .await;
        }

        crate::trade_csv_log::record_market_outcome(market_id, up_won);
    }

    /// `Arc::new(Account::new())` для `main` и PM.
    pub fn new_shared() -> SharedAccount {
        Arc::new(Self::new())
    }

    /// Обновляет пик и max DD от переданной MtM equity (`peak_bankroll` → `max_drawdown_pct`).
    pub async fn update_drawdown(&self, equity: f64) {
        let mut peak = self.peak_bankroll.write().await;
        let mut max_dd = self.max_drawdown_pct.write().await;
        if equity > *peak {
            *peak = equity;
        }
        if *peak > 0.0 {
            let drawdown_pct = (*peak - equity) / *peak * 100.0;
            if drawdown_pct > *max_dd {
                *max_dd = drawdown_pct;
            }
        }
    }
}

impl Default for Account {
    fn default() -> Self {
        Self::new()
    }
}

/// Интервал heartbeat CLOB (~5s; см. [доку heartbeat](https://docs.polymarket.com/developers/CLOB/orders/orders#heartbeat)). Без него сессия снимает ордера ~10s.
const CLOB_HEARTBEAT_INTERVAL_SEC: u64 = 5;

/// Env: EOA hex для CLOB-auth и split; пусто → [`try_authenticate_clob_for_heartbeats`] noop.
pub const POLY_PRIVATE_KEY_ENV: &str = "POLY_PRIVATE_KEY";

/// Env: funder deposit для `POLY_1271`; совпадение с Safe или proxy → тот профиль, не deposit.
pub(crate) const POLY_DEPOSIT_WALLET_ENV: &str = "POLY_DEPOSIT_WALLET";

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
                    crate::tee_eprintln!(
                        "[heartbeat] парсинг {POLY_DEPOSIT_WALLET_ENV}={trimmed:?} провалился: {err:#}; \
                         CLOB heartbeat отключён",
                    );
                    return None;
                }
            };
            if configured == safe {
                crate::tee_eprintln!(
                    "[heartbeat] {POLY_DEPOSIT_WALLET_ENV}={configured:#x} совпадает с Polymarket Safe — \
                     используем GnosisSafe, не Poly1271 deposit",
                );
                return Some(ClobAuthProfile::GnosisSafe { safe });
            }
            if proxy.is_some_and(|proxy_addr| configured == proxy_addr) {
                crate::tee_eprintln!(
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
                        crate::tee_println!(
                            "[heartbeat] CLOB heartbeat OK (heartbeat_id={})",
                            resp.heartbeat_id,
                        );
                        had_first_log = true;
                    } else if !last_was_success {
                        crate::tee_println!(
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
                        crate::tee_eprintln!(
                            "[heartbeat] CLOB heartbeat ошибка #{consecutive_errors}: {err:#} \
                             (открытые ордера могут быть отменены при тишине > 10s)",
                        );
                        had_first_log = true;
                    }
                    last_was_success = false;
                    heartbeat_id = None;

                    // Несколько фейлов подряд — force re-auth.
                    if consecutive_errors >= HEARTBEAT_FAILS_BEFORE_REAUTH {
                        crate::tee_eprintln!(
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
pub(crate) async fn try_authenticate_clob_for_heartbeats_with_force(
    account: &SharedAccount,
    force: bool,
) {
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
            crate::tee_eprintln!(
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
            crate::tee_eprintln!(
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
            crate::tee_eprintln!(
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
                    crate::tee_eprintln!(
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
            crate::tee_println!(
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
            crate::tee_eprintln!(
                "[heartbeat] CLOB {mode} провалился: {err:#}; CLOB heartbeat отключён (для re-auth — следующая попытка через {HEARTBEAT_FAILS_BEFORE_REAUTH} ошибок)",
            );
        }
    }
}

// Ордерный API — [`crate::account_order`] (здесь только `clob_authed` / `clob_signer`).

#[cfg(test)]
mod tests {
    use super::*;

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
             (если упало — смотри stderr-логи `[heartbeat] CLOB authenticate провалился: …`)",
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
