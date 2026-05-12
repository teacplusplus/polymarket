//! Капитал и MtM (`bankroll`, peak, max DD); per-lane позиции и CLOB-клиенты.
//! Один [`SharedAccount`] на процесс: поля под отдельными `RwLock`, auth в [`ArcSwapAny`] (read-mostly).
//! Порядок локов: `bankroll` → `peak_bankroll` → `max_drawdown_pct` → `last_prob` → `positions` → `pending_resolution` → `closing` → `recently_resolved_markets` → один inner на позицию.

use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{INITIAL_BANKROLL, SharedClosingPosition, SharedOpenPosition};
use crate::sim_stats::SimStats;
use crate::real_sim::{RealSimState, interval_label, side_label};
use alloy::signers::Signer as _;
use alloy::signers::local::PrivateKeySigner;
use arc_swap::ArcSwapAny;
use indexmap::IndexSet;
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::Uuid as ClobUuid;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::clob::types::request::UpdateBalanceAllowanceRequest;
use polymarket_client_sdk::clob::types::{AssetType, SignatureType};
use polymarket_client_sdk::types::Address;
use polymarket_client_sdk::{POLYGON, derive_proxy_wallet};
use std::collections::HashMap;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::MissedTickBehavior;

/// Лимит [`recently_resolved_markets`]; переполнение — `shift_remove_index(0)`.
pub const RECENTLY_RESOLVED_MARKETS_CAP: usize = 8;

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
    /// Открытые позиции; тот же `Arc`, что в `pending_resolution` и в записи закрытия (`position`).
    pub positions: Arc<RwLock<HashMap<LaneKey, Vec<SharedOpenPosition>>>>,
    /// Старый маркет до резолюции; не `manage_positions`.
    pub pending_resolution: Arc<RwLock<HashMap<LaneKey, Vec<SharedOpenPosition>>>>,
    /// Закрытия для user-WS; lifecycle в [`crate::history_sim::manage_positions`].
    pub closing: Arc<RwLock<HashMap<LaneKey, Vec<SharedClosingPosition>>>>,
    /// Недавно зарезолвленные `market_id`; анти-повтор открытия ([`RECENTLY_RESOLVED_MARKETS_CAP`]).
    pub recently_resolved_markets: Arc<RwLock<IndexSet<String>>>,
    /// Общий unauth CLOB SDK-клиент (клоны в PM и др.).
    pub clob: Arc<clob::Client>,
    /// Authed-сессия: heartbeat и ордеры ([`crate::account_order`]).
    pub clob_authed: ArcSwapAny<Arc<Option<clob::Client<Authenticated<Normal>>>>>,
    /// EOA-подписант под ордеры; задаётся вместе с `clob_authed`.
    pub clob_signer: ArcSwapAny<Arc<Option<PrivateKeySigner>>>,
    /// `currency` → [`RealSimState`]; лок отдельно от цепочки `bankroll → …`.
    pub real_sim_state_by_currency: Arc<RwLock<HashMap<String, Arc<RwLock<RealSimState>>>>>,
}

impl Account {
    pub fn new() -> Self {
        // SDK v2: production CLOB host (старый `clob.polymarket.com` не тот API).
        let clob = Arc::new(clob::Client::default());
        Self {
            bankroll: Arc::new(RwLock::new(INITIAL_BANKROLL)),
            peak_bankroll: Arc::new(RwLock::new(INITIAL_BANKROLL)),
            max_drawdown_pct: Arc::new(RwLock::new(0.0)),
            last_prob: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            pending_resolution: Arc::new(RwLock::new(HashMap::new())),
            closing: Arc::new(RwLock::new(HashMap::new())),
            recently_resolved_markets: Arc::new(RwLock::new(IndexSet::new())),
            clob,
            clob_authed: ArcSwapAny::new(Arc::new(None)),
            clob_signer: ArcSwapAny::new(Arc::new(None)),
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
    pub async fn register_currency_lanes(
        &self,
        currency: &str,
        lanes: &[(XFrameIntervalKind, CurrencyUpDownOutcome)],
    ) {
        let mut positions = self.positions.write().await;
        let mut pending = self.pending_resolution.write().await;
        let mut closing = self.closing.write().await;
        for (interval, side) in lanes {
            let key = (currency.to_string(), *interval, *side);
            positions.entry(key.clone()).or_default();
            pending.entry(key.clone()).or_default();
            closing.entry(key).or_default();
        }
    }

    /// Бинарный payout по `market_id`: перенос из `positions` в `pending`, затем [`resolve_pending_market_sync`].
    /// `final_price` в CSV resolution; `up_won` — исход UP. Drawdown — на следующем тике.
    ///
    /// Локи: `RealSimState.write`, затем `bankroll` → `pending` → `recently_resolved` в этом порядке.
    pub async fn resolve_pending_market(
        account: &SharedAccount,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
        final_price: f64,
    ) {
        let Some(state) = account.real_sim_state_for_currency(currency).await else {
            return;
        };
        let mut state_guard = state.write().await;

        // Кандидаты на отмену TP после redeem (в sim `tp_order_id` нет — вектор пуст).
        let mut positions_with_tp: Vec<crate::history_sim::SharedOpenPosition> = Vec::new();

        // Carry matching `market_id` в pending. `pnl_finalized` — дроп из positions (без двойного PnL).
        let mut skipped_finalized: usize = 0;
        let mut skipped_non_redeemable: usize = 0;
        {
            let mut positions = account.positions.write().await;
            let mut pending = account.pending_resolution.write().await;
            for ((cur, int_kind, side), pos_vec) in positions.iter_mut() {
                if cur.as_str() != currency || *int_kind != interval {
                    continue;
                }
                let key = (cur.clone(), *int_kind, *side);
                let pending_vec = pending.entry(key).or_default();
                let mut idx = 0;
                while idx < pos_vec.len() {
                    let (
                        matches_market,
                        pnl_finalized,
                        redeemable,
                        open_status,
                        optimistic_fill_replaced,
                        pos_id_for_log,
                    ) = {
                        let pos_g = pos_vec[idx].read().await;
                        (
                            pos_g.market_id == market_id,
                            pos_g.pnl_finalized,
                            pos_g.is_redeemable_at_resolution(),
                            pos_g.open_status,
                            pos_g.optimistic_fill_replaced,
                            pos_g.id.clone(),
                        )
                    };
                    if !matches_market {
                        idx += 1;
                        continue;
                    }
                    if pnl_finalized {
                        let _ = pos_vec.swap_remove(idx);
                        skipped_finalized += 1;
                        crate::tee_println!(
                            "[resolve] skip already-finalized pos: pos_id={pos_id_for_log}, market_id={market_id}, currency={currency}, interval={interval:?} \
                             (PnL уже учтён WS-finalize'ом, не переносим в pending_resolution — иначе двойной счёт)"
                        );
                        continue;
                    }
                    if !redeemable {
                        // OpenFailed и т.п. без шер на Safe — не платим.
                        let _ = pos_vec.swap_remove(idx);
                        skipped_non_redeemable += 1;
                        crate::tee_println!(
                            "[resolve] skip non-redeemable pos: pos_id={pos_id_for_log}, market_id={market_id}, currency={currency}, interval={interval:?}, \
                             open_status={open_status:?}, optimistic_fill_replaced={optimistic_fill_replaced} \
                             (нет реальных шер на Safe — payout не начисляем, дропаем из positions)"
                        );
                        continue;
                    }
                    let pos_arc = pos_vec.swap_remove(idx);
                    let has_tp = pos_arc.read().await.tp_order_id.is_some();
                    if has_tp {
                        positions_with_tp.push(pos_arc.clone());
                    }
                    pending_vec.push(pos_arc);
                }
            }
        }
        if skipped_finalized > 0 {
            crate::tee_println!(
                "[resolve] market_id={market_id} currency={currency} interval={interval:?}: \
                 пропустили {skipped_finalized} уже финализированных позиций при carry в pending_resolution",
            );
        }
        if skipped_non_redeemable > 0 {
            crate::tee_println!(
                "[resolve] market_id={market_id} currency={currency} interval={interval:?}: \
                 пропустили {skipped_non_redeemable} non-redeemable позиций (OpenFailed / PendingOpen без real fills) при carry в pending_resolution",
            );
        }
        if !positions_with_tp.is_empty() {
            crate::account_submit::spawn_cancel_tp_orders_after_resolution(
                account.clone(),
                positions_with_tp,
            );
        }

        let sim_stats = state_guard
            .stats
            .get_mut(&interval)
            .expect("RealSimState.stats: оба интервала пред-инициализированы в new()");
        account
            .resolve_pending_market_sync(
                sim_stats,
                currency,
                interval,
                market_id,
                up_won,
                Some(final_price),
            )
            .await;
    }

    /// Ядро: `pending_resolution` + банкролл + CSV + [`crate::trade_csv_log::record_market_outcome`].
    ///
    /// `final_price`: `None` → `pos.final_price`, иначе override для realtime.
    pub async fn resolve_pending_market_sync(
        &self,
        sim_stats: &mut SimStats,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
        final_price: Option<f64>,
    ) {
        // bankroll → pending → recently_resolved
        let mut recently_resolved = self.recently_resolved_markets.write().await;
        if recently_resolved.insert(market_id.to_string()) {
            while recently_resolved.len() > RECENTLY_RESOLVED_MARKETS_CAP {
                recently_resolved.shift_remove_index(0);
            }
        }
        drop(recently_resolved);

        let mut bankroll = self.bankroll.write().await;
        let mut pending_resolution = self.pending_resolution.write().await;

        for ((cur, int_kind, side), vec) in pending_resolution.iter_mut() {
            if cur.as_str() != currency || *int_kind != interval {
                continue;
            }
            let token_won = match side {
                CurrencyUpDownOutcome::Up => up_won,
                CurrencyUpDownOutcome::Down => !up_won,
            };
            let side_stats = match side {
                CurrencyUpDownOutcome::Up => &mut sim_stats.up,
                CurrencyUpDownOutcome::Down => &mut sim_stats.down,
            };

            let mut i = 0;
            while i < vec.len() {
                let (matches_market, pnl_already_finalized, redeemable) = {
                    let g = vec[i].read().await;
                    (
                        g.market_id == market_id,
                        g.pnl_finalized,
                        g.is_redeemable_at_resolution(),
                    )
                };
                if !matches_market {
                    i += 1;
                    continue;
                }
                // Уже финализировано WS — без повторного payout.
                if pnl_already_finalized {
                    let pos_arc = vec.swap_remove(i);
                    let pos_id_for_log = pos_arc.read().await.id.clone();
                    crate::tee_println!(
                        "[resolve_sync] skip already-finalized pos: pos_id={pos_id_for_log}, \
                         market_id={market_id}, currency={currency}, interval={int_kind:?}, side={side:?} \
                         (PnL уже учтён WS-finalize'ом, payout пропускаем — иначе двойной счёт; \
                         запись `Resolution`/`AutoRedeem` в submit-CSV тоже не пишем — \
                         финальная строка трейда уже записана в `finalize_close_pnl_in_place`)"
                    );
                    continue;
                }
                if !redeemable {
                    let _ = vec.swap_remove(i);
                    continue;
                }
                {
                    let pos_arc = vec.swap_remove(i);
                    let pos = pos_arc.read().await.clone();
                    let pnl = if token_won {
                        pos.shares_held - pos.entry_cost
                    } else {
                        -pos.entry_cost
                    };
                    {
                        let mut pw = pos_arc.write().await;
                        if !pw.pnl_finalized {
                            pw.pnl_finalized = true;
                        }
                    }
                    *bankroll += pnl;
                    side_stats.pnl_usd += pnl;
                    side_stats.trades += 1;
                    if pnl >= 0.0 {
                        side_stats.wins += 1;
                    } else {
                        side_stats.losses += 1;
                    }
                    // Resolution не через close_position — дублируем в closed_trade_entries (replay калибровки).
                    side_stats
                        .closed_trade_entries
                        .push((pos.raw_pred_at_open, pnl > 0.0));
                    if token_won {
                        side_stats.resolution_win += 1;
                        side_stats.pnl_resolution_win += pnl;
                        if pnl >= 0.0 {
                            side_stats.resolution_win_profit += 1;
                        } else {
                            side_stats.resolution_win_loss += 1;
                        }
                    } else {
                        side_stats.resolution_loss += 1;
                        side_stats.pnl_resolution_loss += pnl;
                    }

                    {
                        let interval_str = interval_label(*int_kind);
                        let side_str = side_label(*side);
                        let exit_reason = if token_won {
                            "ResolutionWin"
                        } else {
                            "ResolutionLoss"
                        };
                        let open_unix_ms =
                            pos.event_end_ms.map(|e| e - pos.event_remaining_ms_at_open);
                        let close_unix_ms = pos.event_end_ms;
                        let graph_html_file_uri =
                            crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri(&pos)
                                .map(|p| {
                                    crate::xframe_graph_dump::graph_html_trade_file_uri(
                                        &p,
                                        open_unix_ms,
                                        close_unix_ms,
                                        Some(side_str),
                                    )
                                })
                                .unwrap_or_default();
                        crate::trade_csv_log::write_trade_csv_row(
                            crate::trade_csv_log::TradeCsvRow {
                                polymarket_url: &pos.polymarket_url,
                                price_to_beat: pos.price_to_beat,
                                final_price: final_price.or(pos.final_price),
                                currency: cur,
                                interval: interval_str,
                                side: side_str,
                                market_id,
                                asset_id: &pos.asset_id,
                                exit_reason,
                                buy_price: pos.buy_price,
                                raw_pred: pos.raw_pred_at_open,
                                cal_pred: pos.cal_pred_at_open,
                                kelly_f: pos.kelly_f_at_open,
                                entry_cost: pos.entry_cost,
                                shares_held: pos.shares_held,
                                exit_price: if token_won { 1.0 } else { 0.0 },
                                fee_usdc: 0.0,
                                pnl,
                                frames_held: pos.frames_held,
                                p_win_ema_at_close: pos.p_win_ema,
                                event_remaining_ms_at_open: pos.event_remaining_ms_at_open,
                                event_remaining_ms_at_close: 0,
                                open_unix_ms,
                                close_unix_ms,
                                graph_html_file_uri: graph_html_file_uri.as_str(),
                                pnl_top5_shap: pos.pnl_top5_shap_at_open.as_str(),
                            },
                        );
                        // Расширенный submit-CSV; без лога виртуально no-op.
                        crate::trade_csv_log::write_submit_trade_csv_row(
                            crate::trade_csv_log::SubmitTradeCsvRow {
                                pos_id: &pos.id,
                                polymarket_url: &pos.polymarket_url,
                                price_to_beat: pos.price_to_beat,
                                final_price: final_price.or(pos.final_price),
                                currency: cur,
                                interval: interval_str,
                                side: side_str,
                                market_id,
                                asset_id: &pos.asset_id,
                                exit_reason,
                                fill_role: "AutoRedeem",
                                finalized_via: "Resolution",
                                planned_buy_price: pos.planned_buy_price,
                                buy_price: pos.buy_price,
                                planned_shares_held: pos.planned_shares_held,
                                shares_held: pos.shares_held,
                                planned_entry_cost: pos.planned_entry_cost,
                                entry_cost: pos.entry_cost,
                                exit_price: if token_won { 1.0 } else { 0.0 },
                                fee_usdc: 0.0,
                                pnl,
                                open_order_id: pos.open_order_id.as_deref(),
                                tp_order_id: pos.tp_order_id.as_deref(),
                                close_order_id: None,
                                raw_pred: pos.raw_pred_at_open,
                                cal_pred: pos.cal_pred_at_open,
                                kelly_f: pos.kelly_f_at_open,
                                p_win_ema_at_close: pos.p_win_ema,
                                frames_held: pos.frames_held,
                                event_remaining_ms_at_open: pos.event_remaining_ms_at_open,
                                event_remaining_ms_at_close: 0,
                                open_unix_ms,
                                close_unix_ms,
                                graph_html_file_uri: graph_html_file_uri.as_str(),
                                pnl_top5_shap: pos.pnl_top5_shap_at_open.as_str(),
                            },
                        );
                    }
                    let _ = pos_arc;
                }
                // После swap_remove на `i` на этом индексе новая позиция — без i += 1.
            }
        }
        drop(pending_resolution);
        drop(bankroll);

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
pub(crate) const POLY_PRIVATE_KEY_ENV: &str = "POLY_PRIVATE_KEY";

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
pub(crate) async fn try_authenticate_clob_for_heartbeats(account: &SharedAccount) {
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
        match reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
        {
            Ok(http) => {
                if let Err(err) =
                    crate::poly_chain::ensure_deposit_wallet_deployed(&http, eoa).await
                {
                    crate::tee_eprintln!(
                        "[heartbeat] deposit wallet WALLET-CREATE провалился: {err:#}",
                    );
                }
            }
            Err(err) => {
                crate::tee_eprintln!(
                    "[heartbeat] HTTP-клиент для WALLET-CREATE не создан: {err:#}",
                );
            }
        }
    }
    // Отдельный ephemeral unauth клиент — `authenticate` в SDK требует unique inner.
    let unauth = clob::Client::new(account.clob.host().as_str(), clob::Config::default())
        .expect("failed to create Polymarket CLOB client for auth");

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
    #[ignore = "live network: требует POLY_PRIVATE_KEY; делает HTTP к clob-v2.polymarket.com/auth/api-key"]
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
