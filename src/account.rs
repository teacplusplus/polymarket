//! Единый счёт-капитал на процесс: банкролл, пик equity, max drawdown.
//! Счётчики сделок и per-side статистика — в [`crate::history_sim::SimStats`].
//!
//! Один [`SharedAccount`] (`Arc<Account>`) на все лейны и валюты.
//! [`Account`] держит каждое мутабельное поле под собственным
//! `Arc<RwLock<…>>`, поэтому потребители конкурируют только за тот лок,
//! который реально нужен (например, `bankroll` отдельно от `last_prob`).
//! Read-mostly auth (`clob_authed`, `clob_signer`) живёт в `ArcSwapAny<Arc<…>>`
//! — load без локов, swap атомарный.

use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{
    INITIAL_BANKROLL, SharedClosingPosition, SharedOpenPosition, SimStats,
};
use crate::real_sim::{RealSimState, interval_label, side_label};
use alloy::signers::Signer as _;
use alloy::signers::local::PrivateKeySigner;
use arc_swap::ArcSwapAny;
use indexmap::IndexSet;
use polymarket_client_sdk::POLYGON;
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::Uuid as ClobUuid;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::MissedTickBehavior;

/// Лимит [`Account::recently_resolved_markets`]; при переполнении вытесняется
/// самый старый элемент (`IndexSet::shift_remove_index(0)`).
pub const RECENTLY_RESOLVED_MARKETS_CAP: usize = 8;

/// Разделяемый счёт (`real_sim`, `ProjectManager`): один `Arc` на все воркеры.
/// Никакого внешнего `RwLock`: вся синхронизация — на уровне отдельных полей
/// [`Account`] (per-field `Arc<RwLock<…>>` / `ArcSwapAny<Arc<…>>`).
pub type SharedAccount = Arc<Account>;

/// Алиас под общий ключ `(currency, interval, side)`, по которому маршрутизируются
/// `positions` / `pending_resolution` / `closing` / `last_prob`.
pub type LaneKey = (String, XFrameIntervalKind, CurrencyUpDownOutcome);

/// Реализованный капитал (`bankroll`), пик equity (`peak_bankroll`) и
/// `max_drawdown_pct`. Пик и просадка считаются по MtM equity, не только по cash.
///
/// Каждое мутабельное поле живёт под собственным `Arc<RwLock<…>>`, чтобы
/// потребители брали лок ровно того, что им нужно. Порядок локов при
/// одновременном захвате нескольких — как объявлены ниже:
/// `bankroll → peak_bankroll → max_drawdown_pct → last_prob → positions →
/// pending_resolution → closing → recently_resolved_markets → individual_pos_lock`.
/// Соблюдаем во всех потребителях, иначе deadlock.
///
/// **Inner-локи на отдельные [`OpenPosition`] / [`ClosingPosition`]** (см.
/// [`crate::history_sim::SharedOpenPosition`] / [`crate::history_sim::SharedClosingPosition`])
/// лежат в самом конце канонического порядка: их разрешено брать **только
/// после** взятия HashMap-лока контейнера, и в одной операции — **не более
/// одного** inner-лока одновременно (иначе можно поймать deadlock на
/// одной и той же позиции, попавшей через `Account.positions` и
/// `ClosingPosition.position` одной и той же транзакцией).
///
/// Sync-режимы ([`crate::history_sim`] / [`crate::train_mode`]) пользуются
/// `try_read()` / `try_write()` с `.expect("uncontended")` — там Account живёт
/// в одном потоке без конкурентных тасков. Async-потребители (`real_sim`,
/// `account_ws`, `account_order`) — `read().await` / `write().await` как обычно.
#[derive(Debug)]
pub struct Account {
    pub bankroll: Arc<RwLock<f64>>,
    pub peak_bankroll: Arc<RwLock<f64>>,
    pub max_drawdown_pct: Arc<RwLock<f64>>,
    /// Последний известный implied prob по лейну; для MtM на лейнах без кадра на этом тике.
    /// Ключ с `currency`, чтобы PM разных валют не затирали друг друга.
    pub last_prob: Arc<RwLock<HashMap<LaneKey, f64>>>,
    /// Открытые позиции по лейну. Здесь же, а не `RealSimState`, чтобы Kelly видел entry_cost
    /// по всем валютам/лейнам. Пред-инициализация в `register_currency_lanes` для `get_mut().unwrap()`.
    ///
    /// Каждая позиция — `Arc<RwLock<OpenPosition>>` (см.
    /// [`SharedOpenPosition`]): тот же handle живёт в
    /// [`Self::pending_resolution`] и в [`ClosingPosition::position`], так
    /// что WS-fill пишет в одну запись, видимую отовсюду.
    pub positions: Arc<RwLock<HashMap<LaneKey, Vec<SharedOpenPosition>>>>,
    /// Позиции старого маркета после смены раунда в лейне; закрываются в [`Account::resolve_pending_market`],
    /// не через `manage_positions`.
    pub pending_resolution: Arc<RwLock<HashMap<LaneKey, Vec<SharedOpenPosition>>>>,
    /// Записи о закрытиях позиций (TP/SL/Timeout/EV) для матчинга
    /// real-time подтверждений через user-WS канал
    /// (`wss://ws-subscriptions-clob.polymarket.com/ws/user`,
    /// см. [`spawn_user_ws_listener`]). Создаётся / управляется в
    /// [`crate::history_sim::manage_positions`]; cleanup терминальных
    /// `Closed`/`CloseFailed` записей делается там же на следующем тике.
    ///
    /// Семантика статусов и lifecycle — см. [`ClosingPositionStatus`].
    /// В history_sim/real_sim сюда сразу идёт `Closed` (PnL уже учтён в
    /// `bankroll`), real-торговля будет создавать `PendingClose` и ждать
    /// колбека `apply_user_ws_event`.
    ///
    /// Каждая запись — `Arc<RwLock<ClosingPosition>>` (см.
    /// [`SharedClosingPosition`]); spawned-таски (`account_submit`,
    /// `account_ws`) держат тот же handle, статус апдейтится атомарно.
    pub closing: Arc<RwLock<HashMap<LaneKey, Vec<SharedClosingPosition>>>>,
    /// Уже резолвнутые `condition_id` (см. [`RECENTLY_RESOLVED_MARKETS_CAP`]): не открывать сделку
    /// на маркет после резолюции при гонке HTTP/tick и колбека.
    pub recently_resolved_markets: Arc<RwLock<IndexSet<String>>>,
    /// Unauthenticated CLOB-клиент Polymarket — **единственное место**, где
    /// он создаётся. Все потребители (`ProjectManager.clob`,
    /// [`try_authenticate_clob_for_heartbeats`]) забирают клон отсюда.
    /// Это даёт один общий пул соединений / DNS-кэш / cookie store на
    /// процесс и гарантирует, что все компоненты разговаривают с тем же
    /// CLOB-эндпоинтом.
    ///
    /// Хранится в `Arc`, чтобы синхронные потребители (например,
    /// [`crate::project_manager::ProjectManager::new`]) могли держать
    /// собственный клон без `await` под локом — на hot-path
    /// `pm.clob.order_books(&requests)` идёт без локов вообще.
    /// Внутри SDK `clob::Client` сам по себе обёртка над
    /// `Arc<ClientInner>`, так что внешний `Arc<Arc<…>>` — это два
    /// инкремента счётчика и нулевая аллокация.
    pub clob: Arc<clob::Client>,
    /// Аутентифицированный CLOB-клиент Polymarket — single source of truth
    /// на процесс. Используется для `POST /v1/heartbeats` (см.
    /// [`spawn_heartbeat`]) и для постановки/отмены ордеров через
    /// [`crate::account_order`]. `Arc<None>` если авторизацию не выполняли
    /// (иначе в RealSim см. вызов [`try_authenticate_clob_for_heartbeats`]
    /// в `main` до [`spawn_heartbeat`]) или она упала.
    ///
    /// Read-mostly: hot-path читателей (`spawn_heartbeat`, `account_ws`,
    /// `account_order`, `ProjectManager::clob_authed`) делают `load()` без
    /// локов; запись только из [`try_authenticate_clob_for_heartbeats`].
    pub clob_authed: ArcSwapAny<Arc<Option<clob::Client<Authenticated<Normal>>>>>,
    /// EOA-подписант (`POLY_PRIVATE_KEY` + `POLYGON` chain_id), кэшируется
    /// рядом с [`Self::clob_authed`] чтобы [`crate::account_order::post_order_on_clob`] не лез
    /// в `std::env` на каждый ордер. Заполняется в
    /// [`try_authenticate_clob_for_heartbeats`] одновременно с
    /// `clob_authed`; `Arc<None>` ↔ auth не поднимался / упал.
    /// `PrivateKeySigner: Clone`, hot-path `load()` без локов.
    pub clob_signer: ArcSwapAny<Arc<Option<PrivateKeySigner>>>,
}

impl Account {
    pub fn new() -> Self {
        let clob = Arc::new(
            clob::Client::new("https://clob.polymarket.com", clob::Config::default())
                .expect("failed to create Polymarket CLOB client"),
        );
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
        }
    }

    /// Пустые `positions` / `pending_resolution` / `closing` для всех лейнов валюты
    /// ([`crate::real_sim::run_real_sim`]). `or_default()` идемпотентен при повторном вызове.
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

    /// Закрывает pending по `market_id` бинарной выплатой CTF (как `CloseReason::Resolution` в `close_position`).
    /// Победа токена: `pnl = shares_held - entry_cost`, иначе `pnl = -entry_cost`; комиссии на redeem нет.
    ///
    /// **Параметры:** `account`, `state` — счёт и `RealSimState` этой валюты; `currency` / `interval` —
    /// фильтр лейнов; `market_id` — `condition_id`; `up_won` — см. [`crate::xframe_dump::MarketXFramesDump::up_won`];
    /// `final_price` — фактическая цена закрытия окна, прокидывается в CSV-колонку `final_price`
    /// resolution-строки (на момент входа в позицию неизвестна, появляется только в callback'е
    /// [`crate::xframe_dump::spawn_dump_market_xframes_binary`]).
    ///
    /// **Lock order:** сначала `state.write()` (RealSimState — внешний `RwLock`),
    /// затем поля `Account` в порядке объявления (`bankroll → positions →
    /// pending_resolution → recently_resolved_markets`). Соблюдаем во всех
    /// потребителях, чтобы избежать deadlock'а.
    ///
    /// Drawdown здесь не обновляют — следующий `tick_once` вызовет `update_drawdown`.
    pub async fn resolve_pending_market(
        account: &SharedAccount,
        state: &Arc<RwLock<RealSimState>>,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
        final_price: f64,
    ) {
        let mut state_guard = state.write().await;

        // Собираем live TP-order_id'ы для отмены ниже (см. doc у
        // `crate::account_submit::spawn_cancel_tp_orders_after_resolution`):
        // auto-redeem забирает шеры, но висящие maker TP-лимитки CLOB
        // оставит без явного DELETE, и они будут продолжать висеть до
        // протухания. В виртуальных режимах `tp_order_id` всегда `None` —
        // вектор останется пустым и спавна не будет.
        // `Vec<SharedOpenPosition>` — позиции с активным `tp_order_id`
        // (его и `id` заберёт сама `spawn_cancel_tp_orders_after_resolution`
        // под write-lock'ом, см. doc там).
        let mut positions_with_tp: Vec<crate::history_sim::SharedOpenPosition> = Vec::new();

        // Колбек резолюции может опередить смену `frame`: переносим совпадающие `market_id`
        // из positions в pending. Берём оба лока в порядке объявления полей,
        // плюс per-position write для чтения `market_id` и снятия `tp_order_id`.
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
                    let matches_market = {
                        let pos_g = pos_vec[idx].read().await;
                        pos_g.market_id == market_id
                    };
                    if matches_market {
                        let pos_arc = pos_vec.swap_remove(idx);
                        // Если у позиции есть активный TP — тащим её Arc в
                        // `positions_with_tp`. `take()` для `tp_order_id`
                        // делает `spawn_cancel_tp_orders_after_resolution`
                        // сама (под собственным write-lock'ом) — там же
                        // читает `pos_id`. Здесь просто отмечаем кандидатов;
                        // флаг `is_some` под коротким read-локом достаточен
                        // (под этим же лок-фреймом доступа к pos_arc больше
                        // ни у кого нет — мы её уже `swap_remove`-нули).
                        let has_tp = pos_arc.read().await.tp_order_id.is_some();
                        if has_tp {
                            positions_with_tp.push(pos_arc.clone());
                        }
                        pending_vec.push(pos_arc);
                    } else {
                        idx += 1;
                    }
                }
            }
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

    /// Ядро резолюции: берёт `bankroll` / `pending_resolution` /
    /// `recently_resolved_markets` под write-локами и проводит payout.
    /// Пишет строки в [`crate::trade_csv_log`] и вызывает
    /// [`crate::trade_csv_log::record_market_outcome`].
    ///
    /// `final_price_override` — фактическая цена закрытия окна, попадает в CSV-колонку
    /// `final_price` resolution-строк. `None` — берём `pos.final_price` (исторический режим:
    /// dump уже содержит финальную цену, она проставлена в `OpenPosition.final_price` на входе);
    /// `Some(_)` — переопределяем (real-time режим: на входе финал ещё неизвестен,
    /// прилетает позже из callback'а).
    pub async fn resolve_pending_market_sync(
        &self,
        sim_stats: &mut SimStats,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
        final_price: Option<f64>,
    ) {
        // Lock order: bankroll → pending_resolution → recently_resolved_markets.
        
        let mut recently_resolved = self.recently_resolved_markets.write().await;
        // До PnL: помечаем маркет резолвнутым (гонка HTTP vs колбек; FIFO cap — см. константу).
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
                let matches_market = {
                    let g = vec[i].read().await;
                    g.market_id == market_id
                };
                if matches_market {
                    let pos_arc = vec.swap_remove(i);
                    // Снимаем полный snapshot позиции под одним read-локом —
                    // сразу клонируем нужные поля, чтобы не держать pos-lock
                    // через дальнейший stats / CSV-write.
                    let pos = pos_arc.read().await.clone();
                    let pnl = if token_won {
                        pos.shares_held - pos.entry_cost
                    } else {
                        -pos.entry_cost
                    };
                    *bankroll += pnl;
                    side_stats.pnl_usd += pnl;
                    side_stats.trades += 1;
                    if pnl >= 0.0 {
                        side_stats.wins += 1;
                    } else {
                        side_stats.losses += 1;
                    }
                    // См. doc у `SideStats::closed_trade_entries` в history_sim.rs:
                    // resolution-закрытия идут не через `close_position`, поэтому
                    // дублируем сюда — иначе sim-replay калибровка теряет хвост
                    // позиций, доехавших до резолюции (Res✓/Res✗).
                    side_stats.closed_trade_entries.push((pos.raw_pred_at_open, pnl > 0.0));
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
                        let open_unix_ms = pos.event_end_ms.map(|e| e - pos.event_remaining_ms_at_open);
                        let close_unix_ms = pos.event_end_ms;
                        let graph_html_file_uri = crate::xframe_graph_dump::graph_dump_bin_path_for_trade_csv_uri(&pos)
                            .map(|p| crate::xframe_graph_dump::graph_html_trade_file_uri(&p, open_unix_ms, close_unix_ms, Some(side_str)))
                            .unwrap_or_default();
                        crate::trade_csv_log::write_trade_csv_row(crate::trade_csv_log::TradeCsvRow {
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
                        });
                    }
                    let _ = pos_arc;
                } else {
                    i += 1;
                }
            }
        }
        drop(pending_resolution);
        drop(bankroll);
       

        crate::trade_csv_log::record_market_outcome(market_id, up_won);
    }

    /// `Arc::new(Account::new())` — удобство для `main`/PM.
    pub fn new_shared() -> SharedAccount {
        Arc::new(Self::new())
    }

    /// Пик equity и max DD по переданной MtM equity (вызыватель считает equity на каждом тике).
    /// Async: берёт `peak_bankroll` и `max_drawdown_pct` под write-локами в каноническом порядке.
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

/// Период `POST /v1/heartbeats` к Polymarket CLOB. По
/// [официальной доке](https://docs.polymarket.com/developers/CLOB/orders/orders#heartbeat)
/// окно отмены — 10 секунд (`+` ~5 секунд буфер); SDK-пример рекомендует
/// слать раз в 5 секунд. Без heartbeat'а **все открытые ордера**
/// аутентифицированной сессии будут автоматически отменены.
const CLOB_HEARTBEAT_INTERVAL_SEC: u64 = 5;

/// Имя env-переменной с EOA-приватником для CLOB-аутентификации:
/// та же `POLY_PRIVATE_KEY` (см. `.env`), что используется в
/// [`crate::poly_chain`] для on-chain split. Одна EOA на процесс —
/// одни creds для heartbeat и для подписи `splitPosition`.
///
/// Если переменная пустая/отсутствует —
/// [`try_authenticate_clob_for_heartbeats`] выходит молча, оставляя
/// `Account.clob_authed = Arc::new(None)`; [`spawn_heartbeat`] всё равно
/// поднимается, но clob-тик становится no-op'ом.
pub(crate) const POLY_PRIVATE_KEY_ENV: &str = "POLY_PRIVATE_KEY";

/// Глобальный CLOB heartbeat-таск на процесс (один на все валюты): раз в
/// [`CLOB_HEARTBEAT_INTERVAL_SEC`] секунд шлёт `POST /v1/heartbeats`,
/// удерживая открытые ордера аутентифицированной сессии (без heartbeat'а
/// сервер их автоматически отменит; окно 10s + ~5s буфер, см.
/// [docs](https://docs.polymarket.com/developers/CLOB/orders/orders#heartbeat)).
///
/// Старт — один раз в `main`: после [`Account::new_shared`] вызывается
/// [`try_authenticate_clob_for_heartbeats`], затем [`spawn_heartbeat`]
/// (общий ресурс на процесс, не привязан к валюте/`ProjectManager`/
/// `real_sim`-state'у). Если в окружении нет [`POLY_PRIVATE_KEY_ENV`] или
/// [`try_authenticate_clob_for_heartbeats`] упал — таск всё равно крутится,
/// но ничего не шлёт (clob-тик no-op). Это безопасно, поскольку без
/// аутентификации **открытых ордеров не может быть** в принципе.
///
/// Per-currency `print_sim_stats` snapshot вынесен в отдельную таску
/// [`crate::real_sim::spawn_stats_snapshot`] (он зависит от
/// [`crate::real_sim::RealSimState`], который per-currency).
///
/// `MissedTickBehavior::Delay` — если CLOB-вызов задержался (например,
/// 502 + retry tcp), не догоняем burst'ом 10 heartbeat'ов сразу: следующий
/// тик отсчитывается от момента возврата управления.
pub fn spawn_heartbeat(account: SharedAccount) {
    tokio::spawn(async move {
        // Аутентификация выполняется в `main` до [`spawn_heartbeat`]:
        // [`try_authenticate_clob_for_heartbeats`]. Здесь только снимаем
        // снимок `clob_authed` через ArcSwap.load() — без локов; внутри
        // `Arc<Option<…>>`, поэтому `.as_ref()` даёт `Option<&Client>`.
        let auth_client: Option<clob::Client<Authenticated<Normal>>> =
            (**account.clob_authed.load()).clone();

        let mut clob_tick = tokio::time::interval(Duration::from_secs(CLOB_HEARTBEAT_INTERVAL_SEC));
        clob_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // Первый tick срабатывает мгновенно — пропускаем: на старте сразу
        // постить heartbeat нет смысла (open orders ещё не существуют).
        clob_tick.tick().await;

        // Chain `heartbeat_id`: первый POST с `None`, далее берём UUID из
        // ответа. На любой ошибке сбрасываем в `None`, следующий запрос
        // уйдёт «как первый» (Polymarket-протокол: при невалидном id
        // сервер возвращает 400 + правильный id, но SDK эту деталь не
        // прокидывает — `None` после ошибки = безопасный фолбэк).
        let mut heartbeat_id: Option<ClobUuid> = None;

        // Антишум для лога: подряд успешные heartbeat'ы не печатаем,
        // подряд ошибки — тоже. Печатаем только на смене состояния
        // (success ↔ error) и на самой первой попытке, чтобы один раз
        // подтвердить «heartbeat жив». См. формат сообщений ниже.
        let mut last_was_success = false;
        let mut had_first_log = false;

        loop {
            clob_tick.tick().await;
            let Some(client) = auth_client.as_ref() else {
                // Auth выключен (нет POLY_PRIVATE_KEY или authenticate()
                // упал) — таск крутится no-op'ом, ждать тут нечего, но
                // rolling-проверки на появление auth не делаем:
                // [`try_authenticate_clob_for_heartbeats`] уже отработала в main.
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
                            "[heartbeat] CLOB heartbeat восстановлен (heartbeat_id={})",
                            resp.heartbeat_id,
                        );
                    }
                    last_was_success = true;
                }
                Err(err) => {
                    if last_was_success || !had_first_log {
                        crate::tee_eprintln!(
                            "[heartbeat] CLOB heartbeat ошибка: {err:#} \
                             (открытые ордера могут быть отменены при тишине > 10s)",
                        );
                        had_first_log = true;
                    }
                    last_was_success = false;
                    heartbeat_id = None;
                }
            }
        }
    });
}

/// Поднимает аутентифицированный CLOB-клиент по `POLY_PRIVATE_KEY` (та же
/// EOA, что в [`crate::poly_chain`]) и сохраняет его в
/// [`Account::clob_authed`] — single source of truth на процесс. В RealSim
/// вызывается из `main` до [`spawn_heartbeat`]. Все читатели ([`spawn_heartbeat`] и
/// [`crate::project_manager::ProjectManager::clob_authed`]) берут клиент
/// именно оттуда.
///
/// Идемпотентна: если в Account уже лежит `Arc::new(Some(_))`, функция выходит
/// без сетевого вызова. На любой ошибке (нет env-ключа, парсинг
/// провалился, `clob::Client::new` упал, `authenticate()` вернул
/// ошибку) — лог + Account остаётся с `Arc::new(None)`; clob-тик в
/// [`spawn_heartbeat`] в этом случае молча no-op.
///
/// Аутентификация = `create_or_derive_api_key` (внутри `authenticate()`):
/// EIP-712 ClobAuth на `chain_id=POLYGON`, signature_type=Eoa (дефолт).
/// Этого хватает для heartbeat'а; реальная постановка ордеров через
/// Polymarket Safe потребует отдельного клиента с
/// `signature_type=GnosisSafe + funder=<safe>` (тот же EOA сможет
/// переиспользовать API-ключ).
pub(crate) async fn try_authenticate_clob_for_heartbeats(account: &SharedAccount) {
    if account.clob_authed.load().is_some() {
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
    let address = signer.address();
    // Берём общий unauth-клиент из [`Account::clob`] (создаётся ровно один
    // раз в [`Account::new`]) и клонируем его — `clob::Client` это
    // обёртка над `Arc<ClientInner>`, `clone()` это инкремент счётчика.
    // `authentication_builder` потребляет `self`, поэтому клон обязателен:
    // нельзя «вынуть» клиент из `Arc<clob::Client>` в Account, не
    // ломая остальных потребителей (`ProjectManager.clob` и т.п.).
    let unauth: clob::Client = (*account.clob).clone();
    match unauth.authentication_builder(&signer).authenticate().await {
        Ok(authed) => {
            // signer кэшируем рядом с authed-клиентом — нужен для
            // [`post_order_on_clob`] (`auth_client.sign(&signer, …)`),
            // чтобы не парсить `POLY_PRIVATE_KEY` на каждый ордер.
            // ArcSwap.store даёт атомарную замену без локов.
            account.clob_authed.store(Arc::new(Some(authed)));
            account.clob_signer.store(Arc::new(Some(signer)));
            crate::tee_println!(
                "[heartbeat] CLOB authenticate OK (eoa={address:#x}); heartbeat каждые {CLOB_HEARTBEAT_INTERVAL_SEC}s",
            );
        }
        Err(err) => {
            crate::tee_eprintln!(
                "[heartbeat] CLOB authenticate провалился: {err:#}; CLOB heartbeat отключён",
            );
        }
    }
}

// Постановка ордеров на CLOB (`POST /order`) вынесена в модуль
// [`crate::account_order`]: публичный API — `post_order_on_clob` плюс
// типы `PostOrderRequest`/`PostOrderResult`/`OrderRole`/`OrderAmount`.
// Сюда модуль резолвит зависимости через `Account.clob_authed` и
// `Account.clob_signer` (заполняются [`try_authenticate_clob_for_heartbeats`]),
// `account.rs` сам не импортирует CLOB-ордерные типы, чтобы не тащить их
// в heartbeat-петлю.

#[cfg(test)]
mod tests {
    use super::*;

    /// Live integration-тест полного цикла CLOB-аутентификации.
    /// Выполняет полный цикл как в RealSim перед [`spawn_heartbeat`]:
    /// 1. Читает `POLY_PRIVATE_KEY` через [`dotenvy`].
    /// 2. Вызывает [`try_authenticate_clob_for_heartbeats`] на свежем
    ///    [`Account`] (где `clob_authed = Arc::new(None)`).
    /// 3. Проверяет, что после запроса `Account.clob_authed.load()` отдаёт
    ///    `Some(_)` (т.е. real `POST clob.polymarket.com/auth/api-key/...`
    ///    отдал creds, и SDK построил `Authenticated<Normal>`-клиента).
    /// 4. Проверяет идемпотентность: повторный вызов — no-op
    ///    (короткое замыкание на `clob_authed.load().is_some()`),
    ///    HTTP-запроса не делает.
    ///
    /// **Делает реальный HTTP-запрос к Polymarket CLOB.** Помечен
    /// `#[ignore]`, чтобы не запускался в обычном `cargo test`. Запуск:
    ///
    /// ```bash
    /// POLY_PRIVATE_KEY=0x… \
    ///     cargo test --bin poly account::tests::live_try_authenticate_clob_for_heartbeats -- --ignored --nocapture
    /// ```
    ///
    /// Без `POLY_PRIVATE_KEY` тест пропускается (`Ok(())`), чтобы не
    /// падать в CI.
    #[tokio::test]
    #[ignore = "live network: требует POLY_PRIVATE_KEY; делает HTTP-запрос к clob.polymarket.com/auth/api-key"]
    async fn live_try_authenticate_clob_for_heartbeats() -> anyhow::Result<()> {
        let _ = dotenvy::dotenv();

        // rustls 0.23 требует установленного CryptoProvider'а до первого
        // TLS-запроса. В обычном бинарнике это делает `main`, но
        // `tokio::test` поднимает свой рантайм и `main` не вызывается.
        // `install_default()` идемпотентен: вторая попытка вернёт Err,
        // которую мы намеренно игнорируем.
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

        // 1) Свежий Account → clob_authed=Arc::new(None), инвариант до запроса.
        anyhow::ensure!(
            account.clob_authed.load().is_none(),
            "новый Account должен идти с clob_authed=Arc::new(None)",
        );

        // 2) Полный auth-цикл: signer → unauth client → authentication_builder
        //    → POST /auth/api-key → Account.clob_authed.store(Arc::new(Some(_))).
        try_authenticate_clob_for_heartbeats(&account).await;
        anyhow::ensure!(
            account.clob_authed.load().is_some(),
            "после try_authenticate_clob_for_heartbeats clob_authed обязан быть Some \
             (если упало — смотри stderr-логи `[heartbeat] CLOB authenticate провалился: …`)",
        );

        // 3) Идемпотентность: второй вызов должен короткозамкнуться на
        //    `clob_authed.load().is_some()` и НЕ делать повторный HTTP-запрос.
        //    Сравниваем Arc по указателю — после no-op у нас тот же самый
        //    Arc, не пересозданный.
        let before = account.clob_authed.load_full();
        try_authenticate_clob_for_heartbeats(&account).await;
        let after = account.clob_authed.load_full();
        anyhow::ensure!(
            Arc::ptr_eq(&before, &after),
            "идемпотентность нарушена: clob_authed Arc был пересоздан повторным auth-вызовом",
        );

        // 4) Реальный heartbeat-RPC через тот же клиент, что лежит в
        //    `Account.clob_authed`. Это то, чем занят
        //    [`spawn_heartbeat`] в hot-path: подтверждаем, что
        //    `Authenticated<Normal>`-сессия валидна для
        //    `POST /v1/heartbeats` (а не «технически собралась, но не
        //    принимается сервером»). Первый вызов идёт с `None`
        //    (server-side создаст новый id), второй — с возвращённым
        //    UUID (chained heartbeat — тот же путь, что в hot-path-цикле).
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
