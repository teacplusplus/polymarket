//! Единый счёт-капитал на процесс: банкролл, пик equity, max drawdown.
//! Счётчики сделок и per-side статистика — в [`crate::history_sim::SimStats`].
//!
//! Один [`SharedAccount`] (`Arc<RwLock<Account>>`) на все лейны и валюты,
//! в отличие от старой схемы с отдельным «счётом» на каждый интервал.

use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{ClosingPosition, INITIAL_BANKROLL, OpenPosition, SimStats};
use crate::real_sim::{interval_label, side_label, RealSimState};
use alloy::signers::Signer as _;
use alloy::signers::local::PrivateKeySigner;
use indexmap::IndexSet;
use polymarket_client_sdk::auth::Normal;
use polymarket_client_sdk::auth::Uuid as ClobUuid;
use polymarket_client_sdk::auth::state::Authenticated;
use polymarket_client_sdk::clob;
use polymarket_client_sdk::POLYGON;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::MissedTickBehavior;

/// Лимит [`Account::recently_resolved_markets`]; при переполнении вытесняется
/// самый старый элемент (`IndexSet::shift_remove_index(0)`).
pub const RECENTLY_RESOLVED_MARKETS_CAP: usize = 8;

/// Разделяемый счёт (`real_sim`, `ProjectManager`): один `Arc` на все воркеры.
pub type SharedAccount = Arc<RwLock<Account>>;

/// Реализованный капитал (`bankroll`), пик equity (`peak_bankroll`) и
/// `max_drawdown_pct`. Пик и просадка считаются по MtM equity, не только по cash.
#[derive(Debug)]
pub struct Account {
    pub bankroll: f64,
    pub peak_bankroll: f64,
    pub max_drawdown_pct: f64,
    /// Последний известный implied prob по лейну; для MtM на лейнах без кадра на этом тике.
    /// Ключ с `currency`, чтобы PM разных валют не затирали друг друга.
    pub last_prob: HashMap<(String, XFrameIntervalKind, CurrencyUpDownOutcome), f64>,
    /// Открытые позиции по лейну. Здесь же, а не `RealSimState`, чтобы Kelly видел entry_cost
    /// по всем валютам/лейнам. Пред-инициализация в `register_currency_lanes` для `get_mut().unwrap()`.
    pub positions: HashMap<(String, XFrameIntervalKind, CurrencyUpDownOutcome), Vec<OpenPosition>>,
    /// Позиции старого маркета после смены раунда в лейне; закрываются в [`Account::resolve_pending_market`],
    /// не через `manage_positions`.
    pub pending_resolution: HashMap<(String, XFrameIntervalKind, CurrencyUpDownOutcome), Vec<OpenPosition>>,
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
    pub closing: HashMap<(String, XFrameIntervalKind, CurrencyUpDownOutcome), Vec<ClosingPosition>>,
    /// Уже резолвнутые `condition_id` (см. [`RECENTLY_RESOLVED_MARKETS_CAP`]): не открывать сделку
    /// на маркет после резолюции при гонке HTTP/tick и колбека.
    pub recently_resolved_markets: IndexSet<String>,
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
    /// [`spawn_heartbeat`]) и в перспективе для `post_order` /
    /// `cancel_order`. `None` если авторизацию не выполняли (иначе в
    /// RealSim см. вызов [`try_authenticate_clob_for_heartbeats`] в `main` до
    /// [`spawn_heartbeat`]) или она упала; в `history_sim`/`train` и т.п.
    /// остаётся `None`.
    ///
    /// `clob::Client` — небольшой обёрткой над `Arc<ClientInner<...>>`,
    /// `Clone` дешёвый, поэтому хранится по значению (внешний `Arc` не
    /// нужен). Доступ из любого места — через
    /// [`crate::project_manager::ProjectManager::clob_authed`].
    pub clob_authed: Option<clob::Client<Authenticated<Normal>>>,
    /// EOA-подписант (`POLY_PRIVATE_KEY` + `POLYGON` chain_id), кэшируется
    /// рядом с [`Self::clob_authed`] чтобы [`crate::account_order::post_order_on_clob`] не лез
    /// в `std::env` на каждый ордер. Заполняется в
    /// [`try_authenticate_clob_for_heartbeats`] одновременно с
    /// `clob_authed`; `None` ↔ auth не поднимался / упал.
    /// `PrivateKeySigner: Clone`, поэтому держим по значению.
    pub clob_signer: Option<PrivateKeySigner>,
}

impl Account {
    pub fn new() -> Self {
        let clob = Arc::new(
            clob::Client::new("https://clob.polymarket.com", clob::Config::default())
                .expect("failed to create Polymarket CLOB client"),
        );
        Self {
            bankroll: INITIAL_BANKROLL,
            peak_bankroll: INITIAL_BANKROLL,
            max_drawdown_pct: 0.0,
            last_prob: HashMap::new(),
            positions: HashMap::new(),
            pending_resolution: HashMap::new(),
            closing: HashMap::new(),
            recently_resolved_markets: IndexSet::new(),
            clob,
            clob_authed: None,
            clob_signer: None,
        }
    }

    /// Пустые `positions` / `pending_resolution` / `closing` для всех лейнов валюты
    /// ([`crate::real_sim::run_real_sim`]). `or_default()` идемпотентен при повторном вызове.
    pub fn register_currency_lanes(
        &mut self,
        currency: &str,
        lanes: &[(XFrameIntervalKind, CurrencyUpDownOutcome)],
    ) {
        for (interval, side) in lanes {
            let key = (currency.to_string(), *interval, *side);
            self.positions.entry(key.clone()).or_default();
            self.pending_resolution.entry(key.clone()).or_default();
            self.closing.entry(key).or_default();
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
    /// **Lock order:** `state.write()` → `account.write()`, как в `tick_once`.
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
        let mut account_guard = account.write().await;

        // Колбек резолюции может опередить смену `frame`: переносим совпадающие `market_id` из positions в pending.
        {
            let Account {
                positions,
                pending_resolution,
                ..
            } = &mut *account_guard;
            for ((cur, int_kind, side), pos_vec) in positions.iter_mut() {
                if cur.as_str() != currency || *int_kind != interval {
                    continue;
                }
                let key = (cur.clone(), *int_kind, *side);
                let pending_vec = pending_resolution.entry(key).or_default();
                let mut idx = 0;
                while idx < pos_vec.len() {
                    if pos_vec[idx].market_id == market_id {
                        pending_vec.push(pos_vec.swap_remove(idx));
                    } else {
                        idx += 1;
                    }
                }
            }
        }

        let sim_stats = state_guard
            .stats
            .get_mut(&interval)
            .expect("RealSimState.stats: оба интервала пред-инициализированы в new()");
        account_guard.resolve_pending_market_sync(
            sim_stats,
            currency,
            interval,
            market_id,
            up_won,
            Some(final_price),
        );
    }

    /// Ядро резолюции без локов: из `history_sim` с `&mut Account` или после локов из [`Account::resolve_pending_market`].
    /// Пишет строки в [`crate::trade_csv_log`] и вызывает [`crate::trade_csv_log::record_market_outcome`].
    ///
    /// `final_price_override` — фактическая цена закрытия окна, попадает в CSV-колонку
    /// `final_price` resolution-строк. `None` — берём `pos.final_price` (исторический режим:
    /// dump уже содержит финальную цену, она проставлена в `OpenPosition.final_price` на входе);
    /// `Some(_)` — переопределяем (real-time режим: на входе финал ещё неизвестен,
    /// прилетает позже из callback'а).
    pub fn resolve_pending_market_sync(
        &mut self,
        sim_stats: &mut SimStats,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
        final_price: Option<f64>,
    ) {
        // До PnL: помечаем маркет резолвнутым (гонка HTTP vs колбек; FIFO cap — см. константу).
        if self
            .recently_resolved_markets
            .insert(market_id.to_string())
        {
            while self.recently_resolved_markets.len() > RECENTLY_RESOLVED_MARKETS_CAP {
                self.recently_resolved_markets.shift_remove_index(0);
            }
        }

        let Account {
            bankroll,
            pending_resolution,
            ..
        } = self;

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
                if vec[i].market_id == market_id {
                    let pos = vec.swap_remove(i);
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
                } else {
                    i += 1;
                }
            }
        }

        crate::trade_csv_log::record_market_outcome(market_id, up_won);
    }

    /// `Arc::new(RwLock::new(Account::new()))` — удобство для `main`/PM.
    pub fn new_shared() -> SharedAccount {
        Arc::new(RwLock::new(Self::new()))
    }

    /// Пик equity и max DD по переданной MtM equity (вызыватель считает equity на каждом тике).
    pub fn update_drawdown(&mut self, equity: f64) {
        if equity > self.peak_bankroll {
            self.peak_bankroll = equity;
        }
        if self.peak_bankroll > 0.0 {
            let drawdown_pct = (self.peak_bankroll - equity) / self.peak_bankroll * 100.0;
            if drawdown_pct > self.max_drawdown_pct {
                self.max_drawdown_pct = drawdown_pct;
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
/// `Account.clob_authed = None`; [`spawn_heartbeat`] всё равно
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
        // снимок `clob_authed` на время жизни таска (`None` если auth не было
        // или упал до спавна).
        let auth_client: Option<clob::Client<Authenticated<Normal>>> =
            account.read().await.clob_authed.clone();

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
/// Идемпотентна: если в Account уже лежит `Some(...)`, функция выходит
/// без сетевого вызова. На любой ошибке (нет env-ключа, парсинг
/// провалился, `clob::Client::new` упал, `authenticate()` вернул
/// ошибку) — лог + Account остаётся с `None`; clob-тик в
/// [`spawn_heartbeat`] в этом случае молча no-op.
///
/// Аутентификация = `create_or_derive_api_key` (внутри `authenticate()`):
/// EIP-712 ClobAuth на `chain_id=POLYGON`, signature_type=Eoa (дефолт).
/// Этого хватает для heartbeat'а; реальная постановка ордеров через
/// Polymarket Safe потребует отдельного клиента с
/// `signature_type=GnosisSafe + funder=<safe>` (тот же EOA сможет
/// переиспользовать API-ключ).
pub(crate) async fn try_authenticate_clob_for_heartbeats(account: &SharedAccount) {
    if account.read().await.clob_authed.is_some() {
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
    let unauth: clob::Client = (*account.read().await.clob).clone();
    match unauth.authentication_builder(&signer).authenticate().await {
        Ok(authed) => {
            // signer кэшируем рядом с authed-клиентом — нужен для
            // [`post_order_on_clob`] (`auth_client.sign(&signer, …)`),
            // чтобы не парсить `POLY_PRIVATE_KEY` на каждый ордер.
            let mut guard = account.write().await;
            guard.clob_authed = Some(authed);
            guard.clob_signer = Some(signer);
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
    ///    [`Account`] (где `clob_authed = None`).
    /// 3. Проверяет, что после запроса `Account.clob_authed = Some(_)`
    ///    (т.е. real `POST clob.polymarket.com/auth/api-key/...` отдал
    ///    creds, и SDK построил `Authenticated<Normal>`-клиента).
    /// 4. Проверяет идемпотентность: повторный вызов — no-op
    ///    (короткое замыкание на `clob_authed.is_some()`), HTTP-запроса
    ///    не делает.
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

        // 1) Свежий Account → clob_authed=None, инвариант до запроса.
        anyhow::ensure!(
            account.read().await.clob_authed.is_none(),
            "новый Account должен идти с clob_authed=None",
        );

        // 2) Полный auth-цикл: signer → unauth client → authentication_builder
        //    → POST /auth/api-key → Account.clob_authed = Some(_).
        try_authenticate_clob_for_heartbeats(&account).await;
        anyhow::ensure!(
            account.read().await.clob_authed.is_some(),
            "после try_authenticate_clob_for_heartbeats clob_authed обязан быть Some \
             (если упало — смотри stderr-логи `[heartbeat] CLOB authenticate провалился: …`)",
        );

        // 3) Идемпотентность: второй вызов должен короткозамкнуться на
        //    `clob_authed.is_some()` и НЕ делать повторный HTTP-запрос.
        //    Снимаем Arc<ClientInner> и сравниваем по указателю — после
        //    no-op у нас тот же самый клиент, не пересозданный.
        let before = account.read().await.clob_authed.clone();
        try_authenticate_clob_for_heartbeats(&account).await;
        let after = account.read().await.clob_authed.clone();
        anyhow::ensure!(
            before.is_some() && after.is_some(),
            "оба вызова должны держать clob_authed = Some",
        );

        // 4) Реальный heartbeat-RPC через тот же клиент, что лежит в
        //    `Account.clob_authed`. Это то, чем занят
        //    [`spawn_heartbeat`] в hot-path: подтверждаем, что
        //    `Authenticated<Normal>`-сессия валидна для
        //    `POST /v1/heartbeats` (а не «технически собралась, но не
        //    принимается сервером»). Первый вызов идёт с `None`
        //    (server-side создаст новый id), второй — с возвращённым
        //    UUID (chained heartbeat — тот же путь, что в hot-path-цикле).
        let client = account
            .read()
            .await
            .clob_authed
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
