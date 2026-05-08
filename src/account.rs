//! Единый счёт-капитал на процесс: банкролл, пик equity, max drawdown.
//! Счётчики сделок и per-side статистика — в [`crate::history_sim::SimStats`].
//!
//! Один [`SharedAccount`] (`Arc<RwLock<Account>>`) на все лейны и валюты,
//! в отличие от старой схемы с отдельным «счётом» на каждый интервал.

use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{INITIAL_BANKROLL, OpenPosition, SimStats};
use crate::real_sim::{interval_label, side_label, RealSimState};
use indexmap::IndexSet;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

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
    /// Уже резолвнутые `condition_id` (см. [`RECENTLY_RESOLVED_MARKETS_CAP`]): не открывать сделку
    /// на маркет после резолюции при гонке HTTP/tick и колбека.
    pub recently_resolved_markets: IndexSet<String>,
}

impl Account {
    /// [`INITIAL_BANKROLL`] для `bankroll` и `peak_bankroll` (избегаем ложного 100% DD на старте).
    pub fn new() -> Self {
        Self {
            bankroll: INITIAL_BANKROLL,
            peak_bankroll: INITIAL_BANKROLL,
            max_drawdown_pct: 0.0,
            last_prob: HashMap::new(),
            positions: HashMap::new(),
            pending_resolution: HashMap::new(),
            recently_resolved_markets: IndexSet::new(),
        }
    }

    /// Пустые `positions` / `pending_resolution` для всех лейнов валюты ([`crate::real_sim::run_real_sim`]).
    /// `or_default()` идемпотентен при повторном вызове.
    pub fn register_currency_lanes(
        &mut self,
        currency: &str,
        lanes: &[(XFrameIntervalKind, CurrencyUpDownOutcome)],
    ) {
        for (interval, side) in lanes {
            let key = (currency.to_string(), *interval, *side);
            self.positions.entry(key.clone()).or_default();
            self.pending_resolution.entry(key).or_default();
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
