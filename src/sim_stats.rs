//! Агрегаты симуляции по стороне и версии: [`SideStats`], [`SimStats`], лог-печать.

use crate::tee_println;

/// Статистика по одной стороне (UP/DOWN).
#[derive(Debug, Default)]
pub struct SideStats {
    /// Число закрытых сделок (одна позиция = одна сделка).
    pub(crate) trades: usize,
    /// Сделки с PnL ≥ 0.
    pub(crate) wins: usize,
    /// Сделки с PnL < 0.
    pub(crate) losses: usize,
    /// Суммарный PnL USDC после комиссий.
    pub(crate) pnl_usd: f64,
    /// Сумма taker-комиссий (open + рыночный close).
    pub(crate) fees_paid: f64,
    /// Закрытия по TP ([`crate::xframe::Y_TRAIN_TAKE_PROFIT_PP`]).
    pub(crate) tp_count: usize,
    /// Закрытия по SL.
    pub(crate) sl_count: usize,
    /// Резолюция: токен выиграл (исход верный; PnL может быть < 0).
    pub(crate) resolution_win: usize,
    /// Из них сделки с PnL ≥ 0.
    pub(crate) resolution_win_profit: usize,
    /// Из них с PnL < 0 при верном исходе.
    pub(crate) resolution_win_loss: usize,
    /// Резолюция: токен проиграл (убыток ~ −entry).
    pub(crate) resolution_loss: usize,
    /// Закрытия по таймауту ([`crate::history_sim::POSITION_TIMEOUT_FRAMES`]).
    pub(crate) timeout_count: usize,
    /// Счётчик [`crate::history_sim::CloseReason::EvExitProfit`].
    pub(crate) ev_exit_profit_count: usize,
    /// Счётчик [`crate::history_sim::CloseReason::EvExitLoss`].
    pub(crate) ev_exit_loss_count: usize,
    /// Пропуск входа: мало времени до резолюции ([`crate::history_sim::MIN_ENTRY_REMAINING_MS`]).
    pub(crate) late_entry_skips: usize,
    /// Пропуск входа: нестабильный кадр; закрытие открытых не блокируется.
    pub(crate) unstable_skips: usize,
    /// Пропуск: уже открыта позиция на том же `asset_id`.
    pub(crate) same_asset_open_skips: usize,
    /// Пропуск: Kelly f* ≤ 0.
    pub(crate) kelly_skips: usize,
    /// Пропуск: `entry_prob` в no-trade зоне ([`crate::history_sim::BuyGate::EntryProbOutOfRange`]).
    pub(crate) entry_prob_skips: usize,
    /// Strict: не хватило asks на полный BUY ([`crate::real_sim`]).
    pub(crate) kelly_strict_buy_skips: usize,
    /// Strict: не хватило bids на полный SELL при закрытии.
    pub(crate) kelly_strict_sell_skips: usize,
    /// Кадры с raw ≥ порога (воронка).
    pub(crate) raw_above_threshold: usize,
    /// Сумма raw pred по претендентам (делить на [`Self::raw_above_threshold`]).
    pub(crate) diag_sum_raw: f64,
    /// Сумма cal pred по тем же кадрам.
    pub(crate) diag_sum_calibrated: f64,
    /// Сумма entry_prob по претендентам.
    pub(crate) diag_sum_entry_prob: f64,
    /// Сумма сырого Kelly f* до [`crate::history_sim::KELLY_MULTIPLIER`].
    pub(crate) diag_sum_kelly_f: f64,
    /// Гистограмма entry_prob при успешном открытии (5 бакетов по 0.2).
    pub(crate) histogram_entry_prob: [usize; 5],
    /// Гистограмма cal pred при открытии (та же сетка).
    pub(crate) histogram_cal_pred: [usize; 5],
    /// Сумма PnL закрытий TP.
    pub(crate) pnl_tp: f64,
    /// Сумма PnL закрытий SL.
    pub(crate) pnl_sl: f64,
    /// Сумма PnL закрытий по таймауту.
    pub(crate) pnl_timeout: f64,
    /// Сумма PnL при [`crate::history_sim::CloseReason::EvExitProfit`].
    pub(crate) pnl_ev_exit_profit: f64,
    /// Сумма PnL при [`crate::history_sim::CloseReason::EvExitLoss`].
    pub(crate) pnl_ev_exit_loss: f64,
    /// Сумма PnL резолюций с выигравшим токеном.
    pub(crate) pnl_resolution_win: f64,
    /// Сумма PnL резолюций с проигравшим токеном (≤ 0).
    pub(crate) pnl_resolution_loss: f64,
    /// Для replay-калибровки: (raw на открытии, won); в обычном sim пусто.
    pub(crate) closed_trade_entries: Vec<(f32, bool)>,
    /// Сырые preds resolution в hold-zone (train-калибровка без cal_resolution).
    pub(crate) hold_zone_resolution_predictions: Vec<f32>,
}

/// Агрегаты симуляции по версии; банк и DD — в [`crate::account::Account`].
#[derive(Debug)]
pub struct SimStats {
    /// Обработано маркетов `.bin`.
    pub(crate) events: usize,
    /// Статистика UP.
    pub(crate) up: SideStats,
    /// Статистика DOWN.
    pub(crate) down: SideStats,
}

impl SimStats {
    pub(crate) fn new() -> Self {
        Self {
            events: 0,
            up: SideStats::default(),
            down: SideStats::default(),
        }
    }

    pub(crate) fn total_trades(&self) -> usize {
        self.up.trades + self.down.trades
    }
    pub(crate) fn total_wins(&self) -> usize {
        self.up.wins + self.down.wins
    }
    pub(crate) fn total_losses(&self) -> usize {
        self.up.losses + self.down.losses
    }
    pub(crate) fn total_pnl(&self) -> f64 {
        self.up.pnl_usd + self.down.pnl_usd
    }
    pub(crate) fn total_fees(&self) -> f64 {
        self.up.fees_paid + self.down.fees_paid
    }
    pub(crate) fn total_kelly_skips(&self) -> usize {
        self.up.kelly_skips + self.down.kelly_skips
    }
    pub(crate) fn total_kelly_strict_buy_skips(&self) -> usize {
        self.up.kelly_strict_buy_skips + self.down.kelly_strict_buy_skips
    }
    pub(crate) fn total_kelly_strict_sell_skips(&self) -> usize {
        self.up.kelly_strict_sell_skips + self.down.kelly_strict_sell_skips
    }
    pub(crate) fn total_same_asset_open_skips(&self) -> usize {
        self.up.same_asset_open_skips + self.down.same_asset_open_skips
    }
    pub(crate) fn total_entry_prob_skips(&self) -> usize {
        self.up.entry_prob_skips + self.down.entry_prob_skips
    }
}

pub fn print_side_stats(tag: &str, side_label: &str, s: &SideStats, is_kelly: bool) {
    let n = s.raw_above_threshold.max(1) as f64;
    let diag = if is_kelly {
        format!(
            "raw≥thr={} avg_raw={:.3} avg_cal={:.3} avg_entry={:.3} avg_kelly_f={:.4} kelly_skips={} entry_prob_skips={} same_asset_open_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={}",
            s.raw_above_threshold,
            s.diag_sum_raw / n,
            s.diag_sum_calibrated / n,
            s.diag_sum_entry_prob / n,
            s.diag_sum_kelly_f / n,
            s.kelly_skips,
            s.entry_prob_skips,
            s.same_asset_open_skips,
            s.kelly_strict_buy_skips,
            s.kelly_strict_sell_skips,
        )
    } else {
        format!(
            "raw≥thr={} avg_raw={:.3} avg_entry={:.3} entry_prob_skips={} same_asset_open_skips={} bankroll_too_small_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={}",
            s.raw_above_threshold,
            s.diag_sum_raw / n,
            s.diag_sum_entry_prob / n,
            s.entry_prob_skips,
            s.same_asset_open_skips,
            s.kelly_skips,
            s.kelly_strict_buy_skips,
            s.kelly_strict_sell_skips,
        )
    };
    tee_println!("[sim] {tag} [{side_label}]   {diag}");

    if s.trades == 0 {
        tee_println!("[sim] {tag} [{side_label}]: нет сделок");
        return;
    }
    let win_rate = s.wins as f64 / s.trades as f64 * 100.0;
    let avg_pnl = s.pnl_usd / s.trades as f64;
    tee_println!(
        "[sim] {tag} [{side_label}] \
         | trades={} win={:.1}% \
         | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
         | TP={} SL={} Timeout={} EvExit✓={} EvExit✗={} Res✓={}(profit={}/loss={}) Res✗={} late_skips={} unstable_skips={} same_asset_open_skips={}",
        s.trades,
        win_rate,
        s.pnl_usd,
        avg_pnl,
        s.fees_paid,
        s.tp_count,
        s.sl_count,
        s.timeout_count,
        s.ev_exit_profit_count,
        s.ev_exit_loss_count,
        s.resolution_win,
        s.resolution_win_profit,
        s.resolution_win_loss,
        s.resolution_loss,
        s.late_entry_skips,
        s.unstable_skips,
        s.same_asset_open_skips,
    );

    tee_println!(
        "[sim] {tag} [{side_label}] entry_prob hist (0..0.2 / 0.2..0.4 / 0.4..0.6 / 0.6..0.8 / 0.8..1): {} / {} / {} / {} / {}",
        s.histogram_entry_prob[0],
        s.histogram_entry_prob[1],
        s.histogram_entry_prob[2],
        s.histogram_entry_prob[3],
        s.histogram_entry_prob[4],
    );
    if is_kelly {
        tee_println!(
            "[sim] {tag} [{side_label}] cal_pred  hist (0..0.2 / 0.2..0.4 / 0.4..0.6 / 0.6..0.8 / 0.8..1): {} / {} / {} / {} / {}",
            s.histogram_cal_pred[0],
            s.histogram_cal_pred[1],
            s.histogram_cal_pred[2],
            s.histogram_cal_pred[3],
            s.histogram_cal_pred[4],
        );
    }

    let avg = |sum: f64, cnt: usize| if cnt == 0 { 0.0 } else { sum / cnt as f64 };
    tee_println!(
        "[sim] {tag} [{side_label}] pnl_by_reason: \
         TP={tp_pnl:+.2}$(avg={tp_avg:+.4}) SL={sl_pnl:+.2}$(avg={sl_avg:+.4}) \
         Timeout={to_pnl:+.2}$(avg={to_avg:+.4}) \
         EvExit✓={evp_pnl:+.2}$(avg={evp_avg:+.4}) EvExit✗={evl_pnl:+.2}$(avg={evl_avg:+.4}) \
         Res✓={rw_pnl:+.2}$(avg={rw_avg:+.4}) Res✗={rl_pnl:+.2}$(avg={rl_avg:+.4})",
        tp_pnl = s.pnl_tp,
        tp_avg = avg(s.pnl_tp, s.tp_count),
        sl_pnl = s.pnl_sl,
        sl_avg = avg(s.pnl_sl, s.sl_count),
        to_pnl = s.pnl_timeout,
        to_avg = avg(s.pnl_timeout, s.timeout_count),
        evp_pnl = s.pnl_ev_exit_profit,
        evp_avg = avg(s.pnl_ev_exit_profit, s.ev_exit_profit_count),
        evl_pnl = s.pnl_ev_exit_loss,
        evl_avg = avg(s.pnl_ev_exit_loss, s.ev_exit_loss_count),
        rw_pnl = s.pnl_resolution_win,
        rw_avg = avg(s.pnl_resolution_win, s.resolution_win),
        rl_pnl = s.pnl_resolution_loss,
        rl_avg = avg(s.pnl_resolution_loss, s.resolution_loss),
    );
}

/// Итог прогона; `initial_bankroll` — старт USDC для ROI (`[`crate::history_sim::INITIAL_BANKROLL`]` при history/real_sim).
pub fn print_sim_stats(
    tag: &str,
    sim_stats: &SimStats,
    bankroll_now: f64,
    max_drawdown_pct_now: f64,
    is_kelly: bool,
    initial_bankroll: f64,
) {
    let total_trades = sim_stats.total_trades();
    if total_trades == 0 {
        if is_kelly {
            tee_println!(
                "[sim] {tag}: нет сделок ({} событий, kelly_skips={} entry_prob_skips={} same_asset_open_skips={} kelly_strict_buy_skips={} kelly_strict_sell_skips={})",
                sim_stats.events,
                sim_stats.total_kelly_skips(),
                sim_stats.total_entry_prob_skips(),
                sim_stats.total_same_asset_open_skips(),
                sim_stats.total_kelly_strict_buy_skips(),
                sim_stats.total_kelly_strict_sell_skips(),
            );
        } else {
            tee_println!(
                "[sim] {tag}: нет сделок ({} событий, entry_prob_skips={} same_asset_open_skips={} bankroll_too_small_skips={})",
                sim_stats.events,
                sim_stats.total_entry_prob_skips(),
                sim_stats.total_same_asset_open_skips(),
                sim_stats.total_kelly_skips(),
            );
        }
        print_side_stats(tag, "UP", &sim_stats.up, is_kelly);
        print_side_stats(tag, "DOWN", &sim_stats.down, is_kelly);
        return;
    }

    let total_pnl = sim_stats.total_pnl();
    let total_wins = sim_stats.total_wins();
    let total_fees = sim_stats.total_fees();
    let win_rate = total_wins as f64 / total_trades as f64 * 100.0;
    let avg_pnl = total_pnl / total_trades as f64;
    let roi_pct = (bankroll_now - initial_bankroll) / initial_bankroll * 100.0;

    let total_losses = sim_stats.total_losses();
    if is_kelly {
        tee_println!(
            "[sim] {tag} \
             | events={} trades={} win={:.1}% \
             | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
             | wins={total_wins} losses={total_losses} \
             | kelly_skips={ks} entry_prob_skips={eps} same_asset_open_skips={sas} kelly_strict_buy_skips={ksb} kelly_strict_sell_skips={kss}",
            sim_stats.events,
            total_trades,
            win_rate,
            total_pnl,
            avg_pnl,
            total_fees,
            ks = sim_stats.total_kelly_skips(),
            eps = sim_stats.total_entry_prob_skips(),
            sas = sim_stats.total_same_asset_open_skips(),
            ksb = sim_stats.total_kelly_strict_buy_skips(),
            kss = sim_stats.total_kelly_strict_sell_skips(),
        );
    } else {
        tee_println!(
            "[sim] {tag} \
             | events={} trades={} win={:.1}% \
             | pnl={:+.2}$ avg={:+.4}$/trade fees={:.2}$ \
             | wins={total_wins} losses={total_losses} \
             | entry_prob_skips={eps} same_asset_open_skips={sas} bankroll_too_small_skips={bts}",
            sim_stats.events,
            total_trades,
            win_rate,
            total_pnl,
            avg_pnl,
            total_fees,
            eps = sim_stats.total_entry_prob_skips(),
            sas = sim_stats.total_same_asset_open_skips(),
            bts = sim_stats.total_kelly_skips(),
        );
    }
    tee_println!(
        "[sim]   bankroll: {:.2}$ (start={initial_bankroll}$) ROI={:+.2}% max_drawdown={:.2}%",
        bankroll_now,
        roi_pct,
        max_drawdown_pct_now,
    );

    print_side_stats(tag, "UP", &sim_stats.up, is_kelly);
    print_side_stats(tag, "DOWN", &sim_stats.down, is_kelly);
}
