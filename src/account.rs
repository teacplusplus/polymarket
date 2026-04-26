//! Единый счёт-капитал на весь процесс.
//!
//! `Account` — это **денежная** часть состояния симуляции: текущий
//! банкролл, исторический пик equity и максимальная просадка по equity.
//! Всё остальное (per-side / per-interval счётчики trades / wins / fees /
//! kelly-skips / …) живёт в [`crate::history_sim::SimStats`] и считается
//! отдельно.
//!
//! # Зачем выделено отдельно
//!
//! До рефакторинга `bankroll/peak_bankroll/max_drawdown_pct` лежали внутри
//! `SimStats`, причём в `real_sim` поднималось **по одному** `SimStats` на
//! каждый интервал (5m / 15m). Это давало два независимых псевдо-счёта,
//! каждый со своим `INITIAL_BANKROLL` и своим drawdown'ом — что
//! расходилось с реальностью live-торговли на CLOB, где есть **один
//! кошелёк** на весь процесс. Параллельная нагрузка 5m × 10% +
//! 15m × 10% = 20% реального капитала, но симуляция этого не видела.
//!
//! Теперь `Account` — это **один** объект на процесс (или, при
//! необходимости, на группу [`crate::project_manager::ProjectManager`]),
//! пробрасывается через `Arc<RwLock<Account>>` ([`SharedAccount`]) и
//! агрегирует bankroll **сразу со всех 4 лейнов** `(interval, side)`.
//!
//! # Как использовать
//!
//! ## `real_sim` (несколько `ProjectManager`-ов, async)
//!
//! ```ignore
//! let account = Arc::new(RwLock::new(Account::new()));
//! for currency in CURRENCIES {
//!     let pm = ProjectManager::new(currency.into(), account.clone());
//!     real_sim::run_real_sim(pm).await?;
//! }
//! ```
//!
//! ## `history_sim` (синхронный CLI-режим)
//!
//! ```ignore
//! let mut account = Account::new();
//! simulate_event(..., &mut sim_stats, &mut account);
//! print_sim_stats(&tag, &sim_stats, &account);
//! ```
//!
//! Никаких локов в history_sim не нужно — там симуляция строго
//! однопоточная, прямой `&mut Account` достаточно.

use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{INITIAL_BANKROLL, OpenPosition, SimStats};
use crate::real_sim::{interval_label, side_label, RealSimState};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Удобный псевдоним для разделяемого счёта в async-контексте
/// (`real_sim`, `ProjectManager`). Один и тот же `Arc` можно
/// клонировать в произвольное число воркеров и `ProjectManager`-ов —
/// все увидят одни и те же `bankroll/peak/dd`.
pub type SharedAccount = Arc<RwLock<Account>>;

/// Единый счёт-капитал.
///
/// * `bankroll` — текущий **реализованный** капитал в USDC. Меняется
///   только при закрытиях позиций (open не двигает bankroll, потому что
///   `entry_cost` отражается косвенно через mark-to-market и
///   через `available_bankroll = bankroll − Σ(open.entry_cost)` при
///   расчёте Kelly-сайзинга).
///
/// * `peak_bankroll` — исторический пик **equity** (mark-to-market):
///   `bankroll + Σ(shares_held × current_prob)`. Имя историческое
///   («peak_bankroll»), семантика **equity-based** — иначе drawdown
///   системно занижен на длинных удержаниях, уходящих в красное и
///   закрывающихся через Resolution.
///
/// * `max_drawdown_pct` — максимум `(peak − equity) / peak × 100`,
///   накопленный за всю жизнь счёта.
///
/// `Account` **намеренно** не хранит ни счётчики сделок, ни статистику
/// по сторонам, ни PnL: это обязанность [`crate::history_sim::SimStats`].
/// Так каждое поле имеет одного владельца и нет соблазна обновлять
/// drawdown «не там, где менялся bankroll».
#[derive(Debug)]
pub struct Account {
    pub bankroll: f64,
    pub peak_bankroll: f64,
    pub max_drawdown_pct: f64,
    /// Глобальный реестр последних известных `currency_implied_prob`
    /// по каждому лейну `(currency, interval, side)`. Заполняется
    /// `real_sim`-воркерами в начале каждого `tick_once` соответствующего
    /// лейна и используется для mark-to-market по «чужим» лейнам, по
    /// которым на текущем тике мы кадра не получали.
    ///
    /// Ключ включает `currency`, потому что `Account` единый на процесс
    /// и шарится между всеми [`crate::project_manager::ProjectManager`]-ами
    /// (по одному на валюту) — лейны разных валют не должны
    /// перетирать друг другу `last_prob`.
    ///
    /// Если для лейна записи ещё нет (старт процесса) и при этом там
    /// уже есть позиции — потребитель сам выбирает fallback (обычно
    /// `0.5` как нейтральная оценка); в штатной работе позиции
    /// появляются ПОСЛЕ хотя бы одного `tick_once` лейна, поэтому
    /// fallback почти не срабатывает.
    pub last_prob: HashMap<(String, XFrameIntervalKind, CurrencyUpDownOutcome), f64>,
    /// Открытые позиции всех лейнов всех валют, делящих этот счёт.
    /// Ключ — тройка `(currency, interval, side)`. Тот же ключ, что и у
    /// [`Account::last_prob`] — пары `(positions, last_prob)` всегда
    /// согласованы между собой (для каждой непустой `positions[k]`
    /// гарантированно есть `last_prob[k]`, потому что воркер пишет
    /// `last_prob` в начале каждого `tick_once`, а позиции
    /// открываются/закрываются ПОСЛЕ).
    ///
    /// Карта пред-инициализируется на старте `real_sim` (`run_real_sim`)
    /// 4 пустыми `Vec`-ами для каждой валюты, чтобы в `tick_once` можно
    /// было делать `get_mut(&key).unwrap()` без вставки «по
    /// необходимости».
    ///
    /// Хранение позиций здесь, а не в [`crate::real_sim::RealSimState`],
    /// нужно ровно потому же, почему здесь же лежит `bankroll`: счёт
    /// один на процесс, и Kelly-сайзинг должен видеть **всю** уже
    /// занятую entry_cost — и по своей валюте, и по чужим — иначе
    /// 4 лейна × N валют параллельно «съедят» bankroll кратно
    /// `MAX_BET_FRACTION`.
    pub positions: HashMap<(String, XFrameIntervalKind, CurrencyUpDownOutcome), Vec<OpenPosition>>,
    /// Позиции, **осиротевшие** при смене маркета внутри лейна
    /// (`pos.asset_id != frame.asset_id` на момент `manage_positions`).
    ///
    /// # Зачем отдельная корзина
    ///
    /// Когда внутри лейна `(currency, interval, side)` сменяется
    /// текущий CTF-токен (новый 5m/15m раунд стартовал, а позиция от
    /// прошлого раунда ещё не закрылась), к ней нельзя применять
    /// `sell_gate(new_frame, ...)`: ни `currency_implied_prob`, ни
    /// `event_remaining_ms`, ни hold-zone окно нового маркета не
    /// описывают старую позицию. Поэтому stale-позиции **выводятся
    /// из активной книги** (`positions[k]`) и складываются здесь — у
    /// них больше нет «своего фрейма», их единственное оставшееся
    /// событие — резолюция старого маркета.
    ///
    /// # Закрытие
    ///
    /// Эти позиции закрываются НЕ через `manage_positions`, а через
    /// отдельный post-resolution колбек: когда становится известна
    /// `final_price` уже резолвнувшегося маркета (тот же путь, что
    /// поставляет `xframe_dump::final_price`), вызывается
    /// [`Account::resolve_pending_market`] — он находит все позиции
    /// с подходящим `market_id` во всех лейнах и закрывает их по
    /// этой цене, корректно обновляя `bankroll`.
    ///
    /// Ключ — тот же лейн `(currency, interval, side)`, что и у
    /// активных `positions`. Лейн нужен, потому что `last_prob` и
    /// прочая аналитика по позиции продолжают принадлежать тому же
    /// лейну, даже если маркет в нём уже сменился.
    pub pending_resolution: HashMap<(String, XFrameIntervalKind, CurrencyUpDownOutcome), Vec<OpenPosition>>,
}

impl Account {
    /// Стартовый счёт со всем стартовым капиталом из
    /// [`INITIAL_BANKROLL`]. peak инициализируется тем же значением,
    /// чтобы первый же тик не «накрутил» искусственный 100%-drawdown
    /// от нуля.
    pub fn new() -> Self {
        Self {
            bankroll: INITIAL_BANKROLL,
            peak_bankroll: INITIAL_BANKROLL,
            max_drawdown_pct: 0.0,
            last_prob: HashMap::new(),
            positions: HashMap::new(),
            pending_resolution: HashMap::new(),
        }
    }

    /// Пред-инициализирует записи `positions` (и заодно гарантирует
    /// «пустую» позиционную книгу) для всех 4 лейнов одной валюты.
    /// Вызывается из [`crate::real_sim::run_real_sim`] один раз на PM
    /// перед спавном воркеров, чтобы в `tick_once` можно было делать
    /// `get_mut(&key).unwrap()`.
    ///
    /// `entry().or_default()` — идемпотентный: повторный вызов на ту же
    /// валюту не затирает уже накопленные позиции (на случай, если PM
    /// перезапускается, сохраняя `Account`).
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

    /// Закрывает все «осиротевшие» позиции, ждавшие резолюции
    /// маркета `market_id`, по **бинарной выплате Polymarket CTF**.
    ///
    /// # Семантика выплаты
    ///
    /// Polymarket резолвит каждый токен бинарно:
    ///   * победивший токен → `1.0` USDC за каждый share, `net = shares_held × 1.0`;
    ///   * проигравший токен → `0` USDC, `net = 0`.
    ///
    /// `final_price` (цена базового актива в момент закрытия окна,
    /// см. [`crate::xframe_dump::MarketXFramesDump::final_price`])
    /// в самой формуле выплаты **не участвует** — он используется
    /// только для определения **исхода**: `up_won = final_price ≥ price_to_beat`
    /// (см. [`crate::xframe_dump::MarketXFramesDump::up_won`]).
    /// Из `up_won` и стороны позиции (она лежит в лейн-ключе
    /// `pending_resolution`) выводится `token_won`:
    ///   * `Up`-сторона выиграла, если `up_won = true`;
    ///   * `Down`-сторона выиграла, если `up_won = false`.
    ///
    /// # PnL
    ///
    /// Симметрично с `close_position` в ветке `CloseReason::Resolution`:
    ///   * `won  → net = shares_held; pnl = shares_held - entry_cost`;
    ///   * `lost → net = 0;            pnl = -entry_cost`.
    ///
    /// Комиссия на резолюции **не взимается** (Polymarket gas-free
    /// redemption). Это та же модель, что в `close_position`.
    ///
    /// # Параметры
    ///
    /// * `market_id` — `condition_id` маркета, который только что
    ///   разрешился. Совпадение проверяется по `pos.market_id`.
    /// * `up_won` — исход маркета: `true`, если выиграл `Up`-токен,
    ///   `false` — если `Down`. Получается из `MarketXFramesDump::up_won`
    ///   или эквивалентного источника post-resolution.
    ///
    /// # Параметры
    ///
    /// * `account` — общий счёт (`SharedAccount = Arc<RwLock<Account>>`),
    ///   из которого будут вынуты pending-позиции и в котором
    ///   обновится `bankroll`.
    /// * `state` — per-currency `RealSimState` той валюты, к которой
    ///   принадлежит `market_id`. В нём обновятся per-side счётчики
    ///   `SimStats` (`trades`, `wins`/`losses`, `resolution_win`/`loss`,
    ///   `pnl_usd`) — симметрично с `close_position` для ветки
    ///   `CloseReason::Resolution`. Для других валют свой `state`
    ///   не трогается (там нет позиций по этому `market_id`,
    ///   `pending_resolution` фильтруется по `currency`).
    /// * `currency` — какая валюта резолвится. Используется для
    ///   фильтрации лейн-ключей `pending_resolution`: один и тот же
    ///   `market_id` в `Account` появляется только в записях
    ///   с этой валютой, но мы всё равно фильтруем явно — это
    ///   защищает от теоретической коллизии CTF-id между биржами /
    ///   при ручных тестах.
    /// * `market_id` — `condition_id` маркета, который резолвнулся.
    /// * `up_won` — бинарный исход (см. `MarketXFramesDump::up_won`).
    ///
    /// # Lock order
    ///
    /// `state.write() → account.write()` — тот же порядок, что и в
    /// `tick_once` (`real_sim.rs`). Любой иной порядок может дать
    /// дедлок при параллельных воркерах: pending'и могут резолвиться
    /// прямо в момент чьего-то `tick_once`, и оба пути держат оба лока.
    ///
    /// # Side effects
    ///
    /// * `account.bankroll` += сумма PnL всех закрытых позиций.
    /// * Из `account.pending_resolution[lane]` удалены все позиции
    ///   с совпавшим `pos.market_id` (через `swap_remove`).
    /// * `state.stats[interval].{up|down}` обновлён по каждой
    ///   закрытой позиции точно по тем же полям, что и
    ///   `close_position` для `CloseReason::Resolution`:
    ///   `trades += 1`, `wins`/`losses += 1` по знаку PnL,
    ///   `resolution_win`/`resolution_loss += 1` по `token_won`,
    ///   `pnl_usd += pnl`.
    /// * `tee_println!` per-position и итоговая строка — в стиле
    ///   `[resolve] {currency}/{interval}/{side}: ...`, симметрично с
    ///   sim-диагностиками. Если pending по `market_id` пустой —
    ///   тихо ничего не делает и не печатает (нормальная ситуация:
    ///   маркет резолвнулся, а позиций по нему просто не было).
    ///
    /// # Drawdown
    ///
    /// Здесь **не пересчитывается** — это сделает ближайший
    /// `tick_once` через `update_drawdown` на свежем `bankroll`.
    /// Делать MtM-апдейт изнутри этого колбека без знания текущих
    /// prob чужих лейнов нельзя корректно (см. фазу 2 `tick_once`).
    pub async fn resolve_pending_market(
        account: &SharedAccount,
        state: &Arc<RwLock<RealSimState>>,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
    ) {
        // Lock order: state.write() → account.write(), как в `tick_once`.
        let mut state_guard = state.write().await;
        let mut account_guard = account.write().await;

        // 1) Active → pending. В real_sim резолюция-колбек может прийти
        //    раньше, чем сменится `frame.market_id` в лейне (xframe_dump
        //    спит `max_step` сек после закрытия маркета — и фактически
        //    запускается параллельно с приходом нового раунда). Поэтому
        //    некоторые позиции этого маркета всё ещё могут лежать в
        //    активной книге `account.positions[lane_key]`. Перетаскиваем
        //    их в `pending_resolution[lane_key]` ДО запуска sync core,
        //    чтобы он одним проходом закрыл их по бинарному payout.
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

        // 2) Sync core: проходит pending_resolution по нужному
        //    `(currency, interval, *)`, закрывает по бинарной выплате,
        //    обновляет stats того же интервала. Один `&mut SimStats` —
        //    обе стороны (Up/Down) интервала.
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
        );
    }

    /// Синхронное ядро резолюции: то же поведение, что у
    /// [`Account::resolve_pending_market`], но без `Arc<RwLock<_>>`.
    /// Используется в:
    ///
    /// 1. `history_sim::run_side_simulation` — там симуляция
    ///    однопоточная и владеет `&mut Account` напрямую; локи
    ///    не нужны и были бы лишним indirection'ом. Caller
    ///    сам гарантирует, что surviving позиции лейна перенесены
    ///    из локального активного `Vec` в
    ///    `account.pending_resolution[lane_key]` ПЕРЕД вызовом.
    ///
    /// 2. Async-обёрткой [`Account::resolve_pending_market`] —
    ///    после захвата обоих локов и переноса active→pending
    ///    она делегирует собственно резолюцию сюда.
    ///
    /// Всё остальное (формула выплаты, обновление `bankroll` /
    /// `SideStats`) — единое для обоих путей.
    ///
    /// **Логирование всегда включено**: каждая закрытая по резолюции
    /// позиция печатает `[resolve] … {WIN|LOSS} shares=… cost=… pnl=…
    /// bankroll=…` через `tee_println!`, плюс summary `closed= total_pnl=`
    /// в конце. Раньше был `log: bool`-флаг, чтобы глушить эти строки
    /// в `history_sim` (там маркетов сотни), но сейчас на всех
    /// путях прогон пишется ещё и в файл (`xframes/last_history_sim.txt`,
    /// `xframes/last_real_sim.txt`) — терять из лога per-market PnL,
    /// который драйвит итоговый bankroll, нельзя ни в backtest'e, ни в
    /// live: на этих строках строится трассировка «как именно сложилась
    /// final-метрика».
    pub fn resolve_pending_market_sync(
        &mut self,
        sim_stats: &mut SimStats,
        currency: &str,
        interval: XFrameIntervalKind,
        market_id: &str,
        up_won: bool,
    ) {
        // Split borrow: одновременно нужны `pending_resolution` (итерация)
        // и `bankroll` (PnL).
        let Account {
            bankroll,
            pending_resolution,
            ..
        } = self;

        let mut closed = 0usize;
        let mut total_pnl = 0.0_f64;

        for ((cur, int_kind, side), vec) in pending_resolution.iter_mut() {
            // Фильтр по `(currency, interval)`: 5m и 15m раунды независимы
            // по `market_id`, и резолюция 5m не должна задевать 15m
            // позиции. Фильтр по валюте — защита от теоретической
            // CTF-id коллизии.
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
                        // net = shares_held × 1.0, без комиссии.
                        pos.shares_held - pos.entry_cost
                    } else {
                        // net = 0; теряем весь entry_cost.
                        -pos.entry_cost
                    };
                    *bankroll += pnl;
                    total_pnl += pnl;
                    closed += 1;

                    // Симметрично c прежней `close_position` веткой
                    // `CloseReason::Resolution`: те же поля SideStats.
                    side_stats.pnl_usd += pnl;
                    side_stats.trades += 1;
                    if pnl >= 0.0 {
                        side_stats.wins += 1;
                    } else {
                        side_stats.losses += 1;
                    }
                    if token_won {
                        side_stats.resolution_win += 1;
                        // Разбивка по знаку pnl: `resolution_win` —
                        // token-outcome счётчик («токен победил»),
                        // но при дорогих входах (`entry_prob` ~ 1.0)
                        // выплата `shares_held × $1` минус
                        // `entry_cost` ± entry-fee может оказаться в
                        // минусе. Симметрично с `wins/losses`-сплитом,
                        // только в рамках token_won.
                        if pnl >= 0.0 {
                            side_stats.resolution_win_profit += 1;
                        } else {
                            side_stats.resolution_win_loss += 1;
                        }
                    } else {
                        // `resolution_loss`: проигравший токен
                        // погашается за $0, потеря всегда равна
                        // `entry_cost`, дополнительной разбивки по
                        // знаку pnl не нужно.
                        side_stats.resolution_loss += 1;
                    }

                    {
                        let tag = format!(
                            "{}/{}/{}",
                            cur,
                            interval_label(*int_kind),
                            side_label(*side),
                        );
                        let outcome = if token_won { "WIN" } else { "LOSS" };
                        crate::tee_println!(
                            "[resolve] {tag} market={market_id} {outcome} \
                             shares={shares:.4} cost={cost:.4} pnl={pnl:+.4} bankroll={bankroll:.4}",
                            tag = tag,
                            market_id = market_id,
                            outcome = outcome,
                            shares = pos.shares_held,
                            cost = pos.entry_cost,
                            pnl = pnl,
                            bankroll = *bankroll,
                        );
                    }
                } else {
                    i += 1;
                }
            }
        }

        if closed > 0 {
            crate::tee_println!(
                "[resolve] {currency}/{interval} market={market_id} closed={closed} \
                 total_pnl={total_pnl:+.4} bankroll={bankroll:.4}",
                currency = currency,
                interval = interval_label(interval),
                market_id = market_id,
                closed = closed,
                total_pnl = total_pnl,
                bankroll = *bankroll,
            );
        }
    }

    /// Удобный конструктор для async-контекста: сразу оборачивает
    /// `Account` в `Arc<RwLock<_>>`.
    ///
    /// Используется один раз в `main.rs` перед спавном
    /// `ProjectManager`-ов; полученный `SharedAccount` клонируется
    /// (cheap Arc-clone) в каждый PM.
    pub fn new_shared() -> SharedAccount {
        Arc::new(RwLock::new(Self::new()))
    }

    /// Обновляет `peak_bankroll` и `max_drawdown_pct` по mark-to-market
    /// equity. Должна вызываться на **каждом** тике, а не только на
    /// сделке — между `open` и `close` реализованный `bankroll` не
    /// двигается, и без mark-to-market drawdown системно занижен.
    ///
    /// Вызыватель сам считает equity:
    /// `equity = bankroll + Σ(shares_held × current_prob)` по всем
    /// открытым позициям всех лейнов, делящих этот счёт.
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
