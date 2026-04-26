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
use crate::history_sim::{INITIAL_BANKROLL, OpenPosition};
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
            self.positions
                .entry((currency.to_string(), *interval, *side))
                .or_default();
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
