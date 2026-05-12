//! Режим обучения: читает дампы [`crate::xframe_dump::MarketXFramesDump`] из папки `xframes/` **по одному файлу**
//! (без удержания всех `frames_up`/`frames_down` в RAM), строит матрицы признаков и меток, обучает XGBoost с байесовской оптимизацией гиперпараметров
//! и сохраняет модель рядом с папкой версии.

use crate::account::Account;
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{
    HOLD_TO_END_THRESHOLD_SEC, MIN_ENTRY_REMAINING_MS, load_market_xframes, run_side_simulation,
    window_bounds_from_dump_path,
};
use crate::sim_stats::{SideStats, SimStats};
use crate::project_manager::FRAME_BUILD_INTERVALS_SEC;
use crate::tee_log::TEE_LOG;
use crate::xframe::{
    SIZE, XFrame, Y_TRAIN_HORIZON_FRAMES, Y_TRAIN_STOP_LOSS_PP, Y_TRAIN_TAKE_PROFIT_PP,
    apply_side_symmetry, calc_y_train_pnl, calc_y_train_resolution,
};
use crate::xframe_dump::MarketXFramesDump;
use crate::{tee_eprintln, tee_println};
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, ParamValue, Study};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use xgb::parameters::learning::{
    EvaluationMetric, LearningTaskParametersBuilder, Metrics, Objective,
};
use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
use xgb::parameters::{BoosterParametersBuilder, BoosterType, TrainingParametersBuilder};
use xgb::{Booster, DMatrix};

/// Число итераций байесовского оптимизатора (TPE) для PnL-модели.
const OPTIMIZER_TRIALS_PNL: usize = 100;
/// Число итераций байесовского оптимизатора (TPE) для Resolution-модели.
const OPTIMIZER_TRIALS_RESOLUTION: usize = 100;
/// Максимальное число раундов бустинга при финальном обучении.
const BOOST_ROUNDS: u32 = 500;
/// Число раундов без улучшения AUC до остановки (early stopping).
const EARLY_STOPPING_PATIENCE: u32 = 20;
/// Базовое число раундов бустинга на TPE-пробу при референсном [`EVAL_REFERENCE_ETA`].
/// Реальный бюджет раундов масштабируется в [`eval_boost_rounds`] обратно
/// пропорционально `eta`, чтобы медленные модели (малый `eta`) успевали сойтись
/// и не проседали по AUC из-за недоучивания.
const EVAL_BOOST_ROUNDS: u32 = 80;
/// Верхняя граница раундов на TPE-пробу: даже при очень малом `eta` не уходим
/// в квадратичный по времени оптимайзер. См. [`eval_boost_rounds`].
const EVAL_BOOST_ROUNDS_MAX: u32 = 300;
/// Референсный `eta`, относительно которого [`EVAL_BOOST_ROUNDS`] считается
/// «правильным» бюджетом. При меньших `eta` число раундов увеличивается
/// пропорционально `reference_eta / eta`.
const EVAL_REFERENCE_ETA: f32 = 0.1;
/// Нижняя граница `eta` в пространстве поиска TPE. Значения ниже ~0.03
/// при фиксированном `EVAL_BOOST_ROUNDS` гарантированно не сходятся и
/// приводят к вырожденным «лучшим» trial'ам (AUC пробы близок к константе,
/// early stopping финального обучения останавливается на первых раундах).
const ETA_MIN: f32 = 0.01;
/// Верхняя граница `eta` — соответствует прежнему поведению.
const ETA_MAX: f32 = 0.5;
/// Доля валидационной выборки (для optimizer + early stopping).
pub const VAL_FRACTION: f64 = 0.2;
/// Доля тестовой выборки (финальная, честная оценка AUC).
pub const TEST_FRACTION: f64 = 0.2;
/// Понижающий коэффициент `feature_weights` для конкретных фич из [`DOWNWEIGHTED_FEATURES`].
/// `None` — не понижать эти фичи.
const DOWNWEIGHT_FACTOR: Option<f32> = Some(0.1);
/// Понижающий коэффициент для лаговых фич (массивы `delta_n_*[i]`).
/// `None` — не понижать лаговые фичи.
///
/// Чем больше индекс `i` в суффиксе `[i]`, тем ниже фактический вес: множится
/// на [`LAG_DOWNWEIGHT_PER_STEP`] для каждого шага после `[0]` (см.
/// [`lag_downweight_with_index`]).
const LAG_DOWNWEIGHT_FACTOR: Option<f32> = Some(0.3);
/// На каждый следующий индекс лага (`[0]` → `[1]` → …) вес дополнительно
/// умножается на это значение. `1.0` — только базовый [`LAG_DOWNWEIGHT_FACTOR`] без затухания по индексу.
const LAG_DOWNWEIGHT_PER_STEP: f32 = 0.88;
/// Имена фич, которым автоматически понижается `feature_weight` при обучении.
// const DOWNWEIGHTED_FEATURES: &[&str] = &["event_remaining_ms", "sibling_event_remaining_ms", "currency_price_vs_beat_pct", "sibling_currency_price_vs_beat_pct"];
const DOWNWEIGHTED_FEATURES: &[&str] = &[
    "event_remaining_ms",
    "sibling_event_remaining_ms",
    "sibling_currency_price_vs_beat_pct",
    "currency_implied_prob",
];
/// Ниже этого порога сохраняется identity-калибровка.
const CALIBRATION_MIN_AUC: f32 = 0.60;
/// Эпсилон для клиппинга выходов isotonic regression: исключает 0/1 значения,
/// которые сломают logloss и Kelly при логарифмировании.
const CALIBRATION_EPS: f32 = 1e-3;
/// Верхний кап на минимальный суммарный вес одного блока (число сэмплов) в
/// isotonic-калибровке. Эффективный порог адаптивный:
/// `min_weight = min(CALIBRATION_MIN_BLOCK_WEIGHT_CAP, n_entries / 5)`.
/// После PAV последовательно объединяем соседние блоки до достижения этого порога —
/// это регуляризация против переобучения на малых калибровочных сетах.
/// Монотонность при этом сохраняется (weighted-avg двух non-decreasing соседей
/// остаётся в интервале [prev, next]).
///
/// Адаптивность нужна потому, что фиксированные `10` на маленьком sim-replay
/// сете (например, 95 трейдов после `SIM_BUY_THRESHOLD`) гарантируют схлопывание
/// в один bucket → константная калибровка. С `n/5` мы получаем ≈5 ступенек на
/// любом размере сета, при этом для крупных сетов сохраняем полноценные 10.
const CALIBRATION_MIN_BLOCK_WEIGHT_CAP: f64 = 10.0;

/// Минимальное количество трейдов в калибровочном сете sim-replay
/// (см. [`fit_calibration_via_sim_replay`]). Если симулятор на val'е набрал
/// меньше — калибровка fallback'ит на полный per-frame `(preds, y)` сет
/// с предупреждением. Меньше ~20 точек слишком мало для устойчивого PAV:
/// верхний бакет может содержать всего 1–2 трейда, и `cal(raw≥0.7)` будет случайным.
const CALIBRATION_MIN_FILTERED_SAMPLES: usize = 20;

/// Стартовый банкролл для sim-replay калибровки
/// (см. [`fit_calibration_via_sim_replay`]).
///
/// **Намеренно ≫ [`crate::history_sim::INITIAL_BANKROLL`]**: реальный $50
/// банкролл с `BLOCK_SAME_ASSET_OPEN=false` после 1-2 одновременных $30-входов
/// упирается в `available = bankroll - same_side_locked` ≤ 0 и **скипает**
/// последующие signal-frames (или открывает их меньшим размером). Это
/// добавляет bankroll-driven sampling bias в калибровочный сет: «поздние»
/// сигналы маркета и сигналы после серии лоссов недопредставлены не потому,
/// что хуже, а потому что денег нет.
///
/// Калибровка отвечает на вопрос «при `raw=X` какова **истинная** P(win)?» —
/// это свойство сигнала модели, а не истории трейдов. Чтобы убрать bias,
/// банкролл выбран настолько большим, чтобы `min(NO_KELLY_POSITION_SIZE_USD,
/// bankroll) = NO_KELLY_POSITION_SIZE_USD` всегда, а `available ≫ entry_cost`
/// никогда не отбраковывал сигнал (см. [`crate::history_sim::try_open_position`]
/// и [`crate::history_sim::run_side_simulation`]).
const CALIBRATION_REPLAY_BANKROLL_USD: f64 = 1.0e12;

/// Базовый уровень logloss для штрафа в [`TuneObjective::MaximizeAucWithPenalty`].
/// При типичном дисбалансе классов (~25% y=1) константная модель даёт logloss
/// около 0.55, поэтому штрафуем только то, что хуже этого уровня. Если AUC
/// высокий, но `logloss > baseline`, модель «угадывает» порядок, но
/// откалибрована плохо — это снижает usability для Kelly-сайзинга.
const AUC_PENALTY_LOGLOSS_BASELINE: f64 = 0.55;
/// Насколько сильно высокий logloss вычитается из AUC в objective:
/// `score = auc - max(0, logloss - baseline) * weight`. Подобран так, чтобы
/// разница в `0.1` logloss «съедала» примерно `0.005` AUC — соизмеримо с
/// шумом trial-to-trial разброса AUC, но достаточно, чтобы TPE предпочёл
/// калиброванные модели при равном AUC.
const AUC_PENALTY_LOGLOSS_WEIGHT: f64 = 0.05;

/// Цель байесовской оптимизации гиперпараметров XGBoost. Переключается через
/// константу [`TUNE_OBJECTIVE`] — параметризовать через env / CLI смысла нет,
/// решение про objective привязано к билду модели и попадает в обучающий лог.
///
/// * [`TuneObjective::MaximizeAuc`] — классический критерий ranking-power.
///   Игнорирует калибровку: модель может давать высокий AUC, но смещённые
///   probability'ы (например, средняя `cal_pred` ≪ базовая частота),
///   что ломает Kelly. Полезен, если калибровка делается отдельным шагом
///   (как в `xframe.rs` через isotonic) и от booster'а нужна только
///   способность сортировать сэмплы.
///
/// * [`TuneObjective::MinimizeLogLoss`] — оптимизирует именно вероятностные
///   предсказания. На сильно несбалансированных классах (`scale_pos_weight ≫ 1`)
///   часто загоняет модель в константный режим (logloss ≈ baseline), потому
///   что AUC он не контролирует.
///
/// * [`TuneObjective::MaximizeAucWithPenalty`] — компромисс: максимизирует AUC,
///   но вычитает штраф за logloss выше [`AUC_PENALTY_LOGLOSS_BASELINE`]
///   с весом [`AUC_PENALTY_LOGLOSS_WEIGHT`]. Защищает от «high-AUC,
///   broken-calibration»-trial'ов и обычно совпадает с `MaximizeAuc` на
///   хорошо откалиброванных trial'ах (penalty=0). Это default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneObjective {
    MinimizeLogLoss,
    MaximizeAuc,
    MaximizeAucWithPenalty,
}

impl TuneObjective {
    /// Краткое имя метрики, печатается в TPE-логах рядом с `auc`/`logloss`.
    fn label(&self) -> &'static str {
        match self {
            TuneObjective::MinimizeLogLoss => "logloss",
            TuneObjective::MaximizeAuc => "auc",
            TuneObjective::MaximizeAucWithPenalty => "auc_with_logloss_penalty",
        }
    }

    /// Направление оптимизации для [`Study`]: `MaximizeAuc` /
    /// `MaximizeAucWithPenalty` — `Direction::Maximize`,
    /// `MinimizeLogLoss` — `Direction::Minimize`.
    fn direction(&self) -> Direction {
        match self {
            TuneObjective::MinimizeLogLoss => Direction::Minimize,
            TuneObjective::MaximizeAuc => Direction::Maximize,
            TuneObjective::MaximizeAucWithPenalty => Direction::Maximize,
        }
    }

    /// Сравнение свёрнутого скора для early stopping: согласовано с
    /// [`TrialMetrics::score_for`] и [`direction`] (лучше выше при maximize,
    /// лучше ниже при minimize).
    fn score_improved(self, new_score: f64, best_score: f64) -> bool {
        match self.direction() {
            Direction::Maximize => new_score > best_score,
            Direction::Minimize => new_score < best_score,
        }
    }
}

/// Активная цель байесовской оптимизации. Меняется одной правкой константы;
/// см. doc у [`TuneObjective`] для разбора вариантов.
const TUNE_OBJECTIVE: TuneObjective = TuneObjective::MinimizeLogLoss;

/// Макс. отклонение VWAP от L1 при разметке y_train (вход TP / добровольные
/// выходы): передаётся в [`crate::xframe::calc_y_train_pnl`] /
/// [`crate::xframe::calc_y_train_resolution`]. Симуляция исполнения —
/// [`crate::history_sim::SIM_MAX_SLIPPAGE_FROM_L1_PCT`].
pub const Y_TRAIN_MAX_SLIPPAGE_FROM_L1_PCT: f64 = 0.2;

/// Максимальный лаг `delta_n_*` для PnL-модели: `None` — полный вектор
/// [`XFrame::to_x_train_with`]; `Some(n)` — обрезка лагов до `n` первых
/// элементов через [`XFrame::to_x_train_n_with`]. Общий источник истины
/// для тренера и [`crate::history_sim`]: один и тот же feature layout
/// на обучении и инференсе.
pub const PNL_MAX_LAG: Option<usize> = None;
/// Максимальный лаг `delta_n_*` для Resolution-модели (см. [`PNL_MAX_LAG`]).
pub const RESOLUTION_MAX_LAG: Option<usize> = None;

// ─── Калибровка (Isotonic Regression) ────────────────────────────────────────

/// Изотоническая калибровка: кусочно-линейная монотонная (non-decreasing) функция
/// `calibrated = f(raw)`, подогнанная алгоритмом PAV (Pool Adjacent Violators).
///
/// Сохраняется рядом с моделью (`.calibration.bin`).
///
/// # Представление
///
/// После PAV результат — последовательность блоков с неубывающими значениями.
/// Для каждого блока берётся один опорный узел `(x, y)`, где `x` — взвешенное
/// среднее raw-предсказаний в блоке, `y` — доля позитивных меток в блоке
/// (с клиппингом в `[CALIBRATION_EPS, 1 − CALIBRATION_EPS]`).
///
/// `apply(raw)` возвращает линейно-интерполированное значение между соседними
/// опорными узлами; на границах — ближайшее `ys`-значение.
///
/// # Преимущество перед Platt scaling
///
/// Isotonic не предполагает параметрическую форму (sigmoid) и не сжимает
/// «хвосты» распределения raw-предсказаний на скошенных данных.
/// Platt scaling на DOWN-моделях сжимал `raw 0.79 → cal 0.32`, ломая Kelly-фильтр.
/// PAV сохраняет монотонность, но не искажает плотность сигнала.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    /// Опорные raw-значения, строго возрастают.
    pub xs: Vec<f32>,
    /// Калибровочные значения в опорных узлах; неубывающая последовательность
    /// в `[CALIBRATION_EPS, 1 − CALIBRATION_EPS]`.
    pub ys: Vec<f32>,
}

impl Calibration {
    /// Тождественная калибровка `apply(raw) = raw` — используется как fallback
    /// при слабом AUC или пустом калибровочном сете.
    pub fn identity() -> Self {
        Self {
            xs: vec![0.0, 1.0],
            ys: vec![0.0, 1.0],
        }
    }

    /// Применяет isotonic к сырому предсказанию XGBoost.
    ///
    /// Для `raw` вне диапазона опорных узлов возвращает ближайшее ys (края).
    /// Внутри диапазона — линейная интерполяция между соседними узлами.
    pub fn apply(&self, raw_pred: f32) -> f32 {
        let n = self.xs.len();
        if n == 0 {
            return raw_pred;
        }
        if n == 1 {
            return self.ys[0];
        }
        if raw_pred <= self.xs[0] {
            return self.ys[0];
        }
        if raw_pred >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        // Бинарный поиск интервала: xs[idx - 1] ≤ raw_pred ≤ xs[idx].
        let idx = match self.xs.binary_search_by(|probe| {
            probe
                .partial_cmp(&raw_pred)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => return self.ys[i],
            Err(i) => i,
        };
        let (x0, x1) = (self.xs[idx - 1], self.xs[idx]);
        let (y0, y1) = (self.ys[idx - 1], self.ys[idx]);
        let dx = x1 - x0;
        if dx.abs() < f32::EPSILON {
            return y1;
        }
        let t = (raw_pred - x0) / dx;
        y0 + t * (y1 - y0)
    }
}

/// Прогон `run_side_simulation` по val-маркетам с `is_kelly=false` и identity-калибровкой:
/// возвращает пары `(raw_pred_at_open, won)` со всех **закрытых** трейдов
/// ([`crate::history_sim::close_position`] + [`crate::account::Account::resolve_pending_market_sync`]).
///
/// # Зачем
///
/// Per-frame калибровка отвечает на вопрос «какова доля `y=1` среди **всех**
/// кадров с `raw ≥ thr`». В реальной торговле это не тот вопрос: после того
/// как сигнал ушёл выше порога, он держится десятками кадров (модель «думает»
/// о том же рынке), а y-разметка ([`crate::xframe::calc_y_train_pnl`]) пишет
/// `1` только на узком окне TP-горизонта. В сумме per-frame доля позитивных
/// в 2–4 раза ниже, чем фактический win-rate sim'а на тех же данных, и
/// isotonic сжимает `cal(raw≥0.7) → 0.30`, ломая Kelly.
///
/// Здесь мы прогоняем **тот же** [`run_side_simulation`] на val'е (с
/// `is_kelly=false`, фиксированный $30 entry, identity-калибровка) и
/// собираем фактический исход каждой открытой позиции — TP/SL/Timeout/EvExit
/// или Resolution. На этих парах `(raw, won)` PAV даёт честный
/// «raw-скор → реальный win-rate sim'а», а не «raw-скор → доля кадров с y=1».
///
/// # Изоляция от глобального состояния
///
/// На каждый маркет создаётся свежий [`Account`] с очень большим bankroll
/// (чтобы фиксированный $30-entry не урезался по `min(size, bankroll)` после
/// серии трейдов). [`SimStats`] / `positions` / `pending_resolution` тоже
/// per-маркет. CSV-лог трейдов обычно не открыт в момент train_mode
/// ([`crate::trade_csv_log::init_trade_csv_log_file`] дёргается только в
/// [`crate::history_sim::run_sim_mode`]); если открыт — туда попадут лишние
/// строки, в `train_and_history_sim` это не происходит.
/// Hold-zone threshold (sec) для sim-replay калибровки `ModelType::Resolution`.
///
/// Управляет [`compute_p_win_now`] **только в калибровочном прогоне**: production
/// (`run_sim_mode_inner`, `real_sim::tick_once`) продолжает использовать
/// [`HOLD_TO_END_THRESHOLD_SEC`]. Эти константы намеренно разные:
///   * production сейчас фиксирован в `0` — EvExit отключён, resolution-модель
///     в живом sim'е не запрашивается;
///   * для калибровки же resolution на пустом множестве смысла нет, нужно
///     набрать `(raw_resolution, token_won)` точки в реалистичном окне последних
///     30 секунд эвента (где модель в принципе будет применяться, как только
///     production включит EvExit).
const RESOLUTION_CALIBRATION_HOLD_SEC: i64 = 30;

/// Hold-zone threshold (sec) для sim-replay калибровки `ModelType::Pnl`.
/// Жёстко `= 0`: при PnL-калибровке нам нужен «чистый» trade outcome (TP/SL/
/// Timeout/Resolution) без EvExit-выходов, поэтому resolution-ветку
/// глушим, выставляя пустой hold-zone.
const PNL_CALIBRATION_HOLD_SEC: i64 = 0;

/// Sim-replay сбор данных для калибровки одной из двух моделей.
///
/// **Контракт по моделям:**
///
/// * `ModelType::Pnl`: `booster_for_calibration` — сама обучаемая PnL-модель.
///   `pnl_for_entries` игнорируется. `run_side_simulation` запускается в
///   raw-режиме (`is_kelly=false`, identity-калибровка PnL, без resolution).
///   Hold-zone = `PNL_CALIBRATION_HOLD_SEC = 0` (resolution-ветка отключена,
///   EvExit'ы не «съедают» трейды). Возврат — `(raw_pnl_at_open, pnl > 0)` из
///   [`SideStats::closed_trade_entries`].
///
/// * `ModelType::Resolution`: `booster_for_calibration` — обучаемая
///   Resolution-модель. `pnl_for_entries = Some((pnl_booster, pnl_cal))`
///   обязателен — вход в позиции должен идти **через уже откалиброванный
///   PnL-канал в Kelly-режиме**, иначе калибровочные точки берутся из
///   нерепрезентативной популяции кадров (production-сайзинг по Kelly влияет
///   на то, какие маркеты вообще получают позицию). `run_side_simulation`
///   получает `(pnl_booster, Some(pnl_cal), Some(resolution_booster),
///   None /* calibration_resolution */, is_kelly=true)`. Hold-zone =
///   `RESOLUTION_CALIBRATION_HOLD_SEC = 30`. На каждом кадре в hold-zone
///   `compute_p_win_now` (т.к. `calibration_resolution=None`) вернёт **сырой**
///   скор — `run_side_simulation` копит его в
///   [`SideStats::hold_zone_resolution_predictions`]. После прохода каждое
///   значение пар'ится с `token_won` маркета (`up_won` для UP, `!up_won` для
///   DOWN) — единый исход монотонно делит все hold-zone-кадры на «выигрышные»
///   и «проигрышные». PAV сошьёт их в монотонную калибровку.
async fn fit_calibration_via_sim_replay(
    val_paths: &[PathBuf],
    booster_for_calibration: &Booster,
    pnl_for_entries: Option<(&Booster, &Calibration)>,
    side: FrameSide,
    currency: &str,
    interval_kind: XFrameIntervalKind,
    model_type: ModelType,
    tag: &str,
) -> Vec<(f32, bool)> {
    let outcome = side.to_outcome();
    let lane_key = (currency.to_string(), interval_kind, outcome);
    let identity = Calibration::identity();

    // Раскладка ролей моделей и режима sim-replay по `model_type`.
    // `booster_pnl` / `calibration_pnl` идут в PnL-канал `run_side_simulation`,
    // `booster_resolution` — в resolution-канал. Hold-zone определяет окно,
    // в котором собираются точки для калибровки Resolution (см. doc).
    let (booster_pnl, calibration_pnl, booster_resolution, is_kelly, hold_sec) = match model_type {
        ModelType::Pnl => (
            booster_for_calibration,
            &identity,
            None,
            false,
            PNL_CALIBRATION_HOLD_SEC,
        ),
        ModelType::Resolution => {
            let Some((pnl_b, pnl_c)) = pnl_for_entries else {
                tee_eprintln!(
                    "[calibration-sim] {tag}: ModelType::Resolution требует \
                         pnl_for_entries (PnL booster + калибровку для драйва entry); \
                         sim-replay пропущен — будет fallback на per-frame."
                );
                return Vec::new();
            };
            (
                pnl_b,
                pnl_c,
                Some(booster_for_calibration),
                true,
                RESOLUTION_CALIBRATION_HOLD_SEC,
            )
        }
    };

    let mut entries: Vec<(f32, bool)> = Vec::new();

    let mut markets_processed: usize = 0;
    let mut markets_skipped: usize = 0;
    let mut total_frames: usize = 0;

    for path in val_paths {
        let dump = match load_market_xframes(path) {
            Ok(d) => d,
            Err(err) => {
                tee_eprintln!(
                    "[calibration-sim] {tag}: загрузка дампа {} провалена: {err}",
                    path.display()
                );
                markets_skipped += 1;
                continue;
            }
        };
        let frames_vec: Vec<&XFrame<SIZE>> = side.frames(&dump).iter().collect();
        if frames_vec.is_empty() {
            markets_skipped += 1;
            continue;
        }
        total_frames += frames_vec.len();

        // См. [`CALIBRATION_REPLAY_BANKROLL_USD`]: намеренно ≫ INITIAL_BANKROLL,
        // чтобы убрать capacity-фильтр real-bankroll'а из калибровочного сета.
        // Account использует interior mutability (per-field RwLock); пишем
        // через `.write().await` в каноническом порядке (`bankroll → peak_bankroll`).
        // `Arc<Account>` (через `new_shared`) для единообразия с real_sim/PM —
        // `&account` дерефается в `&Account` для всех вызовов ниже.
        let account = Account::new_shared();
        *account.bankroll.write().await = CALIBRATION_REPLAY_BANKROLL_USD;
        *account.peak_bankroll.write().await = CALIBRATION_REPLAY_BANKROLL_USD;
        let mut sim_stats = SimStats::new();

        let event_end_ms =
            window_bounds_from_dump_path(path, interval_kind).map(|b| b.event_end_ms);
        let bin_dump_path = path.to_string_lossy().into_owned();
        let market_id_opt = frames_vec.first().map(|f| f.market_id.clone());
        let up_won = dump.up_won();
        let token_won = match side {
            FrameSide::Up => up_won,
            FrameSide::Down => !up_won,
        };

        {
            let side_stats: &mut SideStats = match side {
                FrameSide::Up => &mut sim_stats.up,
                FrameSide::Down => &mut sim_stats.down,
            };
            run_side_simulation(
                &frames_vec,
                booster_pnl,
                Some(calibration_pnl),
                booster_resolution,
                None, // см. doc: calibration_resolution не передаётся
                &account,
                &lane_key,
                side_stats,
                currency,
                is_kelly,
                "",
                Some(dump.price_to_beat),
                Some(dump.final_price),
                event_end_ms,
                &bin_dump_path,
                hold_sec,
            )
            .await;
        }

        // Хвост позиций, доехавших до конца окна, лежит в `account.pending_resolution`
        // под нашим `lane_key`. resolve_pending_market_sync закрывает их бинарной
        // выплатой и (благодаря патчу в account.rs) push'ит в closed_trade_entries.
        if let Some(market_id) = market_id_opt {
            account
                .resolve_pending_market_sync(
                    &mut sim_stats,
                    currency,
                    interval_kind,
                    &market_id,
                    up_won,
                    None,
                )
                .await;
        }

        let side_stats_ref: &SideStats = match side {
            FrameSide::Up => &sim_stats.up,
            FrameSide::Down => &sim_stats.down,
        };
        match model_type {
            ModelType::Pnl => {
                entries.extend_from_slice(&side_stats_ref.closed_trade_entries);
            }
            ModelType::Resolution => {
                // Все hold-zone кадры одного маркета получают один и тот же
                // `token_won` — резолюция эвента бинарна и общая для всех
                // кадров его hold-окна. Это и есть «правильная» метка для
                // калибровки P(токен_выиграл | признаки кадра).
                entries.extend(
                    side_stats_ref
                        .hold_zone_resolution_predictions
                        .iter()
                        .map(|&p| (p, token_won)),
                );
            }
        }
        markets_processed += 1;
    }

    // Defensive sweep: `close_position` / `Account::resolve_pending_market_sync`
    // выше клали строки в `TRADE_CSV_PENDING` (in-memory буфер). Для каждого
    // маркета `resolve_pending_market_sync` вызывает `record_market_outcome`,
    // который дренирует свой `market_id` (с writer == None строки уходят в
    // drop, см. `trade_csv_log::record_market_outcome`). Но если в каком-то
    // маркете `market_id_opt = None` (пустой dump), `record_market_outcome`
    // не дёрнется и его строки останутся висеть. Чистим буфер до того, как
    // `run_sim_mode` откроет writer и эти orphan-строки попадут в финальный
    // CSV под чужим `regime`.
    crate::trade_csv_log::clear_pending_buffer();

    let n_won = entries.iter().filter(|(_, w)| *w).count();
    let n_lost = entries.len() - n_won;
    let mean_raw_won: f64 = if n_won > 0 {
        entries
            .iter()
            .filter(|(_, w)| *w)
            .map(|(r, _)| *r as f64)
            .sum::<f64>()
            / n_won as f64
    } else {
        0.0
    };
    let mean_raw_lost: f64 = if n_lost > 0 {
        entries
            .iter()
            .filter(|(_, w)| !*w)
            .map(|(r, _)| *r as f64)
            .sum::<f64>()
            / n_lost as f64
    } else {
        0.0
    };
    let win_rate = if !entries.is_empty() {
        n_won as f64 / entries.len() as f64
    } else {
        0.0
    };
    let label = match model_type {
        ModelType::Pnl => "trades",
        ModelType::Resolution => "hold_frames",
    };
    tee_println!(
        "[calibration-sim] {tag} ({model_type:?} hold={hold_sec}s is_kelly={is_kelly}): \
         обработано {markets_processed}/{} маркетов ({} пропущено, {} кадров) | \
         {label}={} won={n_won} lost={n_lost} win_rate={win_rate:.3} \
         mean_raw_won={mean_raw_won:.4} mean_raw_lost={mean_raw_lost:.4}",
        val_paths.len(),
        markets_skipped,
        total_frames,
        entries.len(),
    );

    entries
}

/// Isotonic regression калибровка: подгоняет монотонную неубывающую функцию
/// методом PAV (Pool Adjacent Violators) к парам `(raw_prediction, label)`.
///
/// # Источник данных
///
/// **Основной путь** — sim-replay (см. [`fit_calibration_via_sim_replay`]):
/// прогоняем `run_side_simulation` на val-сплите. Для PnL-модели — в
/// raw-режиме с identity-калибровкой и собираем `(raw_pred_at_open, pnl > 0)`
/// из закрытых трейдов. Для Resolution-модели — в Kelly-режиме с **уже
/// откалиброванным PnL** (передаётся через `pnl_for_entries`), без
/// `calibration_resolution`, и собираем `(raw_resolution, token_won)` на
/// каждом hold-zone-кадре. Подробности — в doc к
/// [`fit_calibration_via_sim_replay`].
///
/// **Fallback** — full per-frame `(preds, y)` (как было до перехода на
/// sim-replay). Срабатывает, если sim-replay набрал меньше
/// [`CALIBRATION_MIN_FILTERED_SAMPLES`] точек или один из классов пуст
/// (например, в test-сплите модель не открывает позиций при
/// `SIM_BUY_THRESHOLD` — диагностический сигнал, что AUC высокий за счёт
/// «низких» кадров, а не сигналов на покупку). Per-frame fallback
/// корректен и для Resolution: `MarketDataset::y` для Resolution-модели —
/// это `calc_y_train_resolution` (тот же `token_won`, только размечается
/// плотно по всем кадрам, а не только по hold-zone).
///
/// Печатает диагностику обоих сетов (per-frame vs sim-replay) — позволяет
/// сравнить distribution shift «глазами» и зафиксировать, какой набор
/// фактически попал в PAV.
#[allow(clippy::too_many_arguments)]
async fn fit_calibration(
    booster: &Booster,
    dmat: &DMatrix,
    val_markets: &[MarketDataset],
    val_paths: &[PathBuf],
    currency: &str,
    interval_kind: XFrameIntervalKind,
    side: FrameSide,
    model_type: ModelType,
    pnl_for_entries: Option<(&Booster, &Calibration)>,
    tag: &str,
) -> anyhow::Result<Calibration> {
    let preds = booster.predict(dmat)?;
    let y: Vec<f32> = val_markets
        .iter()
        .flat_map(|m| m.y.iter().copied())
        .collect();
    debug_assert_eq!(preds.len(), y.len());

    // ── Диагностика на полном per-frame сете (для сравнения) ─────────────
    let n_pos_full = y.iter().filter(|&&v| v >= 1.0).count();
    let n_neg_full = y.len() - n_pos_full;
    let mean_pred_pos_full: f64 = preds
        .iter()
        .zip(y.iter())
        .filter(|(_, yv)| **yv >= 1.0)
        .map(|(&p, _)| p as f64)
        .sum::<f64>()
        / n_pos_full.max(1) as f64;
    let mean_pred_neg_full: f64 = preds
        .iter()
        .zip(y.iter())
        .filter(|(_, yv)| **yv < 1.0)
        .map(|(&p, _)| p as f64)
        .sum::<f64>()
        / n_neg_full.max(1) as f64;
    let cal_auc_full = calc_auc(&preds, &y);
    tee_println!(
        "[calibration] {tag}: full per-frame: n_pos={n_pos_full} n_neg={n_neg_full} \
         mean_pred_pos={mean_pred_pos_full:.4} mean_pred_neg={mean_pred_neg_full:.4} AUC={cal_auc_full:.4}"
    );

    if cal_auc_full < CALIBRATION_MIN_AUC {
        tee_eprintln!(
            "[calibration] {tag}: AUC={cal_auc_full:.4} < {CALIBRATION_MIN_AUC} — модель слишком \
             слабая для калибровки. Используется identity."
        );
        return Ok(Calibration::identity());
    }

    // ── Sim-replay (основной путь) ───────────────────────────────────────
    let entries = fit_calibration_via_sim_replay(
        val_paths,
        booster,
        pnl_for_entries,
        side,
        currency,
        interval_kind,
        model_type,
        tag,
    )
    .await;
    let preds_sim: Vec<f32> = entries.iter().map(|(r, _)| *r).collect();
    let y_sim: Vec<f32> = entries
        .iter()
        .map(|(_, w)| if *w { 1.0 } else { 0.0 })
        .collect();
    let n_pos_sim = entries.iter().filter(|(_, w)| *w).count();
    let n_neg_sim = entries.len() - n_pos_sim;

    // Решаем какой набор кормить в PAV.
    let (cal_preds, cal_y, source_label): (&[f32], &[f32], &'static str) = if entries.len()
        >= CALIBRATION_MIN_FILTERED_SAMPLES
        && n_pos_sim > 0
        && n_neg_sim > 0
    {
        (preds_sim.as_slice(), y_sim.as_slice(), "sim-replay")
    } else {
        tee_eprintln!(
            "[calibration] {tag}: sim-replay набор слишком мал ({} < {CALIBRATION_MIN_FILTERED_SAMPLES}) \
                 или один класс пуст (won={n_pos_sim} lost={n_neg_sim}) — fallback на per-frame.",
            entries.len(),
        );
        if n_pos_full == 0 || n_neg_full == 0 {
            tee_eprintln!(
                "[calibration] {tag}: per-frame набор тоже без двух классов \
                     (n_pos={n_pos_full}, n_neg={n_neg_full}). Используется identity."
            );
            return Ok(Calibration::identity());
        }
        (preds.as_slice(), y.as_slice(), "per-frame")
    };

    let cal = isotonic_fit(cal_preds, cal_y);
    tee_println!(
        "[calibration] {tag}: fit OK ({source_label}) | breakpoints={} | \
         range=[{:.3}…{:.3}] → [{:.3}…{:.3}]",
        cal.xs.len(),
        cal.xs.first().copied().unwrap_or(0.0),
        cal.xs.last().copied().unwrap_or(0.0),
        cal.ys.first().copied().unwrap_or(0.0),
        cal.ys.last().copied().unwrap_or(0.0),
    );

    Ok(cal)
}

/// Ядро isotonic regression: алгоритм PAV (Pool Adjacent Violators).
///
/// # Алгоритм
///
/// 1. Сортируем пары `(raw, label)` по `raw` (asc).
/// 2. Предагрегируем точки с одинаковым `raw` в один начальный блок
///    (сумма меток, суммарный вес).
/// 3. Проходим слева направо, добавляя блоки в стек. Если текущая вершина
///    стека имеет среднее `> ` значение нового блока — это нарушение
///    монотонности, сливаем блоки (взвешенное среднее) и повторяем проверку
///    с новой вершиной. После обработки всех блоков стек содержит неубывающую
///    последовательность.
/// 4. Для каждого блока вычисляем опорный узел: `x = взвешенное среднее raw`,
///    `y = доля позитивных в блоке`, клиппим `y` в `[eps, 1 − eps]`.
///
/// Сложность: `O(N log N)` сортировка + `O(N)` PAV (амортизированно).
fn isotonic_fit(preds: &[f32], y: &[f32]) -> Calibration {
    debug_assert_eq!(preds.len(), y.len());
    if preds.is_empty() {
        return Calibration::identity();
    }

    let mut pairs: Vec<(f32, f32)> = preds
        .iter()
        .zip(y.iter())
        .map(|(&p, &yv)| (p, if yv >= 1.0 { 1.0_f32 } else { 0.0_f32 }))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    #[derive(Clone, Copy)]
    struct Block {
        sum_x: f64,
        sum_y: f64,
        weight: f64,
    }
    impl Block {
        fn value(&self) -> f64 {
            self.sum_y / self.weight
        }
    }

    // Шаг 1: предагрегация точек с идентичным raw в один блок.
    // Это важно: иначе PAV сохранит два блока с одинаковым x,
    // но разными y, создавая дубликаты опорных узлов.
    let mut blocks: Vec<Block> = Vec::with_capacity(pairs.len());
    for &(x, y_i) in &pairs {
        if let Some(last) = blocks.last_mut() {
            let prev_x = last.sum_x / last.weight;
            if (prev_x - x as f64).abs() < 1e-12 {
                last.sum_x += x as f64;
                last.sum_y += y_i as f64;
                last.weight += 1.0;
                continue;
            }
        }
        blocks.push(Block {
            sum_x: x as f64,
            sum_y: y_i as f64,
            weight: 1.0,
        });
    }

    // Шаг 2: собственно PAV — стек блоков с неубывающими значениями.
    let mut stack: Vec<Block> = Vec::with_capacity(blocks.len());
    for block in blocks {
        let mut new_block = block;
        while let Some(&top) = stack.last() {
            if top.value() <= new_block.value() {
                break;
            }
            stack.pop();
            new_block = Block {
                sum_x: top.sum_x + new_block.sum_x,
                sum_y: top.sum_y + new_block.sum_y,
                weight: top.weight + new_block.weight,
            };
        }
        stack.push(new_block);
    }

    // Шаг 3: регуляризация — последовательно аккумулируем блоки в «bucket»,
    // пока суммарный вес не достигнет min_weight. Порог адаптивный:
    // `min(CAP, n/5)` — на маленьком sim-replay сете (≪ 5·CAP) фиксированный
    // CAP=50 склеивал всё в один bucket и калибровка вырождалась в константу.
    // n/5 даёт ≈5 ступенек при любом N; на больших сетах кап ограничивает сверху.
    // Монотонность сохраняется: если v_1 ≤ … ≤ v_k и v_{k+1} ≤ … ≤ v_m,
    // то weighted_avg(v_1..v_k) ≤ v_k ≤ v_{k+1} ≤ weighted_avg(v_{k+1}..v_m).
    let min_weight = (preds.len() as f64 / 5.0).min(CALIBRATION_MIN_BLOCK_WEIGHT_CAP);
    if !stack.is_empty() && min_weight > 1.0 {
        let mut regularized: Vec<Block> = Vec::with_capacity(stack.len());
        let mut acc: Option<Block> = None;
        for block in stack.drain(..) {
            let merged = match acc.take() {
                Some(a) => Block {
                    sum_x: a.sum_x + block.sum_x,
                    sum_y: a.sum_y + block.sum_y,
                    weight: a.weight + block.weight,
                },
                None => block,
            };
            if merged.weight >= min_weight {
                regularized.push(merged);
            } else {
                acc = Some(merged);
            }
        }
        if let Some(a) = acc {
            // Хвост с недобранным весом: сливаем в предыдущий bucket (если есть),
            // иначе сохраняем как единственный блок.
            if let Some(last) = regularized.last_mut() {
                last.sum_x += a.sum_x;
                last.sum_y += a.sum_y;
                last.weight += a.weight;
            } else {
                regularized.push(a);
            }
        }
        stack = regularized;
    }

    let eps = CALIBRATION_EPS;
    let mut xs: Vec<f32> = Vec::with_capacity(stack.len());
    let mut ys: Vec<f32> = Vec::with_capacity(stack.len());
    for b in &stack {
        let x = (b.sum_x / b.weight) as f32;
        let y_val = ((b.sum_y / b.weight) as f32).clamp(eps, 1.0 - eps);
        // Защита от численных дубликатов x (если всё же проскочили):
        // оставляем только строго возрастающие узлы.
        if let Some(&prev_x) = xs.last() {
            if x <= prev_x {
                continue;
            }
        }
        xs.push(x);
        ys.push(y_val);
    }

    if xs.is_empty() {
        return Calibration::identity();
    }
    Calibration { xs, ys }
}

/// Сохраняет калибровку рядом с моделью: `model_path` → `model_path.calibration.bin`.
fn save_calibration(cal: &Calibration, model_path: &Path) -> anyhow::Result<PathBuf> {
    let cal_path = calibration_path(model_path);
    if let Some(parent) = cal_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(&cal_path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, cal)?;
    Ok(cal_path)
}

/// Загружает калибровку из файла рядом с моделью.
pub fn load_calibration(model_path: &Path) -> anyhow::Result<Calibration> {
    let cal_path = calibration_path(model_path);
    let file = fs::File::open(&cal_path)?;
    let reader = BufReader::new(file);
    Ok(bincode::deserialize_from(reader)?)
}

/// Путь к файлу калибровки для данной модели.
pub fn calibration_path(model_path: &Path) -> PathBuf {
    let mut p = model_path.as_os_str().to_owned();
    p.push(".calibration.bin");
    PathBuf::from(p)
}

// ─── XGBoost параметры ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct XgbParams {
    eta: f32,
    max_depth: u32,
    min_child_weight: f32,
    gamma: f32,
    subsample: f32,
    colsample_bytree: f32,
    lambda: f32,
    alpha: f32,
    scale_pos_weight: f32,
}

/// Нога токена: обучение ведётся раздельно по Up и Down фреймам.
#[derive(Debug, Clone, Copy)]
enum FrameSide {
    Up,
    Down,
}

impl FrameSide {
    fn label(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    fn frames<'a>(&self, dump: &'a MarketXFramesDump) -> &'a [XFrame<SIZE>] {
        match self {
            Self::Up => &dump.frames_up,
            Self::Down => &dump.frames_down,
        }
    }

    /// Маппинг на лейн-ключ симулятора: одна и та же сторона
    /// в обоих модулях (UP-токен / DOWN-токен).
    fn to_outcome(self) -> CurrencyUpDownOutcome {
        match self {
            Self::Up => CurrencyUpDownOutcome::Up,
            Self::Down => CurrencyUpDownOutcome::Down,
        }
    }
}

/// Тип модели: определяет какую y-метку использовать при обучении.
#[derive(Debug, Clone, Copy)]
enum ModelType {
    /// PnL-метка ([`calc_y_train_pnl`]): бинарная, учитывает комиссии, TP/SL.
    /// Обучается только для step=1s.
    Pnl,
    /// Resolution-метка ([`calc_y_train_resolution`]): бинарная по исходу события.
    /// Обучается для всех step-интервалов.
    Resolution,
}

impl ModelType {
    fn label(self) -> &'static str {
        match self {
            Self::Pnl => "pnl",
            Self::Resolution => "resolution",
        }
    }
}

/// Точка входа в режим обучения. Ищет валюты в `xframes/`, для каждой версии
/// обучает модели по всем комбинациям interval × step × model_type × side.
///
/// `async fn`, потому что калибровочный sim-replay внутри `train_all_variants` →
/// `train_and_save` → `fit_calibration` → `fit_calibration_via_sim_replay`
/// дёргает `Account::*`-API под `.write().await` per-field RwLock'ов.
pub async fn run_train_mode() -> anyhow::Result<()> {
    let xframes_root = Path::new("xframes");
    if !xframes_root.exists() {
        anyhow::bail!("Папка xframes/ не найдена — сначала запустите сбор данных (STATUS=default)");
    }

    let log_path = xframes_root.join("last_train_mode.txt");
    {
        let file = fs::File::create(&log_path)?;
        let mut guard = TEE_LOG.lock().expect("TEE_LOG poisoned");
        *guard = Some(BufWriter::new(file));
    }
    tee_println!("[train] лог пишется в {}", log_path.display());

    for currency_path in fs_read_dirs(xframes_root)? {
        if !currency_path.is_dir() {
            continue;
        }
        let currency = currency_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        for version_path in fs_read_dirs(&currency_path)? {
            if !version_path.is_dir() {
                continue;
            }
            let version_str = version_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            // Подпапка версии — число (количество признаков); пропускаем прочее.
            if version_str.parse::<usize>().is_err() {
                continue;
            }

            for interval in ["5m", "15m"] {
                let interval_path = version_path.join(interval);
                if !interval_path.is_dir() {
                    continue;
                }

                // Лейбл "5m"/"15m" → enum XFrameIntervalKind: один источник истины
                // с лейном sim'а в [`fit_calibration_via_sim_replay`] (через
                // `lane_key = (currency, interval_kind, outcome)`).
                let interval_kind = match interval {
                    "5m" => XFrameIntervalKind::FiveMin,
                    "15m" => XFrameIntervalKind::FifteenMin,
                    other => {
                        tee_eprintln!(
                            "[train] {currency}/{version_str}/{other}: неизвестный interval, пропуск"
                        );
                        continue;
                    }
                };

                for &step_sec in &FRAME_BUILD_INTERVALS_SEC {
                    let step_path = interval_path.join(format!("{step_sec}s"));
                    if !step_path.is_dir() {
                        continue;
                    }

                    let tag_prefix = format!("{currency}/{version_str}/{interval}/{step_sec}s");
                    tee_println!("[train] {tag_prefix}: сбор путей...");
                    let paths = collect_bin_paths(&step_path)?;
                    if paths.is_empty() {
                        tee_println!("[train] {tag_prefix}: нет маркетов, пропуск");
                        continue;
                    }

                    // Сплит по путям — идентично history_sim; дампы загружаем
                    // только после сплита, чтобы битые/пустые маркеты не сдвигали границы.
                    let (train_count, val_count, test_count) = split_counts(paths.len());
                    let train_paths = &paths[..train_count];
                    let val_paths = &paths[train_count..train_count + val_count];
                    let test_paths = &paths[train_count + val_count..];
                    tee_println!(
                        "[train] {tag_prefix}: маркетов {} → сплит {train_count} train / {val_count} val / {test_count} test",
                        paths.len(),
                    );

                    train_all_variants(
                        train_paths,
                        val_paths,
                        test_paths,
                        &version_path,
                        &tag_prefix,
                        &currency,
                        interval,
                        interval_kind,
                        step_sec,
                    )
                    .await?;
                }
            }
        }
    }

    {
        use std::io::Write;
        let mut guard = TEE_LOG.lock().expect("TEE_LOG poisoned");
        if let Some(mut w) = guard.take() {
            let _ = w.flush();
        }
    }

    Ok(())
}

/// Данные одного маркета (дамп-файла): признаки и метки.
struct MarketDataset {
    x: Vec<f32>,
    y: Vec<f32>,
}

/// Диагностика разметки кадров: сколько кадров пришло на вход, сколько было
/// размечено, и распределение причин отказа от разметки. Заполняется в
/// [`append_frames`] и агрегируется в [`build_market_datasets`].
///
/// # Зачем
///
/// `calc_y_train_pnl` / `calc_y_train_resolution` могут возвращать `None` для
/// нескольких принципиально разных причин («тонкий стакан, не открыли позицию
/// на $200 nominal», «slippage-cap зарезал вход», «не определена
/// `currency_implied_prob`», «за горизонт ничего не случилось» — в зависимости
/// от того, какая версия `y_train` активна). Без счётчика этих причин нельзя
/// понять: упало ли распределение y из-за переключения функции, или из-за
/// смены данных. Принт идёт строкой в `last_train_mode.txt`, рядом с
/// `[train] {tag}: распределение y …`.
#[derive(Debug, Default, Clone, Copy)]
struct AppendStats {
    /// Сколько кадров пришло на вход [`append_frames`] (до фильтров).
    total_frames: usize,
    /// Сколько кадров реально попали в обучающий вектор (`y_out`).
    marked: usize,
    /// Resolution-only: кадры вне hold-zone (`event_remaining_ms <= 0` или
    /// `> HOLD_TO_END_THRESHOLD_SEC * 1000`).
    out_of_hold_zone: usize,
    /// Pnl-only: кадры со слишком малым остатком до резолюции
    /// (`event_remaining_ms < MIN_ENTRY_REMAINING_MS`, включая `<= 0`).
    /// Совпадает с ранним отказом [`crate::history_sim::buy_gate`]
    /// `BuyGate::LateEntry`: модель PnL не должна учиться на тиках,
    /// где исполнитель в любом случае не открывает позицию.
    late_entry: usize,
    /// `calc_y_*` вернул `None` (тонкий стакан / slippage cap / нет
    /// `currency_implied_prob` / прочие причины внутри Y-функции).
    y_none: usize,
    /// Длина вектора признаков не совпала с ожидаемой `feature_count`
    /// (теоретически — несконсистентный layout, на практике 0).
    feature_mismatch: usize,
}

impl AppendStats {
    fn merge(&mut self, other: &AppendStats) {
        self.total_frames += other.total_frames;
        self.marked += other.marked;
        self.out_of_hold_zone += other.out_of_hold_zone;
        self.late_entry += other.late_entry;
        self.y_none += other.y_none;
        self.feature_mismatch += other.feature_mismatch;
    }
}

/// Обучает модели для всех комбинаций `model_type × side` на одном
/// `(currency, version, interval, step_sec)`. Каждая комбинация даёт отдельный
/// файл `model_{interval}_{step_sec}s_{model_type}_{side}.ubj` — формат,
/// который грузит `history_sim`.
async fn train_all_variants(
    train_paths: &[PathBuf],
    val_paths: &[PathBuf],
    test_paths: &[PathBuf],
    version_path: &Path,
    tag_prefix: &str,
    currency: &str,
    interval: &str,
    interval_kind: XFrameIntervalKind,
    step_sec: u64,
) -> anyhow::Result<()> {
    for model_type in [ModelType::Pnl, ModelType::Resolution] {
        // Pnl и Resolution обучаем только на step_sec = 1 с: лейблы обеих
        // моделей считаются через [`crate::xframe::calc_y_train_pnl`] /
        // [`crate::xframe::calc_y_train_resolution`] по горизонту
        // [`Y_TRAIN_HORIZON_FRAMES`] кадров, который на 1s-шаге даёт
        // осмысленные 15 с; на 2s/4s тот же горизонт превращается в 30/60 с
        // и семантика меняется, а `history_sim` всё равно использует
        // только 1s-модели.
        if step_sec != 1 {
            continue;
        }

        for side in [FrameSide::Up, FrameSide::Down] {
            let tag = format!("{tag_prefix}/{}/{}", model_type.label(), side.label());

            let max_lag = match model_type {
                ModelType::Resolution => RESOLUTION_MAX_LAG,
                ModelType::Pnl => PNL_MAX_LAG,
            };

            let (train_markets, train_stats) =
                build_market_datasets(train_paths, side, model_type, max_lag);
            let (val_markets, val_stats) =
                build_market_datasets(val_paths, side, model_type, max_lag);
            let (test_markets, test_stats) =
                build_market_datasets(test_paths, side, model_type, max_lag);

            let total_markets = train_markets.len() + val_markets.len() + test_markets.len();
            if total_markets == 0 {
                tee_println!("[train] {tag}: нет данных, пропуск");
                continue;
            }

            let feature_count = match max_lag {
                Some(n) => XFrame::<SIZE>::count_features_n(n),
                None => XFrame::<SIZE>::count_features(),
            };
            let total_rows: usize = train_markets
                .iter()
                .chain(val_markets.iter())
                .chain(test_markets.iter())
                .map(|m| m.y.len())
                .sum();
            tee_println!(
                "[train] {tag}: маркетов {}/{}/{} (train/val/test), {} строк, {} признаков",
                train_markets.len(),
                val_markets.len(),
                test_markets.len(),
                total_rows,
                feature_count,
            );

            // Воронка разметки: сколько кадров отвалилось до попадания в y.
            // Печатаем на каждом сплите отдельно, чтобы увидеть, не уехал ли,
            // например, `y_none` в test (например, из-за переключения y_train
            // на версию с walk-обходом — на тонком стакане `None` будет чаще).
            print_append_stats(&tag, "train", &train_stats);
            print_append_stats(&tag, "val", &val_stats);
            print_append_stats(&tag, "test", &test_stats);

            let model_path = version_path.join(format!(
                "model_{interval}_{step_sec}s_{}_{}.ubj",
                model_type.label(),
                side.label(),
            ));

            // Для Resolution той же стороны рядом уже должен лежать сохранённый
            // PnL: внешний цикл идёт `[Pnl, Resolution]`, внутренний — `[Up, Down]`.
            // К моменту, когда обучаем `(Resolution, Up)`, файл
            // `model_{interval}_1s_pnl_up.ubj` (+ `.calibration.bin`) уже на
            // диске; аналогично для `Down`. Если по какой-то причине его нет
            // (ошибка/skip предыдущей итерации) — `train_and_save` опустится
            // в fallback на per-frame калибровку.
            let pnl_model_path: Option<PathBuf> = match model_type {
                ModelType::Pnl => None,
                ModelType::Resolution => Some(version_path.join(format!(
                    "model_{interval}_{step_sec}s_{}_{}.ubj",
                    ModelType::Pnl.label(),
                    side.label(),
                ))),
            };

            match train_and_save(
                &train_markets,
                &val_markets,
                &test_markets,
                val_paths,
                currency,
                interval_kind,
                side,
                &model_path,
                pnl_model_path.as_deref(),
                &tag,
                model_type,
                max_lag,
            )
            .await
            {
                Ok(()) => {
                    tee_println!("[train] {tag}: модель сохранена → {}", model_path.display())
                }
                Err(err) => tee_eprintln!("[train] {tag}: ошибка обучения: {err:#}"),
            }
        }
    }
    Ok(())
}

/// Собирает все `.bin` файлы из `step_path/{date}/` в хронологическом порядке
/// (по имени пути). Единственный источник истины для порядка маркетов —
/// используется и тренером, и симулятором.
pub fn collect_bin_paths(step_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if !step_path.is_dir() {
        return Ok(paths);
    }
    for date_path in fs_read_dirs(step_path)? {
        if !date_path.is_dir() {
            continue;
        }
        for file_path in fs_read_dirs(&date_path)? {
            if file_path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
                paths.push(file_path);
            }
        }
    }
    Ok(paths)
}

/// Хронологический 3-way сплит по количеству маркетов.
/// Возвращает `(train_count, val_count, test_count)` так, что
/// `train_count + val_count + test_count == n`. Границы считаются **по путям**,
/// идентично в тренере и симуляторе — одни и те же маркеты всегда
/// попадают в один и тот же сплит.
pub fn split_counts(n: usize) -> (usize, usize, usize) {
    let test_count = ((n as f64) * TEST_FRACTION).ceil() as usize;
    let val_count_raw = ((n as f64) * VAL_FRACTION).ceil() as usize;
    let train_count = n.saturating_sub(test_count + val_count_raw);
    let val_count = val_count_raw.min(n.saturating_sub(train_count));
    let test_count = n - train_count - val_count;
    (train_count, val_count, test_count)
}

/// Одна попытка прочитать и десериализовать дамп по пути. При ошибке печатает в лог и возвращает `None`
/// (как раньше в батч-загрузке). Не держит все маркеты в памяти: вызывать по одному пути в [`build_market_datasets`].
fn try_load_dump_from_path(path: &Path) -> Option<MarketXFramesDump> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(err) => {
            tee_eprintln!("[train] не удалось прочитать {}: {err}", path.display());
            return None;
        }
    };
    match bincode::deserialize::<MarketXFramesDump>(&bytes) {
        Ok(dump) => Some(dump),
        Err(err) => {
            tee_eprintln!("[train] ошибка десериализации {}: {err}", path.display());
            None
        }
    }
}

/// Формирует `MarketDataset` для каждого файла по заданной ноге и типу модели.
/// Дампы читаются **по одному** — после разметки кадры (`frames_up`/`frames_down`) снимаются с памяти.
/// `max_lag` — если `Some(n)`, лаговые массивы обрезаются до первых `n` элементов.
///
/// Возвращает агрегированный [`AppendStats`] по всем дампам — для печати
/// диагностики «сколько кадров отвалилось до разметки». См. [`AppendStats`].
fn build_market_datasets(
    paths: &[PathBuf],
    side: FrameSide,
    model_type: ModelType,
    max_lag: Option<usize>,
) -> (Vec<MarketDataset>, AppendStats) {
    let feature_count = match max_lag {
        Some(n) => XFrame::<SIZE>::count_features_n(n),
        None => XFrame::<SIZE>::count_features(),
    };
    let mut markets = Vec::new();
    let mut total_stats = AppendStats::default();

    for path in paths {
        let Some(dump) = try_load_dump_from_path(path) else {
            continue;
        };
        let mut x = Vec::new();
        let mut y = Vec::new();
        let stats = append_frames(
            side.frames(&dump),
            feature_count,
            model_type,
            dump.price_to_beat,
            dump.final_price,
            max_lag,
            &mut x,
            &mut y,
        );
        total_stats.merge(&stats);
        if !y.is_empty() {
            markets.push(MarketDataset { x, y });
        }
    }

    (markets, total_stats)
}

/// Для каждого кадра в `frames` вычисляет метку по `model_type` и, если она есть,
/// добавляет признаки и метку в `x_out` / `y_out`.
///
/// Возвращает [`AppendStats`] со счётчиками всех ветвей отказа: вне hold-zone
/// (только Resolution), `calc_y_*` вернул `None`, и mismatch размера фич.
fn append_frames(
    frames: &[XFrame<SIZE>],
    feature_count: usize,
    model_type: ModelType,
    price_to_beat: f64,
    final_price: f64,
    max_lag: Option<usize>,
    x_out: &mut Vec<f32>,
    y_out: &mut Vec<f32>,
) -> AppendStats {
    // Граница hold zone в мс (условие идентично [`crate::history_sim::manage_positions`]:
    // `event_remaining_ms > 0 && event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000`).
    // Resolution-модель используется исключительно внутри hold zone, поэтому и
    // обучаем её только на кадрах этого диапазона — обучающее распределение
    // совпадает с инференс-распределением.
    let hold_zone_max_ms: i64 = HOLD_TO_END_THRESHOLD_SEC * 1000;

    let mut stats = AppendStats::default();
    stats.total_frames = frames.len();

    for index in 0..frames.len() {
        let remaining = frames[index].event_remaining_ms;
        match model_type {
            ModelType::Resolution => {
                if remaining <= 0 || remaining > hold_zone_max_ms {
                    stats.out_of_hold_zone += 1;
                    continue;
                }
            }
            ModelType::Pnl => {
                // Симметрично `buy_gate::LateEntry` в `history_sim`: при
                // `event_remaining_ms < MIN_ENTRY_REMAINING_MS` исполнитель
                // не открывает позицию — учить PnL-модель на этих кадрах
                // тоже нечему (распределение разъезжается с инференсом).
                if remaining < MIN_ENTRY_REMAINING_MS {
                    stats.late_entry += 1;
                    continue;
                }
            }
        }

        let label = match model_type {
            ModelType::Pnl => calc_y_train_pnl(
                Y_TRAIN_HORIZON_FRAMES,
                frames,
                index,
                price_to_beat,
                final_price,
                Y_TRAIN_MAX_SLIPPAGE_FROM_L1_PCT,
            ),
            ModelType::Resolution => calc_y_train_resolution(
                Y_TRAIN_HORIZON_FRAMES,
                frames,
                index,
                price_to_beat,
                final_price,
                Y_TRAIN_MAX_SLIPPAGE_FROM_L1_PCT,
            ),
        };
        let Some(label) = label else {
            stats.y_none += 1;
            continue;
        };
        let row = match max_lag {
            Some(n) => frames[index].to_x_train_n_with(n, apply_side_symmetry),
            None => frames[index].to_x_train_with(apply_side_symmetry),
        };
        if row.len() != feature_count {
            stats.feature_mismatch += 1;
            continue;
        }
        x_out.extend_from_slice(&row);
        y_out.push(label);
        stats.marked += 1;
    }

    stats
}

/// Сливает список маркет-датасетов в один плоский `(x, y)`.
fn flatten_markets(markets: &[MarketDataset]) -> (Vec<f32>, Vec<f32>) {
    let total_x: usize = markets.iter().map(|m| m.x.len()).sum();
    let total_y: usize = markets.iter().map(|m| m.y.len()).sum();
    let mut x = Vec::with_capacity(total_x);
    let mut y = Vec::with_capacity(total_y);
    for m in markets {
        x.extend_from_slice(&m.x);
        y.extend_from_slice(&m.y);
    }
    (x, y)
}

/// Обучение на уже расщеплённых по сплитам маркетах.
///
/// Сплит выполнен на уровне путей в [`run_train_mode`] и идентичен тому,
/// что использует [`crate::history_sim`] — один и тот же маркет всегда
/// попадает в один и тот же сплит.
/// - **val** — используется optimizer'ом для подбора гиперпараметров и early stopping.
/// - **test** — held-out, только для финальной честной оценки AUC.
#[allow(clippy::too_many_arguments)]
async fn train_and_save(
    train_markets: &[MarketDataset],
    val_markets: &[MarketDataset],
    test_markets: &[MarketDataset],
    val_paths: &[PathBuf],
    currency: &str,
    interval_kind: XFrameIntervalKind,
    side: FrameSide,
    model_path: &Path,
    pnl_model_path: Option<&Path>,
    tag: &str,
    model_type: ModelType,
    max_lag: Option<usize>,
) -> anyhow::Result<()> {
    let (x_train, y_train) = flatten_markets(train_markets);
    let (x_val, y_val) = flatten_markets(val_markets);
    let (x_test, y_test) = flatten_markets(test_markets);

    let total_rows = y_train.len() + y_val.len() + y_test.len();
    if total_rows == 0 {
        anyhow::bail!("датасет пуст, пропуск");
    }

    let mut all_y = y_train.iter().chain(y_val.iter()).chain(y_test.iter());
    let has_pos = all_y.clone().any(|&v| v > 0.0);
    let has_neg = all_y.any(|&v| v <= 0.0);
    if !has_pos || !has_neg {
        anyhow::bail!("датасет содержит только один класс (AUC невозможен), пропуск");
    }

    let mut dtrain = DMatrix::from_dense(&x_train, y_train.len())?;
    dtrain.set_labels(&y_train)?;
    let mut dval = DMatrix::from_dense(&x_val, y_val.len())?;
    dval.set_labels(&y_val)?;
    let mut dtest = DMatrix::from_dense(&x_test, y_test.len())?;
    dtest.set_labels(&y_test)?;

    let feature_count = x_train.len() / y_train.len();
    let fw = build_feature_weights(feature_count, max_lag);
    dtrain.set_feature_weights(&fw)?;
    dval.set_feature_weights(&fw)?;

    // Optimizer и early stopping работают на val (названа "test" для совместимости с eval_xgboost).
    let eval_sets: [(&DMatrix, &str); 2] = [(&dtrain, "train"), (&dval, "test")];

    let optimizer_trials = match model_type {
        ModelType::Pnl => OPTIMIZER_TRIALS_PNL,
        ModelType::Resolution => OPTIMIZER_TRIALS_RESOLUTION,
    };
    match model_type {
        ModelType::Pnl => tee_println!(
            "[train] {tag}: оптимизация гиперпараметров по AUC на val ({optimizer_trials} итераций, TP={Y_TRAIN_TAKE_PROFIT_PP}, SL={Y_TRAIN_STOP_LOSS_PP})…"
        ),
        ModelType::Resolution => tee_println!(
            "[train] {tag}: оптимизация гиперпараметров по AUC на val ({optimizer_trials} итераций)…"
        ),
    }
    let params = tune_xgboost_optimizer(&eval_sets, &dtrain, optimizer_trials, tag)?;
    tee_println!("[train] {tag}: лучшие параметры: {params:?}");

    let booster = fit_booster_with_early_stopping(&params, &dtrain, &dval, tag)?;

    // Метрики на val (из early stopping)
    print_eval_metrics(&booster, tag, "val");

    // Финальная честная оценка на held-out test (AUC считаем вручную,
    // т.к. booster после load_buffer теряет конфигурацию eval_metrics).
    let test_preds = booster.predict(&dtest)?;
    let test_auc = calc_auc(&test_preds, &y_test);
    let test_logloss = calc_logloss(&test_preds, &y_test);
    tee_println!("[train] {tag}: held-out test: logloss={test_logloss:.5}  AUC={test_auc:.6}");

    print_y_distribution(&y_train, &y_val, &y_test, tag);
    print_contributions(&booster, &dtest, tag, max_lag);

    // ── Isotonic regression: калибровка на VAL set ───────────────────────────
    // Val уже «запачкан» early stopping'ом, но это лучше чем калибровать на test:
    // test обязан оставаться полностью held-out для честной финальной оценки AUC.
    // Кроме того, isotonic имеет O(N) параметров и катастрофически переобучается
    // если калибровочный сет совпадает с тем, по которому меряется AUC.

    // Для Resolution-модели подгружаем PnL `Booster` + `Calibration` той же
    // стороны (записаны на диск более ранней итерацией внешнего цикла
    // `train_all_variants`). Они нужны, чтобы entry в sim-replay шёл через
    // production-эквивалентный канал (PnL Kelly), а не через сырые скоры
    // самой Resolution-модели. Держим `Option`-обёртки локально, чтобы
    // ссылки внутри `pnl_for_entries` жили до конца вызова `fit_calibration`.
    let (loaded_pnl_booster, loaded_pnl_cal): (Option<Booster>, Option<Calibration>) =
        if matches!(model_type, ModelType::Resolution) {
            match pnl_model_path {
                Some(p) => {
                    let b = crate::history_sim::load_booster(p);
                    let c = load_calibration(p).ok();
                    if b.is_none() {
                        tee_eprintln!(
                            "[train] {tag}: PnL booster для sim-replay калибровки Resolution \
                             не загрузился (путь {}); fit_calibration_via_sim_replay упадёт в \
                             fallback на per-frame.",
                            p.display(),
                        );
                    }
                    if c.is_none() {
                        tee_eprintln!(
                            "[train] {tag}: PnL calibration для sim-replay калибровки Resolution \
                             не загрузилась (путь {}.calibration.bin); \
                             fit_calibration_via_sim_replay упадёт в fallback на per-frame.",
                            p.display(),
                        );
                    }
                    (b, c)
                }
                None => {
                    tee_eprintln!(
                        "[train] {tag}: ModelType::Resolution без `pnl_model_path` — \
                         sim-replay калибровка пропущена, ждём fallback на per-frame."
                    );
                    (None, None)
                }
            }
        } else {
            (None, None)
        };
    let pnl_for_entries: Option<(&Booster, &Calibration)> =
        match (loaded_pnl_booster.as_ref(), loaded_pnl_cal.as_ref()) {
            (Some(b), Some(c)) => Some((b, c)),
            _ => None,
        };

    match fit_calibration(
        &booster,
        &dval,
        val_markets,
        val_paths,
        currency,
        interval_kind,
        side,
        model_type,
        pnl_for_entries,
        tag,
    )
    .await
    {
        Ok(cal) => {
            tee_println!(
                "[train] {tag}: calibration: breakpoints={} \
                 (примеры: raw 0.50→{:.3}, 0.70→{:.3}, 0.85→{:.3}, 0.95→{:.3})",
                cal.xs.len(),
                cal.apply(0.50),
                cal.apply(0.70),
                cal.apply(0.85),
                cal.apply(0.95),
            );
            match save_calibration(&cal, model_path) {
                Ok(path) => {
                    tee_println!("[train] {tag}: калибровка сохранена → {}", path.display())
                }
                Err(err) => tee_eprintln!("[train] {tag}: ошибка сохранения калибровки: {err:#}"),
            }
        }
        Err(err) => tee_eprintln!("[train] {tag}: ошибка калибровки (isotonic): {err:#}"),
    }

    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent)?;
    }
    booster.save(model_path)?;
    Ok(())
}

/// Печатает метрики logloss и AUC на train и val/test выборках.
/// `eval_label` — человеко-читаемое имя второй выборки ("val" или "test").
fn print_eval_metrics(booster: &Booster, tag: &str, eval_label: &str) {
    let results = &booster.eval_dmat_results;
    let get = |metric: &str, split: &str| -> String {
        results
            .get(metric)
            .and_then(|splits| splits.get(split))
            .map(|val| format!("{val:.5}"))
            .unwrap_or_else(|| "—".to_string())
    };
    tee_println!(
        "[train] {tag}: метрики: train-logloss:{:>8}  {eval_label}-logloss:{:>8}  train-auc:{:>8}  {eval_label}-auc:{:>8}",
        get("logloss", "train"),
        get("logloss", "test"),
        get("auc", "train"),
        get("auc", "test"),
    );
}

/// Вычисляет и печатает SHAP-вклад каждой фичи на первой строке тестовой выборки,
/// отсортированный по убыванию абсолютного вклада.
fn print_contributions(booster: &Booster, dtest: &DMatrix, tag: &str, max_lag: Option<usize>) {
    let Ok((shap_values, (num_rows, num_cols))) = booster.predict_contributions(dtest) else {
        tee_eprintln!("[train] {tag}: не удалось вычислить SHAP contributions");
        return;
    };
    if num_rows == 0 {
        return;
    }

    let n_features = num_cols - 1; // последний столбец — bias
    let total_abs: f32 = (0..n_features)
        .map(|feat_idx| shap_values[feat_idx].abs())
        .sum();

    let mut contributions: Vec<(String, f32, f32)> = (0..n_features)
        .filter_map(|feat_idx| {
            let shap = shap_values[feat_idx];
            let name = match max_lag {
                Some(n) => XFrame::<SIZE>::feature_name_n(feat_idx, n),
                None => XFrame::<SIZE>::feature_name(feat_idx),
            }?;
            let percent = if total_abs > 0.0 {
                shap.abs() / total_abs * 100.0
            } else {
                0.0
            };
            Some((name.to_string(), shap, percent))
        })
        .collect();

    contributions.sort_by(|(_, _, pct_a), (_, _, pct_b)| {
        pct_b
            .partial_cmp(pct_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    tee_println!("[train] {tag}: SHAP contributions (первая строка теста, топ-20):");
    for (name, shap, percent) in contributions.iter().take(20) {
        tee_println!("  {:>8.4}  {:>6.2}%  {name}", shap, percent);
    }
    let bias = shap_values[num_cols - 1];
    tee_println!("  {:>8.4}           __bias__", bias);
}

/// Печатает воронку разметки: сколько кадров пришло, сколько размечено,
/// и распределение причин пропуска. См. [`AppendStats`].
fn print_append_stats(tag: &str, split: &str, s: &AppendStats) {
    if s.total_frames == 0 {
        return;
    }
    let marked_pct = s.marked as f64 / s.total_frames as f64 * 100.0;
    let y_none_pct = s.y_none as f64 / s.total_frames as f64 * 100.0;
    let out_pct = s.out_of_hold_zone as f64 / s.total_frames as f64 * 100.0;
    let late_entry_pct = s.late_entry as f64 / s.total_frames as f64 * 100.0;
    tee_println!(
        "[train] {tag}: append_stats ({split}): frames={} marked={} ({:.1}%) y_none={} ({:.1}%) out_of_hold_zone={} ({:.1}%) late_entry={} ({:.1}%) feature_mismatch={}",
        s.total_frames,
        s.marked,
        marked_pct,
        s.y_none,
        y_none_pct,
        s.out_of_hold_zone,
        out_pct,
        s.late_entry,
        late_entry_pct,
        s.feature_mismatch,
    );
}

/// Печатает распределение меток в train и test выборках.
fn print_y_distribution(y_train: &[f32], y_val: &[f32], y_test: &[f32], tag: &str) {
    fn count_values(labels: &[f32]) -> std::collections::BTreeMap<String, usize> {
        let mut counts = std::collections::BTreeMap::new();
        for &val in labels {
            let key = format!("{val:.1}");
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    let print_counts = |split: &str, labels: &[f32]| {
        let counts = count_values(labels);
        let total = labels.len();
        tee_println!("[train] {tag}: распределение y ({split}, всего={total}):");
        for (val, count) in &counts {
            let percent = *count as f64 / total as f64 * 100.0;
            tee_println!("  y={val}: {count:>6}  ({percent:>5.1}%)");
        }
    };

    print_counts("train", y_train);
    print_counts("val", y_val);
    print_counts("test", y_test);
}

/// Метрики одного TPE-trial'а на eval-сете (имя сета — `"test"`, см. caller'а
/// `tune_xgboost_optimizer`). Считаются за одно обучение в [`eval_xgboost`]
/// и затем сворачиваются в скаляр для [`Study`] через [`Self::score_for`].
#[derive(Debug, Clone, Copy)]
struct TrialMetrics {
    auc: f64,
    logloss: f64,
}

impl TrialMetrics {
    /// Сворачивает метрики в одно число для оптимизатора согласно
    /// активной [`TuneObjective`]. Для `MaximizeAucWithPenalty` штраф
    /// линеен по превышению `logloss` над [`AUC_PENALTY_LOGLOSS_BASELINE`]
    /// с весом [`AUC_PENALTY_LOGLOSS_WEIGHT`]; ниже baseline — штраф 0.
    fn score_for(&self, obj: TuneObjective) -> f64 {
        match obj {
            TuneObjective::MaximizeAuc => self.auc,
            TuneObjective::MinimizeLogLoss => self.logloss,
            TuneObjective::MaximizeAucWithPenalty => {
                let penalty = (self.logloss - AUC_PENALTY_LOGLOSS_BASELINE).max(0.0)
                    * AUC_PENALTY_LOGLOSS_WEIGHT;
                self.auc - penalty
            }
        }
    }
}

/// Байесовская оптимизация гиперпараметров XGBoost. Метрика-цель и направление
/// берутся из [`TUNE_OBJECTIVE`] (см. doc у [`TuneObjective`]). Каждый trial
/// измеряет AUC и LogLoss на eval-сете `"test"`; сворачиваются в `score`
/// через [`TrialMetrics::score_for`], TPE оптимизирует по нему.
fn tune_xgboost_optimizer(
    eval_sets: &[(&DMatrix, &str); 2],
    dtrain: &DMatrix,
    trials: usize,
    tag: &str,
) -> anyhow::Result<XgbParams> {
    let sampler = TpeSampler::new();
    let objective = TUNE_OBJECTIVE;
    let study: Study<f64> = Study::with_sampler(objective.direction(), sampler);

    study.optimize_with_sampler(trials, |trial| {
        let params = XgbParams {
            eta: trial.suggest_float("eta", ETA_MIN as f64, ETA_MAX as f64)? as f32,
            max_depth: trial.suggest_int("max_depth", 2, 40)? as u32,
            min_child_weight: trial.suggest_float("min_child_weight", 0.0, 20.0)? as f32,
            gamma: trial.suggest_float("gamma", 0.0, 10.0)? as f32,
            subsample: trial.suggest_float("subsample", 0.1, 1.0)? as f32,
            colsample_bytree: trial.suggest_float("colsample_bytree", 0.1, 1.0)? as f32,
            lambda: trial.suggest_float("lambda", 0.0, 20.0)? as f32,
            alpha: trial.suggest_float("alpha", 0.0, 80.0)? as f32,
            scale_pos_weight: trial.suggest_float("scale_pos_weight", 4.0, 30.0)? as f32,
        };
        let metrics = eval_xgboost(&params, eval_sets, dtrain)
            .map_err(|_err| optimizer::Error::InvalidStep)?;
        let score = metrics.score_for(objective);
        tee_println!(
            "[train] {tag} trial #{}: {label}={score:.6} (auc={auc:.6} logloss={logloss:.6})",
            trial.id(),
            label = objective.label(),
            score = score,
            auc = metrics.auc,
            logloss = metrics.logloss,
        );
        Ok::<f64, optimizer::Error>(score)
    })?;

    let best = study.best_trial()?;
    tee_println!(
        "[train] {tag}: лучший trial ({label}): value={} params={:?}",
        best.value,
        best.params,
        label = objective.label(),
    );
    Ok(params_from_map(&best.params))
}

/// Быстрое обучение для оценки параметров: возвращает AUC и LogLoss на
/// eval-сете `"test"` (`eval_sets[1]` в caller'е). Сворачивание в скаляр
/// для TPE — [`TrialMetrics::score_for`].
fn eval_xgboost(
    params: &XgbParams,
    eval_sets: &[(&DMatrix, &str); 2],
    dtrain: &DMatrix,
) -> Result<TrialMetrics, Box<dyn std::error::Error>> {
    let rounds = eval_boost_rounds(params.eta);
    let booster = fit_booster(params, dtrain, eval_sets, rounds)?;
    let auc = booster
        .eval_dmat_results
        .get("auc")
        .and_then(|metric| metric.get("test"))
        .copied()
        .unwrap_or(0.0) as f64;
    let logloss = booster
        .eval_dmat_results
        .get("logloss")
        .and_then(|metric| metric.get("test"))
        .copied()
        .unwrap_or(f32::INFINITY) as f64;
    Ok(TrialMetrics { auc, logloss })
}

/// Бюджет раундов на TPE-пробу: обратная пропорция к `eta` относительно
/// [`EVAL_REFERENCE_ETA`] с клиппингом в `[EVAL_BOOST_ROUNDS, EVAL_BOOST_ROUNDS_MAX]`.
///
/// Мотивация: сходимость градиентного бустинга ≈ `T * eta = const`,
/// поэтому при фиксированном `T = EVAL_BOOST_ROUNDS` пробы с малым `eta`
/// систематически недоучиваются, и TPE видит шум вместо реального AUC.
fn eval_boost_rounds(eta: f32) -> u32 {
    let eta = eta.max(ETA_MIN);
    let scaled = (EVAL_BOOST_ROUNDS as f32 * EVAL_REFERENCE_ETA / eta).ceil();
    (scaled as u32).clamp(EVAL_BOOST_ROUNDS, EVAL_BOOST_ROUNDS_MAX)
}

/// Обучение с early stopping: критерий улучшения совпадает с [`TUNE_OBJECTIVE`]
/// (через [`TrialMetrics::score_for`], как в [`tune_xgboost_optimizer`]);
/// останавливается после `EARLY_STOPPING_PATIENCE` раундов без улучшения на val
/// (`dtest`); возвращает booster с лучшим снимком по этому критерию.
fn fit_booster_with_early_stopping(
    params: &XgbParams,
    dtrain: &DMatrix,
    dtest: &DMatrix,
    tag: &str,
) -> anyhow::Result<Booster> {
    let objective = TUNE_OBJECTIVE;
    let booster_params = build_booster_params(params)?;
    let cached = [dtrain, dtest];
    let mut bst = Booster::new_with_cached_dmats(&booster_params, &cached)?;

    let mut best_score: Option<f64> = None;
    let mut best_metrics: Option<TrialMetrics> = None;
    let mut best_snapshot: Vec<u8> = Vec::new();
    let mut best_round: u32 = 0;
    let mut rounds_without_improvement: u32 = 0;
    // Метрики на момент лучшего раунда: metric -> {split -> val}.
    // Сохраняем здесь, а не переоцениваем после load_buffer —
    // так как load_buffer не восстанавливает eval_metric параметры booster'а.
    let mut best_eval_results: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, f32>,
    > = Default::default();

    for round in 0..BOOST_ROUNDS {
        bst.update(dtrain, round as i32)?;

        let test_metrics = bst.evaluate(dtest)?;
        let auc = test_metrics.get("auc").copied().unwrap_or(0.0) as f64;
        let logloss = test_metrics
            .get("logloss")
            .copied()
            .unwrap_or(f32::INFINITY) as f64;
        let metrics = TrialMetrics { auc, logloss };
        let score = metrics.score_for(objective);

        let improved = best_score.map_or(true, |b| objective.score_improved(score, b));
        if improved {
            best_score = Some(score);
            best_metrics = Some(metrics);
            best_round = round;
            rounds_without_improvement = 0;
            best_snapshot = bst.save_buffer(true)?;

            // Сохраняем метрики train и test в момент лучшего раунда по objective
            best_eval_results.clear();
            let train_metrics = bst.evaluate(dtrain)?;
            for (metric, val) in train_metrics {
                best_eval_results
                    .entry(metric)
                    .or_default()
                    .insert("train".to_string(), val);
            }
            for (metric, val) in test_metrics {
                best_eval_results
                    .entry(metric)
                    .or_default()
                    .insert("test".to_string(), val);
            }
        } else {
            rounds_without_improvement += 1;
            if rounds_without_improvement >= EARLY_STOPPING_PATIENCE {
                let metrics_at_best =
                    best_metrics.expect("best_metrics after at least one improving round");
                let best_objective_score =
                    best_score.expect("best_score after at least one improving round");
                tee_println!(
                    "[train] {tag}: early stopping на раунде {round}: лучший {label}={best_objective_score:.6} (auc={val_auc:.6} logloss={val_logloss:.6}) на раунде {best_round}",
                    label = objective.label(),
                    val_auc = metrics_at_best.auc,
                    val_logloss = metrics_at_best.logloss,
                );
                break;
            }
        }
    }

    if best_snapshot.is_empty() {
        anyhow::bail!("не удалось получить ни одного валидного раунда бустинга");
    }
    let mut result_bst = Booster::load_buffer(&best_snapshot)?;
    result_bst.eval_dmat_results = best_eval_results;
    Ok(result_bst)
}

fn build_booster_params(params: &XgbParams) -> anyhow::Result<xgb::parameters::BoosterParameters> {
    let learning_params = LearningTaskParametersBuilder::default()
        .objective(Objective::BinaryLogistic)
        .eval_metrics(Metrics::Custom(vec![
            EvaluationMetric::LogLoss,
            EvaluationMetric::AUC,
        ]))
        .build()?;

    let tree_params = TreeBoosterParametersBuilder::default()
        .eta(params.eta)
        .max_depth(params.max_depth)
        .min_child_weight(params.min_child_weight)
        .gamma(params.gamma)
        .subsample(params.subsample)
        .colsample_bytree(params.colsample_bytree)
        .lambda(params.lambda)
        .alpha(params.alpha)
        .scale_pos_weight(params.scale_pos_weight)
        .tree_method(TreeMethod::Hist)
        .build()?;

    Ok(BoosterParametersBuilder::default()
        .learning_params(learning_params)
        .booster_type(BoosterType::Tree(tree_params))
        .verbose(false)
        .build()?)
}

fn fit_booster(
    params: &XgbParams,
    dtrain: &DMatrix,
    eval_sets: &[(&DMatrix, &str); 2],
    rounds: u32,
) -> anyhow::Result<Booster> {
    let booster_params = build_booster_params(params)?;

    let training_params = TrainingParametersBuilder::default()
        .dtrain(dtrain)
        .booster_params(booster_params)
        .evaluation_sets(Some(eval_sets))
        .boost_rounds(rounds)
        .build()?;

    Ok(Booster::train(&training_params)?)
}

fn params_from_map(map: &HashMap<String, ParamValue>) -> XgbParams {
    XgbParams {
        eta: get_f32(map, "eta"),
        max_depth: get_u32(map, "max_depth"),
        min_child_weight: get_f32(map, "min_child_weight"),
        gamma: get_f32(map, "gamma"),
        subsample: get_f32(map, "subsample"),
        colsample_bytree: get_f32(map, "colsample_bytree"),
        lambda: get_f32(map, "lambda"),
        alpha: get_f32(map, "alpha"),
        scale_pos_weight: get_f32(map, "scale_pos_weight"),
    }
}

fn get_f32(map: &HashMap<String, ParamValue>, key: &str) -> f32 {
    match &map[key] {
        ParamValue::Float(val) => *val as f32,
        ParamValue::Int(val) => *val as f32,
        _ => panic!("ожидался float/int для {key}"),
    }
}

fn get_u32(map: &HashMap<String, ParamValue>, key: &str) -> u32 {
    match &map[key] {
        ParamValue::Int(val) => *val as u32,
        _ => panic!("ожидался int для {key}"),
    }
}

// ─── Feature weights ─────────────────────────────────────────────────────────

/// Индекс лага из суффикса вида `field[4]` в имени фичи (см. `XFeatures`).
fn lag_bracket_index(name: &str) -> Option<usize> {
    let inner = name.split_once('[')?.1;
    inner.split_once(']')?.0.parse().ok()
}

/// Эффективный понижающий вес лаговой фичи с учётом индекса в скобках.
#[inline]
fn lag_downweight_with_index(base: f32, name: &str) -> f32 {
    let i = lag_bracket_index(name).unwrap_or(0);
    base * LAG_DOWNWEIGHT_PER_STEP.powi(i as i32)
}

/// Строит вектор `feature_weights` длины `n_features`.
/// - Фичи из [`DOWNWEIGHTED_FEATURES`] получают вес из [`DOWNWEIGHT_FACTOR`], если он `Some`.
/// - Лаговые фичи (имя содержит `[`) получают вес из [`LAG_DOWNWEIGHT_FACTOR`], если он `Some`, с затуханием по индексу лага ([`LAG_DOWNWEIGHT_PER_STEP`]).
/// - Если фича попадает в оба условия, берётся минимальный из применимых весов; при обоих `None` вес остаётся 1.0.
fn build_feature_weights(n_features: usize, max_lag: Option<usize>) -> Vec<f32> {
    let mut weights = vec![1.0_f32; n_features];
    let mut n_explicit = 0usize;
    let mut n_lag = 0usize;
    for idx in 0..n_features {
        let name = match max_lag {
            Some(n) => XFrame::<SIZE>::feature_name_n(idx, n),
            None => XFrame::<SIZE>::feature_name(idx),
        };
        if let Some(name) = name {
            let is_lag = name.contains('[');
            let base_name = name.split('[').next().unwrap_or(name);
            let is_explicit = DOWNWEIGHTED_FEATURES.contains(&base_name);

            if is_explicit && is_lag {
                let w = match (DOWNWEIGHT_FACTOR, LAG_DOWNWEIGHT_FACTOR) {
                    (Some(d), Some(l)) => Some(d.min(lag_downweight_with_index(l, name))),
                    (Some(d), None) => Some(d),
                    (None, Some(l)) => Some(lag_downweight_with_index(l, name)),
                    (None, None) => None,
                };
                if let Some(w) = w {
                    weights[idx] = w;
                    n_explicit += 1;
                    n_lag += 1;
                }
            } else if is_explicit {
                if let Some(d) = DOWNWEIGHT_FACTOR {
                    weights[idx] = d;
                    n_explicit += 1;
                }
            } else if is_lag {
                if let Some(l) = LAG_DOWNWEIGHT_FACTOR {
                    weights[idx] = lag_downweight_with_index(l, name);
                    n_lag += 1;
                }
            }
        }
    }
    if n_explicit > 0 || n_lag > 0 {
        tee_println!(
            "[train] feature_weights: explicit={n_explicit} (factor={DOWNWEIGHT_FACTOR:?}), \
             lag={n_lag} (base={LAG_DOWNWEIGHT_FACTOR:?}, per_lag_step={LAG_DOWNWEIGHT_PER_STEP})"
        );
    }
    weights
}

// ─── Метрики (ручной расчёт) ─────────────────────────────────────────────────

/// AUC-ROC по предсказаниям и меткам (Wilcoxon–Mann–Whitney).
///
/// Сортирует пары `(pred, label)` по pred **asc** (rank 1 = наименьшее
/// предсказание), затем считает сумму рангов позитивного класса.
fn calc_auc(preds: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, bool)> = preds
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| (p, y >= 1.0))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = pairs.iter().filter(|(_, y)| *y).count() as f64;
    let n_neg = pairs.iter().filter(|(_, y)| !*y).count() as f64;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.0;
    }

    let mut sum_ranks = 0.0_f64;
    let mut rank = 1.0_f64;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let avg_rank = (rank + rank + (j - i - 1) as f64) / 2.0;
        for k in i..j {
            if pairs[k].1 {
                sum_ranks += avg_rank;
            }
        }
        rank += (j - i) as f64;
        i = j;
    }

    let auc = (sum_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);
    auc as f32
}

/// Binary cross-entropy (logloss).
fn calc_logloss(preds: &[f32], labels: &[f32]) -> f32 {
    if preds.is_empty() {
        return 0.0;
    }
    let eps = 1e-7_f32;
    let sum: f32 = preds
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| {
            let p = p.clamp(eps, 1.0 - eps);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum();
    sum / preds.len() as f32
}

/// Возвращает список путей к подпапкам/файлам в `dir`, отсортированных по имени.
fn fs_read_dirs(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    entries.sort();
    Ok(entries)
}
