//! Режим обучения: читает дампы [`crate::xframe_dump::MarketXFramesDump`] из папки `xframes/`,
//! строит матрицы признаков и меток, обучает XGBoost с байесовской оптимизацией гиперпараметров
//! и сохраняет модель рядом с папкой версии.

use crate::history_sim::HOLD_TO_END_THRESHOLD_SEC;
use crate::project_manager::FRAME_BUILD_INTERVALS_SEC;
use crate::tee_log::TEE_LOG;
use crate::xframe::{
    apply_side_symmetry, calc_y_train_pnl, calc_y_train_resolution, XFrame, SIZE,
    Y_TRAIN_HORIZON_FRAMES, Y_TRAIN_TAKE_PROFIT_PP, Y_TRAIN_STOP_LOSS_PP,
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
use xgb::parameters::learning::{EvaluationMetric, LearningTaskParametersBuilder, Metrics, Objective};
use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
use xgb::parameters::{BoosterParametersBuilder, BoosterType, TrainingParametersBuilder};
use xgb::{Booster, DMatrix};

/// Число итераций байесовского оптимизатора.
const OPTIMIZER_TRIALS: usize = 100;
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
const ETA_MIN: f32 = 0.03;
/// Верхняя граница `eta` — соответствует прежнему поведению.
const ETA_MAX: f32 = 0.3;
/// Доля валидационной выборки (для optimizer + early stopping).
pub const VAL_FRACTION: f64 = 0.1;
/// Доля тестовой выборки (финальная, честная оценка AUC).
pub const TEST_FRACTION: f64 = 0.1;
/// Понижающий коэффициент `feature_weights` для конкретных фич из [`DOWNWEIGHTED_FEATURES`].
const DOWNWEIGHT_FACTOR: f32 = 0.1;
/// Понижающий коэффициент для лаговых фич (массивы `delta_n_*[i]`).
const LAG_DOWNWEIGHT_FACTOR: f32 = 0.3;
/// Имена фич, которым автоматически понижается `feature_weight` при обучении.
// const DOWNWEIGHTED_FEATURES: &[&str] = &["event_remaining_ms", "sibling_event_remaining_ms", "currency_price_vs_beat_pct", "sibling_currency_price_vs_beat_pct"];
const DOWNWEIGHTED_FEATURES: &[&str] = &["event_remaining_ms", "sibling_event_remaining_ms", "sibling_currency_price_vs_beat_pct"];
/// Ниже этого порога сохраняется identity-калибровка.
const CALIBRATION_MIN_AUC: f32 = 0.60;
/// Эпсилон для клиппинга выходов isotonic regression: исключает 0/1 значения,
/// которые сломают logloss и Kelly при логарифмировании.
const CALIBRATION_EPS: f32 = 1e-3;
/// Минимальный суммарный вес одного блока (число сэмплов) в isotonic-калибровке.
/// После PAV последовательно объединяем соседние блоки до достижения этого порога —
/// это регуляризация против переобучения на малых калибровочных сетах.
/// Монотонность при этом сохраняется (weighted-avg двух non-decreasing соседей
/// остаётся в интервале [prev, next]).
const CALIBRATION_MIN_BLOCK_WEIGHT: f64 = 50.0;

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
        Self { xs: vec![0.0, 1.0], ys: vec![0.0, 1.0] }
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
            probe.partial_cmp(&raw_pred).unwrap_or(std::cmp::Ordering::Equal)
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

/// Isotonic regression калибровка: подгоняет монотонную неубывающую функцию
/// методом PAV (Pool Adjacent Violators) к парам `(raw_prediction, label)`.
///
/// Печатает диагностику для обнаружения инверсии/distribution shift.
fn fit_calibration(booster: &Booster, dmat: &DMatrix, y: &[f32], tag: &str) -> anyhow::Result<Calibration> {
    let preds = booster.predict(dmat)?;

    let n_pos = y.iter().filter(|&&v| v >= 1.0).count();
    let n_neg = y.len() - n_pos;
    let mean_pred_pos: f64 = preds.iter().zip(y.iter())
        .filter(|(_, yv)| **yv >= 1.0)
        .map(|(&p, _)| p as f64)
        .sum::<f64>() / n_pos.max(1) as f64;
    let mean_pred_neg: f64 = preds.iter().zip(y.iter())
        .filter(|(_, yv)| **yv < 1.0)
        .map(|(&p, _)| p as f64)
        .sum::<f64>() / n_neg.max(1) as f64;
    let cal_auc = calc_auc(&preds, y);
    tee_println!(
        "[calibration] {tag}: n_pos={n_pos} n_neg={n_neg} mean_pred_pos={mean_pred_pos:.4} \
         mean_pred_neg={mean_pred_neg:.4} AUC={cal_auc:.4}"
    );

    if cal_auc < CALIBRATION_MIN_AUC {
        tee_eprintln!(
            "[calibration] {tag}: AUC={cal_auc:.4} < {CALIBRATION_MIN_AUC} — модель слишком \
             слабая для калибровки. Используется identity."
        );
        return Ok(Calibration::identity());
    }

    if n_pos == 0 || n_neg == 0 {
        tee_eprintln!(
            "[calibration] {tag}: в калибровочном сете есть только один класс \
             (n_pos={n_pos}, n_neg={n_neg}). Используется identity."
        );
        return Ok(Calibration::identity());
    }

    let cal = isotonic_fit(&preds, y);
    tee_println!(
        "[calibration] {tag}: fit OK | breakpoints={} | \
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

    let mut pairs: Vec<(f32, f32)> = preds.iter().zip(y.iter())
        .map(|(&p, &yv)| (p, if yv >= 1.0 { 1.0_f32 } else { 0.0_f32 }))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    #[derive(Clone, Copy)]
    struct Block { sum_x: f64, sum_y: f64, weight: f64 }
    impl Block {
        fn value(&self) -> f64 { self.sum_y / self.weight }
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
        blocks.push(Block { sum_x: x as f64, sum_y: y_i as f64, weight: 1.0 });
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
    // пока суммарный вес не достигнет CALIBRATION_MIN_BLOCK_WEIGHT.
    // Монотонность сохраняется: если v_1 ≤ … ≤ v_k и v_{k+1} ≤ … ≤ v_m,
    // то weighted_avg(v_1..v_k) ≤ v_k ≤ v_{k+1} ≤ weighted_avg(v_{k+1}..v_m).
    let min_weight = CALIBRATION_MIN_BLOCK_WEIGHT;
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
            Self::Up   => "up",
            Self::Down => "down",
        }
    }

    fn frames<'a>(&self, dump: &'a MarketXFramesDump) -> &'a [XFrame<SIZE>] {
        match self {
            Self::Up   => &dump.frames_up,
            Self::Down => &dump.frames_down,
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
pub fn run_train_mode() -> anyhow::Result<()> {
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
                    let val_paths   = &paths[train_count..train_count + val_count];
                    let test_paths  = &paths[train_count + val_count..];
                    tee_println!(
                        "[train] {tag_prefix}: маркетов {} → сплит {train_count} train / {val_count} val / {test_count} test",
                        paths.len(),
                    );

                    let train_dumps = load_dumps_for_paths(train_paths);
                    let val_dumps   = load_dumps_for_paths(val_paths);
                    let test_dumps  = load_dumps_for_paths(test_paths);

                    train_all_variants(
                        &train_dumps, &val_dumps, &test_dumps,
                        &version_path, &tag_prefix, interval, step_sec,
                    )?;
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

/// Обучает модели для всех комбинаций `model_type × side` на одном
/// `(currency, version, interval, step_sec)`. Каждая комбинация даёт отдельный
/// файл `model_{interval}_{step_sec}s_{model_type}_{side}.ubj` — формат,
/// который грузит `history_sim`.
fn train_all_variants(
    train_dumps: &[MarketXFramesDump],
    val_dumps:   &[MarketXFramesDump],
    test_dumps:  &[MarketXFramesDump],
    version_path: &Path,
    tag_prefix: &str,
    interval: &str,
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

            let train_markets = build_market_datasets(train_dumps, side, model_type, max_lag);
            let val_markets   = build_market_datasets(val_dumps,   side, model_type, max_lag);
            let test_markets  = build_market_datasets(test_dumps,  side, model_type, max_lag);

            let total_markets = train_markets.len() + val_markets.len() + test_markets.len();
            if total_markets == 0 {
                tee_println!("[train] {tag}: нет данных, пропуск");
                continue;
            }

            let feature_count = match max_lag {
                Some(n) => XFrame::<SIZE>::count_features_n(n),
                None => XFrame::<SIZE>::count_features(),
            };
            let total_rows: usize = train_markets.iter().chain(val_markets.iter()).chain(test_markets.iter())
                .map(|m| m.y.len())
                .sum();
            tee_println!(
                "[train] {tag}: маркетов {}/{}/{} (train/val/test), {} строк, {} признаков",
                train_markets.len(), val_markets.len(), test_markets.len(),
                total_rows, feature_count,
            );

            let model_path = version_path.join(format!(
                "model_{interval}_{step_sec}s_{}_{}.ubj",
                model_type.label(),
                side.label(),
            ));

            match train_and_save(
                &train_markets, &val_markets, &test_markets,
                &model_path, &tag, model_type, max_lag,
            ) {
                Ok(()) => tee_println!("[train] {tag}: модель сохранена → {}", model_path.display()),
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

/// Загружает дампы для заданного списка путей. Ошибки чтения/десериализации
/// молча пропускаются (печатаются в stderr) — идентичное поведение
/// с [`crate::history_sim`]: битый файл одинаково игнорируется обоими.
fn load_dumps_for_paths(paths: &[PathBuf]) -> Vec<MarketXFramesDump> {
    let mut dumps = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(err) => {
                tee_eprintln!("[train] не удалось прочитать {}: {err}", path.display());
                continue;
            }
        };
        match bincode::deserialize::<MarketXFramesDump>(&bytes) {
            Ok(dump) => dumps.push(dump),
            Err(err) => tee_eprintln!("[train] ошибка десериализации {}: {err}", path.display()),
        }
    }
    dumps
}

/// Формирует `MarketDataset` для каждого дампа по заданной ноге и типу модели.
/// `max_lag` — если `Some(n)`, лаговые массивы обрезаются до первых `n` элементов.
fn build_market_datasets(dumps: &[MarketXFramesDump], side: FrameSide, model_type: ModelType, max_lag: Option<usize>) -> Vec<MarketDataset> {
    let feature_count = match max_lag {
        Some(n) => XFrame::<SIZE>::count_features_n(n),
        None => XFrame::<SIZE>::count_features(),
    };
    let mut markets = Vec::new();

    for dump in dumps {
        let mut x = Vec::new();
        let mut y = Vec::new();
        append_frames(side.frames(dump), feature_count, model_type, dump.price_to_beat, dump.final_price, max_lag, &mut x, &mut y);
        if !y.is_empty() {
            markets.push(MarketDataset { x, y });
        }
    }

    markets
}

/// Для каждого кадра в `frames` вычисляет метку по `model_type` и, если она есть,
/// добавляет признаки и метку в `x_out` / `y_out`.
fn append_frames(
    frames: &[XFrame<SIZE>],
    feature_count: usize,
    model_type: ModelType,
    price_to_beat: f64,
    final_price: f64,
    max_lag: Option<usize>,
    x_out: &mut Vec<f32>,
    y_out: &mut Vec<f32>,
) {
    // Граница hold zone в мс (условие идентично [`crate::history_sim::manage_positions`]:
    // `event_remaining_ms > 0 && event_remaining_ms <= HOLD_TO_END_THRESHOLD_SEC * 1000`).
    // Resolution-модель используется исключительно внутри hold zone, поэтому и
    // обучаем её только на кадрах этого диапазона — обучающее распределение
    // совпадает с инференс-распределением.
    let hold_zone_max_ms: i64 = HOLD_TO_END_THRESHOLD_SEC * 1000;

    for index in 0..frames.len() {
        if matches!(model_type, ModelType::Resolution) {
            let remaining = frames[index].event_remaining_ms;
            if remaining <= 0 || remaining > hold_zone_max_ms {
                continue;
            }
        }

        let label = match model_type {
            ModelType::Pnl => calc_y_train_pnl(Y_TRAIN_HORIZON_FRAMES, frames, index, price_to_beat, final_price),
            ModelType::Resolution => calc_y_train_resolution(Y_TRAIN_HORIZON_FRAMES, frames, index, price_to_beat, final_price),
        };
        let Some(label) = label else {
            continue;
        };
        let row = match max_lag {
            Some(n) => frames[index].to_x_train_n_with(n, apply_side_symmetry),
            None => frames[index].to_x_train_with(apply_side_symmetry),
        };
        if row.len() != feature_count {
            continue;
        }
        x_out.extend_from_slice(&row);
        y_out.push(label);
    }
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
fn train_and_save(
    train_markets: &[MarketDataset],
    val_markets: &[MarketDataset],
    test_markets: &[MarketDataset],
    model_path: &Path,
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

    match model_type {
        ModelType::Pnl => tee_println!(
            "[train] {tag}: оптимизация гиперпараметров по AUC на val ({OPTIMIZER_TRIALS} итераций, TP={Y_TRAIN_TAKE_PROFIT_PP}, SL={Y_TRAIN_STOP_LOSS_PP})…"
        ),
        ModelType::Resolution => tee_println!(
            "[train] {tag}: оптимизация гиперпараметров по AUC на val ({OPTIMIZER_TRIALS} итераций)…"
        ),
    }
    let params = tune_xgboost_optimizer(&eval_sets, &dtrain, OPTIMIZER_TRIALS, tag)?;
    tee_println!("[train] {tag}: лучшие параметры: {params:?}");

    let booster = fit_booster_with_early_stopping(&params, &dtrain, &dval, tag)?;

    // Метрики на val (из early stopping)
    print_eval_metrics(&booster, tag, "val");

    // Финальная честная оценка на held-out test (AUC считаем вручную,
    // т.к. booster после load_buffer теряет конфигурацию eval_metrics).
    let test_preds = booster.predict(&dtest)?;
    let test_auc = calc_auc(&test_preds, &y_test);
    let test_logloss = calc_logloss(&test_preds, &y_test);
    tee_println!(
        "[train] {tag}: held-out test: logloss={test_logloss:.5}  AUC={test_auc:.6}"
    );

    print_y_distribution(&y_train, &y_val, &y_test, tag);
    print_contributions(&booster, &dtest, tag, max_lag);

    // ── Isotonic regression: калибровка на VAL set ───────────────────────────
    // Val уже «запачкан» early stopping'ом, но это лучше чем калибровать на test:
    // test обязан оставаться полностью held-out для честной финальной оценки AUC.
    // Кроме того, isotonic имеет O(N) параметров и катастрофически переобучается
    // если калибровочный сет совпадает с тем, по которому меряется AUC.
    match fit_calibration(&booster, &dval, &y_val, tag) {
        Ok(cal) => {
            tee_println!(
                "[train] {tag}: calibration: breakpoints={} \
                 (примеры: raw 0.50→{:.3}, 0.70→{:.3}, 0.85→{:.3}, 0.95→{:.3})",
                cal.xs.len(),
                cal.apply(0.50), cal.apply(0.70), cal.apply(0.85), cal.apply(0.95),
            );
            match save_calibration(&cal, model_path) {
                Ok(path) => tee_println!("[train] {tag}: калибровка сохранена → {}", path.display()),
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
        pct_b.partial_cmp(pct_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    tee_println!("[train] {tag}: SHAP contributions (первая строка теста, топ-20):");
    for (name, shap, percent) in contributions.iter().take(20) {
        tee_println!("  {:>8.4}  {:>6.2}%  {name}", shap, percent);
    }
    let bias = shap_values[num_cols - 1];
    tee_println!("  {:>8.4}           __bias__", bias);
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


/// Байесовская оптимизация гиперпараметров XGBoost (максимизация AUC на тесте).
fn tune_xgboost_optimizer(
    eval_sets: &[(&DMatrix, &str); 2],
    dtrain: &DMatrix,
    trials: usize,
    tag: &str,
) -> anyhow::Result<XgbParams> {
    let sampler = TpeSampler::new();
    // Максимизируем AUC на тестовой выборке: для торговой модели с дисбалансом классов
    // AUC лучше отражает способность разделять классы, чем logloss.
    let study: Study<f64> = Study::with_sampler(Direction::Maximize, sampler);

    study.optimize_with_sampler(trials, |trial| {
        let params = XgbParams {
            eta: trial.suggest_float("eta", ETA_MIN as f64, ETA_MAX as f64)? as f32,
            max_depth: trial.suggest_int("max_depth", 2, 8)? as u32,
            min_child_weight: trial.suggest_float("min_child_weight", 1.0, 20.0)? as f32,
            gamma: trial.suggest_float("gamma", 0.0, 10.0)? as f32,
            subsample: trial.suggest_float("subsample", 0.5, 1.0)? as f32,
            colsample_bytree: trial.suggest_float("colsample_bytree", 0.5, 1.0)? as f32,
            lambda: trial.suggest_float("lambda", 0.0, 20.0)? as f32,
            alpha: trial.suggest_float("alpha", 0.0, 80.0)? as f32,
            scale_pos_weight: trial.suggest_float("scale_pos_weight", 5.0, 30.0)? as f32,
        };
        let score = eval_xgboost(&params, eval_sets, dtrain)
            .map_err(|_err| optimizer::Error::InvalidStep)?;
        tee_println!("[train] {tag} trial #{}: auc={score:.6}", trial.id());
        Ok::<f64, optimizer::Error>(score)
    })?;

    let best = study.best_trial()?;
    tee_println!("[train] {tag}: лучший trial: value={} params={:?}", best.value, best.params);
    Ok(params_from_map(&best.params))
}

/// Быстрое обучение для оценки параметров.
fn eval_xgboost(
    params: &XgbParams,
    eval_sets: &[(&DMatrix, &str); 2],
    dtrain: &DMatrix,
) -> Result<f64, Box<dyn std::error::Error>> {
    let rounds = eval_boost_rounds(params.eta);
    let booster = fit_booster(params, dtrain, eval_sets, rounds)?;
    let auc = booster
        .eval_dmat_results
        .get("auc")
        .and_then(|metric| metric.get("test"))
        .copied()
        .unwrap_or(0.0) as f64;
    Ok(auc)
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

/// Обучение с early stopping: останавливается, когда AUC на тесте не улучшается
/// `EARLY_STOPPING_PATIENCE` раундов подряд; возвращает booster с лучшим AUC.
fn fit_booster_with_early_stopping(
    params: &XgbParams,
    dtrain: &DMatrix,
    dtest: &DMatrix,
    tag: &str,
) -> anyhow::Result<Booster> {
    let booster_params = build_booster_params(params)?;
    let cached = [dtrain, dtest];
    let mut bst = Booster::new_with_cached_dmats(&booster_params, &cached)?;

    let mut best_auc: f32 = 0.0;
    let mut best_snapshot: Vec<u8> = Vec::new();
    let mut best_round: u32 = 0;
    let mut rounds_without_improvement: u32 = 0;
    // Метрики на момент лучшего раунда: metric -> {split -> val}.
    // Сохраняем здесь, а не переоцениваем после load_buffer —
    // так как load_buffer не восстанавливает eval_metric параметры booster'а.
    let mut best_eval_results: std::collections::BTreeMap<String, std::collections::BTreeMap<String, f32>> =
        Default::default();

    for round in 0..BOOST_ROUNDS {
        bst.update(dtrain, round as i32)?;

        let test_metrics = bst.evaluate(dtest)?;
        let auc = test_metrics.get("auc").copied().unwrap_or(0.0);

        if auc > best_auc {
            best_auc = auc;
            best_round = round;
            rounds_without_improvement = 0;
            best_snapshot = bst.save_buffer(true)?;

            // Сохраняем метрики train и test в момент лучшего AUC
            best_eval_results.clear();
            let train_metrics = bst.evaluate(dtrain)?;
            for (metric, val) in train_metrics {
                best_eval_results.entry(metric).or_default().insert("train".to_string(), val);
            }
            for (metric, val) in test_metrics {
                best_eval_results.entry(metric).or_default().insert("test".to_string(), val);
            }
        } else {
            rounds_without_improvement += 1;
            if rounds_without_improvement >= EARLY_STOPPING_PATIENCE {
                tee_println!(
                    "[train] {tag}: early stopping на раунде {round}: лучший AUC={best_auc:.6} на раунде {best_round}"
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

/// Строит вектор `feature_weights` длины `n_features`.
/// - Фичи из [`DOWNWEIGHTED_FEATURES`] получают вес [`DOWNWEIGHT_FACTOR`].
/// - Лаговые фичи (имя содержит `[`) получают вес [`LAG_DOWNWEIGHT_FACTOR`].
/// - Если фича попадает в оба условия, берётся минимальный вес.
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
                weights[idx] = DOWNWEIGHT_FACTOR.min(LAG_DOWNWEIGHT_FACTOR);
                n_explicit += 1;
                n_lag += 1;
            } else if is_explicit {
                weights[idx] = DOWNWEIGHT_FACTOR;
                n_explicit += 1;
            } else if is_lag {
                weights[idx] = LAG_DOWNWEIGHT_FACTOR;
                n_lag += 1;
            }
        }
    }
    if n_explicit > 0 || n_lag > 0 {
        tee_println!(
            "[train] feature_weights: explicit={n_explicit} (factor={DOWNWEIGHT_FACTOR}), \
             lag={n_lag} (factor={LAG_DOWNWEIGHT_FACTOR})"
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
