//! Режим обучения: читает дампы [`crate::xframe_dump::MarketXFramesDump`] из папки `xframes/`,
//! строит матрицы признаков и меток, обучает XGBoost с байесовской оптимизацией гиперпараметров
//! и сохраняет модель рядом с папкой версии.

use crate::xframe::{calc_y_train, XFrame, SIZE, Y_TRAIN_TAKE_PROFIT_PP, Y_TRAIN_STOP_LOSS_PP};
use crate::xframe_dump::MarketXFramesDump;
use optimizer::sampler::tpe::TpeSampler;
use optimizer::{Direction, ParamValue, Study};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use xgb::parameters::learning::{EvaluationMetric, LearningTaskParametersBuilder, Metrics, Objective};
use xgb::parameters::tree::{TreeBoosterParametersBuilder, TreeMethod};
use xgb::parameters::{BoosterParametersBuilder, BoosterType, TrainingParametersBuilder};
use xgb::{Booster, DMatrix};

/// Число итераций байесовского оптимизатора.
const OPTIMIZER_TRIALS: usize = 50;
/// Число раундов бустинга при финальном обучении.
const BOOST_ROUNDS: u32 = 200;
/// Число раундов бустинга при каждом шаге оптимизатора (быстрее).
const EVAL_BOOST_ROUNDS: u32 = 50;
/// Горизонт предсказания: через сколько кадров оцениваем изменение вероятности.
const Y_TRAIN_HORIZON_FRAMES: usize = 10;
/// Доля тестовой выборки.
const TEST_FRACTION: f64 = 0.1;

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

/// Точка входа в режим обучения. Ищет валюты в `xframes/`, для каждой версии
/// (подпапка с числом признаков) собирает все `.bin` файлы, обучает модель
/// и сохраняет её в `xframes/{currency}/{version}/model.ubj`.
pub fn run_train_mode() -> anyhow::Result<()> {
    let xframes_root = Path::new("xframes");
    if !xframes_root.exists() {
        anyhow::bail!("Папка xframes/ не найдена — сначала запустите сбор данных (STATUS=default)");
    }

    for currency_entry in fs_read_dirs(xframes_root)? {
        let currency_path = currency_entry;
        let currency = currency_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        for version_entry in fs_read_dirs(&currency_path)? {
            let version_path = version_entry;
            let version_str = version_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            // Подпапка версии — число (количество признаков); пропускаем прочее.
            if version_str.parse::<usize>().is_err() {
                continue;
            }

            println!("[train] {currency} / version={version_str}: загрузка дампов...");
            let (x_train, y_train) = collect_dataset(&version_path)?;

            if y_train.is_empty() {
                println!("[train] {currency} / version={version_str}: нет данных, пропуск");
                continue;
            }

            println!(
                "[train] {currency} / version={version_str}: {} строк, {} признаков",
                y_train.len(),
                XFrame::<SIZE>::count_features(),
            );

            let model_path = version_path.join("model.ubj");
            match train_and_save(&x_train, &y_train, &model_path) {
                Ok(()) => println!(
                    "[train] {currency} / version={version_str}: модель сохранена → {}",
                    model_path.display()
                ),
                Err(err) => eprintln!(
                    "[train] {currency} / version={version_str}: ошибка обучения: {err:#}"
                ),
            }
        }
    }

    Ok(())
}

/// Рекурсивно обходит подпапки версии (`{date}/`), читает все `.bin` файлы
/// и склеивает кадры из `frames_up` и `frames_down` в единые векторы `x` и `y`.
fn collect_dataset(version_path: &Path) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let feature_count = XFrame::<SIZE>::count_features();
    let mut x_all: Vec<f32> = Vec::new();
    let mut y_all: Vec<f32> = Vec::new();

    for date_entry in fs_read_dirs(version_path)? {
        let date_path = date_entry;
        if !date_path.is_dir() {
            continue;
        }
        for file in fs_read_dirs(&date_path)? {
            if file.extension().and_then(|ext| ext.to_str()) != Some("bin") {
                continue;
            }
            let bytes = match fs::read(&file) {
                Ok(bytes) => bytes,
                Err(err) => {
                    eprintln!("[train] не удалось прочитать {}: {err}", file.display());
                    continue;
                }
            };
            let dump: MarketXFramesDump = match bincode::deserialize(&bytes) {
                Ok(dump) => dump,
                Err(err) => {
                    eprintln!("[train] ошибка десериализации {}: {err}", file.display());
                    continue;
                }
            };
            append_frames(&dump.frames_up, feature_count, &mut x_all, &mut y_all);
            append_frames(&dump.frames_down, feature_count, &mut x_all, &mut y_all);
        }
    }

    Ok((x_all, y_all))
}

/// Для каждого кадра в `frames` вычисляет метку `calc_y_train` и, если она есть,
/// добавляет признаки и метку в `x_out` / `y_out`.
fn append_frames(
    frames: &[XFrame<SIZE>],
    feature_count: usize,
    x_out: &mut Vec<f32>,
    y_out: &mut Vec<f32>,
) {
    for index in 0..frames.len() {
        let Some(label) = calc_y_train(Y_TRAIN_HORIZON_FRAMES, frames, index) else {
            continue;
        };
        let row = frames[index].to_x_train();
        if row.len() != feature_count {
            continue;
        }
        x_out.extend_from_slice(&row);
        y_out.push(label);
    }
}

/// Разбивает датасет на train/test, запускает `tune_xgboost_optimizer`, проводит
/// финальное обучение с оптимальными параметрами и сохраняет модель.
fn train_and_save(
    x_all: &[f32],
    y_all: &[f32],
    model_path: &Path,
) -> anyhow::Result<()> {
    let feature_count = XFrame::<SIZE>::count_features();
    let num_rows = y_all.len();
    assert_eq!(x_all.len(), num_rows * feature_count, "несовпадение размеров датасета");

    // Хронологический сплит: train — первые (1 - TEST_FRACTION) кадров по времени,
    // test — последние TEST_FRACTION. Рандомный shuffle недопустим для временных рядов:
    // будущие кадры попали бы в train, прошлые — в test, что даёт data leakage.
    let train_size = num_rows - ((num_rows as f64) * TEST_FRACTION) as usize;
    let train_idx: Vec<usize> = (0..train_size).collect();
    let test_idx: Vec<usize> = (train_size..num_rows).collect();

    let (x_train, y_train) = gather_rows(x_all, y_all, &train_idx, feature_count);
    let (x_test, y_test) = gather_rows(x_all, y_all, &test_idx, feature_count);

    let mut dtrain = DMatrix::from_dense(&x_train, y_train.len())?;
    dtrain.set_labels(&y_train)?;
    let mut dtest = DMatrix::from_dense(&x_test, y_test.len())?;
    dtest.set_labels(&y_test)?;

    let eval_sets: [(&DMatrix, &str); 2] = [(&dtrain, "train"), (&dtest, "test")];

    println!(
        "[train] оптимизация гиперпараметров по AUC ({OPTIMIZER_TRIALS} итераций, TP={Y_TRAIN_TAKE_PROFIT_PP}, SL={Y_TRAIN_STOP_LOSS_PP})…"
    );
    let params = tune_xgboost_optimizer(&eval_sets, &dtrain, OPTIMIZER_TRIALS)?;
    println!("[train] лучшие параметры: {params:?}");

    let booster = fit_booster(&params, &dtrain, &eval_sets, BOOST_ROUNDS)?;

    print_eval_metrics(&booster);
    print_y_distribution(&y_train, &y_test);
    print_contributions(&booster, &dtest);

    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent)?;
    }
    booster.save(model_path)?;
    Ok(())
}

/// Печатает финальные метрики logloss и AUC на train и test выборках.
fn print_eval_metrics(booster: &Booster) {
    let results = &booster.eval_dmat_results;
    let get = |metric: &str, split: &str| -> String {
        results
            .get(metric)
            .and_then(|splits| splits.get(split))
            .map(|val| format!("{val:.5}"))
            .unwrap_or_else(|| "—".to_string())
    };
    println!(
        "[train] метрики: train-logloss:{:>8}  test-logloss:{:>8}  train-auc:{:>8}  test-auc:{:>8}",
        get("logloss", "train"),
        get("logloss", "test"),
        get("auc", "train"),
        get("auc", "test"),
    );
}

/// Вычисляет и печатает SHAP-вклад каждой фичи на первой строке тестовой выборки,
/// отсортированный по убыванию абсолютного вклада.
fn print_contributions(booster: &Booster, dtest: &DMatrix) {
    let Ok((shap_values, (num_rows, num_cols))) = booster.predict_contributions(dtest) else {
        eprintln!("[train] не удалось вычислить SHAP contributions");
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
            let name = XFrame::<SIZE>::feature_name(feat_idx)?;
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

    println!("[train] SHAP contributions (первая строка теста, топ-20):");
    for (name, shap, percent) in contributions.iter().take(20) {
        println!("  {:>8.4}  {:>6.2}%  {name}", shap, percent);
    }
    let bias = shap_values[num_cols - 1];
    println!("  {:>8.4}           __bias__", bias);
}

/// Печатает распределение меток в train и test выборках.
fn print_y_distribution(y_train: &[f32], y_test: &[f32]) {
    fn count_values(labels: &[f32]) -> std::collections::BTreeMap<String, usize> {
        let mut counts = std::collections::BTreeMap::new();
        for &val in labels {
            let key = format!("{val:.1}");
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    let print_counts = |label: &str, labels: &[f32]| {
        let counts = count_values(labels);
        let total = labels.len();
        println!("[train] распределение y ({label}, всего={total}):");
        for (val, count) in &counts {
            let percent = *count as f64 / total as f64 * 100.0;
            println!("  y={val}: {count:>6}  ({percent:>5.1}%)");
        }
    };

    print_counts("train", y_train);
    print_counts("test", y_test);
}

/// Собирает строки датасета по индексам.
fn gather_rows(x: &[f32], y: &[f32], indices: &[usize], feature_count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x_out = Vec::with_capacity(indices.len() * feature_count);
    let mut y_out = Vec::with_capacity(indices.len());
    for &row_idx in indices {
        x_out.extend_from_slice(&x[row_idx * feature_count..(row_idx + 1) * feature_count]);
        y_out.push(y[row_idx]);
    }
    (x_out, y_out)
}

/// Байесовская оптимизация гиперпараметров XGBoost (минимизация logloss на тесте).
fn tune_xgboost_optimizer(
    eval_sets: &[(&DMatrix, &str); 2],
    dtrain: &DMatrix,
    trials: usize,
) -> anyhow::Result<XgbParams> {
    let sampler = TpeSampler::new();
    // Максимизируем AUC на тестовой выборке: для торговой модели с дисбалансом классов
    // AUC лучше отражает способность разделять классы, чем logloss.
    let study: Study<f64> = Study::with_sampler(Direction::Maximize, sampler);

    study.optimize_with_sampler(trials, |trial| {
        let params = XgbParams {
            eta: trial.suggest_float("eta", 0.01, 0.3)? as f32,
            max_depth: trial.suggest_int("max_depth", 2, 8)? as u32,
            min_child_weight: trial.suggest_float("min_child_weight", 1.0, 20.0)? as f32,
            gamma: trial.suggest_float("gamma", 0.0, 10.0)? as f32,
            subsample: trial.suggest_float("subsample", 0.5, 1.0)? as f32,
            colsample_bytree: trial.suggest_float("colsample_bytree", 0.5, 1.0)? as f32,
            lambda: trial.suggest_float("lambda", 0.0, 20.0)? as f32,
            alpha: trial.suggest_float("alpha", 0.0, 20.0)? as f32,
            scale_pos_weight: trial.suggest_float("scale_pos_weight", 5.0, 20.0)? as f32,
        };
        let score = eval_xgboost(&params, eval_sets, dtrain)
            .map_err(|_err| optimizer::Error::InvalidStep)?;
        println!("[train] trial #{}: auc={score:.6}", trial.id());
        Ok::<f64, optimizer::Error>(score)
    })?;

    let best = study.best_trial()?;
    println!("[train] лучший trial: value={} params={:?}", best.value, best.params);
    Ok(params_from_map(&best.params))
}

/// Быстрое обучение для оценки параметров.
fn eval_xgboost(
    params: &XgbParams,
    eval_sets: &[(&DMatrix, &str); 2],
    dtrain: &DMatrix,
) -> Result<f64, Box<dyn std::error::Error>> {
    let booster = fit_booster(params, dtrain, eval_sets, EVAL_BOOST_ROUNDS)?;
    let auc = booster
        .eval_dmat_results
        .get("auc")
        .and_then(|metric| metric.get("test"))
        .copied()
        .unwrap_or(0.0) as f64;
    Ok(auc)
}

fn fit_booster(
    params: &XgbParams,
    dtrain: &DMatrix,
    eval_sets: &[(&DMatrix, &str); 2],
    rounds: u32,
) -> anyhow::Result<Booster> {
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

    let booster_params = BoosterParametersBuilder::default()
        .learning_params(learning_params)
        .booster_type(BoosterType::Tree(tree_params))
        .verbose(false)
        .build()?;

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

/// Возвращает список путей к подпапкам/файлам в `dir`, отсортированных по имени.
fn fs_read_dirs(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    entries.sort();
    Ok(entries)
}
