//! Линковка с локально собранным libxgboost.so (CUDA/CPU).
//! `-lxgboost` резолвится в системный CPU-only libxgboost.so.0 из apt.
//!
//! Cargo при `run`/`test` добавляет `target/<profile>/deps` в `LD_LIBRARY_PATH`.
//! Там часто остаётся старый prebuilt libxgboost.so (~15 MB, CPU-only) — он перебивает
//! наш CUDA-бинарник, если линковать через DT_RUNPATH. Поэтому: DT_RPATH + symlink в deps.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn main() {
    // Вшиваем версию сборки (время компиляции + git) в бинарь. Печатается на старте программы,
    // чтобы сразу видеть, что запущен свежий бинарь, а не устаревшая копия (см. инцидент с
    // CARGO_TARGET_DIR: собиралось в sandbox-кэш, а запускалась старая ./target/release/poly).
    // ВАЖНО: делаем это ДО раннего return ниже — иначе без фичи xgb-local переменные не вшьются.
    // Сознательно НЕ эмитим `cargo:rerun-if-changed`: тогда сохраняется дефолтное поведение
    // (build.rs перезапускается при изменении любого файла пакета), и POLY_BUILD_UTC совпадает
    // с реальным моментом пересборки бинаря.
    emit_build_version();

    if env::var("CARGO_FEATURE_XGB_LOCAL").is_err() {
        return;
    }

    let Some(lib_dir) = local_xgboost_lib_dir() else {
        return;
    };
    let lib_path = lib_dir.join("libxgboost.so");
    if !lib_path.exists() {
        println!(
            "cargo:warning=local xgboost library not found at {} (will appear after first xgboost-sys build)",
            lib_path.display()
        );
        return;
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-arg={}", lib_path.display());
    // RPATH (не RUNPATH): ищется до LD_LIBRARY_PATH, иначе cargo run/test подхватит
    // stale prebuilt из target/<profile>/deps/libxgboost.so.
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    if let Err(err) = sync_deps_xgboost_symlink(&lib_path) {
        println!("cargo:warning=failed to sync deps/libxgboost.so symlink: {err}");
    }
}

/// Вшивает в бинарь компайл-тайм переменные версии через `cargo:rustc-env` (читаются в
/// рантайме макросом `env!`): время компиляции в UTC, короткий git-хэш и флаг «грязного»
/// рабочего дерева. git-вызовы best-effort — при отсутствии git значения = "unknown".
fn emit_build_version() {
    let build_utc = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();
    println!("cargo:rustc-env=POLY_BUILD_UTC={build_utc}");

    let git_hash =
        run_git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = match run_git(&["status", "--porcelain"]) {
        Some(s) if !s.trim().is_empty() => "-dirty",
        _ => "",
    };
    println!("cargo:rustc-env=POLY_GIT_HASH={git_hash}{dirty}");
}

/// Запускает `git <args>` в каталоге пакета и возвращает trimmed stdout при успехе.
fn run_git(args: &[&str]) -> Option<String> {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Заменяет `target/<profile>/deps/libxgboost.so` симлинком на локальную CUDA/CPU-сборку.
fn sync_deps_xgboost_symlink(correct_lib: &Path) -> io::Result<()> {
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("target"));
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let deps_lib = target_dir.join(profile).join("deps").join("libxgboost.so");
    let Some(deps_dir) = deps_lib.parent() else {
        return Ok(());
    };
    if !deps_dir.exists() {
        return Ok(());
    }

    // Уже указывает на нужный файл.
    if fs::read_link(&deps_lib).is_ok_and(|p| p == correct_lib) {
        return Ok(());
    }

    let _ = fs::remove_file(&deps_lib);
    std::os::unix::fs::symlink(correct_lib, &deps_lib)
}

fn local_xgboost_lib_dir() -> Option<PathBuf> {
    let base = env::var("XGBOOST_BUILD_CACHE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            env::var("HOME").ok().map(|home| {
                PathBuf::from(home)
                    .join(".cache")
                    .join("poly-xgboost-build")
            })
        })?;

    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".into());
    // Вариант кэша должен совпадать с xgboost-sys/build.rs: «cuda» только если
    // фича включена И nvcc реально найден, иначе там собирается CPU-сборка в .../cpu/.
    let variant = if env::var("CARGO_FEATURE_CUDA").is_ok() && cuda_toolkit_root().is_some() {
        "cuda"
    } else {
        "cpu"
    };
    let lib_dir = base.join(variant).join(profile).join("lib");
    if lib_dir.exists() {
        Some(lib_dir)
    } else {
        None
    }
}

/// Найден ли nvcc — повторяет логику `xgboost-sys/build.rs::cuda_toolkit_root`,
/// чтобы выбор cuda/cpu-варианта кэша совпадал с реальной сборкой XGBoost.
fn cuda_toolkit_root() -> Option<String> {
    if let Ok(p) = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .or_else(|_| env::var("CUDA_ROOT"))
    {
        if Path::new(&p).join("bin/nvcc").exists() {
            return Some(p);
        }
    }

    for candidate in [
        "/usr/lib/nvidia-cuda-toolkit",
        "/usr/local/cuda",
        "/usr/lib/cuda",
    ] {
        if Path::new(candidate).join("bin/nvcc").exists() {
            return Some(candidate.to_string());
        }
    }

    if Path::new("/usr/bin/nvcc").exists() {
        return Some("/usr/lib/nvidia-cuda-toolkit".to_string());
    }

    None
}
