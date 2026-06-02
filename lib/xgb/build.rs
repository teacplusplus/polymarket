use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(xgb_cuda_built)");

    if !cfg!(feature = "cuda") {
        return;
    }

    if cuda_toolkit_root().is_none() {
        println!(
            "cargo:warning=CUDA feature enabled but nvcc not found; \
             XGBoost will be CPU-only (install nvidia-cuda-toolkit for GPU training)"
        );
        return;
    }

    // Не линкуем cudart: libxgboost.so уже содержит cudart_static, второй экземпляр
    // в Rust-бинарнике ломает инициализацию CUDA в XGBoost (device=cuda → CPU fallback).
    println!("cargo:rustc-cfg=xgb_cuda_built");
}

/// Корень CUDA toolkit, если `nvcc` доступен.
fn cuda_toolkit_root() -> Option<String> {
    if let Ok(p) = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .or_else(|_| env::var("CUDA_ROOT"))
    {
        let root = PathBuf::from(&p);
        if root.join("bin/nvcc").exists() || Path::new("/usr/bin/nvcc").exists() {
            return Some(p);
        }
    }

    for candidate in [
        "/usr/lib/nvidia-cuda-toolkit",
        "/usr/local/cuda",
        "/usr/lib/cuda",
    ] {
        let root = Path::new(candidate);
        if root.join("bin/nvcc").exists() {
            return Some(candidate.to_string());
        }
    }

    if Path::new("/usr/bin/nvcc").exists() {
        return Some("/usr/lib/nvidia-cuda-toolkit".to_string());
    }

    None
}
