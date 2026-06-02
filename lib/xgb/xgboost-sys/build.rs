use bindgen;
use std::env;
use std::path::{Path, PathBuf};

const GITHUB_URL: &str = "https://github.com/marcomq/rust-xgboost/raw/refs/tags/v3.0.1/xgboost-sys/lib/";

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let xgb_root = Path::new("xgboost").canonicalize().unwrap();

    let wrapper_h = xgb_root.join("include").join("xgboost").join("c_api.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper_h.to_string_lossy())
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!("-I{}", xgb_root.join("dmlc-core").join("include").display()));

    #[cfg(feature = "cuda")]
    let bindings = {
        if let Some(cuda) = cuda_toolkit_root() {
            bindings.clang_arg(format!("-I{}/include", cuda))
        } else {
            bindings
        }
    };
    let bindings = bindings.generate().expect("Unable to generate bindings.");

    let out_path = PathBuf::from(&out_dir);
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    if target.contains("apple") {
        println!(
            "cargo:rustc-link-search=native={}/opt/libomp/lib",
            &std::env::var("HOMEBREW_PREFIX").unwrap_or("/opt/homebrew".into())
        );
    }

    #[cfg(feature = "use_prebuilt_xgb")]
    {
        if let Ok(xgboost_lib_dir) = std::env::var("XGBOOST_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
        } else {
            let deps_path = dunce::canonicalize(Path::new(&format!("{}/../../../deps", out_dir))).unwrap();
            let deps_path = deps_path.to_string_lossy();
            println!("cargo:rustc-link-search=native={}", deps_path);
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                let path = format!("{GITHUB_URL}/mac_arm64");
                if !std::fs::exists(format!("{deps_path}/libxgboost.dylib")).unwrap() {
                    web_copy(
                        &format!("{path}/libxgboost.dylib"),
                        &format!("{deps_path}/libxgboost.dylib"),
                    )
                    .unwrap();
                    web_copy(&format!("{path}/libdmlc.a"), &format!("{deps_path}/libdmlc.a")).unwrap();
                }
            } else if cfg!(target_os = "linux") {
                let path = if cfg!(target_arch = "aarch64") {
                    format!("{GITHUB_URL}/linux_arm64")
                } else {
                    format!("{GITHUB_URL}/linux_amd64")
                };
                if !std::fs::exists(format!("{deps_path}/libxgboost.so")).unwrap() {
                    web_copy(&format!("{path}/libxgboost.so"), &format!("{deps_path}/libxgboost.so")).unwrap();
                    web_copy(&format!("{path}/libdmlc.a"), &format!("{deps_path}/libdmlc.a")).unwrap();
                }
            } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
                let path = format!("{GITHUB_URL}/win_amd64");
                if !std::fs::exists(format!("{deps_path}/xgboost.dll")).unwrap() {
                    web_copy(&format!("{path}/xgboost.dll"), &format!("{deps_path}/xgboost.dll")).unwrap();
                    web_copy(&format!("{path}/xgboost.lib"), &format!("{deps_path}/xgboost.lib")).unwrap();
                }
            } else {
                if let Ok(homebrew_path) = std::env::var("HOMEBREW_PREFIX") {
                    let xgboost_lib_dir = format!("{}/opt/xgboost/lib", &homebrew_path);
                    println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
                } else {
                    panic!("Please set $XGBOOST_LIB_DIR")
                }
            }
        }
    }

    #[cfg(feature = "local_build")]
    {
        // compile XGBOOST with cmake and ninja

        let cache_dir = xgb_persistent_cache_dir();
        let _ = std::fs::create_dir_all(&cache_dir);
        println!("cargo:warning=XGBoost cmake cache: {}", cache_dir.display());

        // CMake
        let mut cfg = cmake::Config::new(&xgb_root);
        cfg.out_dir(&cache_dir)
            .generator("Ninja")
            .define("CMAKE_BUILD_TYPE", "RelWithDebInfo");

        // Ускоряет повторные пересборки C++/CUDA (apt install ccache).
        if Path::new("/usr/bin/ccache").exists() {
            cfg.define("CMAKE_CXX_COMPILER_LAUNCHER", "ccache")
                .define("CMAKE_CUDA_COMPILER_LAUNCHER", "ccache");
        }

        #[cfg(feature = "cuda")]
        if let Some(cuda) = cuda_toolkit_root() {
            println!("cargo:warning=Building XGBoost with CUDA (toolkit root: {cuda})");
            cfg.define("USE_CUDA", "ON")
                .define("BUILD_WITH_CUDA", "ON")
                .define("BUILD_WITH_CUDA_CUB", "ON")
                .define("CUDAToolkit_ROOT", &cuda);
            if let Some(host) = cuda_host_cxx_compiler() {
                println!("cargo:warning=Using {host} as CUDA host compiler");
                cfg.define("CMAKE_CUDA_HOST_COMPILER", host);
            }
        } else {
            #[cfg(feature = "cuda")]
            println!(
                "cargo:warning=CUDA feature enabled but nvcc not found; \
                 building CPU-only XGBoost (install nvidia-cuda-toolkit for GPU)"
            );
        }

        let dst = cfg.build();

        let lib_dir = dst.join("lib");
        let lib64_dir = dst.join("lib64");
        println!("cargo:rustc-link-search=native={}", dst.display());
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
        if lib64_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib64_dir.display());
        }
        println!("cargo:rustc-link-lib=static=dmlc");
    }

    // link to appropriate C++ lib
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=omp");
    } else {
        #[cfg(target_os = "linux")]
        {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=stdc++fs");
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }

    #[cfg(not(feature = "local_build"))]
    {
        println!("cargo:rustc-link-lib=dylib=xgboost");
    }
}

/// Постоянный каталог cmake-сборки XGBoost (не удаляется при `cargo clean`).
/// Переопределение: `XGBOOST_BUILD_CACHE=/path/to/dir`.
#[cfg(feature = "local_build")]
fn xgb_persistent_cache_dir() -> PathBuf {
    let base = env::var("XGBOOST_BUILD_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".cache").join("poly-xgboost-build")
        });
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".into());
    #[cfg(feature = "cuda")]
    let variant = if cuda_toolkit_root().is_some() {
        "cuda"
    } else {
        "cpu"
    };
    #[cfg(not(feature = "cuda"))]
    let variant = "cpu";
    base.join(variant).join(profile)
}

/// Корень CUDA toolkit, если `nvcc` доступен.
#[cfg(feature = "cuda")]
fn cuda_toolkit_root() -> Option<String> {
    if let Ok(p) = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .or_else(|_| env::var("CUDA_ROOT"))
    {
        let root = Path::new(&p);
        if root.join("bin/nvcc").exists() {
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

/// CUDA 12 из apt не поддерживает GCC 14; g++-12 обычно есть в Ubuntu 24.04.
#[cfg(feature = "cuda")]
fn cuda_host_cxx_compiler() -> Option<&'static str> {
    for candidate in ["/usr/bin/g++-12", "/usr/bin/g++-11"] {
        if Path::new(candidate).exists() {
            return Some(candidate);
        }
    }
    None
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "use_prebuilt_xgb")]
fn web_copy(web_src: &str, target: &str) -> Result<()> {
    dbg!(&web_src);
    let resp = reqwest::blocking::get(web_src)?;
    let body = resp.bytes()?;
    std::fs::write(target, &body)?;
    Ok(())
}
