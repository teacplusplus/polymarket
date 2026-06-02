/// XGBoost собран с поддержкой CUDA (nvcc был доступен при сборке).
pub fn cuda_built() -> bool {
    cfg!(xgb_cuda_built)
}

/// CUDA доступна в рантайме: сборка с CUDA + видимая GPU + рабочий драйвер.
pub fn cuda_runtime_available() -> bool {
    cuda_built() && cuda_device_count() > 0
}

/// Значение параметра `device` для XGBoost (`cpu` или `cuda`).
pub fn preferred_device() -> &'static str {
    if cuda_runtime_available() {
        "cuda"
    } else {
        "cpu"
    }
}

/// Число видимых CUDA-устройств. Без CUDA-сборки или при ошибке драйвера — 0.
pub fn cuda_device_count() -> i32 {
    if !cuda_built() {
        return 0;
    }
    cuda_device_count_impl()
}

/// Driver API через `dlopen`, без линковки `cudart`.
/// Статический `cudart` в Rust-бинарнике конфликтует с `cudart_static` внутри libxgboost.so
/// и ломает `cudaGetDeviceCount` в XGBoost (device=cuda откатывается на CPU).
#[cfg(xgb_cuda_built)]
fn cuda_device_count_impl() -> i32 {
    use std::ffi::CString;

    unsafe extern "C" {
        fn dlopen(filename: *const libc::c_char, flag: libc::c_int) -> *mut libc::c_void;
        fn dlsym(handle: *mut libc::c_void, symbol: *const libc::c_char) -> *mut libc::c_void;
        fn dlclose(handle: *mut libc::c_void) -> libc::c_int;
    }

    const RTLD_LAZY: libc::c_int = 1;
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> i32;

    unsafe {
        let lib_name = CString::new("libcuda.so.1").unwrap();
        let handle = dlopen(lib_name.as_ptr(), RTLD_LAZY);
        if handle.is_null() {
            return 0;
        }

        let cu_init_name = CString::new("cuInit").unwrap();
        let cu_count_name = CString::new("cuDeviceGetCount").unwrap();
        let cu_init_ptr = dlsym(handle, cu_init_name.as_ptr());
        let cu_count_ptr = dlsym(handle, cu_count_name.as_ptr());
        if cu_init_ptr.is_null() || cu_count_ptr.is_null() {
            let _ = dlclose(handle);
            return 0;
        }
        let cu_init: CuInit = std::mem::transmute(cu_init_ptr);
        let cu_device_get_count: CuDeviceGetCount = std::mem::transmute(cu_count_ptr);

        if cu_init(0) != 0 {
            let _ = dlclose(handle);
            return 0;
        }

        let mut count: i32 = 0;
        let ok = cu_device_get_count(&mut count) == 0;
        let _ = dlclose(handle);
        if ok { count } else { 0 }
    }
}

#[cfg(not(xgb_cuda_built))]
fn cuda_device_count_impl() -> i32 {
    0
}
