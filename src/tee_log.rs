//! Универсальный «tee»-лог: дублирует консольный вывод в файл.
//!
//! Макросы [`tee_println!`] и [`tee_eprintln!`] форматируют строку один раз,
//! выводят её в `stdout`/`stderr` и пишут ту же строку в файл, на который
//! указывает [`TEE_LOG`]. Инициализация и закрытие файла — ответственность
//! вызывающего кода (обычно в точке входа режима).
//!
//! Если [`TEE_LOG`] ещё не инициализирован (`None`) — макросы работают как
//! обычный `println!`/`eprintln!`, просто без файловой копии.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

/// Файловый писатель для дублирования консольного вывода.
/// `const`-инициализация через [`Mutex::new`] — без внешних крейтов.
pub static TEE_LOG: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Пишет одну строку в [`TEE_LOG`] (если файл инициализирован) и сразу флашит.
/// Используется внутри [`tee_println!`]/[`tee_eprintln!`].
pub fn tee_log_write(line: &str) {
    if let Ok(mut guard) = TEE_LOG.lock() {
        if let Some(w) = guard.as_mut() {
            let _ = writeln!(w, "{}", line);
            let _ = w.flush();
        }
    }
}

/// `println!`, который дополнительно пишет ту же строку в [`TEE_LOG`].
#[macro_export]
macro_rules! tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        println!("{}", __line);
        $crate::tee_log::tee_log_write(&__line);
    }};
}

/// `eprintln!`, который дополнительно пишет ту же строку в [`TEE_LOG`].
#[macro_export]
macro_rules! tee_eprintln {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        eprintln!("{}", __line);
        $crate::tee_log::tee_log_write(&__line);
    }};
}
