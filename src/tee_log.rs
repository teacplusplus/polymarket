//! Универсальный «tee»-лог: дублирует консольный вывод в файл.
//!
//! Макросы [`tee_println!`] и [`tee_eprintln!`] форматируют строку один раз,
//! выводят её в `stdout`/`stderr` и пишут ту же строку в файл, на который
//! указывает [`TEE_LOG`]. Инициализация и закрытие файла — ответственность
//! вызывающего кода (обычно в точке входа режима).
//!
//! Если [`TEE_LOG`] ещё не инициализирован (`None`) — макросы работают как
//! обычный `println!`/`eprintln!`, просто без файловой копии.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
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

/// Открывает (или перезаписывает) файл `path`, кладёт его `BufWriter` в
/// [`TEE_LOG`] и пишет первую строку-маркер `«[<tag>] лог пишется в …»`.
/// Возвращает ошибку только если не удалось создать сам файл; директорию
/// создаём best-effort (`create_dir_all` без bail на ошибке — точно так же
/// раньше работал inline-код в точках входа режимов).
///
/// Идемпотентен в смысле «последний победил»: повторный вызов заменит
/// предыдущий писатель в `TEE_LOG`, prev `BufWriter` сдропается на месте
/// и сам флашнется. На практике вызывается один раз на процесс — в
/// точке входа конкретного режима (`run_sim_mode` для history_sim,
/// `AppMode::RealSim` ветка `main` для real_sim).
pub fn init_tee_log_file(path: &Path, tag: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = File::create(path)?;
    {
        let mut guard = TEE_LOG.lock().expect("TEE_LOG poisoned");
        *guard = Some(BufWriter::new(file));
    }
    crate::tee_println!("[{tag}] лог пишется в {}", path.display());
    Ok(())
}

/// Флашит и закрывает писатель в [`TEE_LOG`], если он был открыт.
/// Используется в финале однократных режимов (history_sim), где
/// контролируемое закрытие даёт гарантию, что хвост лога ушёл на диск
/// до выхода из `main`. Для долгоживущих режимов (real_sim) не нужен —
/// `BufWriter` флашится в Drop статика при штатном выходе процесса.
pub fn finish_tee_log() {
    if let Ok(mut guard) = TEE_LOG.lock() {
        if let Some(mut w) = guard.take() {
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
