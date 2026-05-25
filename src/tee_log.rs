//! Универсальный «tee»-лог: дублирует консольный вывод в файл.
//!
//! Макросы [`tee_println!`] и [`tee_eprintln!`] форматируют строку один раз,
//! выводят её в `stdout`/`stderr` и пишут ту же строку в файл, на который
//! указывает [`TEE_LOG`]. Инициализация и закрытие файла — ответственность
//! вызывающего кода (обычно в точке входа режима).
//!
//! Если [`TEE_LOG`] ещё не инициализирован (`None`) — [`tee_println!`]/[`tee_eprintln!`]
//! работают как обычный `println!`/`eprintln!`, просто без файловой копии.
//!
//! Если [`STREAM_TEE_LOG`], [`USER_STREAM_TEE_LOG`], [`SIM_STATS_TEE_LOG`] или [`TEST_TEE_LOG`]
//! не открыты через `init_*` — соответствующие макросы не пишут в файл (форматирование
//! строки всё равно выполняется).

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
pub fn init_tee_log_file(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = File::create(path)?;
    {
        let mut guard = TEE_LOG.lock().expect("TEE_LOG poisoned");
        *guard = Some(BufWriter::new(file));
    }
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

// ---------------------------------------------------------------------------
// Отдельный «stream» tee-канал для observability-логов (live-тесты и т.п.).
//
// Идея — параллельный писатель, который **не** дублирует вывод в stdout/stderr
// и **не** шарится с основным [`TEE_LOG`]: подробные `[order_invoke/...]`
// логи пишутся ТОЛЬКО в файл, заданный [`init_stream_tee_log_file`]. Это держит
// прод-вывод чистым и одновременно даёт полный sequence событий для разбора.
// ---------------------------------------------------------------------------

/// Параллельный файловый писатель для stream-логов; не пересекается с [`TEE_LOG`].
/// Семантика идентична: `Mutex<Option<BufWriter<File>>>`, `None` = «не инициализирован,
/// все записи через [`stream_tee_log_write`] становятся no-op».
pub static STREAM_TEE_LOG: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Пишет одну строку в [`STREAM_TEE_LOG`] (если файл инициализирован) и сразу флашит.
/// Используется внутри [`stream_tee_println!`]/[`stream_tee_eprintln!`]; в stdout/stderr
/// **не** дублирует (в отличие от [`tee_log_write`]) — это отдельный stream-канал.
pub fn stream_tee_log_write(line: &str) {
    if let Ok(mut guard) = STREAM_TEE_LOG.lock() {
        if let Some(w) = guard.as_mut() {
            let _ = writeln!(w, "{}", line);
            let _ = w.flush();
        }
    }
}

/// Аналог [`init_tee_log_file`] для [`STREAM_TEE_LOG`]: открывает (или
/// перезаписывает) `path`, кладёт его `BufWriter` в [`STREAM_TEE_LOG`] и пишет
/// первую строку-маркер «[<tag>] stream-log пишется в …». Тот же контракт «последний
/// победил»; на практике вызывается один раз перед размещением ордеров.
pub fn init_stream_tee_log_file(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = File::create(path)?;
    {
        let mut guard = STREAM_TEE_LOG.lock().expect("STREAM_TEE_LOG poisoned");
        *guard = Some(BufWriter::new(file));
    }
    Ok(())
}

/// Флашит и закрывает писатель в [`STREAM_TEE_LOG`], если он был открыт. Полезно в
/// конце прогона, чтобы гарантировать, что хвост лога ушёл на диск до выхода.
pub fn finish_stream_tee_log() {
    if let Ok(mut guard) = STREAM_TEE_LOG.lock() {
        if let Some(mut w) = guard.take() {
            let _ = w.flush();
        }
    }
}

/// Записывает форматированную строку **только** в [`STREAM_TEE_LOG`] (если открыт);
/// в stdout не пишет — в отличие от [`tee_println!`]. Если файл не инициализирован,
/// макрос становится почти no-op (формат строки выполнится, запись будет проглочена).
#[macro_export]
macro_rules! stream_tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::stream_tee_log_write(&__line);
    }};
}

/// Записывает форматированную строку **только** в [`STREAM_TEE_LOG`] (если открыт);
/// в stderr не пишет — в отличие от [`tee_eprintln!`]. Используется для отметки
/// неуспешных WS/HTTP веток внутри stream-трассы.
#[macro_export]
macro_rules! stream_tee_eprintln {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::stream_tee_log_write(&__line);
    }};
}

// ---------------------------------------------------------------------------
// Отдельный «user-stream» tee-канал для user-WS и CLOB heartbeat
// ([`crate::account_ws`] `[user_ws] …`, [`crate::account`] `[heartbeat] …`):
// только файл, без stdout/stderr — как [`STREAM_TEE_LOG`], но свой путь
// (`xframes/last_user_stream.txt` в live/submit-режимах).
// ---------------------------------------------------------------------------

/// Параллельный файловый писатель для user-WS логов; не пересекается с [`TEE_LOG`] и [`STREAM_TEE_LOG`].
pub static USER_STREAM_TEE_LOG: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Пишет одну строку в [`USER_STREAM_TEE_LOG`] (если файл инициализирован) и сразу флашит.
pub fn user_stream_tee_log_write(line: &str) {
    if let Ok(mut guard) = USER_STREAM_TEE_LOG.lock() {
        if let Some(w) = guard.as_mut() {
            let _ = writeln!(w, "{}", line);
            let _ = w.flush();
        }
    }
}

/// Аналог [`init_stream_tee_log_file`] для [`USER_STREAM_TEE_LOG`]: маркер
/// «[<tag>] user-stream-log пишется в …».
pub fn init_user_stream_tee_log_file(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = File::create(path)?;
    {
        let mut guard = USER_STREAM_TEE_LOG.lock().expect("USER_STREAM_TEE_LOG poisoned");
        *guard = Some(BufWriter::new(file));
    }
    Ok(())
}

/// Флашит и закрывает писатель в [`USER_STREAM_TEE_LOG`], если он был открыт.
pub fn finish_user_stream_tee_log() {
    if let Ok(mut guard) = USER_STREAM_TEE_LOG.lock() {
        if let Some(mut w) = guard.take() {
            let _ = w.flush();
        }
    }
}

/// Записывает строку **только** в [`USER_STREAM_TEE_LOG`] (если открыт); в stdout не пишет.
#[macro_export]
macro_rules! user_stream_tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::user_stream_tee_log_write(&__line);
    }};
}

/// Записывает строку **только** в [`USER_STREAM_TEE_LOG`] (если открыт); в stderr не пишет.
#[macro_export]
macro_rules! user_stream_tee_eprintln {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::user_stream_tee_log_write(&__line);
    }};
}

// ---------------------------------------------------------------------------
// Отдельный «sim-stats» tee-канал: снимки [`crate::sim_stats::SimStats`] при закрытии
// позиций в real_sim / real_sim_with_submit (`xframes/last_sim_stats.txt`).
// ---------------------------------------------------------------------------

/// Файловый писатель для sim-stats снимков; не пересекается с другими tee-каналами.
pub static SIM_STATS_TEE_LOG: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Пишет одну строку в [`SIM_STATS_TEE_LOG`] (если файл инициализирован) и сразу флашит.
pub fn sim_stats_tee_log_write(line: &str) {
    if let Ok(mut guard) = SIM_STATS_TEE_LOG.lock() {
        if let Some(w) = guard.as_mut() {
            let _ = writeln!(w, "{}", line);
            let _ = w.flush();
        }
    }
}

/// `true`, если [`SIM_STATS_TEE_LOG`] открыт (real_sim / real_sim_with_submit).
pub fn sim_stats_tee_log_is_open() -> bool {
    SIM_STATS_TEE_LOG
        .lock()
        .ok()
        .is_some_and(|g| g.is_some())
}

/// Аналог [`init_stream_tee_log_file`] для [`SIM_STATS_TEE_LOG`]: маркер
/// «[<tag>] sim-stats-log пишется в …».
pub fn init_sim_stats_tee_log_file(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = File::create(path)?;
    {
        let mut guard = SIM_STATS_TEE_LOG.lock().expect("SIM_STATS_TEE_LOG poisoned");
        *guard = Some(BufWriter::new(file));
    }
    Ok(())
}

/// Флашит и закрывает писатель в [`SIM_STATS_TEE_LOG`], если он был открыт.
pub fn finish_sim_stats_tee_log() {
    if let Ok(mut guard) = SIM_STATS_TEE_LOG.lock() {
        if let Some(mut w) = guard.take() {
            let _ = w.flush();
        }
    }
}

/// Записывает строку **только** в [`SIM_STATS_TEE_LOG`] (если открыт); в stdout не пишет.
#[macro_export]
macro_rules! sim_stats_tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::sim_stats_tee_log_write(&__line);
    }};
}

// ---------------------------------------------------------------------------
// Отдельный «test» tee-канал (unit / интеграционные сценарии): только файл,
// без stdout/stderr — как [`STREAM_TEE_LOG`], но с собственным файлом и маркером.
// ---------------------------------------------------------------------------

/// Файловый писатель для test-only логов; не пересекается с [`TEE_LOG`] и [`STREAM_TEE_LOG`].
pub static TEST_TEE_LOG: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Пишет одну строку в [`TEST_TEE_LOG`] (если файл инициализирован) и сразу флашит.
pub fn test_tee_log_write(line: &str) {
    if let Ok(mut guard) = TEST_TEE_LOG.lock() {
        if let Some(w) = guard.as_mut() {
            let _ = writeln!(w, "{}", line);
            let _ = w.flush();
        }
    }
}

/// Аналог [`init_stream_tee_log_file`] для [`TEST_TEE_LOG`]: маркер «[<tag>] test-log пишется в …».
pub fn init_test_tee_log_file(path: &Path, tag: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = File::create(path)?;
    {
        let mut guard = TEST_TEE_LOG.lock().expect("TEST_TEE_LOG poisoned");
        *guard = Some(BufWriter::new(file));
    }
    crate::test_tee_println!("[{tag}] test-log пишется в {}", path.display());
    Ok(())
}

/// Закрывает писатель [`TEST_TEE_LOG`], флаш перед снятием.
pub fn finish_test_tee_log() {
    if let Ok(mut guard) = TEST_TEE_LOG.lock() {
        if let Some(mut w) = guard.take() {
            let _ = w.flush();
        }
    }
}

/// Записывает строку **только** в [`TEST_TEE_LOG`] (если открыт); в stdout не пишет.
#[macro_export]
macro_rules! test_tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::test_tee_log_write(&__line);
    }};
}

/// Записывает строку **только** в [`TEST_TEE_LOG`] (если открыт); в stderr не пишет.
#[macro_export]
macro_rules! test_tee_eprintln {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::test_tee_log_write(&__line);
    }};
}
