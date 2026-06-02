//! Универсальный «tee»-лог: дублирует консольный вывод в файл.
//!
//! Макросы [`tee_println!`] и [`tee_eprintln!`] форматируют строку один раз,
//! выводят её в `stdout`/`stderr` и ставят ту же строку в очередь на запись
//! в файл, на который указывает соответствующий канал. Инициализация и закрытие
//! файла — ответственность вызывающего кода (обычно в точке входа режима).
//!
//! Запись на диск выполняется фоновой задачей (`tokio::spawn`): продюсеры
//! только отправляют строки в `mpsc`, писатель при появлении данных хотя бы в
//! одном канале сливает **все** накопившиеся строки из **всех** активных каналов
//! в соответствующие файлы.
//!
//! Если канал ещё не инициализирован — [`tee_println!`]/[`tee_eprintln!`]
//! работают как обычный `println!`/`eprintln!`, без файловой копии.
//! Stream/user/sim-stats/test макросы без `init_*` — no-op на запись в файл.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use tokio::sync::mpsc;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum TeeKind {
    Main,
    Stream,
    UserStream,
    SimStats,
    Test,
    TradeCsv,
    SubmitTradeCsv,
}

enum WriterMsg {
    Register {
        kind: TeeKind,
        rx: mpsc::UnboundedReceiver<String>,
        path: PathBuf,
        ack: std::sync::mpsc::Sender<()>,
    },
    Close {
        kind: TeeKind,
        ack: std::sync::mpsc::Sender<()>,
    },
}

struct ChannelSlot {
    kind: TeeKind,
    rx: mpsc::UnboundedReceiver<String>,
    writer: BufWriter<File>,
}

struct TeeChannel {
    sender: Mutex<Option<mpsc::UnboundedSender<String>>>,
}

impl TeeChannel {
    const fn new() -> Self {
        Self {
            sender: Mutex::new(None),
        }
    }

    fn send_line(&self, line: &str) {
        let tx = match self.sender.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => return,
        };
        if let Some(tx) = tx {
            if tx.send(line.to_owned()).is_ok() {
                writer_wake();
            }
        }
    }

    fn set_sender(&self, tx: mpsc::UnboundedSender<String>) {
        if let Ok(mut guard) = self.sender.lock() {
            *guard = Some(tx);
        }
    }

    fn take_sender(&self) -> Option<mpsc::UnboundedSender<String>> {
        self.sender.lock().ok().and_then(|mut g| g.take())
    }
}

static WRITER_CMD_TX: OnceLock<mpsc::UnboundedSender<WriterMsg>> = OnceLock::new();
static WRITER_WAKE: OnceLock<tokio::sync::Notify> = OnceLock::new();

static TEE_CHANNEL: TeeChannel = TeeChannel::new();
static STREAM_CHANNEL: TeeChannel = TeeChannel::new();
static USER_STREAM_CHANNEL: TeeChannel = TeeChannel::new();
static SIM_STATS_CHANNEL: TeeChannel = TeeChannel::new();
static TEST_CHANNEL: TeeChannel = TeeChannel::new();
static TRADE_CSV_CHANNEL: TeeChannel = TeeChannel::new();
static SUBMIT_TRADE_CSV_CHANNEL: TeeChannel = TeeChannel::new();

fn writer_wake() {
    if let Some(n) = WRITER_WAKE.get() {
        n.notify_one();
    }
}

fn channel_for(kind: TeeKind) -> &'static TeeChannel {
    match kind {
        TeeKind::Main => &TEE_CHANNEL,
        TeeKind::Stream => &STREAM_CHANNEL,
        TeeKind::UserStream => &USER_STREAM_CHANNEL,
        TeeKind::SimStats => &SIM_STATS_CHANNEL,
        TeeKind::Test => &TEST_CHANNEL,
        TeeKind::TradeCsv => &TRADE_CSV_CHANNEL,
        TeeKind::SubmitTradeCsv => &SUBMIT_TRADE_CSV_CHANNEL,
    }
}

fn ensure_writer() {
    WRITER_CMD_TX.get_or_init(|| {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let _ = WRITER_WAKE.set(tokio::sync::Notify::new());
        tokio::spawn(tee_log_writer_loop(cmd_rx));
        cmd_tx
    });
}

fn writer_cmd() -> &'static mpsc::UnboundedSender<WriterMsg> {
    ensure_writer();
    WRITER_CMD_TX.get().expect("tee_log writer cmd_tx")
}

fn register_channel(kind: TeeKind, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    File::create(path)?;
    let (tx, rx) = mpsc::unbounded_channel();
    let (ack_tx, ack_rx) = std::sync::mpsc::channel();
    writer_cmd()
        .send(WriterMsg::Register {
            kind,
            rx,
            path: path.to_path_buf(),
            ack: ack_tx,
        })
        .map_err(|_| std::io::Error::other("tee_log writer task stopped"))?;
    ack_rx
        .recv_timeout(Duration::from_secs(30))
        .map_err(|_| std::io::Error::other("tee_log writer register timeout"))?;
    channel_for(kind).set_sender(tx);
    Ok(())
}

fn close_channel(kind: TeeKind) {
    if channel_for(kind).take_sender().is_none() {
        return;
    }
    let (ack_tx, ack_rx) = std::sync::mpsc::channel();
    let _ = writer_cmd().send(WriterMsg::Close {
        kind,
        ack: ack_tx,
    });
    let _ = ack_rx.recv_timeout(Duration::from_secs(30));
}

fn drain_slot(slot: &mut ChannelSlot) {
    while let Ok(line) = slot.rx.try_recv() {
        let _ = writeln!(slot.writer, "{}", line);
    }
}

fn flush_slot(slot: &mut ChannelSlot) {
    let _ = slot.writer.flush();
}

async fn tee_log_writer_loop(mut cmd_rx: mpsc::UnboundedReceiver<WriterMsg>) {
    let mut channels: Vec<ChannelSlot> = Vec::new();
    let wake = WRITER_WAKE.get().expect("WRITER_WAKE");

    loop {
        while let Ok(msg) = cmd_rx.try_recv() {
            match msg {
                WriterMsg::Register { kind, rx, path, ack } => {
                    channels.retain(|slot| slot.kind != kind);
                    match File::create(&path) {
                        Ok(file) => {
                            channels.push(ChannelSlot {
                                kind,
                                rx,
                                writer: BufWriter::new(file),
                            });
                        }
                        Err(e) => {
                            eprintln!("tee_log: не удалось открыть {}: {e}", path.display());
                        }
                    }
                    let _ = ack.send(());
                }
                WriterMsg::Close { kind, ack } => {
                    if let Some(idx) = channels.iter().position(|s| s.kind == kind) {
                        let mut slot = channels.remove(idx);
                        drain_slot(&mut slot);
                        flush_slot(&mut slot);
                    }
                    let _ = ack.send(());
                }
            }
        }

        let mut wrote_any = false;
        for slot in &mut channels {
            while let Ok(line) = slot.rx.try_recv() {
                let _ = writeln!(slot.writer, "{}", line);
                wrote_any = true;
            }
        }
        if wrote_any {
            for slot in &mut channels {
                flush_slot(slot);
            }
            continue;
        }

        tokio::select! {
            msg = cmd_rx.recv() => {
                if let Some(msg) = msg {
                    match msg {
                        WriterMsg::Register { kind, rx, path, ack } => {
                            channels.retain(|slot| slot.kind != kind);
                            if let Ok(file) = File::create(&path) {
                                channels.push(ChannelSlot { kind, rx, writer: BufWriter::new(file) });
                            }
                            let _ = ack.send(());
                        }
                        WriterMsg::Close { kind, ack } => {
                            if let Some(idx) = channels.iter().position(|s| s.kind == kind) {
                                let mut slot = channels.remove(idx);
                                drain_slot(&mut slot);
                                flush_slot(&mut slot);
                            }
                            let _ = ack.send(());
                        }
                    }
                } else {
                    break;
                }
            }
            () = wake.notified() => {}
        }
    }
}

/// Пишет одну строку в очередь основного tee-канала (если инициализирован).
pub fn tee_log_write(line: &str) {
    TEE_CHANNEL.send_line(line);
}

/// Обновляет строку прогресса на месте (без `\n`, в tee-файл не пишет).
pub fn tee_progress_update(line: &str) {
    use std::io::{IsTerminal, Write};
    if !std::io::stdout().is_terminal() {
        return;
    }
    print!("\r{line}\x1b[K");
    let _ = std::io::stdout().flush();
}

/// Завершает in-place прогресс — следующий вывод идёт с новой строки.
pub fn tee_progress_finish() {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        println!();
    }
}

/// Открывает (или перезаписывает) файл `path` и регистрирует основной tee-канал.
pub fn init_tee_log_file(path: &Path) -> std::io::Result<()> {
    register_channel(TeeKind::Main, path)
}

/// Сливает очередь и закрывает основной tee-канал.
pub fn finish_tee_log() {
    close_channel(TeeKind::Main);
}

/// `println!`, который дополнительно пишет ту же строку в tee-файл.
#[macro_export]
macro_rules! tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        println!("{}", __line);
        $crate::tee_log::tee_log_write(&__line);
    }};
}

/// In-place прогресс в терминале (`\r`, без tee).
#[macro_export]
macro_rules! tee_progress {
    ($($arg:tt)*) => {{
        $crate::tee_log::tee_progress_update(&format!($($arg)*));
    }};
}

/// `eprintln!`, который дополнительно пишет ту же строку в tee-файл.
#[macro_export]
macro_rules! tee_eprintln {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        eprintln!("{}", __line);
        $crate::tee_log::tee_log_write(&__line);
    }};
}

// ---------------------------------------------------------------------------
// Stream tee (observability, без stdout/stderr)
// ---------------------------------------------------------------------------

pub fn stream_tee_log_write(line: &str) {
    STREAM_CHANNEL.send_line(line);
}

pub fn init_stream_tee_log_file(path: &Path) -> std::io::Result<()> {
    register_channel(TeeKind::Stream, path)
}

pub fn finish_stream_tee_log() {
    close_channel(TeeKind::Stream);
}

#[macro_export]
macro_rules! stream_tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::stream_tee_log_write(&__line);
    }};
}

#[macro_export]
macro_rules! stream_tee_eprintln {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::stream_tee_log_write(&__line);
    }};
}

// ---------------------------------------------------------------------------
// User-stream tee
// ---------------------------------------------------------------------------

pub fn user_stream_tee_log_write(line: &str) {
    USER_STREAM_CHANNEL.send_line(line);
}

pub fn init_user_stream_tee_log_file(path: &Path) -> std::io::Result<()> {
    register_channel(TeeKind::UserStream, path)
}

pub fn finish_user_stream_tee_log() {
    close_channel(TeeKind::UserStream);
}

#[macro_export]
macro_rules! user_stream_tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::user_stream_tee_log_write(&__line);
    }};
}

#[macro_export]
macro_rules! user_stream_tee_eprintln {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::user_stream_tee_log_write(&__line);
    }};
}

// ---------------------------------------------------------------------------
// Sim-stats tee
// ---------------------------------------------------------------------------

pub fn sim_stats_tee_log_write(line: &str) {
    SIM_STATS_CHANNEL.send_line(line);
}

pub fn init_sim_stats_tee_log_file(path: &Path) -> std::io::Result<()> {
    register_channel(TeeKind::SimStats, path)
}

pub fn finish_sim_stats_tee_log() {
    close_channel(TeeKind::SimStats);
}

#[macro_export]
macro_rules! sim_stats_tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::sim_stats_tee_log_write(&__line);
    }};
}

// ---------------------------------------------------------------------------
// Test tee
// ---------------------------------------------------------------------------

pub fn test_tee_log_write(line: &str) {
    TEST_CHANNEL.send_line(line);
}

pub fn init_test_tee_log_file(path: &Path, tag: &str) -> std::io::Result<()> {
    register_channel(TeeKind::Test, path)?;
    crate::test_tee_println!("[{tag}] test-log пишется в {}", path.display());
    Ok(())
}

pub fn finish_test_tee_log() {
    close_channel(TeeKind::Test);
}

#[macro_export]
macro_rules! test_tee_println {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::test_tee_log_write(&__line);
    }};
}

#[macro_export]
macro_rules! test_tee_eprintln {
    ($($arg:tt)*) => {{
        let __line = format!($($arg)*);
        $crate::tee_log::test_tee_log_write(&__line);
    }};
}

// ---------------------------------------------------------------------------
// Per-trade CSV (history_sim / real_sim)
// ---------------------------------------------------------------------------

/// Ставит готовую CSV-строку в очередь `last_*_trades.csv` (если канал открыт).
pub fn trade_csv_log_write(line: &str) {
    TRADE_CSV_CHANNEL.send_line(line);
}

/// Открывает (или перезаписывает) trade-CSV файл.
pub fn init_trade_csv_log_file(path: &Path) -> std::io::Result<()> {
    register_channel(TeeKind::TradeCsv, path)
}

/// Сливает очередь и закрывает trade-CSV канал.
pub fn finish_trade_csv_log() {
    close_channel(TeeKind::TradeCsv);
}

// ---------------------------------------------------------------------------
// Submit-orders CSV (real_sim_with_submit)
// ---------------------------------------------------------------------------

/// Ставит готовую CSV-строку в очередь submit-CSV (если канал открыт).
pub fn submit_trade_csv_log_write(line: &str) {
    SUBMIT_TRADE_CSV_CHANNEL.send_line(line);
}

/// Открывает (или перезаписывает) submit-CSV файл.
pub fn init_submit_trade_csv_log_file(path: &Path) -> std::io::Result<()> {
    register_channel(TeeKind::SubmitTradeCsv, path)
}

/// Сливает очередь и закрывает submit-CSV канал.
pub fn finish_submit_trade_csv_log() {
    close_channel(TeeKind::SubmitTradeCsv);
}
