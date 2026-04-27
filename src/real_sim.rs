//! Режим **реальной** симуляции (`STATUS=real_sim`).
//!
//! Зеркалит торговую логику [`crate::history_sim`], но работает по живому
//! потоку данных [`ProjectManager`]: фанаут
//! `build_frames_from_buffer_lane_once(lane=0)` пушит стабильные [`XFrame`]
//! в 4 `mpsc`-канала `(интервал, сторона)`, а 4 tokio-таска слушают свой
//! `Receiver` и на каждом полученном кадре вызывают [`tick_once`].
//!
//! # Контракт
//!
//! * Для каждой комбинации `интервал × сторона` (`5m`/`15m` × `up`/`down`)
//!   поднимается отдельный tokio-таск. Таск **не** использует таймер — он
//!   блокируется на `rx.recv().await` и реагирует на событие (новый стабильный
//!   кадр лейна 1 с), что убирает поллинг и рассинхрон с фанаутом.
//! * Четыре таска делят общее состояние через
//!   [`Arc<RwLock<RealSimState>>`]: внутри — два [`SimStats`] (по одному на
//!   интервал; банкролл / drawdown общий для UP+DOWN внутри интервала) и
//!   четыре `Vec<OpenPosition>` (по одному на `(интервал, сторона)`). Смена
//!   `market_id` для маршрута `(интервал, сторона)` хранится локально в таске;
//!   `stats.events` увеличивает только воркер `Up` на интервал (одно событие на
//!   пару UP/DOWN).
//! * Перед каждым тиком по HTTP снимается стакан Polymarket CLOB
//!   ([`clob::Client::order_book`]) и сравнивается с топ-уровнем WS-кадра.
//!   При расхождении > `frame.tick_size * 2` считаем, что WS **отстаёт**:
//!   печатаем ошибку, **не открываем** новые позиции в этот тик, но
//!   продолжаем вести закрытия ([`manage_positions`]) — выйти из позиции
//!   важнее, чем дождаться WS.
//! * После каждой сделки (buy/sell) — полный дамп статистики через
//!   [`print_sim_stats`] (идентично `history_sim`).
//!
//! # Источники данных (без новых кешей)
//!
//! * Кадры: [`LaneFrame`] из `lane_frame_channels` (фанаут лейна 1 с);
//!   `market_id`, `asset_id`, `frame` берутся прямо из кадра.
//! * Стакан HTTP: единый координатор [`run_book_coordinator`] батчит
//!   запросы 4 воркеров в один `clob.order_books(&[...])` — вместо 4
//!   независимых GET `/book` каждый воркер шлёт `BookRequest` в общий
//!   `mpsc` и ждёт ответ через персональный `oneshot`.

use crate::account::{Account, SharedAccount};
use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{
    BuyGate, any_position_would_sell, buy_gate, compute_p_win_now, compute_pnl_inference,
    load_booster, manage_positions, print_sim_stats, try_open_position, OpenPosition,
    SimStats, StrictBook,
};
use crate::xframe::BookLevel;
use crate::project_manager::{LaneFrame, ProjectManager};
use crate::train_mode::{load_calibration, Calibration};
use crate::util::current_timestamp_ms;
use crate::xframe::{XFrame, SIZE};

use anyhow::{anyhow, Result};
use futures_util::FutureExt;
use indexmap::IndexSet;
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::response::OrderBookSummaryResponse;
use polymarket_client_sdk::types::U256;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, RwLock};
use xgb::Booster;

/// Ёмкость очереди `LaneFrame` на одну пару `(interval, side)` — фанаут из
/// `project_manager::build_frames_from_buffer_lane_once(lane=0)` идёт через
/// `try_send`; буфер даёт запас на задержку воркера (HTTP ордербук + predict).
const LANE_FRAME_CHANNEL_CAP: usize = 64;

/// Ёмкость очереди запросов стаканов в [`run_book_coordinator`]. 4 воркера
/// шлют максимум по одному запросу за кадр (≤ 4 в очереди в стационарном
/// режиме); запас на пик при медленном HTTP — небольшой, чтобы не копить
/// «старые» asset_id за прошлые тики.
const BOOK_REQUEST_CHANNEL_CAP: usize = 64;

/// **Idle-окно** склейки в [`run_book_coordinator`]: после каждого
/// принятого `BookRequest` сбрасывается тайм-аут ожидания следующего.
/// Если за этот интервал больше ничего не пришло — считаем пачку
/// собранной и идём в HTTP. Покрывает реальный inter-arrival jitter
/// между воркерами одного фанаут-тика (≈ ms), но не штрафует тики, где
/// `needs_http` сработал только у 1–2 лейнов.
const BOOK_BATCH_IDLE_MS: u64 = 5;

/// **Абсолютный дедлайн** одного батча в [`run_book_coordinator`] от
/// момента первого запроса. Страховка от патологического случая, когда
/// запросы идут плотным потоком и idle-таймер всё время ресетится:
/// после этого срока выходим в HTTP даже если очередь не «успокоилась».
const BOOK_BATCH_MAX_MS: u64 = 50;

/// Тайм-аут одного `clob.order_books(&[...])` HTTP-запроса в
/// [`run_book_coordinator`]. Polymarket CLOB обычно отвечает за
/// 50–200 ms; всё, что больше 2 секунд — почти наверняка зависший
/// сокет / DNS-таймаут / 5xx без авторазрыва. Без явного тайм-аута
/// один такой запрос блокировал бы координатор на десятки секунд:
/// все 4 воркера висят на `oneshot::Receiver` в `fetch_http_strict_book`,
/// фанаут лейнов копит кадры в очередях `LANE_FRAME_CHANNEL_CAP`,
/// и выйти из ситуации без рестарта процесса нельзя.
///
/// На срабатывании отвечаем `None` всем `oneshot`-получателям батча —
/// `tick_once` пойдёт без strict-fill (WS-fallback для закрытий, скип
/// для buy через `kelly_strict_buy_skips`). На следующем кадре
/// координатор соберёт новый батч.
const BOOK_HTTP_TIMEOUT_MS: u64 = 2000;

/// Тайм-аут ожидания ответа координатора в [`fetch_http_strict_book`]
/// со стороны воркера. Должен покрывать `BOOK_BATCH_IDLE_MS` +
/// `BOOK_BATCH_MAX_MS` + `BOOK_HTTP_TIMEOUT_MS` + запас на
/// планировщик; берём `3 × BOOK_HTTP_TIMEOUT_MS` как простой и
/// достаточный буфер. Если за это время `oneshot::Sender` не пришёл
/// (координатор завис или потерял запрос) — воркер выходит без
/// strict_book на этом тике, не блокируя дальше всю свою цепочку.
const BOOK_REPLY_TIMEOUT_MS: u64 = BOOK_HTTP_TIMEOUT_MS * 3;

/// Максимум `market_id`, которые держим в [`RealSimState::seen_market_ids`]
/// **на каждый** `XFrameIntervalKind`. При insert'е сверх этого порога
/// самый старый (по порядку первой регистрации) вытесняется через
/// `IndexSet::shift_remove_index(0)`, чтобы set не пух бесконечно за
/// время жизни процесса.
const SEEN_MARKET_IDS_CAP: usize = 8;

/// Полный набор 4 ключей фанаута 1s-кадров: `(interval, side)`.
const LANE_FRAME_ROUTES: [(XFrameIntervalKind, CurrencyUpDownOutcome); 4] = [
    (XFrameIntervalKind::FifteenMin, CurrencyUpDownOutcome::Down),
    (XFrameIntervalKind::FifteenMin, CurrencyUpDownOutcome::Up),
    (XFrameIntervalKind::FiveMin, CurrencyUpDownOutcome::Down),
    (XFrameIntervalKind::FiveMin, CurrencyUpDownOutcome::Up),
];

// ─── Общее состояние ───────────────────────────────────────────────────────────

/// Каналы 1s-кадров `(interval, side) → (Sender, dummy_rx)` — живут внутри
/// [`RealSimState`] и являются единственной связкой между
/// `project_manager::build_frames_from_buffer_lane_once(lane=0)` и 4 воркерами
/// `real_sim`. Карта создаётся **пустой** в [`LaneFrameChannels::new`]; в
/// режиме `real_sim` [`run_real_sim`] заполняет её по мере спавна воркеров,
/// а в остальных режимах (`default`/`train`/`history_sim`) карта так и
/// остаётся пустой — фанаут `get` возвращает `None` и кадры молча отбрасывает.
///
/// В каждой записи настоящий `Receiver` **не хранится**: реальный `rx`
/// отдаётся воркеру прямо в момент создания канала (там же, где делается
/// `mpsc::channel`), а в карту кладётся «заглушечный» `Receiver` из свежего
/// `mpsc::channel(1)` — нужен лишь для того, чтобы не менять тип карты
/// (фанаут всё равно трогает только `Sender`).
pub struct LaneFrameChannels {
    pub channels: Arc<
        RwLock<
            HashMap<
                (XFrameIntervalKind, CurrencyUpDownOutcome), mpsc::Sender<LaneFrame>
            >,
        >,
    >,
}

impl LaneFrameChannels {
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// Разделяемое состояние симуляции: 2 [`SimStats`] (один на интервал,
/// per-interval счётчики trades/wins/...) + push-канал от `ProjectManager`:
/// `build_frames_from_buffer_lane_once(lane=0)` шлёт сюда `LaneFrame` каждый
/// стабильный тик. Сам state живёт внутри
/// [`crate::project_manager::ProjectManager::real_sim_state`] (`Arc<RwLock<_>>`).
///
/// **Денежная** часть состояния (bankroll, drawdown, last_prob,
/// открытые позиции) живёт в [`crate::account::Account`], который
/// шарится между ВСЕМИ `ProjectManager`-ами процесса. Это нужно для
/// корректного Kelly-сайзинга и mark-to-market по единому портфелю
/// поверх всех валют и всех 4 лейнов: 4×N параллельных воркеров видят
/// общий `bankroll` и общую сумму `entry_cost` всех позиций.
pub struct RealSimState {
    /// Агрегированная статистика по интервалам (per-interval счётчики).
    /// Карта инициализируется оба ключа сразу, воркеры делают
    /// `get_mut(&kind).unwrap()`.
    pub stats: HashMap<XFrameIntervalKind, SimStats>,
    pub lane_frame_channels: LaneFrameChannels,
    /// Множество уже посчитанных в `stats[interval].events` маркетов —
    /// дедуп-щит для bump'а events. Инкрементируем `events` только если
    /// `seen_market_ids[interval].insert(market_id)` вернул `true`.
    ///
    /// # Зачем
    ///
    /// Раньше `events += 1` бампился только Up-стороной. Это давало
    /// двойной баг:
    ///   1. Если Up-канал отстал и первой сменилась Down-сторона —
    ///      `events` за этот маркет вообще не инкрементировался.
    ///   2. Если первый кадр маркета приходил с `prob = None` (типично
    ///      сразу после ws-reconnect), мы делали early-return ДО
    ///      обновления `last_market_id`, и на следующем кадре того
    ///      же маркета `market_changed = true` снова → дубль bump'а.
    ///
    /// `IndexSet` решает оба случая: первая сторона, увидевшая новый
    /// `market_id` в этом интервале, инкрементирует; вторая (и любые
    /// повторы) — нет. Идемпотентно.
    ///
    /// # Почему `IndexSet`, а не `HashSet`
    ///
    /// Для предотвращения неограниченного роста при долгой жизни
    /// процесса: `IndexSet` сохраняет порядок вставки, что позволяет
    /// дешево вытеснять «самый старый» маркет через
    /// `shift_remove_index(0)` при превышении [`SEEN_MARKET_IDS_CAP`].
    /// Семантика — FIFO-ring: всегда помним последние ~1024 маркетов на
    /// интервал; более старые забываются. Цена забывания — теоретически
    /// возможный лишний bump'а events для маркета, кадр которого
    /// прилетел спустя сотни тиков (нереальный случай при
    /// `BOOK_HTTP_TIMEOUT_MS = 2 сек`).
    pub seen_market_ids: HashMap<XFrameIntervalKind, IndexSet<String>>,
}

impl RealSimState {
    pub fn new() -> Self {
        let mut stats = HashMap::with_capacity(2);
        stats.insert(XFrameIntervalKind::FiveMin, SimStats::new());
        stats.insert(XFrameIntervalKind::FifteenMin, SimStats::new());
        let mut seen_market_ids = HashMap::with_capacity(2);
        seen_market_ids.insert(XFrameIntervalKind::FiveMin, IndexSet::new());
        seen_market_ids.insert(XFrameIntervalKind::FifteenMin, IndexSet::new());
        Self {
            stats,
            lane_frame_channels: LaneFrameChannels::new(),
            seen_market_ids,
        }
    }
}

/// Модели одной стороны одного интервала (PnL обязательная, Resolution — опциональная).
struct SideModels {
    booster_pnl: Arc<Booster>,
    calibration_pnl: Option<Calibration>,
    booster_resolution: Option<Arc<Booster>>,
    calibration_resolution: Option<Calibration>,
}

pub(crate) fn interval_label(kind: XFrameIntervalKind) -> &'static str {
    match kind {
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    }
}

pub(crate) fn side_label(side: CurrencyUpDownOutcome) -> &'static str {
    match side {
        CurrencyUpDownOutcome::Up => "up",
        CurrencyUpDownOutcome::Down => "down",
    }
}

// ─── Точка входа ───────────────────────────────────────────────────────────────

/// Запускает `real_sim`: загружает модели из последней версии
/// `xframes/{currency}/{version}/` и поднимает 4 tokio-таска
/// (5m × up/down, 15m × up/down). Каждый таск слушает свой `mpsc::Receiver`
/// из [`LaneFrameChannels`] и обрабатывает кадр в [`tick_once`].
pub async fn run_real_sim(project_manager: Arc<ProjectManager>) -> Result<()> {
    let currency_arc = project_manager.currency.clone();
    let currency = currency_arc.as_str().to_string();
    let version_path = latest_version_path(&currency)
        .ok_or_else(|| anyhow!(
            "нет ни одной версии в xframes/{currency}/ — сначала соберите данные (STATUS=default) и обучите модели (STATUS=train)"
        ))?;
    let version = dir_name(&version_path);
    let tag_prefix = format!("{currency}/{version}");

    crate::tee_println!(
        "[real_sim] версия моделей: {tag_prefix} (из {})",
        version_path.display(),
    );

    // Единое общее состояние (по ТЗ — "Введи новую структуру с Arc<RwLock>"):
    // живёт внутри `ProjectManager` и создаётся прямо в `ProjectManager::new`.
    // Воркеры и фанаут 1s-кадров делят один и тот же `Arc<RwLock<RealSimState>>`.
    let state = project_manager.real_sim_state.clone();
    let account = project_manager.account.clone();
    let channels = state.read().await.lane_frame_channels.channels.clone();

    // Пред-инициализируем `Account.positions` для 4 лейнов этой валюты,
    // чтобы в `tick_once` можно было делать `get_mut(&key).unwrap()`.
    // Идемпотентно: повторные запуски `run_real_sim` для одной и той же
    // валюты не затрут уже накопленные позиции.
    account
        .write()
        .await
        .register_currency_lanes(&currency, &LANE_FRAME_ROUTES);

    // Один координатор на все 4 воркера: батчит параллельные запросы
    // ордербуков в один `clob.order_books(&[...])`. Каждый воркер получает
    // клон `book_tx` и шлёт `BookRequest` с персональным `oneshot::Sender`.
    let (book_tx, book_rx) = mpsc::channel::<BookRequest>(BOOK_REQUEST_CHANNEL_CAP);
    {
        let project_manager = project_manager.clone();
        tokio::spawn(async move {
            run_book_coordinator(project_manager, book_rx).await;
        });
    }

    for (interval_kind, side) in LANE_FRAME_ROUTES {
        let label = interval_label(interval_kind);
        let side_lbl = side_label(side);
        let models = load_side_models(&version_path, label, side_lbl).ok_or_else(|| {
            anyhow!("не удалось загрузить pnl-модель {label}/{side_lbl}")
        })?;
        crate::tee_println!(
            "[real_sim] {tag_prefix}/{label}/{side_lbl}: pnl ✓  resolution={}",
            if models.booster_resolution.is_some() { "✓" } else { "✗" },
        );

        // Канал создаётся здесь же: настоящий `rx` отдаём прямо воркеру (без
        // промежуточных копий/перемещений через HashMap), а в общую карту
        // регистрируем `(tx, dummy_rx)` — фанаут будет слать через `tx`, а
        // `dummy_rx` нужен только чтобы не менять тип карты.
        let (tx, rx) = mpsc::channel::<LaneFrame>(LANE_FRAME_CHANNEL_CAP);
        channels
            .write()
            .await
            .insert((interval_kind, side), tx);

        spawn_side_worker(
            book_tx.clone(),
            state.clone(),
            account.clone(),
            currency_arc.clone(),
            interval_kind,
            side,
            models,
            tag_prefix.clone(),
            rx,
        );
    }

    Ok(())
}

// ─── Воркер (интервал × сторона) ───────────────────────────────────────────────

fn spawn_side_worker(
    book_tx: mpsc::Sender<BookRequest>,
    state: Arc<RwLock<RealSimState>>,
    account: SharedAccount,
    currency: Arc<String>,
    interval_kind: XFrameIntervalKind,
    side: CurrencyUpDownOutcome,
    models: SideModels,
    tag_prefix: String,
    mut rx: mpsc::Receiver<LaneFrame>,
) {
    tokio::spawn(async move {
        let tag = format!(
            "{tag_prefix}/{}/{}",
            interval_label(interval_kind),
            side_label(side),
        );
        let mut last_market_id: Option<String> = None;
        while let Some(lane_frame) = rx.recv().await {
            // C6: оборачиваем тик в `catch_unwind`, чтобы паника внутри
            // `tick_once` (например, в `predict_frame` на повреждённой
            // модели или при integer overflow в EMA) не убивала весь
            // tokio-таск воркера. Без этого один битый кадр уносил весь
            // лейн до конца жизни процесса — фанаут продолжает писать
            // в канал, кадры копятся до cap'а и теряются (см. C3).
            //
            // `AssertUnwindSafe` корректно: после паники мы не используем
            // мутабельное состояние, которое могло остаться в неконсистентном
            // виде, кроме `last_market_id` (просто `Option<String>`).
            // Account / state защищены RwLock'ами — паника в середине
            // write-секции освободит лок, и параллельные воркеры увидят
            // частично-обновлённое состояние; это хуже, чем dataloss, но
            // лучше «зависшего навсегда» лейна. Для проблем такого рода
            // нужен на уровне выше healthcheck + restart процесса.
            let result = AssertUnwindSafe(tick_once(
                &book_tx,
                &state,
                &account,
                currency.as_str(),
                interval_kind,
                side,
                &models,
                &tag,
                &mut last_market_id,
                lane_frame,
            ))
            .catch_unwind()
            .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    crate::tee_eprintln!("[real_sim] {tag}: tick error: {err:#}");
                }
                Err(payload) => {
                    let msg = panic_payload_message(&payload);
                    crate::tee_eprintln!(
                        "[real_sim] {tag}: tick PANIC ({msg}) — кадр пропущен, воркер живой"
                    );
                }
            }
        }
        crate::tee_eprintln!("[real_sim] {tag}: канал закрыт — воркер завершён");
    });
}

/// Извлекает текстовое сообщение из panic-payload, чтобы залогировать причину
/// в C6-обёртке. `std::panic::catch_unwind` отдаёт `Box<dyn Any + Send>` —
/// типичный кейс — `&'static str` или `String`.
fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Один тик воркера: из пришедшего `LaneFrame` берём маркет/asset/frame,
/// сверяем WS vs HTTP, вызываем `manage_positions` (всегда) и
/// `try_open_position` (если WS не отстаёт и это не последний тик).
async fn tick_once(
    book_tx: &mpsc::Sender<BookRequest>,
    state: &Arc<RwLock<RealSimState>>,
    account: &SharedAccount,
    currency: &str,
    interval_kind: XFrameIntervalKind,
    side: CurrencyUpDownOutcome,
    models: &SideModels,
    tag: &str,
    last_market_id: &mut Option<String>,
    lane_frame: LaneFrame,
) -> Result<()> {
    let LaneFrame {
        market_id,
        asset_id,
        frame,
    } = lane_frame;

    // `market_changed` считаем до early-return, т.к. он нужен и для
    // bookkeeping-апдейта `events` ниже (а тот — обязан произойти даже
    // в кадрах без `currency_implied_prob`, иначе целые маркеты могут
    // «пропасть» из счётчика, если в их первом фрейме prob ещё не
    // подоспела).
    //
    // На `event_remaining_ms ≤ 0` `tick_once` сам ничего не закрывает:
    // `sell_gate` возвращает Hold для всех живых позиций, а резолюция
    // (бинарная выплата CTF $1/$0 без комиссии) уезжает в отдельный
    // колбек [`crate::account::Account::resolve_pending_market`],
    // который дёргается из `xframe_dump` по реальному `final_price`.
    let market_changed = last_market_id.as_deref() != Some(market_id.as_str());

    // ── Bookkeeping #1: bump `events` через seen_market_ids dedupe ────────────
    // `events` отражает «сколько маркетов мы наблюдали для этого интервала».
    // Раньше bump делал только Up-воркер — это давало два бага:
    //   1. Если Up-канал отстал и Down первой увидел смену маркета,
    //      `events` за этот маркет вообще не инкрементировался.
    //   2. Если первый кадр маркета пришёл с `prob = None` (типично
    //      сразу после ws-reconnect), мы делали early-return ДО апдейта
    //      `last_market_id` — на следующем кадре того же маркета
    //      `market_changed = true` снова, и был дубль bump'а.
    //
    // Теперь обе стороны (и любые их повторы) идут через dedupe: первая
    // сторона, успевшая зарегистрировать `market_id` в
    // `state.seen_market_ids[interval]`, инкрементирует. Все последующие
    // (Down + повторы Up из-за пропущенного `prob`) — нет.
    //
    // `last_market_id` обновляем СРАЗУ (и здесь, и на early-return): он
    // нужен для оптимизации (`market_changed` гасится на следующем тике
    // того же маркета и мы не лезем в `state.write()` зря).
    if market_changed {
        let mut state_guard = state.write().await;
        let RealSimState {
            seen_market_ids,
            stats,
            ..
        } = &mut *state_guard;
        let seen = seen_market_ids.entry(interval_kind).or_default();
        if seen.insert(market_id.clone()) {
            // FIFO-вытеснение: вышли за cap → выкидываем самый старый.
            // `shift_remove_index(0)` сохраняет порядок (O(n) копирование,
            // но n ≤ SEEN_MARKET_IDS_CAP, так что для 1024 элементов это
            // микросекунды и происходит максимум раз на тик).
            while seen.len() > SEEN_MARKET_IDS_CAP {
                seen.shift_remove_index(0);
            }
            stats
                .get_mut(&interval_kind)
                .expect("stats map initialized for both intervals")
                .events += 1;
        }
    }

    let Some(raw_prob) = frame.currency_implied_prob else {
        return Ok(());
    };
    // D4: защита от bogus prob (NaN / out-of-range). `effective_implied_prob`
    // в `xframe.rs` уже clamp'ит свой результат, но кадр мог прийти с
    // нефинитным значением через парсер ws (теоретически). Если так —
    // пропускаем кадр целиком: писать NaN в `last_prob` нельзя — это
    // отравит mark-to-market всех зависимых лейнов на следующих тиках.
    if !raw_prob.is_finite() || raw_prob <= 0.0 || raw_prob >= 1.0 {
        crate::tee_eprintln!(
            "[real_sim] {tag}: bogus currency_implied_prob={raw_prob} \
             (market={market_id}) — кадр пропущен"
        );
        *last_market_id = Some(market_id);
        return Ok(());
    }
    let currency_implied_prob = raw_prob.clamp(0.001, 0.999);

    let lane_key = (currency.to_string(), interval_kind, side);

    // ── Bookkeeping #2: записать last_prob в общий реестр Account ─────────────
    // Делаем это **сразу** после того как узнали `currency_implied_prob`, в
    // СОБСТВЕННОМ узком `account.write()` — параллельные воркеры других
    // лейнов, висящие на `account.read()` для своего snapshot'а, тут же
    // увидят свежее значение в своих mark-to-market. Если оставить запись
    // в большой общий `account.write()` для торговли, она будет видна
    // только после HTTP-задержки (десятки ms на ордербук), что замусоривает
    // mark-to-market параллельных лейнов «старой» prob.
    //
    // Здесь пишем именно WS-prob: `effective_implied_prob` (HTTP-mid /
    // last trade) станет доступен только после `fetch_http_strict_book`
    // ниже, а другим лейнам нужно свежее значение СРАЗУ. После HTTP мы
    // ПЕРЕПИШЕМ `last_prob` на effective под общим write-локом фазы 1
    // (см. ниже), так что чужие лейны на следующих тиках уже увидят
    // strict-prob.
    {
        let mut account_guard = account.write().await;
        account_guard
            .last_prob
            .insert(lane_key.clone(), currency_implied_prob);
    }

    // ── Снапшот состояния для гейтов ──────────────────────────────────────────
    // * `has_positions` — нужно ли **звать** `manage_positions`. Даже если
    //   ни одна позиция не закрывается по WS, вызов нужен для обновления
    //   `frames_held`/`p_win_ema` и для переноса stale-позиций в
    //   `pending_resolution` при смене маркета.
    // * `needs_http` — нужен ли **HTTP-запрос** стакана. Дёргаем CLOB только
    //   когда реально будем исполнять ордер через `book_fill_*_strict`:
    //     - `needs_sell` (см. `any_position_would_sell`) — рыночное
    //       закрытие TP/SL/Timeout/EV-exit через strict-sell;
    //     - `buy_gate == Proceed` — модель хочет открыться (strict-buy).
    //   Если позиции просто висят без триггера и входить не планируем —
    //   HTTP не делаем; `manage_positions` отработает на WS-fallback для
    //   одного только bookkeeping'а (никаких закрытий не сработает,
    //   предикат симметричен фактическим условиям из `manage_positions`).
    // * `available_bankroll_pre` — доступный для НОВОГО входа капитал:
    //   `account.bankroll − Σ(entry_cost) по ВСЕМ лейнам ВСЕХ валют` —
    //   `Account.positions` теперь портфельный, и Kelly видит экспозицию
    //   на ОБЩИЙ bankroll сразу со всех PM. Без этой агрегации `4×N`
    //   параллельных воркеров (4 лейна × N валют) раздули бы суммарную
    //   экспозицию до `4N × MAX_BET_FRACTION` исходного капитала.
    let (
        has_positions,
        needs_sell,
        available_bankroll_pre,
        dd_halt_active,
        account_max_dd_pct,
        market_already_resolved,
    ) = {
        let account_guard = account.read().await;
        let this_positions = account_guard
            .positions
            .get(&lane_key)
            .expect("Account.positions pre-populated by run_real_sim");
        let total_locked: f64 = account_guard
            .positions
            .values()
            .flat_map(|v| v.iter())
            .map(|p| p.entry_cost)
            .sum();
        let available = (account_guard.bankroll - total_locked).max(0.0);
        // Kill-switch по mark-to-market drawdown: если `EMERGENCY_HALT_DRAWDOWN_PCT
        // = Some(p)` и `account.max_drawdown_pct ≥ p` — новые входы (и только
        // новые) блокируем до конца жизни процесса. `max_drawdown_pct` —
        // историческая величина (не сбрасывается на восстановлении), так что
        // halt — необратимое состояние без явного рестарта. `manage_positions`
        // (TP/SL/EV/Timeout/резолюция) продолжает работать как обычно: выйти
        // из позиции важнее, чем дождаться улучшения equity.
        let dd_halt = match crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT {
            Some(threshold) => account_guard.max_drawdown_pct >= threshold,
            None => false,
        };
        // C2-защита: если этот `market_id` уже резолвнулся (колбек
        // `xframe_dump::spawn_dump_market_xframes_binary` отработал), не
        // открываем по нему новые позиции. Между приходом резолюционного
        // колбека и возвратом нашего HTTP-запроса в `tick_once` мог
        // образоваться зазор: HTTP подвис на пару секунд, за это время
        // резолюция вызвала `Account::resolve_pending_market`, а кадры
        // нашего лейна с этого маркета уже стоят в очереди воркера с
        // `event_remaining_ms > 0` (CLOB закрыл маркет раньше, чем
        // `sleep(max_step)` дорастил резолюционный колбек). Без этого
        // фильтра `try_open_position` создал бы фантомную локальную
        // позицию на маркете, который on-chain больше не торгуется.
        let market_resolved = account_guard
            .recently_resolved_markets
            .contains(market_id.as_str());
        (
            !this_positions.is_empty(),
            any_position_would_sell(this_positions, &frame),
            available,
            dd_halt,
            account_guard.max_drawdown_pct,
            market_resolved,
        )
    };

    // ── Booster-инференсы — СТРОГО ВНЕ write-локов ────────────────────────────
    // Дорогая часть decision-tree (XGBoost predict + калибровка) не зависит от
    // `state`/`account`: входы — `frame` и иммутабельные `&Booster`/
    // `Option<&Calibration>` из `RealSimWorkerCfg`. Считаем ОДИН раз за тик и
    // прокидываем готовые значения и в дешёвый `buy_gate`-предикат для
    // `may_open` (см. ниже), и в `try_open_position` / `manage_positions` под
    // write-локами. Без этого:
    //   * `predict_frame` для PnL-модели запускался ДВА раза за тик (один раз
    //     здесь под `may_open`, второй — в `try_open_position` внутри
    //     `state.write() + account.write()`);
    //   * `predict_frame` resolution-модели запускался под теми же локами
    //     внутри `manage_positions`.
    // На 4×N параллельных воркеров это материальная задержка критсекции —
    // выносим наружу.
    let pnl_inference = compute_pnl_inference(
        &frame,
        &models.booster_pnl,
        models.calibration_pnl.as_ref(),
    );
    // `compute_p_win_now` больше не гейтится по `has_positions` —
    // resolution-инференс считается каждый тик в hold-zone безусловно,
    // чтобы у только что открытой позиции к следующему `manage_positions`
    // EMA `p_win` уже стартовала с готовой инициализацией. Стоимость —
    // не более одного inference resolution-модели на кадр в hold-zone
    // даже при пустых позициях, что для real_sim вне локов пренебрежимо.
    let p_win_now = compute_p_win_now(
        &frame,
        models.booster_resolution.as_deref(),
        models.calibration_resolution.as_ref(),
    );

    // `buy_gate` сам отказывает, если событие уже завершилось или до резолюции;
    // `available_bankroll_pre` уже учитывает экспозицию всех лейнов всех валют,
    // так что Kelly не сможет раздуть позицию поверх занятого капитала.
    //
    // На этом первом вызове `strict_book = None` — мы ещё не сходили
    // в HTTP, и решаем, нужен ли он вообще (`needs_http = needs_sell ||
    // may_open`). Поэтому `buy_gate` использует `frame.currency_implied_prob`
    // (WS-prob); strict-prob по mid HTTP-стакана будет передан в
    // фактический `try_open_position` ниже, после `fetch_http_strict_book`.
    let buy_gate_proceed = matches!(
        buy_gate(&frame, pnl_inference, available_bankroll_pre, None),
        BuyGate::Proceed { .. }
    );
    let may_open = !dd_halt_active && !market_already_resolved && buy_gate_proceed;
    if buy_gate_proceed && dd_halt_active {
        // Один лог на каждый «пропущенный по halt» вход. Halt — необратимое
        // состояние, и поток входов модели в просадке естественно скуднее,
        // так что спама не будет.
        crate::tee_eprintln!(
            "[real_sim] {tag}: halt by drawdown — новые позиции заблокированы (порог={:?}%, max_dd_pct={:.2}%), закрытия продолжаем",
            crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT,
            account_max_dd_pct
        );
    }
    if buy_gate_proceed && market_already_resolved {
        // Маркет уже резолвнулся (см. C2-фильтр выше). Логируем, чтобы было
        // видно, что модель «промахивается» в окно после резолюции —
        // обычно из-за HTTP-задержки `fetch_http_strict_book`. Если строк
        // много — значит зазор между CLOB-закрытием маркета и нашим
        // `xframe_dump`-колбеком слишком велик, и стоит уменьшить
        // `BOOK_HTTP_TIMEOUT_MS` или ускорить `spawn_dump_market_xframes_binary`.
        crate::tee_eprintln!(
            "[real_sim] {tag}: skip open — market={market_id} уже резолвнулся, кадр пришёл с задержкой"
        );
    }
    let needs_http = needs_sell || may_open;

    // ── HTTP-ордербук: один запрос за кадр, переиспользуется для всего ───────
    // * проверки отставания WS (best-bid/ask сверяются с `frame.book_*_l1_*`);
    // * СТРОГОГО исполнения buy/sell через `StrictBook` (см. `history_sim`).
    // Если HTTP недоступен — `strict_book` не собирается, `manage_positions`/
    // `try_open_position` уйдут с `None` (WS-fallback). Блокировать закрытия
    // важнее, чем блокировать всё — иначе позиция «зависнет» при сетевом шуме.
    // `StrictBook` владеет собственными `Vec<BookLevel>` — здесь он живёт
    // до конца кадра и подаётся в `manage_positions`/`try_open_position`
    // через `.as_ref()`.
    let strict_book: Option<StrictBook> = if needs_http {
        fetch_http_strict_book(book_tx, &asset_id, tag).await
    } else {
        None
    };

    // Свежесть WS определяется по верхним 3 уровням HTTP-стакана относительно L1/L2/L3 WS-кадра.
    let ws_lagging = match strict_book.as_ref() {
        Some(book) => {
            let lagging = is_ws_lagging(book, &frame);
            if lagging {
                crate::tee_eprintln!(
                    "[real_sim] {tag}: WS отстаёт — ордербук по HTTP расходится с last XFrame (market={market_id} asset={asset_id}); новые позиции пропускаем, ведём только закрытия"
                );
            }
            lagging
        }
        None => false,
    };

    // ── Торговля + Mark-to-market под write-локами ────────────────────────────
    // Лок-порядок: state.write() → account.write(). Никогда наоборот, иначе
    // 4×N параллельных воркеров (4 лейна × N валют) могут зайти в дедлок.
    // Bookkeeping (`events` bump и `last_prob` insert) уже сделан выше в
    // отдельных узких локах — здесь только торговля и MtM.
    //
    // Локи держим минимальное время:
    //   * `state_guard` дропаем СРАЗУ после трейдинга — MtM использует
    //     только `Account`, и держать там state.write() — значит зря
    //     блокировать `events`-bump параллельных воркеров.
    //   * `account_guard` живёт до конца блока, после чего тоже дропается.
    //   * Печать (`print_sim_stats`) выполняется уже под READ-локами вне
    //     этого блока — `tee_println!` может блокироваться на stdout/файле,
    //     держать там write — большой штраф для других воркеров.
    let mut sold = false;
    let mut bought = false;
    // B1: «эффективная» prob — Polymarket-style по HTTP-стакану (mid L1 при
    // спреде ≤ 10¢, иначе last trade), с фоллбэком на WS-prob. Это ровно
    // та же шкала, на которой `try_open_position` пишет `OpenPosition.entry_prob`,
    // и на которой `sell_gate` считает `delta = current_prob − entry_prob`.
    // Используем её и для перезаписи `last_prob` ниже, и для MtM нашего лейна
    // в фазе 2 — чтобы entry/exit/MtM жили на одной шкале.
    let effective_prob = crate::history_sim::effective_implied_prob(&frame, strict_book.as_ref())
        .unwrap_or(currency_implied_prob);
    {
        let mut state_guard = state.write().await;
        let mut account_guard = account.write().await;

        // B1: ПЕРЕзаписать `last_prob` на effective_prob. Первая (WS) запись
        // была в bookkeeping #2 — она нужна параллельным воркерам, ждущим
        // MtM прямо сейчас. Эта вторая запись даёт чужим лейнам уже
        // strict-prob к их **следующему** тику, что ровняет шкалу
        // entry_prob ↔ MtM по портфелю.
        account_guard
            .last_prob
            .insert(lane_key.clone(), effective_prob);

        // B2: re-check halt под write-локом. Snapshot `dd_halt_active` был
        // сделан ДО HTTP (десятки/сотни ms назад). За это время параллельные
        // воркеры могли пересчитать `update_drawdown` и пробить порог.
        // Без re-check'а до 1 «лишнего» входа после фактического halt'а;
        // под write-локом мы уже видим самое свежее значение и можем
        // отказаться от открытия.
        let dd_halt_now = match crate::history_sim::EMERGENCY_HALT_DRAWDOWN_PCT {
            Some(threshold) => account_guard.max_drawdown_pct >= threshold,
            None => false,
        };
        if !dd_halt_active && dd_halt_now && may_open {
            crate::tee_eprintln!(
                "[real_sim] {tag}: halt by drawdown сработал между snapshot'ом и HTTP — \
                 новый вход отменяем (max_dd_pct={:.2}%)",
                account_guard.max_drawdown_pct
            );
        }
        let may_open = may_open && !dd_halt_now;

        // ── Фаза 1: торговля ──────────────────────────────────────────────────
        // «Купили/продали» получаем напрямую из возвратов `manage_positions` /
        // `try_open_position` — никакого до/после диффа `stats.trades` /
        // `positions.len()` (buy+sell за один тик оставит `len` тем же).
        if has_positions || may_open {
            // Lock на чужие лейны (cross-lane locked) — это сумма по ВСЕМ
            // лейнам ВСЕХ валют, кроме своего; считаем ДО split-borrow, пока
            // `account_guard.{positions,pending_resolution}` доступны только по `&`.
            //
            // ВАЖНО: учитываем и `pending_resolution`-позиции. Их
            // `entry_cost` физически всё ещё «работает» в ожидании
            // post-resolution колбека — деньги в bankroll до закрытия
            // не вернутся. Без этого учёта Kelly считал бы pending-капитал
            // свободным и параллельно раздул бы экспозицию того же
            // bankroll'а в новые позиции на других лейнах.
            let cross_lanes_locked: f64 = account_guard
                .positions
                .iter()
                .filter(|(k, _)| *k != &lane_key)
                .flat_map(|(_, v)| v.iter())
                .map(|p| p.entry_cost)
                .chain(
                    account_guard
                        .pending_resolution
                        .iter()
                        .filter(|(k, _)| *k != &lane_key)
                        .flat_map(|(_, v)| v.iter())
                        .map(|p| p.entry_cost),
                )
                .sum();

            // `stats` живёт в `RealSimState`, `bankroll`/`positions` — в
            // `Account`. Берём split-borrow по обоим guard'ам сразу: это
            // разные структуры за разными RwLock'ами, conflict'а нет.
            let stats: &mut SimStats = state_guard
                .stats
                .get_mut(&interval_kind)
                .expect("stats map initialized for both intervals");
            let side_stats = match side {
                CurrencyUpDownOutcome::Up => &mut stats.up,
                CurrencyUpDownOutcome::Down => &mut stats.down,
            };

            let Account {
                bankroll,
                positions: account_positions,
                pending_resolution: account_pending,
                ..
            } = &mut *account_guard;
            // `get_many_mut` мы не имеем (msrv), но ключи заведомо разные
            // («positions» и «pending_resolution» — это разные карты), и
            // лейн-ключ один и тот же, так что коллизии нет: достаём
            // мутабельные ссылки из РАЗНЫХ HashMap'ов независимо.
            let this_positions: &mut Vec<OpenPosition> = account_positions
                .get_mut(&lane_key)
                .expect("Account.positions pre-populated by run_real_sim");
            let this_pending: &mut Vec<OpenPosition> = account_pending
                .get_mut(&lane_key)
                .expect("Account.pending_resolution pre-populated by run_real_sim");

            // 1) Жизненный цикл уже открытых позиций: инкремент `frames_held`,
            //    EMA `p_win`, проверка TP/SL/Timeout/EV. Вызываем **всегда**,
            //    когда позиции есть, — даже без HTTP: «тихий» тик сам ничего
            //    не закроет по WS-fallback, т.к. предикат `needs_sell`
            //    симметричен условиям `manage_positions`. `strict_book.as_ref()`
            //    будет `Some` только при `needs_http=true`. Резолюция по
            //    итогу события сюда не приходит: на `event_remaining_ms ≤ 0`
            //    `sell_gate` возвращает Hold, а закрытие по бинарному
            //    payout исполняет колбек
            //    [`crate::account::Account::resolve_pending_market`].
            //
            //    Stale-позиции (asset_id ≠ frame.asset_id, например на смене
            //    5m/15m раунда внутри лейна) на этом тике уезжают в
            //    `this_pending` и больше не блокируют ни sell_gate, ни
            //    Kelly-сайзинг этого лейна. На `event_remaining_ms ≤ 0`
            //    `sell_gate` сама возвращает Hold — позиции дождутся
            //    резолюционного колбека `Account::resolve_pending_market`,
            //    который придёт от `xframe_dump` после реального
            //    `final_price`.
            if has_positions {
                sold = manage_positions(
                    this_positions,
                    this_pending,
                    &frame,
                    // `is_last = false`: real_sim — live-поток без
                    // понятия «последний кадр». Параметр актуален
                    // только для history_sim (truncated-дамп fallback).
                    false,
                    p_win_now,
                    side_stats,
                    bankroll,
                    strict_book.as_ref(),
                    tag,
                );
            }

            // 2) BUY: пропускаем, если WS отстаёт. На `event_remaining_ms ≤ 0`
            //    `may_open` уже `false` (внутри `buy_gate` сработал `LateEntry`).
            //    Пересчитываем `available_bankroll`: `manage_positions` мог
            //    закрыть нашу позицию — освободившийся entry_cost снова
            //    доступен Kelly. `cross_lanes_locked` зафиксирован снапшотом
            //    выше (другие лейны/валюты внутри ЭТОГО account.write()-блока
            //    не меняются — параллельные `tick_once` стоят на `account.write()`).
            if may_open && !ws_lagging {
                // Same-lane locked: активные + pending этого лейна. Те же
                // мотивы, что и для cross-lane (см. комментарий выше) —
                // pending'и держат `entry_cost`, считать их свободными
                // нельзя.
                let same_locked_post: f64 = this_positions
                    .iter()
                    .chain(this_pending.iter())
                    .map(|p| p.entry_cost)
                    .sum();
                let available_bankroll_post = (*bankroll - cross_lanes_locked - same_locked_post).max(0.0);
                bought = try_open_position(
                    &frame,
                    pnl_inference,
                    this_positions,
                    side_stats,
                    available_bankroll_post,
                    strict_book.as_ref(),
                    tag,
                );
            }
        }

        // `state_guard` больше не нужен — отпускаем явно ДО MtM, чтобы
        // параллельные воркеры могли делать `events`-bump (`state.write()`)
        // и читать `stats`, не дожидаясь нашего MtM по 4-N лейнам.
        drop(state_guard);

        // ── Фаза 2: Mark-to-market equity drawdown ────────────────────────────
        // Считается на КАЖДОМ тике, не только на сделке. Между open и close
        // реализованный bankroll не двигается, и без MtM `max_drawdown_pct`
        // системно занижен на длинных удержаниях, уходящих в красное и
        // закрывающихся через резолюционный колбек.
        //
        // Поскольку `bankroll`, `positions` и `last_prob` живут в одном
        // `Account`, equity считается **истинно портфельный** — по всем
        // валютам и всем 4 лейнам каждой:
        //     equity = account.bankroll + Σ_(c,i,s) Σ_pos shares_held × prob[c,i,s]
        // Для текущего лейна prob — `effective_prob` (Polymarket-style по
        // HTTP-стакану, fallback на WS), та же шкала, что у
        // `OpenPosition.entry_prob`. Для чужих — `account.last_prob[(c,i,s)]`,
        // который обновляется в начале каждого `tick_once` соответствующего
        // лейна, плюс перезаписывается на effective ниже по этому же блоку
        // (см. B1) — т.е. на следующих тиках чужие лейны увидят strict-prob.
        // Каждое значение clamp'им (D4): защита от NaN/out-of-range, который
        // мог проскочить в `last_prob` других лейнов до D4-фильтра.
        // Если записи ещё нет (старт процесса) и при этом там уже есть
        // позиции — берём `0.5` (нейтрально), но в штатной работе позиции
        // открываются только ПОСЛЕ хотя бы одного `tick_once`, так что
        // fallback почти не срабатывает.
        let total_value: f64 = {
            let Account {
                positions: account_positions,
                pending_resolution: account_pending,
                last_prob,
                ..
            } = &*account_guard;
            let active: f64 = account_positions
                .iter()
                .map(|((c, i, s), pos_vec)| {
                    let prob_raw = if c.as_str() == currency && *i == interval_kind && *s == side {
                        effective_prob
                    } else {
                        last_prob
                            .get(&(c.clone(), *i, *s))
                            .copied()
                            .unwrap_or(0.5)
                    };
                    let prob = if prob_raw.is_finite() {
                        prob_raw.clamp(0.001, 0.999)
                    } else {
                        0.5
                    };
                    pos_vec.iter().map(|p| p.shares_held * prob).sum::<f64>()
                })
                .sum();
            // pending-позиции: текущий prob их СТАРОГО маркета мы не
            // знаем (фреймов по нему больше не приходит, а `last_prob`
            // лейна уже описывает новый маркет). Используем `entry_prob`
            // как нейтральный mark-to-market — это фактически «капитал
            // заблокирован, итог пока неизвестен». Даёт MtM, эквивалентный
            // entry_cost (с точностью до комиссии): equity не двигается
            // от факта перевода в pending, что и нужно — реальный PnL
            // прилетит post-resolution колбеком.
            let pending: f64 = account_pending
                .values()
                .flat_map(|v| v.iter())
                .map(|p| p.shares_held * p.entry_prob)
                .sum();
            active + pending
        };
        let equity = account_guard.bankroll + total_value;
        account_guard.update_drawdown(equity);

        // `account_guard` дропается на закрытии этого блока.
    }

    // ── Фаза 3: print on trade под READ-локами ────────────────────────────────
    // Срабатывает только при реальной сделке (редко) и не блокирует
    // параллельных писателей — `state.read()`/`account.read()` совместимы с
    // другими читателями. Между дропом write-локов выше и этим блоком
    // другой воркер мог успеть сделать свою сделку или MtM, поэтому печать
    // отражает «текущее состояние», а не строгий снапшот сразу после нашей
    // сделки. Для метрики мониторинга это допустимо; цена строгой
    // консистентности — держать write-локи на медленном `tee_println!` —
    // непропорционально велика.
    if bought || sold {
        let state_guard = state.read().await;
        let account_guard = account.read().await;
        let stats = state_guard
            .stats
            .get(&interval_kind)
            .expect("stats map initialized for both intervals");
        let action = if bought && sold {
            "buy+sell"
        } else if bought {
            "buy"
        } else {
            "sell"
        };
        crate::tee_println!(
            "[real_sim] {tag}: {action} @ t={} market={market_id} prob={currency_implied_prob:.4}",
            current_timestamp_ms(),
        );
        print_sim_stats(tag, stats, &account_guard);
    }

    *last_market_id = Some(market_id);
    Ok(())
}

// ─── HTTP ордербук + проверка свежести WS ──────────────────────────────────────

/// Запрос ордербука к [`run_book_coordinator`]: `asset_id` строкой
/// (парсится в `U256` уже внутри координатора, чтобы все ошибки
/// валидации жили в одном месте) + персональный `oneshot::Sender` для
/// ответа. Координатор отвечает `Some(StrictBook)` при успехе батча и
/// `None` на любой ошибке (HTTP, парс U256, mismatch размеров ответа).
struct BookRequest {
    asset_id: String,
    reply: oneshot::Sender<Option<StrictBook>>,
}

/// Единая точка входа для получения HTTP-ордербука за тик в виде готового
/// [`StrictBook`]. Не делает HTTP сама: посылает `BookRequest` в
/// [`run_book_coordinator`] и ждёт ответ через `oneshot`. Координатор
/// группирует параллельные запросы 4 воркеров в один
/// `clob.order_books(&[...])` — вместо 4 независимых GET.
///
/// Ошибку HTTP **не** пробрасываем наверх: trading-цикл умеет работать без
/// строгого стакана (`strict_book = None` → `manage_positions`/
/// `try_open_position` используют WS-fallback). Блокировать закрытия из-за
/// сетевого шума хуже, чем временно выключить strict-fill — логируем и
/// возвращаем `None`.
async fn fetch_http_strict_book(
    book_tx: &mpsc::Sender<BookRequest>,
    asset_id: &str,
    tag: &str,
) -> Option<StrictBook> {
    let (reply_tx, reply_rx) = oneshot::channel();
    let req = BookRequest {
        asset_id: asset_id.to_string(),
        reply: reply_tx,
    };
    if book_tx.send(req).await.is_err() {
        crate::tee_eprintln!(
            "[real_sim] {tag}: book-coord канал закрыт — strict-fill выключен на тик"
        );
        return None;
    }
    // Тайм-аут страхует от ситуации, когда координатор ещё жив, но
    // потерял наш `oneshot::Sender` (например, на нештатном выходе из
    // батч-цикла) или его HTTP-запрос завис дольше `BOOK_HTTP_TIMEOUT_MS`,
    // но по какой-то причине без явного ответа `None`. Без этого
    // воркер блокировался бы на `reply_rx.await` бесконечно, копя кадры
    // в `LANE_FRAME_CHANNEL_CAP` и в итоге роняя фанаут (`try_send` →
    // `Full`).
    match tokio::time::timeout(Duration::from_millis(BOOK_REPLY_TIMEOUT_MS), reply_rx).await {
        Ok(Ok(book)) => book,
        Ok(Err(_)) => {
            crate::tee_eprintln!(
                "[real_sim] {tag}: book-coord уронил oneshot до ответа — strict-fill выключен на тик"
            );
            None
        }
        Err(_) => {
            crate::tee_eprintln!(
                "[real_sim] {tag}: ожидание ответа book-coord > {BOOK_REPLY_TIMEOUT_MS}ms — strict-fill выключен на тик"
            );
            None
        }
    }
}

/// Координатор HTTP-ордербуков: один таск на весь `real_sim`. Слушает
/// `mpsc::Receiver<BookRequest>` и склеивает запросы в батчи.
///
/// Алгоритм одного цикла:
/// 1. Блокируется на `recv()` первого запроса (если канал закрыт — выходит).
/// 2. Открывает окно [`BOOK_BATCH_WINDOW_MS`] через `tokio::time::sleep`,
///    чтобы дать остальным воркерам положить свои запросы в очередь.
/// 3. Через `try_recv()` выгребает всё накопившееся.
/// 4. Дедуплицирует по `asset_id` (на 4 разных лейна обычно 4 уникальных
///    asset_id, но в редких случаях возможна повторная отправка — отвечаем
///    одним и тем же клонированным `StrictBook`).
/// 5. Делает один `clob.order_books(&[...])` на все валидные `U256`-asset’ы.
///    Невалидные — отвечает `None` сразу, не включая в батч.
/// 6. Ответы CLOB приходят **в том же порядке**, что и запросы, поэтому
///    `zip(valid_ids, responses)` корректно сопоставляет (`order_books`
///    в SDK явно гарантирует это, см. сорцы `polymarket-client-sdk`).
///
/// При закрытии всех `Sender`-ов (process shutdown) `recv()` вернёт `None`
/// и таск аккуратно завершится.
async fn run_book_coordinator(
    project_manager: Arc<ProjectManager>,
    mut rx: mpsc::Receiver<BookRequest>,
) {
    while let Some(first) = rx.recv().await {
        let mut batch: Vec<BookRequest> = vec![first];

        // Adaptive batching: idle-окно + абсолютный дедлайн + потолок по
        // числу лейнов. **Не** ждём фиксированный таймаут, потому что на
        // конкретном тике `needs_http` может сработать только у 1–3
        // лейнов из 4 (зависит от позиций / `buy_gate`). Пустое ожидание
        // в этих случаях даром накручивало бы латентность.
        //
        // На каждой итерации `timeout_at(idle_deadline.min(absolute), recv)`:
        //   - возвращает `Ready(Some(_))` **мгновенно**, если запрос уже в
        //     очереди — никакого паразитного `sleep`;
        //   - на пустой очереди ждёт до ближайшего из двух дедлайнов и
        //     возвращает `Err(_)` → прерываем сборку.
        // Idle-дедлайн **сбрасывается** после каждого успешного
        // `recv` — т.е. пока приходят запросы плотнее
        // `BOOK_BATCH_IDLE_MS`, мы их собираем; как только пауза дольше —
        // считаем пачку собранной. Абсолютный дедлайн страхует от того,
        // что плотный поток может ресетить idle-таймер «вечно». Потолок
        // `LANE_FRAME_ROUTES.len()` корректен: каждый воркер блокирован
        // на собственном `oneshot::Receiver` в `fetch_http_strict_book`,
        // больше одного in-flight запроса на лейн быть не может.
        let absolute_deadline = tokio::time::Instant::now() + Duration::from_millis(BOOK_BATCH_MAX_MS);
        while batch.len() < LANE_FRAME_ROUTES.len() {
            let idle_deadline = tokio::time::Instant::now() + Duration::from_millis(BOOK_BATCH_IDLE_MS);
            let next_deadline = idle_deadline.min(absolute_deadline);
            match tokio::time::timeout_at(next_deadline, rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                Ok(None) | Err(_) => break, // канал закрыт ИЛИ idle/absolute истёк
            }
        }

        // Группируем sender’ы по asset_id (обычно 1 sender на asset, но
        // защитимся от повторных запросов на тот же токен — клонируем
        // StrictBook на каждого).
        let mut by_asset: HashMap<String, Vec<oneshot::Sender<Option<StrictBook>>>> = HashMap::new();
        for req in batch {
            by_asset.entry(req.asset_id).or_default().push(req.reply);
        }

        // Парсим U256 → строим список валидных запросов; невалидные
        // asset_id обслуживаем сразу (None) и в HTTP не шлём.
        let mut requests: Vec<OrderBookSummaryRequest> = Vec::with_capacity(by_asset.len());
        let mut valid_ids: Vec<String> = Vec::with_capacity(by_asset.len());
        let invalid_ids: Vec<String> = by_asset
            .keys()
            .filter(|aid| U256::from_str(aid).is_err())
            .cloned()
            .collect();
        for aid in invalid_ids {
            crate::tee_eprintln!("[real_sim/book-coord] невалидный asset_id={aid} — отвечаем None");
            if let Some(senders) = by_asset.remove(&aid) {
                for s in senders {
                    let _ = s.send(None);
                }
            }
        }
        for aid in by_asset.keys() {
            // .expect: невалидные уже отсеяны выше.
            let token_id = U256::from_str(aid).expect("invalid asset_ids filtered above");
            requests.push(
                OrderBookSummaryRequest::builder()
                    .token_id(token_id)
                    .build(),
            );
            valid_ids.push(aid.clone());
        }

        if requests.is_empty() {
            continue;
        }

        // Один HTTP вместо 4: батч-эндпоинт CLOB `POST /books`.
        // Заворачиваем в `tokio::time::timeout`, чтобы зависший
        // сокет / 5xx без авторазрыва не блокировал координатор на
        // десятки секунд (см. doc у `BOOK_HTTP_TIMEOUT_MS`).
        let n = requests.len();
        let http_result = tokio::time::timeout(
            Duration::from_millis(BOOK_HTTP_TIMEOUT_MS),
            project_manager.clob.order_books(&requests),
        )
        .await;
        match http_result {
            Ok(Ok(responses)) if responses.len() == n => {
                for (aid, resp) in valid_ids.iter().zip(responses.iter()) {
                    let book = parse_book_levels(resp);
                    if let Some(senders) = by_asset.remove(aid) {
                        // 99.9% случаев тут ровно 1 sender, но если
                        // дедуплицировали несколько — клонируем.
                        let mut iter = senders.into_iter();
                        if let Some(last) = iter.next_back() {
                            for s in iter {
                                let _ = s.send(Some(book.clone()));
                            }
                            let _ = last.send(Some(book));
                        }
                    }
                }
            }
            Ok(Ok(responses)) => {
                crate::tee_eprintln!(
                    "[real_sim/book-coord] order_books вернул {} ответов на {n} запросов — отбрасываем батч",
                    responses.len(),
                );
                for senders in by_asset.into_values() {
                    for s in senders {
                        let _ = s.send(None);
                    }
                }
            }
            Ok(Err(err)) => {
                crate::tee_eprintln!(
                    "[real_sim/book-coord] order_books({n} assets) failed: {err:#}"
                );
                for senders in by_asset.into_values() {
                    for s in senders {
                        let _ = s.send(None);
                    }
                }
            }
            Err(_) => {
                crate::tee_eprintln!(
                    "[real_sim/book-coord] order_books({n} assets) timed out > {BOOK_HTTP_TIMEOUT_MS}ms — отбрасываем батч"
                );
                for senders in by_asset.into_values() {
                    for s in senders {
                        let _ = s.send(None);
                    }
                }
            }
        }
    }
    crate::tee_eprintln!("[real_sim/book-coord] mpsc закрыт — координатор завершён");
}

/// Превращает `OrderBookSummaryResponse` в пару `(bids, asks)` формата
/// [`StrictBook`]: уровни [`BookLevel`] в порядке **от лучшего к худшему**.
///
/// Polymarket CLOB отдаёт `bids`/`asks` в обратном порядке («худшее → лучшее»,
/// best = последний элемент), поэтому здесь мы их реверсим. Уровни с
/// неположительной ценой/размером или нечитаемым `Decimal` отбрасываются.
/// Преобразует ответ CLOB `POST /books` в [`StrictBook`].
///
/// Помимо bids/asks (отсортированных «лучший → худший») сохраняем:
/// * `last_trade_price` — для воспроизведения Polymarket-style логики
///   `mid L1 ≤ 10¢ / иначе last trade` в
///   [`crate::history_sim::effective_implied_prob`] (без него HTTP-prob
///   систематически расходился бы с фичей `XFrame.currency_implied_prob`,
///   которая идёт в модель).
/// * `min_order_size` — статичный атрибут маркета, которого нет в
///   WS-канале CLOB, но без которого strict-исполнение
///   ([`crate::history_sim::book_fill_buy_strict`] /
///   `book_fill_sell_strict`) не может зеркалить «CLOB отклонит ордер
///   меньше минимума» — без него локальная бухгалтерия открывала бы
///   фантомные позиции, которых нет on-chain.
fn parse_book_levels(book: &OrderBookSummaryResponse) -> StrictBook {
    let to_level = |o: &polymarket_client_sdk::clob::types::response::OrderSummary| {
        let price = o.price.to_string().parse::<f64>().ok()?;
        let size = o.size.to_string().parse::<f64>().ok()?;
        if price <= 0.0 || size <= 0.0 {
            return None;
        }
        Some(BookLevel { price, size })
    };
    let bids: Vec<BookLevel> = book.bids.iter().rev().filter_map(to_level).collect();
    let asks: Vec<BookLevel> = book.asks.iter().rev().filter_map(to_level).collect();
    let last_trade_price = book
        .last_trade_price
        .and_then(|d| d.to_string().parse::<f64>().ok())
        .filter(|p| p.is_finite() && *p > 0.0);
    let min_order_size = book
        .min_order_size
        .to_string()
        .parse::<f64>()
        .ok()
        .filter(|s| s.is_finite() && *s > 0.0);
    StrictBook {
        bids,
        asks,
        last_trade_price,
        min_order_size,
    }
}

/// Возвращает `true`, если HTTP-ордербук расходится с тремя верхними уровнями
/// WS-кадра (`book_{bid,ask}_l{1,2,3}_price`) больше, чем на `frame.tick_size
/// * 2` (или на 0.01 по умолчанию при отсутствии `tick_size`). Индикатор
/// отставания WS: HTTP всегда свежий, WS может быть задержан буфером или
/// реконнектом.
///
/// Сравниваются все три уровня, потому что L1 может случайно совпасть (тик
/// попал на прежнее значение), а L2/L3 уже «видит» пропущенные апдейты
/// глубины — расхождение на них ловит такой stale чаще. Правило на каждом
/// уровне одинаковое: `(Some, Some)` → сравниваем цену, `(None, Some)` или
/// `(Some, None)` → depth у сторон разная → расхождение, `(None, None)` →
/// уровня просто нет ни там, ни там (ок).
fn is_ws_lagging(book: &StrictBook, frame: &XFrame<SIZE>) -> bool {
    let tol = frame.tick_size.unwrap_or(0.01).max(1e-6) * 2.0;
    let diverges = |ws: Option<f64>, http: Option<f64>| -> bool {
        match (ws, http) {
            (Some(a), Some(b)) => (a - b).abs() > tol,
            (None, Some(_)) | (Some(_), None) => true,
            (None, None) => false,
        }
    };

    // `book.bids`/`book.asks` уже в порядке «лучший → худший» (см.
    // `parse_book_levels`), поэтому индексы 0/1/2 соответствуют L1/L2/L3.
    let http_level = |side: &[BookLevel], idx: usize| side.get(idx).map(|l| l.price);
    let http_bid = |i| http_level(&book.bids, i);
    let http_ask = |i| http_level(&book.asks, i);

    let ws_bid = [
        frame.book_bid_l1_price,
        frame.book_bid_l2_price,
        frame.book_bid_l3_price,
    ];
    let ws_ask = [
        frame.book_ask_l1_price,
        frame.book_ask_l2_price,
        frame.book_ask_l3_price,
    ];

    let bid_bad = (0..ws_bid.len()).any(|i| diverges(ws_bid[i], http_bid(i)));
    let ask_bad = (0..ws_ask.len()).any(|i| diverges(ws_ask[i], http_ask(i)));

    if bid_bad || ask_bad {
        crate::tee_eprintln!(
            "[real_sim] WS vs HTTP ордербук (tol={tol:.4}):\n  \
             bid WS  L1/L2/L3 = {:?}/{:?}/{:?}\n  \
             bid HTTP L1/L2/L3 = {:?}/{:?}/{:?}\n  \
             ask WS  L1/L2/L3 = {:?}/{:?}/{:?}\n  \
             ask HTTP L1/L2/L3 = {:?}/{:?}/{:?}",
            ws_bid[0], ws_bid[1], ws_bid[2],
            http_bid(0), http_bid(1), http_bid(2),
            ws_ask[0], ws_ask[1], ws_ask[2],
            http_ask(0), http_ask(1), http_ask(2),
        );
        true
    } else {
        false
    }
}

// ─── Утилиты (поиск последней версии моделей) ──────────────────────────────────

fn latest_version_path(currency: &str) -> Option<PathBuf> {
    let base = Path::new("xframes").join(currency);
    let mut versions: Vec<(usize, PathBuf)> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if !path.is_dir() {
                return None;
            }
            let name = path.file_name()?.to_string_lossy().to_string();
            let n = name.parse::<usize>().ok()?;
            Some((n, path))
        })
        .collect();
    versions.sort_by_key(|(n, _)| *n);
    versions.pop().map(|(_, p)| p)
}

fn load_side_models(version_path: &Path, interval: &str, side: &str) -> Option<SideModels> {
    let pnl_path = version_path.join(format!("model_{interval}_1s_pnl_{side}.ubj"));
    let resolution_path =
        version_path.join(format!("model_{interval}_1s_resolution_{side}.ubj"));

    let booster_pnl = load_booster(&pnl_path)?;
    let calibration_pnl = load_calibration(&pnl_path).ok();
    let booster_resolution = load_booster(&resolution_path).map(Arc::new);
    let calibration_resolution = load_calibration(&resolution_path).ok();

    Some(SideModels {
        booster_pnl: Arc::new(booster_pnl),
        calibration_pnl,
        booster_resolution,
        calibration_resolution,
    })
}

fn dir_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}
