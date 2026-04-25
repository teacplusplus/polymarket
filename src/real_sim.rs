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

use crate::constants::{CurrencyUpDownOutcome, XFrameIntervalKind};
use crate::history_sim::{
    BuyGate, any_position_would_sell, buy_gate, load_booster, manage_positions, print_sim_stats,
    try_open_position, BookLevel, OpenPosition, SimStats, StrictBook,
};
use crate::project_manager::{LaneFrame, ProjectManager};
use crate::train_mode::{load_calibration, Calibration};
use crate::util::current_timestamp_ms;
use crate::xframe::{XFrame, SIZE};

use anyhow::{anyhow, Result};
use polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest;
use polymarket_client_sdk::clob::types::response::OrderBookSummaryResponse;
use polymarket_client_sdk::types::U256;
use std::collections::HashMap;
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
/// банкролл/drawdown общий для UP+DOWN внутри интервала) и 4 списка открытых
/// позиций. Структуры и поля — из [`crate::history_sim`], не дублируем.
/// Плюс под-структура [`LaneFrameChannels`] — push-канал от `ProjectManager`:
/// `build_frames_from_buffer_lane_once(lane=0)` шлёт сюда `LaneFrame` каждый
/// стабильный тик. Сам state живёт внутри
/// [`crate::project_manager::ProjectManager::real_sim_state`] (`Arc<RwLock<_>>`).
pub struct RealSimState {
    /// Агрегированная статистика по интервалам (bankroll/drawdown/up-down счётчики).
    /// Банкролл общий для UP+DOWN одного интервала — поэтому ключ это
    /// [`XFrameIntervalKind`], а side-счётчики лежат внутри `SimStats.up/down`.
    /// Карта инициализируется оба ключа сразу, воркеры делают `get_mut(&kind).unwrap()`.
    pub stats: HashMap<XFrameIntervalKind, SimStats>,
    /// Открытые позиции 4 лейнов `(интервал, сторона)` в одной карте.
    /// Ключи совпадают с [`LANE_FRAME_ROUTES`] и [`LaneFrameChannels::channels`],
    /// карта инициализируется полным набором пустых `Vec`, поэтому
    /// воркеры могут безопасно делать `.get_mut(&key).unwrap()` без
    /// вставки «по необходимости».
    pub positions: HashMap<(XFrameIntervalKind, CurrencyUpDownOutcome), Vec<OpenPosition>>,
    pub lane_frame_channels: LaneFrameChannels,
}

impl RealSimState {
    pub fn new() -> Self {
        let mut stats = HashMap::with_capacity(2);
        stats.insert(XFrameIntervalKind::FiveMin, SimStats::new());
        stats.insert(XFrameIntervalKind::FifteenMin, SimStats::new());
        let mut positions = HashMap::with_capacity(LANE_FRAME_ROUTES.len());
        for key in LANE_FRAME_ROUTES {
            positions.insert(key, Vec::new());
        }
        Self {
            stats,
            positions,
            lane_frame_channels: LaneFrameChannels::new(),
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

fn interval_label(kind: XFrameIntervalKind) -> &'static str {
    match kind {
        XFrameIntervalKind::FiveMin => "5m",
        XFrameIntervalKind::FifteenMin => "15m",
    }
}

fn side_label(side: CurrencyUpDownOutcome) -> &'static str {
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
    let currency = project_manager.currency.as_str().to_string();
    let version_path = latest_version_path(&currency)
        .ok_or_else(|| anyhow!(
            "нет ни одной версии в xframes/{currency}/ — сначала соберите данные (STATUS=default) и обучите модели (STATUS=train)"
        ))?;
    let version = dir_name(&version_path);
    let tag_prefix = format!("{currency}/{version}");

    println!(
        "[real_sim] версия моделей: {tag_prefix} (из {})",
        version_path.display(),
    );

    // Единое общее состояние (по ТЗ — "Введи новую структуру с Arc<RwLock>"):
    // живёт внутри `ProjectManager` и создаётся прямо в `ProjectManager::new`.
    // Воркеры и фанаут 1s-кадров делят один и тот же `Arc<RwLock<RealSimState>>`.
    let state = project_manager.real_sim_state.clone();
    let channels = state.read().await.lane_frame_channels.channels.clone();

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
        println!(
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
            if let Err(err) = tick_once(
                &book_tx,
                &state,
                interval_kind,
                side,
                &models,
                &tag,
                &mut last_market_id,
                lane_frame,
            ).await {
                eprintln!("[real_sim] {tag}: tick error: {err:#}");
            }
        }
        eprintln!("[real_sim] {tag}: канал закрыт — воркер завершён");
    });
}

/// Один тик воркера: из пришедшего `LaneFrame` берём маркет/asset/frame,
/// сверяем WS vs HTTP, вызываем `manage_positions` (всегда) и
/// `try_open_position` (если WS не отстаёт и это не последний тик).
async fn tick_once(
    book_tx: &mpsc::Sender<BookRequest>,
    state: &Arc<RwLock<RealSimState>>,
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

    let Some(currency_implied_prob) = frame.currency_implied_prob else {
        return Ok(());
    };

    // Событие завершилось, как только `event_remaining_ms ≤ 0`
    // (в `xframe.rs` значение всегда рассчитывается как `end_ms - now`,
    // сентинел `-1` не используется). В этом случае `sell_gate` вместо
    // TP/SL/EV закрывает все позиции по бинарной выплате CTF ($1/шер
    // победителю, $0 проигравшему — без комиссии). Победителя
    // определяем по `currency_implied_prob ≥ 0.5` (в реальности CLOB
    // за секунды до резолюции уже схлопывается к 0/1) и только при
    // `event_over` — пока событие идёт, исход неизвестен, `won = None`
    // честно отражает это, а `sell_gate` на этой ветке к `won` вообще
    // не обращается.
    let event_over = frame.event_remaining_ms <= 0;
    let won: Option<bool> = event_over.then(|| currency_implied_prob >= 0.5);
    let market_changed = last_market_id.as_deref() != Some(market_id.as_str());

    // ── Два независимых гейта ────────────────────────────────────────────────
    // * `has_positions` — нужно ли **звать** `manage_positions`. Даже если
    //   ни одна позиция не закрывается по WS, вызов нужен для обновления
    //   `frames_held`/`p_win_ema` и, на `event_over`, для Resolution (которой
    //   HTTP не требуется — `close_position` не обращается к `strict_book`
    //   в этой ветке).
    // * `needs_http` — нужен ли **HTTP-запрос** стакана. Дёргаем CLOB только
    //   когда реально будем исполнять ордер через `book_fill_*_strict`:
    //     - `needs_sell` (см. `any_position_would_sell`) — любое закрытие,
    //       кроме Resolution: TP/SL/Timeout/EV-exit — через strict-sell;
    //     - `buy_gate == Proceed` — модель хочет открыться (strict-buy).
    //   Если позиции просто висят без триггера и входить не планируем —
    //   HTTP не делаем; `manage_positions` отработает на WS-fallback для
    //   одного только bookkeeping'а (никаких закрытий не сработает,
    //   предикат симметричен фактическим условиям из `manage_positions`).
    let (has_positions, needs_sell, bankroll_snapshot) = {
        let guard = state.read().await;
        let positions = guard
            .positions
            .get(&(interval_kind, side))
            .expect("positions map initialized for all LANE_FRAME_ROUTES");
        let bankroll = guard
            .stats
            .get(&interval_kind)
            .expect("stats map initialized for both intervals")
            .bankroll;
        (
            !positions.is_empty(),
            any_position_would_sell(positions, &frame, event_over),
            bankroll,
        )
    };

    // `buy_gate` сам отказывает, если событие уже завершилось или до резолюции

    let may_open = matches!(
        buy_gate(
            &frame,
            &models.booster_pnl,
            models.calibration_pnl.as_ref(),
            bankroll_snapshot,
        ),
        BuyGate::Proceed { .. }
    );
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
                eprintln!(
                    "[real_sim] {tag}: WS отстаёт — ордербук по HTTP расходится с last XFrame (market={market_id} asset={asset_id}); новые позиции пропускаем, ведём только закрытия"
                );
            }
            lagging
        }
        None => false,
    };

    let mut guard = state.write().await;

    // Смена активного маркета для этого `(интервал, сторона)` — bump `events`
    // один раз на интервал (только таск `Up`: один polymarket market на пару).
    if market_changed && side == CurrencyUpDownOutcome::Up {
        guard
            .stats
            .get_mut(&interval_kind)
            .expect("stats map initialized for both intervals")
            .events += 1;
    }

    // Торговая часть нужна, только если есть что вести или открывать. Если
    // `has_positions=false && may_open=false` — ни `manage_positions`, ни
    // `try_open_position` всё равно ничего бы не сделали; пропускаем целиком.
    if has_positions || may_open {
        // «Купили/продали» получаем напрямую из возвратов `manage_positions` /
        // `try_open_position` — никакого до/после диффа `stats.trades` /
        // `positions.len()` (buy+sell за один тик оставит `len` тем же).
        let mut sold = false;
        let mut bought = false;
        {
            // Destructure `RealSimState` за одну операцию — получаем независимые
            // `&mut` к каждому полю (split borrow `stats` и `positions` —
            // разные поля одной структуры), и только после этого выбираем
            // по ключам конкретные `SimStats` и `Vec<OpenPosition>`.
            let RealSimState {
                stats,
                positions,
                ..
            } = &mut *guard;

            let stats: &mut SimStats = stats
                .get_mut(&interval_kind)
                .expect("stats map initialized for both intervals");
            let positions: &mut Vec<OpenPosition> = positions
                .get_mut(&(interval_kind, side))
                .expect("positions map initialized for all LANE_FRAME_ROUTES");
            let side_stats = match side {
                CurrencyUpDownOutcome::Up => &mut stats.up,
                CurrencyUpDownOutcome::Down => &mut stats.down,
            };

            // 1) Жизненный цикл уже открытых позиций: инкремент `frames_held`,
            //    EMA `p_win`, проверка TP/SL/Timeout/Resolution/EV. Вызываем
            //    **всегда**, когда позиции есть, — даже без HTTP: Resolution
            //    (`event_remaining_ms ≤ 0`) исполняется без `strict_book`, а
            //    «тихий» тик сам ничего не закроет по WS-fallback, т.к.
            //    предикат `needs_sell` симметричен условиям `manage_positions`.
            //    `strict_book.as_ref()` будет `Some` только при `needs_http=true`.
            if has_positions {
                sold = manage_positions(
                    positions,
                    &frame,
                    event_over,
                    won,
                    models.booster_resolution.as_deref(),
                    models.calibration_resolution.as_ref(),
                    side_stats,
                    &mut stats.bankroll,
                    strict_book.as_ref(),
                    tag,
                );
            }

            // 2) BUY: пропускаем, если WS отстаёт. На `event_over` `may_open`
            //    уже `false` (внутри `buy_gate` сработал `LateEntry`), поэтому
            //    отдельный guard здесь не нужен.
            if may_open && !ws_lagging {
                bought = try_open_position(
                    &frame,
                    &models.booster_pnl,
                    models.calibration_pnl.as_ref(),
                    positions,
                    side_stats,
                    stats.bankroll,
                    strict_book.as_ref(),
                    tag,
                );
            }
        }


        // Если случилась buy или sell — печатаем полную статистику (1:1 с history_sim).
        if bought || sold {
            guard
                .stats
                .get_mut(&interval_kind)
                .expect("stats map initialized for both intervals")
                .update_drawdown();
            let stats = guard
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
            println!(
                "[real_sim] {tag}: {action} @ t={} market={market_id} prob={currency_implied_prob:.4}",
                current_timestamp_ms(),
            );
            print_sim_stats(tag, stats);
        }
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
        eprintln!(
            "[real_sim] {tag}: book-coord канал закрыт — strict-fill выключен на тик"
        );
        return None;
    }
    match reply_rx.await {
        Ok(book) => book,
        Err(_) => {
            eprintln!(
                "[real_sim] {tag}: book-coord уронил oneshot до ответа — strict-fill выключен на тик"
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
            eprintln!("[real_sim/book-coord] невалидный asset_id={aid} — отвечаем None");
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
        let n = requests.len();
        match project_manager.clob.order_books(&requests).await {
            Ok(responses) if responses.len() == n => {
                for (aid, resp) in valid_ids.iter().zip(responses.iter()) {
                    let (bids, asks) = parse_book_levels(resp);
                    let book = StrictBook { bids, asks };
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
            Ok(responses) => {
                eprintln!(
                    "[real_sim/book-coord] order_books вернул {} ответов на {n} запросов — отбрасываем батч",
                    responses.len(),
                );
                for senders in by_asset.into_values() {
                    for s in senders {
                        let _ = s.send(None);
                    }
                }
            }
            Err(err) => {
                eprintln!(
                    "[real_sim/book-coord] order_books({n} assets) failed: {err:#}"
                );
                for senders in by_asset.into_values() {
                    for s in senders {
                        let _ = s.send(None);
                    }
                }
            }
        }
    }
    eprintln!("[real_sim/book-coord] mpsc закрыт — координатор завершён");
}

/// Превращает `OrderBookSummaryResponse` в пару `(bids, asks)` формата
/// [`StrictBook`]: уровни [`BookLevel`] в порядке **от лучшего к худшему**.
///
/// Polymarket CLOB отдаёт `bids`/`asks` в обратном порядке («худшее → лучшее»,
/// best = последний элемент), поэтому здесь мы их реверсим. Уровни с
/// неположительной ценой/размером или нечитаемым `Decimal` отбрасываются.
fn parse_book_levels(book: &OrderBookSummaryResponse) -> (Vec<BookLevel>, Vec<BookLevel>) {
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
    (bids, asks)
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
        eprintln!(
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
