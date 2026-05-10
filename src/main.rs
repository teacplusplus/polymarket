pub mod constants;
pub mod util;
pub mod gamma_question;
pub mod currency_updown_sibling;
pub mod xframe;
pub mod project_manager;
pub mod market_snapshot;
pub mod run_log;
pub mod currency_ws;
pub mod data_ws;
pub mod xframe_dump;
pub mod xframe_graph_dump;
pub mod train_mode;
pub mod tee_log;
pub mod history_sim;
pub mod real_sim;
pub mod account_submit;
pub mod account;
pub mod account_order;
pub mod account_ws;
pub mod migration;
pub mod migration_price_to_beat;
pub mod migration_graph_html;
pub mod trade_csv_log;
pub mod poly_chain;

use anyhow::Result;
use account::Account;
use project_manager::ProjectManager;

/// Список валют: независимые `ProjectManager`-ы в `Default`/`RealSim`, обход дампов
/// в миграциях (`migrate`, `migrate_price_to_beat`, `migrate_graph_html`).
/// Новую валюту достаточно добавить сюда.
pub const CURRENCIES: &[&str] = &["btc"];

/// Режим запуска, считанный из переменной окружения `STATUS` (`.env`).
#[derive(Debug)]
enum AppMode {
    /// Бесконечный сбор рыночных данных через WebSocket.
    Default,
    /// Однократное обучение XGBoost по накопленным дампам и завершение.
    Train,
    /// Историческая симуляция торговли по накопленным дампам с подсчётом P&L.
    HistorySim,
    /// Сначала [`AppMode::Train`] (обучение по накопленным дампам),
    /// потом сразу — [`AppMode::HistorySim`] (тест на test-сплите) в
    /// одном процессе. Удобно для итераций «правлю гипотезы → смотрю,
    /// что вышло», без ручной композиции `STATUS=train && STATUS=history_sim`.
    /// Каждый шаг пишет свой отдельный tee-лог
    /// (`xframes/last_train_mode.txt`, `xframes/last_history_sim.txt`),
    /// чтобы анализ обучения и симуляции остался независимым; CSV-лог
    /// per-trade (`last_history_sim_trades.csv`) пишет только sim-фаза.
    TrainAndHistorySim,
    /// Реальная (виртуальная) торговля по живому WS потоку: поднимает тот же
    /// `ProjectManager` что и `Default`, плюс 4 tokio-воркера раз-в-секунду
    /// (5m × 15m × up/down) с логикой из `history_sim`. **Виртуальные fill'ы**:
    /// `book_fill_*_strict` сразу даёт `shares_held`/`buy_price`, PnL финализируется
    /// в `manage_positions`/`Account::resolve_pending_market`. CLOB-ордера НЕ
    /// отправляются, heartbeat / user-WS / auth не поднимаются — для этого см.
    /// [`AppMode::RealSimWithSubmit`].
    RealSim,
    /// То же, что [`AppMode::RealSim`], но с **реальной отправкой ордеров**
    /// на Polymarket CLOB через [`crate::account_order::post_order_on_clob`] /
    /// [`crate::account_order::cancel_order_on_clob`]:
    /// - вход taker BUY со slippage cap'ом ([`SIM_MAX_SLIPPAGE_FROM_L1_PCT`]);
    /// - TP — сразу maker SELL лимиткой по `pos.buy_price + Y_TRAIN_TAKE_PROFIT_PP`
    ///   (выставляется из WS-колбека `apply_user_ws_event_value` сразу после
    ///   BUY-MATCHED + дублируется delayed-verify-таском, дедуп через
    ///   `OpenPosition.tp_placement_attempted`);
    /// - SL/Timeout/EvExit — отмена TP + taker SELL без slippage cap'а;
    /// - PnL финализируется по реальным fills из user-WS `trade` events
    ///   (см. [`crate::account_ws`]).
    ///
    /// Включает heartbeat ([`crate::account::spawn_heartbeat`]), auth-цикл
    /// ([`crate::account::try_authenticate_clob_for_heartbeats`]) и user-WS
    /// листенер ([`crate::account_ws::spawn_user_ws_listener`]).
    RealSimWithSubmit,
    /// Одноразовая миграция дампов `xframes/...` под актуальную раскладку
    /// `XFrame` (см. `migration::run_migration`). Вызывается вручную через
    /// `STATUS=migrate`; идемпотентна — повторный запуск ничего не сделает.
    Migrate,
    /// Одноразовая миграция `price_to_beat` уже сохранённых дампов
    /// `xframes/{currency}/<size>/...` (см.
    /// `migration_price_to_beat::run_price_to_beat_migration`). Перетягивает
    /// точный `priceToBeat` через Vatic API `targets/timestamp`
    /// (`https://api.vatic.trading`) и пересчитывает зависимые поля кадра
    /// (`currency_price_vs_beat_pct`, `sibling_currency_price_vs_beat_pct`).
    /// Запускается через `STATUS=migrate_price_to_beat`; идемпотентна —
    /// повторный запуск на уже исправленных дампах их не меняет.
    MigratePriceToBeat,
    /// Для каждого `xframes/{currency}/<schema>/.../*.bin` создаёт зеркальный
    /// `graph/.../*.html` (см. [`crate::xframe_graph_dump::try_write_graph_html_from_bin_dump`]).
    /// Запуск: `STATUS=migrate_graph_html`.
    MigrateGraphHtml,
}

impl AppMode {
    fn from_env() -> Self {
        match std::env::var("STATUS").as_deref() {
            Ok("train")                 => AppMode::Train,
            Ok("history_sim")           => AppMode::HistorySim,
            Ok("train_and_history_sim") => AppMode::TrainAndHistorySim,
            Ok("real_sim")              => AppMode::RealSim,
            Ok("real_sim_with_submit")  => AppMode::RealSimWithSubmit,
            Ok("migrate")               => AppMode::Migrate,
            Ok("migrate_price_to_beat") => AppMode::MigratePriceToBeat,
            Ok("migrate_graph_html")    => AppMode::MigrateGraphHtml,
            _                           => AppMode::Default,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    match util::detect_country_and_ip().await {
        Some(info) => {
            let country = info.country.as_deref().unwrap_or("?");
            let ip      = info.ip.as_deref().unwrap_or("?");
            println!("Страна: {country}, IP: {ip}");
        }
        None => println!("Страна: не удалось определить (ifconfig.co/json)"),
    }

    let mode = AppMode::from_env();
    println!("Режим запуска: {mode:?}");

    match mode {
        AppMode::Train => {
            train_mode::run_train_mode().await?;
        }
        AppMode::HistorySim => {
            history_sim::run_sim_mode().await?;
        }
        AppMode::TrainAndHistorySim => {
            // Train пишет в `xframes/last_train_mode.txt`, sim — в
            // `xframes/last_history_sim.txt`. Между двумя фазами
            // явно дёргаем `tee_log::finish_tee_log`, чтобы первый файл
            // полностью смылся на диск (на случай если `run_sim_mode`
            // упадёт — обучение всё равно сохранится).
            train_mode::run_train_mode().await?;
            tee_log::finish_tee_log();
            history_sim::run_sim_mode().await?;
        }
        AppMode::Migrate => {
            migration::run_migration()?;
        }
        AppMode::MigratePriceToBeat => {
            // rustls нужен для HTTPS-запросов на api.vatic.trading через
            // `reqwest`; ставим default-провайдер один раз на процесс.
            rustls::crypto::ring::default_provider()
                .install_default()
                .expect("rustls: install ring CryptoProvider (needed for HTTPS)");
            migration_price_to_beat::run_price_to_beat_migration().await?;
        }
        AppMode::MigrateGraphHtml => {
            migration_graph_html::run_graph_html_migration()?;
        }
        AppMode::Default => {
            rustls::crypto::ring::default_provider()
                .install_default()
                .expect("rustls: install ring CryptoProvider (needed for WebSocket TLS)");

            // Как в `RealSim`: `run_log` и прочий вывод через `tee_*` дублируется
            // в файл (пока иначе при `Default` в лог на диск ничего не уходило).
            tee_log::init_tee_log_file(
                std::path::Path::new("xframes/last_default.txt"),
                "default",
            )?;

            // Единый счёт-капитал на все валюты процесса.
            // Создаётся ДО спавна `ProjectManager`-ов и клонируется в каждый
            // через `Arc` — drawdown/bankroll едины поверх всех 4 лейнов
            // (5m up/down × 15m up/down) и всех валют.
            let account = Account::new_shared();

            for currency in CURRENCIES {
                // ProjectManager::new спаунит фоновые таски, удерживающие
                // собственные `Arc`-клоны — возвращаемый Arc можно сразу
                // отпустить, пайплайн продолжит жить. Карта каналов
                // `lane_frame_channels` у `real_sim_state` остаётся пустой,
                // фанаут просто молча отбрасывает кадры.
                let _ = ProjectManager::new((*currency).to_string(), account.clone());
            }

            std::future::pending::<()>().await;
        }
        AppMode::RealSim => {
            rustls::crypto::ring::default_provider()
                .install_default()
                .expect("rustls: install ring CryptoProvider (needed for WebSocket TLS)");

            // TEE_LOG — единый файл прогона на ВЕСЬ процесс real_sim'а
            // (а не на валюту), чтобы все 4×N tokio-воркеров писали в
            // один и тот же `BufWriter<File>`. Открываем ДО спавна
            // `run_real_sim`, чтобы первые `tee_*`-вызовы (`[real_sim]
            // версия моделей`, init-сообщения воркеров) уже попали в
            // файл. Закрытие — на завершении процесса; `BufWriter` сам
            // флашится в Drop'е статика.
            tee_log::init_tee_log_file(
                std::path::Path::new("xframes/last_real_sim.txt"),
                "real_sim",
            )?;

            // Per-trade CSV-лог: одна строка на каждое закрытие позиции
            // (см. `trade_csv_log` модульный комментарий и
            // `Account::resolve_pending_market_sync`). Без этой инициализации
            // `write_trade_csv_row` молча копит строки в `TRADE_CSV_PENDING`
            // и `record_market_outcome` дренирует их в drop — никаких
            // per-trade данных на диск не попадает.
            trade_csv_log::init_trade_csv_log_file(
                std::path::Path::new("xframes/last_real_sim_trades.csv"),
            )?;
            trade_csv_log::set_current_regime("real_sim");

            // См. комментарий в `AppMode::Default` — общий счёт на процесс.
            // В этом режиме CLOB-ордера НЕ отправляются, поэтому
            // `try_authenticate_clob_for_heartbeats` / `spawn_heartbeat` /
            // `spawn_user_ws_listener` НЕ запускаются — все эти таски нужны
            // только для real-торговли (см. [`AppMode::RealSimWithSubmit`]).
            let account = Account::new_shared();

            for currency in CURRENCIES {
                let project_manager =
                    ProjectManager::new((*currency).to_string(), account.clone());
                // submit=false → виртуальные fill'ы / закрытия как раньше
                // (см. doc у [`real_sim::run_real_sim`]).
                real_sim::run_real_sim(project_manager, false).await?;
            }

            std::future::pending::<()>().await;
        }
        AppMode::RealSimWithSubmit => {
            rustls::crypto::ring::default_provider()
                .install_default()
                .expect("rustls: install ring CryptoProvider (needed for WebSocket TLS)");

            // Отдельные tee/CSV-файлы для submit-режима — чтобы анализировать
            // прогоны с реальной торговлей независимо от чисто виртуальных.
            tee_log::init_tee_log_file(
                std::path::Path::new("xframes/last_real_sim_with_submit.txt"),
                "real_sim_with_submit",
            )?;
            trade_csv_log::init_trade_csv_log_file(
                std::path::Path::new("xframes/last_real_sim_with_submit_trades.csv"),
            )?;
            trade_csv_log::set_current_regime("real_sim_with_submit");

            let account = Account::new_shared();

            // CLOB L2-auth + кэш `clob_signer` до фоновых тасков: heartbeat и
            // user-WS читают готовый [`Account::clob_authed`] без гонки «ждём
            // auth внутри heartbeat». Без auth'а submit-режим бесполезен —
            // `post_order_on_clob` упадёт на `clob_authed=None`, ордера не
            // поедут. Лог об отсутствии `POLY_PRIVATE_KEY` поднимется внутри.
            account::try_authenticate_clob_for_heartbeats(&account).await;

            // Глобальный CLOB heartbeat-таск (раз в 5s `POST /v1/heartbeats`),
            // удерживает открытые ордера от автоматической отмены сервером.
            // Без heartbeat'а CLOB снимает все наши ордера через 10s тишины,
            // а мы как раз держим maker TP-лимитки и pending taker'ы.
            account::spawn_heartbeat(account.clone());

            // Глобальный user-WS листенер на процесс — основной канал
            // подтверждения постановок/исполнений и аккумуляции реальных
            // fills для PnL (см. `crate::account_ws::apply_user_ws_event_value`).
            // Без него мы остаёмся слепы относительно реальных shares/cost.
            account_ws::spawn_user_ws_listener(account.clone());

            for currency in CURRENCIES {
                let project_manager =
                    ProjectManager::new((*currency).to_string(), account.clone());
                // submit=true → реальные `post_order_on_clob` / `cancel_order_on_clob`,
                // PnL финализируется по WS `trade` events.
                real_sim::run_real_sim(project_manager, true).await?;
            }

            std::future::pending::<()>().await;
        }
    }

    Ok(())
}
