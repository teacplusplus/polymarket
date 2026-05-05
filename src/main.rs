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
pub mod account;
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
    /// (5m × 15m × up/down) с логикой из `history_sim`.
    RealSim,
    /// Одноразовая миграция дампов `xframes/...` под актуальную раскладку
    /// `XFrame` (см. `migration::run_migration`). Вызывается вручную через
    /// `STATUS=migrate`; идемпотентна — повторный запуск ничего не сделает.
    Migrate,
    /// Одноразовая миграция `price_to_beat` уже сохранённых дампов
    /// `xframes/{currency}/<size>/...` (см.
    /// `migration_price_to_beat::run_price_to_beat_migration`). Перетягивает
    /// точный `priceToBeat` со страницы `polymarket.com/event/{slug}` (HTTP,
    /// не Gamma API) и пересчитывает зависимые поля кадра
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
            train_mode::run_train_mode()?;
        }
        AppMode::HistorySim => {
            history_sim::run_sim_mode()?;
        }
        AppMode::TrainAndHistorySim => {
            // Train пишет в `xframes/last_train_mode.txt`, sim — в
            // `xframes/last_history_sim.txt`. Между двумя фазами
            // явно дёргаем `tee_log::finish_tee_log`, чтобы первый файл
            // полностью смылся на диск (на случай если `run_sim_mode`
            // упадёт — обучение всё равно сохранится).
            train_mode::run_train_mode()?;
            tee_log::finish_tee_log();
            history_sim::run_sim_mode()?;
        }
        AppMode::Migrate => {
            migration::run_migration()?;
        }
        AppMode::MigratePriceToBeat => {
            // rustls нужен для HTTPS-запросов на polymarket.com через
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

            // См. комментарий в `AppMode::Default` — общий счёт на процесс.
            let account = Account::new_shared();

            for currency in CURRENCIES {
                let project_manager =
                    ProjectManager::new((*currency).to_string(), account.clone());
                real_sim::run_real_sim(project_manager).await?;
            }

            std::future::pending::<()>().await;
        }
    }

    Ok(())
}
