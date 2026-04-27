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
pub mod train_mode;
pub mod tee_log;
pub mod history_sim;
pub mod real_sim;
pub mod account;
pub mod migration;

use anyhow::Result;
use account::Account;
use project_manager::ProjectManager;

/// Список валют, для которых поднимаются независимые `ProjectManager`-ы
/// (свой WS + свой кэш xframes + свои воркеры `real_sim`). Новую валюту
/// достаточно добавить сюда — точки входа режимов `Default` и `RealSim` сами
/// пройдут по массиву.
const CURRENCIES: &[&str] = &["btc"];

/// Режим запуска, считанный из переменной окружения `STATUS` (`.env`).
#[derive(Debug)]
enum AppMode {
    /// Бесконечный сбор рыночных данных через WebSocket.
    Default,
    /// Однократное обучение XGBoost по накопленным дампам и завершение.
    Train,
    /// Историческая симуляция торговли по накопленным дампам с подсчётом P&L.
    HistorySim,
    /// Реальная (виртуальная) торговля по живому WS потоку: поднимает тот же
    /// `ProjectManager` что и `Default`, плюс 4 tokio-воркера раз-в-секунду
    /// (5m × 15m × up/down) с логикой из `history_sim`.
    RealSim,
    /// Одноразовая миграция дампов `xframes/...` под актуальную раскладку
    /// `XFrame` (см. `migration::run_migration`). Вызывается вручную через
    /// `STATUS=migrate`; идемпотентна — повторный запуск ничего не сделает.
    Migrate,
}

impl AppMode {
    fn from_env() -> Self {
        match std::env::var("STATUS").as_deref() {
            Ok("train")         => AppMode::Train,
            Ok("history_sim")   => AppMode::HistorySim,
            Ok("real_sim")      => AppMode::RealSim,
            Ok("migrate")       => AppMode::Migrate,
            _                   => AppMode::Default,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let mode = AppMode::from_env();
    println!("Режим запуска: {mode:?}");

    match mode {
        AppMode::Train => {
            train_mode::run_train_mode()?;
        }
        AppMode::HistorySim => {
            history_sim::run_sim_mode()?;
        }
        AppMode::Migrate => {
            migration::run_migration()?;
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
