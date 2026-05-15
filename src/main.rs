pub mod account;
pub mod account_exit;
pub mod account_order;
pub mod account_order_completion;
pub mod account_submit;
pub mod account_ws;
pub mod constants;
pub mod currency_updown_sibling;
pub mod currency_ws;
pub mod data_ws;
pub mod gamma_question;
pub mod history_sim;
pub mod market_snapshot;
pub mod migration;
pub mod migration_graph_html;
pub mod migration_price_to_beat;
pub mod poly_chain;
pub mod project_manager;
pub mod real_sim;
pub mod run_log;
pub mod sim_stats;
pub mod tee_log;
pub mod trade_csv_log;
pub mod train_mode;
pub mod util;
pub mod xframe;
pub mod xframe_dump;
pub mod xframe_graph_dump;

use account::Account;
use anyhow::Result;
use project_manager::ProjectManager;

/// Валюты процесса: по одному [`ProjectManager`] на элемент (Default/RealSim/миграции).
pub const CURRENCIES: &[&str] = &["btc"];

/// Режим из переменной окружения `STATUS` (`.env`).
#[derive(Debug)]
enum AppMode {
    /// Бесконечный сбор данных по WS.
    Default,
    /// Обучение XGBoost по дампам.
    Train,
    /// Симуляция на дампах (P&L).
    HistorySim,
    /// Подряд Train и HistorySim (разные tee-лога).
    TrainAndHistorySim,
    /// Живой WS, виртуальные fill'ы без CLOB.
    RealSim,
    /// Живой WS + ордера на CLOB (heartbeat, user-WS, auth).
    RealSimWithSubmit,
    /// Миграция полей [`crate::xframe::XFrame`] в дампах.
    Migrate,
    /// Backfill `price_to_beat` (Vatic API).
    MigratePriceToBeat,
    /// Генерация HTML графиков из `.bin`.
    MigrateGraphHtml,
}

impl AppMode {
    fn from_env() -> Self {
        match std::env::var("STATUS").as_deref() {
            Ok("train") => AppMode::Train,
            Ok("history_sim") => AppMode::HistorySim,
            Ok("train_and_history_sim") => AppMode::TrainAndHistorySim,
            Ok("real_sim") => AppMode::RealSim,
            Ok("real_sim_with_submit") => AppMode::RealSimWithSubmit,
            Ok("migrate") => AppMode::Migrate,
            Ok("migrate_price_to_beat") => AppMode::MigratePriceToBeat,
            Ok("migrate_graph_html") => AppMode::MigrateGraphHtml,
            _ => AppMode::Default,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    match util::detect_country_and_ip().await {
        Some(info) => {
            let country = info.country.as_deref().unwrap_or("?");
            let ip = info.ip.as_deref().unwrap_or("?");
            let region = info.region.as_deref().unwrap_or("");
            let region_suffix = if region.is_empty() {
                String::new()
            } else {
                format!(", регион: {region}")
            };
            println!(
                "Polymarket geoblock: blocked={}, страна: {country}, IP: {ip}{region_suffix}",
                info.blocked
            );
        }
        None => println!("Polymarket geoblock: не удалось определить (api/geoblock)"),
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
            train_mode::run_train_mode().await?;
            tee_log::finish_tee_log();
            history_sim::run_sim_mode().await?;
        }
        AppMode::Migrate => {
            migration::run_migration()?;
        }
        AppMode::MigratePriceToBeat => {
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

            tee_log::init_tee_log_file(
                std::path::Path::new("xframes/last_default.txt"),
                "default",
            )?;

            let account = Account::new_shared();

            for currency in CURRENCIES {
                let _ = ProjectManager::new((*currency).to_string(), account.clone());
            }

            std::future::pending::<()>().await;
        }
        AppMode::RealSim => {
            rustls::crypto::ring::default_provider()
                .install_default()
                .expect("rustls: install ring CryptoProvider (needed for WebSocket TLS)");

            tee_log::init_tee_log_file(
                std::path::Path::new("xframes/last_real_sim.txt"),
                "real_sim",
            )?;

            trade_csv_log::init_trade_csv_log_file(std::path::Path::new(
                "xframes/last_real_sim_trades.csv",
            ))?;
            trade_csv_log::set_current_regime("real_sim");

            let account = Account::new_shared();

            for currency in CURRENCIES {
                let project_manager = ProjectManager::new((*currency).to_string(), account.clone());
                real_sim::run_real_sim(project_manager, false).await?;
            }

            std::future::pending::<()>().await;
        }
        AppMode::RealSimWithSubmit => {
            rustls::crypto::ring::default_provider()
                .install_default()
                .expect("rustls: install ring CryptoProvider (needed for WebSocket TLS)");

            tee_log::init_tee_log_file(
                std::path::Path::new("xframes/last_real_sim_with_submit.txt"),
                "real_sim_with_submit",
            )?;
            trade_csv_log::init_submit_trade_csv_log_file(std::path::Path::new(
                "xframes/last_real_sim_with_submit_trades.csv",
            ))?;
            trade_csv_log::set_current_regime("real_sim_with_submit");

            let account = Account::new_shared();

            account::try_authenticate_clob_for_heartbeats(&account).await;
            assert!(
                account.clob_authed.load().is_some(),
                "[main/RealSimWithSubmit] CLOB auth не поднялся — submit-режим без \
                 авторизации бесполезен. Проверьте `POLY_PRIVATE_KEY` в окружении \
                 (см. `account::POLY_PRIVATE_KEY_ENV`) и сообщение от \
                 `try_authenticate_clob_for_heartbeats` в логах выше.",
            );

            account::spawn_heartbeat(account.clone());
            account_ws::spawn_user_ws_listener(account.clone());

            for currency in CURRENCIES {
                let project_manager = ProjectManager::new((*currency).to_string(), account.clone());
                real_sim::run_real_sim(project_manager, true).await?;
            }

            wait_for_shutdown_signal().await;
            crate::tee_println!("[main] получен shutdown-сигнал → graceful exit");
            account_exit::graceful_exit(account.clone()).await;
            std::process::exit(0);
        }
    }

    Ok(())
}

/// SIGINT или SIGTERM (unix); только для [`AppMode::RealSimWithSubmit`].
#[cfg(unix)]
async fn wait_for_shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(err) => {
            crate::tee_eprintln!(
                "[main] не удалось зарегистрировать SIGTERM-хэндлер: {err:#}; \
                 ждём только SIGINT (Ctrl+C)"
            );
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            crate::tee_println!("[main] SIGINT (Ctrl+C)");
        }
        _ = sigterm.recv() => {
            crate::tee_println!("[main] SIGTERM");
        }
    }
}

/// Не-unix: только Ctrl+C.
#[cfg(not(unix))]
async fn wait_for_shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    crate::tee_println!("[main] SIGINT (Ctrl+C)");
}
