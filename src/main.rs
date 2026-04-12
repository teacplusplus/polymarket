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

use anyhow::Result;
use project_manager::ProjectManager;

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("rustls: install ring CryptoProvider (needed for WebSocket TLS)");

    let _project_manager = ProjectManager::new("btc".to_string());

    std::future::pending::<()>().await;
    Ok(())
}
