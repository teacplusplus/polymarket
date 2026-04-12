pub mod constants;
pub mod util;
pub mod gamma_question;
pub mod btc_updown_sibling;
pub mod xframe;
pub mod project_manager;
pub mod market_snapshot;
pub mod run_log;
pub mod btcusdt_ws;
pub mod data_ws;
pub mod xframe_dump;

use anyhow::Result;
use project_manager::{ProjectManager, FIFTEEN_MIN_SEC, FIVE_MIN_SEC};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::util::{
    current_timestamp_ms, fetch_gamma_event_data_for_slug,
    fetch_price_to_beat_from_polymarket_event_page, GammaEventSlugData,
};
use crate::constants::XFrameIntervalKind;

async fn run_btc_updown_interval(
    project_manager: Arc<ProjectManager>,
    period_sec: i64,
    slug_mid: &'static str,
) {
    let xframe_interval_kind = match period_sec {
        ps if ps == FIVE_MIN_SEC => XFrameIntervalKind::FiveMin,
        ps if ps == FIFTEEN_MIN_SEC => XFrameIntervalKind::FifteenMin,
        _ => XFrameIntervalKind::FifteenMin,
    };

    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tick.tick().await;
        let now_ms = current_timestamp_ms();
        let poly_sec = now_ms / 1000;
        let window_start_sec = (poly_sec / period_sec) * period_sec;     
        let ws_end_sec = window_start_sec + period_sec;

        if now_ms >= ws_end_sec * 1000 {
            continue;
        }

        let mut ws_handle: Option<tokio::task::JoinHandle<()>> = None;
        let mut subscribed: Vec<String> = Vec::new();
        let mut ws_session_market_id: Option<String> = None;
        let mut ws_session_gamma_question: Option<String> = None;

        while current_timestamp_ms() < ws_end_sec * 1000 {
            let now_poly_ms = current_timestamp_ms();
            if now_poly_ms >= ws_end_sec * 1000 {
                break;
            }

            let slug = format!("btc-updown-{slug_mid}-{window_start_sec}");
            let http_gamma = project_manager.http.clone();
            let http_price = project_manager.http.clone();
            let slug_gamma = slug.clone();
            let slug_price = slug.clone();

            let (gamma_join_res, price_join_res) = tokio::join!(
                tokio::spawn(async move {
                    fetch_gamma_event_data_for_slug(http_gamma.as_ref(), &slug_gamma).await
                }),
                tokio::spawn(async move {
                    fetch_price_to_beat_from_polymarket_event_page(http_price.as_ref(), &slug_price).await
                }),
            );

            let (
                ids,
                market_event_start_ms,
                market_event_end_ms,
                gamma_question,
                btc_up_down_by_asset_id,
            ) = match gamma_join_res {
                Ok(Ok(GammaEventSlugData {
                    clob_token_ids,
                    btc_up_down_by_asset_id,
                    market_event_start_ms,
                    market_event_end_ms,
                    gamma_question,
                })) => (
                    clob_token_ids,
                    market_event_start_ms,
                    market_event_end_ms,
                    gamma_question,
                    btc_up_down_by_asset_id,
                ),
                Ok(Err(e)) => {
                    run_log::gamma_fetch_err(slug_mid, &slug, &e);
                    continue;
                }
                Err(e) => {
                    run_log::gamma_fetch_err(slug_mid, &slug, &format!("join gamma: {e}"));
                    continue;
                }
            };

            let price_to_beat = match price_join_res {
                Ok(Ok(price_to_beat)) => Some(price_to_beat),
                Ok(Err(e)) => {
                    run_log::gamma_fetch_err(slug_mid, &slug, &e);
                    continue;
                }
                Err(e) => {
                    run_log::gamma_fetch_err(slug_mid, &slug, &format!("join price_to_beat: {e}"));
                    continue;
                }
            };

            let wall_end_ms = market_event_end_ms
                .values()
                .copied()
                .flatten()
                .max()
                .unwrap_or(ws_end_sec * 1000);

            let market_ids: Vec<String> = market_event_end_ms.keys().cloned().collect();

            project_manager
                .merge_market_event_data(
                    market_event_start_ms,
                    market_event_end_ms,
                    price_to_beat,
                    gamma_question.clone(),
                    btc_up_down_by_asset_id,
                )
                .await;

            let should_restart_ws = ids != subscribed;

            if should_restart_ws {
                if let Some(h) = ws_handle.take() {
                    run_log::ws_stop_replace(slug_mid, &slug, subscribed.len());
                    h.abort();
                }

                let remain_ms = (wall_end_ms - current_timestamp_ms()).max(0) as u64;
                let session_deadline = Instant::now() + Duration::from_millis(remain_ms);

                run_log::ws_start(
                    slug_mid,
                    &slug,
                    price_to_beat,
                    &market_ids,
                    &ids,
                    remain_ms,
                    wall_end_ms,
                );

                match data_ws::spawn_bounded_market_ws(
                    project_manager.clone(),
                    ids.clone(),
                    session_deadline,
                    xframe_interval_kind,
                ) {
                    Ok(h) => {
                        project_manager.on_btc_updown_ws_connected(
                            period_sec,
                            window_start_sec,
                            &market_ids,
                            gamma_question.as_deref()).await;

                        project_manager
                            .record_btc_updown_ws_connect(&market_ids)
                            .await;

                        ws_handle = Some(h);
                        subscribed = ids.clone();
                        ws_session_market_id = market_ids.first().cloned();
                        ws_session_gamma_question = gamma_question.clone();
                    }
                    Err(e) => {
                        run_log::ws_spawn_err(slug_mid, &slug, &e);
                        continue;
                    }
                }
            }

            tick.tick().await;
        }

        if let Some(h) = ws_handle.take() {
            let slug = format!("btc-updown-{slug_mid}-{window_start_sec}");
            run_log::ws_window_end_wait(slug_mid, &slug, subscribed.len());
            let _ = h.await;
            run_log::ws_task_joined(slug_mid, &slug);
            if let Some(mid) = ws_session_market_id {
                xframe_dump::spawn_dump_market_xframes_binary(
                    project_manager.clone(),
                    mid,
                    ws_session_gamma_question,
                );
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("rustls: install ring CryptoProvider (needed for WebSocket TLS)");

    let project_manager = ProjectManager::new();
    tokio::spawn(run_btc_updown_interval(
        project_manager.clone(),
        FIVE_MIN_SEC,
        "5m",
    ));
    tokio::spawn(run_btc_updown_interval(
        project_manager.clone(),
        FIFTEEN_MIN_SEC,
        "15m",
    ));

    std::future::pending::<()>().await;
    Ok(())
}
