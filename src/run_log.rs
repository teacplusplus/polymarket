pub const WS_LOG_ENABLED: bool = true;
pub const XFRAME_LOG_ENABLED: bool = false;
pub const BTCUSDT_SEC_BAR_LOG_ENABLED: bool = false;

pub fn gamma_fetch_err(slug_mid: &str, slug: &str, err: impl std::fmt::Display) {
    if !WS_LOG_ENABLED {
        return;
    }
    eprintln!("[{slug_mid}] Gamma slug={slug}: {err}");
}

pub fn ws_stop_replace(slug_mid: &str, slug: &str, prev_token_count: usize) {
    if !WS_LOG_ENABLED {
        return;
    }
    eprintln!(
        "[{slug_mid}] ws: останавливаю прежнее подключение (смена подписки), slug={slug}, было {prev_token_count} clob token id"
    );
}

pub fn ws_start(
    slug_mid: &str,
    slug: &str,
    market_ids: &[String],
    asset_ids: &[String],
    remain_ms: u64,
    wall_end_ms: i64,
) {
    if !WS_LOG_ENABLED {
        return;
    }
    let markets = if market_ids.is_empty() {
        String::from("(нет condition_id в Gamma)")
    } else {
        market_ids.join(", ")
    };
    let assets = if asset_ids.is_empty() {
        String::from("(нет)")
    } else {
        asset_ids.join(", ")
    };
    eprintln!(
        "[{slug_mid}] ws: запускаю market ws | slug={slug} | market (condition_id)=[{markets}] | asset_id (clob)=[{assets}] | session ~{remain_ms} ms до wall_end_ms={wall_end_ms}"
    );
}

pub fn ws_spawn_err(slug_mid: &str, slug: &str, err: impl std::fmt::Display) {
    if !WS_LOG_ENABLED {
        return;
    }
    eprintln!("[{slug_mid}] ws: ошибка spawn для slug={slug}: {err}");
}

pub fn ws_window_end_wait(slug_mid: &str, slug: &str, token_count: usize) {
    if !WS_LOG_ENABLED {
        return;
    }
    eprintln!(
        "[{slug_mid}] ws: конец окна по времени, жду завершение task | slug={slug} | {token_count} clob token id"
    );
}

pub fn ws_task_joined(slug_mid: &str, slug: &str) {
    if !WS_LOG_ENABLED {
        return;
    }
    eprintln!("[{slug_mid}] ws: task market ws завершён | slug={slug}");
}

pub fn market_ws_session_err(err: impl std::fmt::Display) {
    if !WS_LOG_ENABLED {
        return;
    }
    eprintln!("market ws session ended with error: {err}");
}

pub fn xframe_built(
    market_id: &str,
    asset_id: &str,
) {
    if !XFRAME_LOG_ENABLED {
        return;
    }
    eprintln!(
        "xframe: built | market={market_id} | asset={asset_id}"
    );
}

pub fn btcusdt_sec_bar_inserted(
    bucket_sec: i64,
    price: f64,
    price_timestamp_ms: i64,
    message_timestamp_ms: i64,
    bars_in_history: usize,
) {
    if !BTCUSDT_SEC_BAR_LOG_ENABLED {
        return;
    }
    eprintln!(
        "btcusdt sec bar: bucket_sec={bucket_sec} price={price} price_ts_ms={price_timestamp_ms} rtds_msg_ts_ms={message_timestamp_ms} bars_in_history={bars_in_history}"
    );
}