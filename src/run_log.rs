pub const WS_LOG_ENABLED: bool = true;
pub const XFRAME_LOG_ENABLED: bool = true;
pub const RTDS_CURRENCY_SEC_BAR_LOG_ENABLED: bool = false;
pub const XFRAME_DUMP_LOG_ENABLED: bool = false;
pub const CURRENCY_UP_DOWN_SIBLING_SLOTS_LOG_ENABLED: bool = false;

/// Логирует, когда после обновления WS одновременно заполнены `fifteen_min` и `five_min`: впервые оба слота, или оба сменились на новые `market_id` / `window_start_sec`.
pub fn currency_updown_sibling_slots_updated(
    ts_ms: i64,
    after_fifteen: Option<(String, i64)>,
    after_five: Option<(String, i64)>,
) {
    if !CURRENCY_UP_DOWN_SIBLING_SLOTS_LOG_ENABLED {
        return;
    }
    let after_both = after_fifteen.is_some() && after_five.is_some();
    let pair_full = after_both;
    if !pair_full {
        return;
    }
    eprintln!(
        "{ts_ms} currency_updown_sibling: оба слота выставлены | стало 15m={after_fifteen:?} 5m={after_five:?}"
    );
}

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
    price_to_beat: Option<f64>,
    market_ids: &[String],
    asset_ids: &[String],
    remain_ms: u64,
    wall_end_ms: i64,
) {
    if !WS_LOG_ENABLED {
        return;
    }
    let polymarket_event_url = format!("https://polymarket.com/event/{slug}");
    let price_to_beat_str = price_to_beat
        .map(|p| format!("{p}"))
        .unwrap_or_else(|| "—".to_string());
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
        "[{slug_mid}] ws: запускаю market ws | polymarket={polymarket_event_url} | price_to_beat={price_to_beat_str} | market (condition_id)=[{markets}] | asset_id (clob)=[{assets}] | session ~{remain_ms} ms до wall_end_ms={wall_end_ms}"
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

/// Финальный кадр после merge other/sibling, непосредственно перед записью в `xframes_by_market`.
pub fn xframe_stored(
    frame: &crate::xframe::XFrame<{ crate::xframe::SIZE }>,
) {
    if !XFRAME_LOG_ENABLED {
        return;
    }
    // let frame_one_line = format!("{frame:?}").replace('\n', " ").replace('\r', "");
    // eprintln!(
    //     "{frame_one_line}"
    // );
}

pub fn xframe_dump_written(
    path: &std::path::Path,
    market_id: &str,
    frame_count: usize,
    bytes: usize,
) {
    if !XFRAME_DUMP_LOG_ENABLED {
        return;
    }
    eprintln!(
        "xframe_dump: wrote {bytes} bytes, {frame_count} frames | market_id={market_id} | path={}",
        path.display()
    );
}

pub fn rtds_currency_sec_bar_inserted(
    pair_symbol: &str,
    bucket_sec: i64,
    price: f64,
    price_timestamp_ms: i64,
    message_timestamp_ms: i64,
    bars_in_history: usize,
) {
    if !RTDS_CURRENCY_SEC_BAR_LOG_ENABLED {
        return;
    }
    eprintln!(
        "rtds sec bar ({pair_symbol}): bucket_sec={bucket_sec} price={price} price_ts_ms={price_timestamp_ms} rtds_msg_ts_ms={message_timestamp_ms} bars_in_history={bars_in_history}"
    );
}