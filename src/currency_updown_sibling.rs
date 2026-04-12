//! Пара up/down по валюте 5m ↔ 15m: длительности окон и состояние sibling-маркетов для [`crate::project_manager::ProjectManager`].

use crate::constants::{CurrencyUpDownInterval, FIFTEEN_MIN_SEC, FIVE_MIN_SEC};
use crate::gamma_question::currency_updown_question_window_start_unix_sec;
use crate::run_log;
use crate::util::current_timestamp_ms;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct CurrencyUpDownSiblingSlot {
    pub market_id: String,
    /// Старт окна из текста Gamma `question` (ET → Unix-сек), см. [`crate::gamma_question::currency_updown_question_window_start_unix_sec`].
    pub window_start_sec: i64,
}

/// Активные маркеты по WS: один слот 15m и один 5m; [`update_currency_updown_sibling_slots`] заполняет слот после подписки на market WS.
/// Множество `asset_id` по `condition_id` для sibling merge — в [`crate::project_manager::ProjectManager::market_asset_ids_by_market`].
#[derive(Debug, Default)]
pub struct CurrencyUpDownSiblingState {
    pub fifteen_min: Option<CurrencyUpDownSiblingSlot>,
    pub five_min: Option<CurrencyUpDownSiblingSlot>,
}

impl CurrencyUpDownSiblingState {
    /// Пара `condition_id` (5m, 15m), если оба слота заполнены, окна согласованы и маркеты разные.
    pub fn paired_five_and_fifteen_market_ids(&self) -> Option<(String, String)> {
        let five = self.five_min.as_ref()?;
        let fifteen = self.fifteen_min.as_ref()?;
        if five.market_id == fifteen.market_id {
            return None;
        }
        if !five_min_belongs_to_fifteen_window(five.window_start_sec, fifteen.window_start_sec) {
            return None;
        }
        Some((five.market_id.clone(), fifteen.market_id.clone()))
    }
}

/// `window_start_sec` в слотах — Unix-старт из `question`; 5m должна быть одной из трёх пятиминуток внутри 15m.
pub fn five_min_belongs_to_fifteen_window(five_start_unix: i64, fifteen_start_unix: i64) -> bool {
    let offset_from_fifteen_start_sec = five_start_unix - fifteen_start_unix;
    offset_from_fifteen_start_sec >= 0
        && offset_from_fifteen_start_sec < FIFTEEN_MIN_SEC
        && offset_from_fifteen_start_sec.rem_euclid(FIVE_MIN_SEC) == 0
}

/// Вызывать после успешной подписки на CLOB market WS для окна up/down по валюте: пишет в слот 5m или 15m `market_id` и окно из Gamma `question`.
/// `slug_window_start_unix_sec` — секунда из slug (опора для года/DST при разборе `question`).
/// Старт окна в слоте — из [`crate::gamma_question::currency_updown_question_window_start_unix_sec`].
/// Пару для merge sibling-признаков читает [`CurrencyUpDownSiblingState::paired_five_and_fifteen_market_ids`].
pub async fn update_currency_updown_sibling_slots(
    sibling_state: Arc<RwLock<CurrencyUpDownSiblingState>>,
    interval_sec: i64,
    slug_window_start_unix_sec: i64,
    market_ids: &[String],
    gamma_question: Option<&str>,
) {
    let Some(condition_id) = market_ids.iter().filter(|s| !s.is_empty()).min().cloned() else {
        return;
    };

    let Some(question) = gamma_question.filter(|s| !s.is_empty()) else {
        eprintln!("currency_updown_sibling: пустой gamma_question — слот не обновлён");
        return;
    };

    let Some((parsed_start_unix, duration_min)) =
        currency_updown_question_window_start_unix_sec(question, slug_window_start_unix_sec)
    else {
        eprintln!("currency_updown_sibling: не разобрать окно из question для market_id={condition_id}");
        return;
    };

    let Some(horizon) = CurrencyUpDownInterval::try_from_interval_sec(interval_sec) else {
        return;
    };
    let expected_dur_min = horizon.duration_minutes();
    if duration_min != expected_dur_min {
        eprintln!("currency_updown_sibling: в question длительность {duration_min} мин, ожидалось {expected_dur_min} для interval_sec={interval_sec} (market_id={condition_id})");
        return;
    }

    let mut sibling_state_lock = sibling_state.write().await;
    match horizon {
        CurrencyUpDownInterval::FifteenMin => {
            let prev_fifteen_start = sibling_state_lock
                .fifteen_min
                .as_ref()
                .map(|s| s.window_start_sec);
            if sibling_state_lock.five_min.is_some()
                && prev_fifteen_start.is_some_and(|prev| parsed_start_unix > prev)
            {
                sibling_state_lock.five_min = None;
            }
            sibling_state_lock.fifteen_min = Some(CurrencyUpDownSiblingSlot {
                market_id: condition_id,
                window_start_sec: parsed_start_unix,
            });
        }
        CurrencyUpDownInterval::FiveMin => {
            if let Some(ref fifteen_slot) = sibling_state_lock.fifteen_min {
                if !five_min_belongs_to_fifteen_window(parsed_start_unix, fifteen_slot.window_start_sec) {
                    return;
                }
            }
            sibling_state_lock.five_min = Some(CurrencyUpDownSiblingSlot {
                market_id: condition_id,
                window_start_sec: parsed_start_unix,
            });
        }
    }
    let after_fifteen = sibling_state_lock
        .fifteen_min
        .as_ref()
        .map(|s| (s.market_id.clone(), s.window_start_sec));
    let after_five = sibling_state_lock
        .five_min
        .as_ref()
        .map(|s| (s.market_id.clone(), s.window_start_sec));
    run_log::currency_updown_sibling_slots_updated(
        current_timestamp_ms(),
        after_fifteen,
        after_five,
    );
}
