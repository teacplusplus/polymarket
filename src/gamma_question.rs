//! Разбор поля `question` Gamma для up/down по валюте (окно времени в ET).

use chrono::{Datelike, LocalResult, NaiveDate, NaiveTime, TimeZone};
use chrono_tz::America::New_York;

use crate::constants::CurrencyUpDownDelayClass;

/// Парсит `question` Gamma в духе `Bitcoin Up or Down - April 8, 3:35PM-3:40PM ET`:  
/// если окно ровно 5 минут, возвращает класс пятиминутки внутри 15-минутного блока.
pub fn currency_up_down_five_min_slot_from_gamma_question(
    question: &str,
) -> Option<CurrencyUpDownDelayClass> {
    let (_, market_info) = question.split_once(" - ")?;
    let (_, time_part) = market_info.split_once(", ")?;
    let (start_time, end_with_tz) = time_part.split_once('-')?;
    let end_time = end_with_tz.split_whitespace().next()?;

    let (start_h, start_m) = parse_gamma_question_time_12h(start_time)?;
    let (end_h, end_m) = parse_gamma_question_time_12h(end_time)?;

    let start_total = (start_h as i32) * 60 + start_m as i32;
    let mut end_total = (end_h as i32) * 60 + end_m as i32;
    if end_total < start_total {
        end_total += 24 * 60;
    }
    let duration_min = end_total - start_total;
    if duration_min != 5 {
        return None;
    }

    let minutes_since_midnight = (start_h as i32) * 60 + start_m as i32;
    let rem = minutes_since_midnight.rem_euclid(15);
    if rem % 5 != 0 {
        return None;
    }
    CurrencyUpDownDelayClass::from_i32(rem / 5)
}

fn english_month_abbrev_or_full(month_token: &str) -> Option<u32> {
    match month_token.trim().to_ascii_lowercase().as_str() {
        "january" | "jan" => Some(1),
        "february" | "feb" => Some(2),
        "march" | "mar" => Some(3),
        "april" | "apr" => Some(4),
        "may" => Some(5),
        "june" | "jun" => Some(6),
        "july" | "jul" => Some(7),
        "august" | "aug" => Some(8),
        "september" | "sep" | "sept" => Some(9),
        "october" | "oct" => Some(10),
        "november" | "nov" => Some(11),
        "december" | "dec" => Some(12),
        _ => None,
    }
}

/// Старт окна из `question` Gamma в духе `Bitcoin Up or Down - April 8, 3:35PM-3:40PM ET` (интерпретация в America/New_York).
/// `reference_unix_sec` задаёт календарный год и границы DST (обычно `window_start` из slug).
/// Возвращает `(unix_start_sec, duration_min)` для окна ровно 5 или 15 минут.
pub fn currency_updown_question_window_start_unix_sec(
    question: &str,
    reference_unix_sec: i64,
) -> Option<(i64, i64)> {
    let (_, market_info) = question.split_once(" - ")?;
    let (date_part, time_part) = market_info.split_once(", ")?;
    let mut date_parts = date_part.split_whitespace();
    let month = english_month_abbrev_or_full(date_parts.next()?)?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    let (start_str, end_rest) = time_part.split_once('-')?;
    let end_time = end_rest.split_whitespace().next()?;
    let (start_hour, start_minute) = parse_gamma_question_time_12h(start_str.trim())?;
    let (end_hour, end_minute) = parse_gamma_question_time_12h(end_time.trim())?;
    let start_total = (start_hour as i32) * 60 + start_minute as i32;
    let mut end_total = (end_hour as i32) * 60 + end_minute as i32;
    if end_total < start_total {
        end_total += 24 * 60;
    }
    let duration_min = (end_total - start_total) as i64;
    if duration_min != 5 && duration_min != 15 {
        return None;
    }
    let ref_et = New_York.timestamp_opt(reference_unix_sec, 0).single()?;
    let year = ref_et.year();
    let naive_date = NaiveDate::from_ymd_opt(year, month, day)?;
    let naive_time = NaiveTime::from_hms_opt(start_hour, start_minute, 0)?;
    let naive_dt = naive_date.and_time(naive_time);
    let unix_start_sec = match New_York.from_local_datetime(&naive_dt) {
        LocalResult::Single(dt) => dt.timestamp(),
        LocalResult::Ambiguous(_, latest) => latest.timestamp(),
        LocalResult::None => return None,
    };
    Some((unix_start_sec, duration_min))
}

fn parse_gamma_question_time_12h(value: &str) -> Option<(u32, u32)> {
    let (hour_raw, minute_ampm) = value.split_once(':')?;
    if minute_ampm.len() < 4 {
        return None;
    }
    let hour_12 = hour_raw.parse::<u32>().ok()?;
    let minute = minute_ampm[..2].parse::<u32>().ok()?;
    let am_pm = &minute_ampm[2..];
    let hour_24 = match am_pm {
        "AM" if hour_12 == 12 => 0,
        "AM" => hour_12,
        "PM" if hour_12 == 12 => 12,
        "PM" => hour_12 + 12,
        _ => return None,
    };
    Some((hour_24, minute))
}
