// //! Redeem-01 tail rule.
// //!
// //! This module contains the live/project decision rule only. It does not replay
// //! bot activity and does not depend on precomputed market/asset tables.
//
// use crate::account::SharedAccount;
// use crate::constants::{CurrencyUpDownOutcome, FIFTEEN_MIN_SEC, FIVE_MIN_SEC, XFrameIntervalKind};
// use crate::history_sim::{OpenPosition, SharedOpenPosition};
// use crate::project_manager::ProjectManager;
// use std::sync::Arc;
//
// /// Синтетический [`OpenPosition`] для рынка, который ещё **не наступил** (нет ни
// /// [`crate::xframe::XFrame`], ни [`crate::history_sim::StrictBook`]). В отличие от
// /// [`crate::history_sim::open_position`] никакого fill по книге нет: размер/цена
// /// считаются от переданной `assumed_buy_price`
// /// (`shares = position_size / price`, planned fee = 0 для maker), а CSV-поля модели
// /// (`raw/cal_pred`, `kelly_f`, SHAP) заполняются нулями/пусто — они станут
// /// неактуальны/перезапишутся после фактического fill при промоушене.
// /// `event_remaining_ms_at_open` берём как полное окно интервала (рынок ещё впереди).
// #[allow(clippy::too_many_arguments)]
// fn future_open_position(
//     asset_id: &str,
//     market_id: &str,
//     currency: &str,
//     interval_kind: XFrameIntervalKind,
//     side: CurrencyUpDownOutcome,
//     polymarket_url: &str,
//     event_end_ms: Option<i64>,
//     position_size: f64,
//     assumed_buy_price: f64,
//     opened_in_hold_zone: bool,
//     redeem_01: bool,
//     redeem_x: bool,
//     price_to_beat: Option<f64>,
//     final_price: Option<f64>,
//     graph_dump_bin_path: &str,
//     gamma_question_at_open: Option<&str>,
// ) -> Option<OpenPosition> {
//     if !(position_size > 0.0 && position_size.is_finite()) {
//         return None;
//     }
//     let buy_price = assumed_buy_price.clamp(0.001, 0.999);
//     let nominal_shares = position_size / buy_price;
//     if !(nominal_shares > 0.0 && nominal_shares.is_finite()) {
//         return None;
//     }
//     let fee_usdc = 0.0;
//     let shares_held = nominal_shares;
//
//     let id = uuid::Uuid::new_v4().to_string();
//
//     Some(OpenPosition {
//         id,
//         asset_id: asset_id.to_string(),
//         market_id: market_id.to_string(),
//         shares_held,
//         planned_shares_held: shares_held,
//         entry_prob: buy_price,
//         buy_price,
//         planned_buy_price: buy_price,
//         sell_vwap_entry: buy_price,
//         position_size,
//         planned_entry_cost: position_size,
//         best_bid_at_entry: None,
//         frames_held: 0,
//         opened_in_hold_zone,
//         redeem_01,
//         redeem_x,
//         raw_pred_at_open: 0.0,
//         cal_pred_at_open: 0.0,
//         kelly_f_at_open: 0.0,
//         event_remaining_ms_at_open: interval_kind.interval_ms(),
//         xframe_interval_type_at_open: interval_kind.as_i32(),
//         currency_up_down_outcome_at_open: side.as_i32(),
//         currency: currency.to_string(),
//         polymarket_url: polymarket_url.to_string(),
//         price_to_beat,
//         final_price,
//         event_end_ms,
//         graph_dump_bin_path: graph_dump_bin_path.to_string(),
//         gamma_question_at_open: gamma_question_at_open.map(|s| s.to_string()),
//         pnl_top5_shap_at_open: String::new(),
//         open_order_id: None,
//         open_buy_invoke: None,
//         maker_tp_position: None,
//         taker_positions: Vec::new(),
//         close_after_submit_finalized: false,
//         entry_fee_usdc: fee_usdc,
//         planned_fee_usdc: fee_usdc,
//     })
// }
//
// /// Сторона + цена входа одного maker-ордера future-лесенки.
// #[derive(Debug, Clone, Copy)]
// pub(crate) struct FutureLadderEntry {
//     pub side: CurrencyUpDownOutcome,
//     /// Цена входа (она же maker-limit), напр. `0.01..=0.09`.
//     pub price: f64,
// }
//
// /// Контекст одного рынка для постановки future-лесенки.
// #[derive(Debug, Clone)]
// pub(crate) struct FutureLadderMarket {
//     pub market_id: String,
//     pub up_asset_id: String,
//     pub down_asset_id: String,
//     pub currency: String,
//     pub interval_kind: XFrameIntervalKind,
//     pub polymarket_url: String,
//     pub event_end_ms: Option<i64>,
//     pub gamma_question: Option<String>,
// }
//
// /// Выставляет **батч maker BUY-ордеров** на рынок, который **ещё не наступил**:
// /// для каждого [`FutureLadderEntry`] строит синтетический [`OpenPosition`]
// /// ([`future_open_position`], `redeem_01 = true`), кладёт его в
// /// [`crate::account::Account::future_positions`] и одним вызовом
// /// [`crate::account_submit::spawn_open_buy`] отправляет весь батч как **maker**
// /// (`delta_price = Some(0.0)` ⇒ limit ровно по `price`; размер в shares
// /// берётся из [`FUTURE_LADDER_SHARES_BY_CENT`] по ценовому уровню 1..=9c,
// /// `position_size = shares_for_level * price`).
// ///
// /// Когда рынок стартует по времени (`now >= event_end_ms − interval_ms`), позиции
// /// промоутятся в [`crate::account::Account::positions`]
// /// ([`crate::account::Account::promote_started_future_positions`]) и дальше живут
// /// как обычные. Плановую fee не добавляем: открытие идёт maker-ордером.
// ///
// /// Порядок в батче = порядок `entries` (вызывающий передаёт от дешёвых к дорогим).
// pub(crate) async fn future_open_positions(
//     account: &SharedAccount,
//     project_manager: Option<&Arc<ProjectManager>>,
//     submit_mode: crate::account_submit::SubmitMode,
//     market: &FutureLadderMarket,
//     entries: &[FutureLadderEntry],
// ) {
//     if entries.is_empty() {
//         return;
//     }
//
//     let mut requests = Vec::with_capacity(entries.len());
//     for entry in entries {
//         let asset_id = match entry.side {
//             CurrencyUpDownOutcome::Up => market.up_asset_id.as_str(),
//             CurrencyUpDownOutcome::Down => market.down_asset_id.as_str(),
//         };
//         let level_shares = future_ladder_shares_for_price(entry.price);
//         if !(level_shares > 0.0 && level_shares.is_finite()) {
//             crate::tee_eprintln!(
//                 "[future] open maker BUY market_id={}: невалидный level_shares={} для price={}",
//                 market.market_id,
//                 level_shares,
//                 entry.price,
//             );
//             continue;
//         }
//         let position_size = level_shares * entry.price;
//         let graph_dump_bin_path = market
//             .gamma_question
//             .as_deref()
//             .map(|gq| {
//                 let stem = crate::util::sanitized_filename_from_gamma_question(Some(gq));
//                 crate::xframe_dump::synthetic_xframes_dump_bin_path_for_csv_link(
//                     &market.currency,
//                     market.interval_kind,
//                     &stem,
//                     market.event_end_ms,
//                 )
//             })
//             .flatten()
//             .map(|path| path.to_string_lossy().into_owned())
//             .unwrap_or_default();
//
//         let Some(pos) = future_open_position(
//             asset_id,
//             &market.market_id,
//             &market.currency,
//             market.interval_kind,
//             entry.side,
//             &market.polymarket_url,
//             market.event_end_ms,
//             position_size,
//             entry.price,
//             false,
//             true,
//             false,
//             None,
//             None,
//             &graph_dump_bin_path,
//             market.gamma_question.as_deref(),
//         ) else {
//             crate::tee_eprintln!(
//                 "[future] open maker BUY asset_id={asset_id} market_id={}: невалидный \
//                  position_size={position_size} / price={} — пропуск",
//                 market.market_id,
//                 entry.price,
//             );
//             continue;
//         };
//
//         let lane_key = (market.currency.clone(), market.interval_kind, entry.side);
//         let pos_id = pos.id.clone();
//         let pos_arc: SharedOpenPosition = std::sync::Arc::new(tokio::sync::RwLock::new(pos));
//         {
//             let mut future = account.future_positions.write().await;
//             future
//                 .entry(lane_key)
//                 .or_default()
//                 .insert(pos_id, pos_arc.clone());
//         }
//         requests.push(crate::account_submit::OpenBuyRequest {
//             position: pos_arc,
//             price: Some(entry.price),
//             // maker limit ровно по `price` (delta = 0): spawn_open_buy → OrderRole::Maker,
//             // size в shares = position_size / price = level_shares.
//             delta_price: Some(0.0),
//         });
//     }
//
//     if requests.is_empty() {
//         return;
//     }
//     crate::account_submit::spawn_open_buy(
//         account.clone(),
//         project_manager.cloned(),
//         requests,
//         None,
//         None,
//         submit_mode,
//     );
// }
//
// /// Сайз в shares по ценовым уровням 1..=9c (индекс 0 = 1c).
// const FUTURE_LADDER_SHARES_BY_CENT: [f64; 9] =
//     [1000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0];
// /// Горизонт планирования будущих рынков (≈24ч): дальше книги ещё нет.
// const FUTURE_LADDER_HORIZON_MS: i64 = 24 * 60 * 60 * 1_000;
// /// Дешёвые уровни — ставим первыми (фаза 1: 7×2 = 14 ордеров ≤ 15).
// const FUTURE_LADDER_PHASE1_CENTS: [i64; 7] = [1, 2, 3, 4, 5, 6, 7];
// /// Остаток лесенки (фаза 2: 2×2 = 4 ордера).
// const FUTURE_LADDER_PHASE2_CENTS: [i64; 2] = [8, 9];
// /// Если gamma не дала `acceptingOrdersTimestamp` — грубый прогноз момента открытия
// /// книги (за столько до старта окна). Основной источник — сам timestamp из gamma.
// const FUTURE_LADDER_FALLBACK_OPEN_LEAD_MS: i64 = 24 * 60 * 60 * 1_000;
// /// С какого лида от `now` начинать перебор будущих окон. Книги у этих рынков
// /// открываются ~24ч заранее, поэтому стартуем с 23ч: на холодном старте не лезем во
// /// весь суточный бэклог ближних окон, а работаем у фронта свежеоткрывшихся книг
// /// (окна в полосе `[now+23ч; now+24ч]`, верх ограничен [`FUTURE_LADDER_HORIZON_MS`]).
// const FUTURE_LADDER_SCAN_LEAD_SEC: i64 = 23 * 60 * 60;
//
// /// [`FutureLadderEntry`] на обе стороны (up+down) для набора центовых уровней,
// /// от дешёвых к дорогим.
// fn future_ladder_entries(cents: &[i64]) -> Vec<FutureLadderEntry> {
//     let mut out = Vec::with_capacity(cents.len() * 2);
//     for &c in cents {
//         let price = c as f64 / 100.0;
//         out.push(FutureLadderEntry {
//             side: CurrencyUpDownOutcome::Up,
//             price,
//         });
//         out.push(FutureLadderEntry {
//             side: CurrencyUpDownOutcome::Down,
//             price,
//         });
//     }
//     out
// }
//
// fn future_ladder_shares_for_price(price: f64) -> f64 {
//     // Нормализуем к центам (ожидаем уровни 0.01..=0.09).
//     let cents = (price * 100.0).round() as i64;
//     if (1..=9).contains(&cents) {
//         FUTURE_LADDER_SHARES_BY_CENT[(cents - 1) as usize]
//     } else {
//         0.0
//     }
// }
//
// /// up/down `asset_id` из карты gamma-события (нужны обе стороны).
// fn future_ladder_resolve_sides(
//     data: &crate::util::CurrencyEventSlugData,
// ) -> Option<(String, String)> {
//     let mut up = None;
//     let mut down = None;
//     for (asset_id, code) in &data.currency_up_down_by_asset_id {
//         match code {
//             CurrencyUpDownOutcome::Up => up = Some(asset_id.clone()),
//             CurrencyUpDownOutcome::Down => down = Some(asset_id.clone()),
//         }
//     }
//     match (up, down) {
//         (Some(up), Some(down)) => Some((up, down)),
//         _ => None,
//     }
// }
//
// /// Прогнозируемый момент открытия книги: gamma `acceptingOrdersTimestamp`, fallback —
// /// `event_start_ms - FUTURE_LADDER_FALLBACK_OPEN_LEAD_MS`.
// fn future_ladder_open_at_ms(
//     data: &crate::util::CurrencyEventSlugData,
//     fallback_market_start_ms: i64,
// ) -> i64 {
//     data.accepting_orders_timestamp_ms
//         .unwrap_or(fallback_market_start_ms - FUTURE_LADDER_FALLBACK_OPEN_LEAD_MS)
// }
//
// /// Окно, с которого начинать перебор: якорь `now + 23ч`, выровненный вниз к сетке
// /// окон периода. В установившемся режиме поднимается до конца последнего уже
// /// обработанного окна (`last_event_end_ms`), чтобы не пересканировать обработанное.
// fn future_ladder_next_window_start_sec(
//     last_event_end_ms: Option<i64>,
//     now_sec: i64,
//     period_sec: i64,
// ) -> i64 {
//     let anchor_sec = now_sec + FUTURE_LADDER_SCAN_LEAD_SEC;
//     let scan_base = (anchor_sec / period_sec) * period_sec;
//     last_event_end_ms
//         .map(|end_ms| end_ms / 1_000)
//         .unwrap_or(scan_base)
//         .max(scan_base)
// }
//
// /// Луп future-лесенки: раз в секунду ведёт фронт перебора будущих 5m/15m окон
// /// выбранной монеты, начиная с `now + 23ч` ([`FUTURE_LADDER_SCAN_LEAD_SEC`]) и до
// /// горизонта `now + 24ч` ([`FUTURE_LADDER_HORIZON_MS`]). На каждое окно запускает
// /// отдельный `tokio::spawn`, который **спит до момента открытия книги**
// /// (`acceptingOrdersTimestamp`; `delay = 0`, если книга уже открыта) и в этот момент
// /// выставляет maker BUY-лесенку 1..9c на обе стороны (up/down) с сайзом из
// /// [`FUTURE_LADDER_SHARES_BY_CENT`].
// ///
// /// Внутри спавна — два батча (≤15 ордеров/батч): сначала **1..7c** (14 ордеров),
// /// затем **8..9c** (4 ордера). За тик планируется не более одного окна на период;
// /// `last_*_event_end_ms` двигает фронт вперёд, чтобы окна не пересканировались.
// ///
// /// Кэш рынков/asset_id/timestamp — через [`ProjectManager`] (`event_data_by_market`,
// /// `slug_to_market_id`, `currency_up_down_by_asset_id`, `market_asset_ids_by_market`).
// pub(crate) async fn run_redeem_01_tail_future_ladder_loop(
//     account: SharedAccount,
//     project_manager: Arc<ProjectManager>,
//     submit_mode: crate::account_submit::SubmitMode,
// ) {
//     let coin_lower = project_manager.currency.to_ascii_lowercase();
//     // Последний рынок, по которому уже выставляли лесенку, отдельно по 5m/15m.
//     // Следующий кандидат считается от `event_end_ms` последнего поставленного рынка.
//     let mut last_5m_event_end_ms: Option<i64> = None;
//     let mut last_15m_event_end_ms: Option<i64> = None;
//     let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
//     tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
//
//     loop {
//         tick.tick().await;
//         let now_ms = crate::util::current_timestamp_ms();
//         let now_sec = now_ms / 1_000;
//
//         for (period_label, period_sec, interval_kind, last_event_end_ms) in [
//             (
//                 "5m",
//                 FIVE_MIN_SEC,
//                 XFrameIntervalKind::FiveMin,
//                 &mut last_5m_event_end_ms,
//             ),
//             (
//                 "15m",
//                 FIFTEEN_MIN_SEC,
//                 XFrameIntervalKind::FifteenMin,
//                 &mut last_15m_event_end_ms,
//             ),
//         ] {
//             let interval_ms = period_sec * 1_000;
//             let mut window_start =
//                 future_ladder_next_window_start_sec(*last_event_end_ms, now_sec, period_sec);
//             loop {
//                 let window_end_ms = (window_start + period_sec) * 1_000;
//                 if window_end_ms - now_ms > FUTURE_LADDER_HORIZON_MS {
//                     break;
//                 }
//                 let slug = format!("{coin_lower}-updown-{period_label}-{window_start}");
//
//                 // Кэш PM первым; gamma дёргаем только если в кэше нет записи или в ней
//                 // не хватает market_id / обеих сторон (≤1 HTTP на окно — без лагов на
//                 // последовательных рынках).
//                 let mut data = project_manager
//                     .try_restore_currency_event_from_slug_cache(&slug)
//                     .await;
//                 let needs_fetch = match &data {
//                     Some(d) => d.market_id.is_none() || future_ladder_resolve_sides(d).is_none(),
//                     None => true,
//                 };
//                 if needs_fetch {
//                     if let Some(fresh) = project_manager
//                         .fetch_currency_event_from_gamma_and_merge(&slug, period_label)
//                         .await
//                     {
//                         data = Some(fresh);
//                     }
//                 }
//                 let Some(data) = data else {
//                     // gamma рынок ещё не создала — дальние окна тем более.
//                     break;
//                 };
//
//                 let start_ms = data.event_start_ms.unwrap_or(window_start * 1_000);
//                 let predicted_open_ms = future_ladder_open_at_ms(&data, start_ms);
//                 let delay_ms = predicted_open_ms - now_ms;
//                 if delay_ms > interval_ms {
//                     // Откроется позже, чем через ближайший интервал (<5m/<15m) →
//                     // пока рано планировать, проверим на следующих тиках.
//                     break;
//                 }
//
//                 // Резолвим market_id и up/down asset_id (нужны обе стороны).
//                 let Some(market_id) = data.market_id.clone() else {
//                     window_start += period_sec;
//                     continue;
//                 };
//                 let Some((up_asset_id, down_asset_id)) = future_ladder_resolve_sides(&data) else {
//                     window_start += period_sec;
//                     continue;
//                 };
//                 let polymarket_url = crate::util::polymarket_event_url(&slug);
//                 let market = FutureLadderMarket {
//                     market_id,
//                     up_asset_id,
//                     down_asset_id,
//                     currency: coin_lower.clone(),
//                     interval_kind,
//                     polymarket_url,
//                     event_end_ms: data.event_end_ms.or(Some(window_end_ms)),
//                     gamma_question: data.gamma_question.clone(),
//                 };
//                 *last_event_end_ms = market.event_end_ms.or(Some(window_end_ms));
//
//                 // Спим до момента открытия книги и в этот момент ставим лесенку.
//                 let sleep_ms = delay_ms.max(0) as u64;
//                 let account_for_task = account.clone();
//                 let pm_for_task = project_manager.clone();
//                 tokio::spawn(async move {
//                     if sleep_ms > 0 {
//                         tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;
//                     }
//                     let phase1 = future_ladder_entries(&FUTURE_LADDER_PHASE1_CENTS);
//                     let phase2 = future_ladder_entries(&FUTURE_LADDER_PHASE2_CENTS);
//                     // Фаза 1 (1..7c), затем фаза 2 (8..9c).
//                     future_open_positions(
//                         &account_for_task,
//                         Some(&pm_for_task),
//                         submit_mode,
//                         &market,
//                         &phase1,
//                     )
//                     .await;
//                     future_open_positions(
//                         &account_for_task,
//                         Some(&pm_for_task),
//                         submit_mode,
//                         &market,
//                         &phase2,
//                     )
//                     .await;
//                 });
//                 break; // одно окно на период за тик
//             }
//         }
//     }
// }
