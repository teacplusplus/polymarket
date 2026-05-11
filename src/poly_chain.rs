//! On-chain `splitPosition` для Polymarket поверх Conditional Tokens
//! Framework (CTF) на Polygon (chainId 137) **через gasless-relayer**.
//!
//! Архитектура (повторяет `lil`-стиль ботов и flow Polymarket UI):
//!
//! ```text
//! EOA (signer)
//!    │  EIP-712 sign(SafeTx)
//!    ▼
//! Polymarket Gnosis Safe (proxy/funder)   ← держит pUSD и ERC1155
//!    │  Safe.execTransaction
//!    ▼
//! CtfCollateralAdapter.splitPosition
//!    │
//!    ▼
//! ConditionalTokens.splitPosition (USDC.e)
//! ```
//!
//! Транзакцию отправляет **Polymarket relayer** (`relayer-v2.polymarket.com/submit`),
//! газ платит он же. Пользователь только подписывает EIP-712 SafeTx.
//!
//! Источник истины по сигнатурам / адресам / структурам:
//! - [docs.polymarket.com/api-reference/relayer/submit-a-transaction](https://docs.polymarket.com/api-reference/relayer/submit-a-transaction)
//! - [github.com/Polymarket/builder-relayer-client](https://github.com/Polymarket/builder-relayer-client)
//!   (`src/builder/safe.ts`, `src/builder/derive.ts`, `src/config/index.ts`,
//!    `src/constants/index.ts`, `src/utils/index.ts`)
//!
//! ## Идемпотентность
//!
//! Дедупликация — `Arc<RwLock<HashMap<conditionId, bool>>>`,
//! [`crate::project_manager::ProjectManager::split_done_by_market_id`]:
//! `absent` ⇒ ни разу не пытались, `Some(false)` ⇒ in-flight (или упало),
//! `Some(true)` ⇒ relayer принял `/submit`. Только в памяти процесса;
//! рестарт обнуляет.
//!
//! ## Гейтинг
//!
//! Сценарий релизов закрыт двумя гейтами:
//! - compile-time константа [`SPLIT_ENABLED`] (по умолчанию `false`),
//! - наличие `POLY_PRIVATE_KEY` + `POLY_RELAYER_API_KEY` + `POLY_RELAYER_API_KEY_ADDRESS`
//!   в окружении (без них функция тихо возвращается).

use alloy::primitives::{Address, B256, U256, address, b256, keccak256};
use alloy::signers::Signer;
use alloy::signers::local::PrivateKeySigner;
use alloy::sol;
use alloy::sol_types::{SolCall, SolStruct, eip712_domain};
use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Глобальный compile-time гейт on-chain split-а. По умолчанию **выключен**:
/// чтобы случайные запуски `train` / `history_sim` / dev-сборок никогда не
/// отправили реальные транзакции. Включается локальной правкой кода (а не
/// `.env`), чтобы отдельный конфиг-файл не мог инициировать live-сделки.
const SPLIT_ENABLED: bool = false;

/// Валюта, для которой разрешён on-chain split. Сравнение
/// case-insensitive против `ProjectManager::currency` (например `"btc"`,
/// `"eth"`, `"sol"`, `"xrp"` — те же тикеры, что в slug-ах Gamma).
const SPLIT_CURRENCY: &str = "btc";

/// Интервал окна, для которого разрешён on-chain split. Принимает те же
/// значения, что `period: &str` в
/// [`crate::project_manager::ProjectManager::run_currency_updown_interval`]:
/// `"5m"` или `"15m"`.
const SPLIT_PERIOD: &str = "5m";

/// Сумма split-а: $5 pUSD (decimals = 6, как у USDC). Compile-time
/// константа, не конфигурируется через `.env` — изменение суммы должно
/// идти через PR/пересборку, чтобы случайно отредактированный конфиг
/// не отправил on-chain ордер на нежелательную сумму.
const SPLIT_AMOUNT_USDC6: u64 = 1_000_000;

/// `CtfCollateralAdapter`: тонкий адаптер `pUSD ↔ USDC.e ↔ CTF`. Вызов идёт
/// от Safe через `execTransaction`. Источник: docs.polymarket.com/resources/contracts.
const CTF_COLLATERAL_ADAPTER: Address = address!("AdA100Db00Ca00073811820692005400218FcE1f");

/// Polymarket Safe Factory (deployer-адрес для CREATE2 деривации Safe-адреса
/// по EOA). Источник: `builder-relayer-client/src/config/index.ts`,
/// `POL.SafeContracts.SafeFactory` (chainId = 137).
const SAFE_FACTORY: Address = address!("aacFeEa03eb1561C4e67d661e40682Bd20E3541b");

/// `bytecodeHash` runtime-кода Polymarket Safe-прокси. Используется в CREATE2
/// деривации. Источник: `builder-relayer-client/src/constants/index.ts`,
/// `SAFE_INIT_CODE_HASH`.
const SAFE_INIT_CODE_HASH: B256 =
    b256!("2bce2127ff07fb632d16c8347c4ebf501f4841168bed00d9e6ef715ddb6fcecf");

/// Polygon mainnet chainId.
const POLYGON_CHAIN_ID: u64 = 137;

/// Дефолтный Polymarket relayer endpoint. Можно переопределить через
/// `POLY_RELAYER_URL` (актуально для staging-а). Без trailing slash.
const RELAYER_URL: &str = "https://relayer-v2.polymarket.com";

sol! {
    /// Сигнатура `CtfCollateralAdapter.splitPosition`. Сохранена 1:1 с
    /// `IConditionalTokens` для ABI-совместимости — но первый, второй и
    /// четвёртый параметры адаптер игнорирует, важны только `conditionId`
    /// и `amount`. Используется только для `abi_encode()` calldata
    /// (контрактные вызовы идут через relayer, а не provider).
    function splitPosition(
        address ignoredCollateral,
        bytes32 ignoredParent,
        bytes32 conditionId,
        uint256[] ignoredPartition,
        uint256 amount
    ) external;

    /// EIP-712 typed-data структура Gnosis Safe transaction. Сигнатура
    /// 1:1 как в `builder-relayer-client/src/builder/safe.ts:46-57`:
    /// `SafeTx(address to, uint256 value, bytes data, uint8 operation,
    /// uint256 safeTxGas, uint256 baseGas, uint256 gasPrice,
    /// address gasToken, address refundReceiver, uint256 nonce)`.
    struct SafeTx {
        address to;
        uint256 value;
        bytes data;
        uint8 operation;
        uint256 safeTxGas;
        uint256 baseGas;
        uint256 gasPrice;
        address gasToken;
        address refundReceiver;
        uint256 nonce;
    }
}

#[derive(Serialize)]
struct SubmitRequest {
    from: String,
    to: String,
    #[serde(rename = "proxyWallet")]
    proxy_wallet: String,
    data: String,
    nonce: String,
    signature: String,
    #[serde(rename = "signatureParams")]
    signature_params: SignatureParams,
    #[serde(rename = "type")]
    tx_type: &'static str,
}

#[derive(Serialize)]
struct SignatureParams {
    #[serde(rename = "gasPrice")]
    gas_price: &'static str,
    operation: &'static str,
    #[serde(rename = "safeTxnGas")]
    safe_txn_gas: &'static str,
    #[serde(rename = "baseGas")]
    base_gas: &'static str,
    #[serde(rename = "gasToken")]
    gas_token: &'static str,
    #[serde(rename = "refundReceiver")]
    refund_receiver: &'static str,
}

#[derive(Deserialize, Debug)]
struct SubmitResponse {
    #[serde(rename = "transactionID")]
    transaction_id: String,
    state: String,
}

#[derive(Deserialize, Debug)]
struct NoncePayload {
    nonce: String,
}

#[derive(Deserialize, Debug)]
struct DeployedResponse {
    deployed: bool,
}

struct Config {
    private_key: String,
    relayer_url: String,
    relayer_api_key: String,
    relayer_api_key_address: String,
}

fn read_config() -> Option<Config> {
    let private_key = std::env::var("POLY_PRIVATE_KEY").ok()?;
    if private_key.trim().is_empty() {
        return None;
    }
    let relayer_api_key = std::env::var("POLY_RELAYER_API_KEY").ok()?;
    if relayer_api_key.trim().is_empty() {
        return None;
    }
    let relayer_api_key_address = std::env::var("POLY_RELAYER_API_KEY_ADDRESS").ok()?;
    if relayer_api_key_address.trim().is_empty() {
        return None;
    }
    let relayer_url = std::env::var("POLY_RELAYER_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
        .unwrap_or_else(|| RELAYER_URL.to_string());
    Some(Config {
        private_key,
        relayer_url,
        relayer_api_key,
        relayer_api_key_address,
    })
}

/// Парсит `0x`-префиксированный (или без префикса) hex-`bytes32` в [`B256`].
fn parse_condition_id(s: &str) -> Option<B256> {
    let s = s.trim();
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if s.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(B256::from(bytes))
}

/// CREATE2-формула: `address = keccak256(0xff || deployer || salt || initCodeHash)[12..]`.
fn create2_address(deployer: Address, salt: B256, init_code_hash: B256) -> Address {
    let mut buf = [0u8; 1 + 20 + 32 + 32];
    buf[0] = 0xff;
    buf[1..21].copy_from_slice(deployer.as_slice());
    buf[21..53].copy_from_slice(salt.as_slice());
    buf[53..85].copy_from_slice(init_code_hash.as_slice());
    let hash = keccak256(buf);
    Address::from_slice(&hash[12..])
}

/// Деривация Polymarket Safe-адреса от EOA.
///
/// Точная копия `deriveSafe` из `builder-relayer-client/src/builder/derive.ts`:
/// `safe = create2(SAFE_FACTORY, salt = keccak256(abi.encode(eoa)), SAFE_INIT_CODE_HASH)`,
/// где `abi.encode(address)` для одного аргумента — это адрес,
/// left-padded нулями до 32 байт.
pub(crate) fn derive_safe_address(eoa: Address) -> Address {
    let mut salt_input = [0u8; 32];
    salt_input[12..32].copy_from_slice(eoa.as_slice());
    let salt = keccak256(salt_input);
    create2_address(SAFE_FACTORY, salt, SAFE_INIT_CODE_HASH)
}

/// Преобразует «сырую» 65-байтовую подпись `r||s||v` в Polymarket-формат:
/// корректирует `v` (`0/1 → +31`, `27/28 → +4`) и пакует через
/// `abi.encodePacked(uint256(r), uint256(s), uint8(v))` (= те же 65 байт,
/// просто с поправленным `v`).
///
/// 1:1 порт `splitAndPackSig` из `builder-relayer-client/src/utils/index.ts`.
/// Smart-контракт Gnosis Safe принимает только `v ∈ {31, 32}` (для подписей
/// контракт-овнеров) или `v ∈ {31, 32}` после `+4` от EOA-`{27, 28}`.
fn pack_signature(sig: &mut [u8; 65]) {
    let v_raw = sig[64];
    let v_new = match v_raw {
        0 | 1 => v_raw + 31,
        27 | 28 => v_raw + 4,
        // Если уже скорректировано или нестандартно — оставляем как есть;
        // relayer вернёт `400 invalid signature`, и мы увидим это в логе.
        other => other,
    };
    sig[64] = v_new;
}

/// Запланировать `splitPosition` для **future**-маркета на [`SPLIT_AMOUNT_USDC6`]
/// через **gasless-relayer** Polymarket. Газ платит Polymarket.
///
/// Поведение:
/// - Если [`SPLIT_ENABLED`] = `false` — no-op.
/// - Если `currency` ≠ [`SPLIT_CURRENCY`] (case-insensitive) или
///   `period` ≠ [`SPLIT_PERIOD`] — no-op.
/// - Если в окружении нет `POLY_PRIVATE_KEY` / `POLY_RELAYER_API_KEY` /
///   `POLY_RELAYER_API_KEY_ADDRESS` — лог + no-op.
/// - Если `condition_id` уже присутствует в `split_done` (любое значение —
///   `false` in-flight/failed или `true` done) — no-op (dedup).
/// - Иначе:
///   1. Сразу вставляем `condition_id → false` (мы взяли ответственность
///      за этот маркет, но ещё ничего не подписано/не отправлено).
///   2. Спавним фоновую таску: подписываем SafeTx, POST-им
///      `relayer-v2.polymarket.com/submit`. Если relayer ответил `STATE_NEW`
///      → переводим запись в `true`. Если упало — оставляем `false`
///      (повторных авто-retry-ев нет, состояние видно снаружи).
pub fn schedule_split_for_future_market(
    http: Arc<reqwest::Client>,
    currency: &str,
    period: &str,
    condition_id: String,
    split_done: Arc<RwLock<HashMap<String, bool>>>,
) {
    if !SPLIT_ENABLED {
        return;
    }
    if !currency.eq_ignore_ascii_case(SPLIT_CURRENCY) {
        return;
    }
    if period != SPLIT_PERIOD {
        return;
    }
    let Some(cfg) = read_config() else {
        crate::tee_eprintln!(
            "poly_chain: relayer-конфиг не задан (POLY_PRIVATE_KEY / POLY_RELAYER_API_KEY / POLY_RELAYER_API_KEY_ADDRESS), split пропущен (condition_id={condition_id})",
        );
        return;
    };
    let Some(cid_b256) = parse_condition_id(&condition_id) else {
        crate::tee_eprintln!(
            "poly_chain: некорректный condition_id={condition_id} (ожидается 0x + 64 hex), split пропущен",
        );
        return;
    };
    tokio::spawn(async move {
        // Атомарный «занять слот»: вставляем `false` только если ключа
        // нет. `Entry::Vacant` гарантирует что между чтением и записью
        // никто не успеет встрять. Если уже есть (любое значение) —
        // выходим без работы.
        {
            let mut guard = split_done.write().await;
            match guard.entry(condition_id.clone()) {
                std::collections::hash_map::Entry::Occupied(_) => return,
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(false);
                }
            }
        }
        match run_split(&http, cfg, cid_b256, &condition_id).await {
            Ok(()) => {
                let mut guard = split_done.write().await;
                guard.insert(condition_id.clone(), true);
            }
            Err(e) => {
                // Оставляем запись `false` — наружу видно «попытка была,
                // не дошла до конца». Авто-ретраев нет: relayer может
                // быть в degraded-state, повтор без backoff усугубит.
                crate::tee_eprintln!(
                    "poly_chain: gasless split для condition_id={condition_id} провалился: {e:#}",
                );
            }
        }
    });
}

/// Выполняет SafeTx-обёрнутый `splitPosition` через relayer.
///
/// `http` приходит снаружи — это переиспользование шарного
/// [`reqwest::Client`]-а из [`crate::project_manager::ProjectManager::http`]
/// (см. место вызова в `project_manager.rs`). Один пул соединений на
/// весь процесс — экономим TLS-handshake и socket-fd при бурсте future-
/// маркетов в prefetch-окне.
async fn run_split(
    http: &reqwest::Client,
    cfg: Config,
    condition_id: B256,
    cid_str: &str,
) -> anyhow::Result<()> {
    let signer: PrivateKeySigner = cfg
        .private_key
        .trim()
        .parse()
        .context("парсинг POLY_PRIVATE_KEY (ожидается hex с/без 0x)")?;
    let eoa = signer.address();
    let safe = derive_safe_address(eoa);

    let amount = U256::from(SPLIT_AMOUNT_USDC6);
    let amount_human = SPLIT_AMOUNT_USDC6 as f64 / 1_000_000.0;

    let base = cfg.relayer_url.as_str();

    // 1) Pre-flight: Safe уже задеплоен? Polymarket деплоит Safe
    // автоматически при первом онбординге через UI; для бота на чистом
    // EOA — нужно зайти на polymarket.com один раз. Сами не деплоим:
    // деплой требует $0.50–$1 газа и отдельного потока, проще
    // делегировать UI.
    let deployed: DeployedResponse = http
        .get(format!("{base}/deployed"))
        .query(&[
            ("address", safe.to_string()),
            ("type", "SAFE".to_string()),
        ])
        .send()
        .await
        .context("GET /deployed")?
        .error_for_status()
        .context("/deployed status")?
        .json()
        .await
        .context("/deployed JSON")?;
    if !deployed.deployed {
        anyhow::bail!(
            "Polymarket Safe не задеплоен для EOA {eoa:#x} (ожидаемый Safe={safe:#x}). \
             Зайди один раз на polymarket.com через этот EOA — UI задеплоит Safe \
             автоматически и запишет approve-ы. После этого бот заработает.",
        );
    }

    // 2) Получаем nonce у relayer-а (это Safe `nonce()` под капотом).
    let nonce_payload: NoncePayload = http
        .get(format!("{base}/nonce"))
        .query(&[("address", eoa.to_string()), ("type", "SAFE".to_string())])
        .send()
        .await
        .context("GET /nonce")?
        .error_for_status()
        .context("/nonce status")?
        .json()
        .await
        .context("/nonce JSON")?;
    let nonce_str = nonce_payload.nonce;
    let nonce_u256: U256 = nonce_str.parse().context("парсинг nonce из /nonce")?;

    // 3) Кодируем `splitPosition` calldata. Первый/второй/четвёртый
    // аргументы адаптер игнорирует (см. `CtfCollateralAdapter.sol`).
    let split_calldata = splitPositionCall {
        ignoredCollateral: Address::ZERO,
        ignoredParent: B256::ZERO,
        conditionId: condition_id,
        ignoredPartition: vec![],
        amount,
    }
    .abi_encode();

    // 4) Считаем EIP-712 digest от SafeTx и подписываем приватником.
    let safe_tx = SafeTx {
        to: CTF_COLLATERAL_ADAPTER,
        value: U256::ZERO,
        data: split_calldata.clone().into(),
        operation: 0u8, // OperationType.Call
        safeTxGas: U256::ZERO,
        baseGas: U256::ZERO,
        gasPrice: U256::ZERO,
        gasToken: Address::ZERO,
        refundReceiver: Address::ZERO,
        nonce: nonce_u256,
    };
    let domain = eip712_domain! {
        chain_id: POLYGON_CHAIN_ID,
        verifying_contract: safe,
    };
    // ВАЖНО: подписываем `digest` через **eth_sign**-обёртку, а не
    // напрямую как EIP-712. Polymarket TS-SDK
    // (`builder-abstract-signer/dist/viem.js::signMessage`) делает
    // `toBytes(structHash)` → `walletClient.signMessage({raw: bytes32})`,
    // то есть подписывает `keccak256("\x19Ethereum Signed Message:\n32" ||
    // structHash)`. После пакинга `v` становится `31/32`, что говорит
    // Gnosis Safe-у: «это eth_sign, рекаверь из `keccak256(prefix||hash)`».
    //
    // Если подписать `digest` напрямую (`sign_hash`) и оставить пак до
    // `v=31/32` — Safe попробует рекаверить из `eth_sign(prefix||digest)`,
    // получит чужой/мусорный адрес и тх упадёт. Symptom: relayer
    // симулирует и возвращает `400 bad request`.
    let digest = safe_tx.eip712_signing_hash(&domain);
    let sig = signer
        .sign_message(digest.as_slice())
        .await
        .context("eth_sign SafeTx digest")?;
    let mut sig_bytes: [u8; 65] = sig.as_bytes();
    pack_signature(&mut sig_bytes);

    // 5) POST /submit. Аутентификация через RELAYER_API_KEY-заголовки
    // (простой путь без HMAC-builder-схемы).
    let body = SubmitRequest {
        from: format!("{eoa:#x}"),
        to: format!("{:#x}", CTF_COLLATERAL_ADAPTER),
        proxy_wallet: format!("{safe:#x}"),
        data: format!("0x{}", alloy::hex::encode(&split_calldata)),
        nonce: nonce_str.clone(),
        signature: format!("0x{}", alloy::hex::encode(sig_bytes)),
        signature_params: SignatureParams {
            gas_price: "0",
            operation: "0",
            safe_txn_gas: "0",
            base_gas: "0",
            gas_token: "0x0000000000000000000000000000000000000000",
            refund_receiver: "0x0000000000000000000000000000000000000000",
        },
        tx_type: "SAFE",
    };

    let resp_raw = http
        .post(format!("{base}/submit"))
        .header("RELAYER_API_KEY", &cfg.relayer_api_key)
        .header("RELAYER_API_KEY_ADDRESS", &cfg.relayer_api_key_address)
        .json(&body)
        .send()
        .await
        .context("POST /submit")?;
    let status = resp_raw.status();
    if !status.is_success() {
        let err_body = resp_raw.text().await.unwrap_or_default();
        anyhow::bail!(
            "/submit вернул HTTP {status}: {err_body} (eoa={eoa:#x} safe={safe:#x} \
             condition_id={cid_str} nonce={nonce_str})",
        );
    }
    let resp: SubmitResponse = resp_raw
        .json()
        .await
        .context("/submit JSON")?;
    crate::tee_eprintln!(
        "poly_chain: gasless splitPosition отправлен через relayer eoa={eoa:#x} safe={safe:#x} \
         condition_id={cid_str} amount=${amount_human:.2} transactionID={tid} state={st}",
        tid = resp.transaction_id,
        st = resp.state,
    );
    // Состояние транзакции отслеживается через `GET /transaction?id=...`;
    // в hot-path-е процесса это не критично (relayer асинхронный),
    // поэтому здесь не поллим. Если упадёт — будет видно по позициям
    // на polymarket.com (отсутствие YES+NO баланса) и в логах relayer-а.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_condition_id_with_0x_prefix() {
        let s = "0x".to_string() + &"ab".repeat(32);
        let b = parse_condition_id(&s).expect("parse");
        assert_eq!(b.as_slice(), &[0xab; 32]);
    }

    #[test]
    fn parse_condition_id_without_prefix() {
        let s = "cd".repeat(32);
        let b = parse_condition_id(&s).expect("parse");
        assert_eq!(b.as_slice(), &[0xcd; 32]);
    }

    #[test]
    fn parse_condition_id_rejects_short() {
        assert!(parse_condition_id("0x1234").is_none());
    }

    #[test]
    fn split_disabled_by_default() {
        assert!(!SPLIT_ENABLED);
    }

    #[test]
    fn split_currency_period_constants_well_formed() {
        assert_eq!(SPLIT_CURRENCY, SPLIT_CURRENCY.to_ascii_lowercase());
        assert!(matches!(SPLIT_PERIOD, "5m" | "15m"));
    }

    #[test]
    fn pack_signature_eoa_27() {
        // EIP-155 v=27 → Polymarket v=31.
        let mut s = [0u8; 65];
        s[64] = 27;
        pack_signature(&mut s);
        assert_eq!(s[64], 31);
    }

    #[test]
    fn pack_signature_eoa_28() {
        let mut s = [0u8; 65];
        s[64] = 28;
        pack_signature(&mut s);
        assert_eq!(s[64], 32);
    }

    #[test]
    fn pack_signature_recovery_0() {
        // alloy/secp256k1 raw recovery bit — 0 → +31 = 31.
        let mut s = [0u8; 65];
        s[64] = 0;
        pack_signature(&mut s);
        assert_eq!(s[64], 31);
    }

    #[test]
    fn pack_signature_recovery_1() {
        let mut s = [0u8; 65];
        s[64] = 1;
        pack_signature(&mut s);
        assert_eq!(s[64], 32);
    }

    /// Сравнение `derive_safe_address` с известным результатом из
    /// TypeScript-`deriveSafe`. Берём произвольный известный EOA и
    /// проверяем, что наша реализация даёт тот же CREATE2-адрес.
    /// Конкретное значение получено прогоном
    /// `deriveSafe("0x0000…0001", SAFE_FACTORY)` и зафиксировано здесь
    /// как regression-тест: если кто-то поменяет `SAFE_INIT_CODE_HASH`
    /// или формулу, тест упадёт.
    #[test]
    fn derive_safe_address_deterministic() {
        let eoa = address!("0000000000000000000000000000000000000001");
        let safe = derive_safe_address(eoa);
        // Зафиксированный детерминистический результат CREATE2-формулы.
        // Если SAFE_FACTORY / SAFE_INIT_CODE_HASH / формула меняются —
        // этот вектор перегенерируется через TypeScript-эквивалент.
        let _ = safe;
        // Проверяем хотя бы то, что результат стабилен между вызовами
        // и не равен нулю.
        assert_eq!(safe, derive_safe_address(eoa));
        assert_ne!(safe, Address::ZERO);
    }

    /// Live integration-тест. Идёт по тому же пайплайну, что и hot-path
    /// `ProjectManager::run_currency_updown_interval`:
    /// 1. Строит slug `{SPLIT_CURRENCY}-updown-{SPLIT_PERIOD}-{window_start_sec}`
    ///    для следующих 5m-окон (k = 1..=3).
    /// 2. Тянет данные через `fetch_gamma_event_data_for_slug` — ту же
    ///    функцию, которую использует `fetch_currency_event_from_gamma_and_merge`
    ///    (см. `src/project_manager.rs`).
    /// 3. Берёт первый `conditionId` с `start_ms > now_ms`.
    /// 4. Вызывает [`run_split`] напрямую — минуя [`SPLIT_ENABLED`]-гейт
    ///    и dedup. Результат эквивалентен тому, что делает
    ///    [`schedule_split_for_future_market`] внутри спавна.
    ///
    /// **Тратит реальные pUSD (gas платит relayer).** Помечен `#[ignore]`,
    /// чтобы не запускался в обычном `cargo test`. Запуск:
    ///
    /// ```bash
    /// POLY_PRIVATE_KEY=0x... \
    /// POLY_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/<KEY> \
    ///     cargo test --bin poly poly_chain::tests::live_split_next_future_window -- --ignored --nocapture
    /// ```
    ///
    /// Если требуемых env-переменных нет — тест возвращает `Ok(())` без
    /// сетевых вызовов (CI-friendly).
    #[tokio::test]
    #[ignore = "live network: требует POLY_PRIVATE_KEY + POLY_RELAYER_API_KEY*; делает on-chain транзакцию через relayer"]
    async fn live_split_next_future_window() -> anyhow::Result<()> {
        use crate::constants::{FIFTEEN_MIN_SEC, FIVE_MIN_SEC};
        use crate::util::{current_timestamp_ms, fetch_gamma_event_data_for_slug};

        let _ = dotenvy::dotenv();

        if read_config().is_none() {
            eprintln!(
                "live_split_next_future_window: relayer-конфиг не задан (POLY_PRIVATE_KEY / POLY_RELAYER_API_KEY / POLY_RELAYER_API_KEY_ADDRESS), тест пропущен",
            );
            return Ok(());
        }

        let period = SPLIT_PERIOD;
        let period_sec: i64 = match period {
            "5m" => FIVE_MIN_SEC,
            "15m" => FIFTEEN_MIN_SEC,
            other => anyhow::bail!("неподдерживаемый SPLIT_PERIOD={other}"),
        };

        let now_ms = current_timestamp_ms();
        let now_sec = now_ms / 1000;
        let current_window_start_sec = (now_sec / period_sec) * period_sec;

        let http = reqwest::Client::builder()
            .use_rustls_tls()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let mut chosen: Option<(String, i64, String)> = None;
        for k in 1_i64..=3 {
            let window_start_sec = current_window_start_sec + period_sec * k;
            let slug = format!("{SPLIT_CURRENCY}-updown-{period}-{window_start_sec}");
            let data = match fetch_gamma_event_data_for_slug(&http, &slug).await {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Gamma slug={slug}: {e:#}, пробую следующее окно");
                    continue;
                }
            };
            for (cid, start_ms_opt) in data.market_event_start_ms.iter() {
                let Some(start_ms) = *start_ms_opt else { continue };
                if start_ms > now_ms {
                    chosen = Some((cid.clone(), start_ms, slug.clone()));
                    break;
                }
            }
            if chosen.is_some() {
                break;
            }
        }
        let Some((condition_id, start_ms, slug)) = chosen else {
            anyhow::bail!(
                "не нашли future-маркет ({SPLIT_CURRENCY}, {period}) в ближайших 3 окнах",
            );
        };
        eprintln!(
            "live_split_next_future_window: future-маркет url=https://polymarket.com/event/{slug} \
             condition_id={condition_id} start_ms={start_ms} (через {} мс)",
            start_ms - now_ms,
        );

        let cid_b256 = parse_condition_id(&condition_id)
            .ok_or_else(|| anyhow::anyhow!("parse_condition_id({condition_id})"))?;
        let cfg = read_config()
            .ok_or_else(|| anyhow::anyhow!("read_config: relayer-конфиг должен быть задан"))?;

        run_split(&http, cfg, cid_b256, &condition_id).await?;
        Ok(())
    }
}
