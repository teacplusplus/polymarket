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
use alloy::sol;
use alloy::sol_types::SolValue;
use anyhow::Context as _;
use serde::Serialize;
use std::time::Duration;

/// Polymarket Safe Factory (deployer-адрес для CREATE2 деривации Safe-адреса
/// по EOA). Источник: `builder-relayer-client/src/config/index.ts`,
/// `POL.SafeContracts.SafeFactory` (chainId = 137).
const SAFE_FACTORY: Address = address!("aacFeEa03eb1561C4e67d661e40682Bd20E3541b");

/// `bytecodeHash` runtime-кода Polymarket Safe-прокси. Используется в CREATE2
/// деривации. Источник: `builder-relayer-client/src/constants/index.ts`,
/// `SAFE_INIT_CODE_HASH`.
const SAFE_INIT_CODE_HASH: B256 =
    b256!("2bce2127ff07fb632d16c8347c4ebf501f4841168bed00d9e6ef715ddb6fcecf");

/// Deposit wallet factory / implementation (Polygon mainnet). Источник:
/// `builder-relayer-client/src/config/index.ts`, `POL.DepositWalletContracts`.
const DEPOSIT_WALLET_FACTORY: Address = address!("00000000000Fb5C9ADea0298D729A0CB3823Cc07");
const DEPOSIT_WALLET_IMPLEMENTATION: Address = address!("58CA52ebe0DadfdF531Cde7062e76746de4Db1eB");
const ERC1967_CONST1: B256 =
    b256!("cc3735a920a3ca505d382bbc545af43d6000803e6038573d6000fd5b3d6000f3");
const ERC1967_CONST2: B256 =
    b256!("5155f3363d3d373d3d363d7f360894a13ba1a3210667c828492db98dca3e2076");

/// Polymarket Relayer API (`POST /submit`). Сервер из официальной OpenAPI:
/// <https://docs.polymarket.com/api-reference/relayer/submit-a-transaction>
/// Без trailing slash.
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

struct Config {
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
    Some(Config {
        relayer_url: RELAYER_URL.to_string(),
        relayer_api_key,
        relayer_api_key_address,
    })
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

/// CREATE2-адрес deposit wallet от EOA. Порт `deriveDepositWallet` из
/// `builder-relayer-client/src/builder/derive.ts`.
pub(crate) fn derive_deposit_wallet_address(eoa: Address) -> Address {
    let mut wallet_id = [0u8; 32];
    wallet_id[12..32].copy_from_slice(eoa.as_slice());
    let wallet_id_b256 = B256::from(wallet_id);
    let args = (DEPOSIT_WALLET_FACTORY, wallet_id_b256).abi_encode();
    let salt = keccak256(&args);
    let init_code_hash = deposit_wallet_init_code_hash(DEPOSIT_WALLET_IMPLEMENTATION, &args);
    create2_address(DEPOSIT_WALLET_FACTORY, salt, init_code_hash)
}

fn deposit_wallet_init_code_hash(implementation: Address, args: &[u8]) -> B256 {
    let prefix = U256::from(0x6100_3d3d_8160_233d_3973_u128);
    let combined: U256 = prefix + (U256::from(args.len()) << 56);
    let combined_be = combined.to_be_bytes::<32>();
    let mut buf = Vec::with_capacity(10 + 20 + 2 + 32 + 32 + args.len());
    buf.extend_from_slice(&combined_be[22..32]);
    buf.extend_from_slice(implementation.as_slice());
    buf.extend_from_slice(&[0x60, 0x09]);
    buf.extend_from_slice(ERC1967_CONST2.as_slice());
    buf.extend_from_slice(ERC1967_CONST1.as_slice());
    buf.extend_from_slice(args);
    keccak256(buf)
}

#[derive(Serialize)]
struct DepositWalletCreateSubmit {
    #[serde(rename = "type")]
    kind: &'static str,
    from: String,
    to: String,
}

/// Пытается задеплоить deposit wallet через relayer `WALLET-CREATE`.
/// Возвращает детерминированный CREATE2-адрес кошелька.
pub(crate) async fn ensure_deposit_wallet_deployed(
    http: &reqwest::Client,
    eoa: Address,
) -> anyhow::Result<Address> {
    let wallet = derive_deposit_wallet_address(eoa);
    let Some(cfg) = read_config() else {
        anyhow::bail!(
            "ensure_deposit_wallet_deployed: relayer-конфиг не задан \
             (POLY_PRIVATE_KEY / POLY_RELAYER_API_KEY / POLY_RELAYER_API_KEY_ADDRESS)"
        );
    };
    let body = DepositWalletCreateSubmit {
        kind: "WALLET-CREATE",
        from: format!("{eoa:#x}"),
        to: format!("{DEPOSIT_WALLET_FACTORY:#x}"),
    };
    let response = http
        .post(format!("{}/submit", cfg.relayer_url))
        .header("RELAYER_API_KEY", &cfg.relayer_api_key)
        .header("RELAYER_API_KEY_ADDRESS", &cfg.relayer_api_key_address)
        .json(&body)
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .context("POST /submit WALLET-CREATE")?;
    if response.status().is_success() {
        return Ok(wallet);
    }
    let status = response.status();
    let err_body = response.text().await.unwrap_or_default();
    if err_body.to_ascii_lowercase().contains("deployed")
        || err_body.to_ascii_lowercase().contains("exists")
    {
        return Ok(wallet);
    }
    anyhow::bail!(
        "WALLET-CREATE вернул HTTP {status}: {err_body} (eoa={eoa:#x}, deposit_wallet={wallet:#x})"
    );
}
