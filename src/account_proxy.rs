//! Временная подстановка `HTTP_PROXY` / `HTTPS_PROXY` для сборки HTTP-клиентов в [`crate::account::Account::new`].
//!
//! В `.env` (после [`dotenvy::dotenv`] в `main`) задайте непустые **`POLY_PROXY_IP`** и **`POLY_PROXY_PORT`**.
//! Опционально **`POLY_PROXY_SCHEME`** (`http` или `https`, по умолчанию `http` — см. комментарии в `.env`),
//! **`POLY_PROXY_PASSWORD`** и **`POLY_PROXY_USERNAME`**.
//! [`PolyProxyEnvGuard::install_from_env`] выставляет переменные перед созданием клиентов;
//! после сборки клиентов вызовите [`PolyProxyEnvGuard::uninstall_from_env`] (как в [`crate::account::Account::new`]).
//! Если забыть вызвать `uninstall_from_env`, при [`Drop`] guard всё равно восстановит окружение.
//!
//! # Safety
//!
//! Мутация окружения процесса потенциально небезопасна при гонках с другими потоками или с библиотеками на C, которые
//! вызывают `getenv` без синхронизации. Используйте guard только при старте (как в [`crate::account::Account::new`]).

const ENV_HTTP_PROXY: &str = "HTTP_PROXY";
const ENV_HTTPS_PROXY: &str = "HTTPS_PROXY";

/// Кодирует сегмент userinfo для URL прокси (не зарезервированные ASCII → percent-encoding).
fn encode_userinfo_segment(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '~') {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            let enc = c.encode_utf8(&mut buf);
            for &byte in enc.as_bytes() {
                use std::fmt::Write as _;
                let _ = write!(out, "%{byte:02X}");
            }
        }
    }
    out
}

fn restore_proxy_env_vars(backups: &mut Vec<(String, Option<String>)>) {
    if backups.is_empty() {
        return;
    }
    // SAFETY: см. раздел «Safety» в документации модуля.
    unsafe {
        for (key, prev) in backups.drain(..) {
            match prev {
                Some(v) => std::env::set_var(&key, v),
                None => std::env::remove_var(&key),
            }
        }
    }
}

/// Схема подключения к хосту прокси: `http` (обычный CONNECT к HTTPS-сайтам) или `https` (TLS до самого прокси).
fn proxy_scheme_from_env() -> &'static str {
    let Ok(raw) = std::env::var("POLY_PROXY_SCHEME") else {
        return "http";
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "https" => "https",
        _ => "http",
    }
}

fn proxy_url_from_parts(
    scheme: &str,
    ip: &str,
    port: &str,
    username: Option<&str>,
    password: Option<&str>,
) -> String {
    let host = ip.trim();
    let host_in_authority = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    let port = port.trim();
    let u = username.map(str::trim).filter(|s| !s.is_empty());
    let p = password.map(str::trim).filter(|s| !s.is_empty());
    match (u, p) {
        (None, None) => format!("{scheme}://{host_in_authority}:{port}"),
        (Some(u), None) => format!(
            "{scheme}://{}@{host_in_authority}:{port}",
            encode_userinfo_segment(u)
        ),
        (None, Some(p)) => format!(
            "{scheme}://:{}@{host_in_authority}:{port}",
            encode_userinfo_segment(p)
        ),
        (Some(u), Some(p)) => format!(
            "{scheme}://{}:{}@{host_in_authority}:{port}",
            encode_userinfo_segment(u),
            encode_userinfo_segment(p)
        ),
    }
}

/// Если в окружении заданы `POLY_PROXY_IP` и `POLY_PROXY_PORT`, на время жизни этого значения
/// выставляет [`ENV_HTTP_PROXY`] и [`ENV_HTTPS_PROXY`] в URL прокси (reqwest подхватывает их по умолчанию).
pub struct PolyProxyEnvGuard {
    backups: Vec<(String, Option<String>)>,
}

impl PolyProxyEnvGuard {
    pub fn install_from_env() -> Self {
        let ip = match std::env::var("POLY_PROXY_IP") {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                return Self {
                    backups: Vec::new(),
                };
            }
        };
        let port = match std::env::var("POLY_PROXY_PORT") {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                return Self {
                    backups: Vec::new(),
                };
            }
        };

        let username = std::env::var("POLY_PROXY_USERNAME").ok();
        let password = std::env::var("POLY_PROXY_PASSWORD").ok();

        let scheme = proxy_scheme_from_env();
        let url =
            proxy_url_from_parts(scheme, &ip, &port, username.as_deref(), password.as_deref());

        let mut backups = Vec::new();
        // SAFETY: см. [`PolyProxyEnvGuard`] — типичный вызов из [`crate::account::Account::new`] при старте;
        // не вызывать параллельно с другими потоками, которые читают окружение через FFI без синхронизации.
        unsafe {
            for key in [ENV_HTTP_PROXY, ENV_HTTPS_PROXY] {
                backups.push((key.to_string(), std::env::var(key).ok()));
                std::env::set_var(key, &url);
            }
        }

        Self { backups }
    }

    /// Восстанавливает сохранённые значения `HTTP_PROXY` / `HTTPS_PROXY`. Guard потребляется без [`Drop`],
    /// чтобы не восстанавливать окружение дважды.
    pub fn uninstall_from_env(mut guard: PolyProxyEnvGuard) {
        restore_proxy_env_vars(&mut guard.backups);
        std::mem::forget(guard);
    }
}

impl Drop for PolyProxyEnvGuard {
    fn drop(&mut self) {
        restore_proxy_env_vars(&mut self.backups);
    }
}
