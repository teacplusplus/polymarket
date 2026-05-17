//! WebSocket-подключение с опциональным HTTP-прокси (**CONNECT**).
//!
//! Переменные как у [`crate::account_proxy`]: `POLY_PROXY_IP`, `POLY_PROXY_PORT`, опционально логин/пароль.
//! **`POLY_PROXY_SCHEME=https`** — сначала TLS к прокси (SNI = хост прокси), затем `CONNECT`, затем TLS к целевому `wss`-хосту.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use rustls::RootCertStore;
use rustls::pki_types::ServerName;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, client_async_tls_with_config,
    tungstenite::{
        Error as WsError, client::IntoClientRequest, error::UrlError, handshake::client::Response,
        http::{Request, Uri},
    },
};

/// Транспорт до целевого TLS (прямой TCP или TCP + TLS к прокси). Нужен единый тип для прямого коннекта и прокси.
pub enum MaybeProxyTransport {
    Plain(TcpStream),
    TlsToProxy(tokio_rustls::client::TlsStream<TcpStream>),
}

impl AsyncRead for MaybeProxyTransport {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            MaybeProxyTransport::Plain(t) => Pin::new(t).poll_read(cx, buf),
            MaybeProxyTransport::TlsToProxy(t) => Pin::new(t).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for MaybeProxyTransport {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        match self.get_mut() {
            MaybeProxyTransport::Plain(t) => Pin::new(t).poll_write(cx, buf),
            MaybeProxyTransport::TlsToProxy(t) => Pin::new(t).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), std::io::Error>> {
        match self.get_mut() {
            MaybeProxyTransport::Plain(t) => Pin::new(t).poll_flush(cx),
            MaybeProxyTransport::TlsToProxy(t) => Pin::new(t).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), std::io::Error>> {
        match self.get_mut() {
            MaybeProxyTransport::Plain(t) => Pin::new(t).poll_shutdown(cx),
            MaybeProxyTransport::TlsToProxy(t) => Pin::new(t).poll_shutdown(cx),
        }
    }
}

pub type PolyWsStream = WebSocketStream<MaybeTlsStream<MaybeProxyTransport>>;

/// Конфигурация HTTP(S)-прокси для WebSocket (`CONNECT`).
#[derive(Debug, Clone)]
pub struct WsProxyConfig {
    pub host: String,
    pub port: u16,
    /// TLS до прокси до отправки `CONNECT` (как `POLY_PROXY_SCHEME=https` для REST).
    pub tls_to_proxy: bool,
    pub username: Option<String>,
    pub password: Option<String>,
}

/// Читает [`WsProxyConfig`] из окружения (`.env` уже должен быть загружен в `main`).
pub fn ws_proxy_from_env() -> Option<WsProxyConfig> {
    let host = std::env::var("POLY_PROXY_IP").ok()?.trim().to_string();
    if host.is_empty() {
        return None;
    }
    let port_raw = std::env::var("POLY_PROXY_PORT").ok()?.trim().to_string();
    if port_raw.is_empty() {
        return None;
    }
    let port: u16 = port_raw.parse().ok()?;

    let tls_to_proxy = matches!(
        std::env::var("POLY_PROXY_SCHEME")
            .ok()
            .as_deref()
            .map(|s| s.trim().eq_ignore_ascii_case("https")),
        Some(true)
    );

    let username = std::env::var("POLY_PROXY_USERNAME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let password = std::env::var("POLY_PROXY_PASSWORD")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    Some(WsProxyConfig {
        host,
        port,
        tls_to_proxy,
        username,
        password,
    })
}

fn connect_authority(target_host: &str, target_port: u16) -> String {
    let host = target_host.trim();
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{target_port}")
    } else {
        format!("{host}:{target_port}")
    }
}

fn proxy_server_name(host: &str) -> Result<ServerName<'static>, WsError> {
    let h = host.trim();
    let inner = h
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(h);
    if let Ok(ip) = inner.parse::<std::net::IpAddr>() {
        return Ok(ServerName::IpAddress(ip.into()));
    }
    ServerName::try_from(inner.to_string()).map_err(|_| {
        WsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "proxy host: invalid DNS name for TLS SNI",
        ))
    })
}

fn proxy_basic_auth_header(username: Option<&str>, password: Option<&str>) -> Option<String> {
    match (username, password) {
        (Some(u), Some(p)) => {
            let cred = format!("{u}:{p}");
            Some(format!(
                "Proxy-Authorization: Basic {}\r\n",
                STANDARD.encode(cred)
            ))
        }
        (Some(u), None) => Some(format!(
            "Proxy-Authorization: Basic {}\r\n",
            STANDARD.encode(format!("{u}:"))
        )),
        (None, Some(p)) => Some(format!(
            "Proxy-Authorization: Basic {}\r\n",
            STANDARD.encode(format!(":{p}"))
        )),
        (None, None) => None,
    }
}

async fn tls_handshake_to_proxy(
    tcp: TcpStream,
    proxy_host: &str,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>, WsError> {
    let mut root_store = RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = Arc::new(
        rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth(),
    );
    let connector = TlsConnector::from(config);
    let server_name = proxy_server_name(proxy_host)?;
    connector
        .connect(server_name, tcp)
        .await
        .map_err(WsError::Io)
}

async fn tcp_to_ws_target(uri: &Uri) -> Result<TcpStream, WsError> {
    let host = uri.host().ok_or(WsError::Url(UrlError::NoHostName))?;
    let port = uri
        .port_u16()
        .or_else(|| match uri.scheme_str() {
            Some("wss") => Some(443),
            Some("ws") => Some(80),
            _ => None,
        })
        .ok_or(WsError::Url(UrlError::UnsupportedUrlScheme))?;
    let spec = connect_authority(host, port);
    TcpStream::connect(spec).await.map_err(WsError::Io)
}

/// Читает ответ прокси до конца заголовков; проверяет статус 200.
async fn read_connect_established<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
) -> Result<(), WsError> {
    use tokio::io::AsyncReadExt;

    let mut buf = Vec::<u8>::new();
    let mut chunk = [0u8; 512];
    loop {
        let n = stream.read(&mut chunk).await.map_err(WsError::Io)?;
        if n == 0 {
            return Err(WsError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "proxy closed before CONNECT response",
            )));
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > 64 * 1024 {
            return Err(WsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "CONNECT response headers too large",
            )));
        }
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let header_text = std::str::from_utf8(&buf).map_err(|e| {
        WsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("CONNECT response not UTF-8: {e}"),
        ))
    })?;
    let status_line = header_text.lines().next().ok_or_else(|| {
        WsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty CONNECT response",
        ))
    })?;
    let ok = status_line.starts_with("HTTP/") && status_line.contains(" 200 ");
    if !ok {
        return Err(WsError::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            format!("CONNECT failed: {status_line}"),
        )));
    }
    Ok(())
}

async fn connect_via_http_proxy(
    request: Request<()>,
    proxy: &WsProxyConfig,
) -> Result<(PolyWsStream, Response), WsError> {
    let uri = request.uri();
    let target_host = uri.host().ok_or(WsError::Url(UrlError::NoHostName))?;
    let target_port = uri
        .port_u16()
        .or_else(|| match uri.scheme_str() {
            Some("wss") => Some(443),
            Some("ws") => Some(80),
            _ => None,
        })
        .ok_or(WsError::Url(UrlError::UnsupportedUrlScheme))?;

    let authority = connect_authority(target_host, target_port);
    let proxy_addr = format!("{}:{}", proxy.host.trim(), proxy.port);
    let tcp = TcpStream::connect(&proxy_addr).await.map_err(WsError::Io)?;

    let mut transport = if proxy.tls_to_proxy {
        MaybeProxyTransport::TlsToProxy(tls_handshake_to_proxy(tcp, proxy.host.trim()).await?)
    } else {
        MaybeProxyTransport::Plain(tcp)
    };

    let auth_block = proxy_basic_auth_header(proxy.username.as_deref(), proxy.password.as_deref())
        .unwrap_or_default();

    let connect_req = format!(
        "CONNECT {authority} HTTP/1.1\r\n\
         Host: {authority}\r\n\
         {auth_block}\
         User-Agent: poly-ws-connect\r\n\
         \r\n",
    );
    transport
        .write_all(connect_req.as_bytes())
        .await
        .map_err(WsError::Io)?;
    read_connect_established(&mut transport).await?;

    client_async_tls_with_config(request, transport, None, None).await
}

/// Прямое подключение (как [`tokio_tungstenite::connect_async`]), с тем же типом потока, что и через прокси.
async fn connect_direct(ws_url: &str) -> Result<(PolyWsStream, Response), WsError> {
    let request = ws_url.into_client_request()?;
    let tcp = tcp_to_ws_target(request.uri()).await?;
    client_async_tls_with_config(request, MaybeProxyTransport::Plain(tcp), None, None).await
}

/// Без прокси — прямой TCP+TLS+WS; с прокси — TCP [+ TLS к прокси] + `CONNECT` + TLS к цели + WS.
pub async fn connect_async_maybe_proxy(
    ws_url: &str,
    proxy: Option<&WsProxyConfig>,
) -> Result<(PolyWsStream, Response), WsError> {
    match proxy {
        None => connect_direct(ws_url).await,
        Some(p) => {
            let request = ws_url.into_client_request()?;
            connect_via_http_proxy(request, p).await
        }
    }
}
