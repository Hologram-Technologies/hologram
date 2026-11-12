//! A minimal HTTP/1.1 client over `std::net` for talking to HTTP-CAS sources: persistent
//! (keep-alive) connections, `Content-Length` bodies with a caller-set ceiling, no TLS. Every body
//! it returns is untrusted; callers verify it against a κ or a piece table before use.

extern crate std;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Cap on response-header bytes from a source.
const MAX_HEADER_BYTES: usize = 64 * 1024;

/// A parsed response. Header names are lower-cased.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    /// First value of header `name` (lower-case).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Why a request failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientError {
    /// Connect, read or write failed, or timed out.
    Io(std::io::ErrorKind),
    /// The response is not well-formed HTTP/1.1, uses chunked encoding, or exceeds the ceiling.
    Protocol(&'static str),
}

impl From<std::io::Error> for ClientError {
    fn from(e: std::io::Error) -> Self {
        ClientError::Io(e.kind())
    }
}

/// Normalize a source address: `http://host:port/` and `host:port` both give `host:port`.
pub fn source_authority(source: &str) -> &str {
    let s = source.strip_prefix("http://").unwrap_or(source);
    s.split('/').next().unwrap_or(s)
}

/// One persistent connection to one source, reopened transparently when the server closes it.
pub struct HttpConn {
    authority: String,
    timeout: Duration,
    conn: Option<BufReader<TcpStream>>,
}

impl HttpConn {
    /// A connection (opened lazily) to `source` (`host:port` or `http://host:port`).
    pub fn new(source: &str, timeout: Duration) -> Self {
        Self {
            authority: source_authority(source).to_string(),
            timeout,
            conn: None,
        }
    }

    fn connect(&self) -> Result<BufReader<TcpStream>, ClientError> {
        let s = TcpStream::connect(&self.authority)?;
        s.set_read_timeout(Some(self.timeout))?;
        s.set_write_timeout(Some(self.timeout))?;
        s.set_nodelay(true)?;
        Ok(BufReader::with_capacity(256 * 1024, s))
    }

    /// Send `method path` with `headers`; read a response whose body is at most `max_body` bytes.
    /// A request on a reused connection that the server had already closed is retried once on a
    /// fresh one.
    pub fn request(
        &mut self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        max_body: u64,
    ) -> Result<Response, ClientError> {
        let reused = self.conn.is_some();
        match self.try_request(method, path, headers, max_body) {
            Err(ClientError::Io(_)) if reused => {
                self.conn = None;
                self.try_request(method, path, headers, max_body)
            }
            r => r,
        }
    }

    fn try_request(
        &mut self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        max_body: u64,
    ) -> Result<Response, ClientError> {
        if self.conn.is_none() {
            self.conn = Some(self.connect()?);
        }
        let result = self.exchange(method, path, headers, max_body);
        match &result {
            Ok(r)
                if !r
                    .header("connection")
                    .is_some_and(|v| v.eq_ignore_ascii_case("close")) => {}
            _ => self.conn = None,
        }
        result
    }

    fn exchange(
        &mut self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        max_body: u64,
    ) -> Result<Response, ClientError> {
        let conn = self.conn.as_mut().expect("connected");
        let mut req = std::format!("{method} {path} HTTP/1.1\r\nHost: {}\r\n", self.authority);
        for (k, v) in headers {
            req.push_str(k);
            req.push_str(": ");
            req.push_str(v);
            req.push_str("\r\n");
        }
        req.push_str("\r\n");
        conn.get_mut().write_all(req.as_bytes())?;

        let mut line = String::new();
        let mut head_bytes = 0usize;
        if conn.read_line(&mut line)? == 0 {
            return Err(ClientError::Io(std::io::ErrorKind::UnexpectedEof));
        }
        head_bytes += line.len();
        let mut parts = line.split_whitespace();
        if !parts.next().is_some_and(|v| v.starts_with("HTTP/1.")) {
            return Err(ClientError::Protocol("status line"));
        }
        let status: u16 = parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(ClientError::Protocol("status code"))?;
        let mut headers = Vec::new();
        loop {
            line.clear();
            if conn.read_line(&mut line)? == 0 {
                return Err(ClientError::Io(std::io::ErrorKind::UnexpectedEof));
            }
            head_bytes += line.len();
            if head_bytes > MAX_HEADER_BYTES {
                return Err(ClientError::Protocol("headers too large"));
            }
            let l = line.trim_end_matches(['\r', '\n']);
            if l.is_empty() {
                break;
            }
            let (k, v) = l
                .split_once(':')
                .ok_or(ClientError::Protocol("header line"))?;
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
        let mut resp = Response {
            status,
            headers,
            body: Vec::new(),
        };
        if resp.header("transfer-encoding").is_some() {
            return Err(ClientError::Protocol("chunked bodies are not supported"));
        }
        let no_body = method == "HEAD" || status == 204 || status == 304 || status < 200;
        if !no_body {
            let len: u64 = match resp.header("content-length") {
                Some(v) => v
                    .parse()
                    .map_err(|_| ClientError::Protocol("content-length"))?,
                None => return Err(ClientError::Protocol("missing content-length")),
            };
            if len > max_body {
                return Err(ClientError::Protocol("body exceeds the expected size"));
            }
            resp.body = alloc::vec![0u8; len as usize];
            // In bounded slices: one huge socket read can fail outright (WSAENOBUFS on Windows).
            for part in resp.body.chunks_mut(1 << 20) {
                conn.read_exact(part)?;
            }
        }
        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_strips_scheme_and_path() {
        assert_eq!(source_authority("127.0.0.1:80"), "127.0.0.1:80");
        assert_eq!(source_authority("http://h:1/"), "h:1");
        assert_eq!(source_authority("http://h:1/x/y"), "h:1");
    }
}
