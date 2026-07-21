//! Small shared HTTP helpers for the registry's Level 3+ handlers (tracer-grade — the values parsed
//! here are κ-labels, ASCII relation/scope names, and short JSON, none of which contain quotes or
//! escapes). When the registry consolidates at ratification these merge with `hologram-net`'s.

use std::io::{Read, Write};
use std::net::TcpStream;

use hologram_space::{Bytes, KappaLabel, KappaLabel71, KappaStore};
use uor_distribution::ErrorCode;

pub(crate) fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

pub(crate) fn header_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let (k, v) = line.split_once(':')?;
    if k.trim().eq_ignore_ascii_case(name) {
        Some(v)
    } else {
        None
    }
}

pub(crate) fn query_value<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find_map(|(k, v)| (k == key).then_some(v))
}

/// Extract a JSON string field `"key":"value"`.
pub(crate) fn json_field(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let after_key = &body[body.find(&pat)? + pat.len()..];
    let after_colon = &after_key[after_key.find(':')? + 1..];
    let after_open = &after_colon[after_colon.find('"')? + 1..];
    Some(after_open[..after_open.find('"')?].to_string())
}

/// Extract a JSON string array `"key":["a","b"]`.
pub(crate) fn json_string_array(body: &str, key: &str) -> Vec<String> {
    let pat = format!("\"{key}\"");
    let Some(i) = body.find(&pat) else {
        return Vec::new();
    };
    let after = &body[i + pat.len()..];
    let (Some(lb), Some(rb)) = (after.find('['), after.find(']')) else {
        return Vec::new();
    };
    let mut rest = &after[lb + 1..rb];
    let mut out = Vec::new();
    while let Some(q1) = rest.find('"') {
        let after_q1 = &rest[q1 + 1..];
        let Some(q2) = after_q1.find('"') else {
            break;
        };
        out.push(after_q1[..q2].to_string());
        rest = &after_q1[q2 + 1..];
    }
    out
}

pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

pub(crate) fn parse71(s: &str) -> Option<KappaLabel71> {
    let bytes: [u8; 71] = s.as_bytes().try_into().ok()?;
    KappaLabel::from_bytes(&bytes).ok()
}

/// Retrieve a blob by κ across σ-axes: the native path first, then the axis-polymorphic surface.
pub(crate) fn load(store: &dyn KappaStore, kappa: &KappaLabel71) -> Option<Bytes> {
    if let Ok(Some(b)) = store.get(kappa) {
        return Some(b);
    }
    if let Ok(Some(b)) = store.get_axis(kappa.as_bytes()) {
        return Some(b);
    }
    None
}

pub(crate) fn read_body(
    stream: &mut TcpStream,
    head: &[u8],
    split: usize,
    content_length: usize,
) -> std::io::Result<Vec<u8>> {
    let want = content_length.min(64 * 1024 * 1024);
    let mut body = Vec::with_capacity(want.min(64 * 1024));
    let already = &head[(split + 4).min(head.len())..];
    body.extend_from_slice(&already[..already.len().min(want)]);
    let mut chunk = [0u8; 8192];
    while body.len() < want {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        let take = (want - body.len()).min(n);
        body.extend_from_slice(&chunk[..take]);
    }
    Ok(body)
}

pub(crate) fn write_resp(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {status} {reason}\r\n");
    for (k, v) in headers {
        head.push_str(k);
        head.push_str(": ");
        head.push_str(v);
        head.push_str("\r\n");
    }
    head.push_str("Connection: close\r\n\r\n");
    stream.write_all(head.as_bytes())?;
    if !body.is_empty() {
        stream.write_all(body)?;
    }
    stream.flush()
}

pub(crate) fn write_error(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    code: ErrorCode,
    message: &str,
) -> std::io::Result<()> {
    let body = format!(
        r#"{{"errors":[{{"code":"{}","message":"{message}"}}]}}"#,
        code.as_str()
    );
    write_resp(
        stream,
        status,
        reason,
        &[
            ("Content-Type", "application/json"),
            ("Content-Length", &body.len().to_string()),
        ],
        body.as_bytes(),
    )
}

/// Parse the request line + `Content-Length` from a request head. Returns
/// `(method, path, query, content_length, header_end_offset)`.
pub(crate) fn parse_request(head: &[u8]) -> (&str, &str, &str, usize, usize) {
    let split = find(head, b"\r\n\r\n").unwrap_or(head.len());
    let head_str = core::str::from_utf8(&head[..split]).unwrap_or("");
    let mut lines = head_str.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let raw_path = parts.next().unwrap_or("");
    let path = raw_path.split('?').next().unwrap_or(raw_path);
    let query = raw_path.split_once('?').map(|(_, q)| q).unwrap_or("");
    let mut content_length = 0usize;
    for line in lines {
        if let Some(v) = header_value(line, "content-length") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    (method, path, query, content_length, split)
}
