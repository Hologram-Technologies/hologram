//! Level 5 — admission filters (spec §6.12, §10). A filter is a content-addressed blob linked to a
//! path scope; it receives content bytes and returns accept/reject. Multiple filters matching a scope
//! MUST all accept; any rejection or execution failure fails **closed** (§10.2).
//!
//! The filter runtime is not prescribed. This tracer models a filter as a `deny:<bytes>` rule — a
//! blob whose content rejects any write containing `<bytes>` — which exercises the all-accept /
//! fail-closed admission semantics without a WASM runtime. Registering/listing/removing filters is
//! the full §6.12 surface; enforcing them on the delegated L1 blob-write path is deferred until the
//! registry owns that path (a consolidation step) — [`admit`] is the enforcement primitive.

use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::{Mutex, OnceLock};

use hologram_space::{address_bytes, KappaStore};
use uor_distribution::ErrorCode;

use crate::http_util::{parse_request, read_body, write_error, write_resp};

struct Filter {
    scope: String,
    kappa: String,
    rule: Vec<u8>,
}

fn filters() -> &'static Mutex<HashMap<String, Vec<Filter>>> {
    static F: OnceLock<Mutex<HashMap<String, Vec<Filter>>>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Evaluate every filter registered under `path` against `content`. All must accept (§10.2); any
/// rejection fails closed. `Err(reason)` names the rejecting filter. This is the enforcement
/// primitive a registry that owns the content-write path calls before admitting a blob.
pub fn admit(path: &str, content: &[u8]) -> Result<(), String> {
    let map = filters().lock().unwrap();
    let Some(list) = map.get(path) else {
        return Ok(());
    };
    for f in list {
        if let Some(needle) = f.rule.strip_prefix(b"deny:") {
            if !needle.is_empty() && contains(content, needle) {
                return Err(format!("filter {} rejected content", f.kappa));
            }
        }
    }
    Ok(())
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    needle.len() <= hay.len() && hay.windows(needle.len()).any(|w| w == needle)
}

/// `PUT /v2/{path}/filters/{scope}` · `GET /v2/{path}/filters/` · `DELETE /v2/{path}/filters/{κ}`
/// (§6.12).
pub fn handle_filters(
    stream: &mut TcpStream,
    head: &[u8],
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    let (method, path, _query, content_length, split) = parse_request(head);
    let Some(fpos) = path.find("/filters") else {
        return write_error(stream, 404, "Not Found", ErrorCode::NameInvalid, "not a filter route");
    };
    let prefix = &path[4..fpos];
    let trailing = path[fpos + "/filters".len()..].trim_start_matches('/');

    match method {
        "PUT" => {
            let scope = trailing;
            let body = read_body(stream, head, split, content_length)?;
            let k = address_bytes(&body);
            let _ = store.put("blake3", &body);
            filters()
                .lock()
                .unwrap()
                .entry(prefix.to_string())
                .or_default()
                .push(Filter {
                    scope: scope.to_string(),
                    kappa: k.as_str().to_string(),
                    rule: body,
                });
            write_resp(
                stream,
                201,
                "Created",
                &[("Content-Length", "0"), ("X-Kappa-Label", k.as_str())],
                b"",
            )
        }
        "GET" => {
            let map = filters().lock().unwrap();
            let empty = Vec::new();
            let list = map.get(prefix).unwrap_or(&empty);
            let items = list
                .iter()
                .map(|f| format!(r#"{{"scope":"{}","kappa":"{}"}}"#, f.scope, f.kappa))
                .collect::<Vec<_>>()
                .join(",");
            let body = format!(r#"{{"filters":[{items}]}}"#);
            write_resp(
                stream,
                200,
                "OK",
                &[
                    ("Content-Type", "application/json"),
                    ("Content-Length", &body.len().to_string()),
                ],
                body.as_bytes(),
            )
        }
        // The filter blob is unlinked (removed from the scope), not deleted — it remains for audit.
        "DELETE" => {
            let mut map = filters().lock().unwrap();
            let list = map.entry(prefix.to_string()).or_default();
            let before = list.len();
            list.retain(|f| f.kappa != trailing);
            let (status, reason) = if list.len() < before {
                (202, "Accepted")
            } else {
                (404, "Not Found")
            };
            write_resp(stream, status, reason, &[("Content-Length", "0")], b"")
        }
        _ => write_resp(
            stream,
            405,
            "Method Not Allowed",
            &[("Content-Length", "0")],
            b"",
        ),
    }
}
