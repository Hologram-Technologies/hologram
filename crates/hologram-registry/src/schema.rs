//! Level 4 — schemas: admission predicates that gate content but never alter its κ (§3.7, §6.10).
//!
//! A schema is a blob defining validation rules for a path scope. It is content-addressed like any
//! blob, and — crucially — **admission only**: it gates whether content is accepted, but does NOT
//! participate in the addressing of the content it validates (§3.7). A schema-validated blob receives
//! the same κ it would receive without the schema.

use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::{Mutex, OnceLock};

use hologram_space::{address_bytes, KappaStore};
use uor_distribution::ErrorCode;

use crate::http_util::{load, parse71, parse_request, read_body, write_error, write_resp};

/// The scope→schema-κ registry: `(path, scope) → schema κ`. Process-local (tracer); a durable,
/// per-registry schema store is the productionization.
fn schemas() -> &'static Mutex<HashMap<(String, String), String>> {
    static S: OnceLock<Mutex<HashMap<(String, String), String>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `PUT`/`GET /v2/{path}/schemas/{scope}` — register/retrieve a schema blob (§6.10).
pub fn handle_schema(
    stream: &mut TcpStream,
    head: &[u8],
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    let (method, path, _query, content_length, split) = parse_request(head);
    let Some(spos) = path.find("/schemas/") else {
        return write_error(stream, 404, "Not Found", ErrorCode::NameInvalid, "not a schema route");
    };
    let prefix = &path[4..spos];
    let scope = &path[spos + "/schemas/".len()..];

    match method {
        "PUT" => {
            let body = read_body(stream, head, split, content_length)?;
            // The schema is a content-addressed blob like any other.
            let k = address_bytes(&body);
            let _ = store.put("blake3", &body);
            schemas()
                .lock()
                .unwrap()
                .insert((prefix.to_string(), scope.to_string()), k.as_str().to_string());
            write_resp(
                stream,
                201,
                "Created",
                &[("Content-Length", "0"), ("X-Kappa-Label", k.as_str())],
                b"",
            )
        }
        "GET" => {
            let kappa = schemas()
                .lock()
                .unwrap()
                .get(&(prefix.to_string(), scope.to_string()))
                .cloned();
            let Some(kappa) = kappa else {
                return write_error(stream, 404, "Not Found", ErrorCode::BlobUnknown, "no schema at scope");
            };
            let Some(k) = parse71(&kappa) else {
                return write_error(stream, 404, "Not Found", ErrorCode::BlobUnknown, "bad schema κ");
            };
            match load(store, &k) {
                Some(b) => write_resp(
                    stream,
                    200,
                    "OK",
                    &[
                        ("Content-Length", &b.as_ref().len().to_string()),
                        ("X-Kappa-Label", &kappa),
                        ("Content-Type", "application/octet-stream"),
                    ],
                    b.as_ref(),
                ),
                None => {
                    write_error(stream, 404, "Not Found", ErrorCode::BlobUnknown, "schema content absent")
                }
            }
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
