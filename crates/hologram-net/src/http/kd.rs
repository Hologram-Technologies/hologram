//! κ-Distribution `/v2/` HTTP binding — **Level 1 blob surface** (spec `003` §6.2–6.4).
//!
//! This is the κ-Distribution protocol's Level 1: a spec-conformant content-addressed blob
//! registry — version check · `HEAD`/`GET`/`PUT`/`DELETE` blobs · multi-label push · chunked
//! upload sessions with recovery · mount · health — layered on the *same* `std::net` server and
//! the *same* content-addressing primitives as the `/cas/` gateway ([`super::live`]).
//!
//! **Identity stays κ-only (Law 2 / SPINE-1).** `{path}` is an opaque routing/authorization prefix
//! that never participates in addressing — a blob's identity is its κ-label and nothing else. This
//! is the same doctrine the UOR platform states for its transport realizations ("identity still
//! comes from UOR-ADDR, not the CID"); the mutable/human-name resolution overlay (tags,
//! `org:path@version`) is a *later* level and lives outside this addressing path.
//!
//! **Upload sessions are ephemeral server state.** The in-flight chunk buffer + byte cursor live in
//! a process-local registry ([`sessions`]); only the *assembled, verified* blob is durably stored,
//! through the existing [`KappaStore`]. The durable object model (edges/pins/witnesses/tags) and a
//! persistent `UploadSessions`/`StreamingStore` contract are later phases (crate `uor-distribution`
//! defines the abstract op traits; hologram implements them).
//!
//! **Non-blocking until ratified.** The whole module is behind the non-default `kd` feature (which
//! implies `live`). The always-green blocking CI jobs never enable `kd`, so this surface is
//! exercised only by the dedicated `kd-conformance` job (plan Phase 0/8, KD conformance class).

extern crate std;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Mutex, OnceLock};

use hologram_space::{verify_kappa, verify_kappa_axis, Bytes, KappaLabel, KappaLabel71, KappaStore};

/// The κ-Distribution protocol version this binding implements (spec §6.2 `GET /v2/`).
pub const KAPPA_DISTRIBUTION_VERSION: &str = "2.0.0";

/// A registry-supplied admission predicate `(path, content) -> accept | reject(reason)`. The kd
/// binding runs it on blob `PUT` before storing (spec §5.1, §10) — this is the seam through which a
/// registry (which owns the filter store) enforces admission filters on the delegated write path.
pub type Admission<'a> = &'a dyn Fn(&str, &[u8]) -> Result<(), String>;

/// DoS guard for a single `PUT`/`PATCH` body (spec §6.4 chunked uploads accumulate many of these).
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

// ─────────────────────────── ephemeral upload-session registry ───────────────────────────

/// An in-flight chunked upload (spec §6.4): the assembled bytes so far and the `{path}` the
/// session was opened under (for the completion `Location`). Discarded on complete/cancel/timeout.
struct Session {
    data: Vec<u8>,
    path: String,
}

/// The process-local session registry. Sessions are keyed by an opaque, unique id
/// ([`new_session_id`]); the URL is opaque to the client (spec §6.4).
fn sessions() -> &'static Mutex<HashMap<String, Session>> {
    static S: OnceLock<Mutex<HashMap<String, Session>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A unique opaque session id. A monotonic counter — deterministic, no randomness (the URL is
/// opaque, so the value is immaterial as long as it is unique within the process).
fn new_session_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    std::format!("sess-{}", N.fetch_add(1, Ordering::Relaxed))
}

// ─────────────────────────── tags: the resolution overlay (Level 2) ───────────────────────────
//
// Tags are mutable `(path, name) → κ` pointers — the ONLY mutable state (spec §3.3). They are the
// UOR-RESOLUTION overlay and, per Law 2 / SPINE-1, are NOT identity: resolving a tag yields a
// κ-label, and that κ (never the tag name) is the object's identity. The registry keeps tags OUTSIDE
// the content-addressed `KappaStore` — this process-local map is a single-node tracer for the
// semantics (content-before-tag, CAS, ordered listing); a durable, consensus-backed `TagStore` is
// the productionization (a later phase, in `hologram-registry` over a persistent backend). Tests use
// a unique `{path}` each to stay isolated in this shared map.

/// The mutable tag registry: `(path, name) → κ-label string`.
fn tags() -> &'static Mutex<HashMap<(String, String), String>> {
    static T: OnceLock<Mutex<HashMap<(String, String), String>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The tags bound under `path`, as `(name, κ-label)` pairs. A registry's GC uses these as reachability
/// roots (spec §9.1: every tagged κ-label is a root), reaching across the L1/L2 delegation boundary.
pub fn tags_for(path: &str) -> Vec<(String, String)> {
    tags()
        .lock()
        .unwrap()
        .iter()
        .filter(|((p, _), _)| p == path)
        .map(|((_, name), k)| (name.clone(), k.clone()))
        .collect()
}

// ─────────────────────────────────────── routing ───────────────────────────────────────────

/// The `/v2/` routes this Level-1 binding recognizes.
enum Route<'a> {
    /// `GET /v2/` — protocol version check (§6.2).
    Version,
    /// `GET /v2/_health/{live,ready,startup}` (§6.14).
    Health,
    /// `POST /v2/{path}/blobs/uploads/` — start an upload session / mount (§6.4, §6.7). Carries the
    /// opaque `{path}` prefix (for the completion `Location`).
    UploadStart(&'a str),
    /// `PATCH`/`GET`/`PUT`/`DELETE /v2/_uploads/{id}` — session chunk / recovery / complete / cancel.
    UploadSession(&'a str),
    /// `GET`/`PUT`/`DELETE /v2/{path}/manifests/{version}` — resolve/bind/delete a tag (or resolve a
    /// direct κ-label when `{version}` contains a colon). Carries `({path}, {version})` (§6.5).
    Manifest(&'a str, &'a str),
    /// `GET /v2/{path}/tags/list` — list the tags at a path (§6.5). Carries `{path}`.
    TagList(&'a str),
    /// `…/blobs/{κ}` — the κ-label is the trailing segment; `{path}` before it is opaque (§6.3).
    Blob(&'a str),
    /// Anything else under `/v2/`.
    Unknown,
}

fn route(path: &str) -> Route<'_> {
    if path == "/v2" || path == "/v2/" {
        return Route::Version;
    }
    if let Some(kind) = path.strip_prefix("/v2/_health/") {
        return match kind {
            "live" | "ready" | "startup" => Route::Health,
            _ => Route::Unknown,
        };
    }
    if let Some(id) = path.strip_prefix("/v2/_uploads/") {
        if !id.is_empty() {
            return Route::UploadSession(id);
        }
    }
    if !path.starts_with("/v2/") {
        return Route::Unknown;
    }
    if path.ends_with("/tags/list") {
        return Route::TagList(&path[4..path.len() - "/tags/list".len()]);
    }
    if let Some(pos) = path.find("/manifests/") {
        let version = &path[pos + "/manifests/".len()..];
        if !version.is_empty() {
            return Route::Manifest(&path[4..pos], version);
        }
    }
    // `/v2/{path}/blobs/uploads/` — must be checked before the generic `/blobs/{κ}` split.
    if let Some(pos) = path.find("/blobs/uploads") {
        return Route::UploadStart(&path[4..pos]);
    }
    // `/v2/{path}/blobs/{κ}` — split on the LAST `/blobs/` so `{path}` may itself have segments.
    if let Some(idx) = path.rfind("/blobs/") {
        let kappa = &path[idx + "/blobs/".len()..];
        if !kappa.is_empty() {
            return Route::Blob(kappa);
        }
    }
    Route::Unknown
}

/// Handle a `/v2/...` request on `stream`. `head` is the request bytes already read by
/// [`super::live`] (through the `\r\n\r\n` terminator, possibly plus leading body bytes); any
/// remaining body is read from `stream` using `Content-Length`. Writes the full response.
pub fn handle_v2(
    stream: &mut TcpStream,
    head: &[u8],
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    handle_v2_admitted(stream, head, store, None)
}

/// [`handle_v2`] with a registry-supplied admission predicate run on blob `PUT` (spec §5.1, §10).
pub fn handle_v2_admitted(
    stream: &mut TcpStream,
    head: &[u8],
    store: &dyn KappaStore,
    admission: Option<Admission>,
) -> std::io::Result<()> {
    let split = find(head, b"\r\n\r\n").unwrap_or(head.len());
    let head_str = String::from_utf8_lossy(&head[..split]);
    let mut lines = head_str.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let raw_path = parts.next().unwrap_or("");
    let path = raw_path.split('?').next().unwrap_or(raw_path);
    let query = raw_path.split_once('?').map(|(_, q)| q).unwrap_or("");

    let mut content_length = 0usize;
    let mut range_start: Option<usize> = None;
    let mut if_match: Option<String> = None;
    let mut if_none_match: Option<String> = None;
    for line in lines {
        if let Some(v) = header_value(line, "content-length") {
            content_length = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = header_value(line, "content-range") {
            range_start = parse_content_range_start(v);
        } else if let Some(v) = header_value(line, "if-match") {
            if_match = Some(v.trim().to_string());
        } else if let Some(v) = header_value(line, "if-none-match") {
            if_none_match = Some(v.trim().to_string());
        }
    }

    match route(path) {
        Route::Version => {
            let body =
                std::format!(r#"{{"kappa-distribution":"{KAPPA_DISTRIBUTION_VERSION}"}}"#);
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
        Route::Health => write_resp(stream, 200, "OK", &[("Content-Length", "0")], b""),
        Route::UploadStart(prefix) => handle_upload_start(stream, prefix, query, store),
        Route::UploadSession(id) => handle_upload_session(
            stream,
            head,
            split,
            content_length,
            range_start,
            method,
            query,
            id,
            store,
        ),
        Route::Manifest(prefix, version) => handle_manifest(
            stream,
            head,
            split,
            content_length,
            method,
            prefix,
            version,
            if_match.as_deref(),
            if_none_match.as_deref(),
            store,
        ),
        Route::TagList(prefix) => handle_tag_list(stream, prefix, query),
        Route::Blob(kappa) => handle_blob(
            stream, head, split, content_length, method, path, query, kappa, store, admission,
        ),
        Route::Unknown => write_error(stream, 404, "Not Found", "NAME_INVALID", "unknown /v2 route"),
    }
}

/// Retrieve a blob by κ across σ-axes: the native (blake3/sha256) path first, then the
/// axis-polymorphic surface for foreign-axis content (multi-label push stores those via `put_axis`).
fn load(store: &dyn KappaStore, kappa: &KappaLabel71) -> Option<Bytes> {
    if let Ok(Some(b)) = store.get(kappa) {
        return Some(b);
    }
    if let Ok(Some(b)) = store.get_axis(kappa.as_bytes()) {
        return Some(b);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn handle_blob(
    stream: &mut TcpStream,
    head: &[u8],
    split: usize,
    content_length: usize,
    method: &str,
    req_path: &str,
    query: &str,
    kappa_str: &str,
    store: &dyn KappaStore,
    admission: Option<Admission>,
) -> std::io::Result<()> {
    let Some(kappa) = parse_kappa71(kappa_str) else {
        return write_error(stream, 400, "Bad Request", "NAME_INVALID", "malformed κ-label");
    };
    let axis = kappa.sigma_axis().unwrap_or("blake3");

    match method {
        "HEAD" => match load(store, &kappa) {
            Some(b) => write_resp(
                stream,
                200,
                "OK",
                &[
                    ("Content-Length", &b.as_ref().len().to_string()),
                    ("X-Kappa-Label", kappa.as_str()),
                    ("X-Kappa-Axis", axis),
                ],
                b"",
            ),
            None => write_resp(stream, 404, "Not Found", &[("Content-Length", "0")], b""),
        },
        "GET" => match load(store, &kappa) {
            Some(b) => write_resp(
                stream,
                200,
                "OK",
                &[
                    ("Content-Length", &b.as_ref().len().to_string()),
                    ("X-Kappa-Label", kappa.as_str()),
                    ("X-Kappa-Axis", axis),
                    ("Content-Type", "application/octet-stream"),
                ],
                b.as_ref(),
            ),
            None => write_error(stream, 404, "Not Found", "BLOB_UNKNOWN", "blob not found"),
        },
        "PUT" => {
            let body = read_body(stream, head, split, content_length)?;
            // Admission filters run before addressing (§5.1, §10.2): if the registry supplies a
            // predicate for this write path and it rejects, fail closed with 422 FILTER_REJECTED —
            // no content enters the store. `{path}` (between `/v2/` and `/blobs/`) is the scope.
            if let Some(admit) = admission {
                let scope = req_path
                    .strip_prefix("/v2/")
                    .and_then(|s| s.split("/blobs/").next())
                    .unwrap_or("");
                if admit(scope, &body).is_err() {
                    return write_error(
                        stream,
                        422,
                        "Unprocessable Entity",
                        "FILTER_REJECTED",
                        "admission filter rejected content",
                    );
                }
            }
            // Verify-on-put (§5.1): re-hash and reject a κ that does not match — the server never
            // trusts the client's asserted address. Multi-label push (`?also=`, §4.6) verifies EVERY
            // provided κ against the content; any mismatch rejects the whole push (store nothing).
            match verify_kappa(&body, &kappa) {
                Ok(true) => {}
                Ok(false) => {
                    return write_error(
                        stream,
                        409,
                        "Conflict",
                        "DIGEST_INVALID",
                        "κ-label does not match content",
                    );
                }
                Err(_) => {
                    return write_error(
                        stream,
                        400,
                        "Bad Request",
                        "NAME_INVALID",
                        "unsupported σ-axis",
                    );
                }
            }
            let also: Vec<&str> = query_values(query, "also");
            for label in &also {
                if verify_kappa_axis(&body, label.as_bytes()) != Ok(true) {
                    return write_error(
                        stream,
                        409,
                        "Conflict",
                        "DIGEST_INVALID",
                        "an also= κ-label does not match content",
                    );
                }
            }
            let existed = store.contains(&kappa);
            if store.put(axis, &body).is_err() {
                return write_error(
                    stream,
                    500,
                    "Internal Server Error",
                    "STORAGE_FAILURE",
                    "put failed",
                );
            }
            // Stored once; index every extra σ-axis label (spec §4.6 "stored once, all indexed").
            for label in &also {
                if let Some(a) = label.split(':').next() {
                    let _ = store.put_axis(a, &body);
                }
            }
            // Idempotent put (§5.1): 200 if it already existed, 201 on first store.
            let (status, reason) = if existed { (200, "OK") } else { (201, "Created") };
            write_resp(
                stream,
                status,
                reason,
                &[
                    ("Content-Length", "0"),
                    ("X-Kappa-Label", kappa.as_str()),
                    ("Location", req_path),
                ],
                b"",
            )
        }
        // Mark GC-eligible (§6.3). With no finalizer pins at L1, content is already GC-eligible, so
        // this acknowledges; real eviction is the GC sweep (Level 5).
        "DELETE" => write_resp(stream, 202, "Accepted", &[("Content-Length", "0")], b""),
        _ => write_resp(
            stream,
            405,
            "Method Not Allowed",
            &[("Content-Length", "0")],
            b"",
        ),
    }
}

/// `POST /v2/{path}/blobs/uploads/` — mount an existing blob (dedup, §6.7) or start a chunked upload
/// session (§6.4). `prefix` is the opaque `{path}`.
fn handle_upload_start(
    stream: &mut TcpStream,
    prefix: &str,
    query: &str,
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    // Mount (§6.7): if the blob is already present, bind it here without re-upload (201). κ-only
    // identity makes this a pure existence check — the blob is globally addressable regardless of
    // `{path}`. If absent, fall through to a normal upload session (202).
    if let Some(mount) = query_value(query, "mount") {
        if let Some(k) = parse_kappa71(mount) {
            if store.contains(&k) || store.contains_axis(k.as_bytes()) {
                let loc = std::format!("/v2/{prefix}/blobs/{}", k.as_str());
                return write_resp(
                    stream,
                    201,
                    "Created",
                    &[("Content-Length", "0"), ("Location", &loc)],
                    b"",
                );
            }
        }
    }
    let id = new_session_id();
    sessions().lock().unwrap().insert(
        id.clone(),
        Session {
            data: Vec::new(),
            path: prefix.to_string(),
        },
    );
    let loc = std::format!("/v2/_uploads/{id}");
    write_resp(
        stream,
        202,
        "Accepted",
        &[
            ("Content-Length", "0"),
            ("Location", &loc),
            ("X-Kappa-Upload-Session", &id),
        ],
        b"",
    )
}

/// `PATCH`/`GET`/`PUT`/`DELETE /v2/_uploads/{id}` — the chunked-upload session lifecycle (§6.4).
#[allow(clippy::too_many_arguments)]
fn handle_upload_session(
    stream: &mut TcpStream,
    head: &[u8],
    split: usize,
    content_length: usize,
    range_start: Option<usize>,
    method: &str,
    query: &str,
    id: &str,
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    match method {
        // Upload a chunk. Chunks MUST be sequential (§6.4): the chunk's start MUST equal the number
        // of bytes already received; an out-of-order chunk is 416 Range Not Satisfiable.
        "PATCH" => {
            let body = read_body(stream, head, split, content_length)?;
            let mut map = sessions().lock().unwrap();
            let Some(sess) = map.get_mut(id) else {
                return write_error(stream, 404, "Not Found", "BLOB_UPLOAD_UNKNOWN", "no session");
            };
            let start = range_start.unwrap_or(sess.data.len());
            if start != sess.data.len() {
                return write_error(
                    stream,
                    416,
                    "Range Not Satisfiable",
                    "BLOB_UPLOAD_INVALID",
                    "out-of-order chunk",
                );
            }
            sess.data.extend_from_slice(&body);
            let range = std::format!("0-{}", sess.data.len().saturating_sub(1));
            let loc = std::format!("/v2/_uploads/{id}");
            write_resp(
                stream,
                202,
                "Accepted",
                &[("Content-Length", "0"), ("Range", &range), ("Location", &loc)],
                b"",
            )
        }
        // Recovery (§6.4 Phase 2a): report the bytes received so the client can resume.
        "GET" => {
            let map = sessions().lock().unwrap();
            let Some(sess) = map.get(id) else {
                return write_error(stream, 404, "Not Found", "BLOB_UPLOAD_UNKNOWN", "no session");
            };
            let range = std::format!("0-{}", sess.data.len().saturating_sub(1));
            let loc = std::format!("/v2/_uploads/{id}");
            write_resp(
                stream,
                204,
                "No Content",
                &[("Content-Length", "0"), ("Range", &range), ("Location", &loc)],
                b"",
            )
        }
        // Complete (§6.4 Phase 3): the `?kappa=` is the hash of the ENTIRE assembled content; the
        // registry verifies the whole blob, not the final chunk.
        "PUT" => {
            let body = read_body(stream, head, split, content_length)?;
            let (assembled, prefix) = {
                let mut map = sessions().lock().unwrap();
                let Some(sess) = map.get_mut(id) else {
                    return write_error(
                        stream,
                        404,
                        "Not Found",
                        "BLOB_UPLOAD_UNKNOWN",
                        "no session",
                    );
                };
                sess.data.extend_from_slice(&body);
                (sess.data.clone(), sess.path.clone())
            };
            let Some(k) = query_value(query, "kappa").and_then(parse_kappa71) else {
                return write_error(stream, 400, "Bad Request", "NAME_INVALID", "missing/bad kappa");
            };
            let axis = k.sigma_axis().unwrap_or("blake3");
            match verify_kappa(&assembled, &k) {
                Ok(true) => {
                    if store.put(axis, &assembled).is_err() {
                        return write_error(
                            stream,
                            500,
                            "Internal Server Error",
                            "STORAGE_FAILURE",
                            "put failed",
                        );
                    }
                    sessions().lock().unwrap().remove(id);
                    let loc = std::format!("/v2/{prefix}/blobs/{}", k.as_str());
                    write_resp(
                        stream,
                        201,
                        "Created",
                        &[
                            ("Content-Length", "0"),
                            ("X-Kappa-Label", k.as_str()),
                            ("Location", &loc),
                        ],
                        b"",
                    )
                }
                _ => {
                    // Assembled content does not match the declared κ — discard the session (§6.4).
                    sessions().lock().unwrap().remove(id);
                    write_error(
                        stream,
                        409,
                        "Conflict",
                        "DIGEST_INVALID",
                        "assembled content does not match κ-label",
                    )
                }
            }
        }
        // Cancel (§6.4 Phase 2b): discard partial content.
        "DELETE" => {
            sessions().lock().unwrap().remove(id);
            write_resp(stream, 204, "No Content", &[("Content-Length", "0")], b"")
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

/// `GET`/`PUT`/`DELETE /v2/{path}/manifests/{version}` — resolve / bind / delete a tag (§6.5).
///
/// **κ-only identity (Law 2).** A `{version}` containing a colon is a κ-label and resolves directly,
/// bypassing tags entirely; otherwise it is a tag name resolved to a κ. Binding stores the content
/// first (content-before-tag, §7.5) then points the tag at the resulting κ. `If-Match`/`If-None-Match`
/// give `tag_set_if` compare-and-swap (§5.7).
#[allow(clippy::too_many_arguments)]
fn handle_manifest(
    stream: &mut TcpStream,
    head: &[u8],
    split: usize,
    content_length: usize,
    method: &str,
    prefix: &str,
    version: &str,
    if_match: Option<&str>,
    if_none_match: Option<&str>,
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    match method {
        "GET" => {
            // Direct κ (contains ':') resolves without touching tags; else resolve the tag pointer.
            let kappa_str = if version.contains(':') {
                version.to_string()
            } else {
                match tags()
                    .lock()
                    .unwrap()
                    .get(&(prefix.to_string(), version.to_string()))
                {
                    Some(k) => k.clone(),
                    None => {
                        return write_error(stream, 404, "Not Found", "TAG_UNKNOWN", "no such tag");
                    }
                }
            };
            let Some(kappa) = parse_kappa71(&kappa_str) else {
                return write_error(stream, 404, "Not Found", "BLOB_UNKNOWN", "bad κ-label");
            };
            match load(store, &kappa) {
                Some(b) => write_resp(
                    stream,
                    200,
                    "OK",
                    &[
                        ("Content-Length", &b.as_ref().len().to_string()),
                        ("X-Kappa-Label", &kappa_str),
                        ("Content-Type", "application/octet-stream"),
                    ],
                    b.as_ref(),
                ),
                None => write_error(stream, 404, "Not Found", "BLOB_UNKNOWN", "content absent"),
            }
        }
        "PUT" => {
            let body = read_body(stream, head, split, content_length)?;
            let k = hologram_space::address_bytes(&body); // blake3 identity
                                                          // Content-before-tag: durably store the content first (§7.5), then bind the pointer.
            if store.put("blake3", &body).is_err() {
                return write_error(
                    stream,
                    500,
                    "Internal Server Error",
                    "STORAGE_FAILURE",
                    "put failed",
                );
            }
            let key = (prefix.to_string(), version.to_string());
            let mut map = tags().lock().unwrap();
            let current = map.get(&key).cloned();
            // tag_set_if compare-and-swap (§5.7): a stale If-Match, or If-None-Match:* on an existing
            // tag, is a TAG_CONFLICT. The pointer moves; identity (the κ) never does.
            if let Some(expected) = if_match {
                if current.as_deref() != Some(expected) {
                    return write_error(
                        stream,
                        409,
                        "Conflict",
                        "TAG_CONFLICT",
                        "If-Match precondition failed",
                    );
                }
            }
            if if_none_match == Some("*") && current.is_some() {
                return write_error(stream, 409, "Conflict", "TAG_CONFLICT", "tag already exists");
            }
            let existed = current.is_some();
            map.insert(key, k.as_str().to_string());
            let (status, reason) = if existed { (200, "OK") } else { (201, "Created") };
            write_resp(
                stream,
                status,
                reason,
                &[("Content-Length", "0"), ("X-Kappa-Label", k.as_str())],
                b"",
            )
        }
        // Remove the tag binding — NOT the content (§5.2 tag_delete).
        "DELETE" => {
            let removed = tags()
                .lock()
                .unwrap()
                .remove(&(prefix.to_string(), version.to_string()))
                .is_some();
            let (status, reason) = if removed {
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

/// `GET /v2/{path}/tags/list` — list a path's tags in ASCIIbetical order, with cursor pagination
/// (`last` is a tag name, never an index) and `after`/`before` range filtering for timestamp tags
/// (§6.5). `n=0` returns an empty list with no `Link` header.
fn handle_tag_list(stream: &mut TcpStream, prefix: &str, query: &str) -> std::io::Result<()> {
    let n: usize = query_value(query, "n").and_then(|s| s.parse().ok()).unwrap_or(100);
    let order = query_value(query, "order").unwrap_or("asc");
    let last = query_value(query, "last");
    let after = query_value(query, "after");
    let before = query_value(query, "before");

    let mut items: Vec<(String, String)> = {
        let map = tags().lock().unwrap();
        map.iter()
            .filter(|((p, _), _)| p == prefix)
            .map(|((_, name), k)| (name.clone(), k.clone()))
            .collect()
    };
    items.sort_by(|a, b| a.0.cmp(&b.0)); // ASCIIbetical
    if order == "desc" {
        items.reverse();
    }
    if let Some(after) = after {
        items.retain(|(name, _)| name.as_str() >= after);
    }
    if let Some(before) = before {
        items.retain(|(name, _)| name.as_str() <= before);
    }
    if let Some(last) = last {
        // Names strictly past the cursor in the requested order.
        if order == "desc" {
            items.retain(|(name, _)| name.as_str() < last);
        } else {
            items.retain(|(name, _)| name.as_str() > last);
        }
    }

    if n == 0 {
        // Empty page, no Link (§6.5 MUST).
        return write_resp(
            stream,
            200,
            "OK",
            &[
                ("Content-Type", "application/json"),
                ("Content-Length", "11"),
            ],
            br#"{"tags":[]}"#,
        );
    }

    let has_more = items.len() > n;
    items.truncate(n);
    let mut body = String::from(r#"{"tags":["#);
    for (i, (name, k)) in items.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        body.push_str(&std::format!(
            r#"{{"name":"{}","kappa":"{}"}}"#,
            json_escape(name),
            json_escape(k)
        ));
    }
    body.push_str("]}");

    let len = body.len().to_string();
    let mut headers: Vec<(&str, &str)> = alloc::vec![
        ("Content-Type", "application/json"),
        ("Content-Length", &len),
    ];
    let link;
    if has_more {
        let cursor = &items.last().unwrap().0;
        link = std::format!("</v2/{prefix}/tags/list?last={cursor}>; rel=\"next\"");
        headers.push(("Link", &link));
    } else {
        link = String::new();
    }
    let _ = &link;
    write_resp(stream, 200, "OK", &headers, body.as_bytes())
}

/// Minimal JSON string escaping for tag names/κ-labels (quotes, backslash, control chars).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&std::format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Read a body of `content_length` bytes: the bytes already buffered past the header terminator,
/// then the remainder from `stream`. Capped at [`MAX_BODY_BYTES`].
fn read_body(
    stream: &mut TcpStream,
    head: &[u8],
    split: usize,
    content_length: usize,
) -> std::io::Result<Vec<u8>> {
    let want = content_length.min(MAX_BODY_BYTES);
    let mut body = Vec::with_capacity(want.min(64 * 1024));
    let body_start = (split + 4).min(head.len());
    let already = &head[body_start..];
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

fn parse_kappa71(s: &str) -> Option<KappaLabel71> {
    let bytes: [u8; 71] = s.as_bytes().try_into().ok()?;
    KappaLabel::from_bytes(&bytes).ok()
}

/// The start offset from a `Content-Range: <start>-<end>` header (tolerates a `bytes ` prefix and a
/// `/total` suffix, spec §6.4).
fn parse_content_range_start(v: &str) -> Option<usize> {
    let v = v.trim();
    let v = v.strip_prefix("bytes").map(str::trim).unwrap_or(v);
    let range = v.split('/').next().unwrap_or(v);
    range.split('-').next()?.trim().parse().ok()
}

fn header_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let (k, v) = line.split_once(':')?;
    if k.trim().eq_ignore_ascii_case(name) {
        Some(v)
    } else {
        None
    }
}

fn query_values<'a>(query: &'a str, key: &str) -> Vec<&'a str> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .filter_map(|(k, v)| (k == key).then_some(v))
        .collect()
}

fn query_value<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find_map(|(k, v)| (k == key).then_some(v))
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Write a response. Callers supply every header (including `Content-Length`) so HEAD can declare a
/// body length it does not send.
fn write_resp(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> std::io::Result<()> {
    let mut head = std::format!("HTTP/1.1 {status} {reason}\r\n");
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

/// Write a spec §6.16 error response: `{"errors":[{"code":…,"message":…}]}` with an
/// uppercase-underscore `code`.
fn write_error(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    code: &str,
    message: &str,
) -> std::io::Result<()> {
    let body = std::format!(r#"{{"errors":[{{"code":"{code}","message":"{message}"}}]}}"#);
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

#[cfg(test)]
mod tests {
    use super::super::live::serve;
    use super::*;
    use alloc::sync::Arc;
    use hologram_space::{address_bytes, address_bytes_axis};
    use hologram_tck::MemKappaStore;

    /// Minimal raw HTTP/1.1 client with optional extra request headers. Returns
    /// `(status, headers, body)`.
    fn request_h(
        addr: &str,
        method: &str,
        path: &str,
        extra: &[(&str, &str)],
        body: &[u8],
    ) -> (u16, Vec<(String, String)>, Vec<u8>) {
        let mut stream = TcpStream::connect(addr).unwrap();
        let mut req = std::format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\n");
        for (k, v) in extra {
            req.push_str(k);
            req.push_str(": ");
            req.push_str(v);
            req.push_str("\r\n");
        }
        req.push_str(&std::format!(
            "Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        ));
        stream.write_all(req.as_bytes()).unwrap();
        if !body.is_empty() {
            stream.write_all(body).unwrap();
        }
        stream.flush().unwrap();
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).unwrap();
        let split = find(&resp, b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&resp[..split]).to_string();
        let mut lines = head.split("\r\n");
        let status: u16 = lines
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        (status, headers, resp[split + 4..].to_vec())
    }

    fn request(
        addr: &str,
        method: &str,
        path: &str,
        body: &[u8],
    ) -> (u16, Vec<(String, String)>, Vec<u8>) {
        request_h(addr, method, path, &[], body)
    }

    fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// KD-1 — verify-on-put + idempotent put + 409 on κ/content mismatch (§5.1).
    #[test]
    fn kd1_verify_on_put_idempotent_and_digest_mismatch() {
        let store = Arc::new(MemKappaStore::new());
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();

        let content = b"kappa-distribution level 1";
        let k = address_bytes(content);
        let path = std::format!("/v2/testns/blobs/{}", k.as_str());

        let (status, headers, _) = request(&addr, "PUT", &path, content);
        assert_eq!(status, 201, "first put is 201 Created");
        assert_eq!(header(&headers, "X-Kappa-Label"), Some(k.as_str()));

        let (status, _, _) = request(&addr, "PUT", &path, content);
        assert_eq!(status, 200, "idempotent re-put is 200 OK");

        let (status, _, body) = request(&addr, "PUT", &path, b"different bytes");
        assert_eq!(status, 409, "κ/content mismatch is rejected");
        assert!(
            String::from_utf8_lossy(&body).contains("DIGEST_INVALID"),
            "409 body carries the DIGEST_INVALID error code"
        );
        server.shutdown();
    }

    /// KD-2 — HEAD/GET round-trip, verify-on-receipt, exact-bytes, 404 for absent (§3.1, §6.3).
    #[test]
    fn kd2_head_get_roundtrip_verify_on_receipt_exact_bytes() {
        let store = Arc::new(MemKappaStore::new());
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();

        let content = b"exact bytes, byte for byte";
        let k = address_bytes(content);
        let path = std::format!("/v2/ns/blobs/{}", k.as_str());
        assert_eq!(request(&addr, "PUT", &path, content).0, 201);

        let (status, headers, body) = request(&addr, "HEAD", &path, b"");
        assert_eq!(status, 200);
        assert_eq!(header(&headers, "X-Kappa-Label"), Some(k.as_str()));
        assert_eq!(header(&headers, "X-Kappa-Axis"), Some("blake3"));
        assert_eq!(
            header(&headers, "Content-Length"),
            Some(content.len().to_string().as_str())
        );
        assert!(body.is_empty(), "HEAD has no body");

        let (status, headers, got) = request(&addr, "GET", &path, b"");
        assert_eq!(status, 200);
        assert_eq!(got, content, "GET returns exact bytes");
        let served = header(&headers, "X-Kappa-Label").unwrap();
        assert_eq!(address_bytes(&got).as_str(), served, "verify-on-receipt");

        let absent = address_bytes(b"nowhere");
        let (status, _, _) =
            request(&addr, "GET", &std::format!("/v2/ns/blobs/{}", absent.as_str()), b"");
        assert_eq!(status, 404);
        server.shutdown();
    }

    /// KD-3 — version check + health endpoints (§6.2, §6.14).
    #[test]
    fn kd3_version_check_and_health() {
        let store = Arc::new(MemKappaStore::new());
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();

        let (status, _, body) = request(&addr, "GET", "/v2/", b"");
        assert_eq!(status, 200);
        assert!(String::from_utf8_lossy(&body).contains(KAPPA_DISTRIBUTION_VERSION));

        for probe in ["live", "ready", "startup"] {
            let (status, _, _) =
                request(&addr, "GET", &std::format!("/v2/_health/{probe}"), b"");
            assert_eq!(status, 200, "health probe {probe} is live");
        }
        server.shutdown();
    }

    /// KD-4 — multi-label push (`?also=`): every κ verified, stored once under all σ-axes; any
    /// mismatch rejects the whole push (§4.6).
    #[test]
    fn kd4_multi_label_push_all_verified_stored_once() {
        let store = Arc::new(MemKappaStore::new());
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();

        let content = b"one blob, two addresses";
        let kb = address_bytes(content); // blake3 (primary)
        let ks_bytes = address_bytes_axis("sha256", content).unwrap();
        let ks = core::str::from_utf8(&ks_bytes).unwrap(); // sha256 (also)

        let path = std::format!("/v2/ns/blobs/{}?also={}", kb.as_str(), ks);
        let (status, headers, _) = request(&addr, "PUT", &path, content);
        assert_eq!(status, 201);
        assert_eq!(header(&headers, "X-Kappa-Label"), Some(kb.as_str()));

        // The blob is retrievable under BOTH addresses (stored once, indexed by each σ-axis).
        assert_eq!(
            request(&addr, "GET", &std::format!("/v2/ns/blobs/{}", kb.as_str()), b"").0,
            200
        );
        let (s, _, got) = request(&addr, "GET", &std::format!("/v2/ns/blobs/{ks}"), b"");
        assert_eq!(s, 200);
        assert_eq!(got, content);

        // A push whose also= label does not match the content is rejected whole — nothing stored.
        let other = b"unrelated payload";
        let kb2 = address_bytes(other);
        let bad_bytes = address_bytes_axis("sha256", b"a different thing").unwrap();
        let bad = core::str::from_utf8(&bad_bytes).unwrap();
        let path2 = std::format!("/v2/ns/blobs/{}?also={}", kb2.as_str(), bad);
        assert_eq!(request(&addr, "PUT", &path2, other).0, 409);
        assert_eq!(
            request(&addr, "GET", &std::format!("/v2/ns/blobs/{}", kb2.as_str()), b"").0,
            404,
            "a rejected push stores nothing"
        );
        server.shutdown();
    }

    /// KD-5 — chunked upload session: assemble + verify-on-complete, out-of-order → 416 → GET
    /// recovery → resume, and cancel (§6.4).
    #[test]
    fn kd5_chunked_upload_recovery_and_cancel() {
        let store = Arc::new(MemKappaStore::new());
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();

        let content = b"assembled from two chunks over the wire";
        let k = address_bytes(content);
        let (c1, c2) = content.split_at(10);

        // Start a session.
        let (status, headers, _) = request(&addr, "POST", "/v2/ns/blobs/uploads/", b"");
        assert_eq!(status, 202);
        let loc = header(&headers, "Location").unwrap().to_string();

        // First chunk at offset 0.
        let (s, _, _) = request_h(&addr, "PATCH", &loc, &[("Content-Range", "0-9")], c1);
        assert_eq!(s, 202);

        // Out-of-order chunk → 416 Range Not Satisfiable.
        let (s, _, _) = request_h(&addr, "PATCH", &loc, &[("Content-Range", "99-108")], b"xxxxxxxxxx");
        assert_eq!(s, 416);

        // Recovery: GET the session → 204 with the bytes-received Range.
        let (s, rh, _) = request(&addr, "GET", &loc, b"");
        assert_eq!(s, 204);
        assert_eq!(header(&rh, "Range"), Some("0-9"), "recovery reports bytes received");

        // Resume from the correct offset.
        let range2 = std::format!("{}-{}", c1.len(), content.len() - 1);
        let (s, _, _) = request_h(&addr, "PATCH", &loc, &[("Content-Range", &range2)], c2);
        assert_eq!(s, 202);

        // Complete: the ?kappa= is the hash of the WHOLE assembled content.
        let complete = std::format!("{loc}?kappa={}", k.as_str());
        let (s, hh, _) = request(&addr, "PUT", &complete, b"");
        assert_eq!(s, 201);
        assert_eq!(header(&hh, "X-Kappa-Label"), Some(k.as_str()));

        // The assembled blob is retrievable and byte-exact.
        let (s, _, got) = request(&addr, "GET", &std::format!("/v2/ns/blobs/{}", k.as_str()), b"");
        assert_eq!(s, 200);
        assert_eq!(got, content);

        // Cancel discards the session.
        let (_, h2, _) = request(&addr, "POST", "/v2/ns/blobs/uploads/", b"");
        let loc2 = header(&h2, "Location").unwrap().to_string();
        assert_eq!(request(&addr, "DELETE", &loc2, b"").0, 204);
        assert_eq!(request(&addr, "GET", &loc2, b"").0, 404, "cancelled session is gone");
        server.shutdown();
    }

    /// KD-6 — mount: an existing blob is bound without re-upload (201); an absent one falls back to
    /// an upload session (202) (§6.7).
    #[test]
    fn kd6_mount_existing_blob_or_fall_back_to_upload() {
        let store = Arc::new(MemKappaStore::new());
        let content = b"already resident in the store";
        let k = store.put("blake3", content).unwrap();
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();

        // Mount a blob that exists → 201 with the blob's namespaced Location, no upload.
        let (s, h, _) = request(
            &addr,
            "POST",
            &std::format!("/v2/ns/blobs/uploads/?mount={}", k.as_str()),
            b"",
        );
        assert_eq!(s, 201);
        assert_eq!(
            header(&h, "Location"),
            Some(std::format!("/v2/ns/blobs/{}", k.as_str()).as_str())
        );

        // Mount a blob that is absent → 202, fall back to a normal upload session.
        let absent = address_bytes(b"not here at all");
        let (s, h, _) = request(
            &addr,
            "POST",
            &std::format!("/v2/ns/blobs/uploads/?mount={}", absent.as_str()),
            b"",
        );
        assert_eq!(s, 202);
        assert!(header(&h, "Location").unwrap().contains("/v2/_uploads/"));
        server.shutdown();
    }

    /// KD-7 — tag bind + resolve round-trip, content-before-tag, and delete keeps the content (§6.5).
    #[test]
    fn kd7_tag_bind_resolve_content_before_tag_and_delete() {
        let store = Arc::new(MemKappaStore::new());
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();

        let content = b"the thing latest points at";
        let k = address_bytes(content);

        // Bind the tag `latest` (stores content + points the tag at it).
        let (s, h, _) = request(&addr, "PUT", "/v2/kd7ns/manifests/latest", content);
        assert_eq!(s, 201);
        assert_eq!(header(&h, "X-Kappa-Label"), Some(k.as_str()));

        // Resolve the tag → the content, with κ echoed.
        let (s, h, got) = request(&addr, "GET", "/v2/kd7ns/manifests/latest", b"");
        assert_eq!(s, 200);
        assert_eq!(got, content);
        assert_eq!(header(&h, "X-Kappa-Label"), Some(k.as_str()));

        // Content-before-tag: the bound κ is retrievable as a blob (content is present).
        assert_eq!(
            request(&addr, "GET", &std::format!("/v2/kd7ns/blobs/{}", k.as_str()), b"").0,
            200
        );

        // Deleting the tag removes the binding but NOT the content.
        assert_eq!(request(&addr, "DELETE", "/v2/kd7ns/manifests/latest", b"").0, 202);
        assert_eq!(request(&addr, "GET", "/v2/kd7ns/manifests/latest", b"").0, 404);
        assert_eq!(
            request(&addr, "GET", &std::format!("/v2/kd7ns/blobs/{}", k.as_str()), b"").0,
            200,
            "tag_delete does not delete content"
        );
        server.shutdown();
    }

    /// KD-8 — tag list: ASCIIbetical order, asc/desc, cursor pagination (last = name), n=0 empty
    /// with no Link (§6.5).
    #[test]
    fn kd8_tag_list_order_and_cursor_pagination() {
        let store = Arc::new(MemKappaStore::new());
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();
        for name in ["a", "b", "c", "d"] {
            let body = std::format!("content-{name}");
            assert_eq!(
                request(&addr, "PUT", &std::format!("/v2/kd8ns/manifests/{name}"), body.as_bytes()).0,
                201
            );
        }

        // asc, n=2 → [a,b] + a Link to the next page.
        let (s, h, body) = request(&addr, "GET", "/v2/kd8ns/tags/list?n=2", b"");
        assert_eq!(s, 200);
        let bs = String::from_utf8_lossy(&body);
        assert!(bs.contains(r#""name":"a""#) && bs.contains(r#""name":"b""#));
        assert!(!bs.contains(r#""name":"c""#));
        assert!(header(&h, "Link").is_some(), "more pages → Link");

        // continuation: last=b → [c,d], no Link.
        let (_, h2, body2) = request(&addr, "GET", "/v2/kd8ns/tags/list?n=2&last=b", b"");
        let bs2 = String::from_utf8_lossy(&body2);
        assert!(bs2.contains(r#""name":"c""#) && bs2.contains(r#""name":"d""#));
        assert!(!bs2.contains(r#""name":"a""#));
        assert!(header(&h2, "Link").is_none(), "last page → no Link");

        // desc order → d first.
        let (_, _, bd) = request(&addr, "GET", "/v2/kd8ns/tags/list?order=desc&n=1", b"");
        assert!(String::from_utf8_lossy(&bd).contains(r#""name":"d""#));

        // n=0 → empty list, no Link.
        let (s, h0, b0) = request(&addr, "GET", "/v2/kd8ns/tags/list?n=0", b"");
        assert_eq!(s, 200);
        assert!(header(&h0, "Link").is_none());
        assert!(String::from_utf8_lossy(&b0).contains(r#""tags":[]"#));
        server.shutdown();
    }

    /// KD-9 (Law-2 guard) — identity is the κ, not the tag: two names over identical content resolve
    /// to the SAME κ; a direct κ resolves without tags; and tag_set_if CAS moves the pointer only,
    /// never the identity (§5.7, §7.5, Law 2 / SPINE-1).
    #[test]
    fn kd9_identity_is_kappa_not_tag_with_cas() {
        let store = Arc::new(MemKappaStore::new());
        let server = serve(store, false).unwrap();
        let addr = server.addr().to_string();

        let content = b"identity is content-derived";
        let k = address_bytes(content);

        // Two different tag names over identical content resolve to the SAME κ — identity is the κ.
        assert_eq!(request(&addr, "PUT", "/v2/kd9ns/manifests/name-one", content).0, 201);
        assert_eq!(request(&addr, "PUT", "/v2/kd9ns/manifests/name-two", content).0, 201);
        let (_, h1, _) = request(&addr, "GET", "/v2/kd9ns/manifests/name-one", b"");
        let (_, h2, _) = request(&addr, "GET", "/v2/kd9ns/manifests/name-two", b"");
        assert_eq!(header(&h1, "X-Kappa-Label"), header(&h2, "X-Kappa-Label"));
        assert_eq!(header(&h1, "X-Kappa-Label"), Some(k.as_str()));

        // A direct κ (version segment contains ':') resolves without touching the tag layer.
        let (s, _, got) =
            request(&addr, "GET", &std::format!("/v2/kd9ns/manifests/{}", k.as_str()), b"");
        assert_eq!(s, 200);
        assert_eq!(got, content);

        // tag_set_if: a stale If-Match is a TAG_CONFLICT (the pointer is not moved).
        let (s, _, body) = request_h(
            &addr,
            "PUT",
            "/v2/kd9ns/manifests/name-one",
            &[("If-Match", address_bytes(b"stale").as_str())],
            b"attempted new content",
        );
        assert_eq!(s, 409);
        assert!(String::from_utf8_lossy(&body).contains("TAG_CONFLICT"));

        // With the correct current κ, the rebind succeeds — a pointer change, not an identity change:
        // the OLD κ's content is unchanged and still retrievable.
        let (s, _, _) = request_h(
            &addr,
            "PUT",
            "/v2/kd9ns/manifests/name-one",
            &[("If-Match", k.as_str())],
            b"replacement content",
        );
        assert!(s == 200 || s == 201);
        assert_eq!(
            request(&addr, "GET", &std::format!("/v2/kd9ns/blobs/{}", k.as_str()), b"").0,
            200,
            "rebinding moved the pointer only; the κ's content is immutable"
        );
        server.shutdown();
    }
}
