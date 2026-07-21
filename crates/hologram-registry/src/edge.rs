//! Level 3 — edges as content-addressed blobs (spec §3.4, §5.3) + the edge index (§6.6).
//!
//! An edge is a typed directed relationship `(source) --relation--> (target)` that is itself a blob.
//! Its canonical bytes come from the **shared** standard [`uor_distribution::edge::edge_canonical`] —
//! the single cross-registry byte contract — and its κ-label is those bytes hashed under the
//! **source's σ-axis** (§3.4), so an edge's identity travels with its source. This module composes the
//! standard's byte form with hologram's σ-axis addressing and [`KappaStore`]; it never re-defines the
//! byte layout (plan risk R1: one canonical form, owned by the standard).

use std::net::TcpStream;
use std::sync::{Mutex, OnceLock};

use hologram_space::{address_bytes_axis, KappaLabel71, KappaStore, StoreError};
use uor_distribution::edge::edge_canonical;
use uor_distribution::ErrorCode;

use crate::http_util::{
    json_escape, json_field, parse71, parse_request, query_value, read_body, write_error, write_resp,
};

/// An edge blob: its canonical bytes and its κ-label (on-the-wire, under the source's σ-axis).
pub struct EdgeBlob {
    /// The Appendix C canonical bytes (from the shared standard).
    pub bytes: Vec<u8>,
    /// The edge's κ-label bytes, computed under the source's σ-axis (§3.4).
    pub kappa: Vec<u8>,
}

/// Build an edge blob: canonical bytes (via the shared standard) + its κ-label under the source's
/// σ-axis (§3.4). `metadata` is opaque, already-deterministic bytes (may be empty).
pub fn edge_blob(
    source: &KappaLabel71,
    relation: &str,
    target: &KappaLabel71,
    metadata: &[u8],
) -> EdgeBlob {
    let bytes = edge_canonical(source.as_bytes(), relation, target.as_bytes(), metadata);
    let axis = source.sigma_axis().unwrap_or("blake3");
    let kappa = address_bytes_axis(axis, &bytes).expect("source σ-axis is a recognized axis");
    EdgeBlob { bytes, kappa }
}

/// Store an edge blob in `store` under the source's σ-axis and return its κ-label. Idempotent — an
/// edge is content-addressed, so re-creating the same `(source, relation, target, metadata)` is a
/// no-op that yields the same κ (§7.6 edge consistency).
pub fn edge_put(
    store: &dyn KappaStore,
    source: &KappaLabel71,
    relation: &str,
    target: &KappaLabel71,
    metadata: &[u8],
) -> Result<Vec<u8>, StoreError> {
    let edge = edge_blob(source, relation, target, metadata);
    let axis = source.sigma_axis().unwrap_or("blake3");
    store.put_axis(axis, &edge.bytes)
}

// ─────────────────────────── the edge index (Level 3, spec §3.4/§6.6) ───────────────────────────
//
// The registry MUST maintain an index that answers queries by source, target, and relation (§3.4).
// This process-local index is a single-node tracer for the semantics; a durable, backend-resident
// index is the productionization. Tests use a unique `{path}` each to stay isolated.

struct EdgeRecord {
    path: String,
    source: String,
    relation: String,
    target: String,
    edge_kappa: String,
}

fn edge_index() -> &'static Mutex<Vec<EdgeRecord>> {
    static I: OnceLock<Mutex<Vec<EdgeRecord>>> = OnceLock::new();
    I.get_or_init(|| Mutex::new(Vec::new()))
}

/// Store an edge blob and index it under `path` (idempotent per `(path, edge κ)`). Returns the edge
/// κ-label string and whether it already existed. Shared by the HTTP handler and the L4 composition
/// provenance edges (composed-of / witness-of).
pub(crate) fn record(
    store: &dyn KappaStore,
    path: &str,
    source: &KappaLabel71,
    relation: &str,
    target: &KappaLabel71,
    metadata: &[u8],
) -> Result<(String, bool), StoreError> {
    let k = edge_put(store, source, relation, target, metadata)?;
    let edge_kappa = String::from_utf8_lossy(&k).to_string();
    let mut idx = edge_index().lock().unwrap();
    let existed = idx
        .iter()
        .any(|r| r.path == path && r.edge_kappa == edge_kappa);
    if !existed {
        idx.push(EdgeRecord {
            path: path.to_string(),
            source: source.as_str().to_string(),
            relation: relation.to_string(),
            target: target.as_str().to_string(),
            edge_kappa: edge_kappa.clone(),
        });
    }
    Ok((edge_kappa, existed))
}

/// All edges recorded under `path`, as `(source, relation, target)` κ-label triples. Used by the GC
/// sweep to walk reachability (§9.2).
pub(crate) fn edges_for(path: &str) -> Vec<(String, String, String)> {
    edge_index()
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.path == path)
        .map(|r| (r.source.clone(), r.relation.clone(), r.target.clone()))
        .collect()
}

/// The witness κ for `target_kappa` under `path`: the source of a `witness-of` edge into it (§6.8).
pub(crate) fn find_witness_source(path: &str, target_kappa: &str) -> Option<String> {
    let idx = edge_index().lock().unwrap();
    idx.iter()
        .find(|r| r.path == path && r.relation == "witness-of" && r.target == target_kappa)
        .map(|r| r.source.clone())
}

/// `PUT`/`GET`/`DELETE /v2/{path}/edges[/{κ}]` — create an edge, query edges involving a κ, or remove
/// an edge (spec §6.6). `head` is the request bytes already read by the server.
pub fn handle_edges(
    stream: &mut TcpStream,
    head: &[u8],
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    let (method, path, query, content_length, split) = parse_request(head);
    let Some(epos) = path.find("/edges") else {
        return write_error(stream, 404, "Not Found", ErrorCode::NameInvalid, "not an edge route");
    };
    let prefix = &path[4..epos];
    let trailing = path[epos + "/edges".len()..].trim_start_matches('/');

    match method {
        "PUT" => {
            let body = read_body(stream, head, split, content_length)?;
            let bs = String::from_utf8_lossy(&body);
            let (Some(src), Some(rel), Some(tgt)) = (
                json_field(&bs, "source"),
                json_field(&bs, "relation"),
                json_field(&bs, "target"),
            ) else {
                return write_error(
                    stream,
                    400,
                    "Bad Request",
                    ErrorCode::NameInvalid,
                    "edge needs source, relation, target",
                );
            };
            let (Some(source), Some(target)) = (parse71(&src), parse71(&tgt)) else {
                return write_error(
                    stream,
                    400,
                    "Bad Request",
                    ErrorCode::NameInvalid,
                    "malformed source/target κ",
                );
            };
            // edge_put MUST verify the source exists (§5.3); the target MAY be absent.
            if !(store.contains(&source) || store.contains_axis(source.as_bytes())) {
                return write_error(
                    stream,
                    409,
                    "Conflict",
                    ErrorCode::EdgeSourceAbsent,
                    "edge source κ absent",
                );
            }
            let (edge_kappa, existed) = match record(store, prefix, &source, &rel, &target, b"") {
                Ok(x) => x,
                Err(_) => {
                    return write_error(
                        stream,
                        500,
                        "Internal Server Error",
                        ErrorCode::Unsupported,
                        "edge store failed",
                    );
                }
            };
            let (status, reason) = if existed { (200, "OK") } else { (201, "Created") };
            write_resp(
                stream,
                status,
                reason,
                &[("Content-Length", "0"), ("X-Kappa-Label", &edge_kappa)],
                b"",
            )
        }
        "GET" => {
            let node = trailing;
            let direction = query_value(query, "direction").unwrap_or("outbound");
            let relation = query_value(query, "relation");
            let idx = edge_index().lock().unwrap();
            let mut body = String::from(r#"{"edges":["#);
            let mut first = true;
            for r in idx.iter().filter(|r| r.path == prefix) {
                let dir_ok = match direction {
                    "inbound" => r.target == node,
                    "both" => r.source == node || r.target == node,
                    _ => r.source == node,
                };
                let rel_ok = relation.is_none_or(|rel| r.relation == rel);
                if dir_ok && rel_ok {
                    if !first {
                        body.push(',');
                    }
                    first = false;
                    body.push_str(&format!(
                        r#"{{"edge_kappa":"{}","source":"{}","relation":"{}","target":"{}","metadata":{{}}}}"#,
                        r.edge_kappa,
                        r.source,
                        json_escape(&r.relation),
                        r.target
                    ));
                }
            }
            body.push_str("]}");
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
        "DELETE" => {
            let mut idx = edge_index().lock().unwrap();
            let before = idx.len();
            idx.retain(|r| !(r.path == prefix && r.edge_kappa == trailing));
            let (status, reason) = if idx.len() < before {
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

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_space::KappaLabel;
    use hologram_tck::MemKappaStore;

    fn label(axis: &str, bytes: &[u8]) -> KappaLabel71 {
        let v = address_bytes_axis(axis, bytes).unwrap();
        let arr: [u8; 71] = v.as_slice().try_into().unwrap();
        KappaLabel::from_bytes(&arr).unwrap()
    }

    /// KD-11 — an edge is a content-addressed blob whose κ is built from the **shared** standard byte
    /// form under the **source's** σ-axis; it is deterministic, idempotent, and storable/retrievable
    /// by its own κ (§3.4, §7.6).
    #[test]
    fn kd11_edge_kappa_inherits_source_axis_via_shared_canonical_form() {
        let store = MemKappaStore::new();
        let source = store.put("blake3", b"source blob").unwrap();
        let target = store.put("blake3", b"target blob").unwrap();

        let e = edge_blob(&source, "derives-from", &target, b"");
        assert_eq!(
            e.bytes,
            edge_canonical(source.as_bytes(), "derives-from", target.as_bytes(), b"")
        );
        assert_eq!(e.kappa, edge_blob(&source, "derives-from", &target, b"").kappa);
        assert!(e.kappa.starts_with(b"blake3:"));

        let k1 = edge_put(&store, &source, "derives-from", &target, b"").unwrap();
        let k2 = edge_put(&store, &source, "derives-from", &target, b"").unwrap();
        assert_eq!(k1, k2);
        assert_eq!(k1, e.kappa);
        assert!(store.contains_axis(&k1), "the edge blob is stored under its own κ");

        let sha_source = label("sha256", b"source blob");
        let cross = edge_blob(&sha_source, "owns", &target, b"");
        assert!(cross.kappa.starts_with(b"sha256:"), "edge κ follows the source axis");
    }
}
