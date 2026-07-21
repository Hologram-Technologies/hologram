//! KD-15/16/17 — the Level 4 surface (composition, witnesses, schemas) over the registry server.
//! Black-box over a socket: operands are created via L1 blobs, then composed via L4.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use hologram_registry::server::serve;
use hologram_space::{address_bytes, address_bytes_axis};
use hologram_tck::MemKappaStore;

fn request(
    addr: &str,
    method: &str,
    path: &str,
    body: &[u8],
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let mut stream = TcpStream::connect(addr).unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(req.as_bytes()).unwrap();
    if !body.is_empty() {
        stream.write_all(body).unwrap();
    }
    stream.flush().unwrap();
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).unwrap();
    let split = resp.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
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

fn header<'a>(h: &'a [(String, String)], name: &str) -> Option<&'a str> {
    h.iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// Extract a JSON string field from a response body.
fn json_get(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let after = &body[body.find(&pat)? + pat.len()..];
    let after = &after[after.find(':')? + 1..];
    let after = &after[after.find('"')? + 1..];
    Some(after[..after.find('"')?].to_string())
}

fn put_blob(addr: &str, path: &str, content: &[u8]) -> String {
    let k = address_bytes(content);
    let (s, _, _) = request(addr, "PUT", &format!("/v2/{path}/blobs/{}", k.as_str()), content);
    assert!(s == 201 || s == 200, "blob put status {s}");
    k.as_str().to_string()
}

fn compose(addr: &str, path: &str, op: &str, operands: &[&str]) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let list = operands
        .iter()
        .map(|o| format!("\"{o}\""))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!("{{\"operands\":[{list}]}}");
    request(addr, "POST", &format!("/v2/{path}/compose/{op}"), body.as_bytes())
}

/// KD-15 — compose applies the operation, returns composed + witness + provenance, g2 is commutative,
/// the composed blob is retrievable, and a σ-axis mismatch is 422 AXIS_MISMATCH (§5.4, §6.9).
#[test]
fn kd15_compose_ops_provenance_and_axis_mismatch() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    let a = put_blob(&addr, "kd15ns", b"operand alpha");
    let b = put_blob(&addr, "kd15ns", b"operand beta");

    // g2(a,b) → 200 with composed + witness + operation.
    let (s, _, body) = compose(&addr, "kd15ns", "g2", &[&a, &b]);
    assert_eq!(s, 200);
    let bs = String::from_utf8_lossy(&body);
    let composed = json_get(&bs, "composed").unwrap();
    let witness = json_get(&bs, "witness").unwrap();
    assert_eq!(json_get(&bs, "operation").as_deref(), Some("g2"));
    assert!(composed.starts_with("blake3:") && witness.starts_with("blake3:"));

    // g2 is commutative: g2(a,b).composed == g2(b,a).composed.
    let (_, _, body2) = compose(&addr, "kd15ns", "g2", &[&b, &a]);
    assert_eq!(json_get(&String::from_utf8_lossy(&body2), "composed"), Some(composed.clone()));

    // The composed blob is retrievable (L1 GET by the composed κ).
    assert_eq!(request(&addr, "GET", &format!("/v2/kd15ns/blobs/{composed}"), b"").0, 200);

    // composed-of edges: querying edges out of the composed κ names both operands.
    let (_, _, edges) = request(
        &addr,
        "GET",
        &format!("/v2/kd15ns/edges/{composed}?direction=outbound&relation=composed-of"),
        b"",
    );
    let es = String::from_utf8_lossy(&edges);
    assert!(es.contains(&a) && es.contains(&b), "composed-of edges reference the operands");

    // A unary op (e8) with one operand also composes.
    assert_eq!(compose(&addr, "kd15ns", "e8", &[&a]).0, 200);

    // σ-axis homogeneity: mixed-axis operands → 422 AXIS_MISMATCH.
    let a_sha = String::from_utf8(address_bytes_axis("sha256", b"operand alpha").unwrap()).unwrap();
    let (s, _, mm) = compose(&addr, "kd15ns", "g2", &[&a, &a_sha]);
    assert_eq!(s, 422);
    assert!(String::from_utf8_lossy(&mm).contains("AXIS_MISMATCH"));
    server.shutdown();
}

/// KD-16 — the witness for a composed κ is retrievable via its witness-of edge, carries the §3.6
/// self-describing header, and its trace replays (re-hashes) to the composed κ (§3.6, §6.8).
#[test]
fn kd16_witness_retrieval_and_replay() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    let a = put_blob(&addr, "kd16ns", b"witness operand");
    let (_, _, body) = compose(&addr, "kd16ns", "e8", &[&a]);
    let composed = json_get(&String::from_utf8_lossy(&body), "composed").unwrap();

    // Retrieve the witness blob for the composed κ.
    let (s, h, wbytes) = request(&addr, "GET", &format!("/v2/kd16ns/witnesses/{composed}"), b"");
    assert_eq!(s, 200);
    assert!(header(&h, "X-Kappa-Label").unwrap().starts_with("blake3:"));

    // §3.6 self-describing header: label_width=71, fingerprint_width=32 (u16 LE).
    assert_eq!(&wbytes[0..2], &71u16.to_le_bytes());
    assert_eq!(&wbytes[2..4], &32u16.to_le_bytes());

    // Replay: re-hashing the witness trace (bytes past the 6-byte header) re-derives the composed κ.
    let trace = &wbytes[6..];
    assert_eq!(address_bytes(trace).as_str(), composed, "witness replays to the composed κ");
    server.shutdown();
}

/// KD-17 — a schema registers/retrieves as a content-addressed blob, and is **admission-only**: the
/// same content addresses to the same κ whether or not a schema is registered for its scope (§3.7,
/// §6.10).
#[test]
fn kd17_schema_register_retrieve_and_admission_only() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    // Register a schema at a scope; it is content-addressed.
    let schema = br#"{"type":"object","required":["scope"]}"#;
    let (s, h, _) = request(&addr, "PUT", "/v2/kd17ns/schemas/datasets", schema);
    assert_eq!(s, 201);
    assert_eq!(header(&h, "X-Kappa-Label"), Some(address_bytes(schema).as_str()));

    // Retrieve it back, byte-exact.
    let (s, _, got) = request(&addr, "GET", "/v2/kd17ns/schemas/datasets", b"");
    assert_eq!(s, 200);
    assert_eq!(got, schema);

    // Admission-only: content addresses identically with a schema (scoped path) and without one.
    let content = b"a dataset record";
    let k = address_bytes(content);
    let (s1, h1, _) = request(&addr, "PUT", &format!("/v2/kd17ns/blobs/{}", k.as_str()), content);
    let (s2, h2, _) = request(&addr, "PUT", &format!("/v2/other/blobs/{}", k.as_str()), content);
    assert!(s1 == 201 || s1 == 200);
    assert!(s2 == 201 || s2 == 200);
    assert_eq!(header(&h1, "X-Kappa-Label"), header(&h2, "X-Kappa-Label"));
    assert_eq!(header(&h1, "X-Kappa-Label"), Some(k.as_str()), "schema does not alter addressing");
    server.shutdown();
}
