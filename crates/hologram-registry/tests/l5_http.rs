//! KD-18/19/20/21 — the Level 5 surface (GC + finalizers, admission filters, federation) over the
//! registry server. Black-box over sockets.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use hologram_registry::federation::federate_fetch;
use hologram_registry::filter::admit;
use hologram_registry::server::serve;
use hologram_space::{address_bytes, KappaStore};
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

fn json_int(body: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{key}\"");
    let after = &body[body.find(&pat)? + pat.len()..];
    let after = &after[after.find(':')? + 1..];
    let num: String = after
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num.parse().ok()
}

fn put_blob(addr: &str, path: &str, content: &[u8]) -> String {
    let k = address_bytes(content);
    let (s, _, _) = request(addr, "PUT", &format!("/v2/{path}/blobs/{}", k.as_str()), content);
    assert!(s == 201 || s == 200);
    k.as_str().to_string()
}

fn put_edge(addr: &str, path: &str, src: &str, rel: &str, tgt: &str) {
    let body = format!(r#"{{"source":"{src}","relation":"{rel}","target":"{tgt}"}}"#);
    let (s, _, _) = request(addr, "PUT", &format!("/v2/{path}/edges/"), body.as_bytes());
    assert!(s == 201 || s == 200);
}

/// KD-18 — GC pin + sweep + status: a pinned blob (and what it `owns`) is reachable and retained; an
/// unpinned, unreachable blob is reported evictable (§6.11, §9).
#[test]
fn kd18_gc_pin_sweep_reachability_status() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    let a = put_blob(&addr, "kd18ns", b"root a");
    let b = address_bytes(b"owned b").as_str().to_string(); // absent target, tolerated
    let c = put_blob(&addr, "kd18ns", b"orphan c");
    let d = address_bytes(b"owned d").as_str().to_string();
    put_edge(&addr, "kd18ns", &a, "owns", &b); // a --owns--> b (reachable from a)
    put_edge(&addr, "kd18ns", &c, "owns", &d); // c --owns--> d (c is not pinned)

    // Pin a as a root.
    let pin_body = format!(r#"{{"kappa":"{a}","ttl":"0","controller":""}}"#);
    assert_eq!(request(&addr, "POST", "/v2/kd18ns/gc/pin", pin_body.as_bytes()).0, 201);

    // Sweep, then read status.
    assert_eq!(request(&addr, "POST", "/v2/kd18ns/gc/sweep", b"").0, 202);
    let (s, _, status) = request(&addr, "GET", "/v2/kd18ns/gc/status", b"");
    assert_eq!(s, 200);
    let st = String::from_utf8_lossy(&status);
    assert_eq!(json_int(&st, "objects_scanned"), Some(4), "{st}");
    assert_eq!(json_int(&st, "objects_reachable"), Some(2), "a + owned b");
    assert_eq!(json_int(&st, "objects_evicted"), Some(2), "orphan c + owned d");
    server.shutdown();
}

/// KD-19 — a finalizer (a pin with a controller) blocks unpin with 409 FINALIZER_OUTSTANDING until it
/// is released (§3.8, §6.11).
#[test]
fn kd19_finalizer_blocks_unpin_until_released() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    let x = put_blob(&addr, "kd19ns", b"finalized blob");
    let pin_body = format!(r#"{{"kappa":"{x}","ttl":"0","controller":"reaper"}}"#);
    let (s, h, _) = request(&addr, "POST", "/v2/kd19ns/gc/pin", pin_body.as_bytes());
    assert_eq!(s, 201);
    let pin_kappa = header(&h, "X-Kappa-Label").unwrap().to_string();

    // Normal unpin is blocked by the outstanding finalizer.
    let unpin = format!(r#"{{"pin_kappa":"{pin_kappa}"}}"#);
    let (s, _, body) = request(&addr, "POST", "/v2/kd19ns/gc/unpin", unpin.as_bytes());
    assert_eq!(s, 409);
    assert!(String::from_utf8_lossy(&body).contains("FINALIZER_OUTSTANDING"));

    // Releasing the finalizer clears it.
    let release = format!(r#"{{"pin_kappa":"{pin_kappa}","release":"true"}}"#);
    assert_eq!(request(&addr, "POST", "/v2/kd19ns/gc/unpin", release.as_bytes()).0, 200);
    server.shutdown();
}

/// KD-20 — admission filters: register/list/remove a filter blob, and the enforcement primitive is
/// all-accept + fail-closed (a `deny:` filter rejects matching content) (§6.12, §10).
#[test]
fn kd20_admission_filters_register_list_remove_and_fail_closed() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    // Register a deny-filter at a scope.
    let (s, h, _) = request(&addr, "PUT", "/v2/kd20ns/filters/datasets", b"deny:SECRET");
    assert_eq!(s, 201);
    let fk = header(&h, "X-Kappa-Label").unwrap().to_string();

    // List shows it.
    let (_, _, list) = request(&addr, "GET", "/v2/kd20ns/filters/", b"");
    let ls = String::from_utf8_lossy(&list);
    assert!(ls.contains("datasets") && ls.contains(&fk));

    // Enforcement: all-accept + fail-closed — denied content is rejected, other content admitted.
    assert!(admit("kd20ns", b"this has SECRET in it").is_err());
    assert!(admit("kd20ns", b"perfectly clean").is_ok());

    // Remove (unlink for audit); it no longer lists.
    assert_eq!(request(&addr, "DELETE", &format!("/v2/kd20ns/filters/{fk}"), b"").0, 202);
    let (_, _, list2) = request(&addr, "GET", "/v2/kd20ns/filters/", b"");
    assert!(!String::from_utf8_lossy(&list2).contains(&fk));
    server.shutdown();
}

/// KD-21 — federation fetch with verify-on-receipt: a blob absent from registry A is fetched from
/// peer registry B, re-hashed against the requested κ, and cached; an absent κ yields no content
/// (§8.3, §11.6).
#[test]
fn kd21_federation_fetch_verifies_on_receipt() {
    // Peer B holds the content and serves /v2/.
    let store_b = Arc::new(MemKappaStore::new());
    let k = store_b.put("blake3", b"content that lives on peer B").unwrap();
    let server_b = serve(store_b).unwrap();
    let peer = server_b.addr().to_string();

    // Registry A starts empty; federate the κ from B, verifying on receipt.
    let a_store = MemKappaStore::new();
    assert!(!a_store.contains(&k));
    assert!(federate_fetch(&peer, "fedns", &k, &a_store), "verified fetch from peer");
    assert!(a_store.contains(&k), "federated blob cached locally after verification");

    // A κ that B does not hold yields no content (404 → false), nothing cached.
    let absent = address_bytes(b"nowhere in the federation");
    assert!(!federate_fetch(&peer, "fedns", &absent, &a_store));
    assert!(!a_store.contains(&absent));
    server_b.shutdown();
}

/// KD-22 — a registered admission filter is **enforced on the blob-write path**: a `PUT` of denied
/// content is rejected `422 FILTER_REJECTED` and stores nothing; clean content is admitted (§5.1,
/// §10). Closes the enforcement seam — the registry runs `filter::admit` over the delegated L1 write.
#[test]
fn kd22_admission_filter_enforced_on_blob_write() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    // Register a deny filter for the path.
    assert_eq!(request(&addr, "PUT", "/v2/kd22ns/filters/all", b"deny:FORBIDDEN").0, 201);

    // A blob whose content matches the deny rule is rejected on the write path — nothing is stored.
    let denied = b"this content is FORBIDDEN here";
    let dk = address_bytes(denied);
    let (s, _, body) = request(&addr, "PUT", &format!("/v2/kd22ns/blobs/{}", dk.as_str()), denied);
    assert_eq!(s, 422);
    assert!(String::from_utf8_lossy(&body).contains("FILTER_REJECTED"));
    assert_eq!(
        request(&addr, "GET", &format!("/v2/kd22ns/blobs/{}", dk.as_str()), b"").0,
        404,
        "filter-rejected content never enters the store"
    );

    // Clean content is admitted normally.
    let clean = b"perfectly acceptable content";
    let ck = address_bytes(clean);
    assert_eq!(
        request(&addr, "PUT", &format!("/v2/kd22ns/blobs/{}", ck.as_str()), clean).0,
        201
    );
    server.shutdown();
}

/// KD-23 — a tagged blob is a GC root: with no pin, a tag keeps its blob reachable through a sweep,
/// while an unreferenced orphan chain is evictable (§9.1). Closes the tags-as-roots seam across the
/// L1/L2 delegation boundary.
#[test]
fn kd23_tagged_blob_is_a_gc_root() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    // Tag a blob (L2 bind) — no pin.
    let (s, _, _) = request(&addr, "PUT", "/v2/kd23ns/manifests/latest", b"kept alive by a tag");
    assert_eq!(s, 201);

    // An orphan chain: an unpinned, untagged source owning a target.
    let orphan_src = put_blob(&addr, "kd23ns", b"orphan source");
    let orphan_tgt = address_bytes(b"orphan target").as_str().to_string();
    put_edge(&addr, "kd23ns", &orphan_src, "owns", &orphan_tgt);

    // Sweep with NO pins: the tagged blob is a reachable root; the orphan chain is evictable.
    assert_eq!(request(&addr, "POST", "/v2/kd23ns/gc/sweep", b"").0, 202);
    let (_, _, status) = request(&addr, "GET", "/v2/kd23ns/gc/status", b"");
    let st = String::from_utf8_lossy(&status);
    assert!(st.contains(r#""objects_reachable":1"#), "tag roots its blob: {st}");
    assert!(st.contains(r#""objects_evicted":2"#), "orphan chain evictable: {st}");
    server.shutdown();
}
