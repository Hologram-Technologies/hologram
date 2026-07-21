//! KD-12/13 — the Level 3 edge HTTP surface (`/v2/{path}/edges`) over the registry server, which
//! **delegates Levels 1-2 blobs/tags to `hologram-net`** and adds L3 edges itself. Black-box: drive
//! the real `/v2/` server over a socket, creating source/target blobs via L1 and edges via L3.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use hologram_registry::server::serve;
use hologram_space::address_bytes;
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

fn put_edge(
    addr: &str,
    path: &str,
    src: &str,
    rel: &str,
    tgt: &str,
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let body = format!(r#"{{"source":"{src}","relation":"{rel}","target":"{tgt}"}}"#);
    request(addr, "PUT", &format!("/v2/{path}/edges/"), body.as_bytes())
}

fn get_edges(
    addr: &str,
    path: &str,
    node: &str,
    direction: &str,
    relation: Option<&str>,
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let mut p = format!("/v2/{path}/edges/{node}?direction={direction}");
    if let Some(r) = relation {
        p.push_str(&format!("&relation={r}"));
    }
    request(addr, "GET", &p, b"")
}

/// KD-12 — edge create verifies the source exists (`409 EDGE_SOURCE_ABSENT`), tolerates an absent
/// target, is idempotent (201→200), and the created edge is retrievable as a blob (§5.3, §6.6).
#[test]
fn kd12_edge_put_source_exists_absent_target_tolerated_idempotent() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    // Source blob via L1 (delegated to hologram-net).
    let src_content = b"edge source blob";
    let src = address_bytes(src_content);
    assert_eq!(
        request(&addr, "PUT", &format!("/v2/kd12ns/blobs/{}", src.as_str()), src_content).0,
        201
    );

    // Target we deliberately do NOT store (absent-target tolerance).
    let tgt = address_bytes(b"absent target blob");

    // Create: source present, target absent → 201, edge κ under the source's axis.
    let (s, h, _) = put_edge(&addr, "kd12ns", src.as_str(), "derives-from", tgt.as_str());
    assert_eq!(s, 201);
    let ek = header(&h, "X-Kappa-Label").unwrap().to_string();
    assert!(ek.starts_with("blake3:"), "edge κ inherits the source σ-axis");

    // Idempotent re-create → 200.
    assert_eq!(
        put_edge(&addr, "kd12ns", src.as_str(), "derives-from", tgt.as_str()).0,
        200
    );

    // The edge blob is retrievable via the L1 blob GET (edges are blobs).
    assert_eq!(request(&addr, "GET", &format!("/v2/kd12ns/blobs/{ek}"), b"").0, 200);

    // Absent source → 409 EDGE_SOURCE_ABSENT.
    let absent_src = address_bytes(b"nonexistent source");
    let (s, _, body) = put_edge(&addr, "kd12ns", absent_src.as_str(), "owns", tgt.as_str());
    assert_eq!(s, 409);
    assert!(String::from_utf8_lossy(&body).contains("EDGE_SOURCE_ABSENT"));
    server.shutdown();
}

/// KD-13 — the edge index answers queries by direction (outbound/inbound) and relation, and a delete
/// removes the edge from the index (§3.4, §6.6).
#[test]
fn kd13_edge_index_query_by_direction_relation_and_delete() {
    let store = Arc::new(MemKappaStore::new());
    let server = serve(store).unwrap();
    let addr = server.addr().to_string();

    let a = address_bytes(b"node a");
    let b = address_bytes(b"node b");
    let c = address_bytes(b"node c");
    for (k, content) in [
        (&a, &b"node a"[..]),
        (&b, &b"node b"[..]),
        (&c, &b"node c"[..]),
    ] {
        assert_eq!(
            request(&addr, "PUT", &format!("/v2/kd13ns/blobs/{}", k.as_str()), content).0,
            201
        );
    }

    // a --owns--> b ; a --derives-from--> c
    let (_, h1, _) = put_edge(&addr, "kd13ns", a.as_str(), "owns", b.as_str());
    let e_ab = header(&h1, "X-Kappa-Label").unwrap().to_string();
    put_edge(&addr, "kd13ns", a.as_str(), "derives-from", c.as_str());

    // Outbound from a → both targets b and c.
    let (_, _, out) = get_edges(&addr, "kd13ns", a.as_str(), "outbound", None);
    let outs = String::from_utf8_lossy(&out);
    assert!(outs.contains(b.as_str()) && outs.contains(c.as_str()));

    // Inbound to b → source a.
    let (_, _, inb) = get_edges(&addr, "kd13ns", b.as_str(), "inbound", None);
    assert!(String::from_utf8_lossy(&inb).contains(a.as_str()));

    // Relation filter: outbound from a, relation=owns → only b.
    let (_, _, filt) = get_edges(&addr, "kd13ns", a.as_str(), "outbound", Some("owns"));
    let filts = String::from_utf8_lossy(&filt);
    assert!(filts.contains(b.as_str()) && !filts.contains(c.as_str()));

    // Delete the a--owns-->b edge → gone from the index.
    assert_eq!(
        request(&addr, "DELETE", &format!("/v2/kd13ns/edges/{e_ab}"), b"").0,
        202
    );
    let (_, _, after) = get_edges(&addr, "kd13ns", a.as_str(), "outbound", Some("owns"));
    assert!(
        !String::from_utf8_lossy(&after).contains(b.as_str()),
        "deleted edge is no longer indexed"
    );
    server.shutdown();
}
