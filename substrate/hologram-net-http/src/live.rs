//! Live HTTP/1.1 CAS transport over `std::net` — no async runtime, no heavy HTTP crate. The
//! server answers `GET /cas/{kappa}` from a [`KappaStore`]; the client ([`HttpKappaSync`]) is a
//! [`KappaSync`] that fetches from configured gateways and **verifies every byte on receipt**
//! (SPINE-4 / §6.4). This is the verifiable realization of the §6.3 protocol; the uor-native TCP
//! transport (`hologram-net-tcp`) layers κ-XOR Kademlia content discovery above it.

extern crate std;

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use hologram_space::{Bytes, KappaLabel, KappaLabel71, KappaStore, KappaSync, SyncError};

use crate::{accept_received, cas_path, serve_get, CasResponse};

/// A running CAS gateway server bound to an ephemeral localhost port. Drop or call [`shutdown`] to
/// stop it.
pub struct CasServer {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl CasServer {
    /// The bound address (use as a gateway URL/host:port for the client).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
    pub fn shutdown(mut self) {
        self.stop();
    }
    fn stop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Nudge the accept loop with a self-connection so it observes the flag promptly.
        let _ = TcpStream::connect(self.addr);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for CasServer {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.stop();
        }
    }
}

/// Serve `GET /cas/{kappa}` from `store` on an ephemeral localhost port (tests). If `forge` is set,
/// the server returns garbage bodies for any request — used to prove the client's verify-on-receipt
/// rejects a malicious gateway.
pub fn serve(store: Arc<dyn KappaStore>, forge: bool) -> std::io::Result<CasServer> {
    serve_addr(store, "127.0.0.1:0", forge)
}

/// Serve `GET /cas/{kappa}` from `store` on a specified `addr` (e.g. `0.0.0.0:8080` for a node, spec
/// §6.5).
pub fn serve_addr(
    store: Arc<dyn KappaStore>,
    addr: &str,
    forge: bool,
) -> std::io::Result<CasServer> {
    let listener = TcpListener::bind(addr)?;
    let addr = listener.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    let handle = std::thread::spawn(move || {
        for conn in listener.incoming() {
            if flag.load(Ordering::SeqCst) {
                break;
            }
            if let Ok(mut stream) = conn {
                let _ = handle_conn(&mut stream, store.as_ref(), forge);
            }
        }
    });
    Ok(CasServer {
        addr,
        shutdown,
        handle: Some(handle),
    })
}

/// Cap on request-header bytes — a DoS guard, not a functional limit on κ paths (which are tiny).
const MAX_HEADER_BYTES: usize = 64 * 1024;

fn handle_conn(stream: &mut TcpStream, store: &dyn KappaStore, forge: bool) -> std::io::Result<()> {
    // Read until the end of the request headers (`\r\n\r\n`), not a single fixed-size read — a
    // request line can be any length and may arrive fragmented.
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > MAX_HEADER_BYTES {
            break;
        }
    }
    let req = String::from_utf8_lossy(&buf);
    let path = req.split_whitespace().nth(1).unwrap_or("");
    if forge {
        return write_response(stream, 200, b"forged-content");
    }
    // Discovery extension (spec §6.3): GET /cas/?prefix=<p>&limit=<n> → newline-separated κ-labels.
    if path.starts_with("/cas/?") || path == "/cas/" {
        let body = serve_discover(store, path);
        return write_response(stream, 200, body.as_bytes());
    }
    match serve_get(store, path) {
        CasResponse::Ok(body) => write_response(stream, 200, body.as_ref()),
        CasResponse::NotFound => write_response(stream, 404, b""),
        CasResponse::BadRequest => write_response(stream, 400, b""),
    }
}

/// Discovery: the locally-present κ-labels whose `<axis>:<hex>` form starts with `prefix`, up to
/// `limit`, newline-separated.
fn serve_discover(store: &dyn KappaStore, path: &str) -> String {
    let query = path.strip_prefix("/cas/?").unwrap_or("");
    let mut prefix = "";
    let mut limit = 64usize;
    for kv in query.split('&') {
        if let Some(p) = kv.strip_prefix("prefix=") {
            prefix = p;
        } else if let Some(l) = kv.strip_prefix("limit=") {
            limit = l.parse().unwrap_or(64);
        }
    }
    let mut out = String::new();
    for k in store
        .iterate()
        .into_iter()
        .filter(|k| k.as_str().starts_with(prefix))
        .take(limit)
    {
        out.push_str(k.as_str());
        out.push('\n');
    }
    out
}

fn write_response(stream: &mut TcpStream, status: u16, body: &[u8]) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "Bad Request",
    };
    let head = std::format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nCache-Control: public, immutable, max-age=31536000\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// A [`KappaSync`] that fetches κ-labels from configured HTTP-CAS gateways (`host:port`), verifying
/// each response by σ-axis re-derivation before returning it (SPINE-4).
#[derive(Default)]
pub struct HttpKappaSync {
    gateways: spin_lock::Mutex<Vec<String>>,
}

// Tiny std Mutex shim so this stays std-only without pulling spin into net-http's deps.
mod spin_lock {
    pub use std::sync::Mutex;
}

impl HttpKappaSync {
    pub fn new(gateways: Vec<String>) -> Self {
        Self {
            gateways: std::sync::Mutex::new(gateways),
        }
    }

    fn fetch_one(gateway: &str, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        let mut stream =
            TcpStream::connect(gateway).map_err(|_| SyncError::BackendFailure("connect"))?;
        let req = std::format!(
            "GET {} HTTP/1.1\r\nHost: {gateway}\r\nConnection: close\r\n\r\n",
            cas_path(kappa)
        );
        stream
            .write_all(req.as_bytes())
            .map_err(|_| SyncError::BackendFailure("write"))?;
        let mut resp = Vec::new();
        stream
            .read_to_end(&mut resp)
            .map_err(|_| SyncError::BackendFailure("read"))?;
        let split =
            find_subslice(&resp, b"\r\n\r\n").ok_or(SyncError::BackendFailure("no headers"))?;
        let head = &resp[..split];
        let body = &resp[split + 4..];
        let status = parse_status(head).ok_or(SyncError::BackendFailure("bad status"))?;
        match status {
            200 => match accept_received(kappa, body) {
                Ok(b) => Ok(Some(b)),
                Err(_) => Err(SyncError::VerificationFailed), // forged gateway rejected (§6.4)
            },
            404 => Ok(None),
            _ => Err(SyncError::BackendFailure("http status")),
        }
    }
}

impl HttpKappaSync {
    /// Query one gateway's discovery endpoint; `None` on transport error.
    fn discover_one(gateway: &str, prefix: &str, limit: usize) -> Option<Vec<KappaLabel71>> {
        let mut stream = TcpStream::connect(gateway).ok()?;
        let req = std::format!(
            "GET /cas/?prefix={prefix}&limit={limit} HTTP/1.1\r\nHost: {gateway}\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(req.as_bytes()).ok()?;
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).ok()?;
        let split = find_subslice(&resp, b"\r\n\r\n")?;
        let body = core::str::from_utf8(&resp[split + 4..]).ok()?;
        let mut out = Vec::new();
        for line in body.lines() {
            if let Ok(arr) = <[u8; 71]>::try_from(line.as_bytes()) {
                if let Ok(k) = KappaLabel::from_bytes(&arr) {
                    out.push(k);
                }
            }
        }
        Some(out)
    }
}

fn parse_status(head: &[u8]) -> Option<u16> {
    let line = core::str::from_utf8(head).ok()?.lines().next()?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[async_trait::async_trait]
impl KappaSync for HttpKappaSync {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        let gateways = self.gateways.lock().unwrap().clone();
        if gateways.is_empty() {
            return Err(SyncError::NotEnabled);
        }
        let mut last = SyncError::AllSourcesFailed;
        for g in gateways {
            match Self::fetch_one(&g, kappa) {
                Ok(Some(b)) => return Ok(Some(b)),
                Ok(None) => last = SyncError::AllSourcesFailed,
                Err(e) => last = e, // try the next source; a forged source is skipped, not trusted
            }
        }
        // If the only outcome was "not present anywhere", that's Ok(None) per the trait.
        if matches!(last, SyncError::AllSourcesFailed) {
            Ok(None)
        } else {
            Err(last)
        }
    }
    async fn announce(&self, _kappa: &KappaLabel71) {
        // HTTP-CAS is **pull-only by design** (spec §6.3): a peer becomes discoverable when
        // remotes GET its `/cas/?prefix=` endpoint, not via a push announcement. The DHT-style
        // `Provide` push lives on the uor-native TCP transport (`hologram-net-tcp`, DHT class).
        // This is the architected behavior — not a stub: announce on HTTP-CAS has no wire effect
        // because the protocol has no announcement wire.
    }
    async fn discover(&self, prefix: Option<&[u8]>, limit: usize) -> Vec<KappaLabel71> {
        let prefix_str = prefix
            .and_then(|p| core::str::from_utf8(p).ok())
            .unwrap_or("");
        let gateways = self.gateways.lock().unwrap().clone();
        let mut seen: std::collections::HashSet<[u8; 71]> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for g in gateways {
            for label in Self::discover_one(&g, prefix_str, limit)
                .into_iter()
                .flatten()
            {
                if out.len() >= limit {
                    return out;
                }
                if seen.insert(*label.as_array()) {
                    out.push(label);
                }
            }
        }
        out
    }
    async fn add_peer(&self, _addr: &str) -> Result<(), SyncError> {
        // HTTP-CAS doesn't consume `host:port` peers; `FederatedKappaSync` routes those to the
        // `hologram-net-tcp` backend (arch §11.2). Returning Err here makes the routing
        // unambiguous — peers go to TCP, gateways come here.
        Err(SyncError::BackendFailure(
            "http does not consume host:port peers — use FederatedKappaSync",
        ))
    }
    async fn add_gateway(&self, url: &str) -> Result<(), SyncError> {
        self.gateways.lock().unwrap().push(url.to_string());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_space::{address_bytes, get_with_fetch};

    #[test]
    fn live_fetch_roundtrips_and_verifies_then_caches() {
        pollster::block_on(async {
            let origin = Arc::new(MemKappaStore::new());
            let k = origin.put("blake3", b"served-over-the-wire").unwrap();
            let server = serve(origin.clone(), false).unwrap();

            let sync = HttpKappaSync::new(std::vec![server.addr().to_string()]);
            // Eviction-tolerant read: local miss → live HTTP fetch → verify → cache.
            let local = MemKappaStore::new();
            let got = get_with_fetch(&local, &sync, &k).await.unwrap();
            assert_eq!(got.unwrap().as_ref(), b"served-over-the-wire");
            assert!(local.contains(&k), "verified bytes cached locally");

            // Absent κ → 404 → Ok(None).
            let absent = hologram_space::address_bytes(b"absent");
            assert_eq!(sync.fetch(&absent).await.unwrap(), None);
            server.shutdown();
        });
    }

    #[test]
    fn live_fetch_rejects_a_forging_gateway() {
        pollster::block_on(async {
            let origin = Arc::new(MemKappaStore::new());
            let k = origin.put("blake3", b"authentic").unwrap();
            let evil = serve(origin, true).unwrap(); // serves garbage for every request

            let sync = HttpKappaSync::new(std::vec![evil.addr().to_string()]);
            assert_eq!(sync.fetch(&k).await, Err(SyncError::VerificationFailed));
            evil.shutdown();
        });
    }

    #[test]
    fn multi_node_fetch_fallback_and_cross_node_discovery() {
        pollster::block_on(async {
            // Three real nodes over TCP. B holds κ1; C holds κ2; A holds neither.
            let store_b = Arc::new(MemKappaStore::new());
            let store_c = Arc::new(MemKappaStore::new());
            let k1 = store_b.put("blake3", b"lives-on-node-B").unwrap();
            let k2 = store_c.put("blake3", b"lives-on-node-C").unwrap();
            let node_b = serve(store_b, false).unwrap();
            let node_c = serve(store_c, false).unwrap();

            // Node A knows both peers (C first so we exercise the 404→next-peer fallback for k1).
            let a = HttpKappaSync::new(std::vec![
                node_c.addr().to_string(),
                node_b.addr().to_string()
            ]);

            // k1: C 404s → falls through to B → verified bytes.
            assert_eq!(
                a.fetch(&k1).await.unwrap().unwrap().as_ref(),
                b"lives-on-node-B"
            );
            // k2: served by C, verified.
            assert_eq!(
                a.fetch(&k2).await.unwrap().unwrap().as_ref(),
                b"lives-on-node-C"
            );
            // A κ no peer has → Ok(None).
            assert_eq!(a.fetch(&address_bytes(b"nowhere")).await.unwrap(), None);

            // Cross-node discovery: merge what B and C advertise (deduped).
            let found = a.discover(Some(b"blake3:"), 64).await;
            assert!(
                found.contains(&k1) && found.contains(&k2),
                "discovery merges peers' κ-labels"
            );
            node_b.shutdown();
            node_c.shutdown();
        });
    }

    use hologram_store_mem::MemKappaStore;
}
