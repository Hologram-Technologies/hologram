//! Live HTTP/1.1 CAS transport over `std::net` — no async runtime, no heavy HTTP crate. The
//! server ([`serve_with`]) answers `GET`/`HEAD /cas/{kappa}` from a [`KappaStore`] with ranges,
//! ETags, CORS and keep-alive, one thread per connection; the client ([`HttpKappaSync`]) is a
//! [`KappaSync`] that fetches from configured gateways and **verifies every byte on receipt**
//! (SPINE-4 / §6.4). This is the verifiable realization of the §6.3 protocol; the uor-native TCP
//! transport (`hologram-net-tcp`) layers κ-XOR Kademlia content discovery above it.

extern crate std;

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Duration;

use hologram_space::{
    Bytes, KappaLabel, KappaLabel71, KappaStore, KappaSync, PieceTreeBuilder, SyncError,
};

#[cfg(doc)]
use hologram_space::PieceTree;

use super::{accept_received, cas_path, parse_cas_resource, parse_range, CasResource, RangeSpec};

/// Server settings. [`Default`] is what [`serve`] and [`serve_addr`] use.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// Answer every request with a garbage body (tests: proves clients verify on receipt).
    pub forge: bool,
    /// Connections served at once; one more is answered `503` and closed.
    pub max_connections: usize,
    /// How long an idle keep-alive connection is held open.
    pub idle_timeout: Duration,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            forge: false,
            max_connections: 256,
            idle_timeout: Duration::from_secs(30),
        }
    }
}

/// A running CAS gateway. Drop or call `shutdown` to stop accepting; connections already open
/// finish their current request and close.
pub struct CasServer {
    addr: SocketAddr,
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

impl CasServer {
    /// The bound address (use as a gateway URL/host:port for the client).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
    /// Connections being served right now.
    pub fn active_connections(&self) -> usize {
        self.shared.active.load(Ordering::SeqCst)
    }
    pub fn shutdown(mut self) {
        self.stop();
    }
    fn stop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
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

struct Shared {
    store: Arc<dyn KappaStore>,
    opts: ServeOptions,
    active: AtomicUsize,
    shutdown: AtomicBool,
    /// Encoded piece tables already computed, by κ (32 bytes per MiB of content).
    trees: Mutex<HashMap<[u8; 71], Arc<Vec<u8>>>>,
}

/// Serve the HTTP-CAS protocol from `store` on an ephemeral localhost port (tests). If `forge` is
/// set, the server returns garbage bodies for any request — used to prove the client's
/// verify-on-receipt rejects a malicious gateway.
pub fn serve(store: Arc<dyn KappaStore>, forge: bool) -> std::io::Result<CasServer> {
    serve_addr(store, "127.0.0.1:0", forge)
}

/// Serve on a specified `addr` (e.g. `0.0.0.0:8080` for a node, spec §6.5).
pub fn serve_addr(
    store: Arc<dyn KappaStore>,
    addr: &str,
    forge: bool,
) -> std::io::Result<CasServer> {
    serve_with(
        store,
        addr,
        ServeOptions {
            forge,
            ..ServeOptions::default()
        },
    )
}

/// Serve on `addr` with explicit [`ServeOptions`].
///
/// Routes (all answer `GET`, `HEAD` and the CORS preflight `OPTIONS`):
/// - `/cas/{kappa}`: the content. `ETag` is the quoted κ (`If-None-Match` gives 304), one
///   `Range` gives 206 or 416, `Accept-Ranges: bytes`, immutable caching. Bodies are streamed from
///   the store 1 MiB at a time, so a multi-gigabyte blob is never held in memory.
/// - `/cas/{kappa}/pieces`: the content's 1 MiB [`PieceTree`] table, which hashes to the κ, so a
///   client can verify each range it fetches.
/// - `/cas/?prefix=&limit=`: discovery (newline-separated κ-labels).
///
/// Each connection runs on its own thread with HTTP/1.1 keep-alive, up to
/// [`ServeOptions::max_connections`] at once.
pub fn serve_with(
    store: Arc<dyn KappaStore>,
    addr: &str,
    opts: ServeOptions,
) -> std::io::Result<CasServer> {
    let listener = TcpListener::bind(addr)?;
    let addr = listener.local_addr()?;
    let shared = Arc::new(Shared {
        store,
        opts,
        active: AtomicUsize::new(0),
        shutdown: AtomicBool::new(false),
        trees: Mutex::new(HashMap::new()),
    });
    let accept = shared.clone();
    let handle = std::thread::spawn(move || {
        for conn in listener.incoming() {
            if accept.shutdown.load(Ordering::SeqCst) {
                break;
            }
            let Ok(mut stream) = conn else { continue };
            if accept.active.fetch_add(1, Ordering::SeqCst) >= accept.opts.max_connections {
                accept.active.fetch_sub(1, Ordering::SeqCst);
                let _ = write_head(&mut stream, 503, &[], 0, false);
                continue;
            }
            let shared = accept.clone();
            std::thread::spawn(move || {
                let _ = serve_connection(stream, &shared);
                shared.active.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    Ok(CasServer {
        addr,
        shared,
        handle: Some(handle),
    })
}

/// Cap on request-header bytes — a DoS guard, not a functional limit on κ paths (which are tiny).
const MAX_HEADER_BYTES: usize = 64 * 1024;

/// Body bytes read from the store per write to the socket.
const BODY_WINDOW: u64 = 1 << 20;

/// CORS headers on every response: any origin may read content-addressed bytes (they are public
/// by κ and verified by the reader), and the range headers are exposed to scripts.
const CORS: &[(&str, &str)] = &[
    ("Access-Control-Allow-Origin", "*"),
    (
        "Access-Control-Expose-Headers",
        "Content-Length, Content-Range, Accept-Ranges, ETag",
    ),
];

struct Request {
    method: String,
    path: String,
    keep_alive: bool,
    headers: Vec<(String, String)>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Read one request head. `Ok(None)` on a clean close between requests.
fn read_request(conn: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    // Tolerate stray CRLFs between pipelined requests (RFC 9112 §2.2).
    loop {
        line.clear();
        if conn.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        if !line.trim().is_empty() {
            break;
        }
    }
    let mut total = line.len();
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let version = parts.next().unwrap_or("");
    let mut keep_alive = version == "HTTP/1.1";
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if conn.read_line(&mut h)? == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        total += h.len();
        if total > MAX_HEADER_BYTES {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        let h = h.trim_end_matches(['\r', '\n']);
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            let k = k.trim().to_ascii_lowercase();
            let v = v.trim().to_string();
            if k == "connection" {
                if v.eq_ignore_ascii_case("close") {
                    keep_alive = false;
                } else if v.eq_ignore_ascii_case("keep-alive") {
                    keep_alive = true;
                }
            }
            headers.push((k, v));
        }
    }
    Ok(Some(Request {
        method,
        path,
        keep_alive,
        headers,
    }))
}

fn serve_connection(stream: TcpStream, shared: &Shared) -> std::io::Result<()> {
    stream.set_read_timeout(Some(shared.opts.idle_timeout))?;
    stream.set_nodelay(true)?;
    let mut out = stream.try_clone()?;
    let mut conn = BufReader::new(stream);
    while let Some(req) = read_request(&mut conn)? {
        // A request with a body is not part of this protocol; answer and close rather than
        // guess where the next request starts.
        let has_body = req.header("transfer-encoding").is_some()
            || req
                .header("content-length")
                .is_some_and(|v| v.trim() != "0");
        let keep = req.keep_alive && !has_body && !shared.shutdown.load(Ordering::SeqCst);
        let ok = respond(&mut out, shared, &req, keep)?;
        if !keep || !ok {
            break;
        }
    }
    Ok(())
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        206 => "Partial Content",
        304 => "Not Modified",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        416 => "Range Not Satisfiable",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

fn write_head(
    out: &mut TcpStream,
    status: u16,
    headers: &[(&str, &str)],
    content_length: u64,
    keep_alive: bool,
) -> std::io::Result<()> {
    let mut head = std::format!("HTTP/1.1 {status} {}\r\n", reason(status));
    for (k, v) in CORS.iter().chain(headers) {
        head.push_str(k);
        head.push_str(": ");
        head.push_str(v);
        head.push_str("\r\n");
    }
    if status != 304 && status != 204 {
        head.push_str(&std::format!("Content-Length: {content_length}\r\n"));
    }
    head.push_str(if keep_alive {
        "Connection: keep-alive\r\n\r\n"
    } else {
        "Connection: close\r\n\r\n"
    });
    out.write_all(head.as_bytes())
}

/// Answer one request. Returns `Ok(false)` when the connection must close (a body could not be
/// completed after its head was sent).
fn respond(
    out: &mut TcpStream,
    shared: &Shared,
    req: &Request,
    keep: bool,
) -> std::io::Result<bool> {
    let head_only = req.method == "HEAD";
    match req.method.as_str() {
        "GET" | "HEAD" => {}
        "OPTIONS" => {
            write_head(
                out,
                204,
                &[
                    ("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS"),
                    (
                        "Access-Control-Allow-Headers",
                        "Range, If-None-Match, If-Range",
                    ),
                    ("Access-Control-Max-Age", "86400"),
                ],
                0,
                keep,
            )?;
            return Ok(true);
        }
        _ => {
            write_head(out, 405, &[("Allow", "GET, HEAD, OPTIONS")], 0, keep)?;
            return Ok(true);
        }
    }
    if shared.opts.forge {
        let body: &[u8] = b"forged-content";
        write_head(out, 200, &[], body.len() as u64, keep)?;
        if !head_only {
            out.write_all(body)?;
        }
        return Ok(true);
    }
    let store = shared.store.as_ref();
    match parse_cas_resource(&req.path) {
        None => write_head(out, 400, &[], 0, keep)?,
        Some(CasResource::Discover) => {
            let body = serve_discover(store, &req.path);
            write_head(
                out,
                200,
                &[("Content-Type", "text/plain; charset=utf-8")],
                body.len() as u64,
                keep,
            )?;
            if !head_only {
                out.write_all(body.as_bytes())?;
            }
        }
        Some(CasResource::Pieces(kappa)) => match piece_table(shared, &kappa) {
            Ok(Some(table)) => {
                let etag = std::format!("\"{}/pieces\"", kappa.as_str());
                write_head(
                    out,
                    200,
                    &[
                        ("Content-Type", "application/vnd.hologram.piece-tree"),
                        ("ETag", &etag),
                        ("Cache-Control", "public, immutable, max-age=31536000"),
                    ],
                    table.len() as u64,
                    keep,
                )?;
                if !head_only {
                    out.write_all(&table)?;
                }
            }
            Ok(None) => write_head(out, 404, &[], 0, keep)?,
            // The stored bytes no longer hash to their κ: never vouch for them.
            Err(()) => write_head(out, 500, &[], 0, keep)?,
        },
        Some(CasResource::Blob(kappa)) => return serve_blob(out, store, req, &kappa, keep),
    }
    Ok(true)
}

fn serve_blob(
    out: &mut TcpStream,
    store: &dyn KappaStore,
    req: &Request,
    kappa: &KappaLabel71,
    keep: bool,
) -> std::io::Result<bool> {
    let len = match store.content_len(kappa) {
        Ok(Some(len)) => len,
        Ok(None) => {
            write_head(out, 404, &[], 0, keep)?;
            return Ok(true);
        }
        Err(_) => {
            write_head(out, 500, &[], 0, keep)?;
            return Ok(true);
        }
    };
    let etag = std::format!("\"{}\"", kappa.as_str());
    let common = [
        ("Content-Type", "application/octet-stream"),
        ("ETag", etag.as_str()),
        ("Accept-Ranges", "bytes"),
        ("Cache-Control", "public, immutable, max-age=31536000"),
    ];
    if req
        .header("if-none-match")
        .is_some_and(|v| v.split(',').any(|t| t.trim() == etag || t.trim() == "*"))
    {
        write_head(out, 304, &common, 0, keep)?;
        return Ok(true);
    }
    // `If-Range` with another validator means "send it all": the client's partial copy is stale.
    let range_header = match req.header("if-range") {
        Some(v) if v.trim() != etag => None,
        _ => req.header("range"),
    };
    let (status, start, end) = match parse_range(range_header, len) {
        RangeSpec::Full => (200, 0, len),
        RangeSpec::Partial { start, end } => (206, start, end),
        RangeSpec::Unsatisfiable => {
            let cr = std::format!("bytes */{len}");
            let mut h = common.to_vec();
            h.push(("Content-Range", &cr));
            write_head(out, 416, &h, 0, keep)?;
            return Ok(true);
        }
    };
    let cr = std::format!("bytes {}-{}/{len}", start, end.saturating_sub(1));
    let mut h = common.to_vec();
    if status == 206 {
        h.push(("Content-Range", &cr));
    }
    write_head(out, status, &h, end - start, keep)?;
    if req.method == "HEAD" {
        return Ok(true);
    }
    let mut pos = start;
    while pos < end {
        let n = (end - pos).min(BODY_WINDOW);
        match store.get_range(kappa, pos, n) {
            Ok(Some(bytes)) => out.write_all(&bytes)?,
            // The content vanished or failed mid-body: the head promised more bytes, so the only
            // honest move is to close the connection (the client sees a short body).
            _ => return Ok(false),
        }
        pos += n;
    }
    Ok(true)
}

/// The encoded piece table for `kappa`, computed once by streaming the content through
/// [`PieceTreeBuilder`] and checked against the κ before it is ever served. `Err` if the stored
/// bytes do not hash to their κ.
#[allow(clippy::result_unit_err)]
fn piece_table(shared: &Shared, kappa: &KappaLabel71) -> Result<Option<Arc<Vec<u8>>>, ()> {
    if let Some(t) = shared.trees.lock().unwrap().get(kappa.as_array()) {
        return Ok(Some(t.clone()));
    }
    let store = shared.store.as_ref();
    let Some(len) = store.content_len(kappa).map_err(|_| ())? else {
        return Ok(None);
    };
    let mut builder = PieceTreeBuilder::new();
    let mut pos = 0;
    while pos < len {
        let n = (len - pos).min(BODY_WINDOW);
        let bytes = store.get_range(kappa, pos, n).map_err(|_| ())?.ok_or(())?;
        builder.update(&bytes);
        pos += n;
    }
    let tree = builder.finish();
    tree.verify_root(kappa).map_err(|_| ())?;
    let table = Arc::new(tree.encode());
    shared
        .trees
        .lock()
        .unwrap()
        .insert(*kappa.as_array(), table.clone());
    Ok(Some(table))
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

    use hologram_tck::MemKappaStore;
}
