#![cfg(feature = "live")]
//! HTTP-CAS for large blobs: the server's HTTP surface (HEAD, Range, ETag, CORS, keep-alive,
//! concurrent connections, piece tables) and the parallel multi-source verified fetch against
//! honest, lying, dead and partial sources. Every lie is a planted defect the client must refuse.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use hologram_net::http::client::HttpConn;
use hologram_net::http::fetch::{
    fetch_piece_tree, fetch_pieces, fetch_to_vec, FetchError, FetchOptions,
};
use hologram_net::http::live::{serve, serve_with, ServeOptions};
use hologram_net::http::{cas_path, pieces_path};
use hologram_space::{
    address_bytes, Bytes, KappaLabel71, KappaStore, MemKappaStore, PieceSink, PieceTree,
    StoreError, PIECE_LEN,
};
use hologram_store::file::FileKappaStore;

fn blob(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

fn opts() -> FetchOptions {
    FetchOptions {
        concurrency: 6,
        max_failures: 2,
        timeout: Duration::from_secs(5),
    }
}

/// A store that serves one wrong byte at `flip_at` — a lying source.
struct Lying {
    inner: Arc<dyn KappaStore>,
    flip_at: u64,
}

impl KappaStore for Lying {
    fn put(&self, axis: &str, b: &[u8]) -> Result<KappaLabel71, StoreError> {
        self.inner.put(axis, b)
    }
    fn get(&self, k: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        let len = self.inner.content_len(k)?.unwrap_or(0);
        self.get_range(k, 0, len)
    }
    fn contains(&self, k: &KappaLabel71) -> bool {
        self.inner.contains(k)
    }
    fn pin(&self, k: &KappaLabel71) -> Result<(), StoreError> {
        self.inner.pin(k)
    }
    fn unpin(&self, k: &KappaLabel71) -> Result<(), StoreError> {
        self.inner.unpin(k)
    }
    fn iterate(&self) -> Vec<KappaLabel71> {
        self.inner.iterate()
    }
    fn pinned_roots(&self) -> Vec<KappaLabel71> {
        self.inner.pinned_roots()
    }
    fn approximate_count(&self) -> usize {
        self.inner.approximate_count()
    }
    fn approximate_bytes(&self) -> u64 {
        self.inner.approximate_bytes()
    }
    fn content_len(&self, k: &KappaLabel71) -> Result<Option<u64>, StoreError> {
        self.inner.content_len(k)
    }
    fn get_range(&self, k: &KappaLabel71, off: u64, len: u64) -> Result<Option<Bytes>, StoreError> {
        let Some(b) = self.inner.get_range(k, off, len)? else {
            return Ok(None);
        };
        let mut v = b.to_vec();
        if (off..off + len).contains(&self.flip_at) {
            v[(self.flip_at - off) as usize] ^= 0x01;
        }
        Ok(Some(Arc::from(v)))
    }
}

fn mem_with(b: &[u8]) -> (Arc<dyn KappaStore>, KappaLabel71) {
    let s = Arc::new(MemKappaStore::new());
    let k = s.put("blake3", b).unwrap();
    (s, k)
}

/// Send raw request bytes on `stream` and read one response: (status line, headers, body).
fn raw_exchange(
    conn: &mut BufReader<TcpStream>,
    req: &str,
) -> (String, Vec<(String, String)>, Vec<u8>) {
    conn.get_mut().write_all(req.as_bytes()).unwrap();
    let mut status = String::new();
    conn.read_line(&mut status).unwrap();
    let mut headers = Vec::new();
    loop {
        let mut l = String::new();
        conn.read_line(&mut l).unwrap();
        let l = l.trim_end();
        if l.is_empty() {
            break;
        }
        let (k, v) = l.split_once(':').unwrap();
        headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
    }
    let len: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .map_or(0, |(_, v)| v.parse().unwrap());
    let head_only = req.starts_with("HEAD ");
    let mut body = vec![0u8; if head_only { 0 } else { len }];
    conn.read_exact(&mut body).unwrap();
    (status.trim_end().to_string(), headers, body)
}

fn h<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

// ───────────────────────────── D: the server's HTTP surface ─────────────────────────────

#[test]
fn head_range_etag_cors_and_keep_alive_on_one_connection() {
    let b = blob(3 * PIECE_LEN as usize + 10, 1);
    let (store, k) = mem_with(&b);
    let server = serve(store, false).unwrap();
    let path = cas_path(&k);
    let mut c = BufReader::new(TcpStream::connect(server.addr()).unwrap());
    let req =
        |method: &str, extra: &str| format!("{method} {path} HTTP/1.1\r\nHost: x\r\n{extra}\r\n");

    // HEAD: full metadata, no body.
    let (st, hd, body) = raw_exchange(&mut c, &req("HEAD", ""));
    assert_eq!(st, "HTTP/1.1 200 OK");
    assert_eq!(h(&hd, "content-length"), Some(b.len().to_string().as_str()));
    let etag = format!("\"{}\"", k.as_str());
    assert_eq!(h(&hd, "etag"), Some(etag.as_str()));
    assert_eq!(h(&hd, "accept-ranges"), Some("bytes"));
    assert_eq!(h(&hd, "content-type"), Some("application/octet-stream"));
    assert_eq!(h(&hd, "access-control-allow-origin"), Some("*"));
    assert!(h(&hd, "access-control-expose-headers")
        .unwrap()
        .contains("Content-Range"));
    assert_eq!(h(&hd, "connection"), Some("keep-alive"));
    assert!(body.is_empty());

    // Same socket: ranges of every form.
    let (st, hd, body) = raw_exchange(&mut c, &req("GET", "Range: bytes=100-199\r\n"));
    assert_eq!(st, "HTTP/1.1 206 Partial Content");
    assert_eq!(
        h(&hd, "content-range"),
        Some(format!("bytes 100-199/{}", b.len()).as_str())
    );
    assert_eq!(body, &b[100..200]);
    let (_, _, body) = raw_exchange(&mut c, &req("GET", "Range: bytes=-5\r\n"));
    assert_eq!(body, &b[b.len() - 5..]);
    let tail = format!("Range: bytes={}-\r\n", b.len() - 3);
    let (_, _, body) = raw_exchange(&mut c, &req("GET", &tail));
    assert_eq!(body, &b[b.len() - 3..]);
    // Spanning a piece boundary, from a blob larger than one body window.
    let span = format!("Range: bytes={}-{}\r\n", PIECE_LEN - 7, 2 * PIECE_LEN + 7);
    let (_, _, body) = raw_exchange(&mut c, &req("GET", &span));
    assert_eq!(body, &b[PIECE_LEN as usize - 7..2 * PIECE_LEN as usize + 8]);

    // Past the end: 416 with the length.
    let past = format!("Range: bytes={}-\r\n", b.len());
    let (st, hd, body) = raw_exchange(&mut c, &req("GET", &past));
    assert_eq!(st, "HTTP/1.1 416 Range Not Satisfiable");
    assert_eq!(
        h(&hd, "content-range"),
        Some(format!("bytes */{}", b.len()).as_str())
    );
    assert!(body.is_empty());

    // Multiple ranges are ignored: the whole body with 200.
    let (st, _, body) = raw_exchange(&mut c, &req("GET", "Range: bytes=0-1,5-6\r\n"));
    assert_eq!(st, "HTTP/1.1 200 OK");
    assert_eq!(body, b);

    // Conditional requests on the κ ETag.
    let inm = format!("If-None-Match: {etag}\r\n");
    let (st, hd, body) = raw_exchange(&mut c, &req("GET", &inm));
    assert_eq!(st, "HTTP/1.1 304 Not Modified");
    assert_eq!(h(&hd, "etag"), Some(etag.as_str()));
    assert!(body.is_empty());
    let stale = "If-Range: \"blake3:old\"\r\nRange: bytes=0-0\r\n";
    let (st, _, body) = raw_exchange(&mut c, &req("GET", stale));
    assert_eq!(
        st, "HTTP/1.1 200 OK",
        "a stale If-Range gets the whole body"
    );
    assert_eq!(body.len(), b.len());
    let fresh = format!("If-Range: {etag}\r\nRange: bytes=0-0\r\n");
    let (st, _, _) = raw_exchange(&mut c, &req("GET", &fresh));
    assert_eq!(st, "HTTP/1.1 206 Partial Content");

    // Connection: close is honoured.
    let (_, hd, _) = raw_exchange(&mut c, &req("HEAD", "Connection: close\r\n"));
    assert_eq!(h(&hd, "connection"), Some("close"));
    let mut rest = Vec::new();
    c.read_to_end(&mut rest).unwrap();
    assert!(rest.is_empty(), "server closed after Connection: close");
}

#[test]
fn preflight_errors_and_piece_table_resource() {
    let b = blob(2 * PIECE_LEN as usize + 1, 2);
    let (store, k) = mem_with(&b);
    let server = serve(store, false).unwrap();
    let mut c = BufReader::new(TcpStream::connect(server.addr()).unwrap());

    let (st, hd, _) = raw_exchange(
        &mut c,
        &format!(
            "OPTIONS {} HTTP/1.1\r\nOrigin: https://hub.example\r\nAccess-Control-Request-Headers: range\r\n\r\n",
            cas_path(&k)
        ),
    );
    assert_eq!(st, "HTTP/1.1 204 No Content");
    assert_eq!(
        h(&hd, "access-control-allow-methods"),
        Some("GET, HEAD, OPTIONS")
    );
    assert!(h(&hd, "access-control-allow-headers")
        .unwrap()
        .contains("Range"));

    let absent = cas_path(&address_bytes(b"absent"));
    let (st, _, _) = raw_exchange(&mut c, &format!("GET {absent} HTTP/1.1\r\n\r\n"));
    assert_eq!(st, "HTTP/1.1 404 Not Found");
    let (st, _, _) = raw_exchange(&mut c, "GET /cas/junk HTTP/1.1\r\n\r\n");
    assert_eq!(st, "HTTP/1.1 400 Bad Request");
    let (st, hd, _) = raw_exchange(&mut c, &format!("DELETE {} HTTP/1.1\r\n\r\n", cas_path(&k)));
    assert_eq!(st, "HTTP/1.1 405 Method Not Allowed");
    assert_eq!(h(&hd, "allow"), Some("GET, HEAD, OPTIONS"));

    let (st, _, body) = raw_exchange(&mut c, &format!("GET {} HTTP/1.1\r\n\r\n", pieces_path(&k)));
    assert_eq!(st, "HTTP/1.1 200 OK");
    let tree = PieceTree::decode(&body).unwrap();
    tree.verify_root(&k).unwrap();
    assert_eq!(tree, PieceTree::from_bytes(&b));
    let (st, _, _) = raw_exchange(
        &mut c,
        &format!(
            "GET {} HTTP/1.1\r\n\r\n",
            pieces_path(&address_bytes(b"absent"))
        ),
    );
    assert_eq!(st, "HTTP/1.1 404 Not Found");
}

#[test]
fn a_store_that_lies_gets_no_piece_table() {
    let b = blob(2 * PIECE_LEN as usize, 3);
    let (inner, k) = mem_with(&b);
    let lying: Arc<dyn KappaStore> = Arc::new(Lying { inner, flip_at: 5 });
    let server = serve(lying, false).unwrap();
    let mut c = HttpConn::new(&server.addr().to_string(), Duration::from_secs(5));
    let r = c.request("GET", &pieces_path(&k), &[], 1 << 20).unwrap();
    assert_eq!(
        r.status, 500,
        "a server never vouches for bytes that miss their κ"
    );
}

#[test]
fn an_idle_connection_does_not_block_others_and_the_cap_is_enforced() {
    let (store, k) = mem_with(&blob(100_000, 4));
    let server = serve_with(
        store,
        "127.0.0.1:0",
        ServeOptions {
            max_connections: 3,
            ..ServeOptions::default()
        },
    )
    .unwrap();
    let addr = server.addr().to_string();
    // Two clients connect and say nothing (the old server served connections one at a time,
    // so this alone would have stalled everyone behind them).
    let idle1 = TcpStream::connect(&addr).unwrap();
    let idle2 = TcpStream::connect(&addr).unwrap();
    let mut c = HttpConn::new(&addr, Duration::from_secs(5));
    let r = c.request("GET", &cas_path(&k), &[], 1 << 20).unwrap();
    assert_eq!(r.status, 200);
    assert_eq!(r.body.len(), 100_000);
    // A fourth concurrent connection is refused with 503.
    let mut extra = BufReader::new(TcpStream::connect(&addr).unwrap());
    let mut line = String::new();
    extra.read_line(&mut line).unwrap();
    assert!(line.starts_with("HTTP/1.1 503"), "{line}");
    drop((idle1, idle2));
}

#[test]
fn many_clients_in_parallel_get_correct_ranges() {
    let b = Arc::new(blob(8 * PIECE_LEN as usize, 5));
    let (store, k) = mem_with(&b);
    let server = serve(store, false).unwrap();
    let addr = server.addr().to_string();
    std::thread::scope(|s| {
        for t in 0..16u64 {
            let (addr, b) = (addr.clone(), b.clone());
            s.spawn(move || {
                let mut c = HttpConn::new(&addr, Duration::from_secs(10));
                for i in 0..8u64 {
                    let start = ((t * 7 + i * 131_071) % (b.len() as u64 - 4096)) as usize;
                    let range = format!("bytes={}-{}", start, start + 4095);
                    let r = c
                        .request("GET", &cas_path(&k), &[("Range", &range)], 4096)
                        .unwrap();
                    assert_eq!(r.status, 206);
                    assert_eq!(r.body, &b[start..start + 4096]);
                }
            });
        }
    });
}

#[test]
fn a_file_backed_server_streams_a_large_blob() {
    let dir = TempDir::new("serve");
    let store = Arc::new(FileKappaStore::open(&dir.0).unwrap());
    let b = blob(20 * PIECE_LEN as usize + 5, 6);
    let k = store.put("blake3", &b).unwrap();
    let server = serve(store, false).unwrap();
    let mut c = HttpConn::new(&server.addr().to_string(), Duration::from_secs(10));
    let r = c.request("GET", &cas_path(&k), &[], u64::MAX).unwrap();
    assert_eq!(r.status, 200);
    assert!(r.body == b);
}

// ───────────────────────────── E: parallel multi-source verified fetch ─────────────────────────────

#[test]
fn fetch_spreads_pieces_over_honest_sources() {
    let b = blob(12 * PIECE_LEN as usize + 99, 10);
    let servers: Vec<_> = (0..3)
        .map(|_| serve(mem_with(&b).0, false).unwrap())
        .collect();
    let k = address_bytes(&b);
    let sources: Vec<String> = servers.iter().map(|s| s.addr().to_string()).collect();
    let (got, report) = fetch_to_vec(&k, &sources, &opts()).unwrap();
    assert!(got == b);
    assert_eq!(report.pieces_fetched, 13);
    assert_eq!(report.bytes_fetched, b.len() as u64);
    assert!(
        report.sources.iter().all(|s| s.pieces > 0 && !s.dropped),
        "{report:?}"
    );
}

#[test]
fn a_lying_source_is_dropped_and_its_pieces_refetched_elsewhere() {
    let b = blob(10 * PIECE_LEN as usize, 11);
    let (honest, k) = mem_with(&b);
    // Planted defects: one source flips a byte in piece 4, another serves garbage for everything,
    // and a third address has nobody listening.
    let liar = serve(
        Arc::new(Lying {
            inner: honest.clone(),
            flip_at: 4 * PIECE_LEN + 12_345,
        }),
        false,
    )
    .unwrap();
    let garbage = serve(honest.clone(), true).unwrap();
    let dead = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().to_string()
    };
    let good = serve(honest, false).unwrap();
    let sources = vec![
        liar.addr().to_string(),
        garbage.addr().to_string(),
        dead.clone(),
        good.addr().to_string(),
    ];

    // The table comes from the first source whose table hashes to κ: the liar's server refuses to
    // compute one (500), the garbage server's does not decode, the dead one does not answer.
    let tree = fetch_piece_tree(&k, &sources, &opts()).unwrap();
    let sink = hologram_net::http::fetch::MemPieceSink::new(&tree);
    let done = vec![false; tree.piece_count() as usize];
    let report = fetch_pieces(&k, &tree, &sources, &done, &sink, &opts()).unwrap();
    assert!(sink.into_bytes() == b, "every byte is the honest one");

    let by = |a: &str| report.sources.iter().find(|s| s.source == a).unwrap();
    assert!(by(&sources[1]).dropped && by(&sources[1]).forged >= 1);
    assert!(by(&sources[2]).dropped && by(&sources[2]).failures >= 1);
    assert_eq!(by(&sources[3]).forged, 0);
    let liar_r = by(&sources[0]);
    assert!(
        liar_r.forged == 0 || liar_r.dropped,
        "a source that forged a piece is dropped: {liar_r:?}"
    );
    assert_eq!(report.pieces_fetched, 10);
}

#[test]
fn fetch_fails_closed_when_no_source_is_honest() {
    let b = blob(3 * PIECE_LEN as usize, 12);
    let (honest, k) = mem_with(&b);
    let garbage = serve(honest.clone(), true).unwrap();
    let sources = vec![garbage.addr().to_string()];
    assert_eq!(
        fetch_piece_tree(&k, &sources, &opts()),
        Err(FetchError::NoPieceTree)
    );

    // With a valid table but only liars for the bytes, the fetch stops on a piece, never
    // delivering an unverified one.
    let tree = PieceTree::from_bytes(&b);
    let liar = serve(
        Arc::new(Lying {
            inner: honest,
            flip_at: 0,
        }),
        false,
    )
    .unwrap();
    let sink = hologram_net::http::fetch::MemPieceSink::new(&tree);
    let err = fetch_pieces(
        &k,
        &tree,
        &[liar.addr().to_string()],
        &[false; 3],
        &sink,
        &FetchOptions {
            concurrency: 1,
            ..opts()
        },
    )
    .unwrap_err();
    assert_eq!(err, FetchError::PieceUnavailable(0));
    assert!(!sink.done()[0], "the forged piece never reached the sink");

    // A table for other content is refused before any byte is fetched.
    let other = PieceTree::from_bytes(&blob(3 * PIECE_LEN as usize, 13));
    assert_eq!(
        fetch_pieces(&k, &other, &[], &[false; 3], &sink, &opts()),
        Err(FetchError::NoPieceTree)
    );
}

/// A sink that accepts `budget` pieces, then fails: an interrupted download.
struct Interrupting<'a> {
    inner: &'a dyn PieceSink,
    budget: AtomicI64,
}

impl PieceSink for Interrupting<'_> {
    fn put_piece(&self, index: u64, bytes: &[u8]) -> Result<(), StoreError> {
        if self.budget.fetch_sub(1, Ordering::SeqCst) <= 0 {
            return Err(StoreError::QuotaExceeded);
        }
        self.inner.put_piece(index, bytes)
    }
}

#[test]
fn an_interrupted_download_resumes_into_the_file_store() {
    let b = blob(9 * PIECE_LEN as usize + 3, 14);
    let (origin, k) = mem_with(&b);
    let a = serve(origin.clone(), false).unwrap();
    let c = serve(origin, false).unwrap();
    let sources = vec![a.addr().to_string(), c.addr().to_string()];
    let dir = TempDir::new("resume");

    {
        let store = FileKappaStore::open(&dir.0).unwrap();
        let tree = fetch_piece_tree(&k, &sources, &opts()).unwrap();
        let part = store.begin_partial(&k, tree.clone()).unwrap();
        let sink = Interrupting {
            inner: &part,
            budget: AtomicI64::new(4),
        };
        let err = fetch_pieces(&k, &tree, &sources, &part.done(), &sink, &opts()).unwrap_err();
        assert_eq!(err, FetchError::Sink(StoreError::QuotaExceeded));
        assert_eq!(part.done().iter().filter(|d| **d).count(), 4);
        // The process stops here; nothing is installed.
        assert!(!store.contains(&k));
    }

    let store = FileKappaStore::open(&dir.0).unwrap();
    let part = store.resume_partial(&k).unwrap().unwrap();
    let report = fetch_pieces(&k, part.tree(), &sources, &part.done(), &part, &opts()).unwrap();
    assert_eq!(report.pieces_skipped, 4);
    assert_eq!(report.pieces_fetched, 6);
    assert_eq!(part.commit().unwrap(), k);
    assert_eq!(store.verify_blob(&k).unwrap(), Some(true));
    assert_eq!(
        store
            .get_range(&k, 5 * PIECE_LEN, 10)
            .unwrap()
            .unwrap()
            .as_ref(),
        &b[5 * PIECE_LEN as usize..5 * PIECE_LEN as usize + 10]
    );
}

#[test]
fn a_source_without_the_blob_is_dropped_not_retried() {
    let b = blob(4 * PIECE_LEN as usize, 15);
    let (honest, k) = mem_with(&b);
    let empty = serve(Arc::new(MemKappaStore::new()), false).unwrap();
    let good = serve(honest, false).unwrap();
    let sources = vec![empty.addr().to_string(), good.addr().to_string()];
    let (got, report) = fetch_to_vec(&k, &sources, &opts()).unwrap();
    assert!(got == b);
    let e = &report.sources[0];
    assert!(e.dropped && e.pieces == 0 && e.failures >= 1, "{e:?}");
}

#[test]
fn small_and_empty_blobs_fetch() {
    for b in [Vec::new(), b"tiny".to_vec(), blob(PIECE_LEN as usize, 16)] {
        let (store, k) = mem_with(&b);
        let s = serve(store, false).unwrap();
        let (got, _) = fetch_to_vec(&k, &[s.addr().to_string()], &opts()).unwrap();
        assert_eq!(got, b);
    }
}

/// A fresh directory under the system temp dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "hologram-net-http-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
