//! End-to-end throughput for one large blob (default 1 GiB): store write, ranged reads, and
//! HTTP-CAS serving and verified parallel fetch over loopback.
//!
//! `cargo run --release -p hologram-net --features live --example large_blob_throughput [-- <MiB> [dir]]`
//!
//! Memory stays small: the blob is generated as it is written, HTTP bodies are discarded as they
//! arrive, and "verify only" fetches check every piece and drop it. Reads hit the OS page cache
//! after the write, so read and HTTP numbers measure the code path, not the disk. Hashing-only
//! numbers are in `cargo run --release -p hologram-space --example piece_throughput`.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hologram_net::http::cas_path;
use hologram_net::http::client::HttpConn;
use hologram_net::http::fetch::{fetch_piece_tree, fetch_pieces, FetchOptions};
use hologram_net::http::live::serve;
use hologram_space::{KappaLabel71, KappaStore, PieceSink, StoreError, PIECE_LEN};
use hologram_store::file::{kappa_file_name, FileKappaStore};

/// Verifies nothing itself (the fetcher already did) and keeps nothing: isolates network + verify.
struct Discard(AtomicU64);

impl PieceSink for Discard {
    fn put_piece(&self, _i: u64, bytes: &[u8]) -> Result<(), StoreError> {
        self.0.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

fn gbps(bytes: u64, d: Duration) -> f64 {
    bytes as f64 / d.as_secs_f64() / 1e9
}

/// GET the whole blob on a raw socket, discarding the body as it streams in.
fn get_whole_discarding(addr: &str, k: &KappaLabel71) -> u64 {
    let mut s = BufReader::new(TcpStream::connect(addr).unwrap());
    write!(
        s.get_mut(),
        "GET {} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
        cas_path(k)
    )
    .unwrap();
    let mut line = String::new();
    loop {
        line.clear();
        s.read_line(&mut line).unwrap();
        if line.trim().is_empty() {
            break;
        }
    }
    let mut buf = vec![0u8; 1 << 20];
    let mut got = 0u64;
    loop {
        let n = s.read(&mut buf).unwrap();
        if n == 0 {
            return got;
        }
        got += n as u64;
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mib: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let base = args
        .next()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("hologram-large-blob-bench"));
    let _ = std::fs::remove_dir_all(&base);
    let n = mib << 20;
    println!("blob: {mib} MiB");

    // ── store: streaming put (hash + piece table + write + fsync + rename) ──
    let origin = Arc::new(FileKappaStore::open(base.join("origin")).unwrap());
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut chunk = vec![0u8; 4 << 20];
    let t = Instant::now();
    let mut w = origin.put_writer("blake3", None).unwrap();
    let mut left = n;
    while left > 0 {
        let take = left.min(chunk.len() as u64) as usize;
        for word in chunk[..take].chunks_exact_mut(8) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            word.copy_from_slice(&x.to_le_bytes());
        }
        w.write(&chunk[..take]).unwrap();
        left -= take as u64;
    }
    let k = w.finish().unwrap();
    let d = t.elapsed();
    println!(
        "store put_writer (hash+table+fsync)  {:6.2} GB/s  ({:.2} s)",
        gbps(n, d),
        d.as_secs_f64()
    );
    let t = Instant::now();
    assert_eq!(origin.verify_blob(&k).unwrap(), Some(true));
    println!(
        "store verify_blob (re-hash file)     {:6.2} GB/s",
        gbps(n, t.elapsed())
    );

    // ── store: ranged reads ──
    let t = Instant::now();
    let mut off = 0;
    while off < n {
        let len = (n - off).min(PIECE_LEN);
        assert_eq!(
            origin.get_range(&k, off, len).unwrap().unwrap().len() as u64,
            len
        );
        off += len;
    }
    println!(
        "store get_range 1 MiB sequential     {:6.2} GB/s",
        gbps(n, t.elapsed())
    );
    let mut seed = 7u64;
    let reads = 5000u64;
    let t = Instant::now();
    for _ in 0..reads {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        origin
            .get_range(&k, seed % (n - 4096), 4096)
            .unwrap()
            .unwrap();
    }
    println!(
        "store get_range 4 KiB random         {:6.1} µs/read",
        t.elapsed().as_secs_f64() * 1e6 / reads as f64
    );
    std::fs::remove_file(origin.root().join("trees").join(kappa_file_name(&k))).unwrap();
    let t = Instant::now();
    let tree = origin.piece_tree(&k).unwrap().unwrap();
    println!(
        "store piece_tree from file (cores)   {:6.2} GB/s",
        gbps(n, t.elapsed())
    );

    // ── HTTP-CAS over loopback ──
    let servers: Vec<_> = (0..3)
        .map(|_| serve(origin.clone(), false).unwrap())
        .collect();
    let all: Vec<String> = servers.iter().map(|s| s.addr().to_string()).collect();
    let one = all[..1].to_vec();

    let t = Instant::now();
    assert_eq!(get_whole_discarding(&one[0], &k), n);
    println!(
        "http GET whole, 1 conn (no verify)   {:6.2} GB/s",
        gbps(n, t.elapsed())
    );

    let mut c = HttpConn::new(&one[0], Duration::from_secs(60));
    let t = Instant::now();
    let mut off = 0;
    while off < n {
        let end = (off + PIECE_LEN).min(n);
        let range = format!("bytes={}-{}", off, end - 1);
        let r = c
            .request("GET", &cas_path(&k), &[("Range", &range)], PIECE_LEN)
            .unwrap();
        assert_eq!(r.status, 206);
        off = end;
    }
    println!(
        "http Range 1 MiB, 1 keep-alive conn  {:6.2} GB/s",
        gbps(n, t.elapsed())
    );

    let done = vec![false; tree.piece_count() as usize];
    for (label, sources, workers) in [
        ("1 source,  1 worker ", &one, 1usize),
        ("1 source,  8 workers", &one, 8),
        ("3 sources, 8 workers", &all, 8),
        ("3 sources, 16 workers", &all, 16),
    ] {
        let opts = FetchOptions {
            concurrency: workers,
            ..FetchOptions::default()
        };
        let sink = Discard(AtomicU64::new(0));
        let t = Instant::now();
        fetch_pieces(&k, &tree, sources, &done, &sink, &opts).unwrap();
        let d = t.elapsed();
        assert_eq!(sink.0.load(Ordering::Relaxed), n);
        println!("fetch+verify, {label}       {:6.2} GB/s", gbps(n, d));
    }

    {
        let dest = FileKappaStore::open(base.join("dest")).unwrap();
        let opts = FetchOptions {
            concurrency: 8,
            ..FetchOptions::default()
        };
        let t = Instant::now();
        let tree = fetch_piece_tree(&k, &all, &opts).unwrap();
        let part = dest.begin_partial(&k, tree).unwrap();
        fetch_pieces(&k, part.tree(), &all, &part.done(), &part, &opts).unwrap();
        part.commit().unwrap();
        let d = t.elapsed();
        println!(
            "fetch+verify → disk, 3 sources, 8 workers {:6.2} GB/s  ({:.2} s, incl. fsync)",
            gbps(n, d),
            d.as_secs_f64()
        );
    }
    drop(servers);
    let _ = std::fs::remove_dir_all(&base);
}
