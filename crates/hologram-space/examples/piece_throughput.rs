//! Throughput of the κ verification paths on one in-memory blob (default 1 GiB).
//!
//! `cargo run --release -p hologram-space --example piece_throughput [-- <MiB>]`
//!
//! Reports: whole-blob BLAKE3 (`address_bytes`), the streaming verifier, building the 1 MiB piece
//! table, and verifying every piece against it (one thread, then all cores).
use std::time::Instant;

use hologram_space::{address_bytes, PieceTree, StreamVerifier, PIECE_LEN};

fn main() {
    let mib: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let n = mib << 20;
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let blob: Vec<u8> = (0..n / 8)
        .flat_map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x.to_le_bytes()
        })
        .collect();
    let gbps = |secs: f64| n as f64 / secs / 1e9;

    let t = Instant::now();
    let k = address_bytes(&blob);
    println!(
        "blake3 one-shot (address_bytes)   {:6.2} GB/s",
        gbps(t.elapsed().as_secs_f64())
    );

    let t = Instant::now();
    let mut v = StreamVerifier::new(&k).unwrap();
    for c in blob.chunks(256 << 10) {
        v.update(c).unwrap();
    }
    v.finish().unwrap();
    println!(
        "stream verify, 256 KiB writes     {:6.2} GB/s",
        gbps(t.elapsed().as_secs_f64())
    );

    let t = Instant::now();
    let tree = PieceTree::from_bytes(&blob);
    println!(
        "piece table build, 1 thread       {:6.2} GB/s",
        gbps(t.elapsed().as_secs_f64())
    );
    tree.verify_root(&k).unwrap();

    let t = Instant::now();
    for i in 0..tree.piece_count() {
        let r = tree.piece_range(i).unwrap();
        tree.verify_piece(i, &blob[r.start as usize..r.end as usize])
            .unwrap();
    }
    println!(
        "piece verify, 1 thread            {:6.2} GB/s",
        gbps(t.elapsed().as_secs_f64())
    );

    let threads = std::thread::available_parallelism().map_or(1, |p| p.get());
    let t = Instant::now();
    let next = std::sync::atomic::AtomicU64::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= tree.piece_count() {
                    break;
                }
                let r = tree.piece_range(i).unwrap();
                tree.verify_piece(i, &blob[r.start as usize..r.end as usize])
                    .unwrap();
            });
        }
    });
    println!(
        "piece verify, {threads:2} threads          {:6.2} GB/s",
        gbps(t.elapsed().as_secs_f64())
    );
    println!(
        "piece table size: {} bytes for {} MiB ({} pieces of {} B)",
        tree.encode().len(),
        mib,
        tree.piece_count(),
        PIECE_LEN
    );
}
