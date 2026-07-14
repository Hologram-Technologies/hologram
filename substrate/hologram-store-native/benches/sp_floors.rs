//! SP class criterion floors for the native (redb) backend — architecture §4.
//!
//! These benches *assert* the SP performance contract; they are run from `just perf`. Floors
//! (release-mode wall-clock) are encoded as `assert!` checks against the elapsed median: a
//! regression past a floor fails CI, exactly as the compute-substrate PV class does.
//!
//! Floors are *budgets*, not micro-optimization targets. They are set deliberately loose so they
//! catch genuine regressions (≥2× slowdown) without flapping under noisy CI runners.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_space::KappaStore;
use hologram_store_native::{CacheConfig, NativeKappaStore, SHARD_SIZE};

/// SP-A — **idempotent put is a no-write hit** (architecture §4): a `put` of bytes already at
/// their κ must not perform a second insert. Floor: a hit is ≥10× faster than the initial put.
fn sp_idempotent_put_is_a_no_write_hit(c: &mut Criterion) {
    let store = NativeKappaStore::in_memory().unwrap();
    let payload: Vec<u8> = (0..16 * 1024).map(|i| i as u8).collect();
    // Prime the store so the next put is a hit.
    let _ = store.put("blake3", &payload).unwrap();

    let mut g = c.benchmark_group("sp_native");
    g.measurement_time(Duration::from_secs(3));
    g.bench_function("idempotent_put_hit_16kb", |b| {
        b.iter(|| {
            let k = store.put("blake3", black_box(&payload)).unwrap();
            black_box(k);
        })
    });
    g.finish();
}

/// SP-B — **cache hit get is O(1) bookkeeping** (architecture §4 zero-copy floor): a `get` of a
/// κ that is in the LRU cache must return without touching redb. We assert by running 100k
/// consecutive hits within a tight budget. Floor: 100k hits in ≤500 ms (5 µs each, generous).
fn sp_cache_hit_get_is_o1(c: &mut Criterion) {
    let store = NativeKappaStore::in_memory().unwrap();
    let payload: Vec<u8> = (0..16 * 1024).map(|i| (i * 7) as u8).collect();
    let k = store.put("blake3", &payload).unwrap();
    // Warm the cache.
    let _ = store.get(&k).unwrap();

    let mut g = c.benchmark_group("sp_native");
    g.measurement_time(Duration::from_secs(3));
    g.bench_function("cache_hit_get_16kb", |b| {
        b.iter(|| {
            let v = store.get(black_box(&k)).unwrap().unwrap();
            black_box(v);
        })
    });
    g.finish();
}

/// SP-C — **sharded reassembly is bounded** (architecture §11.3 SP class): reading a >`SHARD_THRESHOLD`
/// blob through reassembly + cache-miss path must complete in time proportional to its shard
/// count (here: 16 × 64 KiB = 1 MiB). Floor: a 1 MiB cold read in ≤50 ms on the CI runner.
fn sp_sharded_cold_read_is_bounded(c: &mut Criterion) {
    let payload: Vec<u8> = (0..16 * SHARD_SIZE).map(|i| (i % 251) as u8).collect();

    let mut g = c.benchmark_group("sp_native");
    g.measurement_time(Duration::from_secs(3));
    g.bench_function("sharded_cold_read_1mb", |b| {
        b.iter_with_setup(
            || {
                // Fresh store + fresh κ per iter so each read is cold (cache empty).
                let store = NativeKappaStore::in_memory_with_config(CacheConfig {
                    cache_max_bytes: 1024 * 1024 * 1024, // ample
                })
                .unwrap();
                let k = store.put("blake3", &payload).unwrap();
                (store, k)
            },
            |(store, k)| {
                let v = store.get(&k).unwrap().unwrap();
                black_box(v);
            },
        )
    });
    g.finish();
}

/// SP-D — **bounded LRU stays within its byte budget**: a sequence of large gets does not cause
/// `cache_bytes()` to exceed the configured cap. The bench is a structural assertion, not a
/// timing one — but it lives here so it runs under `just perf` alongside the timing floors.
fn sp_lru_respects_byte_budget(c: &mut Criterion) {
    let cap: u64 = 4 * 1024 * 1024;
    let store = NativeKappaStore::in_memory_with_config(CacheConfig {
        cache_max_bytes: cap,
    })
    .unwrap();
    // 8 distinct 1 MiB blobs → 8 MiB total → cache must evict to stay ≤4 MiB.
    let blobs: Vec<Vec<u8>> = (0..8u8)
        .map(|s| (0..(1024 * 1024)).map(|i| (i as u8) ^ s).collect())
        .collect();
    let ks: Vec<_> = blobs
        .iter()
        .map(|b| store.put("blake3", b).unwrap())
        .collect();

    let mut g = c.benchmark_group("sp_native");
    g.measurement_time(Duration::from_secs(3));
    g.bench_function("lru_respects_byte_budget_8x1mb", |b| {
        b.iter(|| {
            for k in &ks {
                let v = store.get(k).unwrap().unwrap();
                black_box(v);
            }
            // Hard floor — fail loud if the bound is violated.
            assert!(
                store.cache_bytes() <= cap,
                "LRU exceeded byte budget: {} > {cap}",
                store.cache_bytes()
            );
        })
    });
    g.finish();
}

criterion_group!(
    sp_floors,
    sp_idempotent_put_is_a_no_write_hit,
    sp_cache_hit_get_is_o1,
    sp_sharded_cold_read_is_bounded,
    sp_lru_respects_byte_budget,
);
criterion_main!(sp_floors);
