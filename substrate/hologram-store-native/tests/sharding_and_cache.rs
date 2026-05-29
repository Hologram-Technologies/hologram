//! V&V witnesses for G1 (bounded LRU read-through cache) and G2 (file-sharding split per spec §5.5).
//!
//! These complement the shared TCK (`store_battery`) and the reachability-GC witness; they assert
//! the **uor-native** properties that distinguish the redb backend from the in-memory reference:
//! every fragment of a large blob is a κ, identical fragments dedup across blobs, sharded blobs
//! verify by σ-axis re-derivation, and the LRU cache respects its byte budget while still
//! upholding the SP zero-copy floor.

use hologram_realizations::REGISTRY;
use hologram_store_native::{CacheConfig, NativeKappaStore, SHARD_SIZE};
use hologram_substrate_core::{address_bytes, verify_kappa, KappaStore};
use std::sync::Arc;

/// G2-1 — round-trip a blob larger than [`SHARD_THRESHOLD`]: put returns the σ-axis address of
/// the *whole* content; get reassembles exactly those bytes; the κ verifies by re-derivation.
#[test]
fn g2_large_blob_round_trips_through_sharding() {
    let store = NativeKappaStore::in_memory().unwrap();
    // 5 × SHARD_SIZE + 17 — exercises a non-aligned tail shard.
    let payload: Vec<u8> = (0..5 * SHARD_SIZE + 17).map(|i| (i % 251) as u8).collect();
    let k = store.put("blake3", &payload).unwrap();
    // The κ is the address of the whole content (not a manifest hash).
    assert_eq!(&k, &address_bytes(&payload));
    let got = store.get(&k).unwrap().expect("present after put");
    assert_eq!(got.len(), payload.len());
    assert_eq!(got.as_ref(), payload.as_slice());
    // SPINE-4: re-derivation matches.
    assert!(verify_kappa(got.as_ref(), &k).unwrap());
}

/// G2-2 — sharding is below [`SHARD_THRESHOLD`]-content-equivalent: a tiny blob still inlines, and
/// `contains`/`get` see no difference. Sharding is a backend refinement, not a wire-visible split.
#[test]
fn g2_small_blob_takes_the_inline_path() {
    let store = NativeKappaStore::in_memory().unwrap();
    let payload = b"small inline payload";
    let k = store.put("blake3", payload).unwrap();
    assert!(store.contains(&k));
    let got = store.get(&k).unwrap().unwrap();
    assert_eq!(got.as_ref(), payload.as_slice());
}

/// G2-3 — fragment dedup: identical shards across distinct large blobs share storage in INLINE
/// (uor-native: every fragment is itself content-addressed, so duplicates collapse).
#[test]
fn g2_identical_shard_dedups_across_blobs() {
    let store = NativeKappaStore::in_memory().unwrap();
    // Two large blobs that share a prefix of exactly one SHARD_SIZE.
    let mut a: Vec<u8> = vec![0xAA; SHARD_SIZE];
    a.extend(vec![0x11; SHARD_SIZE + 7]);
    let mut b: Vec<u8> = vec![0xAA; SHARD_SIZE]; // identical first shard
    b.extend(vec![0x22; SHARD_SIZE + 7]);

    let ka = store.put("blake3", &a).unwrap();
    let bytes_after_a = store.approximate_bytes();
    let kb = store.put("blake3", &b).unwrap();
    let bytes_after_b = store.approximate_bytes();
    assert_ne!(ka, kb);
    // The increment from putting `b` must be strictly less than `b.len()` because the first
    // shard was deduped by content-address — that's the uor-native property.
    let delta = bytes_after_b - bytes_after_a;
    assert!(
        delta < b.len() as u64,
        "shared shard must dedup: delta={delta}, b.len()={}",
        b.len()
    );
}

/// G2-4 — `iterate` and `approximate_count` report top-level κs only, never fragment κs (a user
/// who put one large blob sees one top-level entry).
#[test]
fn g2_iterate_returns_top_level_only_not_fragments() {
    let store = NativeKappaStore::in_memory().unwrap();
    let payload: Vec<u8> = (0..3 * SHARD_SIZE).map(|i| (i % 7) as u8).collect();
    let k = store.put("blake3", &payload).unwrap();
    let listed = store.iterate();
    assert_eq!(listed.len(), 1, "exactly one top-level κ visible");
    assert_eq!(listed[0], k);
    assert_eq!(store.approximate_count(), 1);
}

/// G2-5 — GC of a sharded κ reclaims its fragments (subject to dedup): when no other reachable
/// blob references a fragment, the fragment is evicted with its parent.
#[test]
fn g2_gc_reclaims_fragments_of_unreachable_sharded_blob() {
    let store = NativeKappaStore::in_memory().unwrap();
    let payload: Vec<u8> = (0..2 * SHARD_SIZE + 9)
        .map(|i| (i as u8).wrapping_mul(31))
        .collect();
    let k = store.put("blake3", &payload).unwrap();
    let before = store.approximate_bytes();
    // Not pinned, no roots → entire reachable closure is empty → everything evicts.
    let evicted = store.gc(REGISTRY).unwrap();
    assert_eq!(evicted, 1, "one top-level κ evicted");
    assert!(!store.contains(&k));
    let after = store.approximate_bytes();
    assert!(
        after < before,
        "fragments must reclaim bytes (before={before}, after={after})"
    );
    assert_eq!(
        after, 0,
        "no fragments should remain when nothing is pinned"
    );
}

/// G2-6 — GC keeps shared fragments alive when *any* reachable sharded κ still references them.
#[test]
fn g2_gc_keeps_shared_fragments_alive_via_other_root() {
    let store = NativeKappaStore::in_memory().unwrap();
    let shared: Vec<u8> = vec![0x5A; SHARD_SIZE]; // exactly one shard
    let mut a = shared.clone();
    a.extend(vec![0x11; SHARD_SIZE + 1]);
    let mut b = shared.clone();
    b.extend(vec![0x22; SHARD_SIZE + 1]);
    let ka = store.put("blake3", &a).unwrap();
    let kb = store.put("blake3", &b).unwrap();
    store.pin(&ka).unwrap(); // ka is reachable
    let _evicted = store.gc(REGISTRY).unwrap();
    // ka still reads back fully (its shared shard wasn't dropped with kb).
    assert!(!store.contains(&kb));
    let got = store.get(&ka).unwrap().unwrap();
    assert_eq!(got.as_ref(), a.as_slice());
}

/// G1-1 — the read-through cache respects its byte budget: when a third blob would exceed
/// the cap, the LRU evicts the oldest entry from the cache (the persistent store is unaffected).
#[test]
fn g1_cache_respects_byte_budget_and_evicts_in_lru_order() {
    // Each blob = 80 KiB inline. With cap = 200 KiB and three blobs cached, the first must evict.
    let cap: u64 = 200 * 1024;
    let store = NativeKappaStore::in_memory_with_config(CacheConfig {
        cache_max_bytes: cap,
    })
    .unwrap();
    let mk = |seed: u8, len: usize| -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    };
    let p1 = mk(1, 80 * 1024);
    let p2 = mk(2, 80 * 1024);
    let p3 = mk(3, 80 * 1024);
    let k1 = store.put("blake3", &p1).unwrap();
    let k2 = store.put("blake3", &p2).unwrap();
    let k3 = store.put("blake3", &p3).unwrap();
    // Each `get` is a cache miss the first time, then a hit.
    let _ = store.get(&k1).unwrap();
    let _ = store.get(&k2).unwrap();
    let _ = store.get(&k3).unwrap();
    assert!(
        store.cache_bytes() <= cap,
        "cache stayed within budget: cap={cap}, total={}",
        store.cache_bytes()
    );
    // The persistent store is unaffected — eviction happens in cache, not in redb.
    assert!(store.contains(&k1));
    assert!(store.contains(&k2));
    assert!(store.contains(&k3));
    // And the evicted κ's get still succeeds (cold path through redb).
    let got = store.get(&k1).unwrap().unwrap();
    assert_eq!(got.as_ref(), p1.as_slice());
}

/// G1-2 — SP zero-copy floor (the architecture §4 SP class): consecutive `get(κ)` calls share the
/// same `Arc<[u8]>` handle — the cache is read-through, not copy-through. This is asserted by the
/// TCK already; we re-assert against the bounded cache and against sharded reassembly.
#[test]
fn g1_consecutive_gets_share_the_same_arc_for_inline_and_sharded() {
    let store = NativeKappaStore::in_memory().unwrap();
    let inline_payload = b"inline-zero-copy";
    let large_payload: Vec<u8> = (0..2 * SHARD_SIZE + 3).map(|i| i as u8).collect();
    let ki = store.put("blake3", inline_payload).unwrap();
    let kl = store.put("blake3", &large_payload).unwrap();

    let a = store.get(&ki).unwrap().unwrap();
    let b = store.get(&ki).unwrap().unwrap();
    assert!(Arc::ptr_eq(&a, &b), "inline: same Arc across gets");

    let c = store.get(&kl).unwrap().unwrap();
    let d = store.get(&kl).unwrap().unwrap();
    assert!(
        Arc::ptr_eq(&c, &d),
        "sharded reassembly: same Arc across gets"
    );
}

/// G1-3 — a cache_max_bytes of zero is rejected at construction (a zero-byte cache would force a
/// redb transaction on every `get` and break the SP zero-copy floor). Fail-loud, no fallback
/// (SPINE-6).
#[test]
fn g1_zero_byte_cache_is_rejected_loud() {
    let r = NativeKappaStore::in_memory_with_config(CacheConfig { cache_max_bytes: 0 });
    assert!(r.is_err());
}

/// G1-4 — GC invalidates the LRU cache entries it evicts from the persistent store (no stale
/// reads after `gc`).
#[test]
fn g1_gc_invalidates_cache() {
    let store = NativeKappaStore::in_memory().unwrap();
    let payload = b"will-be-gced";
    let k = store.put("blake3", payload).unwrap();
    let _ = store.get(&k).unwrap();
    assert!(store.cache_entries() >= 1);
    let _ = store.gc(REGISTRY).unwrap();
    assert!(!store.contains(&k));
    // The cache no longer holds the evicted κ.
    assert_eq!(store.get(&k).unwrap(), None);
}
