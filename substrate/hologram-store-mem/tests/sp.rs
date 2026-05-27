//! SP — substrate performance floors (architecture §4). Machine-independent witnesses that the
//! reference store upholds hologram's performance contract: content-addressing is the efficiency
//! mechanism, not a tax. Mirrors the compute substrate's structural (not timing) perf proofs.

use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::KappaStore;

/// SP-1 — `get` is zero-copy: two reads of the same κ share one allocation (an `Arc` clone, not a
/// byte copy). This is the storage analog of the compute substrate's single-buffer pool.
#[test]
fn sp1_get_is_zero_copy() {
    let store = MemKappaStore::new();
    let k = store.put("blake3", &vec![7u8; 1 << 16]).unwrap(); // 64 KiB
    let a = store.get(&k).unwrap().unwrap();
    let b = store.get(&k).unwrap().unwrap();
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "two gets must return the same allocation (zero-copy), not independent copies"
    );
}

/// SP-2 — idempotent `put` does no second write: re-putting identical bytes returns the same κ and
/// leaves the *same* stored buffer in place (spec §10.2; no duplicate storage).
#[test]
fn sp2_idempotent_put_does_not_rewrite() {
    let store = MemKappaStore::new();
    let bytes = vec![3u8; 4096];
    let k1 = store.put("blake3", &bytes).unwrap();
    let before = store.get(&k1).unwrap().unwrap();
    let k2 = store.put("blake3", &bytes).unwrap();
    let after = store.get(&k2).unwrap().unwrap();
    assert_eq!(k1, k2);
    assert_eq!(store.approximate_count(), 1);
    assert!(
        std::sync::Arc::ptr_eq(&before, &after),
        "idempotent put must not replace the stored buffer"
    );
}

/// SP-3 — reachability GC is bounded by the reachable set, not total storage: a large pinned cone
/// is fully retained and an equally-large unreachable set is fully reclaimed in one pass.
#[test]
fn sp3_gc_is_bounded_by_reachable_set() {
    use hologram_realizations::{ContainerManifest, REGISTRY};
    use hologram_substrate_core::Realization;
    let store = MemKappaStore::new();

    // 300 reachable leaves behind 100 pinned manifests; 300 unreachable orphans.
    for i in 0..100u32 {
        let c = store.put("blake3", &[i as u8, 0]).unwrap();
        let s = store.put("blake3", &[i as u8, 1]).unwrap();
        let p = store.put("blake3", &[i as u8, 2]).unwrap();
        let m = ContainerManifest {
            code: c,
            initial_state: s,
            parameters: p,
        };
        let mk = store.put("blake3", &m.canonicalize()).unwrap();
        store.pin(&mk).unwrap();
    }
    for i in 0..300u32 {
        store.put("blake3", &i.to_le_bytes()).unwrap();
    }
    let reclaimed = store.gc(REGISTRY);
    assert_eq!(
        reclaimed, 300,
        "exactly the unreachable orphans are reclaimed"
    );
    // 100 manifests + 300 operand leaves remain reachable.
    assert_eq!(store.approximate_count(), 400);
}
