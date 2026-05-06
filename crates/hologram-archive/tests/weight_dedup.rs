//! Weight content-addressing (spec X.3): identical bodies share storage,
//! distinct bodies are keyed by separate BLAKE3 fingerprints.

use hologram_archive::{WeightStore, WeightFingerprint};

#[test]
fn identical_bytes_share_storage() {
    let mut store = WeightStore::new();
    let a = store.insert(vec![1u8, 2, 3, 4]);
    let b = store.insert(vec![1u8, 2, 3, 4]);
    assert_eq!(a, b);
    assert_eq!(store.len(), 1);
}

#[test]
fn distinct_bytes_get_distinct_fingerprints() {
    let mut store = WeightStore::new();
    let a = store.insert(vec![1u8, 2, 3]);
    let b = store.insert(vec![1u8, 2, 4]);
    assert_ne!(a, b);
    assert_eq!(store.len(), 2);
}

#[test]
fn fingerprint_roundtrips_via_get() {
    let mut store = WeightStore::new();
    let body = vec![9u8, 8, 7, 6, 5, 4, 3, 2, 1];
    let fp = store.insert(body.clone());
    assert_eq!(store.get(fp), Some(body.as_slice()));
}

#[test]
fn fingerprint_is_deterministic_blake3() {
    // Stable across calls — used for cross-compilation memoization.
    let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let f1 = WeightFingerprint::of(&bytes);
    let f2 = WeightFingerprint::of(&bytes);
    assert_eq!(f1, f2);
}
