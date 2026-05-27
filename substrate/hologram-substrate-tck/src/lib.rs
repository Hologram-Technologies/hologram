#![cfg_attr(not(feature = "std"), no_std)]
//! # hologram-substrate-tck — Test Conformance Kit
//!
//! The reusable [`KappaStore`] conformance battery. **Every** storage backend (mem, native,
//! OPFS, bare-metal) runs the *same* battery, so conformance is defined once and validated against
//! each substrate identically (the TR substrate-tripling discipline at the trait level).
//!
//! These assert the trait-level ST/SPINE invariants that hold for *any* backend (idempotency,
//! eviction-tolerant `get`, fail-loud unknown axis, pin/unpin, content round-trip). Backend-specific
//! properties (zero-copy `Arc` identity, reachability `gc`) are tested by each backend directly,
//! since they are not trait methods.

extern crate alloc;

use hologram_substrate_core::{address_bytes, KappaStore};

/// Run the full trait-level conformance battery against `store`. Panics on the first violation
/// (intended to be called from a `#[test]`). The store must start empty for the count assertions.
pub fn store_battery(store: &dyn KappaStore) {
    idempotency(store);
    eviction_tolerant_get(store);
    unknown_axis_fails_loud(store);
    pin_unpin(store);
    content_roundtrip(store);
}

/// ST-2 / spec §10.2 — identical `(axis, bytes)` ⇒ identical κ-label, no duplicate write.
pub fn idempotency(store: &dyn KappaStore) {
    let before = store.approximate_count();
    let a = store.put("blake3", b"tck-idempotency").unwrap();
    let b = store.put("blake3", b"tck-idempotency").unwrap();
    assert_eq!(a, b, "idempotent put must return the same κ-label");
    assert_eq!(
        store.approximate_count(),
        before + 1,
        "idempotent put must not duplicate storage"
    );
}

/// SPINE-5 / spec §5.2 — a κ that was never stored is `Ok(None)` (eviction-tolerant), not an error;
/// there is no delete primitive (the trait has no `delete` — compile-time guarantee, §10.5).
pub fn eviction_tolerant_get(store: &dyn KappaStore) {
    let absent = address_bytes(b"tck-never-stored-xyz");
    assert_eq!(store.get(&absent).unwrap(), None);
    assert!(!store.contains(&absent));
}

/// SPINE-6 — an unwired σ-axis is rejected, never silently coerced (no fallback).
pub fn unknown_axis_fails_loud(store: &dyn KappaStore) {
    assert!(
        store.put("sha256", b"tck").is_err(),
        "an unsupported σ-axis must fail loud, not fall back to blake3"
    );
}

/// ST / spec §5.3 — pin establishes a root, unpin removes it; unpin of a non-pinned κ is an error.
pub fn pin_unpin(store: &dyn KappaStore) {
    let k = store.put("blake3", b"tck-pin").unwrap();
    store.pin(&k).unwrap();
    assert!(store.pinned_roots().iter().any(|r| r == &k));
    store.unpin(&k).unwrap();
    assert!(!store.pinned_roots().iter().any(|r| r == &k));
    assert!(store.unpin(&k).is_err(), "unpin of a non-pinned κ must error (NotPinned)");
}

/// Content integrity — bytes put are bytes got, and the κ re-derives to itself.
pub fn content_roundtrip(store: &dyn KappaStore) {
    let payload = b"tck-roundtrip-payload-0123456789";
    let k = store.put("blake3", payload).unwrap();
    let got = store.get(&k).unwrap().expect("present after put");
    assert_eq!(got.as_ref(), payload);
    assert!(hologram_substrate_core::verify_kappa(got.as_ref(), &k).unwrap());
}
