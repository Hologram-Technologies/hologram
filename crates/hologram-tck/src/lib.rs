#![cfg_attr(not(feature = "std"), no_std)]
//! # hologram-tck — Technology Compatibility Kit
//!
//! The reusable [`KappaStore`] conformance battery. The reference in-memory store
//! ([`MemKappaStore`]) now lives in `hologram-space` (with the trait it implements) and is
//! re-exported here for conformance authors — runtime consumers reach it without a normal
//! dependency on this test kit.
//! **Every** storage backend (mem, native, OPFS, bare-metal) runs the *same* battery, so
//! conformance is defined once and validated against each substrate identically (the TR
//! substrate-tripling discipline at the trait level).
//!
//! These assert the trait-level ST/SPINE invariants that hold for *any* backend (idempotency,
//! eviction-tolerant `get`, fail-loud unknown axis, pin/unpin, content round-trip). Backend-specific
//! properties (zero-copy `Arc` identity, reachability `gc`) are tested by each backend directly,
//! since they are not trait methods.

extern crate alloc;

/// The reference in-memory [`KappaStore`] — the conformance oracle every real backend is
/// differentially compared against. Re-exported from `hologram-space` (its home), so
/// `hologram_tck::MemKappaStore` keeps resolving for conformance authors.
pub use hologram_space::MemKappaStore;

use hologram_space::{address_bytes, KappaStore, StoreError};

/// Run the full trait-level conformance battery against `store`. Panics on the first violation
/// (intended to be called from a `#[test]`). The store must start empty for the count assertions.
pub fn store_battery(store: &dyn KappaStore) {
    idempotency(store);
    eviction_tolerant_get(store);
    unknown_axis_fails_loud(store);
    pin_unpin(store);
    content_roundtrip(store);
    axis_polymorphic_round_trip(store);
    zero_copy_get_returns_arc_handle(store);
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
/// uor-addr 0.2.0 ships blake3 / sha256 / sha3-256 / keccak256 / sha512; any axis outside that
/// registry must Err. `md5` is the canonical out-of-registry example.
pub fn unknown_axis_fails_loud(store: &dyn KappaStore) {
    assert!(
        store.put("md5", b"tck").is_err(),
        "an unsupported σ-axis must fail loud, not fall back"
    );
    assert!(
        store.put_axis("md5", b"tck").is_err(),
        "axis-polymorphic surface also fails loud on an unsupported σ-axis"
    );
}

/// AS — axis-polymorphic stored content (architecture §3.1 G-B1). The reference backend opts in to
/// `put_axis`/`get_axis` for all five uor-addr-supported axes; a foreign-axis κ stored on this
/// substrate must round-trip identically to bytes received from any other axis-compliant peer.
pub fn axis_polymorphic_round_trip(store: &dyn KappaStore) {
    let bytes = b"axis-polymorphic-tck-fixture".as_slice();
    for axis in &["blake3", "sha256", "sha3-256", "keccak256", "sha512"] {
        match store.put_axis(axis, bytes) {
            Ok(label) => {
                assert!(
                    store.contains_axis(&label),
                    "{axis}: contains_axis after put_axis"
                );
                let got = store.get_axis(&label).unwrap().unwrap();
                assert_eq!(
                    got.as_ref(),
                    bytes,
                    "{axis}: get_axis round-trips the stored bytes"
                );
            }
            // Backend may opt out — UnknownAxis is the documented default. But for the reference
            // store-mem battery, this branch must NOT be taken for any of the five axes.
            Err(StoreError::UnknownAxis) => {}
            Err(e) => panic!("{axis}: put_axis raised an unexpected error: {:?}", e),
        }
    }
}

/// SP — `get` returns a cheap `Arc<[u8]>` handle; consecutive `get`s of the same κ share storage
/// (no per-call byte copy). The architecture §4 SP zero-copy floor: get is O(1) bookkeeping over
/// the same buffer, **never a memcpy of the payload**. Verified by Arc-pointer identity.
pub fn zero_copy_get_returns_arc_handle(store: &dyn KappaStore) {
    let k = store.put("blake3", b"tck-zero-copy-sentinel").unwrap();
    let a = store.get(&k).unwrap().unwrap();
    let b = store.get(&k).unwrap().unwrap();
    assert!(
        alloc::sync::Arc::ptr_eq(&a, &b),
        "consecutive get(κ) calls must share the same Arc — zero-copy SP floor"
    );
    // The content matches what we put — sanity, not the zero-copy assertion.
    assert_eq!(a.as_ref(), b"tck-zero-copy-sentinel");
}

/// ST / spec §5.3 — pin establishes a root, unpin removes it; unpin of a non-pinned κ is an error.
pub fn pin_unpin(store: &dyn KappaStore) {
    let k = store.put("blake3", b"tck-pin").unwrap();
    store.pin(&k).unwrap();
    assert!(store.pinned_roots().iter().any(|r| r == &k));
    store.unpin(&k).unwrap();
    assert!(!store.pinned_roots().iter().any(|r| r == &k));
    assert!(
        store.unpin(&k).is_err(),
        "unpin of a non-pinned κ must error (NotPinned)"
    );
}

/// Content integrity — bytes put are bytes got, and the κ re-derives to itself.
pub fn content_roundtrip(store: &dyn KappaStore) {
    let payload = b"tck-roundtrip-payload-0123456789";
    let k = store.put("blake3", payload).unwrap();
    let got = store.get(&k).unwrap().expect("present after put");
    assert_eq!(got.as_ref(), payload);
    assert!(hologram_space::verify_kappa(got.as_ref(), &k).unwrap());
}
