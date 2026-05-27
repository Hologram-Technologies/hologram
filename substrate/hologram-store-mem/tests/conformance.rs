//! Conformance witnesses for the Hologram deployment substrate — each validated against an
//! **external authority**, never self-reference (see specs/docs/container-substrate-vv.md).
//!
//! Classes: AS (σ-axis vs BLAKE3 reference), ST (KappaStore idempotency / append-only /
//! reachability eviction), RZ (references() inverse projection), SPINE (uor-native invariants).

use hologram_realizations::{ContainerManifest, REGISTRY};
use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::{address_bytes, verify_kappa, KappaStore, Realization};

/// Format a 32-byte digest as the canonical `blake3:<64 hex>` κ-label string.
fn blake3_label_str(digest: &[u8; 32]) -> String {
    let mut s = String::from("blake3:");
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ───────────────────────────── AS — σ-axis vs the BLAKE3 reference ─────────────────────────────

#[test]
fn as_sigma_axis_matches_independent_blake3_reference() {
    // External authority: the upstream `blake3` crate (the algorithm authors' impl).
    for input in [&b""[..], b"hologram", b"the quick brown fox", &[0u8; 4096][..]] {
        let reference = blake3::hash(input);
        let expected = blake3_label_str(reference.as_bytes());
        let ours = address_bytes(input);
        assert_eq!(
            ours.as_str(),
            expected,
            "substrate σ-axis must equal the BLAKE3 reference for {} bytes",
            input.len()
        );
        // SPINE-4: verification is re-derivation through the σ-axis.
        assert!(verify_kappa(input, &ours).unwrap());
    }
}

#[test]
fn spine4_verify_rejects_tampered_bytes() {
    let k = address_bytes(b"authentic");
    assert!(!verify_kappa(b"tampered!", &k).unwrap());
}

// ───────────────────────────── ST — KappaStore semantics ─────────────────────────────

#[test]
fn st2_put_is_idempotent_no_duplicate_write() {
    // spec §10.2: same bytes ⇒ same κ, no second write.
    let store = MemKappaStore::new();
    let a = store.put("blake3", b"payload").unwrap();
    let b = store.put("blake3", b"payload").unwrap();
    assert_eq!(a, b);
    assert_eq!(store.approximate_count(), 1, "idempotent put must not duplicate storage");
}

#[test]
fn st_get_absent_is_none_not_error_and_no_delete_primitive() {
    // SPINE-5: local absence is None (eviction-tolerant), not an error; there is no delete.
    let store = MemKappaStore::new();
    let absent = address_bytes(b"never-stored");
    assert_eq!(store.get(&absent).unwrap(), None);
    assert!(!store.contains(&absent));
    // (No `delete` method exists on the KappaStore trait — append-only surface, §10.5.)
}

#[test]
fn st_unknown_axis_fails_loud_no_fallback() {
    // SPINE-6: an unwired σ-axis is rejected, never silently coerced to blake3.
    let store = MemKappaStore::new();
    assert!(store.put("sha256", b"x").is_err());
}

#[test]
fn st_pin_unpin_roundtrip() {
    let store = MemKappaStore::new();
    let k = store.put("blake3", b"pinme").unwrap();
    store.pin(&k).unwrap();
    assert_eq!(store.pinned_roots(), vec![k]);
    store.unpin(&k).unwrap();
    assert!(store.pinned_roots().is_empty());
    assert!(store.unpin(&k).is_err(), "unpin of a non-pinned κ is NotPinned");
}

// ───────────────────────────── ST/§10.8 + RZ — reachability eviction ─────────────────────────────

#[test]
fn st10_8_gc_retains_reachable_evicts_unreachable() {
    let store = MemKappaStore::new();

    // Three leaf operands + a manifest that composes them (its canonical form embeds their κ).
    let code = store.put("blake3", b"wasm-module-bytes").unwrap();
    let state = store.put("blake3", b"initial-state").unwrap();
    let params = store.put("blake3", b"params").unwrap();
    let manifest = ContainerManifest { code, initial_state: state, parameters: params };
    let manifest_bytes = manifest.canonicalize();
    let manifest_k = store.put("blake3", &manifest_bytes).unwrap();

    // An unrelated blob, reachable from nothing.
    let orphan = store.put("blake3", b"orphan-garbage").unwrap();

    // Pin only the manifest. GC must keep the manifest + its three operands (recovered via the
    // registry's references() inverse projection) and evict the orphan — never a reachable κ.
    store.pin(&manifest_k).unwrap();
    let evicted = store.gc(REGISTRY);

    assert_eq!(evicted, 1, "exactly the orphan is evicted");
    assert!(store.contains(&manifest_k));
    assert!(store.contains(&code) && store.contains(&state) && store.contains(&params));
    assert!(!store.contains(&orphan));
}

#[test]
fn rz_references_inverse_projection_is_exact() {
    // RZ/§10.10: references() recovers exactly the operands the canonical form embedded.
    let code = address_bytes(b"c");
    let state = address_bytes(b"s");
    let params = address_bytes(b"p");
    let m = ContainerManifest { code, initial_state: state, parameters: params };
    let refs = ContainerManifest::references(&m.canonicalize()).unwrap();
    assert_eq!(refs, vec![code, state, params]);
}

/// The in-memory reference passes the shared TCK identically to every other backend (TR).
#[test]
fn mem_passes_the_kappastore_tck() {
    hologram_substrate_tck::store_battery(&MemKappaStore::new());
}
