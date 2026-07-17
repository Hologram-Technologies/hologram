//! Parser hardening (spec `refactor/03` §Parser hardening — a **standing requirement**).
//!
//! The realization decoders parse **network-supplied bytes**, so they must never panic on hostile
//! input: every length is bounds-checked against the declared size, with no unbounded allocation.
//! This deterministic mutation suite feeds each P4–P6 decoder (and the generic registry dispatch —
//! the true network entry point) a corpus of truncations, byte mutations, and pseudo-random noise,
//! asserting each returns `Ok`/`Err` and **never panics**. It is CI-permanent (runs in the normal
//! `cargo test`), the always-green complement to out-of-tree cargo-fuzz targets.

use hologram_space::{
    address_bytes, references, AppManifest, AttestationKey, AuditEvent, CapabilitySet,
    KappaLabel71, Layer, LifecycleTransition, Network, NetworkTier, Realization, RevocationEvent,
    REGISTRY,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Deterministic xorshift64 — a reproducible corpus without `rand`/`Date`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

fn k(seed: &[u8]) -> KappaLabel71 {
    address_bytes(seed)
}

/// Assert `decode` never panics across a mutation corpus derived from `seed`.
fn assert_panic_free(label: &str, seed: &[u8], decode: impl Fn(&[u8])) {
    let run = |input: &[u8]| catch_unwind(AssertUnwindSafe(|| decode(input))).is_ok();

    // 1) The valid form and every truncated prefix (off-by-one / short-read hunting).
    for n in 0..=seed.len() {
        assert!(
            run(&seed[..n]),
            "{label} panicked on truncated prefix of len {n}"
        );
    }
    // 2) Single-byte mutations at every offset — flips length prefixes, counts, tags, IRIs.
    for i in 0..seed.len() {
        for delta in [0x01u8, 0x3f, 0x80, 0xff] {
            let mut m = seed.to_vec();
            m[i] = m[i].wrapping_add(delta);
            assert!(run(&m), "{label} panicked on byte mutation at offset {i}");
        }
    }
    // 3) Pseudo-random noise of varied lengths (arbitrary bytes off the wire).
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.len() as u64);
    for len in [0usize, 1, 5, 71, 137, 512, 4096] {
        for _ in 0..64 {
            let buf: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
            assert!(run(&buf), "{label} panicked on random input of len {len}");
        }
    }
}

fn app_manifest_seed() -> Vec<u8> {
    AppManifest {
        primary: Some(0),
        requires: k(b"caps"),
        layers: vec![
            Layer::wasm(k(b"w"), "_start"),
            Layer::tensor(k(b"t"), "sess"),
            Layer::rootfs(k(b"r"), "boot", "riscv64"),
            Layer::view(k(b"v"), "portable"),
        ],
        children: vec![(k(b"child-app"), k(b"child-caps"))],
    }
    .canonicalize()
}

fn network_seed() -> Vec<u8> {
    Network {
        membership: vec![k(b"a"), k(b"b")],
        policy: k(b"p"),
        parent: Some(k(b"parent")),
        tier: NetworkTier::Private,
        key_ref: Some(k(b"net-key")), // both tail optionals present exercises the flags decode
    }
    .canonicalize()
}

#[test]
fn app_manifest_decoder_never_panics() {
    let seed = app_manifest_seed();
    assert_panic_free("AppManifest::decode", &seed, |b| {
        let _ = AppManifest::decode(b);
    });
    assert_panic_free("AppManifest::references", &seed, |b| {
        let _ = AppManifest::references(b);
    });
}

#[test]
fn network_decoder_never_panics() {
    let seed = network_seed();
    assert_panic_free("Network::decode", &seed, |b| {
        let _ = Network::decode(b);
    });
}

#[test]
fn attestation_key_decoder_never_panics() {
    let seed = AttestationKey::new(0, b"public-key-material".to_vec()).canonicalize();
    assert_panic_free("AttestationKey::decode", &seed, |b| {
        let _ = AttestationKey::decode(b);
    });
}

#[test]
fn audit_event_decoder_never_panics() {
    let seed =
        AuditEvent::record(LifecycleTransition::Spawn, k(b"subj"), Some(k(b"prev"))).canonicalize();
    assert_panic_free("AuditEvent::transition_of", &seed, |b| {
        let _ = AuditEvent::transition_of(b);
    });
}

#[test]
fn revocation_decoder_never_panics() {
    // `decode` is network-facing (the `is_revoked` walk decodes store bytes that may be foreign).
    let seed = RevocationEvent {
        revoked_key: k(b"revoked"),
        revoker_key: k(b"revoker"),
        predecessor: Some(k(b"prev")),
        reason: 3,
        signature: vec![0xAB; 64],
    }
    .canonicalize();
    assert_panic_free("RevocationEvent::decode", &seed, |b| {
        let _ = RevocationEvent::decode(b);
    });
}

#[test]
fn capability_set_decoder_never_panics() {
    // Exercise the existing budget/group decoder too — it splits refs by payload-declared counts.
    let seed = CapabilitySet::new(hologram_space::Capabilities {
        storage_roots: vec![k(b"s0"), k(b"s1")],
        storage_quota_bytes: 1000,
        network_fetch: true,
        network_announce: false,
        publish_channels: vec![k(b"pub")],
        subscribe_channels: vec![],
        memory_max_bytes: 2000,
        cpu_time_per_event_ms: 10,
        priority_weight: 3,
    })
    .canonicalize();
    assert_panic_free("CapabilitySet::to_capabilities", &seed, |b| {
        let _ = CapabilitySet::to_capabilities(b);
    });
}

#[test]
fn generic_registry_dispatch_never_panics_on_arbitrary_bytes() {
    // The store's single network entry point: `references(bytes, REGISTRY)` dispatches by embedded
    // IRI over hostile bytes. Feed it every realization's mutated form + pure noise.
    for seed in [app_manifest_seed(), network_seed()] {
        assert_panic_free("references(_, REGISTRY)", &seed, |b| {
            let _ = references(b, REGISTRY);
        });
    }
}

#[test]
fn forged_oversized_counts_error_not_allocate() {
    // A manifest whose declared layer count is enormous must be rejected by the bounds check, never
    // trigger an unbounded allocation (spec 03: bounds-checked; no unbounded inflation). The layout
    // is `IRI ‖ 0x00 ‖ u32(n_refs) ‖ refs ‖ u32(payload_len) ‖ payload`, and the payload begins
    // `primary:u32 ‖ n_layers:u32 ‖ …`; forge n_layers to u32::MAX.
    let seed = app_manifest_seed();
    let iri = b"https://hologram.foundation/realization/app-manifest";
    let nul = iri.len();
    let n_refs = u32::from_le_bytes(seed[nul + 1..nul + 5].try_into().unwrap()) as usize;
    let payload_len_at = nul + 1 + 4 + n_refs * 71;
    let payload_at = payload_len_at + 4;
    let n_layers_at = payload_at + 4; // after the primary u32
    let mut forged = seed.clone();
    forged[n_layers_at..n_layers_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    // Must be a clean error, and must not have panicked/aborted getting here.
    assert!(
        AppManifest::decode(&forged).is_err(),
        "forged huge n_layers must be rejected"
    );
}
