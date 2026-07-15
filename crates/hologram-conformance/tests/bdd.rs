//! Cucumber runner. Discovers every `.feature` under `features/suites`.
//!
//! Pending scenarios (no matching steps) are reported as skipped and do NOT fail
//! the run. As each phase (P0–P6) implements a suite, add its step definitions and
//! enable `.fail_on_skipped()` for that suite's tag (see features/README.md).
use hologram_conformance::ConformanceWorld;

use cucumber::{given, then, when, World};
use hologram_space::{address_bytes, verify_kappa, Capabilities, Realization};
use hologram_space::{CapabilitySet, ContainerManifest};
use hologram_spike_sp3::{Client, SpikeSpace};
use hologram_tck::MemKappaStore;

#[given("the conformance harness is wired")]
fn harness_wired(_w: &mut ConformanceWorld) {}

#[then("it runs at least one scenario")]
fn runs_one(_w: &mut ConformanceWorld) {}

// ───────────────────────────── GV-1 — R1 traceability ─────────────────────────────
// A ContainerManifest realization embeds its operand κs (SPINE-2/3); `references()`
// is the inverse projection recovering exactly those operands — the full provenance
// closure, with no side tables. Witnessed against `hologram-realizations`.

#[given("a new realization built from known operand κs")]
fn gv1_build(w: &mut ConformanceWorld) {
    let code = address_bytes(b"gv1-code-module");
    let initial_state = address_bytes(b"gv1-initial-state");
    let parameters = address_bytes(b"gv1-parameters");
    w.operand_kappas = vec![
        code.as_bytes().to_vec(),
        initial_state.as_bytes().to_vec(),
        parameters.as_bytes().to_vec(),
    ];
    let manifest = ContainerManifest {
        code,
        initial_state,
        parameters,
    };
    w.canonical = manifest.canonicalize();
}

#[when("I call references() on it")]
fn gv1_references(w: &mut ConformanceWorld) {
    let refs = <ContainerManifest as Realization>::references(&w.canonical)
        .expect("references() must decode a well-formed manifest");
    w.references = Some(refs.iter().map(|k| k.as_bytes().to_vec()).collect());
}

#[then("the returned set equals the full provenance closure with no side tables")]
fn gv1_assert(w: &mut ConformanceWorld) {
    let refs = w
        .references
        .as_ref()
        .expect("references() must have been called by the When step");
    assert_eq!(
        refs, &w.operand_kappas,
        "references() must yield exactly the embedded operand κs — no more (no side tables), no fewer"
    );
}

// ───────────────── LAW-3 — contracts are hologram's, spaces are anyone's ─────────────────
// The `hologram-spike-sp3` crate is a *separate* crate that implements the `hologram-space`
// contract using only its public API — exactly what an external repo's space does. `Client`
// (generic over `Space`) accepts it with no special-casing. Sealed traits or crate-private
// seams would make this fail to compile, so it compiling + running is the witness (D21).

#[given("the hologram-space contract with no sealed traits or crate-private seams")]
fn law3_given(_w: &mut ConformanceWorld) {}

#[when("a space is implemented in an external repository")]
fn law3_impl(w: &mut ConformanceWorld) {
    // Downstream type (`SpikeSpace`, from another crate) accepted by the generic `Client`.
    let client = Client::new(SpikeSpace::new());
    // Reaching a contract-mediated operation proves the impl satisfies the trait bounds.
    w.law3_accepted = !client.compile().is_empty();
}

#[then("it compiles against the published crates and is accepted with no in-tree privilege")]
fn law3_assert(w: &mut ConformanceWorld) {
    assert!(
        w.law3_accepted,
        "a downstream crate must implement the space contract and be accepted by Client \
         with no privileged access — the contract is open (D21)"
    );
}

// ───────────────────── LAW-1 — SPINE-1: canonical bytes or nothing ─────────────────────
// A realization's identity IS the σ-axis address of its canonical bytes — there is no
// identity without canonical bytes. Identity is never trusted: it is verified by
// re-derivation. Authentic bytes re-derive to the κ (true); any tampering fails (false).
// Witnessed against `hologram-substrate-core::verify_kappa` + a `ContainerManifest`.

#[given("a realization addressed only by its canonical bytes")]
fn law1_given(w: &mut ConformanceWorld) {
    let manifest = ContainerManifest {
        code: address_bytes(b"law1-code"),
        initial_state: address_bytes(b"law1-state"),
        parameters: address_bytes(b"law1-params"),
    };
    w.canonical = manifest.canonicalize();
}

#[when("its identity is checked")]
fn law1_check(w: &mut ConformanceWorld) {
    let kappa = address_bytes(&w.canonical);
    let authentic = verify_kappa(&w.canonical, &kappa).expect("verify authentic");
    let mut tampered_bytes = w.canonical.clone();
    tampered_bytes[0] ^= 0xff; // flip a byte — no longer the canonical form
    let tampered = verify_kappa(&tampered_bytes, &kappa).expect("verify tampered");
    w.law1_verify = Some((authentic, tampered));
}

#[then("re-derivation of the canonical bytes verifies, and any tampering is rejected")]
fn law1_assert(w: &mut ConformanceWorld) {
    let (authentic, tampered) = w
        .law1_verify
        .expect("the When step must have checked identity by re-derivation");
    assert!(
        authentic,
        "authentic canonical bytes must re-derive to the κ (SPINE-1)"
    );
    assert!(
        !tampered,
        "tampered bytes must fail re-derivation — identity is never trusted, only re-derived"
    );
}

// ─────────────────────── SP-3 — space composition (P0.5 spike) ───────────────────────
// A `Client` over the `Space` contract drives compile→store→boot: a synchronous compile,
// a synchronous store, and the async network/boot seam calling into synchronous compute.
// The async `when` awaits `boot` directly — the one async↔sync boundary (LAW-4).
// Witnessed against `hologram-spike-sp3` (the SP-3 slice).

#[given("a Client over a space with a synchronous store and an async network seam")]
fn sp3_given(_w: &mut ConformanceWorld) {}

#[when("it drives compile then store then boot")]
async fn sp3_run(w: &mut ConformanceWorld) {
    let client = Client::new(SpikeSpace::new());
    let holo = client.compile();
    let kappa = client.store_holo(&holo).expect("store the compiled .holo");
    let vals: [i64; 4] = [0, 42, -7, 1024];
    let mut input = Vec::new();
    for &v in &vals {
        input.extend_from_slice(&v.to_le_bytes());
    }
    // `boot` is async (the network/boot seam) and internally runs the sync compute.
    w.sp3_output = Some(client.boot(&kappa, &input).await);
}

#[then("the workload runs end to end through the async-to-sync boundary")]
fn sp3_assert(w: &mut ConformanceWorld) {
    let out = w
        .sp3_output
        .as_ref()
        .expect("the When step must have driven compile→store→boot");
    assert_eq!(
        out,
        &vec![0.0, 42.0, -7.0, 1024.0],
        "compile→store→boot must compute the i64→f32 cast — the slice composes async \
         storage/boot with sync compute end to end"
    );
}

// ───────────────────── SP-1 — passing the TCK is conformance ─────────────────────
// The `hologram-tck` battery is the single definition of conformance, run identically
// against every backend (substrate-tripling). Witnessed against the reference store:
// if the shared battery passes, the store is conformant (`store_battery` panics on the
// first violation). Same pattern as GV-1 — the invariant witnessed against a reference.

#[given("a space implementing the hologram-space traits")]
fn sp1_given(_w: &mut ConformanceWorld) {}

#[when("it runs the hologram-tck battery")]
fn sp1_run(w: &mut ConformanceWorld) {
    let store = MemKappaStore::new();
    hologram_tck::store_battery(&store);
    // Reached only if every battery assertion held.
    w.sp1_tck_passed = true;
}

#[then("passing the TCK is the definition of conformance")]
fn sp1_assert(w: &mut ConformanceWorld) {
    assert!(
        w.sp1_tck_passed,
        "the reference store must pass the shared hologram-tck battery — passing the \
         TCK is the definition of conformance"
    );
}

// ─────────────────── MG-5 — κ-stability golden vectors (P1 preflight) ───────────────────
// Frozen reference κs captured 2026-07-14 against the current substrate. Ground rule 5:
// crate moves/renames MUST NOT change any canonical byte form or κ — every phase re-derives
// these and must match bit-for-bit. A mismatch is a κ break (a versioned format change,
// never a move). Witnessed against hologram-substrate-core + hologram-realizations.

const GV_SIGMA_1_KAPPA: &str =
    "blake3:6d3fd64e3ed30b2904d0dfe85ded905fe7bcf88d4c872156ec1729ac1b741745";
const GV_SIGMA_2_KAPPA: &str =
    "blake3:1669bd4584b6af8519f71e9f9116a4ad6aaae70950a70787ee5718c00558e18d";
const GV_MANIFEST_KAPPA: &str =
    "blake3:2d4f5ff9117227d79c5cd31d6774af96246896bd80a013d69b20c714246a7224";
const GV_CAPS_KAPPA: &str =
    "blake3:efd7908e447824e02df07049a68f6c5018663484bf87e63d5a17e4ae43ce02b0";

fn golden_manifest() -> ContainerManifest {
    ContainerManifest {
        code: address_bytes(b"gv-code-module"),
        initial_state: address_bytes(b"gv-initial-state"),
        parameters: address_bytes(b"gv-parameters"),
    }
}

fn golden_caps() -> CapabilitySet {
    CapabilitySet::new(Capabilities {
        storage_roots: vec![address_bytes(b"gv-root")],
        publish_channels: Vec::new(),
        subscribe_channels: Vec::new(),
        storage_quota_bytes: 1024,
        memory_max_bytes: 4096,
        cpu_time_per_event_ms: 10,
        priority_weight: 1,
        network_fetch: true,
        network_announce: false,
    })
}

#[given("frozen golden vectors for the σ-axis and the realization canonical forms")]
fn mg5_given(_w: &mut ConformanceWorld) {}

#[when("each is re-derived from the same inputs")]
fn mg5_rederive(w: &mut ConformanceWorld) {
    let sigma1 = address_bytes(b"hologram-golden-vector/sigma/1").to_string();
    let sigma2 = address_bytes(b"hologram-golden-vector/sigma/2").to_string();
    let manifest = address_bytes(&golden_manifest().canonicalize()).to_string();
    let caps = address_bytes(&golden_caps().canonicalize()).to_string();
    w.mg5_stable = sigma1 == GV_SIGMA_1_KAPPA
        && sigma2 == GV_SIGMA_2_KAPPA
        && manifest == GV_MANIFEST_KAPPA
        && caps == GV_CAPS_KAPPA;
}

#[then("every vector yields its frozen κ, bit-for-bit")]
fn mg5_assert(w: &mut ConformanceWorld) {
    assert!(
        w.mg5_stable,
        "a golden vector re-derived to a different κ — canonical bytes or the σ-axis \
         changed. This is a κ break (a versioned format change), never a crate move."
    );
}

#[tokio::main]
async fn main() {
    ConformanceWorld::cucumber()
        .fail_on_skipped_with(|feat, _rule, sc| {
            feat.tags
                .iter()
                .chain(sc.tags.iter())
                .any(|t| t.trim_start_matches('@') == "status:enforced")
        })
        .run_and_exit(hologram_conformance::SUITES_DIR)
        .await;
}
