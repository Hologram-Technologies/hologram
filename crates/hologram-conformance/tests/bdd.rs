//! Cucumber runner. Discovers every `.feature` under `features/suites`.
//!
//! Pending scenarios (no matching steps) are reported as skipped and do NOT fail
//! the run. As each phase (P0–P6) implements a suite, add its step definitions and
//! enable `.fail_on_skipped()` for that suite's tag (see features/README.md).
use hologram_conformance::ConformanceWorld;

use cucumber::{given, then, when, World};
// The **real** facade `Client`, driven over `SpikeSpace` — an external-crate `Space` impl —
// so LAW-3 (open contract) and SP-3 (composition) witness the shipping surface, not a spike.
use hologram::Client;
use hologram_space::{address_bytes, verify_kappa, Capabilities, Realization};
use hologram_space::{CapabilitySet, ContainerManifest};
// SP-4/SP-5: the reference HAL + Surface seams, exercised directly through the contract's
// public API (external witness: the shipping reference impls in `hologram-space`).
use hologram_space::{
    Clock, Entropy, Intent, ManualClock, NoopSpawner, NullSurface, SeededEntropy, Spawner, Surface,
    SurfaceError,
};
use hologram_spike_sp3::SpikeSpace;
use hologram_tck::MemKappaStore;

/// The smallest graph that computes: an i64→f32 cast of a rank-1 tensor — the SP-3 workload.
fn cast_graph() -> hologram::graph::Graph {
    use hologram::graph::node::Node;
    use hologram::graph::registry::{DTypeId, ShapeDescriptor};
    use hologram::graph::{Graph, GraphOp, InputSource, OpKind};
    use smallvec::SmallVec;
    const DTYPE_F32: u8 = 8;
    const DTYPE_I64: u8 = 5;
    let mut graph = Graph::new();
    let sh = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let inp = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I64),
        output_shape: sh,
    });
    graph.add_input(inp);
    let cast = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Cast),
        inputs: SmallVec::from_iter([InputSource::Node(inp)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(cast)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    graph.add_output(out);
    graph
}

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
    w.law3_accepted = client.compile(cast_graph()).is_ok();
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
    let holo = client.compile(cast_graph()).expect("compile the workload");
    let kappa = client
        .provision(&holo)
        .expect("provision the compiled .holo");
    let vals: [i64; 4] = [0, 42, -7, 1024];
    let mut input = Vec::new();
    for &v in &vals {
        input.extend_from_slice(&v.to_le_bytes());
    }
    // `run` is async (resolve, the network/boot seam) and internally runs the sync compute.
    let outputs = client
        .run(&kappa, &[input.as_slice()])
        .await
        .expect("run the workload");
    let cast: Vec<f32> = outputs[0]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    w.sp3_output = Some(cast);
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

// ───────────────────── SP-4 — deterministic HAL seams (spec 02 §4) ─────────────────────
// The reference Entropy/Clock/Spawner are the hermetic-V&V seams: equally-seeded entropy
// reproduces the same stream, `ManualClock` advances only when told, and `NoopSpawner` drops
// the future it is handed. All three are what make conformance runs reproducible. Witnessed
// against `hologram-space`'s reference impls through their public API.

#[given("the reference Entropy, Clock, and Spawner seams")]
fn sp4_given(_w: &mut ConformanceWorld) {}

#[when(
    "entropy is drawn from two equally-seeded sources, the clock is advanced, and a background task is spawned"
)]
fn sp4_exercise(w: &mut ConformanceWorld) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // Entropy: two sources with the same seed must produce byte-identical, non-trivial streams.
    const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
    let (mut a, mut b) = ([0u8; 32], [0u8; 32]);
    SeededEntropy::new(SEED).fill(&mut a);
    SeededEntropy::new(SEED).fill(&mut b);
    let entropy_identical = a == b && a != [0u8; 32];

    // Clock: starts where set, moves forward only on `advance`, and holds between reads.
    let clock = ManualClock::new(1_000);
    let t0 = clock.now_millis();
    clock.advance(500);
    let t1 = clock.now_millis();
    let t2 = clock.now_millis(); // no advance between these two reads
    let clock_explicit_only = t0 == 1_000 && t1 == 1_500 && t2 == t1;

    // Spawner: `NoopSpawner` drops the future — its side effect must never run.
    let ran = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&ran);
    NoopSpawner.spawn(Box::pin(async move {
        flag.store(true, Ordering::SeqCst);
    }));
    let spawn_inert = !ran.load(Ordering::SeqCst);

    w.sp4_hal = Some((entropy_identical, clock_explicit_only, spawn_inert));
}

#[then(
    "the two entropy streams are identical, the clock reflects only explicit advances, and the spawned task is inert"
)]
fn sp4_assert(w: &mut ConformanceWorld) {
    let (entropy, clock, spawn) = w
        .sp4_hal
        .expect("the When step must have exercised the HAL seams");
    assert!(
        entropy,
        "equally-seeded SeededEntropy must reproduce the same non-trivial stream — V&V reproducible"
    );
    assert!(
        clock,
        "ManualClock must reflect only explicit advances — hermetic, controllable time"
    );
    assert!(
        spawn,
        "NoopSpawner must drop the spawned future — background work is inert in a hermetic space"
    );
}

// ─────────────────── SP-5 — headless surface conformance (spec 02 §5) ───────────────────
// Headless is a first-class profile, not an exemption: `NullSurface` projects the canonical
// empty-projection κ (the κ of no bytes) and refuses `intent` with a typed `Headless` error.
// A space with no display still satisfies the contract. Witnessed against `NullSurface`.

#[given("a headless space's Surface")]
fn sp5_given(_w: &mut ConformanceWorld) {}

#[when("a workload is projected and an operator intent is submitted")]
async fn sp5_drive(w: &mut ConformanceWorld) {
    let surface = NullSurface;
    let workload = address_bytes(b"sp5-headless-workload");
    let projected = surface.project(&workload).await;
    let project_is_empty_kappa =
        matches!(&projected, Ok(k) if k.as_bytes() == address_bytes(&[]).as_bytes());
    let refused = surface
        .intent(&workload, Intent::TerminalInput(b"ls\n".to_vec()))
        .await;
    let intent_refused_headless = matches!(refused, Err(SurfaceError::Headless));
    w.sp5_surface = Some((project_is_empty_kappa, intent_refused_headless));
}

#[then(
    "projection yields the canonical empty-projection κ and intent is refused with a typed headless error"
)]
fn sp5_assert(w: &mut ConformanceWorld) {
    let (empty, refused) = w
        .sp5_surface
        .expect("the When step must have driven the headless surface");
    assert!(
        empty,
        "a headless project() must yield the canonical empty-projection κ (the κ of no bytes)"
    );
    assert!(
        refused,
        "a headless intent() must be refused with SurfaceError::Headless — a typed error, not a panic"
    );
}

// ───────────────── MG-7 — holospaces CC V&V absorbed into the unified ledger ─────────────────
// The holospaces component-conformance catalog is absorbed as hologram's non-BDD `CC` class,
// each row witnessed by a cargo test in the ported space. This scenario runs the artifact-free
// CC bijection audit (the same `cc::check_cc_bijection` the standalone `cc_gate` test runs) over
// the real ledger + witness tree: it proves the absorption is honest — no CC row claims a
// component conforms without a present, named witness test. The actual CC passes (fast/artifact
// via `cargo test`, QEMU boots + browser via the `holospaces-vv-heavy` CI job) are gated
// separately; this witnesses the *catalog integrity* that makes those gates meaningful.

#[given("the holospaces CC catalog in the unified conformance ledger")]
fn mg7_given(_w: &mut ConformanceWorld) {}

#[when("the CC bijection audit binds every row to its witness test")]
fn mg7_audit(w: &mut ConformanceWorld) {
    use hologram_conformance::{catalog, cc, CC_TESTS_DIR, CONFORMANCE_MD};
    let md = std::fs::read_to_string(CONFORMANCE_MD).expect("read CONFORMANCE.md");
    let rows = catalog::parse_catalog(&md);
    let witnesses = cc::collect_cc_witnesses(std::path::Path::new(CC_TESTS_DIR))
        .expect("walk the holospaces cc tests");
    w.mg7_cc_bound = Some(cc::check_cc_bijection(&rows, &witnesses).is_ok());
}

#[then("every CC row binds to a present, named witness — none by self-reference")]
fn mg7_assert(w: &mut ConformanceWorld) {
    assert!(
        w.mg7_cc_bound
            .expect("the When step must have run the CC bijection audit"),
        "every CC catalog row must bind to a present, named witness test — the holospaces V&V is \
         absorbed honestly (MG-7)"
    );
}

// ───────────────── MG-8 — holospaces CS docs-conformance absorbed into the ledger ─────────────────
// holospaces' specification conformance is absorbed as hologram's non-BDD `CS` class (CS-1..6),
// each witnessed by a V1–V8 validator script. This scenario runs the artifact-free CS bijection
// audit over the ledger + the docs scripts tree — proving the absorption is honest (no CS row
// claims the docs conform without a present validator). The actual V1–V8 runs are gated by the
// `docs-conformance` CI job (the docs toolchain); this witnesses the catalog integrity.

#[given("holospaces' specification-conformance (CS) validators and their external standards")]
fn mg8_given(_w: &mut ConformanceWorld) {}

#[when("the docs V&V is absorbed into hologram's conformance framework")]
fn mg8_audit(w: &mut ConformanceWorld) {
    use hologram_conformance::{catalog, cc, CONFORMANCE_MD, CS_SCRIPTS_DIR};
    let md = std::fs::read_to_string(CONFORMANCE_MD).expect("read CONFORMANCE.md");
    let rows = catalog::parse_catalog(&md);
    w.mg8_cs_bound =
        Some(cc::check_cs_bijection(&rows, std::path::Path::new(CS_SCRIPTS_DIR)).is_ok());
}

#[then("every CS row runs V1-V8 against its external standard, never self-reference")]
fn mg8_assert(w: &mut ConformanceWorld) {
    assert!(
        w.mg8_cs_bound
            .expect("the When step must have run the CS bijection audit"),
        "every CS catalog row must bind to a present V1–V8 validator script — the holospaces docs \
         V&V is absorbed honestly (MG-8)"
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
