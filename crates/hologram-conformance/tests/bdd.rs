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
// HF-1/HF-2: the `.holo` v3 application container (spec 03) — the AppManifest realization + the
// real archive container, so opening a tensor-only archive and nesting a child are witnessed
// end-to-end through the shipping format.
use hologram_archive::{HoloLoader, HoloWriter};
use hologram_space::{AppManifest, Layer, LayerKind};
// NW-1/NW-2: a Network is a κ-realization (membership + policy operands); its tier gates capability
// at the protocol boundary (spec 04).
use hologram_space::{Network, NetworkOp, NetworkTier};
// GV-3/GV-4: a signing key bound to a κ-addressed identity as content; a capability policy gating
// network ops at the boundary with per-capability accounting (spec 07 R3/R4).
use hologram_space::AttestationKey;
// GV-2: lifecycle transitions emit through one audit seam onto an append-only κ-chain (spec 07 R2).
use hologram_space::{AuditEvent, KappaLabel71, LifecycleTransition};
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

// ───────────────────────────── HF-1 — `.holo` v3 is the one container ─────────────────────────────
// A tensor-only archive is the degenerate single-layer case of the one format: a `.holo` v3 whose
// AppManifest names exactly one tensor-plan layer (no exit code, no children). Witnessed end-to-end
// through the real archive container — write a v3 archive, load it, decode the manifest realization.

#[given("a tensor-only archive")]
fn hf1_given(w: &mut ConformanceWorld) {
    // A compiled tensor graph κ + the required-capabilities κ (a declaration, never an entitlement).
    let graph = address_bytes(b"hf1-tensor-graph");
    let requires = address_bytes(b"hf1-requires");
    let manifest = AppManifest::single_tensor_plan(graph, "session", requires);
    let mut writer = HoloWriter::new();
    writer.set_app_manifest(manifest.canonicalize());
    w.canonical = writer
        .finish()
        .expect("write a tensor-only .holo v3 archive");
}

#[when("I open it as a .holo v3 application")]
fn hf1_open(w: &mut ConformanceWorld) {
    let plan = HoloLoader::from_bytes(&w.canonical)
        .expect("a v3 archive loads")
        .into_plan()
        .expect("its section table parses");
    let manifest_bytes = plan
        .app_manifest()
        .expect("the v3 archive carries an AppManifest section");
    let manifest = AppManifest::decode(manifest_bytes).expect("the manifest realization decodes");
    // "Opening as an application" validates the manifest's execution invariants.
    manifest.validate().expect("the manifest is loadable");
    let first_is_tensor_plan = manifest
        .layers
        .first()
        .is_some_and(|l| l.kind == LayerKind::TensorPlan);
    w.hf1_degenerate = Some((
        manifest.layers.len(),
        first_is_tensor_plan,
        manifest.primary.is_none(),
    ));
}

#[then("it is the degenerate single-layer case of the one format")]
fn hf1_assert(w: &mut ConformanceWorld) {
    assert_eq!(
        w.hf1_degenerate,
        Some((1, true, true)),
        "a tensor-only archive must open as exactly one tensor-plan layer with no exit code (no \
         primary) — the degenerate single-layer case of the one `.holo` v3 format"
    );
}

// ───────────────────────────── HF-2 — capability-attenuated nesting ─────────────────────────────
// A parent app nests a child by κ ref with a delegated CapabilitySet; the delegated authority must
// be a subset of the parent's — attenuation only, amplification unrepresentable. Witnessed by the
// capability lattice's `admits` relation over a real AppManifest child edge.

/// Full `Capabilities` builder (the struct has no `Default`); channel lists inferred empty.
fn hf2_caps(storage: &[&[u8]], quota: u64, fetch: bool) -> Capabilities {
    Capabilities {
        storage_roots: storage.iter().map(|s| address_bytes(s)).collect(),
        storage_quota_bytes: quota,
        network_fetch: fetch,
        network_announce: false,
        publish_channels: vec![],
        subscribe_channels: vec![],
        memory_max_bytes: quota,
        cpu_time_per_event_ms: 10,
        priority_weight: 4,
    }
}

#[given("a parent app with a CapabilitySet")]
fn hf2_given(w: &mut ConformanceWorld) {
    // Parent authority: two storage roots, a real budget, fetch allowed. Carried to the When step
    // as CapabilitySet canonical bytes so the World stays free of domain types.
    let parent = hf2_caps(&[b"root-A", b"root-B"], 1000, true);
    w.canonical = CapabilitySet::new(parent).canonicalize();
}

#[when("it nests a child by κ ref with a delegated CapabilitySet")]
fn hf2_nest(w: &mut ConformanceWorld) {
    let parent = CapabilitySet::to_capabilities(&w.canonical).expect("decode parent caps");

    // The delegated child: a subset of the parent (fewer roots, tighter budget, fetch dropped).
    let child = hf2_caps(&[b"root-A"], 500, false);
    let child_caps_kappa = CapabilitySet::new(child.clone()).kappa();
    let child_app_kappa = address_bytes(b"hf2-child-app");

    // The nesting is expressed in the κ-graph: the parent AppManifest carries the child as a
    // `(app κ, delegated caps κ)` edge — nested "by κ ref", recovered by references().
    let parent_manifest = AppManifest {
        primary: None,
        requires: CapabilitySet::new(parent.clone()).kappa(),
        layers: vec![hologram_space::Layer::tensor(
            address_bytes(b"hf2-code"),
            "run",
        )],
        children: vec![(child_app_kappa, child_caps_kappa)],
    };
    let refs = <AppManifest as Realization>::references(&parent_manifest.canonicalize())
        .expect("the parent manifest decodes");
    assert!(
        refs.contains(&child_app_kappa) && refs.contains(&child_caps_kappa),
        "the child must be nested by κ ref — its app κ and delegated caps κ are edges in the \
         parent's reachability closure"
    );

    // An over-broad child that reaches a root the parent lacks — amplification, must be refused.
    let overbroad = hf2_caps(&[b"root-A", b"root-C"], 500, false);

    let refs_subset = child
        .storage_roots
        .iter()
        .all(|r| parent.storage_roots.contains(r));
    w.hf2_attenuation = Some((
        parent.admits(&child),
        refs_subset,
        !parent.admits(&overbroad),
    ));
}

#[then("the child's refs and capabilities are a subset of the parent's")]
fn hf2_assert(w: &mut ConformanceWorld) {
    assert_eq!(
        w.hf2_attenuation,
        Some((true, true, true)),
        "the delegated child must be admitted (caps ⊆ parent), its refs a subset of the parent's, \
         and an over-broad child refused — attenuation only, amplification unrepresentable"
    );
}

// ───────────────────────────── HF-3 — per-layer certificates ─────────────────────────────
// A `.holo` v3's per-layer certificate is each layer's κ-identity, bound into the app's committed
// identity (the manifest κ addresses the bytes that embed every layer κ). Inspecting through the
// Client surface returns one verdict per layer, each verifying, from a **thin** archive (manifest
// only, no payload) — so certificates travel with the manifest and inspection never strips them.

#[given("a .holo v3 with per-layer certificates")]
fn hf3_given(w: &mut ConformanceWorld) {
    // A four-layer app; each layer κ is its per-layer certificate, bound into the manifest κ.
    let manifest = AppManifest {
        primary: Some(0),
        requires: address_bytes(b"hf3-requires"),
        layers: vec![
            Layer::wasm(address_bytes(b"hf3-wasm"), "_start"),
            Layer::tensor(address_bytes(b"hf3-plan"), "sess"),
            Layer::rootfs(address_bytes(b"hf3-rootfs"), "boot", "riscv64"),
            Layer::view(address_bytes(b"hf3-view"), "portable"),
        ],
        children: vec![],
    };
    // A THIN archive: the manifest section only, no payloads — verification must not need the fat
    // profile, and certificates travel with the manifest.
    let mut writer = HoloWriter::new();
    writer.set_app_manifest(manifest.canonicalize());
    w.canonical = writer
        .finish()
        .expect("write a thin .holo v3 with per-layer certs");
}

#[when("I inspect it through the Client surface")]
fn hf3_inspect(w: &mut ConformanceWorld) {
    let client = Client::new(SpikeSpace::new());
    let holo = hologram::Holo::from_bytes(w.canonical.clone());
    let inspection = client
        .inspect(&holo)
        .expect("inspect the .holo v3 through the Client");
    w.hf3_inspection = Some((inspection.all_verified(), inspection.layers.len()));
}

#[then("every certificate verifies and none is stripped")]
fn hf3_assert(w: &mut ConformanceWorld) {
    assert_eq!(
        w.hf3_inspection,
        Some((true, 4)),
        "inspecting a .holo v3 must return a verified certificate for every one of its layers \
         (none stripped) — from the thin profile, so certs travel with the manifest"
    );
}

// ───────────────────────────── NW-1 — a Network is a κ-realization ─────────────────────────────
// A Network embeds its membership + policy operand κs (SPINE-2/3); references() is the inverse
// projection recovering exactly those operands — no side tables. Witnessed against the `Network`
// realization in `hologram-space`.

#[given("a Network built from a membership set and a policy")]
fn nw1_build(w: &mut ConformanceWorld) {
    let op_a = address_bytes(b"nw1-operator-a");
    let op_b = address_bytes(b"nw1-operator-b");
    let policy = address_bytes(b"nw1-policy-capset");
    w.operand_kappas = vec![
        op_a.as_bytes().to_vec(),
        op_b.as_bytes().to_vec(),
        policy.as_bytes().to_vec(),
    ];
    let network = Network {
        membership: vec![op_a, op_b],
        policy,
        parent: None,
        tier: NetworkTier::Restricted,
        key_ref: None,
    };
    w.canonical = network.canonicalize();
}

#[when("I call references() on its realization")]
fn nw1_references(w: &mut ConformanceWorld) {
    let refs = Network::references(&w.canonical).expect("a well-formed Network decodes");
    w.references = Some(refs.iter().map(|k| k.as_bytes().to_vec()).collect());
}

#[then("it yields the membership and policy operand κs with no side tables")]
fn nw1_assert(w: &mut ConformanceWorld) {
    let refs = w
        .references
        .as_ref()
        .expect("references() must have been called by the When step");
    assert_eq!(
        refs, &w.operand_kappas,
        "references() must yield exactly the membership + policy operand κs — no side tables"
    );
}

// ───────────────────────────── NW-2 — tiers gate at the boundary ─────────────────────────────
// A network tier decides store/fetch/announce from `(tier, is_member)` alone — the gate is given no
// business data, so the check is structurally at the protocol boundary. Public admits anyone;
// restricted/private require membership. Witnessed against `NetworkTier::admits`.

#[given("public, restricted, and private network tiers")]
fn nw2_given(_w: &mut ConformanceWorld) {}

#[when("a peer attempts store/fetch/announce")]
fn nw2_attempt(w: &mut ConformanceWorld) {
    // The gate is applied uniformly to every op, from (tier, is_member) only.
    let ops = [NetworkOp::Store, NetworkOp::Fetch, NetworkOp::Announce];
    let boundary = ops.iter().all(|&op| {
        // Public: open to a non-member. Restricted/Private: refused unless a member.
        NetworkTier::Public.admits(op, false)
            && !NetworkTier::Restricted.admits(op, false)
            && NetworkTier::Restricted.admits(op, true)
            && !NetworkTier::Private.admits(op, false)
            && NetworkTier::Private.admits(op, true)
    });
    w.nw2_boundary = Some(boundary);
}

#[then("the capability check happens at the protocol boundary, not in business logic")]
fn nw2_assert(w: &mut ConformanceWorld) {
    assert_eq!(
        w.nw2_boundary,
        Some(true),
        "every op's admission must be decided from (tier, membership) alone — a protocol-boundary \
         gate: public admits anyone, restricted/private require membership, never business logic"
    );
}

// ───────────────────────────── GV-3 — R3 attestation: keys bind to κ-identity ─────────────────────────────
// A signing key is self-sovereign key material published as content; its identity IS its κ (the
// address of its canonical form) — verifiable by re-derivation, deterministic (one identity), never
// a second identity surface smuggled in through a certificate. Witnessed against `AttestationKey`.

#[given("a space signing a session attestation")]
fn gv3_given(w: &mut ConformanceWorld) {
    // The space's signing key (ed25519 = scheme 0, key material as content).
    let key = AttestationKey::new(0, b"gv3-ed25519-public-key".to_vec());
    w.canonical = key.canonicalize();
}

#[when("the signing key is published")]
fn gv3_publish(w: &mut ConformanceWorld) {
    // Publishing binds the key to a κ-addressed identity: its κ is the address of its content.
    let key_kappa = address_bytes(&w.canonical);
    let identity_is_kappa = verify_kappa(&w.canonical, &key_kappa).unwrap_or(false);
    // One identity surface: republishing the same key content yields the same κ; a leaf identity
    // has no operand κs and no separate certificate identity.
    let deterministic = address_bytes(&w.canonical) == key_kappa;
    let single_surface = AttestationKey::references(&w.canonical)
        .map(|r| r.is_empty())
        .unwrap_or(false);
    w.gv3_key_bound = Some(identity_is_kappa && deterministic && single_surface);
}

#[then("it is bound to a κ-addressed identity as content, never a second identity surface")]
fn gv3_assert(w: &mut ConformanceWorld) {
    assert_eq!(
        w.gv3_key_bound,
        Some(true),
        "a published signing key's identity must BE its κ — verifiable content, deterministic, with \
         no operand or certificate second identity surface (law 2 applies to attestation, R3)"
    );
}

// ───────────────────────────── GV-4 — R4 data governance: capability at the boundary ─────────────────────────────
// Governance is capability policies with quotas: who may store/fetch/announce, decided at the
// import/protocol boundary from the capability alone, with per-capability (not global) accounting.
// Witnessed against `Capabilities::admits_network_op`.

#[given("a network capability policy with quotas")]
fn gv4_given(w: &mut ConformanceWorld) {
    // Stash the policy as CapabilitySet bytes so the World stays free of domain types.
    let policy = Capabilities {
        storage_roots: vec![],
        storage_quota_bytes: 1000,
        network_fetch: true,
        network_announce: false,
        publish_channels: vec![],
        subscribe_channels: vec![],
        memory_max_bytes: 0,
        cpu_time_per_event_ms: 0,
        priority_weight: 0,
    };
    w.canonical = CapabilitySet::new(policy).canonicalize();
}

#[when("a peer stores, fetches, or announces content")]
fn gv4_attempt(w: &mut ConformanceWorld) {
    let policy = CapabilitySet::to_capabilities(&w.canonical).expect("decode the policy");
    // The check is at the boundary — decided from the capability alone, per op.
    let fetch_ok = policy.admits_network_op(NetworkOp::Fetch, 0);
    let announce_refused = !policy.admits_network_op(NetworkOp::Announce, 0);
    let store_within = policy.admits_network_op(NetworkOp::Store, 500);
    let store_over = !policy.admits_network_op(NetworkOp::Store, 2000);
    // Accounting is per-capability: a second capability's quota is independent, not a global counter.
    let other = Capabilities {
        storage_quota_bytes: 5000,
        ..policy.clone()
    };
    let per_capability = other.admits_network_op(NetworkOp::Store, 2000);
    w.gv4_boundary =
        Some(fetch_ok && announce_refused && store_within && store_over && per_capability);
}

#[then("the capability check is at the import/protocol boundary and accounting is per-capability")]
fn gv4_assert(w: &mut ConformanceWorld) {
    assert_eq!(
        w.gv4_boundary,
        Some(true),
        "store/fetch/announce must be admitted from the capability alone (a boundary check), with \
         each capability's quota its own — per-capability accounting, never a global counter (R4)"
    );
}

// ───────────────────────────── GV-2 — R2 auditability: one seam, no bypass ─────────────────────────────
// The audit trail is a κ-chained, append-only event log (SPINE-5 gives tamper-evidence for free).
// Every lifecycle transition (spawn/suspend/resume/terminate) emits through the one audit seam
// (`AuditEvent::record`) — the same seam `hologram-runtime`'s `Session` drives on every transition.
// The transition enum is closed, so no path can bypass it. Witnessed against `AuditEvent`.

#[given("lifecycle transitions spawn, suspend, resume, terminate")]
fn gv2_given(_w: &mut ConformanceWorld) {}

#[when("each transition occurs")]
fn gv2_transitions(w: &mut ConformanceWorld) {
    let subject = address_bytes(b"gv2-container");
    // The whole closed set of transitions — each emits through the one seam onto the κ-chain.
    let transitions = [
        LifecycleTransition::Spawn,
        LifecycleTransition::Suspend,
        LifecycleTransition::Resume,
        LifecycleTransition::Terminate,
    ];
    let mut head: Option<KappaLabel71> = None;
    let mut chain: Vec<KappaLabel71> = Vec::new();
    let mut pointable = true;
    for t in transitions {
        let event = AuditEvent::record(t, subject, head);
        let bytes = event.canonicalize();
        // Pointable at the κ-chain: references() recovers the subject and the predecessor link.
        let refs = AuditEvent::references(&bytes).expect("an audit event decodes");
        let links = refs[0] == subject && head.is_none_or(|prev| refs.get(1) == Some(&prev));
        pointable &= links && AuditEvent::transition_of(&bytes) == Ok(t);
        let kappa = event.kappa();
        chain.push(kappa);
        head = Some(kappa);
    }
    // No bypass: every one of the four transitions emitted a distinct, linked event.
    let distinct = chain
        .iter()
        .enumerate()
        .all(|(i, a)| chain[i + 1..].iter().all(|b| a != b));
    w.gv2_audit = Some(pointable && distinct && chain.len() == 4);
}

#[then("it emits through one seam that can be pointed at the κ-chain and no path bypasses it")]
fn gv2_assert(w: &mut ConformanceWorld) {
    assert_eq!(
        w.gv2_audit,
        Some(true),
        "every lifecycle transition must emit through the one audit seam onto an append-only, \
         κ-chained log (each event links to its predecessor) — no path bypasses it (R2)"
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
