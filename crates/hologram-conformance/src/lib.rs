//! BDD conformance harness for the hologram refactor.
//!
//! - `cucumber` runner entrypoint: `tests/bdd.rs`
//! - static honesty meta-gate: `tests/meta_gate.rs`
//! - catalog parser: [`catalog`]; feature parser: [`feature`]; gate: [`report`]
pub mod catalog;
pub mod cc;
pub mod feature;
pub mod report;

/// Absolute path to the repo-root `features/suites` tree, resolved at compile time.
pub const SUITES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../features/suites");

/// Absolute path to the repo-root `CONFORMANCE.md`.
pub const CONFORMANCE_MD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../CONFORMANCE.md");

/// Absolute path to the ported holospaces space's `tests/` tree (the `cc*.rs` CC witnesses),
/// resolved at compile time — the source the CC bijection audit walks (MG-7).
pub const CC_TESTS_DIR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../spaces/holospaces/tests");

/// Absolute path to the ported holospaces docs V&V validator scripts (the V1–V8 CS witnesses),
/// resolved at compile time — the source the CS bijection audit binds against (MG-8).
pub const CS_SCRIPTS_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../specs/holospaces/scripts"
);

/// Per-scenario async context. The only async↔sync seam (law 4): step bodies that
/// touch tensor compute stay synchronous inside this async World.
///
/// State is stored as raw κ-label bytes so this crate's library stays free of
/// domain types; the realizations are pulled in only as a dev-dependency by the
/// step definitions in `tests/bdd.rs`.
#[derive(Debug, Default, cucumber::World)]
pub struct ConformanceWorld {
    /// GV-1: the operand κ-labels a realization was built from (raw 71-byte forms).
    pub operand_kappas: Vec<Vec<u8>>,
    /// GV-1: the realization's canonical bytes.
    pub canonical: Vec<u8>,
    /// GV-1: what `references()` recovered (raw 71-byte κ-label forms).
    pub references: Option<Vec<Vec<u8>>>,
    /// SP-3: the output values produced by driving the spike slice's
    /// compile→store→boot (stored as primitives to keep the lib domain-type-free).
    pub sp3_output: Option<Vec<f32>>,
    /// SP-1: set once the reference store has run the full `hologram-tck` battery
    /// without a violation (the battery panics on the first failure).
    pub sp1_tck_passed: bool,
    /// LAW-1 (SPINE-1): `(authentic_verifies, tampered_verifies)` from re-deriving a
    /// realization's identity — authentic must be `true`, tampered `false`.
    pub law1_verify: Option<(bool, bool)>,
    /// LAW-3 (D21): set once a downstream crate's `Space` impl is accepted by `Client`
    /// and reaches a contract-mediated operation — proof the contract has no sealed traits.
    pub law3_accepted: bool,
    /// MG-5 (ground rule 5): set once every frozen golden vector re-derives to its
    /// recorded κ, bit-for-bit — the κ-stability safety net for crate moves.
    pub mg5_stable: bool,
    /// SP-4: `(entropy_identical, clock_explicit_only, spawn_inert)` from exercising the
    /// reference HAL seams — equally-seeded entropy must match, the clock must move only on
    /// `advance`, and `NoopSpawner` must drop (never run) the spawned future.
    pub sp4_hal: Option<(bool, bool, bool)>,
    /// SP-5: `(project_is_empty_kappa, intent_refused_headless)` from driving `NullSurface` —
    /// projection must yield the empty-projection κ and intent must be refused with `Headless`.
    pub sp5_surface: Option<(bool, bool)>,
    /// MG-7: `true` once the CC bijection audit binds every `CC` catalog row to a present witness
    /// test in the ported holospaces space — the honesty check that the V&V is really absorbed.
    pub mg7_cc_bound: Option<bool>,
    /// MG-8: `true` once the CS bijection audit binds every `CS` catalog row to a present V1–V8
    /// validator script — the honesty check that the docs V&V is really absorbed.
    pub mg8_cs_bound: Option<bool>,
    /// HF-1: `(layer_count, only_layer_is_tensor_plan, primary_is_none)` recovered by opening a
    /// tensor-only `.holo` v3 archive as an application — the degenerate single-layer case is
    /// `(1, true, true)`.
    pub hf1_degenerate: Option<(usize, bool, bool)>,
    /// HF-2: `(child_admitted, refs_subset, overbroad_refused)` from nesting a child app by κ ref
    /// with a delegated CapabilitySet — capability-attenuation holds iff `(true, true, true)`.
    pub hf2_attenuation: Option<(bool, bool, bool)>,
    /// HF-3: `(all_verified, inspected_layer_count)` from inspecting a `.holo` v3 through the
    /// `Client` surface — every per-layer certificate must verify and every layer be returned
    /// (none stripped), so this is `(true, expected_layer_count)`.
    pub hf3_inspection: Option<(bool, usize)>,
    /// NW-2: `true` once the tier gate has been shown to decide store/fetch/announce from
    /// `(tier, is_member)` alone — a protocol-boundary check (public admits all; restricted/private
    /// require membership), never business logic.
    pub nw2_boundary: Option<bool>,
}
