//! BDD conformance harness for the hologram refactor.
//!
//! - `cucumber` runner entrypoint: `tests/bdd.rs`
//! - static honesty meta-gate: `tests/meta_gate.rs`
//! - catalog parser: [`catalog`]; feature parser: [`feature`]; gate: [`report`]
pub mod catalog;
pub mod feature;
pub mod report;

/// Absolute path to the repo-root `features/suites` tree, resolved at compile time.
pub const SUITES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../features/suites");

/// Absolute path to the repo-root `CONFORMANCE.md`.
pub const CONFORMANCE_MD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../CONFORMANCE.md");

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
}
