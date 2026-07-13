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
#[derive(Debug, Default, cucumber::World)]
pub struct ConformanceWorld {
    /// Set by a `When` step; asserted by a `Then` step. Placeholder until real
    /// contract handles land per phase.
    pub last_outcome: Option<String>,
}
