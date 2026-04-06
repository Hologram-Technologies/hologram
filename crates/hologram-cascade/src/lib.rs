//! Cascade state machine: 7-stage evaluation pipeline with certificate memoization.
//!
//! Implements the UOR cascade pipeline from uor-foundation v0.1.4:
//! - Stage 0 (Init, Ω⁰): initialize state vector, check certificate cache
//! - Stage 1 (Declare, Ω¹): resolver selection
//! - Stage 2 (Factorize, Ω²): ground to ring address (CSE, fusion, folding)
//! - Stage 3 (Resolve, Ω³): resolve scheduling constraints
//! - Stage 4 (Attest, Ω⁴): verify no contradictions
//! - Stage 5 (Extract, Ω⁵): extract coherent output (re-entry on cache hit)
//! - Stage 6 (Converge, π): terminal state, emit certificate
//!
//! # Memoization
//!
//! The cascade caches `Certificate` values keyed by `(unit_address, quantum_level)`.
//! On cache hit at Stage 0, the cascade skips directly to Stage 5 (Extract),
//! bypassing stages 1-4 entirely.
//!
//! # Architecture
//!
//! This crate is in the **kernel layer** — it depends only on `hologram-core`
//! and `blake3`. No backward dependencies on bridge or user crates.

pub mod certificate;
pub mod dispatch_decl;
pub mod effect_decl;
pub mod engine;
pub mod liveness;
pub mod precision;
pub mod qedl;
pub mod stage;
pub mod tape_builder;
pub mod workspace;

pub use certificate::{Certificate, CertificateStore};
pub use engine::{run_cascade, run_cascade_with_graph, run_cascade_with_graph_opts, CascadeResult};
pub use liveness::{compute_liveness, LivenessInterval};
pub use precision::promote_prim_ring_levels;
pub use qedl::{EncodingId, QedlBoundary};
pub use stage::{CascadeStage, CascadeState, HaltReason, Transition};
pub use workspace::{plan_workspace, WorkspaceLayout};
