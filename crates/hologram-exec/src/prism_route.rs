//! Prism-pipeline-routed execute entry (wiki ADR-022 D5 / spec VIII).
//!
//! `InferenceSession::execute` is hologram's compute walker — it steps
//! the schedule, dispatches kernels through the chosen `Backend`, and
//! emits raw `OutputBuffer`s. That path is unchanged.
//!
//! `InferenceSession::execute_attested` wraps the same compute and
//! additionally routes a `Grounded<Digest<32>>` attestation through
//! `prism::pipeline::run` per wiki ADR-022 D5. Two attestation
//! surfaces are returned and serve distinct purposes:
//!
//! 1. `prism_attestation: Grounded<Digest<32>>` — prism's sealed
//!    witness that the inference unit was admitted by the canonical
//!    reduction-stage sequence (preflight feasibility, budget
//!    solvency, package coherence, dispatch coverage, timing). Its
//!    `content_fingerprint()` is folded over the unit's TYPE-SHAPE
//!    (witt level, budget, result-type IRI, constraints) per the
//!    prism API contract (`fold_unit_digest`); it is identical for
//!    any two sessions with the same shape.
//! 2. `archive_fingerprint: [u8; 32]` — hologram's canonical 32-byte
//!    BLAKE3 footer fingerprint over the `.holo` archive's bytes
//!    (spec X.1, routed through `prism::crypto::Blake3Hasher`). This
//!    is the per-content anchor; it differs between any two distinct
//!    archives.
//!
//! Consumers that need "this prism-admitted unit, of this content"
//! pair the two — together they form hologram's TC-03 commitment
//! (the wiki cost-model's content-anchored attestation).

use alloc::vec::Vec;

use prism::crypto::Digest;
use prism::pipeline::{self as prism_pipeline, PipelineFailure};
use prism::seal::Grounded;
use prism::vocabulary::{CompileUnitBuilder, VerificationDomain, WittLevel};
// `literal_u64` is a foundation-only helper (not curated through
// `prism::operation`); reach it via `prism`'s façade re-export of
// the substrate crate to keep the dep tree single-rooted.
use hologram_ops::{HoloTerm, HOLOGRAM_INLINE_BYTES};
use hologram_types::HologramHasher;
use prism::uor_foundation::pipeline::literal_u64;

/// Content-fingerprint width threaded through prism's pipeline. Single
/// source of truth is the application's `HostBounds` selection
/// (BLAKE3-canonical 32 bytes); every backend agrees on this width.
const FP_MAX: usize =
    <hologram_types::HologramHostBoundsCpu as uor_foundation::HostBounds>::FINGERPRINT_MAX_BYTES;

use crate::buffer::{InputBuffer, OutputBuffer};
use crate::error::ExecError;
use crate::session::{InferenceSession, SessionBackend};

/// Static root term consumed by `pipeline::run` — a single W32 literal.
/// Prism's `pipeline::run` folds only the unit's type-shape fields into
/// the `Grounded` content fingerprint (witt + budget + IRI + constraints +
/// kind); the root_term is structural metadata for resolvers, not digest
/// input. The per-content anchor is `AttestedExecution::archive_fingerprint`.
const ROUTE_TERMS: &[HoloTerm] = &[literal_u64(0, WittLevel::W32)];

/// Static verification-domain set for the attestation unit. Hologram
/// operates in the algebraic verification domain (the ring-arithmetic
/// fold-rule per ADR-050).
static ROUTE_DOMAINS: &[VerificationDomain] = &[VerificationDomain::Algebraic];

/// One execution's compute outputs paired with the two attestation
/// surfaces: prism's sealed shape-witness and hologram's content anchor.
/// Together they form the TC-03 content-anchored attestation per the
/// wiki cost-model.
pub struct AttestedExecution {
    /// The compute results — identical to what `execute()` returns.
    pub outputs: Vec<OutputBuffer>,
    /// Prism's sealed `Grounded<Digest<32>>` witness that the inference
    /// unit was admitted by the canonical reduction-stage sequence. Its
    /// `content_fingerprint()` is folded over the unit's TYPE-SHAPE
    /// (witt + budget + result-type IRI + constraints); identical for
    /// any two sessions with the same shape.
    pub prism_attestation: Grounded<'static, Digest<32>, HOLOGRAM_INLINE_BYTES, FP_MAX>,
    /// Hologram's per-content anchor: the archive's canonical 32-byte
    /// BLAKE3 footer fingerprint (spec X.1, computed via the prism
    /// `Blake3Hasher`). Differs between any two distinct archives —
    /// this is the field that anchors the attestation to "this
    /// specific model" rather than "any model of this shape".
    pub archive_fingerprint: [u8; 32],
}

impl<B: SessionBackend> InferenceSession<B> {
    /// Execute the session through the prism pipeline (wiki ADR-022 D5).
    ///
    /// Runs hologram's compute walker to produce the output buffers,
    /// then builds a `Validated<CompileUnit>` whose result type is
    /// `Digest<32>` and invokes `prism::pipeline::run` to emit a
    /// `Grounded<Digest<32>>` attestation. The Grounded value carries
    /// the canonical content fingerprint per the prism cost-model
    /// commitment (TC-03 / wiki §10).
    ///
    /// # Errors
    ///
    /// Returns `ExecError::Backend` on compute failure, or
    /// `ExecError::PipelineFailure` if prism's reduction-stage
    /// sequence rejects the unit (e.g. preflight feasibility,
    /// budget solvency, dispatch coverage).
    pub fn execute_attested(
        &mut self,
        inputs: &[InputBuffer],
    ) -> Result<AttestedExecution, ExecError> {
        // Run the compute first (this populates the workspace's
        // output slots through the schedule walker).
        let outputs = self.execute(inputs)?;

        // Build the attestation unit. Witt-level ceiling = W32 (matches
        // the route literal); thermodynamic budget is sized for one
        // inference pass. Result type is hologram's canonical
        // `Digest<32>` from `prism::crypto`.
        let unit = CompileUnitBuilder::new()
            .root_term(ROUTE_TERMS)
            .witt_level_ceiling(WittLevel::W32)
            .thermodynamic_budget(1024)
            .target_domains(ROUTE_DOMAINS)
            .result_type::<Digest<32>>()
            .validate()
            .map_err(|sv| ExecError::Pipeline(PipelineFailure::ShapeViolation { report: sv }))?;

        // Route through prism::pipeline::run. The reduction-stage
        // sequence runs preflight checks + emits a sealed
        // `Grounded<Digest<32>>` whose `content_fingerprint()` is
        // folded over the unit's TYPE-SHAPE (witt + budget + IRI +
        // constraints) — identical for every session with the same
        // shape, per prism's `fold_unit_digest` contract.
        let prism_attestation =
            prism_pipeline::run::<Digest<32>, _, HologramHasher, HOLOGRAM_INLINE_BYTES, FP_MAX>(
                unit,
            )
            .map_err(ExecError::Pipeline)?;

        // Pair the prism witness with hologram's per-content anchor.
        // Consumers verify "this prism-admitted shape" AND "of this
        // archive's content" together (TC-03).
        Ok(AttestedExecution {
            outputs,
            prism_attestation,
            archive_fingerprint: self.archive_fingerprint(),
        })
    }
}
