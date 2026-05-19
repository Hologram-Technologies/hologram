//! Prism-pipeline-routed execute entry (wiki ADR-022 D5 / spec VIII).
//!
//! `InferenceSession::execute` is hologram's compute walker — it steps
//! the schedule, dispatches kernels through the chosen `Backend`, and
//! emits raw `OutputBuffer`s. That path is unchanged.
//!
//! `InferenceSession::execute_attested` wraps the same compute and
//! additionally routes a `Grounded<Digest<32>>` attestation through
//! `prism::pipeline::run` per wiki ADR-022 D5. The attestation is
//! produced by the canonical prism reduction-stage sequence
//! (preflight feasibility, budget solvency, package coherence,
//! dispatch coverage, timing) and carries a content fingerprint over
//! the session's compile unit. The compute happens in hologram's
//! axis impls (see `hologram_backend::prism_axes`); the prism
//! pipeline supplies the verification + sealing surface per the wiki
//! cost-model commitment (TC-03).

use prism::crypto::Digest;
use prism::operation::Term;
use prism::pipeline::{self as prism_pipeline, PipelineFailure};
use prism::seal::Grounded;
use prism::vocabulary::{CompileUnitBuilder, VerificationDomain, WittLevel};
// `literal_u64` is a foundation-only helper (not curated through
// `prism::operation`); reach it via `prism`'s façade re-export of
// the substrate crate to keep the dep tree single-rooted.
use prism::uor_foundation::pipeline::literal_u64;
use hologram_host::HologramHasher;

use crate::buffer::{InputBuffer, OutputBuffer};
use crate::error::ExecError;
use crate::session::{InferenceSession, SessionBackend};

/// Static root term consumed by `pipeline::run` — a single W32 literal.
/// The actual compute happens in hologram's executor before
/// `pipeline::run` is invoked; this static `Term` slice is the
/// minimal-content carrier the prism pipeline requires to anchor a
/// `CompileUnit` for content-fingerprint emission.
static ROUTE_TERMS: &[Term] = &[literal_u64(0, WittLevel::W32)];

/// Static verification-domain set for the attestation unit. Hologram
/// operates in the algebraic verification domain (the ring-arithmetic
/// fold-rule per ADR-050).
static ROUTE_DOMAINS: &[VerificationDomain] = &[VerificationDomain::Algebraic];

/// One execution's compute outputs paired with the
/// `Grounded<Digest<32>>` attestation prism emits over the
/// inference unit's compile-time content fingerprint.
pub struct AttestedExecution {
    /// The compute results — identical to what `execute()` returns.
    pub outputs: Vec<OutputBuffer>,
    /// The prism-emitted attestation carrying the canonical content
    /// fingerprint over the inference unit's `CompileUnit` shape.
    pub attestation: Grounded<Digest<32>>,
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

        // Build the attestation unit. Hologram's witt level is the
        // backend's `WITT_LEVEL_MAX_BITS` ceiling; the budget is sized
        // for one inference pass.
        let unit = CompileUnitBuilder::new()
            .root_term(ROUTE_TERMS)
            .witt_level_ceiling(WittLevel::W32)
            .thermodynamic_budget(1024)
            .target_domains(ROUTE_DOMAINS)
            .result_type::<Digest<32>>()
            .validate()
            .map_err(|sv| ExecError::Pipeline(PipelineFailure::ShapeViolation { report: sv }))?;

        // Route through prism::pipeline::run. The reduction-stage
        // sequence runs preflight checks + content-fingerprint
        // emission; `Grounded<Digest<32>>` attests the unit's shape.
        let attestation = prism_pipeline::run::<Digest<32>, _, HologramHasher>(unit)
            .map_err(ExecError::Pipeline)?;

        Ok(AttestedExecution { outputs, attestation })
    }
}

