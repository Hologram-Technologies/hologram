use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("archive load failed: {0}")]
    Archive(#[from] hologram_archive::ArchiveError),
    #[error("backend dispatch failed")]
    Backend,
    #[error("input shape mismatch")]
    InputMismatch,
    #[error("workspace exhausted")]
    WorkspaceExhausted,
    /// `prism::pipeline::run` rejected the inference unit during the
    /// reduction-stage sequence (preflight feasibility / budget
    /// solvency / package coherence / dispatch coverage / timing).
    /// Per wiki ADR-022 D5 this is the canonical attestation-failure
    /// path; the compute may or may not have run before this.
    #[error("prism pipeline rejected the inference unit")]
    Pipeline(prism::pipeline::PipelineFailure),
    /// A weight slot declared `weight_layout = OUTPUT_MAJOR` — its bytes arrive
    /// as `[n,k]` — but no output-major kernel can serve the call (unregistered
    /// quant tier, `k` beyond the exact-accumulation bound, per-tensor or
    /// group-wise scales, a codebook tier without its codebook, or `act_quant`
    /// left at `W8A32`).
    ///
    /// Every other decode path reads `[k,n]`, so there is no correct fallback:
    /// silently taking one would transpose the weight and return a plausible,
    /// wrong answer. See `docs/numerics/w8a8.md`.
    #[error("weight slot declares output-major layout but no output-major kernel can serve it")]
    UnsatisfiableWeightLayout,
}
