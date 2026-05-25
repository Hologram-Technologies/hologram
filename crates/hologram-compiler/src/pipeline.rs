//! Prism pipeline integration (spec VII.2 steps 4-6, wiki ADR-031).
//!
//! Builds a per-node `CompileUnit`, validates it via prism's
//! `CompileUnitBuilder`, and runs `pipeline::run_tower_completeness`
//! to produce a `Validated<LiftChainCertificate>`. Every imported
//! type reaches hologram through the prism façade (`prism::pipeline`,
//! `prism::seal`, `prism::operation`, `prism::vocabulary`); the
//! direct foundation namespace is referenced only for items the
//! prism crate has not yet curated (such as the
//! `pipeline::run_tower_completeness` free function).

use crate::error::CompileError;
use hologram_host::HologramHasher;
use hologram_ops::{HoloTerm, HOLOGRAM_INLINE_BYTES};
use prism::pipeline::ConstrainedTypeShape;
use prism::seal::Validated;
use prism::uor_foundation::enforcement::{
    Binding, CompileUnit, CompileUnitBuilder, GenericImpossibilityWitness, LiftChainCertificate,
};
use prism::uor_foundation::pipeline as upstream_pipeline;
use prism::vocabulary::{VerificationDomain, WittLevel};

/// Content-fingerprint width threaded through the prism completeness tower.
/// Single source of truth is the application's `HostBounds` selection
/// (BLAKE3-canonical 32 bytes); every backend agrees on this width.
const FP_MAX: usize =
    <hologram_host::HologramHostBoundsCpu as uor_foundation::HostBounds>::FINGERPRINT_MAX_BYTES;

/// Per-node compile-unit construction inputs.
pub struct PerNodeUnit<'a> {
    pub root_term: &'a [HoloTerm],
    pub bindings: &'a [Binding],
    pub witt_level: WittLevel,
    pub budget: u64,
    pub target_domains: &'a [VerificationDomain],
}

/// Build + validate the `CompileUnit`. Returns the `Validated<CompileUnit>`
/// (sealed by upstream).
pub fn build_unit<'a>(
    input: &PerNodeUnit<'a>,
) -> Result<Validated<CompileUnit<'a, HOLOGRAM_INLINE_BYTES>>, CompileError> {
    // `result_type::<T>()` takes a compile-time `ConstrainedTypeShape`. Op
    // selection is a runtime `OpKind`, so the result type is the
    // runtime-adaptable `RuntimeResultType` marker (hologram's tensor type);
    // per-op identity is carried by the certificate-cache key
    // (`compute_fingerprint` over op IRI + witt level + backend), not by the
    // unit's result-type parameter.
    CompileUnitBuilder::new()
        .root_term(input.root_term)
        .bindings(input.bindings)
        .witt_level_ceiling(input.witt_level)
        .thermodynamic_budget(input.budget)
        .target_domains(input.target_domains)
        .result_type::<RuntimeResultType>()
        .validate()
        .map_err(|sv| CompileError::ShapeViolation { iri: sv.shape_iri })
}

/// Runtime-adaptable result-type marker. Its IRI is the generic Tensor IRI;
/// per-op IRIs flow through the certificate-cache key, not through this type.
struct RuntimeResultType;

impl ConstrainedTypeShape for RuntimeResultType {
    const IRI: &'static str = "https://hologram.uor.foundation/type/tensor";
    const SITE_COUNT: usize = 0;
    const CONSTRAINTS: &'static [prism::pipeline::ConstraintRef] = &[];
    const CYCLE_SIZE: u64 = 1;
}

/// Run `pipeline::run_tower_completeness` against the result type at the
/// requested Witt level. Returns the `Validated<LiftChainCertificate>`.
pub fn run_completeness(
    witt_level: WittLevel,
) -> Result<Validated<LiftChainCertificate<FP_MAX>>, GenericImpossibilityWitness> {
    upstream_pipeline::run_tower_completeness::<RuntimeResultType, HologramHasher, FP_MAX>(
        &RuntimeResultType,
        witt_level,
    )
}
