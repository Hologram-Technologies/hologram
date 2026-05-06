//! Upstream pipeline integration (spec VII.2 steps 4-6).
//!
//! Builds a per-node `CompileUnit`, validates it, and runs
//! `pipeline::run_tower_completeness` to produce a
//! `Validated<LiftChainCertificate>`.

use uor_foundation::WittLevel;
use uor_foundation::enforcement::{
    CompileUnit, CompileUnitBuilder, Term, Validated,
    GenericImpossibilityWitness, LiftChainCertificate,
};
use uor_foundation::enums::VerificationDomain;
use uor_foundation::pipeline as upstream_pipeline;
use hologram_host::HologramHasher;
use crate::error::CompileError;

/// Per-node compile-unit construction inputs.
pub struct PerNodeUnit<'a> {
    pub root_term: &'a [Term],
    pub bindings: &'a [uor_foundation::enforcement::Binding],
    pub witt_level: WittLevel,
    pub budget: u64,
    pub target_domains: &'a [VerificationDomain],
    pub result_type_iri: &'static str,
}

/// Build + validate the `CompileUnit`. Returns the `Validated<CompileUnit>`
/// (sealed by upstream).
pub fn build_unit<'a>(input: &PerNodeUnit<'a>) -> Result<Validated<CompileUnit<'a>>, CompileError> {
    // We thread the IRI through via the typed `result_type::<T>()` setter on
    // `CompileUnitBuilder`. Because the IRI is per-op and known only at
    // dispatch time, we use a generic `result_type` via the `PhantomShape`
    // helper below.
    let _ = input.result_type_iri; // routed through cache key, not the builder
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

impl uor_foundation::pipeline::ConstrainedTypeShape for RuntimeResultType {
    const IRI: &'static str = "https://hologram.uor.foundation/type/tensor";
    const SITE_COUNT: usize = 0;
    const CONSTRAINTS: &'static [uor_foundation::pipeline::ConstraintRef] = &[];
}

/// Run `pipeline::run_tower_completeness` against the result type at the
/// requested Witt level. Returns the `Validated<LiftChainCertificate>`.
pub fn run_completeness(
    witt_level: WittLevel,
) -> Result<Validated<LiftChainCertificate>, GenericImpossibilityWitness> {
    upstream_pipeline::run_tower_completeness::<RuntimeResultType, HologramHasher>(
        &RuntimeResultType,
        witt_level,
    )
}
