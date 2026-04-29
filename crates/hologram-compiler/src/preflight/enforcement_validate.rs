//! Enforcement-level validation using uor-foundation v0.1.4 builders.
//!
//! Uses `enforcement::CompileUnitBuilder.validate()` as a declarative
//! preflight step, complementing the existing shape validation.
//!
//! The builder checks:
//! - Tier 1: rootTerm present, quantumLevelCeiling present,
//!   thermodynamicBudget present, targetDomains non-empty.
//! - Tier 2: budget solvency, level coherence.

use hologram_core::term::enforcement_bridge::arena_to_enforcement_terms;
use hologram_core::term::HoloCompileUnit;
use uor_foundation::enforcement::{CompileUnitBuilder, ShapeViolation};

/// Run enforcement-level validation on a `HoloCompileUnit`.
///
/// Converts the hologram term arena to enforcement terms, then uses the
/// v0.1.4 `CompileUnitBuilder` to validate structural and value constraints.
///
/// Returns `Ok(())` on success, or the structured `ShapeViolation` on failure.
pub fn enforcement_validate(unit: &HoloCompileUnit) -> Result<(), ShapeViolation> {
    let level = unit.quantum_level;
    let terms = arena_to_enforcement_terms(&unit.arena, level);

    CompileUnitBuilder::new()
        .root_term(&terms)
        .witt_level_ceiling(level)
        .thermodynamic_budget(unit.thermodynamic_budget as u64)
        .target_domains(&unit.target_domains_array[..unit.target_domain_count as usize])
        // v0.3.0 added a required `result_type` shape. Hologram doesn't
        // yet track per-unit result types — declare the open
        // `ConstrainedTypeInput` shape so the validator accepts the unit.
        // A future ADR can wire real ConstrainedTypeShape impls through
        // preflight when hologram has real type metadata to attach.
        .result_type::<uor_foundation::enforcement::ConstrainedTypeInput>()
        .validate()
        .map(|_validated| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::PrimOp;
    use hologram_core::term::{TermArena, TermKind};
    use uor_foundation::enums::VerificationDomain;
    use uor_foundation::{ViolationKind, WittLevel as QuantumLevel};

    #[test]
    fn enforcement_validate_passes_valid_unit() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(42));
        let unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::W8,
            6.0,
            &[VerificationDomain::Algebraic],
        );
        assert!(enforcement_validate(&unit).is_ok());
    }

    #[test]
    fn enforcement_validate_passes_complex_unit() {
        let mut arena = TermArena::new();
        let a = arena.alloc(TermKind::IntLit(3));
        let b = arena.alloc(TermKind::IntLit(5));
        let root = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: a,
            rhs: b,
        });
        let unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::W16,
            12.0,
            &[
                VerificationDomain::Algebraic,
                VerificationDomain::Thermodynamic,
            ],
        );
        assert!(enforcement_validate(&unit).is_ok());
    }

    #[test]
    fn enforcement_validate_rejects_empty_domains() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(42));
        let unit = HoloCompileUnit::new(arena, root, QuantumLevel::W8, 6.0, &[]);
        let err = enforcement_validate(&unit);
        assert!(err.is_err());
        let violation = err.unwrap_err();
        assert_eq!(violation.kind, ViolationKind::Missing);
        assert!(violation.property_iri.contains("targetDomains"));
    }
}
