//! Preflight pipeline for CompileUnit admission.
//!
//! Implements the cascade admission sequence from uor-foundation v0.1.3:
//! 1. Shape validation (`conformance:CompileUnitShape`)
//! 2. Budget solvency check (`cascade:BudgetSolvencyCheck`, preflightOrder 0, CS_6)
//! 3. Unit address computation (`op:CS_7`)

pub mod budget_solvency;
pub mod shape;
pub mod type_check;
pub mod unit_address;

pub use budget_solvency::{check_budget_solvency, minimum_budget};
pub use shape::{validate_shape, ShapeError};
pub use type_check::check_type_constraints;
pub use unit_address::compute_unit_address;

use hologram_core::term::{HoloCompileUnit, PreflightStatus};

/// Error produced during preflight checks.
#[derive(Debug, Clone, PartialEq)]
pub enum PreflightError {
    /// Shape validation failed.
    Shape(ShapeError),
    /// CS_6: thermodynamic budget is below the Landauer minimum.
    BudgetInsufficient {
        declared: f64,
        minimum: f64,
    },
}

impl core::fmt::Display for PreflightError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape(e) => write!(f, "shape validation failed: {}", e),
            Self::BudgetInsufficient { declared, minimum } => write!(
                f,
                "budget solvency failed: declared {} < minimum {}",
                declared, minimum
            ),
        }
    }
}

impl std::error::Error for PreflightError {}

impl From<ShapeError> for PreflightError {
    fn from(e: ShapeError) -> Self {
        Self::Shape(e)
    }
}

/// Run the full preflight pipeline on a CompileUnit.
///
/// Sequence (per uor-foundation v0.1.3 cascade admission):
/// 1. Shape validation (`CompileUnitShape` — 4 PropertyConstraints)
/// 2. Budget solvency check (CS_6, preflightOrder 0)
/// 3. Unit address computation (CS_7)
///
/// On success, the unit's `preflight` status and `unit_address` fields are updated.
/// On failure, returns the first error encountered (fail-fast).
pub fn run_preflight(unit: &mut HoloCompileUnit) -> Result<(), PreflightError> {
    // Step 0: Shape validation (structural precondition).
    validate_shape(unit)?;

    // Step 1: Budget solvency (preflightOrder 0, identity CS_6).
    if check_budget_solvency(unit) {
        unit.preflight.mark_passed(PreflightStatus::BUDGET_SOLVENCY);
    } else {
        unit.preflight.mark_failed(PreflightStatus::BUDGET_SOLVENCY);
        return Err(PreflightError::BudgetInsufficient {
            declared: unit.thermodynamic_budget,
            minimum: minimum_budget(unit.quantum_level),
        });
    }

    // Step 2: Type constraint validation (preflightOrder 2).
    match check_type_constraints(unit) {
        Ok(()) => unit.preflight.mark_passed(PreflightStatus::DISPATCH_COVERAGE),
        Err(msg) => {
            unit.preflight.mark_failed(PreflightStatus::DISPATCH_COVERAGE);
            return Err(PreflightError::Shape(ShapeError::InvalidTypeDecl(msg)));
        }
    }

    // Step 3: Unit address computation (identity CS_7).
    unit.unit_address = compute_unit_address(&unit.arena, unit.root_term);
    unit.address = hologram_core::term::HoloAddress::from_hash(unit.unit_address);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::{PrimOp, RingLevel};
    use hologram_core::term::{HoloCompileUnit, TermArena, TermKind};
    use uor_foundation::enums::VerificationDomain;

    #[test]
    fn preflight_passes_valid_unit() {
        let mut arena = TermArena::new();
        let a = arena.alloc(TermKind::IntLit(1));
        let b = arena.alloc(TermKind::IntLit(2));
        let root = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: a,
            rhs: b,
        });
        let mut unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        assert!(run_preflight(&mut unit).is_ok());
        assert!(unit.preflight.is_passed(PreflightStatus::BUDGET_SOLVENCY));
        assert_ne!(unit.unit_address, [0u8; 32]); // address was computed
    }

    #[test]
    fn preflight_rejects_insufficient_budget() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(0));
        let mut unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            5.0, // below Q0 minimum of 5.545
            &[VerificationDomain::Algebraic],
        );

        let result = run_preflight(&mut unit);
        assert!(result.is_err());
        match result.unwrap_err() {
            PreflightError::BudgetInsufficient { .. } => {}
            other => panic!("expected BudgetInsufficient, got {:?}", other),
        }
        assert!(!unit.preflight.is_passed(PreflightStatus::BUDGET_SOLVENCY));
    }

    #[test]
    fn preflight_rejects_missing_domains() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(0));
        let mut unit = HoloCompileUnit::new(arena, root, RingLevel::Q0, 6.0, &[]);

        let result = run_preflight(&mut unit);
        assert!(result.is_err());
        assert!(matches!(result, Err(PreflightError::Shape(ShapeError::NoTargetDomains))));
    }

    #[test]
    fn preflight_rejects_literal_out_of_range() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(256)); // > 255, out of Q0 range
        let mut unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let result = run_preflight(&mut unit);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(PreflightError::Shape(ShapeError::LiteralOutOfRange { .. }))
        ));
    }

    #[test]
    fn unit_address_deterministic() {
        let mut arena1 = TermArena::new();
        let r1 = arena1.alloc(TermKind::IntLit(42));
        let mut u1 = HoloCompileUnit::new(
            arena1,
            r1,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );
        run_preflight(&mut u1).unwrap();

        let mut arena2 = TermArena::new();
        let r2 = arena2.alloc(TermKind::IntLit(42));
        let mut u2 = HoloCompileUnit::new(
            arena2,
            r2,
            RingLevel::Q0,
            10.0, // different budget — should not affect address
            &[
                VerificationDomain::Algebraic,
                VerificationDomain::Thermodynamic,
            ], // different domains — should not affect address
        );
        run_preflight(&mut u2).unwrap();

        assert_eq!(
            u1.unit_address, u2.unit_address,
            "same rootTerm must produce same address regardless of budget/domains"
        );
    }

    #[test]
    fn different_terms_different_addresses() {
        let mut arena1 = TermArena::new();
        let r1 = arena1.alloc(TermKind::IntLit(1));
        let mut u1 = HoloCompileUnit::new(
            arena1,
            r1,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );
        run_preflight(&mut u1).unwrap();

        let mut arena2 = TermArena::new();
        let r2 = arena2.alloc(TermKind::IntLit(2));
        let mut u2 = HoloCompileUnit::new(
            arena2,
            r2,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );
        run_preflight(&mut u2).unwrap();

        assert_ne!(u1.unit_address, u2.unit_address);
    }
}
