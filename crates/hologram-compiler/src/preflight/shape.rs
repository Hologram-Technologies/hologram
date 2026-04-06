//! CompileUnitShape validation — the 4 PropertyConstraints from
//! `uor_foundation::bridge::conformance_`.
//!
//! Validates structural invariants before cascade admission.

use hologram_core::term::{HoloCompileUnit, TermArena, TermId, TermKind};
use uor_foundation::QuantumLevel;

/// Shape validation error.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// `compileUnit_rootTerm_constraint`: minCount 1, maxCount 1.
    MissingRootTerm,
    /// `compileUnit_thermodynamicBudget_constraint`: must be positive and finite.
    InvalidBudget(f64),
    /// `compileUnit_targetDomains_constraint`: minCount 1.
    NoTargetDomains,
    /// All literals must fit within the declared quantum level's ring.
    LiteralOutOfRange {
        term_id: u32,
        value: i64,
        max_value: u64,
        level: QuantumLevel,
    },
    /// Type declaration is invalid.
    InvalidTypeDecl(String),
}

impl core::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingRootTerm => write!(f, "rootTerm is missing or out of bounds"),
            Self::InvalidBudget(v) => write!(f, "thermodynamicBudget is invalid: {}", v),
            Self::NoTargetDomains => write!(f, "targetDomains is empty (at least 1 required)"),
            Self::LiteralOutOfRange {
                term_id,
                value,
                max_value,
                level,
            } => write!(
                f,
                "literal {} = {} exceeds {:?} ring max {}",
                term_id, value, level, max_value
            ),
            Self::InvalidTypeDecl(msg) => write!(f, "invalid type declaration: {}", msg),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Validate a CompileUnit against `conformance:CompileUnitShape`.
///
/// Checks the 4 PropertyConstraints:
/// - `rootTerm`: exactly 1, valid index in arena
/// - `unitQuantumLevel`: exactly 1 (satisfied by construction — `RingLevel` is not `Option`)
/// - `thermodynamicBudget`: exactly 1, positive and finite
/// - `targetDomains`: at least 1
///
/// Additionally validates quantum level consistency: all integer literals
/// must fit within the declared ring's range.
pub fn validate_shape(unit: &HoloCompileUnit) -> Result<(), ShapeError> {
    // rootTerm: exactly 1, must be a valid arena index.
    if unit.root_term.0 >= unit.arena.len() {
        return Err(ShapeError::MissingRootTerm);
    }

    // unitQuantumLevel: exactly 1 — satisfied by construction.

    // thermodynamicBudget: exactly 1, must be positive and finite.
    if unit.thermodynamic_budget <= 0.0
        || unit.thermodynamic_budget.is_nan()
        || unit.thermodynamic_budget.is_infinite()
    {
        return Err(ShapeError::InvalidBudget(unit.thermodynamic_budget));
    }

    // targetDomains: at least 1.
    if unit.target_domain_count == 0 {
        return Err(ShapeError::NoTargetDomains);
    }

    // Quantum level consistency: all literals must fit in declared ring.
    validate_literal_range(&unit.arena, unit.root_term, unit.quantum_level)?;

    Ok(())
}

/// Walk the term tree and verify all integer literals fit within the ring's range.
///
/// O(n) where n = number of term nodes (single iterative pass over the arena).
fn validate_literal_range(
    arena: &TermArena,
    _root: TermId,
    level: QuantumLevel,
) -> Result<(), ShapeError> {
    let max_value: u64 = match level.index() {
        0 => 255,
        1 => 65535,
        2 => 0x00FF_FFFF,
        _ => u32::MAX as u64,
    };

    // Iterate all nodes in the arena (O(n), sequential, cache-friendly).
    for (id, node) in arena.iter() {
        match node.kind {
            TermKind::IntLit(v) => {
                if v < 0 || (v as u64) > max_value {
                    return Err(ShapeError::LiteralOutOfRange {
                        term_id: id.0,
                        value: v,
                        max_value,
                        level,
                    });
                }
            }
            TermKind::QuantumLit { value, .. } => {
                if (value as u64) > max_value {
                    return Err(ShapeError::LiteralOutOfRange {
                        term_id: id.0,
                        value: value as i64,
                        max_value,
                        level,
                    });
                }
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::term::{HoloCompileUnit, TermArena, TermKind};
    use uor_foundation::enums::VerificationDomain;

    fn make_unit(root_kind: TermKind, level: QuantumLevel, budget: f64) -> HoloCompileUnit {
        let mut arena = TermArena::new();
        let root = arena.alloc(root_kind);
        HoloCompileUnit::new(arena, root, level, budget, &[VerificationDomain::Algebraic])
    }

    #[test]
    fn valid_unit_passes() {
        let unit = make_unit(TermKind::IntLit(42), QuantumLevel::Q0, 6.0);
        assert!(validate_shape(&unit).is_ok());
    }

    #[test]
    fn invalid_budget_nan() {
        let unit = make_unit(TermKind::IntLit(42), QuantumLevel::Q0, f64::NAN);
        assert!(matches!(
            validate_shape(&unit),
            Err(ShapeError::InvalidBudget(v)) if v.is_nan()
        ));
    }

    #[test]
    fn invalid_budget_zero() {
        let unit = make_unit(TermKind::IntLit(42), QuantumLevel::Q0, 0.0);
        assert_eq!(validate_shape(&unit), Err(ShapeError::InvalidBudget(0.0)));
    }

    #[test]
    fn invalid_budget_negative() {
        let unit = make_unit(TermKind::IntLit(42), QuantumLevel::Q0, -1.0);
        assert_eq!(validate_shape(&unit), Err(ShapeError::InvalidBudget(-1.0)));
    }

    #[test]
    fn no_target_domains() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(0));
        let unit = HoloCompileUnit::new(arena, root, QuantumLevel::Q0, 6.0, &[]);
        assert_eq!(validate_shape(&unit), Err(ShapeError::NoTargetDomains));
    }

    #[test]
    fn literal_out_of_range_q0() {
        let unit = make_unit(TermKind::IntLit(256), QuantumLevel::Q0, 6.0);
        assert!(matches!(
            validate_shape(&unit),
            Err(ShapeError::LiteralOutOfRange { .. })
        ));
    }

    #[test]
    fn literal_fits_q1() {
        let unit = make_unit(TermKind::IntLit(256), QuantumLevel::Q1, 12.0);
        assert!(validate_shape(&unit).is_ok());
    }

    #[test]
    fn negative_literal_rejected() {
        let unit = make_unit(TermKind::IntLit(-1), QuantumLevel::Q0, 6.0);
        assert!(matches!(
            validate_shape(&unit),
            Err(ShapeError::LiteralOutOfRange { .. })
        ));
    }
}
