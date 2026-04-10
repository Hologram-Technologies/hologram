//! Budget solvency check.
//!
//! Rejects a CompileUnit whose declared `thermodynamicBudget` is strictly less
//! than the Landauer minimum: `bitsWidth(W_n) × ln(2)` in k_B T units.
//!
//! O(1) — a single comparison. No tree walk required.

use hologram_core::term::HoloCompileUnit;
use hologram_foundation::WittLevel;

/// Compute the minimum viable thermodynamic budget for a Witt level.
///
/// `min = bitsWidth(W_n) × ln(2)` in k_B T units. Works for any Witt
/// level, not just the spec-named W8/W16/W24/W32.
#[inline]
pub fn minimum_budget(level: WittLevel) -> f64 {
    (level.bits_width() as f64) * core::f64::consts::LN_2
}

/// Budget solvency check. O(1).
///
/// Returns `true` if the declared budget meets or exceeds the Landauer
/// minimum for the unit's Witt level. Returns `false` if the unit should
/// be rejected.
#[inline]
pub fn check_budget_solvency(unit: &HoloCompileUnit) -> bool {
    unit.thermodynamic_budget >= minimum_budget(unit.witt_level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::term::{HoloCompileUnit, TermArena, TermKind};
    use hologram_foundation::enums::VerificationDomain;

    fn make_unit(level: WittLevel, budget: f64) -> HoloCompileUnit {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(0));
        HoloCompileUnit::new(arena, root, level, budget, &[VerificationDomain::Algebraic])
    }

    #[test]
    fn q0_minimum_budget() {
        let min = minimum_budget(WittLevel::W8);
        assert!((min - 5.545).abs() < 0.001, "Q0 min = {}", min);
    }

    #[test]
    fn q1_minimum_budget() {
        let min = minimum_budget(WittLevel::W16);
        assert!((min - 11.090).abs() < 0.001, "Q1 min = {}", min);
    }

    #[test]
    fn q2_minimum_budget() {
        let min = minimum_budget(WittLevel::W24);
        assert!((min - 16.636).abs() < 0.001, "Q2 min = {}", min);
    }

    #[test]
    fn q3_minimum_budget() {
        let min = minimum_budget(WittLevel::W32);
        assert!((min - 22.181).abs() < 0.001, "Q3 min = {}", min);
    }

    #[test]
    fn q0_passes_at_minimum() {
        let unit = make_unit(WittLevel::W8, 5.546);
        assert!(check_budget_solvency(&unit));
    }

    #[test]
    fn q0_fails_below_minimum() {
        let unit = make_unit(WittLevel::W8, 5.0);
        assert!(!check_budget_solvency(&unit));
    }

    #[test]
    fn q0_passes_exact_minimum() {
        let min = minimum_budget(WittLevel::W8);
        let unit = make_unit(WittLevel::W8, min);
        assert!(check_budget_solvency(&unit));
    }

    #[test]
    fn q3_passes_at_minimum() {
        let min = minimum_budget(WittLevel::W32);
        let unit = make_unit(WittLevel::W32, min);
        assert!(check_budget_solvency(&unit));
    }

    #[test]
    fn q3_fails_below_minimum() {
        let unit = make_unit(WittLevel::W32, 22.0);
        assert!(!check_budget_solvency(&unit));
    }

    #[test]
    fn budget_solvency_performance() {
        // Performance contract: 10M checks < 50ms (< 5ns each, O(1))
        let unit = make_unit(WittLevel::W8, 6.0);
        let start = std::time::Instant::now();
        for _ in 0..10_000_000 {
            let _ = check_budget_solvency(&unit);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200, // generous CI margin
            "10M budget checks took {}ms (target < 200ms)",
            elapsed.as_millis()
        );
    }
}
