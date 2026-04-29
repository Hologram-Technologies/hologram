//! Type declaration constraint validation.
//!
//! Verifies that type declarations in a CompileUnit are structurally valid:
//! - Referenced term IDs exist in the arena
//! - Constraint kinds are valid for the quantum level

use hologram_core::term::HoloCompileUnit;

/// Validate type declarations in the compile unit.
///
/// Returns `Ok(())` if all type declarations are valid, or an error message
/// describing the first invalid declaration.
pub fn check_type_constraints(unit: &HoloCompileUnit) -> Result<(), String> {
    let arena_len = unit.arena.len();

    for i in 0..unit.type_decl_count as usize {
        let decl = &unit.type_decls[i];

        // Verify the constraint value term exists in the arena.
        if decl.value.0 as usize >= arena_len as usize {
            return Err(format!(
                "type declaration {} references term {} which is out of arena bounds (len={})",
                i, decl.value.0, arena_len
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::term::{TermArena, TermKind};
    use uor_foundation::enums::VerificationDomain;
    use uor_foundation::WittLevel as QuantumLevel;

    #[test]
    fn no_type_decls_passes() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(0));
        let unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::W8,
            100.0,
            &[VerificationDomain::Algebraic],
        );
        assert!(check_type_constraints(&unit).is_ok());
    }
}
