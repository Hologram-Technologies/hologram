//! Typed effect declarations for the cascade pipeline.
//!
//! Per PRISM Section 3.7, effects are typed context endomorphisms applied
//! through the cascade's stage machinery. Uses the enforcement module's
//! `EffectDeclarationBuilder` for validation.
//!
//! Fixed-capacity, zero-allocation storage matching hologram patterns.

use uor_foundation::enforcement::{EffectDeclarationBuilder, ShapeViolation};

/// Maximum effect declarations per cascade run.
pub const MAX_EFFECTS: usize = 32;

/// A validated effect declaration. Fixed-size, Copy, no heap.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedEffect {
    /// Budget delta (positive = fiber increment, negative = decrement).
    pub budget_delta: i64,
    /// Whether this effect commutes with effects on disjoint fibers.
    pub commutes: bool,
    /// Number of target fibers.
    pub fiber_count: u8,
    /// Target fiber indices (up to 8 fibers per effect).
    pub target_fibers: [u32; 8],
}

/// Fixed-capacity effect declaration store. No heap allocation.
#[derive(Debug)]
pub struct EffectStore {
    effects: [ValidatedEffect; MAX_EFFECTS],
    len: u8,
}

impl Default for EffectStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectStore {
    /// Create an empty effect store.
    #[inline]
    pub const fn new() -> Self {
        Self {
            effects: [ValidatedEffect {
                budget_delta: 0,
                commutes: true,
                fiber_count: 0,
                target_fibers: [0; 8],
            }; MAX_EFFECTS],
            len: 0,
        }
    }

    /// Validate and register an effect declaration.
    /// Returns the index of the registered effect.
    pub fn register(
        &mut self,
        name: &str,
        target_fibers: &[u32],
        budget_delta: i64,
        commutes: bool,
    ) -> Result<u8, ShapeViolation> {
        let _validated = EffectDeclarationBuilder::new()
            .name(name)
            .target_sites(target_fibers)
            .budget_delta(budget_delta)
            .commutes(commutes)
            .validate()?;

        if self.len as usize >= MAX_EFFECTS {
            return Err(ShapeViolation {
                shape_iri: "https://uor.foundation/conformance/EffectShape",
                constraint_iri: "hologram:effect_capacity",
                property_iri: "hologram:effectCount",
                expected_range: "xsd:nonNegativeInteger",
                min_count: 0,
                max_count: MAX_EFFECTS as u32,
                kind: uor_foundation::ViolationKind::CardinalityViolation,
            });
        }

        let idx = self.len;
        let mut fibers = [0u32; 8];
        let count = target_fibers.len().min(8);
        fibers[..count].copy_from_slice(&target_fibers[..count]);

        self.effects[idx as usize] = ValidatedEffect {
            budget_delta,
            commutes,
            fiber_count: count as u8,
            target_fibers: fibers,
        };
        self.len += 1;
        Ok(idx)
    }

    /// Get an effect by index. O(1).
    #[inline]
    pub fn get(&self, index: u8) -> Option<&ValidatedEffect> {
        if index < self.len {
            Some(&self.effects[index as usize])
        } else {
            None
        }
    }

    /// Number of registered effects.
    #[inline]
    pub const fn len(&self) -> u8 {
        self.len
    }

    /// Whether the store is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_store_register_and_get() {
        let mut store = EffectStore::new();
        let idx = store.register("blit", &[0, 1, 2], 0, true).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(store.len(), 1);
        let effect = store.get(0).unwrap();
        assert_eq!(effect.budget_delta, 0);
        assert!(effect.commutes);
        assert_eq!(effect.fiber_count, 3);
    }

    #[test]
    fn effect_store_empty() {
        let store = EffectStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.get(0).is_none());
    }
}
