//! Dispatch declaration registry for the cascade pipeline.
//!
//! Per PRISM Section 2.2, dispatch rules map intents to resolvers with
//! priority ordering. Fixed-capacity, no heap allocation.

use uor_foundation::enforcement::{DispatchDeclarationBuilder, ShapeViolation, Term};

/// Maximum dispatch rules per cascade.
pub const MAX_DISPATCH_RULES: usize = 16;

/// A validated dispatch rule. Fixed-size, Copy, no heap.
#[derive(Debug, Clone, Copy)]
pub struct DispatchRule {
    /// Resolver identifier (index into a resolver table).
    pub resolver_id: u16,
    /// Priority (lower = higher priority).
    pub priority: u16,
}

/// Fixed-capacity dispatch registry. Rules sorted by priority.
#[derive(Debug)]
pub struct DispatchRegistry {
    rules: [DispatchRule; MAX_DISPATCH_RULES],
    len: u8,
}

impl DispatchRegistry {
    /// Create an empty registry.
    #[inline]
    pub const fn new() -> Self {
        Self {
            rules: [DispatchRule {
                resolver_id: 0,
                priority: u16::MAX,
            }; MAX_DISPATCH_RULES],
            len: 0,
        }
    }

    /// Validate and register a dispatch rule.
    /// Rules are insertion-sorted by priority (O(n), n <= 16).
    pub fn register(
        &mut self,
        predicate: &[Term],
        resolver_name: &str,
        resolver_id: u16,
        priority: u16,
    ) -> Result<(), ShapeViolation> {
        let _validated = DispatchDeclarationBuilder::new()
            .predicate(predicate)
            .target_resolver(resolver_name)
            .priority(priority as u32)
            .validate()?;

        if self.len as usize >= MAX_DISPATCH_RULES {
            return Err(ShapeViolation {
                shape_iri: "https://uor.foundation/conformance/DispatchShape",
                constraint_iri: "hologram:dispatch_capacity",
                property_iri: "hologram:dispatchRuleCount",
                expected_range: "xsd:nonNegativeInteger",
                min_count: 0,
                max_count: MAX_DISPATCH_RULES as u32,
                kind: uor_foundation::ViolationKind::CardinalityViolation,
            });
        }

        // Insertion sort by priority
        let new_rule = DispatchRule {
            resolver_id,
            priority,
        };
        let mut pos = self.len as usize;
        while pos > 0 && self.rules[pos - 1].priority > priority {
            self.rules[pos] = self.rules[pos - 1];
            pos -= 1;
        }
        self.rules[pos] = new_rule;
        self.len += 1;
        Ok(())
    }

    /// Get the highest-priority dispatch rule. O(1).
    #[inline]
    pub fn highest_priority(&self) -> Option<&DispatchRule> {
        if self.len > 0 {
            Some(&self.rules[0])
        } else {
            None
        }
    }

    /// Get all registered rules (sorted by priority).
    #[inline]
    pub fn rules(&self) -> &[DispatchRule] {
        &self.rules[..self.len as usize]
    }

    /// Number of registered rules.
    #[inline]
    pub const fn len(&self) -> u8 {
        self.len
    }

    /// Whether the registry is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_registry_empty() {
        let reg = DispatchRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.highest_priority().is_none());
    }
}
