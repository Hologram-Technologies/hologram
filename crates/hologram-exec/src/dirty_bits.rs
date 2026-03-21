//! Dirty-bit tracking for incremental execution.
//!
//! During autoregressive decode, most intermediate buffers are unchanged
//! between steps (only the new token's path through the graph changes).
//! DirtyBits tracks which nodes have been modified, enabling skip-if-unchanged
//! for nodes whose inputs haven't changed.
//!
//! This is Phase 13 of the Compile-Time-First Acceleration plan.

/// Bit vector tracking which node indices have been modified.
///
/// During decode, mark a node dirty when its output buffer changes.
/// Before dispatching a node, check if ALL its inputs are clean —
/// if so, the output is also unchanged and dispatch can be skipped.
pub struct DirtyBits {
    /// One bit per node index. `true` = output has changed.
    bits: Vec<u64>,
}

impl DirtyBits {
    /// Create a dirty-bit tracker for `n` nodes, all initially clean.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let words = n.div_ceil(64);
        Self {
            bits: vec![0u64; words],
        }
    }

    /// Mark a node as dirty (its output has changed).
    #[inline]
    pub fn mark_dirty(&mut self, node_idx: u32) {
        let word = node_idx as usize / 64;
        let bit = node_idx as usize % 64;
        if word < self.bits.len() {
            self.bits[word] |= 1 << bit;
        }
    }

    /// Check if a node is dirty.
    #[inline]
    #[must_use]
    pub fn is_dirty(&self, node_idx: u32) -> bool {
        let word = node_idx as usize / 64;
        let bit = node_idx as usize % 64;
        if word < self.bits.len() {
            (self.bits[word] >> bit) & 1 != 0
        } else {
            false
        }
    }

    /// Check if ANY of the given node indices are dirty.
    #[inline]
    #[must_use]
    pub fn any_dirty(&self, indices: &[u32]) -> bool {
        indices.iter().any(|&idx| self.is_dirty(idx))
    }

    /// Mark all nodes as clean (reset for next decode step).
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
    }

    /// Mark all nodes as dirty (used for prefill where everything changes).
    pub fn mark_all_dirty(&mut self) {
        for word in &mut self.bits {
            *word = u64::MAX;
        }
    }
}

impl Default for DirtyBits {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_all_clean() {
        let db = DirtyBits::new(100);
        for i in 0..100 {
            assert!(!db.is_dirty(i));
        }
    }

    #[test]
    fn mark_and_check() {
        let mut db = DirtyBits::new(200);
        db.mark_dirty(42);
        db.mark_dirty(100);
        assert!(db.is_dirty(42));
        assert!(db.is_dirty(100));
        assert!(!db.is_dirty(41));
        assert!(!db.is_dirty(43));
    }

    #[test]
    fn clear_resets() {
        let mut db = DirtyBits::new(100);
        db.mark_dirty(5);
        db.mark_dirty(50);
        db.clear();
        assert!(!db.is_dirty(5));
        assert!(!db.is_dirty(50));
    }

    #[test]
    fn any_dirty_checks_group() {
        let mut db = DirtyBits::new(100);
        db.mark_dirty(10);
        assert!(db.any_dirty(&[5, 10, 15]));
        assert!(!db.any_dirty(&[5, 15, 20]));
    }

    #[test]
    fn mark_all_dirty() {
        let mut db = DirtyBits::new(100);
        db.mark_all_dirty();
        for i in 0..100 {
            assert!(db.is_dirty(i));
        }
    }

    #[test]
    fn boundary_bits() {
        let mut db = DirtyBits::new(128);
        db.mark_dirty(63); // Last bit of first word
        db.mark_dirty(64); // First bit of second word
        assert!(db.is_dirty(63));
        assert!(db.is_dirty(64));
        assert!(!db.is_dirty(62));
        assert!(!db.is_dirty(65));
    }
}
