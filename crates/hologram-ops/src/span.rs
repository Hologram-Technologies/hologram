//! Concrete location of a tensor inside a flat workspace buffer.
//!
//! `SlotSpan` is the address layer's resolved form: a planner takes
//! symbolic `AddressRef`s and produces `SlotSpan`s the executor uses
//! directly. Lengths and offsets are in **elements**, not bytes.

/// Offset and length, in elements, into the planner's flat workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotSpan {
    /// Offset into the buffer storage.
    pub offset: usize,
    /// Length in elements.
    pub len: usize,
}

impl SlotSpan {
    /// Construct an empty span at the given offset.
    #[inline]
    #[must_use]
    pub const fn empty(offset: usize) -> Self {
        Self { offset, len: 0 }
    }
}
