//! Typed addressing primitives.
//!
//! These types describe *which* object — never *how* to compute it. They are
//! the bridge between the LUT / identity layer (sourced from `uor-foundation`)
//! and the planner. Address values are stable symbolic references **before**
//! planning; the planner resolves them into concrete `SlotSpan`s.
//!
//! See [ADR-043](../../specs/adrs/043-lut-addressed-transform-chains.md).

/// Index into a `TransformChain`'s tensor table.
///
/// Stable across planning. The planner uses `TensorId.0 as usize` as an
/// O(1) index into the resolved `AddressTable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u32);

/// Index into a `TransformChain`'s node list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Identity of an abstract memory region.
///
/// Reserved for future use; lets the planner co-locate tensors that share
/// a backing region (e.g. workspace arenas, KV-cache slabs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId(pub u32);

/// Identity of a layout descriptor (row-major, tiled, transposed, …).
///
/// Layouts are looked up in a separate LUT so kernels never re-derive
/// strides at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayoutId(pub u32);

/// Default layout: row-major, contiguous, no transposition.
pub const DEFAULT_LAYOUT: LayoutId = LayoutId(0);

/// A symbolic, typed reference to a tensor slot.
///
/// `AddressRef` is stable from chain construction through planning. The
/// planner consumes these refs and produces concrete `SlotSpan`s; kernels
/// only ever see the resolved spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AddressRef {
    /// Which tensor this address refers to.
    pub tensor: TensorId,
    /// Which layout view of that tensor.
    pub layout: LayoutId,
}

impl AddressRef {
    /// Construct an address ref against the default (row-major) layout.
    #[inline]
    #[must_use]
    pub const fn of(tensor: TensorId) -> Self {
        Self {
            tensor,
            layout: DEFAULT_LAYOUT,
        }
    }

    /// Construct an address ref with an explicit layout.
    #[inline]
    #[must_use]
    pub const fn with_layout(tensor: TensorId, layout: LayoutId) -> Self {
        Self { tensor, layout }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_of_uses_default_layout() {
        let r = AddressRef::of(TensorId(7));
        assert_eq!(r.tensor, TensorId(7));
        assert_eq!(r.layout, DEFAULT_LAYOUT);
    }

    #[test]
    fn address_with_layout_keeps_layout() {
        let r = AddressRef::with_layout(TensorId(2), LayoutId(3));
        assert_eq!(r.tensor.0, 2);
        assert_eq!(r.layout.0, 3);
    }

    #[test]
    fn ids_are_copy_and_orderable() {
        let mut ids = [TensorId(2), TensorId(0), TensorId(1)];
        ids.sort();
        assert_eq!(ids, [TensorId(0), TensorId(1), TensorId(2)]);
    }
}
