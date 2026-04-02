//! Append-only arena allocator for term nodes.
//!
//! O(1) allocation, O(1) indexed access, zero-allocation steady state.
//!
//! - `std` / default mode: `Vec`-backed with dynamic growth.
//! - `no_alloc` feature: fixed-capacity 4096-node buffer (64 KB).

use super::{TermId, TermKind, TermNode, TypeId};

#[cfg(not(feature = "no_alloc"))]
use super::{FloatOpRef, ViewRef};
#[cfg(not(feature = "no_alloc"))]
use crate::op::FloatOp;
#[cfg(not(feature = "no_alloc"))]
use crate::view::ElementWiseView;

/// Maximum arena capacity in `no_alloc` mode.
/// 4096 nodes * ~16 bytes = 64 KB, fits in L1 cache.
#[cfg(feature = "no_alloc")]
const ARENA_CAPACITY: usize = 4096;

/// Append-only arena for [`TermNode`] values.
///
/// Nodes are allocated sequentially and referenced by [`TermId`] index.
/// The arena never frees or reuses slots — it grows monotonically.
///
/// Side tables store large types (FloatOp, ElementWiseView) that would
/// bloat the 16-byte TermKind if inlined. Referenced via indices
/// (`FloatOpRef`, `ViewRef`).
pub struct TermArena {
    #[cfg(not(feature = "no_alloc"))]
    nodes: alloc::vec::Vec<TermNode>,

    #[cfg(feature = "no_alloc")]
    nodes: [TermNode; ARENA_CAPACITY],
    #[cfg(feature = "no_alloc")]
    len: u32,

    /// Side table for FloatOp values (indexed by `FloatOpRef`).
    #[cfg(not(feature = "no_alloc"))]
    float_ops: alloc::vec::Vec<FloatOp>,
    /// Side table for fused ElementWiseView values (indexed by `ViewRef`).
    #[cfg(not(feature = "no_alloc"))]
    views: alloc::vec::Vec<ElementWiseView>,
}

#[cfg(not(feature = "no_alloc"))]
extern crate alloc;

impl TermArena {
    /// Create a new empty arena.
    #[cfg(not(feature = "no_alloc"))]
    pub fn new() -> Self {
        Self {
            nodes: alloc::vec::Vec::new(),
            float_ops: alloc::vec::Vec::new(),
            views: alloc::vec::Vec::new(),
        }
    }

    /// Create a new empty arena with pre-allocated capacity.
    #[cfg(not(feature = "no_alloc"))]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: alloc::vec::Vec::with_capacity(cap),
            float_ops: alloc::vec::Vec::new(),
            views: alloc::vec::Vec::new(),
        }
    }

    /// Create a new empty arena (fixed-capacity mode).
    #[cfg(feature = "no_alloc")]
    pub const fn new() -> Self {
        Self {
            nodes: [TermNode {
                kind: TermKind::IntLit(0),
                ty: TypeId::UNCONSTRAINED,
            }; ARENA_CAPACITY],
            len: 0,
        }
    }

    /// Allocate a new term node and return its [`TermId`].
    ///
    /// O(1) amortized. Panics on `no_alloc` overflow (> 4096 nodes).
    #[inline]
    pub fn alloc(&mut self, kind: TermKind) -> TermId {
        self.alloc_typed(kind, TypeId::UNCONSTRAINED)
    }

    /// Allocate a new term node with a type annotation.
    #[inline]
    pub fn alloc_typed(&mut self, kind: TermKind, ty: TypeId) -> TermId {
        let id = self.len();

        #[cfg(not(feature = "no_alloc"))]
        {
            self.nodes.push(TermNode { kind, ty });
        }

        #[cfg(feature = "no_alloc")]
        {
            assert!(
                (id as usize) < ARENA_CAPACITY,
                "TermArena overflow: {} >= {}",
                id,
                ARENA_CAPACITY
            );
            self.nodes[id as usize] = TermNode { kind, ty };
            self.len += 1;
        }

        TermId(id)
    }

    /// Access a node by its [`TermId`]. Panics if out of bounds.
    #[inline]
    pub fn get(&self, id: TermId) -> &TermNode {
        let idx = id.0 as usize;
        debug_assert!(
            idx < self.len() as usize,
            "TermId {} out of bounds (len {})",
            idx,
            self.len()
        );

        #[cfg(not(feature = "no_alloc"))]
        {
            &self.nodes[idx]
        }

        #[cfg(feature = "no_alloc")]
        {
            &self.nodes[idx]
        }
    }

    /// Number of allocated nodes.
    #[inline]
    pub fn len(&self) -> u32 {
        #[cfg(not(feature = "no_alloc"))]
        {
            self.nodes.len() as u32
        }

        #[cfg(feature = "no_alloc")]
        {
            self.len
        }
    }

    /// Returns `true` if the arena contains no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Store a `FloatOp` in the side table and return its reference.
    #[cfg(not(feature = "no_alloc"))]
    pub fn alloc_float_op(&mut self, op: FloatOp) -> FloatOpRef {
        let idx = self.float_ops.len();
        self.float_ops.push(op);
        FloatOpRef(idx as u32)
    }

    /// Retrieve a `FloatOp` by its reference.
    #[cfg(not(feature = "no_alloc"))]
    pub fn get_float_op(&self, r: FloatOpRef) -> &FloatOp {
        &self.float_ops[r.0 as usize]
    }

    /// Store an `ElementWiseView` in the side table and return its reference.
    #[cfg(not(feature = "no_alloc"))]
    pub fn alloc_view(&mut self, view: ElementWiseView) -> ViewRef {
        let idx = self.views.len();
        self.views.push(view);
        ViewRef(idx as u32)
    }

    /// Retrieve an `ElementWiseView` by its reference.
    #[cfg(not(feature = "no_alloc"))]
    pub fn get_view(&self, r: ViewRef) -> &ElementWiseView {
        &self.views[r.0 as usize]
    }

    /// Iterate over all allocated nodes with their IDs.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (TermId, &TermNode)> {
        let len = self.len() as usize;

        #[cfg(not(feature = "no_alloc"))]
        let slice = &self.nodes[..len];

        #[cfg(feature = "no_alloc")]
        let slice = &self.nodes[..len];

        slice
            .iter()
            .enumerate()
            .map(|(i, node)| (TermId(i as u32), node))
    }
}

#[cfg(not(feature = "no_alloc"))]
impl Default for TermArena {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for TermArena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TermArena")
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec::Vec;
    use super::*;
    use crate::op::PrimOp;

    #[test]
    fn alloc_and_get() {
        let mut arena = TermArena::new();
        let id0 = arena.alloc(TermKind::IntLit(42));
        let id1 = arena.alloc(TermKind::BrailleLit(0xFF));
        let id2 = arena.alloc(TermKind::UnaryApp {
            op: PrimOp::Neg,
            arg: id0,
        });

        assert_eq!(id0.0, 0);
        assert_eq!(id1.0, 1);
        assert_eq!(id2.0, 2);
        assert_eq!(arena.len(), 3);

        assert_eq!(arena.get(id0).kind, TermKind::IntLit(42));
        assert_eq!(arena.get(id1).kind, TermKind::BrailleLit(0xFF));
        match arena.get(id2).kind {
            TermKind::UnaryApp { op, arg } => {
                assert_eq!(op, PrimOp::Neg);
                assert_eq!(arg, id0);
            }
            other => panic!("expected UnaryApp, got {:?}", other),
        }
    }

    #[test]
    fn alloc_typed() {
        let mut arena = TermArena::new();
        let ty = TypeId(7);
        let id = arena.alloc_typed(TermKind::IntLit(1), ty);
        assert_eq!(arena.get(id).ty, ty);
    }

    #[test]
    fn empty_arena() {
        let arena = TermArena::new();
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn iter_all() {
        let mut arena = TermArena::new();
        for i in 0..10 {
            arena.alloc(TermKind::IntLit(i));
        }
        let collected: Vec<_> = arena.iter().collect();
        assert_eq!(collected.len(), 10);
        for (i, (id, node)) in collected.iter().enumerate() {
            assert_eq!(id.0, i as u32);
            assert_eq!(node.kind, TermKind::IntLit(i as i64));
        }
    }

    #[test]
    fn binary_app_references() {
        let mut arena = TermArena::new();
        let a = arena.alloc(TermKind::IntLit(1));
        let b = arena.alloc(TermKind::IntLit(2));
        let sum = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: a,
            rhs: b,
        });

        match arena.get(sum).kind {
            TermKind::BinaryApp { op, lhs, rhs } => {
                assert_eq!(op, PrimOp::Add);
                assert_eq!(arena.get(lhs).kind, TermKind::IntLit(1));
                assert_eq!(arena.get(rhs).kind, TermKind::IntLit(2));
            }
            other => panic!("expected BinaryApp, got {:?}", other),
        }
    }

    #[test]
    fn with_capacity() {
        let arena = TermArena::with_capacity(1024);
        assert!(arena.is_empty());
    }
}
