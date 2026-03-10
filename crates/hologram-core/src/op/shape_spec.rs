//! Symbolic output shape specifications for FloatOp.
//!
//! Each `FloatOp` declares its output shape behavior via `ShapeSpec`. The
//! executor resolves these against actual input shapes at runtime, replacing
//! the scattered shape logic that was previously duplicated across
//! `executor.rs` and `float_dispatch.rs`.
//!
//! This module is runtime-only — `ShapeSpec` is not serialized into `.holo`
//! archives. The archive format continues to use `node_shapes: Vec<(NodeId,
//! Vec<usize>)>` with 0-sentinels for symbolic dimensions.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// How to compute a single output dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeDim {
    /// Copy dimension from `input[input_idx].shape[axis]`.
    /// Negative `axis` counts from end (-1 = last dimension).
    FromInput { input: u8, axis: i8 },
    /// Fixed compile-time constant (e.g., embedding dim, head count).
    Fixed(u32),
    /// Computed at runtime from `total_elements / product_of_known_dims`.
    Inferred,
}

/// Symbolic output shape specification for a `FloatOp`.
///
/// The executor resolves these against actual input shapes at runtime.
/// This replaces the scattered, per-op shape logic with a single
/// declarative source of truth on each `FloatOp` variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeSpec {
    /// Output shape = `input[i].shape` (unary elementwise, norms, cast, etc.)
    SameAs(u8),
    /// Output shape = broadcast of `input[a]` and `input[b]` shapes.
    /// Uses the longer shape (higher rank).
    Broadcast(u8, u8),
    /// Output shape = `input[i].shape` with last dimension removed (reductions).
    DropLastDim(u8),
    /// Output shape described per-dimension via `ShapeDim` entries.
    Dims(Vec<ShapeDim>),
    /// Requires op-specific logic (MatMul, Reshape, Transpose, etc.).
    /// The executor delegates to dedicated handlers for these ops.
    Custom,
}

impl ShapeSpec {
    /// Create a `Dims` spec with a single inferred dimension (1-D output).
    #[must_use]
    pub fn inferred_1d() -> Self {
        Self::Dims(vec![ShapeDim::Inferred])
    }

    /// Create a `Dims` spec with `[Inferred, Fixed(dim)]` (e.g., Embed, Gather).
    #[must_use]
    pub fn inferred_by_fixed(dim: u32) -> Self {
        Self::Dims(vec![ShapeDim::Inferred, ShapeDim::Fixed(dim)])
    }
}
