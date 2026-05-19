//! Hologram type vocabulary (spec Part IV).
//!
//! Hologram is a Prism application (wiki ADR-031). The canonical
//! shape carriers — `MatrixShape<R, C, E>`, `VectorShape<N, E>`,
//! `Digest<N>` — are imported from the prism standard-library
//! Layer-3 sub-crates (`prism::tensor`, `prism::crypto`) and reach
//! hologram callers through this crate's re-exports.
//!
//! What `hologram-types` adds beyond the prism vocabulary:
//!
//! - `DType*` markers (F32, F16, BF16, F64, I64, I32, I8, I4, U64, U8,
//!   Bool) with a `DType` trait carrying `BIT_WIDTH` and `KIND` for the
//!   compiler's per-op dtype resolution. The corresponding upstream
//!   ConstrainedTypeShape declarations supply `CYCLE_SIZE` per ADR-032.
//! - `Dim<N>` and `Shape1/Shape2` markers for the graph IR's rank-1 /
//!   rank-2 type-level shapes. Higher ranks compose through `partition_product!`
//!   per ADR-033/044 (prism's canonical pattern); see `hologram-ops`
//!   axis declarations for the composition.
//!
//! All other ConstrainedTypeShape carriers (matrix/vector shapes,
//! fingerprints, tensor data carriers, region/layout/weight/schedule
//! markers) come directly from the prism façade. Hologram declares
//! no parallel duplicates per the user's ADR-031 directive.

#![no_std]

pub mod dtype;
pub mod shape;

pub use dtype::{
    DType, DTypeBf16, DTypeBool, DTypeF16, DTypeF32, DTypeF64, DTypeI32, DTypeI4, DTypeI64,
    DTypeI8, DTypeKind, DTypeU64, DTypeU8,
};
pub use shape::{Dim, Shape1, Shape2};

// Re-export the canonical prism shape carriers so hologram callers
// reach them through this crate's surface without an extra import.
pub use prism::crypto::Digest;
pub use prism::tensor::{MatrixShape, VectorShape};

/// IRI prefix for the hologram-side type namespace. Hologram-introduced
/// types (Dim, Shape1, Shape2, dtypes) live under this prefix; all
/// other shape carriers use prism's `https://uor.foundation/type/...`
/// namespace.
pub const IRI_PREFIX: &str = "https://hologram.uor.foundation/type/";
