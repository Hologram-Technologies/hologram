//! Hologram domain type vocabulary (spec Part IV).
//!
//! All hologram types are `ConstrainedTypeShape` declarations. There is no
//! parallel hand-rolled type system. The IRI namespace
//! `https://hologram.uor.foundation/type/...` is a Prism extension (per
//! ADR-013 and spec C-3): hologram introduces types, never new `PrimitiveOp`s.

#![no_std]

pub mod dtype;
pub mod shape;
pub mod tensor;
pub mod region;
pub mod layout;
pub mod weight;
pub mod schedule;
pub mod fingerprint;
pub mod witness;

pub use dtype::{
    DType, DTypeKind,
    DTypeF32, DTypeF16, DTypeBf16, DTypeF64,
    DTypeI64, DTypeI32, DTypeI8, DTypeI4, DTypeU64, DTypeU8, DTypeBool,
};
pub use shape::{
    Dim, DimSymbolic,
    Shape1, Shape2, Shape3, Shape4, Shape5, Shape6, Shape7, Shape8,
    ShapeArray,
};
pub use tensor::Tensor;
pub use region::Region;
pub use layout::Layout;
pub use weight::{Weight, Constant};
pub use schedule::Schedule;
pub use fingerprint::Fingerprint;
pub use witness::WitnessRecord;

/// IRI prefix for the hologram type namespace.
pub const IRI_PREFIX: &str = "https://hologram.uor.foundation/type/";
