//! Dtype declarations (spec IV.4).
//!
//! Each dtype is a leaf `ConstrainedTypeShape` whose IRI carries bit-width
//! and signedness. The constraint list is empty; structural information is
//! recovered through the `DType` marker trait, not through `Site::position`.

use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};

/// Marker trait identifying a hologram dtype.
pub trait DType: ConstrainedTypeShape {
    const BIT_WIDTH: u32;
    const KIND: DTypeKind;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeKind {
    Float,
    Bfloat,
    SignedInt,
    UnsignedInt,
    Bool,
}

macro_rules! declare_dtype {
    ($ty:ident, $iri:literal, $bw:expr, $kind:expr) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $ty;

        impl ConstrainedTypeShape for $ty {
            const IRI: &'static str = $iri;
            const SITE_COUNT: usize = 1;
            const CONSTRAINTS: &'static [ConstraintRef] = &[];
        }

        impl DType for $ty {
            const BIT_WIDTH: u32 = $bw;
            const KIND: DTypeKind = $kind;
        }
    };
}

declare_dtype!(DTypeF32,  "https://hologram.uor.foundation/type/dtype/f32",  32, DTypeKind::Float);
declare_dtype!(DTypeF16,  "https://hologram.uor.foundation/type/dtype/f16",  16, DTypeKind::Float);
declare_dtype!(DTypeBf16, "https://hologram.uor.foundation/type/dtype/bf16", 16, DTypeKind::Bfloat);
declare_dtype!(DTypeF64,  "https://hologram.uor.foundation/type/dtype/f64",  64, DTypeKind::Float);
declare_dtype!(DTypeI64,  "https://hologram.uor.foundation/type/dtype/i64",  64, DTypeKind::SignedInt);
declare_dtype!(DTypeI32,  "https://hologram.uor.foundation/type/dtype/i32",  32, DTypeKind::SignedInt);
declare_dtype!(DTypeI8,   "https://hologram.uor.foundation/type/dtype/i8",    8, DTypeKind::SignedInt);
declare_dtype!(DTypeU64,  "https://hologram.uor.foundation/type/dtype/u64",  64, DTypeKind::UnsignedInt);
declare_dtype!(DTypeU8,   "https://hologram.uor.foundation/type/dtype/u8",    8, DTypeKind::UnsignedInt);
declare_dtype!(DTypeBool, "https://hologram.uor.foundation/type/dtype/bool",  1, DTypeKind::Bool);
