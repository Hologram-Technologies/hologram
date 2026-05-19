//! Op types: PrimOp (10 primitives), LutOp (21+ activation functions),
//! and FloatOp (typed f32 tensor operations for AI inference).

mod float_op;
mod float_op_methods;
mod lut_op;
mod prim;
pub mod shape_projection;
mod shape_spec;

pub use float_op::{
    bits_to_f32, f32_to_bits, FloatDType, FloatOp, FloatOpShape, TensorMeta, RUNTIME,
};
pub use lut_op::LutOp;
pub use prim::PrimOp;
pub use shape_spec::{ShapeDim, ShapeSpec};

// Ring-native activation ops from hologram-ring (parametric ring foundation).
pub use hologram_ring::activation::ActivationOp;

// Canonical QuantumLevel comes from uor-foundation v0.1.4.
pub use uor_foundation::QuantumLevel;

/// Ring quantum level for ring-arithmetic execution.
///
/// Selects which ring Z/2^nZ to operate in:
/// - Q0: Z/256Z (8-bit)
/// - Q1: Z/65536Z (16-bit)
/// - Q2: Z/2^24Z (24-bit)
/// - Q3: Z/2^32Z (32-bit)
///
/// This enum is retained for rkyv serialization compatibility in `GraphOp`.
/// New code should use `QuantumLevel` directly via `QuantumLevelExt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[repr(u8)]
pub enum RingLevel {
    Q0 = 0,
    Q1 = 1,
    Q2 = 2,
    Q3 = 3,
}

impl RingLevel {
    /// Convert from `uor_foundation::QuantumLevel` to `RingLevel`.
    /// Returns `None` for quantum levels beyond Q3.
    #[inline]
    pub const fn from_quantum(q: uor_foundation::QuantumLevel) -> Option<Self> {
        match q.index() {
            0 => Some(Self::Q0),
            1 => Some(Self::Q1),
            2 => Some(Self::Q2),
            3 => Some(Self::Q3),
            _ => None,
        }
    }

    /// Convert to `uor_foundation::QuantumLevel`.
    #[inline]
    pub const fn to_quantum(self) -> uor_foundation::QuantumLevel {
        match self {
            Self::Q0 => uor_foundation::QuantumLevel::Q0,
            Self::Q1 => uor_foundation::QuantumLevel::Q1,
            Self::Q2 => uor_foundation::QuantumLevel::Q2,
            Self::Q3 => uor_foundation::QuantumLevel::Q3,
        }
    }

    /// Byte width for this ring level.
    #[inline]
    pub const fn byte_width(self) -> u8 {
        (self as u8) + 1
    }
}

impl From<uor_foundation::QuantumLevel> for RingLevel {
    #[inline]
    fn from(q: uor_foundation::QuantumLevel) -> Self {
        Self::from_quantum(q).unwrap_or(Self::Q3)
    }
}

impl From<RingLevel> for uor_foundation::QuantumLevel {
    #[inline]
    fn from(r: RingLevel) -> Self {
        r.to_quantum()
    }
}

/// Extension trait for `QuantumLevel` providing byte-width dispatch.
pub trait QuantumLevelExt {
    /// Byte width for this quantum level: `index + 1`.
    /// Q0 = 1, Q1 = 2, Q2 = 3, Q3 = 4, Q7 = 8.
    fn byte_width(self) -> u8;
}

impl QuantumLevelExt for uor_foundation::QuantumLevel {
    #[inline(always)]
    fn byte_width(self) -> u8 {
        (self.index() + 1) as u8
    }
}

// The legacy unified `Op` sum-type enum was removed (no production
// callers; dead since `GraphOp` and `SemanticOp` superseded it). The
// canonical op identity now lives as `hologram_ops::Op` (a trait); the
// per-domain enums (`PrimOp`, `LutOp`, `FloatOp`) remain as the
// byte/float dispatch shapes, which is what consumers actually use.
