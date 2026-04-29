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

// Canonical Witt level comes from uor-foundation v0.3.0. Per Plan 074 §2a
// we re-export under the legacy `QuantumLevel` name to keep hologram-ai
// and other downstream consumers stable. The 0.1.x `index()` method is
// replaced by `witt_length()` returning bit count.
pub use uor_foundation::WittLevel as QuantumLevel;

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
    /// Convert from `uor_foundation::WittLevel` to `RingLevel`.
    /// Returns `None` for Witt levels beyond W32 (RingLevel covers
    /// Q0/Q1/Q2/Q3 = 8/16/24/32 bits).
    #[inline]
    pub const fn from_quantum(q: uor_foundation::WittLevel) -> Option<Self> {
        match q.witt_length() {
            8 => Some(Self::Q0),
            16 => Some(Self::Q1),
            24 => Some(Self::Q2),
            32 => Some(Self::Q3),
            _ => None,
        }
    }

    /// Convert to `uor_foundation::WittLevel`.
    #[inline]
    pub const fn to_quantum(self) -> uor_foundation::WittLevel {
        match self {
            Self::Q0 => uor_foundation::WittLevel::W8,
            Self::Q1 => uor_foundation::WittLevel::W16,
            Self::Q2 => uor_foundation::WittLevel::W24,
            Self::Q3 => uor_foundation::WittLevel::W32,
        }
    }

    /// Byte width for this ring level.
    #[inline]
    pub const fn byte_width(self) -> u8 {
        (self as u8) + 1
    }
}

impl From<uor_foundation::WittLevel> for RingLevel {
    #[inline]
    fn from(q: uor_foundation::WittLevel) -> Self {
        Self::from_quantum(q).unwrap_or(Self::Q3)
    }
}

impl From<RingLevel> for uor_foundation::WittLevel {
    #[inline]
    fn from(r: RingLevel) -> Self {
        r.to_quantum()
    }
}

/// Extension trait for `WittLevel` providing byte-width dispatch.
pub trait QuantumLevelExt {
    /// Byte width: 1 for W8, 2 for W16, 3 for W24, 4 for W32, etc.
    fn byte_width(self) -> u8;
}

impl QuantumLevelExt for uor_foundation::WittLevel {
    #[inline(always)]
    fn byte_width(self) -> u8 {
        (self.witt_length() / 8) as u8
    }
}

// The legacy unified `Op` sum-type enum was removed (no production
// callers; dead since `GraphOp` and `SemanticOp` superseded it). The
// canonical op identity now lives as `hologram_ops::Op` (a trait); the
// per-domain enums (`PrimOp`, `LutOp`, `FloatOp`) remain as the
// byte/float dispatch shapes, which is what consumers actually use.
