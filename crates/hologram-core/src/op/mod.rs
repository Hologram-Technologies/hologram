//! Op types: PrimOp (10 primitives), LutOp (21+ activation functions),
//! and FloatOp (typed f32 tensor operations for AI inference).

mod float_op;
mod lut_op;
mod prim;
pub mod shape_projection;
mod shape_spec;

pub use float_op::{
    bits_to_f32, f32_to_bits, FloatDType, FloatOp, OpCategory, TensorMeta, RUNTIME,
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

/// Unified operation enum for all byte-level operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub enum Op {
    /// One of the 10 UOR primitive operations.
    Prim(PrimOp),
    /// An activation/scientific function via LUT.
    Lut(LutOp),
    /// A typed f32 tensor operation for AI inference.
    Float(FloatOp),
}

impl Op {
    /// Arity of this operation (1 = unary, 2 = binary).
    #[inline]
    #[must_use]
    pub const fn arity(&self) -> u8 {
        match self {
            Self::Prim(p) => p.arity(),
            Self::Lut(_) => 1,
            Self::Float(f) => f.arity(),
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Prim(p) => p.name(),
            Self::Lut(l) => l.name(),
            Self::Float(f) => f.name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_arity() {
        assert_eq!(Op::Prim(PrimOp::Neg).arity(), 1);
        assert_eq!(Op::Prim(PrimOp::Add).arity(), 2);
        assert_eq!(Op::Lut(LutOp::Sigmoid).arity(), 1);
    }

    #[test]
    fn op_name() {
        assert_eq!(Op::Prim(PrimOp::Neg).name(), "neg");
        assert_eq!(Op::Lut(LutOp::Relu).name(), "relu");
    }
}
