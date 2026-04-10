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

/// Re-export of v0.2.0's `WittLevel` struct (from `hologram_foundation`).
///
/// In v0.1.4 this was an enum named `QuantumLevel` with variants `Q0..Q3`
/// and an `index()` method returning a small integer. In v0.2.0 it is a
/// struct with `witt_length()` returning the *bit width* (8, 16, 24, 32).
/// Constants `WittLevel::W8`, `W16`, `W24`, `W32` are provided as
/// associated constants on the struct, plus `WittLevel::new(n)` for
/// arbitrary widths.
pub use hologram_foundation::WittLevel;

/// Wire-format representation of the four spec-named [`WittLevel`]s
/// (W8 / W16 / W24 / W32) for use inside rkyv-serialized archive data.
///
/// `WittLevel` itself is a struct over `u32` and does not derive rkyv;
/// embedding it directly in archive node payloads would require pulling
/// rkyv into `uor-foundation`. `RingLevel` is the small enum that lives
/// in `GraphOp::RingPrim*` variants and `TermKind::QuantumLit` so those
/// archive sections stay rkyv-serializable. Convert via
/// [`Self::to_witt_level`] / [`Self::from_witt_level`] at the boundary.
///
/// **Variant→bit-width mapping:** Q0=8, Q1=16, Q2=24, Q3=32.
///
/// Per the v0.2.0 conformance-first contract there is no implicit
/// `From<WittLevel> for RingLevel` — that conversion is fallible (it
/// rejects widths outside the four spec-named levels) and must be made
/// at every call site so the rejection branch is visible.
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
    /// Convert from a v0.2.0 [`WittLevel`] to a `RingLevel`.
    /// Returns `None` for Witt levels with bit widths outside the four
    /// spec-named levels (8/16/24/32). Callers must handle the `None`
    /// branch explicitly — there is no silent fallback.
    #[inline]
    pub const fn from_witt_level(w: WittLevel) -> Option<Self> {
        match w.witt_length() {
            8 => Some(Self::Q0),
            16 => Some(Self::Q1),
            24 => Some(Self::Q2),
            32 => Some(Self::Q3),
            _ => None,
        }
    }

    /// Convert to a v0.2.0 [`WittLevel`] (constants are 8/16/24/32 bits).
    #[inline]
    pub const fn to_witt_level(self) -> WittLevel {
        match self {
            Self::Q0 => WittLevel::W8,
            Self::Q1 => WittLevel::W16,
            Self::Q2 => WittLevel::W24,
            Self::Q3 => WittLevel::W32,
        }
    }

    /// Byte width for this ring level.
    #[inline]
    pub const fn byte_width(self) -> u8 {
        (self as u8) + 1
    }

    /// Bit width for this ring level. Equivalent to the corresponding
    /// `WittLevel::witt_length()`.
    #[inline]
    pub const fn bit_width(self) -> u8 {
        self.byte_width() * 8
    }
}

impl From<RingLevel> for WittLevel {
    #[inline]
    fn from(r: RingLevel) -> Self {
        r.to_witt_level()
    }
}

/// Extension trait for [`WittLevel`] providing byte-width dispatch.
///
/// Replaces the v0.1.4 `WittLevelExt` trait. Methods are renamed to
/// match v0.2.0 vocabulary; semantics are unchanged.
pub trait WittLevelExt {
    /// Byte width for this Witt level: `witt_length / 8`.
    /// W8 = 1, W16 = 2, W24 = 3, W32 = 4, W64 = 8.
    fn byte_width(self) -> u8;
}

impl WittLevelExt for WittLevel {
    #[inline(always)]
    fn byte_width(self) -> u8 {
        (self.witt_length() / 8) as u8
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

    #[test]
    fn ring_level_witt_roundtrip() {
        for r in [RingLevel::Q0, RingLevel::Q1, RingLevel::Q2, RingLevel::Q3] {
            let w: WittLevel = r.into();
            assert_eq!(RingLevel::from_witt_level(w), Some(r));
        }
    }

    #[test]
    fn witt_level_byte_width() {
        assert_eq!(WittLevel::W8.byte_width(), 1);
        assert_eq!(WittLevel::W16.byte_width(), 2);
        assert_eq!(WittLevel::W24.byte_width(), 3);
        assert_eq!(WittLevel::W32.byte_width(), 4);
    }

    #[test]
    fn ring_level_byte_width() {
        assert_eq!(RingLevel::Q0.byte_width(), 1);
        assert_eq!(RingLevel::Q3.byte_width(), 4);
    }
}
