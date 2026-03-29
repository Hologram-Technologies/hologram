//! Involution: the two generators of the dihedral group D_{2^n}.

use crate::level::QuantumLevel;
use crate::word::RingWord;
use crate::PrismPrimitives;
use uor_foundation::enums::GeometricCharacter;

/// The two involutions of the ring R_n.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Involution<Q: QuantumLevel> {
    /// Ring reflection: neg(x) = (-x) mod 2^n.
    Neg,
    /// Hypercube reflection: bnot(x) = bitwise NOT.
    Bnot,
    /// PhantomData carrier (never constructed).
    #[doc(hidden)]
    _Phantom(core::marker::PhantomData<Q>),
}

impl<Q: QuantumLevel> Involution<Q> {
    /// Apply this involution. Compiles to a single ALU instruction.
    #[inline]
    #[must_use]
    pub fn apply(self, x: Q::Word) -> Q::Word {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
            Self::_Phantom(_) => unreachable!(),
        }
    }
}

// ── UOR Operation trait (per concrete level via macro) ───────────────────

macro_rules! impl_involution_uor {
    ($Q:ty, $neg_static:ident, $bnot_static:ident) => {
        static $neg_static: Involution<$Q> = Involution::Neg;
        static $bnot_static: Involution<$Q> = Involution::Bnot;

        impl uor_foundation::kernel::op::Operation<PrismPrimitives> for Involution<$Q> {
            fn arity(&self) -> u64 {
                1
            }
            fn has_geometric_character(&self) -> GeometricCharacter {
                match self {
                    Self::Neg => GeometricCharacter::RingReflection,
                    Self::Bnot => GeometricCharacter::HypercubeReflection,
                    Self::_Phantom(_) => unreachable!(),
                }
            }
            type OperationTarget = Involution<$Q>;
            fn inverse(&self) -> &Self::OperationTarget {
                match self {
                    Self::Neg => &$neg_static,
                    Self::Bnot => &$bnot_static,
                    Self::_Phantom(_) => unreachable!(),
                }
            }
            fn composed_of(&self) -> &str {
                match self {
                    Self::Neg => "neg",
                    Self::Bnot => "bnot",
                    Self::_Phantom(_) => unreachable!(),
                }
            }
        }

        impl uor_foundation::kernel::op::UnaryOp<PrismPrimitives> for Involution<$Q> {}
        impl uor_foundation::kernel::op::Involution<PrismPrimitives> for Involution<$Q> {}
    };
}

use crate::level::{Q0, Q1, Q15, Q3, Q7};

impl_involution_uor!(Q0, INV_NEG_Q0, INV_BNOT_Q0);
impl_involution_uor!(Q1, INV_NEG_Q1, INV_BNOT_Q1);
impl_involution_uor!(Q3, INV_NEG_Q3, INV_BNOT_Q3);
impl_involution_uor!(Q7, INV_NEG_Q7, INV_BNOT_Q7);
impl_involution_uor!(Q15, INV_NEG_Q15, INV_BNOT_Q15);
