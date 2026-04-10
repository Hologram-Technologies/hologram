//! Involution: the two generators of the dihedral group D_{2^n}.

use crate::level::WittLevelMarker;
use crate::word::RingWord;
use crate::PrismPrimitives;
use hologram_foundation::enums::GeometricCharacter;

/// The two involutions of the ring R_n.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Involution<W: WittLevelMarker> {
    /// Ring reflection: neg(x) = (-x) mod 2^n.
    Neg,
    /// Hypercube reflection: bnot(x) = bitwise NOT.
    Bnot,
    /// PhantomData carrier (never constructed).
    #[doc(hidden)]
    _Phantom(core::marker::PhantomData<W>),
}

impl<W: WittLevelMarker> Involution<W> {
    /// Apply this involution. Compiles to a single ALU instruction.
    #[inline]
    #[must_use]
    pub fn apply(self, x: W::Word) -> W::Word {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
            Self::_Phantom(_) => unreachable!(),
        }
    }
}

// ── UOR Operation trait (per concrete level via macro) ───────────────────

macro_rules! impl_involution_uor {
    ($W:ty, $neg_static:ident, $bnot_static:ident) => {
        static $neg_static: Involution<$W> = Involution::Neg;
        static $bnot_static: Involution<$W> = Involution::Bnot;

        impl hologram_foundation::op::Operation<PrismPrimitives> for Involution<$W> {
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
            type OperationTarget = Involution<$W>;
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

        impl hologram_foundation::op::UnaryOp<PrismPrimitives> for Involution<$W> {}
        impl hologram_foundation::op::Involution<PrismPrimitives> for Involution<$W> {}
    };
}

use crate::level::{W128, W16, W32, W64, W8};

impl_involution_uor!(W8, INV_NEG_W8, INV_BNOT_W8);
impl_involution_uor!(W16, INV_NEG_W16, INV_BNOT_W16);
impl_involution_uor!(W32, INV_NEG_W32, INV_BNOT_W32);
impl_involution_uor!(W64, INV_NEG_W64, INV_BNOT_W64);
impl_involution_uor!(W128, INV_NEG_W128, INV_BNOT_W128);
