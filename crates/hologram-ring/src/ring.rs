//! PrismRing: zero-sized marker type implementing UOR Ring traits.

use crate::datum::Datum;
use crate::involution::Involution;
use crate::level::{WittLevelMarker, W128, W16, W32, W64, W8};
use crate::PrismPrimitives;

/// Zero-sized marker type for the ring R_n at Witt level W.
#[derive(Debug, Clone, Copy)]
pub struct PrismRing<W: WittLevelMarker>(core::marker::PhantomData<W>);

impl<W: WittLevelMarker> PrismRing<W> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<W: WittLevelMarker> Default for PrismRing<W> {
    fn default() -> Self {
        Self::new()
    }
}

// ── UOR trait implementations (require std for OnceLock statics) ─────────
//
// Everything below is gated behind `std` feature. The core ring arithmetic
// (RingWord, PrimOp, ActivationOp, accumulate, observables) is unconditionally
// no_std — available on bare metal, WASM, and embedded.

#[cfg(feature = "std")]
#[allow(unused_imports)]
mod uor_impls {
    use super::*;
    use crate::word::RingWord;

    pub struct PrismMultTable<W: WittLevelMarker>(core::marker::PhantomData<W>);

    impl<W: WittLevelMarker> hologram_foundation::division::MultiplicationTable<PrismPrimitives>
        for PrismMultTable<W>
    {
    }

    // ── PrismDivisionAlgebra ─────────────────────────────────────────────────

    /// Unified division algebra enum bridging Witt levels across the CD chain.
    #[derive(Debug, Clone, Copy)]
    pub enum PrismDivisionAlgebra {
        W8(PrismRing<W8>),
        W16(PrismRing<W16>),
        W32(PrismRing<W32>),
        W64(PrismRing<W64>),
    }

    impl hologram_foundation::division::NormedDivisionAlgebra<PrismPrimitives>
        for PrismDivisionAlgebra
    {
        fn algebra_dimension(&self) -> u64 {
            match self {
                Self::W8(_) => 1,
                Self::W16(_) => 2,
                Self::W32(_) => 4,
                Self::W64(_) => 8,
            }
        }
        fn is_commutative(&self) -> bool {
            !matches!(self, Self::W32(_) | Self::W64(_))
        }
        fn is_associative(&self) -> bool {
            !matches!(self, Self::W64(_))
        }
        fn basis_elements(&self) -> &str {
            match self {
                Self::W8(_) => "{1}",
                Self::W16(_) => "{1, i}",
                Self::W32(_) => "{1, i, j, k}",
                Self::W64(_) => "{1, e1, e2, e3, e4, e5, e6, e7}",
            }
        }
        type MultiplicationTable = PrismMultTable<W8>; // marker ZST
        fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
            &MULT_TABLE_W8
        }
    }

    // ── Per-level statics ────────────────────────────────────────────────────

    static MULT_TABLE_W8: PrismMultTable<W8> = PrismMultTable(core::marker::PhantomData);
    static MULT_TABLE_W16: PrismMultTable<W16> = PrismMultTable(core::marker::PhantomData);
    static MULT_TABLE_W32: PrismMultTable<W32> = PrismMultTable(core::marker::PhantomData);
    static MULT_TABLE_W64: PrismMultTable<W64> = PrismMultTable(core::marker::PhantomData);
    static MULT_TABLE_W128: PrismMultTable<W128> = PrismMultTable(core::marker::PhantomData);

    static DA_W8: PrismDivisionAlgebra =
        PrismDivisionAlgebra::W8(PrismRing(core::marker::PhantomData));
    static DA_W16: PrismDivisionAlgebra =
        PrismDivisionAlgebra::W16(PrismRing(core::marker::PhantomData));
    static DA_W32: PrismDivisionAlgebra =
        PrismDivisionAlgebra::W32(PrismRing(core::marker::PhantomData));
    static DA_W64: PrismDivisionAlgebra =
        PrismDivisionAlgebra::W64(PrismRing(core::marker::PhantomData));

    // ── Macro for per-level Ring + Group + NDA + CD impls ────────────────────

    macro_rules! impl_ring_for_level {
        ($W:ty, $word_ty:ty, $gen_val:expr, $bits:expr, $index:expr,
     $witt_level:expr, $modulus:expr, $dim:expr,
     $commutative:expr, $associative:expr, $basis:expr,
     $mult_table:expr,
     $neg_static:ident, $bnot_static:ident, $gen_static:ident, $generators_static:ident) => {
            static $neg_static: Involution<$W> = Involution::Neg;
            static $bnot_static: Involution<$W> = Involution::Bnot;
            static $generators_static: [Involution<$W>; 2] = [Involution::Neg, Involution::Bnot];
            // Generator datum needs runtime construction; use a function instead.
            // We use a OnceLock for lazy initialization.

            impl hologram_foundation::schema::Ring<PrismPrimitives> for PrismRing<$W> {
                /// v0.2.0 renamed `ring_quantum()` to `ring_witt_length()`.
                fn ring_witt_length(&self) -> u64 {
                    $bits
                }
                fn modulus(&self) -> u64 {
                    $modulus
                }
                type Datum = Datum<$W>;
                fn generator(&self) -> &Self::Datum {
                    use std::sync::OnceLock;
                    static $gen_static: OnceLock<Datum<$W>> = OnceLock::new();
                    $gen_static.get_or_init(|| Datum::<$W>::new(<$word_ty>::ONE))
                }
                type Involution = Involution<$W>;
                fn negation(&self) -> &Self::Involution {
                    &$neg_static
                }
                fn complement(&self) -> &Self::Involution {
                    &$bnot_static
                }
                /// v0.2.0 renamed `at_quantum_level()` to `at_witt_level()`
                /// and the return type to `WittLevel`.
                fn at_witt_level(&self) -> hologram_foundation::WittLevel {
                    $witt_level
                }
            }

            impl hologram_foundation::op::Group<PrismPrimitives> for PrismRing<$W> {
                type Operation = Involution<$W>;
                fn generated_by(&self) -> &[Self::Operation] {
                    &$generators_static
                }
                fn order(&self) -> u64 {
                    if $bits >= 64 {
                        u64::MAX
                    } else {
                        1u64 << $bits
                    }
                }
            }

            impl hologram_foundation::op::DihedralGroup<PrismPrimitives> for PrismRing<$W> {}

            impl hologram_foundation::division::NormedDivisionAlgebra<PrismPrimitives>
                for PrismRing<$W>
            {
                fn algebra_dimension(&self) -> u64 {
                    $dim
                }
                fn is_commutative(&self) -> bool {
                    $commutative
                }
                fn is_associative(&self) -> bool {
                    $associative
                }
                fn basis_elements(&self) -> &str {
                    $basis
                }
                type MultiplicationTable = PrismMultTable<$W>;
                fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
                    &$mult_table
                }
            }

            // v0.2.0 keeps AlgebraCommutator/AlgebraAssociator as marker
            // traits — no methods to implement.
            impl hologram_foundation::division::AlgebraCommutator<PrismPrimitives>
                for PrismRing<$W>
            {
            }

            impl hologram_foundation::division::AlgebraAssociator<PrismPrimitives>
                for PrismRing<$W>
            {
            }
        };
    }

    impl_ring_for_level!(
        W8,
        u8,
        1u8,
        8,
        0,
        hologram_foundation::WittLevel::W8,
        256,
        1,
        true,
        true,
        "{1}",
        MULT_TABLE_W8,
        NEG_W8,
        BNOT_W8,
        GEN_W8,
        GENS_W8
    );

    impl_ring_for_level!(
        W16,
        u16,
        1u16,
        16,
        1,
        hologram_foundation::WittLevel::W16,
        65536,
        2,
        true,
        true,
        "{1, i}",
        MULT_TABLE_W16,
        NEG_W16,
        BNOT_W16,
        GEN_W16,
        GENS_W16
    );

    impl_ring_for_level!(
        W32,
        u32,
        1u32,
        32,
        3,
        hologram_foundation::WittLevel::W32,
        4_294_967_296,
        4,
        false,
        true,
        "{1, i, j, k}",
        MULT_TABLE_W32,
        NEG_W32,
        BNOT_W32,
        GEN_W32,
        GENS_W32
    );

    impl_ring_for_level!(
        W64,
        u64,
        1u64,
        64,
        7,
        hologram_foundation::WittLevel::new(64),
        0,
        8,
        false,
        false,
        "{1, e1, e2, e3, e4, e5, e6, e7}",
        MULT_TABLE_W64,
        NEG_W64,
        BNOT_W64,
        GEN_W64,
        GENS_W64
    );

    impl_ring_for_level!(
        W128,
        u128,
        1u128,
        128,
        15,
        hologram_foundation::WittLevel::new(128),
        0,
        1,
        true,
        true,
        "{1}",
        MULT_TABLE_W128,
        NEG_W128,
        BNOT_W128,
        GEN_W128,
        GENS_W128
    );

    // ── CayleyDicksonConstruction (per level pair) ───────────────────────────

    impl hologram_foundation::division::CayleyDicksonConstruction<PrismPrimitives> for PrismRing<W8> {
        type NormedDivisionAlgebra = PrismDivisionAlgebra;
        fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
            &DA_W8
        }
        fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
            &DA_W16
        }
        fn adjoined_element(&self) -> &str {
            "i"
        }
        fn conjugation_rule(&self) -> &str {
            "i\u{00b2} = \u{2212}1 mod 256"
        }
    }

    impl hologram_foundation::division::CayleyDicksonConstruction<PrismPrimitives> for PrismRing<W16> {
        type NormedDivisionAlgebra = PrismDivisionAlgebra;
        fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
            &DA_W16
        }
        fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
            &DA_W32
        }
        fn adjoined_element(&self) -> &str {
            "j"
        }
        fn conjugation_rule(&self) -> &str {
            "j\u{00b2} = \u{2212}1 mod 65536"
        }
    }

    impl hologram_foundation::division::CayleyDicksonConstruction<PrismPrimitives> for PrismRing<W32> {
        type NormedDivisionAlgebra = PrismDivisionAlgebra;
        fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
            &DA_W32
        }
        fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
            &DA_W64
        }
        fn adjoined_element(&self) -> &str {
            "l"
        }
        fn conjugation_rule(&self) -> &str {
            "l\u{00b2} = \u{2212}1 mod 2^32"
        }
    }
} // mod uor_impls

#[cfg(feature = "std")]
pub use uor_impls::*;
