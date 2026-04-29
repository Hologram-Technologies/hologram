//! PrismRing: zero-sized marker type implementing UOR Ring traits.

use crate::datum::Datum;
use crate::involution::Involution;
use crate::level::{QuantumLevel, Q0, Q1, Q15, Q3, Q7};
use crate::PrismPrimitives;

/// Zero-sized marker type for the ring R_n at quantum level Q.
#[derive(Debug, Clone, Copy)]
pub struct PrismRing<Q: QuantumLevel>(core::marker::PhantomData<Q>);

impl<Q: QuantumLevel> PrismRing<Q> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<Q: QuantumLevel> Default for PrismRing<Q> {
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

    pub struct PrismMultTable<Q: QuantumLevel>(core::marker::PhantomData<Q>);

    impl<Q: QuantumLevel> uor_foundation::kernel::division::MultiplicationTable<PrismPrimitives>
        for PrismMultTable<Q>
    {
    }

    // ── PrismDivisionAlgebra ─────────────────────────────────────────────────

    /// Unified division algebra enum bridging quantum levels across the CD chain.
    #[derive(Debug, Clone, Copy)]
    pub enum PrismDivisionAlgebra {
        Q0(PrismRing<Q0>),
        Q1(PrismRing<Q1>),
        Q3(PrismRing<Q3>),
        Q7(PrismRing<Q7>),
    }

    impl uor_foundation::kernel::division::NormedDivisionAlgebra<PrismPrimitives>
        for PrismDivisionAlgebra
    {
        fn algebra_dimension(&self) -> u64 {
            match self {
                Self::Q0(_) => 1,
                Self::Q1(_) => 2,
                Self::Q3(_) => 4,
                Self::Q7(_) => 8,
            }
        }
        fn is_commutative(&self) -> bool {
            !matches!(self, Self::Q3(_) | Self::Q7(_))
        }
        fn is_associative(&self) -> bool {
            !matches!(self, Self::Q7(_))
        }
        fn basis_elements(&self) -> &str {
            match self {
                Self::Q0(_) => "{1}",
                Self::Q1(_) => "{1, i}",
                Self::Q3(_) => "{1, i, j, k}",
                Self::Q7(_) => "{1, e1, e2, e3, e4, e5, e6, e7}",
            }
        }
        type MultiplicationTable = PrismMultTable<Q0>; // marker ZST
        fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
            &MULT_TABLE_Q0
        }
    }

    // ── Per-level statics ────────────────────────────────────────────────────

    static MULT_TABLE_Q0: PrismMultTable<Q0> = PrismMultTable(core::marker::PhantomData);
    static MULT_TABLE_Q1: PrismMultTable<Q1> = PrismMultTable(core::marker::PhantomData);
    static MULT_TABLE_Q3: PrismMultTable<Q3> = PrismMultTable(core::marker::PhantomData);
    static MULT_TABLE_Q7: PrismMultTable<Q7> = PrismMultTable(core::marker::PhantomData);
    static MULT_TABLE_Q15: PrismMultTable<Q15> = PrismMultTable(core::marker::PhantomData);

    static DA_Q0: PrismDivisionAlgebra =
        PrismDivisionAlgebra::Q0(PrismRing(core::marker::PhantomData));
    static DA_Q1: PrismDivisionAlgebra =
        PrismDivisionAlgebra::Q1(PrismRing(core::marker::PhantomData));
    static DA_Q3: PrismDivisionAlgebra =
        PrismDivisionAlgebra::Q3(PrismRing(core::marker::PhantomData));
    static DA_Q7: PrismDivisionAlgebra =
        PrismDivisionAlgebra::Q7(PrismRing(core::marker::PhantomData));

    // ── Macro for per-level Ring + Group + NDA + CD impls ────────────────────

    macro_rules! impl_ring_for_level {
        ($Q:ty, $word_ty:ty, $gen_val:expr, $bits:expr, $index:expr,
     $uor_level:expr, $modulus:expr, $dim:expr,
     $commutative:expr, $associative:expr, $basis:expr,
     $mult_table:expr,
     $neg_static:ident, $bnot_static:ident, $gen_static:ident, $generators_static:ident) => {
            static $neg_static: Involution<$Q> = Involution::Neg;
            static $bnot_static: Involution<$Q> = Involution::Bnot;
            static $generators_static: [Involution<$Q>; 2] = [Involution::Neg, Involution::Bnot];
            // Generator datum needs runtime construction; use a function instead
            // We use a thread-local or lazy approach — but for simplicity, just
            // return a reference that lives long enough via Box::leak in a once_cell.

            impl uor_foundation::kernel::schema::Ring<PrismPrimitives> for PrismRing<$Q> {
                fn ring_witt_length(&self) -> u64 {
                    $bits
                }
                fn modulus(&self) -> u64 {
                    $modulus
                }
                type Datum = Datum<$Q>;
                fn generator(&self) -> &Self::Datum {
                    // Leak a static Datum. Called at most once per level.
                    use std::sync::OnceLock;
                    static $gen_static: OnceLock<Datum<$Q>> = OnceLock::new();
                    $gen_static.get_or_init(|| Datum::<$Q>::new(<$word_ty>::ONE))
                }
                type Involution = Involution<$Q>;
                fn negation(&self) -> &Self::Involution {
                    &$neg_static
                }
                fn complement(&self) -> &Self::Involution {
                    &$bnot_static
                }
                fn at_witt_level(&self) -> uor_foundation::WittLevel {
                    $uor_level
                }
            }

            impl uor_foundation::kernel::op::Group<PrismPrimitives> for PrismRing<$Q> {
                type Operation = Involution<$Q>;
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

            impl uor_foundation::kernel::op::DihedralGroup<PrismPrimitives> for PrismRing<$Q> {}

            impl uor_foundation::kernel::division::NormedDivisionAlgebra<PrismPrimitives>
                for PrismRing<$Q>
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
                type MultiplicationTable = PrismMultTable<$Q>;
                fn algebra_multiplication_table(&self) -> &Self::MultiplicationTable {
                    &$mult_table
                }
            }

            // uor-foundation 0.1.4 made AlgebraCommutator/AlgebraAssociator marker
            // traits — no methods to implement.
            impl uor_foundation::kernel::division::AlgebraCommutator<PrismPrimitives>
                for PrismRing<$Q>
            {
            }

            impl uor_foundation::kernel::division::AlgebraAssociator<PrismPrimitives>
                for PrismRing<$Q>
            {
            }
        };
    }

    impl_ring_for_level!(
        Q0,
        u8,
        1u8,
        8,
        0,
        uor_foundation::WittLevel::W8,
        256,
        1,
        true,
        true,
        "{1}",
        MULT_TABLE_Q0,
        NEG_Q0,
        BNOT_Q0,
        GEN_Q0,
        GENS_Q0
    );

    impl_ring_for_level!(
        Q1,
        u16,
        1u16,
        16,
        1,
        uor_foundation::WittLevel::W16,
        65536,
        2,
        true,
        true,
        "{1, i}",
        MULT_TABLE_Q1,
        NEG_Q1,
        BNOT_Q1,
        GEN_Q1,
        GENS_Q1
    );

    impl_ring_for_level!(
        Q3,
        u32,
        1u32,
        32,
        3,
        uor_foundation::WittLevel::W24,
        4_294_967_296,
        4,
        false,
        true,
        "{1, i, j, k}",
        MULT_TABLE_Q3,
        NEG_Q3,
        BNOT_Q3,
        GEN_Q3,
        GENS_Q3
    );

    impl_ring_for_level!(
        Q7,
        u64,
        1u64,
        64,
        7,
        uor_foundation::WittLevel::W32,
        0,
        8,
        false,
        false,
        "{1, e1, e2, e3, e4, e5, e6, e7}",
        MULT_TABLE_Q7,
        NEG_Q7,
        BNOT_Q7,
        GEN_Q7,
        GENS_Q7
    );

    impl_ring_for_level!(
        Q15,
        u128,
        1u128,
        128,
        15,
        uor_foundation::WittLevel::new(128),
        0,
        1,
        true,
        true,
        "{1}",
        MULT_TABLE_Q15,
        NEG_Q15,
        BNOT_Q15,
        GEN_Q15,
        GENS_Q15
    );

    // ── CayleyDicksonConstruction (per level pair) ───────────────────────────

    impl uor_foundation::kernel::division::CayleyDicksonConstruction<PrismPrimitives>
        for PrismRing<Q0>
    {
        type NormedDivisionAlgebra = PrismDivisionAlgebra;
        fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
            &DA_Q0
        }
        fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
            &DA_Q1
        }
        fn adjoined_element(&self) -> &str {
            "i"
        }
        fn conjugation_rule(&self) -> &str {
            "i\u{00b2} = \u{2212}1 mod 256"
        }
    }

    impl uor_foundation::kernel::division::CayleyDicksonConstruction<PrismPrimitives>
        for PrismRing<Q1>
    {
        type NormedDivisionAlgebra = PrismDivisionAlgebra;
        fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
            &DA_Q1
        }
        fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
            &DA_Q3
        }
        fn adjoined_element(&self) -> &str {
            "j"
        }
        fn conjugation_rule(&self) -> &str {
            "j\u{00b2} = \u{2212}1 mod 65536"
        }
    }

    impl uor_foundation::kernel::division::CayleyDicksonConstruction<PrismPrimitives>
        for PrismRing<Q3>
    {
        type NormedDivisionAlgebra = PrismDivisionAlgebra;
        fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra {
            &DA_Q3
        }
        fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra {
            &DA_Q7
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
