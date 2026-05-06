//! Shape declarations (spec IV.5).

use core::marker::PhantomData;
use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef, AFFINE_MAX_COEFFS};

/// A static dimension carrying a known size N.
///
/// Affine constraint asserts `1·site_0 = N`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim<const N: u64>;

const fn dim_coefficients() -> [i64; AFFINE_MAX_COEFFS] {
    let mut c = [0i64; AFFINE_MAX_COEFFS];
    c[0] = 1;
    c
}

impl<const N: u64> ConstrainedTypeShape for Dim<N> {
    const IRI: &'static str = "https://hologram.uor.foundation/type/shape/dim";
    const SITE_COUNT: usize = 1;
    const CONSTRAINTS: &'static [ConstraintRef] = &[
        ConstraintRef::Affine {
            coefficients: dim_coefficients(),
            coefficient_count: 1,
            bias: N as i64,
        },
    ];
}

/// A symbolic dimension. No `Affine` pinning; resolved at graph-build time.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DimSymbolic<const ID: u64>;

impl<const ID: u64> ConstrainedTypeShape for DimSymbolic<ID> {
    const IRI: &'static str = "https://hologram.uor.foundation/type/shape/dim_symbolic";
    const SITE_COUNT: usize = 1;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}

/// Rank-N shape markers. `SITES` is the aggregate site count, supplied by
/// the caller per spec IV.3 (stable const generics).
macro_rules! declare_shape {
    ($name:ident, $iri:literal, [$($d:ident),+]) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name<$($d,)+ const SITES: usize>(PhantomData<($($d,)+)>);

        impl<$($d,)+ const SITES: usize> Default for $name<$($d,)+ SITES> {
            fn default() -> Self { Self(PhantomData) }
        }

        impl<$($d,)+ const SITES: usize> ConstrainedTypeShape for $name<$($d,)+ SITES>
        where
            $($d: ConstrainedTypeShape,)+
        {
            const IRI: &'static str = $iri;
            const SITE_COUNT: usize = SITES;
            const CONSTRAINTS: &'static [ConstraintRef] = &[];
        }
    };
}

declare_shape!(Shape1, "https://hologram.uor.foundation/type/shape/rank1", [D0]);
declare_shape!(Shape2, "https://hologram.uor.foundation/type/shape/rank2", [D0, D1]);
declare_shape!(Shape3, "https://hologram.uor.foundation/type/shape/rank3", [D0, D1, D2]);
declare_shape!(Shape4, "https://hologram.uor.foundation/type/shape/rank4", [D0, D1, D2, D3]);
declare_shape!(Shape5, "https://hologram.uor.foundation/type/shape/rank5", [D0, D1, D2, D3, D4]);
declare_shape!(Shape6, "https://hologram.uor.foundation/type/shape/rank6", [D0, D1, D2, D3, D4, D5]);
declare_shape!(Shape7, "https://hologram.uor.foundation/type/shape/rank7", [D0, D1, D2, D3, D4, D5, D6]);
declare_shape!(Shape8, "https://hologram.uor.foundation/type/shape/rank8", [D0, D1, D2, D3, D4, D5, D6, D7]);

/// Heap variant for rank > 8. `SITES` is the aggregate; `RANK` is the rank.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeArray<const RANK: usize, const SITES: usize>;

impl<const RANK: usize, const SITES: usize> ConstrainedTypeShape for ShapeArray<RANK, SITES> {
    const IRI: &'static str = "https://hologram.uor.foundation/type/shape/array";
    const SITE_COUNT: usize = SITES;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
