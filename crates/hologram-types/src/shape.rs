//! Hologram-specific shape markers (spec IV.5).
//!
//! Hologram contributes only `Dim<N>` (a single static dimension) and
//! `Shape1` / `Shape2` (rank-1 / rank-2 wrappers) — the small shapes
//! the graph IR's per-node dtype-and-shape resolver consumes. Higher
//! ranks and matrix/vector shape carriers come from prism-tensor's
//! `MatrixShape<R, C, E>` and `VectorShape<N, E>` per wiki ADR-031
//! (re-exported from this crate's `lib.rs`).

use core::marker::PhantomData;
use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef, AFFINE_MAX_COEFFS};

/// A static dimension carrying a known size N.
///
/// Affine constraint asserts `1·site_0 = N`. Per ADR-032 the
/// dimension admits N distinct index residues.
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
    const CONSTRAINTS: &'static [ConstraintRef] = &[ConstraintRef::Affine {
        coefficients: dim_coefficients(),
        coefficient_count: 1,
        bias: N as i64,
    }];
    const CYCLE_SIZE: u64 = N;
}

/// Rank-1 / rank-2 shape markers. `SITES` is the aggregate site count,
/// supplied by the caller per spec IV.3 (stable const generics).
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
            const CYCLE_SIZE: u64 = 1;
        }
    };
}

declare_shape!(
    Shape1,
    "https://hologram.uor.foundation/type/shape/rank1",
    [D0]
);
declare_shape!(
    Shape2,
    "https://hologram.uor.foundation/type/shape/rank2",
    [D0, D1]
);
