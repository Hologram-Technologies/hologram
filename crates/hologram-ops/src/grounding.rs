//! Boundary `Grounding` impls (spec VII.7).
//!
//! Per invariant I-10: `Grounding` lives only at the input boundary —
//! parsing host bytes into typed values. Operations between hologram types
//! are Term trees, never Grounding programs.

use core::marker::PhantomData;
use uor_foundation::enforcement::{
    combinators, BinaryGroundingMap, GroundedCoord, Grounding, GroundingProgram,
};

/// Parses one element of a model weight from raw bytes into a `GroundedCoord`.
/// `D` selects the dtype's Witt level; `B` is the active host bounds (informational).
pub struct WeightLoaderGrounding<D, B>(PhantomData<(D, B)>);

impl<D, B> Default for WeightLoaderGrounding<D, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D, B> Grounding for WeightLoaderGrounding<D, B>
where
    D: 'static,
    B: 'static,
{
    type Output = GroundedCoord;
    type Map = BinaryGroundingMap;

    fn program(&self) -> GroundingProgram<Self::Output, Self::Map> {
        GroundingProgram::from_primitive(combinators::read_bytes::<GroundedCoord>())
    }
}

/// Parses an inline graph constant from bytes. Same shape as `WeightLoaderGrounding`,
/// distinct marker so resolvers can route differently.
pub struct ConstantGrounding<D, B>(PhantomData<(D, B)>);

impl<D, B> Default for ConstantGrounding<D, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D, B> Grounding for ConstantGrounding<D, B>
where
    D: 'static,
    B: 'static,
{
    type Output = GroundedCoord;
    type Map = BinaryGroundingMap;

    fn program(&self) -> GroundingProgram<Self::Output, Self::Map> {
        GroundingProgram::from_primitive(combinators::read_bytes::<GroundedCoord>())
    }
}
