//! prism-ring: Parametric ring R_n for the Prism algebraic grounding.
//!
//! Every operation is a ring primitive or a composition of ring primitives.
//! The ring is parametric over quantum level Q_k with bit width n = 8*(k+1)
//! and ring R_n = Z/(2^n)Z.
//!
//! All types are zero-cost: ZST level markers, const fn ring arithmetic,
//! monomorphized generics, no dynamic dispatch.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod accumulate;
pub mod activation;
pub mod address;
pub mod datum;
pub mod encoding;
pub mod involution;
pub mod level;
pub mod observables;
pub mod prim;
pub mod ring;
pub mod word;

// Re-exports
pub use accumulate::accumulate;
pub use activation::ActivationOp;
pub use encoding::Encoding;
pub use involution::Involution;
pub use level::{QuantumLevel, Q0, Q1, Q15, Q3, Q7};
pub use observables::{curvature, domain, rank, stratum};
pub use prim::PrimOp;
pub use word::RingWord;

/// The UOR primitive type family for Prism.
pub struct PrismPrimitives;

impl uor_foundation::Primitives for PrismPrimitives {
    type String = str;
    type Integer = i64;
    type NonNegativeInteger = u64;
    type PositiveInteger = u64;
    type Decimal = f64;
    type Boolean = bool;
}
