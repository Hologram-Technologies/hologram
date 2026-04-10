//! prism-ring: Parametric ring R_n for the Prism algebraic grounding.
//!
//! Every operation is a ring primitive or a composition of ring primitives.
//! The ring is parametric over Witt level W_n (bit width n) where R_n =
//! Z/(2^n)Z.
//!
//! All types are zero-cost: ZST level markers, const fn ring arithmetic,
//! monomorphized generics, no dynamic dispatch.
//!
//! # v0.2.0 reframing
//!
//! In v0.1.4 the levels were named after their quantum-level *index*
//! (`Q0`, `Q1`, `Q3`, `Q7`, `Q15`). v0.2.0 renames them after their *bit
//! width* (`W8`, `W16`, `W32`, `W64`, `W128`) to align with
//! `hologram_foundation::WittLevel`'s naming convention. The local trait is
//! `WittLevelMarker` to disambiguate from the foundation's runtime
//! `WittLevel` struct. **Perf: NEUTRAL** — pure rename, no behavioral
//! change.

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
pub use level::{WittLevelMarker, W128, W16, W32, W64, W8};
pub use observables::{curvature, domain, rank, stratum};
pub use prim::PrimOp;
pub use word::RingWord;

/// The UOR primitive type family for Prism.
pub struct PrismPrimitives;

impl hologram_foundation::Primitives for PrismPrimitives {
    type String = str;
    type Integer = i64;
    type NonNegativeInteger = u64;
    type PositiveInteger = u64;
    type Decimal = f64;
    type Boolean = bool;
}
