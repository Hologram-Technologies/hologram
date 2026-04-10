//! Core LUT tables, views, ring algebra, and encoding for O(1) computation.
//!
//! This crate provides the mathematical foundation: precomputed lookup tables,
//! the `ElementWiseView` function composition system, byte-domain ring operations,
//! and the pi-F-lambda encoding pipeline.
//!
//! # Foundation dependency
//!
//! `hologram-core` depends only on `hologram-foundation` (the v0.2.0
//! re-export shim, no_std, traits-only). All tables are compile-time
//! constants in `.rodata`.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod buffer;
pub mod carry;
pub mod datum;
pub mod encoding;
pub mod error;
pub mod lut;
pub mod op;
pub mod q1;
pub mod q2;
pub mod q3;
pub mod quantum;
pub mod ring;
pub mod term;
pub mod view;

/// Hologram primitive type family for v0.2.0 foundation traits.
///
/// Maps XSD primitives to Rust types suitable for O(1) LUT computation.
pub struct HoloPrimitives;

impl hologram_foundation::Primitives for HoloPrimitives {
    type String = str;
    type Integer = i64;
    type NonNegativeInteger = u64;
    type PositiveInteger = u64;
    type Decimal = f64;
    type Boolean = bool;
}
