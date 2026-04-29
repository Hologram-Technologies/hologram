//! Core LUT tables, views, ring algebra, and encoding for O(1) computation.
//!
//! This crate provides the mathematical foundation: precomputed lookup tables,
//! the `ElementWiseView` function composition system, byte-domain ring operations,
//! and the pi-F-lambda encoding pipeline.
//!
//! # Zero Dependencies
//!
//! `hologram-core` depends only on `uor-foundation` (no_std, traits-only).
//! All tables are compile-time constants in `.rodata`.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod buffer;
pub mod carry;
pub mod datum;
pub mod element;
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

/// Hologram host-types family for uor-foundation traits.
///
/// Per ADR-052, the new 0.3.0 `HostTypes` trait replaced the previous
/// `Primitives` trait. Hologram adopts the `DefaultHostTypes` shape:
/// `Decimal = f64`, `HostString = str`, `WitnessBytes = [u8]`. The
/// removed `Integer`/`NonNegativeInteger`/`PositiveInteger`/`Boolean`
/// slots had no `HostTypes` analogue — call sites that relied on them
/// now consume concrete `i64`/`u64`/`bool` directly.
pub struct HoloPrimitives;

impl uor_foundation::HostTypes for HoloPrimitives {
    type Decimal = f64;
    type HostString = str;
    type WitnessBytes = [u8];
}
