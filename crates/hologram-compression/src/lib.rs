//! Lossless compression via UOR ring algebra observables.
//!
//! Uses algebraic properties of Z/256Z (stratum, torus, ring differential)
//! to decompose data and factor out structural redundancy, approaching the
//! irreducible information content bounded by the Landauer limit.
//!
//! # Compression Modes
//!
//! - **Generic** (Mode 0): Ring-differential coding + ANS entropy backend
//! - **Stratum** (Mode 1): Stratum-partitioned entropy coding (SPEC)
//! - **Float** (Mode 2): IEEE 754 byte-plane transpose + per-plane SPEC/RDC
//! - **Quantized** (Mode 3): Orbit-torus blocked coding

#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod codec;
pub mod entropy;
pub mod float_plane;
pub mod header;
pub mod permute;
pub mod pipeline;
pub mod ring_diff;
pub mod stratum;
pub mod torus_block;

pub use codec::{CompressedBlock, CompressionMode, CompressionStats};
pub use pipeline::{compress, decompress};
