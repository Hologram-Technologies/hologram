//! Hologram substitution-axis impls per spec Part III.
//!
//! Provides the three-axis selection (HostTypes / HostBounds / Hasher) for
//! the canonical hologram backends. `ActiveCpuBounds` resolves at compile
//! time to the strongest CPU bounds available on the build target.

#![no_std]

mod types;
mod bounds;
mod hasher;

pub use types::HologramHostTypes;
pub use bounds::{
    HologramHostBoundsCpu,
    HologramHostBoundsAvx2,
    HologramHostBoundsAvx512,
    HologramHostBoundsNeon,
    HologramHostBoundsMetal,
    HologramHostBoundsWgpu,
};
pub use hasher::HologramHasher;

/// Active CPU bounds for this build. Resolves at compile time:
/// AVX-512 ▸ AVX2 ▸ NEON ▸ scalar.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type ActiveCpuBounds = HologramHostBoundsAvx512;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
pub type ActiveCpuBounds = HologramHostBoundsAvx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type ActiveCpuBounds = HologramHostBoundsNeon;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
pub type ActiveCpuBounds = HologramHostBoundsCpu;
