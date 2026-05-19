//! Hologram substitution-axis selections per spec Part III.
//!
//! Hologram is a Prism application (wiki ADR-031): it imports the
//! canonical substitution axes through the prism façade rather than
//! reimplementing them. Two of the three axes are upstream-canonical:
//!
//! - `HologramHostTypes` is a type alias for [`uor_foundation::DefaultHostTypes`].
//! - `HologramHasher` is a type alias for [`prism::crypto::Blake3Hasher`] —
//!   the prism-crypto standard-library Layer-3 `HashAxis` impl that
//!   simultaneously satisfies the `Hasher<32>` content-addressing trait
//!   (wiki ADR-031, ADR-055).
//!
//! Only `HostBounds` is hologram-specific: each backend pins
//! `WITT_LEVEL_MAX_BITS` to its natural register width and the remaining
//! ADR-037 capacity bounds are sized for trillion-param + UHD streaming.

#![no_std]

// Anchor the Prism standard-library façade and SDK so the dep tree
// records them at hologram-host (the substitution-axis layer). Per
// wiki ADR-031, hologram is a Prism application; downstream hologram
// crates reach axis declarations, verb declarations, and SDK macros
// through these re-exports.
pub use prism;
pub use uor_foundation_sdk as sdk;

mod bounds;

pub use bounds::{
    HologramHostBoundsAvx2, HologramHostBoundsAvx512, HologramHostBoundsCpu,
    HologramHostBoundsMetal, HologramHostBoundsNeon, HologramHostBoundsWgpu,
};

/// Hologram's `HostTypes` selection. Aliased to upstream's canonical
/// `DefaultHostTypes` — hologram has no need for non-default Decimal /
/// HostString / WitnessBytes representations (spec III.1).
pub type HologramHostTypes = uor_foundation::DefaultHostTypes;

/// Hologram's canonical `Hasher<32>` selection. Aliased to
/// `prism::crypto::Blake3Hasher` — the prism-crypto Layer-3 BLAKE3
/// `HashAxis` impl (wiki ADR-031). Reaches `Hasher<32>` via the
/// upstream `Hasher` trait impl on `Blake3Hasher`, and `HashAxis` via
/// the same type — so a hologram-built `AxisTuple` admits this hasher
/// at the canonical first axis position.
pub type HologramHasher = prism::crypto::Blake3Hasher;

/// Active CPU bounds for this build. Resolves at compile time:
/// AVX-512 ▸ AVX2 ▸ NEON ▸ scalar.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type ActiveCpuBounds = HologramHostBoundsAvx512;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub type ActiveCpuBounds = HologramHostBoundsAvx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type ActiveCpuBounds = HologramHostBoundsNeon;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
pub type ActiveCpuBounds = HologramHostBoundsCpu;
