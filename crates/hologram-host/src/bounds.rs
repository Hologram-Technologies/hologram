//! Per-backend `HostBounds` impls (spec III.2).
//!
//! `WITT_LEVEL_MAX_BITS` is pinned to each host's natural register width:
//! the largest single-instruction algebraic operation the host issues.
//! The remaining ADR-037 capacity bounds are shared across backends and
//! sized for hologram's trillion-param + UHD-streaming target workloads.

use uor_foundation::HostBounds;

/// Emit a `HostBounds` impl for a per-backend marker that varies only in
/// `WITT_LEVEL_MAX_BITS`. All other ADR-037 capacities are shared and
/// sized for hologram's target workloads (trillion-param model loading,
/// per-frame UHD streaming, deep transformer decode stacks).
macro_rules! hologram_host_bounds {
    ($name:ident, $witt_bits:expr) => {
        impl HostBounds for $name {
            // Fingerprint width is BLAKE3-canonical 32 bytes (ADR-001, ADR-052).
            const FINGERPRINT_MIN_BYTES: usize = 32;
            const FINGERPRINT_MAX_BYTES: usize = 32;

            // Trace capacity is sized for UHD per-frame streaming workloads.
            const TRACE_MAX_EVENTS: usize = 16_384;

            // Per-backend register-width-driven Witt-level ceiling.
            const WITT_LEVEL_MAX_BITS: u32 = $witt_bits;

            // ADR-037 capacities for hologram-class workloads.
            const FOLD_UNROLL_THRESHOLD: usize = 8;
            const BETTI_DIMENSION_MAX: usize = 16;
            const NERVE_CONSTRAINTS_MAX: usize = 16;
            const NERVE_SITES_MAX: usize = 16;
            const JACOBIAN_SITES_MAX: usize = 16;
            const RECURSION_TRACE_DEPTH_MAX: usize = 32;
            const OP_CHAIN_DEPTH_MAX: usize = 16;
            const AFFINE_COEFFS_MAX: usize = 8;
            const CONJUNCTION_TERMS_MAX: usize = 8;
            const UNFOLD_ITERATIONS_MAX: usize = 4_096;
        }
    };
}

/// CPU scalar (x86-64 / AArch64 GPR). One u64 per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsCpu;
hologram_host_bounds!(HologramHostBoundsCpu, 64);

/// AVX2 (256-bit YMM). `Limbs<4>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsAvx2;
hologram_host_bounds!(HologramHostBoundsAvx2, 256);

/// AVX-512 (512-bit ZMM). `Limbs<8>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsAvx512;
hologram_host_bounds!(HologramHostBoundsAvx512, 512);

/// ARM NEON (128-bit Q-register). `Limbs<2>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsNeon;
hologram_host_bounds!(HologramHostBoundsNeon, 128);

/// Apple Metal (per-lane scalar). Cross-lane reductions are pipeline structure,
/// not single algebraic ops.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsMetal;
hologram_host_bounds!(HologramHostBoundsMetal, 64);

/// WebGPU (WGSL per-lane scalar). Same per-lane reasoning as Metal.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsWgpu;
hologram_host_bounds!(HologramHostBoundsWgpu, 64);
