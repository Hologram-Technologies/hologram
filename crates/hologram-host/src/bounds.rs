//! Per-backend `HostBounds` impls (spec III.2).
//!
//! `WITT_LEVEL_MAX_BITS` is pinned to each host's natural register width:
//! the largest single-instruction algebraic operation the host issues.

use uor_foundation::HostBounds;

const FP: usize = 32;
const TR_MAX: usize = 16_384;

/// CPU scalar (x86-64 / AArch64 GPR). One u64 per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsCpu;

impl HostBounds for HologramHostBoundsCpu {
    const FINGERPRINT_MIN_BYTES: usize = FP;
    const FINGERPRINT_MAX_BYTES: usize = FP;
    const TRACE_MAX_EVENTS: usize = TR_MAX;
    const WITT_LEVEL_MAX_BITS: u32 = 64;
}

/// AVX2 (256-bit YMM). `Limbs<4>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsAvx2;

impl HostBounds for HologramHostBoundsAvx2 {
    const FINGERPRINT_MIN_BYTES: usize = FP;
    const FINGERPRINT_MAX_BYTES: usize = FP;
    const TRACE_MAX_EVENTS: usize = TR_MAX;
    const WITT_LEVEL_MAX_BITS: u32 = 256;
}

/// AVX-512 (512-bit ZMM). `Limbs<8>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsAvx512;

impl HostBounds for HologramHostBoundsAvx512 {
    const FINGERPRINT_MIN_BYTES: usize = FP;
    const FINGERPRINT_MAX_BYTES: usize = FP;
    const TRACE_MAX_EVENTS: usize = TR_MAX;
    const WITT_LEVEL_MAX_BITS: u32 = 512;
}

/// ARM NEON (128-bit Q-register). `Limbs<2>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsNeon;

impl HostBounds for HologramHostBoundsNeon {
    const FINGERPRINT_MIN_BYTES: usize = FP;
    const FINGERPRINT_MAX_BYTES: usize = FP;
    const TRACE_MAX_EVENTS: usize = TR_MAX;
    const WITT_LEVEL_MAX_BITS: u32 = 128;
}

/// Apple Metal (per-lane scalar). Cross-lane reductions are pipeline structure,
/// not single algebraic ops.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsMetal;

impl HostBounds for HologramHostBoundsMetal {
    const FINGERPRINT_MIN_BYTES: usize = FP;
    const FINGERPRINT_MAX_BYTES: usize = FP;
    const TRACE_MAX_EVENTS: usize = TR_MAX;
    const WITT_LEVEL_MAX_BITS: u32 = 64;
}

/// WebGPU (WGSL per-lane scalar). Same per-lane reasoning as Metal.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsWgpu;

impl HostBounds for HologramHostBoundsWgpu {
    const FINGERPRINT_MIN_BYTES: usize = FP;
    const FINGERPRINT_MAX_BYTES: usize = FP;
    const TRACE_MAX_EVENTS: usize = TR_MAX;
    const WITT_LEVEL_MAX_BITS: u32 = 64;
}
