//! Spec XII.3: every HostBounds impl satisfies WITT_LEVEL_MAX_BITS = expected register width.

use uor_foundation::HostBounds;
use hologram_host::*;

#[test]
fn cpu_scalar_is_w64() {
    assert_eq!(HologramHostBoundsCpu::WITT_LEVEL_MAX_BITS, 64);
}

#[test]
fn avx2_is_w256() {
    assert_eq!(HologramHostBoundsAvx2::WITT_LEVEL_MAX_BITS, 256);
}

#[test]
fn avx512_is_w512() {
    assert_eq!(HologramHostBoundsAvx512::WITT_LEVEL_MAX_BITS, 512);
}

#[test]
fn neon_is_w128() {
    assert_eq!(HologramHostBoundsNeon::WITT_LEVEL_MAX_BITS, 128);
}

#[test]
fn metal_is_w64() {
    assert_eq!(HologramHostBoundsMetal::WITT_LEVEL_MAX_BITS, 64);
}

#[test]
fn wgpu_is_w64() {
    assert_eq!(HologramHostBoundsWgpu::WITT_LEVEL_MAX_BITS, 64);
}

#[test]
fn fingerprint_is_32_bytes_everywhere() {
    assert_eq!(HologramHostBoundsCpu::FINGERPRINT_MAX_BYTES, 32);
    assert_eq!(HologramHostBoundsAvx2::FINGERPRINT_MAX_BYTES, 32);
    assert_eq!(HologramHostBoundsAvx512::FINGERPRINT_MAX_BYTES, 32);
    assert_eq!(HologramHostBoundsNeon::FINGERPRINT_MAX_BYTES, 32);
    assert_eq!(HologramHostBoundsMetal::FINGERPRINT_MAX_BYTES, 32);
    assert_eq!(HologramHostBoundsWgpu::FINGERPRINT_MAX_BYTES, 32);
}

#[test]
fn trace_capacity_supports_uhd() {
    const _: () = assert!(HologramHostBoundsCpu::TRACE_MAX_EVENTS >= 16_384);
}
