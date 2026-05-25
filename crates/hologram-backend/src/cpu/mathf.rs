//! `no_std` scalar IEEE-754 helpers for the CPU execution kernels.
//!
//! hologram's execution-form kernels are raw `f32` compute (their formal
//! spec lives in the `hologram-ops` Term trees / prism numeric axes). The
//! inherent `f32::{floor, abs, mul_add, max, min, clamp}` methods live in
//! `std`; on `no_std` targets (wasm / embedded) this `FloatExt` trait fills
//! the same surface through `libm` — exactly as the foundation's
//! `DecimalTranscendental` impls route `ln` / `exp` / `sqrt` through `libm`.
//!
//! The module and its import are compiled only under `not(feature = "std")`.
//! Under `std` the inherent `f32` methods win by Rust's method-resolution
//! priority, so the hot paths keep their hardware rounding / FMA lowering
//! and no call site changes.

/// `f32` methods the CPU kernels need that are otherwise `std`-only.
pub trait FloatExt {
    fn floor(self) -> Self;
    fn abs(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;
}

impl FloatExt for f32 {
    #[inline]
    fn floor(self) -> f32 {
        libm::floorf(self)
    }
    #[inline]
    fn abs(self) -> f32 {
        libm::fabsf(self)
    }
    #[inline]
    fn mul_add(self, a: f32, b: f32) -> f32 {
        libm::fmaf(self, a, b)
    }
    #[inline]
    fn max(self, other: f32) -> f32 {
        libm::fmaxf(self, other)
    }
    #[inline]
    fn min(self, other: f32) -> f32 {
        libm::fminf(self, other)
    }
    #[inline]
    fn clamp(self, lo: f32, hi: f32) -> f32 {
        libm::fminf(libm::fmaxf(self, lo), hi)
    }
}
