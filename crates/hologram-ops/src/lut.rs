//! Compile-time LUTs for activation specialization (spec V.6).
//!
//! Activation kernels at low Witt levels (W8, W16) admit precomputed
//! lookup tables. The LUT is a backend optimization — it does not replace
//! the Term tree (which remains the formal spec).

extern crate alloc;
use alloc::boxed::Box;

/// Reference scalar evaluation contract for an activation marker.
/// Used both for LUT generation and for kernel equivalence testing.
pub trait ActivationFn {
    fn eval_w8(x: u8) -> u8;
    fn eval_w16(x: u16) -> u16;
    fn eval_f32(x: f32) -> f32;
}

/// Build a 256-entry W8 LUT for the activation `F`.
/// Backends call this at session-load time to materialize the precomputed
/// table; the LUT is then consulted instead of evaluating the activation
/// per element. The Term tree remains the formal spec; the LUT is an
/// equivalence-proven optimization.
pub fn build_w8_lut<F: ActivationFn>() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = F::eval_w8(i as u8);
        i += 1;
    }
    t
}

/// Build a 65,536-entry W16 LUT for the activation `F`. Larger than the
/// W8 form (128 KiB); use only when the activation is on a hot path.
pub fn build_w16_lut<F: ActivationFn>() -> Box<[u16; 65_536]> {
    let mut t = Box::new([0u16; 65_536]);
    let mut i: usize = 0;
    while i < 65_536 {
        t[i] = F::eval_w16(i as u16);
        i += 1;
    }
    t
}

