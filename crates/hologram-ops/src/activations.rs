//! `ActivationFn` impls for the LUT specialization (spec V.6).
//!
//! Reference scalar evaluation contract for activation markers, used both
//! to build precomputed LUTs and to verify backend kernel equivalence.

use crate::lut::ActivationFn;

#[inline]
fn s_to_unit_u8(x: u8) -> f32 { x as f32 / 255.0 }
#[inline]
fn unit_to_u8(x: f32) -> u8 {
    let c = x.clamp(0.0, 1.0);
    (c * 255.0 + 0.5) as u8
}
#[inline]
fn s_to_unit_u16(x: u16) -> f32 { x as f32 / 65535.0 }
#[inline]
fn unit_to_u16(x: f32) -> u16 {
    let c = x.clamp(0.0, 1.0);
    (c * 65535.0 + 0.5) as u16
}

/// Marker types matched against the elementwise_unary marker family.
/// Each carries the scalar reference implementation of its activation.
#[derive(Debug, Default, Clone, Copy)]
pub struct Relu;
impl ActivationFn for Relu {
    fn eval_w8(x: u8) -> u8 { x }
    fn eval_w16(x: u16) -> u16 { x }
    fn eval_f32(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Sigmoid;
impl ActivationFn for Sigmoid {
    fn eval_w8(x: u8) -> u8 {
        let f = (s_to_unit_u8(x) - 0.5) * 8.0;
        unit_to_u8(1.0 / (1.0 + libm::expf(-f)))
    }
    fn eval_w16(x: u16) -> u16 {
        let f = (s_to_unit_u16(x) - 0.5) * 8.0;
        unit_to_u16(1.0 / (1.0 + libm::expf(-f)))
    }
    fn eval_f32(x: f32) -> f32 {
        1.0 / (1.0 + libm::expf(-x))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Tanh;
impl ActivationFn for Tanh {
    fn eval_w8(x: u8) -> u8 {
        let f = (s_to_unit_u8(x) - 0.5) * 4.0;
        unit_to_u8((libm::tanhf(f) + 1.0) / 2.0)
    }
    fn eval_w16(x: u16) -> u16 {
        let f = (s_to_unit_u16(x) - 0.5) * 4.0;
        unit_to_u16((libm::tanhf(f) + 1.0) / 2.0)
    }
    fn eval_f32(x: f32) -> f32 { libm::tanhf(x) }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Gelu;
impl ActivationFn for Gelu {
    fn eval_w8(x: u8) -> u8 {
        let f = (s_to_unit_u8(x) - 0.5) * 8.0;
        unit_to_u8((Self::eval_f32(f) + 1.0) * 0.0625)
    }
    fn eval_w16(x: u16) -> u16 {
        let f = (s_to_unit_u16(x) - 0.5) * 8.0;
        unit_to_u16((Self::eval_f32(f) + 1.0) * 0.0625)
    }
    fn eval_f32(x: f32) -> f32 {
        // Approximate GELU: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))
        let c = 0.797_884_6_f32; // sqrt(2/pi)
        let x3 = x * x * x;
        0.5 * x * (1.0 + libm::tanhf(c * (x + 0.044_715 * x3)))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Silu;
impl ActivationFn for Silu {
    fn eval_w8(x: u8) -> u8 {
        let f = (s_to_unit_u8(x) - 0.5) * 8.0;
        unit_to_u8((Self::eval_f32(f) + 4.0) * 0.125)
    }
    fn eval_w16(x: u16) -> u16 {
        let f = (s_to_unit_u16(x) - 0.5) * 8.0;
        unit_to_u16((Self::eval_f32(f) + 4.0) * 0.125)
    }
    fn eval_f32(x: f32) -> f32 {
        x / (1.0 + libm::expf(-x))
    }
}
