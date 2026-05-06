//! ActivationFn impl smoke tests.

use hologram_ops::activations::{Relu, Sigmoid, Tanh, Gelu, Silu};
use hologram_ops::lut::{ActivationFn, build_w8_lut_runtime};

#[test]
fn relu_is_identity_unsigned() {
    for x in [0u8, 1, 127, 255] {
        assert_eq!(Relu::eval_w8(x), x);
    }
}

#[test]
fn sigmoid_is_monotonic_at_w8() {
    let mut prev = Sigmoid::eval_w8(0);
    for x in 1u8..=255 {
        let cur = Sigmoid::eval_w8(x);
        assert!(cur >= prev, "sigmoid not monotonic at {x}: {prev} -> {cur}");
        prev = cur;
    }
}

#[test]
fn tanh_at_origin_is_half() {
    let mid = Tanh::eval_w8(128);
    assert!(mid.abs_diff(128) <= 4);
}

#[test]
fn gelu_silu_compile() {
    let _ = Gelu::eval_f32(0.5);
    let _ = Silu::eval_f32(0.5);
}

#[test]
fn lut_construction() {
    let lut = build_w8_lut_runtime::<Relu>();
    for (i, &v) in lut.iter().enumerate() {
        assert_eq!(v, i as u8);
    }
}
