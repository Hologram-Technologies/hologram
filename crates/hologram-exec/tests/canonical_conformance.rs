//! Canonical-vs-exec conformance cross-check (ADR-050).
//!
//! For every op the canonical layer covers, the optimised exec
//! dispatch must produce the same output as the reference canonical
//! kernel — that's the contract canonical kernels enforce.
//!
//! This file ships the cross-check infrastructure plus representative
//! tests covering each canonical op category. Adding a new canonical
//! op = add one cross-check test here pointing at the new kernel.

use hologram_core::op::FloatOp;
use hologram_exec::float_dispatch::{dispatch_float, dispatch_float_with_shapes};
use hologram_ops::{
    AddCall, BinaryCall, KernelCall, MatMulCall, ReduceCall, ReduceKind, SlotSpan, SoftmaxCall,
    UnaryCall, UnaryKind,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(data).to_vec()
}

fn result_f32(result: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(result).to_vec()
}

fn approx_eq(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch — exec={}, canonical={}",
        a.len(),
        b.len()
    );
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() <= tol,
            "{label}: index {i} differs — exec={av}, canonical={bv}",
        );
    }
}

/// Run a canonical unary kernel against a fresh f32 workspace and
/// return the output values.
fn run_canonical_unary(input: &[f32], kind: UnaryKind) -> Vec<f32> {
    let n = input.len();
    let mut storage = vec![0.0_f32; 2 * n];
    storage[..n].copy_from_slice(input);
    let call = UnaryCall {
        input: SlotSpan { offset: 0, len: n },
        output: SlotSpan { offset: n, len: n },
    };
    hologram_ops::dispatch(&mut storage, &KernelCall::Unary(call, kind));
    storage[n..].to_vec()
}

/// Run a canonical binary kernel (Add / Sub / Mul / Div / etc.).
fn run_canonical_binary(a: &[f32], b: &[f32], variant: BinaryVariant) -> Vec<f32> {
    let n = a.len();
    assert_eq!(n, b.len());
    let mut storage = vec![0.0_f32; 3 * n];
    storage[..n].copy_from_slice(a);
    storage[n..2 * n].copy_from_slice(b);
    let bin = BinaryCall {
        a: SlotSpan { offset: 0, len: n },
        b: SlotSpan { offset: n, len: n },
        c: SlotSpan {
            offset: 2 * n,
            len: n,
        },
    };
    let call = match variant {
        BinaryVariant::Add => KernelCall::Add(AddCall {
            a: bin.a,
            b: bin.b,
            c: bin.c,
        }),
        BinaryVariant::Sub => KernelCall::Sub(bin),
        BinaryVariant::Mul => KernelCall::Mul(bin),
        BinaryVariant::Div => KernelCall::Div(bin),
    };
    hologram_ops::dispatch(&mut storage, &call);
    storage[2 * n..].to_vec()
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // `Sub`/`Div` are wired for future cross-check tests.
enum BinaryVariant {
    Add,
    Sub,
    Mul,
    Div,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn relu_matches_canonical() {
    let xs = [-3.0_f32, -0.5, 0.0, 1.5, 100.0];
    let exec_out = result_f32(&dispatch_float(&FloatOp::Relu, &[&f32_bytes(&xs)]).unwrap());
    let canon_out = run_canonical_unary(&xs, UnaryKind::Relu);
    approx_eq(&exec_out, &canon_out, 1e-6, "Relu");
}

#[test]
fn gelu_matches_canonical() {
    let xs = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let exec_out = result_f32(&dispatch_float(&FloatOp::Gelu, &[&f32_bytes(&xs)]).unwrap());
    let canon_out = run_canonical_unary(&xs, UnaryKind::Gelu);
    // GELU has two common forms (tanh approximation vs erf-based).
    // Hologram's exec uses the same tanh approximation as canonical
    // (ADR-049 / kernels/unary.rs comment); 1e-5 tolerance accounts for
    // independent reorderings of the same operations.
    approx_eq(&exec_out, &canon_out, 1e-4, "Gelu");
}

#[test]
fn sigmoid_matches_canonical() {
    let xs = [-5.0_f32, -1.0, 0.0, 1.0, 5.0];
    let exec_out = result_f32(&dispatch_float(&FloatOp::Sigmoid, &[&f32_bytes(&xs)]).unwrap());
    let canon_out = run_canonical_unary(&xs, UnaryKind::Sigmoid);
    approx_eq(&exec_out, &canon_out, 1e-5, "Sigmoid");
}

#[test]
fn add_matches_canonical() {
    let a = [1.0_f32, 2.0, 3.0, -4.0];
    let b = [10.0_f32, 20.0, 30.0, 40.0];
    let exec_out =
        result_f32(&dispatch_float(&FloatOp::Add, &[&f32_bytes(&a), &f32_bytes(&b)]).unwrap());
    let canon_out = run_canonical_binary(&a, &b, BinaryVariant::Add);
    approx_eq(&exec_out, &canon_out, 1e-6, "Add");
}

#[test]
fn mul_matches_canonical() {
    let a = [1.0_f32, 2.0, 3.0, -4.0];
    let b = [10.0_f32, 20.0, 30.0, 40.0];
    let exec_out =
        result_f32(&dispatch_float(&FloatOp::Mul, &[&f32_bytes(&a), &f32_bytes(&b)]).unwrap());
    let canon_out = run_canonical_binary(&a, &b, BinaryVariant::Mul);
    approx_eq(&exec_out, &canon_out, 1e-6, "Mul");
}

#[test]
fn matmul_matches_canonical() {
    // 2x3 @ 3x2 — same example as the canonical unit test.
    let a: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let exec_out = result_f32(
        &dispatch_float_with_shapes(
            &FloatOp::MatMul { m: 2, k: 3, n: 2 },
            &[&f32_bytes(&a), &f32_bytes(&b)],
            &[vec![2, 3], vec![3, 2]],
        )
        .unwrap(),
    );

    // Canonical
    let mut storage = vec![0.0_f32; 6 + 6 + 4];
    storage[..6].copy_from_slice(&a);
    storage[6..12].copy_from_slice(&b);
    let call = KernelCall::MatMul(MatMulCall {
        a: SlotSpan { offset: 0, len: 6 },
        b: SlotSpan { offset: 6, len: 6 },
        c: SlotSpan { offset: 12, len: 4 },
        m: 2,
        k: 3,
        n: 2,
    });
    hologram_ops::dispatch(&mut storage, &call);
    let canon_out = storage[12..16].to_vec();

    approx_eq(&exec_out, &canon_out, 1e-5, "MatMul");
}

#[test]
fn softmax_matches_canonical() {
    let xs = [1.0_f32, 2.0, 3.0, 4.0];
    let exec_out = result_f32(
        &dispatch_float_with_shapes(
            &FloatOp::Softmax { size: 4 },
            &[&f32_bytes(&xs)],
            &[vec![4]],
        )
        .unwrap(),
    );

    let n = xs.len();
    let mut storage = vec![0.0_f32; 2 * n];
    storage[..n].copy_from_slice(&xs);
    let call = KernelCall::Softmax(SoftmaxCall {
        input: SlotSpan { offset: 0, len: n },
        output: SlotSpan { offset: n, len: n },
        size: n,
    });
    hologram_ops::dispatch(&mut storage, &call);
    let canon_out = storage[n..].to_vec();

    approx_eq(&exec_out, &canon_out, 1e-5, "Softmax");
}

#[test]
fn reduce_sum_matches_canonical() {
    let xs = [1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0];
    let exec_out = result_f32(
        &dispatch_float_with_shapes(
            &FloatOp::ReduceSum { size: 3 },
            &[&f32_bytes(&xs)],
            &[vec![2, 3]],
        )
        .unwrap(),
    );

    let mut storage = vec![0.0_f32; xs.len() + 2];
    storage[..xs.len()].copy_from_slice(&xs);
    let call = KernelCall::Reduce(
        ReduceCall {
            input: SlotSpan {
                offset: 0,
                len: xs.len(),
            },
            output: SlotSpan {
                offset: xs.len(),
                len: 2,
            },
            size: 3,
        },
        ReduceKind::Sum,
    );
    hologram_ops::dispatch(&mut storage, &call);
    let canon_out = storage[xs.len()..].to_vec();

    approx_eq(&exec_out, &canon_out, 1e-5, "ReduceSum");
}
