//! Kernel ↔ Term-tree reference-evaluator equivalence (spec VII.3).
//!
//! For each direct-PrimitiveOp wrapper, we build the canonical Term tree
//! (via the same emit pattern hologram-ops uses), evaluate it against
//! a scalar reference evaluator, and compare to the CPU kernel result.
//! This is the equivalence check that anchors interpretation A: the Term
//! tree is the formal spec; the kernel is the execution form; tests
//! verify they agree.

use hologram_backend::{
    CpuBackend, Backend, KernelCall, BufferRef,
    UnaryCall, BinaryCall, Workspace,
};
use hologram_ops::{ScalarEvaluatorU64, ReferenceEvaluator};
use uor_foundation::PrimitiveOp;
use uor_foundation::enforcement::{TermArena, Term, TermList};

struct Ws { slots: Vec<Vec<u8>> }
impl Workspace for Ws {
    fn read(&self, b: BufferRef) -> &[u8] { &self.slots[b.slot as usize] }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let i = b.slot as usize;
        let len = self.slots[i].len();
        &mut self.slots[i][..len]
    }
}
fn buf(slot: u32, len: u32) -> BufferRef { BufferRef { slot, offset: 0, length: len } }

fn eval_unary_term(prim: PrimitiveOp, x: u64) -> u64 {
    let mut arena: TermArena<8> = TermArena::new();
    let v = arena.push(Term::Variable { name_index: 0 }).unwrap();
    let root = arena.push(Term::Application {
        operator: prim, args: TermList { start: v, len: 1 },
    }).unwrap();
    ScalarEvaluatorU64::evaluate(&arena, root, &[x]).unwrap()
}

fn eval_binary_term(prim: PrimitiveOp, a: u64, b: u64) -> u64 {
    let mut arena: TermArena<8> = TermArena::new();
    let av = arena.push(Term::Variable { name_index: 0 }).unwrap();
    arena.push(Term::Variable { name_index: 1 }).unwrap();
    let root = arena.push(Term::Application {
        operator: prim, args: TermList { start: av, len: 2 },
    }).unwrap();
    ScalarEvaluatorU64::evaluate(&arena, root, &[a, b]).unwrap()
}

fn run_unary_kernel(make: impl Fn(UnaryCall) -> KernelCall, inp: &[u8]) -> Vec<u8> {
    let mut ws = Ws { slots: vec![inp.to_vec(), vec![0u8; inp.len()]] };
    let call = make(UnaryCall {
        input: buf(0, inp.len() as u32),
        output: buf(1, inp.len() as u32),
        element_count: inp.len() as u32,
        witt_bits: 8, dtype: 1,
    });
    let mut backend: CpuBackend<Ws> = CpuBackend::new();
    backend.dispatch(&call, &mut ws).unwrap();
    ws.slots[1].clone()
}

fn run_binary_kernel(
    make: impl Fn(BinaryCall) -> KernelCall, a: &[u8], b: &[u8],
) -> Vec<u8> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut ws = Ws { slots: vec![a.to_vec(), b.to_vec(), vec![0u8; n]] };
    let call = make(BinaryCall {
        a: buf(0, n as u32), b: buf(1, n as u32), output: buf(2, n as u32),
        element_count: n as u32, witt_bits: 8, dtype: 1,
    });
    let mut backend: CpuBackend<Ws> = CpuBackend::new();
    backend.dispatch(&call, &mut ws).unwrap();
    ws.slots[2].clone()
}

#[test]
fn neg_kernel_matches_term() {
    for x in 0u8..=255 {
        let term_v = eval_unary_term(PrimitiveOp::Neg, x as u64) as u8;
        let kernel = run_unary_kernel(KernelCall::Neg, &[x]);
        assert_eq!(kernel[0], term_v, "neg({x})");
    }
}

#[test]
fn bnot_kernel_matches_term() {
    for x in 0u8..=255 {
        let term_v = eval_unary_term(PrimitiveOp::Bnot, x as u64) as u8;
        let kernel = run_unary_kernel(KernelCall::Bnot, &[x]);
        assert_eq!(kernel[0], term_v, "bnot({x})");
    }
}

#[test]
fn succ_kernel_matches_term() {
    for x in 0u8..=255 {
        let term_v = eval_unary_term(PrimitiveOp::Succ, x as u64) as u8;
        let kernel = run_unary_kernel(KernelCall::Succ, &[x]);
        assert_eq!(kernel[0], term_v, "succ({x})");
    }
}

#[test]
fn pred_kernel_matches_term() {
    for x in 0u8..=255 {
        let term_v = eval_unary_term(PrimitiveOp::Pred, x as u64) as u8;
        let kernel = run_unary_kernel(KernelCall::Pred, &[x]);
        assert_eq!(kernel[0], term_v, "pred({x})");
    }
}

#[test]
fn add_kernel_matches_term() {
    let cases = [(0u8, 0u8), (1, 2), (50, 100), (255, 1), (200, 200), (123, 213)];
    for &(a, b) in &cases {
        let term_v = eval_binary_term(PrimitiveOp::Add, a as u64, b as u64) as u8;
        let kernel = run_binary_kernel(KernelCall::Add, &[a], &[b]);
        assert_eq!(kernel[0], term_v, "add({a}, {b})");
    }
}

#[test]
fn sub_kernel_matches_term() {
    let cases = [(10u8, 3u8), (3, 10), (255, 1), (0, 1), (100, 100)];
    for &(a, b) in &cases {
        let term_v = eval_binary_term(PrimitiveOp::Sub, a as u64, b as u64) as u8;
        let kernel = run_binary_kernel(KernelCall::Sub, &[a], &[b]);
        assert_eq!(kernel[0], term_v, "sub({a}, {b})");
    }
}

#[test]
fn mul_kernel_matches_term() {
    let cases = [(0u8, 0u8), (3, 4), (200, 5), (16, 16), (255, 2)];
    for &(a, b) in &cases {
        let term_v = eval_binary_term(PrimitiveOp::Mul, a as u64, b as u64) as u8;
        let kernel = run_binary_kernel(KernelCall::Mul, &[a], &[b]);
        assert_eq!(kernel[0], term_v, "mul({a}, {b})");
    }
}

#[test]
fn xor_and_or_kernels_match_term() {
    for &(a, b) in &[(0u8, 0u8), (1, 1), (170, 85), (255, 255), (12, 200)] {
        let xor_v = eval_binary_term(PrimitiveOp::Xor, a as u64, b as u64) as u8;
        let and_v = eval_binary_term(PrimitiveOp::And, a as u64, b as u64) as u8;
        let or_v  = eval_binary_term(PrimitiveOp::Or,  a as u64, b as u64) as u8;
        assert_eq!(run_binary_kernel(KernelCall::Xor, &[a], &[b])[0], xor_v);
        assert_eq!(run_binary_kernel(KernelCall::And, &[a], &[b])[0], and_v);
        assert_eq!(run_binary_kernel(KernelCall::Or,  &[a], &[b])[0], or_v);
    }
}

#[test]
fn add_bulk_matches_term_pointwise() {
    // Bulk vector to confirm the SIMD path (where applicable) and the
    // byte-domain scalar path agree per element with the term reference.
    let n = 1024usize;
    let a: Vec<u8> = (0..n).map(|i| (i & 0xFF) as u8).collect();
    let b: Vec<u8> = (0..n).map(|i| ((i * 7) & 0xFF) as u8).collect();
    let kernel = run_binary_kernel(KernelCall::Add, &a, &b);
    for i in 0..n {
        let want = eval_binary_term(PrimitiveOp::Add, a[i] as u64, b[i] as u64) as u8;
        assert_eq!(kernel[i], want);
    }
}
