//! Verify each op marker's `emit_term` produces a well-formed Term tree
//! whose `Term::Application::operator` is restricted to the closed
//! `PrimitiveOp` set (spec I-1).

use hologram_host::HologramHostBoundsCpu;
use hologram_types::{DTypeF32, Dim, Shape1, Shape2};
use uor_foundation::enforcement::{Term, TermArena};
use uor_foundation::pipeline::ConstrainedTypeShape;
use uor_foundation::{HostBounds, PrimitiveOp, WittLevel};

use hologram_ops::{
    activation_reduce::*, backward::*, conv::*, direct::*, elementwise_binary::*,
    elementwise_unary::*, layout::*, linalg::*, normalization::*, pooling::*, reduction::*,
    structured::*, utility::*,
};

/// Walk an arena and assert every `Term::Application::operator` is one of
/// the closed 10 `PrimitiveOp` discriminants — spec invariant I-1.
fn assert_closed_under_primitives<const CAP: usize>(arena: &TermArena<CAP>) {
    for slot in arena.as_slice().iter().flatten() {
        if let Term::Application { operator, .. } = slot {
            // The PrimitiveOp enum is exhaustively closed; this match is
            // a static check that the operator is one of the 10 variants.
            match *operator {
                // Spec I-1: hologram emits only `PrimitiveOp` discriminants;
                // upstream `uor-foundation 0.4.15` extends the closed set to 18
                // (per ADR-040: byte-level comparators + Euclidean div/mod,
                // modular exponentiation, byte concatenation). The exhaustive
                // arm asserts every `Term::Application` operator hologram emits
                // is one of these — no extra-substrate operator slips in.
                PrimitiveOp::Neg
                | PrimitiveOp::Bnot
                | PrimitiveOp::Succ
                | PrimitiveOp::Pred
                | PrimitiveOp::Add
                | PrimitiveOp::Sub
                | PrimitiveOp::Mul
                | PrimitiveOp::Xor
                | PrimitiveOp::And
                | PrimitiveOp::Or
                | PrimitiveOp::Le
                | PrimitiveOp::Lt
                | PrimitiveOp::Ge
                | PrimitiveOp::Gt
                | PrimitiveOp::Concat
                | PrimitiveOp::Div
                | PrimitiveOp::Mod
                | PrimitiveOp::Pow => {}
            }
        }
    }
}

fn assert_nonempty<const CAP: usize>(arena: &TermArena<CAP>) {
    let any = arena.as_slice().iter().any(|s| s.is_some());
    assert!(any, "emit_term produced an empty arena");
}

#[test]
fn direct_neg_emit_is_well_formed() {
    let mut arena: Box<TermArena<8>> = Box::new(TermArena::new());
    let v0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    NegOp::emit_term(&mut arena, WittLevel::W8, v0).unwrap();
    assert_closed_under_primitives(&arena);
    assert_nonempty(&arena);
}

#[test]
fn direct_add_emit_is_well_formed() {
    let mut arena: Box<TermArena<8>> = Box::new(TermArena::new());
    let v0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    arena.push(Term::Variable { name_index: 1 }).unwrap();
    AddOp::emit_term(&mut arena, WittLevel::W8, v0).unwrap();
    assert_closed_under_primitives(&arena);
    assert_nonempty(&arena);
}

#[test]
fn elementwise_unary_emit_is_well_formed() {
    let mut arena: Box<TermArena<32>> = Box::new(TermArena::new());
    let v0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    SigmoidOp::emit_term(&mut arena, WittLevel::W8, v0).unwrap();
    assert_closed_under_primitives(&arena);
}

#[test]
fn elementwise_binary_emit_is_well_formed() {
    let mut arena: Box<TermArena<32>> = Box::new(TermArena::new());
    let v0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    arena.push(Term::Variable { name_index: 1 }).unwrap();
    DivOp::emit_term(&mut arena, WittLevel::W8, v0).unwrap();
    assert_closed_under_primitives(&arena);
}

#[test]
fn reduction_emit_is_well_formed() {
    type Op = ReduceSumOp<Shape1<Dim<128>, 1>, Shape1<Dim<0>, 1>, DTypeF32, HologramHostBoundsCpu>;
    let mut arena: Box<TermArena<32>> = Box::new(TermArena::new());
    let v0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    Op::emit_term(&mut arena, WittLevel::W8, v0).unwrap();
    assert_closed_under_primitives(&arena);
    let _ = HologramHostBoundsCpu::WITT_LEVEL_MAX_BITS;
    let _ = <Shape1<Dim<128>, 1> as ConstrainedTypeShape>::IRI;
}

#[test]
fn layout_emit_is_single_variable() {
    type Op = ReshapeOp<
        Shape2<Dim<8>, Dim<8>, 2>,
        Shape2<Dim<16>, Dim<4>, 2>,
        DTypeF32,
        HologramHostBoundsCpu,
    >;
    let mut arena: Box<TermArena<8>> = Box::new(TermArena::new());
    Op::emit_term(&mut arena, WittLevel::W8, 0).unwrap();
    // Layout ops emit only Variable nodes (no Application).
    for slot in arena.as_slice().iter().flatten() {
        match slot {
            Term::Variable { .. } => {}
            other => panic!("layout op emitted non-Variable: {:?}", other),
        }
    }
}

#[test]
fn matmul_emit_uses_only_primitive_ops() {
    type Op = MatMulOp<32, 32, 32, DTypeF32, HologramHostBoundsCpu>;
    let mut arena: Box<TermArena<64>> = Box::new(TermArena::new());
    let av = arena.push(Term::Variable { name_index: 0 }).unwrap();
    let bv = arena.push(Term::Variable { name_index: 1 }).unwrap();
    Op::emit_term(&mut arena, WittLevel::W8, av, bv).unwrap();
    assert_closed_under_primitives(&arena);
}

#[test]
fn conv2d_emit_is_well_formed() {
    type S = Shape2<Dim<8>, Dim<8>, 2>;
    type Op = Conv2dOp<S, S, S, S, DTypeF32, HologramHostBoundsCpu>;
    let mut arena: Box<TermArena<128>> = Box::new(TermArena::new());
    let xv = arena.push(Term::Variable { name_index: 0 }).unwrap();
    Op::emit_term(&mut arena, WittLevel::W8, xv, xv).unwrap();
    assert_closed_under_primitives(&arena);
}

#[test]
fn norm_emit_is_well_formed() {
    type S = Shape2<Dim<8>, Dim<8>, 2>;
    type Op = LayerNormOp<S, DTypeF32, HologramHostBoundsCpu>;
    let mut arena: Box<TermArena<128>> = Box::new(TermArena::new());
    let v0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    let v1 = arena.push(Term::Variable { name_index: 1 }).unwrap();
    let v2 = arena.push(Term::Variable { name_index: 2 }).unwrap();
    Op::emit_term(&mut arena, WittLevel::W8, v0, v1, v2).unwrap();
    assert_closed_under_primitives(&arena);
}

#[test]
fn softmax_emit_is_well_formed() {
    type S = Shape2<Dim<8>, Dim<8>, 2>;
    type Axes = Shape1<Dim<0>, 1>;
    type Op = SoftmaxOp<S, Axes, DTypeF32, HologramHostBoundsCpu>;
    let mut arena: Box<TermArena<64>> = Box::new(TermArena::new());
    let v0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    Op::emit_term(&mut arena, WittLevel::W8, v0).unwrap();
    assert_closed_under_primitives(&arena);
}

#[test]
fn pooling_emit_is_well_formed() {
    type S = Shape2<Dim<8>, Dim<8>, 2>;
    type Op = MaxPool2dOp<S, S, S, DTypeF32, HologramHostBoundsCpu>;
    let mut arena: Box<TermArena<64>> = Box::new(TermArena::new());
    let v0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    Op::emit_term(&mut arena, WittLevel::W8, v0).unwrap();
    assert_closed_under_primitives(&arena);
}

#[test]
fn attention_emit_is_well_formed() {
    type S = Shape2<Dim<8>, Dim<8>, 2>;
    type Op = AttentionOp<S, S, S, DTypeF32, HologramHostBoundsCpu>;
    let mut arena: Box<TermArena<128>> = Box::new(TermArena::new());
    let q = arena.push(Term::Variable { name_index: 0 }).unwrap();
    Op::emit_term(&mut arena, WittLevel::W8, q, q, q).unwrap();
    assert_closed_under_primitives(&arena);
}

#[test]
fn utility_layout_emit_is_well_formed() {
    type S = Shape2<Dim<8>, Dim<8>, 2>;
    type Op = PadOp<S, S, DTypeF32, HologramHostBoundsCpu>;
    let mut arena: Box<TermArena<8>> = Box::new(TermArena::new());
    Op::emit_term(&mut arena, WittLevel::W8, 0).unwrap();
    for slot in arena.as_slice().iter().flatten() {
        match slot {
            Term::Variable { .. } => {}
            other => panic!("layout PadOp emitted non-Variable: {:?}", other),
        }
    }
}

#[test]
fn backward_emit_is_well_formed() {
    let mut arena: Box<TermArena<32>> = Box::new(TermArena::new());
    let g0 = arena.push(Term::Variable { name_index: 0 }).unwrap();
    MatMulGradAOp::emit_term(&mut arena, WittLevel::W8, g0).unwrap();
    assert_closed_under_primitives(&arena);
}
