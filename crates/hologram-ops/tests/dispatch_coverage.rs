//! Dispatch coverage (spec V.3 + I-1): every `OpKind` variant emits a
//! non-empty Term tree whose `Term::Application::operator` is restricted
//! to the closed 10-PrimitiveOp set.

use hologram_ops::HoloArena;
use hologram_ops::{emit_op_term, OpKind};
use uor_foundation::enforcement::Term;
use uor_foundation::{PrimitiveOp, WittLevel};

fn assert_closed_primitive_set<const CAP: usize>(arena: &HoloArena<CAP>) {
    for slot in arena.as_slice().iter().flatten() {
        if let Term::Application { operator, .. } = slot {
            // PrimitiveOp is exhaustively closed (spec I-1); this match
            // statically attests the operator is one of the 10 variants.
            match *operator {
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

fn try_emit(kind: OpKind) -> bool {
    // Box the arena: `HoloArena<256>` holds 256 × `Option<Term>` where
    // each `Term::Literal` carries a 4 KiB `TermValue` buffer in
    // upstream 0.4.15. On-stack instantiation in a loop blows the
    // default thread stack.
    // Arena CAP picked to cover the largest hologram op marker (Attention
    // at CAP = 96 per spec V.5) plus headroom for the per-arity variable
    // prologue, while keeping the on-stack `[Option<Term>; CAP]` size
    // (each `Term::Literal` carries a 4 KiB `TermValue` buffer in
    // upstream 0.4.15) below the default thread stack ceiling.
    let mut arena: HoloArena<128> = HoloArena::new();
    let arity = kind.primary_arity();
    let v0 = arena.push(Term::Variable { name_index: 0 }).expect("v0");
    for i in 1..arity {
        arena
            .push(Term::Variable {
                name_index: i as u32,
            })
            .expect("vi");
    }
    let res = emit_op_term(kind, &mut arena, WittLevel::W32, v0);
    assert_closed_primitive_set(&arena);
    res.is_some()
}

const ALL_OP_KINDS: &[OpKind] = &[
    OpKind::Neg,
    OpKind::Bnot,
    OpKind::Succ,
    OpKind::Pred,
    OpKind::Add,
    OpKind::Sub,
    OpKind::Mul,
    OpKind::Xor,
    OpKind::And,
    OpKind::Or,
    OpKind::Relu,
    OpKind::Sigmoid,
    OpKind::Tanh,
    OpKind::Gelu,
    OpKind::Silu,
    OpKind::Elu,
    OpKind::Selu,
    OpKind::Exp,
    OpKind::Log,
    OpKind::Log1p,
    OpKind::Sqrt,
    OpKind::Reciprocal,
    OpKind::Sin,
    OpKind::Cos,
    OpKind::Tan,
    OpKind::Asin,
    OpKind::Acos,
    OpKind::Atan,
    OpKind::Ceil,
    OpKind::Floor,
    OpKind::Round,
    OpKind::Erf,
    OpKind::IsNaN,
    OpKind::Sign,
    OpKind::Abs,
    OpKind::Div,
    OpKind::Pow,
    OpKind::Mod,
    OpKind::Min,
    OpKind::Max,
    OpKind::Equal,
    OpKind::Less,
    OpKind::LessOrEqual,
    OpKind::Greater,
    OpKind::GreaterOrEqual,
    OpKind::MatMul,
    OpKind::Gemm,
    OpKind::Conv2d,
    OpKind::ConvTranspose2d,
    OpKind::LayerNorm,
    OpKind::RmsNorm,
    OpKind::GroupNorm,
    OpKind::InstanceNorm,
    OpKind::AddRmsNorm,
    OpKind::ReduceSum,
    OpKind::ReduceMean,
    OpKind::ReduceProd,
    OpKind::ReduceMin,
    OpKind::ReduceMax,
    OpKind::Reshape,
    OpKind::Transpose,
    OpKind::Concat,
    OpKind::Slice,
    OpKind::Softmax,
    OpKind::LogSoftmax,
    OpKind::MaxPool2d,
    OpKind::AvgPool2d,
    OpKind::GlobalAvgPool,
    OpKind::Attention,
    OpKind::FusedSwiGlu,
    OpKind::Pad,
    OpKind::Expand,
    OpKind::Resize,
    OpKind::CumSum,
    OpKind::RotaryEmbedding,
    OpKind::Clip,
    OpKind::Lrn,
    OpKind::Where,
    OpKind::Im2Col,
    OpKind::Col2Im,
    OpKind::Dequantize,
];

#[test]
fn every_op_kind_dispatches_to_a_well_formed_tree() {
    for &kind in ALL_OP_KINDS {
        assert!(try_emit(kind), "emit_op_term failed for {:?}", kind);
    }
}

#[test]
fn op_kind_catalog_size_matches_spec() {
    // Locks the catalog cardinality. The 26 backward `*Grad` markers were
    // removed when autodiff moved to forward-op composition; `Im2Col`/`Col2Im`
    // were added as the conv-composition layout primitives. Adjust if and only
    // if the catalog is intentionally extended.
    assert_eq!(ALL_OP_KINDS.len(), 81);
}

#[test]
fn every_op_emit_fits_in_declared_cap() {
    // Spec V.5: each op declares an arena CAP. The emitted Term tree
    // must fit within that CAP. Drives a fresh arena per op, populates
    // arity variables, calls dispatch, and asserts the slot count
    // stays at or below `OpKind::cap()`.
    for &kind in ALL_OP_KINDS {
        // Box the arena: `HoloArena<256>` holds 256 × `Option<Term>` where
        // each `Term::Literal` carries a 4 KiB `TermValue` buffer in
        // upstream 0.4.15. On-stack instantiation in a loop blows the
        // default thread stack.
        // Arena CAP picked to cover the largest hologram op marker (Attention
        // at CAP = 96 per spec V.5) plus headroom for the per-arity variable
        // prologue, while keeping the on-stack `[Option<Term>; CAP]` size
        // (each `Term::Literal` carries a 4 KiB `TermValue` buffer in
        // upstream 0.4.15) below the default thread stack ceiling.
        let mut arena: HoloArena<128> = HoloArena::new();
        let arity = kind.primary_arity();
        let v0 = arena.push(Term::Variable { name_index: 0 }).expect("v0");
        for i in 1..arity {
            arena
                .push(Term::Variable {
                    name_index: i as u32,
                })
                .expect("vi");
        }
        let pre = arena.as_slice().iter().filter(|s| s.is_some()).count();
        let _ = hologram_ops::emit_op_term(kind, &mut arena, WittLevel::W32, v0)
            .unwrap_or_else(|| panic!("emit_op_term failed for {:?}", kind));
        let post = arena.as_slice().iter().filter(|s| s.is_some()).count();
        let used = post - pre;
        let cap = kind.cap();
        assert!(
            used <= cap,
            "{:?} emitted {} term slots, exceeds declared CAP {}",
            kind,
            used,
            cap
        );
    }
}
