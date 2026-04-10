//! Ring algebraic fusion tests.
//!
//! Tests that the fusion pass recognizes ring algebraic identities.

use hologram_core::op::PrimOp;
use hologram_ir::graph::{Graph, GraphOp};
use hologram_ir::GraphBuilder;

/// Helper: build a 2-op chain Input → op1 → op2 → Output, fuse, return fused graph.
fn fuse_chain(op1: PrimOp, op2: PrimOp) -> Graph {
    let mut g = GraphBuilder::new()
        .node(GraphOp::Input)
        .node(GraphOp::Prim(op1))
        .node(GraphOp::Prim(op2))
        .node(GraphOp::Output)
        .edge(0, 1)
        .edge(1, 2)
        .edge(2, 3)
        .build();
    let _ = hologram_ir::analysis::analyze(&mut g);
    g
}

fn count_op(g: &Graph, op: PrimOp) -> usize {
    g.nodes()
        .filter(|n| matches!(&n.op, GraphOp::Prim(p) if *p == op))
        .count()
}

// ── Involution cancellation (already implemented) ────────────────────────

#[test]
fn neg_neg_cancels() {
    let g = fuse_chain(PrimOp::Neg, PrimOp::Neg);
    assert_eq!(count_op(&g, PrimOp::Neg), 0, "neg∘neg should cancel");
}

#[test]
fn bnot_bnot_cancels() {
    let g = fuse_chain(PrimOp::Bnot, PrimOp::Bnot);
    assert_eq!(count_op(&g, PrimOp::Bnot), 0, "bnot∘bnot should cancel");
}

// ── Succ/Pred cancellation ───────────────────────────────────────────────

#[test]
fn succ_pred_cancels() {
    let g = fuse_chain(PrimOp::Succ, PrimOp::Pred);
    assert_eq!(
        count_op(&g, PrimOp::Succ) + count_op(&g, PrimOp::Pred),
        0,
        "succ∘pred should cancel"
    );
}

#[test]
fn pred_succ_cancels() {
    let g = fuse_chain(PrimOp::Pred, PrimOp::Succ);
    assert_eq!(
        count_op(&g, PrimOp::Succ) + count_op(&g, PrimOp::Pred),
        0,
        "pred∘succ should cancel"
    );
}

// ── Cross-involution composition ─────────────────────────────────────────

#[test]
fn neg_bnot_fuses() {
    // Chain: Input → Bnot → Neg → Output = Neg(Bnot(x)) = succ(x)
    let g = fuse_chain(PrimOp::Bnot, PrimOp::Neg);
    // After fusion, the separate Neg and Bnot should not both remain
    let neg_count = count_op(&g, PrimOp::Neg);
    let bnot_count = count_op(&g, PrimOp::Bnot);
    assert!(
        neg_count == 0 || bnot_count == 0,
        "neg∘bnot should fuse (found {neg_count} neg + {bnot_count} bnot)"
    );
}

#[test]
fn bnot_neg_fuses() {
    // Chain: Input → Neg → Bnot → Output = Bnot(Neg(x)) = pred(x)
    let g = fuse_chain(PrimOp::Neg, PrimOp::Bnot);
    let neg_count = count_op(&g, PrimOp::Neg);
    let bnot_count = count_op(&g, PrimOp::Bnot);
    assert!(
        neg_count == 0 || bnot_count == 0,
        "bnot∘neg should fuse (found {neg_count} neg + {bnot_count} bnot)"
    );
}
