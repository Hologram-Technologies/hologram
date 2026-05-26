//! Backward emission (spec V.4 / ADR-043) — autodiff by **composition**.
//!
//! Gradients are not new primitives: `append_backward` composes each op's
//! vector-Jacobian product from existing forward ops (the chain rule is
//! categorical composition). These tests assert the structural contract; the
//! numerical correctness of every VJP is grad-checked against finite
//! differences in `hologram-exec/tests/autodiff.rs`.

use hologram_compiler::{compile_with_backward, BackendKind};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{append_backward, BackwardError, Graph, GraphOp, InputSource, NodeId, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

fn input(g: &mut Graph, shape: hologram_graph::registry::ShapeId) -> NodeId {
    let id = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    g.add_input(id);
    id
}

fn build_sigmoid_graph() -> (Graph, NodeId) {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let x = input(&mut graph, shape);
    let y = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Sigmoid),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(y)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);
    (graph, y)
}

#[test]
fn append_backward_composes_from_forward_ops_only() {
    let (mut graph, y) = build_sigmoid_graph();
    let original = graph.node_count();
    let grads = append_backward(&mut graph, y).unwrap();
    assert_eq!(grads.len(), 1, "one differentiable input");
    assert!(graph.node_count() > original, "backward adds nodes");

    // The σ VJP is g·y·(1−y): composed purely from forward Sub/Mul (+ a 1.0
    // constant). Gradients run on the already-verified forward kernels — there
    // are no `*Grad` op-kinds in the catalog at all (they were removed when
    // autodiff moved to composition), so every appended node is a forward op.
    let added = &graph.nodes()[original..];
    assert!(
        added.iter().any(|n| matches!(n.op, GraphOp::Op(OpKind::Mul)))
            && added.iter().any(|n| matches!(n.op, GraphOp::Op(OpKind::Sub))),
        "sigmoid VJP composes Sub + Mul"
    );
}

#[test]
fn unimplemented_vjp_fails_loud() {
    // An op whose VJP is not composed yet errors explicitly — it is never
    // silently approximated.
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let x = input(&mut graph, shape);
    let y = graph.add_node(Node {
        // `Xor` is a bitwise op on the discrete byte ring — it has no
        // real-valued input and so no calculus derivative; append_backward
        // fails loud rather than fabricate one. (Verifies the fail-loud
        // mechanism. Every *differentiable* op — arithmetic, activations,
        // matmul/gemm, conv, attention, norms, pools, resize, lrn, rope, mod,
        // and even the predicate ops' 0-gradient — is grad-checked in
        // hologram-exec/tests/autodiff.rs. Only the discrete byte-algebra ops
        // (And/Or/Xor/Bnot/Succ/Pred) and Dequantize remain gradient-free.)
        op: GraphOp::Op(OpKind::Xor),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(y)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);
    assert_eq!(
        append_backward(&mut graph, y),
        Err(BackwardError::NoGradient(OpKind::Xor))
    );
}

#[test]
fn compile_with_backward_returns_input_gradients() {
    let (graph, y) = build_sigmoid_graph();
    let (output, input_grads) =
        compile_with_backward(graph, y, BackendKind::Cpu, WittLevel::W32).unwrap();
    assert_eq!(input_grads.len(), 1);
    assert!(!output.archive.is_empty());
    // Forward + composed backward compiles to more than the 3 forward nodes.
    assert!(output.stats.total_nodes > 3);
}
