//! Backward emission (spec V.4 / ADR-043) — exercises
//! `compile_with_backward` and the `OpKind::primary_grad` mapping.

use hologram_compiler::{compile_with_backward, BackendKind};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{append_backward, Graph, GraphOp, InputSource, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

fn build_simple_graph() -> (Graph, hologram_graph::NodeId, hologram_graph::NodeId) {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(x);
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
    (graph, x, y)
}

#[test]
fn append_backward_grows_graph_with_grad_nodes() {
    let (mut graph, _x, y) = build_simple_graph();
    let original_count = graph.node_count();
    let grads = append_backward(&mut graph, y).unwrap();
    assert_eq!(grads.len(), 1, "expected one input gradient");
    assert!(
        graph.node_count() > original_count,
        "backward emission should add nodes"
    );
    // Confirm a UnaryGrad node was added (Sigmoid → UnaryGrad per V.4).
    let added_unary_grad = graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op, GraphOp::Op(OpKind::UnaryGrad)));
    assert!(
        added_unary_grad,
        "expected a UnaryGrad node in backward subgraph"
    );
}

#[test]
fn primary_grad_mapping_is_consistent() {
    // Spec V.4: every differentiable op has a defined gradient. Spot-check
    // the canonical mappings.
    assert_eq!(OpKind::MatMul.primary_grad(), Some(OpKind::MatMulGradA));
    assert_eq!(OpKind::Conv2d.primary_grad(), Some(OpKind::Conv2dGradX));
    assert_eq!(OpKind::Softmax.primary_grad(), Some(OpKind::SoftmaxGrad));
    assert_eq!(
        OpKind::LayerNorm.primary_grad(),
        Some(OpKind::LayerNormGrad)
    );
    assert_eq!(OpKind::Sigmoid.primary_grad(), Some(OpKind::UnaryGrad));
    assert_eq!(OpKind::Mul.primary_grad(), Some(OpKind::MulGrad));
    // Non-differentiable / identity-passthrough.
    assert_eq!(OpKind::Add.primary_grad(), None);
    assert_eq!(OpKind::Reshape.primary_grad(), None);
}

#[test]
fn compile_with_backward_returns_input_gradients() {
    let (graph, _x, y) = build_simple_graph();
    let (output, input_grads) =
        compile_with_backward(graph, y, BackendKind::Cpu, WittLevel::W32).unwrap();
    assert_eq!(input_grads.len(), 1);
    assert!(!output.archive.is_empty());
    // The augmented graph compiles to more kernel calls than the forward.
    assert!(output.stats.total_nodes > 3);
}
