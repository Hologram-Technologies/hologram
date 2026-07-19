//! Parallel-within-level execution (spec VIII.2). Only runs when the
//! `parallel` cargo feature is enabled.

#![cfg(feature = "parallel")]

use hologram_compiler::{compile, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

fn f32_to_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn parallel_within_level_produces_same_result() {
    // Build a graph with two independent unary ops on the same input —
    // both share a schedule level. With the `parallel` feature on, the
    // executor dispatches them concurrently; the result must match the
    // sequential reference.
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(x);
    let abs = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Abs),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let _sign = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Sign),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(abs)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let input = vec![-1.0f32, 0.0, 2.5, -3.5];
    let bytes = f32_to_le(&input);
    let outputs = session.execute(&[InputBuffer { bytes: &bytes }]).unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    assert_eq!(result, vec![1.0, 0.0, 2.5, 3.5]);
}
