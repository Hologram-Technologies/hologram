//! End-to-end coverage across op categories with real f32 data.

use hologram_compiler::{compile, BackendKind};
use hologram_backend::CpuBackend;
use hologram_exec::{InferenceSession, BufferArena, InputBuffer};
use hologram_graph::{
    Graph, GraphOp, InputSource, OpKind,
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
};
use smallvec::SmallVec;
use prism::vocabulary::WittLevel;

const DTYPE_F32: u8 = 8;

fn f32_to_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn build_unary_graph(kind: OpKind, n: u64) -> (Graph, hologram_graph::NodeId) {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(n));
    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(x);
    let op = graph.add_node(Node {
        op: GraphOp::Op(kind),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(op)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);
    (graph, op)
}

fn run_unary(kind: OpKind, n: u64, input: Vec<f32>) -> Vec<f32> {
    let (graph, _) = build_unary_graph(kind, n);
    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let bytes = f32_to_le(&input);
    let outputs = session.execute(&[InputBuffer { bytes: &bytes }]).unwrap();
    le_to_f32(&outputs[0].bytes)
}

#[test]
fn sigmoid_f32() {
    let result = run_unary(OpKind::Sigmoid, 4, vec![0.0, 1.0, -1.0, 100.0]);
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 0.731_058_6).abs() < 1e-3);
    assert!((result[2] - 0.268_941_4).abs() < 1e-3);
    assert!(result[3] > 0.999);
}

#[test]
fn tanh_f32() {
    let result = run_unary(OpKind::Tanh, 3, vec![0.0, 1.0, -1.0]);
    assert!(result[0].abs() < 1e-6);
    assert!((result[1] - 0.761_594).abs() < 1e-3);
    assert!((result[2] + 0.761_594).abs() < 1e-3);
}

#[test]
fn exp_f32() {
    let result = run_unary(OpKind::Exp, 3, vec![0.0, 1.0, 2.0]);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - core::f32::consts::E).abs() < 1e-3);
    assert!((result[2] - 7.389_056).abs() < 1e-3);
}

#[test]
fn sqrt_f32() {
    let result = run_unary(OpKind::Sqrt, 4, vec![0.0, 1.0, 4.0, 9.0]);
    assert!(result[0].abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-5);
    assert!((result[3] - 3.0).abs() < 1e-5);
}

#[test]
fn abs_f32() {
    let result = run_unary(OpKind::Abs, 4, vec![-3.0, 0.0, 5.0, -2.5]);
    assert_eq!(result, vec![3.0, 0.0, 5.0, 2.5]);
}

#[test]
fn div_f32() {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(3));
    let a = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(a);
    let b = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(b);
    let div = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Div),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(div)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);
    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let outputs = session.execute(&[
        InputBuffer { bytes: &f32_to_le(&[10.0, 20.0, 5.0]) },
        InputBuffer { bytes: &f32_to_le(&[2.0, 4.0, 5.0]) },
    ]).unwrap();
    assert_eq!(le_to_f32(&outputs[0].bytes), vec![5.0, 5.0, 1.0]);
}

#[test]
fn min_max_f32() {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let a = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(a);
    let b = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(b);
    let min_n = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Min),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(min_n)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);
    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let outputs = session.execute(&[
        InputBuffer { bytes: &f32_to_le(&[1.0, 5.0, 3.0, -2.0]) },
        InputBuffer { bytes: &f32_to_le(&[3.0, 2.0, 3.0, -1.0]) },
    ]).unwrap();
    assert_eq!(le_to_f32(&outputs[0].bytes), vec![1.0, 2.0, 3.0, -2.0]);
}

#[test]
fn reshape_f32_passes_through() {
    let mut graph = Graph::new();
    let shape_in = graph.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 3));
    let shape_out = graph.shape_registry_mut().intern(ShapeDescriptor::rank2(3, 2));
    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape_in,
    });
    graph.add_input(x);
    let r = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Reshape),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape_out,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(r)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape_out,
    });
    graph.add_output(out);
    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let outputs = session.execute(&[
        InputBuffer { bytes: &f32_to_le(&input) },
    ]).unwrap();
    assert_eq!(le_to_f32(&outputs[0].bytes), input);
}
