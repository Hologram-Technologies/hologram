//! End-to-end execution with real graph shapes, real f32 data, and
//! verification of numerical output.
//!
//! Constructs graphs programmatically (not through the line parser) to
//! exercise the shape-resolution and dtype-aware byte-sizing paths.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::node::Node;
use hologram_graph::{
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind,
};
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
fn unary_relu_f32_real_data() {
    // Graph: input_x -> relu -> output
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(8));

    let in_node = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(in_node);

    let relu_node = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(in_node)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });

    let out_node = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(relu_node)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out_node);

    // Compile.
    let out = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();

    // Load + execute with real f32 input.
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();
    assert_eq!(session.input_count(), 1);
    assert_eq!(session.output_count(), 1);

    let input_bytes = f32_to_le(&[-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, -7.0]);
    let outputs = session
        .execute(&[InputBuffer {
            bytes: &input_bytes,
        }])
        .unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].bytes.len(), 8 * 4);
    let result = le_to_f32(&outputs[0].bytes);
    // ReLU on the input vector: negatives clamp to 0, non-negatives pass through.
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 5.0, 0.0]);
}

#[test]
fn softmax_rank3_normalizes_over_last_axis() {
    // Regression: norm/softmax shape derivation used to fire only for rank-2
    // inputs, leaving `feature = 0` for the common rank-3 `[batch, seq, hidden]`
    // transformer layout — the kernel then short-circuits on `feature == 0` and
    // silently emits an untouched (zeroed) output. The derivation now takes the
    // last axis as `feature` and the product of preceding axes as `batch`, so
    // any rank ≥ 1 normalizes correctly.
    let mut graph = Graph::new();
    // [batch=2, seq=2, hidden=3] -> 4 rows of 3 features.
    let shape = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank3(2, 2, 3));

    let in_node = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(in_node);

    let softmax = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Softmax),
        inputs: SmallVec::from_iter([InputSource::Node(in_node)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });

    let out_node = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(softmax)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out_node);

    let out = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();

    // Four rows of three equal logits -> each row softmaxes to [1/3, 1/3, 1/3].
    let input_bytes = f32_to_le(&[0.0; 12]);
    let outputs = session
        .execute(&[InputBuffer {
            bytes: &input_bytes,
        }])
        .unwrap();

    let result = le_to_f32(&outputs[0].bytes);
    assert_eq!(result.len(), 12);
    for v in &result {
        assert!((v - 1.0 / 3.0).abs() < 1e-6, "expected 1/3, got {v}");
    }
    // Each row (last axis = 3) sums to 1 — proves normalization actually ran.
    for row in result.chunks_exact(3) {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "row should sum to 1, got {s}");
    }
}

#[test]
fn matmul_f32_real_data() {
    // Graph: input_a, input_b -> matmul -> output
    let mut graph = Graph::new();
    let shape_a = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(2, 3));
    let shape_b = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(3, 2));
    let shape_out = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(2, 2));

    let a = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape_a,
    });
    graph.add_input(a);
    let b = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape_b,
    });
    graph.add_input(b);

    let mm = graph.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape_out,
    });

    let out_node = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape_out,
    });
    graph.add_output(out_node);

    let out = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();

    // A = [[1,2,3],[4,5,6]]  (2x3)
    // B = [[7,8],[9,10],[11,12]]  (3x2)
    // A·B = [[58, 64], [139, 154]]  (2x2)
    let a_bytes = f32_to_le(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_bytes = f32_to_le(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let outputs = session
        .execute(&[
            InputBuffer { bytes: &a_bytes },
            InputBuffer { bytes: &b_bytes },
        ])
        .unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].bytes.len(), 4 * 4);
    let result = le_to_f32(&outputs[0].bytes);
    assert!((result[0] - 58.0).abs() < 1e-3, "got {}", result[0]);
    assert!((result[1] - 64.0).abs() < 1e-3, "got {}", result[1]);
    assert!((result[2] - 139.0).abs() < 1e-3, "got {}", result[2]);
    assert!((result[3] - 154.0).abs() < 1e-3, "got {}", result[3]);

    // Cross-check by reading the MatMul node's slot directly (alias
    // confirmation — the output port should resolve to the same data).
    let mm_bytes = session.workspace().read_slot(mm.0 as usize).unwrap();
    let mm_floats = le_to_f32(&mm_bytes[..16]);
    assert_eq!(result, mm_floats);
}

#[test]
fn binary_add_f32_real_data() {
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

    let add = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out_node = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(add)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out_node);

    let out = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();

    let a_bytes = f32_to_le(&[1.0, 2.0, 3.0, 4.0]);
    let b_bytes = f32_to_le(&[10.0, 20.0, 30.0, 40.0]);
    let outputs = session
        .execute(&[
            InputBuffer { bytes: &a_bytes },
            InputBuffer { bytes: &b_bytes },
        ])
        .unwrap();

    assert_eq!(outputs.len(), 1);
    let result = le_to_f32(&outputs[0].bytes);
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);

    // Confirm aliasing: the Output port reads the Add node's slot.
    let add_bytes = session.workspace().read_slot(add.0 as usize).unwrap();
    let add_floats = le_to_f32(&add_bytes[..16]);
    assert_eq!(result, add_floats);
}
