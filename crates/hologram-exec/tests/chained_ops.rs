//! Multi-op chained pipelines verifying that intermediate kernel outputs
//! correctly feed downstream kernels.

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

#[test]
fn three_op_chain_add_then_relu_then_mul() {
    // input x, input y → add → relu → mul-by-constant → output
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));

    let scale = graph.constants_mut().insert(hologram_graph::constant::ConstantEntry {
        bytes: f32_to_le(&[2.0, 2.0, 2.0, 2.0]),
        dtype: DTypeId(DTYPE_F32),
        shape,
    });

    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(x);
    let y = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(y);

    let add = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Node(y)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let relu = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(add)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let mul = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Mul),
        inputs: SmallVec::from_iter([
            InputSource::Node(relu),
            InputSource::Constant(scale),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mul)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();

    let x_bytes = f32_to_le(&[-3.0, 1.0, 2.0, -1.0]);
    let y_bytes = f32_to_le(&[1.0, 2.0, 3.0, -1.0]);
    // x+y = [-2, 3, 5, -2]; relu = [0, 3, 5, 0]; *2 = [0, 6, 10, 0]
    let outputs = session.execute(&[
        InputBuffer { bytes: &x_bytes },
        InputBuffer { bytes: &y_bytes },
    ]).unwrap();

    let result = le_to_f32(&outputs[0].bytes);
    assert_eq!(result, vec![0.0, 6.0, 10.0, 0.0]);
}

#[test]
fn five_op_chain_add_relu_add_relu_add() {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(2));

    let one = graph.constants_mut().insert(hologram_graph::constant::ConstantEntry {
        bytes: f32_to_le(&[1.0, 1.0]),
        dtype: DTypeId(DTYPE_F32),
        shape,
    });

    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(x);

    // 5 sequential add-then-relu's.
    let mut last = x;
    for _ in 0..5 {
        let add = graph.add_node(Node {
            op: GraphOp::Op(OpKind::Add),
            inputs: SmallVec::from_iter([
                InputSource::Node(last),
                InputSource::Constant(one),
            ]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: shape,
        });
        let relu = graph.add_node(Node {
            op: GraphOp::Op(OpKind::Relu),
            inputs: SmallVec::from_iter([InputSource::Node(add)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: shape,
        });
        last = relu;
    }
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(last)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();

    let input_bytes = f32_to_le(&[10.0, -3.0]);
    let outputs = session.execute(&[InputBuffer { bytes: &input_bytes }]).unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    // x = [10, -3]; +1 → [11, -2]; relu → [11, 0]
    // ... repeat 4 more times: each add adds 1, relu clamps non-neg.
    // Final: x + 5 (since both went non-neg by iter 1 in case 2).
    // [10,-3] → [11,-2]→[11,0] → [12,1]→[12,1] → [13,2]→[13,2] → [14,3]→[14,3] → [15,4]→[15,4]
    assert_eq!(result, vec![15.0, 4.0]);
}
