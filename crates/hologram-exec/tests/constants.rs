//! Constants section round-trip: a graph with a Constant node should
//! pre-fill the workspace slot at session-load time.

use hologram_compiler::{compile, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    constant::ConstantEntry as GraphConstantEntry,
    node::Node,
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
fn constant_added_to_input_via_add_op() {
    // Graph: input_x + constant -> add -> output
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));

    // Register a constant of [10, 20, 30, 40] f32.
    let constant_id = graph.constants_mut().insert(GraphConstantEntry {
        bytes: f32_to_le(&[10.0, 20.0, 30.0, 40.0]),
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

    let add = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(constant_id)]),
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

    let input_bytes = f32_to_le(&[1.0, 2.0, 3.0, 4.0]);
    let outputs = session
        .execute(&[InputBuffer {
            bytes: &input_bytes,
        }])
        .unwrap();

    assert_eq!(outputs.len(), 1);
    let result = le_to_f32(&outputs[0].bytes);
    // [1, 2, 3, 4] + [10, 20, 30, 40] = [11, 22, 33, 44]
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}
