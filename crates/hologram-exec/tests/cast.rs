//! End-to-end numeric `Cast`: an i64 input cast to f32 compiles to a
//! `KernelCall::Cast` and executes through the CPU cast kernel — the general
//! int→float conversion (distinct from `Dequantize`), value-preserving.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DTYPE_I64: u8 = 5;

#[test]
fn cast_i64_to_f32_end_to_end() {
    let mut graph = Graph::new();
    let sh = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let inp = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I64),
        output_shape: sh,
    });
    graph.add_input(inp);
    let cast = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Cast),
        inputs: SmallVec::from_iter([InputSource::Node(inp)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(cast)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();

    let vals: [i64; 4] = [0, 42, -7, 1024];
    let mut bytes = Vec::new();
    for &v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let outputs = session.execute(&[InputBuffer { bytes: &bytes }]).unwrap();
    let got: Vec<f32> = outputs[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, vec![0.0, 42.0, -7.0, 1024.0]);
}
