//! End-to-end runtime-indexed `Gather` / embedding lookup.
//!
//! A `[V, D]` table gathered by an `i64 [S]` index vector along axis 0 compiles
//! to a `KernelCall::Gather` and executes through the CPU gather kernel,
//! producing the `[S, D]` selected rows — `out[k, :] = table[indices[k], :]`.
//! This is the embedding path real language models need (int64 `input_ids`,
//! vocab ≫ 256), and the `O(S·D)` replacement for the `OneHot·MatMul` desugar.

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

fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn gather_embedding_int64_indices_axis0() {
    // table [V=3, D=2]: row r = [10r, 10r + 1].
    // indices [S=4] = [2, 0, 2, 1]  → rows [[20,21],[0,1],[20,21],[10,11]].
    let mut graph = Graph::new();
    let table_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(3, 2));
    let idx_sh = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let out_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(4, 2));

    let table = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: table_sh,
    });
    graph.add_input(table);
    let indices = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I64),
        output_shape: idx_sh,
    });
    graph.add_input(indices);

    let gather = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Gather),
        inputs: SmallVec::from_iter([InputSource::Node(table), InputSource::Node(indices)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: out_sh,
    });
    // axis defaults to 0 (no GatherAttrs needed for the embedding case).
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(gather)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: out_sh,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();

    let mut table_bytes = Vec::new();
    for r in 0..3i32 {
        for c in 0..2i32 {
            table_bytes.extend_from_slice(&((r * 10 + c) as f32).to_le_bytes());
        }
    }
    let mut idx_bytes = Vec::new();
    for &i in &[2i64, 0, 2, 1] {
        idx_bytes.extend_from_slice(&i.to_le_bytes());
    }

    let outputs = session
        .execute(&[
            InputBuffer {
                bytes: &table_bytes,
            },
            InputBuffer { bytes: &idx_bytes },
        ])
        .unwrap();
    let got = le_to_f32(&outputs[0].bytes);
    assert_eq!(
        got,
        vec![20.0, 21.0, 0.0, 1.0, 20.0, 21.0, 10.0, 11.0],
        "gathered rows mismatch"
    );
}
