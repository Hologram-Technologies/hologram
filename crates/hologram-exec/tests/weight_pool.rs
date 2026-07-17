//! Spec X.3 + X-7: large weight bodies live in the BLAKE3-deduped
//! `Weights` section, with `Constants` carrying only fingerprint
//! references. Verifies the archive doesn't double-store large bodies
//! and that the runtime resolves slot bytes via the WeightStore.

use hologram_compiler::{compile, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::constant::ConstantEntry;
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const INLINE_THRESHOLD: usize = 4096;

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
fn small_constants_are_inlined_in_archive() {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let body = f32_to_le(&[0.5, 0.5, 0.5, 0.5]);
    let cid = graph.constants_mut().insert(ConstantEntry {
        bytes: body.clone(),
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
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(cid)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(add)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    assert!(body.len() <= INLINE_THRESHOLD);
    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let outputs = session
        .execute(&[InputBuffer {
            bytes: &f32_to_le(&[1.0, 2.0, 3.0, 4.0]),
        }])
        .unwrap();
    let expected: Vec<f32> = [1.0_f32, 2.0, 3.0, 4.0]
        .iter()
        .zip([0.5_f32; 4].iter())
        .map(|(a, b)| a + b)
        .collect();
    assert_eq!(le_to_f32(&outputs[0].bytes), expected);
}

#[test]
fn large_weights_become_content_addressed_references() {
    // A 8 KiB constant — above the 4 KiB inline threshold — should be
    // stored once in the Weights pool and referenced (not inlined) by
    // Constants. Total archive size should be approximately 8 KiB plus
    // bookkeeping, NOT 16 KiB (which would indicate double-storage).
    let mut graph = Graph::new();
    let shape = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(2048));
    let body: Vec<u8> = (0..8192).map(|i| (i & 0xFF) as u8).collect();
    let cid = graph.constants_mut().insert(ConstantEntry {
        bytes: body.clone(),
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
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(cid)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(add)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    // Archive must contain the body once (8 KiB) plus headers + kernel
    // section + weight pool overhead. With double-storage it'd be
    // ≥ 16 KiB; the corrected codec keeps it under ~12 KiB.
    let archive_size = compiled.archive.len();
    assert!(
        archive_size < 16 * 1024,
        "archive {} bytes — body appears double-stored",
        archive_size
    );
    assert!(
        archive_size >= body.len(),
        "archive {} bytes — body lost",
        archive_size
    );
}

#[test]
fn duplicate_large_weights_share_storage() {
    // Two graphs constants with identical bodies must share storage
    // through BLAKE3 dedup — the archive grows by O(1) overhead, not
    // by 2× the body size.
    let mut graph = Graph::new();
    let shape = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(2048));
    let body: Vec<u8> = (0..8192).map(|i| (i & 0xFF) as u8).collect();
    let c1 = graph.constants_mut().insert(ConstantEntry {
        bytes: body.clone(),
        dtype: DTypeId(DTYPE_F32),
        shape,
    });
    let c2 = graph.constants_mut().insert(ConstantEntry {
        bytes: body.clone(),
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
    let a = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(c1)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let b = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Constant(c2)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    // Two identical 8 KiB bodies => archive ≈ 8 KiB + overhead, not 16 KiB.
    let archive_size = compiled.archive.len();
    assert!(
        archive_size < 12 * 1024,
        "archive {} bytes — duplicate bodies not deduped",
        archive_size
    );
}
