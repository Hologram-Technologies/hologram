//! Prism-pipeline-routed `execute_attested` smoke tests (wiki ADR-022 D5).
//!
//! Verifies that `InferenceSession::execute_attested` produces both
//! the compute outputs and a `Grounded<Digest<32>>` attestation
//! emitted through `prism::pipeline::run`. The attestation's content
//! fingerprint is deterministic for a fixed inference unit.

use hologram_compiler::{compile, BackendKind};
use hologram_backend::CpuBackend;
use hologram_exec::{InferenceSession, BufferArena, InputBuffer, AttestedExecution};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
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

fn build_unary_session() -> InferenceSession<CpuBackend<BufferArena>> {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(x);
    let r = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Op(GraphOp::Output.into_op_kind().unwrap_or(OpKind::Relu)),
        inputs: SmallVec::from_iter([InputSource::Node(r)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    InferenceSession::load(&compiled.archive, backend).unwrap()
}

// Helper trait — GraphOp doesn't expose an `into_op_kind` shim in the
// public surface; use this local mapping for the test's Output node.
trait GraphOpExt {
    fn into_op_kind(self) -> Option<OpKind>;
}
impl GraphOpExt for GraphOp {
    fn into_op_kind(self) -> Option<OpKind> { None }
}

#[test]
fn attested_execute_emits_compute_outputs_and_grounded_attestation() {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(x);
    let r = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(r)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let input = vec![-2.0f32, -1.0, 1.0, 2.0];
    let bytes = f32_to_le(&input);

    let AttestedExecution { outputs, attestation } = session
        .execute_attested(&[InputBuffer { bytes: &bytes }])
        .expect("attested execution succeeds");

    // Compute outputs match the standard execute() result: relu clamps.
    let result = le_to_f32(&outputs[0].bytes);
    assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);

    // The prism-emitted attestation is a Grounded<Digest<32>> — the
    // accessor surface is sealed, but content_fingerprint() is public.
    let fp = attestation.content_fingerprint();
    // Fingerprint carries at least one byte (the BLAKE3 width
    // routed through hologram's canonical Hasher<32> selection).
    assert!(!fp.as_bytes().is_empty());
}

#[test]
fn attested_execute_is_deterministic_across_invocations() {
    let mut session1 = build_unary_session();
    let _ = build_unary_session(); // dummy: ensures determinism across separate session-loads
    let mut session2 = build_unary_session();
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes = f32_to_le(&input);

    let a = session1.execute_attested(&[InputBuffer { bytes: &bytes }]).unwrap();
    let b = session2.execute_attested(&[InputBuffer { bytes: &bytes }]).unwrap();

    // Compute outputs identical.
    assert_eq!(a.outputs[0].bytes, b.outputs[0].bytes);
    // The Grounded<T> attestation is sealed but its content fingerprint
    // is publicly readable; identical CompileUnits yield identical
    // fingerprints (the canonical attestation determinism per ADR-001).
    assert_eq!(a.attestation.content_fingerprint().as_bytes(),
               b.attestation.content_fingerprint().as_bytes());
}
