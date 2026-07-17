//! Prism-pipeline-routed `execute_attested` smoke tests (wiki ADR-022 D5).
//!
//! Verifies that `InferenceSession::execute_attested` produces both
//! the compute outputs and a `Grounded<Digest<32>>` attestation
//! emitted through `prism::pipeline::run`. The attestation's content
//! fingerprint is deterministic for a fixed inference unit.

use hologram_compiler::{compile, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{AttestedExecution, BufferArena, InferenceSession, InputBuffer};
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
    fn into_op_kind(self) -> Option<OpKind> {
        None
    }
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

    let AttestedExecution {
        outputs,
        prism_attestation,
        archive_fingerprint,
    } = session
        .execute_attested(&[InputBuffer { bytes: &bytes }])
        .expect("attested execution succeeds");

    // Compute outputs match the standard execute() result: relu clamps.
    let result = le_to_f32(&outputs[0].bytes);
    assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);

    // The prism witness is a sealed Grounded<Digest<32>>; its
    // content_fingerprint() is folded over the unit's TYPE-SHAPE.
    let fp = prism_attestation.content_fingerprint();
    assert!(!fp.as_bytes().is_empty());

    // The archive fingerprint is hologram's per-content anchor. It is
    // a 32-byte BLAKE3 digest over the archive bytes (spec X.1) — and
    // it is non-trivial (not all-zero) for any real archive.
    assert_ne!(archive_fingerprint, [0u8; 32]);
}

#[test]
fn attested_execute_is_deterministic_across_invocations() {
    let mut session1 = build_unary_session();
    let _ = build_unary_session(); // dummy: ensures determinism across separate session-loads
    let mut session2 = build_unary_session();
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes = f32_to_le(&input);

    let a = session1
        .execute_attested(&[InputBuffer { bytes: &bytes }])
        .unwrap();
    let b = session2
        .execute_attested(&[InputBuffer { bytes: &bytes }])
        .unwrap();

    // Compute outputs identical.
    assert_eq!(a.outputs[0].bytes, b.outputs[0].bytes);
    // Identical CompileUnits yield identical prism attestation
    // fingerprints (canonical determinism per ADR-001).
    assert_eq!(
        a.prism_attestation.content_fingerprint().as_bytes(),
        b.prism_attestation.content_fingerprint().as_bytes()
    );
    // Identical archives yield identical content anchors.
    assert_eq!(a.archive_fingerprint, b.archive_fingerprint);
}

#[test]
fn attested_execute_anchors_to_archive_fingerprint() {
    // Two graphs with different op kinds produce two different archives
    // (different footer fingerprints). The `archive_fingerprint` field
    // on `AttestedExecution` MUST differ — this is hologram's per-content
    // anchor, distinct from prism's type-shape fingerprint.
    //
    // Prism's `Grounded.content_fingerprint()` is folded only over the
    // unit's TYPE-SHAPE (witt + budget + IRI + constraints) per
    // `fold_unit_digest`, so two sessions with the same shape but
    // different content yield identical prism fingerprints — the
    // content anchor lives on hologram's side of the pair.

    fn build_session_op(op: OpKind) -> InferenceSession<CpuBackend<BufferArena>> {
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
            op: GraphOp::Op(op),
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
        InferenceSession::load(&compiled.archive, CpuBackend::<BufferArena>::new()).unwrap()
    }

    let mut relu = build_session_op(OpKind::Relu);
    let mut sigmoid = build_session_op(OpKind::Sigmoid);

    // Archive fingerprints differ (different op kinds → different
    // kernel-call payloads → different footer).
    assert_ne!(relu.archive_fingerprint(), sigmoid.archive_fingerprint());

    let bytes = f32_to_le(&[0.5f32, 1.5, 2.5, 3.5]);
    let a = relu
        .execute_attested(&[InputBuffer { bytes: &bytes }])
        .unwrap();
    let b = sigmoid
        .execute_attested(&[InputBuffer { bytes: &bytes }])
        .unwrap();

    // The per-content anchors differ — TC-03's content-anchoring
    // commitment holds.
    assert_ne!(a.archive_fingerprint, b.archive_fingerprint);
    // The prism witnesses' shape-fingerprints are identical because
    // both sessions admit the same `CompileUnit` shape (W32 + same
    // budget + same result-type IRI + same constraints). This is by
    // prism's design — verify the documented separation of concerns.
    assert_eq!(
        a.prism_attestation.content_fingerprint().as_bytes(),
        b.prism_attestation.content_fingerprint().as_bytes()
    );
}
