//! Content-addressed execution: identical computation is addressed once
//! and memoized, so a re-execution with identical inputs reuses cached
//! content rather than recomputing — while producing byte-identical
//! output. Distinct inputs produce new content addresses.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
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

/// add → relu → mul-by-constant, the same graph as `chained_ops`, but
/// here we re-execute to exercise the content-addressed memo.
fn build_session() -> InferenceSession<CpuBackend<BufferArena>> {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));

    let scale = graph
        .constants_mut()
        .insert(hologram_graph::constant::ConstantEntry {
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
        inputs: SmallVec::from_iter([InputSource::Node(relu), InputSource::Constant(scale)]),
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
    InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap()
}

#[test]
fn identical_reexecution_is_fully_memoized() {
    let mut session = build_session();
    let x = f32_to_le(&[-3.0, 1.0, 2.0, -1.0]);
    let y = f32_to_le(&[1.0, 2.0, 3.0, -1.0]);

    let first = session
        .execute(&[InputBuffer { bytes: &x }, InputBuffer { bytes: &y }])
        .unwrap();
    assert_eq!(le_to_f32(&first[0].bytes), vec![0.0, 6.0, 10.0, 0.0]);
    let after_first = session.content_store_len();

    // Re-execute with identical inputs: every value's derivation address
    // is already in the store, so nothing new is interned (all memo
    // hits) and the output is byte-identical.
    let second = session
        .execute(&[InputBuffer { bytes: &x }, InputBuffer { bytes: &y }])
        .unwrap();
    assert_eq!(second[0].bytes, first[0].bytes);
    assert_eq!(
        session.content_store_len(),
        after_first,
        "identical re-execution must not address any new content"
    );
}

#[test]
fn runtime_weight_footprint_is_the_deduplicated_set() {
    // `resident_bytes`/`resident_count` report the *distinct* content-addressed
    // set — the deduplicated runtime memory footprint — covering both the pinned
    // constant (`scale`) and the transient inputs/intermediates.
    let mut session = build_session();
    assert_eq!(session.resident_count(), session.content_store_len());

    let x = f32_to_le(&[-3.0, 1.0, 2.0, -1.0]);
    let y = f32_to_le(&[1.0, 2.0, 3.0, -1.0]);
    session
        .execute(&[InputBuffer { bytes: &x }, InputBuffer { bytes: &y }])
        .unwrap();

    let bytes_after = session.resident_bytes();
    let count_after = session.resident_count();
    assert!(
        bytes_after > 0,
        "footprint must include the pinned constant"
    );
    assert_eq!(session.resident_count(), session.content_store_len());

    // Identical re-execution is all memo hits: the deduplicated footprint stays
    // flat (no new buffer for content that already has an address).
    session
        .execute(&[InputBuffer { bytes: &x }, InputBuffer { bytes: &y }])
        .unwrap();
    assert_eq!(session.resident_bytes(), bytes_after);
    assert_eq!(session.resident_count(), count_after);
}

#[test]
fn addressed_io_matches_byte_io_and_never_rehashes() {
    let mut session = build_session();
    let x = f32_to_le(&[-3.0, 1.0, 2.0, -1.0]);
    let y = f32_to_le(&[1.0, 2.0, 3.0, -1.0]);

    // Byte boundary establishes the expected result.
    let expected = session
        .execute(&[InputBuffer { bytes: &x }, InputBuffer { bytes: &y }])
        .unwrap();

    // Content-addressed I/O: intern once, then operate on labels.
    let lx = session.intern_input(&x);
    let ly = session.intern_input(&y);
    let out_labels = session.execute_addressed(&[lx, ly]).unwrap();
    assert_eq!(out_labels.len(), 1);
    let bytes = session.resolve(&out_labels[0]).expect("output resolvable");
    assert_eq!(
        &bytes[..expected[0].bytes.len()],
        expected[0].bytes.as_slice()
    );

    // Re-running on the same labels is a pure graph-memo hit: no new
    // content is addressed (nothing rehashed, nothing recomputed).
    let before = session.content_store_len();
    let again = session.execute_addressed(&[lx, ly]).unwrap();
    assert_eq!(again, out_labels);
    assert_eq!(session.content_store_len(), before);
}

#[test]
fn distinct_inputs_address_new_content() {
    let mut session = build_session();
    let x = f32_to_le(&[-3.0, 1.0, 2.0, -1.0]);
    let y = f32_to_le(&[1.0, 2.0, 3.0, -1.0]);
    session
        .execute(&[InputBuffer { bytes: &x }, InputBuffer { bytes: &y }])
        .unwrap();
    let baseline = session.content_store_len();

    // Different input content ⇒ new leaf + new derivation addresses.
    let x2 = f32_to_le(&[5.0, 5.0, 5.0, 5.0]);
    let out = session
        .execute(&[InputBuffer { bytes: &x2 }, InputBuffer { bytes: &y }])
        .unwrap();
    // x2+y = [6,7,8,4]; relu = same; *2 = [12,14,16,8]
    assert_eq!(le_to_f32(&out[0].bytes), vec![12.0, 14.0, 16.0, 8.0]);
    assert!(
        session.content_store_len() > baseline,
        "novel inputs must address new content"
    );
}
