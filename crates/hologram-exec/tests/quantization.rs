//! Quantized weight round-trip (spec X-5).
//!
//! - INT8 weights with per-tensor scale/zero-point.
//! - Packed INT4 weights.
//!
//! Both compile to a `KernelCall::Dequantize` and execute through the
//! CPU dequant kernel, producing F32 output bytes.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind, QuantAttrs};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DTYPE_I8: u8 = 2;
const DTYPE_I4: u8 = 10;

fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn dequantize_int8_round_trip() {
    // y = (q − zp) · scale  with scale = 0.5, zp = 0 over q = [-2, 0, 2, 4].
    // Expected: [-1.0, 0.0, 1.0, 2.0]
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let q_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I8),
        output_shape: shape,
    });
    graph.add_input(q_in);
    let dq = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([InputSource::Node(q_in)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            scale_bits: 0.5f32.to_bits(),
            zero_point: 0,
            axis: -1,
        },
    );
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    // Input: i8 values -2, 0, 2, 4 packed as bytes via two's-complement.
    let q_bytes: Vec<u8> = vec![(-2i8) as u8, 0, 2, 4];
    let outputs = session.execute(&[InputBuffer { bytes: &q_bytes }]).unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    assert!((result[0] - (-1.0)).abs() < 1e-6, "got {:?}", result);
    assert!((result[1] - 0.0).abs() < 1e-6);
    assert!((result[2] - 1.0).abs() < 1e-6);
    assert!((result[3] - 2.0).abs() < 1e-6);
}

#[test]
fn dequantize_int8_per_channel_round_trip() {
    // Weight `[2, 3]` quantized per output channel (axis 0): row 0 uses
    // scale 0.5 / zp 0, row 1 uses scale 0.25 / zp 2. The scale/zero-point
    // vectors are the dequantize node's 2nd/3rd (constant) operands.
    use hologram_graph::constant::ConstantEntry;
    let mut graph = Graph::new();
    let shape = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(2, 3));
    let vsh = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(2));
    let q_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I8),
        output_shape: shape,
    });
    graph.add_input(q_in);
    let scale = graph.constants_mut().insert(ConstantEntry {
        bytes: [0.5f32, 0.25]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect(),
        dtype: DTypeId(DTYPE_F32),
        shape: vsh,
    });
    let zp = graph.constants_mut().insert(ConstantEntry {
        bytes: [0i32, 2].iter().flat_map(|z| z.to_le_bytes()).collect(),
        dtype: DTypeId(2),
        shape: vsh,
    });
    let dq = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([
            InputSource::Node(q_in),
            InputSource::Constant(scale),
            InputSource::Constant(zp),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            scale_bits: 0,
            zero_point: 0,
            axis: 0,
        },
    );
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let q_bytes: Vec<u8> = vec![(-2i8) as u8, 0, 2, 4, 6, 8];
    let outputs = session.execute(&[InputBuffer { bytes: &q_bytes }]).unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    // row 0: (q−0)·0.5 = [−1, 0, 1] ; row 1: (q−2)·0.25 = [0.5, 1.0, 1.5]
    let want = [-1.0f32, 0.0, 1.0, 0.5, 1.0, 1.5];
    for (g, w) in result.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-6, "got {result:?} want {want:?}");
    }
}

#[test]
fn dequantize_int4_packed_unpacks_correctly() {
    // INT4 packs two values per byte. Encode q = [-2, 1, 0, -1]:
    //   element 0 = -2 → 0b1110 (low nibble of byte 0)
    //   element 1 =  1 → 0b0001 (high nibble of byte 0)
    //   element 2 =  0 → 0b0000 (low nibble of byte 1)
    //   element 3 = -1 → 0b1111 (high nibble of byte 1)
    // → bytes = [0x1E, 0xF0]
    // With scale = 1.0, zp = 0 → expected [-2, 1, 0, -1].
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let q_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I4),
        output_shape: shape,
    });
    graph.add_input(q_in);
    let dq = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([InputSource::Node(q_in)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I4,
            scale_bits: 1.0f32.to_bits(),
            zero_point: 0,
            axis: -1,
        },
    );
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let q_bytes: Vec<u8> = vec![0x1E, 0xF0];
    let outputs = session.execute(&[InputBuffer { bytes: &q_bytes }]).unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    assert!((result[0] - (-2.0)).abs() < 1e-6, "el0 got {}", result[0]);
    assert!((result[1] - 1.0).abs() < 1e-6, "el1 got {}", result[1]);
    assert!((result[2] - 0.0).abs() < 1e-6, "el2 got {}", result[2]);
    assert!((result[3] - (-1.0)).abs() < 1e-6, "el3 got {}", result[3]);
}

#[test]
fn dequant_matmul_fuses_and_matches_unfused() {
    // A[2,3] · dequant(Wq[3,2]) with a *dynamic* quantized weight (graph input).
    // The `dequantize → matmul` fusion fires, eliding the dense f32 weight; the
    // result equals dequantizing then multiplying separately.
    let mut graph = Graph::new();
    let a_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(2, 3));
    let w_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(3, 2));
    let o_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(2, 2));
    let a_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: a_sh,
    });
    graph.add_input(a_in);
    let wq = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I8),
        output_shape: w_sh,
    });
    graph.add_input(wq);
    let dq = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([InputSource::Node(wq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: w_sh,
    });
    graph.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            scale_bits: 0.5f32.to_bits(),
            zero_point: 0,
            axis: -1,
        },
    );
    let mm = graph.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a_in), InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: o_sh,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: o_sh,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    assert_eq!(
        session.dequant_fused_count(),
        1,
        "dequant→matmul must fuse to MatMulDequant"
    );

    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let wq_bytes: Vec<u8> = vec![1u8, 2, 3, 4, 5, 6]; // i8 = W·2 (scale 0.5)
    let outputs = session
        .execute(&[
            InputBuffer { bytes: &a_bytes },
            InputBuffer { bytes: &wq_bytes },
        ])
        .unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    // W = [[0.5,1],[1.5,2],[2.5,3]] ; A·W = [[11,14],[24.5,32]].
    let want = [11.0f32, 14.0, 24.5, 32.0];
    for (g, w) in result.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-5, "got {result:?} want {want:?}");
    }
}

#[test]
fn dequantize_int8_with_nonzero_zero_point() {
    // Asymmetric INT8: scale = 0.25, zp = 5 → y = (q − 5) · 0.25
    // q = [5, 9, 13, 1] → [0.0, 1.0, 2.0, -1.0]
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let q_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I8),
        output_shape: shape,
    });
    graph.add_input(q_in);
    let dq = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([InputSource::Node(q_in)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            scale_bits: 0.25f32.to_bits(),
            zero_point: 5,
            axis: -1,
        },
    );
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let q_bytes: Vec<u8> = vec![5, 9, 13, 1];
    let outputs = session.execute(&[InputBuffer { bytes: &q_bytes }]).unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    assert!((result[0] - 0.0).abs() < 1e-6, "got {:?}", result);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    assert!((result[3] - (-1.0)).abs() < 1e-6);
}
