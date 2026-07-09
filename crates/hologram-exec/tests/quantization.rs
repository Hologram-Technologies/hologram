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
use hologram_graph::constant::ConstantEntry;
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind, QuantAttrs};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DTYPE_I8: u8 = 2;
const DTYPE_I4: u8 = 10;
const DTYPE_E8CB: u8 = 11;

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
            ..Default::default()
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
fn dequantize_uint8_round_trip() {
    // ONNX's default asymmetric uint8: y = (q − zp) · scale, q unsigned 0..=255.
    // scale = 0.5, zp = 128 over q = [128, 130, 126, 200] → [0, 1, -1, 36].
    const DTYPE_U8: u8 = 1;
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let q_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_U8),
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
            quant_dtype: DTYPE_U8,
            scale_bits: 0.5f32.to_bits(),
            zero_point: 128,
            axis: -1,
            ..Default::default()
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
    // Raw unsigned bytes 128, 130, 126, 200 (no two's-complement reinterpretation).
    let q_bytes: Vec<u8> = vec![128, 130, 126, 200];
    let outputs = session.execute(&[InputBuffer { bytes: &q_bytes }]).unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    assert!((result[0] - 0.0).abs() < 1e-6, "got {result:?}");
    assert!((result[1] - 1.0).abs() < 1e-6, "got {result:?}");
    assert!((result[2] - (-1.0)).abs() < 1e-6, "got {result:?}");
    assert!((result[3] - 36.0).abs() < 1e-6, "got {result:?}");
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
fn dequant_e8cb_matmul_fuses_omajor_and_matches_reference() {
    // E8-codebook (DTYPE_E8CB) decode projection: A[1,k] · dequant(indices[k,n]).
    // The per-channel constant index weight (m=1) triggers the omajor W8A8
    // fusion to `matmul_e8cb_omajor`. This exercises the compiler's
    // `[k/8,n] → [n,k/8]` index transpose AND the per-model **codebook operand**
    // (the Dequantize node's 4th input) end-to-end, against a scalar restatement
    // of the spec (i8 activation quant → index→codebook LUT → exact i32 dot).
    //
    // The codebook is the model's own data: this test declares one, and the
    // reference below decodes against the same bytes. Nothing about it is baked
    // into the engine.
    const CB_ENTRIES: usize = 256;
    const CB_GROUP: usize = 8;
    let codebook: Vec<i8> = (0..CB_ENTRIES * CB_GROUP)
        .map(|i| (((i * 37 + 11) % 255) as i32 - 127) as i8)
        .collect();
    // Spread of (k, n): n<4 scalar tail, the exact 4-col body, a ragged tail,
    // and many groups/columns — exercising the compiler's `[k/8,n] → [n,k/8]`
    // index transpose end-to-end across shapes.
    for &(k, n) in &[(16usize, 3usize), (64, 8), (128, 17), (256, 32)] {
        let g = k / 8;
        let mut graph = Graph::new();
        let a_sh = graph
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(1, k as u64));
        let w_sh = graph
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(k as u64, n as u64));
        let v_sh = graph
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank1(n as u64));
        let o_sh = graph
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(1, n as u64));

        let a_in = graph.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: a_sh,
        });
        graph.add_input(a_in);

        // Index weight: [k/8, n] row-major (element gk*n + j).
        let idx: Vec<u8> = (0..g * n).map(|i| ((i * 53 + 7) % 256) as u8).collect();
        let scales: Vec<f32> = (0..n).map(|j| 0.03 + j as f32 * 0.01).collect();
        let wc = graph.constants_mut().insert(ConstantEntry {
            bytes: idx.clone(),
            dtype: DTypeId(DTYPE_E8CB),
            shape: w_sh,
        });
        let sc = graph.constants_mut().insert(ConstantEntry {
            bytes: scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(DTYPE_F32),
            shape: v_sh,
        });
        let zc = graph.constants_mut().insert(ConstantEntry {
            bytes: vec![0u8; n * 4],
            dtype: DTypeId(DTYPE_I8),
            shape: v_sh,
        });
        let cb_sh = graph
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(CB_ENTRIES as u64, CB_GROUP as u64));
        let cbc = graph.constants_mut().insert(ConstantEntry {
            bytes: codebook.iter().map(|&v| v as u8).collect(),
            dtype: DTypeId(DTYPE_I8),
            shape: cb_sh,
        });
        let dq = graph.add_node(Node {
            op: GraphOp::Op(OpKind::Dequantize),
            inputs: SmallVec::from_iter([
                InputSource::Constant(wc),
                InputSource::Constant(sc),
                InputSource::Constant(zc),
                // 4th input: the model's codebook.
                InputSource::Constant(cbc),
            ]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: w_sh,
        });
        graph.set_quant_attrs(
            dq,
            QuantAttrs {
                quant_dtype: DTYPE_E8CB,
                scale_bits: 0,
                zero_point: 0,
                axis: 1,
                // E8CB's public entry quantizes the activation internally — it is
                // W1A8, not W1A32 — so a graph using it opts into the activation
                // rounding like any other W8A8 weight. `weight_layout` stays
                // ROW_MAJOR: these are constant `[k/8, n]` index bytes and the
                // compiler transposes them itself when it fuses.
                act_quant: hologram_types::act_quant::W8A8_TOKEN_SYM,
                ..Default::default()
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
            "e8cb dequant→matmul must fuse to the omajor MatMulDequant"
        );

        let a: Vec<f32> = (0..k).map(|i| (i as f32 - 7.5) * 0.4).collect();
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result =
            le_to_f32(&session.execute(&[InputBuffer { bytes: &a_bytes }]).unwrap()[0].bytes);

        // Scalar reference: symmetric i8 activation quant (round half away), then
        // exact i32 dot of q against the codebook-looked-up weights.
        let amax = a.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let inv = 127.0 / amax;
        let scale_a = amax / 127.0;
        let q: Vec<i32> = a
            .iter()
            .map(|&v| {
                let t = v * inv;
                let r = if t >= 0.0 {
                    (t + 0.5) as i32
                } else {
                    (t - 0.5) as i32
                };
                r.clamp(-127, 127)
            })
            .collect();
        for j in 0..n {
            let mut s = 0i32;
            for gk in 0..g {
                let e = idx[gk * n + j] as usize * 8;
                for t in 0..8 {
                    s += q[gk * 8 + t] * codebook[e + t] as i32;
                }
            }
            let want = s as f32 * (scale_a * scales[j]);
            assert!(
                (result[j] - want).abs() < 1e-4,
                "k={k} n={n} col {j}: got {} want {want}",
                result[j]
            );
        }
    }
}

#[test]
fn dequant_gelu_fuses_to_densified_table() {
    // Dequantize(i8, per-tensor) → Gelu. The dequant output is an f32
    // intermediate whose realized domain is the i8 source (256 values), so the
    // composition densifies into one quantized-domain table — the PM_7 LUT win
    // applied to the f32 quantized-inference path. Must fuse and stay correct.
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
            ..Default::default()
        },
    );
    let act = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Gelu),
        inputs: SmallVec::from_iter([InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(act)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    assert_eq!(
        session.dequant_activation_fused_count(),
        1,
        "dequant→gelu must densify to one DequantActivation"
    );

    // q = [-2, 0, 2, 4] → dequant [-1, 0, 1, 2] → gelu(·).
    let q_bytes: Vec<u8> = vec![(-2i8) as u8, 0, 2, 4];
    let outputs = session.execute(&[InputBuffer { bytes: &q_bytes }]).unwrap();
    let result = le_to_f32(&outputs[0].bytes);
    let gelu = |x: f32| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
    for (g, x) in result.iter().zip([-1.0f32, 0.0, 1.0, 2.0]) {
        assert!((g - gelu(x)).abs() < 1e-5, "got {result:?}");
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
            ..Default::default()
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

/// Build `A[1,k] · dequant(Wq)` where the **weight is a graph input**, not a
/// constant — the weightless-compile shape: the graph carries no weight bytes,
/// they are bound at execute time. `declare` opts the weight slot into the
/// output-major W8A8 decode form.
///
/// Returns the executed output.
fn weightless_dequant_matmul(
    k: usize,
    n: usize,
    a: &[f32],
    w_bytes: &[u8],
    declare: bool,
) -> Vec<f32> {
    use hologram_types::{act_quant, weight_layout};
    let mut graph = Graph::new();
    let a_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, k as u64));
    let w_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let v_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));
    let o_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, n as u64));

    let a_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: a_sh,
    });
    graph.add_input(a_in);
    // The weight: a graph input, bound later. No bytes at compile time.
    let wq = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I8),
        output_shape: w_sh,
    });
    graph.add_input(wq);

    let scales: Vec<f32> = (0..n).map(|j| 0.02 + j as f32 * 0.003).collect();
    let sc = graph.constants_mut().insert(ConstantEntry {
        bytes: scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
        dtype: DTypeId(DTYPE_F32),
        shape: v_sh,
    });
    let zc = graph.constants_mut().insert(ConstantEntry {
        bytes: vec![0u8; n * 4], // symmetric
        dtype: DTypeId(DTYPE_I8),
        shape: v_sh,
    });
    let dq = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([
            InputSource::Node(wq),
            InputSource::Constant(sc),
            InputSource::Constant(zc),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: w_sh,
    });
    graph.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            axis: 1,
            // The weight-slot declaration under test.
            weight_layout: if declare {
                weight_layout::OUTPUT_MAJOR
            } else {
                weight_layout::ROW_MAJOR
            },
            act_quant: if declare {
                act_quant::W8A8_TOKEN_SYM
            } else {
                act_quant::W8A32
            },
            ..Default::default()
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
        "the load-time dequant→matmul fusion must fire for a weightless weight"
    );
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = session
        .execute(&[
            InputBuffer { bytes: &a_bytes },
            InputBuffer { bytes: w_bytes },
        ])
        .unwrap();
    le_to_f32(&outs[0].bytes)
}

/// The regression this test exists for: a weight bound **after** compile could
/// never reach the fused output-major W8A8 decode GEMV, because the only path to
/// it required the compiler to hold the weight's constant bytes and transpose
/// them. Every weightless model — which is every model that wants dedupable,
/// pageable archives — silently took the slow W8A32 `[k,n]` path.
///
/// A weight slot now *declares* `{layout: output-major, act: W8A8}` and the
/// load-time fusion builds the same call the constant-weight path builds. The
/// declared result must equal the exact-integer W8A8 oracle.
#[test]
fn weightless_weight_can_declare_output_major_w8a8_and_reaches_the_integer_gemv() {
    let (k, n) = (16usize, 4usize);
    let a: Vec<f32> = (0..k).map(|i| (i as f32 - 7.5) * 0.31).collect();
    // Row-major logical weight w[kk][j], and its output-major image.
    let w: Vec<i8> = (0..k * n).map(|i| ((i * 37) % 197) as i32 as i8).collect();
    let mut w_omajor = vec![0i8; k * n];
    for kk in 0..k {
        for j in 0..n {
            w_omajor[j * k + kk] = w[kk * n + j];
        }
    }
    let w_om_bytes: Vec<u8> = w_omajor.iter().map(|&v| v as u8).collect();
    let got = weightless_dequant_matmul(k, n, &a, &w_om_bytes, true);

    // Exact-integer W8A8 oracle: per-token symmetric i8 activation quant, exact
    // i32 dot, one fused per-column writeback.
    let scales: Vec<f32> = (0..n).map(|j| 0.02 + j as f32 * 0.003).collect();
    let amax = a.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let inv = 127.0 / amax;
    let scale_a = amax / 127.0;
    let q: Vec<i32> = a
        .iter()
        .map(|&v| {
            let t = v * inv;
            let r = if t >= 0.0 {
                (t + 0.5) as i32
            } else {
                (t - 0.5) as i32
            };
            r.clamp(-127, 127)
        })
        .collect();
    for j in 0..n {
        let mut s = 0i32;
        for (kk, &qv) in q.iter().enumerate() {
            s += qv * w[kk * n + j] as i32;
        }
        let want = s as f32 * (scale_a * scales[j]);
        assert!(
            (got[j] - want).abs() < 1e-4,
            "col {j}: got {} want {want} (declared W8A8 must hit the integer GEMV)",
            got[j]
        );
    }
}

/// W8A8 rounds the activation, so it changes the computed value. It must
/// therefore never be an invisible upgrade: an undeclared weight keeps the exact
/// `dequantize → matmul` (W8A32) semantics, bit-for-bit.
#[test]
fn w8a8_is_opt_in_undeclared_weights_keep_w8a32_semantics() {
    let (k, n) = (16usize, 4usize);
    let a: Vec<f32> = (0..k).map(|i| (i as f32 - 7.5) * 0.31).collect();
    let w: Vec<i8> = (0..k * n).map(|i| ((i * 37) % 197) as i32 as i8).collect();
    let w_bytes: Vec<u8> = w.iter().map(|&v| v as u8).collect();
    // Undeclared: row-major bytes, W8A32.
    let got = weightless_dequant_matmul(k, n, &a, &w_bytes, false);

    let scales: Vec<f32> = (0..n).map(|j| 0.02 + j as f32 * 0.003).collect();
    for j in 0..n {
        // Exact f32 reference: dequantize then multiply, no activation rounding.
        let mut s = 0f32;
        for (kk, &av) in a.iter().enumerate() {
            s += av * (w[kk * n + j] as f32 * scales[j]);
        }
        assert!(
            (got[j] - s).abs() < 1e-3,
            "col {j}: got {} want {s} — an undeclared weight must not be silently upgraded to W8A8",
            got[j]
        );
    }
}

/// A load-bound weight declaring `OUTPUT_MAJOR` promises its bytes arrive
/// `[n,k]`. If no output-major kernel can serve the call — here because the
/// scales are per-tensor, so there is no per-channel `scales[n]` to index — then
/// **there is no correct fallback**: the W8A32 dequant loop and the standalone
/// Dequantize kernel both read `[k,n]`, and taking either would transpose the
/// weight by accident and return a plausible, wrong answer.
///
/// Every precondition is static, so the compiler refuses. This is the "no
/// silent-wrong" property: a declaration the substrate cannot honour is an
/// error, not a quiet downgrade. (`hologram-exec`'s loader re-checks the same
/// predicate for archives this compiler did not produce.)
#[test]
fn declared_output_major_weight_that_no_kernel_can_serve_fails_loud() {
    use hologram_types::{act_quant, weight_layout};

    let (k, n) = (16usize, 4usize);
    let mut graph = Graph::new();
    let a_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, k as u64));
    let w_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let o_sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, n as u64));

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
            scale_bits: 0.05f32.to_bits(),
            zero_point: 0,
            axis: -1, // per-tensor: the output-major GEMV needs per-channel
            weight_layout: weight_layout::OUTPUT_MAJOR,
            act_quant: act_quant::W8A8_TOKEN_SYM,
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

    let err = compile(graph, BackendKind::Cpu, WittLevel::W32)
        .err()
        .expect("declared OUTPUT_MAJOR with per-tensor scales must not compile");
    let msg = format!("{err}");
    assert!(
        msg.contains("OUTPUT_MAJOR") && msg.contains("axis = 1"),
        "error must name the offending predicate, got: {msg}"
    );
}
