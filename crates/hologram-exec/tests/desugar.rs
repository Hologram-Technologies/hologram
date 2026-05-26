//! **Path B — composite-op desugaring, end-to-end V&V.**
//!
//! A composite op carries no kernel of its own; the compiler desugars it into
//! its primitive-op pipeline (`Clip(x,lo,hi) = Min(Max(x,lo),hi)`), which runs
//! on the already-verified primitive kernels. This test compiles a Clip graph,
//! executes it through the content-addressed session, and checks the result
//! against an independent `clamp` reference — proving the pipeline lowering is
//! numerically correct end-to-end (not the old silent identity / fail-loud).

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, compile_from_source, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    constant::ConstantEntry,
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, LrnAttrs, NodeId, OpKind,
};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

fn f32_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn source_grammar_clip_end_to_end() {
    // The text frontend expresses shapes + constant operands: a Clip with its
    // bounds as `const` tensors, compiled from source and executed. Proves the
    // grammar drives a real UOR-native op end-to-end (Clip → Min∘Max).
    let src = "\
input x :4
const lo :4 = -0.5,-0.5,-0.5,-0.5
const hi :4 = 0.5,0.5,0.5,0.5
op clip x lo hi :4 as=y
output y
";
    let x = [-1.0f32, 0.0, 0.3, 1.0];
    let want = [-0.5f32, 0.0, 0.3, 0.5];
    let compiled = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );
    for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
        assert!(
            (gv - wv).abs() < 1e-6,
            "source clip[{i}]: got {gv}, want {wv}"
        );
    }
}

#[test]
fn clip_desugars_and_clamps_end_to_end() {
    let n = 16usize;
    // Spread across the bounds so clamping is observable on both sides.
    let x: Vec<f32> = (0..n)
        .map(|i| (i as f32) / (n as f32) * 2.0 - 1.0)
        .collect();
    let (lo_val, hi_val) = (-0.3f32, 0.4f32);
    let want: Vec<f32> = x.iter().map(|&v| v.clamp(lo_val, hi_val)).collect();

    // Clip's min/max are full-size bound tensors (the primitive Min/Max kernels
    // are elementwise — broadcast of a scalar bound is a separate concern).
    let lo = vec![lo_val; n];
    let hi = vec![hi_val; n];

    let mut g = Graph::new();
    let s = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));
    let lo_c = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&lo),
        dtype: DTypeId(DTYPE_F32),
        shape: s,
    });
    let hi_c = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&hi),
        dtype: DTypeId(DTYPE_F32),
        shape: s,
    });
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_input(xi);
    let clip = g.add_node(Node {
        op: GraphOp::Op(OpKind::Clip),
        inputs: SmallVec::from_iter([
            InputSource::Node(xi),
            InputSource::Constant(lo_c),
            InputSource::Constant(hi_c),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(clip)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );

    assert_eq!(got.len(), n, "output length");
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-6,
            "clip[{i}]: got {g}, want {w} (x={}, [{lo_val},{hi_val}])",
            x[i]
        );
    }
    // Clamping actually happened (some elements moved) — not a passthrough.
    assert!(
        got.iter().zip(&x).any(|(&g, &xv)| (g - xv).abs() > 1e-6),
        "no element was clamped — desugaring degenerated to identity"
    );
}

#[test]
fn reshape_is_zero_movement_readdressing() {
    // Input → Reshape → Relu → Output. Reshape is byte-identity (a row-major
    // relabel), so the executor binds its output slot to the input's buffer
    // with NO dispatch and NO copy (counted in last_skipped); Relu then reads
    // the same bytes. This is the UOR-native addressing realization — reshape
    // is re-addressing, not a memcpy.
    let n = 8usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) - 4.0).collect();
    let want: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();

    let mut g = Graph::new();
    let s1 = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));
    let s2 = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(2, (n / 2) as u64));
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s1,
    });
    g.add_input(xi);
    let rs = g.add_node(Node {
        op: GraphOp::Op(OpKind::Reshape),
        inputs: SmallVec::from_iter([InputSource::Node(xi)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s2,
    });
    let relu = g.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(rs)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s2,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(relu)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s2,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );

    for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
        assert!(
            (gv - wv).abs() < 1e-6,
            "reshape→relu[{i}]: got {gv}, want {wv}"
        );
    }
    // The interior Reshape was elided (bound to the input buffer), not dispatched.
    assert!(
        sess.last_skipped() >= 1,
        "reshape was not zero-movement — it dispatched a copy (last_skipped={})",
        sess.last_skipped()
    );
}

#[test]
fn slice_is_zero_movement_projectfield() {
    // data[4,3] → Slice rows [1,3) (axis-0) → Relu → output. Slice is a
    // ProjectField view: the executor binds the output to the input's
    // sub-region with no dispatch/copy (last_skipped), and Relu reads it.
    let data: Vec<f32> = (0..12).map(|i| i as f32 - 6.0).collect();
    // rows 1,2 = elements 3..9, then relu.
    let want: Vec<f32> = data[3..9].iter().map(|&v| v.max(0.0)).collect();

    let mut g = Graph::new();
    let s_data = g.shape_registry_mut().intern(ShapeDescriptor::rank2(4, 3));
    let s_out = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 3));
    let s_idx = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
    let starts = g.constants_mut().insert(ConstantEntry {
        bytes: 1i64.to_le_bytes().to_vec(),
        dtype: DTypeId(5), // I64
        shape: s_idx,
    });
    let ends = g.constants_mut().insert(ConstantEntry {
        bytes: 3i64.to_le_bytes().to_vec(),
        dtype: DTypeId(5),
        shape: s_idx,
    });
    let di = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_data,
    });
    g.add_input(di);
    let sl = g.add_node(Node {
        op: GraphOp::Op(OpKind::Slice),
        inputs: SmallVec::from_iter([
            InputSource::Node(di),
            InputSource::Constant(starts),
            InputSource::Constant(ends),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    let relu = g.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(sl)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(relu)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&data),
            }])
            .unwrap()[0]
            .bytes,
    );
    assert_eq!(got.len(), 6, "slice output length");
    for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
        assert!(
            (gv - wv).abs() < 1e-6,
            "slice→relu[{i}]: got {gv}, want {wv}"
        );
    }
    assert!(
        sess.last_skipped() >= 1,
        "slice was not zero-movement (last_skipped={})",
        sess.last_skipped()
    );
}

#[test]
fn resize_nearest_upsamples() {
    // x[1,1,2,2] → [1,1,4,4], 2× nearest-neighbor: each pixel replicated 2×2.
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let want = [
        1.0f32, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];

    let mut g = Graph::new();
    let s_in = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(1, 1, 2, 2));
    let s_out = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(1, 1, 4, 4));
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_in,
    });
    g.add_input(xi);
    let rz = g.add_node(Node {
        op: GraphOp::Op(OpKind::Resize),
        inputs: SmallVec::from_iter([InputSource::Node(xi)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(rz)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );
    assert_eq!(got, want, "resize nearest did not upsample correctly");
}

#[test]
fn lrn_normalizes_over_channel_window() {
    // x[1,3,1] (3 channels), size=3, α=1, β=1, bias=0. Window [c−1,c+1]:
    // out[c] = x[c] / ((1/3)·Σ_window x²).
    let x = [1.0f32, 2.0, 3.0];
    let want = [
        1.0 / (5.0 / 3.0),  // c0: window {1,4}
        2.0 / (14.0 / 3.0), // c1: window {1,4,9}
        3.0 / (13.0 / 3.0), // c2: window {4,9}
    ];

    let mut g = Graph::new();
    let s = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(1, 3, 1, 1));
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_input(xi);
    let lrn = g.add_node(Node {
        op: GraphOp::Op(OpKind::Lrn),
        inputs: SmallVec::from_iter([InputSource::Node(xi)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.set_lrn_attrs(
        NodeId(lrn.0),
        LrnAttrs {
            size: 3,
            alpha_bits: 1.0f32.to_bits(),
            beta_bits: 1.0f32.to_bits(),
            bias_bits: 0.0f32.to_bits(),
        },
    );
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(lrn)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );
    for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
        assert!((gv - wv).abs() < 1e-5, "lrn[{i}]: got {gv}, want {wv}");
    }
}

#[test]
fn rope_rotates_halves() {
    // x[2,4] (head_dim=4, half=2). With cos=1, sin=0.5 the rotate-half form
    // gives, per row: out[0]=x0−x2·.5, out[1]=x1−x3·.5, out[2]=x2+x0·.5,
    // out[3]=x3+x1·.5.
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let cos = [1.0f32; 8];
    let sin = [0.5f32; 8];
    let want = [-0.5f32, 0.0, 3.5, 5.0, 1.5, 2.0, 9.5, 11.0];

    let mut g = Graph::new();
    let s = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 4));
    let cos_c = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&cos),
        dtype: DTypeId(DTYPE_F32),
        shape: s,
    });
    let sin_c = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&sin),
        dtype: DTypeId(DTYPE_F32),
        shape: s,
    });
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_input(xi);
    let rope = g.add_node(Node {
        op: GraphOp::Op(OpKind::RotaryEmbedding),
        inputs: SmallVec::from_iter([
            InputSource::Node(xi),
            InputSource::Constant(cos_c),
            InputSource::Constant(sin_c),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(rope)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );
    for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
        assert!((gv - wv).abs() < 1e-6, "rope[{i}]: got {gv}, want {wv}");
    }
}

#[test]
fn expand_broadcasts() {
    // x[1,3] expand → [4,3]: the size-1 axis 0 broadcasts (each row = x).
    let x = [10.0f32, 20.0, 30.0];
    let want = [
        10.0f32, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0,
    ];

    let mut g = Graph::new();
    let s_in = g.shape_registry_mut().intern(ShapeDescriptor::rank2(1, 3));
    let s_out = g.shape_registry_mut().intern(ShapeDescriptor::rank2(4, 3));
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_in,
    });
    g.add_input(xi);
    let ex = g.add_node(Node {
        op: GraphOp::Op(OpKind::Expand),
        inputs: SmallVec::from_iter([InputSource::Node(xi)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(ex)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );
    assert_eq!(got, want, "expand did not broadcast");
}

#[test]
fn transpose_permutes_axes() {
    // x[2,3] transposed (default reverse perm) → [3,2]: out[j,i] = x[i,j].
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // rows [1,2,3],[4,5,6]
    let want = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]; // [[1,4],[2,5],[3,6]]

    let mut g = Graph::new();
    let s_in = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 3));
    let s_out = g.shape_registry_mut().intern(ShapeDescriptor::rank2(3, 2));
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_in,
    });
    g.add_input(xi);
    // Single-input Transpose → default axis reversal.
    let tr = g.add_node(Node {
        op: GraphOp::Op(OpKind::Transpose),
        inputs: SmallVec::from_iter([InputSource::Node(xi)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(tr)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );
    assert_eq!(got, want, "transpose did not permute axes");
}

#[test]
fn pad_places_data_in_zeroed_buffer() {
    // data[2,3] padded axis-0 by (1 before, 1 after) → [4,3]: a zero row, the
    // data, a zero row. Pad = placement into a zeroed output at offset lo.
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let want = [
        0.0f32, 0.0, 0.0, // pad-lo row
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // data
        0.0, 0.0, 0.0, // pad-hi row
    ];
    // ONNX pads (rank-2): [begin0, begin1, end0, end1] = [1,0,1,0] as i64.
    let mut pads_bytes = Vec::new();
    for v in [1i64, 0, 1, 0] {
        pads_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let mut g = Graph::new();
    let s_in = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 3));
    let s_out = g.shape_registry_mut().intern(ShapeDescriptor::rank2(4, 3));
    let s_pads = g.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let pads = g.constants_mut().insert(ConstantEntry {
        bytes: pads_bytes,
        dtype: DTypeId(5), // I64
        shape: s_pads,
    });
    let di = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_in,
    });
    g.add_input(di);
    let pad = g.add_node(Node {
        op: GraphOp::Op(OpKind::Pad),
        inputs: SmallVec::from_iter([InputSource::Node(di), InputSource::Constant(pads)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(pad)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_out,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&data),
            }])
            .unwrap()[0]
            .bytes,
    );
    assert_eq!(got, want, "pad did not place data in a zeroed buffer");
}

#[test]
fn concat_primitive_places_a_then_b() {
    // Concat is the closed PrimitiveOp::Concat constructor: out = a ∥ b.
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0];
    let want = [1.0f32, 2.0, 3.0, 4.0, 5.0];

    let mut g = Graph::new();
    let sa = g.shape_registry_mut().intern(ShapeDescriptor::rank1(3));
    let sb = g.shape_registry_mut().intern(ShapeDescriptor::rank1(2));
    let so = g.shape_registry_mut().intern(ShapeDescriptor::rank1(5));
    let ai = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sa,
    });
    g.add_input(ai);
    let bi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sb,
    });
    g.add_input(bi);
    let cc = g.add_node(Node {
        op: GraphOp::Op(OpKind::Concat),
        inputs: SmallVec::from_iter([InputSource::Node(ai), InputSource::Node(bi)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(cc)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[
                InputBuffer {
                    bytes: &f32_to_le(&a),
                },
                InputBuffer {
                    bytes: &f32_to_le(&b),
                },
            ])
            .unwrap()[0]
            .bytes,
    );
    assert_eq!(got, want, "concat did not place a ∥ b");
}

fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            o[i * n + j] = acc as f32;
        }
    }
    o
}

#[test]
fn swiglu_desugars_and_computes_end_to_end() {
    // SwiGLU(x, W_gate, W_up) = Silu(x·W_gate) ⊙ (x·W_up). Desugars to
    // MatMul, Silu, MatMul, Mul — all existing verified kernels.
    let (m, k, n) = (2usize, 3usize, 4usize);
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let wg: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 - 0.2).collect();
    let wu: Vec<f32> = (0..k * n).map(|i| 0.3 - (i as f32) * 0.04).collect();

    // Reference: silu(x·Wg) ⊙ (x·Wu).
    let g = ref_matmul(&x, &wg, m, k, n);
    let u = ref_matmul(&x, &wu, m, k, n);
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&gv, &uv)| {
            let silu = gv / (1.0 + (-gv).exp());
            silu * uv
        })
        .collect();

    let mut gr = Graph::new();
    let sx = gr
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, k as u64));
    let sw = gr
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let so = gr
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, n as u64));
    let wg_c = gr.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&wg),
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    let wu_c = gr.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&wu),
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    let xi = gr.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    gr.add_input(xi);
    let sg = gr.add_node(Node {
        op: GraphOp::Op(OpKind::FusedSwiGlu),
        inputs: SmallVec::from_iter([
            InputSource::Node(xi),
            InputSource::Constant(wg_c),
            InputSource::Constant(wu_c),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let out = gr.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(sg)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    gr.add_output(out);

    let compiled = compile(gr, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let got = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes,
    );

    assert_eq!(got.len(), m * n);
    let scale = want.iter().fold(0f64, |mx, &v| mx.max(f64::from(v).abs())) + 1e-9;
    let err = got
        .iter()
        .zip(&want)
        .map(|(&gv, &wv)| (f64::from(gv) - f64::from(wv)).abs() / scale)
        .fold(0f64, f64::max);
    assert!(
        err <= 1e-4,
        "SwiGLU diverged from reference (err {err:.3e})"
    );
    assert!(
        got.iter().any(|&v| v.abs() > 1e-6),
        "SwiGLU output all-zero — desugaring degenerated"
    );
}
