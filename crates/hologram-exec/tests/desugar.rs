//! **Path B — composite-op desugaring, end-to-end V&V.**
//!
//! A composite op carries no kernel of its own; the compiler desugars it into
//! its primitive-op pipeline (`Clip(x,lo,hi) = Min(Max(x,lo),hi)`), which runs
//! on the already-verified primitive kernels. This test compiles a Clip graph,
//! executes it through the content-addressed session, and checks the result
//! against an independent `clamp` reference — proving the pipeline lowering is
//! numerically correct end-to-end (not the old silent identity / fail-loud).

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    constant::ConstantEntry,
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind,
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
fn clip_desugars_and_clamps_end_to_end() {
    let n = 16usize;
    // Spread across the bounds so clamping is observable on both sides.
    let x: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32) * 2.0 - 1.0).collect();
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
        assert!((gv - wv).abs() < 1e-6, "reshape→relu[{i}]: got {gv}, want {wv}");
    }
    // The interior Reshape was elided (bound to the input buffer), not dispatched.
    assert!(
        sess.last_skipped() >= 1,
        "reshape was not zero-movement — it dispatched a copy (last_skipped={})",
        sess.last_skipped()
    );
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
    assert!(err <= 1e-4, "SwiGLU diverged from reference (err {err:.3e})");
    assert!(
        got.iter().any(|&v| v.abs() > 1e-6),
        "SwiGLU output all-zero — desugaring degenerated"
    );
}
