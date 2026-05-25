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
