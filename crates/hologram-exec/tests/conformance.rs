//! **Content-addressed execution conformance — scaling V&V (class SC-2).**
//!
//! Demonstrates the content-addressed runtime holds at arbitrary scale and
//! is not short-cutting or breaking down: at each size the executed output
//! matches an independent f64-reference matmul, and a re-execution
//! (graph-memo hit) is **byte-identical** to the first — proving the reuse
//! path returns the true result at scale, not a degenerate stand-in.

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
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
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

fn matmul_session(m: u64, k: u64, n: u64) -> InferenceSession<CpuBackend<BufferArena>> {
    let mut g = Graph::new();
    let sa = g.shape_registry_mut().intern(ShapeDescriptor::rank2(m, k));
    let sb = g.shape_registry_mut().intern(ShapeDescriptor::rank2(k, n));
    let so = g.shape_registry_mut().intern(ShapeDescriptor::rank2(m, n));
    let a = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sa,
    });
    g.add_input(a);
    let b = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sb,
    });
    g.add_input(b);
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    g.add_output(out);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap()
}

#[test]
fn sc2_content_addressed_matmul_conforms_and_reuses_across_scale() {
    for (idx, &(m, k, n)) in [
        (8usize, 8usize, 8usize),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ]
    .iter()
    .enumerate()
    {
        let a = fill(m * k, 0x51 + idx as u64);
        let b = fill(k * n, 0x73 + idx as u64);
        let mut session = matmul_session(m as u64, k as u64, n as u64);
        let inputs = [
            InputBuffer {
                bytes: &f32_to_le(&a),
            },
            InputBuffer {
                bytes: &f32_to_le(&b),
            },
        ];

        let first = session.execute(&inputs).unwrap();
        let got = le_to_f32(&first[0].bytes);
        let want = ref_matmul(&a, &b, m, k, n);

        // External correctness at this scale.
        let scale = want.iter().fold(0f64, |mx, &w| mx.max(f64::from(w).abs())) + 1e-9;
        let err = got
            .iter()
            .zip(&want)
            .map(|(&gv, &wv)| (f64::from(gv) - f64::from(wv)).abs() / scale)
            .fold(0f64, f64::max);
        assert!(
            err <= 1e-4,
            "{m}×{k}×{n}: content-addressed output diverged from reference (err {err:.3e})"
        );

        // Reuse at scale: graph-memo hit is byte-identical to the first run
        // (not a degenerate short-cut).
        let second = session.execute(&inputs).unwrap();
        assert_eq!(
            second[0].bytes, first[0].bytes,
            "{m}×{k}×{n}: memoized re-execution diverged from the first"
        );
    }
}

// ─── SC-4: matmul against a CONSTANT weight (the inference case) ──────
//
// Regression guard for a compiler bug the V&V exposed: `lower.rs` resolved
// operand shapes only for `InputSource::Node`, so a matmul whose weight is
// an `InputSource::Constant` inferred `m=k=n=0` and silently became a
// no-op (zeros). SC-2 missed it (two Input operands). This builds the real
// inference shape — activation · constant-weight — and checks the output
// against the f64 reference.
#[test]
fn sc4_matmul_against_constant_weight_conforms() {
    for (idx, &(m, k, n)) in [(2usize, 3usize, 4usize), (16, 32, 8), (64, 64, 64)]
        .iter()
        .enumerate()
    {
        let a = fill(m * k, 0xC0 + idx as u64);
        let w = fill(k * n, 0xD0 + idx as u64);

        let mut g = Graph::new();
        let sa = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(m as u64, k as u64));
        let sw = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(k as u64, n as u64));
        let so = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(m as u64, n as u64));
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: f32_to_le(&w),
            dtype: DTypeId(DTYPE_F32),
            shape: sw,
        });
        let ai = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: sa,
        });
        g.add_input(ai);
        let mm = g.add_node(Node {
            op: GraphOp::Op(OpKind::MatMul),
            inputs: SmallVec::from_iter([InputSource::Node(ai), InputSource::Constant(wc)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: so,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(mm)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: so,
        });
        g.add_output(out);
        let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
        let mut sess: InferenceSession<CpuBackend<BufferArena>> =
            InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

        let got = le_to_f32(
            &sess
                .execute(&[InputBuffer {
                    bytes: &f32_to_le(&a),
                }])
                .unwrap()[0]
                .bytes,
        );
        let want = ref_matmul(&a, &w, m, k, n);
        let scale = want.iter().fold(0f64, |mx, &x| mx.max(f64::from(x).abs())) + 1e-9;
        let err = got
            .iter()
            .zip(&want)
            .map(|(&gv, &wv)| (f64::from(gv) - f64::from(wv)).abs() / scale)
            .fold(0f64, f64::max);
        assert!(
            err <= 1e-4,
            "{m}×{k}×{n} weight-matmul diverged (err {err:.3e})"
        );
        assert!(
            got.iter().any(|&v| v.abs() > 1e-6),
            "{m}×{k}×{n} weight-matmul output is all-zero (no-op regression)"
        );
    }
}

// ─── SC-5: conv2d against a CONSTANT weight (the other dim-from-weight op) ─
//
// Conv infers channels_out / k_h / k_w from its weight operand, so it had
// the *same* constant-operand no-op bug as matmul. This exercises the
// compiler path with a constant conv weight and checks vs the f64 ref.
#[allow(clippy::too_many_arguments)]
fn ref_conv(
    x: &[f32],
    w: &[f32],
    b: usize,
    cin: usize,
    cout: usize,
    hi: usize,
    wi: usize,
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    let (ho, wo) = (hi - kh + 1, wi - kw + 1);
    let mut o = vec![0f32; b * cout * ho * wo];
    for bi in 0..b {
        for co in 0..cout {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc = 0f64;
                    for ci in 0..cin {
                        for y in 0..kh {
                            for xk in 0..kw {
                                acc += f64::from(x[((bi * cin + ci) * hi + oh + y) * wi + ow + xk])
                                    * f64::from(w[((co * cin + ci) * kh + y) * kw + xk]);
                            }
                        }
                    }
                    o[((bi * cout + co) * ho + oh) * wo + ow] = acc as f32;
                }
            }
        }
    }
    o
}

#[test]
fn sc5_conv2d_against_constant_weight_conforms() {
    let (b, cin, cout, hi, wi, kh, kw) = (2usize, 3usize, 4usize, 12usize, 12usize, 3usize, 3usize);
    let (ho, wo) = (hi - kh + 1, wi - kw + 1);
    let x = fill(b * cin * hi * wi, 0xE0);
    let w = fill(cout * cin * kh * kw, 0xE1);

    let mut g = Graph::new();
    let sx = g.shape_registry_mut().intern(ShapeDescriptor::rank4(
        b as u64, cin as u64, hi as u64, wi as u64,
    ));
    let sw = g.shape_registry_mut().intern(ShapeDescriptor::rank4(
        cout as u64,
        cin as u64,
        kh as u64,
        kw as u64,
    ));
    let so = g.shape_registry_mut().intern(ShapeDescriptor::rank4(
        b as u64,
        cout as u64,
        ho as u64,
        wo as u64,
    ));
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&w),
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    g.add_input(xi);
    let cv = g.add_node(Node {
        op: GraphOp::Op(OpKind::Conv2d),
        inputs: SmallVec::from_iter([InputSource::Node(xi), InputSource::Constant(wc)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(cv)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
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
    let want = ref_conv(&x, &w, b, cin, cout, hi, wi, kh, kw);
    let scale = want.iter().fold(0f64, |mx, &v| mx.max(f64::from(v).abs())) + 1e-9;
    let err = got
        .iter()
        .zip(&want)
        .map(|(&gv, &wv)| (f64::from(gv) - f64::from(wv)).abs() / scale)
        .fold(0f64, f64::max);
    assert!(
        err <= 1e-4,
        "conv-vs-constant-weight diverged (err {err:.3e})"
    );
    assert!(
        got.iter().any(|&v| v.abs() > 1e-6),
        "conv-vs-constant-weight output all-zero (no-op regression)"
    );
}

// ─── SG-1: sub-graph content addressing (the high-leverage reuse path) ───
//
// Whole-graph memoization only fires when the *entire* input set repeats.
// Sub-graph addressing addresses every node by the witnessed composition
// of its operands' κ-labels, so a sub-graph whose operands are unchanged
// is recognized and its compute elided even when the top-level input set
// differs — the prefix/KV-cache case. This builds a two-branch graph
//
//     p = matmul(a, b)      q = matmul(c, d)      out = p + q
//
// runs it once (all three nodes dispatch), then re-runs with only `d`
// changed. The whole-graph memo *misses* (input set differs), but the
// `a·b` branch is unchanged: its node label matches the first run, so its
// matmul is elided (skipped), while `c·d'` and the add recompute. We
// assert the elision happened (`last_skipped == 1`, the heavy matmul) and
// that the output still equals the independent f64 reference p + q'.
fn ref_add(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a + b).collect()
}

#[test]
fn sg1_subgraph_reuse_elides_unchanged_branch_across_inputs() {
    let n = 64usize;
    let mut g = Graph::new();
    let s = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(n as u64, n as u64));
    let mk_input = |g: &mut Graph| {
        let id = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: s,
        });
        g.add_input(id);
        id
    };
    let a = mk_input(&mut g);
    let b = mk_input(&mut g);
    let c = mk_input(&mut g);
    let d = mk_input(&mut g);
    let p = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let q = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(c), InputSource::Node(d)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let sum = g.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([InputSource::Node(p), InputSource::Node(q)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(sum)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_output(out);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

    let av = fill(n * n, 0x1);
    let bv = fill(n * n, 0x2);
    let cv = fill(n * n, 0x3);
    let dv = fill(n * n, 0x4);
    let dv2 = fill(n * n, 0x5);

    let run = |sess: &mut InferenceSession<CpuBackend<BufferArena>>, d: &[u8]| {
        sess.execute(&[
            InputBuffer {
                bytes: &f32_to_le(&av),
            },
            InputBuffer {
                bytes: &f32_to_le(&bv),
            },
            InputBuffer {
                bytes: &f32_to_le(&cv),
            },
            InputBuffer { bytes: d },
        ])
        .unwrap()[0]
            .bytes
            .clone()
    };

    // First run: nothing resident, every node is novel.
    let _ = run(&mut sess, &f32_to_le(&dv));
    assert_eq!(
        sess.last_dispatched(),
        3,
        "first run must compute all nodes"
    );
    assert_eq!(sess.last_skipped(), 0);

    // Second run: only `d` changes. Whole-graph memo misses, but the a·b
    // branch is unchanged → its matmul is elided; c·d' and the add run.
    let got = le_to_f32(&run(&mut sess, &f32_to_le(&dv2)));
    assert_eq!(
        sess.last_skipped(),
        1,
        "the unchanged a·b matmul must be elided by sub-graph addressing"
    );
    assert_eq!(
        sess.last_dispatched(),
        2,
        "only the c·d' matmul and the add should recompute"
    );

    // Correctness: the reused-prefix result still equals the f64 reference.
    let want = ref_add(
        &ref_matmul(&av, &bv, n, n, n),
        &ref_matmul(&cv, &dv2, n, n, n),
    );
    let scale = want.iter().fold(0f64, |mx, &w| mx.max(f64::from(w).abs())) + 1e-9;
    let err = got
        .iter()
        .zip(&want)
        .map(|(&gv, &wv)| (f64::from(gv) - f64::from(wv)).abs() / scale)
        .fold(0f64, f64::max);
    assert!(
        err <= 1e-4,
        "sub-graph-reused output diverged (err {err:.3e})"
    );
}

// ─── SG-2: common-subexpression elision within a single execution ───────
//
// Sub-graph addressing also eliminates *intra-graph* redundancy: if the
// same computation appears twice in one graph (same op, same operand
// labels), the second occurrence hits the store the first produced — so
// the redundant compute is elided within a single `execute`. Graph:
//
//     p = matmul(a, b)      q = matmul(a, b)      out = p + q
//
// `p` and `q` are the identical computation, so exactly one matmul runs;
// the second is recognized by label. Output must equal 2·(a·b).
#[test]
fn sg2_common_subexpression_elided_within_one_execution() {
    let n = 48usize;
    let mut g = Graph::new();
    let s = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(n as u64, n as u64));
    let a = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_input(a);
    let b = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_input(b);
    let p = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let q = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let sum = g.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([InputSource::Node(p), InputSource::Node(q)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(sum)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_output(out);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

    let av = fill(n * n, 0x9);
    let bv = fill(n * n, 0xA);
    let got = le_to_f32(
        &sess
            .execute(&[
                InputBuffer {
                    bytes: &f32_to_le(&av),
                },
                InputBuffer {
                    bytes: &f32_to_le(&bv),
                },
            ])
            .unwrap()[0]
            .bytes,
    );

    // Exactly one of the two identical matmuls runs; the other is elided.
    assert_eq!(
        sess.last_skipped(),
        1,
        "the duplicate matmul must be elided as a common subexpression"
    );

    let ab = ref_matmul(&av, &bv, n, n, n);
    let want: Vec<f32> = ab.iter().map(|&v| v + v).collect();
    let scale = want.iter().fold(0f64, |mx, &w| mx.max(f64::from(w).abs())) + 1e-9;
    let err = got
        .iter()
        .zip(&want)
        .map(|(&gv, &wv)| (f64::from(gv) - f64::from(wv)).abs() / scale)
        .fold(0f64, f64::max);
    assert!(err <= 1e-4, "CSE-elided output diverged (err {err:.3e})");
}

// ─── FU: content-addressed fusion (the UOR-native execution pass) ────────
//
// `matmul → elementwise-activation` is collapsed into one fused op whose
// activation runs in the matmul epilogue, so the activation's intermediate
// is never materialized or addressed — the fused node carries a single
// κ-derivation. FU-1: the fused result matches the independent f64
// reference AND the intermediate is elided (one kernel, not two). FU-2:
// fusion is semantics-preserving (byte-identical to the unfused result)
// and is *guarded* — a matmul whose output has another observer is not
// fused.

/// `matmul(input[m,k], const w[k,n]) → activation → output`. Fuses to one op.
fn matmul_act_session(
    m: usize,
    k: usize,
    n: usize,
    act: OpKind,
    w: &[f32],
) -> InferenceSession<CpuBackend<BufferArena>> {
    let mut g = Graph::new();
    let sa = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, k as u64));
    let sw = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let so = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, n as u64));
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(w),
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    let ai = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sa,
    });
    g.add_input(ai);
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(ai), InputSource::Constant(wc)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let a = g.add_node(Node {
        op: GraphOp::Op(act),
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(a)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    g.add_output(out);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap()
}

fn act_ref(act: OpKind, x: f32) -> f32 {
    let xd = f64::from(x);
    let v = match act {
        OpKind::Relu => xd.max(0.0),
        OpKind::Gelu => 0.5 * xd * (1.0 + (0.797_884_6 * (xd + 0.044_715 * xd * xd * xd)).tanh()),
        OpKind::Silu => xd / (1.0 + (-xd).exp()),
        _ => xd,
    };
    v as f32
}

#[test]
fn fu1_fused_matmul_activation_conforms_and_elides_intermediate() {
    for act in [OpKind::Relu, OpKind::Gelu, OpKind::Silu] {
        for (idx, &(m, k, n)) in [(4usize, 8usize, 6usize), (16, 32, 24), (64, 64, 64)]
            .iter()
            .enumerate()
        {
            let a = fill(m * k, 0x10 + idx as u64);
            let w = fill(k * n, 0x20 + idx as u64);
            let mut sess = matmul_act_session(m, k, n, act, &w);

            // The intermediate is elided: matmul+activation became ONE op.
            assert_eq!(
                sess.fused_count(),
                1,
                "matmul→{act:?} must fuse to one content-addressed op"
            );
            assert_eq!(
                sess.kernel_count(),
                1,
                "fused op is the only kernel — the activation intermediate is elided"
            );

            let got = le_to_f32(
                &sess
                    .execute(&[InputBuffer {
                        bytes: &f32_to_le(&a),
                    }])
                    .unwrap()[0]
                    .bytes,
            );
            let mm = ref_matmul(&a, &w, m, k, n);
            let want: Vec<f32> = mm.iter().map(|&v| act_ref(act, v)).collect();
            let scale = want.iter().fold(0f64, |mx, &x| mx.max(f64::from(x).abs())) + 1e-9;
            let err = got
                .iter()
                .zip(&want)
                .map(|(&gv, &wv)| (f64::from(gv) - f64::from(wv)).abs() / scale)
                .fold(0f64, f64::max);
            assert!(
                err <= 1e-3,
                "{act:?} {m}×{k}×{n}: fused matmul+activation diverged (err {err:.3e})"
            );
        }
    }
}

#[test]
fn fu2_fusion_is_semantics_preserving_and_guarded() {
    let (m, k, n) = (32usize, 48usize, 24usize);
    let a = fill(m * k, 0x77);
    let w = fill(k * n, 0x88);

    // A: matmul → gelu → output. Fuses.
    let mut fused = matmul_act_session(m, k, n, OpKind::Gelu, &w);
    assert_eq!(fused.fused_count(), 1, "A must fuse");

    // B: matmul → {gelu→out0, relu→out1}. The matmul output now has two
    // observers, so the fusion guard must suppress fusion.
    let mut g = Graph::new();
    let sa = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, k as u64));
    let sw = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let so = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, n as u64));
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&w),
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    let ai = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sa,
    });
    g.add_input(ai);
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(ai), InputSource::Constant(wc)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let gnode = g.add_node(Node {
        op: GraphOp::Op(OpKind::Gelu),
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let rnode = g.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let o0 = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(gnode)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    g.add_output(o0);
    let o1 = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(rnode)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    g.add_output(o1);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut unfused: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    assert_eq!(
        unfused.fused_count(),
        0,
        "matmul with two observers must NOT fuse (the intermediate is needed)"
    );

    // Fusion is semantics-preserving: the fused gelu output is byte-identical
    // to the unfused gelu output (same matmul, same activation).
    let xa = f32_to_le(&a);
    let fa = fused.execute(&[InputBuffer { bytes: &xa }]).unwrap()[0]
        .bytes
        .clone();
    let ub = unfused.execute(&[InputBuffer { bytes: &xa }]).unwrap()[0]
        .bytes
        .clone();
    assert_eq!(
        fa, ub,
        "fused matmul+gelu must be byte-identical to the unfused computation"
    );
}
