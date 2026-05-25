//! **Performance V&V (class PV) — production-load throughput / latency.**
//!
//! Microbenchmarks (PV-1/3) bound a single kernel. Production inference is
//! a *graph* of mixed ops run under *sustained load*, so PV-2/PV-4 measure
//! the surface a server actually pays:
//!
//! * **PV-2** — content-addressed reuse is not a bottleneck: a whole-graph
//!   memo hit must be ≥ 8× cheaper than recompute (a same-machine ratio,
//!   machine-independent).
//! * **PV-4** — a production-representative workload (a stacked
//!   transformer-MLP: per layer `matmul → gelu → matmul → residual`, the
//!   FLOP-dominant block of real models) sustains a throughput floor and
//!   does **not break down across sizes** (arbitrary models/sizes), and
//!   under serving load with reuse its effective latency collapses. We
//!   report GFLOP/s, FLOP/core-cycle, and per-inference latency.
//!
//! Release-only (`cargo test --release`); compiles to zero tests in debug.
#![cfg(not(debug_assertions))]

use std::time::Instant;

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
const DIM: u64 = 128;

fn f32_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Best-effort core clock (GHz) for the FLOP/core-cycle report. Reads the
/// first `cpu MHz` from `/proc/cpuinfo`; falls back to a nominal 3.0 GHz.
/// Used only for the *printed* efficiency figure, never for an assertion
/// (so the V&V stays machine-independent).
fn core_ghz() -> f64 {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("cpu MHz"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().parse::<f64>().ok())
        })
        .map(|mhz| mhz / 1000.0)
        .unwrap_or(3.0)
}

/// Build a stacked transformer-MLP inference graph — the FLOP-dominant
/// block of a production transformer, repeated `layers` times:
///
/// ```text
///   x_{l+1} = x_l + W2_l · gelu(W1_l · x_l)
/// ```
///
/// `x` is `[seq, d]`; each layer expands to `hidden = 4·d` and back, so the
/// op mix is the production one: matmul (dominant), gelu (activation), add
/// (residual). `W1`/`W2` are constant weights (addressed once at load).
/// Parameterized by `(seq, d, layers)` so it scales to arbitrary sizes.
fn mlp_stack_session(seq: u64, d: u64, layers: usize) -> InferenceSession<CpuBackend<BufferArena>> {
    let hidden = 4 * d;
    let mut g = Graph::new();
    let s_io = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(seq, d));
    let s_w1 = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(d, hidden));
    let s_hidden = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(seq, hidden));
    let s_w2 = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(hidden, d));

    let x0 = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_io,
    });
    g.add_input(x0);

    let mut x = x0;
    for l in 0..layers {
        let w1_bytes: Vec<u8> = (0..(d * hidden) as usize)
            .flat_map(|i| (((i + l) as f32).sin() * 0.02).to_le_bytes())
            .collect();
        let w1 = g.constants_mut().insert(ConstantEntry {
            bytes: w1_bytes,
            dtype: DTypeId(DTYPE_F32),
            shape: s_w1,
        });
        let w2_bytes: Vec<u8> = (0..(hidden * d) as usize)
            .flat_map(|i| (((i + l) as f32).cos() * 0.02).to_le_bytes())
            .collect();
        let w2 = g.constants_mut().insert(ConstantEntry {
            bytes: w2_bytes,
            dtype: DTypeId(DTYPE_F32),
            shape: s_w2,
        });
        let up = g.add_node(Node {
            op: GraphOp::Op(OpKind::MatMul),
            inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(w1)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: s_hidden,
        });
        let act = g.add_node(Node {
            op: GraphOp::Op(OpKind::Gelu),
            inputs: SmallVec::from_iter([InputSource::Node(up)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: s_hidden,
        });
        let down = g.add_node(Node {
            op: GraphOp::Op(OpKind::MatMul),
            inputs: SmallVec::from_iter([InputSource::Node(act), InputSource::Constant(w2)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: s_io,
        });
        x = g.add_node(Node {
            op: GraphOp::Op(OpKind::Add),
            inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Node(down)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: s_io,
        });
    }
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s_io,
    });
    g.add_output(out);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap()
}

/// Matmul FLOPs of one MLP-stack inference (the dominant, well-defined
/// compute): per layer `2·seq·d·hidden` (up) + `2·seq·hidden·d` (down),
/// `hidden = 4d` ⇒ `16·seq·d²` per layer.
fn mlp_stack_flops(seq: u64, d: u64, layers: usize) -> f64 {
    16.0 * seq as f64 * (d as f64) * (d as f64) * layers as f64
}

/// Best-of-N cold (all-novel) per-inference seconds + the GFLOP/s it
/// implies, driving distinct inputs so every request is a full recompute
/// (the worst-case production compute rate).
fn mlp_cold_gflops(seq: u64, d: u64, layers: usize, runs: u32) -> (f64, f64) {
    let mut sess = mlp_stack_session(seq, d, layers);
    let elems = (seq * d) as usize;
    let base: Vec<f32> = (0..elems).map(|i| (i as f32).sin() * 0.1).collect();
    // Warm (compile caches, page-in) with one distinct input.
    {
        let mut x = base.clone();
        x[0] = -1.0;
        let _ = sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap();
    }
    let mut best = f64::INFINITY;
    for it in 0..runs {
        let mut x = base.clone();
        x[0] = it as f32 + 1.0; // distinct ⇒ graph-memo miss ⇒ full recompute
        let bytes = f32_to_le(&x);
        let t = Instant::now();
        let _ = sess.execute(&[InputBuffer { bytes: &bytes }]).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    let gflops = mlp_stack_flops(seq, d, layers) / best / 1e9;
    (best, gflops)
}

/// A depth-4 128³ matmul chain against constant weights — enough compute
/// that a genuine recompute dwarfs a memo lookup.
fn chain_session() -> InferenceSession<CpuBackend<BufferArena>> {
    let mut g = Graph::new();
    let shape = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(DIM, DIM));
    let x = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    g.add_input(x);
    let mut acc = x;
    for layer in 0..4u32 {
        let w_bytes: Vec<u8> = (0..(DIM * DIM) as usize)
            .flat_map(|i| ((i as u32 + layer) as f32 * 0.001).to_le_bytes())
            .collect();
        let w = g.constants_mut().insert(ConstantEntry {
            bytes: w_bytes,
            dtype: DTypeId(DTYPE_F32),
            shape,
        });
        acc = g.add_node(Node {
            op: GraphOp::Op(OpKind::MatMul),
            inputs: SmallVec::from_iter([InputSource::Node(acc), InputSource::Constant(w)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: shape,
        });
    }
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(acc)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    g.add_output(out);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap()
}

#[test]
fn pv2_content_addressed_reuse_beats_recompute() {
    let mut session = chain_session();
    let elems = (DIM * DIM) as usize;
    let base: Vec<f32> = (0..elems).map(|i| i as f32 * 1e-3).collect();

    // Recompute path: a novel input each iteration ⇒ graph-memo miss ⇒
    // the full 4-matmul chain runs. Best-of-N (min) for stability.
    let mut recompute = f64::INFINITY;
    for it in 0..5u32 {
        let mut x = base.clone();
        x[0] = it as f32 + 1.0; // force a distinct input ⇒ miss
        let inputs = [InputBuffer {
            bytes: &f32_to_le(&x),
        }];
        let t = Instant::now();
        let _ = session.execute(&inputs).unwrap();
        recompute = recompute.min(t.elapsed().as_secs_f64());
    }

    // Reuse path: the same input ⇒ graph-memo hit ⇒ no recompute.
    let fixed = f32_to_le(&base);
    let inputs = [InputBuffer { bytes: &fixed }];
    let _ = session.execute(&inputs).unwrap(); // prime
    let mut reuse = f64::INFINITY;
    for _ in 0..50 {
        let t = Instant::now();
        let _ = session.execute(&inputs).unwrap();
        reuse = reuse.min(t.elapsed().as_secs_f64());
    }

    let speedup = recompute / reuse;
    assert!(
        speedup >= 8.0,
        "content-addressed reuse only {speedup:.1}× faster than recompute \
         (recompute {recompute:.2e}s, reuse {reuse:.2e}s) — the reuse path is a bottleneck \
         or is secretly recomputing"
    );
}

/// **PV-4 — production-workload throughput, latency, and scaling.**
///
/// A stacked transformer-MLP (the production op mix + structure) run under
/// sustained load. Three things a serving deployment cares about, each an
/// assertion plus a printed figure:
///
/// 1. **Throughput floor** — the cold (all-novel) compute rate clears a
///    conservative GFLOP/s floor, so no part of the production graph is a
///    silent bottleneck. We also print FLOP/core-cycle (the efficiency the
///    user is maximizing).
/// 2. **No breakdown across sizes** — hologram targets *arbitrary* models
///    and sizes, so throughput at a larger `d` must not collapse relative
///    to a smaller one (catching a path that degrades super-linearly /
///    falls off the SIMD fast path / thrashes as the working set grows).
/// 3. **Reuse collapses latency under load** — a served request that
///    repeats (cache / replay) resolves by whole-graph memo ≥ 8× faster
///    than a cold recompute: the content-addressing payoff under load.
#[test]
fn pv4_production_mlp_throughput_latency_and_scaling() {
    let ghz = core_ghz();
    let (seq, layers) = (64u64, 4usize);

    // The production workload exercises content-addressed fusion: each
    // layer's `up = matmul; gelu(up)` collapses to one fused op, so the
    // 4·d-wide activation intermediate is never materialized or addressed.
    let probe = mlp_stack_session(seq, 256, layers);
    assert_eq!(
        probe.fused_count(),
        layers,
        "production MLP-stack must fuse one matmul→gelu per layer"
    );
    eprintln!(
        "PV-4 fusion: {} matmul→gelu ops fused (one per layer); {} total kernels",
        probe.fused_count(),
        probe.kernel_count()
    );

    // Two production-ish sizes; B has 4× the per-layer FLOPs of A.
    let (lat_a, g_a) = mlp_cold_gflops(seq, 128, layers, 12);
    let (lat_b, g_b) = mlp_cold_gflops(seq, 256, layers, 12);

    eprintln!(
        "PV-4 production MLP-stack (seq={seq}, layers={layers}), cold/all-novel:\n  \
         d=128: {lat_a_ms:.3} ms/infer, {g_a:.2} GFLOP/s, {fca:.2} FLOP/core-cycle @ {ghz:.2} GHz\n  \
         d=256: {lat_b_ms:.3} ms/infer, {g_b:.2} GFLOP/s, {fcb:.2} FLOP/core-cycle @ {ghz:.2} GHz",
        lat_a_ms = lat_a * 1e3,
        lat_b_ms = lat_b * 1e3,
        fca = g_a / ghz,
        fcb = g_b / ghz,
    );

    // (1) Throughput floor. A working vectorized graph does tens of
    // GFLOP/s; 2.0 only fails on a catastrophic bottleneck.
    assert!(
        g_a >= 2.0 && g_b >= 2.0,
        "production MLP-stack below throughput floor (d=128 {g_a:.2}, d=256 {g_b:.2} GFLOP/s) — a part is a bottleneck"
    );

    // (2) No breakdown across sizes: the larger model must keep at least a
    // quarter of the smaller's per-FLOP throughput (arbitrary sizes hold;
    // catches super-linear degradation / cache-thrash collapse).
    assert!(
        g_b >= 0.25 * g_a,
        "throughput collapsed from {g_a:.2} GFLOP/s (d=128) to {g_b:.2} GFLOP/s (d=256) — \
         the production path breaks down as size grows"
    );

    // (3) Reuse collapses latency under serving load: a repeated request
    // hits the whole-graph memo. Best-of-N min for stability.
    let mut sess = mlp_stack_session(seq, 256, layers);
    let elems = (seq * 256) as usize;
    let fixed = f32_to_le(
        &(0..elems)
            .map(|i| (i as f32).cos() * 0.1)
            .collect::<Vec<_>>(),
    );
    let _ = sess.execute(&[InputBuffer { bytes: &fixed }]).unwrap(); // prime
    let mut reuse = f64::INFINITY;
    for _ in 0..200 {
        let t = Instant::now();
        let _ = sess.execute(&[InputBuffer { bytes: &fixed }]).unwrap();
        reuse = reuse.min(t.elapsed().as_secs_f64());
    }
    let served_speedup = lat_b / reuse;
    eprintln!(
        "  reuse (served-request memo hit): {reuse_us:.3} µs/infer ⇒ {served_speedup:.0}× vs cold",
        reuse_us = reuse * 1e6,
    );
    assert!(
        served_speedup >= 8.0,
        "served-request reuse only {served_speedup:.1}× faster than cold recompute — \
         the content-addressing payoff under load has regressed"
    );
}
