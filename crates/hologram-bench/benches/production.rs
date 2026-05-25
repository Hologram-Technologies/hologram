//! Production-workload benchmark — a stacked transformer-MLP.
//!
//! Microbenches (matmul.rs) bound one kernel; production inference is a
//! *graph* of mixed ops (matmul → gelu → matmul → residual, repeated per
//! layer — the FLOP-dominant block of a real transformer) run under
//! serving load. This bench measures the two regimes a deployment pays:
//!
//! * **cold** — a distinct input every iteration ⇒ full recompute (the
//!   worst-case production compute rate, the GFLOP/s the model sustains);
//! * **served** — a repeated request ⇒ whole-graph content-addressed memo
//!   hit (the cache/replay payoff: latency collapses to a label lookup).
//!
//! Swept across sizes because hologram targets arbitrary models/sizes.
//! Divide `16 · seq · d² · layers` (matmul FLOPs) by the reported time for
//! GFLOP/s; divide by core GHz for FLOP/core-cycle.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
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

/// `x_{l+1} = x_l + W2_l · gelu(W1_l · x_l)`, repeated `layers` times.
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
        let w1: Vec<u8> = (0..(d * hidden) as usize)
            .flat_map(|i| (((i + l) as f32).sin() * 0.02).to_le_bytes())
            .collect();
        let w1 = g.constants_mut().insert(ConstantEntry {
            bytes: w1,
            dtype: DTypeId(DTYPE_F32),
            shape: s_w1,
        });
        let w2: Vec<u8> = (0..(hidden * d) as usize)
            .flat_map(|i| (((i + l) as f32).cos() * 0.02).to_le_bytes())
            .collect();
        let w2 = g.constants_mut().insert(ConstantEntry {
            bytes: w2,
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

/// Cold path: distinct input each iteration ⇒ full recompute.
fn bench_cold(c: &mut Criterion, seq: u64, d: u64, layers: usize) {
    let mut sess = mlp_stack_session(seq, d, layers);
    let elems = (seq * d) as usize;
    let mut x: Vec<f32> = (0..elems).map(|i| (i as f32).sin() * 0.1).collect();
    let mut ctr: u32 = 0;
    let name = format!("production::mlp_cold_seq{seq}_d{d}_l{layers}");
    c.bench_function(&name, |bencher| {
        bencher.iter(|| {
            ctr = ctr.wrapping_add(1);
            x[0] = ctr as f32; // distinct ⇒ memo miss ⇒ recompute
            let bytes = f32_to_le(&x);
            let out = sess
                .execute(black_box(&[InputBuffer { bytes: &bytes }]))
                .unwrap();
            black_box(out);
        });
    });
}

/// Served path: repeated request ⇒ whole-graph memo hit.
fn bench_served(c: &mut Criterion, seq: u64, d: u64, layers: usize) {
    let mut sess = mlp_stack_session(seq, d, layers);
    let elems = (seq * d) as usize;
    let x = f32_to_le(
        &(0..elems)
            .map(|i| (i as f32).cos() * 0.1)
            .collect::<Vec<_>>(),
    );
    let inputs = [InputBuffer { bytes: &x }];
    let _ = sess.execute(&inputs).unwrap(); // prime the memo
    let name = format!("production::mlp_served_seq{seq}_d{d}_l{layers}");
    c.bench_function(&name, |bencher| {
        bencher.iter(|| {
            let out = sess.execute(black_box(&inputs)).unwrap();
            black_box(out);
        });
    });
}

fn benches(c: &mut Criterion) {
    // Two sizes (d=128, d=256) to expose throughput scaling; layers=2 to
    // keep the cold path's criterion sampling bounded.
    bench_cold(c, 64, 128, 2);
    bench_cold(c, 64, 256, 2);
    bench_served(c, 64, 256, 2);
}

criterion_group!(prod, benches);
criterion_main!(prod);
