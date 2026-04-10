//! Structural analysis (fusion-detection) pass benchmarks on graphs of varying sizes.
//!
//! Renamed from "fusion benchmarks" under the v0.2.0 conformance-first
//! refactor: the passes are now framed as finders, not optimizations. The
//! benchmark file name is preserved for historical continuity, but the
//! function under benchmark is `analysis::analyze`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_core::op::LutOp;
use hologram_ir::analysis;
use hologram_ir::builder::GraphBuilder;
use hologram_ir::graph::GraphOp;

fn build_linear_graph(size: usize) -> hologram_ir::Graph {
    let ops = [
        LutOp::Relu,
        LutOp::Sigmoid,
        LutOp::Tanh,
        LutOp::Sin,
        LutOp::Cos,
    ];
    let mut builder = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0); // 0

    for i in 0..size {
        let op = ops[i % ops.len()];
        builder = builder.node_with_inputs(GraphOp::Lut(op), &[i]);
    }
    let last = size;
    builder = builder
        .node_with_inputs(GraphOp::Output, &[last])
        .output("y", last + 1);
    builder.build()
}

fn bench_analyze_10(c: &mut Criterion) {
    c.bench_function("analysis::analyze(10_nodes)", |b| {
        b.iter_batched(
            || build_linear_graph(10),
            |mut g| analysis::analyze(black_box(&mut g)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_analyze_100(c: &mut Criterion) {
    c.bench_function("analysis::analyze(100_nodes)", |b| {
        b.iter_batched(
            || build_linear_graph(100),
            |mut g| analysis::analyze(black_box(&mut g)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_analyze_1000(c: &mut Criterion) {
    c.bench_function("analysis::analyze(1000_nodes)", |b| {
        b.iter_batched(
            || build_linear_graph(1000),
            |mut g| analysis::analyze(black_box(&mut g)),
            criterion::BatchSize::LargeInput,
        )
    });
}

criterion_group!(
    benches,
    bench_analyze_10,
    bench_analyze_100,
    bench_analyze_1000,
);
criterion_main!(benches);
