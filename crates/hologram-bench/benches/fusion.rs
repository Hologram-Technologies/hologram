//! Fusion pass benchmarks on graphs of varying sizes.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_core::op::LutOp;
use hologram_graph::builder::GraphBuilder;
use hologram_graph::fusion;
use hologram_graph::graph::GraphOp;

fn build_linear_graph(size: usize) -> hologram_graph::Graph {
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

fn bench_fuse_10(c: &mut Criterion) {
    c.bench_function("fusion::fuse(10_nodes)", |b| {
        b.iter_batched(
            || build_linear_graph(10),
            |mut g| fusion::fuse(black_box(&mut g)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_fuse_100(c: &mut Criterion) {
    c.bench_function("fusion::fuse(100_nodes)", |b| {
        b.iter_batched(
            || build_linear_graph(100),
            |mut g| fusion::fuse(black_box(&mut g)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_fuse_1000(c: &mut Criterion) {
    c.bench_function("fusion::fuse(1000_nodes)", |b| {
        b.iter_batched(
            || build_linear_graph(1000),
            |mut g| fusion::fuse(black_box(&mut g)),
            criterion::BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_fuse_10, bench_fuse_100, bench_fuse_1000,);
criterion_main!(benches);
