//! Compiler pipeline benchmarks at varying graph sizes.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holo_compiler::compile;
use holo_compiler::liveness::compute_liveness;
use holo_compiler::workspace::plan_workspace;
use holo_core::op::LutOp;
use holo_graph::builder::GraphBuilder;
use holo_graph::graph::GraphOp;
use holo_graph::schedule::ExecutionSchedule;

fn build_chain(size: usize) -> holo_graph::Graph {
    let ops = [LutOp::Relu, LutOp::Sigmoid, LutOp::Tanh];
    let mut b = GraphBuilder::new().node(GraphOp::Input);
    for i in 0..size {
        b = b.node_with_inputs(GraphOp::Lut(ops[i % ops.len()]), &[i]);
    }
    b.node_with_inputs(GraphOp::Output, &[size]).build()
}

fn bench_compile_10(c: &mut Criterion) {
    c.bench_function("compile/10_nodes", |b| {
        b.iter_batched(
            || build_chain(10),
            |g| compile(black_box(g)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_compile_50(c: &mut Criterion) {
    c.bench_function("compile/50_nodes", |b| {
        b.iter_batched(
            || build_chain(50),
            |g| compile(black_box(g)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_compile_100(c: &mut Criterion) {
    c.bench_function("compile/100_nodes", |b| {
        b.iter_batched(
            || build_chain(100),
            |g| compile(black_box(g)),
            criterion::BatchSize::LargeInput,
        )
    });
}

fn bench_liveness_10(c: &mut Criterion) {
    c.bench_function("liveness/10_nodes", |b| {
        let g = build_chain(10);
        let sched = ExecutionSchedule::build(&g).unwrap();
        b.iter(|| compute_liveness(black_box(&sched), black_box(&g)))
    });
}

fn bench_liveness_50(c: &mut Criterion) {
    c.bench_function("liveness/50_nodes", |b| {
        let g = build_chain(50);
        let sched = ExecutionSchedule::build(&g).unwrap();
        b.iter(|| compute_liveness(black_box(&sched), black_box(&g)))
    });
}

fn bench_liveness_100(c: &mut Criterion) {
    c.bench_function("liveness/100_nodes", |b| {
        let g = build_chain(100);
        let sched = ExecutionSchedule::build(&g).unwrap();
        b.iter(|| compute_liveness(black_box(&sched), black_box(&g)))
    });
}

fn bench_workspace_10(c: &mut Criterion) {
    c.bench_function("workspace/10_intervals", |b| {
        let g = build_chain(10);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);
        b.iter(|| plan_workspace(black_box(&intervals)))
    });
}

fn bench_workspace_50(c: &mut Criterion) {
    c.bench_function("workspace/50_intervals", |b| {
        let g = build_chain(50);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);
        b.iter(|| plan_workspace(black_box(&intervals)))
    });
}

criterion_group!(
    benches,
    bench_compile_10,
    bench_compile_50,
    bench_compile_100,
    bench_liveness_10,
    bench_liveness_50,
    bench_liveness_100,
    bench_workspace_10,
    bench_workspace_50,
);
criterion_main!(benches);
