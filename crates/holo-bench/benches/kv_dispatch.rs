//! KvStore dispatch benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holo_core::op::{LutOp, PrimOp};
use holo_exec::KvStore;
use holo_graph::graph::GraphOp;

fn bench_dispatch_unary_lut(c: &mut Criterion) {
    let op = GraphOp::Lut(LutOp::Sigmoid);
    let input = vec![128u8; 256];
    c.bench_function("kv::dispatch_unary_lut(256B)", |b| {
        b.iter(|| KvStore::dispatch(black_box(&op), &[black_box(&input)]))
    });
}

fn bench_dispatch_unary_4k(c: &mut Criterion) {
    let op = GraphOp::Lut(LutOp::Relu);
    let input = vec![100u8; 4096];
    c.bench_function("kv::dispatch_unary_lut(4KB)", |b| {
        b.iter(|| KvStore::dispatch(black_box(&op), &[black_box(&input)]))
    });
}

fn bench_dispatch_unary_64k(c: &mut Criterion) {
    let op = GraphOp::Lut(LutOp::Tanh);
    let input = vec![64u8; 65536];
    c.bench_function("kv::dispatch_unary_lut(64KB)", |b| {
        b.iter(|| KvStore::dispatch(black_box(&op), &[black_box(&input)]))
    });
}

fn bench_dispatch_binary_add(c: &mut Criterion) {
    let op = GraphOp::Prim(PrimOp::Add);
    let lhs = vec![100u8; 256];
    let rhs = vec![50u8; 256];
    c.bench_function("kv::dispatch_binary_add(256B)", |b| {
        b.iter(|| KvStore::dispatch(black_box(&op), &[black_box(&lhs), black_box(&rhs)]))
    });
}

fn bench_dispatch_binary_mul(c: &mut Criterion) {
    let op = GraphOp::Prim(PrimOp::Mul);
    let lhs = vec![13u8; 4096];
    let rhs = vec![17u8; 4096];
    c.bench_function("kv::dispatch_binary_mul(4KB)", |b| {
        b.iter(|| KvStore::dispatch(black_box(&op), &[black_box(&lhs), black_box(&rhs)]))
    });
}

fn bench_dispatch_all_lut_ops(c: &mut Criterion) {
    let input = vec![128u8; 256];
    let mut group = c.benchmark_group("kv_dispatch_lut_ops");
    for op in &LutOp::ALL {
        let graph_op = GraphOp::Lut(*op);
        group.bench_function(op.name(), |b| {
            b.iter(|| KvStore::dispatch(black_box(&graph_op), &[black_box(&input)]))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dispatch_unary_lut,
    bench_dispatch_unary_4k,
    bench_dispatch_unary_64k,
    bench_dispatch_binary_add,
    bench_dispatch_binary_mul,
    bench_dispatch_all_lut_ops,
);
criterion_main!(benches);
