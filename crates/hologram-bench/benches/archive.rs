//! HoloWriter::build + load_from_bytes round-trip benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_archive::{load_from_bytes, HoloWriter};
use hologram_core::op::{LutOp, PrimOp};
use hologram_ir::builder::GraphBuilder;
use hologram_ir::graph::GraphOp;

fn build_graph(size: usize) -> hologram_ir::Graph {
    // Build a linear chain of `size` unary ops
    let mut builder = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0); // 0

    let ops = [
        LutOp::Relu,
        LutOp::Sigmoid,
        LutOp::Tanh,
        LutOp::Sin,
        LutOp::Cos,
    ];
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

fn bench_archive_write_small(c: &mut Criterion) {
    let g = build_graph(5);
    c.bench_function("archive::write(5_nodes)", |b| {
        b.iter(|| HoloWriter::new().set_graph(black_box(&g)).build())
    });
}

fn bench_archive_write_medium(c: &mut Criterion) {
    let g = build_graph(50);
    c.bench_function("archive::write(50_nodes)", |b| {
        b.iter(|| HoloWriter::new().set_graph(black_box(&g)).build())
    });
}

fn bench_archive_roundtrip_small(c: &mut Criterion) {
    let g = build_graph(5);
    let _archive = HoloWriter::new().set_graph(&g).build().unwrap();
    c.bench_function("archive::roundtrip(5_nodes)", |b| {
        b.iter(|| {
            let bytes = HoloWriter::new().set_graph(black_box(&g)).build().unwrap();
            load_from_bytes(black_box(&bytes)).unwrap()
        })
    });
}

fn bench_archive_load(c: &mut Criterion) {
    let g = build_graph(50);
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    c.bench_function("archive::load(50_nodes)", |b| {
        b.iter(|| load_from_bytes(black_box(&archive)))
    });
}

fn bench_archive_diamond(c: &mut Criterion) {
    // Diamond graph: more complex topology
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 2
        .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[0]) // 3
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2]) // 4
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[4, 3]) // 5
        .node_with_inputs(GraphOp::Output, &[5]) // 6
        .output("y", 6)
        .build();
    c.bench_function("archive::roundtrip_diamond(7_nodes)", |b| {
        b.iter(|| {
            let bytes = HoloWriter::new().set_graph(black_box(&g)).build().unwrap();
            load_from_bytes(black_box(&bytes)).unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_archive_write_small,
    bench_archive_write_medium,
    bench_archive_roundtrip_small,
    bench_archive_load,
    bench_archive_diamond,
);
criterion_main!(benches);
