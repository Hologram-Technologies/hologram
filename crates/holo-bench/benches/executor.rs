//! KvExecutor benchmarks: execute graphs of varying topology.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holo_archive::HoloWriter;
use holo_core::op::{LutOp, PrimOp};
use holo_exec::{build_schedule, execute_bytes, GraphInputs};
use holo_graph::builder::GraphBuilder;
use holo_graph::fusion;
use holo_graph::graph::GraphOp;

fn make_archive(g: &mut holo_graph::Graph) -> Vec<u8> {
    let _ = fusion::fuse(g);
    HoloWriter::new().set_graph(g).build().unwrap()
}

fn bench_executor_linear(c: &mut Criterion) {
    // Linear chain: Input → Relu → Sigmoid → Tanh → Output
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[1])
        .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();
    let archive = make_archive(&mut g);

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    c.bench_function("exec::linear_chain(4_nodes, 256B)", |b| {
        b.iter(|| execute_bytes(black_box(&archive), black_box(&inputs)))
    });
}

fn bench_executor_diamond(c: &mut Criterion) {
    // Diamond: Input → (Relu, Sigmoid) → Add → Output
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();
    let archive = make_archive(&mut g);

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    c.bench_function("exec::diamond(5_nodes, 256B)", |b| {
        b.iter(|| execute_bytes(black_box(&archive), black_box(&inputs)))
    });
}

fn bench_executor_wide_parallel(c: &mut Criterion) {
    // Wide: Input → 8x(LutOp) → sum all → Output
    let ops = [
        LutOp::Relu,
        LutOp::Sigmoid,
        LutOp::Tanh,
        LutOp::Sin,
        LutOp::Cos,
        LutOp::Abs,
        LutOp::Sqrt,
        LutOp::Gelu,
    ];
    let mut builder = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0); // 0

    // Add 8 parallel LUT nodes (indices 1..=8)
    for op in &ops {
        builder = builder.node_with_inputs(GraphOp::Lut(*op), &[0]);
    }

    // Pairwise add: (1+2), (3+4), (5+6), (7+8)
    builder = builder
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2]) // 9
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[3, 4]) // 10
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[5, 6]) // 11
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[7, 8]) // 12
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[9, 10]) // 13
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[11, 12]) // 14
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[13, 14]) // 15
        .node_with_inputs(GraphOp::Output, &[15]) // 16
        .output("y", 16);

    let mut g = builder.build();
    let archive = make_archive(&mut g);

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    c.bench_function("exec::wide_parallel(17_nodes, 256B)", |b| {
        b.iter(|| execute_bytes(black_box(&archive), black_box(&inputs)))
    });
}

fn bench_executor_large_buffer(c: &mut Criterion) {
    // Simple chain but with 64KB buffer
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();
    let archive = make_archive(&mut g);

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 65536]);

    c.bench_function("exec::linear(3_nodes, 64KB)", |b| {
        b.iter(|| execute_bytes(black_box(&archive), black_box(&inputs)))
    });
}

fn bench_schedule_build(c: &mut Criterion) {
    // Benchmark just the schedule building step
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();
    let _ = fusion::fuse(&mut g);
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = holo_archive::load_from_bytes(&archive).unwrap();

    c.bench_function("exec::build_schedule(5_nodes)", |b| {
        b.iter(|| build_schedule(black_box(plan.graph())))
    });
}

criterion_group!(
    benches,
    bench_executor_linear,
    bench_executor_diamond,
    bench_executor_wide_parallel,
    bench_executor_large_buffer,
    bench_schedule_build,
);
criterion_main!(benches);
