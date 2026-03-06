//! Benchmarks: async compile + execute vs sync, measuring overhead.

use criterion::{criterion_group, criterion_main, Criterion};
use holo_archive::writer::holo_writer::HoloWriter;
use holo_async::{AsyncCompiler, AsyncExecutor};
use holo_compiler::CompilerBuilder;
use holo_core::op::LutOp;
use holo_exec::{execute_bytes, GraphInputs};
use holo_graph::builder::GraphBuilder;
use holo_graph::graph::GraphOp;

fn build_chain_graph(depth: usize) -> holo_graph::Graph {
    let mut b = GraphBuilder::new().input("x");
    b = b.node_from_graph_input(GraphOp::Input, 0);
    for i in 0..depth {
        let op = if i % 2 == 0 {
            GraphOp::Lut(LutOp::Relu)
        } else {
            GraphOp::Lut(LutOp::Sigmoid)
        };
        b = b.node_with_inputs(op, &[i]);
    }
    b = b.node_with_inputs(GraphOp::Output, &[depth]);
    b.output("y", depth + 1).build()
}

fn sync_compile(c: &mut Criterion) {
    let g = build_chain_graph(10);
    c.bench_function("sync_compile_10nodes", |b| {
        b.iter(|| CompilerBuilder::new(g.clone()).fuse(true).build().unwrap());
    });
}

fn async_compile(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let g = build_chain_graph(10);
    c.bench_function("async_compile_10nodes", |b| {
        b.iter(|| {
            rt.block_on(async {
                AsyncCompiler::new(g.clone())
                    .compile()
                    .await
                    .unwrap()
                    .unwrap()
            })
        });
    });
}

fn sync_execute(c: &mut Criterion) {
    let archive = HoloWriter::new()
        .set_graph(&build_chain_graph(10))
        .build()
        .unwrap();
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 64]);
    c.bench_function("sync_execute_10nodes", |b| {
        b.iter(|| execute_bytes(&archive, &inputs).unwrap());
    });
}

fn async_execute(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let archive = HoloWriter::new()
        .set_graph(&build_chain_graph(10))
        .build()
        .unwrap();
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 64]);
    c.bench_function("async_execute_10nodes", |b| {
        b.iter(|| {
            rt.block_on(async {
                AsyncExecutor::execute(archive.clone(), inputs.clone())
                    .await
                    .unwrap()
                    .unwrap()
            })
        });
    });
}

criterion_group!(
    benches,
    sync_compile,
    async_compile,
    sync_execute,
    async_execute
);
criterion_main!(benches);
