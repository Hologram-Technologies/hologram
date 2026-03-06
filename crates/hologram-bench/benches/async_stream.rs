//! Benchmarks: streaming execution throughput vs batch.

use criterion::{criterion_group, criterion_main, Criterion};
use hologram_archive::writer::holo_writer::HoloWriter;
use hologram_async::execute_stream;
use hologram_core::op::LutOp;
use hologram_exec::{execute_bytes, GraphInputs};
use hologram_graph::builder::GraphBuilder;
use hologram_graph::graph::GraphOp;

fn build_chain_archive(depth: usize) -> Vec<u8> {
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
    let g = b.output("y", depth + 1).build();
    HoloWriter::new().set_graph(&g).build().unwrap()
}

fn batch_execute(c: &mut Criterion) {
    let archive = build_chain_archive(20);
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);
    c.bench_function("batch_execute_20nodes", |b| {
        b.iter(|| execute_bytes(&archive, &inputs).unwrap());
    });
}

fn stream_execute(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let archive = build_chain_archive(20);
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);
    c.bench_function("stream_execute_20nodes", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (mut rx, handle) = execute_stream(archive.clone(), inputs.clone());
                while rx.recv().await.is_some() {}
                handle.await.unwrap().unwrap()
            })
        });
    });
}

criterion_group!(benches, batch_execute, stream_execute);
criterion_main!(benches);
