//! Tape executor benchmarks: execute graphs of varying topology.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_archive::HoloWriter;
use hologram_core::op::{FloatOp, LutOp, PrimOp};
use hologram_fused_component::mmap::{build_tape_from_plan, execute_tape};
use hologram_fused_component::{build_schedule, GraphInputs};
use hologram_ir::analysis;
use hologram_ir::builder::GraphBuilder;
use hologram_ir::graph::GraphOp;

fn make_tape_and_plan(
    g: &mut hologram_ir::Graph,
) -> (
    hologram_archive::LoadedPlan,
    hologram_fused_component::tape::EnumTape,
) {
    let _ = analysis::analyze(g);
    let archive = HoloWriter::new().set_graph(g).build().unwrap();
    let plan = hologram_archive::load_from_bytes(&archive).unwrap();
    let tape = build_tape_from_plan(&plan).unwrap();
    (plan, tape)
}

fn bench_executor_linear(c: &mut Criterion) {
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[1])
        .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();
    let (plan, tape) = make_tape_and_plan(&mut g);

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    c.bench_function("exec::linear_chain(4_nodes, 256B)", |b| {
        b.iter(|| execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs)))
    });
}

fn bench_executor_diamond(c: &mut Criterion) {
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();
    let (plan, tape) = make_tape_and_plan(&mut g);

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    c.bench_function("exec::diamond(5_nodes, 256B)", |b| {
        b.iter(|| execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs)))
    });
}

fn bench_executor_wide_parallel(c: &mut Criterion) {
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
        .node_from_graph_input(GraphOp::Input, 0);

    for op in &ops {
        builder = builder.node_with_inputs(GraphOp::Lut(*op), &[0]);
    }

    builder = builder
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[3, 4])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[5, 6])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[7, 8])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[9, 10])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[11, 12])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[13, 14])
        .node_with_inputs(GraphOp::Output, &[15])
        .output("y", 16);

    let mut g = builder.build();
    let (plan, tape) = make_tape_and_plan(&mut g);

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    c.bench_function("exec::wide_parallel(17_nodes, 256B)", |b| {
        b.iter(|| execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs)))
    });
}

fn bench_executor_large_buffer(c: &mut Criterion) {
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();
    let (plan, tape) = make_tape_and_plan(&mut g);

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 65536]);

    c.bench_function("exec::linear(3_nodes, 64KB)", |b| {
        b.iter(|| execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs)))
    });
}

fn bench_schedule_build(c: &mut Criterion) {
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();
    let _ = analysis::analyze(&mut g);
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = hologram_archive::load_from_bytes(&archive).unwrap();

    c.bench_function("exec::build_schedule(5_nodes)", |b| {
        b.iter(|| build_schedule(black_box(plan.graph())))
    });
}

fn bench_page_faults(c: &mut Criterion) {
    use std::io::Write;

    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();
    let weights = vec![0u8; 256 * 1024];
    let archive = HoloWriter::new()
        .set_graph(&g)
        .set_weights(weights)
        .build()
        .unwrap();

    let dir = std::env::temp_dir();
    let path = dir.join("hologram_bench_pagefault.holo");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&archive).unwrap();
    }

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    c.bench_function("exec::mmap_load_execute(256KB_weights)", |b| {
        b.iter(|| {
            let loader = hologram_archive::HoloLoader::open(&path).unwrap();
            let plan = loader.load().unwrap();
            let tape = build_tape_from_plan(&plan).unwrap();
            black_box(execute_tape(&tape, &plan, &inputs))
        })
    });

    std::fs::remove_file(&path).ok();
}

fn bench_enum_tape_linear(c: &mut Criterion) {
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[0])
        .node_with_inputs(GraphOp::Float(FloatOp::Sigmoid), &[1])
        .node_with_inputs(GraphOp::Float(FloatOp::Tanh), &[2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();
    let _ = analysis::analyze(&mut g);
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = hologram_archive::load_from_bytes(&archive).unwrap();
    let tape = build_tape_from_plan(&plan).unwrap();

    let input_f32: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32) * 0.1).to_le_bytes())
        .collect();
    let mut inputs = GraphInputs::new();
    inputs.set(0, input_f32);

    c.bench_function("tape::linear_chain(4_float_nodes, 256B)", |b| {
        b.iter(|| execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs)))
    });
}

fn bench_tape_relu_64kb(c: &mut Criterion) {
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();
    let _ = analysis::analyze(&mut g);
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = hologram_archive::load_from_bytes(&archive).unwrap();
    let tape = build_tape_from_plan(&plan).unwrap();

    let input_f32: Vec<u8> = (0..16384) // 64KB of f32
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let mut inputs = GraphInputs::new();
    inputs.set(0, input_f32);

    c.bench_function("tape::relu(64KB)", |b| {
        b.iter(|| execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs)))
    });
}

fn bench_transformer_layer(c: &mut Criterion) {
    use hologram_ir::constant::ConstantData;

    let hidden = 2048usize;
    let ffn = 5632usize;
    let epsilon = f32::to_bits(1e-5);

    let make_weight = |rows: usize, cols: usize| -> Vec<u8> { vec![0x3f; rows * cols * 4] };

    let norm_weight = make_weight(1, hidden);
    let qkv_weight = make_weight(hidden, hidden);
    let out_weight = make_weight(hidden, hidden);
    let gate_weight = make_weight(hidden, ffn);
    let up_weight = make_weight(hidden, ffn);
    let down_weight = make_weight(ffn, hidden);

    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .constant_with_shape(ConstantData::Bytes(norm_weight.clone()), vec![hidden])
        .node_with_inputs(
            GraphOp::Float(FloatOp::RmsNorm {
                size: hidden as u32,
                epsilon,
            }),
            &[0, 1],
        )
        .constant_with_shape(ConstantData::Bytes(qkv_weight), vec![hidden, hidden])
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: hidden as u32,
                n: hidden as u32,
            }),
            &[2, 3],
        )
        .constant_with_shape(ConstantData::Bytes(out_weight), vec![hidden, hidden])
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: hidden as u32,
                n: hidden as u32,
            }),
            &[4, 5],
        )
        .node_with_inputs(GraphOp::Float(FloatOp::Add), &[0, 6])
        .constant_with_shape(ConstantData::Bytes(norm_weight), vec![hidden])
        .node_with_inputs(
            GraphOp::Float(FloatOp::RmsNorm {
                size: hidden as u32,
                epsilon,
            }),
            &[7, 8],
        )
        .constant_with_shape(ConstantData::Bytes(gate_weight), vec![hidden, ffn])
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: hidden as u32,
                n: ffn as u32,
            }),
            &[9, 10],
        )
        .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[11])
        .constant_with_shape(ConstantData::Bytes(up_weight), vec![hidden, ffn])
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: hidden as u32,
                n: ffn as u32,
            }),
            &[9, 13],
        )
        .node_with_inputs(GraphOp::Float(FloatOp::Mul), &[12, 14])
        .constant_with_shape(ConstantData::Bytes(down_weight), vec![ffn, hidden])
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: ffn as u32,
                n: hidden as u32,
            }),
            &[15, 16],
        )
        .node_with_inputs(GraphOp::Float(FloatOp::Add), &[7, 17])
        .node_with_inputs(GraphOp::Output, &[18])
        .output("y", 19)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = hologram_archive::load_from_bytes(&archive).unwrap();
    let tape = build_tape_from_plan(&plan).unwrap();

    let input_f32: Vec<u8> = (0..hidden)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let mut inputs = GraphInputs::new();
    inputs.set(0, input_f32);

    c.bench_function("transformer::tape(decode_step)", |b| {
        b.iter(|| execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs)))
    });
}

criterion_group!(
    benches,
    bench_executor_linear,
    bench_executor_diamond,
    bench_executor_wide_parallel,
    bench_executor_large_buffer,
    bench_schedule_build,
    bench_page_faults,
    bench_enum_tape_linear,
    bench_tape_relu_64kb,
    bench_transformer_layer,
);
criterion_main!(benches);
