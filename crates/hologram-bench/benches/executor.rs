//! KvExecutor benchmarks: execute graphs of varying topology.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_archive::HoloWriter;
use hologram_core::op::{LutOp, PrimOp};
use hologram_exec::{build_schedule, execute_bytes, GraphInputs};
use hologram_graph::builder::GraphBuilder;
use hologram_graph::fusion;
use hologram_graph::graph::GraphOp;

fn make_archive(g: &mut hologram_graph::Graph) -> Vec<u8> {
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
    let plan = hologram_archive::load_from_bytes(&archive).unwrap();

    c.bench_function("exec::build_schedule(5_nodes)", |b| {
        b.iter(|| build_schedule(black_box(plan.graph())))
    });
}

fn bench_page_faults(c: &mut Criterion) {
    use std::io::Write;

    // Build a moderately large archive to trigger measurable page faults.
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();
    let weights = vec![0u8; 256 * 1024]; // 256KB weights section
    let archive = HoloWriter::new()
        .set_graph(&g)
        .set_weights(weights)
        .build()
        .unwrap();

    // Write to a temp file for mmap-based loading.
    let dir = std::env::temp_dir();
    let path = dir.join("hologram_bench_pagefault.holo");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&archive).unwrap();
    }

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    // Benchmark mmap load + execute cycle.
    // To measure page faults, run with:
    //   perf stat -e major-faults,minor-faults cargo bench --bench executor -- page_faults
    // or on macOS:
    //   /usr/bin/time -l cargo bench --bench executor -- page_faults
    c.bench_function("exec::mmap_load_execute(256KB_weights)", |b| {
        b.iter(|| {
            let loader = hologram_archive::HoloLoader::open(&path).unwrap();
            let plan = loader.load().unwrap();
            black_box(hologram_exec::mmap::execute_plan(&plan, &inputs))
        })
    });

    std::fs::remove_file(&path).ok();
}

fn bench_enum_tape_linear(c: &mut Criterion) {
    // Same linear chain as bench_executor_linear, but via the enum-dispatch tape path.
    use hologram_core::op::FloatOp;

    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[0])
        .node_with_inputs(GraphOp::Float(FloatOp::Sigmoid), &[1])
        .node_with_inputs(GraphOp::Float(FloatOp::Tanh), &[2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();
    let _ = hologram_graph::fusion::fuse(&mut g);
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = hologram_archive::load_from_bytes(&archive).unwrap();
    let tape = hologram_exec::mmap::build_tape_from_plan(&plan).unwrap();

    // Build f32 input: 64 floats = 256 bytes
    let input_f32: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32) * 0.1).to_le_bytes())
        .collect();
    let mut inputs = GraphInputs::new();
    inputs.set(0, input_f32);

    c.bench_function("tape::linear_chain(4_float_nodes, 256B)", |b| {
        b.iter(|| {
            hologram_exec::mmap::execute_tape(
                black_box(&tape),
                black_box(&plan),
                black_box(&inputs),
            )
        })
    });
}

fn bench_enum_tape_vs_kvexecutor(c: &mut Criterion) {
    // Compare tape path vs KvExecutor path on the same graph.
    use hologram_core::op::FloatOp;

    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();
    let _ = hologram_graph::fusion::fuse(&mut g);
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    let input_f32: Vec<u8> = (0..16384) // 64KB of f32
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let mut inputs = GraphInputs::new();
    inputs.set(0, input_f32);

    let plan = hologram_archive::load_from_bytes(&archive).unwrap();
    let tape = hologram_exec::mmap::build_tape_from_plan(&plan).unwrap();

    let mut group = c.benchmark_group("tape_vs_kv");
    group.bench_function("kvexecutor(relu, 64KB)", |b| {
        b.iter(|| execute_bytes(black_box(&archive), black_box(&inputs)))
    });
    group.bench_function("enum_tape(relu, 64KB)", |b| {
        b.iter(|| {
            hologram_exec::mmap::execute_tape(
                black_box(&tape),
                black_box(&plan),
                black_box(&inputs),
            )
        })
    });
    group.finish();
}

fn bench_inline_vs_generic(c: &mut Criterion) {
    // Compare inline dispatch (Phase 9a) vs generic Float dispatch on same op.
    use hologram_core::op::FloatOp;

    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();
    let _ = hologram_graph::fusion::fuse(&mut g);
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = hologram_archive::load_from_bytes(&archive).unwrap();

    // Tape builder now maps Relu → InlineRelu automatically.
    let tape = hologram_exec::mmap::build_tape_from_plan(&plan).unwrap();

    let input_f32: Vec<u8> = (0..16384) // 64KB
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let mut inputs = GraphInputs::new();
    inputs.set(0, input_f32);

    c.bench_function("inline::relu(64KB)", |b| {
        b.iter(|| {
            hologram_exec::mmap::execute_tape(
                black_box(&tape),
                black_box(&plan),
                black_box(&inputs),
            )
        })
    });
}

/// Build a synthetic single-layer transformer graph (TinyLlama-scale).
///
/// Graph structure: Input → RmsNorm → Q/K/V MatMul → Add (residual)
///   → RmsNorm → FFN gate MatMul → Silu → FFN up MatMul → Mul → FFN down MatMul → Add → Output
///
/// Simplified from the full spec: omits Attention (which requires specific head layouts)
/// and uses the FFN + norm chain that dominates real inference time.
fn build_transformer_layer() -> (Vec<u8>, Vec<u8>) {
    use hologram_core::op::FloatOp;
    use hologram_graph::constant::ConstantData;

    let hidden = 2048usize;
    let ffn = 5632usize;
    let epsilon = f32::to_bits(1e-5);

    // Random weight data (f32 bytes). Content doesn't matter for benchmarking.
    let make_weight = |rows: usize, cols: usize| -> Vec<u8> {
        vec![0x3f; rows * cols * 4] // ~0.75 as f32 bytes (approximate)
    };

    let norm_weight = make_weight(1, hidden); // [hidden]
    let qkv_weight = make_weight(hidden, hidden); // [hidden, hidden] (simplified: Q only)
    let out_weight = make_weight(hidden, hidden); // [hidden, hidden]
    let gate_weight = make_weight(hidden, ffn); // [hidden, ffn]
    let up_weight = make_weight(hidden, ffn); // [hidden, ffn]
    let down_weight = make_weight(ffn, hidden); // [ffn, hidden]

    // Node indices:
    // 0: Input
    // 1: norm_w1 (constant)
    // 2: RmsNorm (attention)
    // 3: qkv_w (constant)
    // 4: MatMul Q projection
    // 5: out_w (constant)
    // 6: MatMul output projection
    // 7: Add (residual 1)
    // 8: norm_w2 (constant)
    // 9: RmsNorm (FFN)
    // 10: gate_w (constant)
    // 11: MatMul gate
    // 12: Silu
    // 13: up_w (constant)
    // 14: MatMul up
    // 15: Mul (gate * up)
    // 16: down_w (constant)
    // 17: MatMul down
    // 18: Add (residual 2)
    // 19: Output

    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0: Input
        .constant_with_shape(ConstantData::Bytes(norm_weight.clone()), vec![hidden]) // 1: norm_w1
        .node_with_inputs(
            GraphOp::Float(FloatOp::RmsNorm {
                size: hidden as u32,
                epsilon,
            }),
            &[0, 1],
        ) // 2: RmsNorm
        .constant_with_shape(ConstantData::Bytes(qkv_weight), vec![hidden, hidden]) // 3: qkv_w
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: hidden as u32,
                n: hidden as u32,
            }),
            &[2, 3],
        ) // 4: Q MatMul
        .constant_with_shape(ConstantData::Bytes(out_weight), vec![hidden, hidden]) // 5: out_w
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: hidden as u32,
                n: hidden as u32,
            }),
            &[4, 5],
        ) // 6: out MatMul
        .node_with_inputs(GraphOp::Float(FloatOp::Add), &[0, 6]) // 7: Add residual
        .constant_with_shape(ConstantData::Bytes(norm_weight), vec![hidden]) // 8: norm_w2
        .node_with_inputs(
            GraphOp::Float(FloatOp::RmsNorm {
                size: hidden as u32,
                epsilon,
            }),
            &[7, 8],
        ) // 9: RmsNorm FFN
        .constant_with_shape(ConstantData::Bytes(gate_weight), vec![hidden, ffn]) // 10: gate_w
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: hidden as u32,
                n: ffn as u32,
            }),
            &[9, 10],
        ) // 11: gate MatMul
        .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[11]) // 12: Silu
        .constant_with_shape(ConstantData::Bytes(up_weight), vec![hidden, ffn]) // 13: up_w
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: hidden as u32,
                n: ffn as u32,
            }),
            &[9, 13],
        ) // 14: up MatMul
        .node_with_inputs(GraphOp::Float(FloatOp::Mul), &[12, 14]) // 15: Mul gate*up
        .constant_with_shape(ConstantData::Bytes(down_weight), vec![ffn, hidden]) // 16: down_w
        .node_with_inputs(
            GraphOp::Float(FloatOp::MatMul {
                m: 1,
                k: ffn as u32,
                n: hidden as u32,
            }),
            &[15, 16],
        ) // 17: down MatMul
        .node_with_inputs(GraphOp::Float(FloatOp::Add), &[7, 17]) // 18: Add residual
        .node_with_inputs(GraphOp::Output, &[18]) // 19: Output
        .output("y", 19)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    // f32 input: [1, 1, 2048] = 8KB
    let input_f32: Vec<u8> = (0..hidden)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();

    (archive, input_f32)
}

fn bench_transformer_layer(c: &mut Criterion) {
    let (archive, input_f32) = build_transformer_layer();

    let mut inputs = GraphInputs::new();
    inputs.set(0, input_f32);

    let plan = hologram_archive::load_from_bytes(&archive).unwrap();
    let tape = hologram_exec::mmap::build_tape_from_plan(&plan).unwrap();

    let mut group = c.benchmark_group("transformer");

    group.bench_function("kvexecutor(decode_step)", |b| {
        b.iter(|| execute_bytes(black_box(&archive), black_box(&inputs)))
    });

    group.bench_function("enum_tape(decode_step)", |b| {
        b.iter(|| {
            hologram_exec::mmap::execute_tape(
                black_box(&tape),
                black_box(&plan),
                black_box(&inputs),
            )
        })
    });

    group.finish();
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
    bench_enum_tape_vs_kvexecutor,
    bench_inline_vs_generic,
    bench_transformer_layer,
);
criterion_main!(benches);
