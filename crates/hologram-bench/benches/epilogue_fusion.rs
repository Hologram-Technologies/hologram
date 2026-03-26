//! A/B benchmark: unfused (MatMul + Silu as separate ops) vs fused
//! (single FusedMatMulActivation). Measures the memory round-trip
//! overhead eliminated by epilogue fusion.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hologram_archive::HoloWriter;
use hologram_core::op::FloatOp;
use hologram_exec::mmap::{build_tape_from_plan, execute_tape};
use hologram_exec::GraphInputs;
use hologram_graph::builder::GraphBuilder;
use hologram_graph::graph::GraphOp;

fn make_matmul_silu_graph(m: u32, k: u32, n: u32, fuse: bool) -> hologram_graph::Graph {
    use hologram_graph::constant::ConstantData;

    let weight_bytes = vec![0x3fu8; (k as usize) * (n as usize) * 4];
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .constant_with_shape(
            ConstantData::Bytes(weight_bytes),
            vec![k as usize, n as usize],
        ) // 1
        .node_with_inputs(GraphOp::Float(FloatOp::MatMul { m, k, n }), &[0, 1]) // 2
        .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[2]) // 3
        .node_with_inputs(GraphOp::Output, &[3]) // 4
        .output("y", 4)
        .build();

    if fuse {
        let _ = hologram_graph::fusion::fuse(&mut g);
    }
    g
}

fn bench_epilogue_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("epilogue_fusion");

    // Sizes where the activation overhead is a meaningful fraction.
    // At 1x64x64, matmul ~microseconds, activation overhead visible.
    let sizes: &[(u32, u32, u32)] = &[(1, 64, 64), (1, 256, 256), (1, 512, 512), (1, 2048, 2048)];

    for &(m, k, n) in sizes {
        let label = format!("{m}x{k}x{n}");

        // Unfused: MatMul + Silu as 2 separate tape instructions.
        let g_unfused = make_matmul_silu_graph(m, k, n, false);
        let archive = HoloWriter::new().set_graph(&g_unfused).build().unwrap();
        let plan_unfused = hologram_archive::load_from_bytes(&archive).unwrap();
        let tape_unfused = build_tape_from_plan(&plan_unfused).unwrap();

        let input_f32: Vec<u8> = (0..(m as usize * k as usize))
            .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
            .collect();
        let mut inputs = GraphInputs::new();
        inputs.set(0, input_f32.clone());

        group.bench_with_input(BenchmarkId::new("unfused", &label), &(), |b, _| {
            b.iter(|| {
                execute_tape(
                    black_box(&tape_unfused),
                    black_box(&plan_unfused),
                    black_box(&inputs),
                )
            })
        });

        // Fused: single FusedMatMulActivation tape instruction.
        let g_fused = make_matmul_silu_graph(m, k, n, true);
        let archive = HoloWriter::new().set_graph(&g_fused).build().unwrap();
        let plan_fused = hologram_archive::load_from_bytes(&archive).unwrap();
        let tape_fused = build_tape_from_plan(&plan_fused).unwrap();

        let mut inputs_fused = GraphInputs::new();
        inputs_fused.set(0, input_f32);

        group.bench_with_input(BenchmarkId::new("fused", &label), &(), |b, _| {
            b.iter(|| {
                execute_tape(
                    black_box(&tape_fused),
                    black_box(&plan_fused),
                    black_box(&inputs_fused),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_epilogue_fusion);
criterion_main!(benches);
