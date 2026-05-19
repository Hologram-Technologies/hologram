//! Benchmarks mirroring the calculator example's four workflows:
//! encoding round-trips, LUT composition, graph I/O, and full pipeline.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_archive::HoloWriter;
use hologram_core::encoding::{AngleEncoding, Encoding, SignedEncoding, UnsignedEncoding};
use hologram_core::op::{LutOp, PrimOp};
use hologram_core::view::ElementWiseView;
use hologram_exec::mmap::{build_tape_from_plan, execute_tape};
use hologram_exec::GraphInputs;
use hologram_graph::builder::GraphBuilder;
use hologram_graph::fusion;
use hologram_graph::graph::GraphOp;

// ---------------------------------------------------------------------------
// Group 1: Encoding round-trip (Pi-F-Lambda)
// ---------------------------------------------------------------------------

fn bench_encoding_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding_roundtrip");

    let angle = AngleEncoding;
    let signed = SignedEncoding;
    let unsigned = UnsignedEncoding;

    group.bench_function("lut_sin_single", |b| {
        b.iter(|| {
            let byte_in = angle.embed(black_box(1.0_f64));
            let byte_out = LutOp::Sin.apply(byte_in);
            signed.lift(byte_out)
        })
    });

    group.bench_function("native_sin_single", |b| b.iter(|| black_box(1.0_f64).sin()));

    group.bench_function("lut_sqrt_single", |b| {
        b.iter(|| {
            let byte_in = unsigned.embed(black_box(0.5_f64));
            let byte_out = LutOp::Sqrt.apply(byte_in);
            unsigned.lift(byte_out)
        })
    });

    group.bench_function("native_sqrt_single", |b| {
        b.iter(|| black_box(0.5_f64).sqrt())
    });

    group.bench_function("lut_sin_256", |b| {
        b.iter(|| {
            let mut sum = 0.0_f64;
            for b_val in 0u8..=255 {
                let x = angle.lift(b_val);
                let byte_in = angle.embed(x);
                let byte_out = LutOp::Sin.apply(byte_in);
                sum += signed.lift(byte_out);
            }
            black_box(sum)
        })
    });

    group.bench_function("native_sin_256", |b| {
        b.iter(|| {
            let mut sum = 0.0_f64;
            for b_val in 0u8..=255 {
                let x = angle.lift(b_val);
                sum += x.sin();
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: LUT composition
// ---------------------------------------------------------------------------

fn bench_lut_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("lut_composition");

    let cos_view = ElementWiseView::from_table(*LutOp::Cos.table());
    let sin_view = ElementWiseView::from_table(*LutOp::Sin.table());
    let composed = cos_view.then(&sin_view);

    group.bench_function("chained_sin_cos", |b| {
        b.iter(|| LutOp::Sin.apply(LutOp::Cos.apply(black_box(128u8))))
    });

    group.bench_function("fused_sin_cos", |b| {
        b.iter(|| composed.apply(black_box(128u8)))
    });

    group.bench_function("native_sin_cos", |b| {
        b.iter(|| black_box(0.5_f64).cos().sin())
    });

    group.bench_function("fused_256", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for byte in 0u8..=255 {
                sum += composed.apply(black_box(byte)) as u64;
            }
            black_box(sum)
        })
    });

    let angle = AngleEncoding;
    group.bench_function("native_sin_cos_256", |b| {
        b.iter(|| {
            let mut sum = 0.0_f64;
            for b_val in 0u8..=255 {
                let x = angle.lift(b_val);
                sum += x.cos().sin();
            }
            black_box(sum)
        })
    });

    group.bench_function("build_composed_view", |b| {
        b.iter(|| {
            let c = ElementWiseView::from_table(black_box(*LutOp::Cos.table()));
            let s = ElementWiseView::from_table(black_box(*LutOp::Sin.table()));
            black_box(c.then(&s))
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: Graph I/O (multi-output fan-out)
// ---------------------------------------------------------------------------

fn build_multi_output_archive() -> Vec<u8> {
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Abs), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .node_with_inputs(GraphOp::Output, &[2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("relu", 4)
        .output("sigmoid", 5)
        .output("abs", 6)
        .build();
    HoloWriter::new().set_graph(&g).build().unwrap()
}

fn bench_graph_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_io");

    group.bench_function("build_and_serialize", |b| {
        b.iter(|| black_box(build_multi_output_archive()))
    });

    let archive = build_multi_output_archive();

    let inputs_256 = {
        let mut inp = GraphInputs::new();
        inp.set(0, (0..=255).collect());
        inp
    };

    group.bench_function("execute_multi_output(256B)", |b| {
        b.iter(|| {
            let plan = hologram_archive::load_from_bytes(&archive).unwrap();
            let tape = build_tape_from_plan(&plan).unwrap();
            execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs_256))
        })
    });

    let data_256: Vec<f64> = (0..=255).map(|i| i as f64 / 255.0).collect();
    group.bench_function("native_multi_output(256)", |b| {
        b.iter(|| {
            let d = black_box(&data_256);
            let relu: Vec<f64> = d.iter().map(|&x| x.max(0.0)).collect();
            let sigmoid: Vec<f64> = d.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
            let abs: Vec<f64> = d.iter().map(|&x| x.abs()).collect();
            black_box((relu, sigmoid, abs))
        })
    });

    let inputs_64k = {
        let mut inp = GraphInputs::new();
        inp.set(0, vec![128u8; 65536]);
        inp
    };

    group.bench_function("execute_multi_output(64KB)", |b| {
        b.iter(|| {
            let plan = hologram_archive::load_from_bytes(&archive).unwrap();
            let tape = build_tape_from_plan(&plan).unwrap();
            execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs_64k))
        })
    });

    let data_64k: Vec<f64> = (0..65536).map(|i| (i % 256) as f64 / 255.0).collect();
    group.bench_function("native_multi_output(64K)", |b| {
        b.iter(|| {
            let d = black_box(&data_64k);
            let relu: Vec<f64> = d.iter().map(|&x| x.max(0.0)).collect();
            let sigmoid: Vec<f64> = d.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
            let abs: Vec<f64> = d.iter().map(|&x| x.abs()).collect();
            black_box((relu, sigmoid, abs))
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: Full pipeline (build → fuse → serialize → execute)
// ---------------------------------------------------------------------------

fn build_fused_sin_cos_archive() -> Vec<u8> {
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Sin), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Cos), &[1])
        .node_with_inputs(GraphOp::Output, &[2])
        .output("y", 3)
        .build();
    let _ = fusion::fuse(&mut g).unwrap();
    HoloWriter::new().set_graph(&g).build().unwrap()
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    group.bench_function("build_fuse_serialize", |b| {
        b.iter(|| black_box(build_fused_sin_cos_archive()))
    });

    let inputs_256 = {
        let mut inp = GraphInputs::new();
        inp.set(0, (0..=255).collect());
        inp
    };

    group.bench_function("end_to_end(256B)", |b| {
        b.iter(|| {
            let archive = build_fused_sin_cos_archive();
            black_box({
                let plan = hologram_archive::load_from_bytes(&archive).unwrap();
                let tape = build_tape_from_plan(&plan).unwrap();
                execute_tape(&tape, &plan, &inputs_256).unwrap()
            })
        })
    });

    let inputs_64k = {
        let mut inp = GraphInputs::new();
        inp.set(0, vec![128u8; 65536]);
        inp
    };

    group.bench_function("end_to_end(64KB)", |b| {
        b.iter(|| {
            let archive = build_fused_sin_cos_archive();
            black_box({
                let plan = hologram_archive::load_from_bytes(&archive).unwrap();
                let tape = build_tape_from_plan(&plan).unwrap();
                execute_tape(&tape, &plan, &inputs_64k).unwrap()
            })
        })
    });

    let archive = build_fused_sin_cos_archive();

    group.bench_function("execute_fused(256B)", |b| {
        b.iter(|| {
            let plan = hologram_archive::load_from_bytes(&archive).unwrap();
            let tape = build_tape_from_plan(&plan).unwrap();
            execute_tape(black_box(&tape), black_box(&plan), black_box(&inputs_256))
        })
    });

    let angle = AngleEncoding;
    group.bench_function("native_cos_sin(256)", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(256);
            for b_val in 0u8..=255 {
                let x = angle.lift(b_val);
                out.push(x.sin().cos());
            }
            black_box(out)
        })
    });

    let data_64k: Vec<f64> = (0..65536).map(|i| angle.lift((i % 256) as u8)).collect();
    group.bench_function("native_cos_sin(64K)", |b| {
        b.iter(|| {
            let out: Vec<f64> = black_box(&data_64k)
                .iter()
                .map(|&x| x.sin().cos())
                .collect();
            black_box(out)
        })
    });

    group.bench_function("binary_add_end_to_end", |b| {
        b.iter(|| {
            let mut g = GraphBuilder::new()
                .input("a")
                .input("b")
                .node_from_graph_input(GraphOp::Input, 0)
                .node_from_graph_input(GraphOp::Input, 1)
                .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[0, 1])
                .node_with_inputs(GraphOp::Output, &[2])
                .output("sum", 3)
                .build();
            let _ = fusion::fuse(&mut g).unwrap();
            let archive = HoloWriter::new().set_graph(&g).build().unwrap();
            let mut inputs = GraphInputs::new();
            inputs.set(0, vec![10u8, 100, 200, 250]);
            inputs.set(1, vec![5u8, 50, 100, 200]);
            black_box({
                let plan = hologram_archive::load_from_bytes(&archive).unwrap();
                let tape = build_tape_from_plan(&plan).unwrap();
                execute_tape(&tape, &plan, &inputs).unwrap()
            })
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_encoding_roundtrip,
    bench_lut_composition,
    bench_graph_io,
    bench_full_pipeline,
);
criterion_main!(benches);
