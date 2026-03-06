//! Q1 benchmarks: LUT lookups, view operations, and comparison with Q0 and native f64.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_core::lut::activation as q0_act;
use hologram_core::q1::activation as q1_act;
use hologram_core::q1::view::ElementWiseView16;
use hologram_core::view::ElementWiseView;

fn bench_q1_vs_q0_vs_f64_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("q1_vs_q0_vs_f64/sigmoid");

    // Q0: 256-byte table lookup
    group.bench_function("q0_lut", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(q0_act::sigmoid_lut(black_box(i)));
            }
        })
    });

    // Q1: 128KB table lookup
    group.bench_function("q1_lut", |b| {
        b.iter(|| {
            for i in (0u32..65536).step_by(256) {
                black_box(q1_act::sigmoid_q1(black_box(i as u16)));
            }
        })
    });

    // Native f64 sigmoid
    group.bench_function("f64_native", |b| {
        b.iter(|| {
            for i in (0u32..65536).step_by(256) {
                let x = (i as f64 - 32768.0) / 4096.0;
                black_box(1.0 / (1.0 + (-x).exp()));
            }
        })
    });

    group.finish();
}

fn bench_q1_vs_q0_vs_f64_sin(c: &mut Criterion) {
    let mut group = c.benchmark_group("q1_vs_q0_vs_f64/sin");

    group.bench_function("q0_lut", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(q0_act::sin_lut(black_box(i)));
            }
        })
    });

    group.bench_function("q1_lut", |b| {
        b.iter(|| {
            for i in (0u32..65536).step_by(256) {
                black_box(q1_act::sin_q1(black_box(i as u16)));
            }
        })
    });

    group.bench_function("f64_native", |b| {
        b.iter(|| {
            for i in (0u32..65536).step_by(256) {
                let angle = i as f64 * std::f64::consts::TAU / 65536.0;
                black_box(angle.sin());
            }
        })
    });

    group.finish();
}

fn bench_q1_batch_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("q1_batch");
    let input: Vec<u16> = (0..1024).map(|i| (i * 64) as u16).collect();

    group.bench_function("sigmoid_1024_q1", |b| {
        b.iter(|| {
            let mut out = vec![0u16; 1024];
            for (i, &v) in input.iter().enumerate() {
                out[i] = q1_act::sigmoid_q1(black_box(v));
            }
            black_box(out);
        })
    });

    group.bench_function("sigmoid_1024_f64", |b| {
        b.iter(|| {
            let mut out = vec![0.0f64; 1024];
            for (i, &v) in input.iter().enumerate() {
                let x = (v as f64 - 32768.0) / 4096.0;
                out[i] = 1.0 / (1.0 + (-x).exp());
            }
            black_box(out);
        })
    });

    group.finish();
}

fn bench_view16_apply_single(c: &mut Criterion) {
    let view = ElementWiseView16::from_static(&q1_act::SIGMOID_65536);

    c.bench_function("view16/apply_single", |b| {
        b.iter(|| black_box(view.apply(black_box(32768))))
    });
}

fn bench_view16_compose(c: &mut Criterion) {
    let sig = ElementWiseView16::from_static(&q1_act::SIGMOID_65536);
    let relu = ElementWiseView16::from_static(&q1_act::RELU_65536);

    c.bench_function("view16/compose", |b| {
        b.iter(|| black_box(sig.then(black_box(&relu))))
    });
}

fn bench_view16_apply_slice(c: &mut Criterion) {
    let view = ElementWiseView16::from_static(&q1_act::SIGMOID_65536);
    let data: Vec<u16> = (0..1024).map(|i| (i * 64) as u16).collect();

    c.bench_function("view16/apply_slice_1k", |b| {
        b.iter_batched(
            || data.clone(),
            |mut d| {
                view.apply_slice(black_box(&mut d));
                d
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_arith_q1_vs_q0(c: &mut Criterion) {
    let mut group = c.benchmark_group("arith_q1_vs_q0");

    group.bench_function("add/q0", |b| {
        b.iter(|| {
            for a in 0..=255u8 {
                for b_val in (0u8..=255).step_by(16) {
                    black_box(hologram_core::lut::arith::add_q0(
                        black_box(a),
                        black_box(b_val),
                    ));
                }
            }
        })
    });

    group.bench_function("add/q1", |b| {
        b.iter(|| {
            for a in (0u32..65536).step_by(256) {
                for b_val in (0u32..65536).step_by(4096) {
                    black_box(hologram_core::q1::add_q1(
                        black_box(a as u16),
                        black_box(b_val as u16),
                    ));
                }
            }
        })
    });

    group.bench_function("mul/q0", |b| {
        b.iter(|| {
            for a in 0..=255u8 {
                for b_val in (0u8..=255).step_by(16) {
                    black_box(hologram_core::lut::arith::mul_q0(
                        black_box(a),
                        black_box(b_val),
                    ));
                }
            }
        })
    });

    group.bench_function("mul/q1", |b| {
        b.iter(|| {
            for a in (0u32..65536).step_by(256) {
                for b_val in (0u32..65536).step_by(4096) {
                    black_box(hologram_core::q1::mul_q1(
                        black_box(a as u16),
                        black_box(b_val as u16),
                    ));
                }
            }
        })
    });

    group.finish();
}

fn bench_view16_vs_view8(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_compose");

    let q0_sig = ElementWiseView::from_table(*q0_act::activation_table_by_id(0).unwrap());
    let q0_relu = ElementWiseView::from_table(*q0_act::activation_table_by_id(4).unwrap());

    group.bench_function("q0/compose", |b| {
        b.iter(|| black_box(q0_sig.then(black_box(&q0_relu))))
    });

    let q1_sig = ElementWiseView16::from_static(&q1_act::SIGMOID_65536);
    let q1_relu = ElementWiseView16::from_static(&q1_act::RELU_65536);

    group.bench_function("q1/compose", |b| {
        b.iter(|| black_box(q1_sig.then(black_box(&q1_relu))))
    });

    group.finish();
}

fn bench_memory_budget(c: &mut Criterion) {
    // This is a verification benchmark, not a performance benchmark.
    // Validates that Q1 total memory stays under 4MB.
    c.bench_function("memory_budget/verify", |b| {
        b.iter(|| {
            let q1_total = q1_act::Q1_TABLE_COUNT * 131072; // 21 * 128KB
            assert!(q1_total < 4 * 1024 * 1024);

            let q0_arith = 4 * 65536; // 4 arithmetic tables @ 64KB
            let q0_act = 21 * 256; // 21 activation tables @ 256B
            let q0_obs = 7 * 256; // 7 observable tables @ 256B
            let q0_total = q0_arith + q0_act + q0_obs;

            let combined = q0_total + q1_total;
            assert!(combined < 4 * 1024 * 1024, "combined = {} bytes", combined);
            black_box(combined);
        })
    });
}

criterion_group!(
    benches,
    bench_q1_vs_q0_vs_f64_sigmoid,
    bench_q1_vs_q0_vs_f64_sin,
    bench_q1_batch_sigmoid,
    bench_view16_apply_single,
    bench_view16_compose,
    bench_view16_apply_slice,
    bench_arith_q1_vs_q0,
    bench_view16_vs_view8,
    bench_memory_budget,
);
criterion_main!(benches);
