//! LUT table lookup benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holo_core::lut::activation;
use holo_core::lut::arith;
use holo_core::lut::q0;
use holo_core::op::LutOp;

fn bench_q0_stratum(c: &mut Criterion) {
    c.bench_function("q0::stratum_q0(127)", |b| {
        b.iter(|| q0::stratum_q0(black_box(127)))
    });
}

fn bench_arith_add(c: &mut Criterion) {
    c.bench_function("arith::add_q0(100, 200)", |b| {
        b.iter(|| arith::add_q0(black_box(100), black_box(200)))
    });
}

fn bench_arith_mul(c: &mut Criterion) {
    c.bench_function("arith::mul_q0(13, 17)", |b| {
        b.iter(|| arith::mul_q0(black_box(13), black_box(17)))
    });
}

fn bench_sigmoid_lut(c: &mut Criterion) {
    c.bench_function("activation::sigmoid_lut(128)", |b| {
        b.iter(|| activation::sigmoid_lut(black_box(128)))
    });
}

fn bench_sigmoid_vs_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid_comparison");
    group.bench_function("lut_lookup", |b| {
        b.iter(|| activation::sigmoid_lut(black_box(128)))
    });
    group.bench_function("f64_compute", |b| {
        b.iter(|| {
            let x: f64 = black_box(0.0);
            1.0 / (1.0 + (-x).exp())
        })
    });
    group.finish();
}

fn bench_all_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_lookup");
    for op in &LutOp::ALL {
        group.bench_function(op.name(), |b| {
            b.iter(|| op.apply(black_box(128)))
        });
    }
    group.finish();
}

fn bench_activation_batch(c: &mut Criterion) {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    c.bench_function("sigmoid_1024_scalars", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for &byte in black_box(&data) {
                sum += activation::sigmoid_lut(byte) as u64;
            }
            sum
        })
    });
}

criterion_group!(
    benches,
    bench_q0_stratum,
    bench_arith_add,
    bench_arith_mul,
    bench_sigmoid_lut,
    bench_sigmoid_vs_f64,
    bench_all_activations,
    bench_activation_batch,
);
criterion_main!(benches);
