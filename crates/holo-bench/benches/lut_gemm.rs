//! Criterion benchmarks for LUT-GEMM kernels.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holo_exec::lut_gemm::matmul::{lut_gemm_4bit, lut_gemm_8bit, naive_matmul};
use holo_exec::lut_gemm::quantize::{quantize_4bit, quantize_8bit};

fn bench_naive_matmul(c: &mut Criterion) {
    let (m, k, n) = (4, 64, 64);
    let a = vec![1.0f32; m * k];
    let b = vec![0.5f32; k * n];
    let mut out = vec![0.0f32; m * n];
    c.bench_function("naive_matmul(4x64x64)", |bench| {
        bench.iter(|| naive_matmul(black_box(&a), black_box(&b), &mut out, m, k, n))
    });
}

fn bench_lut_gemm_q4_16x16(c: &mut Criterion) {
    let (k, n) = (16, 16);
    let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let qw = quantize_4bit(&weights, k as u32, n as u32);
    let activations = vec![1.0f32; k];
    let mut output = vec![0.0f32; n];
    c.bench_function("lut_gemm_q4(1x16x16)", |bench| {
        bench.iter(|| lut_gemm_4bit(black_box(&activations), black_box(&qw), &mut output))
    });
}

fn bench_lut_gemm_q4_64x64(c: &mut Criterion) {
    let (k, n) = (64, 64);
    let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let qw = quantize_4bit(&weights, k as u32, n as u32);
    let activations = vec![1.0f32; 4 * k];
    let mut output = vec![0.0f32; 4 * n];
    c.bench_function("lut_gemm_q4(4x64x64)", |bench| {
        bench.iter(|| lut_gemm_4bit(black_box(&activations), black_box(&qw), &mut output))
    });
}

fn bench_lut_gemm_q4_256x256(c: &mut Criterion) {
    let (k, n) = (256, 256);
    let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.001).collect();
    let qw = quantize_4bit(&weights, k as u32, n as u32);
    let activations = vec![1.0f32; 4 * k];
    let mut output = vec![0.0f32; 4 * n];
    c.bench_function("lut_gemm_q4(4x256x256)", |bench| {
        bench.iter(|| lut_gemm_4bit(black_box(&activations), black_box(&qw), &mut output))
    });
}

fn bench_lut_gemm_q8_16x16(c: &mut Criterion) {
    let (k, n) = (16, 16);
    let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let qw = quantize_8bit(&weights, k as u32, n as u32);
    let activations = vec![1.0f32; k];
    let mut output = vec![0.0f32; n];
    c.bench_function("lut_gemm_q8(1x16x16)", |bench| {
        bench.iter(|| lut_gemm_8bit(black_box(&activations), black_box(&qw), &mut output))
    });
}

fn bench_lut_gemm_q8_64x64(c: &mut Criterion) {
    let (k, n) = (64, 64);
    let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let qw = quantize_8bit(&weights, k as u32, n as u32);
    let activations = vec![1.0f32; 4 * k];
    let mut output = vec![0.0f32; 4 * n];
    c.bench_function("lut_gemm_q8(4x64x64)", |bench| {
        bench.iter(|| lut_gemm_8bit(black_box(&activations), black_box(&qw), &mut output))
    });
}

fn bench_lut_gemm_q8_256x256(c: &mut Criterion) {
    let (k, n) = (256, 256);
    let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.001).collect();
    let qw = quantize_8bit(&weights, k as u32, n as u32);
    let activations = vec![1.0f32; 4 * k];
    let mut output = vec![0.0f32; 4 * n];
    c.bench_function("lut_gemm_q8(4x256x256)", |bench| {
        bench.iter(|| lut_gemm_8bit(black_box(&activations), black_box(&qw), &mut output))
    });
}

fn bench_quantize_q4(c: &mut Criterion) {
    let (k, n) = (64, 64);
    let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
    c.bench_function("quantize_q4(64x64)", |bench| {
        bench.iter(|| quantize_4bit(black_box(&weights), k as u32, n as u32))
    });
}

fn bench_quantize_q8(c: &mut Criterion) {
    let (k, n) = (64, 64);
    let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
    c.bench_function("quantize_q8(64x64)", |bench| {
        bench.iter(|| quantize_8bit(black_box(&weights), k as u32, n as u32))
    });
}

criterion_group!(
    benches,
    bench_naive_matmul,
    bench_lut_gemm_q4_16x16,
    bench_lut_gemm_q4_64x64,
    bench_lut_gemm_q4_256x256,
    bench_lut_gemm_q8_16x16,
    bench_lut_gemm_q8_64x64,
    bench_lut_gemm_q8_256x256,
    bench_quantize_q4,
    bench_quantize_q8,
);
criterion_main!(benches);
