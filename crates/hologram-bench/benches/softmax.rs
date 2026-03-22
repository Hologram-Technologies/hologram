//! Online softmax vs row-based softmax benchmark.
//!
//! Compares the two softmax implementations for decode (seq=1) scenarios
//! where the online variant avoids allocating the scores matrix.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Row-based softmax: standard two-pass (max, then exp/sum/normalize).
fn softmax_row(x: &[f32], size: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for row in out.chunks_mut(size) {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        let inv_sum = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
    out
}

/// Online softmax: single-pass (running max + rescale).
fn softmax_online(x: &[f32], size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks(size).zip(out.chunks_mut(size)) {
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        for &v in row_in {
            if v > max_val {
                sum *= (max_val - v).exp();
                max_val = v;
            }
            sum += (v - max_val).exp();
        }
        let inv_sum = 1.0 / sum;
        for (o, &v) in row_out.iter_mut().zip(row_in) {
            *o = ((v - max_val).exp()) * inv_sum;
        }
    }
    out
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_decode");

    for seq_k in [128, 512, 2048, 8192] {
        // Decode scenario: seq_q=1, so we have one row of length seq_k
        let data: Vec<f32> = (0..seq_k).map(|i| (i as f32) * 0.01 - 0.5).collect();

        group.bench_with_input(BenchmarkId::new("row_based", seq_k), &data, |b, data| {
            b.iter(|| black_box(softmax_row(black_box(data), seq_k)))
        });

        group.bench_with_input(BenchmarkId::new("online", seq_k), &data, |b, data| {
            b.iter(|| black_box(softmax_online(black_box(data), seq_k)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_softmax);
criterion_main!(benches);
