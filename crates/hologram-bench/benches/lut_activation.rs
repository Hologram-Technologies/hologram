//! LUT-accelerated low-precision activation vs computed (PM_7 Q1 tier).
//!
//! At a 16-bit quantum level a transcendental activation is fully materialized
//! as a 65536-entry table (content-addressed, built once). This bench shows the
//! table lookup beats `widen → tanh/exp → narrow` over a large bf16 tensor —
//! the genuine speedup PM_7's CpuL2 tier names.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_compute::cpu::dtype::{read_bf16, write_bf16, DTYPE_BF16};
use hologram_compute::cpu::lut::unary_lut;
use hologram_compute::kernel_call::{lut_act, UnaryCall};
use hologram_compute::{BufferRef, Workspace};
use hologram_exec::{BufferArena, SlotSpan};

fn rb(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 0,
    }
}

/// tanh-approx GELU (matches `float_kernels::gelu_f`, the table's source).
fn gelu(x: f32) -> f32 {
    let c = (2.0f32 / core::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

fn bench_lut_vs_compute(c: &mut Criterion) {
    let n = 1usize << 20; // 1M bf16 elements
    let bytes = n * 2;
    let r = bytes.next_multiple_of(64) as u64;
    let arena = || {
        let mut a = BufferArena::with_capacity(
            2 * r as usize,
            vec![
                SlotSpan {
                    offset: 0,
                    length: r,
                },
                SlotSpan {
                    offset: r,
                    length: r,
                },
            ],
        );
        // Seed input slot 0 with a sweep of bf16 values.
        let buf = a.write_slot(0).expect("slot");
        for i in 0..n {
            write_bf16(buf, i, (i as f32 % 17.0) * 0.5 - 4.0);
        }
        a
    };
    let call = UnaryCall {
        input: rb(0),
        output: rb(1),
        element_count: n as u64,
        witt_bits: 16,
        dtype: DTYPE_BF16,
    };

    let mut ws = arena();
    c.bench_function("bf16_gelu_lut_1M (table lookup)", |b| {
        b.iter(|| {
            unary_lut(black_box(&call), &mut ws, lut_act::GELU).unwrap();
        })
    });

    // Computed baseline: widen → gelu → narrow, same bf16 data + buffers.
    let mut ws2 = arena();
    c.bench_function("bf16_gelu_compute_1M (widen+tanh+narrow)", |b| {
        b.iter(|| {
            let (reads, out) = ws2.split_borrow(&[rb(0)], rb(1)).unwrap();
            let inp = reads[0];
            for i in 0..n {
                write_bf16(out, i, gelu(read_bf16(inp, i)));
            }
            black_box(&out);
        })
    });
}

criterion_group!(benches, bench_lut_vs_compute);
criterion_main!(benches);
