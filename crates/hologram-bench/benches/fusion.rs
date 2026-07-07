//! Fusion benchmarks — fused vs unfused, head to head (spec XII.4 / FU class).
//!
//! Demonstrates the runtime fusions are an *improvement*, not just neutral:
//!   * `dequantize → matmul` vs the fused `MatMulDequant` — the fused op
//!     dequantizes the weight into a reused thread-local panel inside the
//!     matmul, eliding the materialized dense f32 weight (a slot write + read).
//!   * `expand → mul` vs the fused `BroadcastBinary` — the fused op reads the
//!     small operand with stride-0 indexing in place, eliding the materialized
//!     broadcast tensor (a slot write + read).
//!
//! Both run on a real `BufferArena` (64-byte-aligned slots) so the zero-copy
//! f32 views and reused scratch are exercised exactly as in production.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_backend::cpu::dtype::DTYPE_F32;
use hologram_backend::{
    broadcast_op, Backend, BinaryCall, BroadcastBinaryCall, BufferRef, CpuBackend, DequantizeCall,
    ExpandCall, KernelCall, MatMulCall, MatMulDequantCall,
};
use hologram_exec::{BufferArena, SlotSpan};

fn rb(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 0,
    }
}

/// `n` equal-sized 64-byte-aligned slots.
fn arena(slot_bytes: usize, slots: usize) -> BufferArena {
    let r = slot_bytes.next_multiple_of(64) as u64;
    let spans: Vec<SlotSpan> = (0..slots)
        .map(|i| SlotSpan {
            offset: i as u64 * r,
            length: r,
        })
        .collect();
    BufferArena::with_capacity(slots * r as usize, spans)
}

fn seed_f32(ws: &mut BufferArena, slot: usize, n: usize, base: f32) {
    let buf = ws.write_slot(slot).expect("slot");
    for i in 0..n {
        buf[i * 4..i * 4 + 4].copy_from_slice(&((i as f32) * 0.001 + base).to_le_bytes());
    }
}

// ── dequantize → matmul ───────────────────────────────────────────────

fn bench_dequant_matmul(c: &mut Criterion) {
    let d = 256usize;
    // Slots: 0=A(f32) 1=Bq(i8) 2=Wf32(intermediate) 3=out.
    let mut ws = arena(d * d * 4, 4);
    seed_f32(&mut ws, 0, d * d, 0.5);
    {
        let bq = ws.write_slot(1).expect("slot");
        for (i, byte) in bq.iter_mut().take(d * d).enumerate() {
            *byte = (i % 17) as u8;
        }
    }
    let mut be: CpuBackend<BufferArena> = CpuBackend::new();
    let scale = 0.02f32.to_bits();

    let dq = KernelCall::Dequantize(DequantizeCall {
        input: rb(1),
        scales: DequantizeCall::NO_VEC,
        zero_points: DequantizeCall::NO_VEC,
        output: rb(2),
        element_count: (d * d) as u64,
        channels: 0,
        inner: 0,
        quant_dtype: 2,
        dtype: DTYPE_F32,
        scale_bits: scale,
        zero_point: 0,
    });
    let mm = KernelCall::MatMul(MatMulCall {
        a: rb(0),
        b: rb(2),
        output: rb(3),
        m: d as u32,
        k: d as u32,
        n: d as u32,
        dtype: DTYPE_F32,
        b_packed: false,
    });
    let fused = KernelCall::MatMulDequant(MatMulDequantCall {
        a: rb(0),
        bq: rb(1),
        scales: DequantizeCall::NO_VEC,
        zero_points: DequantizeCall::NO_VEC,
        output: rb(3),
        m: d as u32,
        k: d as u32,
        n: d as u32,
        channels: 0,
        inner: 0,
        quant_dtype: 2,
        dtype: DTYPE_F32,
        scale_bits: scale,
        zero_point: 0,
        bq_omajor: false,
        act_quant: 0,
        act: 0,
        residual: MatMulDequantCall::NO_RESIDUAL,
    });

    c.bench_function("dequant_then_matmul_256 (unfused)", |b| {
        b.iter(|| {
            be.dispatch(black_box(&dq), &mut ws).unwrap();
            be.dispatch(black_box(&mm), &mut ws).unwrap();
        })
    });
    c.bench_function("matmul_dequant_256 (fused)", |b| {
        b.iter(|| {
            be.dispatch(black_box(&fused), &mut ws).unwrap();
        })
    });
}

// ── expand → mul ──────────────────────────────────────────────────────

fn bench_expand_binary(c: &mut Criterion) {
    let d = 512usize;
    // Slots: 0=small([1,d]) 1=other([d,d]) 2=expanded(intermediate) 3=out.
    let mut ws = arena(d * d * 4, 4);
    seed_f32(&mut ws, 0, d, 1.0);
    seed_f32(&mut ws, 1, d * d, 0.5);
    let mut be: CpuBackend<BufferArena> = CpuBackend::new();

    let mut in_dims = [0u32; 8];
    let mut out_dims = [0u32; 8];
    in_dims[0] = 1;
    in_dims[1] = d as u32;
    out_dims[0] = d as u32;
    out_dims[1] = d as u32;

    let expand = KernelCall::Expand(ExpandCall {
        input: rb(0),
        output: rb(2),
        rank: 2,
        in_dims,
        out_dims,
        dtype: DTYPE_F32,
    });
    let mul = KernelCall::Mul(BinaryCall {
        a: rb(2),
        b: rb(1),
        output: rb(3),
        element_count: (d * d) as u64,
        witt_bits: 32,
        dtype: DTYPE_F32,
    });
    let fused = KernelCall::BroadcastBinary(BroadcastBinaryCall {
        small: rb(0),
        other: rb(1),
        output: rb(3),
        rank: 2,
        in_dims,
        out_dims,
        op: broadcast_op::MUL,
        small_is_lhs: true,
        dtype: DTYPE_F32,
    });

    c.bench_function("expand_then_mul_512 (unfused)", |b| {
        b.iter(|| {
            be.dispatch(black_box(&expand), &mut ws).unwrap();
            be.dispatch(black_box(&mul), &mut ws).unwrap();
        })
    });
    c.bench_function("broadcast_mul_512 (fused)", |b| {
        b.iter(|| {
            be.dispatch(black_box(&fused), &mut ws).unwrap();
        })
    });
}

criterion_group!(benches, bench_dequant_matmul, bench_expand_binary);
criterion_main!(benches);
