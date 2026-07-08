//! MatMul kernel benchmark (spec XII.4).
//!
//! Exercises the CPU matmul kernel at byte-domain (W8) and f32 widths.
//! The f32 benches run on a real `BufferArena` workspace so the
//! split-borrow + `bytemuck::cast_slice` zero-copy path is exercised.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_backend::cpu::dtype::DTYPE_F32;
use hologram_backend::{Backend, BufferRef, CpuBackend, KernelCall, MatMulCall};
use hologram_exec::{BufferArena, SlotSpan};

fn ref_buf(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 0,
    }
}

/// Build a 3-slot `BufferArena` (A, B, output) with `slot_bytes`
/// per slot, rounded up to a 64-byte boundary so the per-slot offset
/// stays 64-byte aligned (needed for `bytemuck::cast_slice::<u8, f32>`
/// zero-copy in the matmul fast path).
fn make_arena(slot_bytes: usize) -> BufferArena {
    let rounded = slot_bytes.next_multiple_of(64);
    let slots = vec![
        SlotSpan {
            offset: 0,
            length: rounded as u64,
        },
        SlotSpan {
            offset: rounded as u64,
            length: rounded as u64,
        },
        SlotSpan {
            offset: 2 * rounded as u64,
            length: rounded as u64,
        },
    ];
    BufferArena::with_capacity(3 * rounded, slots)
}

/// Seed `slot` with `n` ascending f32 values (LE-encoded).
fn seed_f32(ws: &mut BufferArena, slot: usize, n: usize) {
    let bytes = ws.write_slot(slot).expect("slot in range");
    for i in 0..n {
        let v = ((i as f32) * 0.001).to_le_bytes();
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v);
    }
}

fn bench_matmul_w8_64(c: &mut Criterion) {
    c.bench_function("matmul_w8_64x64x64", |b| {
        let n = 64usize;
        let mut ws = make_arena(n * n);
        // Seed A, B with all-ones bytes.
        for slot in 0..2 {
            let buf = ws.write_slot(slot).expect("slot in range");
            for byte in buf.iter_mut().take(n * n) {
                *byte = 1;
            }
        }
        let mut backend: CpuBackend<BufferArena> = CpuBackend::new();
        let call = KernelCall::MatMul(MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            m: 64,
            k: 64,
            n: 64,
            dtype: 0,
            b_packed: false,
        });
        b.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}

fn bench_matmul_f32_64(c: &mut Criterion) {
    c.bench_function("matmul_f32_64x64x64 (zero-copy)", |bench| {
        let n = 64usize;
        let mut ws = make_arena(n * n * 4);
        seed_f32(&mut ws, 0, n * n);
        seed_f32(&mut ws, 1, n * n);
        let mut backend: CpuBackend<BufferArena> = CpuBackend::new();
        let call = KernelCall::MatMul(MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            m: 64,
            k: 64,
            n: 64,
            dtype: DTYPE_F32,
            b_packed: false,
        });
        bench.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}

fn bench_matmul_f32_128(c: &mut Criterion) {
    c.bench_function("matmul_f32_128x128x128 (zero-copy)", |bench| {
        let n = 128usize;
        let mut ws = make_arena(n * n * 4);
        seed_f32(&mut ws, 0, n * n);
        seed_f32(&mut ws, 1, n * n);
        let mut backend: CpuBackend<BufferArena> = CpuBackend::new();
        let call = KernelCall::MatMul(MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            m: 128,
            k: 128,
            n: 128,
            dtype: DTYPE_F32,
            b_packed: false,
        });
        bench.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}

/// f32 matmul at a size whose operands exceed L2 — exercises the
/// cache-oblivious recursion's scaling (efficiency should hold, not fall off,
/// since the recursion keeps each sub-problem's working set in-cache: misses
/// stay compulsory, not capacity).
fn bench_matmul_f32_square(c: &mut Criterion, n: usize) {
    c.bench_function(&format!("matmul_f32_{n}x{n}x{n} (zero-copy)"), |bench| {
        let mut ws = make_arena(n * n * 4);
        seed_f32(&mut ws, 0, n * n);
        seed_f32(&mut ws, 1, n * n);
        let mut backend: CpuBackend<BufferArena> = CpuBackend::new();
        let call = KernelCall::MatMul(MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            m: n as u32,
            k: n as u32,
            n: n as u32,
            dtype: DTYPE_F32,
            b_packed: false,
        });
        bench.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}

fn bench_matmul_f32_256(c: &mut Criterion) {
    bench_matmul_f32_square(c, 256);
}
fn bench_matmul_f32_512(c: &mut Criterion) {
    bench_matmul_f32_square(c, 512);
}

/// **Decode-shaped f32 GEMV** (M=1 · K×N, one token vector times a weight
/// matrix). The substrate's worst case: the GEMM microkernel's register tile
/// (MR rows) does not engage at M=1, so the row-remainder path decides the
/// speed. A scalar remainder collapses this to a few percent of memory
/// bandwidth; the vectorized FMA remainder holds it near roofline. This bench
/// guards that shape from regressing back to scalar.
fn bench_matmul_f32_gemv(c: &mut Criterion, k: usize, n: usize) {
    c.bench_function(&format!("matmul_f32_gemv_1x{k}x{n} (decode)"), |bench| {
        let mut ws = make_arena((k * n).max(n) * 4);
        seed_f32(&mut ws, 0, k); // A is 1×k
        seed_f32(&mut ws, 1, k * n); // B is k×n
        let mut backend: CpuBackend<BufferArena> = CpuBackend::new();
        let call = KernelCall::MatMul(MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            m: 1,
            k: k as u32,
            n: n as u32,
            dtype: DTYPE_F32,
            b_packed: false,
        });
        bench.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}
fn bench_matmul_f32_gemv_2048(c: &mut Criterion) {
    bench_matmul_f32_gemv(c, 2048, 2048);
}

criterion_group!(
    benches,
    bench_matmul_w8_64,
    bench_matmul_f32_64,
    bench_matmul_f32_128,
    bench_matmul_f32_256,
    bench_matmul_f32_512,
    bench_matmul_f32_gemv_2048
);
criterion_main!(benches);
