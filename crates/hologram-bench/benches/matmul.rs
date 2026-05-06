//! MatMul kernel benchmark (spec XII.4).
//!
//! Exercises the CPU matmul kernel at byte-domain (W8) and f32 widths.

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use hologram_backend::{
    CpuBackend, Backend, KernelCall, BufferRef, MatMulCall, Workspace,
};
use hologram_backend::cpu::dtype::DTYPE_F32;

struct Ws { slots: Vec<Vec<u8>> }
impl Workspace for Ws {
    fn read(&self, b: BufferRef) -> &[u8] { &self.slots[b.slot as usize] }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let i = b.slot as usize;
        let len = self.slots[i].len();
        &mut self.slots[i][..len]
    }
}

fn ref_buf(slot: u32) -> BufferRef { BufferRef { slot, offset: 0, length: 0 } }

fn bench_matmul_w8_64(c: &mut Criterion) {
    c.bench_function("matmul_w8_64x64x64", |b| {
        let n = 64usize;
        let mut ws = Ws { slots: vec![
            vec![1u8; n * n], vec![1u8; n * n], vec![0u8; n * n],
        ]};
        let mut backend: CpuBackend<Ws> = CpuBackend::new();
        let call = KernelCall::MatMul(MatMulCall {
            a: ref_buf(0), b: ref_buf(1), output: ref_buf(2),
            m: 64, k: 64, n: 64, dtype: 0,
        });
        b.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}

fn bench_matmul_f32_64(c: &mut Criterion) {
    c.bench_function("matmul_f32_64x64x64", |bench| {
        let n = 64usize;
        let mut a_bytes = vec![0u8; n * n * 4];
        let mut b_bytes = vec![0u8; n * n * 4];
        for i in 0..n * n {
            let v = ((i as f32) * 0.001).to_le_bytes();
            a_bytes[i * 4..i * 4 + 4].copy_from_slice(&v);
            b_bytes[i * 4..i * 4 + 4].copy_from_slice(&v);
        }
        let mut ws = Ws { slots: vec![
            a_bytes, b_bytes, vec![0u8; n * n * 4],
        ]};
        let mut backend: CpuBackend<Ws> = CpuBackend::new();
        let call = KernelCall::MatMul(MatMulCall {
            a: ref_buf(0), b: ref_buf(1), output: ref_buf(2),
            m: 64, k: 64, n: 64, dtype: DTYPE_F32,
        });
        bench.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}

fn bench_matmul_f32_128(c: &mut Criterion) {
    c.bench_function("matmul_f32_128x128x128", |bench| {
        let n = 128usize;
        let mut a_bytes = vec![0u8; n * n * 4];
        let mut b_bytes = vec![0u8; n * n * 4];
        for i in 0..n * n {
            let v = ((i as f32) * 0.001).to_le_bytes();
            a_bytes[i * 4..i * 4 + 4].copy_from_slice(&v);
            b_bytes[i * 4..i * 4 + 4].copy_from_slice(&v);
        }
        let mut ws = Ws { slots: vec![
            a_bytes, b_bytes, vec![0u8; n * n * 4],
        ]};
        let mut backend: CpuBackend<Ws> = CpuBackend::new();
        let call = KernelCall::MatMul(MatMulCall {
            a: ref_buf(0), b: ref_buf(1), output: ref_buf(2),
            m: 128, k: 128, n: 128, dtype: DTYPE_F32,
        });
        bench.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}

criterion_group!(benches, bench_matmul_w8_64, bench_matmul_f32_64, bench_matmul_f32_128);
criterion_main!(benches);
