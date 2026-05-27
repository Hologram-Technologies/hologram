//! Densified `Dequantize → activation` vs the unfused scalar path (PM_7
//! densification, generalized to the f32 quantized-inference case).
//!
//! A quantized (i8) tensor feeding a transcendental activation stores its output
//! as f32 — too wide to table directly — but its *realized* domain is only 256
//! values. So `activation((q − zp)·scale)` densifies into a 256-entry table
//! indexed by the quantized byte: one lookup per element instead of `dequant →
//! widen → tanh → narrow`. This bench shows the fused table beats the unfused
//! pair over a large tensor, the throughput win keyed on the realized quantum
//! level rather than the storage width.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_backend::cpu::dtype::{DTYPE_F32, DTYPE_I8};
use hologram_backend::{
    lut_act, Backend, BufferRef, CpuBackend, DequantActivationCall, DequantizeCall, KernelCall,
    SplitReads, UnaryCall, Workspace,
};

struct Ws {
    slots: Vec<Vec<u8>>,
}
impl Workspace for Ws {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let s = b.slot as usize;
        let n = self.slots[s].len();
        &mut self.slots[s][..n]
    }
    fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(SplitReads<'a>, &'a mut [u8])> {
        let w = write.slot as usize;
        if reads.iter().any(|r| r.slot as usize == w) {
            return None;
        }
        let (lo, hi) = self.slots.split_at_mut(w);
        let (wbuf, rest) = hi.split_first_mut()?;
        let rs = reads
            .iter()
            .map(|r| {
                let i = r.slot as usize;
                if i < w {
                    &lo[i][..]
                } else {
                    &rest[i - w - 1][..]
                }
            })
            .collect();
        Some((rs, wbuf.as_mut_slice()))
    }
}

fn rb(s: u32) -> BufferRef {
    BufferRef {
        slot: s,
        offset: 0,
        length: 0,
    }
}

fn bench_dequant_activation(c: &mut Criterion) {
    let n = 1usize << 20; // 1M i8 elements
    let qbytes: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let scale_bits = 0.02f32.to_bits();
    let zero_point = 3i32;
    let mut be: CpuBackend<Ws> = CpuBackend::new();

    // Fused: one densified table over the quantized domain.
    let fused = KernelCall::DequantActivation(DequantActivationCall {
        input: rb(0),
        output: rb(1),
        element_count: n as u64,
        quant_dtype: DTYPE_I8,
        act: lut_act::GELU,
        dtype: DTYPE_F32,
        scale_bits,
        zero_point,
    });
    let mut wf = Ws {
        slots: vec![qbytes.clone(), vec![0u8; n * 4]],
    };
    c.bench_function("dequant_gelu_i8_1M_fused (densified table)", |b| {
        b.iter(|| {
            be.dispatch(black_box(&fused), &mut wf).unwrap();
        })
    });

    // Unfused: dequantize to f32, then scalar transcendental GELU.
    let deq = KernelCall::Dequantize(DequantizeCall {
        input: rb(0),
        scales: DequantizeCall::NO_VEC,
        zero_points: DequantizeCall::NO_VEC,
        output: rb(1),
        element_count: n as u64,
        channels: 0,
        inner: 0,
        quant_dtype: DTYPE_I8,
        dtype: DTYPE_F32,
        scale_bits,
        zero_point,
    });
    let gelu = KernelCall::Gelu(UnaryCall {
        input: rb(1),
        output: rb(2),
        element_count: n as u64,
        witt_bits: 32,
        dtype: DTYPE_F32,
    });
    let mut wu = Ws {
        slots: vec![qbytes, vec![0u8; n * 4], vec![0u8; n * 4]],
    };
    c.bench_function("dequant_gelu_i8_1M_unfused (dequant + scalar gelu)", |b| {
        b.iter(|| {
            be.dispatch(black_box(&deq), &mut wu).unwrap();
            be.dispatch(black_box(&gelu), &mut wu).unwrap();
        })
    });
}

criterion_group!(benches, bench_dequant_activation);
criterion_main!(benches);
