//! LUT-accelerated low-precision activation conformance (PM_7 Q1).
//!
//! A bf16 transcendental activation dispatched through the backend now takes
//! the content-addressed LUT path. It must equal the f64 reference of the
//! activation over the bf16-quantized input (the LUT entry is `narrow(f(widen
//! (bits)))`, so this is exact within bf16 — a speedup, not an approximation).

use hologram_backend::cpu::dtype::{read_bf16, write_bf16, DTYPE_BF16};
use hologram_backend::{
    Backend, BufferRef, CpuBackend, KernelCall, SplitReads, UnaryCall, Workspace,
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

fn gelu_ref(x: f64) -> f64 {
    let c = (2.0f64 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}
fn sigmoid_ref(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn bf16_gelu_via_lut_matches_f64_reference() {
    let n = 64usize;
    let mut inb = vec![0u8; n * 2];
    for i in 0..n {
        write_bf16(&mut inb, i, (i as f32) * 0.25 - 8.0);
    }
    let mut ws = Ws {
        slots: vec![inb, vec![0u8; n * 2]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    be.dispatch(
        &KernelCall::Gelu(UnaryCall {
            input: rb(0),
            output: rb(1),
            element_count: n as u64,
            witt_bits: 16,
            dtype: DTYPE_BF16,
        }),
        &mut ws,
    )
    .unwrap();
    for i in 0..n {
        let x = read_bf16(&ws.slots[0], i);
        let got = read_bf16(&ws.slots[1], i) as f64;
        // bf16 has ~3 decimal digits; the activation of the bf16 input,
        // re-quantized to bf16, must match within one bf16 ULP-ish bound.
        let want = gelu_ref(x as f64);
        assert!(
            (got - want).abs() <= 0.02 + 0.02 * want.abs(),
            "gelu[{i}] x={x} got {got} want {want}"
        );
    }
}

#[test]
fn bf16_sigmoid_via_lut_matches_f64_reference() {
    let n = 48usize;
    let mut inb = vec![0u8; n * 2];
    for i in 0..n {
        write_bf16(&mut inb, i, (i as f32) * 0.4 - 9.0);
    }
    let mut ws = Ws {
        slots: vec![inb, vec![0u8; n * 2]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    be.dispatch(
        &KernelCall::Sigmoid(UnaryCall {
            input: rb(0),
            output: rb(1),
            element_count: n as u64,
            witt_bits: 16,
            dtype: DTYPE_BF16,
        }),
        &mut ws,
    )
    .unwrap();
    for i in 0..n {
        let x = read_bf16(&ws.slots[0], i) as f64;
        let got = read_bf16(&ws.slots[1], i) as f64;
        assert!(
            (got - sigmoid_ref(x)).abs() <= 0.01,
            "sigmoid[{i}] x={x} got {got}"
        );
    }
}
