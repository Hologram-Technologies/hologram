//! IEEE-754 native CPU kernel correctness tests.

use hologram_compute::cpu::dtype::DTYPE_F32;
use hologram_compute::SplitReads;
use hologram_compute::{
    Backend, BinaryCall, BufferRef, CpuBackend, KernelCall, MatMulCall, UnaryCall, Workspace,
};

struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}

impl Workspace for TestWorkspace {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize][..]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let slot = b.slot as usize;
        let len = self.slots[slot].len();
        let _ = b;
        &mut self.slots[slot][..len]
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
        let (wbuf, hi_rest) = hi.split_first_mut()?;
        let rs = reads
            .iter()
            .map(|r| {
                let i = r.slot as usize;
                if i < w {
                    &lo[i][..]
                } else {
                    &hi_rest[i - w - 1][..]
                }
            })
            .collect();
        Some((rs, wbuf.as_mut_slice()))
    }
}

fn buf(slot: u32, length: u64) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length,
    }
}

fn f32_to_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn add_f32_native() {
    let a = f32_to_le(&[1.0, 2.0, 3.0, 4.0]);
    let b = f32_to_le(&[10.0, 20.0, 30.0, 40.0]);
    let mut ws = TestWorkspace {
        slots: vec![a, b, vec![0u8; 16]],
    };
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::Add(BinaryCall {
        a: buf(0, 16),
        b: buf(1, 16),
        output: buf(2, 16),
        element_count: 4,
        witt_bits: 32,
        dtype: DTYPE_F32,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = le_to_f32(&ws.slots[2]);
    assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn matmul_f32_2x2() {
    let a = f32_to_le(&[1.0, 2.0, 3.0, 4.0]);
    let b = f32_to_le(&[5.0, 6.0, 7.0, 8.0]);
    let mut ws = TestWorkspace {
        slots: vec![a, b, vec![0u8; 16]],
    };
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::MatMul(MatMulCall {
        a: buf(0, 16),
        b: buf(1, 16),
        output: buf(2, 16),
        m: 2,
        k: 2,
        n: 2,
        dtype: DTYPE_F32,
        b_packed: false,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = le_to_f32(&ws.slots[2]);
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn relu_f32_native() {
    let a = f32_to_le(&[-3.0, -1.0, 0.0, 2.0, 5.0]);
    let mut ws = TestWorkspace {
        slots: vec![a, vec![0u8; 20]],
    };
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::Relu(UnaryCall {
        input: buf(0, 20),
        output: buf(1, 20),
        element_count: 5,
        witt_bits: 32,
        dtype: DTYPE_F32,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = le_to_f32(&ws.slots[1]);
    assert_eq!(out, vec![0.0, 0.0, 0.0, 2.0, 5.0]);
}

#[test]
fn sigmoid_f32_native() {
    let a = f32_to_le(&[0.0]);
    let mut ws = TestWorkspace {
        slots: vec![a, vec![0u8; 4]],
    };
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::Sigmoid(UnaryCall {
        input: buf(0, 4),
        output: buf(1, 4),
        element_count: 1,
        witt_bits: 32,
        dtype: DTYPE_F32,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = le_to_f32(&ws.slots[1]);
    // sigmoid(0) == 0.5
    assert!((out[0] - 0.5).abs() < 1e-6);
}

#[test]
fn tanh_f32_native() {
    let a = f32_to_le(&[0.0, 1.0]);
    let mut ws = TestWorkspace {
        slots: vec![a, vec![0u8; 8]],
    };
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::Tanh(UnaryCall {
        input: buf(0, 8),
        output: buf(1, 8),
        element_count: 2,
        witt_bits: 32,
        dtype: DTYPE_F32,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = le_to_f32(&ws.slots[1]);
    assert!((out[0] - 0.0).abs() < 1e-6);
    assert!((out[1] - 0.7615942_f32).abs() < 1e-3);
}
