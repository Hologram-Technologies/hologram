//! Smoke tests for the wgpu backend (spec IX.4).
//!
//! Cfg-gated behind the `wgpu` feature. Tests are skipped silently when no
//! GPU adapter is available (common on headless CI containers).

#![cfg(feature = "wgpu")]

use hologram_backend::cpu::dtype::DTYPE_F32;
use hologram_backend::SplitReads;
use hologram_backend::WgpuBackend;
use hologram_backend::{Backend, BinaryCall, BufferRef, KernelCall, MatMulCall, Workspace};

struct Ws {
    slots: Vec<Vec<u8>>,
}
impl Workspace for Ws {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let i = b.slot as usize;
        let len = self.slots[i].len();
        &mut self.slots[i][..len]
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
fn buf(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 0,
    }
}

fn f32_to_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn wgpu_add_f32_smoke() {
    let mut backend: WgpuBackend<Ws> = match WgpuBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("wgpu unavailable — skipping");
            return;
        }
    };
    let a = f32_to_le(&[1.0, 2.0, 3.0, 4.0]);
    let b = f32_to_le(&[10.0, 20.0, 30.0, 40.0]);
    let mut ws = Ws {
        slots: vec![a, b, vec![0u8; 16]],
    };
    let call = KernelCall::Add(BinaryCall {
        a: buf(0),
        b: buf(1),
        output: buf(2),
        element_count: 4,
        witt_bits: 32,
        dtype: DTYPE_F32,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = le_to_f32(&ws.slots[2]);
    assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn wgpu_matmul_f32_2x2_smoke() {
    let mut backend: WgpuBackend<Ws> = match WgpuBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("wgpu unavailable — skipping");
            return;
        }
    };
    let a = f32_to_le(&[1.0, 2.0, 3.0, 4.0]);
    let b = f32_to_le(&[5.0, 6.0, 7.0, 8.0]);
    let mut ws = Ws {
        slots: vec![a, b, vec![0u8; 16]],
    };
    let call = KernelCall::MatMul(MatMulCall {
        a: buf(0),
        b: buf(1),
        output: buf(2),
        m: 2,
        k: 2,
        n: 2,
        dtype: DTYPE_F32,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = le_to_f32(&ws.slots[2]);
    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn wgpu_falls_back_to_cpu_for_byte_kernels() {
    let mut backend: WgpuBackend<Ws> = match WgpuBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("wgpu unavailable — skipping");
            return;
        }
    };
    // Byte-domain Add (dtype=1, U8) should route through CPU fallback inside
    // the wgpu backend's dispatch.
    let mut ws = Ws {
        slots: vec![vec![1u8, 2, 3], vec![10u8, 20, 30], vec![0u8; 3]],
    };
    let call = KernelCall::Add(BinaryCall {
        a: buf(0),
        b: buf(1),
        output: buf(2),
        element_count: 3,
        witt_bits: 8,
        dtype: 1,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    assert_eq!(ws.slots[2], vec![11u8, 22, 33]);
}
