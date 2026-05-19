//! AddRmsNorm semantics: out = rms_norm(x + residual).

use hologram_backend::cpu::dtype::DTYPE_F32;
use hologram_backend::{Backend, BufferRef, CpuBackend, KernelCall, NormCall, Workspace};

struct Ws {
    slots: Vec<Vec<u8>>,
}
impl Workspace for Ws {
    fn read(&self, b: BufferRef) -> &[u8] {
        if b.slot == u32::MAX {
            return &[];
        }
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
    ) -> Option<(Vec<&'a [u8]>, &'a mut [u8])> {
        let w = write.slot as usize;
        if reads
            .iter()
            .any(|r| r.slot != u32::MAX && r.slot as usize == w)
        {
            return None;
        }
        let (lo, hi) = self.slots.split_at_mut(w);
        let (wbuf, hi_rest) = hi.split_first_mut()?;
        let rs = reads
            .iter()
            .map(|r| -> &[u8] {
                if r.slot == u32::MAX {
                    return &[];
                }
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

fn f32_vec(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn read_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn add_rms_norm_sums_residual_before_normalizing() {
    // x = [1,1,1,1], residual = [1,1,1,1] => x+r = [2,2,2,2]
    // rms = sqrt(mean(4,4,4,4)+eps) = ~2 → out = [2/2*g, ...] = [g, g, g, g]
    let x = f32_vec(&[1.0, 1.0, 1.0, 1.0]);
    let resid = f32_vec(&[1.0, 1.0, 1.0, 1.0]);
    let gamma = f32_vec(&[1.0, 1.0, 1.0, 1.0]);
    let mut ws = Ws {
        slots: vec![
            x,
            gamma,
            vec![], // beta unused for rms
            resid,
            vec![0u8; 16],
        ],
    };
    let mut backend: CpuBackend<Ws> = CpuBackend::new();
    let call = KernelCall::AddRmsNorm(NormCall {
        x: buf(0),
        gamma: buf(1),
        beta: NormCall::NO_RESIDUAL,
        residual: buf(3),
        output: buf(4),
        batch: 1,
        feature: 4,
        epsilon_bits: 0,
        dtype: DTYPE_F32,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = read_f32_vec(&ws.slots[4]);
    // After add: [2,2,2,2]; mean(sq) = 4; rms = 2; out = [1,1,1,1] (gamma=1).
    for v in &out {
        assert!((v - 1.0).abs() < 1e-3, "expected ~1.0 got {v}");
    }
}

#[test]
fn add_rms_norm_no_residual_falls_back_to_rms() {
    // residual slot = NO_RESIDUAL → behaves like plain RmsNorm.
    let x = f32_vec(&[1.0, 1.0, 1.0, 1.0]);
    let gamma = f32_vec(&[1.0, 1.0, 1.0, 1.0]);
    let mut ws = Ws {
        slots: vec![x, gamma, vec![], vec![0u8; 16]],
    };
    let mut backend: CpuBackend<Ws> = CpuBackend::new();
    let call = KernelCall::AddRmsNorm(NormCall {
        x: buf(0),
        gamma: buf(1),
        beta: NormCall::NO_RESIDUAL,
        residual: NormCall::NO_RESIDUAL,
        output: buf(3),
        batch: 1,
        feature: 4,
        epsilon_bits: 0,
        dtype: DTYPE_F32,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = read_f32_vec(&ws.slots[3]);
    // x=[1,1,1,1] → rms=1 → out=[1,1,1,1]
    for v in &out {
        assert!((v - 1.0).abs() < 1e-3, "expected ~1.0 got {v}");
    }
}
