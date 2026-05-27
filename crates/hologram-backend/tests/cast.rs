//! Numeric `Cast` kernel conformance (ONNX `Cast`).
//!
//! The general int↔float↔int converter: int→float is exact within the
//! destination mantissa, float→int truncates toward zero, int↔int and
//! float↔float change width. These pin the value-preserving semantics against
//! the spec for the dtype pairs models actually use (the int64 `input_ids` →
//! f32 path, f32 → int label paths, and float width changes).

use hologram_backend::cpu::dtype::{
    read_bf16, read_f32, write_f32, DTYPE_BF16, DTYPE_F32, DTYPE_I32, DTYPE_I64,
};
use hologram_backend::{
    Backend, BufferRef, CastCall, CpuBackend, KernelCall, SplitReads, Workspace,
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

fn cast(input: Vec<u8>, n: usize, src: u8, dst: u8, out_bytes: usize) -> Vec<u8> {
    let mut ws = Ws {
        slots: vec![input, vec![0u8; out_bytes]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    be.dispatch(
        &KernelCall::Cast(CastCall {
            input: rb(0),
            output: rb(1),
            element_count: n as u64,
            src_dtype: src,
            dst_dtype: dst,
        }),
        &mut ws,
    )
    .unwrap();
    ws.slots.remove(1)
}

/// int64 → f32: the canonical `input_ids` / index conversion. Exact for the
/// realistic range (|v| < 2²⁴).
#[test]
fn cast_i64_to_f32_is_exact() {
    let vals: [i64; 6] = [0, 1, -1, 255, -300, 1 << 20];
    let mut inp = Vec::new();
    for &v in &vals {
        inp.extend_from_slice(&v.to_le_bytes());
    }
    let out = cast(inp, vals.len(), DTYPE_I64, DTYPE_F32, vals.len() * 4);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(read_f32(&out, i), v as f32, "idx {i}");
    }
}

/// i32 → f32 likewise exact.
#[test]
fn cast_i32_to_f32_is_exact() {
    let vals: [i32; 4] = [7, -7, 1000, -32768];
    let mut inp = Vec::new();
    for &v in &vals {
        inp.extend_from_slice(&v.to_le_bytes());
    }
    let out = cast(inp, vals.len(), DTYPE_I32, DTYPE_F32, vals.len() * 4);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(read_f32(&out, i), v as f32, "idx {i}");
    }
}

/// f32 → i32 truncates toward zero (ONNX Cast semantics), not round.
#[test]
fn cast_f32_to_i32_truncates_toward_zero() {
    let vals: [f32; 5] = [1.9, -1.9, 2.5, -2.5, 0.0];
    let mut inp = vec![0u8; vals.len() * 4];
    for (i, &v) in vals.iter().enumerate() {
        write_f32(&mut inp, i, v);
    }
    let out = cast(inp, vals.len(), DTYPE_F32, DTYPE_I32, vals.len() * 4);
    let got: Vec<i32> = out
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, vec![1, -1, 2, -2, 0]);
}

/// f32 → bf16 (float width change): top 16 bits, value preserved within bf16.
#[test]
fn cast_f32_to_bf16_preserves_value() {
    let vals: [f32; 3] = [1.0, -2.5, 0.5];
    let mut inp = vec![0u8; vals.len() * 4];
    for (i, &v) in vals.iter().enumerate() {
        write_f32(&mut inp, i, v);
    }
    let out = cast(inp, vals.len(), DTYPE_F32, DTYPE_BF16, vals.len() * 2);
    for (i, &v) in vals.iter().enumerate() {
        // These values are exactly representable in bf16.
        assert_eq!(read_bf16(&out, i), v, "idx {i}");
    }
}
