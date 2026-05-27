//! Runtime-indexed `Gather` kernel conformance (ONNX `Gather` / embedding).
//!
//! The kernel realizes `out[o, k, :] = data[o, indices[k], :]` as a direct
//! indexed row copy — `O(outer·num_indices·inner)`, the `axis_dim`-times-cheaper
//! replacement for the `OneHot(indices)·data` matmul. These tests pin the
//! row-exact semantics, ONNX negative-index wrapping, the i64 index width, and
//! the fail-loud out-of-range guard against the spec.

use hologram_backend::cpu::dtype::{read_f32, write_f32, DTYPE_F32, DTYPE_I32, DTYPE_I64};
use hologram_backend::{
    Backend, BufferRef, CpuBackend, GatherCall, KernelCall, SplitReads, Workspace,
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

/// `data [V, D]` f32 table, `indices [S]` i32, axis 0 (embedding lookup):
/// `out[k, :] = data[indices[k], :]`. Covers a repeated index and an
/// out-of-order selection.
#[test]
fn gather_embedding_rows_axis0() {
    let v = 5usize;
    let d = 3usize;
    // data[r, c] = r * 10 + c
    let mut data = vec![0u8; v * d * 4];
    for r in 0..v {
        for c in 0..d {
            write_f32(&mut data, r * d + c, (r * 10 + c) as f32);
        }
    }
    let idx_vals: [i32; 4] = [2, 0, 4, 2];
    let mut idx = Vec::new();
    for &i in &idx_vals {
        idx.extend_from_slice(&i.to_le_bytes());
    }
    let s = idx_vals.len();

    let mut ws = Ws {
        slots: vec![data, idx, vec![0u8; s * d * 4]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    be.dispatch(
        &KernelCall::Gather(GatherCall {
            data: rb(0),
            indices: rb(1),
            output: rb(2),
            outer: 1,
            axis_dim: v as u64,
            inner: d as u64,
            num_indices: s as u64,
            idx_dtype: DTYPE_I32,
            dtype: DTYPE_F32,
        }),
        &mut ws,
    )
    .unwrap();

    for (k, &want_row) in idx_vals.iter().enumerate() {
        for c in 0..d {
            let got = read_f32(&ws.slots[2], k * d + c);
            assert_eq!(got, (want_row as usize * 10 + c) as f32, "row {k} col {c}");
        }
    }
}

/// ONNX permits negative indices counting from the end of the axis; the kernel
/// wraps them by `axis_dim`. `i64` indices (the real-LM `input_ids` width) must
/// read 8 bytes per element.
#[test]
fn gather_negative_indices_and_i64_width() {
    let v = 4usize;
    let d = 2usize;
    let mut data = vec![0u8; v * d * 4];
    for r in 0..v {
        for c in 0..d {
            write_f32(&mut data, r * d + c, (r * 100 + c) as f32);
        }
    }
    // -1 → row 3, -4 → row 0.
    let idx_vals: [i64; 3] = [-1, -4, 1];
    let mut idx = Vec::new();
    for &i in &idx_vals {
        idx.extend_from_slice(&i.to_le_bytes());
    }
    let s = idx_vals.len();
    let mut ws = Ws {
        slots: vec![data, idx, vec![0u8; s * d * 4]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    be.dispatch(
        &KernelCall::Gather(GatherCall {
            data: rb(0),
            indices: rb(1),
            output: rb(2),
            outer: 1,
            axis_dim: v as u64,
            inner: d as u64,
            num_indices: s as u64,
            idx_dtype: DTYPE_I64,
            dtype: DTYPE_F32,
        }),
        &mut ws,
    )
    .unwrap();

    let expect_rows = [3usize, 0, 1];
    for (k, &r) in expect_rows.iter().enumerate() {
        for c in 0..d {
            assert_eq!(
                read_f32(&ws.slots[2], k * d + c),
                (r * 100 + c) as f32,
                "row {k} col {c}"
            );
        }
    }
}

/// An index outside `[-axis_dim, axis_dim)` must fail loud, not gather garbage.
#[test]
fn gather_out_of_range_index_fails_loud() {
    let v = 3usize;
    let d = 1usize;
    let data = vec![0u8; v * d * 4];
    let idx: Vec<u8> = 7i32.to_le_bytes().to_vec(); // 7 >= axis_dim = 3
    let mut ws = Ws {
        slots: vec![data, idx, vec![0u8; d * 4]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    let r = be.dispatch(
        &KernelCall::Gather(GatherCall {
            data: rb(0),
            indices: rb(1),
            output: rb(2),
            outer: 1,
            axis_dim: v as u64,
            inner: d as u64,
            num_indices: 1,
            idx_dtype: DTYPE_I32,
            dtype: DTYPE_F32,
        }),
        &mut ws,
    );
    assert!(r.is_err(), "out-of-range index must error");
}

/// `axis > 0`: a non-trivial `outer` slices the gather per outer block.
/// `data [outer=2, axis_dim=3, inner=2]`, gather axis-1 indices `[2, 0]`.
#[test]
fn gather_axis1_with_outer() {
    let outer = 2usize;
    let axis_dim = 3usize;
    let inner = 2usize;
    let mut data = vec![0u8; outer * axis_dim * inner * 4];
    for o in 0..outer {
        for a in 0..axis_dim {
            for i in 0..inner {
                let lin = (o * axis_dim + a) * inner + i;
                write_f32(&mut data, lin, (o * 1000 + a * 10 + i) as f32);
            }
        }
    }
    let idx_vals: [i32; 2] = [2, 0];
    let mut idx = Vec::new();
    for &i in &idx_vals {
        idx.extend_from_slice(&i.to_le_bytes());
    }
    let num_idx = idx_vals.len();
    let mut ws = Ws {
        slots: vec![data, idx, vec![0u8; outer * num_idx * inner * 4]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    be.dispatch(
        &KernelCall::Gather(GatherCall {
            data: rb(0),
            indices: rb(1),
            output: rb(2),
            outer: outer as u64,
            axis_dim: axis_dim as u64,
            inner: inner as u64,
            num_indices: num_idx as u64,
            idx_dtype: DTYPE_I32,
            dtype: DTYPE_F32,
        }),
        &mut ws,
    )
    .unwrap();

    for o in 0..outer {
        for (k, &a) in idx_vals.iter().enumerate() {
            for i in 0..inner {
                let lin = (o * num_idx + k) * inner + i;
                assert_eq!(
                    read_f32(&ws.slots[2], lin),
                    (o * 1000 + a as usize * 10 + i) as f32,
                    "o {o} k {k} i {i}"
                );
            }
        }
    }
}
