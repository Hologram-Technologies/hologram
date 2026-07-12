#![cfg(feature = "parallel")]
//! Concurrent-session soundness witness (upstream issue, v0.9.1): multiple
//! hosts driving pooled kernels **concurrently in one process** must be
//! exactly as correct as driving them sequentially. v0.9.0's process-global
//! pool let a publisher's help-drain execute *another* walk's tasks — nesting
//! thread-local scratch borrows (`RefCell already borrowed`) and, on the
//! unwind, orphaning the panicked batch's queued tasks in the global queue
//! with raw pointers into a dead walk: silent cross-session corruption.
//!
//! The witness precomputes each worker's expected outputs sequentially
//! (pooled == serial is pinned elsewhere), then hammers the pool from
//! several threads at once — one flooding pooled f32 GEMMs (the publisher
//! holds its `bt` scratch borrow across the drain), the others flooding
//! pooled decode attention (whose tasks borrow the same thread-local) — and
//! asserts every iteration's output is bit-identical to the sequential
//! baseline. Any panic or divergence is the defect.

use hologram_backend::{
    Backend, BufferRef, CpuBackend, DecodeAttentionCall, KernelCall, MatMulCall, SplitReads,
    Workspace,
};
use std::thread;

const DTYPE_F32: u8 = 8;

fn to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 19) % 61) as f32 - 30.0) * 0.017)
        .collect()
}

struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}
impl TestWorkspace {
    fn push(&mut self, data: &[u8]) -> BufferRef {
        let slot = self.slots.len() as u32;
        self.slots.push(data.to_vec());
        BufferRef {
            slot,
            offset: 0,
            length: data.len() as u64,
        }
    }
}
impl Workspace for TestWorkspace {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize][..]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let len = self.slots[b.slot as usize].len();
        let _ = b.length;
        &mut self.slots[b.slot as usize][..len]
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

/// One pooled f32 GEMM: `m` large enough that `matmul_f32_blocked` tiles
/// across the pool while the publisher holds its `bt` scratch borrow.
fn run_gemm(seed: usize) -> Vec<u8> {
    let (m, k, n) = (64usize, 256usize, 512usize);
    let a = f32s(m * k, seed);
    let b = f32s(k * n, seed + 1);
    let mut ws = TestWorkspace { slots: Vec::new() };
    let ra = ws.push(&to_le(&a));
    let rb = ws.push(&to_le(&b));
    let ro = ws.push(&vec![0u8; m * n * 4]);
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(
        &KernelCall::MatMul(MatMulCall {
            a: ra,
            b: rb,
            output: ro,
            m: m as u32,
            k: k as u32,
            n: n as u32,
            dtype: DTYPE_F32,
            b_packed: false,
        }),
        &mut ws,
    )
    .unwrap();
    ws.slots[ro.slot as usize].clone()
}

/// One pooled decode attention: rows·l·d above the parallel work gate, so
/// its tasks run on the pool and each borrows the thread-local scratch.
fn run_attn(seed: usize) -> Vec<u8> {
    let (b, h, hkv, m, past, new, d) = (1u32, 8u32, 2u32, 1u32, 1024u32, 1u32, 128u32);
    let l = (past + new) as usize;
    let mask: Vec<f32> = (0..m as usize * l)
        .map(|i| {
            if i % l < 900 || i % l == past as usize {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rq = ws.push(&to_le(&f32s((b * h * m * d) as usize, seed)));
    let rkp = ws.push(&to_le(&f32s((b * hkv * past * d) as usize, seed + 2)));
    let rvp = ws.push(&to_le(&f32s((b * hkv * past * d) as usize, seed + 3)));
    let rkn = ws.push(&to_le(&f32s((b * hkv * new * d) as usize, seed + 4)));
    let rvn = ws.push(&to_le(&f32s((b * hkv * new * d) as usize, seed + 5)));
    let rm = ws.push(&to_le(&mask));
    let ro = ws.push(&vec![0u8; (b * h * m * d) as usize * 4]);
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(
        &KernelCall::DecodeAttention(DecodeAttentionCall {
            q: rq,
            k_past: rkp,
            v_past: rvp,
            k_new: rkn,
            v_new: rvn,
            mask: rm,
            output: ro,
            batch: b,
            heads: h,
            kv_heads: hkv,
            q_rows: m,
            past_len: past,
            new_len: new,
            head_dim: d,
            scale_bits: 0,
            dtype: DTYPE_F32,
        }),
        &mut ws,
    )
    .unwrap();
    ws.slots[ro.slot as usize].clone()
}

/// **The soundness witness.** Sequential baselines first; then four threads
/// hammer the pool concurrently — two GEMM walks, two decode-attention
/// walks — and every iteration must be bit-identical to its baseline.
/// v0.9.0 fails here with `RefCell already borrowed` (publisher drain
/// executing a foreign task under a held scratch borrow) or with silent
/// divergence from the orphaned-task fallout; either is the defect.
#[test]
fn concurrent_pooled_walks_are_bitwise_equal_to_sequential() {
    const ITERS: usize = 60;
    // Sequential baselines (one walk at a time — pinned-correct regime).
    let gemm_want: Vec<Vec<u8>> = (0..2).map(|t| run_gemm(100 + t)).collect();
    let attn_want: Vec<Vec<u8>> = (0..2).map(|t| run_attn(200 + t)).collect();

    thread::scope(|s| {
        for t in 0..2usize {
            let want = gemm_want[t].clone();
            s.spawn(move || {
                for i in 0..ITERS {
                    let got = run_gemm(100 + t);
                    assert_eq!(
                        got, want,
                        "gemm thread {t} iter {i}: concurrent result diverged from sequential"
                    );
                }
            });
            let want = attn_want[t].clone();
            s.spawn(move || {
                for i in 0..ITERS {
                    let got = run_attn(200 + t);
                    assert_eq!(
                        got, want,
                        "attn thread {t} iter {i}: concurrent result diverged from sequential"
                    );
                }
            });
        }
    });
}
