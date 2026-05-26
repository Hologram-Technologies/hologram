//! **Zero-overhead contract V&V (spec XII.5).**
//!
//! The cache-oblivious engine and every dtype-widen / im2col / score buffer it
//! uses are **reused thread-locals**, so after a single warm-up the compute hot
//! path performs **zero heap allocations per call** — an inference loop pays
//! O(1) total allocations, not O(calls). This is the zero-cost / zero-copy
//! contract made executable: a regression that reintroduces a per-call `Vec`,
//! `to_vec()`, or marshalling copy fails this test immediately.
//!
//! Verified with a per-thread counting global allocator: only the arming
//! thread's allocations inside the measured window are counted, so concurrent
//! tests in the same binary cannot perturb the result. Allocation counts are
//! build-independent, so this runs in both debug and release.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use hologram_backend::cpu::dtype::DTYPE_F32;
use hologram_backend::{
    broadcast_op, AttentionCall, Backend, BroadcastBinaryCall, BufferRef, Conv2dCall, CpuBackend,
    GemmCall, KernelCall, MatMulCall, MatMulDequantCall, SplitReads, Workspace,
};

// ── per-thread allocation-counting allocator ──────────────────────────
struct Counting;
thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}
// Cells (const-init, no heap) keep the counter re-entrancy-free inside `alloc`.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ARMED.with(Cell::get) {
            ALLOCS.with(|c| c.set(c.get() + 1));
        }
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, ns: usize) -> *mut u8 {
        if ARMED.with(Cell::get) {
            ALLOCS.with(|c| c.set(c.get() + 1));
        }
        System.realloc(p, l, ns)
    }
}
#[global_allocator]
static GA: Counting = Counting;

/// Allocations performed by `f` on the current thread.
fn count_allocs(f: impl FnOnce()) -> usize {
    ALLOCS.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    ALLOCS.with(Cell::get)
}

// ── minimal workspace ─────────────────────────────────────────────────
struct Ws {
    slots: Vec<Vec<u8>>,
}
impl Workspace for Ws {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize][..]
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
fn bz(s: u32) -> BufferRef {
    BufferRef {
        slot: s,
        offset: 0,
        length: 0,
    }
}

/// Dispatch once to warm thread-local scratch, then assert the next identical
/// dispatch allocates **nothing**.
fn assert_zero_alloc_after_warmup(op: &str, call: KernelCall, slots: Vec<Vec<u8>>) {
    let mut backend: CpuBackend<Ws> = CpuBackend::new();
    let mut ws = Ws { slots };
    backend.dispatch(&call, &mut ws).unwrap(); // warm-up sizes the scratch
    let n = count_allocs(|| {
        backend.dispatch(&call, &mut ws).unwrap();
    });
    assert_eq!(
        n, 0,
        "{op} performed {n} heap allocation(s) after warm-up — the zero-overhead \
         contract is violated (a per-call Vec / copy crept into the hot path)"
    );
}

#[test]
fn matmul_hotpath_is_zero_alloc() {
    let d = 96usize;
    assert_zero_alloc_after_warmup(
        "matmul",
        KernelCall::MatMul(MatMulCall {
            a: bz(0),
            b: bz(1),
            output: bz(2),
            m: d as u32,
            k: d as u32,
            n: d as u32,
            dtype: DTYPE_F32,
            b_packed: false,
        }),
        vec![
            vec![0x3e; d * d * 4],
            vec![0x3d; d * d * 4],
            vec![0u8; d * d * 4],
        ],
    );
}

#[test]
fn gemm_hotpath_is_zero_alloc() {
    let d = 64usize;
    assert_zero_alloc_after_warmup(
        "gemm",
        KernelCall::Gemm(GemmCall {
            a: bz(0),
            b: bz(1),
            c: bz(2),
            output: bz(3),
            m: d as u32,
            k: d as u32,
            n: d as u32,
            alpha_bits: 1.0f32.to_bits() as u64,
            beta_bits: 1.0f32.to_bits() as u64,
            dtype: DTYPE_F32,
        }),
        vec![
            vec![0x3e; d * d * 4],
            vec![0x3d; d * d * 4],
            vec![0x3c; d * d * 4],
            vec![0u8; d * d * 4],
        ],
    );
}

#[test]
fn conv2d_hotpath_is_zero_alloc() {
    // The im2col patch matrix is a reused thread-local — after warm-up the
    // per-batch GEMM convolution must not allocate.
    let (b, cin, cout, hi, wi, kh, kw) = (2usize, 4, 8, 16, 16, 3, 3);
    let (ho, wo) = (hi - kh + 1, wi - kw + 1);
    assert_zero_alloc_after_warmup(
        "conv2d",
        KernelCall::Conv2d(Conv2dCall {
            x: bz(0),
            w: bz(1),
            output: bz(2),
            batch: b as u32,
            channels_in: cin as u32,
            channels_out: cout as u32,
            h_in: hi as u32,
            w_in: wi as u32,
            h_out: ho as u32,
            w_out: wo as u32,
            k_h: kh as u32,
            k_w: kw as u32,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dtype: DTYPE_F32,
        }),
        vec![
            vec![0x3e; b * cin * hi * wi * 4],
            vec![0x3d; cout * cin * kh * kw * 4],
            vec![0u8; b * cout * ho * wo * 4],
        ],
    );
}

#[test]
fn matmul_dequant_hotpath_is_zero_alloc() {
    // Fused dequant→matmul: A[d,d] f32 · dequant(Bq[d,d] i8). The dequantized
    // panel must come from reused thread-local scratch, not a per-call Vec.
    let d = 96usize;
    assert_zero_alloc_after_warmup(
        "matmul_dequant",
        KernelCall::MatMulDequant(MatMulDequantCall {
            a: bz(0),
            bq: bz(1),
            // Per-tensor (channels = 0): scale/zp vectors are not read.
            scales: bz(u32::MAX),
            zero_points: bz(u32::MAX),
            output: bz(2),
            m: d as u32,
            k: d as u32,
            n: d as u32,
            channels: 0,
            inner: 0,
            quant_dtype: 2, // i8
            dtype: DTYPE_F32,
            scale_bits: 0.5f32.to_bits(),
            zero_point: 0,
        }),
        vec![
            vec![0x3e; d * d * 4], // A (f32)
            vec![0x02; d * d],     // Bq (i8)
            vec![0u8; d * d * 4],  // out
        ],
    );
}

#[test]
fn broadcast_binary_hotpath_is_zero_alloc() {
    // Fused Expand→Mul: small[1,d] broadcast over [d,d] times other[d,d].
    let d = 96usize;
    let mut in_dims = [0u32; 8];
    let mut out_dims = [0u32; 8];
    in_dims[0] = 1;
    in_dims[1] = d as u32;
    out_dims[0] = d as u32;
    out_dims[1] = d as u32;
    assert_zero_alloc_after_warmup(
        "broadcast_binary",
        KernelCall::BroadcastBinary(BroadcastBinaryCall {
            small: bz(0),
            other: bz(1),
            output: bz(2),
            rank: 2,
            in_dims,
            out_dims,
            op: broadcast_op::MUL,
            small_is_lhs: true,
            dtype: DTYPE_F32,
        }),
        vec![
            vec![0x3e; d * 4],     // small (1×d f32)
            vec![0x3d; d * d * 4], // other (d×d f32)
            vec![0u8; d * d * 4],  // out
        ],
    );
}

#[test]
fn attention_hotpath_is_zero_alloc() {
    let (ab, ah, asq, ad) = (2usize, 4, 32, 32);
    let n = ab * ah * asq * ad;
    assert_zero_alloc_after_warmup(
        "attention",
        KernelCall::Attention(AttentionCall {
            q: bz(0),
            k: bz(1),
            v: bz(2),
            output: bz(3),
            batch: ab as u32,
            heads: ah as u32,
            seq: asq as u32,
            head_dim: ad as u32,
            dtype: DTYPE_F32,
        }),
        vec![
            vec![0x3e; n * 4],
            vec![0x3d; n * 4],
            vec![0x3c; n * 4],
            vec![0u8; n * 4],
        ],
    );
}
