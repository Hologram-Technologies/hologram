//! **Performance V&V (class PV) — no silent bottleneck at scale.**
//!
//! Performance budgets that catch a part that has *broken down* into a
//! bottleneck — fallen off the SIMD fast path, gone super-cubic, or
//! degenerated — without being flaky on micro-regressions. Absolute
//! GFLOP/s is machine-dependent, so we use:
//!
//! * **PV-1** a *conservative* throughput floor (best-of-N) that any
//!   working vectorized matmul clears by 10×+, but a scalar-fallback /
//!   pathological path would miss; and
//! * **PV-1b** a scaling-shape bound: 128³ vs 64³ time stays within the
//!   cubic envelope (catching super-cubic blow-up or a degenerate
//!   short-cut that doesn't actually scale).
//!
//! Performance budgets are only meaningful on optimized builds, so this
//! suite is release-only (`cargo test --release`); in debug it compiles to
//! zero tests.
#![cfg(not(debug_assertions))]

use std::time::Instant;

use hologram_backend::cpu::dtype::DTYPE_F32;
use hologram_backend::SplitReads;
use hologram_backend::{
    AttentionCall, Backend, BufferRef, Conv2dCall, CpuBackend, KernelCall, MatMulCall, Workspace,
};

struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}
impl Workspace for TestWorkspace {
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

fn buf(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 0,
    }
}

/// Best-of-N wall-clock for one square f32 matmul of dimension `dim`.
fn matmul_best_secs(dim: usize, runs: usize) -> f64 {
    let bytes = dim * dim * 4;
    let a = vec![0x3f; bytes]; // ~0.5 f32 pattern; values irrelevant to timing
    let b = vec![0x3e; bytes];
    let mut ws = TestWorkspace {
        slots: vec![a, b, vec![0u8; bytes]],
    };
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::MatMul(MatMulCall {
        a: buf(0),
        b: buf(1),
        output: buf(2),
        m: dim as u32,
        k: dim as u32,
        n: dim as u32,
        dtype: DTYPE_F32,
        b_packed: false,
    });
    // Warm up, then take the minimum (most stable under CI load).
    backend.dispatch(&call, &mut ws).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        backend.dispatch(&call, &mut ws).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
fn pv1_matmul_throughput_floor_and_scaling() {
    let t64 = matmul_best_secs(64, 9);
    let t256 = matmul_best_secs(256, 5);

    // PV-1: conservative GFLOP/s floor at 256³. A vectorized kernel does
    // tens of GFLOP/s; this floor (1 GFLOP/s) only fails on a
    // catastrophic bottleneck (scalar fallback gone wrong / pathological).
    let flops_256 = 2.0 * (256f64).powi(3);
    let gflops_256 = flops_256 / t256 / 1e9;
    assert!(
        gflops_256 >= 1.0,
        "matmul 256³ at {gflops_256:.2} GFLOP/s — below the 1 GFLOP/s floor; a part has become a bottleneck"
    );

    // PV-1b: scaling envelope. 256³ has 64× the FLOPs of 64³. A correct
    // kernel lands near 64× (often more, from cache effects); we bound it
    // generously in [16×, 400×] to catch a degenerate short-cut (ratio →
    // ~1, not actually scaling) or super-cubic blow-up.
    let ratio = t256 / t64;
    assert!(
        (16.0..=400.0).contains(&ratio),
        "matmul 256³/64³ time ratio {ratio:.1} outside the cubic envelope [16,400] — \
         short-cutting or breaking down at scale"
    );
}

fn best_secs(call: &KernelCall, slots: &[Vec<u8>], runs: usize) -> f64 {
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let mut ws = TestWorkspace {
        slots: slots.to_vec(),
    };
    backend.dispatch(call, &mut ws).unwrap(); // warm
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        backend.dispatch(call, &mut ws).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// PV-3: every *heavy* compute kernel (beyond matmul) carries a budget, so
/// no part is a silent bottleneck. Floors are deliberately conservative
/// (0.1 GFLOP/s — ~100× below a working kernel) to catch catastrophic
/// breakdown without micro-flakiness.
#[test]
fn pv3_conv_and_attention_throughput_floors() {
    // Conv2d: b=4, cin=cout=16, 32×32, 3×3 valid → ho=wo=30.
    let (b, cin, cout, hi, wi, kh, kw) =
        (4usize, 16usize, 16usize, 32usize, 32usize, 3usize, 3usize);
    let (ho, wo) = (hi - kh + 1, wi - kw + 1);
    let conv = KernelCall::Conv2d(Conv2dCall {
        x: buf(0),
        w: buf(1),
        output: buf(2),
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
    });
    let conv_slots = vec![
        vec![0x3e; b * cin * hi * wi * 4],
        vec![0x3d; cout * cin * kh * kw * 4],
        vec![0u8; b * cout * ho * wo * 4],
    ];
    let conv_flops = 2.0 * (b * cout * ho * wo * cin * kh * kw) as f64;
    let conv_g = conv_flops / best_secs(&conv, &conv_slots, 7) / 1e9;
    assert!(
        conv_g >= 0.1,
        "conv2d at {conv_g:.3} GFLOP/s — below floor; bottleneck"
    );

    // Attention: b=2, h=8, s=128, d=64.
    let (ab, ah, asq, ad) = (2usize, 8usize, 128usize, 64usize);
    let n = ab * ah * asq * ad;
    let attn = KernelCall::Attention(AttentionCall {
        q: buf(0),
        k: buf(1),
        v: buf(2),
        output: buf(3),
        batch: ab as u32,
        heads: ah as u32,
        seq: asq as u32,
        head_dim: ad as u32,
        dtype: DTYPE_F32,
    });
    let attn_slots = vec![
        vec![0x3e; n * 4],
        vec![0x3d; n * 4],
        vec![0x3c; n * 4],
        vec![0u8; n * 4],
    ];
    // QKᵀ (2·s²·d) + AV (2·s²·d) per (b,h).
    let attn_flops = 4.0 * (ab * ah * asq * asq * ad) as f64;
    let attn_g = attn_flops / best_secs(&attn, &attn_slots, 7) / 1e9;
    assert!(
        attn_g >= 0.1,
        "attention at {attn_g:.3} GFLOP/s — below floor; bottleneck"
    );
}
