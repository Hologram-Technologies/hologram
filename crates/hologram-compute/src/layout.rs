//! Weight-layout transforms — the data-representation half of hologram's
//! compile-time monomorphism.
//!
//! The layout is defined **here, once**, and is the single source of truth
//! shared by the producer (the compiler's weight-packing pass, which runs
//! [`pack_b_panels_bytes`] on a constant weight) and the consumer (the CPU
//! kernel [`crate::cpu::simd::matmul_f32_packed`], which streams the packed
//! panels contiguously). It is **not** CPU-feature-gated and pulls no SIMD or
//! `bytemuck` dependency — it is pure index arithmetic over raw element bytes,
//! so the (backend-agnostic) compiler can produce it.

use alloc::vec::Vec;

/// Panel width — the kernel's `NR` register-tile column count. B is packed
/// into panels of this many columns.
pub const PANEL: usize = 16;

/// Panel-pack a row-major `k×n` matrix of `elem`-byte elements into
/// `⌈n/PANEL⌉` column panels, each `k`-contiguous: the packed element at
/// `(p·k + kk)·PANEL + c` is the source element at `[kk·n + p·PANEL + c]`,
/// zero-padded where `p·PANEL + c ≥ n`. Dtype-agnostic (operates on element
/// bytes), so the compiler can pack any constant weight without interpreting
/// it. Packed length is `⌈n/PANEL⌉·k·PANEL·elem` bytes.
#[must_use]
pub fn pack_b_panels_bytes(src: &[u8], k: usize, n: usize, elem: usize) -> Vec<u8> {
    let n_panels = n.div_ceil(PANEL);
    let mut out = alloc::vec![0u8; n_panels * k * PANEL * elem];
    for p in 0..n_panels {
        let cols = core::cmp::min(PANEL, n - p * PANEL);
        for kk in 0..k {
            let dst = ((p * k + kk) * PANEL) * elem;
            let s = (kk * n + p * PANEL) * elem;
            out[dst..dst + cols * elem].copy_from_slice(&src[s..s + cols * elem]);
        }
    }
    out
}

/// Packed length in elements for a `k×n` matrix: `⌈n/PANEL⌉·k·PANEL`.
#[must_use]
pub fn packed_len(k: usize, n: usize) -> usize {
    n.div_ceil(PANEL) * k * PANEL
}

/// f32 convenience wrapper (kernel-side / tests). Same layout as
/// [`pack_b_panels_bytes`] with `elem = 4`.
#[must_use]
pub fn pack_b_panels(b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let n_panels = n.div_ceil(PANEL);
    let mut out = alloc::vec![0f32; n_panels * k * PANEL];
    for p in 0..n_panels {
        let cols = core::cmp::min(PANEL, n - p * PANEL);
        for kk in 0..k {
            let dst = (p * k + kk) * PANEL;
            let s = kk * n + p * PANEL;
            out[dst..dst + cols].copy_from_slice(&b[s..s + cols]);
        }
    }
    out
}
