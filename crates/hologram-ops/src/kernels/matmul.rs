//! Canonical `MatMul` op — semantic identity, executable form, and CPU
//! reference kernels (forward + 2 backwards).
//!
//! Reference (correctness-only) implementation. Tiling, vectorisation,
//! and BLAS dispatch are concerns for backend-specialised executors
//! that consume the same `KernelCall` form.

use crate::attrs::MatMulAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Marker struct for the canonical `matmul` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatMul(pub MatMulAttrs);

impl Op for MatMul {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "matmul"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::LinearAlgebra
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::MatMulBackward)
    }
}

/// Pre-resolved arguments for forward matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulCall {
    /// Left operand `A` (`[m, k]`).
    pub a: SlotSpan,
    /// Right operand `B` (`[k, n]`).
    pub b: SlotSpan,
    /// Output `C` (`[m, n]`).
    pub c: SlotSpan,
    /// Rows of `A` and `C`.
    pub m: usize,
    /// Inner dimension.
    pub k: usize,
    /// Cols of `B` and `C`.
    pub n: usize,
}

/// Pre-resolved arguments for `dA += dC @ Bᵀ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulGradACall {
    /// Upstream gradient `dC` (`[m, n]`).
    pub dc: SlotSpan,
    /// Forward `B` (`[k, n]`).
    pub b: SlotSpan,
    /// Gradient slot for `A` (`[m, k]`, accumulated).
    pub da: SlotSpan,
    /// Rows of `A`/`C`.
    pub m: usize,
    /// Inner dimension.
    pub k: usize,
    /// Cols of `B`/`C`.
    pub n: usize,
}

/// Pre-resolved arguments for `dB += Aᵀ @ dC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulGradBCall {
    /// Forward `A` (`[m, k]`).
    pub a: SlotSpan,
    /// Upstream gradient `dC` (`[m, n]`).
    pub dc: SlotSpan,
    /// Gradient slot for `B` (`[k, n]`, accumulated).
    pub db: SlotSpan,
    /// Rows of `A`/`C`.
    pub m: usize,
    /// Inner dimension.
    pub k: usize,
    /// Cols of `B`/`C`.
    pub n: usize,
}

/// Threshold (in floating-point ops, ~`m*k*n*2`) above which the
/// parallel matmul path is worth the rayon launch overhead.
///
/// Empirically calibrated on the decode-step bench: at decode shape
/// (`m == 1`, single token per call) the serial vectorised path runs
/// at ~30 GFLOP/s on a single Apple performance core, and rayon's
/// per-`for_each` overhead — including memory-bandwidth contention
/// when multiple threads share one workspace — measures ~400 µs in
/// practice on this machine. That makes the break-even matmul size
/// `30 GFLOP/s × 400 µs × 0.75 (worst-case scaling) ≈ 36 M flops`,
/// which exceeds *every* matmul in the medium decode block. So for
/// decode workloads the threshold-gated parallel path is effectively
/// never selected, and the feature is best regarded as a tool for
/// training-shape (`m » 1`, large batch) workloads where individual
/// matmuls are 100s of megaflops and the 4-core split actually wins.
///
/// Set deliberately above the largest decode-step matmul (16.8 M
/// flops at the medium shape) so enabling `--features parallel`
/// doesn't regress the decode bench. Lower it for training scenarios.
///
/// Only consulted when the `parallel` feature is enabled.
#[cfg(feature = "parallel")]
const PARALLEL_FLOP_THRESHOLD: usize = 50_000_000;

/// Forward: `C = A @ B` (`A:[m,k]`, `B:[k,n]`, `C:[m,n]`, row-major).
///
/// Loop order is `i-p-j` rather than the textbook `i-j-k`. With
/// row-major operands the inner `j` loop is unit-stride on both `B`
/// and `C`, which is a pattern Rust's autovectoriser handles cleanly
/// (one f32-vector load per iteration, one fma, one store). The
/// textbook order has `B` accessed at stride `n` in the inner loop —
/// every iteration hits a different cache line at large `n`, costing
/// ~5× on the bench's matmul shapes.
///
/// We zero `C` up front so each `(i, p, j)` body is an unconditional
/// fused-multiply-add. Skipping the zero would require a special
/// "first-p-iteration assigns, others accumulate" branch in the inner
/// loop, which kills vectorisation.
#[inline]
pub fn matmul(storage: &mut [f32], call: &MatMulCall) {
    debug_assert_eq!(call.a.len, call.m * call.k);
    debug_assert_eq!(call.b.len, call.k * call.n);
    debug_assert_eq!(call.c.len, call.m * call.n);

    #[cfg(feature = "parallel")]
    {
        // 2 flops per (i,p,j) in the inner loop (mul + add).
        if 2 * call.m * call.k * call.n >= PARALLEL_FLOP_THRESHOLD {
            matmul_parallel(storage, call);
            return;
        }
    }
    matmul_serial(storage, call);
}

/// Single-threaded `i-p-j` matmul. Hot loop in the serial path and
/// also the per-task body of the parallel path (re-used by clamping
/// `m` to one row at a time).
#[inline]
fn matmul_serial(storage: &mut [f32], call: &MatMulCall) {
    let m = call.m;
    let k = call.k;
    let n = call.n;
    let a_off = call.a.offset;
    let b_off = call.b.offset;
    let c_off = call.c.offset;

    storage[c_off..c_off + m * n].fill(0.0);
    if k == 0 {
        return;
    }

    for i in 0..m {
        let c_row = c_off + i * n;
        let a_row = a_off + i * k;
        for p in 0..k {
            let ap = storage[a_row + p];
            let b_row = b_off + p * n;
            for j in 0..n {
                // `s[i] += ap * s[j]` is a sequenced read-then-write
                // on the same slice — the borrow checker is fine with
                // it, and the compiler vectorises the inner loop. The
                // planner guarantees A, B, C spans are disjoint so
                // there's no semantic aliasing either.
                storage[c_row + j] += ap * storage[b_row + j];
            }
        }
    }
}

/// Multi-threaded matmul. Splits work across rayon's pool:
/// * `m == 1` (decode-shape GEMV): chunk `j` into roughly-equal
///   ranges, one per thread. Each task reads the whole of `A` (length
///   `k`) and a column-stripe of `B`, writes a disjoint stripe of `C`.
/// * `m > 1`: one task per row of `C`. Each task reads its row of
///   `A`, all of `B`, writes its disjoint row of `C`.
///
/// Both shapes use raw pointers under the hood because the borrow
/// checker can't see that the planner-allocated A/B/C spans are
/// disjoint regions of `storage`. The pointer is wrapped in a
/// `Send`/`Sync` newtype and cast back inside each task. SAFETY rests
/// on three invariants documented at the call site.
#[cfg(feature = "parallel")]
fn matmul_parallel(storage: &mut [f32], call: &MatMulCall) {
    use rayon::prelude::*;

    let m = call.m;
    let k = call.k;
    let n = call.n;
    let a_off = call.a.offset;
    let b_off = call.b.offset;
    let c_off = call.c.offset;

    storage[c_off..c_off + m * n].fill(0.0);
    if k == 0 {
        return;
    }

    // Stash the storage base pointer as a `usize`. Rust 2021's
    // capture analysis would otherwise look at `Sptr(*mut f32)` and
    // decide it only needs the inner field, which is `!Send + !Sync`.
    // A bare `usize` is trivially Send+Sync; we cast it back to
    // `*mut f32` inside each task. The address only needs to outlive
    // the rayon scope, which it does because `storage` is borrowed
    // for the duration of `matmul_parallel`.
    let storage_addr: usize = storage.as_mut_ptr() as usize;

    if m == 1 {
        // GEMV path: parallelise across columns of C. We iterate
        // over chunk *indices* — `step_by` on rayon's ParIter yields
        // a serial Iterator, so going via the index lets us stay on
        // the parallel side.
        let threads = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(threads).max(64);
        let num_chunks = n.div_ceil(chunk);
        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let j_start = chunk_idx * chunk;
            let j_end = (j_start + chunk).min(n);
            let p = storage_addr as *mut f32;
            for pp in 0..k {
                // SAFETY: `pp < k`, so `a_off + pp` is in A.
                // `j_start..j_end ⊆ [0, n)`, so
                // `b_off + pp*n + j` is in B and `c_off + j` is in
                // C — both planner-disjoint from each other and
                // from A. Only this task writes `c_off + j` for
                // `j ∈ [j_start, j_end)`; sibling tasks own
                // disjoint ranges. Reads of A and B from sibling
                // tasks are immutable accesses to the same memory,
                // which is sound under rayon's scoped concurrency.
                unsafe {
                    let ap = *p.add(a_off + pp);
                    let b_row = b_off + pp * n;
                    for j in j_start..j_end {
                        *p.add(c_off + j) += ap * *p.add(b_row + j);
                    }
                }
            }
        });
    } else {
        // General path: parallelise across rows of C.
        (0..m).into_par_iter().for_each(|i| {
            let p = storage_addr as *mut f32;
            // SAFETY: same contract as the GEMV branch — each task
            // owns a unique row `i` of C; A/B reads are immutable.
            unsafe {
                let c_row = c_off + i * n;
                let a_row = a_off + i * k;
                for pp in 0..k {
                    let ap = *p.add(a_row + pp);
                    let b_row = b_off + pp * n;
                    for j in 0..n {
                        *p.add(c_row + j) += ap * *p.add(b_row + j);
                    }
                }
            }
        });
    }
}

/// Backward w.r.t. A: `dA += dC @ Bᵀ` (`dA:[m,k]`).
#[inline]
pub fn matmul_grad_a(storage: &mut [f32], call: &MatMulGradACall) {
    if call.da.len == 0 {
        return;
    }
    debug_assert_eq!(call.da.len, call.m * call.k);
    for i in 0..call.m {
        for p in 0..call.k {
            let acc = dot_dc_bt(storage, call, i, p);
            storage[call.da.offset + i * call.k + p] += acc;
        }
    }
}

#[inline]
fn dot_dc_bt(storage: &[f32], call: &MatMulGradACall, i: usize, p: usize) -> f32 {
    let mut acc = 0.0_f32;
    for j in 0..call.n {
        let dc = storage[call.dc.offset + i * call.n + j];
        let bv = storage[call.b.offset + p * call.n + j];
        acc += dc * bv;
    }
    acc
}

/// Backward w.r.t. B: `dB += Aᵀ @ dC` (`dB:[k,n]`).
///
/// Same i-p-j loop reorder as the forward — the textbook
/// `dB[p, j] += Σ_i A[i, p] * dC[i, j]` order is bad here because
/// both `A[i, p]` (stride `k`) and `dC[i, j]` (stride `n`) are
/// strided in the inner `i` loop. Reordering the outer loops to
/// `i-p-j` makes the inner `j` loop unit-stride on `dC` and `dB`
/// with `A[i, p]` hoisted as a per-iteration broadcast, which the
/// autovectoriser handles cleanly.
#[inline]
pub fn matmul_grad_b(storage: &mut [f32], call: &MatMulGradBCall) {
    if call.db.len == 0 {
        return;
    }
    debug_assert_eq!(call.db.len, call.k * call.n);
    let m = call.m;
    let k = call.k;
    let n = call.n;
    let a_off = call.a.offset;
    let dc_off = call.dc.offset;
    let db_off = call.db.offset;

    // Note: dB is `+=`-accumulated, not zeroed — caller-managed.
    for i in 0..m {
        let dc_row = dc_off + i * n;
        let a_row = a_off + i * k;
        for p in 0..k {
            let ap = storage[a_row + p];
            let db_row = db_off + p * n;
            for j in 0..n {
                storage[db_row + j] += ap * storage[dc_row + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(off: usize, len: usize) -> SlotSpan {
        SlotSpan { offset: off, len }
    }

    #[test]
    fn op_trait_matmul_declares_backward() {
        let mm = MatMul(MatMulAttrs { m: 2, k: 3, n: 4 });
        assert_eq!(mm.arity(), 2);
        assert_eq!(mm.name(), "matmul");
        assert_eq!(mm.category(), OpCategory::LinearAlgebra);
        assert_eq!(mm.backward(), Some(BackwardRule::MatMulBackward));
    }

    #[test]
    fn op_trait_signature_is_consistent_with_category() {
        let sig = MatMul(MatMulAttrs { m: 1, k: 1, n: 1 }).signature();
        assert_eq!(sig.arity, 2);
        assert_eq!(sig.outputs, 1);
        assert_eq!(sig.category, OpCategory::LinearAlgebra);
        assert!(sig.differentiable);
        assert!(!sig.layout_only);
    }

    #[test]
    fn matmul_2x3_times_3x2_matches_reference() {
        let mut s = vec![0.0_f32; 6 + 6 + 4];
        s[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        s[6..12].copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let call = MatMulCall {
            a: span(0, 6),
            b: span(6, 6),
            c: span(12, 4),
            m: 2,
            k: 3,
            n: 2,
        };
        matmul(&mut s, &call);
        assert_eq!(&s[12..16], &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn matmul_grad_a_accumulates_dc_b_transpose() {
        let mut s = vec![0.0_f32; 6 + 4 + 6];
        s[..6].copy_from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        s[6..10].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let call = MatMulGradACall {
            dc: span(6, 4),
            b: span(0, 6),
            da: span(10, 6),
            m: 2,
            k: 3,
            n: 2,
        };
        matmul_grad_a(&mut s, &call);
        assert_eq!(&s[10..16], &[17.0, 23.0, 29.0, 39.0, 53.0, 67.0]);
    }

    #[test]
    fn matmul_grad_b_accumulates_a_transpose_dc() {
        let mut s = vec![0.0_f32; 6 + 4 + 6];
        s[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        s[6..10].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let call = MatMulGradBCall {
            a: span(0, 6),
            dc: span(6, 4),
            db: span(10, 6),
            m: 2,
            k: 3,
            n: 2,
        };
        matmul_grad_b(&mut s, &call);
        assert_eq!(&s[10..16], &[13.0, 18.0, 17.0, 24.0, 21.0, 30.0]);
    }
}
