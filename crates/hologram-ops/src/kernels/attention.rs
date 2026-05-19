//! Canonical scaled dot-product `Attention` op (ADR-049).
//!
//! Forward: `out = softmax((Q @ Kᵀ) * scale + mask) @ V`.
//!
//! Layout (heads-first, 4-D):
//! - Q : `[batch, num_q_heads, seq_q, head_dim]`
//! - K : `[batch, num_kv_heads, seq_kv, head_dim]`
//! - V : `[batch, num_kv_heads, seq_kv, head_dim]`
//! - out: `[batch, num_q_heads, seq_q, head_dim]`
//!
//! GQA / MQA: each Q head `qh` reads from KV head
//! `qh * num_kv_heads / num_q_heads`. Causal: position `k` masked
//! for query `q` iff `k > q + (seq_kv - seq_q)` (collapses to
//! standard upper-triangular for self-attention).
//!
//! No fusion, no sparsity, no kv-cache integration — those are
//! execution-side concerns or upstream canonical ops (RoPE,
//! RmsNorm).

use crate::attrs::AttentionAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

/// Pre-resolved arguments for the canonical attention kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionCall {
    /// Q span.
    pub q: SlotSpan,
    /// K span.
    pub k: SlotSpan,
    /// V span.
    pub v: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Workspace scratch (length ≥ `seq_kv`). The forward kernel uses
    /// it as the per-row softmax-weights buffer. An empty span means
    /// "no planner-supplied scratch, allocate locally" — used by the
    /// kernel's stand-alone unit tests.
    pub scratch: SlotSpan,
    /// Combined batch (product of leading dims before `num_q_heads`).
    pub batch: u32,
    /// Number of Q heads.
    pub num_q_heads: u32,
    /// Number of KV heads (`num_q_heads % num_kv_heads == 0`).
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Q sequence length.
    pub seq_q: u32,
    /// K/V sequence length.
    pub seq_kv: u32,
    /// `scale` (typically 1/√head_dim), encoded as `f32::to_bits()`.
    pub scale_bits: u32,
    /// Causal mask flag.
    pub causal: bool,
}

/// Marker struct for the canonical scaled-dot-product `attention` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Attention(pub AttentionAttrs);

impl Op for Attention {
    #[inline]
    fn arity(self) -> u8 {
        3
    }
    #[inline]
    fn name(self) -> &'static str {
        "attention"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::LinearAlgebra
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::AttentionBackward)
    }
}

/// Forward: `out = softmax((Q @ Kᵀ) * scale + causal_mask) @ V`.
///
/// Uses `call.scratch` (length ≥ `seq_kv`) as the per-row softmax-
/// weights buffer when supplied; falls back to a local allocation
/// when `call.scratch.len == 0`. The planner reserves the
/// scratch span as part of the workspace so production callers
/// run allocation-free.
pub fn attention(storage: &mut [f32], call: &AttentionCall) {
    let batch = call.batch as usize;
    let nqh = call.num_q_heads as usize;
    let nkh = call.num_kv_heads as usize;
    let hd = call.head_dim as usize;
    let sq = call.seq_q as usize;
    let sk = call.seq_kv as usize;
    debug_assert!(nqh > 0 && nkh > 0 && nqh.is_multiple_of(nkh));
    let scale = f32::from_bits(call.scale_bits);
    let q_stride_per_batch = nqh * sq * hd;
    let kv_stride_per_batch = nkh * sk * hd;
    let out_stride_per_batch = q_stride_per_batch;
    let q_per_kv = nqh / nkh;
    let cross_offset = sk as isize - sq as isize;

    // Scratch handle: either the planner-supplied span (zero-alloc
    // hot path) or a one-shot local Vec (fallback for tests).
    let mut local_scratch: Vec<f32>;
    let scratch_off: usize;
    if call.scratch.len >= sk {
        scratch_off = call.scratch.offset;
        local_scratch = Vec::new();
    } else {
        local_scratch = vec![0.0_f32; sk];
        scratch_off = usize::MAX;
    }
    // Helper closures: read/write `i`-th score either from `storage`
    // (when `scratch_off != usize::MAX`) or `local_scratch`.
    macro_rules! score {
        ($i:expr) => {
            if scratch_off != usize::MAX {
                storage[scratch_off + $i]
            } else {
                local_scratch[$i]
            }
        };
    }
    macro_rules! score_set {
        ($i:expr, $v:expr) => {
            if scratch_off != usize::MAX {
                storage[scratch_off + $i] = $v;
            } else {
                local_scratch[$i] = $v;
            }
        };
    }
    macro_rules! score_mul {
        ($i:expr, $v:expr) => {
            if scratch_off != usize::MAX {
                storage[scratch_off + $i] *= $v;
            } else {
                local_scratch[$i] *= $v;
            }
        };
    }

    for b in 0..batch {
        for qh in 0..nqh {
            let kvh = qh / q_per_kv;
            for q in 0..sq {
                // 1. scores[k] = (Q · K[k]) * scale, with causal mask.
                let q_off = call.q.offset + b * q_stride_per_batch + qh * sq * hd + q * hd;
                let k_plane = call.k.offset + b * kv_stride_per_batch + kvh * sk * hd;
                let v_plane = call.v.offset + b * kv_stride_per_batch + kvh * sk * hd;
                let mut max = f32::NEG_INFINITY;
                for k in 0..sk {
                    if call.causal && (k as isize) > q as isize + cross_offset {
                        score_set!(k, f32::NEG_INFINITY);
                        continue;
                    }
                    let mut dot = 0.0_f32;
                    for d in 0..hd {
                        dot += storage[q_off + d] * storage[k_plane + k * hd + d];
                    }
                    let s = dot * scale;
                    score_set!(k, s);
                    if s > max {
                        max = s;
                    }
                }

                // 2. softmax in-place over scores.
                let mut sum = 0.0_f32;
                for k in 0..sk {
                    let cur = score!(k);
                    let e = if cur.is_finite() {
                        libm::expf(cur - max)
                    } else {
                        0.0
                    };
                    score_set!(k, e);
                    sum += e;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for k in 0..sk {
                    score_mul!(k, inv);
                }

                // 3. out = scores @ V (single row of `head_dim`).
                let out_off = call.output.offset + b * out_stride_per_batch + qh * sq * hd + q * hd;
                for d in 0..hd {
                    let mut acc = 0.0_f32;
                    for k in 0..sk {
                        acc += score!(k) * storage[v_plane + k * hd + d];
                    }
                    storage[out_off + d] = acc;
                }
            }
        }
    }
}

/// Pre-resolved arguments for `Attention` backward.
///
/// Same shape descriptors as `AttentionCall`. `dq`, `dk`, `dv`
/// accumulate; empty grad slots are skipped. Like the forward, this
/// kernel allocates a per-row `[seq_kv]` scratch for the recomputed
/// softmax probabilities (and a second scratch for `dp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionGradCall {
    /// Forward `Q`.
    pub q: SlotSpan,
    /// Forward `K`.
    pub k: SlotSpan,
    /// Forward `V`.
    pub v: SlotSpan,
    /// Upstream gradient `d_out` (same shape as forward output).
    pub d_out: SlotSpan,
    /// Gradient slot for `Q`.
    pub dq: SlotSpan,
    /// Gradient slot for `K`.
    pub dk: SlotSpan,
    /// Gradient slot for `V`.
    pub dv: SlotSpan,
    /// Combined batch.
    pub batch: u32,
    /// Number of Q heads.
    pub num_q_heads: u32,
    /// Number of KV heads.
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Q sequence length.
    pub seq_q: u32,
    /// K/V sequence length.
    pub seq_kv: u32,
    /// `scale`, encoded as `f32::to_bits()`.
    pub scale_bits: u32,
    /// Causal mask flag.
    pub causal: bool,
}

/// Backward of `attention`. Recomputes the row-wise softmax
/// probabilities `p[k]` from forward `Q`/`K` and chains the standard
/// scaled-dot-product attention gradients into `dq`, `dk`, `dv`. GQA
/// is handled by accumulating `dk`/`dv` for the shared KV head across
/// every Q head that pulls from it. No-op when all three grad spans
/// are empty.
pub fn attention_grad(storage: &mut [f32], call: &AttentionGradCall) {
    let want_dq = call.dq.len > 0;
    let want_dk = call.dk.len > 0;
    let want_dv = call.dv.len > 0;
    if !want_dq && !want_dk && !want_dv {
        return;
    }
    let batch = call.batch as usize;
    let nqh = call.num_q_heads as usize;
    let nkh = call.num_kv_heads as usize;
    let hd = call.head_dim as usize;
    let sq = call.seq_q as usize;
    let sk = call.seq_kv as usize;
    debug_assert!(nqh > 0 && nkh > 0 && nqh.is_multiple_of(nkh));
    let scale = f32::from_bits(call.scale_bits);
    let q_stride_per_batch = nqh * sq * hd;
    let kv_stride_per_batch = nkh * sk * hd;
    let out_stride_per_batch = q_stride_per_batch;
    let q_per_kv = nqh / nkh;
    let cross_offset = sk as isize - sq as isize;

    let mut probs = vec![0.0_f32; sk];
    let mut dp = vec![0.0_f32; sk];

    for b in 0..batch {
        for qh in 0..nqh {
            let kvh = qh / q_per_kv;
            for q in 0..sq {
                let q_off = call.q.offset + b * q_stride_per_batch + qh * sq * hd + q * hd;
                let k_plane = call.k.offset + b * kv_stride_per_batch + kvh * sk * hd;
                let v_plane = call.v.offset + b * kv_stride_per_batch + kvh * sk * hd;
                let do_off = call.d_out.offset + b * out_stride_per_batch + qh * sq * hd + q * hd;

                // 1. Recompute scores → probs[k] (same logic as forward).
                let mut max = f32::NEG_INFINITY;
                for k in 0..sk {
                    if call.causal && (k as isize) > q as isize + cross_offset {
                        probs[k] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut dot = 0.0_f32;
                    for d in 0..hd {
                        dot += storage[q_off + d] * storage[k_plane + k * hd + d];
                    }
                    let s = dot * scale;
                    probs[k] = s;
                    if s > max {
                        max = s;
                    }
                }
                let mut sum = 0.0_f32;
                for s in probs.iter_mut().take(sk) {
                    let e = if s.is_finite() {
                        libm::expf(*s - max)
                    } else {
                        0.0
                    };
                    *s = e;
                    sum += e;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for s in probs.iter_mut().take(sk) {
                    *s *= inv;
                }

                // 2. dp[k] = Σ_d d_out[d] * V[k, d]. Also accumulate dV.
                for k in 0..sk {
                    let mut acc = 0.0_f32;
                    for d in 0..hd {
                        acc += storage[do_off + d] * storage[v_plane + k * hd + d];
                    }
                    dp[k] = acc;
                    if want_dv && probs[k] != 0.0 {
                        let dv_row =
                            call.dv.offset + b * kv_stride_per_batch + kvh * sk * hd + k * hd;
                        let p = probs[k];
                        for d in 0..hd {
                            storage[dv_row + d] += p * storage[do_off + d];
                        }
                    }
                }

                // 3. softmax backward: ds[k] = p[k] * (dp[k] - Σ_j p[j]*dp[j]).
                let mut dot_pp = 0.0_f32;
                for k in 0..sk {
                    dot_pp += probs[k] * dp[k];
                }
                if !want_dq && !want_dk {
                    continue;
                }
                for k in 0..sk {
                    let ds = probs[k] * (dp[k] - dot_pp);
                    if ds == 0.0 {
                        continue;
                    }
                    let ds_scaled = ds * scale;
                    if want_dq {
                        for d in 0..hd {
                            storage[call.dq.offset + (q_off - call.q.offset) + d] +=
                                ds_scaled * storage[k_plane + k * hd + d];
                        }
                    }
                    if want_dk {
                        let dk_row =
                            call.dk.offset + b * kv_stride_per_batch + kvh * sk * hd + k * hd;
                        for d in 0..hd {
                            storage[dk_row + d] += ds_scaled * storage[q_off + d];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_self_attn_1head(
        seq: usize,
        head_dim: usize,
        causal: bool,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Vec<f32> {
        let n = seq * head_dim;
        let mut s = vec![0.0_f32; 4 * n];
        s[0..n].copy_from_slice(q);
        s[n..2 * n].copy_from_slice(k);
        s[2 * n..3 * n].copy_from_slice(v);
        let call = AttentionCall {
            q: SlotSpan { offset: 0, len: n },
            k: SlotSpan { offset: n, len: n },
            v: SlotSpan {
                offset: 2 * n,
                len: n,
            },
            output: SlotSpan {
                offset: 3 * n,
                len: n,
            },
            scratch: SlotSpan::empty(0),
            batch: 1,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: head_dim as u32,
            seq_q: seq as u32,
            seq_kv: seq as u32,
            scale_bits: 1.0_f32.to_bits(),
            causal,
        };
        attention(&mut s, &call);
        s[3 * n..4 * n].to_vec()
    }

    #[test]
    fn attention_uniform_q_returns_mean_of_v() {
        // Q = zeros → Q·K = 0 for every key → softmax weights uniform
        // → output = mean(V) per query.
        let q = vec![0.0_f32; 4]; // 2 queries × head_dim 2
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 0.0, 0.0, 20.0];
        let out = run_self_attn_1head(2, 2, false, &q, &k, &v);
        // mean of V rows = ((10 + 0)/2, (0 + 20)/2) = (5, 10), broadcast to both queries.
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 10.0).abs() < 1e-5);
        assert!((out[2] - 5.0).abs() < 1e-5);
        assert!((out[3] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn attention_causal_mask_isolates_first_query() {
        // 2 queries, causal: query 0 can only see key 0 → output[0] = V[0].
        let q = vec![0.0_f32; 4];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 11.0, 20.0, 21.0];
        let out = run_self_attn_1head(2, 2, true, &q, &k, &v);
        // Query 0 only attends key 0 → [10, 11].
        assert!((out[0] - 10.0).abs() < 1e-5);
        assert!((out[1] - 11.0).abs() < 1e-5);
        // Query 1 attends both keys uniformly → mean of V rows.
        assert!((out[2] - 15.0).abs() < 1e-5);
        assert!((out[3] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn attention_gqa_groups_queries_to_kv_heads() {
        // 2 q-heads, 1 kv-head, head_dim=1, seq=1.
        // Q heads: [a0]=1, [a1]=2. K=[1], V=[5]. Both q-heads share kv-head 0.
        // scores both = exp(scale * Q·K) / Z = 1.0 (single key) → out = V = 5.
        let mut s = [
            1.0_f32, 2.0, // Q (heads-first): [qh=0, q=0, d=0], [qh=1, q=0, d=0]
            1.0, // K (single kv-head, seq=1)
            5.0, // V
            0.0, 0.0, // output
        ];
        let call = AttentionCall {
            q: SlotSpan { offset: 0, len: 2 },
            k: SlotSpan { offset: 2, len: 1 },
            v: SlotSpan { offset: 3, len: 1 },
            output: SlotSpan { offset: 4, len: 2 },
            scratch: SlotSpan::empty(0),
            batch: 1,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 1,
            seq_q: 1,
            seq_kv: 1,
            scale_bits: 1.0_f32.to_bits(),
            causal: false,
        };
        attention(&mut s, &call);
        assert!((s[4] - 5.0).abs() < 1e-5);
        assert!((s[5] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn attention_grad_matches_finite_difference() {
        // 1 head, seq=3, head_dim=2, no causal.
        let seq = 3;
        let head_dim = 2;
        let n = seq * head_dim;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.27 - 0.5).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32) - 0.2).collect();
        let d_out: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.41).sin()).collect();

        let total = 7 * n;
        let mut s = vec![0.0_f32; total];
        s[0..n].copy_from_slice(&q);
        s[n..2 * n].copy_from_slice(&k);
        s[2 * n..3 * n].copy_from_slice(&v);
        s[3 * n..4 * n].copy_from_slice(&d_out);
        let call = AttentionGradCall {
            q: SlotSpan { offset: 0, len: n },
            k: SlotSpan { offset: n, len: n },
            v: SlotSpan {
                offset: 2 * n,
                len: n,
            },
            d_out: SlotSpan {
                offset: 3 * n,
                len: n,
            },
            dq: SlotSpan {
                offset: 4 * n,
                len: n,
            },
            dk: SlotSpan {
                offset: 5 * n,
                len: n,
            },
            dv: SlotSpan {
                offset: 6 * n,
                len: n,
            },
            batch: 1,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: head_dim as u32,
            seq_q: seq as u32,
            seq_kv: seq as u32,
            scale_bits: 1.0_f32.to_bits(),
            causal: false,
        };
        attention_grad(&mut s, &call);
        let dq = s[4 * n..5 * n].to_vec();
        let dk = s[5 * n..6 * n].to_vec();
        let dv = s[6 * n..7 * n].to_vec();

        let dot = |y: &[f32]| -> f32 { y.iter().zip(d_out.iter()).map(|(a, b)| a * b).sum() };
        let h = 1e-3_f32;
        for i in 0..n {
            let mut qp = q.clone();
            qp[i] += h;
            let mut qm = q.clone();
            qm[i] -= h;
            let fd = (dot(&run_self_attn_1head(seq, head_dim, false, &qp, &k, &v))
                - dot(&run_self_attn_1head(seq, head_dim, false, &qm, &k, &v)))
                / (2.0 * h);
            assert!(
                (dq[i] - fd).abs() < 5e-2,
                "dq[{}]: got {}, fd {}",
                i,
                dq[i],
                fd
            );

            let mut kp = k.clone();
            kp[i] += h;
            let mut km = k.clone();
            km[i] -= h;
            let fd = (dot(&run_self_attn_1head(seq, head_dim, false, &q, &kp, &v))
                - dot(&run_self_attn_1head(seq, head_dim, false, &q, &km, &v)))
                / (2.0 * h);
            assert!(
                (dk[i] - fd).abs() < 5e-2,
                "dk[{}]: got {}, fd {}",
                i,
                dk[i],
                fd
            );

            let mut vp = v.clone();
            vp[i] += h;
            let mut vm = v.clone();
            vm[i] -= h;
            let fd = (dot(&run_self_attn_1head(seq, head_dim, false, &q, &k, &vp))
                - dot(&run_self_attn_1head(seq, head_dim, false, &q, &k, &vm)))
                / (2.0 * h);
            assert!(
                (dv[i] - fd).abs() < 5e-2,
                "dv[{}]: got {}, fd {}",
                i,
                dv[i],
                fd
            );
        }
    }
}
