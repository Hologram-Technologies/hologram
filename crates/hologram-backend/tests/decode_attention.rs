//! Witnesses for the fused decode attention (`DecodeAttentionCall`).
//!
//! Three claims carry the feature, each asserted bit-for-bit:
//!
//! 1. **Split == precatenated.** Reading keys `past ∥ new` where they lie must
//!    equal the legacy `AttentionCall` over the physically concatenated buffer
//!    — same scores in the same order, so the same bytes. This is what makes
//!    the O(bucket) per-step `Concat` recopy deletable: the split form *is*
//!    the concatenated computation, minus the copy.
//! 2. **Padded bucket + mask == tight.** A fixed bucket with `-inf` mask
//!    entries beyond the realized length must produce byte-identical output to
//!    the tight computation over only the realized keys: the deterministic exp
//!    maps `-inf` to exactly `0.0`, contributing exact zeros to the sum and
//!    the context. This is what makes a *fixed* padded bucket κ-stable: one
//!    call shape, per-step identity riding the mask operand's bytes.
//! 3. **The mask is the only masking authority.** Causal-within-chunk encoded
//!    in the mask must reproduce the legacy causal kernel bit-for-bit at
//!    `m == L` — the new form subsumes the old, it does not approximate it.

use hologram_backend::SplitReads;
use hologram_backend::{
    AttentionCall, Backend, BufferRef, CpuBackend, DecodeAttentionCall, KernelCall, Workspace,
};

const DTYPE_F32: u8 = 8;

/// Minimal test workspace: a slot-indexed Vec<Vec<u8>> (the `cpu_kernels.rs`
/// pattern).
struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}

impl TestWorkspace {
    fn new() -> Self {
        Self { slots: Vec::new() }
    }
    fn push_f32(&mut self, data: &[f32]) -> BufferRef {
        let slot = self.slots.len() as u32;
        self.slots
            .push(data.iter().flat_map(|v| v.to_le_bytes()).collect());
        BufferRef {
            slot,
            offset: 0,
            length: (data.len() * 4) as u64,
        }
    }
    fn read_f32(&self, r: BufferRef) -> Vec<f32> {
        self.slots[r.slot as usize]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
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

fn dispatch(call: &KernelCall, ws: &mut TestWorkspace) {
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(call, ws).unwrap();
}

fn qv(b: usize, h: usize, m: usize, d: usize, seed: usize) -> Vec<f32> {
    (0..b * h * m * d)
        .map(|i| (((i * 13 + seed * 7) % 41) as f32 - 20.0) * 0.043)
        .collect()
}

/// Split == precatenated, and mask-causal == legacy-causal, in one test:
/// at `m == L` with a causal mask, the decode kernel over (past ∥ new) must
/// reproduce the legacy causal `AttentionCall` over the concatenated K/V
/// **bit for bit**.
#[test]
fn split_kv_with_causal_mask_equals_legacy_causal_attention_bitwise() {
    let (b, h, hkv, d) = (2usize, 4usize, 2usize, 16usize);
    let (past, new) = (5usize, 3usize);
    let l = past + new;
    let m = l; // legacy kernel requires q rows == key rows

    let q = qv(b, h, m, d, 1);
    // K/V generated as one concatenated tensor, then split at `past`.
    let k_all = qv(b, hkv, l, d, 2);
    let v_all = qv(b, hkv, l, d, 3);
    let (mut kp, mut kn, mut vp, mut vn) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for bh in 0..b * hkv {
        let base = bh * l * d;
        kp.extend_from_slice(&k_all[base..base + past * d]);
        kn.extend_from_slice(&k_all[base + past * d..base + l * d]);
        vp.extend_from_slice(&v_all[base..base + past * d]);
        vn.extend_from_slice(&v_all[base + past * d..base + l * d]);
    }
    // Causal mask over the full square: query i sees keys j <= i.
    let mask: Vec<f32> = (0..m)
        .flat_map(|i| (0..l).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();

    // Legacy path.
    let mut wa = TestWorkspace::new();
    let (rq, rk, rv) = (wa.push_f32(&q), wa.push_f32(&k_all), wa.push_f32(&v_all));
    let ro = wa.push_f32(&vec![0f32; b * h * m * d]);
    dispatch(
        &KernelCall::Attention(AttentionCall {
            q: rq,
            k: rk,
            v: rv,
            output: ro,
            batch: b as u32,
            heads: h as u32,
            seq: l as u32,
            head_dim: d as u32,
            kv_heads: hkv as u32,
            causal: true,
            scale_bits: 0,
            dtype: DTYPE_F32,
        }),
        &mut wa,
    );
    let legacy = wa.read_f32(ro);

    // Decode path: split KV + the causal mask.
    let mut wd = TestWorkspace::new();
    let (rq, rkp, rvp, rkn, rvn, rm) = (
        wd.push_f32(&q),
        wd.push_f32(&kp),
        wd.push_f32(&vp),
        wd.push_f32(&kn),
        wd.push_f32(&vn),
        wd.push_f32(&mask),
    );
    let ro = wd.push_f32(&vec![0f32; b * h * m * d]);
    dispatch(
        &KernelCall::DecodeAttention(DecodeAttentionCall {
            q: rq,
            k_past: rkp,
            v_past: rvp,
            k_new: rkn,
            v_new: rvn,
            mask: rm,
            output: ro,
            batch: b as u32,
            heads: h as u32,
            kv_heads: hkv as u32,
            q_rows: m as u32,
            past_len: past as u32,
            new_len: new as u32,
            head_dim: d as u32,
            scale_bits: 0,
            dtype: DTYPE_F32,
        }),
        &mut wd,
    );
    let decode = wd.read_f32(ro);

    for (i, (a, bv)) in legacy.iter().zip(&decode).enumerate() {
        assert_eq!(
            a.to_bits(),
            bv.to_bits(),
            "cell {i}: legacy causal {a} vs split+mask {bv}"
        );
    }
}

/// Padded bucket + `-inf` mask == tight computation, bit for bit. This is the
/// property that lets a consumer fix the bucket (one call shape, one κ shape)
/// and let the realized length live in the mask operand's bytes.
#[test]
fn padded_bucket_with_mask_equals_tight_computation_bitwise() {
    let (b, h, hkv, d, m) = (1usize, 6usize, 3usize, 32usize, 1usize);
    let realized = 7usize;
    let bucket = 24usize; // fixed padded past
    let new = 1usize;

    let q = qv(b, h, m, d, 4);
    let k_real = qv(b, hkv, realized, d, 5);
    let v_real = qv(b, hkv, realized, d, 6);
    let kn = qv(b, hkv, new, d, 7);
    let vn = qv(b, hkv, new, d, 8);

    // Padded cache: realized rows then garbage (nonzero, so leakage would show).
    let mut k_pad = vec![0f32; b * hkv * bucket * d];
    let mut v_pad = vec![0f32; b * hkv * bucket * d];
    for bh in 0..b * hkv {
        for r in 0..bucket {
            for c in 0..d {
                let dst = (bh * bucket + r) * d + c;
                if r < realized {
                    k_pad[dst] = k_real[(bh * realized + r) * d + c];
                    v_pad[dst] = v_real[(bh * realized + r) * d + c];
                } else {
                    k_pad[dst] = 999.0; // must never influence the output
                    v_pad[dst] = -999.0;
                }
            }
        }
    }
    let mask_pad: Vec<f32> = (0..m)
        .flat_map(|_| {
            (0..bucket + new).map(|j| {
                if j < realized || j >= bucket {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let mask_tight = vec![0f32; m * (realized + new)];

    let run = |kp: &[f32], vp: &[f32], mask: &[f32], past: usize| -> Vec<f32> {
        let mut w = TestWorkspace::new();
        let (rq, rkp, rvp, rkn, rvn, rm) = (
            w.push_f32(&q),
            w.push_f32(kp),
            w.push_f32(vp),
            w.push_f32(&kn),
            w.push_f32(&vn),
            w.push_f32(mask),
        );
        let ro = w.push_f32(&vec![0f32; b * h * m * d]);
        dispatch(
            &KernelCall::DecodeAttention(DecodeAttentionCall {
                q: rq,
                k_past: rkp,
                v_past: rvp,
                k_new: rkn,
                v_new: rvn,
                mask: rm,
                output: ro,
                batch: b as u32,
                heads: h as u32,
                kv_heads: hkv as u32,
                q_rows: m as u32,
                past_len: past as u32,
                new_len: new as u32,
                head_dim: d as u32,
                scale_bits: 0,
                dtype: DTYPE_F32,
            }),
            &mut w,
        );
        w.read_f32(ro)
    };

    let padded = run(&k_pad, &v_pad, &mask_pad, bucket);
    let tight = run(&k_real, &v_real, &mask_tight, realized);
    for (i, (p, t)) in padded.iter().zip(&tight).enumerate() {
        assert_eq!(
            p.to_bits(),
            t.to_bits(),
            "cell {i}: padded-bucket {p} vs tight {t} — a masked key leaked"
        );
    }
}

/// Decode shape (`m = 1` against a long past) with GQA and an explicit scale:
/// checked against a straightforward reference restated independently, using
/// the same deterministic exp the kernel uses — so the comparison is exact,
/// not tolerance-based.
#[test]
fn decode_step_matches_independent_reference_bitwise() {
    let (b, h, hkv, d, m) = (1usize, 4usize, 2usize, 8usize, 1usize);
    let (past, new) = (33usize, 1usize);
    let l = past + new;
    let q = qv(b, h, m, d, 9);
    let kp = qv(b, hkv, past, d, 10);
    let vp = qv(b, hkv, past, d, 11);
    let kn = qv(b, hkv, new, d, 12);
    let vn = qv(b, hkv, new, d, 13);
    let mask: Vec<f32> = (0..m * l)
        .map(|i| if i % 5 == 3 { f32::NEG_INFINITY } else { 0.0 })
        .collect();
    let scale_mult = 0.25f32; // explicit multiplier: score = dot * 0.25 + mask

    let mut w = TestWorkspace::new();
    let (rq, rkp, rvp, rkn, rvn, rm) = (
        w.push_f32(&q),
        w.push_f32(&kp),
        w.push_f32(&vp),
        w.push_f32(&kn),
        w.push_f32(&vn),
        w.push_f32(&mask),
    );
    let ro = w.push_f32(&vec![0f32; b * h * m * d]);
    dispatch(
        &KernelCall::DecodeAttention(DecodeAttentionCall {
            q: rq,
            k_past: rkp,
            v_past: rvp,
            k_new: rkn,
            v_new: rvn,
            mask: rm,
            output: ro,
            batch: b as u32,
            heads: h as u32,
            kv_heads: hkv as u32,
            q_rows: m as u32,
            past_len: past as u32,
            new_len: new as u32,
            head_dim: d as u32,
            scale_bits: scale_mult.to_bits(),
            dtype: DTYPE_F32,
        }),
        &mut w,
    );
    let got = w.read_f32(ro);

    // Reference: same expression order, same deterministic exp.
    let group = h / hkv;
    let mut want = vec![0f32; b * h * m * d];
    for hi in 0..h {
        let kvh = hi / group;
        let qrow = &q[hi * d..hi * d + d];
        let mut scores = vec![0f32; l];
        for (j, slot) in scores.iter_mut().enumerate() {
            let krow = if j < past {
                &kp[(kvh * past + j) * d..(kvh * past + j) * d + d]
            } else {
                &kn[(kvh * new + (j - past)) * d..(kvh * new + (j - past)) * d + d]
            };
            let dot = hologram_backend::cpu::simd::simd_f32_dot(qrow, krow);
            *slot = dot / (1.0 / scale_mult) + mask[j];
        }
        let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in scores.iter_mut() {
            *v -= mx;
        }
        hologram_backend::cpu::simd::simd_f32_exp_inplace(&mut scores);
        let mut sum = 0f32;
        for &v in &scores {
            sum += v;
        }
        let denom = sum.max(1e-30);
        let orow = &mut want[hi * d..hi * d + d];
        for (j, &sc) in scores.iter().enumerate() {
            let p = sc / denom;
            let vrow = if j < past {
                &vp[(kvh * past + j) * d..(kvh * past + j) * d + d]
            } else {
                &vn[(kvh * new + (j - past)) * d..(kvh * new + (j - past)) * d + d]
            };
            hologram_backend::cpu::simd::simd_f32_axpy(orow, p, vrow);
        }
    }
    for (i, (g, r)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), r.to_bits(), "cell {i}: kernel {g} vs ref {r}");
    }
}
