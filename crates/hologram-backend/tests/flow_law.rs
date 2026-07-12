//! The flow law, witnessed: evolving the decode state by a chunk of `C`
//! tokens in one fused call equals evolving it by `C` single-token steps —
//! **bit for bit**. This is the discrete `orbit_add` of the substrate's
//! decode dynamics: `m = C` chunked prefill (and speculative verify, which is
//! the same shape at `m = K`) and `m = 1` decode are one dynamics driven at
//! different strides, not two numerical paths that happen to be close.
//!
//! Why bitwise equality holds — and what this test pins: the kernel iterates
//! keys `past ∥ new` in index order, and the two drivings present the same
//! visible keys *in the same relative order* (realized past rows, then chunk
//! rows in order, then self). Every erased slot contributes exactly `0.0`
//! through the deterministic exp, and `x + 0.0` is exact, so the serial
//! max/sum/AXPY walk hits identical partial values in both drivings. If a
//! future kernel change breaks any of those properties (iteration order, the
//! exp's exact zero, the sequential reduction), this witness fails.

use hologram_backend::{
    Backend, BufferRef, CpuBackend, DecodeAttentionCall, KernelCall, KvCacheWriteCall, SplitReads,
    Workspace,
};

const DTYPE_F32: u8 = 8;

fn to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn f32s(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 29 + seed * 13) % 43) as f32 - 21.0) * 0.037)
        .collect()
}

/// Slot-indexed test workspace (the established direct-dispatch pattern).
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

#[allow(clippy::too_many_arguments)]
fn attn(
    q: &[f32],
    kp: &[f32],
    vp: &[f32],
    kn: &[f32],
    vn: &[f32],
    mask: &[f32],
    dims: (u32, u32, u32, u32, u32, u32, u32), // b, h, hkv, m, past, new, d
) -> Vec<f32> {
    let (b, h, hkv, m, past, new, d) = dims;
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rq = ws.push(&to_le(q));
    let rkp = ws.push(&to_le(kp));
    let rvp = ws.push(&to_le(vp));
    let rkn = ws.push(&to_le(kn));
    let rvn = ws.push(&to_le(vn));
    let rm = ws.push(&to_le(mask));
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
    le_to_f32(&ws.slots[ro.slot as usize])
}

fn kv_write(cache: &[f32], new: &[f32], pos: u32, planes: u32, bucket: u32, d: u32) -> Vec<f32> {
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rc = ws.push(&to_le(cache));
    let rn = ws.push(&to_le(new));
    let rp = ws.push(&pos.to_le_bytes());
    let ro = ws.push(&vec![0u8; cache.len() * 4]);
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(
        &KernelCall::KvCacheWrite(KvCacheWriteCall {
            cache: rc,
            new: rn,
            pos: rp,
            output: ro,
            planes,
            bucket_rows: bucket,
            new_rows: 1,
            row_bytes: d * 4,
        }),
        &mut ws,
    )
    .unwrap();
    le_to_f32(&ws.slots[ro.slot as usize])
}

/// **The flow law.** One chunked call over `C` new tokens (causal-within-
/// chunk in the mask) equals `C` sequential single-token steps — each step
/// attending, then writing its K/V row into the bucket — bit for bit, on
/// every output row, including GQA head grouping.
#[test]
fn chunked_prefill_equals_sequential_decode_bitwise() {
    let (b, h, hkv, d) = (1u32, 4u32, 2u32, 16u32);
    let (bucket, realized, chunk) = (8u32, 3u32, 4u32);
    let planes = b * hkv;

    // Bucket caches: rows 0..realized are live history; the rest holds
    // finite garbage the masks must erase identically in both drivings.
    let mut cache_k = f32s((planes * bucket * d) as usize, 1);
    let mut cache_v = f32s((planes * bucket * d) as usize, 2);
    for (i, v) in cache_k.iter_mut().enumerate() {
        if (i / d as usize) % bucket as usize >= realized as usize {
            *v = 777.0 + (i % 5) as f32;
        }
    }
    for (i, v) in cache_v.iter_mut().enumerate() {
        if (i / d as usize) % bucket as usize >= realized as usize {
            *v = -555.0 - (i % 7) as f32;
        }
    }
    let q_chunk = f32s((b * h * chunk * d) as usize, 3);
    let k_chunk = f32s((planes * chunk * d) as usize, 4);
    let v_chunk = f32s((planes * chunk * d) as usize, 5);

    // Chunked driving: one call, m = C, causal-within-chunk mask.
    let lc = (bucket + chunk) as usize;
    let mask_chunk: Vec<f32> = (0..chunk as usize * lc)
        .map(|i| {
            let (row, j) = (i / lc, i % lc);
            let visible_past = j < realized as usize;
            let visible_new = j >= bucket as usize && (j - bucket as usize) <= row;
            if visible_past || visible_new {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let out_chunk = attn(
        &q_chunk,
        &cache_k,
        &cache_v,
        &k_chunk,
        &v_chunk,
        &mask_chunk,
        (b, h, hkv, chunk, bucket, chunk, d),
    );

    // Sequential driving: C single-token steps, each writing its row.
    let ls = (bucket + 1) as usize;
    for step in 0..chunk as usize {
        let q_step: Vec<f32> = (0..h as usize)
            .flat_map(|hi| {
                let base = (hi * chunk as usize + step) * d as usize;
                q_chunk[base..base + d as usize].to_vec()
            })
            .collect();
        let k_step: Vec<f32> = (0..planes as usize)
            .flat_map(|p| {
                let base = (p * chunk as usize + step) * d as usize;
                k_chunk[base..base + d as usize].to_vec()
            })
            .collect();
        let v_step: Vec<f32> = (0..planes as usize)
            .flat_map(|p| {
                let base = (p * chunk as usize + step) * d as usize;
                v_chunk[base..base + d as usize].to_vec()
            })
            .collect();
        let visible = realized as usize + step;
        let mask_step: Vec<f32> = (0..ls)
            .map(|j| {
                if j < visible || j == bucket as usize {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let out_step = attn(
            &q_step,
            &cache_k,
            &cache_v,
            &k_step,
            &v_step,
            &mask_step,
            (b, h, hkv, 1, bucket, 1, d),
        );
        // Compare this step's row against the chunk's row `step`, per head.
        for hi in 0..h as usize {
            let seq = &out_step[hi * d as usize..(hi + 1) * d as usize];
            let chk_base = (hi * chunk as usize + step) * d as usize;
            let chk = &out_chunk[chk_base..chk_base + d as usize];
            for (i, (a, c)) in seq.iter().zip(chk).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    c.to_bits(),
                    "step {step} head {hi} component {i}: sequential {a} vs chunked {c}"
                );
            }
        }
        // Advance the state: write this token's K/V row at the frontier.
        cache_k = kv_write(&cache_k, &k_step, realized + step as u32, planes, bucket, d);
        cache_v = kv_write(&cache_v, &v_step, realized + step as u32, planes, bucket, d);
    }
}
