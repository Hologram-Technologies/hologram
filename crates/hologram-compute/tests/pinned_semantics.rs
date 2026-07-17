//! Witnesses that convert former threshold accidents into pinned total
//! semantics — the "exactness replaces thresholds" rule applied to the two
//! places a tolerance literal was silently deciding behavior:
//!
//! - A **fully-masked row** (every key erased by `−∞`) used to flow through
//!   the max-shift as `−∞ − (−∞) = NaN`. Its semantics is now pinned exactly:
//!   decode attention → the zero vector; softmax → all zeros; log-softmax →
//!   all `−∞`. No NaN, no tolerance — the mask is the single visibility
//!   authority and "no key visible" is a legal, total input.
//! - A norm's **declared epsilon** used to be silently rewritten by
//!   `.abs().max(1e-9)`. The declared structure is now authoritative: any
//!   positive finite epsilon is honored exactly (witnessed by a variance
//!   small enough to distinguish `1e-12` from `1e-9`), `0` selects the pinned
//!   default `1e-9`, and a negative/NaN declaration is refused loud.

use hologram_compute::{
    Backend, BufferRef, CpuBackend, DecodeAttentionCall, KernelCall, NormCall, SoftmaxCall,
    SplitReads, Workspace,
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
        .map(|i| (((i * 13 + seed * 7) % 41) as f32 - 20.0) * 0.043)
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

fn decode_attn(mask: &[f32], m: u32, past: u32, new: u32) -> Vec<f32> {
    let (b, h, hkv, d) = (1u32, 2u32, 1u32, 8u32);
    let l = (past + new) as usize;
    assert_eq!(mask.len(), m as usize * l);
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rq = ws.push(&to_le(&f32s((b * h * m * d) as usize, 1)));
    let rkp = ws.push(&to_le(&f32s((b * hkv * past * d) as usize, 2)));
    let rvp = ws.push(&to_le(&f32s((b * hkv * past * d) as usize, 3)));
    let rkn = ws.push(&to_le(&f32s((b * hkv * new * d) as usize, 4)));
    let rvn = ws.push(&to_le(&f32s((b * hkv * new * d) as usize, 5)));
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

/// **Pinned:** a fully-masked query row is exactly the zero vector — every
/// bit — while unmasked rows are untouched by its presence.
#[test]
fn fully_masked_attention_row_is_exactly_the_zero_vector() {
    let (m, past, new) = (2u32, 6u32, 1u32);
    let l = (past + new) as usize;
    // Row 0 fully erased; row 1 sees everything.
    let mask: Vec<f32> = (0..m as usize * l)
        .map(|i| if i < l { f32::NEG_INFINITY } else { 0.0 })
        .collect();
    let out = decode_attn(&mask, m, past, new);
    let d = 8usize;
    let heads = 2usize;
    for hi in 0..heads {
        let row0 = &out[hi * (m as usize) * d..hi * (m as usize) * d + d];
        for (i, v) in row0.iter().enumerate() {
            assert_eq!(
                v.to_bits(),
                0.0f32.to_bits(),
                "head {hi} masked-row component {i} must be exactly +0.0, got {v}"
            );
        }
    }
    // The unmasked row must equal the same computation without the masked
    // row's existence mattering: rerun with row 0 fully visible and compare
    // only row 1 — rows are independent pipelines.
    let mask_all: Vec<f32> = vec![0.0; m as usize * l];
    let out_all = decode_attn(&mask_all, m, past, new);
    for hi in 0..heads {
        let base = hi * (m as usize) * d + d;
        assert_eq!(
            &out[base..base + d]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            &out_all[base..base + d]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "head {hi}: the unmasked row must be independent of the masked row"
        );
    }
}

fn softmax(rows: &[f32], b: u32, f: u32, log_form: bool) -> Vec<f32> {
    let mut ws = TestWorkspace { slots: Vec::new() };
    let ri = ws.push(&to_le(rows));
    let ro = ws.push(&vec![0u8; rows.len() * 4]);
    let call = SoftmaxCall {
        input: ri,
        output: ro,
        batch: b,
        feature: f,
        dtype: DTYPE_F32,
    };
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(
        &if log_form {
            KernelCall::LogSoftmax(call)
        } else {
            KernelCall::Softmax(call)
        },
        &mut ws,
    )
    .unwrap();
    le_to_f32(&ws.slots[ro.slot as usize])
}

/// **Pinned:** softmax of an all-(−∞) row is exactly zero everywhere;
/// log-softmax is exactly −∞ (log 0). A finite sibling row is unaffected.
#[test]
fn softmax_of_a_fully_masked_row_is_pinned_not_nan() {
    let f = 5u32;
    let mut rows = vec![f32::NEG_INFINITY; f as usize];
    rows.extend(f32s(f as usize, 9));
    let sm = softmax(&rows, 2, f, false);
    for (i, v) in sm[..f as usize].iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            0.0f32.to_bits(),
            "softmax[{i}] must be +0.0, got {v}"
        );
    }
    assert!(sm[f as usize..].iter().all(|v| v.is_finite()));
    let lsm = softmax(&rows, 2, f, true);
    for (i, v) in lsm[..f as usize].iter().enumerate() {
        assert!(
            *v == f32::NEG_INFINITY,
            "log_softmax[{i}] must be −∞ exactly, got {v}"
        );
    }
    assert!(lsm[f as usize..].iter().all(|v| v.is_finite()));
}

fn layer_norm(x: &[f32], b: u32, f: u32, epsilon_bits: u64) -> Result<Vec<f32>, ()> {
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rx = ws.push(&to_le(x));
    let rg = ws.push(&to_le(&vec![1.0f32; f as usize]));
    let rb = ws.push(&to_le(&vec![0.0f32; f as usize]));
    let ro = ws.push(&vec![0u8; x.len() * 4]);
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(
        &KernelCall::LayerNorm(NormCall {
            x: rx,
            gamma: rg,
            beta: rb,
            residual: NormCall::NO_RESIDUAL,
            output: ro,
            batch: b,
            feature: f,
            channels: 0,
            num_groups: 0,
            epsilon_bits,
            dtype: DTYPE_F32,
        }),
        &mut ws,
    )
    .map_err(|_| ())?;
    Ok(le_to_f32(&ws.slots[ro.slot as usize]))
}

/// **The declared epsilon is authoritative.** On a row whose variance is
/// tiny (~1e-14), `ε = 1e-12` and `ε = 1e-9` must produce different outputs —
/// under the old silent `.max(1e-9)` floor they were identical, i.e. the
/// kernel was overriding the declared structure. `0` still selects the
/// pinned default `1e-9`, so absent declarations are unchanged.
#[test]
fn norm_honors_the_declared_epsilon_exactly() {
    // Variance ≈ 1e-14: small enough that eps dominates the rsqrt.
    let x: Vec<f32> = (0..8).map(|i| 1.0 + (i as f32) * 1e-7).collect();
    let out_e12 = layer_norm(&x, 1, 8, (1e-12f32).to_bits() as u64).unwrap();
    let out_e9 = layer_norm(&x, 1, 8, (1e-9f32).to_bits() as u64).unwrap();
    let out_default = layer_norm(&x, 1, 8, 0).unwrap();
    assert_ne!(
        out_e12.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        out_e9.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a declared 1e-12 epsilon must not be silently floored to 1e-9"
    );
    assert_eq!(
        out_default.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        out_e9.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "epsilon_bits = 0 selects the pinned default 1e-9"
    );
}

/// A negative, NaN, or infinite declared epsilon is refused loud — never
/// silently rewritten by `.abs()` or a floor.
#[test]
fn garbage_epsilon_declarations_are_refused() {
    let x = f32s(8, 3);
    assert!(layer_norm(&x, 1, 8, (-1e-5f32).to_bits() as u64).is_err());
    assert!(layer_norm(&x, 1, 8, f32::NAN.to_bits() as u64).is_err());
    assert!(layer_norm(&x, 1, 8, f32::INFINITY.to_bits() as u64).is_err());
}
