//! Witnesses for the scalar-mask decode attention (κ121): the visibility law
//! — realized-prefix over the past bucket, causal triangle within the chunk —
//! is compiled in, and the single per-token datum is a 4-byte `valid_len`
//! operand. Three claims, each pinned:
//!
//! 1. **Bit-identity with the mask form** (κ119) over finite bytes, for every
//!    realized length including `0`, a mid-bucket prefix, exactly the bucket,
//!    and beyond it (the post-ring-wrap clamp).
//! 2. **Totality beyond it**: unrealized rows are never *read* — NaN/∞ bytes
//!    there cannot reach the result (they poison the mask form via `0·NaN`).
//! 3. **Refusals**: `q_rows != new_len`, `q_rows == 0`, and a short
//!    `valid_len` operand fail loud.

use hologram_backend::{
    Backend, BufferRef, CpuBackend, DecodeAttentionCall, DecodeAttentionValidCall, KernelCall,
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
        .map(|i| (((i * 23 + seed * 31) % 59) as f32 - 29.0) * 0.021)
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

const DIMS: (u32, u32, u32, u32, u32, u32) = (1, 4, 2, 4, 12, 16); // b, h, hkv, m(=new), past, d

struct Operands {
    q: Vec<f32>,
    kp: Vec<f32>,
    vp: Vec<f32>,
    kn: Vec<f32>,
    vn: Vec<f32>,
}

/// Operand set with `garbage` planted in every past row at index ≥ `valid`.
fn operands(valid: u32, garbage: f32) -> Operands {
    let (b, h, hkv, m, past, d) = DIMS;
    let mut kp = f32s((b * hkv * past * d) as usize, 2);
    let mut vp = f32s((b * hkv * past * d) as usize, 3);
    let vis = valid.min(past) as usize;
    for (i, v) in kp.iter_mut().enumerate() {
        if (i / d as usize) % past as usize >= vis {
            *v = garbage;
        }
    }
    for (i, v) in vp.iter_mut().enumerate() {
        if (i / d as usize) % past as usize >= vis {
            *v = -garbage;
        }
    }
    Operands {
        q: f32s((b * h * m * d) as usize, 1),
        kp,
        vp,
        kn: f32s((b * hkv * m * d) as usize, 4),
        vn: f32s((b * hkv * m * d) as usize, 5),
    }
}

fn run_valid(o: &Operands, valid: u32, scale_bits: u32) -> Result<Vec<f32>, ()> {
    let (b, h, hkv, m, past, d) = DIMS;
    run_valid_dims(o, valid, scale_bits, (b, h, hkv, m, m, past, d))
}

#[allow(clippy::too_many_arguments)]
fn run_valid_dims(
    o: &Operands,
    valid: u32,
    scale_bits: u32,
    dims: (u32, u32, u32, u32, u32, u32, u32), // b, h, hkv, q_rows, new, past, d
) -> Result<Vec<f32>, ()> {
    let (b, h, hkv, q_rows, new, past, d) = dims;
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rq = ws.push(&to_le(&o.q));
    let rkp = ws.push(&to_le(&o.kp));
    let rvp = ws.push(&to_le(&o.vp));
    let rkn = ws.push(&to_le(&o.kn));
    let rvn = ws.push(&to_le(&o.vn));
    let rv = ws.push(&valid.to_le_bytes());
    let ro = ws.push(&vec![0u8; (b * h * q_rows * d) as usize * 4]);
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(
        &KernelCall::DecodeAttentionValid(DecodeAttentionValidCall {
            q: rq,
            k_past: rkp,
            v_past: rvp,
            k_new: rkn,
            v_new: rvn,
            valid_len: rv,
            output: ro,
            batch: b,
            heads: h,
            kv_heads: hkv,
            q_rows,
            past_len: past,
            new_len: new,
            head_dim: d,
            scale_bits,
            dtype: DTYPE_F32,
        }),
        &mut ws,
    )
    .map_err(|_| ())?;
    Ok(le_to_f32(&ws.slots[ro.slot as usize]))
}

/// The equivalent κ119 mask: past col j visible iff `j < min(valid, past)`;
/// new col j visible to row i iff `j ≤ i`.
fn run_mask(o: &Operands, valid: u32, scale_bits: u32) -> Vec<f32> {
    let (b, h, hkv, m, past, d) = DIMS;
    let l = (past + m) as usize;
    let vis = valid.min(past) as usize;
    let mask: Vec<f32> = (0..m as usize * l)
        .map(|i| {
            let (row, j) = (i / l, i % l);
            let visible = if j < past as usize {
                j < vis
            } else {
                j - past as usize <= row
            };
            if visible {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rq = ws.push(&to_le(&o.q));
    let rkp = ws.push(&to_le(&o.kp));
    let rvp = ws.push(&to_le(&o.vp));
    let rkn = ws.push(&to_le(&o.kn));
    let rvn = ws.push(&to_le(&o.vn));
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
            new_len: m,
            head_dim: d,
            scale_bits,
            dtype: DTYPE_F32,
        }),
        &mut ws,
    )
    .unwrap();
    le_to_f32(&ws.slots[ro.slot as usize])
}

/// **Bit-identity with the mask form** at every realized-length regime:
/// empty, mid-bucket, exactly the bucket, and past it (the post-wrap clamp).
/// The packed visible walk and the erased-slot walk are the same serial
/// pipeline over the same values in the same order.
#[test]
fn valid_form_equals_the_mask_form_bitwise() {
    let (_, _, _, _, past, _) = DIMS;
    for valid in [0u32, 3, 7, past, past + 9, u32::MAX] {
        let o = operands(valid, 777.0); // finite garbage beyond the prefix
        let got = run_valid(&o, valid, 0).unwrap();
        let want = run_mask(&o, valid, 0);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "valid={valid} cell {i}: κ121 {g} vs κ119 {w}"
            );
        }
        // Explicit scale resolves identically in both forms.
        let got = run_valid(&o, valid, 0.25f32.to_bits()).unwrap();
        let want = run_mask(&o, valid, 0.25f32.to_bits());
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "valid={valid}: explicit-scale mismatch"
        );
    }
}

/// **Totality**: the kernel never reads an unrealized row, so NaN/∞ bytes
/// there cannot reach the result — output equals the finite-garbage run bit
/// for bit and is fully finite. (The mask form cannot make this promise:
/// `0·NaN = NaN` flows through its erased-slot AXPY.)
#[test]
fn unrealized_rows_are_never_read() {
    let valid = 5u32;
    let clean = operands(valid, 777.0);
    let want = run_valid(&clean, valid, 0).unwrap();
    let poisoned = operands(valid, f32::NAN);
    let got = run_valid(&poisoned, valid, 0).unwrap();
    assert!(
        got.iter().all(|v| v.is_finite()),
        "NaN leaked from unrealized rows"
    );
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "unrealized bytes must be unreachable"
    );
    let poisoned = operands(valid, f32::INFINITY);
    let got = run_valid(&poisoned, valid, 0).unwrap();
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    );
}

/// Refusals: the causal law pairs query row i with chunk key row i, so
/// `q_rows != new_len` (either direction) and `q_rows == 0` are refused; a
/// `valid_len` operand shorter than 4 bytes is refused.
#[test]
fn malformed_geometry_is_refused() {
    let (b, h, hkv, m, past, d) = DIMS;
    let o = operands(3, 777.0);
    // q_rows > new_len and q_rows < new_len: no row↔key correspondence.
    assert!(run_valid_dims(&o, 3, 0, (b, h, hkv, m, m - 1, past, d)).is_err());
    assert!(run_valid_dims(&o, 3, 0, (b, h, hkv, m - 1, m, past, d)).is_err());
    // q_rows == 0.
    assert!(run_valid_dims(&o, 3, 0, (b, h, hkv, 0, 0, past, d)).is_err());
    // Short valid_len operand.
    let (b, h, hkv, m, past, d) = DIMS;
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rq = ws.push(&to_le(&o.q));
    let rkp = ws.push(&to_le(&o.kp));
    let rvp = ws.push(&to_le(&o.vp));
    let rkn = ws.push(&to_le(&o.kn));
    let rvn = ws.push(&to_le(&o.vn));
    let rv = ws.push(&[7u8, 0]); // 2 bytes — too short
    let ro = ws.push(&vec![0u8; (b * h * m * d) as usize * 4]);
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    assert!(be
        .dispatch(
            &KernelCall::DecodeAttentionValid(DecodeAttentionValidCall {
                q: rq,
                k_past: rkp,
                v_past: rvp,
                k_new: rkn,
                v_new: rvn,
                valid_len: rv,
                output: ro,
                batch: b,
                heads: h,
                kv_heads: hkv,
                q_rows: m,
                past_len: past,
                new_len: m,
                head_dim: d,
                scale_bits: 0,
                dtype: DTYPE_F32,
            }),
            &mut ws,
        )
        .is_err());
}
