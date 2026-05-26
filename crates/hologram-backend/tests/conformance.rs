//! **Kernel numeric conformance — external + scaling V&V (class KC/SC).**
//!
//! Validates hologram's compute kernels against an *independent* reference
//! — the operator's IEEE-754 mathematical definition evaluated in `f64`,
//! the numerically-authoritative ground truth (the same discipline used to
//! validate BLAS/NumPy) — **not** hologram's own Term-tree evaluator.
//!
//! Critically, every check runs **across a range of sizes**, including
//! non-power-of-2 dimensions, so the V&V demonstrates that the
//! implementation holds at arbitrary scale and is not short-cutting
//! (a degenerate/memoized path returning wrong or partial results) or
//! breaking down (tail-handling bugs, precision collapse) anywhere:
//!
//! * **KC-1** f32 matmul equals the f64-reference product within an
//!   accumulation-error tolerance, for every (m, k, n) from 2³ up.
//! * **SC-1** correctness holds identically as size scales — the relative
//!   error stays bounded by `~k · ε_f32`, not growing into divergence.
//! * **KC-2** the kernel reads *all* of its operands (no short-cut): a
//!   one-element change in B changes the corresponding output column.

use hologram_backend::cpu::dtype::{DTYPE_F32, DTYPE_F64, DTYPE_I8};
use hologram_backend::SplitReads;
use hologram_backend::{
    AttentionCall, Backend, BinaryCall, BufferRef, Conv2dCall, CpuBackend, DequantizeCall,
    KernelCall, MatMulCall, NormCall, PoolCall, ReduceCall, SoftmaxCall, UnaryCall, Workspace,
};

struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}

impl Workspace for TestWorkspace {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize][..]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let slot = b.slot as usize;
        let len = self.slots[slot].len();
        &mut self.slots[slot][..len]
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

fn buf(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 0,
    }
}

fn f32_to_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Deterministic, reproducible pseudo-random inputs in [-1, 1) — no `rand`
/// dependency, so the conformance corpus is fixed and replayable.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Independent reference: row-major `A(m×k) · B(k×n)` accumulated in f64.
fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

fn run_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut ws = TestWorkspace {
        slots: vec![f32_to_le(a), f32_to_le(b), vec![0u8; m * n * 4]],
    };
    let call = KernelCall::MatMul(MatMulCall {
        a: buf(0),
        b: buf(1),
        output: buf(2),
        m: m as u32,
        k: k as u32,
        n: n as u32,
        dtype: DTYPE_F32,
        b_packed: false,
    });
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    backend.dispatch(&call, &mut ws).unwrap();
    le_to_f32(&ws.slots[2])
}

/// Error against the f64 reference, normalized by the result's magnitude
/// (`max|want|`) — the standard matmul-agreement metric. A per-element
/// relative error is meaningless for near-zero outputs produced by
/// cancellation (tiny denominator); normalizing by the dominant element
/// measures the error at the scale the operator actually operates on.
fn max_rel_err(got: &[f32], want: &[f32]) -> f64 {
    let scale = want.iter().fold(0f64, |m, &w| m.max(f64::from(w).abs())) + 1e-9;
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - f64::from(w)).abs() / scale)
        .fold(0f64, f64::max)
}

#[test]
fn kc1_sc1_matmul_conforms_across_scale() {
    // Tiny → large, with non-power-of-2 and rectangular shapes that expose
    // tail-handling / blocking bugs a power-of-2-only test would miss.
    let shapes = [
        (2usize, 2usize, 2usize),
        (8, 8, 8),
        (31, 17, 29),
        (64, 64, 64),
        (127, 129, 130),
        (128, 128, 128),
        (256, 96, 256),
        (512, 64, 384),
    ];
    for (idx, &(m, k, n)) in shapes.iter().enumerate() {
        let a = fill(m * k, 0x1000 + idx as u64);
        let b = fill(k * n, 0x2000 + idx as u64);
        let got = run_matmul(&a, &b, m, k, n);
        let want = ref_matmul(&a, &b, m, k, n);
        let err = max_rel_err(&got, &want);
        // Normalized f32 matmul error scales ~ √k · ε_f32 (ε ≈ 6e-8), so it
        // stays well under 1e-4 even at k≈512. A short-cut (zeros / partial
        // / memoized-wrong) would blow past this by orders of magnitude —
        // this bound is the line between f32 rounding and breakdown.
        let bound = 1e-4_f64;
        assert!(
            err <= bound,
            "matmul {m}×{k}×{n}: rel err {err:.3e} exceeded bound {bound:.3e} — \
             implementation is short-cutting or breaking down at this scale"
        );
        // Not-degenerate: output must carry real signal, not zeros.
        assert!(
            got.iter().any(|&v| v.abs() > 1e-6),
            "matmul {m}×{k}×{n} produced an all-zero output"
        );
    }
}

#[test]
fn kc2_matmul_reads_all_operands_no_shortcut() {
    let (m, k, n) = (16, 16, 16);
    let a = fill(m * k, 7);
    let mut b = fill(k * n, 9);
    let base = run_matmul(&a, &b, m, k, n);
    // Perturb one element of B at (row 3, col 5) → output column 5 must move.
    b[3 * n + 5] += 1.0;
    let perturbed = run_matmul(&a, &b, m, k, n);
    let changed = (0..m).any(|i| (base[i * n + 5] - perturbed[i * n + 5]).abs() > 1e-6);
    assert!(
        changed,
        "perturbing B did not change the output — kernel is short-cutting an operand"
    );
}

/// KC-1b: low-precision matmul (bf16 / f16) routes through the **same**
/// cache-oblivious f32 engine (widen → engine → narrow). We quantize the
/// operands to the target dtype *first*, then compare the kernel against an
/// f64 reference over those same quantized inputs — isolating the engine's
/// arithmetic (which must match f32 matmul to ~1e-4) from the dtype's
/// inherent rounding. A short-cut (zeros / scalar breakdown) blows past this.
#[test]
fn kc1b_low_precision_matmul_routes_through_engine() {
    use hologram_backend::cpu::dtype::{
        read_bf16, read_f16, write_bf16, write_f16, DTYPE_BF16, DTYPE_F16,
    };
    let shapes = [(8usize, 8usize, 8usize), (31, 17, 29), (128, 96, 130)];
    for (dtype, enc, dec) in [
        (
            DTYPE_BF16,
            write_bf16 as fn(&mut [u8], usize, f32),
            read_bf16 as fn(&[u8], usize) -> f32,
        ),
        (DTYPE_F16, write_f16, read_f16),
    ] {
        for (idx, &(m, k, n)) in shapes.iter().enumerate() {
            let af = fill(m * k, 0x3000 + idx as u64);
            let bf = fill(k * n, 0x4000 + idx as u64);
            // Encode to the target dtype, then decode back to the values the
            // kernel actually sees — these are the reference inputs.
            let mut abytes = vec![0u8; m * k * 2];
            let mut bbytes = vec![0u8; k * n * 2];
            for (i, &v) in af.iter().enumerate() {
                enc(&mut abytes, i, v);
            }
            for (i, &v) in bf.iter().enumerate() {
                enc(&mut bbytes, i, v);
            }
            let aq: Vec<f32> = (0..m * k).map(|i| dec(&abytes, i)).collect();
            let bq: Vec<f32> = (0..k * n).map(|i| dec(&bbytes, i)).collect();
            let want = ref_matmul(&aq, &bq, m, k, n);

            let mut ws = TestWorkspace {
                slots: vec![abytes, bbytes, vec![0u8; m * n * 2]],
            };
            let call = KernelCall::MatMul(MatMulCall {
                a: buf(0),
                b: buf(1),
                output: buf(2),
                m: m as u32,
                k: k as u32,
                n: n as u32,
                dtype,
                b_packed: false,
            });
            let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
            backend.dispatch(&call, &mut ws).unwrap();
            let got: Vec<f32> = (0..m * n).map(|i| dec(&ws.slots[2], i)).collect();

            // Output is re-narrowed to the dtype, so allow one dtype ULP on top
            // of the engine's f32 agreement: bf16 ≈ 8e-3, f16 ≈ 1e-3 relative.
            let bound = if dtype == DTYPE_BF16 { 8e-3 } else { 1.5e-3 };
            let err = max_rel_err(&got, &want);
            assert!(
                err <= bound,
                "low-prec matmul dt={dtype} {m}×{k}×{n}: rel err {err:.3e} > {bound:.3e} — \
                 not routing through the engine correctly"
            );
            assert!(
                got.iter().any(|&v| v.abs() > 1e-6),
                "low-prec matmul dt={dtype} {m}×{k}×{n} produced all zeros"
            );
        }
    }
}

// ─── Shared dispatch helper ───────────────────────────────────────────

fn run(call: KernelCall, mut slots: Vec<Vec<u8>>, out_slot: usize) -> Vec<f32> {
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let mut ws = TestWorkspace {
        slots: std::mem::take(&mut slots),
    };
    backend.dispatch(&call, &mut ws).unwrap();
    le_to_f32(&ws.slots[out_slot])
}

fn check(op: &str, m: usize, k: usize, n: usize, got: &[f32], want: &[f32], bound: f64) {
    let err = max_rel_err(got, want);
    assert!(
        err <= bound,
        "{op} {m}×{k}×{n}: rel err {err:.3e} > {bound:.3e} — short-cutting or breaking down"
    );
}

// ─── KC-3: Softmax (ONNX Softmax over the last axis) ──────────────────

fn ref_softmax(x: &[f32], b: usize, f: usize) -> Vec<f32> {
    let mut o = vec![0f32; b * f];
    for r in 0..b {
        let row = &x[r * f..r * f + f];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = row
            .iter()
            .map(|&v| ((f64::from(v)) - f64::from(max)).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        for j in 0..f {
            o[r * f + j] = (exps[j] / sum) as f32;
        }
    }
    o
}

#[test]
fn kc3_softmax_conforms_across_scale() {
    for (idx, &(b, f)) in [(1usize, 4usize), (8, 31), (16, 128), (4, 1024)]
        .iter()
        .enumerate()
    {
        let x = fill(b * f, 0x300 + idx as u64);
        let got = run(
            KernelCall::Softmax(SoftmaxCall {
                input: buf(0),
                output: buf(1),
                batch: b as u32,
                feature: f as u32,
                dtype: DTYPE_F32,
            }),
            vec![f32_to_le(&x), vec![0u8; b * f * 4]],
            1,
        );
        check("softmax", b, f, 1, &got, &ref_softmax(&x, b, f), 1e-4);
        // rows sum to 1
        for r in 0..b {
            let s: f32 = got[r * f..r * f + f].iter().sum();
            assert!((s - 1.0).abs() < 1e-3, "softmax row {r} sums to {s}");
        }
    }
}

// ─── KC-4: LayerNorm (ONNX LayerNormalization over feature axis) ──────

fn ref_layernorm(x: &[f32], g: &[f32], bta: &[f32], b: usize, f: usize, eps: f32) -> Vec<f32> {
    let mut o = vec![0f32; b * f];
    for r in 0..b {
        let row = &x[r * f..r * f + f];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / f as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / f as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for j in 0..f {
            o[r * f + j] =
                ((f64::from(row[j]) - mean) * inv * f64::from(g[j]) + f64::from(bta[j])) as f32;
        }
    }
    o
}

#[test]
fn kc4_layernorm_conforms_across_scale() {
    let eps = 1e-5f32;
    for (idx, &(b, f)) in [(2usize, 8usize), (8, 33), (16, 256)].iter().enumerate() {
        let x = fill(b * f, 0x400 + idx as u64);
        let g = fill(f, 0x410 + idx as u64);
        let bta = fill(f, 0x420 + idx as u64);
        let got = run(
            KernelCall::LayerNorm(NormCall {
                x: buf(0),
                gamma: buf(1),
                beta: buf(2),
                residual: NormCall::NO_RESIDUAL,
                output: buf(3),
                batch: b as u32,
                feature: f as u32,
                epsilon_bits: u64::from(eps.to_bits()),
                dtype: DTYPE_F32,
            }),
            vec![
                f32_to_le(&x),
                f32_to_le(&g),
                f32_to_le(&bta),
                vec![0u8; b * f * 4],
            ],
            3,
        );
        check(
            "layernorm",
            b,
            f,
            1,
            &got,
            &ref_layernorm(&x, &g, &bta, b, f, eps),
            1e-3,
        );
    }
}

// ─── KC-5: RMSNorm (x / sqrt(mean(x²)+eps) · γ) ───────────────────────

fn ref_rmsnorm(x: &[f32], g: &[f32], b: usize, f: usize, eps: f32) -> Vec<f32> {
    let mut o = vec![0f32; b * f];
    for r in 0..b {
        let row = &x[r * f..r * f + f];
        let ms = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / f as f64;
        let inv = 1.0 / (ms + f64::from(eps)).sqrt();
        for j in 0..f {
            o[r * f + j] = (f64::from(row[j]) * inv * f64::from(g[j])) as f32;
        }
    }
    o
}

#[test]
fn kc5_rmsnorm_conforms_across_scale() {
    let eps = 1e-5f32;
    for (idx, &(b, f)) in [(2usize, 8usize), (8, 33), (16, 256)].iter().enumerate() {
        let x = fill(b * f, 0x500 + idx as u64);
        let g = fill(f, 0x510 + idx as u64);
        let got = run(
            KernelCall::RmsNorm(NormCall {
                x: buf(0),
                gamma: buf(1),
                beta: buf(2),
                residual: NormCall::NO_RESIDUAL,
                output: buf(3),
                batch: b as u32,
                feature: f as u32,
                epsilon_bits: u64::from(eps.to_bits()),
                dtype: DTYPE_F32,
            }),
            vec![
                f32_to_le(&x),
                f32_to_le(&g),
                f32_to_le(&fill(f, 1)),
                vec![0u8; b * f * 4],
            ],
            3,
        );
        check(
            "rmsnorm",
            b,
            f,
            1,
            &got,
            &ref_rmsnorm(&x, &g, b, f, eps),
            1e-3,
        );
    }
}

// ─── KC-6: Gelu (ONNX Gelu approximate="tanh") + Silu (x·σ(x)) ────────

fn ref_gelu(x: f32) -> f32 {
    let x = f64::from(x);
    let c = (2.0f64 / core::f64::consts::PI).sqrt();
    (0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())) as f32
}
fn ref_silu(x: f32) -> f32 {
    let x = f64::from(x);
    (x / (1.0 + (-x).exp())) as f32
}

fn run_unary(call: KernelCall, x: &[f32]) -> Vec<f32> {
    run(call, vec![f32_to_le(x), vec![0u8; x.len() * 4]], 1)
}

#[test]
fn kc6_gelu_silu_conform_across_scale() {
    for &n in &[4usize, 257, 4096] {
        let x = fill(n, 0x600 + n as u64);
        let unary = |op: fn(UnaryCall) -> KernelCall| {
            run_unary(
                op(UnaryCall {
                    input: buf(0),
                    output: buf(1),
                    element_count: n as u64,
                    witt_bits: 32,
                    dtype: DTYPE_F32,
                }),
                &x,
            )
        };
        let gelu = unary(KernelCall::Gelu);
        check(
            "gelu",
            n,
            1,
            1,
            &gelu,
            &x.iter().map(|&v| ref_gelu(v)).collect::<Vec<_>>(),
            1e-3,
        );
        let silu = unary(KernelCall::Silu);
        check(
            "silu",
            n,
            1,
            1,
            &silu,
            &x.iter().map(|&v| ref_silu(v)).collect::<Vec<_>>(),
            1e-4,
        );
    }
}

// ─── KC-7: Conv2d (ONNX Conv, NCHW, valid cross-correlation) ──────────

#[allow(clippy::too_many_arguments)]
fn ref_conv2d(
    x: &[f32],
    w: &[f32],
    b: usize,
    cin: usize,
    cout: usize,
    hi: usize,
    wi: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
) -> (Vec<f32>, usize, usize) {
    let ho = (hi - kh) / sh + 1;
    let wo = (wi - kw) / sw + 1;
    let mut o = vec![0f32; b * cout * ho * wo];
    for bi in 0..b {
        for co in 0..cout {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc = 0f64;
                    for ci in 0..cin {
                        for y in 0..kh {
                            for xk in 0..kw {
                                let ih = oh * sh + y;
                                let iw = ow * sw + xk;
                                let xi = ((bi * cin + ci) * hi + ih) * wi + iw;
                                let wi_ = ((co * cin + ci) * kh + y) * kw + xk;
                                acc += f64::from(x[xi]) * f64::from(w[wi_]);
                            }
                        }
                    }
                    o[((bi * cout + co) * ho + oh) * wo + ow] = acc as f32;
                }
            }
        }
    }
    (o, ho, wo)
}

#[test]
fn kc7_conv2d_conforms_across_scale() {
    for (idx, &(b, cin, cout, hi, wi, kh, kw, sh, sw)) in [
        (
            1usize, 1usize, 1usize, 5usize, 5usize, 3usize, 3usize, 1usize, 1usize,
        ),
        (2, 3, 4, 16, 16, 3, 3, 1, 1),
        (1, 3, 8, 31, 29, 5, 3, 2, 2),
    ]
    .iter()
    .enumerate()
    {
        let x = fill(b * cin * hi * wi, 0x700 + idx as u64);
        let w = fill(cout * cin * kh * kw, 0x710 + idx as u64);
        let (want, ho, wo) = ref_conv2d(&x, &w, b, cin, cout, hi, wi, kh, kw, sh, sw);
        let got = run(
            KernelCall::Conv2d(Conv2dCall {
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
                stride_h: sh as u32,
                stride_w: sw as u32,
                pad_h: 0,
                pad_w: 0,
                dtype: DTYPE_F32,
            }),
            vec![
                f32_to_le(&x),
                f32_to_le(&w),
                vec![0u8; b * cout * ho * wo * 4],
            ],
            2,
        );
        check("conv2d", b, cin, cout, &got, &want, 1e-4);
    }
}

// ─── KC-8: Attention (scaled dot-product, per head, scale = √d) ───────

fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
) -> Vec<f32> {
    let scale = (d as f64).sqrt().max(1.0);
    let mut o = vec![0f32; b * h * s * d];
    for bi in 0..b {
        for hi in 0..h {
            let off = (bi * h + hi) * s * d;
            for qi in 0..s {
                let mut scores = vec![0f64; s];
                for (kj, sc) in scores.iter_mut().enumerate() {
                    let mut acc = 0f64;
                    for di in 0..d {
                        acc += f64::from(q[off + qi * d + di]) * f64::from(k[off + kj * d + di]);
                    }
                    *sc = acc / scale;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|&sc| (sc - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for di in 0..d {
                    let mut acc = 0f64;
                    for (kj, &e) in exps.iter().enumerate() {
                        acc += (e / sum) * f64::from(v[off + kj * d + di]);
                    }
                    o[off + qi * d + di] = acc as f32;
                }
            }
        }
    }
    o
}

#[test]
fn kc8_attention_conforms_across_scale() {
    for (idx, &(b, h, s, d)) in [
        (1usize, 1usize, 4usize, 8usize),
        (2, 4, 16, 32),
        (1, 8, 31, 16),
    ]
    .iter()
    .enumerate()
    {
        let n = b * h * s * d;
        let q = fill(n, 0x800 + idx as u64);
        let k = fill(n, 0x810 + idx as u64);
        let v = fill(n, 0x820 + idx as u64);
        let got = run(
            KernelCall::Attention(AttentionCall {
                q: buf(0),
                k: buf(1),
                v: buf(2),
                output: buf(3),
                batch: b as u32,
                heads: h as u32,
                seq: s as u32,
                head_dim: d as u32,
                dtype: DTYPE_F32,
            }),
            vec![
                f32_to_le(&q),
                f32_to_le(&k),
                f32_to_le(&v),
                vec![0u8; n * 4],
            ],
            3,
        );
        check(
            "attention",
            b,
            h,
            s,
            &got,
            &ref_attention(&q, &k, &v, b, h, s, d),
            1e-4,
        );
    }
}

// ─── KC-9: Dequantize (ONNX DequantizeLinear: (q − zp)·scale) ─────────

#[test]
fn kc9_dequantize_conforms() {
    let n = 256usize;
    let scale = 0.0125f32;
    let zp = 7i32;
    let q: Vec<u8> = (0..n).map(|i| (i as i32 - 128) as i8 as u8).collect();
    let want: Vec<f32> = q
        .iter()
        .map(|&b| ((b as i8 as i32) - zp) as f32 * scale)
        .collect();
    let got = run(
        KernelCall::Dequantize(DequantizeCall {
            input: buf(0),
            output: buf(1),
            element_count: n as u64,
            quant_dtype: DTYPE_I8,
            dtype: DTYPE_F32,
            scale_bits: scale.to_bits(),
            zero_point: zp,
        }),
        vec![q.clone(), vec![0u8; n * 4]],
        1,
    );
    check("dequantize", n, 1, 1, &got, &want, 1e-6);
}

// ─── KC-10: Reduce (ONNX ReduceSum/ReduceMean/ReduceMax, all axes) ────

fn reduce_call(kind: fn(ReduceCall) -> KernelCall, x: &[f32]) -> f32 {
    let n = x.len();
    run(
        kind(ReduceCall {
            input: buf(0),
            output: buf(1),
            element_count: n as u64,
            axis_count: 1,
            keepdims: false,
            dtype: DTYPE_F32,
        }),
        vec![f32_to_le(x), vec![0u8; 64]],
        1,
    )[0]
}

#[test]
fn kc10_reduce_conforms_across_scale() {
    for &n in &[4usize, 257, 4096] {
        let x = fill(n, 0xA00 + n as u64);
        let sum: f64 = x.iter().map(|&v| f64::from(v)).sum();
        let max: f64 = x
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let rel = |got: f32, want: f64| (f64::from(got) - want).abs() / (want.abs() + 1e-6);
        assert!(
            rel(reduce_call(KernelCall::ReduceSum, &x), sum) <= 1e-4,
            "ReduceSum n={n}"
        );
        assert!(
            rel(reduce_call(KernelCall::ReduceMean, &x), sum / n as f64) <= 1e-4,
            "ReduceMean n={n}"
        );
        assert!(
            rel(reduce_call(KernelCall::ReduceMax, &x), max) <= 1e-6,
            "ReduceMax n={n}"
        );
    }
}

// ─── KC-11: Pooling (ONNX MaxPool / AveragePool, NCHW, valid window) ──

#[allow(clippy::too_many_arguments)]
fn ref_pool(
    x: &[f32],
    b: usize,
    ch: usize,
    hi: usize,
    wi: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    max: bool,
) -> (Vec<f32>, usize, usize) {
    let ho = (hi - kh) / sh + 1;
    let wo = (wi - kw) / sw + 1;
    let mut o = vec![0f32; b * ch * ho * wo];
    for bi in 0..b {
        for ci in 0..ch {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc = if max { f64::NEG_INFINITY } else { 0.0 };
                    let mut cnt = 0u32;
                    for y in 0..kh {
                        for xk in 0..kw {
                            let v = f64::from(
                                x[((bi * ch + ci) * hi + oh * sh + y) * wi + ow * sw + xk],
                            );
                            if max {
                                acc = acc.max(v);
                            } else {
                                acc += v;
                            }
                            cnt += 1;
                        }
                    }
                    let r = if max { acc } else { acc / f64::from(cnt) };
                    o[((bi * ch + ci) * ho + oh) * wo + ow] = r as f32;
                }
            }
        }
    }
    (o, ho, wo)
}

#[test]
fn kc11_pooling_conforms_across_scale() {
    for (idx, &(b, ch, hi, wi, kh, kw, sh, sw)) in [
        (
            1usize, 1usize, 8usize, 8usize, 2usize, 2usize, 2usize, 2usize,
        ),
        (2, 3, 31, 29, 3, 3, 2, 2),
    ]
    .iter()
    .enumerate()
    {
        let x = fill(b * ch * hi * wi, 0xB00 + idx as u64);
        for &max in &[true, false] {
            let (want, ho, wo) = ref_pool(&x, b, ch, hi, wi, kh, kw, sh, sw, max);
            let op = if max {
                KernelCall::MaxPool2d
            } else {
                KernelCall::AvgPool2d
            };
            let got = run(
                op(PoolCall {
                    x: buf(0),
                    output: buf(1),
                    batch: b as u32,
                    channels: ch as u32,
                    h_in: hi as u32,
                    w_in: wi as u32,
                    h_out: ho as u32,
                    w_out: wo as u32,
                    k_h: kh as u32,
                    k_w: kw as u32,
                    stride_h: sh as u32,
                    stride_w: sw as u32,
                    dtype: DTYPE_F32,
                }),
                vec![f32_to_le(&x), vec![0u8; b * ch * ho * wo * 4]],
                1,
            );
            check(
                if max { "maxpool" } else { "avgpool" },
                b,
                ch,
                1,
                &got,
                &want,
                1e-4,
            );
        }
    }
}

// ─── KC-DT: dtype-support policy (no silent fallbacks) ────────────────

/// f64 must be **rejected explicitly**, never silently computed at reduced
/// precision or produce zeros. The engine computes in f16/bf16/f32; f64 is a
/// super-f32 storage format with no native kernel, so dispatch returns an
/// error rather than the historical silent-zero output.
#[test]
fn kcdt_f64_rejected_never_silent_zero() {
    let d = 4usize;
    let mut ws = TestWorkspace {
        slots: vec![vec![0u8; d * d * 8], vec![0u8; d * d * 8], vec![1u8; d * d * 8]],
    };
    let call = KernelCall::MatMul(MatMulCall {
        a: buf(0),
        b: buf(1),
        output: buf(2),
        m: d as u32,
        k: d as u32,
        n: d as u32,
        dtype: DTYPE_F64,
        b_packed: false,
    });
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let r = backend.dispatch(&call, &mut ws);
    assert!(
        r.is_err(),
        "f64 matmul must error explicitly, not silently compute/zero"
    );
}

/// Division and modulo by zero follow IEEE-754 (±∞ / NaN), not a silent 0.0
/// substitution — a wrong-but-quiet result is worse than a correct special
/// value the caller can detect.

#[test]
fn kcdt_div_mod_by_zero_is_ieee() {
    let a = [1.0f32, -1.0, 0.0, 5.0];
    let b = [0.0f32, 0.0, 0.0, 2.0];
    let mut ws = TestWorkspace {
        slots: vec![f32_to_le(&a), f32_to_le(&b), vec![0u8; 16]],
    };
    let call = KernelCall::Div(BinaryCall {
        a: buf(0),
        b: buf(1),
        output: buf(2),
        element_count: 4,
        witt_bits: 32,
        dtype: DTYPE_F32,
    });
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    backend.dispatch(&call, &mut ws).unwrap();
    let out = le_to_f32(&ws.slots[2]);
    assert!(out[0].is_infinite() && out[0] > 0.0, "1/0 must be +∞");
    assert!(out[1].is_infinite() && out[1] < 0.0, "-1/0 must be -∞");
    assert!(out[2].is_nan(), "0/0 must be NaN");
    assert!((out[3] - 2.5).abs() < 1e-6, "5/2 must be 2.5");
}

// ─── KC-7b / KC-8b: low-precision conv & attention route through engine ──

fn enc_bf16(vals: &[f32]) -> (Vec<u8>, Vec<f32>) {
    use hologram_backend::cpu::dtype::{read_bf16, write_bf16};
    let mut bytes = vec![0u8; vals.len() * 2];
    for (i, &v) in vals.iter().enumerate() {
        write_bf16(&mut bytes, i, v);
    }
    let q = (0..vals.len()).map(|i| read_bf16(&bytes, i)).collect();
    (bytes, q)
}

#[test]
fn kc7b_bf16_conv_routes_through_engine() {
    use hologram_backend::cpu::dtype::{read_bf16, DTYPE_BF16};
    let (b, cin, cout, hi, wi, kh, kw, sh, sw) = (2usize, 3, 4, 16, 16, 3, 3, 1, 1);
    let x = fill(b * cin * hi * wi, 0x900);
    let w = fill(cout * cin * kh * kw, 0x910);
    let (xb, xq) = enc_bf16(&x);
    let (wb, wq) = enc_bf16(&w);
    let (want, ho, wo) = ref_conv2d(&xq, &wq, b, cin, cout, hi, wi, kh, kw, sh, sw);
    let mut ws = TestWorkspace {
        slots: vec![xb, wb, vec![0u8; b * cout * ho * wo * 2]],
    };
    let call = KernelCall::Conv2d(Conv2dCall {
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
        stride_h: sh as u32,
        stride_w: sw as u32,
        pad_h: 0,
        pad_w: 0,
        dtype: DTYPE_BF16,
    });
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    backend.dispatch(&call, &mut ws).unwrap();
    let got: Vec<f32> = (0..b * cout * ho * wo)
        .map(|i| read_bf16(&ws.slots[2], i))
        .collect();
    // bf16 accumulation in f32 then narrow: error dominated by bf16 rounding.
    let err = max_rel_err(&got, &want);
    assert!(err <= 3e-2, "bf16 conv rel err {err:.3e} > 3e-2");
    assert!(got.iter().any(|&v| v.abs() > 1e-6), "bf16 conv all zeros");
}

#[test]
fn kc8b_bf16_attention_routes_through_engine() {
    use hologram_backend::cpu::dtype::{read_bf16, DTYPE_BF16};
    let (ab, ah, asq, ad) = (2usize, 2, 8, 16);
    let n = ab * ah * asq * ad;
    let qf = fill(n, 0xA00);
    let kf = fill(n, 0xA10);
    let vf = fill(n, 0xA20);
    let (qb, qq) = enc_bf16(&qf);
    let (kb, kq) = enc_bf16(&kf);
    let (vb, vq) = enc_bf16(&vf);
    let want = ref_attention(&qq, &kq, &vq, ab, ah, asq, ad);
    let mut ws = TestWorkspace {
        slots: vec![qb, kb, vb, vec![0u8; n * 2]],
    };
    let call = KernelCall::Attention(AttentionCall {
        q: buf(0),
        k: buf(1),
        v: buf(2),
        output: buf(3),
        batch: ab as u32,
        heads: ah as u32,
        seq: asq as u32,
        head_dim: ad as u32,
        dtype: DTYPE_BF16,
    });
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    backend.dispatch(&call, &mut ws).unwrap();
    let got: Vec<f32> = (0..n).map(|i| read_bf16(&ws.slots[3], i)).collect();
    let err = max_rel_err(&got, &want);
    assert!(err <= 3e-2, "bf16 attention rel err {err:.3e} > 3e-2");
    assert!(got.iter().any(|&v| v.abs() > 1e-6), "bf16 attention all zeros");
}
