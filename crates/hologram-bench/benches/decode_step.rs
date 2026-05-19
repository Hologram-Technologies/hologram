//! Transformer-block decode step on the canonical backends.
//!
//! Simulates the forward pass of a single Llama-style decoder block at
//! decode shape (`seq_q = 1`, `seq_kv = S`) using only canonical
//! [`KernelCall`]s, then reports per-iteration latency on every
//! backend that implements `CanonicalBackend`:
//!
//!   * **cpu** — `CpuBackend`, the reference dispatcher (always built).
//!   * **wgpu** — `WgpuBackend`, gated on `--features webgpu`. On macOS
//!     this adapter is Metal under the hood, so the wgpu numbers *are*
//!     the Metal numbers — there is no separate Metal canonical
//!     backend (see `crates/hologram-backend/src/canonical/`). On Linux
//!     it's Vulkan; on Windows, DX12.
//!
//! The block, per iteration:
//!
//! ```text
//!   normed1   = rms_norm(x, w_pre)                          (1) RmsNorm
//!   q,k,v     = normed1 @ {Wq, Wk, Wv}                      (3) MatMul
//!   scores    = q @ K_cacheᵀ                                (1) MatMul
//!   weights   = softmax(scores)                             (1) Softmax
//!   ctx       = weights @ V_cache                           (1) MatMul
//!   attn_out  = ctx @ Wo                                    (1) MatMul
//!   resid     = x + attn_out                                (1) Add
//!   normed2   = rms_norm(resid, w_post)                     (1) RmsNorm
//!   gate, up  = normed2 @ {Wg, Wu}                          (2) MatMul
//!   ffh       = silu(gate) * up                             (1) FusedSwiGlu
//!   ffn_out   = ffh @ Wd                                    (1) MatMul
//!   out       = resid + ffn_out                             (1) Add
//! ```
//!
//! That's 8 MatMuls, 2 RmsNorms, 1 Softmax, 1 SwiGLU, 2 Adds — the
//! decode hot path of every modern dense LLM. Multi-head attention
//! is folded into a single per-head matmul pair (Q@Kᵀ, softmax, w@V):
//! a real implementation would either run `num_heads` of these or use
//! the canonical `Attention` op, but neither has a wgpu native shader
//! today (canonical Attention falls through to host). Single-head
//! shape is representative of the per-head cost and stays fully on
//! the device.
//!
//! Weights and the K/V cache are seeded **once**, before the timing
//! loop, and live in the device-resident workspace for the whole bench
//! run. Per iteration the bench only writes the new token's input `x`
//! and reads the block's output — exactly the host-↔-device traffic
//! a real serving loop pays per generated token.
//!
//! Run:
//!
//! ```bash
//! # CPU only:
//! cargo bench -p hologram-bench --bench decode_step
//! # CPU + wgpu (= Metal on macOS):
//! cargo bench -p hologram-bench --features webgpu --bench decode_step
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_transform::{
    AddCall, AttentionCall, BackendWorkspace, BinaryCall, CanonicalBackend, CpuBackend, KernelCall,
    MatMulCall, NormScaleCall, SlotSpan, SoftmaxCall,
};

/// wgpu storage-buffer offsets must be 64-element-aligned (256 bytes
/// at f32). Aligning every span keeps adjacent slots from violating
/// `min_storage_buffer_offset_alignment`. The CPU backend doesn't care
/// but pays nothing for the extra padding either.
const ALIGN: usize = 64;

#[inline]
fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// Bump-allocator over a flat `f32` workspace. One `alloc(len)` call
/// per slot returns the `SlotSpan` and advances `capacity` by an
/// `ALIGN`-rounded stride.
struct Layout {
    capacity: usize,
}

impl Layout {
    fn new() -> Self {
        Self { capacity: 0 }
    }

    fn alloc(&mut self, len: usize) -> SlotSpan {
        let span = SlotSpan {
            offset: self.capacity,
            len,
        };
        self.capacity += align_up(len);
        span
    }
}

/// Slot map for one decoder block — the planner-equivalent output of
/// laying every tensor onto the resident workspace.
struct Block {
    // Inputs (rewritten per iteration)
    x: SlotSpan,
    out: SlotSpan,
    // Constants (seeded once, read every iteration).
    //
    // `wqkv` is the QKV-projection weight after fusion: a single
    // `[d, 3d]` matrix laid out as row-i = `[Wq_row_i | Wk_row_i |
    // Wv_row_i]`. One MatMul against `normed1` produces a contiguous
    // `[1, 3d]` output that we then read as three adjacent slots
    // (Q, K, V) without copying. Replaces three separate Q/K/V
    // matmuls — saves two dispatches per token.
    wqkv: SlotSpan,
    wo: SlotSpan,
    // `wgu` is the analogous fusion for the SwiGLU MLP's two gating
    // projections — `[d, 2*ff]` row-major, row-i = `[Wg_i | Wu_i]`.
    // One MatMul produces `[gate | up]` contiguously; FusedSwiGlu
    // reads each half as a sub-span. Replaces two matmuls — saves
    // one dispatch per token.
    wgu: SlotSpan,
    wd: SlotSpan,
    rms_pre_w: SlotSpan,
    rms_post_w: SlotSpan,
    k_cache: SlotSpan,
    v_cache: SlotSpan,
    // Intermediates (overwritten each iteration)
    capacity: usize,
}

/// Build the slot layout and the per-iteration call sequence for a
/// `(d, ff, s)` block shape.
///
/// `use_fused_attention` selects the attention path:
/// * `true` — emit one canonical `Attention` op, which `WgpuBackend`
///   routes to the fused `attention_decode` shader. Wins at narrow
///   `head_dim` where dispatch overhead dominates.
/// * `false` — emit the explicit `q@K + softmax + w@V` triple. Wins
///   at wide `head_dim`, where the canonical shader's single
///   workgroup can't keep the GPU busy and the matmul-based path's
///   many workgroups in parallel are faster (the q@K matmul
///   dispatches `seq_kv/64` workgroups, the w@V matmul dispatches
///   `head_dim/64` — together that's many more parallel SIMD groups
///   than a single 256-thread attention workgroup can host on one
///   EU). Until the canonical shader gets multi-workgroup
///   FlashAttention, the bench picks the better path per shape.
fn build_block(
    d: usize,
    ff: usize,
    s: usize,
    use_fused_attention: bool,
) -> (Block, Vec<KernelCall>) {
    let mut l = Layout::new();

    // External / persistent slots.
    let x = l.alloc(d);
    // Fused QKV weight: `[d, 3d]` row-major, row-i = `[Wq_i | Wk_i |
    // Wv_i]`.
    let wqkv = l.alloc(d * 3 * d);
    let wo = l.alloc(d * d);
    // Fused gate/up weight: `[d, 2*ff]` row-major, row-i = `[Wg_i | Wu_i]`.
    let wgu = l.alloc(d * 2 * ff);
    let wd = l.alloc(ff * d);
    let rms_pre_w = l.alloc(d);
    let rms_post_w = l.alloc(d);
    // K and V caches in canonical Attention layout — `[seq_kv, head_dim]`
    // row-major (each row is one cached token's projection). This is
    // the layout the new `attention_decode` shader expects: contiguous
    // head_dim within each k-position. The previous bench layout was
    // `[d, s]` (pre-transposed for the `q @ K` matmul); the dedicated
    // attention path doesn't need that workaround.
    let k_cache = l.alloc(s * d);
    let v_cache = l.alloc(s * d);

    // Per-iter intermediates.
    let normed1 = l.alloc(d);
    // Single `[1, 3d]` slot holding the fused projection output. Q/K/V
    // are sub-spans into it — no copy, just reinterpretation. The
    // sub-span offsets stay 64-element-aligned for every shape we
    // bench (`d` is always a multiple of 64). K and V live at
    // `qkv.offset + d` and `qkv.offset + 2*d` but the bench never
    // reads them — the K/V cache is pre-seeded since we don't model
    // cache append in the canonical op set today.
    let qkv = l.alloc(3 * d);
    debug_assert_eq!(d % ALIGN, 0, "sub-span Q/K/V layout requires d % 64 == 0");
    let q = SlotSpan {
        offset: qkv.offset,
        len: d,
    };
    let ctx = l.alloc(d);
    let attn_out = l.alloc(d);
    let resid = l.alloc(d);
    let normed2 = l.alloc(d);
    // Single `[1, 2*ff]` slot holding the fused gate/up output.
    // `gate` and `up` are sub-spans into it. Sub-span alignment holds
    // for every shape we bench (`ff` is always a multiple of 64).
    let gate_up = l.alloc(2 * ff);
    debug_assert_eq!(
        ff % ALIGN,
        0,
        "sub-span gate/up layout requires ff % 64 == 0"
    );
    let gate = SlotSpan {
        offset: gate_up.offset,
        len: ff,
    };
    let up = SlotSpan {
        offset: gate_up.offset + ff,
        len: ff,
    };
    let ffh = l.alloc(ff);
    let ffn_out = l.alloc(d);
    let out = l.alloc(d);
    // `scores` and `weights` are only consumed by the explicit-
    // decomposition path; the fused-attention path keeps them in
    // workgroup-local memory. Allocated last so the offsets of every
    // other slot stay identical to the fused-attention layout — keeps
    // the two paths' workspace shape-agnostic for cache-locality
    // comparisons.
    let scores = l.alloc(s);
    let weights = l.alloc(s);

    let eps = 1.0e-5_f32.to_bits();

    // Attention sub-sequence: one fused canonical op or the explicit
    // q@K + softmax + w@V triple, depending on shape (see fn header).
    //
    // The two paths interpret the K cache differently — the explicit
    // matmul reads it as `[d, seq_kv]` (transposed), the fused shader
    // as `[seq_kv, d]`. The bytes are identical (the seed is just
    // pseudo-random), so the same fill seeds either path; only the
    // shader's access pattern differs. Math correctness isn't the
    // point of the bench — we're measuring dispatch + compute cost.
    let attention_calls: Vec<KernelCall> = if use_fused_attention {
        // No causal mask needed at `seq_q == 1`: the canonical rule
        // `k > q + (seq_kv - seq_q)` collapses to false for every k.
        vec![KernelCall::Attention(AttentionCall {
            q,
            k: k_cache,
            v: v_cache,
            output: ctx,
            scratch: SlotSpan { offset: 0, len: 0 },
            batch: 1,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: d as u32,
            seq_q: 1,
            seq_kv: s as u32,
            scale_bits: (1.0_f32 / (d as f32).sqrt()).to_bits(),
            causal: false,
        })]
    } else {
        vec![
            KernelCall::MatMul(MatMulCall {
                a: q,
                b: k_cache,
                c: scores,
                m: 1,
                k: d,
                n: s,
            }),
            KernelCall::Softmax(SoftmaxCall {
                input: scores,
                output: weights,
                size: s,
            }),
            KernelCall::MatMul(MatMulCall {
                a: weights,
                b: v_cache,
                c: ctx,
                m: 1,
                k: s,
                n: d,
            }),
        ]
    };

    let mut calls: Vec<KernelCall> = vec![
        // Pre-attention norm.
        KernelCall::RmsNorm(NormScaleCall {
            input: x,
            weight: rms_pre_w,
            output: normed1,
            size: d as u32,
            epsilon: eps,
        }),
        // Fused Q/K/V projection: one matmul against the concatenated
        // weight produces `[Q | K | V]` contiguously into `qkv`. The Q,
        // K, V sub-spans created above point at the three thirds.
        KernelCall::MatMul(MatMulCall {
            a: normed1,
            b: wqkv,
            c: qkv,
            m: 1,
            k: d,
            n: 3 * d,
        }),
    ];
    calls.extend(attention_calls);
    calls.extend([
        // Output projection.
        KernelCall::MatMul(MatMulCall {
            a: ctx,
            b: wo,
            c: attn_out,
            m: 1,
            k: d,
            n: d,
        }),
        // Attention residual.
        KernelCall::Add(AddCall {
            a: x,
            b: attn_out,
            c: resid,
        }),
        // Post-attention norm.
        KernelCall::RmsNorm(NormScaleCall {
            input: resid,
            weight: rms_post_w,
            output: normed2,
            size: d as u32,
            epsilon: eps,
        }),
        // Fused gate/up projection: one matmul against the concatenated
        // weight produces `[gate | up]` contiguously into `gate_up`.
        // The `gate` and `up` sub-spans created above point at each half.
        KernelCall::MatMul(MatMulCall {
            a: normed2,
            b: wgu,
            c: gate_up,
            m: 1,
            k: d,
            n: 2 * ff,
        }),
        // Fused SwiGLU: silu(gate) * up.
        KernelCall::FusedSwiGlu(BinaryCall {
            a: gate,
            b: up,
            c: ffh,
        }),
        // Down projection.
        KernelCall::MatMul(MatMulCall {
            a: ffh,
            b: wd,
            c: ffn_out,
            m: 1,
            k: ff,
            n: d,
        }),
        // FFN residual.
        KernelCall::Add(AddCall {
            a: resid,
            b: ffn_out,
            c: out,
        }),
    ]);

    (
        Block {
            x,
            out,
            wqkv,
            wo,
            wgu,
            wd,
            rms_pre_w,
            rms_post_w,
            k_cache,
            v_cache,
            capacity: l.capacity,
        },
        calls,
    )
}

/// Build the seed for the fused `Wqkv` weight by interleaving three
/// independent `[d, d]` row-major matrices into a `[d, 3d]` row-major
/// layout: row-i of `Wqkv` = `[Wq_i | Wk_i | Wv_i]`. Same per-element
/// values as the unfused version (salts 1/2/3) so a numerically-equivalent
/// reference run would produce the same Q/K/V.
fn fill_wqkv(d: usize) -> Vec<f32> {
    let wq = fill(d * d, 1);
    let wk = fill(d * d, 2);
    let wv = fill(d * d, 3);
    let mut out = vec![0.0_f32; d * 3 * d];
    for ki in 0..d {
        let dst = ki * 3 * d;
        let src = ki * d;
        out[dst..dst + d].copy_from_slice(&wq[src..src + d]);
        out[dst + d..dst + 2 * d].copy_from_slice(&wk[src..src + d]);
        out[dst + 2 * d..dst + 3 * d].copy_from_slice(&wv[src..src + d]);
    }
    out
}

/// Build the seed for the fused `Wgu` weight: `[d, 2*ff]` row-major,
/// row-i = `[Wg_i | Wu_i]`. Mirrors `fill_wqkv` for the gate/up pair.
fn fill_wgu(d: usize, ff: usize) -> Vec<f32> {
    let wg = fill(d * ff, 5);
    let wu = fill(d * ff, 6);
    let mut out = vec![0.0_f32; d * 2 * ff];
    for ki in 0..d {
        let dst = ki * 2 * ff;
        let src = ki * ff;
        out[dst..dst + ff].copy_from_slice(&wg[src..src + ff]);
        out[dst + ff..dst + 2 * ff].copy_from_slice(&wu[src..src + ff]);
    }
    out
}

/// Deterministic pseudo-random seed in `[-1, 1]` derived from the
/// element index — same seed across backends, so any divergence is the
/// backend's, not the data's.
fn fill(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            // Simple LCG-ish scramble — enough variation to keep
            // softmax non-degenerate without pulling in a real RNG.
            let mut h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
            h ^= h >> 13;
            (h as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Seed every persistent slot in the workspace so the per-iter timing
/// only measures the new-token path.
fn seed_constants<W: BackendWorkspace>(ws: &mut W, b: &Block, d: usize, ff: usize, s: usize) {
    let writes: &[(SlotSpan, Vec<f32>)] = &[
        (b.wqkv, fill_wqkv(d)),
        (b.wo, fill(d * d, 4)),
        (b.wgu, fill_wgu(d, ff)),
        (b.wd, fill(ff * d, 7)),
        // RMS weights: small positive perturbation around 1.0 keeps
        // the norm well-conditioned.
        (
            b.rms_pre_w,
            fill(d, 8).iter().map(|v| 1.0 + 0.01 * v).collect(),
        ),
        (
            b.rms_post_w,
            fill(d, 9).iter().map(|v| 1.0 + 0.01 * v).collect(),
        ),
        (b.k_cache, fill(d * s, 10)),
        (b.v_cache, fill(s * d, 11)),
    ];
    for (span, data) in writes {
        ws.write_span(*span, data).unwrap();
    }
}

/// One iteration of the bench harness: stamp the new token in, run the
/// block, drain the output.
///
/// Uses [`CanonicalBackend::run_resident`] (not a per-call
/// `dispatch_resident` loop) so backends that batch into a single
/// command encoder — `WgpuBackend` does — pay one submit per token
/// instead of one per call. The CPU backend's default `run_resident`
/// just loops, so it sees no functional difference.
///
/// `read_span` forces device backends to flush before the iteration
/// times out — without it we'd be timing un-submitted command encoders.
#[inline]
fn run_one_step<B: CanonicalBackend>(
    backend: &mut B,
    ws: &mut B::Workspace,
    block: &Block,
    calls: &[KernelCall],
    x_data: &[f32],
) -> Vec<f32> {
    ws.write_span(block.x, black_box(x_data)).unwrap();
    backend.run_resident(ws, calls).unwrap();
    ws.read_span(block.out).unwrap()
}

/// Variant that runs the block but *skips* the final `read_span`,
/// substituting [`CanonicalBackend::flush`] to wait for GPU completion
/// without paying the staging-copy + map + poll cost.
///
/// Comparing this to [`run_one_step`] isolates readback overhead — the
/// difference is exactly what a real serving loop would save by
/// streaming the residual KV/output state on the device and only
/// reading the final logits at end-of-sequence.
#[cfg(feature = "webgpu")]
#[inline]
fn run_one_step_no_readback<B: CanonicalBackend>(
    backend: &mut B,
    ws: &mut B::Workspace,
    _block: &Block,
    calls: &[KernelCall],
    x_data: &[f32],
) {
    ws.write_span(_block.x, black_box(x_data)).unwrap();
    backend.run_resident(ws, calls).unwrap();
    backend.flush().unwrap();
}

/// Three representative LLM block shapes — toy / small / medium. The
/// largest is roughly TinyLlama per-block (`d=2048, ff=5632, s=2048`)
/// scaled down so a Criterion run finishes in seconds rather than
/// minutes on a workstation CPU.
///
/// The fifth field is `use_fused_attention`: `true` selects the
/// canonical `Attention` op (which `WgpuBackend` routes to a single
/// fused workgroup-cooperative shader); `false` keeps the explicit
/// `q@K + softmax + w@V` triple. The right choice is shape-dependent
/// — see [`build_block`] for the rationale.
const SHAPES: &[(usize, usize, usize, &str, bool)] = &[
    (128, 512, 64, "tiny", true),
    (512, 2048, 256, "small", false),
    (1024, 4096, 512, "medium", false),
];

fn bench_decode_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_step");
    for &(d, ff, s, label, use_fused_attention) in SHAPES {
        let (block, calls) = build_block(d, ff, s, use_fused_attention);
        let x_data = fill(d, 0);

        // Throughput = total f32 reads + writes the sequence touches.
        // Useful as a coarse "did residency cut traffic?" signal even
        // though it under-counts the matmul flops.
        group.throughput(Throughput::Elements(block.capacity as u64));

        // CPU.
        {
            let mut cpu = CpuBackend::new();
            let mut ws = cpu.alloc_workspace(block.capacity).unwrap();
            seed_constants(&mut ws, &block, d, ff, s);
            group.bench_with_input(BenchmarkId::new("cpu", label), &(), |bench, _| {
                bench.iter(|| {
                    let out = run_one_step(&mut cpu, &mut ws, &block, &calls, &x_data);
                    black_box(out);
                });
            });
        }

        // wgpu (= Metal on macOS, Vulkan on Linux, DX12 on Windows).
        bench_wgpu(
            &mut group,
            BenchWgpuArgs {
                block: &block,
                calls: &calls,
                x_data: &x_data,
                d,
                ff,
                s,
                label,
            },
        );
    }
    group.finish();
}

/// Per-shape arguments for [`bench_wgpu`]. Bundles the precomputed
/// plan and the input shapes so the helper stays under
/// `clippy::too_many_arguments`. The criterion group is passed as a
/// separate `&mut` parameter so the caller can keep using it after
/// the helper returns.
// `dead_code` is suppressed: the no-webgpu stub doesn't read the
// fields, but builds are configured with the same struct shape.
#[allow(dead_code)]
struct BenchWgpuArgs<'a> {
    block: &'a Block,
    calls: &'a [KernelCall],
    x_data: &'a [f32],
    d: usize,
    ff: usize,
    s: usize,
    label: &'a str,
}

#[cfg(feature = "webgpu")]
fn bench_wgpu(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    args: BenchWgpuArgs<'_>,
) {
    use hologram_backend::canonical::WgpuBackend;

    let BenchWgpuArgs {
        block,
        calls,
        x_data,
        d,
        ff,
        s,
        label,
    } = args;
    let mut gpu = match WgpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("decode_step: wgpu unavailable, skipping wgpu arm ({e})");
            return;
        }
    };
    let mut ws = gpu.alloc_workspace(block.capacity).unwrap();
    seed_constants(&mut ws, block, d, ff, s);
    group.bench_with_input(BenchmarkId::new("wgpu", label), &(), |bench, _| {
        bench.iter(|| {
            let out = run_one_step(&mut gpu, &mut ws, block, calls, x_data);
            black_box(out);
        });
    });
    // Diagnostic: same block but no final `read_span`. Shows what the
    // step costs when the host doesn't drain the output every token.
    // Both `flush()` and `read_span()` end with the same
    // `device.poll(Maintain::Wait)`, so this isolates the staging
    // copy + buffer-map cost from the GPU-completion wait.
    group.bench_with_input(BenchmarkId::new("wgpu_no_read", label), &(), |bench, _| {
        bench.iter(|| {
            run_one_step_no_readback(&mut gpu, &mut ws, block, calls, x_data);
        });
    });
}

#[cfg(not(feature = "webgpu"))]
fn bench_wgpu(
    _group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    _args: BenchWgpuArgs<'_>,
) {
    // Compiled out: build with `--features webgpu` to include wgpu.
}

criterion_group!(benches, bench_decode_step);
criterion_main!(benches);
