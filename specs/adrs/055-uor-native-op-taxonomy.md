# ADR-055: UOR-native op taxonomy — no fallbacks, pipeline-lowering of composites

**Status:** Accepted 2026-05-25
**Relates to:** ADR-054 (Hologram as a Prism application)

## Context

The CPU backend had accumulated *silent fallbacks* that violate the
UOR-native contract: a kernel must either compute the operator correctly or
refuse — never quietly return a wrong/degraded result. Concretely:

- compute kernels (conv2d, attention, gemm) dropped off the cache-oblivious
  engine into scalar triple-loops for non-f32 dtypes;
- `f64` was tagged a float dtype but `read_float`/`write_float` returned
  `0.0` / no-op, so every f64 kernel silently produced zeros;
- `div`/`mod` by zero returned `0.0` instead of IEEE ±∞/NaN;
- several ops (`Clip`, `RotaryEmbedding`, `Lrn`, `FusedSwiGlu`) had no place
  in the call representation for their parameters and silently behaved as
  identity / plain-matmul;
- the layout ops `Transpose`/`Slice`/`Concat`/`Pad`/`Expand`/`Resize` were
  `memcpy` stubs (`LayoutCall` carries no axis/offset/permutation and a single
  input), silently wrong for any real transform.

## Decision

Every operator falls into exactly one of four classes, and the backend holds
the line that **nothing computes silently-wrong**:

1. **Irreducible structured kernels** — `MatMul`, `Gemm`, `Conv2d`,
   `Attention`, the norms, `Softmax`, pooling, reductions. These are single
   optimized kernels (like BLAS). *All supported float dtypes route through
   the one cache-oblivious f32 engine*: f32 zero-copy, f16/bf16 widen→engine→
   narrow (sub-f32 storage formats whose native semantics *are* f32
   accumulation). No scalar fallback; the residual arm is an explicit error.

2. **True relabels** — `Reshape` only. A row-major buffer's bytes are
   unchanged by a logical shape change, so a dtype-aware byte copy is correct.

3. **Composite ops → primitive pipelines (Path B)** — a composite op has no
   kernel of its own; its meaning *is* a composition of primitives, so the
   compiler **desugars** it (`Graph::desugar_composites`, run at the top of
   `compile()`) into the sequence of primitive graph nodes that computes it,
   reusing the already-verified primitive kernels and the ordinary node→slot
   model (every intermediate is a real node with its own output slot — no
   special buffer machinery, free content-addressing/warm-start). Done:
   - `Clip(x, lo, hi) → Min(Max(x, lo), hi)`
   - `FusedSwiGlu(x, Wg, Wu) → Mul(Silu(MatMul(x, Wg)), MatMul(x, Wu))`
   The rewrite is topology-preserving: it remaps every `InputSource::Node`,
   the input/output port lists and the sparse attr tables; constants/shapes
   are untouched. This is the UOR-native realization of "ops as PrimitiveOp
   pipelines" (ADR-054) — the op's Term-tree structure *is* its lowering.

4. **Declared-but-unimplemented → fail loud** — ops whose correct
   implementation needs representation or primitives that do not yet exist
   return `BackendError::UnsupportedOp` in *every* domain (float and byte),
   rather than a silent approximation:
   - `RotaryEmbedding` — needs `rotate_half` = functional `Slice`+`Neg`+
     `Concat`; blocked on class-(2)→functional layout ops below.
   - `Lrn` — needs a windowed-channel reduction primitive (none exists).
   - layout transforms `Transpose`/`Slice`/`Concat`/`Pad`/`Expand`/`Resize`
     — need a call form carrying axis/offsets/permutation and (for Concat)
     multiple inputs, plus real indexed kernels.
   - the binary/unary gradient ops — backward pipelines (training path).

### dtype-support policy

`call_dtype(&KernelCall)` (exhaustive over the catalog) is the single policy
point: a `f64` call is rejected at the top of float dispatch. Computing a
64-bit type at f32 precision would be a silent downgrade, so f64 is refused
outright (no model frontend emits it today).

## Verification (V&V)

- `tests/zero_overhead.rs` — per-thread counting allocator proves matmul /
  gemm / conv2d / attention perform **0 heap allocations per call after
  warm-up** (the zero-cost/zero-copy contract, executable).
- conformance `KC-1b`/`KC-7b`/`KC-8b` — bf16 matmul/conv/attention route
  through the engine and match the f64 reference; `KCDT-*` — f64 reject and
  IEEE div/mod.
- `hologram-graph` desugar unit tests + `hologram-exec/tests/desugar.rs` —
  Clip clamp and SwiGLU executed end-to-end vs f64 reference (err ≤ 1e-4).

## Consequences

Hologram now has **no silent-wrong path**: an op either computes correctly or
errors. Composite completion is mechanical going forward (add a desugar rule
once its primitives are functional). The remaining roadmap is the connected
"functional layout ops → RoPE; windowed reduction → Lrn; backward pipelines →
gradients; frontend constant/shape grammar" project tracked against this ADR.
