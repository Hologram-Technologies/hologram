# ADR-049: Canonical Scaled Dot-Product Attention

## Status

Accepted (2026-04-27)

## Context

`Attention` is the last canonical op blocking Sprint 37 Phase 3.3
Stage 2. The legacy `FloatOp::Attention` bundles **eight** concerns
into one variant:

```rust
Attention {
    head_dim, num_q_heads, num_kv_heads, scale,
    causal,        // mask flag
    heads_first,   // layout flag (ONNX vs GGUF)
    qk_norm,       // pre-RMSNorm Q and K
    rope,          // pre-RoPE Q and K
    rope_base,     // theta when rope=true
    sparse_v,      // optimisation: skip V accumulation for tiny weights
}
```

That's a *fused execution kernel*, not a single semantic op. Per
ADR-044/045, fused variants stay on the legacy `FloatOp` (or as
`GraphOp::Fused*` planner products) — they are not canonical
sources.

The canonical `Attention` should express **just** the scaled
dot-product attention semantic:
`softmax((Q @ Kᵀ) * scale + mask) @ V`. Anything else — RoPE,
QK-norm, sparse-V — is either a separate canonical op composed
upstream (RoPE, RmsNorm) or an execution-side optimisation that
doesn't belong in semantics (sparse-V).

## Decision

### Canonical surface

```rust
pub struct AttentionAttrs {
    pub head_dim:       u32,
    pub num_q_heads:    u32,
    pub num_kv_heads:   u32,
    pub scale:          u32,    // f32 bits; typically 1/sqrt(head_dim)
    pub causal:         bool,
}
```

Three inputs: `Q`, `K`, `V`. Output: attended values.

### Layout

Canonical layout is **heads-first** in 4-D:

- `Q`: `[batch, num_q_heads, seq_q, head_dim]`
- `K`: `[batch, num_kv_heads, seq_kv, head_dim]`
- `V`: `[batch, num_kv_heads, seq_kv, head_dim]`
- output: `[batch, num_q_heads, seq_q, head_dim]`

Higher leading dims fold into `batch` at the planner. Other layouts
(GGUF-style `[seq, n_heads, head_dim]`, ONNX 3-D) compose by
inserting `Transpose` / `Reshape` upstream.

### What's *not* in the canonical op

- **Layout flags.** The canonical layer specifies one layout. A
  non-canonical layout is a `Transpose` away from canonical; the
  planner can fuse the transpose later.
- **QK normalisation.** `qk_norm = true` is `RmsNorm(Q)` and
  `RmsNorm(K)` upstream — already canonical ops. Composability
  beats fusion at the semantic layer.
- **RoPE integration.** `rope = true` is `RotaryEmbedding(Q)` and
  `RotaryEmbedding(K)` upstream — already canonical ops.
- **Sparse-V.** Pure execution-side perf optimisation. Belongs on
  the kernel implementation flag, not the semantic op identity.
- **Attention mask as input.** Masking has only one semantic
  variant in canonical: `causal: bool`. Arbitrary mask tensors are
  expressed by `Add`-ing a mask before the softmax, which keeps the
  canonical attention op clean. This is the cost-vs-clarity trade
  — canonical wins clarity; specialised masks lose one fused
  kernel call but gain composability.

### Grouped-Query Attention (GQA / MQA)

`num_kv_heads ≤ num_q_heads`, with `num_q_heads % num_kv_heads == 0`.
Each Q head `q_head` reads from KV head
`kv_head = q_head * num_kv_heads / num_q_heads`. This covers
multi-head attention (`num_kv_heads == num_q_heads`),
multi-query attention (`num_kv_heads == 1`), and grouped-query
in between. Validated at planner time.

### Causal mask convention

For cross-attention with `seq_q != seq_kv`, the causal mask hides
positions where the key index is "in the future" relative to the
query. Convention used by Hologram's existing implementation and
the ONNX reference: position `k` is masked for query `q` iff
`k > q + (seq_kv - seq_q)`. For self-attention (seq_q == seq_kv)
this reduces to the standard upper-triangular mask `k > q`.

### Reference kernel

Plain triple-loop:

```text
for b in 0..batch:
  for qh in 0..num_q_heads:
    kv_h = qh * num_kv_heads / num_q_heads
    for q in 0..seq_q:
      # scores[k] = Q[b,qh,q,:] · K[b,kv_h,k,:] * scale
      # apply causal mask, then softmax along k
      # out[b,qh,q,:] = sum_k scores[k] * V[b,kv_h,k,:]
```

No fusion, no sparsity, no specialised math. Backend executors over
the same `KernelCall` are free to emit FlashAttention or any other
optimisation; canonical is the correctness reference.

### Backward

Deferred. Backward attention is a multi-output op (dQ, dK, dV) and
sits at the same multi-output canonical-layer question that ADR-048
defers (`TopK`, `NonZero`). When canonical gets multi-output
support, `AttentionBackward` is one of the first additions.

## Consequences

- Canonical `Attention` is a clean ~5-attribute op, not the
  ~10-attribute legacy bundle.
- Models that use legacy `FloatOp::Attention` with `rope=true` or
  `qk_norm=true` continue to work unchanged: the canonical→legacy
  bridge maps the simple canonical attention to the legacy variant
  with `rope=false, qk_norm=false`. Models built fresh against
  canonical produce explicit `RotaryEmbedding` + `RmsNorm` +
  `Attention` chains that the planner can inline if it wants.
- The legacy → canonical bridge declines to promote when the
  legacy variant has any of the fused flags set
  (`rope`/`qk_norm`/`sparse_v=false` is already non-canonical
  too). Such legacy ops stay on `FloatOp::Attention` until the
  planner explicitly decomposes them into canonical primitives —
  separate work, ADR-worthy on its own.
- Canonical `Attention` completes Phase 3.3 Stage 2 of Sprint 37.
  Canonical surface end state is **75 ops** as projected by ADR-048.

## Alternatives considered

- **Mirror legacy `FloatOp::Attention` 1:1.** Rejected — recreates
  the fused-execution-as-semantic problem ADR-044/045 was supposed
  to fix.
- **Canonical attention without GQA (multi-head only).** Rejected —
  GQA/MQA models (LLaMA-3, Qwen, Mistral) are too common to require
  workarounds. The marginal complexity of `num_kv_heads` is
  trivial.
- **Mask as a 4th input.** Rejected — pulls explicit mask tensors
  into every attention node even when the mask is just a causal
  `bool`. Composing `Add(mask)` before the softmax expresses
  arbitrary masks without bloating the common path.
- **Defer the canonical attention until after backward design.**
  Rejected — forward-only attention is the dominant use case (LLM
  inference), and the backward question is independently bounded
  (it's the same multi-output question for `TopK`/`NonZero`).
