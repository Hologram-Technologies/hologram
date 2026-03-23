# Plan 016: Dynamic Sequence Length — Attention + Slice Fix

## Context

`dispatch_attention` panics with "slice index out of range" when runtime
sequence length differs from compiled sequence length. The goal is to make
ONNX models with dynamic symbolic shapes (e.g., variable seq_len) work
correctly at runtime **without requiring `--seq-len` at compile time**.

After this fix, every op in the standard transformer pipeline has a runtime
inference mechanism for dynamic dimensions — no fixed seq_len needed.

## Root Cause Analysis

The runtime uses a "zero-knowledge" design: ops see raw byte buffers with no
shape metadata. Sequence length is inferred from buffer sizes using baked
structural parameters (head_dim, num_heads). This works IF every op in the
pipeline produces correctly-sized output for the runtime seq_len.

**What already handles dynamic seq correctly:**
- `infer_matmul_k()` — multi-fallback inference from actual buffer sizes
- `resolve_size()` — Softmax, RmsNorm, LayerNorm infer from buffers
- Reshape/Transpose — no-ops in tape execution (data passes through)
- Attention — infers seq from buffer: `seq = buf.len() / (heads * head_dim)`

**What does NOT handle dynamic seq correctly:**
- **`FloatOp::Slice` dispatch** — uses `end` (the slice upper bound) as the
  full axis size. When `end < actual_axis_size` (e.g., slicing Q from combined
  QKV), `src_stride` is computed wrong, producing corrupted output sizes.
  This is the most likely upstream cause: any model using combined QKV
  projection + Split will produce wrong-sized Q/K/V buffers.

## Approach: Two-part fix

### Part 1: Fix Slice dispatch to handle dynamic leading dimensions

**File:** `crates/hologram-exec/src/float_dispatch/mod.rs` (lines 585-634)

The current heuristic `axis_size = end` is wrong when `end < actual_axis_size`.
Fix: infer the actual axis size at runtime by finding the smallest value >= `end`
that evenly divides `n_elems`.

New helper function `infer_slice_axis_size(n_elems, end)`:
1. If `end > 0 && n_elems % end == 0` → return `end` (existing behavior, fast path)
2. Otherwise, find smallest `s` in `(end..=n_elems)` where `n_elems % s == 0`
3. Fallback: return `n_elems` (1-D interpretation)

### Part 2: Add validation in dispatch_attention

**File:** `crates/hologram-exec/src/float_dispatch/attention.rs` (lines 25-39)

Add checks after casting Q/K/V to f32, before seq inference:
1. `q_raw.len() % (num_q_heads * head_dim) == 0` — clean Q division
2. `k_raw.len() % (num_kv_heads * head_dim) == 0` — clean K division
3. `v_raw.len() == k_raw.len()` — K/V consistency
4. `seq_q > 0 && seq_k > 0` — non-zero sequences

Return `ExecError::ShapeMismatch` with diagnostic info.

## Performance Impact

**Zero measurable overhead.** Both changes are O(1) per op invocation.

## Dynamic seq_len coverage after this fix

| Op | Dynamic seq mechanism | Status |
|----|----------------------|--------|
| MatMul | `infer_matmul_k()` runtime fallback | Already works |
| Softmax/RmsNorm/LayerNorm | `resolve_size()` from buffer | Already works |
| Reshape/Transpose | No-ops (data passthrough) | Already works |
| Slice | `infer_slice_axis_size()` | **This fix** |
| Attention | seq inferred from buffer size | Already works (+ validation added) |
| Add/Mul/etc | Element-wise, size-preserving | Already works |
| Concat | `size_a`/`size_b` on hidden dim (not seq) | Already works |
