# Plan: Phase 9 — Zero-Overhead Dispatch

## Context

The EnumTape executor is already 10.6x faster than KvExecutor (4.2 µs vs 44.8 µs on Relu 64KB) after Phase 14's monomorphized SIMD work. Phase 9 eliminates the remaining per-instruction overhead (~205ns/op) in the tape dispatch path. At 150 ops per transformer layer, this reclaims ~30µs — significant for real-time inference.

Phase 9a (inline hot ops) is partially done: InlineRelu, InlineNeg, InlineSigmoid, InlineSilu, InlineTanh, InlineGelu, InlineExp, InlineAdd, InlineMul, InlineSub, InlineDiv already exist. The remaining work adds InlineMatMul, InlineSoftmax, InlineRmsNorm and the zero-copy arena optimizations (9b).

**Priority**: 9a (remaining) + 9b first. Then 9c, 9d, 9e as follow-up.

## Files to Modify

- `crates/hologram-exec/src/tape.rs` — TapeKernel enum, dispatch_kernel, dispatch_kernel_par, execute loop
- `crates/hologram-exec/src/tape_builder.rs` — resolve_float_kernel, consumer count analysis
- `crates/hologram-exec/src/buffer/arena.rs` — move_slot, prewarm, take_owned
- `crates/hologram-exec/src/float_dispatch/mod.rs` — visibility: `mod norm` → `pub(crate) mod norm`, `resolve_size` → `pub(crate)`
- `crates/hologram-exec/src/float_dispatch/norm.rs` — visibility: `pub(super)` → `pub(crate)` for dispatch_softmax_into, dispatch_rms_norm_into
- `crates/hologram-bench/benches/executor.rs` — benchmarks for new inline ops
- `specs/SPRINT.md` — tick completed items, keep 9c-9e as TODO

---

## Step 1: 9a.3 — InlineMatMul { m, k, n }

Eliminates `dispatch_float_into` → `dispatch_custom_into` → `matmul::dispatch_matmul_into` indirection. Preserves Metal GPU path for large matrices.

### tape.rs
Add variant to `TapeKernel`:
```rust
InlineMatMul { m: u32, k: u32, n: u32 },
```

Add match arm in `dispatch_kernel` (after KvRead, before existing inline ops):
- Try `backend.dispatch_float(&FloatOp::MatMul{m,k,n}, inputs, out_buf)` first
- On `Skipped`, call `crate::float_dispatch::matmul::dispatch_matmul_into(inputs, m, k, n, out_buf)` directly
- `matmul` module is already `pub mod` — no visibility changes needed

Add match arm in `dispatch_kernel_par` — CPU-only path (no backend), call `dispatch_matmul_into` directly. InlineMatMul is safe for parallel levels (no shared state).

### tape_builder.rs
In `resolve_float_kernel`, add before the catch-all:
```rust
FloatOp::MatMul { m, k, n } => TapeKernel::InlineMatMul { m: *m, k: *k, n: *n },
```

---

## Step 2: 9a.4 — InlineSoftmax { size } + InlineRmsNorm { size, epsilon }

Same pattern as InlineMatMul. Requires visibility fix for `norm` module.

### float_dispatch/mod.rs
- Change `mod norm;` → `pub(crate) mod norm;`
- Change `fn resolve_size` → `pub(crate) fn resolve_size`

### float_dispatch/norm.rs
- Change `pub(super) fn dispatch_softmax_into` → `pub(crate) fn dispatch_softmax_into`
- Change `pub(super) fn dispatch_rms_norm_into` → `pub(crate) fn dispatch_rms_norm_into`

### tape.rs
Add variants:
```rust
InlineSoftmax { size: u32 },
InlineRmsNorm { size: u32, epsilon: u32 },  // epsilon as f32::to_bits()
```

Add match arms in `dispatch_kernel`:
- Try backend first (Metal handles large tensors via its own size threshold)
- On `Skipped`, resolve size via `crate::float_dispatch::resolve_size(size, inputs)`, then call `norm::dispatch_softmax_into` / `norm::dispatch_rms_norm_into` directly

Add match arms in `dispatch_kernel_par` — CPU-only path.

### tape_builder.rs
In `resolve_float_kernel`:
```rust
FloatOp::Softmax { size } => TapeKernel::InlineSoftmax { size: *size },
FloatOp::RmsNorm { size, epsilon } => TapeKernel::InlineRmsNorm { size: *size, epsilon: *epsilon },
```

---

## Step 3: 9a.5 — Tick completed inline ops + add Abs, Reciprocal

The existing InlineRelu..InlineDiv variants were implemented but SPRINT.md 9a.1/9a.2 are still unchecked. Also add InlineAbs and InlineReciprocal (already monomorphized in Phase 14 but not yet inlined).

### tape.rs
Add variants `InlineAbs` and `InlineReciprocal`. Add match arms in `dispatch_kernel` and `dispatch_kernel_par`.

### tape_builder.rs
```rust
FloatOp::Abs => TapeKernel::InlineAbs,
FloatOp::Reciprocal => TapeKernel::InlineReciprocal,
```

---

## Step 4: 9b.1 — Output Passthrough (arena pointer move)

For `TapeKernel::Output` instructions where the input has exactly one consumer, move the arena buffer directly instead of copying through `out_buf`.

### tape.rs — TapeInstruction
Add field: `pub passthrough: bool` (default false)

### tape_builder.rs
After building all instructions, compute consumer counts by scanning `input_indices`. For `TapeKernel::Output` instructions with a single input that has `consumer_count == 1`, set `passthrough = true`.

### arena.rs
Add method `move_slot(src, dst)` — takes buffer from src slot, puts in dst slot, updates elem_size.

### tape.rs — execute loop
Before the general dispatch path, add early exit for passthrough instructions using `arena.move_slot`.

---

## Step 5: 9b.2 — Pre-warm Arena

Pre-allocate output slots in the arena before first execute, so `swap_insert` has buffers to recycle from the very first instruction.

### tape.rs — EnumTape
Add method `prewarm_arena(&self, arena)` — iterates instructions, pre-allocates `Vec::with_capacity(output_byte_hint)` for each output slot.

Wire into mmap execute path (`crates/hologram-exec/src/mmap/mod.rs`).

---

## Step 6: 9b.3 — In-Place Unary Ops

When a unary op's input has exactly one consumer, overwrite the input buffer in place.

### tape.rs — TapeInstruction
Add field: `pub can_reuse_input: bool` (default false)

### tape_builder.rs
Using consumer count infrastructure: for unary inline ops where the single input has `consumer_count == 1`, set `can_reuse_input = true`.

### tape.rs
Add `inline_unary_inplace(buf, f)` helper and `dispatch_inplace(kernel, buf)` match function. In execute loop, take input buffer, apply in-place, store in output slot.

---

## Step 7: Update SPRINT.md

- Tick 9a.1-9a.5 and 9b.1-9b.3
- Keep 9c, 9d, 9e as `[ ]` TODO items
- Keep 8.1-8.4 as `[ ]` TODO items

---

## Verification

1. `cargo test --workspace` — all existing tests pass
2. `cargo clippy -- -D warnings` — no new warnings
3. Conformance: tape_vs_kv and tape conformance tests verify byte-identical output
4. Benchmarks: `cargo bench --bench executor` — compare tape_vs_kv numbers
5. Full bench: `cargo bench --workspace` for regression check
