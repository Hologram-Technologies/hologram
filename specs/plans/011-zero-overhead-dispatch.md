# Plan: Zero-Overhead O(1) Dispatch — Phase 9

## Context

The tape executor currently has ~205ns of overhead per instruction from 5 abstraction layers. For a 150-op transformer layer, that's ~30µs wasted on dispatch mechanics — not compute. The goal: **zero memory copies, O(1) constant-time dispatch** for the CPU hot path.

## The Zero-Copy Path

For a unary op like Relu on a buffer already in the arena:

**Current (5 layers, ~205ns overhead):**
```
arena.get(id) → &[u8]                    // bounds check
collect into SmallVec                      // push to stack vec
backend.dispatch_float()                   // vtable call → Skipped
dispatch_float_into() → category match     // match OpCategory
  → op match → Relu                        // match FloatOp
    → cast_f32(input)                      // bytemuck alignment check
    → out_buf.resize(0s)                   // zero memory
    → kernel loop                          // ACTUAL COMPUTE
swap_insert into arena                     // mem::take + replace
```

**Target (0 layers, ~0ns overhead):**
```
arena slots[input_idx] as &[f32]           // direct pointer, no check
arena slots[output_idx] as &mut [f32]      // pre-allocated, no check
kernel loop: out[i] = input[i].max(0.0)    // ACTUAL COMPUTE (only this)
```

## Implementation: 5 Sub-Phases

### 9a: Inline Hot Ops
Add `TapeKernel::InlineRelu`, `InlineAdd`, etc. that skip the backend + dispatch_float_into entirely. The tape builder maps common ops to these at build time. The `dispatch_kernel` match goes directly to the kernel function.

### 9b: Zero-Copy Arena
- Output passthrough: just alias the input slot (no copy)
- Pre-allocated output slots: tape builder allocates arena slots at build time, kernel writes directly
- In-place unary: if input has no other consumers, overwrite it (liveness analysis from compiler)

### 9c: Typed Arena
- `arena.get_f32(id) → &[f32]` — cached alignment, no per-call check
- `ArenaBuffer::F32(Vec<f32>)` — store float data natively, skip byte conversion

### 9d: Direct Input Access
- Unary ops: `arena.get(indices[0])` directly, no SmallVec
- Binary ops: two direct gets
- Arity is known from the TapeKernel variant at compile time

### 9e: Unsafe Fast Path
- `set_len` instead of `resize` (skip zeroing)
- Unchecked arena access for tape-validated indices
- `get_unchecked` for known-arity input refs

## Files to Modify

| File | Changes |
|------|---------|
| `crates/hologram-exec/src/tape.rs` | Inline variants + direct arena access in execute loop |
| `crates/hologram-exec/src/tape_builder.rs` | Map hot ops to Inline variants |
| `crates/hologram-exec/src/buffer/arena.rs` | get_f32, get_f32_mut, F32 variant |
| `crates/hologram-exec/src/float_dispatch/elementwise.rs` | Typed kernel functions |
| `specs/SPRINT.md` | Phase 9 tracking |
| `specs/plans/011-zero-overhead-dispatch.md` | This plan |

## Verification

- `cargo test --workspace` — identical results
- `cargo bench --bench executor -- tape_vs_kv` — measure per-phase improvement
- Conformance: inline path vs generic Float path, byte-for-byte match
