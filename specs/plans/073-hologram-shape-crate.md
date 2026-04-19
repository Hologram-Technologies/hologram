# Plan 073: hologram-shape — Runtime Shape Tracking Crate

**Status:** Active
**Created:** 2026-04-19
**Depends on:** Plan 067 (ComputeBackend — provides TensorBuffer with shape field)

## Problem

Variable-length execution is the #1 source of correctness bugs. The executor
compiles tape instructions at a fixed seq_len, then at runtime the actual
seq_len differs. Every op that touches a variable dimension must resolve the
actual size — currently done via heuristic inference from buffer byte lengths.

**Symptoms:**
- Long prompts (>8 tokens) produce garbage output during prefill
- Attention V/K shape mismatches at non-compiled seq lengths
- Transpose shape inference fails when variable dims change
- MatMul dim resolution guesses wrong when batch/seq/hidden are ambiguous
- Slice dispatch uses `n_elems % compiled_axis_size == 0` which false-positives

**Root cause:** Shape information is scattered across three fragile mechanisms:
1. **Compiled TapeKernel params** — baked dims from compile time (e.g., `InlineMatMul { m: 0, k: 2048, n: 4096 }` where m=0 means "variable")
2. **Runtime shape_overrides** — `HashMap<u32, Vec<usize>>` threaded through TapeContext, populated by ShapeContextGraph
3. **Heuristic inference** — `resolve_last_dim()`, `resolve_matmul_dims()` that guess dimensions from buffer byte lengths

When any heuristic guesses wrong, the error propagates silently through
the rest of the computation, producing numerically valid but semantically
incorrect results (garbage logits, wrong attention patterns).

## Solution: hologram-shape Crate

A dedicated crate that tracks tensor shapes alongside data buffers throughout
execution. Every buffer gets an explicit shape — no guessing from byte lengths.

### Crate Structure

```
hologram-shape/
├── Cargo.toml          — minimal deps (smallvec only)
├── src/
│   ├── lib.rs          — public API
│   ├── tensor_shape.rs — TensorShape type
│   ├── registry.rs     — ShapeRegistry (buffer index → shape)
│   ├── infer.rs        — shape inference rules per FloatOp
│   └── validate.rs     — shape validation helpers
```

### Core Types

```rust
/// Tensor shape: concrete dimensions + element dtype.
/// Uses SmallVec to avoid heap allocation for ≤4 dims (covers 99% of tensors).
pub struct TensorShape {
    pub dims: SmallVec<[usize; 4]>,
    pub dtype: DType,
}

/// Maps buffer slot index → TensorShape.
/// Parallel to the executor's buffer array. Every slot that has data
/// also has a shape — no exceptions.
pub struct ShapeRegistry {
    shapes: Vec<Option<TensorShape>>,
}

/// Shape inference: given an op + input shapes, compute the output shape.
/// Returns Err if input shapes are incompatible with the op.
pub fn infer_output_shape(
    op: &FloatOp,
    input_shapes: &[&TensorShape],
) -> Result<TensorShape, ShapeError>;

/// Validate that a buffer's byte length matches its registered shape.
pub fn validate_buffer_shape(
    data: &[u8],
    shape: &TensorShape,
) -> Result<(), ShapeError>;
```

### Integration Points

1. **BufferArena** (`hologram-exec/src/buffer/arena.rs`):
   - Add `shape_registry: ShapeRegistry` field
   - `insert()` and `swap_insert()` take a `TensorShape` parameter
   - `get_shape(id)` returns the shape for any buffer

2. **execute_direct** (`hologram-exec/src/tape.rs`):
   - After each `dispatch_kernel` call, compute output shape via `infer_output_shape`
   - Store shape in registry via `arena.set_shape(output_id, shape)`
   - Pass input shapes to dispatch_kernel (replacing `input_metas`)

3. **dispatch_kernel** (`hologram-exec/src/tape.rs`):
   - Replace `input_metas: &InputMetas` with `input_shapes: &[&TensorShape]`
   - Replace ALL `resolve_last_dim` / `resolve_matmul_dims` calls with direct shape reads
   - MatMul: read M from input_shapes[0].dims[0], not from byte_len / (k * 4)
   - Transpose: read shape from input_shapes[0], not from baked input_shape
   - Slice: read axis_size from input_shapes[0], not from compiled axis_size

4. **Graph inputs** (`hologram-exec/src/mmap/mod.rs`):
   - When seeding the arena with graph inputs, also seed shapes
   - Variable-length inputs get their actual shape (e.g., [1, 13, 2048] for 13 tokens)

5. **Constants/weights**:
   - Shape comes from the archive's weight metadata
   - Registered at load time, immutable during execution

## Phases

### Phase 1: Core Types + Shape Inference (~200 lines)

Create `hologram-shape` crate with:
- `TensorShape` struct
- `ShapeRegistry` struct
- `infer_output_shape` for all elementwise, matmul, softmax, norm, attention ops
- `validate_buffer_shape` helper
- Unit tests for shape inference rules

**Acceptance:** `cargo test -p hologram-shape` passes with ≥30 tests covering
all op categories.

### Phase 2: Wire into BufferArena (~100 lines)

- Add `ShapeRegistry` to `BufferArena`
- Seed shapes from graph inputs and constants at execution start
- Add `get_shape()` / `set_shape()` methods

**Acceptance:** Existing hologram-exec tests still pass.

### Phase 3: Wire into execute_direct (~200 lines)

- After each dispatch_kernel, call `infer_output_shape` and store result
- Pass input shapes to dispatch_kernel
- Add debug assertion: `validate_buffer_shape(output, inferred_shape)`

**Acceptance:** TinyLlama runs with shape validation enabled, no assertion failures
at seq=4.

### Phase 4: Replace Heuristic Resolution (~300 lines, net negative)

- Replace `resolve_last_dim` calls with `input_shapes[i].dims.last()`
- Replace `resolve_matmul_dims` with `(input_shapes[0].dims[0], op.k, op.n)`
- Replace shape_overrides HashMap with direct shape reads
- Delete `shape_resolve.rs` module (359 lines)
- Delete `InputMetas` type and all threading code

**Acceptance:** TinyLlama runs correctly at ALL prompt lengths (4, 13, 36, 77 tokens).
The prefill garbage bug is fixed.

### Phase 5: Propagate to hologram-backend (~100 lines)

- `TensorBuffer<B>` in hologram-backend already has a `shape` field
- Wire `infer_output_shape` into `execute_on_backend` after each dispatch
- CpuBackend and MetalBackend get shape-validated execution

**Acceptance:** `execute_on_backend` produces correct results with shape tracking.

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Variable-length bugs | Recurring (3+ open) | Eliminated by design |
| shape_resolve.rs | 359 lines of heuristics | Deleted |
| InputMetas threading | ~50 lines per caller | Gone |
| shape_overrides HashMap | Per-call clone overhead | Direct O(1) lookup |
| Debug time for shape bugs | Hours (silent corruption) | Immediate (assertion at source) |
| New crate | — | hologram-shape (~500 lines) |

## Non-Goals

- Symbolic shapes (DimExpr) — those stay in hologram-ai-common for compile time
- Shape optimization passes — those stay in the compiler
- Dynamic batch size — batch=1 assumption remains for now
