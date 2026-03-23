# hologram-ai Migration Prompt: Sprint 17 Changes

> Use this prompt in a hologram-ai session to adapt to hologram's Sprint 17 changes.

---

hologram has undergone significant changes in Sprint 17 (Plans 014 + 015). The
deprecated KvExecutor dispatch path has been removed, custom ops now work through
the tape path, and several performance optimizations were made. hologram-ai needs
to adapt to these changes.

## 1. Removed Types and Functions

The following items NO LONGER EXIST in the hologram public API:

**Removed struct:**
- `hologram::KvExecutor`

**Removed functions:**
- `hologram::execute_plan()`
- `hologram::execute_plan_with_shape_hints()`
- `hologram::execute_plan_with_kv_state()`
- `hologram::execute_bytes()`
- `hologram::execute_bytes_with_ops()`
- `hologram::execute_file()`
- `hologram::execute_plan_with_intermediates()` (profile feature)
- `hologram::execute_plan_with_intermediates_and_shape_hints()` (profile feature)
- `hologram::execute_plan_zero_copy()`
- `hologram::execute_bytes_with_progress()`

**Removed internal types:**
- `hologram_exec::eval::shape_propagate` module (deleted)
- `hologram_exec::eval::shape_resolve` module (deleted)
- `hologram_exec::dirty_bits` module (deleted)
- `hologram_exec::profile` module (deleted)
- `IntermediateCapture` struct (deleted)

## 2. Replacement API

The canonical execution path is now tape-only:

```rust
use hologram::{build_tape_from_plan, execute_tape, execute_tape_with_kv, GraphInputs, LoadedPlan};

// Standard execution:
let plan: LoadedPlan = hologram::load_from_bytes(&archive_bytes)?;
let tape = hologram::build_tape_from_plan(&plan)?;
let outputs = hologram::execute_tape(&tape, &plan, &inputs)?;

// With KV cache (autoregressive generation):
let tape = hologram::build_tape_from_plan(&plan)?;
let outputs = hologram::execute_tape_with_kv(&tape, &plan, &inputs, &mut kv_state)?;

// With custom ops:
use hologram::{build_tape_from_plan_with_ops, CustomOpRegistry, CustomOpId};
let mut registry = CustomOpRegistry::new();
registry.register(CustomOpId(1), 3, Arc::new(|inputs, constants| {
    // SDPA implementation
    Ok(result_bytes)
}));
let tape = hologram::build_tape_from_plan_with_ops(&plan, &registry)?;
let outputs = hologram::execute_tape(&tape, &plan, &inputs)?;
```

**Key principle:** Build the tape ONCE at model load time, reuse for all inference
calls. The tape is 17-140x faster than KvExecutor.

## 3. Custom Ops via Tape Path (NEW)

Custom ops (`GraphOp::Custom { id, arity }`) are now supported in the tape path
via `build_tape_from_plan_with_ops`. The handler closures are baked into the tape
as `TapeKernel::Custom` variants at build time — zero dispatch overhead at
inference time (one vtable call vs zero for inline ops, ~1ns difference).

hologram-ai's custom ops (SDPA, SwiGLU, RoPE) should be registered via
`CustomOpRegistry` and passed to `build_tape_from_plan_with_ops`.

## 4. Migration Checklist

Please search the hologram-ai codebase and fix:

1. **Imports:** Find all `use hologram::KvExecutor`, `use hologram::execute_plan`,
   `use hologram::execute_bytes`, etc. Replace with tape API imports.

2. **Execution calls:** Replace `execute_plan(&plan, &inputs)` with
   `build_tape_from_plan(&plan)` + `execute_tape(&tape, &plan, &inputs)`.

3. **KV cache calls:** Replace `KvExecutor::execute_with_plan(graph, &schedule, &inputs, weights)`
   with `execute_tape_with_kv(&tape, &plan, &inputs, &mut kv_state)`.

4. **Custom ops:** Replace `execute_bytes_with_ops(&data, &inputs, &registry)` with
   `build_tape_from_plan_with_ops(&plan, &registry)` + `execute_tape(...)`.

5. **Shape hints:** The old `execute_plan_with_shape_hints` is removed. Shape
   resolution now happens at tape build time. If hologram-ai was passing shape
   hints, this logic needs to move to before tape construction.

6. **Intermediate capture:** `execute_plan_with_intermediates` is removed. If
   hologram-ai's conformance tests used this for node-by-node comparison against
   ORT, they need a new approach (e.g., adding probe nodes to the graph).

7. **Conformance tests:** Ensure all conformance tests (TinyLlama ONNX, GGUF,
   ResNet-50) pass with the tape API. Priority: these are the source of truth.

8. **build_schedule:** This function still exists (`hologram::build_schedule`)
   for the tape builder. It's NOT removed.

## 5. Performance Context

Sprint 17 achieved:
- fusion::fuse(1000_nodes): **1.91 ms → 290 µs** (6.6x faster compilation)
- compile/100_nodes: **79 µs → 60 µs** (-24%)
- Tape execution: **17-140x faster** than KvExecutor (zero dispatch overhead)
- Binary broadcasting: eliminated per-element modulo for same-size operands
- ~3,500 lines of dead code removed

hologram-ai should ensure it's using `build_tape_from_plan` (build once) and
`execute_tape` (run many) rather than any pattern that rebuilds per inference.
