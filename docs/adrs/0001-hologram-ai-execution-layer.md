# ADR-0001: hologram-ai maps its AI IR to hologram-graph + hologram-exec

- Status: Accepted
- Date: 2026-03-06
- Context: Consumer integration (hologram-ai repository)

---

## Context

`hologram-ai` is a compiler and runtime integration layer that imports AI model
artifacts (ONNX, GGUF, GGML) and runs them on the Hologram execution engine.
This ADR documents how hologram-ai uses hologram's public API — recorded here
so hologram maintainers understand the consumer integration contract.

---

## How hologram-ai uses hologram

### Graph construction

hologram-ai-lower (part of `hologram-ai-common`) translates its internal `AiGraph`
IR into a `hologram::Graph`. Each `AiOp` maps to a `GraphOp`:

| AI operation | `GraphOp` |
|---|---|
| MatMul (Q4_0 weights) | `MatMulLut4(ConstantId)` |
| MatMul (Q8_0 weights) | `MatMulLut8(ConstantId)` |
| Activations (Gelu, Relu, Silu, Tanh, Sigmoid) | `Lut(LutOp::…)` |
| Binary ops (Add, Mul, Sub, …) | `Prim(PrimOp::…)` |
| Weight constants | `GraphOp::Constant(ConstantId)` |
| Attention, Norm, RoPE, Embed, Dequantize | `Custom { id, arity }` via `CustomOpRegistry` |

### Execution

```rust
// All sessions from one compiled model share these (read-only after compilation):
let executor = Arc::new(KvExecutor::new());
let registry = Arc::new(CustomOpRegistry::new()); // registered in hologram-ai-lower
// ...register attention, norm, rope, embed, dequant handlers...

// Per-session call (KvExecutor::execute_with_registry is stateless):
let outputs = executor.execute_with_registry(
    &graph,
    &schedule,
    &inputs,
    &registry,
)?;
```

### Weight storage

- Small tensors: `ConstantData::Bytes(weights_bytes)` stored inline in `ConstantStore`
- Large GGUF tensors: `ConstantData::Deferred { … }` + `HoloLoader` for mmap loading

### Archive format

Compiled models can be serialized as `.holo` archives via `HoloWriter` and loaded
back with `HoloLoader`. The `LoadedPlan` keeps the mmap alive for the session lifetime.

---

## What hologram does NOT need to know

hologram has no knowledge of:
- AI model formats (ONNX, GGUF, GGML)
- KV-cache semantics or attention patterns
- Token generation or sampling strategies
- LLM architectures (LLaMA, Mistral, Phi, Qwen, etc.)

All AI-specific logic stays in hologram-ai's `CustomOpRegistry` handlers.

---

## Types consumed from hologram (flat imports via root crate)

```rust
use hologram::{
    // hologram-graph
    Graph, GraphBuilder, GraphOp, NodeId, CustomOpId,
    ConstantData, ConstantId, ConstantStore, ExecutionSchedule,
    // hologram-exec
    KvExecutor, CustomOpRegistry, CustomHandler, BufferArena,
    GraphInputs, GraphOutputs, execute_bytes_with_ops,
    // hologram-archive
    HoloWriter, LoadedPlan, HoloLoader, load_from_bytes,
    // hologram-core
    LutOp, PrimOp,
};
```

All types above are accessible via the root `hologram` crate without importing subcrates.
