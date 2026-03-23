# hologram → hologram-ai Handoff Spec

> **Date**: 2026-03-22
> **Status**: Current — supersedes ADR-0001 for planning purposes
> **Audience**: hologram-ai contributors starting integration work

---

## 1. What hologram Provides Today

hologram is an O(1) compute acceleration runtime. After Phases 8–9, it offers:

- **EnumTape executor** — zero-overhead dispatch, zero per-instruction allocation, pre-resolved kernel routing
- **Multi-backend GPU** — Metal (Apple Silicon, zero-copy), WebGPU/wgpu (cross-platform, deferred batching), CUDA (placeholder)
- **LUT-GEMM** — quantized Q4/Q8 matrix multiply via lookup tables
- **Archive format** — `.holo` files with mmap loading, weight sectioning, parallel schedules
- **KV cache** — prefill + decode lifecycle for autoregressive generation
- **Custom op registry** — consumer-defined ops dispatched alongside built-in kernels

> **Execution path**: Always use the EnumTape executor (`build_tape_from_plan` → `execute_tape`). The older KvExecutor dispatch path is retained for backward compatibility but is 17–140× slower and should not be used for new integration work.

### Performance Baseline (2026-03-22, Apple M-series)

| Benchmark | Value |
|-----------|-------|
| EnumTape Relu 64KB | 2.54 µs |
| KvExecutor Relu 64KB | 44.4 µs |
| Tape vs KvExecutor | **17.5x** |
| TinyLlama decode (hidden=2048) | 2.8 ms (tape) vs 391 ms (KvExecutor) |
| Dispatch overhead per op | ~0 ns |
| Per-inference allocation | 0 bytes (swap-insert arena) |
| Tape linear chain (4 nodes, 256B) | 1.11 µs |
| LUT-GEMM Q4 64×64 | 25.5 µs |
| LUT-GEMM Q8 64×64 | 28.5 µs |

*KvExecutor numbers shown as historical baseline. KvExecutor is the legacy dispatch path from Phases 6–7; it is not recommended for hologram-ai integration.*

---

## 2. Public API Surface

All types below are accessible via `use hologram::*;` (root crate re-exports).

### Graph Construction

```rust
// Build a compute graph.
let mut builder = GraphBuilder::new();
let input = builder.input("tokens");
let embed = builder.node_with_inputs(GraphOp::Float(FloatOp::Embed { dim: 2048, quant: 0 }), &[input]);
let relu = builder.node_with_inputs(GraphOp::Float(FloatOp::Relu), &[embed]);
builder.output("logits", relu);
let graph = builder.build();

// Constants (weights).
let cid = graph.constants_mut().insert(ConstantData::Bytes(weight_bytes));
// Or deferred for large weights:
let cid = graph.constants_mut().insert(ConstantData::Deferred { byte_len, loader_id });
```

**Key types**: `Graph`, `GraphBuilder`, `GraphOp`, `FloatOp`, `LutOp`, `PrimOp`, `NodeId`, `ConstantId`, `ConstantData`, `ConstantStore`, `CustomOpId`

### Compilation

```rust
let output = hologram::compile(&graph)?;
// output.plan — the execution plan
// output.stats — op counts, fusion results
```

**Key types**: `compile()`, `CompilerBuilder`, `CompilationOutput`, `CompilationStats`

### Archive I/O

```rust
// Write .holo archive.
let mut writer = HoloWriter::new();
writer.set_graph(&graph);
writer.set_weights(&weight_bytes);
let archive_bytes = writer.build();

// Load .holo archive (mmap'd).
let loader = HoloLoader::open(&path)?;
let plan = loader.load()?;  // LoadedPlan keeps mmap alive
```

**Key types**: `HoloWriter`, `HoloLoader`, `LoadedPlan`, `HoloHeader`, `LayerHeader`

### Tape Build + Execution

> This is the canonical execution path for all hologram consumers, including hologram-ai.

```rust
// Build tape once at model load time.
let tape = build_tape_from_plan(&plan)?;

// Execute per inference call (reuse tape).
let mut inputs = GraphInputs::new();
inputs.set(0, token_bytes);
let outputs = execute_tape(&tape, &plan, &inputs)?;
let logits = outputs.by_name("logits")?;
```

**Key types**: `EnumTape`, `build_tape_from_plan()`, `execute_tape()`, `TapeContext`, `GraphInputs`, `GraphOutputs`, `BufferArena`

### Backend Selection

```rust
use hologram::backend::{BackendSelector, available_backends};

// Auto (default): Metal → WebGPU → CUDA → CPU
let tape_ctx = TapeContext::new(&constants, weights);

// Force specific backend:
let mut tape_ctx = TapeContext::new(&constants, weights);
tape_ctx.backend = BackendSelector::WebGpu;

// Check what's available:
let backends = available_backends();  // e.g. ["cpu", "metal"]
```

**Key types**: `BackendSelector` (`Auto`, `Cpu`, `Metal`, `Cuda`, `WebGpu`), `ComputeBackend` trait

### KV Cache (Autoregressive Generation)

```rust
let kv = KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq_len);
let tape_ctx = TapeContext::with_kv_cache(&constants, weights, kv);

// Prefill (full prompt).
tape.execute(&mut arena, &tape_ctx)?;

// Decode loop (one token at a time).
for _ in 0..max_tokens {
    tape.execute(&mut arena, &tape_ctx)?;
    // Extract logits → sample next token → feed back as input
}
```

**Key types**: `KvCacheState`, `TapeContext::with_kv_cache()`

> **Why KV Cache?** Without KV caching, each autoregressive decode step recomputes key/value projections for *all* previous tokens — O(n²) total work across a full generation. `KvCacheState` stores K/V tensors from prior steps so each new token only computes its own projections, reducing generation to O(n). This is a data structure, not an executor — it composes with the EnumTape path via `TapeContext::with_kv_cache()`, and the tape executes `FloatOp::KvWrite` / `FloatOp::KvRead` ops to interact with it. Do not confuse it with the legacy `KvExecutor` dispatch mechanism.

### Custom Ops

```rust
let mut registry = CustomOpRegistry::new();
registry.register(CustomOpId(1), CustomHandler::new(|inputs, out_buf| {
    // Multi-head attention implementation
    Ok(())
}));

// Wire custom ops into graph:
let attn = builder.custom_op(CustomOpId(1), 3, &[q, k, v]);
```

**Key types**: `CustomOpRegistry`, `CustomHandler`, `CustomOpId`

---

## 3. Backend Capabilities

| Capability | CPU | Metal | WebGPU | CUDA |
|-----------|-----|-------|--------|------|
| Elementwise (relu, add, …) | all sizes | >4 MB | >4 MB | planned |
| MatMul (SGEMM) | all sizes | >128×128 output | >128×128 output | planned |
| Softmax / RmsNorm | all sizes | >4 MB | >4 MB | planned |
| LUT-GEMM Q4/Q8 | all sizes | — | — | — |
| Zero-copy output | — | `MetalBuffer` (unified mem) | — | — |
| Command batching | — | `flush()` per level | `flush_deferred()` per level | — |
| Deferred readback | — | — | `WgpuDeferred` + batch map | — |
| Feature gate | always | `has_metal` (macOS auto) | `--features webgpu` | `--features cuda` |
| Platforms | all | macOS (Apple Silicon) | Linux, Windows, macOS, browser | Linux, Windows |

**Auto selection priority**: Metal → WebGPU → CUDA → CPU

---

## 4. hologram-ai Current State + Remaining Work

hologram-ai is a **mature compiler** (~33K LOC, 96 source files) with production-ready import, optimization, and lowering pipelines. This section documents what's done and what remains.

### Already Complete

| Component | Crate | Status |
|-----------|-------|--------|
| ONNX import (protobuf → AiGraph) | `hologram-ai-onnx` | Done — all 1156 TinyLlama nodes match ORT |
| GGUF import (v2/v3 binary parsing) | `hologram-ai-gguf` | Done — LLaMA family, logit consistency verified |
| AI IR (`AiGraph`, 50+ `AiOp` variants) | `hologram-ai-common/ir` | Done |
| 24 optimization passes (attention fusion, SwiGLU, etc.) | `hologram-ai-common/opt` | Done |
| AiOp → FloatOp lowering (140K LOC) | `hologram-ai-common/lower` | Done |
| Quantization primitives (Q4_0, Q8_0) | `hologram-ai-quant` | Done |
| BPE tokenizer + HuggingFace json support | `hologram-ai-tokenizer` | Done |
| Compilation pipeline (`ModelCompiler` → `.holo`) | `hologram-ai/compiler` | Done |
| KV cache contiguous mode | `hologram-ai-common/mem` | Done |
| Multi-component archives (LLM + Whisper + CALM) | `hologram-ai-common/sections` | Done |
| Conformance test framework | `hologram-ai-conformance` | Done |
| E2E tests: TinyLlama (ONNX + GGUF), ResNet-50 | `hologram-ai/tests` | Done |

### Remaining Work (hologram-ai side)

| Item | Priority | Notes |
|------|----------|-------|
| **Paged KV cache** | P1 | Plan 016 exists. On-demand page allocation with block tables. Contiguous mode works but wastes memory for long contexts. |
| **Fused kernel support** | P1 | MatMul+Activation, Concat+MatMul passes exist but await fused kernel implementations in hologram base (`FloatOp::FusedMatMulRelu` etc.) |
| **Variable-length prefill** | P2 | Runtime shape projection (blocker resolved, awaiting integration with shape_spec_bridge) |
| **More model coverage** | P2 | BERT, Stable Diffusion, Whisper (ONNX paths exist, need E2E validation) |
| **Performance optimization** | P2 | Plan 017 active — prefill/decode latency, compilation speed |
| **GGML importer** | P3 | Stub only — deferred since GGUF covers the important models |
| **safetensors support** | P3 | Not yet started — needed for HuggingFace ecosystem models |

### What hologram Must Provide (for remaining hologram-ai work)

| hologram item | Needed by | Status |
|--------------|-----------|--------|
| Fused kernel variants (`FusedMatMulRelu`, etc.) | Fused kernel support | Not started — new `FloatOp` variants + dispatch |
| CUDA backend (Phase 8.4) | Server-side GPU inference | Blocked on hardware |
| Weight-page prefetch (Sprint 15, item 2.2) | Large model decode perf | Deferred |

---

## 5. Contract Boundaries

| Responsibility | Owner |
|---------------|-------|
| Tensor arithmetic (f32, quantized) | hologram |
| GPU kernel dispatch + memory management | hologram |
| Compute graph representation + scheduling | hologram |
| Archive format (.holo) + mmap loading | hologram |
| KV cache data structure + read/write ops | hologram |
| Arena-based buffer management | hologram |
| | |
| Model file parsing (ONNX, GGUF, safetensors) | hologram-ai |
| AI IR → Graph lowering | hologram-ai |
| Optimization passes (fusion, const eval, shape prop) | hologram-ai |
| Tokenization (BPE, sentencepiece) | hologram-ai |
| Sampling strategies (top-k/p, temperature) | hologram-ai |
| Generation loop + stopping criteria | hologram-ai |
| Model architecture knowledge (LLaMA, Mistral, etc.) | hologram-ai |
| Weight quantization decisions | hologram-ai |

**Principle**: hologram has zero knowledge of AI model architectures. All AI-specific logic lives in hologram-ai's optimization passes and lowering layer.

---

## 6. Op Mapping Reference (ONNX → AiOp → FloatOp)

For hologram contributors who need to understand how ops flow through:

| ONNX op | hologram-ai `AiOp` | hologram `FloatOp` / `GraphOp` |
|---------|--------------------|---------------------------------|
| MatMul | `AiOp::MatMul` | `FloatOp::MatMul { m, k, n }` |
| Relu, Gelu, Silu, Sigmoid, Tanh | `AiOp::Activation(*)` | `FloatOp::Relu`, etc. |
| Add, Mul, Sub, Div | `AiOp::BinaryOp(*)` | `FloatOp::Add`, etc. |
| Softmax | `AiOp::Softmax` | `FloatOp::Softmax { size }` |
| LayerNorm / RmsNorm | `AiOp::RMSNorm` | `FloatOp::RmsNorm { size, epsilon }` |
| Gather | `AiOp::Gather` | `FloatOp::Gather { dim, dtype }` |
| Concat | `AiOp::Concat` | `FloatOp::Concat { size_a, size_b, dtype }` |
| Reshape / Transpose | `AiOp::Reshape/Transpose` | `FloatOp::Reshape` / `FloatOp::Transpose { perm, ndim }` |
| Cast | `AiOp::Cast` | `FloatOp::Cast { from, to }` |
| SDPA (fused attention) | `AiOp::GroupedQueryAttention` | `Custom { id, arity }` via `CustomOpRegistry` |
| SwiGLU (fused) | `AiOp::FusedSwiGLU` | `Custom { id, arity }` |
| RoPE | `AiOp::RoPE` | `Custom { id, arity }` |
| Embedding | `AiOp::Embedding` | `FloatOp::Embed { dim, quant }` |
| KV cache write/read | `AiOp::KvSlotWrite/Read` | `FloatOp::KvWrite/KvRead` |

### 7. Next Milestone: Paged KV Cache

The next major integration milestone between hologram and hologram-ai.

**Why**: Contiguous KV cache pre-allocates `max_seq_len × n_layers × n_kv_heads × head_dim` per sequence. For long contexts (8K+ tokens) this wastes significant memory. Paged attention allocates blocks on demand.

**hologram side** (new types needed):
- `KvPageTable` — block table mapping logical positions to physical pages
- `FloatOp::KvPagedWrite` / `FloatOp::KvPagedRead` — paged variants of KV ops
- Page allocator in `BufferArena` or separate pool

**hologram-ai side** (Plan 016 exists):
- `KvSlotInjection` pass updated for paged ops
- `MemoryPlanner` paged mode (block size, max pages per sequence)
- Runtime page allocation + eviction policy

---

## 7. FloatOp Reference (Complete)

For hologram-ai's op mapping, here is the full `FloatOp` enum (from `hologram-core`):

**Arithmetic**: Add, Sub, Mul, Div, Pow, Mod, Min, Max
**Unary**: Relu, Gelu, Silu, Tanh, Sigmoid, Exp, Log, Sqrt, Abs, Reciprocal, Neg
**Trigonometric**: Cos, Sin, Sign, Floor, Ceil, Round, Erf
**Comparison**: Equal, Less, Greater, LessOrEqual, GreaterOrEqual
**Boolean**: And, Or, Xor, Not
**Linear algebra**: MatMul { m, k, n }, Gemm, MatMulLut4, MatMulLut8
**Normalization**: RmsNorm { size, epsilon }, AddRmsNorm, LayerNorm, Softmax { size }, LogSoftmax
**Reduction**: ReduceSum { size }, ReduceMean { size }, ReduceMax { size }, ReduceMin { size }
**Shape**: Gather { dim, dtype }, Concat { size_a, size_b, dtype }, Reshape, Transpose { perm, ndim }
**Type**: Cast { from, to }, Embed { dim, quant }, Clip { min, max }, IsNaN
**Other**: Where, Range, Shape { dtype, start, end }

**FloatDType**: F32, F64, F16, BF16, I32, I64, I8, U8, Bool

---

## 8. KvExecutor Deprecation Roadmap

`KvExecutor` is deprecated as of Phase 9. It is the legacy level-by-level dispatch path from Phases 4–7, retained only for backward compatibility. All new integration work **must** use the EnumTape path (`build_tape_from_plan` → `execute_tape`).

**Removal plan**:
1. **CLI migration** — `run_cmd.rs` generation loop: build tape once, reuse per token (high priority — inner loop performance)
2. **Tape profiling** — add intermediate capture to EnumTape (needed before `execute_plan_with_intermediates` can be deprecated)
3. **Test migration** — replace remaining KvExecutor-based conformance tests with tape equivalents
4. **Remove KvExecutor** — delete struct, impl blocks, mmap wrappers, re-exports
