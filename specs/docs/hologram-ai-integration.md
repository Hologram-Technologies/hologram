# hologram-ai Integration Guide

## Overview

hologram-ai should use the **tape execution path** (`build_tape_from_plan` + `execute_tape`) instead of the KvExecutor path (`execute_plan`) for all inference. The tape path is 9.3x faster for elementwise ops, has zero per-instruction allocation, and automatically dispatches to Metal GPU on Apple Silicon.

## Quick Start

```rust
use hologram::*;

// 1. Load the .holo archive (once at model load time).
let loader = HoloLoader::open(&path)?;
let plan = loader.load()?;  // madvise hints applied automatically

// 2. Build the tape (once at model load time).
//    Pre-resolves all kernel dispatch — O(1) per op at runtime.
let tape = build_tape_from_plan(&plan)?;

// 3. Execute per inference call (reuse tape across calls).
let mut inputs = GraphInputs::new();
inputs.set(0, token_embedding_bytes);
let outputs = execute_tape(&tape, &plan, &inputs)?;
let logits = outputs.by_name("logits")?;
```

## Backend Selection

By default, `execute_tape` uses `BackendSelector::Auto` which picks the best available backend:

| Priority | Backend | Auto-detected when |
|----------|---------|-------------------|
| 1 | Metal | macOS (Apple Silicon) |
| 2 | WebGPU | wasm32 target |
| 3 | CUDA | `CUDA_HOME` set or `nvcc` on PATH |
| 4 | CPU | Always available |

### Forcing a Backend

```rust
use hologram::tape::TapeContext;
use hologram::backend::BackendSelector;

let mut tape_ctx = TapeContext::new(&plan.graph().constants, plan.weights());
tape_ctx.backend = BackendSelector::Cpu;  // Force CPU even on Metal-capable machines

tape.execute(&mut arena, &tape_ctx)?;
```

### Checking Available Backends

```rust
use hologram::backend::available_backends;

let backends = available_backends();
// On macOS: ["cpu", "metal"]
// On Linux with CUDA: ["cpu", "cuda"]
// On wasm32: ["cpu", "webgpu"]
```

## Autoregressive Generation (KV Cache)

For LLM token-by-token generation:

```rust
use hologram::tape::TapeContext;
use hologram::KvCacheState;

// Create KV cache for the model architecture.
let kv = KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq_len);

// Create tape context with KV cache.
let tape_ctx = TapeContext::with_kv_cache(
    &plan.graph().constants,
    plan.weights(),
    kv,
);

// Execute prefill (full prompt).
tape.execute(&mut arena, &tape_ctx)?;

// Execute decode (one token at a time).
for _ in 0..max_tokens {
    tape.execute(&mut arena, &tape_ctx)?;
    // Extract next token from arena output...
}
```

## Performance Characteristics

### Tape vs KvExecutor

| Metric | KvExecutor | EnumTape | Speedup |
|--------|-----------|----------|---------|
| Relu 64KB | 44.6 µs | 3.3 µs | 13.5x |
| Dispatch overhead per op | ~2 µs | ~0 ns | ∞ |
| Allocation per inference | O(n) Vec | 0 (swap-insert) | ∞ |
| Matmul dispatch | match chain | inline | direct |

### Metal GPU Dispatch

The Metal backend automatically dispatches ops to the GPU when buffer sizes exceed thresholds:

| Op type | GPU threshold | Below threshold |
|---------|--------------|-----------------|
| Elementwise (relu, add, etc.) | 4 MB | CPU SIMD |
| MatMul | 128×128 output | Accelerate BLAS |
| Softmax / RmsNorm | 4 MB | CPU |

GPU output is stored in the arena as `ArenaBuffer::Metal` — zero-copy on Apple Silicon unified memory. Downstream ops reading from a Metal buffer get a CPU-accessible pointer without any DMA transfer.

## API Reference

### Building the Tape

```rust
// From a loaded plan:
let tape = build_tape_from_plan(&plan)?;

// Or from raw graph + schedule:
let schedule = build_schedule(plan.graph())?;
let tape = hologram::tape_builder::build_tape(plan.graph(), &schedule)?;
```

### Execution Functions

| Function | Use case |
|----------|----------|
| `execute_tape(&tape, &plan, &inputs)` | Standard inference (auto backend) |
| `execute_plan(&plan, &inputs)` | Legacy KvExecutor path (slower) |
| `execute_bytes(&archive, &inputs)` | One-shot from archive bytes |
| `execute_file(&path, &inputs)` | One-shot from .holo file |

### Tape Reuse

The tape is immutable after construction. It can be:
- Shared across threads (`EnumTape` is `Send + Sync`)
- Reused across inference calls (just change inputs)
- Cached for the lifetime of the model

The `TapeContext` is per-call — it holds mutable state (weight cache, KV cache). Create one per inference thread.

## Migration from KvExecutor

Replace:
```rust
// Old (KvExecutor path):
let outputs = execute_plan(&plan, &inputs)?;
```

With:
```rust
// New (tape path):
let tape = build_tape_from_plan(&plan)?;  // Once at load time
let outputs = execute_tape(&tape, &plan, &inputs)?;  // Per inference
```

The tape path is a strict superset — it handles all ops the KvExecutor handles, plus LUT-GEMM, KvCache, and GPU dispatch. The only difference is the tape is built once and reused.
