# hologram: Roadmap

---

## Current State

The hologram runtime is operational with the core pipeline complete:

- Graph IR with full operation set (21 LUT ops, 10 primitives, quantized matmul, custom ops)
- Compiler with three-stage pipeline (parse, fuse, plan & emit)
- KvExecutor with O(1) dispatch and parallel execution
- `.holo` archive format with zero-copy loading
- CLI with compile, run, and inspect commands
- SIMD acceleration (AVX2 / SSE4.2) for LUT operations
- LUT-GEMM with 4-bit and 8-bit quantization
- C ABI and WASM bindings
- Criterion benchmark suite (12 suites)

---

## Phase 1: Stability and Observability

**Goal:** Harden the existing system and improve developer experience.

### Scope

- Structured error types across all crates (replace string errors)
- Compilation statistics API (expose `FusionStats`, `WorkspaceLayout` metrics)
- Archive validation tool (standalone checksum verification)
- Benchmark regression tracking in CI
- Documentation: rustdoc coverage for all public APIs
- `hologram inspect` improvements: show fusion stats, workspace slot count, constant sizes

### Exit Criteria

- All public APIs have rustdoc with examples
- Benchmark suite runs in CI with regression detection
- `hologram inspect` shows complete archive metadata

---

## Phase 2: Advanced Compilation

**Goal:** Improve compilation quality and support more graph patterns.

### Scope

- Multi-pass fusion (iterative fusion until fixpoint)
- Dead node elimination pass
- Subgraph inlining improvements (better flattening heuristics)
- Workspace planning improvements (better bin-packing algorithms)
- Compilation caching (skip recompilation for unchanged graphs)
- Profile-guided optimization (record execution counts, optimize hot paths)

### Exit Criteria

- Fusion fixpoint converges in ≤3 iterations on all benchmark graphs
- Workspace peak memory reduced by ≥15% on representative workloads
- Compilation cache hits avoid full recompilation

---

## Phase 3: Execution Performance

**Goal:** Maximize execution throughput on supported platforms.

### Scope

- ARM NEON SIMD acceleration for LUT operations
- AVX-512 support for wider LUT vectorization
- Memory pool for `BufferArena` (avoid per-execution allocations)
- Streaming execution (process chunks without materializing full buffers)
- LUT-GEMM kernel optimizations (cache-aware blocking, prefetching)
- WASM SIMD support (`wasm32` with SIMD extensions)

### Exit Criteria

- ARM NEON LUT throughput within 90% of x86 AVX2
- BufferArena pooling eliminates allocation overhead for repeated executions
- LUT-GEMM within 2× of optimized BLAS for equivalent quantized workloads

---

## Phase 4: Platform Expansion

**Goal:** Broaden platform support and integration surface.

### Scope

- Metal compute backend for Apple Silicon
- WebGPU backend for browser deployment
- Improved WASM bindings (async execution, streaming)
- Python bindings via PyO3
- Node.js bindings via napi-rs
- Archive streaming (load sections on demand, not all at once)

### Exit Criteria

- Metal backend passes same golden tests as CPU
- Python bindings support full compile-and-execute workflow
- Archive streaming reduces memory footprint for large models by ≥50%

---

## Phase 5: Ecosystem

**Goal:** Support advanced use cases and multi-project integration.

### Scope

- Graph merging (combine multiple compiled graphs into a single archive)
- Distributed execution (split graphs across multiple executors)
- Hot-reload archives (swap `.holo` files without process restart)
- Custom section registry (formalize the custom section interface)
- Versioned archive migration (upgrade old `.holo` files to new format versions)

---

## Explicit Sequencing Rationale

**Stability before performance** because:
- Correctness bugs are harder to find after optimization changes
- Structured errors make debugging faster for all consumers
- Benchmark baselines must exist before optimization claims

**CPU before GPU** because:
- CPU backend runs on all CI machines and platforms
- Numerical correctness is easier to verify on CPU
- GPU backends depend on platform-specific toolchains

**Compilation quality before execution speed** because:
- Better fusion produces fewer nodes → less work for the executor
- Workspace planning improvements reduce memory pressure globally
- Compilation runs once; execution runs many times

---

## Deferred Items (explicitly not in any phase above)

- Training / autograd
- Distributed compilation (compiling across machines)
- Custom hardware backend interfaces (FPGA, ASIC)
- Graph visualization tools
- Interactive debugger for execution traces
