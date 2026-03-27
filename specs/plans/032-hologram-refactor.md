# Hologram Ring-Native Refactoring: Conformance/Test-First Plan

## Context

The hologram workspace (10 crates, ~48K LoC) implements O(1) compute acceleration using precomputed LUT tables at Q0-Q1 and algorithmic ops at Q3+. The architecture has reached its ceiling: LUTs don't scale past Q1 (128KB/table), the float escape hatch (`FloatOp`) breaks ring closure, and the tape-based executor prevents native code generation.

This refactoring makes the entire stack parametric over quantum level, replaces LUTs with ring-primitive compositions, and introduces Cranelift JIT as an **optional acceleration layer** alongside the existing tape executor. The governing principle: **every operation is a ring primitive or a composition of ring primitives. No fallbacks. No escape hatches. No foreign-domain transitions.**

If an operation cannot be expressed as a composition of ring primitives at the chosen quantum level, the quantum level is wrong — not the operation.

All planned types and operations are **zero-cost**: zero-sized type (ZST) level markers, `const fn` ring operations, `#[inline]` on every hot path, monomorphization (no dynamic dispatch), `#![no_std]` in the kernel crate.

The new implementation is built under `prism-*` crate names during development. Once complete and all tests pass, `prism-*` crates are renamed back to `hologram-*`. The public API surface consumed by hologram-ai is preserved. The `hologram` name, archive format (`HOLO`), and binary name remain unchanged.

This plan follows strict test-first discipline: conformance tests are written FIRST to define the contract, then implementation makes the tests pass.

---

## Hard Requirements

These are non-negotiable constraints. Violation of any is a plan rejection.

### HR-1: Memory-Mapped O(1) Weight Access

Archives contain weights >1GiB (commonly 4-8GiB). These MUST run on resource-constrained devices (2GiB RAM). The full archive must never be read or loaded into memory — not during compilation, not during initialization, not during execution.

1. **Zero weight bytes touched during compilation** — metadata only (byte offsets + sizes from `ConstantData::Deferred`)
2. **O(1) weight access per graph op** — pre-computed byte offset baked into instruction, no search/index/hash
3. **No upfront processing of the weight section** — no scan, no index build, no integrity check over weights at load time
4. **Demand-paged mmap preserved** — `Cow::Borrowed(&mmap[offset..])`, physical RAM proportional to working set, not archive size
5. **Prefetch hints for upcoming weight accesses** — `weight_offset_hint` pattern preserved
6. **WeightCache persistence** — borrowed `RefCell<WeightCache>`, no re-deserialization across calls
7. **`madvise(MADV_RANDOM)` on weight section** — no sequential readahead

Current system: 4GiB archive on 2GiB device → ~600MiB physical RAM working set. This must be preserved.

### HR-2: WASM Target Support

`cranelift-jit` requires mmap + executable memory — not available on wasm32. No JIT backend solves this (fundamental WebAssembly constraint). The tape execution path must be preserved as a WASM fallback. `hologram-ffi` with `wasm` feature must continue to work.

### HR-3: blake3 Checksums Only

All archive checksums must use blake3. No CRC. The plan previously referenced "CRC32/Blake3" — only blake3 is carried forward.

### HR-4: Single-Crate Public API

Consumers depend on `hologram` only — never on subcrates directly. All public types re-exported flat from `src/lib.rs`. The `GraphOp<Q>` generic must NOT leak into the public API — use a type alias or erased wrapper. Internal crate names are invisible to consumers.

### HR-5: Q3+ Performance Parity

Q3+ inference throughput must not regress beyond 1.5× of current. LUT-GEMM psumbook advantage (32× fewer multiplies for Q4) and SIMD activation lookup (5-8× faster than polynomial) must be preserved or matched.

---

## Architecture Decision: Hybrid JIT + Tape

**Full JIT replacement (original plan) is rejected** due to HR-1 through HR-5 violations. The revised architecture is **hybrid**: JIT accelerates elementwise chain fusion while tape handles specialized kernels.

### What Runs Where

**Tape-only (never JIT)**:
- All ops that access `ConstantData::Deferred` weights (mmap path with prefetch) — HR-1
- `MatMulLut4/8` + psumbook kernel — preserves 32× multiply reduction — HR-5
- `LutView`, `PrimUnary` at Q0/Q1 — table lookup is O(1), unbeatable — HR-5
- `KvWrite`, `KvRead` — stateful RefCell borrows, must serialize — HR-1
- `Custom` ops — user-defined handlers with arbitrary Rust closures
- All WASM execution — HR-2

**JIT candidates (elementwise chains on native targets)**:
- Chains of 3+ unary/binary float ops on same tensor shape, no weight access, no runtime state
- Composed reductions feeding into elementwise ops
- Norm + activation patterns not already fused in tape

### `TapeKernel::JitSegment`

Extend `TapeKernel` with a JIT variant instead of replacing tape:

```rust
TapeKernel::JitSegment {
    func: JitFnPtr,    // fn(*const *const u8, *mut u8, usize)
    n_inputs: u8,
}
```

The execute loop dispatches JitSegment like any other kernel. Arena, prefetch, parallel execution, decode optimization — all unchanged. JIT is just another kernel type.

### Compiler Decision Logic

New pass: `jit_partition_stage` between `fuse_stage` and `emit_stage`:
1. Walk nodes in topological order
2. Classify as JIT-eligible: elementwise float ops, no weight access, no runtime state
3. Greedily extend chains (same tensor shape, single-consumer successors)
4. Chains of 3+ ops → `JitSegment`; shorter chains → inline tape kernels
5. `#[cfg(target_arch = "wasm32")]` → no-op (all tape) — HR-2

### Crate Organization

```
hologram-ring (kernel, was hologram-core) — parametric ring foundation
hologram-graph (kernel)                   — parametric GraphOp<Q>, fusion, scheduling
hologram-archive (kernel)                 — HOLO v1, blake3 checksums
hologram-exec (bridge)                    — tape execution + TapeKernel::JitSegment
hologram-jit (bridge, NEW)                — Cranelift codegen, feature-gated, never wasm32
hologram-compiler (bridge)                — pipeline + jit_partition_stage
hologram-compression (bridge)             — parametric over RingWord
hologram-ffi (user)                       — unchanged, WASM preserved
hologram-cli (user)                       — add --jit flag
hologram-bench (user)                     — benchmarks for both paths

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
hologram-jit = { path = "crates/hologram-jit", optional = true }

[features]
jit = ["hologram-jit"]  # default on native
```

`hologram-exec` does NOT depend on `hologram-jit`. It only holds `JitFnPtr` (a plain function pointer). The JIT crate is needed at compile/load time only.

---

## hologram-ai Consumer Contract

hologram-ai is the primary consumer (ADR-0001). The refactored hologram must maintain these contract surfaces. All types re-exported flat from `use hologram::*` (HR-4).

| Contract Surface | Status |
|---|---|
| `Graph`, `GraphBuilder`, `GraphOp`, `NodeId` | Preserved (GraphOp variants change — see Phase 8B). `GraphOp<Q>` generic is internal; public API uses type alias. |
| `PrimOp` (10 ring primitives) | Preserved (identical) |
| `ActivationOp` (replaces `LutOp`) | Renamed — hologram-ai adopts new enum |
| `CustomOpId`, `CustomOpRegistry`, `CustomHandler` | Preserved (identical) |
| `ConstantData`, `ConstantId`, `ConstantStore` | Preserved (identical) |
| `compile(&graph) -> CompilationOutput` | Preserved (same API, new internals) |
| `HoloWriter`, `HoloLoader`, `LoadedPlan` | Preserved (new archive format, version 1) |
| Execution: `build_tape` + `execute_tape` | Preserved (tape is the primary path). JIT segments transparent — dispatched inside same tape loop. |
| `KvCacheState` for autoregressive gen | Preserved (tape-backed, never JIT) |
| `WeightCache` (persistent, borrowed RefCell) | Preserved (HR-1) |
| `FloatOp` (70+ variants) | Eliminated — ring-native replacements |
| `LutOp` (21 activations) | Renamed to `ActivationOp` |
| `MatMulLut4/8/16` | Preserved on tape path (psumbook kernel). Graph builder helper `build_matmul_subgraph()` available. |
| `ElementWiseView` | Eliminated from public API (internal to fusion) |

---

## Phase 0: Scaffold prism crates alongside hologram

**Goal:** New crate skeletons exist, workspace compiles, zero hologram changes.

### Tests (written first)
```
prism-ring/tests/scaffold_test.rs
  - crate compiles with #![no_std]
  - placeholder types exist: RingWord, QuantumLevel, PrimOp, Involution, Encoding, Datum, Address
  - PrismPrimitives implements uor_foundation::Primitives
```

### Implementation
1. Create `crates/prism-ring/Cargo.toml` (`#![no_std]`, depends on `uor-foundation = "0.1.1"`)
2. Create empty `crates/prism-graph/`, `prism-compiler/`, `prism-archive/`, `prism-jit/`, `prism-compression/`, `prism-ffi/`, `prism-cli/`, `prism-bench/`
3. Add all to workspace `members` in root `Cargo.toml`
4. Add `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native` to `[workspace.dependencies]` — **feature-gated, cfg(not(wasm32))**
5. Verify `cargo test --workspace` passes (all hologram tests green)

### Files to create
- `crates/prism-ring/Cargo.toml` + `src/lib.rs`
- `crates/prism-graph/Cargo.toml` + `src/lib.rs`
- `crates/prism-compiler/Cargo.toml` + `src/lib.rs`
- `crates/prism-archive/Cargo.toml` + `src/lib.rs`
- `crates/prism-jit/Cargo.toml` + `src/lib.rs`
- `crates/prism-compression/Cargo.toml` + `src/lib.rs`
- `crates/prism-ffi/Cargo.toml` + `src/lib.rs`
- `crates/prism-cli/Cargo.toml` + `src/lib.rs`

---

## Phase 1: prism-ring — The Parametric Ring

The foundation. Everything builds on this. All types in this crate are zero-cost: ZSTs for levels, `const fn` for all ring arithmetic, `#[inline]` on every method, monomorphized generics.

### Phase 1A: RingWord trait + impls

**Tests** (`prism-ring/tests/ring_word_conformance.rs`):

For each W in {u8, u16, u32, u64, u128} — exhaustive for u8, sampled for larger:
1. **Closure**: `wrapping_add(a, b)` produces W
2. **Associativity**: `(a+b)+c == a+(b+c)` for add and mul
3. **Commutativity**: `a+b == b+a`, `a*b == b*a`
4. **Identity**: `a + ZERO == a`, `a * ONE == a`
5. **Additive inverse**: `a + wrapping_neg(a) == ZERO`
6. **Distributivity**: `a*(b+c) == a*b + a*c`
7. **Constants**: `ZERO == 0`, `ONE == 1`, `MAX == 2^BITS - 1`
8. **Bit intrinsics**: `count_ones`, `leading_zeros`, `trailing_zeros` match `core` intrinsics

**Implementation** (`prism-ring/src/word.rs`):
```rust
pub trait RingWord:
    Copy + Eq + Ord +
    core::ops::Add<Output = Self> + core::ops::Sub<Output = Self> +
    core::ops::Mul<Output = Self> + core::ops::BitXor<Output = Self> +
    core::ops::BitAnd<Output = Self> + core::ops::BitOr<Output = Self> +
    core::ops::Not<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;
    const BITS: u32;
    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn count_ones(self) -> u32;
    fn leading_zeros(self) -> u32;
    fn trailing_zeros(self) -> u32;
    fn from_u64(v: u64) -> Self;
    fn to_u64(self) -> u64;
}
```
Implement for u8, u16, u32, u64, u128. Each method is a one-liner delegating to Rust intrinsics.

### Phase 1B–1J

_Unchanged from original plan: QuantumLevel trait, PrimOp, Involution, Datum/Address, Ring/NDA/CD, Observables, Encoding, Accumulation._

### Phase 1I: Activations as ComposedOperations (REVISED)

**Hybrid activation strategy** per HR-5:

```rust
impl ActivationOp {
    #[inline]
    pub fn apply<Q: QuantumLevel>(&self, x: Q::Word) -> Q::Word {
        if Q::BITS <= 16 {
            // Table lookup — pre-evaluated ring function, O(1)
            Self::table::<Q>()[x.to_u64() as usize]
        } else {
            // Piecewise polynomial — computed ring function
            self.polynomial_eval::<Q>(x)
        }
    }
}
```

- Q0 (256 entries, 256B): LUT — fits in single cache line, SIMD via vpshufb/vqtbl1q
- Q1 (65536 entries, 128KB): LUT — fits in L2, still faster than polynomial
- Q3+ (4B+ entries): Piecewise polynomial — tables impractical, polynomial is correct approach

This is ring-consistent: the LUT IS a ring function (W→W), pre-evaluated at compile time. Monomorphized away — zero runtime dispatch.

---

## Phase 2: prism-graph — Parametric Graph IR

_Phase 2A–2D unchanged from original plan: arena graph, GraphOp<Q>, matmul subgraph, algebraic fusion, scheduling._

---

## Phase 3: prism-archive — .holo Format

**Tests** (`prism-archive/tests/archive_conformance.rs`):

1. Header magic: `b"HOLO"`, version 1
2. Quantum level in header: write Q3 → read → `quantum_index == 3`
3. Graph round-trip: write graph → read → same node_count, same topology
4. Constants round-trip: write ring constants → read → identical bytes
5. Weight section: write weights → read → identical bytes, dedup works
6. **blake3 checksum**: corrupt one byte → load returns error (HR-3)
7. Old archives incompatible (new header layout) — no backward compat needed
8. Multi-model: multiple graphs in one archive
9. **Mmap zero-copy preserved** (HR-1): `load_from_bytes_zero_copy` still borrows weight bytes from mmap
10. **madvise hints preserved**: `Advice::Random` on weight section, `Advice::Sequential` on graph

**Implementation**:
- Fork `hologram-archive`, keep magic `HOLO`, version stays 1 (fresh start, no backward compat)
- Serialize new `GraphOp<Q>` via rkyv
- **blake3 only** for all checksums (HR-3)
- Zero-copy mmap path preserved exactly (HR-1)
- Per-weight compression metadata (`compression_scheme: u8`) populated (future: lazy decompression)

**Header Layout** (144 bytes, `#[repr(C)]`, bytemuck Pod):

```rust
pub struct HoloHeader {
    pub magic: [u8; 4],              //   0: b"HOLO"
    pub version: u32,                //   4: 1 (fresh start, no backward compat)
    pub graph_offset: u64,           //   8: byte offset of serialized graph
    pub graph_size: u64,             //  16: byte size of graph section
    pub weights_offset: u64,         //  24: byte offset of weights section
    pub weights_size: u64,           //  32: byte size of weights section
    pub section_table_offset: u64,   //  40: byte offset of section table
    pub section_table_size: u64,     //  48: byte size of section table
    pub total_size: u64,             //  56: total archive size
    pub graph_checksum: [u8; 32],    //  64: blake3 hash of graph bytes
    pub weights_checksum: [u8; 32],  //  96: blake3 hash of weight bytes
    pub section_count: u32,          // 128: entries in section table
    pub flags: u32,                  // 132: compression + feature flags
    pub quantum_index: u32,          // 136: NEW — quantum level (0=Q0, 1=Q1, 3=Q3, 7=Q7)
    pub _reserved: u32,             // 140: padding for alignment
}
// Total: 144 bytes
```

No backward compatibility with the old hologram archive format. This is version 1 of the ring-native format — a fresh start. Old `.holo` files are simply incompatible (different header size, different graph serialization). Models must be recompiled.

### Reusable existing code
- [format/mod.rs](crates/hologram-archive/src/format/mod.rs): Header + section layout
- [loader/mod.rs](crates/hologram-archive/src/loader/mod.rs): Zero-copy load + madvise
- [writer/mod.rs](crates/hologram-archive/src/writer/mod.rs): Archive writing
- [weight/mod.rs](crates/hologram-archive/src/weight/mod.rs): Weight storage + dedup
- [checksum/mod.rs](crates/hologram-archive/src/checksum/mod.rs): blake3 implementation

---

## Phase 4: prism-compiler — Compilation Pipeline

**Pipeline** (REVISED for hybrid):
```
parse → validate → fuse → jit_partition → pattern → schedule → liveness → workspace → emit
```

- `jit_partition_stage`: NEW — identifies elementwise chains eligible for JIT (feature-gated, no-op on wasm32)
- `pattern_stage`: Recognize matmul/conv/attention, annotate tiling + psumbook eligibility
- Removed stages: `precision_stage` (parametric ring), `qedl_stage` (no boundary)

**Key addition — pattern_stage psumbook detection**:
```
PatternAnnotation::QuantizedMatmul { quantization: Q4/Q8, m, k, n }
  → emits psumbook-aware tape kernel (preserves 32× multiply reduction)
```

---

## Phase 5: prism-jit — Cranelift JIT (Feature-Gated)

**REVISED**: JIT is an optional accelerator, not a replacement. Feature-gated behind `jit` feature, never compiled for wasm32.

```toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
cranelift-codegen = "0.115"
cranelift-frontend = "0.115"
cranelift-jit = "0.115"
cranelift-module = "0.115"
cranelift-native = "0.115"
```

### Phase 5A–5B: Cranelift lowering for PrimOp + Activations

_Unchanged from original plan._

### Phase 5C: Cranelift lowering for elementwise chains (REVISED)

**Scope reduced**: JIT compiles fused elementwise chains only — NOT matmul loop nests (those stay on tape with psumbook).

**Tests** (`prism-jit/tests/jit_chain_conformance.rs`):
1. Chain of 3 unary ops: Relu → Sigmoid → Neg — JIT output bit-identical to sequential tape
2. Chain of 5 mixed ops: Add → Relu → Mul → Sigmoid → Sub — bit-identical
3. Shape-preserving: JIT chain on [1024] tensor produces same output as tape
4. **No weight access**: JIT segment receives only arena buffers, never mmap'd weight bytes

### Phase 5D: JIT module — compile segment (REVISED)

```rust
/// Compile a chain of elementwise ops into a native function pointer.
pub fn compile_segment(
    ops: &[FloatOp],
    input_shapes: &[&[usize]],
    output_shape: &[usize],
) -> Option<JitFnPtr>
```

The generated function loops over elements, reading from input pointers and writing to output. All intermediates stay in registers — this is the core JIT advantage.

**ABI**: `fn(inputs: *const *const f32, output: *mut f32, count: usize)`

### Phase 5E: Performance contracts (REVISED)

**Absolute budgets** with 5x CI headroom:
1. JIT compile 100-node linear graph: < 100ms
2. JIT-compiled chain (5 ops) throughput: 1M elements in < 50ms
3. Cache hit: second compile of same graph: < 1ms

**Relative regression gates** (run both old and new, compare):
```
prism-jit/tests/perf_regression.rs:
  - Q4 matmul 512×512: throughput >= 0.8× current hologram LUT-GEMM
  - Q8 matmul 512×512: throughput >= 0.8× current hologram LUT-GEMM
  - Q0 activation batch (64KB): throughput >= 0.9× current ElementWiseView SIMD
  - Q3 activation batch (64KB): throughput >= 0.5× current
  - Full E2E TinyLlama decode token: latency <= 1.5× current
```

---

## Phase 6: prism-compression — Parametric Compression

_Unchanged from original plan, plus:_

**Future work (not blocking Phase 8)**: Per-weight compression with lazy decompression into bounded `WeightCache` with LRU eviction. This allows compressed archives on constrained devices without breaking mmap (HR-1). The per-tensor `compression_scheme: u8` field in weight metadata should be populated.

---

## Phase 7: Peripheral Crates

### prism-ffi
- Tests: C ABI smoke tests, **WASM compilation gate** (HR-2)
- Implementation: `extern "C"` wrappers for compile/execute
- **WASM path uses tape execution only** — no JIT dependency

### prism-cli
- Tests: `--help`, `compile`, `run`, `inspect`
- Add `--jit` flag to enable JIT segments (default on native, unavailable on wasm32)

### prism-bench
- Suites: ring ops per level, graph fusion, JIT compile latency, JIT chain throughput, **tape vs JIT comparison**, matmul at each Q level, archive I/O

---

## Phase 8: End-to-End Integration + hologram-ai Contract

### Phase 8A: Full pipeline E2E

_Unchanged from original plan._

### Phase 8B: hologram-ai contract conformance

**REVISED** — tape execution is primary path, JIT is transparent acceleration:

**Tests** (`tests/hologram_ai_contract_conformance.rs`):
1. **Graph construction**: `Graph`, `GraphBuilder`, `GraphOp`, `NodeId` — monomorphic API (HR-4)
2. **Custom op registry**: unchanged
3. **Compile pipeline**: `compile(&graph) -> CompilationOutput` — unchanged
4. **Archive I/O**: unchanged, blake3 checksums (HR-3)
5. **Tape execution**: `build_tape_from_plan` + `execute_tape` — preserved as primary path
6. **JIT acceleration**: JIT segments dispatched transparently inside tape loop (feature-gated)
7. **KV cache**: `KvCacheState` for autoregressive — tape-backed, never JIT
8. **Mmap weight access**: `ConstantData::Deferred` resolved O(1) via mmap slice (HR-1)
9. **Error types**: `GraphError`, `CompileError`, `ExecError` — updated for ring-native ops

### Phase 8C: Remove old hologram crates, rename prism → hologram

**GATE**: Phase 8C must NOT proceed until:
1. All E2E + contract tests pass
2. All perf regression gates pass (Phase 5E)
3. WASM compilation gate passes (Phase 7)
4. Mmap O(1) weight access verified on constrained device profile (HR-1)

Once gates pass:
1. Remove `crates/hologram-*` (old) from workspace
2. Rename `crates/prism-*` → `crates/hologram-*`
   - `prism-ring` → `hologram-ring` (was `hologram-core`)
   - `prism-graph` → `hologram-graph`
   - `prism-compiler` → `hologram-compiler`
   - `prism-archive` → `hologram-archive`
   - `prism-jit` → `hologram-jit` (NEW, feature-gated)
   - `prism-compression` → `hologram-compression`
   - `prism-ffi` → `hologram-ffi`
   - `prism-cli` → `hologram-cli`
   - `prism-bench` → `hologram-bench`
3. **hologram-exec preserved** — tape execution remains, with `TapeKernel::JitSegment` variant
4. Update `src/lib.rs`: flat re-exports, `GraphOp<Q>` aliased to hide generic (HR-4)
5. Archive magic stays `HOLO`, binary name stays `hologram`
6. Update `CLAUDE.md`, `AGENTS.md`, `README.md`, `hologram.repo.yaml`

---

## What Is Eliminated

| Eliminated | Reason | Replaced By |
|---|---|---|
| `ElementWiseView16` (128KB LUT) | Parametric ring | Polynomial at Q3+, LUT preserved at Q0/Q1 |
| `FloatOp` enum (70+ variants) | Breaks ring closure | Ring-native `Activation`, `Prim`, `Custom` |
| `FusedFloatChain` | Same | Algebraic fusion of ring ops |
| `QedlBoundary` | No float/ring boundary | Encoding at graph I/O only |
| `RingPrimUnary`/`RingPrimBinary` | Redundant | `Prim(PrimOp)` |
| `precision_stage` in compiler | Parametric ring | QuantumLevel parameter |
| `qedl_stage` in compiler | No boundary | Encoding at graph I/O only |
| `CurvatureFlux` runtime tracking | Dynamic selection eliminated | Compile-time quantum level |
| CRC checksums | HR-3 | blake3 only |

## What Survives

| Surviving | Why |
|---|---|
| 10 ring primitives | UOR foundation ontology |
| Tape execution engine | HR-1 (mmap weight access), HR-2 (WASM), HR-5 (psumbook) |
| LUT-GEMM psumbook kernel | HR-5 — 32× fewer multiplies for Q4 |
| ElementWiseView at Q0 (256B LUT) | HR-5 — O(1) per element, SIMD accelerated |
| BufferArena + zero-allocation | Decode optimization (2× throughput) |
| Platform prefetch hints | Weight page fault overlap |
| KvCacheState (tape-backed) | Autoregressive generation |
| Metal/WebGPU backends | Hardware acceleration |
| mmap zero-copy loading | HR-1 |
| Dihedral group, critical identity, observables | Ring algebraic structure |
| Encoding pipeline (π-F-λ) | Graph I/O boundary |
| Graph IR, scheduling, liveness, workspace | Computation representation |
| Algebraic fusion (now ring-native) | Graph optimization |
| UOR trait hierarchy | Ontological grounding |
| rkyv serialization, blake3 archives | Portable persistence |

---

## Quantified Performance Expectations

### LUT-GEMM (preserved on tape, HR-5)

| Metric | Q4 LUT-GEMM | Q8 LUT-GEMM |
|--------|-------------|-------------|
| Multiplies per output element | 16 (centroids) | 256 (centroids) |
| Multiply reduction vs naive | **32×** | **2×** |
| L1 working set | 64B (Psumbook4) | 1KB (Psumbook8) |

### Activation (hybrid, HR-5)

| Path | Throughput |
|------|------------|
| Q0 LUT + AVX2 vpshufb (preserved) | ~1.7B elem/s |
| Q0 LUT + NEON vqtbl1q (preserved) | ~800M elem/s |
| Q3+ piecewise polynomial (new) | ~200M elem/s |

### JIT Chain Fusion (new benefit)

| Scenario | Tape (current) | JIT segment |
|----------|---------------|------------|
| 5-op elementwise chain on 64KB tensor | 5 loops, 5 arena round-trips | 1 fused loop, registers only |
| Estimated speedup | 1× | ~2-3× for chain-heavy subgraphs |

### Tape dispatch overhead: <1% of inference — not the bottleneck

---

## KV Cache Integration

KvCacheState stays on tape path (never JIT):
- `KvWrite`/`KvRead` TapeKernel variants preserved with heads-first/seq-first transpose
- `ExecutionContext.position_offset` drives RoPE + decode optimization
- `advance()` call in execution epilogue preserved
- KV ops serialize (no parallel execution — RefCell borrow safety)

---

## Error Taxonomy Updates

- `InsufficientKernel { op, dtype }` → update to reference ring-native ops + quantum levels
- `ExecError::UnsupportedOp` → update for ring-native op set
- `ArchiveError::InvalidMagic` / `ArchiveError::UnsupportedVersion` for old archives
- PX_5 model (Insufficient vs Contradictory) preserved conceptually

---

## Archive Migration

- No backward compatibility — old archives are simply incompatible (different header layout, different graph serialization)
- Version stays 1 (fresh start for the ring-native format)
- Models must be recompiled from source (ONNX/GGUF → new hologram pipeline)
- No migration tool needed — recompilation is the migration

---

## Test Mapping: Existing → New

| Existing Test | New Location | Notes |
|---|---|---|
| `ring_conformance.rs` | `prism-ring/tests/ring_word_conformance.rs` + `primop_conformance.rs` + `ring_uor_conformance.rs` | Parametric over all levels |
| `q3_conformance.rs` | `prism-ring/tests/ring_uor_conformance.rs` (CD chain) | |
| `carry_conformance.rs` | `prism-ring/tests/encoding_conformance.rs` | |
| `perf_contract.rs` | `prism-jit/tests/perf_contract.rs` + `perf_regression.rs` | Absolute + relative gates |
| `float_conformance.rs` | **Eliminated** — refs reused as oracle in `activation_conformance.rs` | |
| `gemm_conformance.rs` | `prism-graph/tests/matmul_pattern_conformance.rs` | Matmul as subgraph |
| `quantize_conformance.rs` | **Eliminated** | |
| `streaming_conformance.rs` | `prism-jit/tests/jit_chain_conformance.rs` | |
| `tests/e2e.rs` | `tests/e2e_prism.rs` | Full pipeline |

---

## Dependency Graph & Parallelism

```
Phase 0 (scaffold)
    │
Phase 1A─→1B─→1C─→1D─→1E─→1F─→1G─→1H─→1I─→1J   (prism-ring, sequential)
    │
    ├── Phase 2A─→2B─→2C─→2D          (prism-graph)
    │       │
    │       ├── Phase 3                 (prism-archive, parallel with 2C-2D)
    │       │
    │       └── Phase 4                 (prism-compiler, needs 2C+3)
    │               │
    │               └── Phase 5A─→5B─→5C─→5D─→5E  (prism-jit, feature-gated)
    │
    ├── Phase 6                         (prism-compression, parallel with 2-5)
    │
    ├── Phase 7                         (ffi/cli/bench, parallel with 5-6)
    │
    └── Phase 8A─→8B─→8C               (E2E + contract + rename, GATED)
```

**Critical path**: 0 → 1 → 2 → 4 → 5 → 8
**Phase 8C gate**: perf regression + WASM + mmap verification

---

## Concurrency Model: Async + Parallel + SIMD

The current system supports three independent concurrency axes. All must be preserved.

### Feature Axes

| Axis | Technology | Feature flag | Purpose |
|------|-----------|-------------|---------|
| **Async** | `tokio::task::spawn_blocking` | `async` | Non-blocking wrappers for compilation + execution |
| **Parallel** | `rayon::par_iter` | `parallel` | Level-wise node parallelism + column-parallel LUT-GEMM |
| **SIMD** | AVX2/SSE4.2/NEON intrinsics | `simd` | Vectorized LUT table application (Q0 activations) |

### How They Compose

```
AsyncExecutor::execute(archive, inputs).await
  → tokio::spawn_blocking (offload to blocking thread)
    → execute_tape() (sync, in blocking thread)
      → for each level:
          if level.len() >= 4 && !needs_shared_state:
            rayon::par_iter on instructions     ← PARALLEL
              → dispatch_kernel:
                  LutView → apply_slice()       ← SIMD (AVX2/NEON)
                  MatMulLut4 → psumbook kernel  ← sequential (RefCell)
          else:
            sequential with prefetch            ← PREFETCH (x86/ARM)
```

### Adaptive Parallel Dispatch

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `PARALLEL_THRESHOLD` | 4 nodes | Min nodes for rayon dispatch |
| `SMALL_BUFFER_BYTES` | 256 bytes | Raises threshold to 16 for tiny ops |
| `PAR_COL_THRESHOLD` | 64 columns | Min columns for parallel LUT-GEMM |

Ops that hold shared mutable state (`MatMulLut4/8`, `KvWrite/KvRead`) force sequential execution for their level. Other ops in the same level still benefit from prefetch overlap.

### SIMD Dispatch (Compile-Time)

| ISA | Intrinsic | Throughput | Target |
|-----|-----------|------------|--------|
| AVX2 | `vpshufb` | 32 bytes/iter, ~0.6 cycles/elem | x86_64 |
| SSE4.2 | `pshufb` | 16 bytes/iter | x86_64 fallback |
| NEON | `vqtbl1q_u8` | 16 bytes/iter, ~1.25 cycles/elem | aarch64 |
| Scalar | `table[byte]` | 1 byte/iter, ~4 cycles/elem | All platforms + WASM |

Detection is **compile-time only** (`#[cfg(target_feature)]`), no runtime CPUID. Scalar remainder always handled.

### What the Refactor Must Preserve

1. **Rayon level parallelism** — JitSegment dispatch must be parallelizable (no shared state)
2. **Column-parallel LUT-GEMM** — psumbook kernel stays on tape with per-thread stack-allocated Psumbook4 (64B, no false sharing)
3. **SIMD activation dispatch** — ElementWiseView at Q0/Q1 preserved with AVX2/NEON paths
4. **Prefetch overlap** — sequential path prefetches next instruction's inputs + weight pages
5. **Async wrappers** — `spawn_blocking` pattern preserved (tape is sync, async is wrapper)
6. **No rayon/tokio conflict** — separate thread pools, no contention

---

## Additional Design Decisions

### no_std / Bare-Metal

- `hologram-ring` (was hologram-core): `#![no_std]`, all `const fn`, zero allocations. `StaticBuf<N>` exists for stack-allocated buffers.
- `hologram-exec`: **requires `std`** — uses `Vec`, `HashMap`, mmap. No bare-metal story.
- **Gap**: No execution path for `no_std` targets beyond ring primitives. Embedded/RTOS devices can use `hologram-ring` for ring arithmetic but not graph execution.
- **Recommendation**: Accept this limitation. Ring primitives on bare-metal, full execution requires std + allocator.

### Custom Op Boundaries

Custom ops (`CustomHandler = Arc<dyn Fn + Send + Sync>`) are opaque closures. They:
- Act as **hard fusion boundaries** — fusion cannot cross them
- Have no shape contract — output shape inferred from byte count
- Are baked into `TapeKernel::Custom(handler)` at tape build time

**Impact on JIT partitioning**: A JIT-eligible chain interrupted by a custom op splits into two JIT segments with the custom op on tape between them. The `jit_partition_stage` must detect custom ops as chain terminators.

### Dynamic Shape Handling

Current shape resolution uses heuristic fallbacks (`shape_resolve.rs`):
1. Input TensorMeta (if available)
2. Compiled baked size
3. Buffer-length inference (`floats / k → m`)
4. Shape overrides from `TapeContext.shape_overrides`

In the ring-native model:
- `FloatDType` eliminated — dtype derived from quantum level (`Q3 → u32`)
- `TensorMeta` simplified: `ndim + dims[8]`, no dtype field (it's always `Q::Word`)
- Shape overrides preserved for variable-length sequences (autoregressive decode)
- KV cache shapes still resolved at runtime from `write_pos`

### Multi-Q Archives

The plan adds `quantum_index: u32` to the archive header — **one Q level per archive**. Mixed-Q graphs are not supported in a single archive. This is simpler than per-node Q selection (which the current precision_stage does) but means:
- All ops in one archive run at the same Q level
- Mixed-precision models need separate archives or a single Q level chosen conservatively

### Profiling

The `profile` feature flag exists but is unused. The refactor should add:
- Per-kernel timing via `std::time::Instant` guards (behind `profile` feature)
- JIT vs tape annotation in profiling output
- `TapeInstruction.kernel_name: &'static str` for diagnostic output

### Cranelift Versioning

Cranelift is pre-1.0 with breaking changes across minor versions. Strategy:
- Pin to a specific Cranelift version (e.g., 0.115.x)
- Track wasmtime release cycle for tested combinations
- Feature-gate behind `jit` — users who don't need JIT avoid the dependency entirely
- Treat Cranelift upgrade as a separate PR with JIT conformance test gate

### JitFnPtr Thread Safety

`JitFnPtr` wraps a raw function pointer to JIT-compiled code. Requirements:
- JIT-compiled code is position-independent and read-only after compilation → **safe to share across threads**
- Implement `unsafe impl Send for JitFnPtr {}` and `unsafe impl Sync for JitFnPtr {}` with safety comment
- `Drop` must deallocate JIT memory via `cranelift_jit::JITModule` (must happen on owning thread or be `Send`)
- Parallel level execution can dispatch JitSegment across rayon threads safely

---

## Build Time Impact

- **Current**: 132 packages, ~566 MB release, ~40-50 sec CI
- **Cranelift addition**: +20-30 sec incremental, ~130 MB
- **Mitigation**: feature-gated, isolated crate, CI cache (`Swatinem/rust-cache@v2`)

---

## Verification

After each phase:
```bash
cargo test --workspace              # all tests pass
cargo clippy --workspace -- -D warnings  # no warnings
cargo fmt --all -- --check          # formatted
```

After Phase 5E (JIT + regression):
```bash
cargo test -p prism-jit             # JIT conformance + perf regression
cargo bench -p prism-bench          # performance baselines
```

After Phase 8A (full E2E):
```bash
cargo test --workspace              # everything including e2e_prism
cargo test --target wasm32-unknown-unknown -p prism-ring  # WASM gate
```
