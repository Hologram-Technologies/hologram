# Plan: Sprint 15 — Graph & mmap Performance Hardening

## Context

Two Meilisearch blog posts reveal performance patterns directly applicable to hologram:

1. **"Patching LMDB"** — 3x speedup by eliminating cache pollution from eager mmap pre-fetching during HNSW graph construction. Core insight: lazy on-demand page loading preserves the working-set pages graph traversal actually needs.

2. **"From Trees to Graphs (Hannoy)"** — 10x speedup switching from tree-based to graph-based vector search. Key lessons: bidirectional edge links for O(degree) reverse lookups, cache-friendly edge storage, incremental graph mutation via link patching.

Both deal with **graph data structures on memory-mapped storage** — the same fundamental pattern hologram uses (computation graph + mmap'd weights via memmap2).

## Phase 1: Hot-Path Allocation Elimination (P0)

### 1.1: Eliminate `to_vec()` copies in tape execute loop

**File**: `crates/hologram-exec/src/tape.rs` (line 117)

**Problem**: `input_bufs.push(data.to_vec())` allocates and copies every input buffer for every instruction. This is the same anti-pattern as Meilisearch's "collect all pointers into a HashMap" — unnecessary materialization in the hottest loop.

**Fix**: Change kernel functions to accept borrowed slices (`&[&[u8]]`) instead of owned `Vec<Vec<u8>>`. For kernels that need mutation, use a single pre-allocated scratch buffer.

**Reuse**: `BufferArena::get()` already returns `&[u8]` — the data is already borrowed. The copy is unnecessary for read-only kernel inputs.

### 1.2: Upgrade prefetch from `black_box` to `_mm_prefetch`

**File**: `crates/hologram-exec/src/tape.rs` (lines 100-110)

**Problem**: `black_box(data.first())` is a blocking load instruction that stalls the pipeline. True `_mm_prefetch` is non-blocking.

**Fix**: Use the platform-specific prefetch wrappers already spec'd in `specs/plans/005-compile-time-acceleration.md` (lines 666-668):
- x86_64: `_mm_prefetch(ptr, _MM_HINT_T0)`
- aarch64: `__prefetch(ptr)`
- WASM: no-op

## Phase 2: mmap Page Discipline (P1)

### 2.1: Add `madvise` hints for mmap'd weight regions

**File**: `crates/hologram-archive/src/loader/` and `crates/hologram-exec/src/mmap/mod.rs`

**Fix**: After mmap creation, call:
- `madvise(MADV_RANDOM)` on weight section (LUT-GEMM access is random within layers)
- `madvise(MADV_SEQUENTIAL)` on graph section (read once at load)
- Use `memmap2::Mmap::advise()` or `libc::madvise`

### 2.2: Weight-page prefetch for next instruction

**File**: `crates/hologram-exec/src/tape.rs`

**Problem**: The tape knows the next instruction's constant offset at compile time, but only prefetches arena buffers, not weight pages.

**Fix**: If `next.kernel` is a LUT-GEMM variant, prefetch the weight region for the next instruction's constant. Requires the tape instruction to carry `constant_offset: Option<u32>` (already partially spec'd as `tile_hint`).

### 2.3: Audit tape builder for eager weight-page touching

**File**: `crates/hologram-exec/src/tape_builder.rs`

**Audit**: Verify that tape building only reads weight offsets/sizes, not weight data. If shape propagation reads weight content, refactor to use metadata only.

## Phase 3: Graph Edge Efficiency (P2)

### 3.1: Reverse-edge index for O(degree) `successors()`

**File**: `crates/hologram-graph/src/graph/mod.rs` (lines 336-344)

**Problem**: `successors()` iterates all nodes — O(n). Hannoy maintains bidirectional links for O(degree) reverse lookups.

**Fix**: Add `successors_cache: Option<Vec<Vec<NodeId>>>` to `Graph`. Build lazily on first `successors()` call, invalidate on mutation. Alternative: build once during `ExecutionSchedule` construction where it's most useful.

### 3.2: SmallVec for InputSlot

**File**: `crates/hologram-graph/src/graph/node.rs`

**Problem**: `inputs: Vec<InputSlot>` heap-allocates even for 1-2 input nodes (the common case). Each Vec is 24 bytes (ptr + len + cap) plus heap allocation.

**Fix**: `inputs: SmallVec<[InputSlot; 2]>` — inlines up to 2 inputs (covers unary + binary ops), spills to heap for variadic ops. `smallvec` is already a workspace dependency.

## Phase 4: Observability (P3)

### 4.1: Page-fault tracking in benchmarks

**File**: `crates/hologram-bench/benches/executor.rs`

**Fix**: Add a benchmark group that measures page faults via `getrusage(RUSAGE_SELF)` before/after execution, or document `perf stat` invocation in Justfile.

## Key Files to Modify

| File | Changes |
|------|---------|
| `crates/hologram-exec/src/tape.rs` | Phases 1.1, 1.2, 2.2 |
| `crates/hologram-exec/src/tape_builder.rs` | Phase 2.3 audit |
| `crates/hologram-archive/src/loader/mmap_loader.rs` | Phase 2.1 |
| `crates/hologram-graph/src/graph/mod.rs` | Phase 3.1 |
| `crates/hologram-graph/src/graph/node.rs` | Phase 3.2 |
| `crates/hologram-bench/benches/executor.rs` | Phase 4.1 |

## Verification

- `cargo test --workspace` — all existing tests pass
- `cargo clippy -- -D warnings` — zero warnings
- `just bench` — baseline before/after comparison
- On Linux: `perf stat -e major-faults,minor-faults,cache-misses cargo bench --bench executor`
- On macOS: `instruments -t "System Trace"` or `sudo dtrace -n 'fbt::vm_fault:entry { @[execname] = count(); }'`
