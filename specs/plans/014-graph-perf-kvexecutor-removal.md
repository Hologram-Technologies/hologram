# Plan 014: Graph Performance + KvExecutor Removal

## Context

After fixing the fusion pass O(N²) successor lookups (commit `6ad9e12`, -41% on 1000-node fusion), we audited the full codebase and found the same `graph.successors()` antipattern in toposort, level building, and CSE. Additionally, the deprecated KvExecutor dispatch path and its ~10 convenience functions are ready for removal — the EnumTape path is 17-140x faster and fully covers all use cases.

---

## Part A: Graph & Scheduling Optimizations

### A1. Toposort — O(N²) `graph.successors()` in Kahn's loop

**Files**:
- `crates/hologram-graph/src/schedule/toposort.rs` — line 38
- `crates/hologram-graph/src/schedule/levels.rs` — line 70
- `crates/hologram-graph/src/graph/validate.rs` — line 65

All three use `graph.successors(id)` (O(V+E) full scan) inside Kahn's main loop. Fix: build `succ_index` once before the loop, use `Graph::successors_from_index()`.

### A2. CSE `rewire_successors()` — O(V×E) per eliminated node

**Files**:
- `crates/hologram-graph/src/fusion/cse.rs` — line 44
- `crates/hologram-graph/src/graph/mod.rs` — lines 571-581

`rewire_successors()` scans ALL slots × ALL inputs. Fix: add `rewire_successors_indexed()` that only visits actual successors from the pre-built index. CSE builds the index once before its loop and passes it through.

### A3. Eliminate double toposort in fusion pass

**File**: `crates/hologram-graph/src/fusion/mod.rs` — lines 48, 75

Reuse the original topo order for CSE. Removed nodes are skipped via `graph.get(id).is_none()`. Topo ordering invariant is preserved since fusion only removes nodes.

---

## Part B: KvExecutor Removal

Remove the entire legacy KvExecutor dispatch path. The EnumTape path (`build_tape_from_plan` → `execute_tape`) is the canonical execution path and is 17-140x faster.

### B1. Remove deprecated convenience functions from `mmap/mod.rs`

**File**: `crates/hologram-exec/src/mmap/mod.rs`

Remove these deprecated functions (all have `#[deprecated(since = "0.10.0")]`):
- `execute_plan()` — lines 26-35
- `execute_plan_with_shape_hints()` — lines 83-101
- `execute_plan_with_kv_state()` — lines 107-127
- `execute_bytes()` — lines 169-177
- `execute_bytes_with_ops()` — lines 182-195
- `execute_bytes_with_progress()` — lines 200-223
- `execute_file()` — lines 228-238
- `execute_plan_zero_copy()` — lines 434-441

Also remove `execute_plan_with_intermediates()` and `execute_plan_with_intermediates_and_shape_hints()` (feature-gated `profile`, uses KvExecutor internally).

Keep `build_tape_from_plan()`, `execute_tape()`, `execute_tape_with_kv()` — these are the canonical API.

Remove internal helpers that only served the deprecated path: `dispatch_layers()`, `execute_graph_entrypoint()`, `verify_checksums()` if only used by deprecated functions.

### B2. Remove `KvExecutor` struct

**File**: `crates/hologram-exec/src/eval/executor.rs`

Remove:
- `KvExecutor` struct (line ~250)
- All `impl KvExecutor` blocks (`execute_with_plan`, `dispatch_level`, `execute_with_intermediates`, `execute_with_intermediates_and_shape_hints`)
- `build_schedule()` function in `eval/schedule_bridge.rs` if only used by KvExecutor

Keep: `GraphInputs`, `GraphOutputs`, `IntermediateCapture` (if used by tape path too)

### B3. Update re-exports

**File**: `crates/hologram-exec/src/lib.rs`
- Remove `KvExecutor` from re-exports (line 29)
- Remove deprecated mmap functions from re-exports (lines 34-37)
- Remove `build_schedule` if removed
- Keep `build_tape_from_plan`, `execute_tape`, `execute_tape_with_kv`

**File**: `src/lib.rs` (root crate)
- Remove `execute_bytes`, `execute_bytes_with_ops`, `execute_plan`, `execute_plan_with_kv_state`, `execute_plan_with_shape_hints`, `execute_file`, `KvExecutor` from re-exports (lines 51-65)
- Remove `#[allow(deprecated)]` annotations
- Remove `execute_plan_with_intermediates*` (profile-gated)
- Keep: `build_tape_from_plan`, `execute_tape`, `execute_tape_with_kv`, `BufferArena`, `CustomHandler`, `CustomOpRegistry`, `GraphInputs`, `GraphOutputs`, `KvCacheState`, `KvStore`

### B4. Update internal consumers

**File**: `crates/hologram-async/src/executor/mod.rs`
- `AsyncExecutor::execute()` calls `execute_bytes()` — rewrite to use `build_tape_from_plan` + `execute_tape`
- Or remove `AsyncExecutor` entirely and provide a tape-based async wrapper

**File**: `crates/hologram-async/src/stream/mod.rs`
- `execute_stream()` calls `execute_bytes_with_progress()` — rewrite to use tape path with level callbacks

**File**: `crates/hologram-ffi/src/exec/mod.rs`
- `hologram_execute_bytes()` calls `execute_bytes()` — rewrite to use tape path
- `hologram_ffi/src/wasm/mod.rs` — same

**File**: `crates/hologram-cli/src/commands/run_cmd.rs`
- `execute()` calls `execute_plan()` (line 76) — use tape path
- `run_generation()` calls `KvExecutor::execute_with_plan()` (line 455) — use `execute_tape_with_kv()`

**File**: `crates/hologram-bench/benches/executor.rs`
- Multiple benches use `execute_bytes()` — update to tape path
- Keep `tape_vs_kv` bench? No — remove KvExecutor half, rename to just tape bench

### B5. Update tests

**File**: `tests/e2e.rs` — ~15 call sites use `execute_bytes()` or `execute_plan()`
**File**: `tests/custom_ops.rs` — ~9 call sites use `execute_bytes_with_ops()`

All need migration to `build_tape_from_plan` + `execute_tape` (or a new `execute_bytes_via_tape()` test helper).

### B6. Clean up the `#[allow(deprecated)]` annotations

After removal, remove all `#[allow(deprecated)]` annotations that were suppressing warnings for these items:
- `src/lib.rs` lines 52, 54
- `crates/hologram-exec/src/lib.rs` lines 28, 33, 43
- `crates/hologram-async/src/executor/mod.rs` line 3, 15
- `crates/hologram-async/src/stream/mod.rs` line 3, 25
- `crates/hologram-ffi/src/exec/mod.rs` line 5, 55
- `crates/hologram-cli/src/commands/run_cmd.rs` lines 12, 37, 359

---

## Part C: SPRINT.md Update

**File**: `specs/SPRINT.md`

Add a new section at the end:

```markdown
## Sprint 17: Performance Hardening + KvExecutor Removal

**Plan**: [plans/014-graph-perf-kvexecutor-removal.md](plans/014-graph-perf-kvexecutor-removal.md)

### Phase 1: Graph Successor Index Optimization
- [ ] **1.1**: Successor index in toposort Kahn's loop (O(N²) → O(V+E))
- [ ] **1.2**: Successor index in build_parallel_levels (O(N²) → O(V+E))
- [ ] **1.3**: Successor index in validate acyclicity check
- [ ] **1.4**: Indexed rewire_successors for CSE pass

### Phase 2: Fusion Pass Optimization
- [ ] **2.1**: Eliminate double toposort — reuse original order for CSE
- [x] **2.2**: Pre-built successor index in fusion pass (commit 6ad9e12)

### Phase 3: KvExecutor Removal
- [ ] **3.1**: Migrate hologram-async to tape path
- [ ] **3.2**: Migrate hologram-ffi to tape path
- [ ] **3.3**: Migrate hologram-cli to tape path
- [ ] **3.4**: Migrate e2e tests to tape path
- [ ] **3.5**: Migrate custom_ops tests to tape path
- [ ] **3.6**: Migrate executor benchmarks to tape path
- [ ] **3.7**: Remove KvExecutor struct + dispatch_level + execute_with_plan
- [ ] **3.8**: Remove deprecated mmap convenience functions
- [ ] **3.9**: Clean up re-exports and #[allow(deprecated)]
```

---

## Part D: hologram-ai Migration Prompt

Generate a prompt for hologram-ai to update its imports after the KvExecutor removal. Save in the plan output.

### Prompt for hologram-ai session:

```
hologram has removed the deprecated KvExecutor dispatch path. The following
items no longer exist in the hologram public API:

REMOVED TYPES:
- hologram::KvExecutor (struct)

REMOVED FUNCTIONS:
- hologram::execute_plan()
- hologram::execute_plan_with_shape_hints()
- hologram::execute_plan_with_kv_state()
- hologram::execute_bytes()
- hologram::execute_bytes_with_ops()
- hologram::execute_file()
- hologram::execute_plan_with_intermediates()          [profile feature]
- hologram::execute_plan_with_intermediates_and_shape_hints()  [profile feature]

REPLACEMENT API (unchanged, still available):
- hologram::build_tape_from_plan(&plan) -> ExecResult<EnumTape>
- hologram::execute_tape(&tape, &plan, &inputs) -> ExecResult<GraphOutputs>
- hologram::execute_tape_with_kv(&tape, &plan, &inputs, &mut kv_state) -> ExecResult<GraphOutputs>

MIGRATION PATTERN:
  // Before:
  let outputs = hologram::execute_plan(&plan, &inputs)?;

  // After:
  let tape = hologram::build_tape_from_plan(&plan)?;
  let outputs = hologram::execute_tape(&tape, &plan, &inputs)?;

  // For autoregressive generation (KV cache):
  // Before:
  let outputs = KvExecutor::execute_with_plan(graph, &schedule, &inputs, weights)?;

  // After:
  let tape = hologram::build_tape_from_plan(&plan)?;
  let outputs = hologram::execute_tape_with_kv(&tape, &plan, &inputs, &mut kv_state)?;

KEY BENEFIT: Build the tape ONCE at model load time, then reuse for all
inference calls. The tape is 17-140x faster than KvExecutor — zero dispatch
overhead, zero per-inference allocation.

Please search the hologram-ai codebase for any imports or uses of the removed
items and migrate them to the tape API. The hologram-ai conformance tests
should be the priority — ensure they pass with the new API.
```

---

## Verification

1. `cargo test --workspace` — all tests pass after each phase
2. `cargo bench -p hologram-bench --bench fusion` — toposort improvement compounds in fusion
3. `cargo bench -p hologram-bench --bench compiler` — compile/liveness benchmarks
4. `cargo bench -p hologram-bench --bench executor` — tape benchmarks (KvExecutor benches removed)
5. `cargo clippy --workspace -- -D warnings` — zero warnings, no `#[allow(deprecated)]` remaining
6. Verify no remaining references to removed functions: `rg 'execute_bytes|execute_plan[^_]|KvExecutor' --type rust`
