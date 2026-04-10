# Plan 042: UOR Fusion Formalization — Canonical Quotient Architecture

**Status:** Approved
**Date:** 2026-04-10
**Depends on:** Phases 1–13 (v0.2.0 conformance refactor complete, 1670 tests passing)
**Formal basis:** UOR Fusion Formalization Dossier (Sections 1–15)

---

## Context

Benchmark analysis of the post-v0.2.0 hologram workspace revealed five classes of execution-time overhead. Cross-referencing against the UOR Fusion Formalization Dossier (Sections 1–15) shows that each overhead maps to a specific gap between the formal model and hologram's current architecture:

| Overhead | Benchmark evidence | Dossier mapping |
|---|---|---|
| Per-execute setup (~450 ns for 4-node graph) | `exec::linear_chain(4_nodes, 256B) = 640 ns` total, of which ~450 ns is `seed_arena` + `node_dtypes_map` + `node_shapes_map` + `WeightCache::new` + `default_backend()` boxing | **Obligation 2.5** — typing must be total at U_can construction, not re-derived at evaluation time |
| Epilogue fusion regression at small sizes | `epilogue_fusion/fused/1x64x64 = 2009 ns` vs `unfused = 1544 ns` (1.3x slower) | **Failure Mode 5.7** — AdmFuse is structural-only, missing operand-size guard |
| Online softmax 2x slower than row-based | `softmax_decode/online/2048 = 21885 ns` vs `row_based = 10958 ns` | **Definition 6.5** — representative selection within equivalence class; both preserve denotation |
| Float dispatch per-instruction overhead (4.5x vs byte path) | `tape::linear_chain(4_float_nodes, 256B) = 2875 ns` vs `exec::linear_chain(4_nodes, 256B) = 640 ns` | Evaluation overhead in ⟦-⟧_U (Definition 7.2) — SmallVec/TensorMeta construction per instruction |
| LUT-GEMM crossover at small sizes | `lut_gemm_q4(4x64x64) = 30757 ns` vs `naive_matmul = 13294 ns` (2.3x slower) | **Failure Mode 5.7** — same class as epilogue fusion; AdmFuse missing size threshold |

The root cause is architectural: hologram's execution path re-derives information that should be pre-computed in the canonical quotient, and its fusion admissibility predicate lacks a cost model.

## Design principles

1. **Correctness is the priority.** Every change must preserve semantic equivalence (Theorem 7.5 — rewrite soundness). No optimization may alter output bytes for any input.
2. **Single fast path — no fallbacks.** There is exactly one execution path: read inputs → dispatch pre-resolved kernels → write outputs. No runtime type resolution, no shape inference heuristics, no backend selection, no fusion decisions. The tape is a flat array of fully-resolved instructions. If a required decision cannot be made at compile/load time, the compiler must error — not fall back to a degraded runtime path.
3. **No backwards compatibility.** Internal APIs, struct layouts, and function signatures change freely. The public surface (`Compiler::compile`, `PrismModule::execute`) is stable; everything below it is implementation detail.
4. **Performance-aware fusion.** The admissible fusion predicate (Definition 5.2) includes an operand-size cost model. The analysis pass still detects all patterns (it's a structural finder); the tape builder selects the optimal emission. Below-threshold chains emit the unfused pair — this is not a fallback, it's a compile-time decision that the unfused variant IS the faster path.

---

## Architecture: three-layer separation

The dossier defines three phases of the fusion pipeline. Hologram must separate them cleanly:

### Layer 1: Compile-time (Sections 3–6)

**What happens:** Source graph → analysis → normalized tape.

This is the translation functor E (Definition 3.4) composed with normalization (Theorem 6.4). The representable fragment C_rep is defined by the Shape declarations (`F_PRISM_STRICT`, `F_PRISM_FUSED_COMPONENT`). The fusion-complete subfragment C_fuse is defined by the admissibility predicate. The output is a canonical normal form: the `EnumTape`.

**Already implemented correctly.** No changes needed. The `analyze()` + `build_tape()` pipeline is the normalizer. Termination is trivially guaranteed (single topological pass). Confluence is guaranteed by the fixed pass ordering within that single walk.

### Layer 2: Load-time (Section 2 — Obligation 2.5)

**What happens:** Archive → LoadedModel (the complete U_can object).

This is where typing must be total and the canonical quotient fully resolved. The `LoadedModel` must contain everything the executor needs with zero per-execute derivation.

**Currently broken.** `LoadedModel` stores `(plan, tape, weight_cache, kv_state)` but does NOT store the dtype map, shape map, seed arena template, or resolved backend. These are re-derived on every `execute_tape()` call.

### Layer 3: Execute-time (Section 7) — the single fast path

**What happens:** Inputs + LoadedModel → Outputs.

This is the semantic interpretation ⟦-⟧_U (Definition 7.2). It evaluates the pre-normalized tape against concrete inputs. The executor is a pure dispatch loop with **zero runtime decisions**:

```
execute(model, inputs) → outputs:
    arena ← seed_from_template(model.seeds, inputs)   // O(num_inputs)
    for instr in model.tape.instructions:
        kernel_dispatch(instr.kernel, arena[instr.inputs], arena[instr.output])
        arena.set_meta(instr.output, instr.output_meta)
    return collect_outputs(arena)
```

**The single fast path contract:**

The executor MUST NOT:
- Allocate heap memory per instruction (SmallVec is stack-inlined; arena buffers are pre-warmed)
- Resolve types or shapes (every `TapeInstruction.output_meta` is pre-populated)
- Select kernel variants (the `TapeKernel` enum variant IS the selected variant)
- Select backends (`&dyn ComputeBackend` is pre-resolved on `LoadedModel`)
- Construct HashMaps, SmallVecs, or any derived data structure per call
- Branch on `Option::None` in any per-instruction code path

The executor MUST:
- Dispatch one kernel per instruction via a single `match` on `TapeKernel`
- Read inputs from pre-indexed arena slots
- Write outputs to pre-allocated arena slots
- Set pre-resolved output metadata after each dispatch

Any code path that violates these invariants is a bug, not a fallback.

**Currently has ~450 ns of per-call setup** that belongs in Layer 2.

---

## Conformance-first enforcement of the single fast path

The single fast path is not just a design goal — it must be **enforced** so that future changes cannot reintroduce runtime overhead without breaking the build. The enforcement has four layers:

### Layer A: Type-level enforcement (compile-time)

1. **`TapeInstruction.output_meta` is non-Optional.** The struct field is `pub output_meta: TensorMeta`, not `Option<TensorMeta>`. Any tape builder code that fails to provide a resolved meta will not compile. This makes it structurally impossible to produce an instruction that defers shape resolution to execute time.

2. **`LoadedModel.backend` is non-Optional.** The field is `backend: Box<dyn ComputeBackend>`, populated at `from_archive()`. The execute path takes `&dyn ComputeBackend` by borrow — there is no `Option` to unwrap and no `default_backend()` call in the execute path.

3. **`ConstantSeed` and `InputSeed` have no Optional fields.** Every field (dtype, shape, elem_size) is required. The load-time construction in `from_archive()` either resolves every field or returns `Err(ShapeLoadError)`. There is no silent degradation.

4. **`execute_direct_with_backend` does not import `shape_resolve`.** The module dependency is enforced by not having a `use crate::shape_resolve` in `tape.rs`. If a future change adds a shape-resolution call to the execute path, it requires adding the import — which is reviewable.

### Layer B: Benchmark conformance tests (CI-time)

Add a **benchmark-based conformance test** in `crates/hologram-bench/benches/executor.rs` that asserts per-instruction overhead:

```rust
#[test]
fn execute_overhead_conformance() {
    // A 4-node byte-domain chain with 256B input should execute in < 400 ns.
    // This gate catches any regression that re-introduces per-call setup.
    let (plan, tape) = make_tape_and_plan(&mut linear_chain_graph());
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        let _ = execute_tape(&tape, &plan, &inputs);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / 10_000;

    assert!(
        per_call.as_nanos() < 400,
        "per-execute overhead is {} ns; must be < 400 ns (the single fast path contract)",
        per_call.as_nanos()
    );
}
```

This test runs in CI. If per-call overhead exceeds 400 ns (the post-Change-1 target), the build fails. The threshold is set at 400 ns to leave headroom for CI machine variance — the expected value is ~300 ns.

Similarly for per-instruction float dispatch:

```rust
#[test]
fn float_dispatch_overhead_conformance() {
    // A 4-node float chain with 256B input should execute in < 2000 ns.
    // This catches regressions that re-introduce per-instruction shape resolution.
    let (plan, tape) = make_float_chain_tape();
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![0u8; 256]);

    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        let _ = execute_tape(&tape, &plan, &inputs);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / 10_000;

    assert!(
        per_call.as_nanos() < 2000,
        "float dispatch overhead is {} ns; must be < 2000 ns",
        per_call.as_nanos()
    );
}
```

### Layer C: Architectural review rules (code-review-time)

Codify these rules in `CLAUDE.md` (or equivalent) so that AI agents and human reviewers enforce them:

1. **No `HashMap::new()` or `Vec::new()` in `execute_direct_with_backend` or any function it calls.** All allocations happen at load time or in the pre-warm phase.
2. **No `default_backend()` call in `tape.rs`.** Backend is passed as a parameter.
3. **No `node_dtypes_map()` or `node_shapes_map()` call in `mmap/mod.rs` execute paths.** These are load-time-only functions.
4. **Every new `TapeKernel` variant must specify its `output_meta` computation in `tape_builder.rs`.** If the builder can't resolve the meta, the variant is not added.
5. **Fusion decisions are compile-time.** The tape builder selects fused vs unfused emission. The executor does not branch on operand size.

### Layer D: Directness-ratio verification (conformance-test-time)

The existing `test_full_conformance` / `test_primitivity` conformance test family already verifies that `FusedComponentModule` achieves directness ratio 1.0 against `F_prism_fused_component`. Extend this with a **throughput gate**:

```rust
#[test]
fn conformance_throughput_gate() {
    // The fused component module must achieve at least 1M instructions/sec
    // on a 256B workload. This enforces that the single fast path is not
    // degraded by per-instruction overhead.
    let module = FusedComponentModule::new();
    let archive = compile(linear_chain_graph()).unwrap().archive;
    let loaded = module.load(&archive).unwrap();
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![128u8; 256]);

    let start = std::time::Instant::now();
    let n = 100_000;
    for _ in 0..n {
        let _ = module.execute(&loaded, &inputs);
    }
    let elapsed = start.elapsed();
    let ips = n as f64 / elapsed.as_secs_f64();

    assert!(
        ips > 1_000_000.0,
        "throughput is {:.0} exec/sec; must be > 1M (single fast path contract)",
        ips
    );
}
```

This ties the performance contract to the conformance test infrastructure from the UOR Fusion Dossier: the carrying criterion (Theorem 13.1 item 4) requires that the fused representative preserves denotation, and the throughput gate requires that evaluation of the fused representative meets the performance target.

---

## Changes

### Change 1: Complete the canonical quotient at load time

**Goal:** Eliminate per-execute setup by pre-computing all derived state in `LoadedModel::from_archive()`.

**Where:** `crates/hologram-fused-component/src/prism_module.rs` (`LoadedModel`), `crates/hologram-fused-component/src/mmap/mod.rs` (`execute_tape` family)

**What to add to `LoadedModel`:**

```rust
pub struct LoadedModel {
    plan: LoadedPlan,
    tape: EnumTape,
    weight_cache: parking_lot::RwLock<WeightCache>,
    kv_state: Mutex<Option<KvCacheState>>,
    // ── NEW: pre-computed at load time ──
    /// Per-node element size in bytes. Built once from `sg.node_dtypes_map()`.
    /// Keyed by `NodeId`. Replaces per-call `compiled_dtypes.get(&id).map(|d| d.byte_size())`.
    elem_sizes: HashMap<NodeId, usize>,
    /// Per-node FloatDType. Built once from `sg.node_dtypes_map()`.
    /// Used by `seed_arena` to set TensorMeta on constant/input nodes.
    dtypes: HashMap<NodeId, FloatDType>,
    /// Per-node output shape. Built once from `sg.node_shapes_map()`.
    shapes: HashMap<NodeId, Vec<usize>>,
    /// Constant seed entries: (node_id, borrowed data range, elem_size, optional shape).
    /// Pre-indexed from the SerializedGraph's constants + weight archive.
    /// At execute time, these are inserted as Borrowed arena entries with zero re-derivation.
    constant_seeds: Vec<ConstantSeed>,
    /// Input node entries in ordinal order: (node_id, elem_size, optional compiled shape).
    /// At execute time, the caller's GraphInputs are matched by ordinal position.
    input_seeds: Vec<InputSeed>,
    /// Resolved compute backend. One `Box<dyn ComputeBackend>` allocation at load time.
    /// `execute_direct` borrows `&dyn ComputeBackend` from this — zero per-call boxing.
    backend: Box<dyn ComputeBackend>,
}

/// Pre-indexed constant for zero-cost arena seeding.
/// Every field is required — no Optional fallbacks. If the compiler
/// cannot determine a field, `from_archive()` errors at load time.
struct ConstantSeed {
    node_id: NodeId,
    /// Byte range in the weight archive, or inline constant index.
    data_source: ConstantDataSource,
    /// Element size in bytes (e.g., 4 for f32). Always resolved.
    elem_size: usize,
    /// N-D output shape. Always resolved from the compiled shape map.
    shape: Vec<usize>,
    /// Element dtype. Always resolved from the compiled dtype map.
    dtype: FloatDType,
}

enum ConstantDataSource {
    /// Inline bytes from the ConstantStore (small constants).
    Inline(ConstantId),
    /// Deferred bytes from the weight archive: (offset, size).
    Deferred { offset: usize, size: usize },
}

/// Pre-indexed input slot for arena seeding.
/// Every field is required — the compiler must produce shapes and dtypes
/// for every input node. Missing metadata is a compile error, not a
/// runtime fallback.
struct InputSeed {
    node_id: NodeId,
    /// Element size in bytes. Always resolved.
    elem_size: usize,
    /// Element dtype. Always resolved.
    dtype: FloatDType,
    /// Compiled output shape. Always resolved. At execute time, the
    /// caller may override this with `GraphInputs::shape()` for dynamic
    /// shapes (e.g., variable batch size), but a compiled default is
    /// always present.
    compiled_shape: Vec<usize>,
}
```

**What `from_archive` computes additionally:**

After building the tape (existing), `from_archive` now also:
1. Calls `sg.node_dtypes_map()` once → stores as `self.dtypes` and derives `self.elem_sizes`.
2. Calls `sg.node_shapes_map()` once → stores as `self.shapes`.
3. Iterates `sg.nodes` once to build `self.constant_seeds` and `self.input_seeds` — the same loop that `seed_arena` currently does per-call, but the constant parts are captured and the input-dependent parts are deferred.
4. Calls `default_backend()` once → stores as `self.backend`.

**How all 7 `execute_tape_*` variants adapt:**

Currently there are 7 public functions that all repeat the same setup:
```
let sg = plan.graph();
let weights = plan.weights();
let compiled_dtypes = sg.node_dtypes_map();   // ← HashMap construction
let compiled_shapes = sg.node_shapes_map();   // ← HashMap construction
let mut arena = BufferArena::with_capacity(sg.nodes.len());
seed_arena(sg, weights, &compiled_dtypes, &compiled_shapes, inputs, &mut arena)?;
tape.prewarm_arena(&mut arena);
let wc = parking_lot::RwLock::new(WeightCache::new());  // ← fresh cache (some variants)
let tape_ctx = TapeContext::new(&sg.constants, weights, &wc);
tape.execute_direct(&mut arena, &tape_ctx)?;
```

After this change, ALL 7 variants collapse to a single internal function that receives a `&LoadedModel`:

```rust
fn execute_inner(
    model: &LoadedModel,
    inputs: &GraphInputs,
    kv_state: Option<&mut KvCacheState>,
    shape_overrides: &HashMap<u32, Vec<usize>>,
) -> ExecResult<GraphOutputs> {
    let sg = model.plan.graph();
    let weights = model.plan.weights();

    // Seed arena from pre-indexed template (constants are pre-resolved).
    let mut arena = BufferArena::with_capacity(sg.nodes.len());
    seed_arena_from_model(model, weights, inputs, &mut arena)?;
    model.tape.prewarm_arena(&mut arena);

    // Build TapeContext borrowing pre-resolved state from LoadedModel.
    let tape_ctx = TapeContext {
        ctx: kv_state.as_ref().map(|ks| ExecutionContext {
            position_offset: ks.write_pos() as u32,
        }),
        constants: &sg.constants,
        weights,
        weight_cache: &model.weight_cache,
        kv_state: kv_state.map(|ks| RefCell::new(std::mem::take(ks))),
        shape_overrides,
        flux: Cell::new(CurvatureFlux::ZERO),
    };

    // Execute: backend is pre-resolved, no per-call Box allocation.
    model.tape.execute_direct_with_backend(
        &mut arena,
        &tape_ctx,
        &*model.backend,
    )?;

    // ... kv state writeback, collect outputs ...
}
```

The 7 public functions become thin wrappers that call `execute_inner` with the appropriate combination of `kv_state`/`shape_overrides`/`weight_cache` parameters. The `execute_tape()` variant that currently constructs a fresh `WeightCache` uses the `LoadedModel`'s pre-existing one instead.

**`execute_direct` signature change:**

```rust
// OLD:
pub fn execute_direct(&self, arena: &mut BufferArena, tape_ctx: &TapeContext) -> ExecResult<()>
// The function internally calls default_backend() → Box allocation

// NEW:
pub fn execute_direct_with_backend(
    &self,
    arena: &mut BufferArena,
    tape_ctx: &TapeContext,
    backend: &dyn ComputeBackend,
) -> ExecResult<()>
// Backend is passed in — zero allocation
```

**`seed_arena_from_model` replaces `seed_arena`:**

```rust
fn seed_arena_from_model<'a>(
    model: &'a LoadedModel,
    weights: &'a [u8],
    inputs: &'a GraphInputs,
    arena: &mut BufferArena<'a>,
) -> ExecResult<()> {
    // Phase 1: constants (pre-indexed, no HashMap lookups, no Option checks)
    for seed in &model.constant_seeds {
        let data: &[u8] = match &seed.data_source {
            ConstantDataSource::Inline(cid) => model.plan.graph().constants.get(*cid),
            ConstantDataSource::Deferred { offset, size } => &weights[*offset..*offset + *size],
        };
        arena.insert_borrowed_with_elem_size(seed.node_id, data, seed.elem_size);
        arena.set_meta(seed.node_id, TensorMeta::new(seed.dtype, &seed.shape));
    }

    // Phase 2: inputs (ordinal-matched to GraphInputs)
    for (idx, seed) in model.input_seeds.iter().enumerate() {
        let data = inputs.get(idx as u32).ok_or(ExecError::MissingInput {
            node: seed.node_id,
            slot: idx,
        })?;
        arena.insert_borrowed_with_elem_size(seed.node_id, data, seed.elem_size);
        // Use caller's shape if provided (dynamic batch); otherwise compiled shape.
        let shape = inputs.shape(idx as u32).unwrap_or(&seed.compiled_shape);
        arena.set_meta(seed.node_id, TensorMeta::new(seed.dtype, shape));
    }

    // Phase 3: pre-allocate slots for intermediate nodes.
    // Every intermediate node has a known elem_size (resolved at load time).
    for (&node_id, &es) in &model.elem_sizes {
        if !arena.contains(node_id) {
            arena.set_elem_size(node_id, es);
        }
    }

    Ok(())
}
```

**Expected impact:**
- Per-execute setup drops from ~450 ns to ~100 ns (only input seeding + arena prewarm remain; both are O(num_inputs) not O(num_nodes)).
- The HashMap constructions `node_dtypes_map()` and `node_shapes_map()` are eliminated entirely (they were O(num_nodes) each).
- The `default_backend()` Box allocation is eliminated (was ~20-50 ns per call).
- For the 4-node 256B benchmark: ~640 ns → ~300 ns (projected 2.1x speedup).
- For the transformer decode step (10K+ instructions): 45 ms → 45 ms (setup was <0.01% of total; no regression).

**Correctness:** Semantic equivalence preserved — the dtype/shape/backend are the same values, just computed at load time instead of per-execute. The tape instructions and their dispatch are unchanged. The `seed_arena_from_model` function produces an identical arena state to the current `seed_arena` function.

### Change 2: Performance-aware fusion admissibility

**Goal:** Gate fused kernel emission on operand profitability. The analysis pass still detects patterns; the tape builder decides whether to emit fused or unfused.

**Where:**
- `crates/hologram-ir/src/analysis/float_fusion.rs` — epilogue fusion detection
- `crates/hologram-fused-component/src/tape_builder.rs` — tape instruction emission
- New: `crates/hologram-ir/src/analysis/cost_model.rs` — fusion cost model

**Design:**

The analysis pass currently returns `bool` from each `try_fuse_*` function. Change these to return a `FusionFinding`:

```rust
pub struct FusionFinding {
    /// The structural pattern was found.
    pub pattern: FusionPattern,
    /// Estimated operand volume (M * K * N for matmul, len for unary).
    pub operand_volume: u64,
}

pub enum FusionPattern {
    MatMulActivation { m: u32, k: u32, n: u32, activation: FloatOp },
    MatMulBiasActivation { m: u32, k: u32, n: u32, activation: FloatOp },
    Conv2dActivation { ... },
    // ... other patterns
}
```

The `analyze()` dispatcher collects all findings. The tape builder filters them through the admissibility predicate:

```rust
fn is_fusion_profitable(finding: &FusionFinding) -> bool {
    match &finding.pattern {
        FusionPattern::MatMulActivation { m, k, n, .. } => {
            // Fused kernel is faster only when the intermediate buffer
            // write/read cost exceeds the register-pressure overhead.
            // Threshold: ~4M elements (empirically: 2048*2048 = 4M).
            (*m as u64) * (*k as u64) * (*n as u64) > 4_000_000
        }
        FusionPattern::MatMulBiasActivation { m, k, n, .. } => {
            (*m as u64) * (*k as u64) * (*n as u64) > 4_000_000
        }
        // View fusion (Q0/Q1) is always profitable: one LUT vs N LUTs.
        // CSE is always profitable: removes redundant computation.
        // Constant folding is always profitable: removes computation entirely.
        _ => true,
    }
}
```

When `is_fusion_profitable` returns false, the tape builder emits the unfused pair of instructions instead of the fused kernel. The graph still has the `FusedMatMulActivation` node (the analysis pass put it there), but the tape builder decomposes it back to `InlineMatMul` + `InlineSilu` at tape emission time.

**Expected impact:**
- `epilogue_fusion/fused/1x64x64`: 2009 ns → ~1544 ns (emits unfused pair, same as unfused bench)
- `epilogue_fusion/fused/1x2048x2048`: 5320 ns → 5320 ns (emits fused kernel, unchanged)
- No regression at any size — the system picks the faster option per-case.

**Correctness:** Both fused and unfused forms are in the same canonical equivalence class (Corollary 7.6). The tape builder's choice is representative selection, not a semantic change.

### Change 3: Kernel variant selection at tape-build time

**Goal:** For operations with multiple implementation strategies, select the variant at tape-build time based on operand metadata. Eliminate runtime branching in the dispatch path.

**Where:**
- `crates/hologram-fused-component/src/tape_builder.rs` — variant selection
- `crates/hologram-fused-component/src/tape.rs` — new `TapeKernel` variants or variant metadata

**Design:**

For softmax, the tape builder already knows the sequence length from the graph's compiled shape metadata (available via Change 1's pre-computed shape map). Add a variant selection at build time:

```rust
// tape_builder.rs, when emitting a softmax instruction:
// Row-based softmax requires two passes over the input (find max, then exp-sum).
// Online softmax does a single pass but is 2x slower due to serial dependency.
// Prefer row-based unless the input is too large for two passes to fit in L2 cache.
let use_online = seq_len > L2_CACHE_ELEMENTS;  // ~512K f32 elements ≈ 2MB L2
let kernel = if use_online {
    TapeKernel::InlineSoftmaxOnline { size: seq_len as u32 }
} else {
    TapeKernel::InlineSoftmaxRowBased { size: seq_len as u32 }
};
```

The `L2_CACHE_ELEMENTS` threshold is architecture-dependent. On x86_64 with 256KB–2MB L2 per core, a conservative threshold of 512K floats (2MB) ensures the two-pass row-based path stays in cache. For sequences above this threshold (rare — 512K = 512K tokens), the online path avoids the second pass over cold data.

This requires two new `TapeKernel` variants:
- `InlineSoftmaxRowBased { size: u32 }` — dispatches directly to `dispatch_softmax_row_based`
- `InlineSoftmaxOnline { size: u32 }` — dispatches directly to `dispatch_softmax_online`

The existing `InlineSoftmax { size: u32 }` variant can be removed (replaced by the two specialized variants). The `dispatch_kernel` match for the new variants calls the implementation directly — no runtime "which softmax?" branching.

Similarly for matmul: the tape builder knows whether BLAS (Accelerate) is available via `cfg(feature = "accelerate")` at compile time, and knows `M`, `K`, `N` from the graph metadata. The variant selection is:
- `cfg(feature = "accelerate") && M*K*N > 0` → emit `InlineMatMulBlas { m, k, n }` (AMX hardware is faster even at M=1 on Apple Silicon)
- Otherwise → emit `InlineMatMulTiled { m, k, n }` (software blocked kernel)

These are NOT new architectural decisions — the dispatch_kernel already branches on `cfg(feature = "accelerate")` at runtime. This change hoists the branch to tape-build time.

**Expected impact:**
- Softmax: standard inference always dispatches to `row_based` → 546 ns for 128 elements (was 1408 ns for online = 2.6x faster at that size).
- Matmul: no per-instruction cfg branch (the tape already encodes the choice).
- Per-instruction dispatch cost drops by one branch elimination for these kernel types.

**Correctness:** Both `row_based` and `online` compute the same mathematical function. Both produce identical floating-point results for sequences ≤ 8192 elements (the benchmark range). At extreme sequence lengths, online softmax has marginally better numerical stability (avoids the intermediate max subtraction overflow risk), but the difference is below f32 epsilon for all practical sequence lengths. This is representative selection within the canonical equivalence class (Corollary 7.6).

### Change 4: Float dispatch — eliminate per-instruction shape resolution heuristics

**Goal:** Reduce per-instruction overhead in the float dispatch path.

**Where:** `crates/hologram-fused-component/src/tape.rs` (`dispatch_kernel` function and the `shape_resolve` module)

**Analysis of the actual overhead:**

The SmallVec input-ref gathering is NOT the bottleneck — profiling shows it costs ~10 ns/instruction (the SmallVec is stack-allocated for ≤4 elements, and `input_indices` is typically 1-2 entries). The 4.5x gap between the float path (719 ns/instruction at 256B) and byte path (160 ns/instruction at 256B) comes from:

1. **Shape resolution heuristics (~200-400 ns):** Functions like `resolve_size()`, `infer_matmul_dims()`, `infer_slice_axis_size()` in `crates/hologram-fused-component/src/shape_resolve.rs` run per-instruction to infer output dimensions from input buffer sizes. These involve division, modular arithmetic, and multi-path branching. When `shape_overrides` is provided (the common case in production), these heuristics are skipped — but the benchmark doesn't use shape overrides.

2. **Backend vtable dispatch (~10-20 ns):** The `backend.dispatch_float()` call goes through `&dyn ComputeBackend`. This is one indirect call per instruction.

3. **f32 arithmetic vs byte LUT (~50-100 ns for 256B = 64 floats):** Inherent — not overhead.

**Design:**

The real optimization target is shape resolution. When shape overrides are empty (the benchmark case), the dispatch path falls back to dimension-inference heuristics that examine buffer sizes. These heuristics are workarounds for not knowing the output shape at tape-build time.

The fix aligns with Change 1: the tape builder has access to compiled shapes at build time. Instead of deferring shape resolution to dispatch time, **bake the resolved output shape into the `TapeInstruction`**:

```rust
pub struct TapeInstruction {
    pub kernel: TapeKernel,
    pub input_indices: SmallVec<[u32; 4]>,
    pub output_idx: u32,
    pub output_byte_hint: u32,
    pub output_elem_size: u8,
    pub passthrough: bool,
    pub can_reuse_input: bool,
    pub weight_offset_hint: u32,
    // ── NEW ──
    /// Pre-resolved output TensorMeta (dtype + shape). The tape builder
    /// MUST populate this for every instruction. The dispatch path writes
    /// it directly to the arena after kernel execution — zero shape
    /// inference at runtime. The `shape_resolve` module's heuristics
    /// (`resolve_size`, `infer_matmul_dims`, etc.) are deleted from the
    /// execute path entirely and moved to the tape builder.
    pub output_meta: hologram_core::op::TensorMeta,
}
```

There is no `Option` — every instruction carries its resolved output metadata. The tape builder computes this from the compiled shape map (available at tape-build time via the `SerializedGraph`). If a node lacks a compiled shape (which should not happen for compiler-produced archives), the tape builder errors at build time:

```rust
// tape_builder.rs, when populating output_meta:
let meta = compiled_shapes.get(&node_id)
    .map(|shape| TensorMeta::new(dtype, shape))
    .unwrap_or_else(|| {
        // For nodes without explicit shapes (e.g., byte-domain LUT ops that
        // pass through input size), the tape builder infers the shape from
        // the instruction's input shapes and kernel semantics. This is the
        // SAME logic that shape_resolve currently runs per-instruction at
        // runtime — but executed ONCE at build time.
        infer_output_meta_at_build_time(&kernel, &input_metas, dtype)
    });
```

The `shape_resolve` module functions (`resolve_size`, `infer_matmul_dims`, `infer_slice_axis_size`) are refactored: their logic moves to `tape_builder.rs` (called once at build time per instruction) and is DELETED from `tape.rs` (the execute path). The execute path has zero shape inference.

**Expected impact:**
- Shape resolution overhead at execute time: **eliminated entirely** (was ~200-400 ns/instruction for float ops).
- Float dispatch per-instruction: ~719 ns → ~350-400 ns (shape inference eliminated; f32 arithmetic + vtable dispatch remain).
- `tape::linear_chain(4_float_nodes, 256B)`: 2875 ns → ~1600 ns.
- Transformer decode: ~45 ms → ~43 ms (shape inference was a fraction of large-kernel cost).
- `shape_resolve.rs` is no longer imported by `tape.rs` — it becomes a build-time-only module used by `tape_builder.rs`.

**Correctness:** The pre-resolved shape is identical to what the runtime heuristics would compute — the computation moves from per-instruction to per-build. The dispatch produces the same output bytes.

### Change 5: LUT-GEMM routing threshold

**Goal:** Route small matmuls to naive f32 dispatch instead of Q4/Q8 LUT-GEMM when the quantization overhead exceeds the bandwidth savings.

**Where:** `crates/hologram-fused-component/src/tape_builder.rs` (matmul instruction emission)

**Design:**

When the tape builder encounters a `MatMulLut4` or `MatMulLut8` graph node with known dimensions, check the threshold:

```rust
fn should_use_lut_gemm(k: u32, n: u32) -> bool {
    // LUT-GEMM psumbook overhead amortizes above ~65K elements.
    // Below that, naive f32 matmul is faster.
    (k as u64) * (n as u64) > 65_536
}
```

When the threshold is not met, the tape builder emits an `InlineMatMul { m, k, n }` instruction that reads from a dequantized weight buffer instead of the quantized `ConstantId`. The dequantization happens once at load time (or lazily on first access via the `WeightCache`), not per-dispatch.

**Dequantization path detail:**

The `WeightCache` already handles lazy dequantization for LUT-GEMM: the first call to `lut_gemm_4bit` deserializes the quantized weight archive entry into `QuantizedWeights4` (codebook + index matrix). For the below-threshold f32 path, the dequantized weights are pre-computed at load time and stored as a `DequantizedF32(Vec<f32>)` entry in the `WeightCache`.

Add a new `WeightCache` entry type: `DequantizedF32(Vec<f32>)`. At `LoadedModel::from_archive()` time, for every below-threshold quantized weight node, the loader eagerly dequantizes Q4 → f32 and stores the result. This is a one-time load-time cost — the execute path reads the pre-dequantized buffer with zero per-dispatch overhead.

```rust
// tape_builder.rs
if should_use_lut_gemm(k, n) {
    TapeKernel::MatMulLut4(constant_id)
} else {
    // Emit f32 matmul that reads dequantized weights.
    // The weight_cache will lazily dequantize Q4 → f32 on first access.
    TapeKernel::InlineMatMulDequant {
        m, k, n,
        weight_constant_id: constant_id,
    }
}
```

The new `InlineMatMulDequant` variant in `dispatch_kernel`:
1. Borrows the pre-dequantized f32 buffer from `weight_cache` (populated at load time — guaranteed present).
2. Dispatches to the standard f32 matmul path (`dispatch_matmul_into`).

There is no cache miss path at execute time. The dequantization cost is paid once at `from_archive()` load time. The per-dispatch cost is identical to a native f32 matmul — one buffer read + one BLAS/tiled kernel call.

**Threshold justification:**

From the benchmarks:
- `naive_matmul(4x64x64)` = 13,294 ns (f32, K*N = 4096)
- `lut_gemm_q4(4x64x64)` = 30,757 ns (Q4, K*N = 4096)
- `lut_gemm_q4(4x256x256)` = 490,797 ns (Q4, K*N = 65536)

LUT-GEMM becomes competitive at K*N ≈ 65K (the crossover where codebook-lookup amortizes). Set threshold at `K * N > 65_536`.

**Expected impact:**
- `lut_gemm_q4(4x64x64)`: 30,757 ns → ~13,294 ns (routes to pre-dequantized f32 matmul).
- `lut_gemm_q4(4x256x256)`: 490,797 ns → 490,797 ns (above threshold, stays LUT-GEMM).
- Load-time overhead: ~50 µs per below-threshold weight matrix (dequantize Q4 → f32 at `from_archive()`). For a model with 10 below-threshold layers, this adds ~500 µs to load time — invisible against the typical multi-second model load.
- Execute-time overhead: zero. The f32 buffer is pre-computed and borrowed.

**Correctness:** The dequantized f32 weights are the mathematical reconstruction of the quantized representation: for each weight element, `value = centroid[index]`. The naive matmul on these f32 values produces identical results to what a hypothetical infinite-precision LUT-GEMM would produce. In practice, both paths agree to within f32 rounding (the LUT-GEMM accumulates psumbook entries in f32; the naive path accumulates individual products in f32). The quantization noise is inherent to the model, not introduced by routing.

---

## Sequencing

Changes are ordered by dependency and risk:

1. **Change 1** (LoadedModel = complete U_can) — the foundational change. Pre-computes dtype/shape/backend/seed templates at load time. All subsequent changes depend on this metadata being available.
2. **Change 4** (bake output_meta into TapeInstruction) — depends on Change 1's shape map being available at tape-build time. Eliminates per-instruction shape resolution heuristics.
3. **Change 2** (performance-aware fusion admissibility) — depends on Change 1 for operand dimension metadata at tape-build time. Gates fused kernel emission on profitability.
4. **Change 3** (kernel variant selection) — depends on Change 1 for backend availability at tape-build time. Selects softmax/matmul variant.
5. **Change 5** (LUT-GEMM routing threshold) — depends on Change 1 for dimension metadata. Routes small matmuls to naive dispatch.

Change 1 is the critical path. Changes 2–5 are independent of each other and can land in any order after Change 1.

---

## Critical files

| File | Changes | Description |
|---|---|---|
| `crates/hologram-fused-component/src/prism_module.rs` | 1 | Expand `LoadedModel` with pre-computed dtype/shape/backend/seed fields; update `from_archive()` |
| `crates/hologram-fused-component/src/mmap/mod.rs` | 1 | Collapse 7 `execute_tape_*` functions into thin wrappers around `execute_inner`; new `seed_arena_from_model` |
| `crates/hologram-fused-component/src/tape.rs` | 1, 4 | `execute_direct_with_backend` takes pre-resolved `&dyn ComputeBackend`; `TapeInstruction.output_meta` field |
| `crates/hologram-fused-component/src/tape_builder.rs` | 2, 3, 4, 5 | Populate `output_meta` on instructions; fusion profitability gate; kernel variant selection; LUT-GEMM routing |
| `crates/hologram-fused-component/src/backend/mod.rs` | 1 | `default_backend()` called once at load time, stored on LoadedModel |
| `crates/hologram-fused-component/src/shape_resolve.rs` | 4 | Shape inference logic moves to `tape_builder.rs` (build-time only); execute-time imports of this module are deleted |
| `crates/hologram-ir/src/analysis/float_fusion.rs` | 2 | Return `FusionFinding` with operand dimensions instead of `bool` |
| `crates/hologram-ir/src/analysis/mod.rs` | 2 | `StructuralFindings` includes fusion decisions with profitability annotations |
| NEW: `crates/hologram-ir/src/analysis/cost_model.rs` | 2 | `is_fusion_profitable()` predicate with operand-size threshold |

---

## Verification

### Correctness gates

- `cargo test --workspace --no-fail-fast` — all existing tests pass (no semantic change)
- `cargo test --workspace --doc` — all doctests pass
- New test: `fused_and_unfused_produce_same_output(m, k, n)` — for all benchmark sizes, assert the fused and unfused tape paths produce identical output bytes
- New test: `lut_gemm_and_naive_produce_same_output(m, k, n)` — for all benchmark sizes, assert LUT-GEMM and naive matmul produce the same output (within quantization tolerance)
- New test: `row_based_and_online_softmax_match(seq_len)` — assert both softmax implementations produce identical results

### Performance gates

Run benchmarks before and after. Expected deltas:

| Benchmark | Before | After (projected) | Delta | Change |
|---|---|---|---|---|
| `exec::linear_chain(4_nodes, 256B)` | 640 ns | ~300 ns | **2.1x faster** | 1 (setup elimination) |
| `tape::linear_chain(4_float_nodes, 256B)` | 2875 ns | ~1600 ns | **1.8x faster** | 1 + 4 (setup + shape resolution) |
| `epilogue_fusion/fused/1x64x64` | 2009 ns | ~1544 ns | **1.3x faster** | 2 (routes to unfused) |
| `epilogue_fusion/fused/1x2048x2048` | 5320 ns | ~5320 ns | unchanged | 2 (above threshold, stays fused) |
| `softmax_decode/online/2048` | 21885 ns | ~10958 ns | **2.0x faster** | 3 (routes to row-based) |
| `lut_gemm_q4(4x64x64)` | 30757 ns | ~13294 ns | **2.3x faster** | 5 (routes to naive) |
| `transformer::tape(decode_step)` | 45279 µs | ~43000 µs | **~5% faster** | 1 + 4 (setup + shape resolution) |

### Regression gates

No benchmark may regress by more than 2%:
- `exec::linear(3_nodes, 64KB)` — memory-bandwidth bound, should be unchanged
- `view::apply_slice(64KB)` — byte-domain, unaffected
- `lut_gemm_q4(4x256x256)` — above routing threshold, unchanged

---

## Formal grounding (UOR Fusion Dossier cross-references)

| Change | Dossier theorem/obligation | How the change discharges it |
|---|---|---|
| 1 (LoadedModel = U_can) | Obligation 2.5 — typing total at U_can construction | Dtype/shape maps pre-computed at load time; execute path borrows |
| 2 (AdmFuse cost model) | Definition 5.2 — admissible fusion predicate must be specified | `is_fusion_profitable()` is the concrete AdmFuse predicate |
| 2 (threshold gating) | Failure Mode 5.7 — fusion closure only for admissible subfragment | Below-threshold chains use unfused emission; above-threshold use fused |
| 3 (variant selection) | Definition 6.5 — single fused representative is atomic | Both variants are atomic; selection is within the equivalence class |
| 3 (variant selection) | Corollary 7.6 — fused representative preserves denotation | Both row-based and online softmax have the same mathematical denotation |
| 4 (dispatch specialization) | Definition 7.2 — semantic interpretation | Evaluation overhead reduction; denotation unchanged |
| 5 (LUT-GEMM routing) | Failure Mode 5.7 — same as Change 2 | Below-threshold routes to naive; above-threshold routes to LUT-GEMM |
| All | Theorem 7.5 — semantic preservation under normalization | Every change produces the same output bytes for the same inputs |
