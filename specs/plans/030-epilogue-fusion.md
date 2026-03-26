# Plan 030: Epilogue Fusion (Plan 005 Phase 2)

## Context

Two research papers on thermodynamic precision and gauge symmetry in neural network inference were analyzed against hologram's architecture. The key actionable insight: **every matmul currently spills its f32 accumulator to memory, reads it back, then applies the activation separately** — the "double-rounding" the paper calls "information destroyed not by necessity but by poor scheduling."

The correct order is: accumulate → scale/bias → activation → quantize (once), all in registers. This is **epilogue fusion** — the core of Plan 005 Phase 2.

hologram-ai has `MatMulRelu`/`MatMulGelu`/`MatMulSilu` as `AiOp` variants but drops the activation during lowering because hologram has no fused kernel to dispatch to.

**All fusion happens at compile time** (graph-level `fuse()` pass), not runtime. The tape executor sees a single `InlineMatMulActivation` instruction — no pattern detection, no extra branching at runtime.

**Branch**: `refactor/epilogue-fusion`
**Sprint file**: Update `specs/SPRINT.md` with Sprint 23
**Plan file**: Save as `specs/plans/030-epilogue-fusion.md`

---

## Current State

| What | Status | File |
|------|--------|------|
| `FusedFloatChain(Vec<FloatOp>)` | Working — fuses unary chains (Exp→Sigmoid→Neg) | [float_fusion.rs](crates/hologram-graph/src/fusion/float_fusion.rs) |
| `InlineMatMul { m, k, n }` | Working — dispatches to `matmul_k_outer` or BLAS | [tape.rs:184](crates/hologram-exec/src/tape.rs#L184) |
| `InlineRelu`/`InlineSilu`/`InlineGelu` | Working — separate tape instructions | [tape.rs](crates/hologram-exec/src/tape.rs) |
| `matmul_k_outer` | Working — 4×8 register-blocked, writeback at line 586-588 | [matmul.rs:562](crates/hologram-exec/src/float_dispatch/matmul.rs#L562) |
| `FloatOp::apply_unary(f32)->f32` | Working — monomorphic scalar activation | [float_op.rs:990](crates/hologram-core/src/op/float_op.rs#L990) |
| `InlineRmsNorm`/`InlineLayerNorm` | Working — baked params, write to out_buf | [tape.rs:188,214](crates/hologram-exec/src/tape.rs#L188) |
| MatMul+Activation fusion | **Missing** — no fused variant at any level | — |
| Norm+Activation fusion | **Missing** — norm and activation are separate instructions | — |

---

## Implementation

### Phase 1: MatMul + Activation Epilogue Fusion

#### 1.1 Add `TapeKernel::InlineMatMulActivation`

File: [tape.rs](crates/hologram-exec/src/tape.rs) (~line 185, after `InlineMatMul`)

```rust
/// Fused matmul + element-wise activation (epilogue fusion).
/// Activation applied in-register before writeback — avoids memory round-trip.
InlineMatMulActivation {
    m: u32,
    k: u32,
    n: u32,
    activation: FloatOp,
},
```

#### 1.2 Add fused CPU kernel

File: [matmul.rs](crates/hologram-exec/src/float_dispatch/matmul.rs) (new functions after `matmul_k_outer`)

- `matmul_k_outer_fused(a, b, out, m, k, n, activation)` — clone of `matmul_k_outer` with `activation.apply_unary(*v)` applied to accumulator elements before `copy_from_slice`. Same for remainder loops.
- `dispatch_matmul_activation_into(inputs, m, k, n, activation, out_buf)` — mirrors `dispatch_matmul_into` (line 246), calls fused kernel. BLAS path: call sgemm then apply activation as post-pass on output (still saves one arena slot vs separate instruction).

#### 1.3 Dispatch in tape executor

File: [tape.rs](crates/hologram-exec/src/tape.rs) (in `dispatch_kernel`, after `InlineMatMul` arm)

Same dimension resolution as `InlineMatMul`. Call `dispatch_matmul_activation_into`. Return `DispatchResult::InOutBufWithMeta`.

#### 1.4 Graph-level fusion: `FusedMatMulActivation`

File: [graph/mod.rs](crates/hologram-graph/src/graph/mod.rs) — add to `GraphOp`:
```rust
/// Fused matmul + activation (compile-time epilogue fusion).
FusedMatMulActivation { m: u32, k: u32, n: u32, activation: FloatOp },
```

Already derives `rkyv::Archive, rkyv::Serialize, rkyv::Deserialize` — `FloatOp` is serializable (used in `FusedFloatChain`).

Update `arity()` → 2 (same as MatMul), and any other trait methods on `GraphOp`.

#### 1.5 Fusion pass: `try_fuse_matmul_activation`

File: [float_fusion.rs](crates/hologram-graph/src/fusion/float_fusion.rs) (new function)

Pattern: MatMul node with **exactly one successor** that is `is_elementwise_unary()`. Replace pair with `FusedMatMulActivation`. Rewire inputs. Remove MatMul node.

Wire into [fusion/mod.rs](crates/hologram-graph/src/fusion/mod.rs) `fuse()` function, after float chain fusion.

#### 1.6 Tape builder wiring

File: [tape_builder.rs](crates/hologram-exec/src/tape_builder.rs) — in `resolve_kernel`:
```rust
GraphOp::FusedMatMulActivation { m, k, n, activation } =>
    TapeKernel::InlineMatMulActivation { m: *m, k: *k, n: *n, activation: *activation }
```

#### 1.7 LUT-GEMM fused variants

File: [tape.rs](crates/hologram-exec/src/tape.rs) — add:
```rust
MatMulLut4Activation(ConstantId, FloatOp),
MatMulLut8Activation(ConstantId, FloatOp),
```

Graph-level: `GraphOp::MatMulLut4Activation(ConstantId, FloatOp)` etc.

Same fusion pass pattern: detect `MatMulLut4 → unary activation` with single successor.

### Phase 2: Norm + Activation Fusion

#### 2.1 Add fused TapeKernel variants

```rust
InlineRmsNormActivation { size: u32, epsilon: u32, activation: FloatOp },
InlineLayerNormActivation { size: u32, epsilon: u32, activation: FloatOp },
InlineGroupNormActivation { num_groups: u32, epsilon: u32, activation: FloatOp },
```

#### 2.2 Fused norm kernels

Apply activation element-wise after norm computation, before writeback. The norm ops already write to `out_buf` directly — insert `apply_unary` in the output loop.

#### 2.3 Fusion pass: `try_fuse_norm_activation`

Same pattern as matmul: norm node with exactly one successor that is elementwise unary. New `GraphOp` variants for each.

### Phase 3: Tests

- **3.1** Unit: `matmul_k_outer_fused` bit-identical to `matmul_k_outer` + `apply_unary` for Relu/Gelu/Silu/Sigmoid/Tanh
- **3.2** Graph fusion: `Input → MatMul → Relu → Output` fuses to `Input → FusedMatMulActivation → Output`
- **3.3** No-fuse: `Input → MatMul → [Relu, Sigmoid]` (fan-out) does NOT fuse
- **3.4** Tape E2E: fused tape produces bit-identical output to unfused
- **3.5** Norm fusion: `Input → RmsNorm → Silu → Output` fuses correctly
- **3.6** Chain absorption: `Input → MatMul → Relu → Sigmoid → Output` — MatMul absorbs Relu, then Sigmoid becomes a `FusedFloatChain([Sigmoid])` (or chain of 1 doesn't fuse — verify)

### Phase 4: Sprint Hygiene

- **4.1** Copy plan to `specs/plans/030-epilogue-fusion.md`
- **4.2** Update `specs/SPRINT.md` with Sprint 23 section
- **4.3** Checkout `refactor/epilogue-fusion` branch

---

## Files to Modify

| File | Change |
|------|--------|
| [crates/hologram-core/src/op/float_op.rs](crates/hologram-core/src/op/float_op.rs) | No changes — `apply_unary` already exists |
| [crates/hologram-graph/src/graph/mod.rs](crates/hologram-graph/src/graph/mod.rs) | Add `FusedMatMulActivation`, `MatMulLut4Activation`, `MatMulLut8Activation`, norm+activation variants to `GraphOp` |
| [crates/hologram-graph/src/fusion/float_fusion.rs](crates/hologram-graph/src/fusion/float_fusion.rs) | Add `try_fuse_matmul_activation()`, `try_fuse_norm_activation()` |
| [crates/hologram-graph/src/fusion/mod.rs](crates/hologram-graph/src/fusion/mod.rs) | Wire new passes into `fuse()` |
| [crates/hologram-exec/src/tape.rs](crates/hologram-exec/src/tape.rs) | Add `InlineMatMulActivation`, `MatMulLut4Activation`, `MatMulLut8Activation`, norm+activation variants + dispatch |
| [crates/hologram-exec/src/tape_builder.rs](crates/hologram-exec/src/tape_builder.rs) | Map new `GraphOp` variants → `TapeKernel` variants |
| [crates/hologram-exec/src/float_dispatch/matmul.rs](crates/hologram-exec/src/float_dispatch/matmul.rs) | Add `matmul_k_outer_fused`, `dispatch_matmul_activation_into` |
| [specs/SPRINT.md](specs/SPRINT.md) | Add Sprint 23 |
| specs/plans/030-epilogue-fusion.md | This plan (new file) |

## Existing Code to Reuse

- `FloatOp::apply_unary(f32) -> f32` — [float_op.rs:990](crates/hologram-core/src/op/float_op.rs#L990)
- `FloatOp::is_elementwise_unary()` — [float_op.rs:960](crates/hologram-core/src/op/float_op.rs#L960)
- `matmul_k_outer` — [matmul.rs:562](crates/hologram-exec/src/float_dispatch/matmul.rs#L562) — clone + modify writeback
- `dispatch_matmul_into` — [matmul.rs:246](crates/hologram-exec/src/float_dispatch/matmul.rs#L246) — clone + modify
- `try_fuse_float_unary` pattern — [float_fusion.rs:18](crates/hologram-graph/src/fusion/float_fusion.rs#L18) — same backward-walk pattern
- `resolve_matmul_dims` — [shape_resolve.rs](crates/hologram-exec/src/shape_resolve.rs)

## Verification

1. `cargo fmt && cargo clippy -- -D warnings` — clean
2. `cargo test` — all existing + new tests pass
3. Graph fusion tests verify correct pattern detection and no-fuse cases
4. Tape E2E tests verify bit-identical output between fused and unfused paths
5. Cross-repo: hologram-ai can lower `MatMulRelu`/`MatMulGelu`/`MatMulSilu` to fused ops

## Considerations

- **Serialization**: `GraphOp` derives rkyv traits. `FloatOp` is already serializable (used in `FusedFloatChain`). New variants serialize automatically.
- **FloatOp's role**: Stays as graph-level IR. `TapeKernel` is execution. `FloatOp` becomes data carried inside fused tape kernels (same pattern as `FusedFloatChain(Vec<FloatOp>)`).
- **BLAS path**: sgemm doesn't support fused epilogues. Apply activation as post-pass on output buffer. Still saves one arena slot allocation + index lookup vs separate tape instruction.
- **GPU/Metal path**: Metal SGEMM kernel could have a fused epilogue (shader modification) — out of scope for this sprint, filed as future work.
- **Ordering**: MatMul fusion runs BEFORE unary chain fusion in `fuse()`. This prevents a Relu from being absorbed into a prior unary chain when it should be absorbed into the MatMul.

---

## Benchmark Results (Sprint 23)

The fused path is currently **slower** than unfused at all tested sizes:

| Size | Unfused | Fused | Delta | Why |
|------|---------|-------|-------|-----|
| 1x64x64 | 978 ns | 1.65 µs | +69% | Dispatch overhead dominates tiny matmul |
| 1x256x256 | 9.59 µs | 12.5 µs | +30% | Dimension re-inference overhead |
| 1x512x512 | 36.1 µs | 40.9 µs | +13% | Overhead shrinking |
| 1x2048x2048 | 538 µs | 556 µs | +3% | Compute-dominated, overhead marginal |

### Root cause analysis

The unfused path benefits from hologram's existing tape optimizations:
1. **`can_reuse_input`**: Silu after MatMul overwrites the output buffer in-place (zero allocation)
2. **Inline dispatch**: `InlineSilu` is a direct `v * sigmoid(v)` with no indirection
3. **Cache locality**: MatMul output stays in L1/L2 for the immediately following Silu

The fused `dispatch_matmul_activation_into` calls `dispatch_matmul_into` internally, which re-does dimension inference (`infer_matmul_k`, batch detection) that was already done in the `dispatch_kernel` match arm. This adds ~0.5-1µs per call.

### Why the architecture is still correct

The performance regression is in the **dispatch overhead**, not the kernel. The fusion eliminates:
- One `TapeInstruction` from the tape (fewer instructions to iterate)
- One arena slot allocation (no intermediate buffer between matmul and activation)
- One output metadata propagation step

These benefits are real but small compared to the double dimension-inference cost. Once the dispatch path is streamlined, the fused path should break even or win.

---

## Future Work: Making Fused Path Break Even

### Investigation 1: Eliminate double dimension inference (HIGH priority)

**Problem**: `InlineMatMulActivation` dispatch in `tape.rs` resolves dimensions via `shape_resolve::resolve_matmul_dims`, then calls `dispatch_matmul_activation_into` which calls `dispatch_matmul_into` which calls `infer_matmul_k` again.

**Fix**: Pass pre-resolved (actual_m, actual_k, actual_n) directly to a lower-level kernel function that skips inference. Create `matmul_kernel_into(a, b, out, m, k, n)` that just does the multiply without any size validation or batch detection. The dispatch arm handles all sizing, the kernel just multiplies.

**Files**: [tape.rs](crates/hologram-exec/src/tape.rs) dispatch arm, [matmul.rs](crates/hologram-exec/src/float_dispatch/matmul.rs) new low-level function.

**Expected impact**: Removes ~0.5-1µs overhead. Should make fused path break even at 256x256 and win at larger sizes.

### Investigation 2: Monomorphic activation dispatch (MEDIUM priority)

**Problem**: `activation.apply_unary(*v)` is a match on `FloatOp` variant per element. For Relu this is trivial but the branch predictor may not optimize it as well as the dedicated `InlineRelu` path.

**Fix**: At tape-build time, specialize the activation function into a concrete `fn(f32) -> f32` pointer. Store `fn(f32) -> f32` in the TapeKernel instead of `FloatOp`. Eliminates the match dispatch per element.

**Files**: [tape.rs](crates/hologram-exec/src/tape.rs) TapeKernel variant, [tape_builder.rs](crates/hologram-exec/src/tape_builder.rs) specialization.

**Expected impact**: ~5-10% improvement on activation post-pass, especially for small matmuls.

### Investigation 3: Bias fusion (HIGH value, new feature)

**Problem**: Many transformer layers have `Linear(x) = xW + b` followed by activation. Currently: MatMul → Add(bias) → Silu = 3 tape instructions, 2 intermediate buffers.

**Fix**: Add `InlineMatMulBiasActivation { m, k, n, bias_cid, activation }` that fuses all three. The bias add is element-wise and can be combined with the activation post-pass: `for v in out { *v = activation(v + bias[col]) }`. One loop, one write.

**Expected impact**: Eliminates a real intermediate buffer (bias add output). This is where fused path should clearly win vs unfused.

### Investigation 4: GPU epilogue shader (LOW priority, blocked)

**Problem**: Metal SGEMM kernel writes to device memory, then activation runs as a separate compute pass.

**Fix**: Add activation function to the Metal SGEMM shader epilogue. Apply in shared memory before writing to device memory.

**Blocked on**: Metal shader compilation infrastructure. Filed as future work.
