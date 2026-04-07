# Plan: Content-Addressed Computation & Layer Reduction

## Context

The hologram runtime has ~12 abstraction layers between `.holo` archives and actual computation. Meanwhile, the byte-domain (Q0) already implements exactly what we want: `ADD_Q0[x][y]` is a precomputed 256×256 table where the "address" of the result is determined by the "addresses" of the inputs. But float operations (`FloatOp::Add`, etc.) bypass this entirely and do native f32 arithmetic.

The goal is twofold:
1. **Reduce layers** — eliminate redundant dispatch and consolidate entry points
2. **Content-addressed compute** — route float operations through the byte-domain LUT path so computation becomes "look up the answer by its address" rather than "compute the answer"

The system already has all the primitives: `Encoding` (pi-F-lambda), `ElementWiseView` (composable 256-byte LUTs), `CurvatureFlux` (dynamic precision promotion), and Q0-Q3 ring levels. They're just disconnected from the float execution path.

---

## Phase 1: Layer Consolidation

**Goal**: 12 layers → 8. No semantic changes, just remove redundancy.

### 1a. Collapse float_dispatch entry points
- **File**: [float_dispatch/mod.rs](crates/hologram-exec/src/float_dispatch/mod.rs)
- Make `dispatch_float_into` the single primary entry point
- `dispatch_float` and `dispatch_float_ctx` become thin wrappers that allocate a Vec and delegate
- Fold broadcast logic from `dispatch_float_with_shapes` into `dispatch_float_into` with an optional `&[Vec<usize>]` param

### 1b. Gate KvStore dispatch behind `#[cfg(test)]`
- **File**: [kv/store.rs](crates/hologram-exec/src/kv/store.rs)
- `KvStore::dispatch_with_shapes` duplicates every routing decision that `dispatch_kernel` in tape.rs already handles
- Keep it as a test/debug path, not production execution

### 1c. Consolidate TapeKernel inline variants
- **File**: [tape.rs](crates/hologram-exec/src/tape.rs)
- The ~15 `Inline*` elementwise variants (`InlineAdd`, `InlineMul`, `InlineRelu`, etc.) can be collapsed into two parametric variants:
  - `InlineUnaryFloat(UnaryFloatFn)` — small enum discriminant
  - `InlineBinaryFloat(BinaryFloatFn)` — small enum discriminant
- Reduces `dispatch_kernel` match arms from ~50 to ~20
- LLVM will still inline the closures through the discriminant

### 1d. Merge CustomOpRegistry into TapeContext
- **File**: [kv/registry.rs](crates/hologram-exec/src/kv/registry.rs), [tape.rs](crates/hologram-exec/src/tape.rs)
- Currently threaded as a separate `Option<&CustomOpRegistry>` — make it a field on `TapeContext`

---

## Phase 2: BinaryElementWiseView — The Missing Primitive

**Goal**: Extend the LUT system from unary to binary operations.

### 2a. Create `BinaryElementWiseView`
- **File**: NEW [view/binary.rs](crates/hologram-core/src/view/binary.rs)
- 64KB `[[u8; 256]; 256]` table for binary byte→byte operations
- `apply(a: u8, b: u8) -> u8` — single 2D array lookup, O(1)
- `then_unary(view: &ElementWiseView) -> BinaryElementWiseView` — compose: apply binary, then unary
- `from_float_op<E: Encoding>(op: fn(f32,f32)->f32, enc: &E) -> Self` — build table by lifting each (a,b) pair to float, applying op, re-embedding
- Cache-line aligned, `#[repr(align(64))]`

### 2b. Create `DynamicEncoding`
- **File**: NEW [encoding/dynamic.rs](crates/hologram-core/src/encoding/dynamic.rs)
- `DynamicEncoding { min: f32, max: f32 }` — maps arbitrary float range to [0, 255]
- Implements existing `Encoding` trait
- Precomputes `scale` and `offset` at construction for fast embed/lift
- `embed_f32(&self, v: f32) -> u8` and `lift_f32(&self, b: u8) -> f32` — avoid f64 intermediate

### 2c. Precompute float-bridge tables
- **File**: NEW [lut/float_bridge.rs](crates/hologram-core/src/lut/float_bridge.rs)
- For each unary FloatOp (Relu, Sigmoid, Tanh, Gelu, Silu, Exp, etc.): precompute `[u8; 256]` table with `SignedEncoding` (these already exist as activation tables — just formalize the bridge)
- For binary FloatOp (Add, Mul, Sub): precompute `[[u8; 256]; 256]` tables
- Total: ~21 unary tables (5.25 KB) + ~4 binary tables (256 KB) = ~261 KB additional static memory

---

## Phase 3: Float-to-LUT Bridge in Tape

**Goal**: Enable float operations to optionally execute via byte-domain LUT.

### 3a. Add LUT-float kernel variants
- **File**: [tape.rs](crates/hologram-exec/src/tape.rs)
- New variants:
  - `TapeKernel::LutUnaryFloat { view: ElementWiseView, encoding: EncodingId }`
  - `TapeKernel::LutBinaryFloat { view: BinaryElementWiseView, enc_lhs: EncodingId, enc_rhs: EncodingId, enc_out: EncodingId }`
- `EncodingId` is a small enum: `Signed | Unsigned | Angle | Dynamic { min: f32, max: f32 }`

### 3b. Execution path for LUT-float
- In `dispatch_kernel`, the LUT-float path:
  1. Cast input bytes to `&[f32]` via bytemuck
  2. Embed each f32 to u8 via encoding
  3. Apply LUT (single array lookup per element)
  4. Lift each u8 back to f32 via encoding
  5. Write to output buffer
- For unary: ~3 ops/element (embed + lookup + lift)
- For binary: ~5 ops/element (2 embeds + lookup + lift + write)

### 3c. Wire into tape builder
- **File**: [tape_builder.rs](crates/hologram-exec/src/tape_builder.rs)
- When compiling a graph node, check if:
  - Op has a LUT equivalent (unary activation or binary elementwise)
  - `QuantizationPolicy` allows Q0 (new field on compilation context)
- If yes, emit `LutUnaryFloat` / `LutBinaryFloat` instead of `InlineAdd` etc.
- `QuantizationPolicy` enum: `Exact | Q0Preferred | Auto`
  - `Exact`: always use float (current behavior)
  - `Q0Preferred`: use LUT when available
  - `Auto`: use `CurvatureFlux` to decide at runtime

---

## Phase 4: Dynamic Precision via CurvatureFlux

**Goal**: Use the existing carry-tracking system to dynamically choose LUT vs float.

### 4a. Per-tensor encoding calibration
- During tape compilation with `Auto` policy, run a calibration pass to determine min/max range per tensor
- Store as `EncodingId::Dynamic { min, max }` in the `TapeInstruction`
- This can be done offline (at `.holo` build time) or lazily (first inference pass)

### 4b. Runtime precision switching
- In `dispatch_kernel`, for `Auto` policy:
  - Check `tape_ctx.flux.required_level()`
  - If Q0 → use LUT path
  - If Q1+ → fall through to float path
- After each LUT op, compute curvature and call `flux.accumulate()`
- Reset flux at layer boundaries (already supported)

### 4c. The conceptual payoff
- Every computation becomes: "what is the address of the result given these input addresses?"
- At Q0: literal table lookup — `result_addr = TABLE[input1_addr][input2_addr]`
- At Q1+: computed but deterministic — same inputs always give same output (content-addressed by definition, just not via LUT)
- `CurvatureFlux` is the "precision pressure gauge" that decides when lookup is sufficient vs when computation is needed

---

## Phase 5: WASM Optimization

**Goal**: Make the LUT path the fast path for WASM targets.

- Extend existing WASM SIMD path in [view/simd.rs](crates/hologram-core/src/view/simd.rs) to cover `BinaryElementWiseView`
- Default `QuantizationPolicy` to `Q0Preferred` on WASM (table lookups outperform f32 SIMD on 128-bit lanes)
- Profile LUT vs float on WASM to validate

---

## Precision Tradeoff Summary

| Path | Precision | Cost/element | Memory | When to use |
|------|-----------|-------------|--------|-------------|
| Q0 LUT | ~7.8 bits | 1 array lookup | 256B unary, 64KB binary | Post-norm activations, embeddings |
| Q1 computed | ~15.8 bits | Native u16 arith | N/A | Mid-network, attention values |
| Float (current) | 23 bits (f32) | Native f32 + BLAS | 4B/element | Accumulation, loss-sensitive paths |
| LUT-GEMM Q4/Q8 | Variable | O(Q) per element | Psumbook + centroids | Weight matrices (already deployed) |

Key: Q0 is NOT a replacement for float. It's the **default fast path** with `CurvatureFlux` promoting to float when precision demands it.

---

## Critical Files

| File | Change |
|------|--------|
| [float_dispatch/mod.rs](crates/hologram-exec/src/float_dispatch/mod.rs) | Collapse 4 entry points → 1 primary |
| [kv/store.rs](crates/hologram-exec/src/kv/store.rs) | Gate behind `#[cfg(test)]` |
| [tape.rs](crates/hologram-exec/src/tape.rs) | Consolidate Inline* variants; add LutFloat kernels |
| [tape_builder.rs](crates/hologram-exec/src/tape_builder.rs) | Wire QuantizationPolicy into kernel selection |
| [view/mod.rs](crates/hologram-core/src/view/mod.rs) | Add `mod binary` |
| NEW [view/binary.rs](crates/hologram-core/src/view/binary.rs) | `BinaryElementWiseView` (64KB tables) |
| NEW [encoding/dynamic.rs](crates/hologram-core/src/encoding/dynamic.rs) | `DynamicEncoding` with runtime min/max |
| NEW [lut/float_bridge.rs](crates/hologram-core/src/lut/float_bridge.rs) | Precomputed float-via-byte tables |
| [carry/mod.rs](crates/hologram-core/src/carry/mod.rs) | Wire flux into tape dispatch decisions |

---

## Verification

1. `cargo test` — all existing tests pass after Phase 1 (no semantic changes)
2. Add roundtrip tests: `lift(embed(x)) ≈ x` within encoding tolerance for `DynamicEncoding`
3. Add accuracy tests: `lut_add(a, b)` vs `f32_add(a, b)` within Q0 tolerance (~0.008)
4. Run TinyLlama E2E with `Q0Preferred` policy — measure output divergence vs `Exact`
5. WASM benchmark: LUT path vs float path latency on representative workload
6. `cargo clippy -- -D warnings` and `cargo fmt` throughout
