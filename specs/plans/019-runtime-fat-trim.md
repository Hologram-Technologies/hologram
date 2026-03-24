# Plan 019: Runtime Fat Trim & Allocation Elimination

## Context

Hologram's core tape executor is already well-optimized (17.5x faster than KvExecutor, inline dispatch, zero-copy arena). However, the float dispatch layer beneath it still allocates intermediate `Vec`s on every call, there are dead code paths (CUDA stub), unused dependencies, and redundant abstractions. This plan trims fat, eliminates allocations in the hot path, and reduces the dependency tree — strictly runtime performance improvements.

---

## Phase 1: Dead Code & Dependency Removal (zero risk)

### 1.1 Remove CUDA backend stub
- **Delete** `crates/hologram-exec/src/backend/cuda.rs` (39 lines, always returns `Skipped`)
- **Edit** `crates/hologram-exec/src/backend/mod.rs`:
  - Remove `#[cfg(has_cuda)] pub mod cuda;`
  - Remove `CudaBackend` from `default_backend()` and `available_backends()`
  - Keep `Cuda` enum variant (it resolves to CPU fallback already)
- **Edit** `crates/hologram-exec/build.rs`: remove CUDA detection block if present

### 1.2 Replace `dirs` crate with cross-platform `home_dir()` helper
- **Edit** `src/config.rs`: add a small `home_dir()` fn:
  ```rust
  fn home_dir() -> Option<PathBuf> {
      #[cfg(windows)]
      { std::env::var("USERPROFILE").ok().map(PathBuf::from) }
      #[cfg(not(windows))]
      { std::env::var("HOME").ok().map(PathBuf::from) }
  }
  ```
  - Line 84: `dirs::home_dir()` → `home_dir()`
  - Line 127: same replacement
- **Edit** `Cargo.toml`: remove `dirs` from `[dependencies]` and `[workspace.dependencies]`
- Saves ~25 transitive dependencies; works on macOS, Linux, and Windows

### 1.3 Gate `serde`/`toml` behind `cli` feature
- **Edit** `Cargo.toml`:
  - Move `serde`, `toml` from required `[dependencies]` to optional
  - Add to `cli` feature: `"dep:serde", "dep:toml"`
  - The config system (`src/config.rs`) is only used by the CLI binary
- This removes serde/toml from the library dependency tree when used as `hologram` lib

### 1.4 Narrow tokio features
- **Edit** `Cargo.toml`: `tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync"] }` (drop "full")

---

## Phase 2: Systematic `.to_vec()` Elimination (high impact)

71 `.to_vec()` calls found across hologram-exec/src (excluding tests). Categorized by fix strategy:

### 2.1 Passthrough `.to_vec()` → tape-builder passthrough (zero dispatch, zero copy)

These return `inputs[0].to_vec()` for identity/no-op cases. Fix by detecting at tape-build time and emitting a passthrough instruction instead of a kernel.

| File | Line(s) | Op | Condition |
|------|---------|-----|-----------|
| `cast.rs` | 11, 21, 86, 124 | Cast | `from == to`, misaligned fallback |
| `shape_ops.rs` | 19, 27, 31 | Reshape | pure reshape (data unchanged) |
| `shape_ops.rs` | 46, 49, 55, 71 | Transpose | identity perm, empty, mismatch |
| `spatial.rs` | 22, 29 | Resize | all scales 1.0 |
| `spatial.rs` | 65, 77 | Pad | all pads 0 |
| `mod.rs` | 447 | Reshape/Transpose/GatherND | passthrough arm |
| `mod.rs` | 667-668 | KvWrite/KvRead | passthrough |

**Action**:
- **tape_builder.rs**: Add passthrough detection for `Cast{from==to}`, identity `Transpose`, identity `Resize`/`Pad` at kernel resolution time
- **float_dispatch/mod.rs**: For the allocating dispatch path, change passthrough arms to return `Cow::Borrowed(inputs[0])` or accept `out_buf` parameter

### 2.2 Norm `_into` variants: write directly to `out_buf` (9 calls)

These all do `let mut out = x.to_vec()` → process in-place → `out_buf.extend_from_slice(cast_slice(&out))`. Eliminates intermediate Vec.

- **File**: `crates/hologram-exec/src/float_dispatch/norm.rs`
- **Lines**: 28, 81, 111, 225, 288, 315, 341, 370, 404
- **Pattern**: Pre-size `out_buf`, get `&mut [f32]` via `bytemuck::cast_slice_mut`, copy input there, process in-place
- **Functions**: `dispatch_softmax`, `dispatch_rms_norm`, `dispatch_log_softmax`, `dispatch_layer_norm`, `dispatch_add_rms_norm` and their `_into` variants
- For allocating variants: use `x.into_owned()` instead of `x.to_vec()`

### 2.3 Attention: eliminate triple `.to_vec()` (3 calls)

- **File**: `crates/hologram-exec/src/float_dispatch/attention.rs`
- **Line 88**: `(q_raw.to_vec(), k_raw.to_vec(), v_raw.to_vec())` — 3 full tensor copies when heads_first=true
- **Fix**: Refactor attention core to accept `&[f32]` slices via `Cow<[f32]>`
- **Line 99**: mask `to_vec()` → `into_owned()`
- **Line 292**: RoPE `to_vec()` → `into_owned()`

### 2.4 Misc ops: `into_owned()` instead of `to_vec()` (4 calls)

- **File**: `crates/hologram-exec/src/float_dispatch/misc.rs`
- Lines 40, 56, 101: scatter_nd, cumsum, reverse_sequence → `into_owned()`

### 2.5–2.15 Lower priority items

- Conv bias, gather/concat shape, spatial scales, helpers strides, matmul shape — small data, low priority
- Tape transpose fallbacks — error path only
- KvStore — old executor, leave as-is unless on runtime path
- Arena/GPU readback — unavoidable hardware boundary
- Test-only calls — skip

### Full Summary

| Priority | Strategy | Calls | Impact |
|----------|----------|-------|--------|
| **P0** | Tape-builder passthrough | 17 | Zero-copy for identity ops |
| **P1** | Norm: write directly to out_buf | 9 | Eliminates hot-path alloc |
| **P2** | Attention: Cow slice refactor | 4 | 3x tensor copy eliminated |
| **P3** | `into_owned()` over `to_vec()` | 8 | Avoids double-copy |
| Unavoidable | Arena/GPU readback, data creation | 13 | Hardware boundary |
| Test-only | Test helpers | 16 | No runtime impact |

---

## Phase 2b: Inline More Ops as TapeKernel Variants

Currently 17 ops have inline TapeKernel variants. The remaining ~40 ops go through the generic `Float(fop)` path adding 2 match layers + 1 vtable call.

### Candidates for inlining (high-frequency in transformer inference):

| Op | Parameters | Inline variant |
|----|------------|----------------|
| `LayerNorm` | `size, epsilon` | `InlineLayerNorm { size: u32, epsilon: u32 }` |
| `AddRmsNorm` | `size, epsilon` | `InlineAddRmsNorm { size: u32, epsilon: u32 }` |
| `LogSoftmax` | `size` | `InlineLogSoftmax { size: u32 }` |
| `Attention` | 6 params | `InlineAttention { ... }` |
| `RotaryEmbedding` | `dim, base, n_heads` | `InlineRoPE { dim: u32, base: u32, n_heads: u32 }` |
| `Gather` | `dim, dtype` | `InlineGather { dim: u32, dtype: FloatDType }` |
| `Concat` | `axis, n_inputs` | `InlineConcat { axis: u32, n_inputs: u32 }` |
| `Transpose` | `perm` | `InlineTranspose { perm: [u8; 4] }` |

### Simple unary/binary ops (trivial to inline):

Log, Sqrt, Cos, Sin, Sign, Floor, Ceil, Round, Erf, Clip, Min, Max

### Skip (rare ops):

Conv2d, ConvTranspose, MaxPool2d, AvgPool2d, GlobalAvgPool, Resize, PadOp, LRN, TopK, CumSum, Compress, ReverseSequence, ScatterND, NonZero, IsNaN, ReduceProd, InstanceNorm

---

## Phase 3: Weight Cache & Dispatch Cleanup (medium impact)

### 3.1 Weight cache: eliminate double hash lookup
- **File**: `crates/hologram-exec/src/kv/weight_cache.rs`
- **Fix**: Use `Entry` API — single hash probe per access

### 3.2 Remove `dispatch_float` wrapper
- **File**: `crates/hologram-exec/src/float_dispatch/mod.rs`
- Inline the 1-line wrapper into all callers

### 3.3 Allocating norm variants: `into_owned()` instead of `to_vec()`
- **File**: `crates/hologram-exec/src/float_dispatch/norm.rs`

---

## Critical Files

| File | Changes |
|------|---------|
| `crates/hologram-exec/src/backend/cuda.rs` | Delete |
| `crates/hologram-exec/src/backend/mod.rs` | Remove CUDA references |
| `crates/hologram-exec/src/float_dispatch/norm.rs` | Rewrite `_into` fns, fix allocating variants |
| `crates/hologram-exec/src/float_dispatch/attention.rs` | Zero-copy heads_first path |
| `crates/hologram-exec/src/float_dispatch/cast.rs` | Identity short-circuit |
| `crates/hologram-exec/src/float_dispatch/spatial.rs` | Passthrough detection |
| `crates/hologram-exec/src/float_dispatch/mod.rs` | Remove wrapper, add passthrough |
| `crates/hologram-exec/src/tape.rs` | New inline TapeKernel variants |
| `crates/hologram-exec/src/tape_builder.rs` | Identity cast → passthrough, new inline mappings |
| `crates/hologram-exec/src/kv/weight_cache.rs` | Entry API fix |
| `Cargo.toml` | Remove dirs, gate serde/toml, narrow tokio |
| `src/config.rs` | Replace dirs::home_dir() |
| `specs/SPRINT.md` | Add Sprint 21 |

## Verification

1. `cargo test` — all existing tests must pass
2. `cargo clippy -- -D warnings` — no new warnings
3. `cargo build --no-default-features` — library builds without serde/toml/dirs
4. `cargo build --features cli` — CLI still builds with config support
5. Benchmark: run existing tape benchmarks to confirm no regression
