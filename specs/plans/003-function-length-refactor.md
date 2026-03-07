# 003 — Function Length & Argument Count Refactor

## Problem

A spec compliance audit found **124 functions** exceeding the 15-line convention and several functions exceeding the 3-argument maximum. These violate the project's coding conventions defined in `CLAUDE.md` and `AGENTS.md`.

## Top Offenders by Crate

### hologram-core (`crates/hologram-core/src/view/simd.rs`)

| Function | Lines | Notes |
|----------|-------|-------|
| `apply_avx2()` | 41 | AVX2 vpshufb intrinsics |
| `apply_to_avx2()` | 42 | AVX2 vpshufb with separate output |
| `apply_sse42()` | 39 | SSE4.2 pshufb intrinsics |
| `apply_to_sse42()` | 40 | SSE4.2 pshufb with separate output |

These SIMD functions are inherently monolithic — they follow a setup/loop/remainder pattern that's tightly coupled by register usage. Splitting may add function call overhead in hot paths. **Consider a pragmatic exception for SIMD intrinsic code.**

### hologram-exec (`crates/hologram-exec/src/`)

| Function | Lines | Args | File |
|----------|-------|------|------|
| `dispatch_level()` | 33 | 6 | `eval/executor.rs` |
| `dispatch_with_constants()` | 30 | — | `kv/store.rs` |

`dispatch_level()` is the worst dual-violation (6 args + 33 lines). Should introduce a `LevelContext` struct to bundle args and split dispatch logic.

### hologram-archive (`crates/hologram-archive/src/`)

| Function | Lines | Args | File |
|----------|-------|------|------|
| `assemble_archive()` | 35 | 5 | `writer/holo_writer.rs` |
| `from_bytes()` | 40 | — | `loader/bytes.rs` |

`assemble_archive` should use a builder or struct for its params. `from_bytes` should split header validation, graph extraction, and weight extraction into helpers.

### hologram-ffi (`crates/hologram-ffi/src/`)

| Function | Lines | File |
|----------|-------|------|
| `hologram_outputs_by_name()` | 32 | `exec/mod.rs` |
| `embed()` | 35 | `encoding/mod.rs` |

FFI functions have inherent boilerplate (null checks, handle borrowing, error codes). May benefit from a macro to reduce repetition.

### hologram-compiler, hologram-bench

Multiple 16-34 line functions across liveness analysis, workspace planning, and benchmark setup code.

## Argument Count Violations

| Function | Args | Location |
|----------|------|----------|
| `dispatch_level()` | 6 | `hologram-exec/src/eval/executor.rs` |
| `assemble_archive()` | 5 | `hologram-archive/src/writer/holo_writer.rs` |
| `execute_with_registry()` | 4 | `hologram-exec/src/eval/executor.rs` |

## Recommended Approach

1. **Benchmark before/after**: Run `just bench` and capture baselines before any refactoring of hot-path functions (SIMD, dispatch, GEMM).

2. **Start with non-hot-path code**: Refactor archive loader, writer, compiler functions first — these are I/O-bound and won't regress performance.

3. **Introduce context structs**: For `dispatch_level()` and `assemble_archive()`, bundle parameters into a struct to reduce argument count.

4. **SIMD exception**: Consider documenting a formal exception for `#[target_feature]` functions in SIMD modules, since splitting these across function boundaries defeats the purpose of target-feature gating.

5. **FFI macro**: Create a helper macro for the boilerplate pattern in FFI functions (null-check handle, borrow, call, set error).

## Scope

This is a significant refactoring effort affecting ~124 functions across 6 crates. Recommend dedicating a full sprint with:
- Pre-refactor benchmark baselines
- Incremental PRs by crate (core first, then exec, archive, ffi, compiler)
- Post-refactor benchmark comparison
- No functional changes — purely structural

## Acceptance Criteria

- All functions ≤ 15 lines (with documented exception for SIMD intrinsics if adopted)
- All functions ≤ 3 arguments
- `just ci` passes
- `just bench` shows no regressions beyond noise (< 5% on hot paths)
