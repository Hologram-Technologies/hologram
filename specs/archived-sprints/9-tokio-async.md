# Sprint 9: Tokio Integration + Async Execution

**Completed**: 2026-03-06
**Tests**: 669 total workspace tests, zero clippy warnings

## Goal

Add async compilation and execution paths backed by Tokio. Allow callers to stream graph evaluation across async tasks for large models and pipelined inference workloads.

## Deliverables

- [x] `hologram-async` crate: `Cargo.toml`, `lib.rs`, `compiler.rs`, `executor.rs`, `stream.rs`
- [x] `AsyncCompiler`: wraps `CompilerBuilder` in `tokio::task::spawn_blocking`; returns `JoinHandle<CompileResult<CompilationOutput>>`
- [x] `AsyncExecutor`: wraps `execute_bytes` in `tokio::task::spawn_blocking`; returns `JoinHandle<ExecResult<GraphOutputs>>`
- [x] `KvExecutor::execute_with_progress<F>`: per-level callback added to `hologram-exec` (no duplication — `execute` delegates to it)
- [x] `execute_bytes_with_progress`: added to `hologram-exec` public API
- [x] Streaming API: `execute_stream() -> (Receiver<LevelResult>, JoinHandle<...>)` via `tokio::sync::mpsc`
- [x] `LevelResult { level_index, nodes_executed }` per-level progress type
- [x] Benchmark: `async_exec.rs` — async vs sync compile + execute (10-node chain)
- [x] Benchmark: `async_stream.rs` — streaming vs batch (20-node chain)
- [x] 16 new tests in `hologram-async` (5 compiler, 5 executor, 6 stream); 2 new in `hologram-exec` (progress callback)
- [x] Sprint 8 archived to `specs/sprints/8-constrained-devices.md`
- [x] Zero clippy warnings; `just ci` green — **669 total workspace tests**

## Implementation Notes

- `KvExecutor::execute_with_progress<F>` extracted 4 private helpers: `build_node_map`, `seed_arena`, `dispatch_level`, `extract_named_outputs`. `execute` delegates with a no-op closure — no duplication.
- `execute_stream` uses `tx.blocking_send` inside `spawn_blocking`. Dropping the receiver silently discards sends; task always runs to completion.
- `AsyncCompiler` defaults `enable_fusion: true`; `.fuse(bool)` override available.
- Workspace crate `hologram-async` depends on: `hologram-compiler`, `hologram-exec`, `hologram-graph`, `tokio`.
- Root `src/lib.rs` re-exports `pub use hologram_async;`.
