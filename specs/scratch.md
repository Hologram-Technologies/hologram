When you're done with the Sprint, Make a git commit, merge to main, checkout a work branch for the next sprint. I want you to plan out the next sprint in @specs/SPRINT.md and the @specs/plans/001-greenfield-refactor.md  (and keep this up-to-date). Next start in on the next sprint

---

What remains for future sprints:

Sprint 15 (14 phases): Tape execution performance

Zero-alloc execution, enum dispatch, 8.5x SIMD speedup, LUT-GEMM + KvCache wiring
Sprint 16 (5 phases): Multi-backend + Metal GPU

ComputeBackend trait with auto-detection (build.rs)
BackendSelector for runtime selection
16 Metal compute kernels: 9 unary + 4 binary + tiled SGEMM + softmax + RmsNorm
Process-global shader caching via OnceLock<Arc<MetalBackend>>
Size thresholds (4MB elementwise, 128×128 matmul)
Conformance tests for relu, matmul, softmax
960 tests passing, clippy clean across the entire session.

What remains for future sprints:

MTLBuffer-backed arena (zero-copy GPU path — requires BufferArena refactor)
Async command buffer batching (requires MTLBuffer arena)
CUDA + WebGPU backends (same trait, different implementations)
Metal attention kernel (multi-head attention with tiled QK^T + softmax + score×V)

---

Keep track of those as well in the @specs/SPRINT.md. Can we reduce the amount of memory copying, hopefully to zero-copy and have the execution be O(1) constant-time lookup as much as possible?