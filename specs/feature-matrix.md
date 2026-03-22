# Feature Matrix — Hologram Target Compatibility

## hologram-core feature flags

| Feature      | Description                                        | Default |
|--------------|----------------------------------------------------|---------|
| `std`        | Enables std + serialize + rkyv validation          | on      |
| `serialize`  | rkyv Archive/Serialize/Deserialize derives + alloc | off     |
| `simd`       | SIMD-accelerated LUT paths (x86 AVX2/SSE4.2)      | on      |
| `no_alloc`   | Marker for StaticBuf-only usage (no heap)          | off     |

## hologram-exec feature flags

| Feature       | Description                                         | Default |
|---------------|-----------------------------------------------------|---------|
| `std`         | Standard library support                            | on      |
| `parallel`    | Rayon-based parallel level execution                | off     |
| `accelerate`  | macOS Accelerate BLAS for MatMul                    | on (macOS) |
| `profile`     | Per-op timing + intermediate capture                | off     |

## Compute Backend Availability

Backends are auto-detected at build time by `build.rs` — no manual feature flags needed.

| Backend | Auto-detected when | GPU kernels | Zero-copy arena | Status |
|---------|--------------------|-------------|-----------------|--------|
| **CPU** | Always | N/A (SIMD autovectorization) | N/A | Production |
| **Metal** | `target_os = "macos"` | 16 kernels (9 unary, 4 binary, SGEMM, softmax, RmsNorm) | MTLBuffer via ArenaBuffer::Metal | Production |
| **WebGPU** | `target_arch = "wasm32"` | Stub (returns Skipped) | Not yet | Planned |
| **CUDA** | `CUDA_HOME` set or `nvcc` on PATH | Stub (returns Skipped) | Not yet | Planned |

### Backend Priority (Auto selection)

Metal > WebGPU > CUDA > CPU

### Metal GPU Dispatch Thresholds

| Op type | Min buffer for GPU dispatch | Below threshold |
|---------|---------------------------|-----------------|
| Elementwise unary/binary | 4 MB (1M floats) | CPU monomorphized SIMD |
| MatMul | 128x128 output (64KB) | Accelerate BLAS |
| Softmax / RmsNorm | 4 MB | CPU dispatch |

## Execution Paths

| Path | Build cost | Per-inference cost | Allocation | Backend support |
|------|-----------|-------------------|------------|-----------------|
| `execute_tape` (EnumTape) | Tape build once | O(1) dispatch per op | Zero (swap-insert) | CPU + Metal + GPU |
| `execute_plan` (KvExecutor) | Schedule per call | O(n) match chain | Per-node Vec | CPU only |
| `execute_bytes` | Parse + schedule | O(n) match chain | Per-node Vec | CPU only |

### Tape Dispatch Optimization Levels

| Level | Ops covered | Overhead per op |
|-------|-------------|-----------------|
| Inline dispatch (Phase 9a) | Relu, Neg, Abs, Sigmoid, Silu, Tanh, Gelu, Exp, Reciprocal, Add, Mul, Sub, Div, MatMul, Softmax, RmsNorm | ~0 ns |
| Generic Float dispatch | All other FloatOps | ~60 ns (backend check + category match) |
| Byte-domain LUT | Lut, FusedView, Prim | ~30 ns (direct table apply) |
| LUT-GEMM | MatMulLut4, MatMulLut8 | ~100 ns (weight cache lookup) |

## Cross-compilation Targets

| Feature        | x86_64-linux | aarch64-macos | wasm32 (no_std) | thumbv7em (ARM) | esp32 |
|----------------|:------------:|:-------------:|:---------------:|:---------------:|:-----:|
| `std`          | yes           | yes            | no               | no               | no     |
| `serialize`    | yes           | yes            | no               | no               | no     |
| `simd`         | AVX2/SSE4.2  | NEON           | partial (simd128)| no               | no     |
| `no_alloc`     | yes           | yes            | yes              | yes              | yes    |
| StaticBuf      | yes           | yes            | yes              | yes              | yes    |
| LUT tables     | yes           | yes            | yes              | yes              | yes    |
| rkyv 0.8       | yes           | yes            | no               | no               | no     |
| rayon parallel | yes           | yes            | no               | no               | no     |
| Metal GPU      | no            | yes (auto)     | no               | no               | no     |
| Accelerate BLAS| no            | yes (auto)     | no               | no               | no     |
| EnumTape       | yes           | yes            | yes (no Metal)   | no               | no     |
| KV cache       | yes           | yes            | yes              | no               | no     |
| TinyVec inputs | yes           | yes            | yes              | yes              | yes    |

## Build Recipes

```bash
# Standard build (std + serialize + simd + accelerate on macOS)
just build

# WASM no_std (no rkyv, no std, LUTs + StaticBuf only)
just wasm-nostd

# ARM bare-metal (Cortex-M4F, no_std)
just embedded

# Full CI (fmt + clippy + tests)
just ci

# Run benchmarks
just bench
```

## Binary Size Estimates (hologram-core, release, no_std)

| Target           | Approx .text size |
|------------------|-------------------|
| wasm32-unknown   | ~40 KB            |
| thumbv7em-none   | ~35 KB            |

These are below the 100 KB Sprint 8 target. The dominant sections are
the precomputed LUT tables in `.rodata` (~64 KB for full Q0 tables).

## Performance Summary (Sprint 15-16)

| Benchmark | KvExecutor | EnumTape | Speedup |
|-----------|-----------|----------|---------|
| Relu 64KB | 44.6 µs | 3.3 µs | 13.5x |
| Linear chain (4 ops, 256B) | 7.5 µs | 1.2 µs | 6.3x |
| Diamond (5 ops, 256B) | 11.4 µs | — | — |
| Wide parallel (17 ops, 256B) | 34.4 µs | — | — |
