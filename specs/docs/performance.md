# Performance Guide — hologram

## Benchmarks

Criterion benchmarks live in `crates/hologram-bench/benches/`:

```bash
just bench           # run all benchmarks
cargo bench          # alternative
cargo bench -- lut   # filter to LUT benchmarks
```

Key benchmarks:

| File | Measures |
|------|----------|
| `lut.rs` | Raw LUT table lookup throughput |
| `view.rs` | `ElementWiseView::apply_slice` with SIMD |
| `fusion.rs` | View composition overhead |
| `kv_dispatch.rs` | `KvStore::dispatch` latency per op |
| `executor.rs` | End-to-end graph execution |
| `lut_gemm.rs` | 4-bit and 8-bit quantized GEMM |

---

## Profiling

Recommended tools:

- **flamegraph**: `cargo install flamegraph && cargo flamegraph --bench executor`
- **perf** (Linux): `perf record cargo bench && perf report`
- **Instruments** (macOS): Profile `cargo bench` binary in Instruments.app

---

## Known Bottlenecks

| Path | Bottleneck | Mitigation |
|------|------------|------------|
| `apply_slice` scalar path | Memory bandwidth | Use SIMD paths (AVX2/SSE4.2) |
| `BufferArena::get` | HashMap lookup | Pre-sized arena; consider arena index |
| rkyv deserialization | Validation overhead | Use `access` (unchecked) in trusted contexts |
| Level scheduling | Rayon overhead on small graphs | Batch small graphs; tune `parallel` feature |

---

## Optimization Techniques

1. **SIMD**: `ElementWiseView::apply_slice` uses AVX2 (32 bytes/iter) or SSE4.2 (16 bytes/iter) when available. Enable with `--features simd`.

2. **Level parallelism**: Nodes within an execution level run concurrently via rayon. Enable with `--features parallel`.

3. **Zero-copy loading**: `.holo` archives use rkyv for zero-copy deserialization. Memory-mapped files avoid load-time copies.

4. **LUT fusion**: Chains of `LutOp` nodes are fused into single `FusedView` nodes at compile time, reducing dispatch overhead.

5. **Cache alignment**: `ElementWiseView` is 64-byte aligned (cache line) for optimal L1 access.

6. **Compile-time tables**: All LUT tables are `const` and placed in `.rodata`, avoiding runtime allocation.

---

## Targets

| Metric | Target | Current |
|--------|--------|---------|
| LUT lookup | < 1 ns/byte | ~0.3 ns/byte (AVX2) |
| View fusion | < 1 µs | ~200 ns |
| Graph dispatch | < 100 ns/node | ~50 ns/node |
| .holo load (mmap) | < 1 ms for 10 MB | ~0.5 ms |
| Binary size (no_std) | < 100 KB | ~40 KB (wasm32) |