# Performance Guide — hologram

## Benchmarks

Criterion benchmarks measure critical paths:

```bash
just bench                              # all suites
cargo bench -p hologram-bench <suite>   # specific suite
```

### Benchmark Suites

| Suite | What it measures | Key metric |
|-------|------------------|------------|
| `lut` | LUT generation, single-byte apply | ns/lookup |
| `view` | Chain composition, SIMD bulk apply | GB/s throughput |
| `kv_dispatch` | KvStore unary/binary dispatch | ns/op |
| `executor` | Full graph execution | ms/graph |
| `lut_gemm` | Q4/Q8 quantized matmul | GFLOPS equivalent |
| `compiler` | Compile pipeline | ms/100 nodes |
| `fusion` | Optimization passes | nodes/second |
| `archive` | Serialize + deserialize | MB/s |
| `async_exec` | Tokio batch throughput | executions/second |
| `ffi` | C/WASM overhead | ns/call |

---

## Profiling

### Flamegraph

```bash
cargo install flamegraph
cargo flamegraph --bench <suite> -- --bench
```

### perf (Linux)

```bash
perf record -g cargo bench -p hologram-bench <suite>
perf report
```

### Instruments (macOS)

```bash
cargo build --release -p hologram-bench
instruments -t "Time Profiler" target/release/deps/hologram_bench-*
```

### DHAT (Heap Profiling)

```bash
cargo install dhat
# Add #[global_allocator] static ALLOC: dhat::Alloc = dhat::Alloc;
cargo run --release
```

---

## Known Bottlenecks

### 1. Large Graph Compilation

- **Symptom**: Compile time grows superlinearly with node count
- **Cause**: Liveness analysis iterates all nodes per level
- **Mitigation**: Batch similar operations; use subgraph templates

### 2. SIMD Fallback

- **Symptom**: Bulk apply slower than expected on older CPUs
- **Cause**: No AVX2; falls back to scalar
- **Detection**: Check `is_x86_feature_detected!("avx2")`
- **Mitigation**: SSE4.2 fallback is implemented but slower

### 3. mmap Page Faults

- **Symptom**: First execution slower than subsequent
- **Cause**: Lazy page-in from mmap
- **Mitigation**: Pre-fault pages with `madvise(MADV_WILLNEED)` or read-ahead

### 4. Rayon Overhead

- **Symptom**: Small graphs slower with parallel feature
- **Cause**: Thread pool overhead exceeds parallel benefit
- **Mitigation**: Disable `parallel` for graphs with < 100 nodes

---

## Optimization Techniques

### Cache-Line Alignment

`ElementWiseView` is 256 bytes (4 cache lines). Ensures:
- No false sharing in parallel execution
- Optimal prefetch behavior
- Aligned SIMD loads

### View Fusion

Chains of unary operations fuse at compile time:

```
Sigmoid → Relu → Tanh → Gelu
           ↓
    Single 256-byte LUT
```

Runtime cost: O(1) regardless of chain length.

### Buffer Reuse

Liveness analysis computes when buffers are dead:

```
Node A outputs to slot 0
Node B reads slot 0, outputs to slot 1
Node C reads slot 1 → slot 0 is dead, reusable
```

Workspace size is minimized via interference graph coloring.

### LUT-GEMM

Quantized matmul avoids multiply-accumulate:

```
Weight matrix: 4-bit indices into 16-entry codebook
Activation: 8-bit indices
Product: Precomputed partial sums in booklet
```

Memory bandwidth dominates; no FMA latency.

### Zero-Copy Serialization

rkyv deserializes in-place:

```
mmap file → validate pointer → cast to &Graph
```

No allocation, no copying. Archive loading is O(1).

---

## Targets

| Metric | Target | Current |
|--------|--------|---------|
| Single LUT lookup | < 1 ns | ~0.5 ns |
| SIMD bulk apply (AVX2) | > 40 GB/s | ~45 GB/s |
| Scalar bulk apply | > 4 GB/s | ~5 GB/s |
| Compile 100-node graph | < 10 ms | ~8 ms |
| Archive load (mmap) | < 1 ms | < 0.1 ms |
| Q4 matmul 256×256 | > 50 GOPS | ~60 GOPS |
| Level parallelism overhead | < 10 µs/level | ~5 µs/level |
| FFI call overhead | < 50 ns | ~30 ns |