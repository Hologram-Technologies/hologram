# Transformer Layer Benchmark Specification

## Purpose

Measure end-to-end tape execution performance on a realistic transformer layer graph built synthetically using `GraphBuilder` — no ONNX loading required. This validates that Sprint 15-16 optimizations (inline dispatch, zero-copy arena, Metal GPU) translate to real model inference speedups.

## Benchmark Graph: Single Transformer Layer

The benchmark constructs a graph equivalent to one decoder layer of a small LLM (e.g., TinyLlama-scale: hidden=2048, n_heads=32, head_dim=64, FFN=5632).

### Graph Structure

```
Input [batch=1, seq=1, hidden=2048]
  │
  ├─ RmsNorm (attention norm)
  │    ├─ MatMul [2048 × 2048] → Q projection
  │    ├─ MatMul [2048 × 256]  → K projection (GQA: 4 KV heads)
  │    └─ MatMul [2048 × 256]  → V projection
  │         └─ Attention [32 Q heads, 4 KV heads, head_dim=64]
  │              └─ MatMul [2048 × 2048] → output projection
  │
  ├─ Add (residual connection)
  │
  ├─ RmsNorm (FFN norm)
  │    ├─ MatMul [2048 × 5632] → gate projection
  │    │    └─ Silu
  │    ├─ MatMul [2048 × 5632] → up projection
  │    │
  │    └─ Mul (gate * up)
  │         └─ MatMul [5632 × 2048] → down projection
  │
  └─ Add (residual connection)
       │
       Output
```

### Op Count: ~15 ops per layer
- 6 × MatMul (dominant: QKV proj, out proj, FFN gate/up/down)
- 2 × RmsNorm
- 2 × Add (residual)
- 1 × Attention
- 1 × Silu
- 1 × Mul (gate)
- 2 × Output/passthrough

### Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_dim | 2048 | TinyLlama scale |
| n_q_heads | 32 | |
| n_kv_heads | 4 | GQA (8:1 ratio) |
| head_dim | 64 | hidden / n_q_heads |
| ffn_dim | 5632 | ~2.75× hidden |
| seq_len | 1 | Decode step (single token) |
| batch | 1 | |

### Implementation Notes

- **Weights**: Random f32 data stored as `ConstantData::Bytes` in the graph's `ConstantStore`. No quantization for this benchmark (f32 baseline).
- **Input**: Random f32 tensor `[1, 1, 2048]` = 8KB.
- **Graph construction**: Use `GraphBuilder` with `Float(MatMul{m,k,n})`, `Float(RmsNorm{size,epsilon})`, `Float(Silu)`, `Float(Add)`, `Float(Attention{...})` ops.
- **Compilation**: Compile via `HoloWriter` → archive → `build_tape_from_plan`.

### Benchmark Variants

| Variant | Path | What it measures |
|---------|------|------------------|
| `transformer::kvexecutor(decode)` | `execute_bytes` | KvExecutor baseline |
| `transformer::enum_tape(decode)` | `execute_tape` | Tape with inline dispatch |
| `transformer::tape_cpu_only(decode)` | `execute_tape` + `BackendSelector::Cpu` | CPU-only tape |

### Expected Metrics

- **KvExecutor**: ~500-1000µs (dominated by MatMul)
- **EnumTape**: ~300-600µs (inline dispatch + zero-alloc overhead elimination)
- **Speedup**: 1.5-2x tape vs KvExecutor for decode step

### Future Extensions

- Prefill benchmark (seq_len=512, batch=1)
- Multi-layer benchmark (30 layers)
- Quantized weights (LUT-GEMM Q4/Q8 path)
- Metal GPU benchmark (above 4MB threshold for matmul)
