# Validation Harness — hologram

## Validation Goals

The validation harness verifies:

1. **Numerical correctness**: LUT outputs match IEEE 754 reference implementations within tolerance
2. **Fusion equivalence**: Fused view produces identical output to sequential application
3. **Serialization fidelity**: Archive round-trip preserves graph structure and weights exactly
4. **Schedule correctness**: Topological order satisfies all dependencies
5. **Parallel determinism**: Parallel execution produces same results as sequential
6. **Quantization accuracy**: Dequantized values match original within quantization error bounds

---

## Test Inputs

### Generated Inputs

Most validation uses programmatically generated inputs:

| Input type | Generation method |
|------------|-------------------|
| Full byte range | `(0u8..=255u8).collect()` |
| Edge cases | `[0, 1, 127, 128, 254, 255]` |
| Random sample | `rand::thread_rng().gen::<[u8; N]>()` |
| Quantization grid | Linear steps covering Q4/Q8 codebook |

### Encoding Domain Inputs

For pi-F-lambda encoding validation:

| Encoding | Input domain |
|----------|--------------|
| AngleEncoding | `[-π, π]` mapped to `[0, 255]` |
| SignedEncoding | `[-range, +range]` mapped to `[0, 255]` |
| UnsignedEncoding | `[0, 1]` mapped to `[0, 255]` |
| RawEncoding | `[0, 255]` pass-through |

---

## Reference Outputs

### LUT Reference

Reference values computed via IEEE 754 `f64` math:

```rust
fn reference_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

LUT entries are precomputed at compile time and compared against runtime lookup.

### Fusion Reference

Sequential application serves as reference for fused chains:

```rust
let sequential = gelu(relu(sigmoid(input)));
let fused = fused_view.apply(input);
assert_eq!(sequential, fused);
```

### Archive Reference

Round-trip validation:

```rust
let original_graph = build_test_graph();
let archive = HoloWriter::new().set_graph(&original_graph).build();
let loaded = HoloLoader::from_bytes(&archive).load()?;
assert_graph_eq(&original_graph, loaded.graph());
```

---

## Tolerance Thresholds

| Property | Tolerance | Rationale |
|----------|-----------|-----------|
| LUT byte output | Exact (0 ULP) | Deterministic table lookup |
| Encoded f64 → byte → f64 | ±1 LSB in byte domain | Quantization error |
| Fused vs sequential | Exact (0 ULP) | Composition is algebraic |
| Archive round-trip | Exact bit equality | rkyv is deterministic |
| Q4 dequantization | ±0.5 / 16 of range | 4-bit quantization error |
| Q8 dequantization | ±0.5 / 256 of range | 8-bit quantization error |
| Parallel vs sequential | Exact (0 ULP) | Same operations, different order |

---

## Running Validation

### Unit Test Validation

```bash
cargo test --workspace
```

Includes all numerical correctness tests.

### Exhaustive LUT Validation

```bash
cargo test -p hologram-core lut_exhaustive -- --ignored
```

Tests all 256 input values for all 21+ LutOp variants.

### Fusion Exhaustive

```bash
cargo test -p hologram-graph fusion_exhaustive -- --ignored
```

Tests composition of all LutOp pairs.

### Archive Fuzz

```bash
cargo test -p hologram-archive fuzz -- --ignored
```

Generates random graphs, archives, and validates round-trip.

### Full Validation Suite

```bash
just ci
```

Runs formatting, linting, and all tests including validation.