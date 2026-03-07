****# Validation Harness — hologram

## Validation Goals

The validation harness verifies:

1. **LUT correctness**: All 256 byte-domain entries for each of the 21 LUT operations (sigmoid, tanh, relu, gelu, silu, sin, cos, etc.) produce correct outputs
2. **Composition fidelity**: Chained operations match fused ElementWiseView results exactly
3. **Quantization accuracy**: Q4 and Q8 quantized operations stay within acceptable relative error bounds against f32 reference
4. **Serialization round-trip**: `.holo` archives written via `HoloWriter` can be loaded and executed with identical graph structure and outputs
5. **Graph semantics**: Fusion statistics (views_fused, constants_folded) match expected optimizations; acyclicity and connectivity hold
6. **Custom op dispatch**: Registry-based custom operations execute correctly through the graph pipeline

---

## Test Inputs

All test inputs are **generated at test time** — no external fixtures or downloaded data:

- **Exhaustive byte ranges**: `for b in 0u8..=255` for LUT table validation
- **Sampled ranges**: `step_by(256)` for Q1 65536-entry tables
- **Linear sequences**: `(0..=255).collect()` for 256-byte exhaustive tests
- **Batch sequences**: `(0..1024).map(|i| (i % 256) as u8).collect()` for throughput tests
- **Generated matrices**: Float weights scaled by `0.01` to `0.001` for GEMM validation
- **Quantized weights**: On-the-fly generation via `quantize_4bit()` and `quantize_8bit()`

Test data types use `bytemuck::cast_slice()` for safe f32 ↔ byte reinterpretation.

---

## Reference Outputs

Reference outputs are computed inline during test execution:

- **LUT operations**: `LutOp::apply(byte)` serves as the reference for table lookups
- **Composed views**: `ElementWiseView::compose().apply(byte)` for fused operation chains
- **Quantized GEMM**: Naive matmul loop computes expected values for Q4/Q8 verification
- **Wrapping arithmetic**: `wrapping_add` for modulo-256 byte-domain reductions

No external golden data files are stored. All expected values are deterministically computed from the same LUT tables and algorithms under test, ensuring self-consistency.

---

## Tolerance Thresholds

| Domain | Threshold | Description |
|--------|-----------|-------------|
| **Byte LUT** | Exact (0) | All 256 entries must match exactly |
| **Q8 relative** | < 0.01 | 1% relative error for 8-bit quantized ops |
| **Q4 relative** | < 0.05 | 5% relative error for 4-bit quantized ops |
| **Sigmoid at 0** | 32000–33500 | Centered around Q1 midpoint (32768) |
| **Sigmoid/Tanh positive** | > 60000 | Near saturation (65535) |
| **Sigmoid/Tanh negative** | < 5000 | Near floor (0) |
| **GELU/SiLU origin** | 120–136 | Origin behavior bounds |

Relative error formula: `rel_err = |got - exp| / max(|exp|, 1e-6)`

---

## Running Validation

```bash
# Full test suite (all crates)
just test

# Full CI (format check + clippy + tests)
just ci

# Single crate tests
cargo test -p hologram-core

# Integration tests (requires ffi feature)
cargo test --test e2e --features ffi

# Criterion benchmarks (timing validation)
just bench

# Filter by test name pattern
cargo test <pattern>
```