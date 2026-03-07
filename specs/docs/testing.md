# Testing Strategy — hologram

## Unit Tests

Unit tests live in `#[cfg(test)]` modules in the same file as the code they test.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() { ... }
}
```

---

## Integration Tests

Integration tests live in `tests/`. Each file in `tests/` is a separate
test binary that can import the crate as an external user would.

### Key Test Files

| File | Purpose |
|------|---------|
| `tests/e2e.rs` | Full pipeline: graph construction → compilation → execution |
| `tests/custom_ops.rs` | Custom operation registration and dispatch verification |

### Test Patterns

- **Linear chain**: Sigmoid → Relu → Output (verifies view fusion)
- **Diamond graph**: Fan-out and fan-in (verifies parallel execution)
- **Custom ops**: Register handler, execute, verify output
- **FFI round-trip**: C/WASM bindings produce correct results

---

## Test Conventions

- Test names are descriptive: `test_<behavior>_<condition>`.
- Use `tempfile::tempdir()` for tests that need the filesystem.
- Do not write to shared state; tests must be order-independent.
- Each test covers one behavior.
- Generated fixtures are preferred over static fixture files.
- Tests verify O(1) execution property: fused chains execute in single lookup.

### Naming Examples

```rust
#[test]
fn test_sigmoid_lut_matches_reference() { ... }

#[test]
fn test_fusion_collapses_chain_to_single_view() { ... }

#[test]
fn test_parallel_levels_execute_independently() { ... }

#[test]
fn test_archive_roundtrip_preserves_graph() { ... }
```

---

## Benchmarks

Criterion benchmarks live in `hologram-bench`. Run via:

```bash
just bench                              # all suites
cargo bench -p hologram-bench <suite>   # specific suite
```

### Benchmark Suites

| Suite | What it measures |
|-------|------------------|
| `lut` | Table generation, single-byte apply, 21 LutOp variants |
| `view` | Composition chains, SIMD `apply_slice()`, rkyv round-trip |
| `kv_dispatch` | KvStore unary/binary at 256 B – 64 KB |
| `executor` | Linear, diamond, wide-parallel graph topologies |
| `lut_gemm` | Q4/Q8 matmul at 16×16 – 256×256 |
| `compiler` | Full compile pipeline at 10/50/100 nodes |
| `fusion` | Constant fold + CSE + view fusion at 10 – 1,000 nodes |
| `archive` | HoloWriter build + HoloLoader round-trip |
| `q1` | 16-bit quantum scaling |
| `async_exec` | Tokio batch throughput |
| `async_stream` | Token-streaming scheduling |
| `ffi` | C/WASM interface overhead |

---

## Running Tests

```bash
cargo test                   # all tests
cargo test --workspace       # all workspace crates
cargo test --test <name>     # single integration test file
cargo test <pattern>         # filter by test name
just test                    # workspace tests via just
just ci                      # full CI gate (fmt + clippy + test)
```

---

## CI Integration

CI runs:

1. `cargo fmt --check` — formatting verification
2. `cargo clippy --workspace -- -D warnings` — lint check
3. `cargo test --workspace` — all unit and integration tests

All three must pass for PR merge.