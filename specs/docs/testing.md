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

Each crate has comprehensive unit tests for its public API and internal invariants.

---

## Integration Tests

Integration tests live in `tests/`. The main integration test is `e2e.rs` which requires the `ffi` feature:

```bash
cargo test --features ffi --test e2e
```

The `hologram-archive` crate has embedded integration tests for round-trip serialization, mmap loading, and pipeline handling.

---

## Test Conventions

- Test names are descriptive: `<module>_<behavior>_<condition>` (e.g., `arena_insert_and_get`).
- Use `std::env::temp_dir()` for tests that need the filesystem.
- Do not write to shared state; tests must be order-independent.
- Each test covers one behavior.
- LUT tables are tested exhaustively: verify all 256 values match expected computation.
- Serialization tests verify round-trip: write → read → compare.

---

## Benchmark Tests

Criterion benchmarks in `crates/hologram-bench/benches/` cover:

| Benchmark | What it measures |
|-----------|-----------------|
| `lut.rs` | LUT table lookup throughput |
| `view.rs` | ElementWiseView apply performance |
| `fusion.rs` | View fusion compilation cost |
| `executor.rs` | Graph execution throughput |
| `kv_dispatch.rs` | KvStore dispatch latency |
| `archive.rs` | .holo serialization/deserialization |
| `compiler.rs` | Full compilation pipeline |
| `lut_gemm.rs` | Quantized matrix multiplication |

---

## Running Tests

```bash
cargo test                       # all tests
cargo test --workspace           # all workspace crates
cargo test -p hologram-core      # single crate
cargo test --test e2e            # single integration test (needs --features ffi)
cargo test <pattern>             # filter by test name

# Via just:
just test                        # cargo test --workspace
just ci                          # full CI: fmt + clippy + test
```