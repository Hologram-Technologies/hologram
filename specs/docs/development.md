# Development Guide — hologram

## Prerequisites

- Rust (stable toolchain — see `rust-toolchain.toml` if present)
- `cargo clippy`, `cargo fmt`
- `just` command runner (install via `cargo install just`)
- For WASM builds: `rustup target add wasm32-unknown-unknown`
- For embedded builds: `rustup target add thumbv7em-none-eabihf`
- For FFI header generation (optional): `cbindgen`

---

## Building

```bash
cargo build
cargo build --release
```

### Workspace Build

```bash
just build              # cargo build --workspace
```

### Target-Specific Builds

```bash
just wasm               # hologram-core for wasm32-unknown-unknown
just wasm-nostd         # no_std, no rkyv
just embedded           # ARM bare-metal (thumbv7em-none-eabihf)
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `simd` | ✓ | AVX2/SSE4.2 SIMD acceleration for bulk LUT apply |
| `parallel` | ✓ | Rayon work-stealing for parallel level execution |
| `std` | ✓ | Standard library (mmap, threading) |
| `cli` | — | CLI binary and subcommands |
| `wasm` | — | WASM bindings via wasm-bindgen |

---

## Testing

```bash
cargo test
cargo test --workspace
```

### Just Commands

```bash
just test               # cargo test --workspace
just ci                 # fmt-check + clippy + test (full CI gate)
```

### Integration Tests

Integration tests live in `tests/`. Key test files:

- `tests/e2e.rs`: Full pipeline tests (graph → compile → execute)
- `tests/custom_ops.rs`: Custom operation registration and dispatch

### Fixtures

Test fixtures are generated programmatically. No external fixture files are required.

---

## Linting and Formatting

```bash
cargo clippy -- -D warnings
cargo fmt --check
```

### Just Commands

```bash
just fmt                # cargo fmt --all
just clippy             # cargo clippy --workspace -- -D warnings
```

CI enforces both. Fix all warnings before opening a PR.

---

## Workflow

1. Create a branch from `main`.
2. Make changes; run tests and Clippy.
3. Ensure functions are ≤ 15 lines.
4. Ensure max 3 function arguments (use builder pattern for more).
5. No TODOs, stubs, or `unimplemented!()`.
6. Run `just ci` to verify full CI gate passes.
7. Open a PR with a clear description.
8. PR requires passing CI.

### Commit Messages

Use conventional commit format:

```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

### Architecture Sync

Before implementing significant functionality:

```bash
holoarch pull           # Pull latest docs from hologram-architecture
holoarch check          # Validate repository conformance
```

---

## Benchmarks

```bash
just bench              # Run all Criterion benchmarks
cargo bench -p hologram-bench <suite>  # Run specific suite
```

Available suites: `lut`, `view`, `kv_dispatch`, `executor`, `lut_gemm`, `compiler`, `fusion`, `archive`, `q1`, `async_exec`, `async_stream`, `ffi`