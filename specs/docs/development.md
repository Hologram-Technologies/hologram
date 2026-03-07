# Development Guide — hologram

## Prerequisites

- Rust stable toolchain (see `rust-toolchain.toml` if present)
- `cargo clippy`, `cargo fmt`
- `just` command runner (install via `cargo install just`)
- For WASM builds: `rustup target add wasm32-unknown-unknown`
- For embedded builds: `rustup target add thumbv7em-none-eabihf`

---

## Building

```bash
cargo build
cargo build --release

# Or via just:
just build          # standard build
just wasm           # wasm32-unknown-unknown build of hologram-core
```

Workspace crates are built together. The root crate re-exports all subcrate APIs.

---

## Testing

```bash
cargo test
cargo test --workspace

# Or via just:
just test           # cargo test --workspace
just ci             # fmt check + clippy + test (full CI)
```

Integration tests requiring the `ffi` feature:
```bash
cargo test --features ffi --test e2e
```

---

## Linting and Formatting

```bash
cargo clippy --workspace -- -D warnings
cargo fmt --check

# Or via just:
just clippy
just fmt
```

CI enforces both. Fix all warnings before opening a PR.

---

## Benchmarks

```bash
cargo bench

# Or via just:
just bench          # criterion benchmarks
```

Benchmarks live in `crates/hologram-bench/benches/`.

---

## Workflow

1. Create a branch from `main`.
2. Make changes; run `just ci` to verify fmt, clippy, and tests pass.
3. Open a PR with a clear description.
4. PR requires passing CI and review.
5. Squash merge to `main`.