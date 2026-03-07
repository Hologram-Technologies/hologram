# Release Process — hologram

## Versioning

This project follows [Semantic Versioning](https://semver.org):
`MAJOR.MINOR.PATCH`.

- **MAJOR**: breaking API or behavior change.
- **MINOR**: new functionality, backward-compatible.
- **PATCH**: bug fixes, no API change.

Note: The `.holo` archive format does not maintain backwards compatibility. Format version changes require a MAJOR version bump.

---

## Release Checklist

- [ ] All tests pass (`cargo test --workspace`)
- [ ] No Clippy warnings (`cargo clippy -- -D warnings`)
- [ ] `just ci` passes (full CI gate)
- [ ] Benchmarks run without regression (`just bench`)
- [ ] CHANGELOG.md updated (generated via `git-cliff`)
- [ ] Version bumped in root `Cargo.toml`
- [ ] Version bumped in all workspace member `Cargo.toml` files
- [ ] PR merged to `main`
- [ ] Release tag created: `v<version>`
- [ ] Architecture docs synced (`holoarch pull`)

---

## Publishing

### Crates.io

Workspace crates are published to crates.io in dependency order:

```bash
# 1. Core (no internal dependencies)
cargo publish -p hologram-core

# 2. Graph (depends on core)
cargo publish -p hologram-graph

# 3. Archive (depends on core, graph)
cargo publish -p hologram-archive

# 4. Exec (depends on core, graph, archive)
cargo publish -p hologram-exec

# 5. Compiler (depends on core, graph, archive)
cargo publish -p hologram-compiler

# 6. Async (depends on compiler, exec)
cargo publish -p hologram-async

# 7. FFI (depends on all)
cargo publish -p hologram-ffi

# 8. CLI (depends on all)
cargo publish -p hologram-cli

# 9. Root crate (re-exports)
cargo publish -p hologram
```

### Binary Releases

Binary releases are built for:

- `x86_64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`
- `x86_64-pc-windows-msvc`

Binaries are attached to GitHub releases.

### WASM Package

WASM builds are published to npm:

```bash
wasm-pack build crates/hologram-ffi --target web --features wasm
wasm-pack publish
```

---

## Changelog Generation

Changelog is generated from conventional commits using `git-cliff`:

```bash
git cliff --output CHANGELOG.md
```

Configuration is in `cliff.toml`.