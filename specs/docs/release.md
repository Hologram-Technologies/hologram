# Release Process — hologram

## Versioning

This project follows [Semantic Versioning](https://semver.org):
`MAJOR.MINOR.PATCH`.

- **MAJOR**: breaking API or format change (e.g., `.holo` format version bump).
- **MINOR**: new functionality, backward-compatible.
- **PATCH**: bug fixes, no API change.

Current version: `0.1.0` (pre-1.0 development).

---

## Release Checklist

- [ ] All tests pass (`just ci`)
- [ ] No Clippy warnings (`cargo clippy --workspace -- -D warnings`)
- [ ] Benchmarks run without regression (`just bench`)
- [ ] Version bumped in workspace `Cargo.toml` and all crate `Cargo.toml` files
- [ ] PR merged to `main`
- [ ] Release tag created: `v<version>`

---

## Publishing

Crates are published to crates.io in dependency order:

1. `hologram-core` (no internal deps)
2. `hologram-graph` (depends on hologram-core)
3. `hologram-archive` (depends on hologram-graph, hologram-core)
4. `hologram-exec` (depends on hologram-graph, hologram-core)
5. `hologram-compiler` (depends on hologram-graph, hologram-archive)
6. `hologram-async` (depends on hologram-exec)
7. `hologram-ffi` (depends on hologram-exec)
8. `hologram-cli` (depends on hologram-compiler, hologram-exec, hologram-archive)
9. `hologram` (root crate, depends on all)

Binary releases are built with `--features full` for maximum functionality.