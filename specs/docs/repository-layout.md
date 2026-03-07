# Repository Layout — hologram

## Top-Level Structure

```
hologram/
├── AGENTS.md         # agent coding rules (holoarch-managed section + project rules)
├── CLAUDE.md         # Claude Code instructions
├── Cargo.toml        # workspace root + root crate manifest
├── justfile          # just command recipes
├── src/              # root crate (re-exports all subcrates)
│   ├── lib.rs
│   └── main.rs       # CLI entry point (requires `cli` feature)
├── crates/           # workspace member crates
├── specs/            # all project documentation
└── tests/            # integration tests
```

---

## specs/ Layout

```
specs/
├── docs/             # project documentation (managed by holoarch)
│   ├── plans/        # planning documents
│   └── adrs/         # Architecture Decision Records
├── plans/            # legacy planning documents
├── sprints/          # archived sprint records
├── SPRINT.md         # current sprint tracking
└── feature-matrix.md # cross-platform feature compatibility
```

Do NOT create a top-level `docs/` directory. All docs go under `specs/docs/`.

---

## Source Layout

```
crates/
├── hologram-core/      # LUT tables, ElementWiseView, ring algebra, encoding (no_std)
│   └── src/
│       ├── lib.rs
│       ├── buffer/     # StaticBuf for no_alloc environments
│       ├── datum/      # Datum type definitions
│       ├── encoding/   # Angle, raw, signed, unsigned encodings
│       ├── error/      # Error types
│       ├── lut/        # LUT tables (activation, arith, q0)
│       ├── op/         # LutOp, PrimOp, Op trait
│       ├── q1/         # Q1 (16-bit) operations
│       ├── ring/       # ByteRing algebra
│       └── view/       # ElementWiseView + SIMD paths
├── hologram-graph/     # Graph IR, subgraphs, fusion, scheduling
│   └── src/
│       ├── builder.rs
│       ├── constant.rs
│       ├── fusion.rs
│       ├── graph/
│       ├── schedule.rs
│       └── subgraph.rs
├── hologram-archive/   # .holo format, rkyv, mmap loading
│   └── src/
│       ├── checksum/
│       ├── entrypoint/
│       ├── format/
│       ├── loader/
│       ├── section/
│       ├── weight/
│       └── writer/
├── hologram-exec/      # KV-lookup executor, buffer arena, parallel execution
│   └── src/
│       ├── buffer/
│       ├── eval/
│       ├── kv/
│       ├── lut_gemm/
│       ├── mmap/
│       └── parallel/
├── hologram-compiler/  # Compilation pipeline
├── hologram-async/     # Async wrappers (tokio)
├── hologram-cli/       # CLI subcommands
├── hologram-ffi/       # C ABI + WASM bindings
└── hologram-bench/     # Criterion benchmarks
```