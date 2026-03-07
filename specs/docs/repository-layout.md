# Repository Layout — hologram

## Top-Level Structure

```
hologram/
AGENTS.md         # agent coding rules (holoarch-managed section + project rules)
CLAUDE.md         # Claude Code instructions
Cargo.toml        # workspace root
Cargo.lock        # locked dependencies
Justfile          # build recipes
cliff.toml        # commit message parsing (changelog)
.holoarch.toml    # holoarch config
.githooks/        # git hooks (format, lint)
README.md         # user guide
specs/            # all project documentation
src/              # root crate (re-exports + CLI entry)
crates/           # workspace member crates
tests/            # integration tests
site/             # documentation website (pnpm)
```

---

## specs/ Layout

```
specs/
docs/             # project documentation (managed by holoarch)
  architecture.md
  upstream-architecture.md
  development.md
  repository-layout.md
  testing.md
  performance.md
  runtime.md
  validation.md
  data-model.md
  cli.md
  release.md
plans/            # planning documents and cross-repo specs
SPRINT.md         # current sprint tracking
sprints/          # archived sprints
```

Do NOT create a top-level `docs/` directory. All docs go under `specs/docs/`.

---

## Source Layout

### Root Crate (`src/`)

```
src/
lib.rs            # unified API re-exporting all subcrates
main.rs           # CLI entry point (feature-gated)
```

### Workspace Crates (`crates/`)

```
crates/
hologram-core/              # LUT tables, encoding, primitives (no_std)
  src/
    lib.rs
    op/                     # LutOp (21+ activations), PrimOp (10 primitives)
    view/                   # ElementWiseView composition
    lut/                    # precomputed activation tables
    encoding/               # Angle, Signed, Unsigned, Raw encoding
    ring/                   # ByteRing modular arithmetic
    buffer/                 # buffer types
    datum/                  # datum types
    quantum/                # Q0/Q1 quantization

hologram-graph/             # expression graph, fusion, scheduling
  src/
    lib.rs
    graph/                  # Graph (arena-based), validation
    node/                   # NodeId, edge management
    edge/                   # connectivity queries
    builder/                # GraphBuilder fluent API
    constant/               # ConstantStore for inline weights
    subgraph/               # SubgraphDef templates
    fusion/                 # constant folding, view fusion, CSE
    schedule/               # topological sort, parallel levels

hologram-archive/           # .holo format, serialization
  src/
    lib.rs
    format/                 # HoloHeader, magic, alignment
    entrypoint/             # LayerHeader, TensorPort
    loader/                 # HoloLoader (mmap), LoadedPlan
    writer/                 # HoloWriter builder
    checksum/               # CRC32 validation
    section/                # section table
    layer/                  # layer metadata
    weight/                 # WeightDType, weight storage

hologram-exec/              # KV executor, LUT-GEMM kernels
  src/
    lib.rs
    eval/                   # KvExecutor, schedule_bridge
    kv/                     # KvStore, CustomOpRegistry
    buffer/                 # BufferArena
    parallel/               # Rayon integration
    lut_gemm/               # Q4/Q8 matrix kernels
    mmap/                   # execute_file(), execute_bytes()

hologram-compiler/          # compilation pipeline
  src/
    lib.rs
    compiler/               # CompilerBuilder (parse → fuse → emit)
    liveness/               # liveness interval computation
    workspace/              # buffer reuse planning

hologram-async/             # Tokio async wrappers
  src/
    lib.rs
    compiler/               # AsyncCompiler
    executor/               # AsyncExecutor
    stream/                 # token-streaming execution

hologram-ffi/               # C ABI + WASM bindings
  src/
    lib.rs
    compiler/               # C bindings for compilation
    graph/                  # C bindings for graph construction
    exec/                   # C bindings for execution
    encoding/               # C bindings for encoding
    error/                  # thread-local error stack
    handle/                 # opaque handle management
    wasm/                   # wasm-bindgen (feature-gated)

hologram-cli/               # CLI subcommands
  src/
    lib.rs
    commands/               # compile, run_cmd, inspect
      inspect/              # detail levels
    fmt/                    # output formatting
    error/                  # CLI error types

hologram-bench/             # Criterion benchmarks
  benches/
    lut.rs
    view.rs
    kv_dispatch.rs
    executor.rs
    lut_gemm.rs
    compiler.rs
    fusion.rs
    archive.rs
    q1.rs
    async_exec.rs
    async_stream.rs
    ffi.rs
```

### Integration Tests (`tests/`)

```
tests/
e2e.rs                      # full pipeline tests
custom_ops.rs               # custom operation tests
```