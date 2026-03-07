# hologram — Project Overview

`hologram` is an O(1) compute acceleration system that replaces iterative
computation with precomputed lookup tables. It provides the core execution
runtime, graph IR, compiler, archive format, and CLI for the Hologram ecosystem.

---

## What it is

A compilation and execution engine. Its job is to:

1. **Represent** computation as a directed graph of operations (`Graph`)
2. **Optimize** the graph via compile-time passes (LUT chain fusion, CSE, constant folding)
3. **Plan memory** — liveness analysis, workspace slot allocation, buffer reuse
4. **Schedule** execution into parallel levels (topological sort → `ExecutionSchedule`)
5. **Emit** optimized `.holo` archives (page-aligned, zero-copy loading)
6. **Execute** graphs via the O(1) key-value dispatch engine (`KvExecutor`)
7. **Extend** with domain-specific operations via `CustomOpRegistry`

---

## What it is not

- Not an AI framework (AI concerns live in `hologram-ai`)
- Not a sandbox system (isolation concerns live in `hologram-sandbox`)
- Not a model format parser (format importers live in consumer projects)
- Not a GPU runtime (execution is CPU-first with SIMD acceleration)

---

## Core Innovation: O(1) Lookup-Based Execution

The pi-F-lambda encoding pipeline:

```
f64 ──[embed: pi]──► u8 ──[LUT: F]──► u8 ──[lift: lambda]──► f64
```

Any unary function is precomputed into a 256-entry lookup table
(`ElementWiseView`). Chains of operations are fused at compile time into a
single table — `sigmoid(relu(gelu(x)))` costs the same as one array lookup.

Properties of `ElementWiseView`:

- 256 bytes = single CPU cache line (64-byte aligned)
- SIMD-accelerated via `vpshufb` (AVX2) or `pshufb` (SSE4.2)
- Composition: O(256) one-time, O(1) per element thereafter

---

## CLI

Hologram ships a command-line interface for compiling, running, and inspecting
`.holo` archives.

```sh
hologram compile --input graph.bin --output build/
hologram inspect build/graph.holo                      # summary
hologram inspect build/graph.holo --detail full         # everything
hologram run build/graph.holo --input 0:deadbeef --input 1:ff
```

The `inspect` command supports varying detail levels (`summary`, `graph`,
`schedule`, `sections`, `weights`, `full`, `json`) via the `--detail` flag.
See [cli.md](cli.md) for the full CLI specification.

---

## Where to read next

| Topic | File |
|-------|------|
| Full architecture | [architecture.md](architecture.md) |
| Crate layout | [crate-layout.md](crate-layout.md) |
| Graph IR | [graph-ir.md](graph-ir.md) |
| Compilation pipeline | [compilation.md](compilation.md) |
| Execution model | [execution.md](execution.md) |
| Archive format | [archive-format.md](archive-format.md) |
| CLI specification | [cli.md](cli.md) |
| Roadmap | [roadmap.md](roadmap.md) |

---

## ADRs

| Number | Decision |
|--------|---------|
| [0001](../../adrs/0001-repo-boundary.md) | Keep hologram sandbox-agnostic and AI-agnostic |
| [0007](../../adrs/0007-hologram-ai-execution-layer.md) | hologram-ai maps to real hologram types |
| [0008](../../adrs/0008-hologram-compiler-invoked-after-lowering.md) | hologram-compiler invoked after lowering |

---

## Design Constraints

The runtime must remain:

- **Sandbox-agnostic** — no direct dependency on process sandboxing, WASM, or microVM implementations
- **Virtualization-agnostic** — no hypervisor-specific code
- **Platform-neutral** — runs on x86_64, ARM, WASM, and bare-metal
- **AI-agnostic** — no AI concepts (attention, tokens, KV-cache, quantization) in the core

Isolation concerns belong in `hologram-sandbox`. AI concerns belong in `hologram-ai`.

---

## Relationship to Subprojects

```
hologram-ai      ──depends on──► hologram  (graph, execution, archives)
hologram-sandbox ──depends on──► hologram  (execution, scheduling)
```

Subprojects consume hologram's types via the root `hologram` crate. They must
not define parallel abstractions (separate graph formats, separate execution
engines, etc.). If a capability exists in hologram, subprojects use it.

**Consumer rules:**

- Hologram is a **read-only dependency** — consumers MUST NOT modify files in
  the `hologram` repository
- Extension is through defined points only: `CustomOpRegistry`,
  `HoloWriter::add_section()`, `GraphBuilder`, `hologram::compile()`,
  `GraphInputs`/`GraphOutputs`
- All types accessed via `hologram::TypeName` — never via internal subcrates
- AI agents in consumer repos MUST NOT edit hologram files unless explicitly
  instructed

See [architecture.md §Consumer Contract](architecture.md) for the full specification.
