# ADR-0008: hologram-compiler is invoked after lowering

**Status:** Accepted
**Date:** 2026-03-06
**Deciders:** hologram-ai architecture team

---

## Context

The hologram-ai pipeline has two optimization stages that are easy to conflate:

1. **AiGraph optimization** — passes in `hologram-ai-common` that transform the
   AI-semantic IR before lowering.
2. **hologram-compiler** — the generic graph compiler in the `hologram` crate that
   operates on `hologram::Graph` (byte-domain IR) after lowering.

The question: does `hologram-ai` need its own post-lowering optimization passes, or
does `hologram::compile()` cover that?

---

## Decision

**`hologram-ai` invokes `hologram::compile(graph)` immediately after `lower()` returns.**

`MemoryPlanner` in `hologram-ai-common` is scoped exclusively to KV-cache layout
(`KvCacheLayout`). Intermediate activation buffer reuse is delegated entirely to
`hologram-compiler`.

---

## Two Complementary Layers

### Layer 1 — Pre-lowering: `hologram-ai-common` opt passes

Run on `AiGraph` (AI-semantic IR). These passes require semantic knowledge of model
architecture that `hologram-compiler` does not possess:

| Pass | What it does |
|------|-------------|
| `AttentionFusion` | Detects Q/K/V projection subgraphs → `MultiHeadAttention` op |
| `GroupedQueryAttentionFusion` | GQA variant |
| `FfnFusion` | gate×up→silu subgraph → `FusedSwiGLU` op |
| `QuantMatMulFusion` | `Dequantize → MatMul` → `QuantizedMatMul` (when backend supports it) |
| `ShapePropagation` | Infer concrete shapes required for lowering |
| `ConstantFolding` | Fold shape arithmetic at compile time |
| `DeadNodeElimination` | Remove unreachable nodes |

`hologram-compiler` never sees `AiOp` variants and cannot perform any of these.

### Layer 2 — Post-lowering: `hologram::compile(graph)`

Run on `hologram::Graph` (byte-domain `GraphOp` IR). These are generic and
AI-agnostic:

| Pass | What it does |
|------|-------------|
| Constant folding | Fold compile-time-known `Constant` chains |
| LUT chain fusion | Adjacent unary LUT ops → single `FusedView` (zero allocations) |
| CSE | Common subexpression elimination |
| Liveness analysis | Tensor live intervals for buffer reuse |
| Workspace slot reuse | First-fit-decreasing bin packing of intermediate buffers |

`hologram-ai` must NOT re-implement these. Any overlap would produce incorrect results
(double optimization) or wasted effort.

---

## Pipeline (corrected)

```
AiGraph (raw)
  → hologram-ai-common opt passes    (semantic AI fusions)
  → KvCacheLayout                    (KV sizing; MemoryPlanner scope)
  → lower(graph, kv_layout, opts)    → LoweringOutput { graph, registry }
  → hologram::compile(lower.graph)   → CompilationOutput { archive, schedule, stats }
  → CompiledModel { archive, schedule, registry, kv_layout }
```

---

## Consequences

- `LoweringOutput` contains `graph: hologram::Graph` and `registry: hologram::CustomOpRegistry`
  — no `ExecutionSchedule`. The schedule is produced by `hologram::compile()`.
- `hologram-ai` (the facade crate) depends on `hologram` with `features = ["compiler"]`.
- `hologram-ai-common` does NOT need the compiler feature — it only builds a `Graph`.
- `MemoryPlanner` is renamed / narrowed to `KvCachePlanner` to reflect its actual scope.
- `CompiledModel` stores `Arc<Vec<u8>>` (archive) alongside `Arc<ExecutionSchedule>`.
  The archive enables future serialization / disk caching of compiled models.

---

## Alternatives Considered

**A: hologram-ai re-implements post-lowering passes.**
Rejected. Duplicates hologram-compiler work. Divergence over time would be inevitable.

**B: Skip hologram-compiler; execute Graph directly.**
Rejected. LUT chain fusion and workspace reuse are significant optimizations. Skipping
them leaves performance on the table and requires hologram-ai to allocate more buffers.

**C: Merge both optimization layers into one.**
Rejected. The two layers operate on fundamentally different IRs (`AiGraph` vs
`hologram::Graph`) and have different semantic domains. Merging would require
`hologram-compiler` to understand AI concepts, violating its AI-agnostic design.
