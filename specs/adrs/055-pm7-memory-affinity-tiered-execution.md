# ADR-055: PM_7 Memory Affinity — Tiered Execution

**Status:** Accepted
**Date:** 2026-05-22
**Deciders:** Ari (project lead)
**Related:** ADR-051 (Workspace Residency), ADR-043 (LUT-Addressed Transform Chains)

## Context

Apple Silicon's unified memory lets CPU and GPU share physical memory with
zero copies. For discrete GPUs (CUDA, WebGPU on non-Apple hardware), data
must be explicitly uploaded and downloaded between host and device memory.
ADR-051 reduced transfer cost from O(size × N_calls) to O(size × 2) via
workspace residency, but the remaining transfers still dominate latency for
small-to-medium models where GPU launch overhead exceeds compute time.

Meanwhile, Hologram's LUT-based architecture makes many operations
CPU-native at O(1): Q0 (256-byte tables in L1) and Q1 (128KB tables in L2)
don't benefit from GPU parallelism at all. Sending these ops to a GPU wastes
transfer bandwidth and adds launch latency.

The fundamental insight: **the compiler already knows each operation's
quantum level (Witt bit-width)**. This determines whether an op is LUT-
accelerable (CPU-native) or algorithmic (GPU-beneficial).

## Decision

Introduce **Prism Identity PM_7 (Memory Affinity)**: the memory tier of a
datum is determined by its quantum level.

### Tier Assignment

```rust
pub enum MemoryTier {
    CpuL1   = 0,  // Q0 (≤8-bit): 256B LUT, L1-resident
    CpuL2   = 1,  // Q1 (9–16-bit): 128KB LUT, L2-resident
    CpuMain = 2,  // Q2 (17–24-bit): ~50MB segmented tables
    Device  = 3,  // Q3+ (≥25-bit): GPU/accelerator dispatch
}
```

The compiler assigns one tier per `KernelCall` based on `witt_bits`,
`element_count`, and whether the op is layout-only. Layout ops always stay
on CPU regardless of bit-width. Small element counts (< 1024) stay on CPU
because GPU launch overhead dominates.

### Archive Integration

A new optional section `TierAssignments` (kind = 13) stores one byte per
kernel call. Archives without this section default all calls to `CpuMain`
(backward compatible, same behavior as before). `FORMAT_VERSION` remains 1.

### Coherence Protocol

PL_2 (Lease Disjointness) guarantees that within a schedule level, each
buffer slot has exactly one writer. This means ownership conflicts are
impossible intra-level, and coherence reduces to a **precomputed static
migration schedule** at level boundaries — no runtime page-fault handling.

```rust
struct LevelMigration {
    cpu_to_device: Vec<u32>,  // slots to upload before this level
    device_to_cpu: Vec<u32>,  // slots to download before this level
}
```

On unified-memory hardware (Apple Silicon), all migrations are no-ops.

### Execution

`HybridExecutor` routes calls by tier:
- `CpuL1 | CpuL2 | CpuMain` → CPU backend
- `Device` → GPU/accelerator backend

For single-backend sessions (CPU-only, or Metal where the same backend
handles both), the tier routing is informational — correctness is identical
to `Executor::run_levels`. The infrastructure exists so that adding a
discrete GPU backend requires only wiring a second backend instance.

### W16 LUT Expansion

For Q1-tier ops (witt_bits 9–16), unary activations are served via
65536-entry lookup tables (128KB each, L2-resident). Tables are built once
per activation and cached for process lifetime via `OnceLock`. Supported
activations: Relu, Sigmoid, Tanh, Gelu, Silu, Neg, Abs, Exp.

Fused unary chains at W16 compose their LUTs into a single table, providing
O(1) per-element evaluation for the entire chain.

## Formal Grounding

| Property | Basis |
|----------|-------|
| Determinism | Witt level resolved at compile time (I-6). Tier is a pure function of `witt_bits`. |
| Coherence | PL_2 (lease disjointness) → no intra-level conflicts. |
| O(1) resolution | Table lookup on `witt_bits` — constant time. |
| Completeness | Every datum maps to exactly one tier; union of tiers covers all datums. |
| Zero-cost on unified memory | Metal backend detects `has_unified_memory()` → all migrations are no-ops. |

## Consequences

### Positive

- On unified-memory hardware: zero overhead (metadata-only, no data copies).
- On discrete GPU: transfers reduced to the theoretical minimum (only slots
  that cross the CPU/Device boundary at level boundaries).
- Q0/Q1 ops never leave CPU — no GPU launch overhead for LUT-accelerable ops.
- Feature-gated (`tiered-exec`) — zero cost when not enabled.
- Backward compatible — old archives work unchanged.

### Negative

- One byte of archive overhead per kernel call.
- W16 LUT tables consume 128KB of resident memory per cached activation.
- Multi-backend dispatch requires `Clone` on backends (future work).

### Neutral

- `InferenceSession` gains two fields (`tiers`, `migrations`) behind `cfg`.
- Existing tests continue to pass unmodified.
