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

### Tier resolution (recomputed at load — no archive change)

Because a tier is a **pure function of the kernel's quantum level**
(`MemoryTier::from_witt(witt_bits, element_count, is_layout_only)`), and the
quantum level (dtype bit-width), element count (output `BufferRef` length), and
layout-only flag are all recoverable from the *decoded* kernel calls, the
session recomputes the per-kernel tiers at load. No archive section, no format
change, backward-compatible by construction — the same inputs that determined
the tier at compile time are present at load. (An earlier draft serialized a
`TierAssignments` section; it was dropped as redundant with the pure-function
property — the uor-native stance is that a derivable quantity is recomputed,
not stored.)

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

What ships and acts today is the **Q0/Q1 LUT acceleration** below — the
concrete realization of the `CpuL1`/`CpuL2` tiers — plus the load-time tier
classification + observability (`tier_report`). These are wired and exercised.

Device routing (dispatching `Device`-tier ops to a discrete accelerator) is a
**future extension**, not shipped here: there is no second backend to route to
on CPU, so a `MigrationBackend`/`HybridExecutor` scaffold would be dead code.
When a real device backend lands, it consumes the existing substrate — the
per-call tiers + the coherence migration schedule — to decide uploads/downloads
at level boundaries; no new tiering machinery is needed. We do not carry an
unused trait in the meantime (no skeleton).

### Q1 LUT-accelerated activations (realized)

The Q1 (CpuL2) tier is concretely realized for **IEEE f16/bf16** — the 16-bit
quantum levels main already executes. A transcendental unary activation over a
16-bit domain has only 65536 possible inputs, so it is materialized once as a
`[u16; 65536]` table (128 KB, L2-resident) whose entry is `narrow(f(widen
(bits)))` — the content-addressed, compute-once form of the function over that
finite quantum domain (the UOR materialize-and-reuse principle applied to a
function). At dispatch a low-precision Sigmoid/Tanh/Gelu/Silu/Exp/Erf becomes a
single table lookup instead of `widen → transcendental → narrow`.

This is **bit-identical** to the compute path (same f32 evaluation, precomputed)
— a pure speedup, validated against the f64 reference. Always-on (not gated):
any f16/bf16 transcendental activation takes the LUT path; f32 still computes
(a 4 GB table is infeasible — f32 is the Device/Q3+ tier). Measured: **bf16
GELU over 1M elements is ~28× faster** (743 µs LUT vs 20.7 ms compute). The
table cache uses `OnceLock` (std); under no_std the activation is computed (a
compile-time choice, not a runtime fallback).

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
