# hologram-types

> The hologram domain type vocabulary, expressed as ConstrainedTypeShape declarations.

Hologram is a Prism application (wiki ADR-031). The canonical shape carriers — `MatrixShape<R, C, E>`, `VectorShape<N, E>`, `Digest<N>` — are imported from the prism standard-library Layer-3 sub-crates (`prism::tensor`, `prism::crypto`) and reach hologram callers through this crate's re-exports.

Beyond the prism façade, this crate declares only the vocabulary that is genuinely hologram-specific: dtype markers, the low-rank type-level shape markers used by the graph IR, and the σ-axis (substitution-axis) host selections absorbed from the former `hologram-host` crate. It declares no parallel duplicates of anything prism already provides.

## What it provides

- `DType` trait plus `DTypeF32` / `DTypeF16` / `DTypeBf16` / `DTypeF64` / `DTypeI64` / `DTypeI32` / `DTypeI8` / `DTypeI4` / `DTypeU64` / `DTypeU8` / `DTypeBool` markers, each carrying `BIT_WIDTH` and `KIND` (`DTypeKind`) for per-op dtype resolution.
- `DTypeId` — the canonical runtime dtype tag, re-exported by the graph registry and the backend rather than respelled.
- `weight_layout` / `act_quant` — weight-slot declaration vocabularies for how a later-bound quantized weight is laid out and which activation treatment it opts into.
- `Dim<N>`, `Shape1`, `Shape2` — rank-1 / rank-2 type-level shape markers for the graph IR; higher ranks compose via prism's `partition_product!`.
- `MemoryTier` — memory-tier marker.
- `host::*` — σ-axis selections (`HologramHasher`, `HologramHostTypes`, `ActiveCpuBounds`, per-backend `HologramHostBounds*`) plus the `prism` / `sdk` re-exports, flattened to the crate root.
- `Digest`, `MatrixShape`, `VectorShape` — re-exported canonical prism shape carriers.
- `IRI_PREFIX` — `https://hologram.uor.foundation/type/`, the namespace for hologram-introduced types.

## Targets & build notes

`no_std`. The absorbed σ-axis types are `no_std`; the `std` feature is an empty passthrough kept for crates that flip the whole stack to std.

Part of the [hologram](../../README.md) workspace.
