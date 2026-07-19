# hologram-graph

> The hologram graph IR: an arena-based DAG of nodes with schedule and registries.

An arena-based DAG where each `Node` carries an `OpKind` (from the closed `hologram-ops` catalog) together with its inputs and dtype/shape metadata. A single `GraphOp` enum unifies all dispatch across the IR.

The crate is the intermediate representation the compiler consumes: it holds the graph structure, constant storage, shape/dtype registries, an execution schedule, and reverse-mode (backward) graph construction.

## What it provides

- `Graph` — the arena-based node DAG.
- `Node`, `NodeId`, `GraphOp`, `InputSource` — the node representation, its id, the unified op enum, and how inputs are sourced.
- Per-op attribute records: `AttentionAttrs`, `ConvAttrs`, `GatherAttrs`, `GemmAttrs`, `LrnAttrs`, `NormAttrs`, `QuantAttrs`, `ReduceAttrs`.
- `append_backward` / `BackwardError` — append a reverse-mode backward subgraph.
- `ConstantStore`, `ConstantId` — constant tensor storage.
- `ShapeRegistry`, `ShapeDescriptor`, `ShapeId`, `DTypeId` — the shape/dtype registries.
- `Schedule` — the execution schedule over the graph.
- `OpKind` — re-exported from `hologram-ops`.

## Targets & build notes

`no_std` with `alloc`. Uses `smallvec` for node input lists and `libm` for scalar math.

Part of the [hologram](../../README.md) workspace.
