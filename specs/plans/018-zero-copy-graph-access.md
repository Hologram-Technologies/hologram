# Plan 018: Zero-Copy Graph Access

## Context

Graph deserialization takes 1.5s per call for a 199MB graph (TinyLlama).
With the pipeline, we parse the graph 3 times during loading (probe, plan,
shape_ctx). This is the last major bottleneck after zero-copy weight loading.

Two independent improvements:
1. **Skip graph compression** — like weights, store graphs uncompressed
2. **rkyv::access instead of from_bytes** — zero-copy archived field access

## Phase 1: Optional graph compression

The graph is always compressed (HoloWriter lines 121-123). Make it optional
like weights. For local inference, skip compression for instant access.

### Changes

**`crates/hologram-archive/src/writer/holo_writer.rs`:**
- Add `compress_graph: bool` field (default false, matching weights)
- Add `.compress_graph()` opt-in method
- Skip compression when `compress_graph == false`

## Phase 2: rkyv::access for zero-copy graph

Replace `rkyv::from_bytes<SerializedGraph>()` (full deserialization into owned
types) with `rkyv::access<ArchivedSerializedGraph>()` (zero-copy pointer into
mmap'd bytes).

### Key type changes

`LoadedPlan::graph` changes from `SerializedGraph` (owned) to either:
- `GraphAccess::Owned(SerializedGraph)` — compressed archives
- `GraphAccess::Archived(AlignedVec)` — uncompressed, access on-demand

The `graph()` method needs to work with both. Since `ArchivedSerializedGraph`
mirrors field layout (ArchivedVec derefs to slice, ArchivedString derefs to str),
most downstream code works unchanged.

## Verification

```bash
cargo test -p hologram-archive -p hologram-exec
cargo bench -p hologram-bench -- archive::load
```
