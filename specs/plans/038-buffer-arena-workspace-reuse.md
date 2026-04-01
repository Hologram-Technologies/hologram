# Plan 038: BufferArena Workspace Reuse

## Context

The compiler's `plan_workspace()` computes optimal buffer slot assignments: nodes
with non-overlapping lifetimes share the same physical buffer slot, reducing peak
memory. But this information is NOT wired into the executor's `BufferArena` — every
node currently gets its own allocation. This wastes 20-40% of peak memory.

## Current State

**Compiler side** (`hologram-compiler/src/workspace/mod.rs`):
- `WorkspaceLayout { slots: Vec<BufferSlot>, assignments: Vec<(NodeId, u32)> }`
- `BufferSlot { slot_id: u32, occupants: Vec<NodeId> }` — nodes sharing a slot
- `plan_workspace(intervals: &[LivenessInterval]) -> WorkspaceLayout`
- Tested: 5 tests covering empty, single, non-overlapping, overlapping intervals

**Executor side** (`hologram-exec/src/buffer/arena.rs`):
- `BufferArena { buffers: Vec<Option<ArenaBuffer>>, ... }` — flat Vec indexed by NodeId
- Each node gets its own index, its own allocation
- `swap_insert_with_elem_size()` allocates MmapBuffer or Vec per node
- `evict()` drops the buffer, freeing memory

## What Needs to Change

### 1. Thread WorkspaceLayout through to EnumTape

`WorkspaceLayout.assignments` maps `NodeId -> slot_id`. This mapping needs to
reach `execute_inner()` so the arena knows which physical slot to use.

- Add `workspace_assignments: Vec<u32>` to `EnumTape` (indexed by node index,
  value = slot_id, or `u32::MAX` for unassigned)
- Populate during `build_tape()` from the compiler's liveness analysis

**Files**: tape.rs (EnumTape struct), tape_builder.rs (build_tape)

### 2. Add Slot Aliasing to BufferArena

When two nodes map to the same slot_id, the second node reuses the first node's
physical buffer (resizing if needed). This requires:

- Add `slot_map: Vec<u32>` to BufferArena (node_index -> slot_id)
- Modify `swap_insert_with_elem_size()`: when `slot_map[idx] != u32::MAX`,
  look up the slot's current physical buffer and reuse it (resize if larger)
- Modify `evict()`: only free the physical buffer if no other node is using it
- Modify `get()/get_mut_f32()`: redirect to the slot's physical buffer

**Key invariant**: Two nodes sharing a slot never have overlapping lifetimes
(guaranteed by `plan_workspace`), so the buffer contents are always valid for
the current occupant.

**Files**: buffer/arena.rs

### 3. Pre-Allocate Slot Buffers at Tape Start

`prewarm_arena()` already pre-allocates output buffers using `output_byte_hint`.
Extend this to pre-allocate one physical buffer per slot (sized to the maximum
occupant's `output_byte_hint`):

```
for slot in workspace_layout.slots {
    let max_size = slot.occupants.iter()
        .map(|node_id| tape.instructions[node_id].output_byte_hint)
        .max();
    arena.allocate_slot(slot.slot_id, max_size);
}
```

This eliminates all per-instruction allocations — the hot loop just writes
into pre-existing buffers.

**Files**: tape.rs (prewarm_arena)

### 4. Handle F16 Compression + Slot Reuse Interaction

When a slot occupant is compressed to F16, the next occupant needs the
full-size f32 buffer. The slot pre-allocation (step 3) handles this: the
physical buffer is always sized to the maximum, so compression/expansion
happens within the slot's allocation.

### 5. Handle Borrowed Buffers

Constants and weights are `Borrowed` (zero-copy from mmap). These should
NOT participate in slot aliasing — they're always available and never freed.
The slot_map should only cover non-borrowed activation buffers.

## Expected Impact

- 20-40% peak activation memory reduction (model-dependent)
- Zero per-instruction allocation overhead (pre-allocated slots)
- No latency regression (slot lookup is O(1) via index)

## Verification

- All existing tests pass (slot reuse is transparent to consumers)
- Peak memory profiling before/after on SD UNet and LLaMA 7B
- New tests: slot aliasing correctness, resize behavior, eviction with shared slots
