Constrained Runtime Profile for Deterministic, Bounded-Memory Execution

Summary

Introduce a constrained execution profile within hologram-exec that:

Executes precompiled tapes deterministically
Uses preallocated memory only (no runtime allocations in hot path)
Streams weights with bounded residency
Restricts supported kernels to a high-performance subset
Reuses workspace memory via static slot assignment
Enables efficient inference on resource-constrained environments

This profile coexists with the full runtime and shares underlying primitives.

Goals
Zero heap allocations during tape execution
Bounded memory usage (activations + weights)
Deterministic execution (no runtime planning)
Minimal runtime branching
High cache locality and sequential IO
Persistent runner reuse across executions
Compatibility with existing tape + arena + mmap systems
Non-Goals
Do not replace the full runtime
Do not redesign tape or compiler from scratch
Do not support all kernels in constrained mode
Do not introduce dynamic graph execution
Do not add runtime heuristics
High-Level Architecture

Add a new module:

crates/hologram-exec/src/constrained/

Structure:

profile.rs
runner.rs
tape_subset.rs
weight_window.rs
residency.rs
region_pack.rs
mod.rs

This module implements a strict execution profile layered on top of existing runtime components.

Core Concepts
ConstrainedProfile

Defines execution limits and policies.

pub struct ConstrainedProfile {
    pub max_weight_bytes: usize,
    pub max_activation_bytes: usize,
    pub weight_policy: WeightPolicy,
    pub backend: BackendSelector,
    pub allow_custom_ops: bool,
    pub allow_fallback_kernels: bool,
}
WeightPolicy
pub enum WeightPolicy {
    FullResident,
    LazyCache,
    BoundedWindow,
    NoCacheStream,
}

Default for constrained mode:

WeightPolicy::BoundedWindow
ConstrainedRunner

Persistent execution object.

pub struct ConstrainedRunner {
    tape: EnumTape,
    arena: BufferArena<'static>,
    weight_window: WeightWindow,
    profile: ConstrainedProfile,
}

Responsibilities:

Prewarm arena once
Reuse buffers across runs
Enforce profile limits
Drive execution loop
WeightWindow

Bounded residency weight manager.

pub struct WeightWindow {
    max_bytes: usize,
    resident: SmallVec<[ResidentWeight; 8]>,
}

Responsibilities:

Load only required weights for upcoming ops
Evict weights after last use
Maintain strict memory cap
Optional prefetch
Constrained Tape Subset

Add validator:

pub fn validate_constrained_tape(tape: &EnumTape) -> ExecResult<()>

Allowed kernels (initial set):

Input / Output / Constant
Add / Sub / Mul
RMSNorm
Softmax
Rotary embedding
Matmul (quantized LUT 2/4/8/16)
Fused matmul + activation
KV read/write
Simple shape-preserving ops

Reject:

Custom ops
GPU-only ops
Unsupported float kernels
Dynamic shape ops without prevalidation
Execution Model

Execution loop:

for op in tape.instructions {
    weight_window.ensure(op.required_weights());
    bind_buffers(op);
    dispatch_kernel(op);
    weight_window.evict(op.released_weights());
}

Constraints:

No allocations
No dynamic dispatch if avoidable
No graph traversal
No constant deserialization outside window
Region-Based Weight Packing (Additive Feature)

Introduce optional packed format:

pub struct PackedWeightSpan {
    pub offset: u64,
    pub len: u32,
    pub region_id: u32,
}

Goals:

Sequential IO
Reduced seek overhead
Natural prefetching
Deterministic eviction

Compiler responsibility:

Group weights by execution region
Emit region-aligned spans

Runtime responsibility:

Read spans via mmap or reader
Load into WeightWindow
Memory Model
Activation Memory
Fully managed by BufferArena
Slot-based reuse
No intermediate allocations
Weight Memory

Controlled by WeightPolicy:

FullResident: load all once
LazyCache: existing behavior
BoundedWindow: constrained mode default
NoCacheStream: load per op
Integration Points

Modify existing code minimally:

Hook into execute_tape
Add alternative entrypoint:
pub fn execute_constrained(...)
Reuse:
EnumTape
BufferArena
TapeKernel dispatch
mmap-backed constants
Implementation Tasks
Phase 1 — Profile + Runner
 Create constrained/profile.rs
 Define ConstrainedProfile + defaults
 Create constrained/runner.rs
 Implement persistent runner
 Add entrypoint API
Phase 2 — Tape Validation
 Create tape_subset.rs
 Implement validation logic
 Add error types for unsupported kernels
 Integrate validation into runner initialization
Phase 3 — Weight Window
 Create weight_window.rs
 Implement bounded residency structure
 Add load/evict API
 Integrate with ConstantStore / mmap reader
 Add memory accounting
Phase 4 — Execution Integration
 Modify execution loop to use WeightWindow
 Ensure zero allocations in hot path
 Inline kernel dispatch where possible
 Add debug assertions for memory limits
Phase 5 — Region Packing (Optional First Pass)
 Define PackedWeightSpan
 Add optional mapping from ConstantId → PackedWeightSpan
 Add sequential read support
 Stub compiler integration
Phase 6 — Testing
 Unit tests for:
tape validation
weight eviction
memory bounds
 Integration test:
run small transformer block
verify bounded memory
 Compare outputs vs full runtime
Phase 7 — Benchmarks

Measure:

peak memory usage
execution latency
weight IO bandwidth
cache hit rate vs window mode

Compare:

full runtime vs constrained runtime
Success Criteria
Execution runs without heap allocation in hot loop
Peak memory is bounded and predictable
Tape execution produces identical outputs
Weight residency never exceeds configured limit
Runtime is measurably faster or equal under constrained memory
Code remains minimal and composable
Future Extensions
Prefetch pipeline for next region
Async weight streaming
SIMD specialization tightening
GPU-constrained profile
UOR-native traversal lowering
Full region-packed archive format
Notes

This is a narrowing and hardening effort, not a rewrite.

The existing tape + arena + mmap system is already correct.

This plan makes it:

smaller
more deterministic
more memory-efficient
easier to deploy in constrained environments