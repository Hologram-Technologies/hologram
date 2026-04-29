# ADR-051: Canonical Backend Workspace Residency

**Status:** Accepted
**Date:** 2026-04-28 (accepted 2026-04-29)
**Deciders:** Ari (project lead)
**Related:** ADR-043 (LUT-Addressed Transform Chains), ADR-050 (Canonical-as-Semantic-Contract), PR #6 (canonical wgpu backend landing)

## Context

The canonical backend layer landed in PR #6 with the
[`CanonicalBackend`](../../crates/hologram-transform/src/backend.rs) trait:

```rust
pub trait CanonicalBackend {
    fn dispatch(&mut self, storage: &mut [f32], call: &KernelCall) -> Result<(), ExecError>;
    fn run(&mut self, storage: &mut [f32], calls: &[KernelCall]) -> Result<(), ExecError> { … }
    fn flush(&mut self) -> Result<(), ExecError>;
    fn name(&self) -> &'static str;
}
```

`storage: &mut [f32]` is a host-side workspace. `CpuBackend` reads/writes
that slice directly — zero overhead, cache-friendly. `WgpuBackend` (the
device backend) treats the slice as a *staging area*: every dispatch
uploads its inputs to fresh device buffers, dispatches the compute
pipeline, and reads results back into the host slice.

For a 100-call plan that's **100 sets of round-trips per execution**.
On a Mac M-series, each round-trip is ~µs-scale per kernel for
small ops and ms-scale for large ones, but the overhead dominates the
compute work for everything except MatMul/Conv at scale. The
`WgpuBackend` is correct (51 variants conformance-validated against
`CpuBackend`) but in production it's slower than `CpuBackend` on most
plans because of the per-call transfer cost.

The intent at landing was: ship correctness first, address residency
in a follow-up. This ADR is that follow-up.

## Decision

Augment `CanonicalBackend` with an opt-in **device-resident workspace
type**, keeping the existing `&mut [f32]` API for the simple case
(tests, conformance harness, host-only backends).

### Trait shape

```rust
pub trait CanonicalBackend {
    /// Backend-owned workspace handle. CPU = `Vec<f32>` (or a wrapper
    /// over a borrowed slice); wgpu = device-side storage buffer.
    type Workspace: BackendWorkspace;

    /// Allocate a workspace sized for the plan. Called once before
    /// `dispatch_resident` runs.
    fn alloc_workspace(
        &self,
        total_elements: usize,
    ) -> Result<Self::Workspace, ExecError>;

    /// Per-call dispatch operating on the device-resident workspace.
    /// Replaces the existing `dispatch(&mut [f32], …)` for backends
    /// that opt in.
    fn dispatch_resident(
        &mut self,
        ws: &mut Self::Workspace,
        call: &KernelCall,
    ) -> Result<(), ExecError>;

    /// Existing per-call API — kept for simple use cases.
    /// Default impl wraps the resident path: alloc + write_span +
    /// dispatch_resident + read_span. Backends free to override.
    fn dispatch(
        &mut self,
        storage: &mut [f32],
        call: &KernelCall,
    ) -> Result<(), ExecError> { … default … }

    fn run_resident(
        &mut self,
        ws: &mut Self::Workspace,
        calls: &[KernelCall],
    ) -> Result<(), ExecError> {
        for call in calls { self.dispatch_resident(ws, call)?; }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ExecError>;
    fn name(&self) -> &'static str;
}

/// Host ↔ workspace bridging operations.
pub trait BackendWorkspace {
    /// Bytes (or elements) currently held by the workspace.
    fn capacity(&self) -> usize;

    /// Stamp host data into a workspace slot at the given span.
    /// Used at plan boundaries (graph inputs / output read-back).
    fn write_span(
        &mut self,
        span: SlotSpan,
        data: &[f32],
    ) -> Result<(), ExecError>;

    /// Read a span back to host. For wgpu this triggers a single
    /// staging-buffer round-trip per call site (planner / executor
    /// schedules these at plan boundaries, not per dispatch).
    fn read_span(&self, span: SlotSpan) -> Result<Vec<f32>, ExecError>;
}
```

### Concrete implementations

- **`CpuBackend`**: `type Workspace = CpuWorkspace`, where
  `CpuWorkspace(Vec<f32>)` is just an owned slice. `write_span` /
  `read_span` are slice copies. `dispatch_resident` is the same body
  as today's `dispatch` but takes `&mut workspace.0` instead of
  `&mut [f32]`.

- **`WgpuBackend`**: `type Workspace = WgpuWorkspace`, where
  `WgpuWorkspace` owns a single `wgpu::Buffer` of size
  `capacity * 4` bytes plus a small staging buffer for read-back.
  Every `dispatch_resident` arm builds a bind group whose entries are
  *views* (`BufferBinding { buffer, offset, size }`) into the
  resident workspace at the call's `SlotSpan` offsets. No per-call
  upload/download — host ↔ device transfers happen only via
  `write_span` / `read_span`.

### Executor flow

```rust
let mut ws = backend.alloc_workspace(plan.workspace.total_elements)?;
seed_inputs(&mut ws, plan, host_inputs)?;     // write_span × N
backend.run_resident(&mut ws, &plan.forward)?;
backend.run_resident(&mut ws, &plan.backward)?; // optional
backend.flush()?;
let out = ws.read_span(plan.address_table.span(output_id))?;
```

The conformance harness (`check_forward`, `check_forward_then_backward`)
gets a parallel `_resident` flavor that uses `alloc_workspace +
write_span + run_resident + read_span` instead of operating on a
host slice directly. Backends opting into residency get conformance
coverage for free.

### Migration

1. **Add the trait machinery** without breaking existing callers.
   `dispatch` / `run` keep their existing signatures (default impls
   wrap `dispatch_resident` for residency-aware backends, fall
   through to direct slice access for `&mut [f32]`-only backends).
2. **CpuBackend** gets the trivial `CpuWorkspace` impl. No
   behavioural change.
3. **WgpuBackend** gets `WgpuWorkspace`. Each existing
   `dispatch_<op>` method is split into:
   - A *bind-group-construction* path that takes `BufferBinding`s
     for inputs/outputs (fed by either `upload_*` for the per-call
     path or workspace views for residency).
   - A *compute-pass-recording* path identical to today.
   The 50+ existing dispatch arms are mechanical to migrate — most
   are 5-line wrappers around `dispatch_and_read_with_workgroups`.
4. **Conformance harness** gains a `check_*_resident` family. The
   existing host-slice harness keeps working for backends that
   don't opt in.
5. **Executor** gains a `run_resident_with` companion to
   `run_*_with`.

### Out of scope

- **Multi-workspace plans**: a single `CompiledPlan` produces a
  single resident workspace today. Plans that need cross-device
  splits or sub-workspaces are out of scope for this ADR.
- **Eviction / paging**: `WgpuWorkspace` allocates `capacity * 4`
  bytes contiguously. Plans whose workspace exceeds device VRAM
  fail at `alloc_workspace`. Eviction is a future concern (would
  need a paging metadata layer in `CompiledPlan`).
- **Cross-backend workspaces**: each `Workspace` is owned by one
  backend; transferring between backends requires explicit host
  round-trip. Cross-device hand-off is also out of scope.

## Consequences

### Positive

- WgpuBackend per-call upload/download cost goes from O(transfer
  size × N calls) to O(transfer size × 2) (input seed + output
  read). For typical plans this is the difference between
  WgpuBackend being slower than CpuBackend and being competitive
  with native compute libraries.
- Sets the contract for any future device backend (Metal, CUDA,
  Atlas ISA): the Workspace type is owned by the backend and
  device-resident.
- The `BackendWorkspace::write_span` / `read_span` API is the
  natural seam for streaming inputs (e.g. KV cache prefill) and
  pipelined outputs.

### Negative

- API surface grows. `CanonicalBackend` gains an associated type;
  users that hold trait objects need a different abstraction
  (probably a `CanonicalBackendDyn` type that erases `Workspace`).
- Migrating WgpuBackend's 50+ dispatch arms is a large but
  mechanical change. Each arm needs the bind-group construction
  refactored to support both upload-and-bind-fresh (legacy `dispatch`
  path) and bind-view-of-resident (new path).
- The conformance harness needs a parallel resident flavor.
  Existing tests stay; new resident tests duplicate the structure.

### Alternatives considered

1. **Keep the existing trait, add a parallel `execute_plan(plan,
   host) -> Result<()>` method on each backend.** Less invasive but
   leaves the device-resident shape implicit, and every backend has
   to invent the same workspace pattern independently.
2. **Replace the host-slice API entirely with the resident type.**
   Cleaner long-term but breaks every test, the conformance
   harness, and the convenience of `CpuBackend::dispatch(&mut
   storage, &call)`. Too disruptive given canonical layer just
   landed.
3. **Hold the current `&mut [f32]` API and add an `unsafe` "I
   promise this slice is device memory" extension trait.** Unsound
   abstraction — host code would deref a device pointer.

## Implementation plan

Tracked separately. Skeleton:

1. Add `BackendWorkspace` trait + `Workspace` associated type +
   default `dispatch` impl (one PR; no behaviour change).
2. `CpuBackend`'s `CpuWorkspace`, opting into the resident path;
   conformance harness gains `_resident` flavors validating
   `CpuBackend`-resident == `CpuBackend`-host (one PR).
3. `WgpuBackend`'s `WgpuWorkspace` + per-arm migration. May split
   across multiple PRs by op family (binary, unary, reductions,
   …), each gated on conformance harness success (multiple PRs).
4. Update integration tests + benchmarks to use the resident path
   for `WgpuBackend`. Document the perf delta.

## References

- [`hologram-transform/src/backend.rs`](../../crates/hologram-transform/src/backend.rs)
  — current trait + `CpuBackend` impl.
- [`hologram-backend/src/canonical/wgpu.rs`](../../crates/hologram-backend/src/canonical/wgpu.rs)
  — current `WgpuBackend` (per-call upload/download).
- [`hologram-transform/src/conformance.rs`](../../crates/hologram-transform/src/conformance.rs)
  — cross-backend conformance harness.
- ADR-043 §"Address layer" — `SlotSpan` / `AddressTable`
  architecture.
- ADR-050 — backend semantics conform to canonical CPU; this ADR
  preserves that conformance via the parallel harness.
