# ADR-0011: Sandbox Execution Contract, Constraint-Based Placement, and Isolation Boundary

- Status: Accepted
- Date: 2026-03-07
- Owners: Architecture

## Context

`hologram-sandbox` needs a formal execution contract defining how it consumes
hologram types, how workloads are placed across isolation backends, and how
data crosses isolation boundaries. Several existing documents
(`docs/src/content/docs/ecosystem/hologram-sandbox.md`, ADR-0001) already
reference this ADR but it was never created.

The core tension: hologram must remain embeddable and sandbox-agnostic, while
`hologram-sandbox` must provide isolation, lifecycle management, scheduling,
and data transport across multiple backend types (process, WASM, microVM).
The execution contract must be thin enough that hologram never needs to know
about sandboxing, yet rich enough that backends can execute workloads without
re-implementing graph dispatch.

## Decision

### 1. Execution contract: SandboxBackend wraps KvExecutor

Sandbox backends implement a `SandboxBackend` trait defined in
`hologram-sandbox-core`. Each backend wraps `hologram::KvExecutor::execute()`
internally. The sandbox NEVER re-implements graph execution — it provides
isolation, lifecycle management, and data transport. Execution always
delegates to hologram's `KvExecutor`.

### 2. No parallel types

`hologram-sandbox` must not define types that duplicate hologram types. No
custom graph IR, no custom execution plan, no custom buffer arena, no custom
archive format. This extends the consumer contract established in
`hologram/architecture.md` section 2a.

### 3. Constraint-based placement, not linear escalation

Backends are execution targets with advertised capabilities — not steps in
an escalation ladder. Workloads declare hard constraints (must be satisfied)
and soft preferences (ranked by desirability). The scheduler filters eligible
backends by hard constraints, ranks by soft preferences, and places.

A single graph can span multiple backends if per-node constraints require it.

### 4. Two-source constraint model

Default constraints may be baked into the `.holo` archive at compile time
(via `HoloWriter::add_section()` custom section). Runtime constraints
provided via API override or augment the `.holo` defaults. The scheduler
merges both sources — runtime overrides take precedence.

Two levels within each source: workload-level defaults and per-node
overrides.

### 5. Data transport across isolation boundaries

Data crosses isolation boundaries via:

- `.holo` archives (`ConstantStore`, `ExecutionSchedule`) — always, for all
  backends
- rkyv-serialized `GraphInputs` / `GraphOutputs` — at process and VM
  boundaries
- Shared memory regions — where the isolation tier permits (process sandbox,
  same-host microVM with virtio-fs)

`BufferArena` is always created fresh inside the sandbox. It is never shared
across isolation boundaries.

### 6. hologram-sandbox-types has no hologram dependency

The `hologram-sandbox-types` crate defines `SandboxId`, `IsolationLevel`,
`BackendCapabilities`, `ResourceConstraints`, and `PlacementPreferences`.
It has no dependency on `hologram`. This allows scheduling logic to be
reasoned about independently of the execution engine.

### 7. Backend selection is explicit

There is no implicit auto-escalation. The caller provides constraints (via
`.holo` defaults, runtime API, or both). The scheduler selects the best
matching backend. If no backend satisfies the hard constraints, execution
fails with a clear error.

### 8. Scheduler crate designed for network extraction

`hologram-sandbox-scheduler` is a separate crate handling local single-node
placement. Its trait interface is designed so a future network scheduler can
implement distributed placement without changing the rest of the stack.

## Consequences

Positive:
- Fills the missing ADR referenced by ecosystem docs
- Establishes a thin, stable contract surface between hologram and sandbox
- Prevents sandbox backends from duplicating execution logic
- Makes data transport model explicit — rkyv at boundaries, zero-copy where possible
- Constraint-based placement is more expressive than linear escalation
- Two-source constraints allow both compile-time defaults and runtime flexibility
- Scheduler isolation enables future network scheduling without architectural change

Negative:
- Requires rkyv serialization at isolation boundaries (not zero-cost, but consistent with hologram's rkyv-only stance)
- Constraint model adds complexity vs simple backend selection
- Per-node placement across multiple backends requires subgraph partitioning logic

## Alternatives Considered

- **Thick sandbox** that re-implements execution internally — rejected: violates no-parallel-types rule
- **Implicit auto-escalation** based on workload analysis — rejected: system cannot reliably determine isolation requirements from workload alone
- **Linear escalation ladder** (embedded → process → WASM → VM) — rejected: does not model real constraint satisfaction; GPU workloads don't fit the linear model
- **Constraints only at runtime** — rejected: compile-time defaults in `.holo` allow model authors to express placement intent
- **Constraints only in .holo** — rejected: operators need runtime override capability for deployment flexibility
- **Separate ADR per sandbox tier** — rejected: the contract is uniform across tiers
