# Plan 075: Compiled Refinement Execution Strategy

**Status:** Prototype implemented
**Created:** 2026-06-01
**Branch:** `plan/refinement-execution-strategy`

## Context

Diffusion language models such as Mercury demonstrate that some workloads
benefit from coarse-to-fine refinement instead of strictly sequential
generation. Hologram should not become a diffusion-model framework or an
agent runtime, but the underlying execution pattern may still be useful:

```text
State0
 -> Pass1
 -> Validate
 -> Pass2
 -> Validate
 -> PassN
 -> Finalize
```

The question is whether this should become a first-class Hologram execution
strategy while preserving the existing constraints:

- [x] Deterministic execution
- [x] Bounded work
- [x] Bounded memory
- [x] Zero-copy state flow where possible
- [x] No runtime graph mutation
- [x] No dynamic scheduling in executor hot paths
- [x] Planner/executor separation
- [x] No new backend virtual dispatch path

## Current Architecture Findings

- [x] Document graph primitives:
  - `GraphOp` is pure dataflow: `Op`, `Input`, `Output`, `Constant`.
  - Canonical operation identity belongs in `hologram-ops::OpKind`.
  - `KernelCall` is the backend dispatch surface, not an orchestration API.

- [x] Document scheduler architecture:
  - Graph scheduling is static and level-based.
  - Current source groups nodes by dependency depth.
  - No runtime graph traversal should be added for refinement.

- [x] Document execution plans:
  - The compiler lowers graph nodes into `KernelCall`s.
  - The archive carries `KernelCalls` plus `ExecPlan`.
  - `InferenceSession` loads and optionally rewrites compiled calls at load time.

- [x] Document state management:
  - `BufferArena` is the content-addressed state store.
  - `execute_addressed` is the existing zero-copy state boundary.
  - Result caching is resident-label reuse, not a separate cache.

- [x] Document optimization phases:
  - Composite desugaring happens before scheduling.
  - Invariant elision happens before scheduling.
  - Load-time fusions rewrite compiled calls before execution.
  - Backward computation is planned as forward `KernelCall`s, not runtime graph traversal.

## Design Decision

Refinement should be a compiled execution strategy in `hologram-exec`, not a
new graph node and not a `KernelCall` variant.

## Non-Goals

- [x] Do not implement a diffusion model runtime.
- [x] Do not implement an agent orchestration loop.
- [x] Do not add runtime planning to `InferenceSession::execute`.
- [x] Do not mutate graphs during execution.
- [x] Do not add a dynamic scheduler in executor hot paths.
- [x] Do not add a side cache separate from `BufferArena` residency.
- [x] Do not add backend trait objects, boxed kernels, or function-pointer tables.

### Rejected: refinement as a node type

- [x] Would mix control flow into pure dataflow graph IR.
- [x] Would force graph validation and compiler lowering to understand loop state.
- [x] Would blur operation identity with execution strategy.

### Rejected: refinement as a canonical op

- [x] Would make orchestration appear as backend work.
- [x] Would pressure `CpuBackend::dispatch` to invoke subgraphs or sessions.
- [x] Would violate the closed, tensor-kernel role of `KernelCall`.

### Possible: refinement as a graph type

- [x] Fully unrolled pass graphs preserve static execution.
- [x] Early convergence and repair policy become awkward.
- [x] Validation would need to be encoded as tensor dataflow or external metadata.
- [x] Useful as a future compiler output, not the smallest first-class primitive.

### Recommended: refinement as an execution strategy

- [x] A refinement plan is compiled before execution.
- [x] The pass order is fixed.
- [x] Validator order is fixed.
- [x] Repair policy is fixed and bounded.
- [x] Runtime state flows through addressed labels.
- [x] The executor repeats compiled plans without mutating graphs.

## Proposed Prototype Scope

Implement the smallest production-quality prototype:

- [x] Add `crates/hologram-exec/src/refinement.rs`.
- [x] Re-export the module from `crates/hologram-exec/src/lib.rs`.
- [x] Keep the prototype single-session.
- [x] Use `InferenceSession::execute_addressed` for pass execution.
- [x] Feed each pass output label set back as the next input label set.
- [x] Require compatible input and output state arity.
- [x] Bound all work by `max_passes` plus bounded repair passes.
- [x] Do not add archive format changes in the prototype.
- [x] Do not add backend dispatch changes.
- [x] Do not add runtime graph mutation.

## Refinement State Contract

Define state explicitly before implementation. A refinement state is not an
opaque user object in the runtime path; it is a bounded set of addressed
Hologram values.

- [x] Define state arity:
  - number of input state ports;
  - number of output state ports;
  - whether input and output arity must match exactly.
- [x] Define state layout:
  - dtype per port;
  - shape per port;
  - byte length per port;
  - alignment expectations inherited from `BufferArena`.
- [x] Define state compatibility:
  - every pass output must be valid as the next pass input;
  - incompatible dtype, shape, or byte length is a plan error;
  - shape repair belongs in compile/planning, not runtime inference.
- [x] Add explicit `RefinementStateContract` for planner-generated plans.
- [x] Allow implicit contract derivation from session ports for convenience.
- [x] Validate explicit contracts before any pass executes.
- [x] Define state ownership:
  - pass inputs are resident labels;
  - pass outputs become the next state labels;
  - previous state labels may be released unless retained for validation/reporting.
- [x] Define aliasing behavior:
  - aliasing is allowed only through `BufferArena` label residency;
  - refinement must not expose mutable byte aliases outside backend dispatch.
- [x] Define boundary behavior:
  - raw input bytes may be interned once at the boundary;
  - final bytes may be resolved or copied only at the caller boundary;
  - pass-to-pass flow should use labels.

## Proposed Runtime API

- [x] Add `RefinementPlan`.
- [x] Add `RefinementPlanBuilder`.
- [x] Add borrowed execution via `RefinementPlan::execute` and `RefinementRunner`.
- [x] Add `CompiledRefinement<B>`.
- [x] Add `RefinementStateContract`.
- [x] Add `RefinementStatePort`.
- [x] Add `ValidatorKind`.
- [x] Add `RepairPolicy`.
- [x] Add `RefinementStatus`.
- [x] Add `RefinementReport`.
- [x] Add `RefinementError`.

The public API should prefer closed runtime enums over trait objects in the
execution path. Planner-facing traits can come later once the compiled plan
format is stable.

## Validation Model

Output labels are witnessed derivation labels, so label equality is stricter
than semantic convergence. An idempotent byte transform can still produce a
new label each pass if the derivation changed.

Convergence states should be reported distinctly:

- [x] `LabelStable`: final state labels match the previous state labels.
- [x] `ByteStable`: final state bytes match the previous state bytes.
- [x] `ValidatorAccepted`: configured validators accepted the state.
- [x] `PassBoundReached`: pass budget ended before convergence.
- [x] `RepairBoundReached`: repair budget ended before convergence.

Prototype validators:

- [x] `StableLabels`: O(number of state ports), useful for identity/address-only state.
- [x] `StableBytes`: zero-copy resolved-slice comparison, useful for semantic fixed points.
- [x] Pass-limit handling: fail when the bounded pass budget is exhausted.

The prototype should preserve zero-copy state flow. `StableBytes` may read
arena slices but must not allocate or copy state buffers.

Validator cost classes:

- [x] O(1): metadata-only or fixed small-field validation.
- [x] O(number of state ports): label equality, arity checks, status checks.
- [x] O(state bytes): exact byte-stability validation.
- [ ] O(compiled validator plan): future validator graphs over shared state.

Strict runtime profiles may disable O(state bytes) validators. Exact byte
stability is useful for the prototype but must be opt-in and visible in the
plan/report.

## Repair Policy

Prototype repair should stay deliberately small:

- [x] `RepairPolicy::None`
- [x] `RepairPolicy::RetryPass { extra_passes: u8 }`

Future repair plans can become compiled repair subplans once shared-pool
multi-session execution exists.

Repair accounting must be explicit:

- [x] Decide whether repair passes count against `max_passes`.
- [x] Decide whether repair has a separate `max_repairs` or `extra_passes`.
- [x] Report repair attempts separately from normal passes.
- [x] Report repaired convergence distinctly from unrepaired convergence.
- [x] Treat repair exhaustion as a bounded terminal status, not an execution loop.

## Failure Taxonomy

- [x] Invalid plan:
  - zero pass budget;
  - incompatible state arity;
  - incompatible dtype or shape contract;
  - unsupported validator/repair combination.
- [x] Execution error:
  - backend dispatch failure;
  - missing resident label;
  - archive/session mismatch.
- [x] Validation failure:
  - configured validator rejected the state;
  - required convergence predicate failed.
- [x] Non-convergence:
  - pass budget reached without accepted state.
- [x] Repair exhausted:
  - repair budget reached without accepted state.
- [x] Resource-bound rejection:
  - plan exceeds configured pass, repair, validator, or state-size limits.

## Memory Lifetime

Repeated refinement can create many witnessed labels even when buffers are
reused. The plan must define retention so refinement cannot grow state
indefinitely.

- [x] Retain current state labels.
- [x] Retain previous state labels only while validators need them.
- [x] Release older state labels after validation/report accounting.
- [x] Avoid retaining every intermediate state by default.
- [x] Report final state labels only, plus optional bounded trace metadata.
- [x] Add tests proving repeated refinement does not grow retained state
      without bound.

## Serialization Boundary

The prototype can avoid archive changes, but the eventual format should be
sketched now so the runtime API does not paint itself into a corner.

- [ ] Define eventual plan identity and version.
- [ ] Define state contract encoding.
- [ ] Define pass graph/session references.
- [ ] Define validator encoding.
- [ ] Define repair policy encoding.
- [ ] Define pass and repair bounds.
- [ ] Define reporting/counter expectations.
- [ ] Decide whether the first representation uses `Extension` metadata or a
      dedicated `RefinementPlan` archive section.

## Shared Arena Prerequisite

Future multi-session validation and repair plans require shared state access.
Separate `InferenceSession`s currently own separate `BufferArena`s, so labels
from one session are not automatically resident in another.

- [ ] Design shared-pool execution before compiled validator sessions.
- [ ] Or design explicit label import with clear copy/hash costs.
- [x] Keep the prototype single-session until this boundary is resolved.
- [x] Document that cross-session zero-copy refinement is future work.

## Determinism Across Backends

Exact byte convergence may differ across CPU, GPU, WASM, and future targets for
floating-point kernels.

- [x] Prototype exact validators are exact-only.
- [ ] Future tolerance validators need explicit dtype and backend policy.
- [x] Reports should include validator kind so backend-specific convergence is
      auditable.
- [x] Cross-backend deterministic tests should start with integer or exact
      byte-preserving graphs.

## Reporting Contract

Refinement reports should be useful without exposing unbounded traces.

- [x] Final status.
- [x] Passes executed.
- [x] Repair passes executed.
- [x] Validator outcomes.
- [x] Final state labels.
- [x] Dispatch count.
- [x] Skipped/resident reuse count.
- [x] Optional bounded failure reason.
- [x] No unbounded per-pass trace by default.

## Resource Bounds

- [x] `max_passes` must be a small bounded plan constant.
- [x] Repair bounds must be small bounded plan constants.
- [x] Validator count must be bounded.
- [x] State port count must be bounded or validated against session ports.
- [x] O(state bytes) validators must be explicit in the plan.
- [x] User input must not be able to create an unbounded refinement loop.

## Portability and API Surface

- [x] Preserve `no_std + alloc` compatibility for `hologram-exec`.
- [x] Avoid APIs that require OS threads, time, randomness, or global state.
- [ ] Keep FFI exposure as a later phase after the Rust API stabilizes.
- [ ] Keep WASM compatibility in mind when choosing report and error types.

## Tests

- [x] Successful convergence.
- [x] Validation failure.
- [x] Bounded pass termination.
- [x] Deterministic execution across repeated runs.
- [x] Repair flow through bounded retry pass.
- [x] Planner-generated plan through the builder.
- [x] Borrowed-session execution without taking session ownership.
- [x] Explicit state contract matches session.
- [x] Explicit state contract rejects shape mismatch.
- [x] Explicit state contract rejects dtype mismatch.
- [x] Explicit state contract rejects byte-length mismatch.
- [x] No pass executes after contract rejection.
- [x] Invalid plan rejection.
- [x] State arity mismatch rejection.
- [x] No graph mutation during refinement execution.
- [x] Memory retention remains bounded across repeated refinement.
- [x] Exact determinism: same archive, labels, and plan produce the same status
      and final labels.
- [x] Validator cost class is visible in the report or plan.
- [x] Existing `hologram-exec` tests continue to pass.

## Documentation

- [x] Add ADR: `specs/adrs/058-compiled-refinement-strategy.md`.
- [x] Update `specs/docs/architecture.md`.
- [x] Add an example walkthrough.
- [x] Explain why refinement exists.
- [x] Explain how refinement differs from normal graph execution.
- [x] Explain how refinement differs from iterative agent loops.
- [x] Explain how refinement differs from diffusion models.
- [x] Explain why the strategy belongs inside Hologram.
- [x] Document the refinement state contract.
- [x] Document the failure taxonomy.
- [x] Document validator cost classes.
- [x] Document memory lifetime and retention rules.
- [x] Avoid stale tape/KV execution vocabulary from older docs.
- [x] Add future roadmap notes.

## Benchmark Notes

- [x] Document expected overhead:
  - fixed pass-loop orchestration;
  - normal `execute_addressed` cost per pass;
  - validator cost by validator type;
  - no new backend dispatch overhead.
- [x] Record dispatch counts and skipped-call counts in refinement reports.
- [x] Defer dedicated Criterion benchmarks until the API stabilizes.

## Future Roadmap

- [ ] Batch refinement over multiple independent state label sets.
- [ ] Parallel refinement over independent candidates.
- [ ] Shared-pool multi-session validation and repair plans.
- [ ] Archive-level `RefinementPlan` section with format-version bump.
- [ ] Graph convergence validators.
- [ ] Planner-generated refinement plans from `hologram-ai`.
- [ ] Code synthesis repair loops.
- [ ] Workflow optimization loops.
- [ ] Constraint solving loops.
- [ ] Schema evolution repair.
- [ ] Multi-agent convergence substrates above Hologram.

## Acceptance Criteria

- [x] Refinement is represented as a first-class execution strategy.
- [x] No `GraphOp` variant is added for refinement.
- [x] No `KernelCall` variant is added for refinement.
- [x] No backend virtual dispatch path is introduced.
- [x] Runtime work is bounded by plan constants.
- [x] Runtime memory use is bounded by compiled state shape and pass count.
- [x] State moves between passes by label, not by copied buffers.
- [x] All public items are documented.
- [x] New tests cover the prototype behavior.
- [x] `cargo fmt --all` passes.
- [x] `cargo test -p hologram-exec` passes.
- [x] `cargo clippy --workspace -- -D warnings` passes.

## Open Questions

- [ ] Should the first archive-level representation use an `Extension` section
      or wait for a dedicated `RefinementPlan` section?
- [ ] Should byte-stability validation be allowed in strict O(1) runtime
      profiles, or should those profiles require label/content validators only?
- [ ] Should repair policies be limited to retry-style policies until shared
      arena execution lands?
- [ ] Should planner-generated plans live in `hologram-compiler` or remain
      entirely external until the archive format is extended?
