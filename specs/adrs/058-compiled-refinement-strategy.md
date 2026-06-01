# ADR-058: Compiled refinement execution strategy

**Status:** Accepted 2026-06-01
**Relates to:** ADR-018 (zero-movement pool), ADR-044/045 (canonical ops),
ADR-056 (UOR-native layout/composite completion)

## Context

Some workloads benefit from bounded coarse-to-fine refinement: produce a
candidate state, validate it, optionally repair it, and repeat until the state
is accepted or the pass budget is exhausted. This pattern appears in diffusion
language models, graph repair, code synthesis, workflow optimization, and
constraint solving.

Hologram is not a diffusion-model framework or an agent runtime. Its executor
must preserve deterministic scheduling, bounded work, bounded memory,
zero-movement state flow, no runtime graph mutation, and no runtime planning in
the `InferenceSession` hot path.

The existing runtime already has the right substrate for refinement:
`InferenceSession::execute_addressed` accepts resident κ-labels and returns
output κ-labels, while `BufferArena` is the only result cache.

## Decision

Refinement is a first-class **compiled execution strategy** in
`hologram-exec`, not a graph node and not a backend kernel.

The initial strategy is intentionally narrow:

1. A `RefinementPlan` fixes `max_passes`, validators, and repair policy before
   execution.
2. A `RefinementPlan` can run against a borrowed `&mut InferenceSession`; an
   owning `CompiledRefinement` wrapper remains available for callers that want
   to bind session and plan together.
3. A plan may carry an explicit `RefinementStateContract`; otherwise the
   contract is derived from the session ports at bind time.
4. Each pass calls `execute_addressed`.
5. The output label set from one pass becomes the input label set for the next.
6. Validators run only at pass boundaries.
7. Repair is bounded and explicit; the prototype supports retrying the same
   pass with `RepairPolicy::RetryPass`.
8. No graph, archive, `KernelCall`, or backend dispatch surface changes are
   required for the prototype.

The runtime validates the state contract up front. A contract records state port
arity plus each port's dtype, element count, shape, and logical byte length.
Both input and output ports must match the contract so a pass output can be fed
back as the next pass input without shape inference.

## Rejected alternatives

- **Refinement as a `GraphOp`.** Rejected because graph nodes are pure dataflow.
  Looping, validation, and repair policy are execution strategy concerns.
- **Refinement as an `OpKind` / `KernelCall`.** Rejected because backend dispatch
  is a closed tensor-kernel surface. A refinement kernel would either invoke
  subgraphs from the backend or encode runtime scheduling in a kernel variant.
- **Refinement as an unrolled graph only.** Useful as a future compiler output,
  but awkward for early convergence and repair accounting. It duplicates pass
  bodies and forces validation semantics into tensor dataflow.
- **Separate refinement cache.** Rejected. The `BufferArena` residency layer is
  the cache; adding another side store would violate the content-addressed
  execution model.

## Consequences

- Hot-path kernel dispatch remains unchanged: no virtual dispatch, no function
  pointer table, no boxed kernels, and no runtime graph traversal.
- Total work is bounded by plan constants. It is not unbounded agent-style
  iteration.
- Exact label convergence and exact byte convergence are separate. Output labels
  are witnessed derivation labels, so byte-stable idempotent transforms may not
  be label-stable.
- `StableBytes` validation is zero-copy but not O(1) in state size; it scans
  logical state bytes. Strict O(1) profiles can use label or metadata validators.
- Multi-session validation and repair plans require future shared-pool execution
  or explicit label import. Separate sessions do not currently share resident
  labels.
- Archive-level refinement plans are deferred. A future `.holo` format version
  can add a dedicated `RefinementPlan` section once multi-pass/shared-pool
  semantics are stable.

## Verification

The prototype is covered by `hologram-exec/tests/refinement.rs`:

- successful label convergence;
- successful byte convergence;
- borrowed-session execution without taking ownership;
- explicit state-contract match and mismatch rejection;
- validation failure via pass-budget exhaustion;
- bounded repair retry;
- deterministic repeated execution;
- planner-style builder construction;
- invalid plan rejection;
- initial state arity rejection;
- graph schedule immutability;
- bounded resident-memory behavior.
