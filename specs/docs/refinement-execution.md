# Refinement Execution

Refinement execution is a bounded strategy for repeatedly applying a compiled
Hologram graph to a state until configured validators accept the result or the
plan budget is exhausted.

It exists for workloads that naturally look like:

```text
State0 -> Pass -> Validate -> Pass -> Validate -> Finalize
```

It does not turn Hologram into a diffusion framework or an agent runtime. The
model policy, planner, tokenizer, sampler, code generator, or repair heuristic
belongs above Hologram. Hologram provides the deterministic execution substrate
for a plan that has already been compiled.

## Execution Model

A `RefinementPlan` runs against an existing `InferenceSession`. Callers can use
the plan directly with `&mut InferenceSession` or bind the two with the owning
`CompiledRefinement` convenience wrapper.

A plan defines:

- a bounded validator list;
- a bounded repair policy.

Each pass calls `InferenceSession::execute_addressed`. The output labels from
that pass become the next pass input labels. No graph is mutated at runtime, no
new `KernelCall` variant is introduced, and the backend dispatch path remains
the ordinary exhaustive `KernelCall` match.

The prototype is single-session. Separate sessions do not share a `BufferArena`,
so cross-session validation and repair graphs are future work.

## State Contract

A refinement state is a bounded set of addressed Hologram values. It is not an
opaque host object in the executor path.

Before execution, the runtime validates:

- input state arity equals output state arity;
- each input/output state port pair has the same dtype;
- each pair has the same element count;
- each pair has the same shape;
- each pair has the same logical byte length.

This ensures every pass output can become the next pass input without runtime
shape inference.

Plans may carry an explicit `RefinementStateContract`. If present, both the
session inputs and outputs must match that contract exactly. If absent, the
runtime derives the contract from the session and requires the input/output
state ports to match each other. Planner-generated plans should prefer explicit
contracts so a session mismatch is rejected before any pass executes.

## Validators

The first implementation has two validators:

| Validator | Cost | Use |
|-----------|------|-----|
| `StableLabels` | O(number of state ports) | Exact label fixed points |
| `StableBytes` | O(state bytes) | Semantic byte fixed points |

`StableBytes` compares resolved arena slices and does not copy state buffers.
It still scans logical state bytes, so it is not suitable for strict O(1)
profiles unless the state is known to be tiny or the profile allows explicit
byte validators.

Label stability and byte stability are intentionally separate. Output labels
are witnessed derivation labels, so an idempotent transform may be byte-stable
without being label-stable.

## Repair

The prototype supports:

- `RepairPolicy::None`;
- `RepairPolicy::RetryPass { extra_passes }`.

`RetryPass` is a bounded retry of the same compiled pass after normal pass
budget exhaustion. Future repair policies can point to separate compiled repair
plans once shared-pool execution exists.

## Example Walkthrough

Consider a one-port state graph:

```text
input x
op relu x as y
output y
```

With initial bytes representing `[-1.0, 2.0]` and a `StableBytes` validator:

1. Pass 1 computes `[0.0, 2.0]`.
2. Validation compares the previous logical bytes to the new logical bytes.
   They differ, so the state has not converged.
3. Pass 2 computes `relu([0.0, 2.0]) = [0.0, 2.0]`.
4. Validation accepts byte stability.
5. The report returns `Converged(ByteStable)`, `passes = 2`, `repairs = 0`,
   final labels, dispatch counts, skip counts, and resident-memory counters.

If `max_passes = 1`, the same plan returns `PassBoundReached`. If the plan also
sets `RepairPolicy::RetryPass { extra_passes: 1 }`, the retry pass converges and
the report returns `Repaired(ByteStable)`.

## Relationship to Other Loops

Normal graph execution is one compiled graph run:

```text
Input -> Graph -> Output
```

Refinement execution is a bounded outer strategy around normal graph execution:

```text
State -> Compiled pass -> Validate -> Compiled pass -> Validate
```

Agent loops are not bounded by Hologram's executor and may include external
tools, prompts, memory, search, or non-deterministic policies. Refinement plans
are fixed before execution.

Diffusion models define a model family and denoising/sampling policy. Hologram
does not implement those policies. It only supports the more general compiled
refinement shape when a downstream planner chooses to use it.

## Future Roadmap

- Batch refinement over independent state label sets.
- Parallel refinement over independent candidate states.
- Shared-pool multi-session validation and repair.
- Archive-level `RefinementPlan` section with a format-version bump.
- Compiled graph validators.
- Planner-generated plans from `hologram-ai`.
- Code synthesis repair loops.
- Workflow optimization loops.
- Constraint solving loops.
- Schema evolution repair.
- Multi-agent convergence substrates above Hologram.

Each extension must preserve bounded work, bounded memory, deterministic
scheduling, no runtime graph mutation, and planner/executor separation.
