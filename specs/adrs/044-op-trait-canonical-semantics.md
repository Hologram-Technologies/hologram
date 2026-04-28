# ADR-044: `Op` Trait as the Canonical Semantic Interface

## Status

Accepted (2026-04-27)

## Context

ADR-043 introduced `hologram-ops` as the canonical semantic vocabulary for
Hologram operations and `hologram-transform` as the chain → plan → execute
stack that consumes it. After the first slice landed, `hologram-ops` exposes
two enums:

- `OpKind` — minimal canonical identity (`Add`, `MatMul`).
- `SemanticOp` — graph-facing payload with semantic attributes (`MatMulAttrs`,
  `NormAttrs`, `Conv2dAttrs`, …).

Every fact about an op (arity, output count, name, signature, default
backward rule, semantic category) lives in `match` arms inside the enum impl.
Adding a new op today requires editing five separate match statements
(`SemanticOp::arity`, `::n_outputs`, `::name`, plus the bridges in
`hologram-graph::graph::op`), and the compiler does not point at all of them
because the matches are exhaustive on the enum, not on a per-op contract.

This is the same conflation ADR-043 set out to remove, just one level deeper:
identity for ops is centralised, but the **interface** every op must satisfy
is fragmented across the enum's impl block.

`FloatOp` has the same shape today, which is why it became unmaintainable.
We do not want `SemanticOp` to inherit that fate.

## Decision

Introduce an **`Op` trait** in `hologram-ops` that is the per-op-type
contract for canonical semantic facts. Each canonical operation is a typed
struct (e.g. `Add`, `MatMul(MatMulAttrs)`) that implements `Op`. The
`SemanticOp` enum stays as the closed dispatch / serialisation surface and
its impls forward to the trait methods.

```
hologram-ops
  ┌──────────────────────────────────────────┐
  │  trait Op  (per-op contract — open)       │
  │     fn arity() -> u8;                     │
  │     fn n_outputs() -> u8;                 │
  │     fn name() -> &'static str;            │
  │     fn signature() -> OpSignature;        │
  │     fn backward() -> Option<BackwardRule>;│
  │     fn category() -> OpCategory;          │
  └──────────────────────────────────────────┘
                    │
                    │ implemented by
                    ▼
  ┌──────────────────────────────────────────┐
  │  struct Add;                              │
  │  struct MatMul(MatMulAttrs);              │
  │  struct LayerNorm(NormAttrs);             │
  │  …                                         │
  └──────────────────────────────────────────┘
                    │
                    │ wrapped by
                    ▼
  ┌──────────────────────────────────────────┐
  │  enum SemanticOp  (closed — serialised)   │
  │     Add(Add),                             │
  │     MatMul(MatMul),                       │
  │     …                                     │
  │  impl SemanticOp {                        │
  │     fn arity(&self) -> u8 {               │
  │         dispatch!(self, Op::arity)        │
  │     }                                     │
  │     …                                     │
  │  }                                         │
  └──────────────────────────────────────────┘
```

### Why a trait, when the enum already works

`SemanticOp` answers "is this op X?" — exhaustive matching, fixed dispatch,
serialisable. That is not in question and is preserved.

The trait answers a different question: "what does op X declare about
itself?" Locating those declarations on the op type — instead of in match
arms inside the enum impl — has three concrete benefits:

1. **Adding a new op is one trait impl.** The compiler tells the author
   exactly which methods are missing. No silent omission of an arity arm.
2. **Per-op tests are local.** Each op type can have its own conformance
   test (`assert_eq!(MatMul::arity(), 2)`) without poking through the enum.
3. **Op authors think about an op as a whole** (forward, backward,
   signature, category) rather than as scattered case rows.

### Why the enum stays

Three constraints make a pure-trait model unworkable:

- **rkyv archival.** Archives ship the closed enum; trait objects do not
  archive. `SemanticOp` is part of the on-disk graph format.
- **Exhaustive matching.** `hologram-graph` rewrites, fusion passes, and
  CLI inspect rely on exhaustive matching to refuse new variants without
  explicit handling. ADR-043 also forbids `Box<dyn>` in the kernel hot
  path; the same logic applies to the planner's pattern-matching surface.
- **Stable dispatch surface.** Downstream crates (`hologram-graph`,
  `hologram-cascade`, `hologram-cli`) match on `SemanticOp` variants. That
  surface should not change with this ADR.

The trait is therefore additive to the enum, not a replacement for it. The
enum's impl block becomes a thin forwarding layer that calls trait methods
on the inner struct.

### Bridge to legacy `FloatOp`

`hologram-graph::graph::op` keeps `legacy_float_op()` and
`semantic_op()` as the bridge between `GraphOp::Compute(SemanticOp)` and
`GraphOp::Float(FloatOp)`. The trait does not change this; it lives strictly
above the bridge. Migration of consumers off `FloatOp` is the topic of a
later ADR.

### What does *not* go on the trait

- **Lowering.** `to_kernel_call`, `to_tape_kernel`, backend dispatch:
  these belong to the planner and to backend-specific crates, not the
  semantic trait. The trait describes meaning, not execution.
- **Constant ids, weight handles, KV state.** Execution-layer concerns.
- **Fused or runtime-only variants** (`FusedMatMulBiasActivation`,
  `MatMulLut4Activation`, `Passthrough`): these are planner products, not
  canonical semantic ops. They stay on `GraphOp` and never gain `Op` impls.

### Alignment with hologram invariants

| Invariant                            | How this ADR preserves it                                      |
|--------------------------------------|----------------------------------------------------------------|
| Closed serialisation surface         | `SemanticOp` enum unchanged on the wire                        |
| Exhaustive pattern matching          | Downstream crates still match on enum variants                 |
| No virtual dispatch in kernels       | Trait is used at planner-time only; kernels still match enums  |
| O(1) lookup, zero-copy               | Trait methods are `#[inline] const fn` where possible          |
| No allocation                        | Trait methods return primitives / `Copy` structs               |

## Consequences

- `hologram-ops` gains `pub trait Op` and per-op marker structs. Existing
  enum API (`SemanticOp::arity`, `SemanticOp::name`, etc.) continues to work
  by forwarding to the trait.
- A small declarative macro (`impl_semantic_op!`) generates the enum
  forwarders, so adding a new op is: define the struct, impl `Op`, add an
  enum variant. No more "did I update every match?" review burden.
- `BackwardRule::for_op(OpKind)` is replaced by `Op::backward()` on each
  op type. `OpKind` was kept temporarily as a chain-layer tag in the
  initial slice, then removed in the follow-up consolidation: the
  `hologram-transform` chain now carries `SemanticOp` directly, and
  `MatMul` dims live on `SemanticOp::MatMul(MatMulAttrs)` (validated by
  the chain builder, read by the planner without re-derivation).
- New ops added to `SemanticOp` after this ADR **must** ship with an `Op`
  impl. The migration is mechanical for existing variants.
- Adding a new canonical op now follows a single procedure: declare the
  marker struct in `hologram-ops` with its `Op` impl, add the
  `SemanticOp` variant + macro arm, and (if the transform planner should
  lower it) add a `KernelCall` variant + planner arm + reference kernel.
  `Sub` and `Mul` were added this way as the first follow-up.
- All 36 canonical `SemanticOp` variants now have reference kernels in
  `hologram-transform` (forward only; backward is a separate ADR). The
  planner is no longer the migration's bottleneck — adding a new op is
  contained to its own kernel module + one planner arm + one builder
  method.
- One pre-existing arity bug was caught while wiring `FusedSwiGlu`:
  `Op::arity()` claimed 1, but the legacy `FloatOp::FusedSwiGLU`
  conformance suite proves it is 2 (`silu(gate) * up`). The trait
  refactor surfaced this — the per-op-type contract makes such bugs
  visible in one place rather than spread across enum match arms.
- `FloatOp` is unaffected. Its eventual deprecation is out of scope here
  and tracked separately.

## Alternatives considered

- **Pure trait, no enum.** Rejected: breaks rkyv archival, exhaustive
  matching, and the closed dispatch surface. Trait objects on the hot path
  also conflict with ADR-043's "no virtual dispatch in kernels" rule.
- **Status quo (enum-only).** Rejected: the cost of adding an op grows
  linearly with the number of facts the enum exposes, and the compiler
  cannot enforce coverage. This is exactly how `FloatOp` rotted.
- **Const-generic / typestate ops with no enum at all.** Rejected as
  premature; archive serialisation and graph rewrite both require a closed
  runtime tag, and typestate would not survive the rkyv boundary.
- **Procedural macro to generate everything from a single source.**
  Rejected for now; a small `macro_rules!` forwarder is enough and avoids
  a proc-macro crate. Revisit if the op set grows past ~50 variants.
