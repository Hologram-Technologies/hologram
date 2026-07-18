# hologram-ops

> The closed catalog of hologram operations as Term-tree emitters, with per-op reference evaluators.

Each canonical op is a marker type plus a const-tagged IRI plus an `emit_term` function that emits a `Term` tree into a caller-provided `HoloArena<CAP>`. The Term tree is the formal specification of the op (spec invariant I-9); backend kernels in `hologram-compute` are the execution form, and equivalence between the two is verified by the per-op reference evaluators.

The catalog is the closed 64-op set organized per spec V.3, grouped by family (activations, reductions, linalg, convolution, normalization, pooling, quantization, layout, elementwise, and so on).

## What it provides

- `emit_op_term` — dispatch entry point that emits the Term tree for a given op.
- `HoloArena`, `HoloTerm`, `HOLOGRAM_INLINE_BYTES` — the Term arena, its term handle, and the inline-storage threshold.
- `OpKind` — the enum tag for the closed op catalog, re-exported by the graph IR.
- `ReferenceEvaluator`, `ScalarEvaluatorU64`, `EvalError` — reference evaluators used to check kernel/Term equivalence.
- Per-family op modules: `activations`, `activation_reduce`, `conv`, `direct`, `elementwise_binary`, `elementwise_unary`, `layout`, `linalg`, `normalization`, `pooling`, `quantization`, `reduction`, `structured`, `utility`, plus `grounding`, `lut`, and `dispatch`.

## Targets & build notes

`no_std`. Uses `libm` for scalar math in the reference evaluators.

Part of the [hologram](../../README.md) workspace.
