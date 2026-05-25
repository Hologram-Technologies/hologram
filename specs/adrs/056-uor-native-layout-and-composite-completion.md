# ADR-056: UOR-native completion of layout, indexed, and gradient ops

**Status:** Accepted 2026-05-25
**Relates to:** ADR-055 (UOR-native op taxonomy), ADR-033 (ProjectField),
ADR-053 (extended PrimitiveOp set incl. `Concat`), ADR-018 (zero-movement pool)

## Context

ADR-055 left a roadmap of declared-but-unimplemented ops that fail loud:
layout transforms (`Transpose`/`Slice`/`Concat`/`Pad`/`Expand`/`Resize`),
indexed ops (`RotaryEmbedding`, `Lrn`), and the gradient ops. The conventional
tensor-library realization — kernels that copy reordered/sliced bytes, with
call structs carrying axis/stride/offset metadata — is **rejected**: it is
data-moving (violates the zero-movement contract, ADR-018) and imposes a
stride/view layer foreign to UOR. This ADR defines the **UOR-native**
realization, grounded in primitives that already exist in the substrate.

## Substrate primitives (verified present)

- **`PrimitiveOp::Concat`** (uor-foundation 0.5.2, ADR-053) — concatenation is
  a *closed primitive*, not a layout hack.
- **`Term::ProjectField { source, byte_offset, byte_length }`** (ADR-033) —
  structural projection of a byte-region of a value: the UOR addressing
  primitive (a *view*, not a copy).
- **`Term::Recurse { measure, base, step }`** — bounded structural recursion
  with a descent measure: the UOR windowed-iteration combinator.
- **`Term::Match`**, **`Term::AxisInvocation { axis, kernel_id, input }`** —
  dispatch + accelerated-kernel invocation.
- **Zero-movement pool** (`BufferArena`): a slot is a *binding* to a buffer;
  `read(BufferRef{slot, offset, length})` already resolves to `(buf, s..e)`,
  so a slot can name a **sub-region** of a buffer with no copy.

## Decision — per-op UOR-native realization

**Addressing class (zero-movement re-binding; no kernel, no copy):**

- **Reshape** — relabel: bind the output slot to the input's buffer, full
  extent. Row-major bytes are unchanged. (Already correct.)
- **Slice** — `ProjectField`: bind the output slot to the input buffer at
  `byte_offset = start·elem`, `byte_length = count·elem`. Contiguous
  (outer-axis) slices are pure re-addressing. Requires **view-binding** in
  `BufferArena` (`bind_view(slot, parent_buf, offset, len)`); `BufferRef`
  already carries `offset`/`length`. Non-contiguous (inner-axis/step) slices
  desugar to a `ProjectField`+`Concat` gather.
- **Expand** (broadcast) — a stride-0 re-addressing view: the output names the
  input buffer; the consuming kernel reads with broadcast addressing. No copy.

**Constructor class (the `Concat` primitive):**

- **Concat** — `PrimitiveOp::Concat`: a binary placement primitive
  `out = a ∥ b` (n-ary concat folds as a left-associated chain of the binary
  primitive). This is the one *intrinsic* constructor; its kernel places `a`
  then `b`. Representation: a binary call (two inputs + output sized to the
  sum), not the single-input `LayoutCall`.
- **Pad** — `Concat(zeros_lo, x, zeros_hi)` along the padded axis: the pad
  regions are constant zero tensors; reuses the `Concat` primitive.

**Pipeline class (Path-B desugar into primitive nodes — ADR-055):**

- **RotaryEmbedding** — `ProjectField` the even/odd halves, then the closed
  form with cos/sin **constant tables** as operands:
  `out_even = x_even·cos − x_odd·sin`, `out_odd = x_even·sin + x_odd·cos`,
  recombined with `Concat`. Pure `Mul`/`Sub`/`Add`/`Concat`/`ProjectField`
  pipeline — no bespoke kernel.
- **Lrn** — `Term::Recurse` windowed channel fold: `square → windowed-sum
  (Recurse) → scale → pow → div`. The window is the recursion's descent
  measure; reuses `Mul`/`Add`/`Pow`/`Div` + `Recurse`.
- **Resize** — bilinear interpolation as `Mul`/`Add` over `ProjectField`-gathered
  neighbors (the `emit_resize` Term shape).
- **Gradient ops** — each op's backward Term tree, desugared into a backward
  primitive-node pipeline (the training-path analogue of Path B).

**Re-indexing class:**

- **Transpose** — UOR-native transpose is *not* a standalone copy. The
  permutation is a re-addressing carried to and **absorbed by the consuming
  kernel** (matmul/attention already read operands in their required layout via
  the packed-weight layout, ADR/weight-monomorphism). A materialized transpose
  with no absorbing consumer desugars to a `ProjectField`+`Concat` gather.

## Execution substrate to build (in dependency order)

1. **View-binding in `BufferArena`** — `bind_view(slot, parent, offset, len)`
   so a slot aliases a sub-region of a producer's buffer (zero-movement). The
   compiler marks addressing-class ops to bind-view rather than allocate +
   run a kernel. Unblocks Slice, Reshape (optimal), Expand.
2. **`Concat` primitive kernel** — binary placement; unblocks Concat, Pad.
3. **`ProjectField` / `Recurse` realized on tensor buffers** — via Path-B
   desugaring into primitive nodes the existing kernels execute (preferred,
   reuses verified kernels) — unblocks RoPE, Lrn, Resize, gradients.

## Verification (V&V) — every op

Each op gets an external-reference conformance test (the operator's
mathematical definition in f64) and, for the addressing class, a zero-movement
assertion (the output slot shares the input's buffer — no allocation, byte
identity) extending `tests/zero_overhead.rs`. No op is "done" until both pass.

## Consequences

Layout/indexed/gradient ops are realized in UOR primitives (Concat,
ProjectField, Recurse) and zero-movement re-addressing — never a conventional
stride/copy layer. The branch is complete when every catalog op is either an
irreducible accelerated kernel, a relabel, a zero-movement view, or a verified
primitive pipeline — with no `UnsupportedOp` remaining.
