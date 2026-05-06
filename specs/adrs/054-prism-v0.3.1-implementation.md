# ADR-054: Hologram as a Prism Application (v0.3.1)

**Status:** Accepted 2026-05-06
**Supersedes:** ADR-046, ADR-047, ADR-048

## Context

`uor-foundation 0.3.1` exposes a parametric Prism whose three substitution
axes — `HostTypes`, `HostBounds`, `Hasher` — are the only externalization
surfaces. Hologram had previously vended its own ring/quantum/term machinery
(in `hologram-ring`, `hologram-core`, `hologram-cascade`, `hologram-transform`,
`hologram-async`, `hologram-compression`, `hologram-shape`) that mirrored
upstream constructs.

## Decision

Hologram is reconciled to be a **consumer** of `uor-foundation 0.3.1`'s
parametric Prism. All hologram types are `ConstrainedTypeShape`
declarations; all operations are Term-arena trees built from upstream's
closed `PrimitiveOp` set (the 10:
`Neg`/`Bnot`/`Succ`/`Pred`/`Add`/`Sub`/`Mul`/`Xor`/`And`/`Or`).

Workspace shape:

| Crate | Role |
|---|---|
| `hologram-host` | Substitution-axis impls per backend (HostTypes, HostBounds, Hasher) |
| `hologram-types` | `ConstrainedTypeShape` declarations (Tensor, Region, Layout, Weight, Schedule, Fingerprint, Witness, dtypes including DTypeI4 for X-5 quantization) |
| `hologram-ops` | Term-arena emitters for the closed 105-op catalog (V.3 + V.4 + X-5) |
| `hologram-graph` | Graph IR + topological-level Schedule + per-node QuantAttrs |
| `hologram-compiler` | Per-node CompileUnit pipeline + certificate cache + backward emission |
| `hologram-exec` | InferenceSession + Executor (schedule-aware, optional rayon-parallel within levels) |
| `hologram-backend` | Per-target dispatch (CPU + Metal + WGPU) |
| `hologram-archive` | `.holo` zero-copy artifact format with BLAKE3 footer |
| `hologram-cli` | Subcommand entry points |
| `hologram-ffi` | C ABI + WASM surfaces |
| `hologram-bench` | Criterion benches |

Crates retired: `hologram-ring`, `hologram-core`, `hologram-cascade`,
`hologram-transform`, `hologram-async`, `hologram-compression`,
`hologram-shape`. Their responsibilities moved to upstream calls or
folded into the surviving crates per spec XI.1.

## Architectural commitments

- **I-1 (closed PrimitiveOp set).** Every Term::Application uses one of the
  10 upstream `PrimitiveOp` discriminants. No new operator vocabulary.
- **I-9 (Term-tree authority).** The Term arena built by `emit_op_term`
  is the formal specification of every operation; backend kernels are
  the execution form; equivalence is verified by tests.
- **I-10 (Grounding boundary).** `Grounding` impls live only at the input
  boundary (WeightLoaderGrounding, ConstantGrounding). Operations
  between hologram types are Term trees, never Grounding programs.
- **VIII.2 (schedule-aware execution).** The compiler emits a per-level
  kernel-call exec plan; the runtime walks it level-by-level and
  optionally parallelizes within levels via rayon (`parallel` feature).
- **V.4 / ADR-043 (backward planned).** `compile_with_backward(graph, output)`
  walks reverse-topological order and emits gradient nodes consuming the
  upstream gradient + forward inputs, accumulating fan-in via Add nodes.
- **X-5 (quantization).** `DTypeI4` and `DTypeI8` weights flow through
  `OpKind::Dequantize` with `(scale, zero_point)` per-tensor parameters
  attached via `Graph::set_quant_attrs`. Symmetric and asymmetric
  schemes; INT4 is two values per byte with sign-extension.
- **X.3 (BLAKE3 weight dedup).** Identical bodies share storage in the
  `Weights` archive section; the compiler dedups every constant body
  through `WeightStore::insert`.

## Consequences

- All hologram code compiles against `uor-foundation 0.3.1` without
  shims.
- The IRI namespace `https://hologram.uor.foundation/type/...` is a
  Prism extension under ADR-013 — types only, never new PrimitiveOps.
- The reconciliation is locked by the verification regime in spec
  Part XII (cargo check / clippy / test / decode-step bench parity /
  disassembly-confirmed zero-cost / forbidden-symbol absence).

See `specs/docs/prism-v0.3.1-implementation.md` for the authoritative
implementation specification.
