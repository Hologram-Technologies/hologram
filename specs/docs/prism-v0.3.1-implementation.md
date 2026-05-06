# Hologram as a Prism Application — v0.3.1 Implementation Specification

**Status:** Authoritative. This file defines the correct implementation of hologram on top of `uor-foundation 0.3.1`. The repository is reconciled to this spec; nothing else is normative. No phases, no compatibility shims.

**Standards version:** 2026.03
**Upstream:** `uor-foundation = "0.3.1"`
**Branch:** `prism-v0.3.1`

---

## Part I — Architecture

### I.1 What hologram is

Hologram is **an application of Prism**. The Prism system is realized by `uor-foundation` (substrate, v0.3.1). Hologram consumes upstream's parametric prism through three substitution choices and produces a zero-cost monomorphic compute runtime for ONNX-class models.

Three layers, top-to-bottom:

1. **Substitution axes.** Hologram provides one `HostTypes`, one `HostBounds` per backend, and one `Hasher` impl. These are the only parametric slots; everything else flows from them.
2. **Type vocabulary.** Every hologram domain type (tensor, region, layout, weight, schedule, fingerprint, …) is a `ConstrainedTypeShape` declaration. **All hologram types are derived from uor-foundation; no parallel type system exists.**
3. **Operation pipelines.** Every hologram operation (matmul, conv2d, attention, layernorm, softmax, …) is a **`Term` arena tree** built via `enforcement::TermArena<CAP>`, with `Term::Application { operator: PrimitiveOp, args }` as the leaf, structural composition via `Term::Recurse` (bounded recursion with descent measure), `Term::Match`, `Term::Unfold`, `Term::Try`, and Witt-level changes via `Term::Lift` / `Term::Project`. Each operation is an iso `A → B` whose Term tree witnesses the structural correspondence between input and output type declarations. **Operations are not `Grounding` impls.** `Grounding` (W4 closure) is reserved for parsing host bytes into typed inputs — used by hologram only at the input boundary (loading model weights, decoding inference inputs).

The compiler instantiates the per-op Term tree against concrete input types, builds a `CompileUnit` via `enforcement::CompileUnitBuilder`, validates it (`pipeline::run_tower_completeness`), and emits a backend-specific native kernel that executes with no virtual dispatch. The Term tree is the **formal specification** of the operation; the native kernel is its **execution form**. The `Validated<LiftChainCertificate>` produced by the pipeline run attests that the kernel's behavior matches the Term tree (interpretation A in §VII.6 below).

### I.2 Invariants

These are absolute. Any code that violates them is incorrect.

- **I-1 (ADR-013).** Hologram introduces zero new `PrimitiveOp` discriminants. The closed set is exactly the upstream 10: `Neg`, `Bnot`, `Succ`, `Pred`, `Add`, `Sub`, `Mul`, `Xor`, `And`, `Or`. Every hologram operation decomposes to these.
- **I-2 (ADR-007).** Substitution occurs only at `HostTypes`, `HostBounds`, `Hasher`. Hologram declares no other substitution surfaces.
- **I-3 (ADR-018).** All capacity bounds flow through `HostBounds`. There are no free-standing capacity constants in hologram code outside per-backend `HostBounds` impls.
- **I-4 (ADR-006).** Hologram is `#![no_std]` where possible. `std` is opt-in via cargo feature, never assumed by core, types, ops, host, or backend crates.
- **I-5 (sealing discipline).** `Datum`, `Triad`, `Derivation`, `FreeRank`, `Validated`, `Grounded`, `Certified` are obtained only via `pipeline::run_tower_completeness` / `pipeline::run_reduction_stages` / `mint_*` / resolver free-functions. Hologram never fabricates them.
- **I-6 (Witt level by host).** A backend's `HostBounds::WITT_LEVEL_MAX_BITS` equals the backend's natural register width — the largest single-instruction algebraic operation the host can issue. It is not a hologram design knob.
- **I-7 (zero-cost).** Every public hologram function compiles down to native instructions for the active backend with no dyn-dispatch, no boxed trait objects, no enum-tagged dispatch in the inner loop. Verification is by reading disassembly on the canonical `decode_step` benchmark.
- **I-8 (no parallel type vocabulary).** Hologram declares no struct that represents a domain concept already expressible as a `ConstrainedTypeShape` over upstream's vocabulary. There is no hologram-side `Datum`, `Ring`, `Element`, `Address`, `Triad`, `WittLevel`, `Limbs`, or `PrimitiveOp` mirror.
- **I-9 (Term tree authority).** The `TermArena`-built Term tree is the formal specification of every hologram operation. The native backend kernel is the execution form. The `LiftChainCertificate` produced by `pipeline::run_tower_completeness` attests these agree. Where they appear to disagree, the Term tree wins and the kernel is wrong.
- **I-10 (Grounding is input-only).** `Grounding` impls in hologram appear only at the input boundary — loading model weights from bytes, decoding ONNX tensors, parsing inference inputs. Operations between hologram types are Term trees, never Grounding programs.

### I.3 Data flow

```
ONNX / source            ─┐
                          │
hologram-graph::Graph    ─┤                          per-node TermArena
   (one Node per op)      │      hologram-compiler   ┌──────────────────┐
                          │      ┌───────────────┐   │ Term::Application│
declared types via       ─┤      │ for each Node:│   │ Term::Recurse    │
ConstrainedTypeShape      │   ─▶ │  emit_term ──┼─▶│ Term::Lift/Project│
(hologram-types)          │      │  build CU    │   │ Term::Match/...   │
                          │      │  validate    │   └────────┬──────────┘
declared bounds via      ─┤      │  run_tower   │            │
HostTypes/Bounds/Hasher   │      │  cache by FP │            │ root_term
(hologram-host)           │      │  lower(B)   ──┼────────┐   │
                          │      └───────────────┘        │   ▼
                          │                               │ uor_foundation
                          │                               │ pipeline::
                          │                  ┌────────────┘ run_tower_completeness
                          │                  │              │
                          │                  ▼              ▼
                          │           native KernelCall    Validated<LiftChainCertificate>
                          │           (per backend)        (per Node)
                          │                  │              │
                          ▼                  ▼              ▼
                       hologram-archive: kernel_calls + certificates + weights + schedule
                                            │
                                            ▼ (.holo file)
                                     hologram-exec
                                            │
                                       hologram-backend
                                       (CPU / Metal / wgpu)
```

The Term tree is the formal spec consumed by upstream's pipeline; the KernelCall is the executable form consumed by the backend. The certificate attests the Term tree is well-formed; per-op tests attest the KernelCall implements the Term tree.

---

## Part II — Workspace Layout

The reconciled workspace is **10 crates**. Crates absent from this list are deleted.

| Crate | Role | Key public items |
|---|---|---|
| `hologram-host` | Substitution-axis impls per backend | `HologramHostTypes`, `HologramHostBoundsCpu`, `HologramHostBoundsAvx2`, `HologramHostBoundsAvx512`, `HologramHostBoundsNeon`, `HologramHostBoundsMetal`, `HologramHostBoundsWgpu`, `HologramHasher` |
| `hologram-types` | `ConstrainedTypeShape` declarations | `Tensor<S, D, L>`, `Region`, `Layout`, `Weight`, `Schedule`, `Fingerprint`, `WitnessRecord`, dtype + shape primitives |
| `hologram-ops` | One `Grounding` impl per canonical op | 54 ops: `MatMulOp`, `Conv2dOp`, `AttentionOp`, `LayerNormOp`, `SoftmaxOp`, … (each is a Prism pipeline) |
| `hologram-graph` | Graph IR + scheduling | `Graph`, `GraphOp`, `NodeId`, `Schedule`, `SubgraphDef`, `ConstantStore` |
| `hologram-compiler` | Graph → archive of compiled pipelines | `Compiler`, `CompilationOutput`, `compile`, `compile_from_source` |
| `hologram-exec` | Runtime executor for compiled archives | `InferenceSession`, `Executor`, `BufferArena` |
| `hologram-backend` | Per-target dispatch | `Backend` trait, `CpuBackend`, `MetalBackend`, `WgpuBackend` |
| `hologram-archive` | `.holo` zero-copy artifact format | `HoloWriter`, `HoloLoader`, `LoadedPlan` |
| `hologram-cli` | Subcommand entry points | `compile`, `execute`, `bench`, `inspect` |
| `hologram-ffi` | C ABI + WASM bindings | C wrapper types, wasm-bindgen surfaces |
| `hologram-bench` | Criterion benchmarks | 23 bench suites (unchanged from current set) |

**Crates deleted:**

- `hologram-ring` — entirely. Ring algebra is upstream parametric machinery; nothing local survives.
- `hologram-core` — split. The `lut`, `view`, `encoding` modules move into `hologram-ops` as compile-time fusion helpers. The `q0`/`q1`/`q2`/`q3`/`carry`/`quantum`/`ring` modules are deleted (replaced by upstream `WittLevel` + `Limbs<N>` + `Ring<H>` + `Datum<H>`). The `term` module is deleted (replaced by upstream `enforcement::Term` + `TermArena`).
- `hologram-shape` — folded into `hologram-types`. Shape inference becomes part of `ConstrainedTypeShape` declaration, not a runtime fallback.
- `hologram-cascade` — deleted. The 7-stage cascade is upstream `pipeline::run_reduction_stages`. The certificate cache (the only hologram-specific value-add) becomes a small in-memory map in `hologram-compiler` keyed on `ContentFingerprint<32>`.
- `hologram-transform` — folded into `hologram-compiler`. The chain → plan → executor split is reframed as "Prism pipeline → CompiledPlan → Backend dispatch."
- `hologram-async` — deleted. Tokio wrappers become 80 lines of thin async glue inside `hologram-exec` behind a `tokio` feature flag.
- `hologram-compression` — deleted. The 100 lines of entropy-based compression become an internal helper in `hologram-archive` behind a `compression` feature flag.

The 12 ADRs in `specs/adrs/` are retained as historical record. A new `ADR-054-prism-v0.3.1-implementation.md` records this reconciliation with a one-line pointer to this spec.

### II.1 Cargo.toml workspace

```toml
[workspace]
resolver = "2"
members = [
    "crates/hologram-host",
    "crates/hologram-types",
    "crates/hologram-ops",
    "crates/hologram-graph",
    "crates/hologram-compiler",
    "crates/hologram-exec",
    "crates/hologram-backend",
    "crates/hologram-archive",
    "crates/hologram-cli",
    "crates/hologram-ffi",
    "crates/hologram-bench",
]

[workspace.package]
version = "0.5.0"
edition = "2021"
license = "MIT OR Apache-2.0"
authors = ["UOR Foundation"]

[workspace.dependencies]
hologram-host     = { path = "crates/hologram-host", default-features = false }
hologram-types    = { path = "crates/hologram-types", default-features = false }
hologram-ops      = { path = "crates/hologram-ops", default-features = false }
hologram-graph    = { path = "crates/hologram-graph", default-features = false }
hologram-compiler = { path = "crates/hologram-compiler" }
hologram-exec     = { path = "crates/hologram-exec" }
hologram-backend  = { path = "crates/hologram-backend" }
hologram-archive  = { path = "crates/hologram-archive" }

uor-foundation = "0.3.1"

blake3   = { version = "1.5", default-features = false }
rkyv     = { version = "0.8", features = ["tinyvec-1"] }
bytemuck = { version = "1.14", features = ["derive"] }
smallvec = "1.13"
tinyvec  = "1.6"
rayon    = "1.8"
parking_lot = "0.12"
thiserror = "2"
clap      = { version = "4", features = ["derive"] }
tokio     = { version = "1", features = ["rt-multi-thread", "macros"] }
wgpu      = "24"
pollster  = "0.4"
criterion = { version = "0.5", features = ["html_reports"] }
libm      = "0.2"
memmap2   = "0.9"
tracing   = "0.1"
```

### II.2 Crate dependency DAG

```
hologram-host        (leaf — depends only on uor-foundation, blake3)
       │
       ▼
hologram-types       (depends on hologram-host)
       │
       ▼
hologram-ops         (depends on hologram-types, hologram-host)
       │
       ▼
hologram-graph       (depends on hologram-ops)
       │
       ▼
hologram-compiler    (depends on hologram-graph, hologram-archive)
hologram-archive     (depends on hologram-graph)
hologram-backend     (depends on hologram-ops, hologram-host)
hologram-exec        (depends on hologram-backend, hologram-archive)
hologram-cli         (depends on hologram-compiler, hologram-exec)
hologram-ffi         (depends on hologram-compiler, hologram-exec)
hologram-bench       (depends on all of the above)
```

No cycles. `hologram-host` is the only crate with no internal dependencies. Removing any crate breaks exactly its downstream subtree.

---

## Part III — Substitution Axes (`hologram-host`)

This is the foundation of every monomorphization. `hologram-host` is `#![no_std]` and depends only on `uor-foundation` and `blake3`.

### III.1 `HologramHostTypes`

Single impl, shared across all backends. All three slots match `DefaultHostTypes` because hologram has no need for non-default Decimal / HostString / WitnessBytes representations.

```rust
// crates/hologram-host/src/types.rs

use uor_foundation::{HostTypes, DecimalTranscendental};

/// Hologram's `HostTypes` impl. Identical layout to `DefaultHostTypes`,
/// but distinct as a marker for downstream substitution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostTypes;

impl HostTypes for HologramHostTypes {
    type Decimal = f64;
    type HostString = str;
    type WitnessBytes = [u8];
    const EMPTY_DECIMAL: f64 = 0.0;
    const EMPTY_HOST_STRING: &'static str = "";
    const EMPTY_WITNESS_BYTES: &'static [u8] = &[];
}
```

### III.2 `HologramHostBounds*` — per-backend

Each backend pins `WITT_LEVEL_MAX_BITS` to its natural register width. Common bounds (`FINGERPRINT_*`, `TRACE_MAX_EVENTS`) are sized for trillion-param + UHD streaming workloads.

```rust
// crates/hologram-host/src/bounds.rs

use uor_foundation::HostBounds;

/// CPU scalar (x86-64 / AArch64 GPR). One u64 per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsCpu;

impl HostBounds for HologramHostBoundsCpu {
    const FINGERPRINT_MIN_BYTES: usize = 32;        // BLAKE3 full width
    const FINGERPRINT_MAX_BYTES: usize = 32;
    const TRACE_MAX_EVENTS: usize = 16_384;          // UHD per-frame capacity
    const WITT_LEVEL_MAX_BITS: u32 = 64;
}

/// AVX2 (256-bit YMM). `Limbs<4>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsAvx2;

impl HostBounds for HologramHostBoundsAvx2 {
    const FINGERPRINT_MIN_BYTES: usize = 32;
    const FINGERPRINT_MAX_BYTES: usize = 32;
    const TRACE_MAX_EVENTS: usize = 16_384;
    const WITT_LEVEL_MAX_BITS: u32 = 256;
}

/// AVX-512 (512-bit ZMM). `Limbs<8>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsAvx512;

impl HostBounds for HologramHostBoundsAvx512 {
    const FINGERPRINT_MIN_BYTES: usize = 32;
    const FINGERPRINT_MAX_BYTES: usize = 32;
    const TRACE_MAX_EVENTS: usize = 16_384;
    const WITT_LEVEL_MAX_BITS: u32 = 512;
}

/// ARM NEON (128-bit Q-register). `Limbs<2>` per algebraic op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsNeon;

impl HostBounds for HologramHostBoundsNeon {
    const FINGERPRINT_MIN_BYTES: usize = 32;
    const FINGERPRINT_MAX_BYTES: usize = 32;
    const TRACE_MAX_EVENTS: usize = 16_384;
    const WITT_LEVEL_MAX_BITS: u32 = 128;
}

/// Apple Metal (64-bit scalar lanes; SIMD-group-wide ops at 32×32 = 1024).
/// Pinned at 64 because algebraic ops are issued per-lane; cross-lane
/// reductions are pipeline structure, not single algebraic operations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsMetal;

impl HostBounds for HologramHostBoundsMetal {
    const FINGERPRINT_MIN_BYTES: usize = 32;
    const FINGERPRINT_MAX_BYTES: usize = 32;
    const TRACE_MAX_EVENTS: usize = 16_384;
    const WITT_LEVEL_MAX_BITS: u32 = 64;
}

/// WebGPU (WGSL). Same per-lane reasoning as Metal: 64-bit scalar.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HologramHostBoundsWgpu;

impl HostBounds for HologramHostBoundsWgpu {
    const FINGERPRINT_MIN_BYTES: usize = 32;
    const FINGERPRINT_MAX_BYTES: usize = 32;
    const TRACE_MAX_EVENTS: usize = 16_384;
    const WITT_LEVEL_MAX_BITS: u32 = 64;
}
```

### III.3 `HologramHasher` — BLAKE3

```rust
// crates/hologram-host/src/hasher.rs

use uor_foundation::enforcement::Hasher;
use blake3::Hasher as Blake3Hasher;

/// BLAKE3-backed `Hasher` impl at the canonical 32-byte width.
///
/// Truncation idempotence: `Hasher::OUTPUT_BYTES = 32` always; downstream
/// callers that need a smaller width slice the result manually. This
/// matches ADR-001 / ADR-052.
#[derive(Clone)]
pub struct HologramHasher {
    inner: Blake3Hasher,
}

impl Hasher<32> for HologramHasher {
    const OUTPUT_BYTES: usize = 32;

    #[inline]
    fn initial() -> Self {
        Self { inner: Blake3Hasher::new() }
    }

    #[inline]
    fn fold_byte(mut self, b: u8) -> Self {
        self.inner.update(&[b]);
        self
    }

    #[inline]
    fn fold_bytes(mut self, bytes: &[u8]) -> Self {
        self.inner.update(bytes);
        self
    }

    #[inline]
    fn finalize(self) -> [u8; 32] {
        self.inner.finalize().into()
    }
}
```

### III.4 Backend selection

The active `HostBounds` is selected by a single `cfg`-gated re-export at the crate root:

```rust
// crates/hologram-host/src/lib.rs

#![no_std]

mod types;
mod bounds;
mod hasher;

pub use types::HologramHostTypes;
pub use bounds::{
    HologramHostBoundsCpu, HologramHostBoundsAvx2, HologramHostBoundsAvx512,
    HologramHostBoundsNeon, HologramHostBoundsMetal, HologramHostBoundsWgpu,
};
pub use hasher::HologramHasher;

/// Active CPU bounds for this build. Resolves at compile time:
/// AVX-512 ▸ AVX2 ▸ NEON ▸ scalar.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type ActiveCpuBounds = HologramHostBoundsAvx512;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
pub type ActiveCpuBounds = HologramHostBoundsAvx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub type ActiveCpuBounds = HologramHostBoundsNeon;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
pub type ActiveCpuBounds = HologramHostBoundsCpu;
```

**Rationale.** A single hologram binary may link multiple backends (CPU + wgpu, CPU + Metal). Each backend instantiates types with its own `HostBounds`; the per-backend monomorphizations coexist in the binary. `ActiveCpuBounds` is the convenience alias for code that does not parameterize over backend.

---

## Part IV — Type Vocabulary (`hologram-types`)

Every hologram domain type is declared via `ConstrainedTypeShape`. There are no hand-rolled struct types representing domain concepts that have a natural `ConstrainedTypeShape` form.

This crate is `#![no_std]`. It depends on `hologram-host` and `uor-foundation`.

### IV.1 Constraint vocabulary in use

Hologram declarations use only the upstream `ConstraintRef` variants. The semantic meaning of each variant in upstream is:

| Variant | Upstream meaning |
|---|---|
| `Residue { modulus, residue }` | value ≡ residue (mod modulus) |
| `Hamming { bound }` | bit-weight bound |
| `Depth { min, max }` | site-depth bound (depth in the constraint graph, not bit width) |
| `Carry { site }` | carry-bit relation at a given site |
| `Site { position }` | site-position restriction (which sites are pinned) |
| `Affine { coefficients, coefficient_count, bias }` | Σ c_i · site_i = bias |
| `SatClauses { clauses, num_vars }` | Boolean clause list |
| `Bound { observable_iri, bound_shape_iri, args_repr }` | parametric observable bound |
| `Conjunction { conjuncts, conjunct_count }` | conjunction of leaf constraints |

Upstream constants:
- `pipeline::AFFINE_MAX_COEFFS = 8` — fixed array length for `Affine` coefficients.
- `pipeline::CONJUNCTION_MAX_TERMS = 8` — fixed array length for `Conjunction`.

Hologram does not extend this vocabulary. Where a hologram type's structure cannot be naturally encoded with these constraint variants, the type's `CONSTRAINTS` is empty and the structural information lives in the IRI + the type's generic parameters; the compiler relies on those, not on the constraint list.

### IV.2 The IRI scheme

Every hologram type's `IRI` follows this pattern:

```
https://hologram.uor.foundation/type/<category>/<name>[<rank|N|...>]
```

Hologram is **introducing this IRI namespace** as a Prism extension; the IRIs are not in upstream's ontology. Resolver behavior is hologram-controlled (the `hologram-compiler`'s validation paths know about these IRIs). Per ADR-013, this introduces **types**, not new `PrimitiveOp` discriminants — types-as-IRIs are an explicit application surface.

Categories:
- `dtype` — element types
- `shape` — shape components (Dim, rank-N tuples)
- `tensor` — tensors
- `region`, `layout`, `weight`, `schedule`, `fingerprint`, `witness` — model and runtime carriers

### IV.3 Const-generic policy

Hologram type declarations rely on the stable `min_const_generics` feature only. The const-generic arithmetic in the v0.5.0-draft Tensor declaration (`SITE_COUNT = S::SITE_COUNT + D::SITE_COUNT`) requires nightly `generic_const_exprs`; **hologram does not commit to nightly**. Site counts are therefore declared either as concrete numbers per monomorphization or as an upper bound passed as a separate const generic.

The committed pattern is the **upper-bound** form:

```rust
// SITE_COUNT is an upper-bound passed at the type level. Concrete instances
// (e.g., Tensor<Shape2<Dim<128>, Dim<128>>, DTypeF32, Bounds, 16384>) provide
// the numeric value at the call site.
pub struct Tensor<S, D, B, const SITES: usize>(core::marker::PhantomData<(S, D, B)>)
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<S, D, B, const SITES: usize> ConstrainedTypeShape for Tensor<S, D, B, SITES>
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    const IRI: &'static str = "https://hologram.uor.foundation/type/tensor";
    const SITE_COUNT: usize = SITES;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
```

The compiler computes `SITES` for each `Graph` node it monomorphizes, threading it as the trailing const generic. Nightly arithmetic features are not required.

### IV.4 Dtype declarations

Each dtype is a leaf `ConstrainedTypeShape` whose IRI carries the bit-width and signedness. The constraint list is empty; bit-width is recovered from the IRI by hologram-side resolvers, **not from `Site::position`**. (Upstream `Site::position` does not mean "bit-width" — it means a site index restriction.)

```rust
// crates/hologram-types/src/dtype.rs

use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};

/// Marker trait identifying a hologram dtype. Used by `Tensor<S, D, B, SITES>`
/// to constrain the dtype slot. Carries the bit width as a const, recovered
/// by `hologram-compiler` during monomorphization.
pub trait DType: ConstrainedTypeShape {
    /// Bit width of one element of this dtype.
    const BIT_WIDTH: u32;
    /// IEEE-754 / signed-integer / unsigned-integer / boolean kind tag.
    const KIND: DTypeKind;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeKind { Float, SignedInt, UnsignedInt, Bool, Bfloat }

pub struct DTypeF32;
impl ConstrainedTypeShape for DTypeF32 {
    const IRI: &'static str = "https://hologram.uor.foundation/type/dtype/f32";
    const SITE_COUNT: usize = 1;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
impl DType for DTypeF32 { const BIT_WIDTH: u32 = 32; const KIND: DTypeKind = DTypeKind::Float; }

pub struct DTypeF16;  // BIT_WIDTH = 16, KIND = Float
pub struct DTypeBf16; // BIT_WIDTH = 16, KIND = Bfloat
pub struct DTypeF64;  // BIT_WIDTH = 64, KIND = Float
pub struct DTypeI64;  // BIT_WIDTH = 64, KIND = SignedInt
pub struct DTypeI32;  // BIT_WIDTH = 32, KIND = SignedInt
pub struct DTypeI8;   // BIT_WIDTH = 8,  KIND = SignedInt
pub struct DTypeU64;  // BIT_WIDTH = 64, KIND = UnsignedInt
pub struct DTypeU8;   // BIT_WIDTH = 8,  KIND = UnsignedInt
pub struct DTypeBool; // BIT_WIDTH = 1,  KIND = Bool
// Each: identical ConstrainedTypeShape impl pattern, distinct IRI suffix.
```

### IV.5 Shape declarations

A shape is a rank-tagged tuple of dimensions. Each dimension is its own `ConstrainedTypeShape` carrying an `Affine` constraint that pins its size.

```rust
// crates/hologram-types/src/shape.rs

use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef, AFFINE_MAX_COEFFS};

/// A static dimension carrying a known size N. `Affine` constraint
/// asserts `1·site_0 = N` — the dimension's site equals N.
pub struct Dim<const N: u64>;

impl<const N: u64> ConstrainedTypeShape for Dim<N> {
    const IRI: &'static str = "https://hologram.uor.foundation/type/shape/dim";
    const SITE_COUNT: usize = 1;
    const CONSTRAINTS: &'static [ConstraintRef] = &[
        ConstraintRef::Affine {
            coefficients: dim_coefficients(),
            coefficient_count: 1,
            bias: N as i64,
        },
    ];
}

const fn dim_coefficients() -> [i64; AFFINE_MAX_COEFFS] {
    let mut c = [0i64; AFFINE_MAX_COEFFS];
    c[0] = 1;
    c
}

/// A symbolic dimension. No `Affine` pinning; resolved at graph-build time.
pub struct DimSymbolic<const ID: u64>;
impl<const ID: u64> ConstrainedTypeShape for DimSymbolic<ID> {
    const IRI: &'static str = "https://hologram.uor.foundation/type/shape/dim_symbolic";
    const SITE_COUNT: usize = 1;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}

/// Rank-N shape markers. Each provides a `ConstrainedTypeShape` impl with
/// an explicit numeric `SITE_COUNT` (sum of component site counts) supplied
/// at the call site as a trailing const generic, per IV.3.
pub struct Shape1<D0, const SITES: usize>(core::marker::PhantomData<D0>);
pub struct Shape2<D0, D1, const SITES: usize>(core::marker::PhantomData<(D0, D1)>);
pub struct Shape3<D0, D1, D2, const SITES: usize>(core::marker::PhantomData<(D0, D1, D2)>);
pub struct Shape4<D0, D1, D2, D3, const SITES: usize>(core::marker::PhantomData<(D0, D1, D2, D3)>);
// Shape5..Shape8 analogous.

impl<D0, D1, const SITES: usize> ConstrainedTypeShape for Shape2<D0, D1, SITES>
where D0: ConstrainedTypeShape, D1: ConstrainedTypeShape,
{
    const IRI: &'static str = "https://hologram.uor.foundation/type/shape/rank2";
    const SITE_COUNT: usize = SITES;
    const CONSTRAINTS: &'static [ConstraintRef] = &[]; // structure is in the type generics
}
```

`CartesianProductShape` (upstream) is implemented for `Shape2..Shape8` to enable Künneth Betti composition. The `Left` / `Right` associated types pair adjacent dimension components.

### IV.6 Tensor type

The central hologram domain type:

```rust
// crates/hologram-types/src/tensor.rs

use uor_foundation::pipeline::{ConstrainedTypeShape, ConstraintRef};
use uor_foundation::HostBounds;

/// Tensor with shape `S`, dtype `D`, host bounds `B`, and aggregate site
/// count `SITES`. `SITES` is the product of dimension sizes; the compiler
/// supplies it at monomorphization.
pub struct Tensor<S, D, B, const SITES: usize>(core::marker::PhantomData<(S, D, B)>)
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<S, D, B, const SITES: usize> ConstrainedTypeShape for Tensor<S, D, B, SITES>
where
    S: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    const IRI: &'static str = "https://hologram.uor.foundation/type/tensor";
    const SITE_COUNT: usize = SITES;
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}
```

**`B::WITT_LEVEL_MAX_BITS` is not encoded as a constraint on Tensor.** It governs which `WittLevel` the compiler passes to `pipeline::run_tower_completeness` and which level the Term tree's `Term::Literal` and `Term::Lift`/`Term::Project` nodes target. Witt-level discipline is enforced at compile-unit construction time, not via constraint-shape predicates.

**Tensor data ≠ tensor type.** `Tensor<S, D, B, SITES>` is the *type*. The actual tensor data (the floating-point or integer bytes) lives in workspace buffers managed by `hologram-exec::BufferArena`. The `ConstrainedTypeShape` declaration describes the type's identity for upstream's pipeline; it never holds a million bytes.

### IV.7 Region, Layout, Weight, Schedule, Fingerprint, Witness

These are auxiliary domain types. Each has a `ConstrainedTypeShape` declaration with a hologram-namespaced IRI; structural information that doesn't fit upstream's constraint vocabulary lives in the type's generic parameters and is consumed by hologram-side resolvers.

```rust
// crates/hologram-types/src/region.rs
pub struct Region;
impl ConstrainedTypeShape for Region {
    const IRI: &'static str = "https://hologram.uor.foundation/type/region";
    const SITE_COUNT: usize = 2; // (offset, length) — two integer sites
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}

// crates/hologram-types/src/layout.rs
pub struct Layout<const RANK: usize, const SITES: usize>;
// `SITES` = RANK (strides) + 2 (alignment, byte_order).
// Empty CONSTRAINTS; structure in generics.

// crates/hologram-types/src/weight.rs
pub struct Weight<D, B, const SITES: usize>(core::marker::PhantomData<(D, B)>)
where D: ConstrainedTypeShape, B: HostBounds;
// SITES = D::SITE_COUNT + 1 (one extra for fingerprint reference).

// crates/hologram-types/src/schedule.rs
pub struct Schedule<const LEVELS: usize>;
// SITE_COUNT = LEVELS

// crates/hologram-types/src/fingerprint.rs
pub struct Fingerprint;
impl ConstrainedTypeShape for Fingerprint {
    const IRI: &'static str = "https://hologram.uor.foundation/type/fingerprint";
    const SITE_COUNT: usize = 32; // 32 bytes = 32 W8 sites
    const CONSTRAINTS: &'static [ConstraintRef] = &[];
}

// crates/hologram-types/src/witness.rs
pub struct WitnessRecord<Op>(core::marker::PhantomData<Op>);
// SITE_COUNT, IRI depend on the witnessed op kind via a per-op impl.
```

`Schedule`, `Layout`, and similar types are not constraint-system-rich — their structure is data, not algebraic relations. They appear here so that hologram-compiler can mention them in `CompileUnit::result_type_iri` and round-trip them through upstream's reduction pipeline (which validates IRI presence and SITE_COUNT but does not interpret structural data).

### IV.8 Type catalog (complete enumeration)

```
Dtypes (10):
  DTypeF32, DTypeF16, DTypeBf16, DTypeF64,
  DTypeI64, DTypeI32, DTypeI8, DTypeU64, DTypeU8, DTypeBool

Shapes:
  Dim<const N: u64>
  DimSymbolic<const ID: u64>
  Shape1, Shape2, Shape3, Shape4, Shape5, Shape6, Shape7, Shape8
  ShapeArray<const RANK: usize, const SITES: usize>  (heap variant, rank > 8)

Tensors:
  Tensor<S, D, B, const SITES: usize>

Memory:
  Region
  Layout<const RANK: usize, const SITES: usize>

Model:
  Weight<D, B, const SITES: usize>
  Constant<D, B, const SITES: usize>
  Schedule<const LEVELS: usize>

Identity:
  Fingerprint  (re-exports `ContentAddress` and `ContentFingerprint<32>` from upstream)

Witness:
  WitnessRecord<Op>          // one per op marker in V.3 / V.4

Activation operand types (LUT specialization markers):
  ActivationOperand<F: ActivationFn>  // one per LUT-eligible activation
```

Anything not in this catalog and not in `uor_foundation::*` is a violation of I-8 and must be deleted. Reconciliation mapping (Part XI.3) lists every current symbol's replacement.

---

## Part V — Operation Pipelines (`hologram-ops`)

Every hologram operation is a **Term arena tree** built from upstream's `enforcement::Term` enum. The tree is the formal specification of the operation; the native backend kernel (Part IX) is its execution form.

This crate is `#![no_std]`. It depends on `hologram-types`, `hologram-host`, and `uor-foundation`. It uses `alloc` only when a `std`-feature is enabled (for tests / hosted builds); core hologram-ops API surfaces are stack-arena-only.

### V.1 The vocabulary

The complete computational vocabulary is `enforcement::Term`:

```rust
pub enum Term {
    Literal     { value: u64, level: WittLevel },
    Variable    { name_index: u32 },
    Application { operator: PrimitiveOp, args: TermList },
    Lift        { operand_index: u32, target: WittLevel },   // canonical injection W_n → W_m, n < m
    Project     { operand_index: u32, target: WittLevel },   // canonical surjection W_m → W_n, m > n
    Match       { scrutinee_index: u32, arms: TermList },
    Recurse     { measure_index: u32, base_index: u32, step_index: u32 },  // bounded recursion
    Unfold      { seed_index: u32, step_index: u32 },        // stream construction
    Try         { body_index: u32, handler_index: u32 },
}
```

`PrimitiveOp` is the closed 10-discriminant set: `Neg`, `Bnot`, `Succ`, `Pred`, `Add`, `Sub`, `Mul`, `Xor`, `And`, `Or`. There is no other operator vocabulary.

Carriers and structures:

- `TermArena<const CAP: usize>` — fixed-capacity stack-resident arena. CAP is per-op, chosen at hologram-ops definition time to bound the term tree depth.
- `TermList { start: u32, len: u32 }` — slice into an arena.
- `Binding { name_index, type_index, value_index, surface, content_address }` — named compile-time constants referenced by `Term::Variable`.
- `BindingsTable` — `&'static [BindingEntry]`, sorted by content address; binary-search-resolved at compile time.
- `CompileUnit<'a> { level: WittLevel, budget, result_type_iri, root_term: &'a [Term], bindings: &'a [Binding], target_domains }` — what `pipeline::run_tower_completeness` consumes (after validation).

**Loops, fan-out, reductions:** expressed via `Term::Recurse`. The `measure_index` term must produce a strictly-decreasing nonnegative quantity per iteration. For a sum over `k ∈ [0, K)`, the measure is `K - k`. There is no `cartesian_fold` or `accumulate` combinator — those are nests of `Recurse` over `Application(Add, …)`.

**Branching:** `Term::Match` over a scrutinee with arm terms. `Term::Try` for failure recovery.

**Streams (UHD frame ingest, token streams):** `Term::Unfold { seed, step }`.

**Cross-Witt-level operations:** `Term::Lift` widens; `Term::Project` narrows. Hologram operations that mix levels (e.g., accumulate W64 products into a W128 partial sum) use these explicitly — there is no implicit promotion.

### V.2 The op-emitter pattern

Each canonical hologram op is a **type marker** plus a **const fn that emits a Term tree** into a caller-provided arena. The marker is the IRI handle; the const fn is the formal definition.

```rust
// crates/hologram-ops/src/matmul.rs

use uor_foundation::enforcement::{Term, TermArena, TermList};
use uor_foundation::{PrimitiveOp, WittLevel};

/// MatMul iso marker: (Tensor[M,K], Tensor[K,N]) → Tensor[M,N].
///
/// The marker carries shape and dtype as type parameters so that the
/// compiler can monomorphize the emitter per concrete (M, K, N, dtype, level).
pub struct MatMulOp<const M: u64, const K: u64, const N: u64, D, B>(
    core::marker::PhantomData<(D, B)>,
)
where
    D: hologram_types::DType,
    B: uor_foundation::HostBounds;

impl<const M: u64, const K: u64, const N: u64, D, B> MatMulOp<M, K, N, D, B>
where
    D: hologram_types::DType,
    B: uor_foundation::HostBounds,
{
    /// IRI under which this iso is declared. Resolves to the same string
    /// hologram-types uses for the result `Tensor` type, suffixed by the
    /// op identity.
    pub const IRI: &'static str =
        "https://hologram.uor.foundation/op/linear-algebra/matmul";

    /// Emit the canonical Term tree for this iso into the provided arena.
    /// Returns the index of the root node.
    ///
    /// The tree shape is:
    ///
    ///   Recurse over (i,j) ∈ [0,M)×[0,N)              (outer)
    ///     base:  Literal { 0, level }
    ///     step:  Application(Add, [acc,
    ///              Recurse over k ∈ [0,K)             (inner)
    ///                base:  Literal { 0, level }
    ///                step:  Application(Add, [acc,
    ///                         Application(Mul,
    ///                           [Variable(a[i,k]),
    ///                            Variable(b[k,j])])])
    ///             ])
    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
    ) -> Option<u32> {
        // Bottom-up arena construction. Returns Some(root_index) on success,
        // None if the arena overflowed.
        // ... (concrete implementation lives in matmul.rs)
        unimplemented!("see matmul.rs for the emitter body")
    }
}
```

The emitter is a `const fn` where the language allows; otherwise an ordinary `fn` since arena `push` is mutable. Per-op `CAP` requirements are enumerated in V.5.

**Why a type marker, not a value:** the marker carries shape constants in its generics. Two MatMul instantiations at different (M, K, N) are distinct types — the compiler monomorphizes the emitter and the downstream native kernel separately. There is no runtime branching on shape.

### V.3 The 64-op catalog

The closed set of canonical ops. Each entry maps to one type-marker definition + one `emit_term` function in `hologram-ops`. The "decomposition" column states the Term-tree skeleton; concrete arena layouts are in the per-op source files.

**Direct PrimitiveOp wrappers (10 ops).** These are not really compositions — they are single `Term::Application { operator: PrimitiveOp::*, args }` trees over the input variables.

```
NegOp, BnotOp, SuccOp, PredOp, AddOp, SubOp, MulOp, XorOp, AndOp, OrOp
```

**Elementwise unary (14 non-primitive ops).** Per-element `Term::Recurse` over the tensor; inner step is the activation's structural decomposition.

```
ReluOp        — Match { x < 0 → 0, otherwise → x }, expressed via Sub + sign-bit And
SigmoidOp     — 1 / (1 + exp(-x))
TanhOp        — (exp(2x) − 1) / (exp(2x) + 1)
GeluOp        — 0.5 · x · (1 + erf(x / √2))
SiluOp        — x · sigmoid(x)
EluOp, SeluOp — piecewise via Match
ExpOp, LogOp, Log1pOp, SqrtOp, ReciprocalOp
              — fixed-degree polynomial / Newton iteration; finite Term::Recurse over Add+Mul
SinOp, CosOp, TanOp, AsinOp, AcosOp, AtanOp
              — CORDIC: Term::Recurse with shift+add steps via PrimitiveOp::{Add, Sub, And, Xor}
CeilOp, FloorOp, RoundOp
              — bit-pattern Application chains using And / Or / Xor / Sub
ErfOp         — Chebyshev-truncated polynomial: Term::Recurse over Add+Mul
IsNaNOp       — bit-pattern And + Equal-to-quiet-NaN-mask
SignOp        — sign-bit isolation via And + shift (shift = repeated PrimitiveOp::Add or BitNot pattern)
AbsOp         — Xor with sign-mask + Add(1) when negative
```

> **Note on float decomposition.** Hologram operates on `Z/(2^n)Z` carriers. Float dtypes (F32, F16, BF16) are encoded by their bit pattern as `WittLevel::W{32,16,16}` integers; arithmetic on them uses bit-level Add / Mul / Xor / And implementing the IEEE-754 algorithm. This is interpretation-A native code in the backend kernel; the Term tree witnesses the bit-level decomposition. See V.7 for the float encoding contract.

**Elementwise binary (13 non-primitive ops).** Per-element `Recurse` with two-input step.

```
DivOp         — Newton-Raphson iteration: bounded Recurse over Mul + Sub
PowOp         — Term tree composing ExpOp + MulOp + LogOp (or fast-path Recurse for integer exponent)
ModOp         — Sub-based: x − ⌊x/y⌋·y, expressed via Mul + Sub
MinOp, MaxOp  — Match on sign of (a − b)
EqualOp, LessOp, LessOrEqualOp, GreaterOp, GreaterOrEqualOp
              — Sub + sign bit isolation + And-to-1
AndOp, OrOp, XorOp — direct PrimitiveOp wrappers (already counted in the 10)
```

**Linear algebra (2 ops).** Doubly-nested `Recurse`; sketch shown in V.2.

```
MatMulOp<M, K, N, D, B>            — outer (i,j), inner k; Mul + Add
GemmOp<M, K, N, D, B>              — MatMulOp + scalar Mul (α) + Mul (β) + Add
```

**Convolution (2 ops).** 4-deep `Recurse` (output_h, output_w, kernel_h, kernel_w) over Mul + Add.

```
Conv2dOp<X, W, P, S, D, B>
ConvTranspose2dOp<X, W, P, S, D, B>
```

**Normalization (5 ops).** Sequential composition of reductions, scalar arithmetic, and per-element scaling.

```
LayerNormOp<S, D, B>      — ReduceMean → Sub → Mul → ReduceMean → Sqrt → Div → Mul → Add
RmsNormOp<S, D, B>        — Mul → ReduceMean → Sqrt → Div → Mul
GroupNormOp<S, D, B>      — group-partitioned LayerNormOp
InstanceNormOp<S, D, B>   — batch-partitioned LayerNormOp
AddRmsNormOp<S, D, B>     — Add + RmsNormOp (fused tree, no intermediate buffer)
```

**Reduction (5 ops).** Single `Recurse` over the reduction axes.

```
ReduceSumOp<S, Axes, D, B>    — Recurse over axes, step = Add
ReduceMeanOp<S, Axes, D, B>   — ReduceSumOp + Mul (1/count, precomputed Binding)
ReduceProdOp<S, Axes, D, B>   — Recurse, step = Mul
ReduceMinOp<S, Axes, D, B>    — Recurse, step = Match (a < b ? a : b)
ReduceMaxOp<S, Axes, D, B>    — Recurse, step = Match (a > b ? a : b)
```

**Layout (4 ops, no compute Term tree).** These are address-relabel operations. Their "Term tree" is a single `Term::Variable` referencing a remapped binding produced by the compiler's address resolver — no `Application` nodes are emitted. The Validated certificate confirms the bijection of the relabel.

```
ReshapeOp<S_in, S_out, D, B>
TransposeOp<S, Perm, D, B>
ConcatOp<Axis, Inputs, D, B>
SliceOp<S_in, Starts, Ends, D, B>
```

**Activation+reduce (2 ops).**

```
SoftmaxOp<S, Axis, D, B>      — ReduceMaxOp → Sub → ExpOp → ReduceSumOp → Div
LogSoftmaxOp<S, Axis, D, B>   — SoftmaxOp + LogOp
```

**Pooling (3 ops).**

```
MaxPool2dOp<X, K, S, D, B>      — windowed Recurse, step = Match (max)
AvgPool2dOp<X, K, S, D, B>      — windowed Recurse, step = Add; final Mul (1/window)
GlobalAvgPoolOp<S, D, B>        — ReduceMeanOp over spatial axes
```

**Structured composition (2 ops).**

```
AttentionOp<Q, K, V, D, B>    — MatMulOp(Q, Kᵀ) → Mul (1/√d, precomputed) → SoftmaxOp → MatMulOp(_, V)
FusedSwiGluOp<X, W, D, B>     — GemmOp + SiluOp + Mul (gate)
```

**Utility (8 ops).**

```
PadOp<S_in, Pad, D, B>            — address-remap layout op
ExpandOp<S_in, S_out, D, B>       — address-remap layout op
ResizeOp<S_in, S_out, D, B>       — bilinear: Mul + Add over neighbor lookups
CumSumOp<S, Axis, D, B>           — prefix-sum: Recurse with step = Add over running accumulator
RotaryEmbeddingOp<S, D, B>        — CosOp + SinOp + Mul + Add (rotation)
ClipOp<S, Lo, Hi, D, B>           — MinOp(MaxOp(x, Lo), Hi)
LrnOp<S, D, B>                    — windowed Recurse (Mul + Add), then ReciprocalOp + Mul
WhereOp<S, D, B>                  — Term::Match { cond → a, otherwise → b }
```

**Total: 10 direct + 14 unary + 13 binary + 27 structured = 64 op markers, each with one `emit_term` function.**

The decompositions above are the **specification** of each op's semantics. They are *not* the executor's hot-path code. Backend kernels (Part IX) implement these semantics natively (CPU SIMD intrinsics, Metal compute kernels, WGSL shaders) and the Validated certificate attests that the kernel's behavior matches the Term tree (Part XII verification).

### V.4 Backward rules

Each differentiable op declares a companion op marker for its backward, with its own `emit_term` function:

```
MatMulGradAOp, MatMulGradBOp                           // transposed MatMulOp variants
Conv2dGradXOp, Conv2dGradWOp                           // transposed Conv2dOp variants
SoftmaxGradOp, LogSoftmaxGradOp
LayerNormGradOp, RmsNormGradOp, GroupNormGradOp
ReduceSumGradOp, ReduceMeanGradOp, ReduceProdGradOp
SubGradOp, MulGradOp, DivGradOp, PowGradOp
MinGradOp, MaxGradOp
ConcatGradOp, SliceGradOp
AvgPool2dGradOp, GlobalAvgPoolGradOp
PadGradOp
AttentionGradOp
FusedSwiGluGradOp
UnaryGradOp                                            // one per unary (Relu, Sigmoid, Tanh, ...)
```

Backward emitters return Term trees for gradient computation. The compiler emits them ahead of time at graph build (ADR-043 — backward is planned, not traversed).

### V.5 TermArena capacity bounds (CAP) per op

Each op pins a `CAP` constant — the maximum number of `Term` nodes its tree can occupy. These are upper bounds, not exact counts; `emit_term` returns `None` if the arena overflows.

| Op category | CAP |
|---|---|
| Direct PrimitiveOp wrappers | 4 |
| Elementwise unary (Relu / Sigmoid / Tanh / etc.) | 32 |
| Transcendentals (Exp / Log / Sin / CORDIC) | 64 |
| Elementwise binary | 16 |
| Layout (no-compute) | 2 |
| MatMul / Gemm | 32 |
| Conv2d / ConvTranspose2d | 64 |
| Reduction | 16 |
| Normalization | 64 |
| Pooling | 32 |
| Softmax / LogSoftmax | 32 |
| Attention | 96 |
| FusedSwiGlu | 64 |
| Utility | 32–64 |

These CAPs are conservative ceilings; they constrain stack-resident arena size. The compiler-side `TermArena<CAP>` for the largest op caps stack usage at `CAP × sizeof(Option<Term>) ≈ CAP × 64 bytes`. The largest single op (Attention at CAP = 96) stays under 6 KiB stack.

### V.6 LUT-fused activation specializations

Activations on low-Witt-level inputs (W8, W16) admit a precomputed-LUT specialization. The LUT does **not** replace the Term tree — the Term tree is the formal spec. The LUT is a backend-kernel optimization: the CPU/Metal/wgpu kernel for `SigmoidOp<DTypeU8, B>` reads from a `&'static [u8; 256]` table built at compile time by evaluating the same arithmetic the Term tree expresses. Both paths produce the same Validated certificate.

LUT generation is owned by `hologram-ops`:

```rust
// crates/hologram-ops/src/lut.rs

/// Compile-time LUT for a unary activation at Witt level W8.
/// Returns a `[u8; 256]` table built by evaluating the activation at every byte input.
/// `F` is one of the activation marker types from V.3; `eval_w8` is its
/// scalar reference implementation (used both for LUT build and for kernel
/// equivalence testing).
pub const fn build_w8_lut<F: ActivationFn>() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = F::eval_w8(i as u8);
        i += 1;
    }
    t
}

/// Reference scalar evaluation for an activation marker. The compiler uses
/// this to (a) build the LUT and (b) test-time-verify the backend kernel
/// against the reference. Not the hot path.
pub trait ActivationFn {
    fn eval_w8(x: u8) -> u8;
    fn eval_w16(x: u16) -> u16;
    fn eval_f32(x: f32) -> f32;
}
```

For W32 and above, no LUT is built; the kernel evaluates the activation arithmetically.

### V.7 Float dtype encoding contract

Hologram supports F32, F16, BF16, F64 dtypes. These are *not* native float types in the Term tree — they are bit patterns at WittLevel::W{32,16,16,64}. The Term tree expresses IEEE-754 arithmetic via bit-level `PrimitiveOp` compositions (sign extraction, exponent alignment, mantissa multiplication, rounding). This is the formal spec.

The backend kernel (Part IX) executes IEEE-754 ops natively (`vfmadd231ps`, `fmla`, etc.) — it does NOT walk the bit-level Term tree at runtime. The Validated certificate attests the kernel's float arithmetic matches the bit-level decomposition modulo IEEE-754's deterministic rounding behavior.

**Decimal-typed observables** (entropy, sigma, jacobian) flow through `H::Decimal = f64` (per `HologramHostTypes` in III.1) and use upstream's `DecimalTranscendental` operations directly — not bit-level decomposition. These observables are NOT compute-path; they are budget / metrics quantities.

### V.8 What does NOT exist in `hologram-ops`

- No `Grounding` impls for canonical ops. (Grounding is W4 input parsing — see VII.7 for the boundary `Grounding` impls hologram does have.)
- No `combinators::cartesian_fold` / `accumulate` / `compose_two` / `fmap` / `select` / `lookup` / `lift`. These do not exist upstream. Loop and reduction structure is `Term::Recurse`.
- No `user::morphism::Composition` impl per op. `Composition<H>` is an ontology-shape trait declared at the type level; hologram operations are Term trees. If a future hologram release wants to publish op-composition laws as ontology entries, that is a separate exercise.
- No new `PrimitiveOp` variants. The closed 10 are the only operators that appear in any `Term::Application`.
- No `GroundingProgram` for compute. `GroundingProgram` is the W4 input-parsing carrier; it does not carry computation.

---

## Part VI — Graph IR (`hologram-graph`)

`hologram-graph` is the IR `hologram-compiler` operates on. It is the only crate that holds mutable graph state. `#![no_std]` + `alloc`. Depends on `hologram-ops` and `hologram-types`.

### VI.1 Graph

```rust
// crates/hologram-graph/src/graph.rs

pub struct Graph {
    nodes: alloc::vec::Vec<Node>,
    inputs: smallvec::SmallVec<[NodeId; 8]>,
    outputs: smallvec::SmallVec<[NodeId; 8]>,
    constants: ConstantStore,
    schedule: Option<Schedule>,
}

pub struct Node {
    op: GraphOp,
    inputs: smallvec::SmallVec<[InputSource; 4]>,
    output_dtype: DTypeId,
    output_shape: ShapeId,
}

pub enum GraphOp {
    /// Reference to a `Grounding` impl in `hologram-ops`. Every variant
    /// of this enum corresponds to exactly one `Grounding` impl in V.2.
    /// Adding a new variant requires adding the corresponding `Grounding`
    /// impl in `hologram-ops` and updating the compiler dispatch table.
    Op(OpKind),

    /// Structural — no compute.
    Input,
    Output,
    Constant(ConstantId),
}

pub enum OpKind {
    // Direct PrimitiveOp wrappers
    Neg, Bnot, Succ, Pred, Add, Sub, Mul, Xor, And, Or,
    // Elementwise unary
    Relu, Sigmoid, Tanh, Gelu, Silu, Elu, Selu, Exp, Log, Log1p, Sqrt,
    Reciprocal, Sin, Cos, Tan, Asin, Acos, Atan, Ceil, Floor, Round, Erf,
    IsNaN, Sign, Abs,
    // Elementwise binary
    Div, Pow, Mod, Min, Max, Equal, Less, LessOrEqual, Greater, GreaterOrEqual,
    // Linear algebra
    MatMul, Gemm,
    // Convolution
    Conv2d, ConvTranspose2d,
    // Normalization
    LayerNorm, RmsNorm, GroupNorm, InstanceNorm, AddRmsNorm,
    // Reduction
    ReduceSum, ReduceMean, ReduceProd, ReduceMin, ReduceMax,
    // Layout
    Reshape, Transpose, Concat, Slice,
    // Activation+reduce
    Softmax, LogSoftmax,
    // Pooling
    MaxPool2d, AvgPool2d, GlobalAvgPool,
    // Structured
    Attention, FusedSwiGlu,
    // Utility
    Pad, Expand, Resize, CumSum, RotaryEmbedding, Clip, Lrn, Where,
    // Backward variants (per V.3) — same enumeration, suffixed.
}
```

`OpKind` is a closed enum. It is the on-disk serialization surface. Adding a hologram operation means: (a) write a new `Grounding` impl in `hologram-ops`, (b) add the variant here, (c) wire the compiler dispatch arm.

`OpKind` has **exactly one variant per `Grounding` impl in V.2 + V.3**. There is no other place where operations are enumerated.

### VI.2 IDs

```rust
pub struct NodeId(u32);
pub struct ConstantId(u32);
pub struct DTypeId(u8);   // index into hologram-types::dtype catalog
pub struct ShapeId(u32);  // intern key into hologram-types::ShapeRegistry
```

Stable across compilation; not exposed to runtime.

### VI.3 Schedule

```rust
pub struct Schedule {
    levels: alloc::vec::Vec<smallvec::SmallVec<[NodeId; 16]>>,
}
```

Schedule is computed by topological sort + level grouping. Nodes at the same level execute in parallel (rayon if enabled).

### VI.4 ShapeRegistry

```rust
pub struct ShapeRegistry {
    shapes: alloc::vec::Vec<ShapeDescriptor>,
}

pub struct ShapeDescriptor {
    rank: u8,
    dims: [u64; 8],   // small-rank fast path
    dims_overflow: Option<alloc::boxed::Box<[u64]>>,
}
```

Backed by `Shape1..Shape8` + `ShapeArray` from `hologram-types`. The registry is a runtime-side intern table that maps `ShapeId` to a concrete descriptor.

---

## Part VII — Compiler (`hologram-compiler`)

The compiler converts `Graph` to a `.holo` archive containing:
- A monomorphic Prism pipeline per op (lowered to the active backend)
- Weight blobs with content-addressed dedup
- An execution schedule
- A composite `Validated<LiftChainCertificate>` produced by `pipeline::run_tower_completeness` over the full compile unit

`#![no_std]` impossible (uses BLAKE3, mmap, threading); plain `std`. Depends on `hologram-graph`, `hologram-archive`, `hologram-host`, `uor-foundation`.

### VII.1 Entry points

```rust
// crates/hologram-compiler/src/lib.rs

pub struct Compiler {
    graph: Graph,
    target: BackendKind,
    cache: CertificateCache,
}

pub enum BackendKind {
    Cpu, Avx2, Avx512, Neon, Metal, Wgpu,
}

pub struct CompilationOutput {
    pub archive: alloc::vec::Vec<u8>,
    pub stats: CompilationStats,
    pub schedule: Schedule,
}

impl Compiler {
    pub fn new(graph: Graph, target: BackendKind) -> Self;
    pub fn compile(self) -> Result<CompilationOutput, CompileError>;
}

/// Convenience: parse UOR source -> Graph -> compile.
pub fn compile_from_source(
    source: &str,
    level: WittLevel,
    target: BackendKind,
) -> Result<CompilationOutput, CompileError>;
```

### VII.2 Granularity: per-node CompileUnit

The compiler builds **one `CompileUnit` per graph Node**, not one per graph. Rationale:

- Each Node has one result type (`Tensor<S, D, B, SITES>`), which fits `CompileUnit::result_type_iri` cleanly.
- Per-node certificates compose into a graph-level certificate via the archive (concatenated `Validated<LiftChainCertificate>` array; the archive's BLAKE3 footer hashes the concatenation).
- Per-node caching by `ContentFingerprint<32>` allows cross-node reuse (e.g., two MatMuls of the same shape share a cached certificate).
- No upstream API requires a graph-wide CompileUnit; the per-node form is the natural granularity.

For each `Node` in topological order:

1. **Lookup the op marker type** for `node.op_kind` in `hologram-ops` (the OpKind → OpMarker map is a const dispatch table generated by a macro in `hologram-compiler::dispatch`).
2. **Resolve concrete type parameters**: shape generics (Dim<N> for static, DimSymbolic for runtime), dtype, host bounds for the active backend, SITES.
3. **Emit the Term tree** by calling `OpMarker::emit_term(&mut arena, witt_level)` with a stack-resident `TermArena<CAP>` sized per V.5.
4. **Build a `CompileUnit`** via `enforcement::CompileUnitBuilder`:
   - `.root_term(arena.as_slice_terms())` — the emitted Term tree
   - `.bindings(&compile_time_bindings)` — input/weight references (compile-time-resolved BindingsTable)
   - `.witt_level_ceiling(witt_level)`
   - `.thermodynamic_budget(estimated_cost_landauer_bits)`
   - `.target_domains(&[VerificationDomain::Boolean, VerificationDomain::Ring])`
   - `.result_type::<Tensor<S, D, B, SITES>>()`
5. **Validate** via `CompileUnitBuilder::validate()` → `Result<Validated<CompileUnit>, ShapeViolation>`.
6. **Run completeness** via `pipeline::run_tower_completeness::<Tensor<S, D, B, SITES>, HologramHasher>(&result_type_marker, witt_level)` → `Result<Validated<LiftChainCertificate>, GenericImpossibilityWitness>`.
7. **Compute content fingerprint**: fold the (op_marker_iri, validated_compile_unit_bytes, backend_kind) tuple through `HologramHasher` → `ContentFingerprint<32>`. Lookup in `CertificateCache`; on hit, reuse cached `(certificate, kernel_call)`.
8. **Lower to backend kernel call**: invoke the active backend's `Lowerer::lower(op_marker, validated_compile_unit) -> KernelCall`. The kernel call is the executable form (Part IX).
9. **Emit (kernel_call, certificate, fingerprint) into the archive**.

### VII.3 Term-tree-vs-kernel relationship (interpretation A)

**This spec commits to interpretation A.** The Term tree is the formal specification. The native kernel is a parallel implementation that the compiler chooses for performance. The `Validated<LiftChainCertificate>` does not include a derivation that the kernel is bit-identical to the Term tree — it certifies that the Term tree itself is well-formed (Boolean satisfiability, ring closure, budget solvency, etc.) under upstream's reduction pipeline.

**Kernel correctness is verified separately**, via the test discipline in §XII.3:

- For each op marker, `hologram-ops` ships a reference scalar evaluator that walks the Term tree (slow, allocation-free).
- Per-op tests assert `backend.dispatch(kernel_call, &workspace) ≈ reference_eval(term_tree, inputs)` modulo the dtype's tolerance.
- Numerical-parity bench (§XII.4) confirms the equivalence end-to-end.

This is interpretation A, not B. **Interpretation B (compiler-walks-Term-tree-and-emits-machine-code)** is **not** committed to in v0.5.0. A future codegen path could remove the duplicated kernel implementations, but it is out of scope here. The kernels are hand-written / template-generated per backend.

### VII.4 CertificateCache

The only state preserved from the deleted `hologram-cascade::CertificateStore`. Lives entirely in compiler memory:

```rust
// crates/hologram-compiler/src/cache.rs

pub struct CertificateCache {
    map: hashbrown::HashMap<ContentFingerprint<32>, CachedCertificate>,
}

pub struct CachedCertificate {
    certificate: Validated<LiftChainCertificate>,
    kernel_call: KernelCall,
}
```

No persistence to disk; the cache exists only for the duration of one `compile()` call. Cross-compilation memoization happens in the archive itself (BLAKE3-deduped weights).

### VII.5 Preflight removed

The legacy `preflight/` module (shape, type_check, budget_solvency, enforcement_validate) is deleted. Its responsibilities are now:
- **Shape checking**: encoded in `ConstrainedTypeShape::CONSTRAINTS`; verified by `pipeline::preflight_feasibility`.
- **Type checking**: same — type identity is the IRI; mismatches are `ShapeViolation::TypeMismatch`.
- **Budget solvency**: `pipeline::preflight_budget_solvency` upstream.
- **Enforcement validation**: `CompileUnitBuilder::validate()`.

Hologram contributes no preflight code beyond wiring the upstream calls.

### VII.6 Term parsing

The legacy `term_parser` and `term_lower` modules are deleted. The IR is `Graph` directly; if a textual input format is needed, it is parsed to `Graph` via a new `hologram-compiler::source::parse(&str) -> Graph` function that emits arena terms via upstream's `enforcement::TermArena`. ONNX import remains TBD as a separate concern (likely `hologram-onnx` crate, post-v0.5.0).

### VII.7 Where `Grounding` impls live

Per I-10, `Grounding` is reserved for input parsing. The boundary `Grounding` impls hologram ships are:

- `WeightLoaderGrounding<D, B>` — parses a `&[u8]` slice of model-weight bytes (e.g., from a `.holo` archive's weight section, or from an external loader) into a `GroundedTuple<N>` of `GroundedCoord`s at the dtype's Witt level. Used at session-load time.
- `OnnxTensorGrounding<D, B>` — parses an ONNX runtime tensor input (post-v0.5.0; placeholder in v0.5.0).
- `ConstantGrounding<D, B>` — parses graph-embedded constants (the `ConstantStore` entries) into typed values at compile time.

Each uses upstream's `combinators::read_bytes`, `combinators::interpret_le_integer`, and `combinators::then` chains. **No `Grounding` impl in hologram returns a Term tree or invokes a `PrimitiveOp`.** Grounding is parse-only.

---

## Part VIII — Executor (`hologram-exec`)

Runs `.holo` archives. Each archive contains a sequence of compiled kernel calls, one per node, plus the schedule and weight blobs.

`std`. Depends on `hologram-backend`, `hologram-archive`, `hologram-host`.

### VIII.1 InferenceSession

```rust
// crates/hologram-exec/src/session.rs

pub struct InferenceSession<B: Backend> {
    archive: LoadedPlan,
    workspace: BufferArena,
    backend: B,
}

impl<B: Backend> InferenceSession<B> {
    pub fn load(bytes: &[u8], backend: B) -> Result<Self, ExecError>;
    pub fn execute(&mut self, inputs: &[InputBuffer]) -> Result<Vec<OutputBuffer>, ExecError>;
}
```

### VIII.2 Execution loop

```rust
// crates/hologram-exec/src/executor.rs

impl<B: Backend> InferenceSession<B> {
    pub fn execute(&mut self, inputs: &[InputBuffer]) -> Result<Vec<OutputBuffer>, ExecError> {
        for level in self.archive.schedule().levels() {
            // Each level runs in parallel (if rayon enabled) or sequentially.
            for node_id in level {
                let kernel_call = self.archive.kernel_call(*node_id);
                self.backend.dispatch(kernel_call, &mut self.workspace)?;
            }
        }
        Ok(self.collect_outputs())
    }
}
```

No virtual dispatch. `Backend::dispatch` is a generic call that monomorphizes per backend; the `kernel_call` is an `enum KernelCall` matched at the bottom of the dispatch chain.

### VIII.3 BufferArena

```rust
// crates/hologram-exec/src/buffer.rs

pub struct BufferArena {
    storage: alloc::vec::Vec<u8>,
    slots: alloc::vec::Vec<SlotSpan>,
}
```

Slots are computed at compile time from liveness analysis (in `hologram-compiler`); the arena performs no runtime allocation in steady state.

---

## Part IX — Backends (`hologram-backend`)

Per-target dispatch. Each backend declares a `HostBounds` from `hologram-host` and a `Lowerer` impl that translates the closed `OpKind` enum to backend-specific kernels.

`std` (most backends require platform APIs). Depends on `hologram-ops`, `hologram-host`.

### IX.1 Backend trait

```rust
// crates/hologram-backend/src/lib.rs

pub trait Backend {
    type Bounds: HostBounds;
    type Buffer;

    fn dispatch(&mut self, call: &KernelCall, workspace: &mut BufferArena)
        -> Result<(), BackendError>;
}

pub enum KernelCall {
    // One variant per OpKind in hologram-graph::OpKind.
    Add(AddCall),
    Mul(MulCall),
    MatMul(MatMulCall),
    Conv2d(Conv2dCall),
    Attention(AttentionCall),
    LayerNorm(LayerNormCall),
    Softmax(SoftmaxCall),
    // ... 64 variants total.
}
```

### IX.2 CPU backend

```rust
// crates/hologram-backend/src/cpu.rs

pub struct CpuBackend;

impl Backend for CpuBackend {
    type Bounds = hologram_host::ActiveCpuBounds;
    type Buffer = alloc::vec::Vec<u8>;

    fn dispatch(&mut self, call: &KernelCall, workspace: &mut BufferArena)
        -> Result<(), BackendError>
    {
        match call {
            KernelCall::Add(c) => cpu_kernels::add(c, workspace),
            KernelCall::MatMul(c) => cpu_kernels::matmul(c, workspace),
            // ... 64 arms, no fallback, no virtual dispatch.
        }
    }
}
```

The `cpu_kernels::*` functions implement the same operation semantics as the corresponding op marker's Term tree (V.3), expressed natively. They are hand-written or template-generated (per ADR-045 conventions: one file per op under `cpu_kernels/<op>.rs`). For trivial ops they are 5–10 lines of native arithmetic. For matmul they are an inner-loop kernel using SIMD intrinsics (`std::arch::x86_64::*`, `std::arch::aarch64::*`) when `cfg(target_feature)` matches.

**Kernel ↔ Term-tree equivalence is verified by tests, not by codegen.** Per VII.3, hologram-ops ships a reference scalar evaluator that walks the Term tree; per-op tests assert kernel output matches the reference within dtype tolerance.

### IX.3 Metal backend

```rust
// crates/hologram-backend/src/metal.rs (cfg(target_os = "macos"))

pub struct MetalBackend {
    device: metal::Device,
    queue: metal::CommandQueue,
}

impl Backend for MetalBackend {
    type Bounds = HologramHostBoundsMetal;
    type Buffer = metal::Buffer;
    // dispatch routes to .metal compute shaders, one per OpKind.
}
```

### IX.4 wgpu backend

```rust
// crates/hologram-backend/src/wgpu.rs (cfg(feature = "wgpu"))

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Resident workspace (ADR-051) lives here.
}

impl Backend for WgpuBackend {
    type Bounds = HologramHostBoundsWgpu;
    type Buffer = wgpu::Buffer;
    // dispatch routes to .wgsl compute shaders, one per OpKind.
}
```

### IX.5 Backend selection

A multi-backend binary contains all enabled backends. The CLI / FFI selects one at session creation time via `BackendKind`. There is no runtime cross-backend dispatch.

---

## Part X — Archive (`hologram-archive`)

`.holo` format. Wraps the compiler's output for distribution.

`std`. Depends on `hologram-graph`, `uor-foundation` (for `Trace` / `Certified` serialization).

### X.1 Format

```
+--------+----+----+----+----+----+----+----+
| MAGIC  | VER|FLG | section_offsets[8]   |
+--------+----+----+----+----+----+----+----+
| Section: kernel_calls[]                  |
| Section: schedule                        |
| Section: weights (BLAKE3-deduped)        |
| Section: shape_registry                  |
| Section: dtype_registry                  |
| Section: certificates (LiftChain)        |
| Section: trace                           |
| Section: metadata                        |
+------------------------------------------+
| FOOTER: full-archive BLAKE3 fingerprint  |
+------------------------------------------+
```

### X.2 Writer/loader

```rust
// crates/hologram-archive/src/writer.rs
pub struct HoloWriter { /* ... */ }
impl HoloWriter {
    pub fn new() -> Self;
    pub fn set_kernel_calls(&mut self, calls: Vec<KernelCall>);
    pub fn set_schedule(&mut self, sched: Schedule);
    pub fn set_weights(&mut self, weights: WeightStore);
    pub fn set_shape_registry(&mut self, registry: ShapeRegistry);
    pub fn set_certificates(&mut self, certs: Vec<Validated<LiftChainCertificate>>);
    pub fn set_trace(&mut self, trace: Trace<16384>);
    pub fn finish(self) -> Vec<u8>;
}

// crates/hologram-archive/src/loader.rs
pub struct HoloLoader<'a> { /* mmap-backed if std */ }
impl<'a> HoloLoader<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ArchiveError>;
    pub fn into_plan(self) -> LoadedPlan<'a>;
}

pub struct LoadedPlan<'a> {
    kernel_calls: &'a [KernelCall],
    schedule: &'a Schedule,
    weights: &'a WeightStore,
    certificates: &'a [Validated<LiftChainCertificate>],
}
```

### X.3 Weight dedup

BLAKE3 fingerprint per weight; identical weights share storage. The weight store is a `HashMap<ContentFingerprint<32>, &[u8]>`.

---

## Part XI — Reconciliation Matrix

This is the master cut/move list. Anything not on this list is **delete**. The reconciliation procedure is:

1. Apply this matrix to every file in `crates/`.
2. Create the `hologram-host` and `hologram-types` crates per Parts III and IV.
3. Rewrite `hologram-ops` per Part V.
4. Trim `hologram-graph` per Part VI.
5. Rewrite `hologram-compiler` per Part VII.
6. Trim `hologram-exec` per Part VIII.
7. Trim `hologram-backend` per Part IX.
8. Trim `hologram-archive` per Part X.
9. Run `cargo check --workspace` until clean.
10. Run `cargo test --workspace` until clean.
11. Run the `decode_step` benchmark and confirm numerical parity.

### XI.1 Per-crate dispositions

| Current crate | Disposition | Notes |
|---|---|---|
| `hologram-ring` | **DELETE entirely** | Replaced by upstream parametric ring + `hologram-host` axes. |
| `hologram-core` | **DELETE most; redistribute remainder** | `lut`, `view`, `encoding`, `op::lut_op`, `op::float_op` move to `hologram-ops`. Everything else deletes. |
| `hologram-shape` | **DELETE; fold into `hologram-types`** | Shape inference becomes `ConstrainedTypeShape`. |
| `hologram-cascade` | **DELETE entirely** | 7-stage pipeline = `pipeline::run_reduction_stages`. Cache logic moves to `hologram-compiler::CertificateCache` (~200 lines). |
| `hologram-transform` | **DELETE; fold into `hologram-compiler`** | chain → plan → executor reframed as Prism pipeline → CompiledPlan → Backend. |
| `hologram-async` | **DELETE; fold into `hologram-exec` behind `tokio` feature** | ~80 lines. |
| `hologram-compression` | **DELETE; fold into `hologram-archive` behind `compression` feature** | ~100 lines. |
| `hologram-graph` | **TRIM** | Drop `GraphOp::Float(FloatOp)`, `GraphOp::Lut(LutOp)`, `GraphOp::Prim(PrimOp)`, `GraphOp::CustomOp(...)`. Single `GraphOp::Op(OpKind)` per Part VI. |
| `hologram-ops` | **REWRITE** | Was an enum-based op catalog. Becomes 64 `Grounding` impls per Part V. |
| `hologram-compiler` | **REWRITE** | Drop preflight, term_parser, term_lower. New `Compiler` per Part VII. |
| `hologram-exec` | **TRIM** | Drop `tape`, `kv_*`, `lut_gemm`, `tape_builder`, `kernel_dispatch`, `float_dispatch`, `eval`, `runner`, `shape_resolve`, `patch_prune`. Keep `BufferArena`, `InferenceSession`, `mmap`. Add `Executor` per Part VIII. |
| `hologram-backend` | **TRIM** | Keep CPU, Metal, wgpu. Drop CUDA stub. New `Backend` trait per Part IX. |
| `hologram-archive` | **TRIM** | Drop `compression` (folded back behind feature). Add `certificates`, `trace` sections per Part X. |
| `hologram-cli` | **TRIM** | Subcommands unchanged. Update imports. |
| `hologram-ffi` | **TRIM** | C ABI + WASM unchanged. Update imports. |
| `hologram-bench` | **KEEP** | All 23 benches survive. Rewire imports per new crate names. |

### XI.2 Symbol-level deletions (workspace-wide)

The following identifiers must not appear anywhere in `crates/` after reconciliation:

```
PrismPrimitives          HoloPrimitives
QuantumLevel             RingLevel
Q0  Q1  Q3  Q7  Q15      RingWord
Datum<Q>                 Address<Q>     // *hologram-side* Datum/Address — upstream's are imported as-is
ByteRing                 PrimOp         // hologram-side PrimOp; replaced by upstream PrimitiveOp
ActivationOp             FloatOp
TapeKernel               KvStore        // KvStore in the v0.4 sense; weight cache moves to archive
HoloCompileUnit          TermArena      // hologram-side; upstream has its own
CertificateStore         CascadeState   // entire cascade machinery
TransformChain           CompiledPlan   // hologram-transform machinery
DispatchDeclarationBuilder    // moved into compiler internals
EffectDeclarationBuilder      // moved into compiler internals
```

Use of any of these identifiers in post-reconciliation code is a regression.

### XI.3 Symbol mapping table

| Current | Reconciled |
|---|---|
| `hologram_ring::PrimOp::*` | `uor_foundation::PrimitiveOp::*` |
| `hologram_ring::QuantumLevel` | `uor_foundation::WittLevel` |
| `hologram_ring::Q0` | `uor_foundation::WittLevel::W8` |
| `hologram_ring::Q1` | `uor_foundation::WittLevel::W16` |
| `hologram_ring::Q3` | `uor_foundation::WittLevel::W32` |
| `hologram_ring::Q7` | `uor_foundation::WittLevel::new(64)` |
| `hologram_ring::Q15` | `uor_foundation::WittLevel::new(128)` |
| `hologram_ring::PrismPrimitives` | `hologram_host::HologramHostTypes` |
| `hologram_core::HoloPrimitives` | `hologram_host::HologramHostTypes` |
| `hologram_ring::Datum<Q>` | (deleted) — at the type plane: the Tensor declaration; at the data plane: bytes in `BufferArena`; at scalar boundary: `enforcement::GroundedCoord::w8/w16/...` |
| `hologram_ring::RingWord` | (deleted) — Term-tree `Term::Literal { value, level }` for compile-time scalars; `BufferArena` slots for runtime bulk data. `Limbs<N>` is sealed and used only by upstream's reduction internals. |
| `hologram_ring::observables::stratum` | hologram-ops internal helper (unchanged math) |
| `hologram_ring::observables::curvature` | same |
| `hologram_ring::observables::rank` | same |
| `hologram_ring::observables::domain` | same |
| `hologram_ring::accumulate::accumulate` | `combinators::accumulate` over upstream PrimitiveOp |
| `hologram_core::FloatDType::F32` | `hologram_types::DTypeF32` |
| `hologram_core::FloatDType::*` | `hologram_types::DType*` |
| `hologram_core::ElementWiseView` | hologram-ops internal LUT helper |
| `hologram_core::Encoding` | hologram-ops internal helper |
| `hologram_cascade::CertificateStore` | `hologram_compiler::CertificateCache` |
| `hologram_cascade::engine::run_cascade` | `uor_foundation::pipeline::run_reduction_stages` |
| `hologram_cascade::stage::*` | upstream reduction stages (no hologram code) |
| `hologram_transform::TransformChain` | hologram-graph::Graph (already exists) |
| `hologram_transform::CompiledPlan` | hologram-archive::LoadedPlan |
| `hologram_transform::Executor` | hologram-exec::Executor |
| `hologram_compiler::preflight::*` | upstream `pipeline::preflight_*` |
| `hologram_compiler::term_parser` | hologram-compiler::source::parse (thin wrapper over `enforcement::TermArena`) |
| `hologram_compiler::term_lower` | merged into `Compiler::compile` |

### XI.4 ADR retirement

The following ADRs are retained as historical record but their normative effect is superseded by this spec:

- ADR-043 (LUT-addressed transform chains) → V.4 LUT-fused operands
- ADR-044 (Op trait canonical semantics) → V.1 pattern, now `Grounding` impls
- ADR-045 (ops as single source of truth) → V.2 catalog
- ADR-046 (canonical to legacy bridge) → no legacy after reconciliation; ADR voided
- ADR-047, 048 (FloatOp deprecation, permanent surface) → FloatOp deleted; ADRs voided
- ADR-049 (canonical attention) → V.2 `AttentionOp`
- ADR-050 (canonical as semantic contract) → invariant I-1
- ADR-051 (workspace residency) → IX.4 wgpu resident workspace retained
- ADR-052 (uor-foundation 0.3.0 domain decisions) → III.3 `HologramHasher`, IV.1 IRI scheme
- ADR-053 (mandatory shape metadata) → IV.4 `Tensor` carries shape via `ConstrainedTypeShape`

A new ADR-054 records this reconciliation:

```
# ADR-054: Hologram as a Prism Application (v0.3.1)

Status: Accepted 2026-05-06
Supersedes: ADR-046, ADR-047, ADR-048

Hologram is reconciled to be a consumer of `uor-foundation 0.3.1`'s parametric
prism. All hologram types are `ConstrainedTypeShape` declarations; all
operations are `Grounding` impls composing the closed `PrimitiveOp` set.
See `specs/docs/prism-v0.3.1-implementation.md` for the authoritative
implementation specification.
```

---

## Part XII — Verification

The reconciliation is complete when all of the following hold.

### XII.1 Compile

```
$ cargo check --workspace
    Finished `dev` profile [unoptimized + debuginfo] target(s) in N.NNs
```

Zero errors, zero warnings.

### XII.2 Lint

```
$ cargo clippy --workspace -- -D warnings
    Finished
```

### XII.3 Test

```
$ cargo test --workspace
    test result: ok. N passed; 0 failed
```

All retained tests must pass. New tests:

- `hologram-host`: every `HostBounds` impl satisfies `WITT_LEVEL_MAX_BITS = expected_register_width(target)`.
- `hologram-host::hasher`: `HologramHasher` implements `Hasher<32>` and produces BLAKE3-equivalent output for known vectors.
- `hologram-types`: every dtype, shape, tensor declaration produces a parseable IRI; `pipeline::preflight_feasibility` accepts every catalog instance.
- `hologram-ops`: every op marker's `emit_term` produces a Term tree whose `Term::Application` operator is restricted to the closed 10 `PrimitiveOp` discriminants (statically asserted by the type signature; `PrimitiveOp` is the only operator type `Term::Application::operator` accepts). For each op, a reference-evaluator equivalence test asserts that the backend kernel output matches the Term-tree reference evaluation within dtype tolerance over a representative input set.
- `hologram-compiler`: a representative `Graph` containing one of every `OpKind` compiles to a `.holo` archive; each per-node certificate validates via `pipeline::run_tower_completeness`.
- `hologram-exec`: round-trip — compile a Graph, execute via CPU backend, compare output to a reference implementation.

### XII.4 Bench parity

```
$ cargo bench --bench decode_step
```

Runs the canonical transformer-block decode-step benchmark. Required:

- **Numerical parity**: output matches the pre-reconciliation result within `1e-6` relative tolerance per element.
- **Throughput parity**: ≥ 95% of pre-reconciliation throughput on the same hardware. (Goal: zero-cost — no slowdown from the parametric machinery.)
- **Code-size parity**: stripped binary size ≤ 110% of pre-reconciliation size. (Goal: code shrinks because of dead code elimination after the cascade and ring-mirror deletions.)

### XII.5 Disassembly check

For invariant I-7 ("zero-cost"). On x86-64 with AVX2, the inner loop of:

```rust
fn matmul_w256(a: &Tensor<Shape2<Dim<128>, Dim<128>>, DTypeF32, HologramHostBoundsAvx2>,
               b: &Tensor<Shape2<Dim<128>, Dim<128>>, DTypeF32, HologramHostBoundsAvx2>)
   -> Tensor<Shape2<Dim<128>, Dim<128>>, DTypeF32, HologramHostBoundsAvx2>
```

must compile to a sequence of `vfmadd231ps` (or equivalent) instructions with no branches into hologram-internal dispatch tables. Verified by `cargo asm` or `cargo objdump --disassemble`.

### XII.6 Symbol absence

```
$ rg --type rust 'PrismPrimitives|QuantumLevel|RingWord|HoloPrimitives|TapeKernel|CascadeState|TransformChain' crates/
```

Returns no results.

```
$ rg --type rust 'hologram_ring::|hologram_core::|hologram_cascade::|hologram_transform::|hologram_async::|hologram_compression::|hologram_shape::' crates/
```

Returns no results (these crates no longer exist).

---

## Part XIII — Tensor data flow

This section closes the gap between hologram's typed surface (sealed Datums, ConstrainedTypeShape declarations, Term trees) and the bulk numeric data (tensors, weights, activations).

### XIII.1 Three planes

Hologram operates on three orthogonal planes:

1. **Type plane.** `ConstrainedTypeShape` declarations identify each tensor / weight / region by IRI + site count + constraints. This is what upstream's pipeline validates. Upstream sees only IRIs and site counts — never bytes of tensor data.
2. **Term plane.** `TermArena<CAP>` holds the formal specification of each operation as a tree of `Term::Application` over `PrimitiveOp` discriminants, parameterized by `Term::Variable` references into a `BindingsTable`. `Term::Variable` references logical sites; the bindings table maps site addresses to compile-time-resolved storage locations. Upstream's pipeline reasons over this tree (SAT classification, reduction, certification).
3. **Data plane.** Tensor / weight / activation bytes live in `BufferArena` slots managed by `hologram-exec`. Backend kernels (Part IX) read and write these slots natively. Upstream never touches the data plane.

The three planes are linked by **content addresses**:

- A `BindingEntry` in the bindings table carries a `ContentAddress` (32-byte BLAKE3 fingerprint).
- The compile-time `BindingsTable` (a `&'static [BindingEntry]`) maps content addresses to compile-time-known byte slices (constants, weight refs).
- The runtime workspace mapping (`hologram-exec::AddressTable`) maps the same content addresses to `BufferArena` slot offsets.
- `Term::Variable { name_index }` indexes into the name table that produced the binding — at codegen time this resolves to a compile-time SlotSpan.

### XIII.2 GroundedCoord / GroundedTuple constraints

Upstream's `GroundedCoord` constructors are `w8`, `w16`, `w24`, `w32`, `w40`, … (one per Witt level). `GroundedTuple<const N: usize>` is a fixed-size array of `GroundedCoord`. Both are **scalar / fixed-size** carriers, not tensor data carriers.

Hologram uses them only for:
- Per-element scalar grounding during input parsing (boundary `Grounding` impls in VII.7).
- Per-element scalar literals in Term trees (`Term::Literal { value: u64, level }`).

Tensor data flow does NOT pass through `GroundedTuple<N>`. A tensor with a million elements is a million `BindingEntry` references in the bindings table (or a single weight-region entry whose body is read by the backend kernel as a contiguous byte slice — the typical case).

### XIII.3 Witt-level discipline at the data plane

Each backend operates at its native Witt level (per HostBounds). When the data plane crosses Witt levels (e.g., a W256 AVX2 backend ingests W64 input bytes from disk):

- The Term tree expresses the level change via `Term::Lift` / `Term::Project`.
- The native kernel performs the level change as a vector-load / vector-narrow instruction.
- The Validated certificate confirms the Term tree's lift/project is well-typed.

There is no implicit promotion at the data plane. Cross-Witt-level operations are explicit at both the Term-tree and kernel levels.

---

## Part XIV — Architectural commitments and open questions

This section enumerates points where the spec makes a load-bearing choice that future implementation may need to revisit, and points where the spec deliberately leaves a gap because the right answer requires implementation experience.

### XIV.1 Committed (load-bearing; do not deviate without revising this spec)

- **C-1.** The Term tree is the formal spec; the native kernel is the execution form. Equivalence is verified by tests (not by codegen). [VII.3, IX.2]
- **C-2.** One `CompileUnit` per graph Node. Per-node `Validated<LiftChainCertificate>` are concatenated into the archive. [VII.2]
- **C-3.** Hologram's IRI namespace `https://hologram.uor.foundation/type/...` is a Prism extension. ADR-013 forbids new `PrimitiveOp` discriminants but does not forbid new types. Hologram's IRIs are explicitly application-introduced types, not new primitives. [IV.2]
- **C-4.** `hologram-types` uses only stable Rust const generics. Aggregate site counts are passed as trailing `const SITES: usize` parameters; nightly `generic_const_exprs` is not required. [IV.3]
- **C-5.** `Grounding` impls appear only at the input boundary. Operations are Term trees. [I-10, V.8, VII.7]
- **C-6.** Each backend has its own `HostBounds` impl. `WITT_LEVEL_MAX_BITS` is the host's natural register width. [III.2, I-6]
- **C-7.** `Hasher<32>` always — `FINGERPRINT_MIN_BYTES = MAX = 32` across all backends. BLAKE3. [III.3]
- **C-8.** Float dtypes are encoded as W{32,16,16,64} integers at the Term-tree level (bit-pattern arithmetic), executed natively by backend kernels (IEEE-754 ops). [V.7]
- **C-9.** The 64-op catalog (V.3 + V.4) is closed for v0.5.0. New ops require a new entry here, a new `OpKind` variant, a new emitter, and a new kernel per backend.

### XIV.2 Open / deferred (right answer comes from implementation)

- **O-1. Term tree CAP for very large operations.** The CAPs in V.5 are upper bounds. If an op's expression exceeds its CAP, `emit_term` returns `None` and compilation fails. The right ceilings will emerge from the per-op implementations; revisit V.5 if any op overflows.

- **O-2. Reference scalar evaluator interface.** The reference evaluator that walks Term trees (used by tests, VII.3) needs a defined trait surface. Sketch:
  ```rust
  pub trait TermEvaluator {
      type Value;
      fn evaluate(arena: &[Term], root: u32, bindings: &BindingsTable) -> Self::Value;
  }
  ```
  The exact signature depends on how variable bindings carry tensor data at test time. Implementation will pick the form.

- **O-3. Bindings table content for tensor weights.** A `BindingEntry` is `(ContentAddress, &'static [u8])` (paraphrasing). For a 1 GB tensor weight, the body is a 32-byte fingerprint that the runtime resolves to a workspace slot — not a 1 GB `&'static [u8]`. The exact bindings-vs-workspace boundary is implementation-decided.

- **O-4. CompileUnit for layout-only ops.** `ReshapeOp`, `TransposeOp`, `ConcatOp`, `SliceOp` produce Term trees with no `Application` nodes (V.3 — "address-remap layout op"). `pipeline::run_tower_completeness` may treat empty Term trees specially or reject them. If it rejects, layout ops emit a single trivial `Application(And, [x, all_ones])` no-op so the certificate validates. The implementation chooses based on what upstream actually accepts.

- **O-5. Backward-rule emission timing.** ADR-043 says backward is planned ahead of time. Whether backward Term trees are emitted at graph-build (per node) or at compile-time (per session) is an implementation choice with caching implications.

- **O-6. wgpu / Metal kernel format.** WGSL and Metal Shading Language source files vs. precompiled SPIR-V vs. runtime-compiled. Choice deferred to the backend implementations; archive format (Part X) accepts a generic byte blob per (op, backend) pair.

- **O-7. Streaming `Trace` ingestion.** `Trace<TR_MAX = 16384>` capacity is sized for UHD per-frame workloads. A streaming-mode where multiple traces concatenate is post-v0.5.0 (Part XV).

- **O-8. WittLevel for >64-bit accumulators on scalar CPU.** The CPU-scalar `HostBounds::WITT_LEVEL_MAX_BITS = 64` caps the largest single-instruction op. If a hot path needs W128 software-emulated arithmetic (multi-limb), it appears as `Term::Lift { target: W128 }` and the kernel falls back to `Limbs<2>` operations (slower than register ops but still correct). Whether to expose a `HologramHostBoundsCpuW128` variant is implementation-decided.

### XIV.3 Explicitly out of scope

- **X-1.** Codegen path that walks Term trees and emits native machine code (interpretation B). Out of scope for v0.5.0.
- **X-2.** Runtime cross-backend dispatch. A binary may link multiple backends; the active one is chosen at session creation, never within a session.
- **X-3.** Persistent (cross-process) certificate caching. CertificateCache is in-memory only.
- **X-4.** ONNX import. Separate `hologram-onnx` crate, post-v0.5.0.
- **X-5.** Quantization formats (4-bit / 8-bit codebook). Re-introduced post-v0.5.0 as `Weight<DTypeI4, B>` / `Weight<DTypeI8, B>` plus a `DequantizeOp` marker.
- **X-6.** CUDA backend. Post-v0.5.0.

---

## Part XV — Open items deferred

The following are intentionally out of scope of this reconciliation. They are tracked here for visibility but do not block v0.5.0:

1. **ONNX import**: separate `hologram-onnx` crate, post-v0.5.0.
2. **Quantization formats** (4-bit, 8-bit codebook KV cache from current `hologram-exec`): reintroduced post-v0.5.0 as a `Weight<DTypeI4, B>` / `Weight<DTypeI8, B>` specialization with a paired `DequantizeOp` Grounding impl. The KV-cache machinery from `hologram-exec/kv_cache.rs` is rewritten on this foundation.
3. **CUDA backend**: post-v0.5.0.
4. **Per-frame UHD streaming**: requires `Trace<TR_MAX>` with `TR_MAX = 16_384`; current spec sizes for it but a streaming-mode test harness is post-v0.5.0.
5. **Trillion-param model loading**: enabled by Witt-level scaling already specified; a representative model is not part of this cut. `hologram-ai` consumes the v0.5.0 surface.

---

## Part XVI — Authority

This document is the source of truth for hologram v0.5.0. When this document conflicts with code, the code is wrong. When this document conflicts with another spec or ADR, this document wins. When this document conflicts with `uor-foundation 0.3.1`, `uor-foundation` wins (and this document is updated).

End of specification.
