# Hologram Ring-Native Refactoring: Conformance/Test-First Plan

## Context

The hologram workspace (10 crates, ~48K LoC) implements O(1) compute acceleration using precomputed LUT tables at Q0-Q1 and algorithmic ops at Q3+. The architecture has reached its ceiling: LUTs don't scale past Q1 (128KB/table), the float escape hatch (`FloatOp`) breaks ring closure, and the tape-based executor prevents native code generation.

This refactoring makes the entire stack parametric over quantum level, replaces LUTs with ring-primitive compositions, and replaces tape dispatch with Cranelift JIT. The governing principle: **every operation is a ring primitive or a composition of ring primitives. No fallbacks. No escape hatches. No foreign-domain transitions.**

If an operation cannot be expressed as a composition of ring primitives at the chosen quantum level, the quantum level is wrong — not the operation.

All planned types and operations are **zero-cost**: zero-sized type (ZST) level markers, `const fn` ring operations, `#[inline]` on every hot path, monomorphization (no dynamic dispatch), `#![no_std]` in the kernel crate.

The new implementation is built under `prism-*` crate names during development. Once complete and all tests pass, `prism-*` crates are renamed back to `hologram-*`. The public API surface consumed by hologram-ai is preserved. The `hologram` name, archive format (`HOLO`), and binary name remain unchanged.

This plan follows strict test-first discipline: conformance tests are written FIRST to define the contract, then implementation makes the tests pass.

## hologram-ai Consumer Contract

hologram-ai is the primary consumer (ADR-0001). The refactored hologram must maintain these contract surfaces:

| Contract Surface | Status |
|---|---|
| `Graph`, `GraphBuilder`, `GraphOp`, `NodeId` | Preserved (GraphOp variants change — see Phase 8B) |
| `PrimOp` (10 ring primitives) | Preserved (identical) |
| `ActivationOp` (replaces `LutOp`) | Renamed — hologram-ai adopts new enum |
| `CustomOpId`, `CustomOpRegistry`, `CustomHandler` | Preserved (identical) |
| `ConstantData`, `ConstantId`, `ConstantStore` | Preserved (identical) |
| `compile(&graph) -> CompilationOutput` | Preserved (same API, new internals) |
| `HoloWriter`, `HoloLoader`, `LoadedPlan` | Preserved (archive version bumped to v2) |
| Execution: `build_tape` + `execute_tape` | Replaced by `JitModule::compile` + `execute` |
| `KvCacheState` for autoregressive gen | Preserved (JIT-backed) |
| `FloatOp` (70+ variants) | Eliminated — ring-native replacements |
| `LutOp` (21 activations) | Renamed to `ActivationOp` |
| `MatMulLut4/8/16` | Replaced by Accumulate+Reduce subgraph |
| `ElementWiseView` | Eliminated (internal to fusion) |

---

## Phase 0: Scaffold prism crates alongside hologram

**Goal:** New crate skeletons exist, workspace compiles, zero hologram changes.

### Tests (written first)
```
prism-ring/tests/scaffold_test.rs
  - crate compiles with #![no_std]
  - placeholder types exist: RingWord, QuantumLevel, PrimOp, Involution, Encoding, Datum, Address
  - PrismPrimitives implements uor_foundation::Primitives
```

### Implementation
1. Create `crates/prism-ring/Cargo.toml` (`#![no_std]`, depends on `uor-foundation = "0.1.1"`)
2. Create empty `crates/prism-graph/`, `prism-compiler/`, `prism-archive/`, `prism-jit/`, `prism-compression/`, `prism-ffi/`, `prism-cli/`, `prism-bench/`
3. Add all to workspace `members` in root `Cargo.toml`
4. Add `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native` to `[workspace.dependencies]`
5. Verify `cargo test --workspace` passes (all hologram tests green)

### Files to create
- `crates/prism-ring/Cargo.toml` + `src/lib.rs`
- `crates/prism-graph/Cargo.toml` + `src/lib.rs`
- `crates/prism-compiler/Cargo.toml` + `src/lib.rs`
- `crates/prism-archive/Cargo.toml` + `src/lib.rs`
- `crates/prism-jit/Cargo.toml` + `src/lib.rs`
- `crates/prism-compression/Cargo.toml` + `src/lib.rs`
- `crates/prism-ffi/Cargo.toml` + `src/lib.rs`
- `crates/prism-cli/Cargo.toml` + `src/lib.rs`

---

## Phase 1: prism-ring — The Parametric Ring

The foundation. Everything builds on this. All types in this crate are zero-cost: ZSTs for levels, `const fn` for all ring arithmetic, `#[inline]` on every method, monomorphized generics.

### Phase 1A: RingWord trait + impls

**Tests** (`prism-ring/tests/ring_word_conformance.rs`):

For each W in {u8, u16, u32, u64, u128} — exhaustive for u8, sampled for larger:
1. **Closure**: `wrapping_add(a, b)` produces W
2. **Associativity**: `(a+b)+c == a+(b+c)` for add and mul
3. **Commutativity**: `a+b == b+a`, `a*b == b*a`
4. **Identity**: `a + ZERO == a`, `a * ONE == a`
5. **Additive inverse**: `a + wrapping_neg(a) == ZERO`
6. **Distributivity**: `a*(b+c) == a*b + a*c`
7. **Constants**: `ZERO == 0`, `ONE == 1`, `MAX == 2^BITS - 1`
8. **Bit intrinsics**: `count_ones`, `leading_zeros`, `trailing_zeros` match `core` intrinsics

Sampling pattern from existing [ring_conformance.rs](crates/hologram-core/tests/ring_conformance.rs): step_by primes (17, 19, 23) for u8 triples, spot-check vectors `[0, 1, 127, 255, 0xFFFF, 0x00FF_FFFF, u32::MAX/2, u32::MAX]` for u32/u64.

**Implementation** (`prism-ring/src/word.rs`):
```rust
pub trait RingWord:
    Copy + Eq + Ord +
    core::ops::Add<Output = Self> + core::ops::Sub<Output = Self> +
    core::ops::Mul<Output = Self> + core::ops::BitXor<Output = Self> +
    core::ops::BitAnd<Output = Self> + core::ops::BitOr<Output = Self> +
    core::ops::Not<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;
    const BITS: u32;
    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn count_ones(self) -> u32;
    fn leading_zeros(self) -> u32;
    fn trailing_zeros(self) -> u32;
    fn from_u64(v: u64) -> Self;
    fn to_u64(self) -> u64;
}
```
Implement for u8, u16, u32, u64, u128. Each method is a one-liner delegating to Rust intrinsics. All methods are `#[inline]` and compile to single ALU instructions.

### Phase 1B: QuantumLevel trait + PrismPrimitives

**Tests** (`prism-ring/tests/quantum_level_conformance.rs`):

For each level {Q0, Q1, Q3, Q7, Q15}:
1. `Q::BITS == 8 * (Q::INDEX + 1)`
2. `<Q::Word as RingWord>::BITS == Q::BITS`
3. Level → Word type mapping: Q0→u8, Q1→u16, Q3→u32, Q7→u64, Q15→u128
4. Each level type is a ZST: `core::mem::size_of::<Q0>() == 0`
5. `PrismPrimitives` implements `uor_foundation::Primitives` with same mapping as `HoloPrimitives`

**Implementation** (`prism-ring/src/level.rs`):
```rust
/// Marker trait for quantum levels. Zero-sized — monomorphized away.
pub trait QuantumLevel: Copy + 'static {
    const BITS: u32;         // 8*(k+1)
    const INDEX: u32;        // k
    type Word: RingWord;
}

#[derive(Debug, Clone, Copy)] pub struct Q0;
#[derive(Debug, Clone, Copy)] pub struct Q1;
#[derive(Debug, Clone, Copy)] pub struct Q3;
#[derive(Debug, Clone, Copy)] pub struct Q7;
#[derive(Debug, Clone, Copy)] pub struct Q15;
```

| Level | INDEX | BITS | Word | Ring | Cranelift | Hardware |
|-------|-------|------|------|------|-----------|----------|
| Q0 | 0 | 8 | u8 | Z/256Z | I8 | byte ops |
| Q1 | 1 | 16 | u16 | Z/65536Z | I16 | half-word |
| Q3 | 3 | 32 | u32 | Z/2^32Z | I32 | 32-bit ALU |
| Q7 | 7 | 64 | u64 | Z/2^64Z | I64 | 64-bit ALU |
| Q15 | 15 | 128 | u128 | Z/2^128Z | I128 | SIMD pair |

Note: INDEX follows the formula strictly. Q2 (24-bit) from hologram is dropped — there is no hardware-native 24-bit type. The new Q3 (32-bit) absorbs hologram's Q2 and Q3 functionality.

```rust
/// The UOR primitive type family for Prism.
pub struct PrismPrimitives;
impl uor_foundation::Primitives for PrismPrimitives {
    type String = str;
    type Integer = i64;
    type NonNegativeInteger = u64;
    type PositiveInteger = u64;
    type Decimal = f64;
    type Boolean = bool;
}
```

### Phase 1C: PrimOp — parametric primitives

**Tests** (`prism-ring/tests/primop_conformance.rs`):

For each of the 10 PrimOps x each QuantumLevel {Q0, Q1, Q3, Q7}:
1. **Known-answer**: `apply_unary`/`apply_binary` matches hand-computed values
2. **Cross-level embedding**: Zero-extend Q0 value into Q1, apply op, truncate back. For operations where no overflow occurs (bitwise ops), result matches Q0 result exactly.
3. **Arity**: 1 for unary (Neg, Bnot, Succ, Pred), 2 for binary (Add, Sub, Mul, Xor, And, Or)
4. **Commutativity**: correct for each binary op (Add, Mul, Xor, And, Or: true; Sub: false)
5. **Associativity**: correct for each binary op (Add, Mul, Xor, And, Or: true; Sub: false)
6. **Identity element**: Add→0, Mul→1, Xor→0, And→MAX, Or→0
7. **Critical identity**: `neg(bnot(x)) == succ(x)` — exhaustive at Q0, sampled at Q1/Q3/Q7
8. **UOR trait conformance**: `PrimOp` implements `Operation<PrismPrimitives>` with correct arity, geometric character, and `composed_of` name. Implements `BinaryOp<PrismPrimitives>` with correct `commutative()`, `associative()`, `identity()`.

Test vectors transplanted from existing [ring_conformance.rs](crates/hologram-core/tests/ring_conformance.rs) and [q3_conformance.rs](crates/hologram-core/tests/q3_conformance.rs).

**Implementation** (`prism-ring/src/prim.rs`):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimOp { Neg, Bnot, Succ, Pred, Add, Sub, Mul, Xor, And, Or }

impl PrimOp {
    #[inline] pub const fn arity(&self) -> u8 { ... }
    #[inline] pub fn apply_unary<W: RingWord>(&self, x: W) -> W { ... }
    #[inline] pub fn apply_binary<W: RingWord>(&self, a: W, b: W) -> W { ... }
}
```
Generic over `RingWord`. No LUT tables. All ops are direct wrapping arithmetic — the same code path at every quantum level. Implements `uor_foundation::kernel::op::Operation<PrismPrimitives>` and `BinaryOp<PrismPrimitives>`.

### Phase 1D: Involution\<Q\> and DihedralGroup

**Tests** (`prism-ring/tests/involution_conformance.rs`):

For each QuantumLevel {Q0, Q1, Q3, Q7}:
1. `Neg` is involutory: `neg(neg(x)) == x` for all sampled x
2. `Bnot` is involutory: `bnot(bnot(x)) == x` for all sampled x
3. **Critical identity**: `neg(bnot(x)) == succ(x)` (the UOR fundamental identity)
4. **Geometric character**: Neg → `RingReflection`, Bnot → `HypercubeReflection`
5. **UOR trait conformance**: `Involution<Q>` implements `uor_foundation::kernel::op::Involution<PrismPrimitives>`, `UnaryOp`, `Operation`
6. **DihedralGroup**: Ring type at each level implements `DihedralGroup<PrismPrimitives>` with `generated_by()` returning [Neg, Bnot], `order()` returning 2^BITS

**Implementation** (`prism-ring/src/involution.rs`):
```rust
/// The two generators of the dihedral group D_{2^n}.
/// Zero-sized — the apply method is const fn, compiles to a single ALU instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Involution<Q: QuantumLevel> {
    Neg,
    Bnot,
    _Phantom(core::marker::PhantomData<Q>),
}

impl<Q: QuantumLevel> Involution<Q> {
    #[inline]
    pub fn apply(self, x: Q::Word) -> Q::Word {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,  // via Not trait on RingWord
            _ => unreachable!(),
        }
    }
}
```
Implements `uor_foundation::kernel::op::{Operation, UnaryOp, Involution}` for `Involution<Q>`.

### Phase 1E: Datum\<Q\> and Address\<Q\>

**Tests** (`prism-ring/tests/datum_conformance.rs`):

For each QuantumLevel {Q0, Q1, Q3, Q7}:
1. `Datum::new(v).value() == v as u64`
2. `Datum::new(v).quantum() == Q::BITS as u64`
3. `Datum::new(v).stratum() == v.count_ones() as u64`
4. `Datum::new(v).spectrum()` is a binary string of length BITS
5. `Datum::PI1.value() == 1` (multiplicative generator)
6. **Address round-trip**: `Address::from_word(v).glyph()` produces valid UTF-8 Braille
7. **UOR trait conformance**: `Datum<Q>` implements `uor_foundation::kernel::schema::Datum<PrismPrimitives>`, `Address<Q>` implements `uor_foundation::kernel::address::Address<PrismPrimitives>`

**Implementation** (`prism-ring/src/datum.rs`, `prism-ring/src/address.rs`):
```rust
/// An element of the ring R_n at quantum level Q.
/// const-constructible, zero allocation.
pub struct Datum<Q: QuantumLevel> {
    value: Q::Word,
    spectrum_buf: [u8; MAX_SPECTRUM_LEN],  // binary string, const-computed
    address: Address<Q>,
}

/// Braille-encoded address for a ring element.
pub struct Address<Q: QuantumLevel> {
    value: Q::Word,
    glyph_buf: [u8; MAX_GLYPH_LEN],  // UTF-8 Braille, const-computed
}
```
All construction is `const fn`. Spectrum and glyph buffers are computed at construction time with zero allocation. Implements UOR `Datum<PrismPrimitives>` and `Address<PrismPrimitives>`.

### Phase 1F: Ring\<Q\> — UOR Ring + NormedDivisionAlgebra + CayleyDicksonConstruction

**Tests** (`prism-ring/tests/ring_uor_conformance.rs`):

For each QuantumLevel {Q0, Q1, Q3, Q7}:
1. `ring.ring_quantum() == Q::BITS as u64`
2. `ring.modulus() == 2^Q::BITS` (as u64, or 0 for overflow at Q7)
3. `ring.at_quantum_level()` matches `uor_foundation::enums::QuantumLevel` mapping
4. `ring.generator().value() == 1` (PI1)
5. `ring.negation()` is `Involution::Neg`, `ring.complement()` is `Involution::Bnot`

For the Cayley-Dickson chain:
6. Q0: dimension 1 (real), source dim 1, target dim 2
7. Q1: dimension 2 (complex), source dim 2, target dim 4
8. Q3: dimension 4 (quaternion) — non-commutative from here
9. Q7: dimension 8 (octonion) — non-associative, terminal in CD chain
10. Each level: `target_dim == 2 * source_dim`
11. Adjoined elements: "i", "j", "k", "e4..e7"

For algebra properties:
12. Q0, Q1: commutative and associative
13. Q3: non-commutative, associative
14. Q7: non-commutative, non-associative
15. Associator at Q7: non-zero for imaginary embeddings, zero for real subalgebra

**Implementation** (`prism-ring/src/ring.rs`):
```rust
/// Zero-sized marker type for the ring R_n at quantum level Q.
#[derive(Debug, Clone, Copy)]
pub struct PrismRing<Q: QuantumLevel>(PhantomData<Q>);

// Ring<PrismPrimitives>
impl<Q: QuantumLevel> uor_foundation::kernel::schema::Ring<PrismPrimitives> for PrismRing<Q> {
    fn ring_quantum(&self) -> u64 { Q::BITS as u64 }
    fn modulus(&self) -> u64 { ... }
    type Datum = Datum<Q>;
    type Involution = Involution<Q>;
    fn generator(&self) -> &Self::Datum { ... }
    fn negation(&self) -> &Self::Involution { ... }
    fn complement(&self) -> &Self::Involution { ... }
    fn at_quantum_level(&self) -> QuantumLevel { ... }
}

// Group<PrismPrimitives> + DihedralGroup<PrismPrimitives>
impl<Q: QuantumLevel> uor_foundation::kernel::op::Group<PrismPrimitives> for PrismRing<Q> {
    type Operation = Involution<Q>;
    fn generated_by(&self) -> &[Self::Operation] { &[Involution::Neg, Involution::Bnot] }
    fn order(&self) -> u64 { 1u64 << Q::BITS }
}
impl<Q: QuantumLevel> uor_foundation::kernel::op::DihedralGroup<PrismPrimitives> for PrismRing<Q> {}

// NormedDivisionAlgebra<PrismPrimitives>
impl<Q: QuantumLevel> NormedDivisionAlgebra<PrismPrimitives> for PrismRing<Q> {
    fn algebra_dimension(&self) -> u64 { 1 << Q::INDEX.min(3) } // 1, 2, 4, 8
    fn is_commutative(&self) -> bool { Q::INDEX < 3 }
    fn is_associative(&self) -> bool { Q::INDEX < 7 }
    fn basis_elements(&self) -> &str { ... }
    type MultiplicationTable = PrismMultTable<Q>;
    ...
}

// CayleyDicksonConstruction<PrismPrimitives> — chain terminates at Q7 (octonions)
impl<Q: QuantumLevel> CayleyDicksonConstruction<PrismPrimitives> for PrismRing<Q> where Q: HasNextLevel {
    type NormedDivisionAlgebra = PrismDivisionAlgebra;
    fn cayley_dickson_source(&self) -> &Self::NormedDivisionAlgebra { ... }
    fn cayley_dickson_target(&self) -> &Self::NormedDivisionAlgebra { ... }
    fn adjoined_element(&self) -> &str { ... }
    fn conjugation_rule(&self) -> &str { ... }
}
```
All static allocations. All singletons. ZSTs throughout. The `PrismDivisionAlgebra` enum unifies across levels (same pattern as existing `HoloDivisionAlgebra`).

### Phase 1G: Observables — generic

**Tests** (`prism-ring/tests/observables_conformance.rs`):

For each QuantumLevel {Q0, Q1, Q3, Q7}:
1. `stratum(x) == x.count_ones()`
2. `curvature(x) == (x ^ x.wrapping_add(ONE)).count_ones()`
3. `rank(x) == x.trailing_zeros()`
4. `domain(x) == x.leading_zeros()`
5. **Boundary**: `stratum(ZERO) == 0`, `stratum(MAX) == BITS`
6. **Stratum range**: `0 <= stratum(x) <= BITS` for all x
7. **Curvature range**: `1 <= curvature(x)` for all x (always at least 1 bit flips)

**Implementation** (`prism-ring/src/observables.rs`):
```rust
#[inline] pub fn stratum<W: RingWord>(x: W) -> u32 { x.count_ones() }
#[inline] pub fn curvature<W: RingWord>(x: W) -> u32 { (x ^ x.wrapping_add(W::ONE)).count_ones() }
#[inline] pub fn rank<W: RingWord>(x: W) -> u32 { x.trailing_zeros() }
#[inline] pub fn domain<W: RingWord>(x: W) -> u32 { x.leading_zeros() }
```
Each function compiles to 1-3 ALU instructions at any quantum level.

### Phase 1H: Encoding — parametric embed/lift

**Tests** (`prism-ring/tests/encoding_conformance.rs`):

For each encoding {Angle, Signed, Unsigned, Raw} x each QuantumLevel:
1. **Round-trip fidelity**: `lift(embed(v))` recovers v to within the quantization step of the level (1/2^BITS)
2. **Monotonicity**: `v1 < v2 => embed(v1) <= embed(v2)` (for Unsigned, Signed)
3. **Range coverage**: embed maps the continuous domain onto the full word space [0, MAX]
4. **Boundary exactness**: `embed(0.0) == ZERO` (Unsigned), `embed(-1.0) == 0 ∧ embed(1.0) == MAX` (Signed), `embed(0.0) == 0 ∧ embed(2π) wraps to 0` (Angle)
5. **UOR trait**: implements `Encoding` concept (embed is the pi map, lift is the lambda map)

**Implementation** (`prism-ring/src/encoding.rs`):
```rust
pub trait Encoding<W: RingWord> {
    fn embed(&self, value: f64) -> W;  // π: continuous → ring
    fn lift(&self, word: W) -> f64;    // λ: ring → continuous
    fn name(&self) -> &'static str;
}
```
Concrete: `AngleEncoding<Q>`, `SignedEncoding<Q>`, `UnsignedEncoding<Q>`, `RawEncoding<Q>`.
The encoding is the pi-F-lambda bridge between the continuous domain and the ring. Once in the ring, all computation is ring-native. The encoding boundary exists only at graph input/output.

### Phase 1I: Activations as ComposedOperations

**Tests** (`prism-ring/tests/activation_conformance.rs`):

This is the critical contract. Every activation is a `ComposedOperation` — a chain of ring primitives. There is no LUT. There is no f64 escape hatch. The ring arithmetic IS the computation.

For each of 21 activations x each QuantumLevel {Q0, Q1, Q3, Q7}:

1. **Decomposition witness**: `decompose()` returns a `Vec<PrimOp>` chain. Applying the chain step-by-step matches `apply()` exactly (bit-identical).
2. **Ring closure**: `apply()` takes a `W` and returns a `W`. No intermediate f64.
3. **Structural properties**:
   - relu: monotonic, `relu(ZERO) == ZERO`, `relu(MAX) depends on sign convention`
   - sigmoid: monotonic, bounded
   - abs: `abs(abs(x)) == abs(x)` (idempotent)
   - square: `square(neg(x)) == square(x)` (even function)
4. **Consistency oracle** (test-only, not runtime): At 256 sample points, encode an f64 value, apply activation in-ring, decode back to f64 — compare against f64 reference. This verifies the polynomial coefficients were correctly computed. The oracle tolerance is the quantization step of the level (1/256 for Q0, 1/65536 for Q1, etc.). The oracle is a TEST, not a fallback.
5. **Piecewise polynomial structure**: For transcendentals (sigmoid, tanh, gelu, silu):
   - Segment boundaries are ring constants (precomputed, stored as `W`)
   - Polynomial coefficients are ring constants
   - Evaluation is `mul` + `add` in the ring (Horner's method)
   - Segment selection is `sub` + sign-bit extraction (comparison via ring ops)
   - All intermediate values are `W` — never leaves the ring

f64 reference implementations for the oracle (transplant from [float_conformance.rs](crates/hologram-exec/tests/float_conformance.rs)):
```rust
// TEST-ONLY oracles — these verify polynomial coefficient correctness
fn sigmoid_ref(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn gelu_ref(x: f64) -> f64 { 0.5 * x * (1.0 + erf(x / SQRT_2)) }
fn silu_ref(x: f64) -> f64 { x * sigmoid_ref(x) }
```

**Implementation** (`prism-ring/src/activation.rs`):
```rust
pub enum ActivationOp { Relu, Abs, Square, Cube, Sigmoid, Tanh, Gelu, Silu, ... /* all 21 */ }

impl ActivationOp {
    /// The composedOf witness: a chain of PrimOps that implements this activation.
    pub fn decompose<Q: QuantumLevel>(&self) -> Vec<PrimOp> { ... }

    /// Apply directly using inlined ring arithmetic. Zero allocation. Zero f64.
    #[inline]
    pub fn apply<Q: QuantumLevel>(&self, x: Q::Word) -> Q::Word { ... }
}
```

Simple activations:
- `relu`: sign-bit mask via `ushr` + `and` (3 ring ops)
- `abs`: conditional neg via sign-bit (3 ring ops)
- `square`: `mul(x, x)` (1 ring op)
- `cube`: `mul(mul(x, x), x)` (2 ring ops)

Transcendentals — ring-native piecewise polynomial:
- Fixed-point representation within the ring word (encoding determines bit split: integer.fractional)
- Polynomial coefficients are ring constants, precomputed per quantum level
- Horner's method: `c0 + x*(c1 + x*(c2 + x*c3))` — each step is `mul` + `add`
- Segment boundaries are ring constants. Comparison: `sub(x, boundary)`, extract sign bit
- Number of segments × polynomial degree chosen so the composed operation is exact to within the quantization step at that level. At Q3 (32-bit), a 4-segment cubic gives ~10^-7 relative error. At Q7 (64-bit), 8-segment cubic gives ~10^-15.
- At Q0 (8 bits), the polynomial evaluation collapses to a small number of ring ops that happen to produce the correct 256 output values. This is algebraically verified, not approximated.

### Phase 1J: Accumulation pattern

**Tests** (`prism-ring/tests/accumulate_conformance.rs`):

For each QuantumLevel:
1. `accumulate(acc, a, b) == wrapping_add(acc, wrapping_mul(a, b))`
2. Iterated: `Σ a[i] * b[i]` via fold over accumulate matches loop of add+mul
3. Matmul 2x2: `C[i,j] = Σ_k A[i,k] * B[k,j]` via accumulate, verify against ring-arithmetic reference

**Implementation** (`prism-ring/src/accumulate.rs`):
```rust
#[inline]
pub fn accumulate<W: RingWord>(acc: W, a: W, b: W) -> W {
    acc.wrapping_add(a.wrapping_mul(b))
}
```
Compiles to `imul` + `iadd` — two ALU instructions at any quantum level.

### Reusable existing code
- `PrimOp` enum shape, arity, commutativity from [prim.rs](crates/hologram-core/src/op/prim.rs)
- UOR trait impl patterns from [byte_ring.rs](crates/hologram-core/src/ring/byte_ring.rs) — Ring, Group, DihedralGroup, NormedDivisionAlgebra, CayleyDicksonConstruction
- Datum/Address patterns from [datum/mod.rs](crates/hologram-core/src/datum/mod.rs)
- Test sampling from [ring_conformance.rs](crates/hologram-core/tests/ring_conformance.rs) — step_by primes for exhaustive, spot vectors for Q3+
- `assert_throughput` from [perf_contract.rs](crates/hologram-core/tests/perf_contract.rs)

---

## Phase 2: prism-graph — Parametric Graph IR

### Phase 2A: Arena graph + GraphOp

**Tests** (`prism-graph/tests/graph_conformance.rs`):

Transplant from existing hologram-graph inline tests:
1. Empty graph: `node_count == 0`, `is_empty`
2. Add/get/remove: generational safety (stale NodeIds return None)
3. Slot reuse: same index, different generation
4. Edge connectivity: `predecessors`, `successors` correct
5. Named I/O: `add_input`, `add_output`, `sources`, `sinks`
6. ConstantStore: add/get round-trip (bit-identical)
7. GraphBuilder: fluent API produces valid graph topology

New tests for new GraphOp:
8. `GraphOp::Accumulate` has arity 3 (acc, a, b)
9. `GraphOp::Reduce { op, axis }` has arity 1 (reduces along axis)
10. `GraphOp::Broadcast { shape }` has arity 1
11. `GraphOp::Reshape { target }` has arity 1
12. **Exhaustive variant check**: no `Lut`, `FusedView`, `FusedView16`, `Float`, `FusedFloatChain`, `MatMulLut*`, `BatchMatMulLut*`, `RingPrimUnary`, `RingPrimBinary` variants exist in the enum. The type system enforces ring closure.

**Implementation**:
- Copy arena/node/edge from [node.rs](crates/hologram-graph/src/graph/node.rs) — NodeId, InputSlot, InputSource, Node unchanged
- New `GraphOp<Q: QuantumLevel>`:
  ```rust
  pub enum GraphOp<Q: QuantumLevel> {
      Input,
      Output,
      Constant(ConstantId),
      Prim(PrimOp),
      Activation(ActivationOp),
      Accumulate,                         // α: fused multiply-add
      Reduce { op: PrimOp, axis: u32 },   // iterated op along axis
      Broadcast { shape: ShapeSpec },      // replicate across dimensions
      Reshape { target: ShapeSpec },       // reinterpret layout
      Fused(FusedOp),                     // algebraically simplified composition
      Custom { id: CustomOpId, arity: u8 },
      CallSubgraph(SubgraphId),
      Passthrough,
      _Phantom(PhantomData<Q>),
  }
  ```
- GraphBuilder: transplant from [builder/mod.rs](crates/hologram-graph/src/builder/mod.rs)
- ConstantStore: transplant from [constant/mod.rs](crates/hologram-graph/src/constant/mod.rs)
- SubgraphDef: transplant from [subgraph/](crates/hologram-graph/src/subgraph/)

### Phase 2B: Matmul as subgraph pattern

**Tests** (`prism-graph/tests/matmul_pattern_conformance.rs`):

1. Build matmul subgraph (Broadcast + Accumulate + Reduce) for M=2, K=3, N=2
2. Evaluate via interpreted ring ops: matches reference `C[i,j] = Σ_k A[i,k] * B[k,j]` (exact in ring)
3. Pattern recognition: `recognize_matmul(&graph)` returns `Some(MatmulPattern { m, k, n })`
4. Psumbook reordering: reordered accumulation (group by quantized weight index, then multiply by centroids) produces identical ring result as standard order (addition is commutative and associative in the ring)
5. Convolution pattern: expressible as matmul with reshaped inputs (im2col)
6. Attention pattern: Q*K^T matmul → activation → V matmul, all ring-native

**Implementation** (`prism-graph/src/patterns/`):
- `matmul.rs`: Build matmul subgraph template
- `conv.rs`: Convolution as reshaped matmul
- `attention.rs`: Multi-head attention as matmul chain
- `recognize.rs`: Pattern matching on subgraph structure → PatternAnnotation

### Phase 2C: Algebraic fusion

**Tests** (`prism-graph/tests/fusion_conformance.rs`):

1. **Chain composition**: Prim(Neg) → Prim(Bnot) → fuses to single Fused node
2. **Involution cancellation**: Neg → Neg → Passthrough (identity, zero cost)
3. **Critical identity fusion**: Bnot → Neg chain → recognized as Succ
4. **Constant folding**: Const(5) → Prim(Succ) → Const(6), at any Q level
5. **CSE**: identical `(op, inputs)` pairs deduplicated (same as existing)
6. **Activation composition**: Activation(Relu) → Activation(Sigmoid) → merged polynomial
7. **Dead code elimination**: nodes with no consumers removed
8. **Semantic preservation**: fused graph produces bit-identical output as unfused for 1000 random inputs at Q3

**Implementation** (`prism-graph/src/fusion/`):
- `compose.rs`: Chain PrimOp sequences into FusedOp
- `cancel.rs`: Involution detection and cancellation
- `identity.rs`: Critical identity recognition (bnot∘neg → succ)
- `constant.rs`: Constant folding via `PrimOp::apply_*`
- `cse.rs`: Transplant from [cse.rs](crates/hologram-graph/src/fusion/cse.rs)
- `dce.rs`: Dead code elimination

### Phase 2D: Scheduling

**Tests** (`prism-graph/tests/schedule_conformance.rs`):

Transplant existing tests:
1. Toposort produces valid topological order
2. Level assignment: no node scheduled before dependencies
3. Critical path length matches manual calculation
4. Parallel levels: all nodes in a level have satisfied dependencies
5. Liveness intervals: `[born, dies]` correct for each node

**Implementation**: Transplant [schedule/](crates/hologram-graph/src/schedule/) — unchanged.

### Reusable existing code
- [node.rs](crates/hologram-graph/src/graph/node.rs): NodeId, InputSlot, InputSource, Node — verbatim
- [builder/mod.rs](crates/hologram-graph/src/builder/mod.rs): GraphBuilder fluent API
- [constant/mod.rs](crates/hologram-graph/src/constant/mod.rs): ConstantStore
- [schedule/](crates/hologram-graph/src/schedule/): toposort, levels, critical_path — verbatim
- [fusion/cse.rs](crates/hologram-graph/src/fusion/cse.rs): CSE pass

---

## Phase 3: prism-archive — .holo Format

**Tests** (`prism-archive/tests/archive_conformance.rs`):

1. Header magic: `b"HOLO"`, version 2 (major version bump for ring-native format)
2. Quantum level in header: write Q3 → read → `quantum_index == 3`
3. Graph round-trip: write graph → read → same node_count, same topology (bit-identical serialization)
4. Constants round-trip: write ring constants → read → identical bytes
5. Weight section: write weights → read → identical bytes, dedup works
6. Checksum: corrupt one byte → load returns error
7. Version: v1 archives → load returns `UnsupportedVersion` (clean break, no backward compat)
8. Multi-model: multiple graphs in one archive

**Implementation**:
- Fork `hologram-archive`, keep magic `HOLO`, bump version to 2
- Add `quantum_index: u32` to header
- Serialize new `GraphOp<Q>` via rkyv (no `ElementWiseView`, `FusedView16`, `FloatOp`)
- Remove all eliminated types from serialization
- v1 archives are not loadable — hologram-ai must recompile models

### Reusable existing code
- [format/mod.rs](crates/hologram-archive/src/format/mod.rs): Header + section layout
- [loader/mod.rs](crates/hologram-archive/src/loader/mod.rs): Zero-copy load
- [writer/mod.rs](crates/hologram-archive/src/writer/mod.rs): Archive writing
- [weight/mod.rs](crates/hologram-archive/src/weight/mod.rs): Weight storage + dedup
- [checksum/mod.rs](crates/hologram-archive/src/checksum/mod.rs): CRC32/Blake3

---

## Phase 4: prism-compiler — Compilation Pipeline

**Tests** (`prism-compiler/tests/compiler_conformance.rs`):

1. Compile empty graph → valid .holo v2 archive
2. Compile linear chain (Input→Prim(Add)→Output) → correct schedule, 1 level
3. Fusion applied: Neg→Bnot chain fused in output
4. Diamond graph: parallel fan-out scheduled into 2+ levels
5. Constant propagation: Const(5)→Succ→ folded to Const(6)
6. Cycle detection: graph with cycle → `CompileError`
7. Liveness: intervals correct, no use-after-free in workspace assignments
8. Workspace: buffer slot reuse respects liveness (slot never live in two intervals)
9. Pattern recognition: Accumulate+Reduce+Broadcast annotated as matmul with tiling metadata

**Tests for pattern stage** (`prism-compiler/tests/pattern_conformance.rs`):
1. Matmul pattern recognized → `PatternAnnotation::Matmul { m, k, n, tile: TileSpec }`
2. Tiling metadata: TM, TN, TK tile sizes derived from target register file
3. Convolution pattern recognized → `PatternAnnotation::Conv { ... }`
4. Attention pattern recognized → `PatternAnnotation::Attention { ... }`
5. Non-pattern subgraph: no false positive match

**Pipeline**:
```
parse → validate → fuse → pattern → schedule → liveness → workspace → emit
```

**Implementation**:
- Fork `hologram-compiler` structure
- Replace fusion stage with prism-graph algebraic fusion
- Add `pattern_stage`: recognize matmul/conv/attention, annotate tiling
- Removed stages: `precision_stage` (parametric ring), `qedl_stage` (no boundary)

### Reusable existing code
- [compiler/mod.rs](crates/hologram-compiler/src/compiler/mod.rs): Pipeline orchestration
- [liveness/mod.rs](crates/hologram-compiler/src/liveness/mod.rs): Liveness intervals
- [workspace/mod.rs](crates/hologram-compiler/src/workspace/mod.rs): Workspace layout

---

## Phase 5: prism-jit — Cranelift JIT

Cranelift is a new dependency (not present in hologram today). No backend trait. No dispatch. One code path: graph → Cranelift IR → native instructions → function pointer.

### Phase 5A: Cranelift lowering for PrimOp

**Tests** (`prism-jit/tests/jit_primop_conformance.rs`):

For each PrimOp x each QuantumLevel {Q0, Q1, Q3, Q7}:
1. JIT-compiled unary op == `PrimOp::apply_unary` — exhaustive at Q0 (all 256), 10K samples at Q1/Q3/Q7
2. JIT-compiled binary op == `PrimOp::apply_binary` — same sampling
3. Type mapping: Q0→I8, Q1→I16, Q3→I32, Q7→I64 (verified via Cranelift IR inspection)
4. **Bit-identical**: JIT result is not "close to" interpreted — it IS identical. Both are wrapping ALU ops.

**Implementation** (`prism-jit/src/lower.rs`):
```rust
fn cranelift_type<Q: QuantumLevel>() -> Type {
    match Q::BITS { 8 => I8, 16 => I16, 32 => I32, 64 => I64, 128 => I128, _ => unreachable!() }
}
fn lower_prim<Q: QuantumLevel>(builder: &mut FunctionBuilder, op: PrimOp, args: &[Value]) -> Value {
    match op {
        PrimOp::Add  => builder.ins().iadd(args[0], args[1]),
        PrimOp::Sub  => builder.ins().isub(args[0], args[1]),
        PrimOp::Mul  => builder.ins().imul(args[0], args[1]),
        PrimOp::Neg  => builder.ins().ineg(args[0]),
        PrimOp::Bnot => builder.ins().bnot(args[0]),
        PrimOp::Succ => { let one = builder.ins().iconst(cranelift_type::<Q>(), 1); builder.ins().iadd(args[0], one) }
        PrimOp::Pred => { let one = builder.ins().iconst(cranelift_type::<Q>(), 1); builder.ins().isub(args[0], one) }
        PrimOp::Xor  => builder.ins().bxor(args[0], args[1]),
        PrimOp::And  => builder.ins().band(args[0], args[1]),
        PrimOp::Or   => builder.ins().bor(args[0], args[1]),
    }
}
```
Each PrimOp lowers to exactly 1 Cranelift IR instruction (Succ/Pred: 1 const + 1 add/sub).

### Phase 5B: Cranelift lowering for activations

**Tests** (`prism-jit/tests/jit_activation_conformance.rs`):

For each ActivationOp x each QuantumLevel {Q0, Q3, Q7}:
1. JIT-compiled activation == `ActivationOp::apply()` — bit-identical for 256 sample points
2. Piecewise polynomial: correct segment branching in generated code (test boundary inputs specifically)
3. **Consistency oracle** (test-only): compare JIT result against f64 reference via encode/decode, verify within quantization step

**Implementation** (`prism-jit/src/lower_activation.rs`):
- Lower `ActivationOp::decompose()` chain as sequence of Cranelift instructions
- Piecewise polynomials: Cranelift `brif` for segment selection, Horner's method for evaluation
- All Cranelift values are `cranelift_type::<Q>()` — integer types, never float

### Phase 5C: Cranelift lowering for Accumulate + loop nests

**Tests** (`prism-jit/tests/jit_accumulate_conformance.rs`):

1. Single accumulate: `acc + a * b` bit-identical to interpreted
2. Reduction loop: `Σ_i a[i]` bit-identical to loop of wrapping_add
3. Matmul 4x4 via tiled loop nest: bit-identical to interpreted matmul
4. Matmul 64x64: bit-identical
5. Broadcast + accumulate: correct address arithmetic (base + stride * index)

**Implementation** (`prism-jit/src/lower_loop.rs`):
- Loop emission: Cranelift basic blocks → branch → body → increment → compare → branch-back
- Tiling: nested loops with tile sizes from compiler PatternAnnotation
- Accumulate in inner loop: `imul` + `iadd`
- Address computation: `base + stride * index` via `imul` + `iadd` on pointer-width integers

### Phase 5D: JIT module — compile + execute

**Tests** (`prism-jit/tests/jit_e2e_conformance.rs`):

1. Linear chain: Input → Relu → Sigmoid → Output — JIT execute bit-identical to interpreted
2. Diamond graph: parallel fan-out — JIT execute bit-identical to interpreted
3. Matmul graph: 8x8 matmul — JIT execute bit-identical to interpreted
4. Multiple executions: same JIT module, different inputs, correct each time
5. Cache: second compile of same graph reuses cached function pointer
6. Host adaptation: Q7 graph compiles on x86_64 (emits ADD r64, r64 equivalent) and on aarch64 (emits ADD Xd, Xn, Xm equivalent) — verified by Cranelift target detection

**Implementation**:
```rust
pub struct JitModule<Q: QuantumLevel> {
    module: cranelift_jit::JITModule,
    fn_ptr: *const u8,
    _phantom: PhantomData<Q>,
}
impl<Q: QuantumLevel> JitModule<Q> {
    pub fn compile(archive: &HoloArchive) -> Result<Self, JitError>;
    pub unsafe fn execute(&self, inputs: &[&[Q::Word]], outputs: &mut [Vec<Q::Word>]);
}
```
No `ComputeBackend`. No `BackendSelector`. No dispatch. The ring IS the execution model.

### Phase 5E: Performance contracts

**Tests** (`prism-jit/tests/perf_contract.rs`):

Using `assert_throughput` pattern from existing [perf_contract.rs](crates/hologram-core/tests/perf_contract.rs) with 5x CI headroom:
1. JIT compile 100-node linear graph: < 100ms
2. JIT-compiled Q0 unary op throughput: 1M ops in < 10ms (matches existing LUT budget)
3. JIT-compiled composed chain (5 ops): 1M ops in < 50ms
4. JIT-compiled accumulate (single): 1M ops in < 10ms
5. JIT-compiled matmul 64x64 at Q7: < 1ms (native 64-bit ALU)
6. Cache hit: second compile of same graph: < 1ms

---

## Phase 6: prism-compression — Parametric Compression

Can proceed in parallel with Phase 5.

**Tests** (`prism-compression/tests/compression_conformance.rs`):

For each QuantumLevel {Q0, Q1, Q3, Q7}:
1. Ring-diff round-trip: compress → decompress == original (bit-identical, lossless)
2. Ring-diff residuals: correlated data → residuals cluster near ZERO
3. Stratum partitioning: `(BITS + 1)` bins at each level, each partition decompresses correctly
4. Torus decomposition: configurable bit-split, round-trip exact
5. ANS encode/decode: round-trip lossless
6. Compression ratio: correlated data at Q7 compresses > 2:1

**Implementation**:
- Parametrize `ring_diff` over `RingWord`: `diff[i] = data[i].wrapping_sub(pred[i])`
- Parametrize stratum histogram: `BITS + 1` bins
- Torus decomposition: configurable bit positions, generic over `RingWord`
- ANS backend unchanged (byte-level entropy coding)

### Reusable existing code
- [ring_diff.rs](crates/hologram-compression/src/ring_diff.rs): Ring-diff algorithm
- [entropy/](crates/hologram-compression/src/entropy/): ANS backend

---

## Phase 7: Peripheral Crates

Can proceed in parallel with Phase 5-6.

### prism-ffi
- Tests: C ABI smoke tests (compile + link), WASM compilation gate
- Implementation: `extern "C"` wrappers for compile/execute, opaque handles for JitModule

### prism-cli
- Tests: `--help` produces output, `compile` subcommand produces .holo file, `run` executes and prints output
- Subcommands: `compile`, `run`, `inspect`, `bench`

### prism-bench
- Tests: criterion benchmarks compile and run
- Suites: ring ops per level, graph fusion, JIT compile latency, JIT execute throughput, matmul at each Q level, archive I/O

---

## Phase 8: End-to-End Integration + hologram-ai Contract

### Phase 8A: Full pipeline E2E

**Tests** (`tests/e2e_prism.rs`):

1. **Linear chain**: build graph → compile → .holo → load → JIT → execute → verify output is bit-identical to interpreted ring ops
2. **Diamond graph**: parallel fan-out through full pipeline
3. **Matmul**: Accumulate/Reduce subgraph → compile → JIT → verify bit-identical to interpreted matmul
4. **Cross-level**: same graph topology at Q0 and Q7, both produce correct ring-native results
5. **Multi-model**: encoder + decoder in single .holo archive, sequential execution
6. **Activation chain**: Input → encode → Relu → Sigmoid → Gelu → decode → Output, verify against oracle

### Phase 8B: hologram-ai contract conformance

hologram-ai consumes hologram's public API per ADR-0001. The refactored implementation must maintain this contract. The prism crates are an internal implementation — the public API stays `hologram::*`.

**Tests** (`tests/hologram_ai_contract_conformance.rs`):

1. **Graph construction**: `Graph`, `GraphBuilder`, `GraphOp`, `NodeId`, `CustomOpId` — hologram-ai builds graphs with `Prim(PrimOp::Add)`, `Activation(ActivationOp::Gelu)`, `Custom { id, arity }`, `Constant(ConstantId)`, `Accumulate`, `Reduce`
2. **Custom op registry**: `CustomOpRegistry::new()`, `.register(id, arity, handler)`, `.dispatch(id, inputs, constants)` — hologram-ai registers SDPA, RoPE, Embedding, Dequantize handlers
3. **Compile pipeline**: `compile(&graph) -> CompilationOutput` — produces a .holo archive
4. **Archive I/O**: `HoloWriter::write()`, `HoloLoader::load()`, `load_from_bytes()`, `load_from_bytes_zero_copy()` — .holo format round-trips
5. **JIT execution**: `JitModule::compile(archive)`, `execute(inputs, outputs)` — replaces `build_tape_from_plan` + `execute_tape`. The hologram-ai-facing API wraps JIT behind the same ergonomic surface.
6. **KV cache**: `KvCacheState` for autoregressive generation — `execute_with_kv()` variant
7. **Constant data**: `ConstantData::Bytes()` and `ConstantData::Deferred { ... }` for weight storage
8. **Error types**: `GraphError`, `CompileError`, `ExecError` — same error taxonomy

**hologram-ai op mapping (updated for ring-native)**:

| AI Operation | Old GraphOp | New GraphOp |
|---|---|---|
| MatMul (any quantization) | `MatMulLut4/8/16(ConstantId)` | Accumulate+Reduce subgraph (compiler recognizes pattern) |
| Activations (Gelu, Relu, Silu, Tanh, Sigmoid) | `Lut(LutOp::...)` | `Activation(ActivationOp::...)` |
| Binary ops (Add, Mul, Sub, ...) | `Prim(PrimOp::...)` | `Prim(PrimOp::...)` (unchanged) |
| Weight constants | `Constant(ConstantId)` | `Constant(ConstantId)` (unchanged) |
| Attention, Norm, RoPE, Embed, Dequantize | `Custom { id, arity }` | `Custom { id, arity }` (unchanged) |

**Key contract changes hologram-ai must adopt**:
- `LutOp::Gelu` → `ActivationOp::Gelu` (enum rename, same semantics)
- `MatMulLut4(cid)` → `build_matmul_subgraph(a, weights_cid, m, k, n)` (graph builder helper)
- `FloatOp::*` → removed. All float operations expressed as ring-native `Activation`, `Prim`, or `Custom`
- `build_tape_from_plan` + `execute_tape` → `JitModule::compile` + `execute` (API shape change)
- `ElementWiseView` → not exposed (internal to fusion)
- `.holo` archive version bumped to v2 (same magic, same loader API, new graph format)

### Phase 8C: Remove old hologram crates, rename prism → hologram

Once all E2E + contract tests pass:
1. Remove `crates/hologram-*` (old implementation) from workspace
2. Rename `crates/prism-*` → `crates/hologram-*` (the prism implementation IS hologram now)
   - `prism-ring` → `hologram-ring` (was `hologram-core`)
   - `prism-graph` → `hologram-graph` (same name, new internals)
   - `prism-compiler` → `hologram-compiler` (same name, new internals)
   - `prism-archive` → `hologram-archive` (same name, new internals)
   - `prism-jit` → `hologram-jit` (new crate, replaces `hologram-exec`)
   - `prism-compression` → `hologram-compression` (same name, new internals)
   - `prism-ffi` → `hologram-ffi` (same name, new internals)
   - `prism-cli` → `hologram-cli` (same name, new internals)
   - `prism-bench` → `hologram-bench` (same name, new internals)
3. Update root `Cargo.toml`: package name stays `hologram`, re-export from renamed crates
4. Update `src/lib.rs`: flat re-exports maintain the `use hologram::*` public API surface
5. Archive magic stays `HOLO` (not `PRSM`) — it's still hologram's format, just version-bumped
6. Update `CLAUDE.md`, `AGENTS.md`, `README.md`, `hologram.repo.yaml`
7. Binary name stays `hologram`

The public API for hologram-ai is `use hologram::{Graph, GraphBuilder, GraphOp, NodeId, CustomOpId, CustomOpRegistry, ConstantData, ConstantId, ConstantStore, HoloWriter, HoloLoader, ...}`. This surface is preserved. The internals are parametric-ring + Cranelift JIT.

---

## What Is Eliminated (complete list)

Every item below existed because the ring was too small or the architecture allowed escape hatches:

| Eliminated | Reason | Replaced By |
|---|---|---|
| `ElementWiseView` (256B LUT) | Ring has sufficient precision | `ComposedOperation` (ring-primitive chain) |
| `ElementWiseView16` (128KB LUT) | Same | Same |
| All activation LUT tables (21×256B, 21×128KB) | Same | Piecewise polynomial in ring arithmetic |
| `LutOp` enum | LUT concept eliminated | `ActivationOp` with `decompose()` |
| `FloatOp` enum (70+ variants) | Breaks ring closure | Ring-native computation |
| `FusedFloatChain` | Same | Algebraic fusion of ring ops |
| `QedlBoundary` | No float/ring boundary exists | Encoding at graph I/O only |
| `ComputeBackend` / `BackendSelector` | Ring ops are ALU, not GPU | Cranelift JIT |
| Metal backend (32K lines) | Same | Same |
| WebGPU backend (42K lines) | Same | Same |
| `TapeKernel` / tape interpreter | Compilation replaces interpretation | Cranelift JIT |
| `BufferArena` | Pointer arithmetic in JIT code | Workspace layout + JIT addressing |
| `MatMulLut4/8/16` graph ops | Special cases eliminated | Accumulate+Reduce subgraph pattern |
| `BatchMatMulLut4/8/16` graph ops | Same | Same |
| `RingPrimUnary`/`RingPrimBinary` graph ops | Redundant with `Prim(PrimOp)` | `Prim(PrimOp)` |
| `lut_gemm/` module (psumbook, orbit) | LUT-GEMM replaced | JIT-compiled tiled matmul loop nest |
| `float_dispatch/` module | Float domain eliminated | Ring-native activations |
| `precision_stage` in compiler | Parametric ring, no promotion | QuantumLevel parameter |
| `qedl_stage` in compiler | No quantize/dequantize boundary | Encoding at graph I/O only |
| `CurvatureFlux` runtime tracking | Dynamic precision selection eliminated | Compile-time quantum level |

## What Survives (algebraically motivated)

| Surviving | Why |
|---|---|
| 10 ring primitives (Neg, Bnot, Succ, Pred, Add, Sub, Mul, Xor, And, Or) | UOR foundation ontology |
| Dihedral group D_{2^n} (Neg and Bnot as generators) | Ring symmetry group |
| Critical identity `succ = neg ∘ bnot` | Fundamental ring relation |
| Observable algebra (stratum, curvature, rank, domain) | Ring element classification |
| Encoding pipeline (π-F-λ) | Bridge between continuous and ring domains |
| Graph IR (nodes, edges, scheduling, liveness, workspace) | Computation representation |
| Algebraic fusion (now purely ring-native) | Graph optimization |
| Accumulation pattern (matmul as iterated add/mul) | Universal compute primitive |
| UOR trait hierarchy (Ring, Datum, Involution, DihedralGroup, NDA, CD) | Ontological grounding |
| Arena-based graph with generational indexing | Memory safety |
| rkyv serialization, content-addressed archives | Portable persistence |
| Cayley-Dickson chain (R→C→H→O) | Algebraic structure of quantum levels |

---

## Test Mapping: Existing → New

| Existing Test | New Location | Notes |
|---|---|---|
| `ring_conformance.rs` | `prism-ring/tests/ring_word_conformance.rs` + `primop_conformance.rs` + `ring_uor_conformance.rs` | Parametric over all levels; adds UOR trait tests |
| `q3_conformance.rs` | `prism-ring/tests/ring_uor_conformance.rs` (CD chain section) | Octonion non-associativity, associator at Q7 |
| `carry_conformance.rs` | `prism-ring/tests/encoding_conformance.rs` | Lift/lower → encoding embed/lift |
| `perf_contract.rs` | `prism-jit/tests/perf_contract.rs` | Budgets for JIT, not LUT |
| `float_conformance.rs` | **Eliminated** — f64 refs reused as test oracle in `activation_conformance.rs` | FloatOps gone; references become oracle |
| `gemm_conformance.rs` | `prism-graph/tests/matmul_pattern_conformance.rs` | Matmul as subgraph, not LUT-GEMM kernel |
| `quantize_conformance.rs` | **Eliminated** | No quantization orbit model |
| `streaming_conformance.rs` | `prism-jit/tests/jit_e2e_conformance.rs` | Streaming via JIT |
| `shape_chain.rs` | `prism-graph/tests/graph_conformance.rs` | Shape propagation in graph |
| `tests/e2e.rs` | `tests/e2e_prism.rs` | Full pipeline with JIT |

---

## Dependency Graph & Parallelism

```
Phase 0 (scaffold)
    │
Phase 1A─→1B─→1C─→1D─→1E─→1F─→1G─→1H─→1I─→1J   (prism-ring, sequential)
    │
    ├── Phase 2A─→2B─→2C─→2D          (prism-graph)
    │       │
    │       ├── Phase 3                 (prism-archive, parallel with 2C-2D)
    │       │
    │       └── Phase 4                 (prism-compiler, needs 2C+3)
    │               │
    │               └── Phase 5A─→5B─→5C─→5D─→5E  (prism-jit)
    │
    ├── Phase 6                         (prism-compression, parallel with 2-5)
    │
    ├── Phase 7                         (ffi/cli/bench, parallel with 5-6)
    │
    └── Phase 8A─→8B─→8C                 (E2E + contract + rename, needs all above)
```

**Critical path**: 0 → 1 → 2 → 4 → 5 → 8

---

## Verification

After each phase:
```bash
cargo test --workspace              # all tests pass
cargo clippy --workspace -- -D warnings  # no warnings
cargo fmt --all -- --check          # formatted
```

After Phase 5D (JIT E2E):
```bash
cargo test -p prism-jit             # JIT conformance
cargo bench -p prism-bench          # performance baselines
```

After Phase 8A (full E2E):
```bash
cargo test --workspace              # everything including e2e_prism
```
