# Plan 074: uor-foundation 0.1.4 → 0.3.0 Upgrade

**Status:** Scope-revised — see addendum below
**Date:** 2026-04-16 (revised 2026-04-28)

## Context

hologram depends on `uor-foundation = "0.1.4"` (workspace root `Cargo.toml:33`). Version 0.3.0 renames `QuantumLevel` → `WittLevel` (a struct with different constants and methods), renames `const_ring_eval_q*` → `const_ring_eval_w*`, removes `ViolationKind`, and reorganizes modules. This upgrade unblocks access to new enforcement APIs and keeps hologram current with the UOR-Framework.

---

## Scope addendum (2026-04-28 — verified against 0.3.0)

The original plan was written assuming 0.2.1 changes; **the actual 0.3.0
surface is significantly larger.** A fresh `cargo check --workspace`
against `uor-foundation = "0.3.0"` produces **148 errors**, of which 109
are downstream cascades of one root: `PrismPrimitives` no longer
implements the (now-renamed) host-types trait. The breaking changes are:

### Replaced traits (not renamed)

- **`Primitives` trait → `HostTypes` trait.** Different field set entirely:
  - Old: `String`, `Integer`, `NonNegativeInteger`, `PositiveInteger`,
    `Decimal`, `Boolean`
  - New: `Decimal`, `HostString: ?Sized`, `WitnessBytes: ?Sized`
  - Affects every `<P: Primitives>` trait bound in `hologram-ring` and
    `hologram-core` (~120 sites).
  - The `String`/`Integer`/`Boolean` slots are gone from the host-types
    trait — anywhere the project relied on them needs a different source.

- **`Address` trait → `Element` trait** (`uor_foundation::kernel::address`).
  Semantic shift, not a rename. New methods require **content-addressable
  digest semantics**:
  - `length()`, `addresses()`, `digest()`, `digest_algorithm()`,
    `canonical_bytes()`, `witt_length()`.
  - `digest_algorithm` must be `"blake3"` (primary) or `"sha256"` (secondary).
  - `canonical_bytes` is per *Amendment 43 §2*: `header(k) || le_bytes(x, k+1)`.
  - **Open question:** which hash algorithm does hologram standardise on
    for ring datums? Plan needs to commit to one and back-port to the
    archive format if persisted.

### Method-signature changes on `Datum<H>` and `Ring<H>`

- `Datum::quantum()` → `Datum::witt_length()` (rename)
- `Datum::glyph()` → `Datum::element()` (semantic shift — returns
  `Element` impl, not the old glyph value)
- **New required methods** with no obvious mechanical mapping:
  - `Datum::value() -> u64`
  - `Datum::stratum() -> u64`  ← needs domain decision
  - `Datum::spectrum() -> u64` ← needs domain decision
- `Ring::ring_quantum()` → `Ring::ring_witt_length()` (rename)
- `Ring::at_quantum_level()` → `Ring::at_witt_level()` (rename)
- Old `Datum::Address` associated type removed; new `Datum::Element`
  associated type bounded by `Element<H>`.

### Renames (mechanical)

- `QuantumLevel` enum → `WittLevel` struct.
  `Q0/Q1/Q2/Q3` → `W8/W16/W24/W32` (mapping in plan §2b).
- `.index()` → `.witt_length()` — semantic shift: `Q0.index()` was 0,
  `W8.witt_length()` is 8.
- `const_ring_eval_q*` → `const_ring_eval_w*` (per original plan).

### Affected files (verified by `cargo check`)

- `hologram-ring`: every UOR trait impl in `lib.rs` (`PrismPrimitives`),
  `ring.rs` (`PrismRing`, `PrismDivisionAlgebra`,
  `PrismMultTable`), `datum.rs`, `address.rs`, `involution.rs`,
  `prim.rs`. Also `level.rs`'s `QuantumLevel` re-exports.
- `hologram-core`: `lib.rs` (`HoloPrimitives`),
  `op/mod.rs` (`RingLevel::to_quantum`/`from_quantum`),
  `op/prim.rs` (`PrimOp::to_foundation`/`from_foundation`),
  `q1`/`q2`/`q3` ring + datum modules,
  `datum/mod.rs` (`ByteDatum::Datum`/`Address` impls), `quantum/mod.rs`,
  `carry/mod.rs`, `term/compile_unit.rs`, `ring/byte_ring.rs`.
- `hologram-cascade`: every file under `src/` consuming
  `uor_foundation::kernel::*`.
- `hologram-compiler`: preflight + lib + term_lower.
- Tests: `hologram-core/tests/q3_conformance.rs`,
  `hologram-core/tests/ring_conformance.rs`.

### Domain decisions required (cannot be answered by mechanical work)

1. **Digest algorithm choice** for `Element::digest_algorithm`. Likely
   `"blake3"` (foundation's primary) but pins to project policy.
2. **Canonical-bytes format** per Amendment 43 §2 — needs the spec read
   to confirm `header(k) || le_bytes(x, k+1)` matches hologram's existing
   ring-element serialisation, or requires migrating the archive format.
3. **`Datum::stratum`, `Datum::spectrum` semantics** for hologram's
   `ByteDatum`, `RingDatum`, `q1`/`q2`/`q3` datums. The 0.3.0 docstring
   says stratum is "the ring-layer index" and spectrum is "bit-pattern
   representation in the hypercube geometry of Z/(2^n)Z" — needs project
   commitment to a specific encoding.
4. **`HostTypes::HostString` / `HostTypes::WitnessBytes`** target types
   for `HoloPrimitives` and `PrismPrimitives` — `str`/`[u8]` per
   `DefaultHostTypes`, or owned types.

### Recommended approach

Given the scope and the open domain questions, this should be a
**dedicated multi-day branch**, not a sprint-side task. Order of
operations:

1. Read uor-foundation 0.3.0 docs + Amendment 43 §2 to resolve the
   digest/canonical-bytes/stratum/spectrum questions.
2. Bump version on a branch; let the compiler enumerate the work.
3. Migrate `Primitives` impls first (root cause of 109/148 errors).
   Pick `HostString = str`, `WitnessBytes = [u8]` unless project policy
   says otherwise.
4. Implement `Datum::value/stratum/spectrum/element` for every
   `*Datum` type in the project — write conformance tests for each.
5. Implement `Element` trait — defer to a `BlakeAddress` impl or
   similar that hashes the canonical bytes once at construction.
6. Apply mechanical renames last (`QuantumLevel`/`Q*`/method names).

Original §2 below describes the 0.2.1-proxy mechanical work that is
still relevant for that step.

---

## Phase 1: Bump version and capture compiler errors

1. Edit `Cargo.toml:33` → `uor-foundation = "0.3.0"`
2. Run `cargo check --workspace 2>&1 | tee /tmp/uor-errors.txt`
3. Use compiler output as the authoritative task list (the plan below is based on 0.2.1 as proxy — 0.3.0 may have additional changes)

---

## Phase 2: Import path + rename fixes (mechanical)

### 2a. QuantumLevel → WittLevel type rename

**Strategy**: Keep hologram's public API stable by re-exporting with an alias:

```rust
// hologram-core/src/op/mod.rs:21
pub use uor_foundation::WittLevel as QuantumLevel;
```

This preserves `hologram_core::op::QuantumLevel` for all internal consumers and hologram-ai. Only code that imports directly from `uor_foundation` needs updating.

**Files importing directly from uor_foundation** (update `QuantumLevel` → `WittLevel`):
- `hologram-ring/src/ring.rs` — trait impl returns
- `hologram-compiler/src/lib.rs:11`
- Test files importing `uor_foundation::QuantumLevel` directly

### 2b. Constant rename: Q0/Q1/Q2/Q3 → W8/W16/W24/W32

Mapping: `QuantumLevel::Q0` → `WittLevel::W8`, `Q1` → `W16`, `Q2` → `W24`, `Q3` → `W32`

**Key file**: `hologram-core/src/op/mod.rs:62-68` — `RingLevel::to_quantum()`:
```rust
pub const fn to_quantum(self) -> uor_foundation::WittLevel {
    match self {
        Self::Q0 => uor_foundation::WittLevel::W8,
        Self::Q1 => uor_foundation::WittLevel::W16,
        Self::Q2 => uor_foundation::WittLevel::W24,
        Self::Q3 => uor_foundation::WittLevel::W32,
    }
}
```

### 2c. Method rename: `.index()` → `.witt_length()`

**Semantic change**: `Q0.index()` = 0, but `W8.witt_length()` = 8.

Update `RingLevel::from_quantum()` at `op/mod.rs:50-57`:
```rust
pub const fn from_quantum(q: uor_foundation::WittLevel) -> Option<Self> {
    match q.witt_length() {
        8 => Some(Self::Q0),
        16 => Some(Self::Q1),
        24 => Some(Self::Q2),
        32 => Some(Self::Q3),
        _ => None,
    }
}
```

Update `QuantumLevelExt` at `op/mod.rs:99-103`:
```rust
impl QuantumLevelExt for uor_foundation::WittLevel {
    fn byte_width(self) -> u8 {
        (self.witt_length() / 8) as u8  // W8→1, W16→2, W24→3, W32→4
    }
}
```

Other `.index()` call sites (precision.rs, certificate.rs, shape.rs) — update comparisons to use `.witt_length()` with appropriate numeric values.

### 2d. const_ring_eval renames

File: `hologram-core/src/ring/const_eval.rs:6-10`

```rust
pub use uor_foundation::enforcement::{
    const_ring_eval_w8, const_ring_eval_w16, const_ring_eval_w32, const_ring_eval_w64,
    const_ring_eval_unary_w8, const_ring_eval_unary_w16, const_ring_eval_unary_w32,
    const_ring_eval_unary_w64,
};
```

Update `eval_binary` and `eval_unary` function bodies (lines 18-42) to use new names.

### 2e. ViolationKind removal

3 sites — adapt to whatever 0.3.0's `ShapeViolation` constructor looks like:
- `hologram-cascade/src/dispatch_decl.rs:69`
- `hologram-cascade/src/effect_decl.rs:79`
- `hologram-compiler/src/preflight/enforcement_validate.rs:87`

---

## Phase 3: Semantic fixes (compiler-driven)

These depend on what 0.3.0 actually exposes — fix based on `cargo check` errors after Phase 2:

- **HostTypes trait**: If `CompileUnitBuilder` requires it, implement on `HoloPrimitives`/`PrismPrimitives`
- **Trait method renames**: e.g. `at_quantum_level()` → `at_witt_level()` in `kernel::schema::Ring`
- **`QuantumLevel::new(k)` → `WittLevel::new(witt_length)`**: Sites using dynamic construction need `new(8 * (k + 1))` transform
- **hologram-ring/src/ring.rs**: Macro-generated impls returning `QuantumLevel` variants — update to `WittLevel::W8` etc.

---

## Phase 3b: Archive backward-compatibility safeguard

**Finding**: QuantumLevel is never rkyv-serialized directly. Archives store `u8` via `.index() as u8` (values 0,1,2,3). Since `WittLevel` doesn't have `.index()`, serialization paths must be updated to preserve the same u8 encoding.

**Affected serialization sites:**
- `hologram-cascade/src/engine.rs:407` — `state.quantum_level.index() as u8` → convert through `RingLevel` instead
- `hologram-cascade/src/certificate.rs` — manual binary encoding via `.index() as u8`

**Fix**: Use `RingLevel::from(level) as u8` or an explicit mapping fn to guarantee 0/1/2/3 encoding regardless of WittLevel internals.

**Add round-trip test** (new test in `hologram-archive/tests/` or `hologram-cascade/tests/`):
```rust
#[test]
fn archive_quantum_level_encoding_backward_compat() {
    // Verify each WittLevel maps to the expected u8 in the archive format
    assert_eq!(RingLevel::from(WittLevel::W8) as u8, 0);
    assert_eq!(RingLevel::from(WittLevel::W16) as u8, 1);
    assert_eq!(RingLevel::from(WittLevel::W24) as u8, 2);
    assert_eq!(RingLevel::from(WittLevel::W32) as u8, 3);
    // Optionally: write a CompileUnitMeta, reload, verify field values
}
```

---

## Phase 4: Verify

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo check --workspace --target wasm32-unknown-unknown  # wasm compat
```

---

## Critical files to modify

| File | Changes |
|------|---------|
| `Cargo.toml:33` | Version bump |
| `crates/hologram-core/src/op/mod.rs` | Re-export alias, `from_quantum`, `to_quantum`, `QuantumLevelExt`, `From` impls |
| `crates/hologram-core/src/ring/const_eval.rs` | Function rename imports + body |
| `crates/hologram-ring/src/ring.rs` | Return value constants in trait impls |
| `crates/hologram-cascade/src/dispatch_decl.rs` | ViolationKind removal |
| `crates/hologram-cascade/src/effect_decl.rs` | ViolationKind removal |
| `crates/hologram-cascade/src/certificate.rs` | `.index()` → RingLevel conversion |
| `crates/hologram-cascade/src/precision.rs` | `.index()` → `.witt_length()` |
| `crates/hologram-cascade/src/engine.rs:407` | Archive encoding via RingLevel |
| `crates/hologram-compiler/src/preflight/shape.rs` | `.index()` → `.witt_length()` dispatch values |
| `crates/hologram-compiler/src/preflight/enforcement_validate.rs` | ViolationKind + possible builder API changes |
| ~10 test files | Constant renames |

---

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| 0.3.0 has more breakage than predicted from 0.2.1 | Phase 1 captures ALL errors; iterate |
| Archive binary format break | QuantumLevel stored as `u8` (not rkyv'd directly). `RingLevel` is hologram-owned `#[repr(u8)]` — unchanged. Serialization paths go through RingLevel. Round-trip test added. |
| hologram-ai downstream break | Re-export alias `QuantumLevel = WittLevel` preserves API |
| wasm compatibility | Phase 4 explicit wasm check |
| engine.rs/certificate.rs encode wrong u8 | Explicit mapping through RingLevel ensures 0/1/2/3 encoding regardless of WittLevel internals |
