# Plan 074: uor-foundation 0.1.4 → 0.3.0 Upgrade

**Status:** In progress
**Date:** 2026-04-16

## Context

hologram depends on `uor-foundation = "0.1.4"` (workspace root `Cargo.toml:33`). Version 0.3.0 renames `QuantumLevel` → `WittLevel` (a struct with different constants and methods), renames `const_ring_eval_q*` → `const_ring_eval_w*`, removes `ViolationKind`, and reorganizes modules. This upgrade unblocks access to new enforcement APIs and keeps hologram current with the UOR-Framework.

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
