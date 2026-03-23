# Plan 015: Dead Code Removal + Tape Custom Ops + Hot Path Optimization

## Context

After Sprint 17 removed KvExecutor, ~1,600 lines of supporting code are dead. Custom ops (`GraphOp::Custom`) were only dispatched through KvExecutor's registry — the tape path errors on them. We'll wire CustomOpRegistry into the tape builder so hologram-ai can register SDPA/SwiGLU/RoPE handlers, delete genuinely dead code, and optimize binary broadcasting.

---

## Part A: Dead Code Removal (~1,500 lines)

### A1. Delete dead eval modules

- **Delete** `crates/hologram-exec/src/eval/shape_propagate.rs` (204 lines)
- **Delete** `crates/hologram-exec/src/eval/shape_resolve.rs` (862 lines)
- **Update** `crates/hologram-exec/src/eval/mod.rs` — remove the two `pub mod` lines

### A2. Delete dead standalone modules

- **Delete** `crates/hologram-exec/src/dirty_bits.rs` (141 lines)
- **Delete** `crates/hologram-exec/src/profile.rs` (244 lines)
- **Update** `crates/hologram-exec/src/lib.rs` — remove `pub mod dirty_bits;` and `#[cfg(feature = "profile")] pub mod profile;`

### A3. Remove old Tape/Instruction/KernelFn from tape.rs

**File:** `crates/hologram-exec/src/tape.rs` — lines ~45-165

Delete `KernelFn` type alias, `Instruction` struct, `Tape` struct, `impl Tape`, and the tests that use them (~lines 1500-1600). These test the old dispatch path, not EnumTape.

---

## Part B: Tape-Compatible Custom Ops

### B1. Add `TapeKernel::Custom` variant

**File:** `crates/hologram-exec/src/tape.rs`

Add to the `TapeKernel` enum:
```rust
/// Custom op — handler baked at tape build time from registry.
Custom(Arc<dyn Fn(&[&[u8]], &ConstantStore) -> ExecResult<Vec<u8>> + Send + Sync>),
```

Add dispatch in `dispatch_kernel()` and `dispatch_kernel_par()`:
```rust
TapeKernel::Custom(handler) => {
    let result = handler(&input_refs, /* constants from ctx */)?;
    out_buf.extend_from_slice(&result);
}
```

### B2. Wire registry into tape builder

**File:** `crates/hologram-exec/src/tape_builder.rs`

Change `build_tape` signature to accept optional registry:
```rust
pub fn build_tape(
    sg: &SerializedGraph,
    schedule: &ExecutionSchedule,
    registry: Option<&CustomOpRegistry>,
) -> ExecResult<EnumTape>
```

In `resolve_kernel`, add a `GraphOp::Custom { id, arity }` arm:
```rust
GraphOp::Custom { id, arity } => {
    let reg = registry.ok_or_else(|| ExecError::UnsupportedOp(
        format!("custom op {} requires a CustomOpRegistry", id.raw())
    ))?;
    let handler = reg.get_handler(*id).ok_or_else(|| ExecError::UnsupportedOp(
        format!("custom op {} not registered", id.raw())
    ))?;
    Ok(TapeKernel::Custom(handler.clone()))
}
```

This means `resolve_kernel` also needs the registry parameter.

### B3. Update `build_tape_from_plan`

**File:** `crates/hologram-exec/src/mmap/mod.rs`

Add a `build_tape_from_plan_with_ops` variant:
```rust
pub fn build_tape_from_plan_with_ops(
    plan: &LoadedPlan,
    registry: &CustomOpRegistry,
) -> ExecResult<EnumTape> {
    let schedule = build_schedule(plan.graph())?;
    crate::tape_builder::build_tape(plan.graph(), &schedule, Some(registry))
}
```

Keep `build_tape_from_plan` passing `None` for the registry (backward compatible).

### B4. Re-export

**File:** `crates/hologram-exec/src/lib.rs` — add `build_tape_from_plan_with_ops` to mmap re-exports
**File:** `src/lib.rs` — add to root re-exports

### B5. Add test

Add a test in `mmap/mod.rs` tests: build a graph with `GraphOp::Custom { id: 1, arity: 1 }`, register a passthrough handler, build tape with registry, execute, verify output matches input.

---

## Part C: Binary Broadcasting Optimization

### C1. Add `binary_broadcast` helper

**File:** `crates/hologram-exec/src/tape.rs`

```rust
/// Binary elementwise with broadcasting. Fast paths avoid per-element modulo.
#[inline(always)]
fn binary_broadcast(a: &[f32], b: &[f32], dst: &mut [f32], f: impl Fn(f32, f32) -> f32) {
    if a.len() == b.len() {
        for (d, (&x, &y)) in dst.iter_mut().zip(a.iter().zip(b.iter())) {
            *d = f(x, y);
        }
    } else if b.len() == 1 {
        let bv = b[0];
        for (d, &x) in dst.iter_mut().zip(a.iter()) {
            *d = f(x, bv);
        }
    } else if a.len() == 1 {
        let av = a[0];
        for (d, &y) in dst.iter_mut().zip(b.iter()) {
            *d = f(av, y);
        }
    } else {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = f(a[i % a.len()], b[i % b.len()]);
        }
    }
}
```

Replace the modulo loops in both `inline_binary()` and `inline_binary_f32()` with a call to `binary_broadcast(a, b, dst, f)`.

Add tests:
- Same-size: `binary_broadcast(&[1.0, 2.0], &[3.0, 4.0], ..., f32::add)` → `[4.0, 6.0]`
- Scalar-b: `binary_broadcast(&[1.0, 2.0, 3.0], &[10.0], ..., f32::add)` → `[11.0, 12.0, 13.0]`
- Scalar-a: `binary_broadcast(&[10.0], &[1.0, 2.0], ..., f32::add)` → `[11.0, 12.0]`
- General: `binary_broadcast(&[1.0, 2.0], &[10.0, 20.0, 30.0], ..., f32::add)` — uses modulo fallback

### C2. Pre-size consumer_counts

**File:** `crates/hologram-exec/src/tape_builder.rs` — `apply_reuse_flags()`

Pre-allocate based on max output_idx instead of dynamic resizing.

---

## Part D: SPRINT.md Update

Add to Sprint 17:
```markdown
### Phase 4: Dead Code Removal
- [ ] **4.1**: Remove shape_propagate.rs + shape_resolve.rs (1066 lines)
- [ ] **4.2**: Remove dirty_bits.rs + profile.rs (385 lines)
- [ ] **4.3**: Remove old Tape/Instruction/KernelFn + their tests

### Phase 5: Tape-Compatible Custom Ops
- [ ] **5.1**: TapeKernel::Custom variant + dispatch
- [ ] **5.2**: Wire CustomOpRegistry into tape_builder
- [ ] **5.3**: build_tape_from_plan_with_ops entry point
- [ ] **5.4**: Custom op E2E test

### Phase 6: Tape Hot Path Optimization
- [ ] **6.1**: binary_broadcast helper — eliminate modulo for same-size/scalar cases
- [ ] **6.2**: Pre-size consumer_counts in apply_reuse_flags
```

---

## Files to modify

1. **Delete:** `crates/hologram-exec/src/eval/shape_propagate.rs`
2. **Delete:** `crates/hologram-exec/src/eval/shape_resolve.rs`
3. **Delete:** `crates/hologram-exec/src/dirty_bits.rs`
4. **Delete:** `crates/hologram-exec/src/profile.rs`
5. **Edit:** `crates/hologram-exec/src/eval/mod.rs` — remove dead module refs
6. **Edit:** `crates/hologram-exec/src/lib.rs` — remove dead modules, add new exports
7. **Edit:** `crates/hologram-exec/src/tape.rs` — remove old Tape, add Custom variant + binary_broadcast
8. **Edit:** `crates/hologram-exec/src/tape_builder.rs` — accept registry, handle Custom, pre-size
9. **Edit:** `crates/hologram-exec/src/mmap/mod.rs` — add build_tape_from_plan_with_ops
10. **Edit:** `crates/hologram-exec/src/kv/registry.rs` — add `get_handler()` method
11. **Edit:** `src/lib.rs` — add build_tape_from_plan_with_ops to root exports
12. **Edit:** `specs/SPRINT.md` — add Phases 4-6

## Verification

1. `cargo test --workspace` — all tests pass
2. `cargo bench -p hologram-bench --bench executor -- diamond` — diamond bench uses binary Add, should improve
3. `cargo clippy --workspace -- -D warnings` — zero warnings
4. New test: custom op passthrough via tape path
