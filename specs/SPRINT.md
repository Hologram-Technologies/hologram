# Sprint Tracking

## Sprint History

(none yet)

---

## Sprint 1: Foundation & Core LUT Engine

### In Progress

(none)

### Blocked

---

## Completed (Running Log)

### Phase 0: Foundation Setup (Sprint 1)
- [x] Convert `Cargo.toml` to workspace + root crate (edition "2021")
- [x] Create all crate skeletons with subdirectory structure
- [x] Create `AGENTS.md` with dev practices, agent roles, sprint workflow
- [x] Create `CLAUDE.md` with project context
- [x] Create `Justfile` with `ci`, `bench`, `test`, `fmt`, `clippy`, `wasm` targets
- [x] Create `.githooks/pre-commit` hook (fmt check + incremental clippy)
- [x] Add workspace dependencies (uor-foundation, rkyv, bytemuck, rayon, criterion, memmap2, crc32fast, smallvec)
- [x] Configure feature flags (std, simd, parallel, wasm)
- [x] Implement `Primitives` for `HoloPrimitives`
- [x] Root `src/lib.rs` re-exports all subcrate APIs
- [x] Create `.gitignore`
- [x] Verify: `cargo build --workspace`, `cargo test`, `cargo clippy -- -D warnings`

### Phase 1: Core LUT Engine (Sprint 1)
- [x] Port Q0 unary tables (stratum, curvature, domain, rank, torus, orbit) to `lut/q0.rs`
- [x] Port Q0 arithmetic tables (add, sub, mul, pow, gf2_mul, gf3_mul) to `lut/arith.rs`
- [x] Port 21 activation tables to `lut/activation/` (basic, modern, scientific + registry)
- [x] Port `ElementWiseView` to `view/mod.rs` (256-byte table, `#[repr(align(64))]`)
- [x] Port SIMD `apply_slice` to `view/simd.rs` (AVX2 vpshufb + SSE4.2 pshufb, feature-gated)
- [x] Implement `.then()` composition in `view/compose.rs`
- [x] Implement `ByteRing` (Z/256Z) in `ring/byte_ring.rs` — implements uor-foundation Ring trait
- [x] Implement `ByteInvolution` (Neg/Bnot) — implements Operation, UnaryOp, Involution traits
- [x] Implement `Encoding` trait + 4 encodings (angle, signed, unsigned, raw) in `encoding/`
- [x] Implement `PrimOp` (10 ops) + `LutOp` (21 ops) + unified `Op` enum in `op/`
- [x] Implement `ByteDatum` + `ByteAddress` in `datum/` — implements uor-foundation Datum, Address traits
- [x] Implement `CoreError` in `error/`
- [x] Add rkyv derives to `ElementWiseView`, `ByteDatum`, `ByteAddress`, `Op`, `PrimOp`, `LutOp` (all with `#[archive(check_bytes)]`)
- [x] Write Criterion benchmarks: `benches/lut.rs` (7 benchmarks), `benches/view.rs` (11 benchmarks incl. rkyv serialize/deserialize)
- [x] 108 tests passing, zero clippy warnings
