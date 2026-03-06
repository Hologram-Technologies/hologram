# Sprint 8: Constrained Device Validation

**Status**: Completed
**Test delta**: +15 (StaticBuf unit tests), ~651 total workspace
**Zero clippy warnings**

---

## Goal

Validate `hologram-core` on constrained targets (WASM no_std, bare-metal ARM). Add `no_alloc`
static-buffer mode. Upgrade rkyv 0.7 → 0.8.15. Document feature availability per target.

## Deliverables

- [x] Verify `hologram-core` compiles `no_std`: `wasm32-unknown-unknown` — no_std, no rkyv
- [x] Verify `hologram-core` compiles for bare-metal ARM: `thumbv7em-none-eabihf` — no_std
- [x] Fix `f64::rem_euclid()` std-only call in `encoding/angle.rs` with manual `% TAU` modulo
- [x] `no_alloc` marker feature in `hologram-core/Cargo.toml`
- [x] `serialize` feature gates rkyv (optional dep); excluded for no_std builds automatically
- [x] `StaticBuf<const N: usize>` in `crates/hologram-core/src/buffer/static_buf.rs`
  — `push/pop/extend_from_slice/as_slice/clear/is_full/capacity`, 15 unit tests
- [x] Binary size: wasm32 ~40 KB `.text`, thumbv7em ~35 KB (well under 100 KB target)
- [x] `Justfile` `embedded` recipe (thumbv7em via rustup toolchain)
- [x] `Justfile` `wasm-nostd` recipe (wasm32 no_std via rustup toolchain)
- [x] `specs/feature-matrix.md`: features per target (x86_64, wasm32, thumbv7em, esp32)
- [x] **rkyv 0.8 upgrade** (added mid-sprint):
  - Upgraded workspace dep from 0.7 → 0.8.15
  - Removed all `#[archive(check_bytes)]` / `#[rkyv(derive(CheckBytes))]` (auto-derived in 0.8)
  - Replaced `to_bytes::<_, N>` → `to_bytes::<rkyv::rancor::Error>` across all crates
  - Replaced `check_archived_root + deserialize` → `rkyv::from_bytes::<T, rkyv::rancor::Error>`
  - Removed `rkyv::Infallible` usage (removed in 0.8)
  - Applied to `tests/e2e.rs` as well (sed had only covered `crates/`)

## Notes

- Homebrew Rust lacks cross-compilation targets; Justfile recipes use rustup toolchain paths explicitly
- rkyv 0.8 auto-derives `CheckBytes` with `Archive` when `bytecheck` feature is enabled — no manual derive needed
- rkyv 0.8 API: single-generic `to_bytes::<E>`, combined `from_bytes` replaces separate check+deserialize
