# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cross-substrate kernel conformance digest (`hologram_compute::cpu::conformance`) and
  its wasm32 driver (`crates/hologram-compute/conformance-wasm`, workspace-excluded):
  `scripts/wasm-conformance.sh` (`just wasm-conformance`, CI job `wasm-conformance`) runs
  one digest body natively and on wasm32 baseline, SIMD128 and relaxed-SIMD, and fails
  unless the f32 dot/sum digests are equal. The `matmul_f32_blocked` digest is reported
  per substrate and does not gate, because it differs (KC-7).

### Changed
- `simd_f32_dot` and `simd_f32_sum` now use one canonical association on every target
  (sixteen lanes over blocks of sixteen, a halving tree, a scalar tail; no FMA), with
  AVX2, SSE2, NEON and SIMD128 twins that keep the same lanes in registers, replacing the
  per-instruction-set kernels that grouped partial sums by lane width. Their results are
  now bit-identical across x86-64, aarch64 and wasm32, and change in the last bits
  relative to 0.13.1 on every target. Attention scores, norm statistics and ReduceSum
  inherit the order. Cost against 0.13.1 on an AVX-512 host: dot about 1.2x slower
  (n = 4096: 106 → 130 ns), sum about 1.8x slower at n = 4096 (53 → 95 ns), because
  the previous kernels used four AVX-512 accumulators. f32 matmul is unchanged and still
  differs across substrates (`docs/numerics/invariance.md`).
- Hardened all public release paths with immutable-byte preflights and idempotent,
  dependency-ordered retries, and made the GitHub source release wait for the
  complete crates.io, npm, and PyPI closure and the exact-tag release V&V run.
- Replaced the unsigned tag as the source-release trust root with a deterministic
  source archive and commit manifest covered by GitHub OIDC/Sigstore build
  provenance; all third-party release actions and toolchains are commit/version
  pinned.
- Made the TypeScript WASM package build use the active pinned Rust toolchain
  instead of silently switching to the floating `stable` alias.

## [0.13.1] - 2026-09-06

### Fixed
- Made the native SDK smoke test load the same platform-specific N-API binary as
  the shipped package loader, covering Linux, macOS, Windows, and the existing
  glibc/musl selection instead of requiring the removed `hologram.node` name.
- Added pinned Node.js 22 tooling to the devcontainer so the native and
  TypeScript SDK release gates run in the repository's prescribed environment.

## [0.13.0] - 2026-09-05

### Added
- `.holo` **format v4** (specs/refactor/03 §v4): `LayerKind::InferenceModel = 4`
  appended to the closed layer-kind set — an engine-agnostic AI-model layer
  (mandatory engine tag in `aux`, mandatory service name in `entry`, no exit
  code). `Layer::inference_model(content, entry, engine)` constructor; new
  `ManifestError` variants `EmptyLayerEntry` / `MissingEngineTag` /
  `DuplicateLayerEntry`; non-empty layer entry names must now be unique within a
  manifest. Writers emit v4; v2/v3 archives remain loadable.
- `hologram ai` CLI group (download / compile / inspect / infer) behind the new
  `ai` cargo feature on hologram-cli (default off), delegating to the sibling
  `hologram-ai` crate — hologram stays engine-agnostic.
- `hologram_ai_*` C ABI surface behind the new `ai` cargo feature on
  hologram-ffi (default off): compile / download / app load / model listing /
  JSON session invoke, with the `HOLOGRAM_ERROR_AI_*` error band (100–111) and
  `ai-*` FEATURES probes.

### Changed
- Raised the workspace, target-specific crates, and language SDKs to `0.13.0`,
  with exact internal crate version edges and a dependency-ordered crates.io
  release plan.
- Raised the declared Rust MSRV from 1.85 to 1.94, matching Wasmtime 47's
  supported compiler floor and the APIs already used by the compute backend.
- Upgraded the native Wasm runtime to Wasmtime 47.0.4 and the SWC parser family
  to the maintained 26/29/45 release line.

### Fixed
- Made crates.io publishing fail closed when credentials are missing and wait
  for each exact dependency version to become downloadable before publishing
  its dependents.
- Updated Rust 1.98 compatibility and public-API snapshots without changing the
  Holo/1 `.holo` v4 archive bytes.

## [0.12.1] - 2026-07-20

### Added
- 

### Changed
- 

### Fixed
- 

## [0.12.0] - 2026-07-20

### Added
- 

### Changed
- 

### Fixed
- 
