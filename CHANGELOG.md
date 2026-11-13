# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-gigabyte content-addressed blobs, end to end:
  - `hologram_space::piece`: a 1 MiB **piece table** cut from BLAKE3's own hash tree. Its root
    is the blob's ordinary `blake3:` κ, so a table from any untrusted source is checked against
    the κ alone, then every piece and every byte range verifies independently
    (`PieceTree::{verify_root, verify_piece, verify_range}`), 32 bytes per MiB. Also
    `StreamVerifier` (confirm a stream against a κ without holding it) and
    `PieceTreeBuilder`. Piece hashing uses blake3's multi-lane SIMD path (no pin change).
  - `KappaStore::{content_len, get_range, put_writer}` with default implementations, so every
    backend keeps compiling; `put_writer` with an expected κ stores nothing unless the bytes
    match. New `StoreError::{OutOfRange, KappaMismatch}`. The TCK gains `large_blob_surface`,
    run by every backend.
  - `hologram-store` feature `file`: `FileKappaStore`, one file per κ, with atomic writes
    (temp file, fsync, rename), ranged reads on cached handles, cached piece tables, scrub,
    pins and reachability GC, and `PartialBlob`: resumable out-of-order downloads that verify
    every piece before accepting it, write in order, and re-verify what is on disk after a
    restart.
  - HTTP-CAS server (`hologram_net::http::live::serve_with`): `HEAD`, one `Range` (206/416),
    the κ as `ETag` (`If-None-Match`/`If-Range`), `Content-Type`, `Accept-Ranges`, CORS
    (including preflight), keep-alive, one thread per connection up to a cap, bodies streamed
    from the store 1 MiB at a time, and `/cas/{κ}/pieces` serving the piece table.
  - `hologram_net::http::fetch`: parallel multi-source fetch over keep-alive `Range` requests
    that verifies each piece before the sink sees it, drops a source that sends one bad piece,
    retries elsewhere, and resumes from a sink's record. `hologram_net::http::client::HttpConn`
    is the keep-alive client it uses.
- `scripts/publish-crates.sh` takes `CRATES="…"` (and `publish-crates.yml` a `crates` input) to
  publish a dependency-closed subset, e.g. the runner closure `hologram-types hologram-space
  hologram-tck hologram-store hologram-net`.
- `just large-blob-bench` runs the 1 GiB throughput examples.

### Changed
- The `std` feature of `hologram-space` now enables `blake3/std` (runtime AVX2/AVX-512
  detection for every BLAKE3 call in the process, not only compile-time features).
- The HTTP-CAS gateway serves connections concurrently with keep-alive instead of one
  connection at a time with `Connection: close`.

## [0.14.0] - 2026-11-13

### Security
- Upgraded the native Wasm runtime from Wasmtime 47.0.4 to 48.0.8, the lowest release
  patched for all 21 RustSec advisories against 47.0.4 (RUSTSEC-2026-0315, -0316, -0325,
  -0326, -0327, -0353 to -0359, -0361 to -0364, -0378, -0379, -0422 to -0424), and the
  Layer Assembler's Zstandard decoder from ruzstd 0.7.3 to 0.9.1 (RUSTSEC-2026-0344,
  decompression bomb).

### Added
- Cross-substrate kernel conformance digest (`hologram_compute::cpu::conformance`) and
  its wasm32 driver (`crates/hologram-compute/conformance-wasm`, workspace-excluded):
  `scripts/wasm-conformance.sh` (`just wasm-conformance`, CI job `wasm-conformance`) runs
  one digest body natively and on wasm32 baseline, SIMD128 and relaxed-SIMD, and fails
  unless the f32 dot/sum digests are equal. The `matmul_f32_blocked` digest is reported
  per substrate and does not gate, because it differs (KC-7).
- **First crates.io release.** Seven crates are published, in dependency order:
  `hologram-types`, `hologram-ops`, `hologram-compute`, `hologram-graph`,
  `hologram-archive`, `hologram-compiler`, `hologram-exec`. `cargo metadata`
  confirms the set is closed: every dependency of these seven is either another
  of the seven or a crates.io crate. Each carries the packaging metadata
  crates.io reads — `description`, `license`, `readme`, `repository`,
  `documentation`, `keywords`, `categories`.
- The remaining twelve workspace crates are marked `publish = false` for this
  release and stay unpublished: `hologram-cli`, `hologram-ffi`, `hologram-bench`,
  `hologram-net`, `hologram-store`, `hologram-tck`, `hologram-emulator`,
  `hologram-runtime`, `hologram-space`, `holospaces`, `holospaces-node`, and the
  `uor-hologram` facade. (`hologram-conformance` was already `publish = false`.)
  Nothing about this is one-way: publishing any of them in a later release is
  purely additive — it adds a name to crates.io and removes nothing. The reverse
  is not true, which is why the first release is the smallest one that works.

### Changed
- Raised the declared Rust MSRV from 1.94 to 1.95, Wasmtime 48's supported compiler floor.
- `simd_f32_dot` and `simd_f32_sum` now use one canonical association on every target
  (sixteen lanes over blocks of sixteen, a halving tree, a scalar tail; no FMA), with
  AVX2, SSE2, NEON and SIMD128 twins that keep the same lanes in registers, replacing the
  per-instruction-set kernels that grouped partial sums by lane width. Their results are
  now bit-identical across x86-64, aarch64 and wasm32, and change in the last bits
  relative to 0.13.1 on every target. Attention scores, norm statistics and ReduceSum
  inherit the order; the pinned cross-substrate golden digest moves to
  `76408ae96e679411`. Cost against 0.13.1 on an AVX-512 host: dot about 1.2x slower
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
- `hologram-archive` now declares its `hologram-space` dev-dependency by path only,
  without a version requirement. Cargo strips a path-only dev-dependency from the
  published manifest; the workspace entry's `=<version>` pin would instead make
  crates.io demand a `hologram-space` that is not published.

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
