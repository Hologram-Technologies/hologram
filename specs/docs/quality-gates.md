# Quality gates — the machine-enforced maintenance contract

Hologram's correctness-and-quality posture (the V&V discipline) is enforced by
GitHub Actions, not by convention. Every gate below is a **required status
check** on `main`: a PR cannot merge unless all pass. Together they guarantee
that what lands is correct, doesn't silently regress performance, keeps a
truthful public-API surface, and upholds the **versioning and rollback
contract** consumers depend on.

This document is the catalog of gates and the contracts they enforce. Each gate
is reproducible locally (a `scripts/*-local.sh` helper or a one-line `cargo`
command) so failures never require pushing to discover.

## Why machine-enforced

A release is a promise: a consumer pinning `tag = "v0.5.0"` must be able to
trust that `v0.5.x` won't break them, that the published numbers reflect the
code, and that a regression can't slip in unnoticed. Humans miss these in
review. The gates make the promise mechanical — the same way the V&V suites
make correctness mechanical rather than aspirational.

## The gate catalog

| Gate | Workflow | Fails when | Local repro |
|---|---|---|---|
| **Format** | `ci.yml` | `cargo fmt --check` finds drift | `cargo fmt --all` |
| **Clippy** | `ci.yml` | any lint (`-D warnings`) | `cargo clippy --workspace --all-targets -- -D warnings` |
| **Tests** (Linux/macOS/Windows) | `ci.yml` | any unit/integration/doc test fails | `cargo test --workspace` |
| **Docs** | `ci.yml` | `cargo doc` warns (`-D warnings`) | `cargo doc --workspace --no-deps --lib` |
| **Security audit** | `ci.yml` | `cargo audit` finds an advisory | `cargo audit` |
| **Cross / no_std** | `ci.yml` | the lib stack fails to build for `aarch64` or `wasm32` (no_std) | `cargo check --target wasm32-unknown-unknown -p … --no-default-features` |
| **V&V suites** | `ci.yml` | a release-mode V&V class (PV/FU/WS/…) fails | `cargo test --release -p … --features …` |
| **Performance regression** | `perf-gate.yml` | a benchmark median regresses past the threshold *and* outside noise | `scripts/perf-gate-local.sh` |
| **Benchmark lifecycle** | `perf-gate.yml` | a tracked benchmark disappears without being deprecated + versioned out | `scripts/perf-gate-local.sh` |
| **Public-API snapshot** | `api-gate.yml` | the committed API snapshot is stale vs the real surface | `scripts/update-api-snapshots.sh` |
| **Semver compliance** | `semver-gate.yml` | a public-API change isn't covered by an adequate version bump | `scripts/semver-gate-local.sh` |

`ci.yml` gates pre-existed; this document also adds the four perf/API/semver
gates below and ties them into one contract.

---

## Performance regression gate

A PR may not regress benchmarked performance relative to the branch it targets.

`perf-gate.yml` benchmarks the **PR merge result** and the **target-branch tip**
back-to-back on the *same runner* (controlling for machine variance — comparing
against numbers from a different VM is too noisy), aggregates each with
`scripts/aggregate-benchmarks.py`, and runs `scripts/compare-benchmarks.py` as a
gating step.

A regression must clear **two** bars, so CI jitter never blocks an honest PR:

1. **Relative**: `pr_median > base_median × (1 + threshold)` — default **10%**.
2. **Noise**: the slowdown also exceeds `noise-sigmas × √(base_std² + pr_std²)` —
   default **2σ**, i.e. it's outside the measured jitter.

Both are env-tunable (`REGRESSION_THRESHOLD`, `NOISE_SIGMAS`). `main` publishes
its post-merge numbers separately via `benchmarks.yml`.

## Benchmark lifecycle (deprecation gate)

The regression gate is only sound if benchmarks can't silently vanish — a
removed benchmark is exactly how a regression (or a whole feature's cost) would
escape the gate. So removal is itself gated.

`benches/manifest.toml` is the registry of every tracked benchmark and its
status:

```toml
[benchmarks."matmul/256"]
status = "active"

[benchmarks."legacy_path/foo"]
status = "deprecated"
deprecated_since = "0.6.0"   # version that deprecated it
```

`compare-benchmarks.py` cross-checks the manifest against the benchmarks present:

- A benchmark in the **baseline but absent from the PR** (a removal) **fails the
  gate** — *unless* the manifest marks it `deprecated`. A benchmark can only
  fall off once it has been **properly deprecated and versioned out**: mark it
  `deprecated` (with `deprecated_since`) in one release, then remove it (and its
  manifest entry) in a later one. This blocks both accidental drops and
  gate-evasion while permitting legitimate lifecycle removal.
- A benchmark **present but missing from the manifest** fails too — new
  benchmarks must be registered (`status = "active"`), so the registry stays the
  source of truth.

This mirrors the API deprecation lifecycle below: nothing tracked disappears
without first being deprecated and versioned out.

## Public-API snapshot gate

The public API of every workspace library crate is **snapshotted to a committed
text file** under `api/<crate>.txt` (generated by
[`cargo-public-api`](https://github.com/cargo-public-api/cargo-public-api)).
`api-gate.yml` regenerates the snapshots and fails if any differs from what's
committed.

Effect: **every change to the public API is an explicit, reviewable diff** in
the PR — a reviewer sees exactly which items were added, changed, or removed,
and an unintended API change can't merge unnoticed. Updating the snapshot is one
command (`scripts/update-api-snapshots.sh`), so the workflow is: change the API,
regenerate, commit the snapshot diff alongside the code.

The snapshot is the human-readable record; the semver gate (next) enforces that
the *version* moves to match.

### The four API-change scenarios

Every public-API change is one of four kinds; the gates handle each, and the
release tooling records each across versions:

| Scenario | Snapshot diff | Semver (0.x) | Recorded as | Lifecycle rule |
|---|---|---|---|---|
| **Add** an item | line added | patch bump | `Added` | register freely |
| **Update** an item (signature) | line changed | minor bump (breaking) | `Changed` | allowed; it's breaking |
| **Deprecate** an item | `#[deprecated]` marker appears | patch bump | `Deprecated` | the *required first step* before removal |
| **Remove** an item | line removed | minor bump (breaking) | `Removed` | only after it was `Deprecated` in a prior release |

The deprecation lifecycle (mirrors the benchmark lifecycle above): you may not
add *and* remove, or remove an undeprecated item — removal is only sound once
consumers have had a deprecated release to migrate off. The snapshot diff makes
the kind explicit; the semver gate forces the matching bump.

### API history across versions (release tooling)

A single current snapshot answers "what is the API now"; the **history**
answers "when did each item arrive, change, or leave". The release tooling
maintains both:

- `api/<crate>.txt` — the *current* snapshot (gated every PR, above).
- `api/history/<version>/<crate>.txt` — the snapshot *archived at each release*.
- `api/CHANGELOG.md` — a per-version, categorized record (Added / Changed /
  Deprecated / Removed), generated by `scripts/api-changelog.py` diffing the new
  snapshots against the previous release's archive.

At release (`version-bump.yml`), the tooling regenerates the snapshots, diffs
them against the last release's archive, prepends a `## vX.Y.Z` section to
`api/CHANGELOG.md`, and archives the new snapshots under `api/history/vX.Y.Z/`.
The changelog is committed with the version bump, so the API's evolution is
queryable in-tree, version by version — the durable record behind the rollback
contract.

## Semver compliance gate

The published version is the **rollback anchor**: consumers pin
`tag = "vX.Y.Z"` and trust the semver contract within a release line.
`semver-gate.yml` runs [`cargo-semver-checks`](https://github.com/obi1kenobi/cargo-semver-checks),
which builds the public API of each library crate for both the PR and the target
branch, classifies the diff, and **fails if the version bump is insufficient**
for the change.

Under the **0.x rules** Cargo applies (hologram is `0.y.z`):

- **Breaking** change (removed/changed public item) ⇒ requires a **minor** bump
  (`0.y` → `0.(y+1)`).
- **Additive** change (new public item) ⇒ requires a **patch** bump (`0.y.z` →
  `0.y.(z+1)`).

A PR that breaks the API while leaving the version untouched fails: the version
would be lying about compatibility. Bump with `cargo set-version --workspace …`
(all crates share one workspace version) in the same PR.

---

## Versioning, deprecation, and rollback contract

- **One workspace version.** All crates inherit `[workspace.package] version`;
  they move together. The release tag `vX.Y.Z` must equal it (`publish.yml`
  verifies this).
- **Rollback.** Because consumers pin a tag, any `vX.Y.Z` remains fetchable
  forever; downgrading is "pin the older tag." The semver gate guarantees a
  given line (`vX.Y.*`) never received a breaking change, so a rollback within a
  line is always safe.
- **Deprecation lifecycle** (APIs *and* benchmarks):
  1. **Deprecate** — mark the item `#[deprecated]` (or the benchmark
     `status = "deprecated"`), bump the version. It still exists; nothing
     downstream breaks.
  2. **Version out** — in a later release, remove it. That's a breaking change,
     so the semver gate requires the minor bump, the API-snapshot diff records
     the removal, and the benchmark-lifecycle gate permits the now-deprecated
     benchmark to fall off.

  Nothing tracked is ever removed in a single step at an unchanged version.

## Enforcement

Each gate's job name must be added as a **required status check**
(Settings → Branches → `main` → branch protection). The gates *run and fail* on
their own; branch protection is what turns a failure into a merge block.
