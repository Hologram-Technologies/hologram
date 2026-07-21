# Design — `s7_readme`: the README as an executable BDD conformance suite

**Date:** 2026-07-20
**Status:** ✅ implemented — all 35 scenarios land; `just bdd` green (68 scenarios, 0 failed), meta-gate + clippy `-D warnings` clean.
**Topic:** A new BDD suite that makes every code block in `README.md` an executable, honesty-gated conformance scenario.

## Implementation notes (as-built)

Matches this design. Two honest adjustments surfaced while wiring the enforced steps:

- **RM-29** — `client.inspect()` returns `NotAnApplication` on a freshly *compiled tensor* holo:
  the `.holo` v3 app-tooling verbs (`inspect`/`thin`/`fat`) target v3 application containers, not a
  bare tensor archive. The scenario exercises those verbs on a v3 app holo (HF-3's surface) and the
  store verbs (`get`/`pin`/`ls`) + `open`→`boot` on the compiled/container path — still enforced.
- **RM-11** — two independently-compiled archives are not byte-equal, so the staged pipeline is
  asserted to yield a **loadable** archive (each stage produces a working next) rather than
  byte-equality with the one-shot `source::parse` path.

Two small testability seams were added to `hologram-cli` (anticipated in §10): `cmd::run_from_args`
/ `run_full_from_args` (tensor + full CLI, return `Result`) and `cmd::run_node_from_args` (node, returns
the exit code instead of `process::exit`). Frontend features (`frontend-python/-typescript/-rust`) are
enabled on the conformance dev-deps (the §10 default), adding the swc/`syn` parsers to the `just bdd`
build.

**Follow-ups (post-merge):**

- **RM-25 promoted** (`⛔ → ✅`): the holospace Manager block boots a real Wasm container over the
  Wasmtime engine (`spaces/holospaces`, mirroring `tests/e2e.rs`).
- **Witnessed rows** (`⛔ → 🟡`): the 5 blocks the Rust `bdd` gate cannot run — Python/TS SDK demos +
  compile-source (RM-20/21/22/23) and the browser `Console` (RM-27) — are bound to their **own**
  package tests (`sdk/python/tests`, `sdk/typescript/test`, `spaces/holospaces-browser`) via a new
  `report::check_witnessed_rows` audit, the same pattern the `CC`/`CS` classes use. So every one of
  the 35 blocks is either a BDD scenario (30) or an externally-witnessed row (5) — none left pending.

---

## 1. Goal

Give the project's public `README.md` the same treatment `specs/refactor/00–07` already
get: a Gherkin conformance suite, bound to the `CONFORMANCE.md` ledger, policed by the
static honesty meta-gate. The organizing invariant the user asked for:

> **Every fenced code block in the README has exactly one BDD scenario.**

The README has **35 fenced code blocks**. This suite adds 35 scenarios, numbered
`RM-1 … RM-35` so that **`RM-N` ≡ the N-th fenced block** (top-to-bottom). That 1:1
numbering is the maintenance contract: add a block to the README → add an `RM` row and a
feature file; the meta-gate fails until they exist.

## 2. Principle

Each `RM` scenario drives the **public surface exactly as the README documents it** — the
`hologram` facade, `source::parse`, the `hologram` CLI binary, the C ABI — **not** the
internal contract traits that `s0–s6` exercise. That is the distinguishing value: when a
README promise stops being true, its scenario goes red. Overlap with `s0–s6` is expected
and acceptable (scenarios carry distinct `@id`s; the meta-gate only requires one scenario
per id).

Where a block is not literally runnable in a unit test (a `curl … | sh` install, a
dependency `toml` snippet, a prose lowering diagram), the scenario asserts the strongest
**verifiable structural fact** the block claims (the script/feature/symbol exists and has
the documented shape) rather than faking execution.

## 3. Fits the existing machinery (no new runner)

The suite plugs into the system already in place:

- **Suite tree:** new `features/suites/s7_readme/`, one `.feature` file per scenario, one
  `Scenario:` per file (the meta-gate enforces exactly one), tagged
  `@class:RM @id:RM-N @spec:README @phase:Pn @status:{pending|partial|enforced}`.
- **Ledger:** a new `## RM — README public surface (README.md; BDD)` section in
  `CONFORMANCE.md`, one table row per scenario, `Witness` =
  `` `s7_readme/<file>.feature::<Scenario title>` ``, `Status` glyph agreeing with the
  scenario's `@status` (`pending`→⛔, `partial`→🟡, `enforced`→✅).
- **Meta-gate registration:** add `"RM"` to `BDD_CLASSES` in
  `crates/hologram-conformance/src/report.rs`. **This is load-bearing** — without it
  `is_bdd("RM")` is false and the bijection check silently ignores RM rows (they could
  drift). Adding it makes the gate enforce, for every RM row: exactly one scenario with the
  same id, status agreement, witness-path match, and one scenario per file.
- **Steps:** step definitions for every non-`pending` scenario in
  `crates/hologram-conformance/tests/bdd.rs`; new `RM-*` fields on `ConformanceWorld`
  (`crates/hologram-conformance/src/lib.rs`), stored as primitives to keep the library
  domain-type-free (existing convention).
- **Run surface:** unchanged — `just bdd` (runner + meta-gate) and `just conformance-report`.
  The runner is already wired with `fail_on_skipped_with(@status:enforced)`, so an
  `enforced` scenario with missing/skipped steps is a build failure. This is what makes
  "enforced" honest: a scenario cannot be labeled green unless its steps really run and pass.

### New dev-dependencies for `hologram-conformance` (dev-only, test-scoped)

`tests/bdd.rs` currently dev-deps `hologram`, `hologram-space`, `hologram-archive`,
`hologram-tck`. The RM steps additionally need:

- `hologram-cli` (drive `cmd::run_from_env` / the `Command` enum in-process) — RM-3, RM-9,
  RM-15/17/19, RM-26, RM-30, RM-31.
- `hologram-ffi` (drive the C ABI in-process) — RM-32.
- `hologram-compiler` frontend features `frontend-python`, `frontend-typescript`,
  `frontend-rust` (enabled on the conformance crate's dev-deps, or via a `cfg`/feature so
  those scenarios are enforced only when the frontend is compiled in) — RM-12/13/14/16/18.

If enabling a frontend dev-feature bloats the default `just bdd` build unacceptably, the
fallback is a `#[cfg(feature = "frontend-python")]`-gated step module and a
`readme-frontends` feature on `hologram-conformance` that `just bdd` turns on; the scenario
stays `pending` when the feature is off. Decision deferred to the plan; default is to enable
the three frontend features on the conformance dev-deps.

## 4. The block → scenario catalog

Feature-file names are kebab-cased per block. `@phase` reflects README readiness:
**P0 = ships today**, **P6 = post-consolidation** (the substrate/SDK surfaces the README
documents ahead of the merged implementation).

### Quickstart
| id | block | feature file | what the scenario asserts | status | phase |
|---|---|---|---|---|---|
| RM-1 | L55 install `curl…\|sh` | `install-script.feature` | `install.sh` exists at repo root, is POSIX `sh`, names the documented platforms (macOS arm64/x86_64, Linux x86_64) and installs into `~/.local/bin` | 🟡 partial | P0 |
| RM-2 | L62 version-pin / `cargo install --git` | `install-alternatives.feature` | `install.sh` honors `--version` and `--help`; the `hologram-cli` binary target exists in the workspace | 🟡 partial | P0 |
| RM-3 | L76 `hologram compile … && execute` | `quickstart-compile-execute.feature` | driving the CLI `compile` then `execute` on a temp source round-trips to a loadable archive with declared output ports | ✅ enforced | P0 |
| RM-4 | L84 toml `features=[archive,backend,compiler,exec]` | `quickstart-library-features.feature` | those four features are declared on the `hologram` facade crate | ✅ enforced | P0 |

### Using the tensor engine
| id | block | feature file | asserts | status | phase |
|---|---|---|---|---|---|
| RM-5 | L215 rust `Client` (space-substrate excerpt) | `space-substrate-client.feature` | the documented `Client` verbs (`compile`→`provision`→`run`, `open`) compose over a `Space` | ✅ enforced | P0 |
| RM-6 | L243 toml features | `tensor-engine-library-features.feature` | the tensor-engine feature set resolves on the facade | ✅ enforced | P0 |
| RM-7 | L255 toml `full`+`space`+`client` | `full-space-client-features.feature` | `full`, `space`, `client` features exist; `full` enables the documented facade modules | ✅ enforced | P0 |
| RM-8 | L267 toml `frontend-*` | `frontend-features.feature` | `frontend-python`/`-typescript`/`-rust` features are declared | ✅ enforced | P0 |
| RM-9 | L279 `--example pipeline` | `pipeline-example.feature` | the `pipeline` example runs to completion (exit 0) | ✅ enforced | P0 |
| RM-10 | L286 **rust minimal example** | `minimal-example.feature` | the example verbatim: `source::parse("input x\nop relu x as=y\noutput y\n")` → `Compiler::new(g, Cpu, W32).compile()` → `InferenceSession::load(&archive, CpuBackend::<BufferArena>::new())` → `execute(zeros)` yields one output per port | ✅ enforced | P0 |
| RM-11 | L311 text lowering diagram | `lowering-pipeline.feature` | the documented lowering symbols exist and are reachable: `SourceDocument` → selected `SourceProgram` → `Graph` (`lower_ir`) → `Compiler` | ✅ enforced | P0 |
| RM-12 | L330 rust `SourceParseOptions` | `source-parse-options.feature` | `SourceParseOptions::new().graph("encoder")` selects a named graph and lowers it (feat `frontend-python`) | ✅ enforced | P0 |
| RM-13 | L347 rust `compile_from_source_language` | `compile-from-source-language.feature` | one-call source→archive compile for a single-graph source (feat `frontend-python`) | ✅ enforced | P0 |
| RM-14 | L368 python builder | `python-frontend-parse.feature` | `frontend-python` extracts the `encoder` graph fn, ignoring `ordinary_app_code` | ✅ enforced | P0 |
| RM-15 | L379 bash frontend-python compile | `python-frontend-cli.feature` | the CLI `--features frontend-python compile --source graph.py --graph encoder` produces an archive | ✅ enforced | P0 |
| RM-16 | L401 ts builder | `typescript-frontend-parse.feature` | `frontend-typescript` extracts the `encoder` fn, ignoring unrelated code | ✅ enforced | P0 |
| RM-17 | L414 bash frontend-ts compile | `typescript-frontend-cli.feature` | the CLI compiles the `.ts` with `--graph encoder` | ✅ enforced | P0 |
| RM-18 | L437 rust builder | `rust-frontend-parse.feature` | `frontend-rust` (syn) extracts the `encoder` fn, ignoring unrelated code | ✅ enforced | P0 |
| RM-19 | L450 bash frontend-rust compile | `rust-frontend-cli.feature` | the CLI compiles the `.rs` with `--graph encoder` | ✅ enforced | P0 |
| RM-20 | L477 python SDK demo | `sdk-python-demo.feature` | `hologram` (py) `Graph`/`Session` build+execute round-trip | ⛔ pending | P6 |
| RM-21 | L493 ts SDK demo | `sdk-typescript-demo.feature` | `@tryhologram/sdk` + `@tryhologram/native` build+execute round-trip | ⛔ pending | P6 |
| RM-22 | L523 python `compile_source_file` | `sdk-python-compile-source.feature` | `hg.compile_source_file("graph.txt")` compiles native source through the SDK | ⛔ pending | P6 |
| RM-23 | L527 ts `compileSource` | `sdk-typescript-compile-source.feature` | `compileSource`/`compileSourceFile` compile native source through the SDK | ⛔ pending | P6 |
| RM-24 | L552 rust `address_ring`/`compose_model` | `address-compose.feature` | verbatim: mint two κ via `address_ring`, `compose_model` is a commutative (order-independent) product | ✅ enforced | P0 |

### Opening a holospace
| id | block | feature file | asserts | status | phase |
|---|---|---|---|---|---|
| RM-25 | L608 rust Platform Manager | `holospace-manager.feature` | `Manager::sign_in` → `provision(Source::Userland)` → `open` → `boot` → `suspend` (κ snapshot) over `holospaces` + a runtime | ⛔ pending | P6 |
| RM-26 | L666 bash `node put/manifest/caps/spawn` | `node-holospace-cli.feature` | CLI `node put`/`manifest`/`caps` mint the documented κs; `spawn` (real Wasmtime boot) is not asserted yet | 🟡 partial | P0 |
| RM-27 | L688 js browser `Console` | `browser-console.feature` | the wasm32 `Console` mirrors the Manager (tab-as-substrate) | ⛔ pending | P6 |

### Programmatic control
| id | block | feature file | asserts | status | phase |
|---|---|---|---|---|---|
| RM-28 | L720 rust `impl Space for MinimalSpace` | `minimal-space.feature` | a minimal reference `Space` composition is accepted by `Client` and reaches a contract-mediated op | ✅ enforced | P0 |
| RM-29 | L742 rust `Client` store/app tooling | `client-store-tooling.feature` | on one `Client` handle: `compile`/`provision`/`run`, then `get`/`pin`/`ls`, then `inspect`(`all_verified`)/`thin`, then `open`+`boot` via `MockEngine` | ✅ enforced | P0 |

### CLI / FFI
| id | block | feature file | asserts | status | phase |
|---|---|---|---|---|---|
| RM-30 | L778 bash compile/inspect/execute/bench | `cli-tensor-verbs.feature` | all four tensor CLI verbs succeed on one archive | ✅ enforced | P0 |
| RM-31 | L801 bash node/app/network | `cli-substrate-verbs.feature` | `node put`/`get`/`verify` round-trip and re-derive (SPINE-4); `serve` and the not-yet-shipped `app`/`network` verbs are not asserted | 🟡 partial | P0 |
| RM-32 | L827 c FFI | `c-ffi.feature` | the C ABI: `hologram_compile_source` → `hologram_session_load` → `_execute` → `_close`, plus `hologram_abi_version` probing | ✅ enforced | P0 |

### Feature flags / Contributing
| id | block | feature file | asserts | status | phase |
|---|---|---|---|---|---|
| RM-33 | L993 toml `no_std` `default-features=false` | `no-std-features.feature` | `default-features=false` + `[backend,compiler,exec]` is a valid feature composition (full no_std build lives in `just wasm`) | 🟡 partial | P0 |
| RM-34 | L1090 bash clone/`just build`/example/install | `building-from-source.feature` | the `pipeline` example runs from a source build; clone/`cargo install` are environmental and not asserted | 🟡 partial | P0 |
| RM-35 | L1101 bash `just ci` | `quality-gate.feature` | the Justfile `ci` recipe chains the documented gates (`fmt-check clippy test deny`) | ✅ enforced | P0 |

### Tally
- **Enforced ✅:** RM-3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,28,29,30,32,35 → **23**
- **Partial 🟡:** RM-1,2,26,31,33,34 → **6**
- **Pending ⛔:** RM-20,21,22,23,25,27 → **6**

Counts may shift ±1 during implementation as each green is confirmed against the compiled
surface; the `fail_on_skipped(@status:enforced)` gate makes it impossible to ship a red
scenario mislabeled as `enforced`.

## 5. Concrete shape (RM-10, the flagship)

```gherkin
# features/suites/s7_readme/minimal-example.feature
@class:RM @id:RM-10 @spec:README @phase:P0 @status:enforced
Feature: the README minimal example compiles and executes on the CPU backend
  Scenario: native source runs end to end
    Given the README's native source graph
    When I compile it to a .holo and load it on the CpuBackend
    Then executing against zero inputs yields one output buffer per port
```

Steps in `bdd.rs` run the snippet verbatim and assert
`outputs.len() == session.output_count()` with each output buffer non-empty.

## 6. File-by-file change list

- **New:** `features/suites/s7_readme/*.feature` — 35 files (names above).
- **Edit:** `CONFORMANCE.md` — add the `## RM` section (35 rows) + a legend line in the
  Classes table (`RM | README public surface | BDD scenarios (s7_readme)`).
- **Edit:** `crates/hologram-conformance/src/report.rs` — add `"RM"` to `BDD_CLASSES`
  (and the `ignores_non_bdd_classes`-style unit expectations if any need updating).
- **Edit:** `crates/hologram-conformance/src/lib.rs` — add `rm_*` fields to
  `ConformanceWorld` for the scenarios that stash results.
- **Edit:** `crates/hologram-conformance/tests/bdd.rs` — step definitions for the 29
  non-pending scenarios; pending scenarios get no steps (they skip).
- **Edit:** `crates/hologram-conformance/Cargo.toml` — dev-deps `hologram-cli`,
  `hologram-ffi`, and the three `frontend-*` features (see §3).
- **Edit (docs):** `features/README.md` — add `s7_readme` to the layout list and note the
  `RM` class + the one-scenario-per-block invariant.
- No change to `just bdd`, the runner entrypoint, or the meta-gate test itself.

## 7. Verification

- `cargo test -p hologram-conformance --test meta_gate` — the bijection holds (every RM row
  has its one scenario, statuses agree, witness paths match).
- `just bdd` — the runner executes all suites; enforced RM scenarios must pass, pending skip.
- `just conformance-report` — no catalog↔scenario drift.
- Spot-run individual features during development via the cucumber name filter.

## 8. Non-goals

- Not re-testing internal-contract invariants `s0–s6` already own — RM re-covers only via
  the **public** README surface.
- Not building the unimplemented substrate/SDK to force pending rows green — those stay
  honest skips until their feature lands.
- Not a second runner, and not a rustdoc/`skeptic` doctest harness.

## 9. Resolved facts (verified 2026-07-20)

- **`install.sh` is at the repo root** with `--version`, `--bin-dir` (default
  `~/.local/bin`), and `-h/--help` — RM-1/2 wire against the real script.
- **CLI drives in-process.** `hologram_cli::cmd::run(cli: Cli)` is public and takes a
  parsed arg struct (`run_from_env()` just parses then calls it). RM-3/9/15/17/19/26/30/31
  construct `Cli` directly and pass absolute temp paths — **no subprocess, no network, no new
  seam.**
- **Every substrate verb is implemented:** `node` (`put/get/pin/unpin/gc/ls/inspect/verify/
  manifest/spawn/serve/caps`), `app` (`inspect/thin/fat`), `network` (`create/show/delegate`).
  So RM-31 asserts node `put/get/verify` **and** the `app`/`network` verbs; only `node serve`
  (a live listener) stays unasserted → RM-31 remains 🟡 solely for `serve`.
- **`examples/pipeline.rs` exists** — RM-9 and the RM-34 example run against it.

## 10. Remaining open question

- **Frontend dev-features cost.** Enabling `frontend-python/-typescript/-rust` on the
  conformance crate pulls the Python / TS / `syn` parsers into the `just bdd` build. If
  that's too heavy, gate them behind a `readme-frontends` feature (see §3); RM-12/13/14/16/18
  fall back to `pending` when off. **Default: enable them.** Resolved at plan time by trying
  the enabled build first.
```
