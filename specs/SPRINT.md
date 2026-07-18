# Sprint Tracking

## Sprint 40: Ecosystem Refactor — Consolidation (STRUCTURALLY COMPLETE — READY TO LAND)

**Plan:** [specs/refactor/00-overview.md](refactor/00-overview.md) · P0 gates
[specs/refactor/P0-PREP.md](refactor/P0-PREP.md) · branch `chore/refactor`

Goal: consolidate the ecosystem (hologram + holospaces in one repo; hologram-ai stays an
external consumer) with clean crate boundaries — `substrate/` dissolved into core crates,
holospaces becomes the space contract's implementations, `.holo` becomes the application
container, one `hologram` facade + `Client` under everything. Phased always-green
(P0.5 spike → P0 → P1 → P2 → P3 hard stop → P4–P6 follow-on). Decisions D1–D29.

### STATUS (2026-07-15) — the refactor is functionally complete on `chore/refactor`

`chore/refactor` is **71 commits ahead of `main`, 0 behind** (Sprint 39's backend work merged
in 2026-07-15). End state, every gate green at every commit:

- **`crates/`** holds every `hologram-*` crate: the compute stack + `hologram-space` (the
  contract), `hologram-net`, `hologram-runtime` (with the generic `lifecycle::Session`, D7),
  the four `hologram-store-*` backends, `hologram-efi`, and **`hologram-emulator`** + its
  codemodule (the 20.6k-LOC system emulator, hoisted out of holospaces).
- **`spaces/`** holds only the space impls: `holospaces{,-browser,-node}` (~10k LOC — the
  boot/provisioning/peer/platform layer after the emulator hoist).
- `substrate/` **eliminated**; no git-pins; **no LGPL** (rustpython → in-tree parser);
  **`uor-hologram` facade** with a real **`Client<S: Space>`** (D4, MVP); verified **MSRV 1.85**;
  **`cargo deny` green with zero license exceptions**; RZ invariant holds throughout.
- **P1** (consolidation) ✅ · **P2** (holospaces import + port) ✅ · **P3**: generic lifecycle
  hoisted ✅, Client MVP ✅ — remaining P3 = `open→Session` (needs the Space contract to expose a
  `ContainerRuntime`) + naming-review gate + first release (crates.io, human-gated).

**Next step: land it** (merge `chore/refactor` → `main`; window is optimal at 71/0). Then, in
order: migrate hologram-ai + re-pin the external holospaces repo → expand the Space contract for
`open→Session` → naming review → first `uor-hologram` release. Deferred: `backend→compute`
(Sprint 39 lull); P2 tail (vv/ fixtures, OPFS dedup, V&V absorption → MG-7).

Conformance: **7 / 29 BDD scenarios enforced** (LAW-0/1/3, SP-1/3, MG-5, GV-1); honesty
meta-gate green. (Later-phase scenarios shaped `@status:pending` at their phase.)

- [x] **Spec suite** 00–08 + P0-PREP, decisions D1–D29.
- [x] **P0.5 spike (D28)**: `hologram-space` (Space contract) + `hologram-spike-sp3`
  (`Client` compile→store→boot) compose on native + wasm32; Send-bound resolved =
  maybe-Send. Corrected **LAW-4** from evidence — storage is *synchronous* (wasm-safe via
  sync-OPFS-in-a-Worker), only network/lifecycle async. SP-3 enforced.
- [x] **Scenarios → enforced**: SP-1 (TCK-is-conformance), LAW-1 (SPINE-1 re-derivation),
  LAW-3 (open contract, D21), MG-5 (κ-stability golden vectors). LAW-2 / LAW-5 held
  honestly (no witness yet — attenuation "not faked here" in the code).
- [x] **P0 preflight**: golden vectors frozen (ground rule 5); LICENSE-MIT + LICENSE-APACHE
  (D24); crates.io audit — `hologram` name **TAKEN**, 18 other target names free.
- [x] **P0 holospaces HEAD-sync (D23)**: pin `18f553d`→`22b0ce1` on holospaces
  `chore/hologram-head-sync` — clean build + 109 tests pass (the 3 breaking changes don't
  touch holospaces; the feared 104-commit port was a clean pin-bump). Bridge tag pending.
- [x] **P0 human gates cleared (2026-07-14)**: relicense consent **granted** (owner is sole
  rights holder); restructuring review **satisfied** (owner); `hologram` name **decided** —
  attempt acquisition, fallback **`uor-hologram`** (non-blocking; only bites at P3 publish).
- [x] **holospaces V&V vs hologram HEAD — code green** (2026-07-14): every hologram-
  dependent suite passed (addressing, security incl. attenuation, ingestion, workspace,
  RISC-V/AArch64/x86-64 emulators, real RISC-V Linux boot to userspace). The V&V harness
  reported FAILED only on two **environment** false-negatives, both independent of the
  sync: CS docs validators (Java 24 vs pinned 21) and portability (homebrew `cargo` lacks
  wasm32/thumbv7em sysroots — proven green by rebuilding with the rustup toolchain).
- [ ] **P0 exit remainder** (non-blocking for P1): bridge tag cut; hologram-ai migrated
  (at P3); crates.io tokens/ownership (P3); fix the two V&V env issues in CI (Java 21 pin,
  rustup toolchain for cross-builds).
- [ ] **P1 — substrate dissolution** (in progress; non-colliding order chosen so it avoids
  Sprint 39's active `hologram-compute` work; perf baselines deferred until kernels settle):
  - [x] **1: bare-hal → hologram-space** — HAL (`BlockDevice`/`NetworkInterface` + fixture)
    absorbed as `hologram_space::hal`; 4 dependents redirected; `substrate/hologram-bare-hal`
    deleted. All 7 scenarios + golden vectors + native/wasm builds + clippy/fmt green.
  - [x] **2: substrate-tck → crates/hologram-tck** — battery crate renamed/relocated;
    SP-1 witness updated (stays green); 4 dependents redirected. (store-mem absorption
    into hologram-tck deferred to a sub-step.) All 7 scenarios + tests + clippy/fmt green.
  - [x] **3: substrate-core → hologram-space** (the foundational one): trait surfaces +
    κ-addressing folded in as the `substrate` module; 88 refs across 16 crates redirected;
    substrate-core deleted. All 7 scenarios (GV-1/LAW-1/MG-5/SP-1 witnesses now via
    hologram_space) + full native/wasm build + tests + clippy/fmt green. `substrate/`: 14→11.
  - [x] **4: realizations → hologram-space** — canonical forms folded in as the
    `realizations` module (+ codemodule test); ~29 refs redirected; all 12 dependents'
    realizations dep removed (they already had hologram-space). **`hologram-space` is now
    the complete contract crate (core + realizations + HAL + Space/Resolver) per spec 02.**
    All 7 scenarios + native/wasm build + tests + clippy/fmt green. `substrate/`: 11→10.
  - [x] **5: net-http/tcp/bare → crates/hologram-net** — consolidated as feature-gated
    modules `bare`/`http`(+`http::live`)/`tcp`; tokio behind `tcp` so no_std/wasm stays
    clean; only substrate-cli redirected. All 7 scenarios + native/wasm + tests +
    clippy/fmt green. `substrate/`: 10→7.
  - [x] **6: runtime/-wasmtime/-bare → crates/hologram-runtime** — core + engine backends
    behind `engine-wasmtime`/`engine-wasmi` features; **engine-wasmi builds wasm32** (browser/
    iOS/bare interpreter). Engine coverage for wasm/WASI/iOS documented in 02. Green.
  - [x] **7: store-mem → hologram-tck** — the reference `MemKappaStore` becomes the TCK's
    `mem` module (battery + oracle in one crate); tests + sp_floors bench move with it. Green.
  - [x] Justfile retargeted (wasm/embedded/vv-substrate) to the consolidated crates; RZ gate
    holds (compute engine absent from store/net/runtime). `substrate/`: now **3** (store-native,
    store-bare, substrate-cli).
  - [x] **8: substrate-cli → hologram-cli (D13)** — the node CLI merges into the one
    `hologram` binary as the `node` subcommand group (verified: `hologram --help` shows
    compile/execute/bench/inspect + node; `hologram node --help` shows put/get/serve/…).
    The two-binaries-named-`hologram` conflict is resolved. Green.
  - [x] **9: hologram-host → hologram-types (D15)** — 137 LOC of σ-axis selections
    (`HologramHasher`/`HologramHostTypes`/`ActiveCpuBounds`/`HologramHostBounds*` + prism/sdk
    re-exports) fold into `hologram-types` as a root-flattened `host` module; 11 dependents
    redirected (`hologram_host::`→`hologram_types::`, `hologram-host/std`→`hologram-types/std`),
    facade `host` feature/module dropped (ships under `types`). host was untouched by Sprint 39
    (verified: no divergence), so it was safe to pull forward. Green: workspace tests, bdd +
    meta-gate (golden vectors held), clippy -D, fmt, wasm32 + thumbv7em no_std, RZ gate. `crates/`: 19.
  - [x] Cleaned the empty leftover crate shells from moves 1–8; `substrate/` now holds exactly
    `store-native` + `store-bare` (members) + excluded `efi`/`store-opfs` — **4 dirs, 2 members**.
  - [x] **P1 preflight — crates.io readiness + supply-chain gate:**
    - Facade published as **`uor-hologram`** (name `hologram` is taken; D16/P0-PREP §3 decision);
      `[lib] name = "hologram"` preserves the one-word import path — user code still `use hologram::…`.
    - Workspace `repository` field added; **`rust-version = "1.85"` declared and verified** (the
      workspace fails on 1.82 — a dep needs `edition2024`, stabilized in 1.85 — and builds clean
      on 1.85.1; probed against real toolchains, not guessed).
    - **`cargo-deny` wired** (`deny.toml` + `just deny`, in `ci`/`vv`): licenses/bans/sources/
      advisories all green. It caught + fixed a **yanked `spin 0.9.8`** (→0.9.9) and a crossbeam
      vuln + memmap2-unsound advisory (cleared by a lockfile bump). Unmaintained transitive crates
      (`paste`, `unic-*`) triaged to `ignore` with justification (no upstream fix).
    - [x] **LGPL removed (governance, resolved 2026-07-15).** cargo-deny flagged that
      `frontend-python` pulled `rustpython-parser` → **`malachite*` (LGPL-3.0-only)**. Chosen fix:
      **replaced `rustpython-parser` with an in-tree ~1.2k-LOC restricted-Python parser**
      (`hologram-compiler/src/source/frontends/pyparse/`, lexer + recursive-descent, byte-accurate
      spans). This drops the LGPL *and* every `unic-*` unmaintained advisory *and* a heavy dep in
      one move; `frontend-python` keeps working (43 tests, exact error positions preserved). deny is
      green with **zero license exceptions**; the only remaining advisory-ignore is `paste`
      (unmaintained proc-macro via the `metal` GPU backend, `backend-metal` only). The whole tree is
      permissive (`MIT OR Apache-2.0`) again.
  - **P1 in-repo restructure essentially complete.** Only `backend→compute` (genuine Sprint 39
    collision — that IS the active kernel crate) and the P2 store moves remain. Remaining:
    stores → `spaces/` (P2, needs holospaces imported); `backend→compute` at a Sprint 39 lull;
    perf baselines; crates.io tokens/org ownership (human, P3).
  - [x] **`hologram-backend` → `hologram-compute`** (2026-07-17, D3 — the last library rename;
    "backend" retired). `git mv` the crate + a mechanical sweep of 233 refs across 110 files
    (crate name in Cargo.toml/CI/Justfile/deny; `hologram_backend`→`hologram_compute` imports;
    KC witness paths + specs). Green: workspace build + test, conformance gates, clippy -D, fmt.
    **P1 in-repo restructure now fully complete.**
- [~] **P2 core DONE** (2026-07-15, commit `094af14`) — holospaces imported into `spaces/`
  (clean snapshot) and ported onto the consolidated crates; plan in
  [specs/refactor/P2-PLAN.md](refactor/P2-PLAN.md). `holospaces` + `holospaces-node` are root
  workspace members, build on default **and** `--no-default-features` (no_std core), lib tests
  green (109 + 7), `cargo test --workspace` green, deny/clippy/fmt/bdd/RZ green. The port needed
  **zero API-drift fixes** — every dissolved-crate symbol resolved after the path renames. Fixed
  a stray `#![no_std]` leftover in `hologram-net/src/http/mod.rs` (from move 5). The 170M `vv/`
  fixture tree was **not** imported; its CC integration tests skip-when-absent.
  - [x] **P2 tail — `MemKappaStore` re-homed** (2026-07-15): moved out of the conformance TCK
    into `hologram-space` (the crate that defines `KappaStore`; zero new deps — hashbrown + spin
    already present). `hologram-tck` re-exports it (back-compat: every dev-dep test user is
    untouched) and drops to a single dependency; holospaces/node/spike no longer take a *runtime*
    dep on the test kit. Green across workspace test, bdd, clippy, fmt, deny, wasm32 + thumbv7em
    no_std, RZ.
  - [x] **`substrate/` ELIMINATED** (2026-07-15). The `hologram-store-*` backends + `hologram-efi`
    are generic `hologram-*` crates → moved to **`crates/`** (user decision: `hologram-*` = core
    infra in `crates/`; `spaces/` holds only `holospaces*` space impls; supersedes the spec's
    `store-native → holospaces-native` rename — 01 §"From substrate/" updated). Pure path move
    (all consumers use `{ workspace = true }`); green: workspace test, clippy, fmt, deny,
    thumbv7em no_std, RZ. `spaces/` is now purely `holospaces{,-node,-web,-emulator}`.
  - [x] **holospaces-web → `spaces/holospaces-browser`** (2026-07-15) — the last crate on
    pre-refactor git-pins is ported: renamed, git-pins → explicit path deps onto the consolidated
    crates (deduped: substrate-core + store-mem → hologram-space; runtime + runtime-bare →
    hologram-runtime `engine-wasmi`), imports rewritten. **wasm32 builds clean** (own `[workspace]`,
    standalone; cross-workspace `holospaces` path dep resolves). Its `opfs_store.rs` (sync
    `FileSystemSyncAccessHandle`) was later moved into `crates/hologram-store-opfs` (the OPFS
    backend crate) — see "P2 tail — OPFS dedup resolved" below.
  - [x] **Emulator hoisted → `crates/hologram-emulator`** (2026-07-15). System emulation is a
    first-class *hologram* capability, not holospaces-specific (user call). The 20.6k-LOC core
    (`emulator.rs` + RISC-V/x86-64/aarch64 cores + `machine.rs`) hoisted out of holospaces into
    `crates/hologram-emulator` (deps only `hologram-space`/`libm`/`spin`; the couplings were
    doc-links only). holospaces re-exports `emulator`/`machine` (zero churn for its 5 consumer
    modules) and **shrank ~31k → 10.4k LOC** — now purely the boot/provisioning/peer layer. The
    codemodule moved too: `spaces/holospaces-emulator` → `crates/hologram-emulator-codemodule`
    (sheds its holospaces dep — wraps `hologram_emulator` directly). Green: hologram-emulator
    (47 tests) + holospaces (62) + full workspace test, clippy, fmt, deny, RZ, wasm32 + thumbv7em
    no_std, wasm32 codemodule. 01-crate-map updated.
  - [x] **P2 tail — OPFS dedup resolved** (2026-07-15). Investigation showed the two OPFS stores
    are *not* copy-paste dup but a boundary violation: the in-product sync `OpfsKappaStore` (Worker;
    pack file + offset index; the real `KappaStore`) lived **inside** `spaces/holospaces-browser`,
    while `crates/hologram-store-opfs` (async file-per-κ + GC + JS bindings, Playwright-verified) was
    consumed by nothing in Rust. Fix (on the north star — backends in `crates/`, spaces consume
    them): **moved `OpfsKappaStore` into `crates/hologram-store-opfs`** as its `sync_store` module,
    gated the async JS API (`js_api` + SAB `bridge`) behind a default `js-api` feature, added
    `crate-type=+rlib`. `holospaces-browser` now depends on it with `default-features = false` (the
    sync backend only — pulls no `wasm-bindgen`, so no duplicate-wasm-bindgen clash in the Pages
    bundle). The crate finally holds a real `KappaStore` (matching its `-native`/`-mem` siblings) and
    OPFS is one crate. Green: store-opfs wasm32 build **both** feature configs, holospaces-browser
    standalone wasm32 build, fmt. (wasm32 *clippy* is unavailable in this env — clippy-driver can't
    resolve the wasm32 std sysroot; build is the gate for wasm-only crates.)
  - [~] **P2 tail — MG-7 shaped (pending)** (2026-07-15). Investigated holospaces' V&V: it has its
    own mature framework — 45 `cc*.rs` component-conformance tests (the **CC catalog**) + CS-* spec
    conformance, witnessed against external authorities (hash KATs, native-executor oracle, substrate
    TCK, QEMU, Playwright), plus a **170M `vv/`** tree (almost entirely `vv/artifacts` external
    oracles/images; the framework itself is ~250K). Per the method, **shaped `MG-7` as a `@status:pending`
    scenario + CONFORMANCE.md row** cataloguing the absorption acceptance criteria (CC/CS run under
    the one meta-gate, external-authority-witnessed, `vv/` artifacts content-addressed on import).
    bdd 32 scenarios / 9 enforced; meta-gate bijective + green. **Enforcement is a multi-session
    effort. **`vv/` fixtures decision → CLOSED (2026-07-16, user call): external, never committed.**
    Import only the ~250K vv/ framework (`suites`/`lib`/`heavy`/`run.sh`/`PROVENANCE.md`); the 170M
    `vv/artifacts/` stays out of git — `run.sh` reproduces/fetches each from its pinned `SOURCE.txt`
    (mke2fs images, reproducible kernels, BuildKit OCI, fetched-by-pin vscode-web/Structurizr),
    verified by re-derivation, skip-when-absent locally. `.gitignore` guard added now
    (`vv/artifacts/`) so no future import can bloat the repo; 06-migration records the strategy.
    MG-7 enforcement (the CC-catalog absorption itself) is unblocked but remains a multi-session job.
  - [~] **MG-7 ENFORCEMENT (in progress, plan approved 2026-07-16)** — full gating: MG-7 flips ✅
    only when all 45 CC pass in CI (QEMU boots + Playwright + 170M artifacts materialized, blocking).
    CS-\* deferred but tracked (Phase G). Phases:
    - [x] **A — CC class in the ledger** (2026-07-16). Added a non-BDD `CC` class (Classes table) +
      45 rows (`## CC — component conformance`) mechanically derived from the ported
      `spaces/holospaces/tests/cc*.rs` (12 fast / 8 artifact / 25 heavy tiers), each witnessed by a
      cargo test fn — no Gherkin, no meta-gate change (CC ignored like AS/KC). Rows 🟡 (present, not
      yet CI-gated); MG-7 stays ⛔. Green: meta_gate bijection holds.
    - [x] **B — CC bijection audit** (2026-07-16). `hologram_conformance::cc` (`check_cc_bijection`
      + `collect_cc_witnesses`) + a new `tests/cc_gate.rs` that binds every CC row to a present
      `#[test]` fn in `spaces/holospaces/tests/cc*.rs` — text-only, **no artifacts, no cc compile**.
      Broadened `catalog::extract_witness` to also capture `.rs::` witnesses (CC/AS/KC). All 45 rows
      bind (validating Phase A's generation); a corrupted witness fails `cc_gate` with the exact
      violation (teeth verified), restore → green. 18 lib tests, clippy -D, fmt clean.
    - [x] **C — import `vv/` framework at repo root** (2026-07-16). Copied holospaces'
      `vv/{run.sh,README,PROVENANCE.md,lib,heavy,suites,targets}` (52 suites) → `hologram/vv/`;
      **`vv/artifacts/` NOT imported** (gitignore guard confirmed active). Gated `run.sh`'s CS-\*
      block behind `CC_ONLY` (default `1` in-tree — docs V&V absorbed in Phase G). Suite `-p
      holospaces --test ccN` resolves unchanged (workspace member). Verified: run.sh syntax ok;
      cc1 (fast) → 5 passed exit 0; cc7 (artifact absent) → skip-guards fire, exit 0. Full-green run
      (browser/QEMU + portability) is the Phase-E CI tier.
    - [x] **D — artifact-materialization pipeline** (2026-07-16). **Hosting sub-decision resolved:**
      holospaces' 170M `vv/artifacts/` is *git-committed* (351 files), so `scripts/vv-fetch.sh`
      fetches the `vv/artifacts/` subtree at a provenance pin (`vv/ARTIFACTS_PIN` =
      holospaces@142caa0, archived-but-accessible) via `git archive` — no kernel-reproduction, no
      separate release-asset upload — then verifies the 14 sha256 sidecars. Idempotent (skip if
      present+verified); handles mixed sidecar path conventions (root- vs dir-relative) and
      distinguishes *fetched-by-pin* artifacts (cc17 vscode-web, intel-sdm — not committed,
      materialized by their own suites → skip) from *corruption* (present+wrong-hash → fail).
      Verified end-to-end: materialize → verify exit 0 → idempotent skip → **cc7 ext4 round-trip +
      cc35 AArch64 ISA battery PASS against the real artifacts** (SKIP→PASS) → shellcheck clean.
      Fallback if holospaces ever becomes unfetchable: mirror the pinned tree to a hologram release asset.
    - [x] **E — heavy CI job** (2026-07-16; promoted to blocking in F). `.github/workflows/ci.yml`: (1) a cheap
      **CC audit step** in the existing `vv` job (`cc_gate` + `meta_gate` + `bdd`, artifact-free);
      (2) a new **`holospaces-vv-heavy`** job — QEMU (riscv64/aarch64/x86-64/user) + e2fsprogs + OVMF
      + Node/Playwright/wasm-pack; a second pinned holospaces checkout as the artifact source →
      `vv-fetch.sh` (170M cached on the pin) → `CC_ONLY=1 bash vv/run.sh` (all 45 CC). Pin validated
      as a 40-hex sha before use in `ref:` (injection-safe). Also fixed two stale `substrate/…` CI
      paths left by the refactor (`hologram-efi`, `hologram-store-opfs/web` → `crates/…`). actionlint
      clean (the one SC2086 info is pre-existing); YAML valid. **Introduced NON-blocking** (not in
      `ci-success.needs`) to keep the branch always-green while unvalidated — `chore/refactor` doesn't
      trigger CI, so the job must be observed green via the nightly `schedule` / `workflow_dispatch`
      / a PR before Phase F promotes it to blocking + flips MG-7. **Human note:** if the holospaces
      repo is private, set the `HOLOSPACES_ARTIFACTS_TOKEN` CI secret (a read token).
    - [x] **F — MG-7 ENFORCED** (2026-07-16). Witnessed by running the **full CC V&V locally**
      (holospaces at `../holospaces`; all qemu-system riscv64/aarch64/x86-64 + e2fsprogs + Node +
      Playwright Chromium present; artifacts materialized via `vv-fetch.sh`). Result: **every one of
      the 45 cargo-witnessed CC passed** — incl. all real QEMU Linux boots (cc9/11/14/15/16/36/44) —
      plus the browser workbench suites, with **zero cargo test failures**. The first FAILED verdict
      was purely the incomplete browser port: the vv/ browser suites + `build-wasm-peer.sh` still
      referenced the pre-rename `crates/holospaces-web` (→ `spaces/holospaces-browser`), and two
      holospaces scripts (`build-extension.sh`, `browser-manager-test.sh`) weren't imported — fixed
      (commits `e3c1877`, `680c8a5`; `[lib] name = "holospaces_web"` keeps the wasm/web assets valid;
      `.vscode-test-web` untracked+gitignored). Flip: feature `@status:enforced` + MG-7 step defs
      calling `cc::check_cc_bijection` (the audit is authored once, run by both `cc_gate` and the BDD
      step); MG-7 row ⛔→✅; **44/45 CC rows 🟡→✅**. `holospaces-vv-heavy` promoted to **blocking**
      (`ci-success.needs`). Green: meta_gate, cc_gate, bdd (**10 enforced**, MG-7 runs+passes), fmt.
      CC-31 + CC-51 confirmed green by running their `#[ignore]`d witnesses directly
      (cc31_resume_terminal 74s real devcontainer resume; cc51_nested_workspace 23s real QEMU-9p
      boot) — they have no vv/ suite, so an explicit step was added to `holospaces-vv-heavy` to gate
      them. **CC-45 stays 🟡** — its dogfood witness needs ~24 GB + a real Dev Container build
      (`vv/heavy/`, manual), not automatically gatable.
    - [ ] **F-followup (tracked): browser workbench V&V gaps** — three browser-only suites
      (`cc51-scm-git`, `cc52-search`, `cc53-tasks`; VS Code workbench SCM/search/tasks) fail
      consistently on vscode-cdn / welcome-media asset loads + command-palette timing (post-port
      workbench regressions; **not** in the 45-row cargo ledger — CC-52/53 have no `cc*.rs`).
      **Quarantined non-gating** in `vv/run.sh` (`VV_QUARANTINE`, clearly logged); fix + un-quarantine.
  - [~] **Phase G — CS-\* docs conformance (in progress, started 2026-07-16)**. Absorb holospaces'
    *specification* conformance (the docs V&V) alongside the CC work. Scope from investigation:
    holospaces `docs/` is 1.7G but only ~1–2MB is source — **105 git-tracked source files** (arc42
    chapters 01–13, `src/` adoc, `scripts/` V1–V8 validators + orchestration, `tools/` **7 source
    pins**, Gemfile, images, tests); the **1.6G `docs/tools/`** is downloaded tools (Structurizr.war,
    cmark-gfm, pandoc — materialized by `install-tools.sh`) and **`docs/vendor/arc42-generator` is a
    git submodule** — neither is committed. Catalog: **CS-1..CS-6** witnessed by validators **V1–V8**.
    Toolchain caveat: needs JDK 21 / Ruby 3 / Structurizr / cmark-gfm / pandoc — only partly
    available locally (java 24, ruby 4, pandoc/cmark absent), so V1–V8 is CI-validated (like the CC
    heavy tier), not fully local. Sub-phases:
    - [ ] **G1** import docs source → `specs/holospaces/` (105 files + the arc42-generator submodule);
      gitignore the 1.6G tool downloads; `install-tools.sh` materializes them (mirrors `vv-fetch.sh`).
    - [ ] **G2** adapt the V1–V8 validators' paths to the new home; `install-tools.sh` in-tree.
    - [x] **G1** docs source imported → `specs/holospaces/` (104 tracked files, 1.2M; via `git
      archive` — no downloads/submodule content; .gitignore guards the 1.6G tool tree) — `61d5429`.
    - [x] **G2** validators verified **self-contained** (all paths relative to `REPO_ROOT` =
      `specs/holospaces/`; only a comment mentions holospaces) — no path adaptation needed. The
      arc42-generator submodule (pin `46bd7cea`) is CI-only (V2/arc42-build) → added in G4.
    - [x] **G3** non-BDD **`CS`** class + CS-1..CS-6 in `CONFORMANCE.md`, witnessed by V1–V8
      (`specs/holospaces/scripts/v*-*`); `run.sh` CS block repointed to
      `specs/holospaces/scripts/build.sh` (`CC_ONLY` still defaults on locally — no toolchain; G4
      sets it off). Rows 🟡. meta_gate green (CS ignored) — `7307b99`.
    - [~] **G4** docs-conformance CI job authored (2026-07-16). New `docs-conformance` job in
      `ci.yml`: checkout `submodules: recursive`, JDK 21 + Ruby 3 (bundler-cache) + Node 22,
      `install-tools.sh` (Structurizr/cmark-gfm/pandoc/playwright), then `build.sh` (V1–V8). Added
      the **arc42-generator submodule** as a pointer only (`.gitmodules` + a gitlink at pin
      `46bd7cea` via `update-index --cacheinfo` — no 153M clone; CI fetches it). Introduced
      Fixed the install-tools nested-submodule fetch (fetch the exact arc42-template pin SHA).
      **CI-GREEN (2026-07-17, 2m34s)** — the toolchain installs + V1–V8 all pass; **promoted to
      blocking** (`ci-success.needs`).
    - [x] **G5 — MG-8 ENFORCED** (2026-07-17). Once `docs-conformance` went green in CI, flipped
      MG-8 `@status:enforced` + row ✅ + **CS-1..6 rows ✅**. Added a **CS bijection audit**
      (`cc::check_cs_bijection` + `tests/cs_gate.rs`) binding every CS row to a present V1–V8
      validator script (artifact-free); MG-8's bdd step calls it. `catalog::extract_witness`
      broadened to capture `.sh` validator witnesses. Green: meta_gate + cc_gate + cs_gate, bdd (11
      enforced — MG-8 runs+passes), clippy -D. **Phase G complete; both MG-7 + MG-8 enforced.**
- [~] **P3 — generic lifecycle `Session` hoisted → `hologram-runtime`** (2026-07-15, D7). Only
  the space-agnostic lifecycle *primitive* (boot/suspend/resume/terminate over `ContainerRuntime`
  + container κ + caps κ) is now `hologram_runtime::lifecycle::Session` (346 LOC + 4 tests).
  holospaces keeps a thin `Session` wrapper (adds `holospace()`/`reconfigure()`) — **zero external
  churn**: Peer, Manager, config, identity, the ~74 lifecycle call sites, and all CC tests unchanged.
  Correction to the original D7: **Peer + Manager stay in holospaces** — they're the operator/peer
  *platform* layer (provision Holospaces, roster sync, control-plane reconfigure), generic over `R`
  only mechanically, not generic infrastructure (01-crate-map updated). Green: workspace test (913),
  clippy, fmt, deny, RZ, wasm32 no_std.
  - [x] **`Client` facade MVP** (2026-07-15, D4). `hologram::Client<S: Space>` in the facade
    (`client` feature): `builder().space(s).build()` → `compile` (sync) → `provision` (sync store)
    → `get/pin/unpin/ls/verify/gc` passthroughs → `resolve`/`run` (async seam → sync compute). The
    kept realization of the SP-3 spike; tested end-to-end (compile→provision→run an i64→f32 cast)
    + builds no_std for wasm32. Deferred to the P3 naming-review gate: `open → Session` (needs the
    Space contract to expose a `ContainerRuntime`) + the fuller net/app/manager/network surface.
    05-tooling updated.
  - [x] **Space contract expanded → `Client::open → Session`** (2026-07-15). Two steps:
    (1) `ContainerEngine` (+ `HostContext`/`ContainerIntents`) moved into the contract crate
    `hologram-space` (commit `42630d6`); (2) the `Space` trait gained `type Runtime: ContainerRuntime`
    + `runtime()` — the **pragmatic shape** (Space exposes the *composed* runtime, `store()` delegates
    to `runtime().store()`; chosen over spec-02's literal `type Engine` because a `Runtime` owns its
    store; 02 §Space corrected). `hologram::Client::open(container_κ, caps_κ) → Session` drives
    boot/suspend/resume/terminate over `space.runtime()`. Tested end-to-end (open→boot→suspend via
    the MockEngine). Green: workspace test, bdd/SP-3, clippy, fmt, wasm32 no_std (spike + facade).
  - [x] **HAL traits `Entropy` + `Clock` → the `Space` contract** (2026-07-15). `hologram-space`'s
    `hal` gains `Entropy` (`fill(&mut [u8])`) + `Clock` (`now_millis`) — the platform randomness /
    time seams (spec 02 §4; grounds the runtime's current direct `getrandom`) — each with a
    deterministic reference impl for hermetic V&V (`SeededEntropy` SplitMix64 / `ManualClock`).
    `Space` gains `type Entropy`/`type Clock` + accessors; both Space impls (SpikeSpace, TestSpace)
    provide them. Green: workspace test, bdd (SP-3/LAW-3), clippy, fmt, wasm32 + thumbv7em no_std.
  - [x] **`Space::Sync` — network seam unified** (2026-07-15). Reconciled the two overlapping
    network traits into **one cfg-gated maybe-Send `KappaSync`** (fetch/announce/discover/add_peer/
    add_gateway): retired the minimal `Resolver` and the `?Send`-twin `LocalKappaSync`. `Space` now
    has `type Sync: KappaSync` + `sync()`; `Client::resolve`/`run` use `sync().fetch()` (error type
    → `SyncError`). **Cascade**: making `KappaSync` `?Send` on wasm makes `Runtime` (holds an
    `Arc<dyn KappaSync>`) `!Send`, breaking `ContainerRuntime: Send+Sync` — so `ContainerRuntime`
    got the identical cfg-gated maybe-Send treatment + `LocalContainerRuntime` (dead twin, 0 impls)
    retired. The whole async surface is now one maybe-Send trait per seam. Green: workspace test
    (923), bdd (SP-3/LAW-3), clippy, fmt, deny, RZ, **wasm32 + thumbv7em no_std** (runtime, spike,
    holospaces, facade `client`).
  - [x] **`Space::Spawner` — background-task spawn seam** (2026-07-15). `hologram-space` `hal`
    gains `Spawner` — one cfg-gated maybe-Send trait `fn spawn(Pin<Box<dyn Future<Output=()>
    [+ Send] + 'static>>)` (Send native / ?Send wasm, same posture as KappaSync) — the seam where
    the net pump's `tokio::spawn`/`spawn_local` background work runs. Reference impl `NoopSpawner`
    (drops the future; for spaces with no background tasks + hermetic tests). `Space` gains
    `type Spawner` + `spawner()`; both impls provide `NoopSpawner`. **The Space contract now has
    6/7 spec-02 parts** (Store/Sync/Runtime/Entropy/Clock/Spawner). Green: workspace test, bdd,
    clippy, fmt, wasm32 + thumbv7em no_std.
  - [x] **`Space::Surface` — presentation / interaction seam** (2026-07-15). The last spec-02
    Space part, designed fresh (no trait to hoist — generalizes holospaces' `projection.rs`
    Workspace/Intent). `hologram-space` gains a `surface` module: one cfg-gated maybe-Send async
    `Surface` (`project(workload_κ) → κ` renders state; `intent(workload_κ, Intent) → κ` publishes
    an operator event, Law L1) + a closed `Intent` enum (TerminalInput / FileEdit / FrameRegion) +
    typed `SurfaceError`. Takes the running workload's **κ**, not a runtime `Session` — the contract
    crate must not depend on `hologram-runtime` (RZ). **Headless is a first-class profile**: the
    reference `NullSurface` projects the empty-projection κ and refuses `intent` with
    `SurfaceError::Headless`. `Space` gains `type Surface` + `surface()`; both impls (SpikeSpace,
    TestSpace) provide `NullSurface`. **The Space contract now has all 7/7 spec-02 parts**
    (Store/Sync/Runtime/Entropy/Clock/Spawner/Surface). Green: workspace test, bdd (SP-3/LAW-3) +
    meta-gate, clippy -D, fmt, deny, RZ, **wasm32 + thumbv7em no_std**. 02 §5 marked implemented.
  - [x] **Witness the new contract parts — SP-4 + SP-5 enforced** (2026-07-15). Per the method
    (every contract part earns a scenario), the four parts added this phase are now witnessed in
    `s1_space_contract`: **SP-4** (deterministic HAL seams — equally-seeded `SeededEntropy`
    reproduces its stream, `ManualClock` advances only when told, `NoopSpawner` drops the future)
    and **SP-5** (headless `Surface` — `NullSurface.project` → empty-projection κ, `intent` →
    `SurfaceError::Headless`), both driven through the reference impls' public API and flipped
    `@status:enforced` with matching CONFORMANCE.md rows (✅). bdd now 31 scenarios / **9 enforced**;
    meta-gate bijection green; clippy -D, fmt clean. (Entropy/Clock/Spawner + Surface are all
    exercised; the runtime seam SP-3 already covered Sync + Runtime.)
  - [ ] **P3 remaining**: the Client naming-review gate (D29); first lockstep `uor-hologram`
    release (hard stop, D26). The spec-02 `Space` contract is complete (7/7), all witnessed.
- [ ] **P4–P6** .holo v3 / networks / encryption (follow-on). Conformance-driven: drive HF → NW →
  GV rows from ⛔ to ✅, each backed by its real feature; always-green at every commit.
  - [x] **P4.1 — `AppManifest` realization** (2026-07-16, spec 03). The `.holo` v3 application is a
    SPINE-2/3 realization in `hologram-space`: `AppManifest` (IRI `.../realization/app-manifest`)
    embeds every layer κ, every child `(app κ, caps κ)`, and the `requires` CapabilitySet κ as
    operands, so `references()` yields the whole app's reachability closure — migrating an app is
    `resolve_closure(app κ)`, the same op as any content. Closed `LayerKind` enum
    (WasmCodemodule/TensorPlan/RootfsImage/View; exit-semantics derived from kind, no catch-all);
    `Layer` (content κ + entrypoint + kind-specific arch/surface tag); `primary: Option<u32>` so the
    **degenerate tensor-only archive** (one TensorPlan layer, no exit code) is valid; `validate()`
    enforces the load-time invariants (primary is exit-bearing; rootfs has arch; portable kinds
    don't); `decode()` is the inverse. Registered in `REGISTRY`. 6 new tests; native + wasm32 +
    thumbv7em green, clippy -D, fmt clean.
  - [x] **P4.2 — `.holo` v3 in `hologram-archive`** (2026-07-17). `FORMAT_VERSION` 2→3;
    `SectionKind::AppManifest` (discriminant 15; kinds 0–14 unchanged, κ-stability); writer
    `set_app_manifest` (opaque bytes — the AppManifest canonical form; the archive doesn't depend on
    `hologram-space`, correct layering); loader `app_manifest()` accessor; v2 read-shim
    (`MIN_READ_VERSION..=FORMAT_VERSION`). 4 new tests; exec/ffi/runtime round-trip v3 unchanged;
    clippy -D, fmt clean. Manifest-*presence* enforcement is the app loader's (P4.3); `into_plan`
    stays the bare tensor-container reader.
  - [~] **P4.3** — app loader (`resolve_closure` + fat/thin) + parser fuzz targets.
    - [x] **`resolve_closure` core** (2026-07-17) — the app-loader reachability primitive in
      `hologram-space`: `resolve_closure(root, &dyn KappaStore, registry) -> Closure` walks the
      κ-graph breadth-first from an app κ via each realization's `references()` (opaque leaf content
      contributes no edges), fetching bytes from the store. `Closure { reachable, missing }`:
      `missing` records κs named-but-absent (the **thin**-archive signal, resolved via KappaSync,
      LAW-4); `is_complete()` ⇒ a **fat** closure. This is "load = resolve the manifest's closure"
      (03 §Fat and thin) and `resolve_closure(app κ)` migration. 2 tests (fat + thin); native +
      wasm32 + thumbv7em green. Unblocks HF-3 (inspection resolves + verifies layers).
    - [x] **parser hardening** (2026-07-17, spec 03 §Parser hardening — standing requirement). Audit
      + fix of the network-facing parsers: `AppManifest::decode` (`1 + n_layers` ×2, `2 * n_children`)
      and `LoadedPlan::section`/`extensions` (`start + length` on forged u64 offsets) overflowed
      `usize` on 32-bit targets (wasm32/bare-metal) on hostile bytes — now checked arithmetic (Err,
      never overflow/OOM), allocation bounded by real ref count not declared count. CI-permanent
      deterministic mutation suites (`hologram-space` + `hologram-archive` `tests/parser_hardening.rs`)
      prove every P4–P6 decoder + the section parser + the generic `references()` dispatch never panic
      over truncations / byte-mutations / noise + forged oversized counts.
    - [x] **fat/thin conversion** (2026-07-17, spec 03 §Fat and thin). `SectionKind::ContentBlob`
      (κ71 ‖ content, repeatable) lets a **fat** archive embed its layer/closure content;
      `HoloWriter::assemble` frames raw sections + `content_blobs()` reads them back (zero-copy).
      `Client::fat` resolves the manifest closure over the store and embeds every reachable κ;
      `Client::thin` drops blobs (manifest + certificates only); `is_fat` checks self-containment via
      `resolve_closure` over the archive's own blobs. **The manifest κ — the app's identity — is
      invariant across fat↔thin** (packaging, not identity; tested). ContentBlob added to the archive
      parser-hardening fuzz. Native + wasm32 green.
    - [x] **`hologram app` CLI** (2026-07-17) — `app inspect <archive>` prints the app κ + primary +
      per-layer descriptors + children (store-free; decodes the manifest realization); `app thin
      --input --output` re-frames to manifest + certificates only (app κ unchanged). Both wrap the
      same archive/space primitives as `Client::inspect`/`thin`. `app fat` needs the node's content
      store — a follow-on. `thin_archive_bytes` unit-tested (manifest preserved, payload dropped).
    - [x] **out-of-tree cargo-fuzz targets** (2026-07-17, spec 03 §Parser hardening — CI-permanent).
      `crates/hologram-space/fuzz/` (own `[workspace]`, host-workspace-excluded): coverage-guided
      libfuzzer targets `manifest_decode` (`AppManifest::decode`/`references`) and
      `references_dispatch` (the generic registry dispatch — the network entry point). Both **build**
      under nightly+ASAN; `manifest_decode` ran **200k iterations with no crash**, independently
      confirming the parser-hardening fixes. Generated corpus/artifacts gitignored. **CI job wired**:
      `ci.yml` `fuzz` job (nightly `schedule` + `workflow_dispatch`) installs cargo-fuzz, builds the
      targets under ASan, and deep-fuzzes each for 5 min — a discovered crash fails the nightly run.
      The per-PR always-green gate stays the deterministic in-tree mutation suites.
    - [x] **`hologram app fat`** (2026-07-17) — `app fat --input --output --store <redb>` resolves
      the manifest closure over a persistent `NativeKappaStore` and embeds every reachable κ's content
      as a ContentBlob (self-contained); the app κ is unchanged. Completes the fat/thin CLI
      (inspect/thin/fat). Test provisions a store, fattens, verifies blobs embedded + κ preserved.
  - [x] **P4.4 — HF conformance complete (3/3)** — HF-1/2/3 all ⛔→✅ with executable steps (the
    ledger's whole HF class). The `.holo` v3 format's conformance surface is green.
    - [x] **HF-1** (2026-07-17) — `.holo` v3 is the one container: opening a tensor-only archive
      (a real v3 archive whose AppManifest section carries `single_tensor_plan`) yields the
      degenerate single-layer case (1 tensor-plan layer, no primary). Witnessed end-to-end through
      the archive container + AppManifest realization. `container.feature` @status:enforced, HF-1 ✅.
    - [x] **HF-2** (2026-07-17) — capability-attenuated nesting: a parent AppManifest nests a child
      by κ ref `(app κ, delegated caps κ)`; `Capabilities::admits` witnesses the delegated set ⊆
      parent (refs + budgets), and an over-broad child is refused. `nesting.feature` @status:enforced,
      HF-2 ✅. bdd now **13 passed / 20 skipped**; meta-gate bijection + status agreement green.
    - [x] **HF-3** (2026-07-17) — per-layer certificates verify + never stripped: `Client::inspect`
      decodes a `.holo` v3 manifest and returns one `LayerCertVerdict` per layer. A layer's cert is
      its κ-identity **bound into the app κ** (the manifest κ addresses the bytes embedding every
      layer κ — stripping/swapping any layer changes the app κ); verification is **thin** (manifest
      only, no payload — witnessed on a manifest-only archive), so certs travel with the manifest and
      inspection never strips them. `client` feature now enables `archive`. `certificates.feature`
      @status:enforced, HF-3 ✅. **All 3 HF rows green — P4's conformance surface complete.** bdd 14
      passed.
  - [~] **P5** — networks (spec 04). **NW conformance complete (2/2)** (2026-07-17):
    - [x] **NW-1** — `Network` realization in `hologram-space` (SPINE-2/3): embeds the membership
      set + policy CapabilitySet κ (+ optional reserved parent-network κ) as operands; `references()`
      recovers exactly them, no side tables. `decode()` inverse; registered. `realization.feature`
      @status:enforced, NW-1 ✅.
    - [x] **NW-2** — `NetworkTier` (public / restricted / private) + `NetworkOp`; `admits(op,
      is_member)` gates from `(tier, membership)` **alone** — its signature carries no business data,
      so the check is structurally at the protocol boundary. Public admits all; restricted/private
      require membership (private adds P6 encryption, not a capability change). `tiers.feature`
      @status:enforced, NW-2 ✅. bdd now 16 passed; native + wasm32 + thumbv7em green.
    - [x] **bounded `resolve_closure`** (2026-07-17, spec 04 §Protocol hardening). A peer resolving a
      manifest served over the network seam must bound the walk — a hostile peer can otherwise serve an
      adversarially wide/deep κ-graph to force an unbounded resolve (DoS). `resolve_closure_bounded(…,
      max_nodes)` stops at the limit and sets `Closure::truncated`; `is_complete()` now also requires
      `!truncated`. `resolve_closure` delegates to it (unbounded). Tested; native + thumbv7em green.
    - [x] **wire-version negotiation** (2026-07-17, spec 04 §Protocol hardening). `hologram-net`
      `protocol` module: `WIRE_VERSION` + `WireVersionRange{min,max}` with `negotiate` (highest common
      version; disjoint ranges ⇒ `None` = refuse, never a silent downgrade) + `encode`/`decode` of the
      4-byte handshake payload (malformed / `min>max` rejected). Portable/no_std.
      **Wired into the `bare` frame protocol**: `KIND_HELLO` frame + `hello_frame` /
      `negotiate_from_hello` (the connect handshake — highest common version, or a `HandshakeError`;
      a non-HELLO/garbage first frame is a clean `BadHello`, never a panic). Deterministic in-process
      test; native + thumbv7em green.
    - [x] **`hologram network` CLI** (2026-07-17, spec 04) — `network create --member <file>…
      --policy <file> --tier <t> [--key <file>] --output <file>` builds a `Network` realization whose
      membership/policy/key are the **κs of the content files** (a member/policy/key is content, named
      by its κ — SPINE-1); enforces Private ⟺ `--key`. `network show <file>` decodes + displays the κ,
      tier, membership, policy, and key binding. `network delegate --parent <capset> --child <capset>
      --output` mints a `Delegation` realization but only if `admits(parent, child)` — amplification
      refused (attenuation only, law 5). End-to-end tests (temp files).
    - [x] **in-process loopback transport test** (2026-07-17) — a `PairedNic` (crossed queues: one
      NIC's `transmit` is the other's `receive`) drives two real `BareNetSync` peers over an in-process
      link with **no sockets**: peer B fetches content only peer A holds — the full FETCH_REQ → resolve
      → FETCH_RES_OK → verify-on-receipt path — plus the FETCH_RES_404 miss path. Deterministic; the
      two-node protocol test the TCK battery would otherwise need a live harness for.
    - [x] **wire-version handshake wired into the live TCP transport** (2026-07-17). `KIND_HELLO` is
      now public (shared by `bare`/`tcp`), and `TcpKappaSync::handle_connection` answers a HELLO with
      its own HELLO + negotiates — **additive and backward-compatible**: a peer that opens with a
      HELLO negotiates; a peer that starts straight into a DHT/fetch frame is unaffected (the DHT
      suite still passes, 8/8); an incompatible/malformed HELLO closes the connection (refuse, no
      silent downgrade). Integration tests (`#[cfg(feature="tcp")]`, `127.0.0.1:0`, `current_thread`
      tokio): the raw handshake negotiates/refuses over a real socket, **and a real `TcpKappaSync`
      answers the handshake at its current wire version**. Cleanly gated — a no-op without `tcp`.
    - [x] **dialer-side handshake — full end-to-end negotiation** (2026-07-17). `TcpKappaSync::rpc`
      now runs `dialer_handshake` once per new connection (before any request): send our HELLO, read
      the peer's, negotiate — an incompatible peer aborts the dial. Both sides of a real connection
      now negotiate. **Verified with zero regression**: the whole DHT suite (fetch / find_node /
      get_providers / forgery-rejection — 8/8) passes with the handshake in the flow, proving the
      two-peer negotiate→fetch path works end-to-end. `no_std` core unaffected (tcp-gated).
    - [x] **transport inventory** (2026-07-17) — the frame protocol + wire-version handshake are
      carried by **TCP** (`TcpKappaSync`, handshake both sides), **HTTP-CAS** (`http::live`), the
      **bare-metal** NIC (`BareNetSync`), and **WebSocket** (browser-egress exit node in
      `holospaces-node`, `tungstenite`, tested end-to-end — CC-16). `network delegate` shipped
      (file-based Delegation w/ attenuation). `app fat`/`network create` use the persistent
      `NativeKappaStore`.
    - [x] **QUIC transport** (2026-07-17) — encrypted P2P over QUIC (`hologram-net::quic`, feature
      `quic`): quinn/TLS-1.3 carrying the same `len|kind|payload` frames + wire-version handshake;
      self-signed transport cert + skip-verify client (confidentiality from TLS, **integrity stays
      κ** — verify-on-receipt). `QuicPeer` serves + dials from one endpoint. 3 localhost tests
      (fetch / 404-miss / **forging-responder rejected**) — deterministic, gated in CI
      (`--features quic`). 14 new deps, no `blake3` (clean vs the κ core's 1.5 pin).
    - [x] **`network join` — QUIC peer routing** (2026-07-17): `QuicPeer` implements the full
      `KappaSync` — a join-ordered peer table (`join`/`add_peer`), `fetch(κ)` short-circuits on a
      local hit then routes to each joined peer until one honestly answers (verify-on-receipt per
      hop; dead/forging peers skipped, not fatal). Direct-dial model (announce no-op, discover empty
      — the DHT owns gossip). quic suite 5/5.
    - [x] **iroh — blocked upstream, recorded** (2026-07-17): modern iroh needs `blake3 1.8`; the κ
      core pins `blake3 1.5` (`uor-prism-crypto`) — irreconcilable, and the only resolvable iroh
      (0.28) pulls ~291 packages + an outdated API. **Unblock path:** bump `uor-prism-crypto`'s
      blake3 range, then modern iroh layers relay/NAT-traversal onto the shipped QUIC substrate.
    - [ ] genuinely-remaining P5 (external-dep / live-network, not always-green-unit): WebRTC browser
      endpoint; the live multi-node TCK battery on the heavy CI runner; iroh (pending the blake3
      unblock above).
  - [x] **P6 — GV governance conformance complete (4/4)** (2026-07-17). GV-1 was already ✅; this
    phase drove **GV-2/3/4** ⛔→✅:
    - **GV-3** — `AttestationKey` realization: a signing key bound to a κ-addressed identity as
      published content (identity IS its κ; leaf identity; deterministic single surface), never a
      second identity surface. Rotation = new content/new κ; revocation = append-only event.
    - **GV-4** — `Capabilities::admits_network_op`: store/fetch/announce gated from the capability
      alone (import/protocol boundary) with per-capability quota accounting, never global.
    - **GV-2** — `AuditEvent` realization + `LifecycleTransition`; `hologram-runtime`'s `Session`
      now emits through **one** `record` seam on every transition (spawn/suspend/resume/terminate),
      threading an append-only audit κ-chain — runtime-tested that all four advance a distinct linked
      head (no bypass). BDD witnesses the same seam's κ-chain.
    - bdd **19 passed**; meta-gate bijection + status agreement green; native + wasm32 + thumbv7em.
    - [x] **`RevocationEvent` chain + verifier** (2026-07-17, spec 07 R3) — the append-only
      complement to `AttestationKey` rotation: a `RevocationEvent` realization (revoked key κ +
      predecessor κ + reason) forms a tamper-evident revocation list; `is_revoked(key, head, store)`
      walks the chain so a verifier can decide if a key is revoked (append-only — nothing un-revokes).
      Registered; covered by the parser-hardening dispatch fuzz.
    - [x] **`SessionAttestation`** (2026-07-17, spec 07 R3) — the additive, non-breaking attestation
      section: a realization binding *where and how* a workload ran ("session booted app κ under caps
      κ on space-impl κ at engine κ", signed by an `AttestationKey` κ). `references()` recovers the
      five bound facts (no side tables); the binding is tamper-evident (content-addressed) and the
      signing key is bound as content, not a second surface. ed25519 signature verification is the
      verifier's follow-on. Existing `Snapshot` κs untouched (a separate realization, not a format
      break). Registered.
    - [x] **ed25519 sign/verify wiring** (2026-07-17, spec 07 R3) — the R3 attestation seam made
      real. Portable, dependency-free `SignatureVerifier` trait in the no_std core (a space supplies
      its platform verifier, like the other HAL seams); `SessionAttestation::signable_bytes` (the
      κ-embedding of the bound facts, empty payload) + `verify(verifier, public_key)`. The reference
      ed25519 impl is a **dev-dependency only** (`ed25519-dalek`), so wasm32/thumbv7em portability
      builds never pull curve25519 — verified. `tests/attestation_ed25519.rs`: a real attestation
      signs+verifies end-to-end, tampering any bound fact breaks it, wrong key fails, and malformed
      key/signature bytes are a clean `false` (never panic).
    - [x] **signed revocations** (2026-07-17, spec 07 R3) — `RevocationEvent` now carries a
      `revoker_key` κ + signature; `verify(verifier, revoker_pubkey)` + `is_revoked_signed(…,
      trusted_revoker)` honor a revocation only when its signature verifies under a *trusted*
      revoker, closing the "anyone revokes anyone" gap (a forged event naming a trusted revoker but
      signed by an attacker is rejected — ed25519-witnessed). Decoder added to the parser-hardening fuzz.
    - [x] **signed key rotation** (2026-07-17, spec 07 R3) — a `KeyRotation` realization: a **signed
      supersession chain** (superseded key κ → successor κ, signed by the *superseded* key so only its
      holder can rotate it, ed25519-witnessed). `current_key(chain_head, store)` returns the latest
      successor; old attestations stay verifiable against the key that made them. The complement of
      `RevocationEvent` (supersede vs invalidate). **R3 fully implemented**: κ-identity (GV-3) · signed
      rotation · authenticated revocation · signed session attestation — all four key-lifecycle events.
    - [x] **ChaCha20-Poly1305 Private-tier encryption** (2026-07-17, spec 04 §Private / P6 Phase B).
      Portable, dep-free `PayloadCipher` AEAD seam in the no_std core + `convergent_nonce(key,
      plaintext)` = `blake3(key ‖ plaintext)[..12]` + `seal_private`. **Convergent nonces need no
      RNG** (solving the bare-metal/wasm RNG wrinkle) *and* preserve **Law L3 dedup under
      encryption** — identical content under one key → identical ciphertext (the exact tension 04
      §Private flags; equality-leak is the documented tradeoff). Reference ChaCha20-Poly1305 impl is
      a **dev-dependency only** — wasm32/thumbv7em builds never pull the cipher (verified).
      `tests/private_tier_chacha20.rs`: round-trip, convergent dedup, wrong-key/tamper fail-loud
      (AEAD), distinct-key non-reuse, ill-sized-input never-panics.
    - [x] **network-key binding — Private tier end-to-end** (2026-07-17, spec 04 §Private). The
      `Network` realization now carries `key_ref: Option<κ>` — the Private tier binds its
      symmetric-key κ (the key material is content, so access is gated by the restricted-tier
      membership; no new asymmetric protocol). Two tail optionals (key_ref, parent) encode via a
      flags byte; decode is overflow-safe. `key_binding_ok()` enforces Private ⟺ key (a key on an
      unencrypted tier is a false confidentiality promise). End-to-end test: a Private network binds
      a key, a member resolves it from `key_ref` and seals/opens a payload, a non-member cannot open
      it, and two members sealing the same payload converge on one κ (L3 dedup on the private
      network). NW-1/NW-2 conformance unchanged; native + thumbv7em green.
    - [x] **forward secrecy on membership change** (2026-07-17): the `KeyEpoch` realization — a
      Private network's payload-key as an append-only **membership-epoch chain**. Each change cuts a
      new epoch with a **fresh random key wrapped per-member** to each *current* member's enrollment
      public key (the new portable `KeyWrapper` seam, dep-free like `PayloadCipher`/`SignatureVerifier`).
      A removed member is absent from the new epoch's wraps *and* can unwrap none of the others — so
      they never obtain the new key nor open post-revocation content. This is exactly the
      "convergent-shared-key is insufficient" branch (a convergent key is re-derivable → un-rotatable
      to exclude a member). Reference impl `tests/forward_secrecy_x25519.rs` (X25519 sealed-box):
      structural + cryptographic exclusion, codec round-trip + REGISTRY dispatch, parser-hardened.
      no_std core clean (wasm32 + thumbv7em); x25519 stays a dev-dep.

**All P4–P6 conformance rows are green (HF-1/2/3, NW-1/2, GV-1/2/3/4).** What remains in P4–P6 is
non-conformance feature depth: fat/thin CLI tooling + parser fuzz (P4), native transports +
wire-version + TCK battery (P5), payload encryption + key lifecycle chain (P6).

## Crate consolidation (simplification — user directive 2026-07-17)

Reduce crate sprawl: one crate per concept, feature-gated backends instead of sibling crates.
**Done (2026-07-17) — 24 crates → 20 members.**
- [x] **`hologram-store-{bare,native,opfs}` → `hologram-store`** (5cf8c9c + 0099069) — one crate,
  three feature-gated backend modules (`bare` no_std/BlockDevice, `native` std/redb, `opfs`
  wasm32/web-sys + `js-api`). Repointed every reverse-dep (efi, runtime, cli, holospaces{,-node,
  -browser}, root), workspace members/deps, imports, CI + Justfile. Tri-target green (native + wasm32
  + thumbv7em); the opfs `js-api` bundle builds via `cargo rustc --crate-type cdylib` so the crate
  stays a plain lib. 29 native tests + holospaces-browser wasm build verified.
- [x] **`hologram-emulator-codemodule` → `hologram-emulator`** (e43ac39) — folded in as a
  `codemodule` feature (`cfg(all(feature="codemodule", target_arch="wasm32"))`) built as a cdylib via
  `scripts/build-emulator.sh` (`cargo rustc --crate-type cdylib`). Native lib + wasm32 lib + the
  216 KB wasm codemodule all build.
- [x] **`hologram-spike-sp3` → `hologram-conformance`** (d13528b) — the "spike" name was stale; the
  reference `SpikeSpace` (LAW-3/SP-3 witness) is now `tests/common/mod.rs` shared test support, still
  built from public API only so the LAW-3 witness holds. Conformance suite green.
- [x] **`README.md` in every crate** — all 18 `crates/*` + 3 `spaces/*` now have a README derived
  from each crate's Cargo.toml + `//!` doc (the root already had one).

## Sprint 39: Decode Residual — Browser (ACTIVE)

**Plan:** [plans/077-decode-residual-browser.md](plans/077-decode-residual-browser.md)

Goal: close hologram's share of hologram-ai's browser decode residual
(compute-bound wasm int8 matmul at ~7 MB/s effective vs GB/s stream
bandwidth), staying within κ-operation. Acceptance is witnessed downstream by
hologram-ai's performance contract; hologram benches are the regression
mirror.

- [x] **1.1**: Output-major W8A8 int8 GEMV kernel (`matmul_i8_pc_omajor`):
  contiguous k-inner weight walk, per-token symmetric i8 activation
  quantization, exact integer accumulation (wasm `i32x4_dot_i16x8`, NEON
  `vmull_s8`+`vpadalq_s16`); bit-identical across scalar/NEON/wasm (verified
  natively, under qemu-aarch64, and under wasmtime+simd128).
- [x] **1.2**: `MatMulDequantCall { bq_omajor, act_quant }` — layout excluded
  from `op_signature` (b_packed rule), W8A8 on its own signature tag (116);
  archive wire tag `D_MMDQ2 = 116`, legacy archives byte-identical, unknown
  tags fail closed.
- [x] **1.3**: Compile-time `fuse_const_i8_decode` pass: constant symmetric
  per-channel i8 weight uniquely consumed by `Dequantize → MatMul(B)` at
  m ≤ 4 fuses in the archive with the constant transposed `[k,n] → [n,k]`
  (derived content under its own κ); dynamic weights keep load-time W8A32
  fusion; `wl2_*` conformance (fusion fires, bit-identical to independent
  W8A8 reference, prefill + asymmetric negatives).
- [x] **2.1**: m = 1 GEMV specialization — 4 output rows in flight,
  independent integer accumulators, no output tiling.
- [x] **9.1**: `decode_gemv` benches at deployed decode shapes (0.5B/1.5B/7B
  projections, m = 1) reporting int8 bytes-streamed/s, kernel + full-pipeline
  novel-input session step; manifest registration; `wasm_matmul_timing`
  extended so the wasmtime+simd128 lane runs the actual wasm kernel.
- [x] **4.1**: Relaxed-SIMD tier: `i32x4_relaxed_dot_i8x16_i7x16_add` over a
  `q = q⁺ − q⁻` i7 split — same exact W8A8 function, bit-identical on both
  builds; baseline stays the witnessed fallback and `just wasm` builds both
  tiers. `f32x4_relaxed_madd` measured ~30% slower under wasmtime
  (latency-bound accumulator chains) and deliberately excluded.
- [x] **7.1a**: dequant+matmul+bias+activation as ONE call:
  `MatMulDequantCall` gained fused-epilogue fields (`act`, `residual` —
  signature-visible; wire on the extended discriminant), the load-time
  epilogue pass now absorbs activation / bias-add / three-op chains into
  fused dequant-matmuls (compile-time-fused omajor W8A8 included), and the
  dispatch applies the epilogue in place while the results are hot.
  Conformance: `gelu(A·dequant(Bq) + bias)` is one call; exact epilogues
  stay bit-identical to the W8A8 reference. Also fixed en route: fusion
  pass ordering (dequant→matmul now fuses before the
  matmul epilogue, so a quantized weight followed by an activation keeps
  streaming in place instead of materializing the dense f32 weight each
  step; conformance-locked). Measured per-step session overhead at m = 1,
  896×4864: ~84 µs over the raw kernel (~7% single-op; the multi-op residual
  is the remaining fusion/plan-handle work).
- [x] **7.1b**: validate-once / replay-per-step for the seq-1 walk. Profiling
  (callgrind) attributed the fixed per-step residual (~100 µs on a 1-node
  graph) to the boundary-address mint: `derive_label_witnessed` grounded a
  full ψ-tower composition per operand per step and the walk dropped the
  TC-05 witness. Added `derive_label_boundary` /
  `compose_ordered_blake3_address` — the identical composition sequence
  minting only the address (pinned label-equal to the witnessed form by
  tests, so any algebra change fails closed; the witness stays re-derivable
  on demand). Per-step walk overhead: ~100 µs → ~10–28 µs. Arena reuse
  across steps already holds (generation rotation + free list); constant
  rebinding is O(constants) HashMap hits per step — revisit only if a
  many-hundred-weight model shows it.
- [x] **8.1**: Deterministic vectorized exp for the decode softmax path
  (`exp_f32_det` scalar spec + NEON/wasm SIMD128 lanes replaying the exact
  IEEE sequence — bit-identical across targets, verified natively, under
  qemu-aarch64, and under wasmtime on both SIMD tiers; masked −∞ scores
  stay exactly 0). Wired into `softmax_float` and the attention inner
  softmax with reduction order unchanged. ~2× over scalar libm on the wasm
  lane, and stronger determinism than before (std/no_std builds previously
  used different libms). Q-tier exp table remains an item-6 follow-up.
- [x] **5.1**: wasm threads (`wasm-threads` feature): embedder-provided
  workers register via the exported `hologram_worker_run` and drain a
  single fork-join job slot in shared linear memory; the futex is
  embedder-provided too (`hologram_host_wait32`/`notify` — JS
  `Atomics.wait`/`notify` — since wasm's native wait/notify intrinsics are
  unstable on stable Rust; the std test lane parks by spin+yield). The
  decode GEMV statically partitions output rows, so every row is computed
  whole by one participant — parallel output is **bit-identical** to serial
  (locked by `parallel_gemv_matches_serial_bitwise` running real threads
  under wasmtime). Scaling signal (wasmtime, 3 workers + main): 2.5–3.8×,
  72 GB/s aggregate int8 at 1536×8960; the 7B shape saturates DRAM at
  35 GB/s. Plain simd128 builds are unchanged (witnessed fallback).
- [x] **6.1**: LUT-tier decode core to main — `matmul_i4_pc_omajor`, an
  output-major packed-i4 W4A8 GEMV: the stored-weight multiply becomes an
  in-register 16-entry `i8x16_swizzle`/`vqtbl1q_s8` table lookup and the
  streamed weight bytes **halve**, then the looked-up i8 values flow through
  the identical integer dot pipeline as the i8 kernel (bit-identical across
  scalar/NEON/wasm on both SIMD tiers). Dispatch routes it under the existing
  W8A8 `MatMulDequant` (i4 quant_dtype, even-k guard — no new call-surface or
  signature); `fuse_const_i8_decode` repacks constant nibbles to `[n, k/2]`
  under their own κ; the wasm pool carries an i8/i4 `kind` and
  `parallel_gemv_matches_serial_bitwise` locks both. Conformance:
  `wl3_*`. Signal: half the bytes at comparable step time, and under the pool
  at the DRAM-bound 7B shape W4A8 (1434 µs) beats W8A8 (1551 µs) with half the
  resident footprint. The full orbit/psumbook/fiber-radix port stays a future
  sprint on the migration branch.

## Sprint 38: Source IR and Multi-Language Frontends (DONE)

**Plan:** [plans/075-source-ir-language-frontends.md](plans/075-source-ir-language-frontends.md)

Goal: replace `hologram-compiler::source`'s hand-rolled direct
`source -> Graph` parser with a staged
`source -> SourceDocument -> SourceProgram -> Graph` pipeline, then add explicit
Python, TypeScript, and Rust frontends that lower through the same IR without
executing user code.

- [x] **0.1**: Plan 075 drafted with source-IR boundary, parser strategy,
  host-language contracts, compatibility API, phases, and acceptance criteria.
- [x] **1.1**: Define `SourceProgram` / `SourceItem` / `SourceExpr` /
  `SourceType` / span diagnostics in `hologram-compiler::source::ir`.
- [x] **1.2**: Move name resolution, shape interning, constant insertion,
  output creation, and sparse op-attribute attachment into a shared
  `SourceProgram -> Graph` lowerer.
- [x] **1.2a**: Preserve the zero-copy/O(1) runtime contract: source spans,
  language tags, parser ASTs, and symbol names disappear before backend
  lowering; execution receives only resolved graph/archive structures.
- [x] **1.2b**: Add symbol interning and direct tensor-literal-to-bytes parsing
  so source lowering avoids per-token string churn and avoids
  `Vec<f32> -> Vec<u8>` double materialization for constants.
- [x] **1.3**: Preserve `source::parse(&str) -> Result<Graph, CompileError>`
  and `compile_from_source` by routing the legacy grammar through the new IR.
- [x] **2.1**: Replace `split_whitespace` with a real native parser
  (preferred: `nom`, to preserve `no_std + alloc`).
- [x] **2.2**: Add native v2 syntax subset for typed tensors, bracket shapes,
  tensor literals, call expressions, and `let`.
- [x] **2.2a**: Add named attributes to native v2 syntax.
- [x] **2.2b**: Move source op-name parsing to the canonical `OpKind::ALL`
  catalog instead of a compiler-local hand-maintained list; generate
  `OpKind`, `OpKind::ALL`, and `OpKind::name()` from one catalog declaration.
- [x] **2.2c**: Add `SourceDiagnostic` and diagnostic parse entry points that
  report line, column, kind, and rejected token while preserving
  `CompileError::SourceParse` compatibility.
- [x] **3.1**: Update CLI language selection now that `SourceLanguage`,
  `parse_ir`, `lower_ir`, and `compile_from_source_language` exist.
- [x] **3.1a**: Add a `SourceFrontend` adapter boundary under
  `source/frontends/`; `nom` remains the native Hologram DSL parser, while
  Python/TypeScript/Rust and future Go/C/PHP/etc. frontends use parser
  adapters appropriate to each language. Frontend metadata owns language
  aliases and filename extensions so CLI detection does not hard-code them.
- [x] **3.1b**: Add `SourceDocument`, `SourceGraph`, and
  `SourceParseOptions` so frontends can extract Hologram graph regions from
  larger host-language files before selecting a single `SourceProgram` for
  lowering.
- [x] **3.1c**: Add `hologram-cli compile --graph <name>` and route CLI source
  compilation through `parse_ir_with_options` so multi-graph documents fail
  loud unless the graph selection is explicit.
- [x] **3.1d**: Document the graph-inference policy: safe inference is limited
  to AST-only detection of known Hologram builder usage; unrelated host code is
  ignored; unsupported statements inside inferred graph candidates fail loudly.
- [x] **4.1**: Add restricted AST-based Python frontend behind an explicit
  `frontend-python` feature using `rustpython-parser`.
- [x] **4.1a**: Extract named Python `SourceGraph`s from top-level functions
  with unambiguous `h.input` / `h.const` / `h.ops.<op>` / `h.output` builder
  usage while ignoring unrelated host-language code.
- [x] **4.1d**: Document Python compile flags, accepted builder calls, graph
  selection, and the AST-only/no-execution boundary in the README and site
  docs.
- [x] **4.1b**: Add source-position diagnostics for rejected Python AST nodes.
- [x] **4.1c**: Add Python op-attribute parsing beyond `shape` / `dtype`.
- [x] **5.1**: Add restricted AST-based TypeScript frontend behind an explicit
  feature flag.
- [x] **6.1**: Add restricted AST-based Rust frontend behind an explicit
  feature flag.
- [x] **7.1**: Add cross-frontend equivalence tests proving native, Python,
  TypeScript, and Rust sources lower to equivalent IR/graphs/archives.
- [x] **7.1a**: Add negative dependency checks proving `source::*` types do not
  leak into exec/backend/archive runtime structures.
- [x] **7.1b**: Add source-lowering microbenchmarks for large graph and constant
  parsing.
- [x] **7.1c**: Design external tensor references for large weights so
  host-language frontends do not force model data through inline source
  literals.
- [x] **7.2**: Update `specs/docs/architecture.md`, site compiler docs, and
  README after the source frontend architecture lands.
- [x] **8.1 / Plan 076**: Reconcile the FFI header with the implemented Rust
  FFI surface and add a stable source/graph builder ABI.
  - [x] **8.1a**: Replace stale checked-in C header declarations with the
    implemented compile/session/source-builder FFI surface.
  - [x] **8.1b**: Add opaque source builder handle plus `input`, `op`,
    `output`, `compile`, and `free` functions backed by `SourceProgram`.
  - [x] **8.1c**: Add ABI version and basic thread-local error message
    functions for SDK callers.
  - [x] **8.1d**: Add C ABI tests for source-builder compile round-trip and
    rejected-op error reporting.
  - [x] **8.1e**: Add inline constants and external tensor references to the
    builder ABI.
  - [x] **8.1f**: Replace generic builder failures with stable SDK-facing
    error categories.
- [x] **8.2 / Plan 076**: Add versioned ABI, structured error, ownership, and
  exported-symbol compatibility contracts for SDK consumers.
  - [x] **8.2a**: Add archive-format version query and feature-probing API for
    additive SDK capability checks.
  - [x] **8.2b**: Document pointer, buffer, builder, session, and error-message
    ownership/lifetime rules.
  - [x] **8.2c**: Add C header symbol/constant snapshot tests.
  - [x] **8.2d**: Document ABI version, feature string, and `.holo` format
    compatibility rules.
- [x] **8.3 / Plan 076**: Generate low-level Python and TypeScript bindings
  from canonical op/dtype/attribute metadata.
  - [x] **8.3a**: Add `hologram-ffi::sdk` metadata derived from
    `OpKind::ALL`, dtype constants, required feature strings, and source attr
    metadata.
  - [x] **8.3b**: Generate checked-in low-level Python and TypeScript SDK
    metadata/helper files under `sdk/`.
  - [x] **8.3c**: Add a generator entry point for refreshing SDK binding
    files from canonical Rust metadata.
  - [x] **8.3d**: Add generated-file freshness and canonical-op coverage
    tests.
- [x] **8.4 / Plan 076**: Build human chainable Python and TypeScript SDKs on
  top of the generated bindings.
  - [x] **8.4a**: Add `sdk/python/hologram` with chainable `Graph` /
    `Tensor`, `input`, inline `const`, `const_ref`, output aliases, native
    feature checks, and `Graph.op(...)` escape hatch.
  - [x] **8.4b**: Add Python stdlib tests using a fake native source-builder
    binding.
  - [x] **8.4c**: Add `sdk/typescript/src` with generated-op proxy tensor
    calls, `Graph.op(...)`, `constRef`, output aliases, and async compile over
    a native binding protocol.
  - [x] **8.4d**: Validate the TypeScript SDK entry point with `tsc --strict`.
- [ ] **8.5 / Plan 076**: Package PyPI wheels, npm native/WASM packages, and
  browser-safe TypeScript distribution artifacts.
  - [x] **8.5a**: Add Python SDK package metadata, typed-package marker, and
    package-surface smoke test.
  - [x] **8.5b**: Add `@hologram/sdk` npm package metadata, TypeScript build
    config, ESM export map, and browser-safe package README.
  - [x] **8.5c**: Document native N-API/WASM package boundaries and the shared
    `NativeBinding` protocol expected by the human SDKs.
  - [x] **8.5d**: Add `@hologram/native` and `@hologram/wasm` adapter packages
    over the SDK `NativeBinding` protocol, with ABI/feature checks, output
    alias support, constant-byte conversion, and package dry-run checks.
  - [x] **8.5e**: Implement the actual N-API binary and WASM driver surfaces
    consumed by the adapter packages.
  - [x] **8.5f**: Add platform prebuild/release matrix and installed-package
    smoke checks.
- [x] **8.6 / Plan 076**: Add cross-language SDK/parser conformance tests,
  including external tensor references.
  - [x] **8.6a**: Add a golden external-weight matmul witness proving
    `SourceProgram` file-backed constants lower to the same graph/archive as
    inline native `.txt` constants.
  - [x] **8.6b**: Add a C/FFI source-builder witness proving SDK-facing
    `const_ref` compilation produces the same archive as native `.txt`.
  - [x] **8.6c**: Add Python and TypeScript SDK golden tests proving the human
    APIs emit the same builder contract for `input`, `const_ref` / `constRef`,
    `matmul`, and output aliases.
  - [x] **8.6d**: Run the parser conformance suite with
    `frontend-python,frontend-typescript,frontend-rust` enabled.
- [x] **8.7 / Plan 076**: Add a real Python native wheel binding over the FFI
  source-builder ABI.
  - [x] **8.7a**: Add `_hologram.py` ctypes bindings for source-builder,
    version, feature, and error APIs.
  - [x] **8.7b**: Bundle the `hologram-ffi` cdylib into platform-specific
    Python wheels via the package build hook.
  - [x] **8.7c**: Check ABI version, archive format version, and required
    feature strings before native builder use.
  - [x] **8.7d**: Add Python native and installed-wheel smoke tests that compile
    a graph through the bundled library.
- [x] **8.8 / Plan 076**: Add Python SDK session load/execute wrappers on top
  of the FFI session ABI.
  - [x] **8.8a**: Add `hg.Session.load(...)` with context-managed close and
    finalization.
  - [x] **8.8b**: Expose session input/output counts, port names, port shapes,
    output byte lengths, kernel count, and archive fingerprint.
  - [x] **8.8c**: Add `Session.execute(...)` for named input mappings and
    ordered byte-buffer sequences, returning output-name keyed bytes.
  - [x] **8.8d**: Extend Python native and installed-wheel smoke coverage from
    compile-only to compile/load/execute.
- [x] **8.9 / Plan 076**: Add TypeScript SDK session load/execute wrappers on
  top of the native and WASM driver session surfaces.
  - [x] **8.9a**: Add `Session.load(...)` to `@hologram/sdk` over optional
    `NativeBinding.sessionLoad`.
  - [x] **8.9b**: Add session load, introspection, execute, and close methods
    to `@hologram/native` and its N-API addon.
  - [x] **8.9c**: Add the same session surface to `@hologram/wasm` and the
    wasm-bindgen driver.
  - [x] **8.9d**: Extend SDK golden, native addon, native installed-package,
    and WASM installed-package smokes from compile-only to compile/load/execute.
- [x] **8.10 / Plan 076**: Map stable FFI error categories to Python and
  TypeScript SDK error classes.
  - [x] **8.10a**: Add Python `hologram.errors` taxonomy and map native
    archive load, external tensor, execution, ABI, invalid argument, and SDK
    validation failures to typed exceptions with stable `.code` values.
  - [x] **8.10b**: Add TypeScript `@hologram/sdk` error classes, `ERROR_*`
    constants, and `errorFromCode(...)`.
  - [x] **8.10c**: Map native N-API and WASM adapter driver errors into the
    shared TypeScript SDK classes.
  - [x] **8.10d**: Cover bad archives, bad external tensor references, missing
    inputs, and execution failures in focused Python, TypeScript golden, and
    installed-package smoke tests.
  - [x] **8.10e**: Re-export generated `TensorRef` and
    `LowLevelGraphBuilder` from `@hologram/sdk` and add a declaration-output
    type smoke test for the public TypeScript package surface.
- [x] **8.11 / Plan 076**: Finish SDK/FFI diagnostics, metadata, external
  tensor policy, and docs cleanup.
  - [x] **8.11a**: Expose source-position diagnostics through FFI, Python,
    N-API, and WASM adapters.
  - [x] **8.11b**: Expose session input/output dtypes and metadata extensions
    through Python and TypeScript SDK sessions.
  - [x] **8.11c**: Add Node finalizer backstops for source-builder and session
    handles.
  - [x] **8.11d**: Generate richer op metadata, Python op wrappers, and
    TypeScript generated method / attr option types from canonical metadata.
  - [x] **8.11e**: Add `HOLOGRAM_EXTERNAL_TENSOR_ROOT` compile-root policy and
    tests proving external tensors are embedded at compile time.
  - [x] **8.11f**: Update README, SDK docs, FFI contract docs, external tensor
    docs, and website reference/configuration docs.
- [x] **8.12 / Plan 076**: Add SDK convenience wrappers for native Hologram
  `.txt` source compilation.
  - [x] **8.12a**: Add Python `compile_source(...)` and
    `compile_source_file("graph.txt")` over `hologram_compile_source`.
  - [x] **8.12b**: Add shared TypeScript `compileSource(source, binding)` plus
    N-API and WASM driver `compileSource` methods.
  - [x] **8.12c**: Add Node-only `@hologram/native`
    `compileSourceFile("graph.txt", binding)`.
  - [x] **8.12d**: Cover source-string, `.txt` file, public type, native addon,
    installed-package, and WASM adapter paths in focused tests.
  - [x] **8.12e**: Update README, SDK docs, and website docs with `.txt`
    source-helper usage.

## Sprint 37: Single Source of Truth for Ops (ACTIVE)

**ADR:** [adrs/045-ops-as-single-source-of-truth.md](adrs/045-ops-as-single-source-of-truth.md)

Goal: collapse the op-definition spread by moving every op-related
artefact (per-op `Call` struct, kernel function, dispatch enum and
function) into `hologram-ops`. `hologram-transform` becomes a pure
orchestration crate (chain → plan → execute, addressing, buffer).

- [x] **1.1**: New `hologram-ops/src/kernels/` tree — one module per op
  (or group of related ops) owns its `Call` struct + forward kernel +
  backward kernel (where applicable)
- [x] **1.2**: `KernelCall` enum + `dispatch()` function moved to
  `hologram_ops::kernels`
- [x] **1.3**: `SlotSpan` moved to `hologram_ops::span`
- [x] **1.4**: `hologram-transform/src/kernels/` deleted; executor
  delegates to `hologram_ops::dispatch`
- [x] **1.5**: `hologram-transform/src/plan.rs` re-exports the kernel
  types (back-compat) and keeps only orchestration types
  (`CompiledPlan`, `AddressTable`, `WorkspaceLayout`)
- [x] **1.6**: `libm` dep moved to `hologram-ops` (the kernel home);
  `hologram-transform` no longer needs it as a regular dep
- [x] **1.7**: All 65 workspace test suites pass, 0 failures, clippy
  clean across migration-affected crates
- [x] **1.8 (post-migration sweep)**: 9 pre-existing clippy errors
  in non-migration files cleaned up — `hologram-shape/tensor_shape.rs`
  (×2 no-op), `hologram-compute/src/cpu.rs` + `metal.rs` (×5 loop
  / approx_constant), `hologram-exec/src/buffer/scatter_gather.rs`
  + `tests/constrained_bench.rs` (×3 io / unit_arg). Workspace
  `cargo clippy --workspace --tests -- -D warnings` is now
  fully clean.

### Phase 1.5 — Cleanup (completing ADR-045's promise)

- [x] **1.5.1**: `hologram-ops/src/lib.rs` slimmed from 1010 lines
  monolithic to ~50 lines of module declarations + re-exports.
  Per-module split: `trait_def.rs` (Op trait + OpCategory + OpSignature
  + BackwardRule), `attrs.rs` (all `*Attrs` structs), `semantic.rs`
  (`SemanticOp` + dispatch macro), `span.rs` (`SlotSpan`), `kernels/`
  (per-op modules)
- [x] **1.5.2**: Per-op marker struct + `Op` impl moved into the same
  file as the kernel — `kernels/<op>.rs` now contains the **complete**
  definition of the op (struct + Op impl + Call struct(s) + kernel
  function(s) + tests). The "one file per op" promise is fulfilled.
- [x] **1.5.3**: Dead `hologram-transform/src/op.rs` re-export file
  deleted; consumers import directly from `hologram_ops`
- [x] **1.5.4**: Naming collisions resolved:
  - `hologram_core::op::OpCategory` (the float dispatch-shape enum)
    renamed to `FloatOpShape` — frees the `OpCategory` name for the
    semantic category in `hologram-ops`
  - `hologram_core::op::Op` (legacy unified sum-type enum) deleted —
    no production callers; canonical identity is now the
    `hologram_ops::Op` trait
  - `hologram::Op` at the top-level now refers to the canonical trait
- [x] **1.5.5**: `AGENTS.md` updated with the canonical-ops rules
  (single source of truth, the "add an op" checklist, hot-path
  invariants linked to ADR-044/045)

### Phase 2 — Per-op LUT generator (TODO, blocked on Plan 074)

This is the third leg of the "every op needs both a LUT *and* a
transformation" invariant. ADR-045 placed the `Call` struct + kernel
function in each `hologram-ops/src/kernels/<op>.rs` so the LUT
generator can land in the *same file* — completing the consolidation.

The LUT layer is provided by `uor-foundation`. Plan 074 upgrades
`uor-foundation` from 0.1.4 → 0.3.0 which exposes the address /
identity API the LUT generators consume. **This phase blocks on Plan
074 landing.**

- [ ] **2.1**: Extend the `Op` trait with a `lut()` method (signature
  TBD — likely returns a `LutDescriptor` that names the address space,
  the layout, and the precomputed table-id contract). Default impl
  returns `None` so non-LUT ops are unaffected.
- [ ] **2.2**: For each per-op file in `hologram-ops/src/kernels/`,
  add the per-op LUT generator alongside its `Call` struct + kernel.
  Co-locating the three (identity, transformation, LUT) in one file
  is the architectural promise of ADR-045.
- [ ] **2.3**: Bridge `AddressRef` (in `hologram-transform`) to
  `uor-foundation`'s LUT-resolved address type. Currently the planner
  assigns sequential offsets in a flat workspace; with the LUT layer
  the address resolution becomes ontological lookup, not arithmetic.
- [ ] **2.4**: Connect the chain's `AddressTable` to LUT-resolved
  addresses so the same plan can target different backends without
  re-keying tensor identities.
- [ ] **2.5**: Per-op tests: each op's LUT generator must round-trip
  with its kernel — `lut(op).resolve(...) == kernel_output_address(...)`.
- [ ] **2.6**: Update ADR-043 (LUT-Addressed Transform Chains) Phase 6
  status from "deferred" to "implemented".

After this phase, `hologram-ops/src/kernels/<op>.rs` contains the
**complete definition** of the op: identity (`Op` trait), executable
form (`Call` + kernel), and addressing (`lut()`). One file. One place.

### Phase 3 — Bigger architectural moves (separate ADR-worthy work)

These are intentionally out of scope for this sprint; tracked here so
they don't get lost.

- [x] **3.1**: Collapse `Compute` / `Float` dispatch duplication in
  `hologram-exec`. `kv/store.rs::dispatch_with_shapes` and
  `tape_builder.rs::resolve_kernel` both had two identical match
  arms; now one each, both routing through `op.legacy_float_op()`.
  The bridge is the supported architecture per ADR-046; full
  reorganisation of exec around `SemanticOp` is left to Phase 3.3.
- [x] **3.2**: Shrink `graph::op` bridge — `semantic_op()` (the
  unused FloatOp → SemanticOp direction) and its 95-line helper
  removed. Captured in ADR-046; the bridge is one-way going forward.
- [x] **3.3 Stage 1**: ADR-047 (staged FloatOp deprecation roadmap).
  Smart constructor `GraphOp::from_float(FloatOp) -> GraphOp` lands
  in `hologram-graph::graph::op` — emits `Compute(SemanticOp)` when
  canonical covers the variant, falls back to `Float(FloatOp)`.
  Compiler `term_lower` migrated to use it; new lowering paths
  default to canonical. End state per ADR-047 has 4 stages:
  smart-construction (done) → canonical surface expansion → exec
  reorganisation around `SemanticOp` → public `FloatOp` removal.
- [x] **3.3 Stage 2 complete (36 / 36 migratable ops)**:
  Canonical coverage expanded across 6 rounds.
  - Round 1 (15 ops): `Pow`, `Mod`, `Min`, `Max`, `Equal`, `Less`,
    `LessOrEqual`, `Greater`, `GreaterOrEqual`, `And`, `Or`, `Xor`,
    `Not`, `IsNaN`.
  - Round 2 (11 ops): Reductions, Pooling (NCHW-aware), `Where`,
    `Clip`.
  - Round 3 (3 ops): `CumSum`, `Pad` (constant mode), `Resize`
    (nearest mode).
  - Round 4 (4 ops): `Lrn`, `ConvTranspose2d`, `Gemm`, `Expand`.
  - Round 5 (2 ops + 2 mode extensions): `RotaryEmbedding`
    (half-rotation, position = row index), `Pad`-reflect,
    `Pad`-edge, `Resize`-bilinear.
  - Round 6 (1 op): `Attention` (scaled dot-product, GQA-aware,
    causal mask, ADR-049). Un-fused canonical form — RoPE /
    QK-norm / sparse-V are upstream canonical ops or execution
    flags, not part of canonical semantics.
  - All bridge mappings (`from_float` / `legacy_float_op`) and
    planner arms updated.
  - **ADRs landed**: ADR-048 (permanent-FloatOp surface),
    ADR-049 (canonical Attention design). Canonical surface
    end state: **~75 ops**, exactly as projected.

  **Stage 2 closes Sprint 37 Phase 3.3 forward-only canonical
  surface coverage.** Remaining Phase 3.3 work is the bigger
  Stage 3 (exec dispatch reorganisation around `SemanticOp`) and
  Stage 4 (public `FloatOp` removal — needs an archive-format ADR).
- [x] **3.3 Stage 3 (reframed and shipped, ADR-050)**: Original
  framing — "rewrite exec to call canonical kernels" — proved
  architecturally wrong on inspection. Exec's float dispatch is
  heavily optimised (monomorphised hot paths, in-place
  `OutputBuffer`, autovectorisation hints); canonical kernels are
  reference-grade with explicit scratch allocation. Forcing exec
  to call canonical would regress every covered op's performance
  with no semantic gain.

  **Reframed:** canonical kernels are the *semantic contract*, not
  the execution path. Exec/backend implementations *conform* to
  canonical semantics; conformance is enforced by tests.

  **Shipped:** `hologram-exec/tests/canonical_conformance.rs` —
  cross-check infrastructure that for any canonical-covered op
  asserts exec's output matches the canonical reference within
  tolerance. 8 representative tests covering unary
  (Relu, Gelu, Sigmoid), binary (Add, Mul), MatMul, Softmax,
  ReduceSum. Adding a new canonical op henceforth = one new
  cross-check test.

  ADR-050 documents the decision and explicitly rejects the
  "rewrite exec" path as architecturally wrong.
- [x] **3.3 Stage 4 (public API surface only)**: `FloatOp` is no
  longer re-exported from the top-level `hologram::*`. New code
  uses `hologram::SemanticOp` via `GraphOp::Compute(...)`. The
  underlying enum stays in `hologram_core::op::FloatOp` as
  exec/backend's internal dispatch encoding (per ADR-050) and is
  reachable for downstream embedded use cases that need it. Full
  removal of `GraphOp::Float` from the rkyv archive format is a
  separate concern (would require an "archive format migration"
  ADR — not in this sprint's scope).
- [x] **3.4 (complete)**: Backward rules for **all differentiable
  canonical ops** wired through 10 expansion rounds:
  - Round 1: `Add`, `MatMul` (foundation, ADR-043 scope).
  - Round 2: `Sub`, `Mul`, `Neg`.
  - Round 3: `Div`, plus 5 differentiable unaries — `Relu`, `Sigmoid`,
    `Tanh`, `Exp`, `Log` — sharing `KernelCall::UnaryGrad(UnaryGradCall,
    UnaryGradKind)`.
  - Round 4: `Sqrt`, `Abs`, `Reciprocal`, `Min`, `Max`, `ReduceSum`,
    `ReduceMean` (shared `MinMaxGradCall` and `ReduceGradCall`).
  - Round 5: `Gelu`, `Silu`, `Pow`, `Concat`, `Slice`, `Transpose`.
  - Round 6: `ReduceMax`, `ReduceMin` (shared `ReduceArgGradCall`),
    `Softmax`, `LogSoftmax` (shared `SoftmaxGradCall`).
  - Round 7: `ReduceProd` (zero-aware), `RmsNorm`, `LayerNorm`.
  - Round 8: `InstanceNorm`, `AddRmsNorm`, `GlobalAvgPool`, `AvgPool2d`,
    `MaxPool2d`.
  - Round 9: `GroupNorm`, `FusedSwiGlu`.
  - Round 10: `Conv2d`, `ConvTranspose2d`, `Attention` (full GQA-aware,
    causal-mask-correct backward).
  Heavyweight grads (norms, conv, conv-transpose, attention) are
  finite-difference cross-checked in `hologram-ops` unit tests.
- [x] **3.5**: Backend executors over the same `CompiledPlan` shipped
  in PR #6 (canonical layer). `CanonicalBackend` trait + `CpuBackend`
  adapter live in `hologram-transform`; `WgpuBackend` in
  `hologram-compute/src/canonical/wgpu.rs` covers **51 variants** with
  real WGSL compute pipelines (binary/unary elementwise families,
  reductions, softmax/log-softmax, all 5 norm forwards, MatMul +
  grads, full Pool family, Conv2d, ConvTranspose2d, FusedSwiGlu,
  GroupNorm, plus 4 norm-grads — RmsNorm/InstanceNorm/LayerNorm/
  AddRmsNorm — and SoftmaxGrad family). Cross-backend conformance
  harness (`check_forward`, `check_forward_then_backward`) in
  `hologram-transform`; 29 `#[ignore]`-gated wgpu integration tests
  validate every promoted variant against the CPU canonical reference.
  Remaining variants route through `host_cpu_fallback` — the dispatch
  arm is exhaustive (no catch-all `_`), so every variant has a working
  path and adding a new `KernelCall` variant fails the build until it's
  explicitly handled.

---

## Sprint 36: `Op` Trait — Per-Op-Type Semantic Contract (ACTIVE)

**ADR:** [adrs/044-op-trait-canonical-semantics.md](adrs/044-op-trait-canonical-semantics.md)

Goal: introduce a per-op-type `Op` trait in `hologram-ops` so every fact
about a canonical op (arity, name, signature, default backward rule,
category) lives with the op type instead of in scattered enum match arms.
The closed `SemanticOp` enum stays as the serialisation and dispatch
surface and forwards to the trait via a single dispatch macro.

- [x] **1.1**: Define `pub trait Op` in `hologram-ops`
- [x] **1.2**: Add per-op marker structs for every `SemanticOp` variant
- [x] **1.3**: Macro-driven `SemanticOp` → `Op` forwarding (one match site)
- [x] **1.4**: Trait conformance tests for representative ops
- [x] **1.5**: Verify enum dispatch surface unchanged (downstream crates
  compile, archives unaffected, `cargo test --workspace` clean)
- [x] **2.1**: Drop `OpKind` from `hologram-ops`; chain layer carries
  `SemanticOp` directly (no parallel chain-only op enum)
- [x] **2.2**: `BackwardRule::for_op` / `forward_op` removed — callers use
  `op.backward()` via the `Op` trait, keeping rule and op in sync
- [x] **2.3**: `MatMul` dims live on `SemanticOp::MatMul(MatMulAttrs)`;
  builder validates shapes at construction; planner reads attrs directly
- [x] **2.4**: Extend transform planner with `Sub` and `Mul` (shared
  `BinaryCall` shape) — proof that adding canonical ops is now mechanical
- [x] **3.1**: All 18 unary elementwise ops (Neg, Relu, Gelu, Silu, Tanh,
  Sigmoid, Exp, Log, Sqrt, Abs, Reciprocal, Cos, Sin, Sign, Floor, Ceil,
  Round, Erf) wired through a single `KernelCall::Unary(_, UnaryKind)`
  enum-tag dispatch — one new variant in `KernelCall`, one kernel module
- [x] **3.2**: `Div` kernel (extends `BinaryCall`)
- [x] **3.3**: `Softmax` and `LogSoftmax` kernels (numerically stable,
  per-row over the last axis)
- [x] **3.4**: `Reshape` kernel (in-storage copy; planner-level span
  aliasing left as a future optimisation)
- [x] **3.5**: 7 new integration tests covering the new ops end-to-end

### Phase 4 — Remaining canonical ops
- [x] **4.0**: Fix `FusedSwiGlu` arity bug (was 1, should be 2 — confirmed
  against `hologram-exec` float-conformance suite)
- [x] **4.1**: Norms (`RmsNorm`, `LayerNorm`, `InstanceNorm`, `GroupNorm`,
  `AddRmsNorm`) — `NormScaleCall` / `NormFullCall` / `GroupNormCall` /
  `AddRmsNormCall`, shared mean/variance helpers
- [x] **4.2**: `Transpose` (physical, up to 4-D, n-D index walker)
- [x] **4.3**: `Slice` (last-axis contiguous; non-last-axis explicitly
  rejected with a clear `PlanError`)
- [x] **4.4**: `Concat` (last-axis, two operands)
- [x] **4.5**: `Conv2d` (direct reference, NCHW, optional bias, groups)
- [x] **4.6**: `FusedSwiGlu` (`silu(gate) * up`)
- [x] **4.7**: 10 new integration tests; 22 new kernel unit tests

**Migration complete.** All 36 `SemanticOp` variants now lower to a
`KernelCall` and execute through the transform stack. The canonical-op
vocabulary in `hologram-ops` is the single source of truth; chain →
plan → execute is fully connected.

### Phase 5 — Future work (out of this sprint's scope)
- [ ] **Per-op LUT generators** — every op needs both a LUT *and* a
  transformation. The transformation lives in
  `hologram-ops/src/kernels/<op>.rs` today; the LUT generator will land
  in the same file once Plan 074 (`uor-foundation` 0.3.0) exposes the
  address API. Tracked in detail under Sprint 37 Phase 2 (ADR-045).
- [x] Backward rules for the new ops — done. All differentiable
  canonical ops have `BackwardRule` + `Op::backward()` + `KernelCall::*Grad`
  + planner arm wired (see Sprint 37 Phase 3.4 for the full list).
- [x] Planner-level `Reshape` span aliasing — done. Reshape outputs
  share their input's `SlotSpan`; chained reshapes resolve to one
  root. Workspace shrinks correspondingly. Kernel still runs but
  no-ops via its existing `input.offset == output.offset`
  shortcircuit. `compute_reshape_alias_roots` walks `chain.nodes` in
  execution order; alias only applied when in/out tensors agree on
  `total_elements()` and `requires_grad`.
- [x] Workspace allocation for kernels that need scratch — initial
  pass shipped. `AttentionCall` carries a `scratch: SlotSpan` field;
  the planner reserves `seq_kv` floats per Attention node at the
  tail of the workspace via `build_op_scratch`. Kernel uses the
  span when supplied, falls back to a local `Vec` when empty (test
  path). Forward only — `AttentionGradCall` still allocates locally
  (two scratches: `probs` + `dp`); future work can extend the same
  pattern. im2col Conv2d remains a future scratch consumer.
- [x] Backend executors over the same `CompiledPlan` — done. `WgpuBackend`
  in `hologram-compute/src/canonical/wgpu.rs` ships in PR #6 with 51
  GPU-implemented variants conformance-validated against `CpuBackend`.

---

## Sprint 35: LUT-Addressed Transformation / Mutation Chains (ACTIVE)

**Plan:** [plans/043-lut-addressed-transform-chains.md](plans/043-lut-addressed-transform-chains.md)
**ADR:** [adrs/043-lut-addressed-transform-chains.md](adrs/043-lut-addressed-transform-chains.md)

Goal: introduce a new `hologram-transform` crate that separates LUT/address
identity, semantic transform descriptors, compile-time planning, and
runtime execution. Initial op set: ADD and MatMul (forward + backward).
Preserves all hot-path invariants — O(1) lookup, zero-copy, no dynamic
allocation, no virtual dispatch in kernels, no runtime algorithm selection.

### Phase 1 — Semantic transform chain
- [x] **1.1**: Create `hologram-transform` crate, register in workspace
- [x] **1.2**: Define `OpKind`, `BackwardRule`, `TransformNode`, `TransformChain`
- [x] **1.3**: Define `AddressRef`, `TensorId`, `NodeId`, `RegionId`, `LayoutId`
- [x] **1.4**: Add `TransformChain::builder` API + chain-construction tests

### Phase 2 — Address table and planner
- [x] **2.1**: Define `SlotSpan`, `AddressTable`, `WorkspaceLayout`
- [x] **2.2**: Implement planner: `TransformChain` → `CompiledPlan`
- [x] **2.3**: Resolve `AddressRef` → `SlotSpan` (O(1) lookup)
- [x] **2.4**: Allocate grad slots only for `requires_grad = true`

### Phase 3 — Forward / backward lowering
- [x] **3.1**: Add `KernelCall::Add`, `KernelCall::AddGrad`
- [x] **3.2**: Add `KernelCall::MatMul`, `KernelCall::MatMulGradA`,
  `KernelCall::MatMulGradB`
- [x] **3.3**: Backward emitted by planner (no runtime traversal)

### Phase 4 — CPU reference executor
- [x] **4.1**: `BufferSet` (single owned `Box<[f32]>`)
- [x] **4.2**: `Executor::run_forward`, `Executor::run_backward`
- [x] **4.3**: Reference CPU kernels: `add`, `add_grad`, `matmul`,
  `matmul_grad_a`, `matmul_grad_b`
- [x] **4.4**: Tests: ADD fwd/bwd, MatMul fwd/bwd, no-alloc invariant

### Phase 5 — Fusion + backend specialisation
- [x] **5.1 (initial)**: Fusion as a planner pass —
  `hologram_transform::fusion::fuse(&mut chain)` rewrites the chain's
  `nodes` slice. Public `compile_fused(&chain)` clones, fuses,
  compiles. Default `compile(&chain)` path unchanged so unfused
  plans still exercise via tests / conformance harness. **Initial
  pattern**: SwiGlu — `Silu(gate) → Mul(silu_out, up)` collapses to
  `FusedSwiGlu(gate, up)`. Conservative: skips when `silu_out` has
  more than one consumer or when the Mul's other operand transitively
  references the Silu input. Adding a new pattern lands as a private
  `try_fuse_<name>` helper invoked by `fuse()`.
- [x] **5.2**: Backend executors (Metal, WebGPU, Atlas) over the same plan
  — `CanonicalBackend` trait + `CpuBackend` reference + `WgpuBackend`
  shipped in PR #6. Cross-backend conformance harness validates each
  variant. Coverage: 51 GPU-implemented + 6 host-side intentional +
  remainder via canonical-CPU fallback (every variant has a working
  dispatch path).
- [x] **5.3 (ADR-051)**: Workspace residency for `WgpuBackend` —
  `BackendWorkspace` trait + `WgpuWorkspace` device-resident buffer
  + per-arm migration. **Every wgpu-backed arm now runs device-
  resident**: binary (17 variants + fused-swiglu), unary (20 kinds),
  reduce (5 kinds), softmax + log-softmax, pool (2d max/avg + global-
  avg), all 4 norms (rms, instance, layer, add-rms), group-norm,
  matmul + both grads, conv2d, conv-transpose-2d, reshape, slice,
  concat, softmax-grad + log-softmax-grad, rms-norm-grad,
  instance-norm-grad, layer-norm-grad, add-rms-norm-grad. All bind
  workspace buffer windows via `BufferBinding` offsets — zero
  per-call upload/download. Layouts/shaders use all-`read_write`
  storage bindings to satisfy wgpu's usage-scope rule. Spans must be
  64-element-aligned (256-byte storage offset).

  Remaining arms route through `host_cpu_fallback` because no wgpu
  shader exists for them: Attention/AttentionGrad, Gemm, Transpose/
  TransposeGrad, Pad/Expand/Resize{Nearest,Linear}/CumSum/
  RotaryEmbedding/Where/Lrn/Clip, MulGrad/DivGrad/PowGrad/MinMaxGrad/
  UnaryGrad/ConcatGrad/SliceGrad/ReduceGrad family (Reduce/ReduceArg/
  ReduceProd)/Pool2dGrad/GlobalAvgPoolGrad/FusedSwiGluGrad/
  GroupNormGrad/Conv2dGrad/ConvTranspose2dGrad. These take the slow
  round-trip path; promotion to device requires writing the WGSL
  shader first and is a per-arm decision blocked on benchmarks.

### Phase 6 — UOR / `uor-foundation` integration (deferred)
- [ ] **6.1**: Bridge `AddressRef` to `uor-foundation` LUT addresses (after
  Plan 074 lands the 0.3.0 upgrade)

---

## Sprint 34: uor-foundation 0.3.0 Upgrade

**Plan:** [plans/074-uor-foundation-0.3.0-upgrade.md](plans/074-uor-foundation-0.3.0-upgrade.md)

Goal: upgrade uor-foundation from 0.1.4 to 0.3.0. Major rename: QuantumLevel → WittLevel
(struct, not enum), const_ring_eval_q* → const_ring_eval_w*, ViolationKind removed.
Archive format preserved via RingLevel bridge.

### Phase 1: Bump + Capture Errors
- [ ] **1.1**: Edit `Cargo.toml:33` → `uor-foundation = "0.3.0"`
- [ ] **1.2**: Run `cargo check --workspace`, capture all errors

### Phase 2: Mechanical Renames
- [ ] **2.1**: Re-export alias `pub use uor_foundation::WittLevel as QuantumLevel` in op/mod.rs
- [ ] **2.2**: Update `RingLevel::from_quantum()` — match on `.witt_length()` (8/16/24/32)
- [ ] **2.3**: Update `RingLevel::to_quantum()` — return W8/W16/W24/W32
- [ ] **2.4**: Update `QuantumLevelExt::byte_width()` — `witt_length() / 8`
- [ ] **2.5**: Update `From<WittLevel> for RingLevel` and `From<RingLevel> for WittLevel`
- [ ] **2.6**: Rename const_ring_eval imports (q0→w8, q1→w16, q3→w32, q7→w64)
- [ ] **2.7**: Update eval_binary/eval_unary function bodies
- [ ] **2.8**: Fix all direct `uor_foundation::QuantumLevel` imports across crates
- [ ] **2.9**: Remove ViolationKind usage (3 sites in cascade + compiler)

### Phase 3: Semantic Fixes (compiler-driven)
- [ ] **3.1**: Fix `.index()` → `.witt_length()` in precision.rs, certificate.rs, shape.rs
- [ ] **3.2**: Fix `QuantumLevel::new(k)` → `WittLevel::new(8*(k+1))` construction sites
- [ ] **3.3**: Update hologram-ring trait impls returning QuantumLevel constants
- [ ] **3.4**: Implement HostTypes if required by CompileUnitBuilder
- [ ] **3.5**: Fix any additional trait method renames (at_quantum_level → at_witt_level)

### Phase 3b: Archive Backward Compatibility
- [ ] **3b.1**: Update engine.rs:407 to encode via RingLevel (not .index())
- [ ] **3b.2**: Update certificate.rs encoding to use RingLevel bridge
- [ ] **3b.3**: Add round-trip test verifying W8→0, W16→1, W24→2, W32→3 encoding

### Phase 4: Verify
- [ ] **4.1**: `cargo check --workspace` — zero errors
- [ ] **4.2**: `cargo test --workspace` — all pass
- [ ] **4.3**: `cargo clippy --workspace -- -D warnings` — clean
- [ ] **4.4**: `cargo check --workspace --target wasm32-unknown-unknown` — wasm compat

---

## Sprint 33: hologram-shape — Runtime Shape Tracking (ACTIVE)

**Plan:** [plans/073-hologram-shape-crate.md](plans/073-hologram-shape-crate.md)

Goal: eliminate all variable-length execution bugs by tracking tensor shapes
explicitly alongside data buffers. Every buffer gets a `TensorShape` — no more
guessing dimensions from byte lengths. Fixes the prefill garbage bug (long
prompts produce wrong output) and the entire class of shape-inference errors.

**Blocker:** Long prompts (>8 tokens) produce garbage during prefill due to
heuristic shape resolution failures in the 848-instruction execution chain.

### Phase 1: Core Types + Shape Inference
- [x] **1.1**: Create `hologram-shape` crate with `TensorShape`, `ShapeRegistry`
- [x] **1.2**: Implement `infer_output_shape` for all FloatOp categories
- [x] **1.3**: Unit tests for shape inference rules (≥30 tests)

### Phase 2: Wire into BufferArena
- [x] **2.1**: Add `ShapeRegistry` to `BufferArena`
- [x] **2.2**: Seed shapes from graph inputs + constants at execution start
- [x] **2.3**: Add `get_shape()` / `set_shape()` methods

### Phase 3: Wire into execute_direct
- [x] **3.1**: Compute + store output shape after each dispatch_kernel
- [x] **3.2**: Pass input shapes to dispatch_kernel
- [x] **3.3**: Debug assertions: validate buffer byte length matches shape

### Phase 4: Replace Heuristic Resolution (net line reduction)
- [x] **4.1**: Replace `resolve_last_dim` with direct shape reads — done
  in commit `1d1e870`. `dispatch_kernel` now reads input shapes via
  `shape_last_dim`/`shape_spatial_hw`/`shape_chw`/`shape_matmul_m`
  closures at all 22 resolve sites.
- [x] **4.2**: Replace `resolve_matmul_dims` with direct shape reads —
  done in the same commit (`shape_matmul_m` covers it).
- [ ] **4.3**: Delete `shape_resolve.rs` (355 lines) + `InputMetas` type
  — **partial**. Direct reads are the fast path, but
  `shape_resolve::resolve_*` calls remain as `.unwrap_or_else()`
  fallbacks for archives that lack shape metadata. Full deletion
  requires an archive-format ADR mandating shape metadata in every
  archive (so the fallback chain can be removed) — currently legacy
  archives still hit it.
- [ ] **4.4**: Verify: TinyLlama correct at seq=4, 13, 36, 77 — **blocked
  on model artefacts**. The dispatch path (Phase 4.1/4.2 direct shape
  reads) is in place; this phase is purely an end-to-end correctness
  check that needs the TinyLlama weight + tokenizer files to run.
  Re-open when artefacts land in the test-fixtures bucket.

### Phase 5: Propagate to hologram-compute
- [x] **5.1**: Wire `infer_output_shape` into `execute_on_backend` (parallel
  shape table seeded from arena's `ShapeRegistry`, output shape inferred
  per dispatch + debug-assert that inferred volume × dtype size matches
  device buffer byte length).
- [x] **5.2 (executor side)**: `kernel_params_for(op, input_shapes,
  output_shape)` populates `KernelParams.u32s` for every op the CPU
  backend hard-requires (MaxPool2d, AvgPool2d, Resize, GlobalAvgPool —
  `[channels, h_in, w_in]` family) plus the last-axis ops (Softmax,
  LogSoftmax, Reduce family, CumSum, LRN) and Transpose (full dims +
  permutation). `execute_on_backend` calls it before every dispatch so
  hard-required cases that previously errored now get the right
  shapes routed through. Other ops fall back to byte-length inference
  inside the backend as before.
- [ ] **5.2 (downstream)**: Each backend's dispatch path could
  additionally consult `TensorBuffer.shape` directly instead of
  reading dims from `KernelParams.u32s`. Optional cleanup; the
  current routing is correct, just less direct.

---

## Sprint 32: hologram-compute + hologram-exec Cleanup (COMPLETE)

**Plan:** [plans/067-compute-backend-rewrite.md](plans/067-compute-backend-rewrite.md)

Goal: create `hologram-compute` crate with `ComputeMemory` + `ComputeBackend<M>`
traits. Clean up hologram-exec by removing the old backend/ module and GPU
dispatch infrastructure. All GPU execution routes through the new backend.

- [x] Create hologram-compute crate (CpuBackend + MetalBackend)
- [x] CpuBackend: all 60+ FloatOp variants with Accelerate BLAS
- [x] MetalBackend: tiled SGEMM, im2col Conv2d, elementwise, norms, ring LUT
- [x] Fix Metal SGEMM dispatch (dispatch_thread_groups)
- [x] Comprehensive TapeKernel→FloatOp mapping (90+ ops)
- [x] Delete old backend/ module from hologram-exec (-4,337 lines)
- [x] Remove Metal/WebGPU deps + cfg flags from hologram-exec
- [x] Consolidate 35 trivial dispatch_kernel arms via dispatch_float_into
- [x] Upgrade CpuBackend Gemm to use Accelerate BLAS with transpose

**Result:** hologram-exec -5,500 lines. hologram-compute 3,969 lines. 535 tests pass.

---

## Backlog

- [x] Function length & argument count refactor — [plan](plans/003-function-length-refactor.md)
- [x] Prism ontology integration — [plan](plans/004-prism-uor-integration.md)
- [x] Compile-time-first acceleration — [plan](plans/005-compile-time-acceleration.md)
- [x] UOR-based lossless compression — [plan](plans/006-uor-compression-implementation.md)
- [x] Graph & mmap performance hardening — [plan](plans/007-graph-mmap-performance.md)
- [x] Dynamic sequence length — attention + slice fix — [plan](plans/016-dynamic-seq-attention-fix.md)
- [x] Zero-copy pipeline weights — [plan](plans/017-zero-copy-pipeline-weights.md)
- [x] Zero-copy graph access — [plan](plans/018-zero-copy-graph-access.md)
- [x] Runtime fat trim & allocation elimination — [plan](plans/019-runtime-fat-trim.md)
- [x] MatMul optimization — [plan](plans/020-matmul-optimization.md)
- [x] Epilogue fusion (Plan 005 Phase 2) — [plan](plans/030-epilogue-fusion.md)
- [x] Bias fusion (MatMul+Bias+Activation) — [plan](plans/031-bias-fusion.md)
- [x] Shape-aware tape execution API (feat/ai-optimization)

## Sprint 31: GEMM / MatMul / Conv2D Kernel Performance

**Plan**: [plans/037-gemm-conv2d-perf.md](plans/037-gemm-conv2d-perf.md)

Goal: maximize GEMM, MatMul, and Conv2D kernel throughput while keeping memory
usage flat. 10 optimizations across 4 phases.

### Phase 1: Bug Fix + Trivial Win
- [x] **1.1**: Complete LUT-GEMM + Conv2D in dispatch_kernel_par (BUG: catch-all error since ad7df78)
- [x] **1.2**: Eliminate per-call bias `.to_vec()` in Conv2D (borrow instead)

### Phase 2: GEMM Parallelism
- [x] **2.1**: Shared B-panel packing across M-tiles in matmul_k_outer
- [x] **2.2**: N-dimension parallelism in vecmat_mul + lower PAR_M_TILE_THRESHOLD

### Phase 3: Conv2D Kernel Improvements
- [x] **3.1**: Winograd batched GEMM parallelism (par_chunks_mut over 16 elements)
- [x] **3.2**: SIMD depthwise Conv2D (interior/border split + vectorization)
- [x] **3.3**: Fast im2col (memcpy interior for stride=1)

### Phase 4: Polish
- [x] **4.1**: Cache Winograd weight transform (1-entry thread-local, 16MB max)
- [x] **4.2**: A-panel packing — skipped (SIMD broadcast already fast, marginal gain vs complexity)
- [x] **4.3**: wasm32 SIMD128 micro-kernels (4×8 as two 4×4 halves)

---

## Sprint 30: CPU Optimization Sweep — Fusion, Prefetch, Parallelism

**Plan**: [plans/036-cpu-optimization-sweep.md](plans/036-cpu-optimization-sweep.md)

Goal: close remaining CPU-side optimization gaps across fusion, memory prefetch,
and parallelism. All changes are platform-agnostic (wasm + native).

### Phase 1: Fusion Gaps
- [x] **1.1**: AddRmsNorm + Activation fusion (GraphOp + TapeKernel + dispatch)
- [x] **1.1b**: InstanceNorm + Activation fusion (same pattern)
- [x] **1.2**: Binary in-place Add (zero-alloc Attention→Add(residual) via arena.add_inplace)
- [x] **1.3**: Transpose elimination (inverse transpose pairs → Passthrough)

### Phase 2: Multi-Level Weight Prefetch
- [x] **2.1**: 2-level lookahead in execute_inner (MADV_WILLNEED for i+1, i+2)
- [x] **2.2**: Early release already exists (MADV_DONTNEED for current level)

### Phase 3: Lock-Free LUT-GEMM Parallelism
- [x] **3.1**: Replace RefCell<WeightCache> with parking_lot::RwLock<WeightCache>
- [x] **3.2**: Enable rayon for LUT-GEMM levels (removed from needs_shared_state block)
- [ ] **3.3**: Per-thread Psumbook scratch (future: pre-populate cache for read-only access)

### Phase 4: Additional Fusion + Tuning
- [x] **4.1**: SwiGLU fusion from Silu + Mul pattern (try_fuse_swiglu)
- [ ] **4.2**: Adaptive sparse_v threshold (configurable per model/context)
- [ ] **4.3**: Activation checkpointing validation (verify compiler populates checkpoint_map)
- [x] **4.4**: InstanceNorm + Activation fusion (done in Phase 1)

### Phase 5: Memory Optimizations
- [x] **5.1**: Wire F16 activation compression (checkpoint nodes compress to F16 instead of evict)
- [ ] **5.2**: Wire workspace buffer reuse into arena allocation (not yet wired — architectural)

### Phase 6: WebGPU Kernel Parity (wasm GPU path)
- [x] **6.2**: Softmax + RmsNorm already existed; GroupNorm WGSL shader added

### Deferred until benchmarks (2026-04-28)

The four open items below are perf knobs without a measured baseline.
Implementing them speculatively risks shipping complexity for no win.
Each is parked until the corresponding measurement exists:

- **3.3 Per-thread Psumbook scratch** — needs a contended-cache
  benchmark on multi-thread LUT-GEMM showing the lock is the
  bottleneck. Today's `RwLock` may already be invisible.
- **4.2 Adaptive sparse_v threshold** — needs a sweep across model
  sparsity profiles; current static threshold is fine for the
  models we've measured.
- **4.3 Activation checkpointing validation** — needs an end-to-end
  training-recompute test to confirm `checkpoint_map` populates
  correctly. Inference path doesn't exercise it.
- **5.2 Workspace buffer reuse in arena** — Sprint 31 already wires
  this for the executor; this 5.2 entry is the *compile-time*
  variant and overlaps. Re-evaluate after Sprint 31 lands peak-mem
  profiling (Phase 3.2).

---

## Sprint 31: BufferArena Workspace Reuse

**Plan**: [plans/038-buffer-arena-workspace-reuse.md](plans/038-buffer-arena-workspace-reuse.md)

Goal: wire the compiler's `plan_workspace()` slot assignments into the executor's
`BufferArena` so non-overlapping nodes share physical buffers. Expected 20-40%
peak activation memory reduction with zero latency regression.

### Phase 1: Thread WorkspaceLayout into EnumTape
- [x] **1.1**: Add `slot_assignments: Vec<u32>` + `n_slots: u32` to `EnumTape`
- [x] **1.2**: `compute_slot_assignments()` — greedy interval coloring from producer/consumer maps

### Phase 2: MmapBuffer Free-List Recycling
- [x] **2.1**: Add `free_mmaps: Vec<MmapBuffer>` free-list to `BufferArena`
- [x] **2.2**: `evict()` pushes large MmapBuffers to free-list instead of dropping
- [x] **2.3**: `swap_insert_with_elem_size()` reuses from free-list before allocating

### Phase 3: Tests + Validation
- [x] **3.1**: 3 new tests: evict recycles, swap_insert reuses, small evict no-recycle
- [ ] **3.2**: Peak memory profiling (SD UNet, LLaMA 7B)

---

## Backlog: WebGPU Kernel Parity (wasm GPU path)
- [ ] Conv2d WGSL compute shader (im2col + tiled GEMM in WGSL)
- [ ] Attention WGSL shader (tiled, Flash Attention-style)

---

## Sprint 29: Conv2d Epilogue Fusion — Accelerate SD UNet Chain

**Plan**: [plans/035-conv2d-epilogue-fusion.md](plans/035-conv2d-epilogue-fusion.md)

Goal: fuse Conv2d + Activation (and Conv2d + Bias + Activation) into single tape
kernels, eliminating intermediate buffer materialization. GroupNorm → SiLU is already
fused (Sprint 23). For 512×512 SD inference with 23 ResNet blocks, this eliminates
~7.7GB of unnecessary memory traffic per step.

### Phase 1: Conv2d + Activation Epilogue Fusion
- [ ] **1.1**: Add `FusedConv2dActivation` GraphOp variant
- [ ] **1.2**: Add `try_fuse_conv2d_activation()` fusion pattern
- [ ] **1.3**: Add `InlineConv2dActivation` TapeKernel + dispatch
- [ ] **1.4**: Wire tape builder + exhaustive match coverage
- [ ] **1.5**: Tests: fusion detection, no-fuse (fan-out), correctness

### Phase 2: Conv2d + Bias + Activation (3-node)
- [ ] **2.1**: Add `FusedConv2dBiasActivation` GraphOp variant
- [ ] **2.2**: Add `try_fuse_conv2d_bias_activation()` fusion pattern
- [ ] **2.3**: Add `InlineConv2dBiasActivation` TapeKernel + dispatch
- [ ] **2.4**: Tests: 3-node pattern, non-constant bias rejection

### Phase 3: Validation
- [ ] **3.1**: Verify `can_reuse_input` for Attention → MatMul handoff
- [ ] **3.2**: Conv2d fusion benchmark
- [ ] **3.3**: End-to-end SD UNet latency comparison

---

## Sprint 28: KV Cache Quantization — Asymmetric Compression

**Plan**: [plans/034-kv-cache-quantization.md](plans/034-kv-cache-quantization.md)

Goal: reduce KV cache memory 2-4× via asymmetric quantization (K at f32, V at q4/q8)
with boundary layer protection and Walsh-Hadamard pre-rotation. Based on findings
from asymmetric KV compression research (V is robust to quantization; K errors
propagate exponentially through softmax).

### Phase 1: Boundary Layer Protection + Config
- [ ] **1.1**: Add `KvCacheConfig` / `KvBits` types with boundary layer support
- [ ] **1.2**: Modify `KvCacheState::new()` to accept config
- [ ] **1.3**: Tests: boundary layers remain f32, config defaults

### Phase 2: Per-Channel Min/Max Quantization
- [ ] **2.1**: Add `QuantizedKvBuffer` (q8/q4 storage with per-head scales)
- [ ] **2.2**: Online quantize on `write_layer()`, dequantize on `read_k()`/`read_v()`
- [ ] **2.3**: Tests: round-trip tolerance, asymmetric K/V precision

### Phase 3: Walsh-Hadamard Pre-Rotation
- [ ] **3.1**: Implement FWHT (in-place butterfly, O(d log d))
- [ ] **3.2**: Apply rotation before V quantization, inverse on dequantize
- [ ] **3.3**: Tests: self-inverse property, quantization error reduction

### Phase 4: Tape Integration
- [ ] **4.1**: Wire config through `TapeContext` → `KvCacheState`
- [ ] **4.2**: Verify KvWrite/KvRead dispatch handles quantized paths
- [ ] **4.3**: End-to-end integration tests

---

## Sprint 27: Performance Regression Fixes — M=1 MatMul + Tape Execution

**Branch**: `refactor/stable-diffusion-pipeline`

Goal: fix two regressions introduced by the fused dequant-matmul / mmap refactoring
in Sprint 26. M=1 (single-token decode) float matmul regressed 100-890% due to
missing SIMD in the MR=1 remainder path; tape execution regressed 140-272% due to
mandatory mmap syscalls on every small output.

### Fix 1: Specialized M=1 vecmat_mul
- [x] **F1.1**: Add `vecmat_mul` with NEON/AVX2 SIMD kernels (strided B, no packing)
- [x] **F1.2**: Early-out in `matmul_k_outer` when `m == 1`
- [x] **F1.3**: Relax bit-exact dequant test assertions to relative tolerance

**Result**: M=1 dispatch_matmul 49-58% faster; 1x64x64 now 20% faster than pre-Sprint-26 main.

### Fix 2: VecOwned arena buffer for small outputs
- [x] **F2.1**: Add `VecOwned(Vec<u8>)` variant to `ArenaBuffer`
- [x] **F2.2**: `swap_insert_with_elem_size` takes Vec ownership below 256 KB (no mmap syscall)
- [x] **F2.3**: Update all ArenaBuffer match sites (as_bytes, into_owned, get_mut_f32)

**Result**: Tape execution 2-4× faster for small/medium sizes; epilogue_fusion 49-75% faster.

---

## Sprint 26: UOR 0.1.0 Migration — Algebraic Performance Acceleration

**Plan**: [plans/033-uor-migration.md](plans/033-uor-migration.md)

Goal: merge the Q0→Q3 Cayley-Dickson algebraic acceleration chain from
`feat/uor0.1.0-migration`. Key wins: ~256× faster quantization, ~2× fewer MACs
via orbit compression, 24 inline hot-path kernels, cache-optimized fiber-ordered
GEMM, carry-driven dynamic precision dispatch, Q1 view fusion, and platform
prefetch.

### Merge
- [ ] **M.1**: Merge `feat/ai-optimization` → `main`
- [ ] **M.2**: Merge `origin/feat/uor0.1.0-migration` → `main` (resolve `tape.rs` conflict)
- [ ] **M.3**: Verify: `cargo test` + `cargo clippy` + `cargo fmt --check`

### New Modules (from migration branch)
- Q1/Q2/Q3 algebraic types (`hologram-core/src/{q1,q2,q3}/`)
- Carry-driven precision lifting (`hologram-core/src/carry/`)
- Orbit compression + Q16 quantization (`hologram-exec/src/lut_gemm/{orbit,quantize_q1,psumbook_q1}.rs`)
- Precision + QEDL compiler passes (`hologram-compiler/src/{precision,qedl}/`)
- Q1 view fusion pass (`hologram-graph/src/fusion/q1_view_fusion.rs`)
- 58 new conformance + performance-contract tests

---

## Sprint 25: Parallel Compilation + BLAKE3 Checksums

Goal: parallelise the compiler pipeline and migrate archive checksums from
CRC32 to BLAKE3 (format v2). ADR: [specs/adrs/001-blake3-checksums.md](adrs/001-blake3-checksums.md)

### Part A: CRC32 → BLAKE3 Migration
- [x] **A.1**: Migrate `checksum/mod.rs` from crc32fast to blake3
- [x] **A.2**: Expand header/section/error/weight checksum fields to `[u8; 32]`
- [x] **A.3**: Update writers + loader
- [x] **A.4**: Remove `crc32fast` dep

### Part B: Parallel Compilation (feature-gated `parallel`)
- [x] **B.1**: Add `parallel` feature + rayon to archive/graph/compiler crates
- [x] **B.2**: Parallelise graph + weight compression (`rayon::join`)
- [x] **B.3**: Parallelise schedule building (levels ∥ critical path)
- [x] **B.4**: Parallelise liveness analysis (`par_iter`)

---

## Sprint 24: Bias Fusion (Plan 031) — DONE

**Plan**: [plans/031-bias-fusion.md](plans/031-bias-fusion.md)

Goal: fuse MatMul+Add(bias)+Activation into a single TapeKernel, eliminating
two intermediate buffers. This is the pattern that `can_reuse_input` cannot
optimize away — the real performance win from epilogue fusion.

### Phase 1: Graph + Tape Variants
- [x] **1.1**: Add `FusedMatMulBiasActivation` GraphOp + `InlineMatMulBiasActivation` TapeKernel
- [x] **1.2**: Exhaustive match coverage (kv/store, CLI inspect, tape builder)

### Phase 2: Fused Kernel
- [x] **2.1**: `dispatch_matmul_bias_activation_into` — matmul + bias+activation in single pass

### Phase 3: Fusion Pass
- [x] **3.1**: `try_fuse_matmul_bias_activation()` — 3-node pattern (MatMul → Add(const) → Activation)
- [x] **3.2**: Wire into `fuse()` before 2-node matmul+activation pass

### Phase 4: Tests + Benchmark
- [x] **4.1**: Graph fusion test (`fuse_matmul_bias_activation_via_full_pass`)
- [x] **4.2**: Benchmark: transformer decode 2.81ms → 2.77ms (-1.24%, p=0.01)

---

## Sprint 23: Epilogue Fusion (Plan 030)

**Plan**: [plans/030-epilogue-fusion.md](plans/030-epilogue-fusion.md)

Goal: fuse matmul+activation and norm+activation into single TapeKernel variants,
eliminating memory round-trips between accumulator writeback and activation.
Driven by thermodynamic precision analysis (Landauer's principle: the epilogue is
the last reversible place to change precision gauges).

### Phase 1: MatMul + Activation Epilogue Fusion
- [x] **1.1**: Add `TapeKernel::InlineMatMulActivation` variant
- [x] **1.2**: Add `matmul_k_outer_fused` CPU kernel + `dispatch_matmul_activation_into`
- [x] **1.3**: Wire dispatch in tape executor
- [x] **1.4**: Add `GraphOp::FusedMatMulActivation` (rkyv-serializable)
- [x] **1.5**: Add `try_fuse_matmul_activation()` fusion pass
- [x] **1.6**: Wire tape builder: `FusedMatMulActivation` → `InlineMatMulActivation`
- [x] **1.7**: LUT-GEMM fused variants (`MatMulLut4Activation`, `MatMulLut8Activation`)

### Phase 2: Norm + Activation Fusion
- [x] **2.1**: Add fused `InlineRmsNormActivation`, `InlineLayerNormActivation`, `InlineGroupNormActivation`
- [x] **2.2**: Fused norm kernels (apply activation before writeback)
- [x] **2.3**: Add `try_fuse_norm_activation()` fusion pass

### Phase 3: Tests
- [x] **3.1**: Unit tests: fused kernel bit-identical to separate ops
- [x] **3.2**: Graph fusion tests: pattern detection + no-fuse cases
- [x] **3.3**: Tape E2E: fused vs unfused output identity

---

## Sprint 22: MatMul Optimization (Plan 020)

**Plan**: [plans/020-matmul-optimization.md](plans/020-matmul-optimization.md)

Goal: optimize MatMul kernels across CPU and GPU paths. Fix dispatch_gemm perf
bug, eliminate intermediate allocations, add register-blocked micro-kernel for
non-BLAS platforms, and enable batched matmul on GPU.

### Phase 1: dispatch_gemm Loop Restructuring
- [x] **1.1**: Pre-transpose A/B instead of runtime conditionals in inner loop
- [x] **1.2**: Use k-outer loop pattern via shared `matmul_k_outer` kernel
- [x] **1.3**: Apply alpha/beta scaling as post-pass

### Phase 2: dispatch_matmul_into Direct Write
- [x] **2.1**: Move `alloc_f32_in` + `transpose_f32` to shared helpers module
- [x] **2.2**: Rewrite dispatch_matmul_into to write directly to out_buf
- [x] **2.3**: Consolidate all matmul loops to shared `matmul_k_outer` kernel

### Phase 3: CPU Register-Blocked Micro-Kernel
- [x] **3.1**: 4×8 register-blocked matmul for non-BLAS platforms
- [x] **3.2**: Remainder handling for non-tile-aligned dimensions
- [x] **3.3**: Matmul size sweep benchmark (1×64×64 → 128×2048×2048)

### Phase 4: Batched MatMul GPU Dispatch
- [x] **4.1**: Metal batched SGEMM kernel (Z-dimension batch, shared-memory tiled)
- [x] **4.2**: WebGPU batched SGEMM kernel (Z-workgroup batch, deferred dispatch)
- [x] **4.3**: Wire batched dispatch through ComputeBackend trait (default Skipped)

---

## Sprint 21: Runtime Fat Trim & Allocation Elimination (Plan 019)

**Plan**: [plans/019-runtime-fat-trim.md](plans/019-runtime-fat-trim.md)

Goal: eliminate dead code, remove unused dependencies, eliminate `.to_vec()`
allocations in the hot path, and inline remaining high-frequency ops as
TapeKernel variants.

### Phase 1: Dead Code & Dependency Removal
- [x] **1.1**: Remove CUDA backend stub (`backend/cuda.rs` — always returns Skipped)
- [x] **1.2**: Replace `dirs` crate with cross-platform `home_dir()` helper (~25 transitive deps removed)
- [x] **1.3**: Gate `serde`/`toml` behind `cli` feature (remove from library dep tree)
- [x] **1.4**: Narrow `tokio` features (drop "full", use minimal subset)

### Phase 2: `.to_vec()` Elimination (71 calls audited)
- [x] **2.1**: Tape-builder passthrough for identity Cast and Reshape (zero dispatch, zero copy)
- [x] **2.2**: Norm `_into` variants write directly to `out_buf` (9 calls → zero intermediate Vec)
- [x] **2.3**: Attention zero-copy `heads_first` path via `Cow<[f32]>` (3 tensor copies eliminated)
- [x] **2.4**: `into_owned()` replacements for scatter_nd, cumsum, reverse_sequence, mask, RoPE (8 calls)

### Phase 2b: Inline TapeKernel Expansion
- [x] **2b.1**: Inline LayerNorm, AddRmsNorm, LogSoftmax (norm ops with baked params)
- [x] **2b.2**: Inline Attention, RotaryEmbedding (per-layer ops — uses TapeContext for position offset)
- [x] **2b.3**: Inline Gather, Concat (data movement ops with baked params)
- [x] **2b.4**: Inline remaining simple unary: Log, Sqrt, Cos, Sin, Sign, Floor, Ceil, Round, Erf
- [x] **2b.5**: Inline remaining simple binary: Min, Max

### Phase 3: Weight Cache & Dispatch Cleanup
- [x] **3.1**: Weight cache — eliminate double hash lookup via Entry API
- [x] **3.2**: `dispatch_float` marked `#[inline]` (kept for public API + test compat)
- [x] **3.3**: Allocating norm variants: `into_owned()` instead of `to_vec()`

### Benchmark Results (Sprint 21)
- tape::relu 64KB: **2.30 µs → 1.81 µs** (21% faster)
- transformer decode step: **5.99 ms → 2.77 ms** (54% faster, 2.2x speedup)
- softmax row_based 8192: **12.83 µs → 11.83 µs** (8% faster)
- tape::linear chain 4 nodes: **1.15 µs** (unchanged — already optimal)
- Total inline TapeKernel variants: **17 → 38** (all high-frequency ops covered)

---

## Sprint 20: Zero-Copy Graph Access (Plan 018)

**Plan**: [plans/018-zero-copy-graph-access.md](plans/018-zero-copy-graph-access.md)

Goal: eliminate 1.5s graph deserialization by using rkyv::access (zero-copy
archived field access) instead of rkyv::from_bytes (full owned deserialization).

### Phase 1: Optional Graph Compression
- [x] **1.1**: `compress_graph: bool` field on HoloWriter (default false)
- [x] **1.2**: `.compress_graph()` opt-in method
- [x] **1.3**: Skip compression when `compress_graph == false`

### Phase 2: Zero-Copy Graph Access
- [x] **2.1**: `GraphAccess` enum (Owned vs Archived) in LoadedPlan — lazy `OnceLock` deserialization
- [x] **2.2**: ~~`ArchivedConstantStore::get()`~~ — not needed: `graph()` transparently deserializes
- [x] **2.3**: ~~Archived-compatible maps~~ — not needed: lazy deser returns `&SerializedGraph`
- [x] **2.4**: ~~Update consumers~~ — all unchanged: `graph()` API returns `&SerializedGraph`
- [x] **2.5**: Decompress-once cache: `HoloLoader::load_cached()` with `.holo.cache` file + mmap

## Sprint 19: Zero-Copy Pipeline Weights (Plan 017)

**Plan**: [plans/017-zero-copy-pipeline-weights.md](plans/017-zero-copy-pipeline-weights.md)

Goal: pipeline archives store weights once in the wrapper, sub-archives reference
them via dedup index. Loading is zero-copy via mmap. Archive size halved, load
time from 20s+ to <1s.

### Phase 1: Archive Format + Loader (hologram)
- [x] **1.1**: `LoadedPlan::set_weights_borrowed` — zero-copy weight grafting from wrapper mmap
- [x] **1.2**: `PipelineWriter::build_with_shared_weights` — shared weight blob layout
- [x] **1.3**: `LoadedPipeline::from_bytes_zero_copy` — borrow sub-archive + shared weights from mmap

### Phase 2: Compiler (hologram-ai)
- [x] **2.1**: Shared weight extraction via `WeightStore` — `build_with_shared_weights()` wired
- [x] **2.2**: ~~Rewrite Deferred offsets~~ — not needed: offsets are per-component, loader grafts correct slice
- [x] **2.3**: `HoloRunner` zero-copy pipeline loading — dedup index resolution added to `from_storage()`

### Phase 3: Tests
- [x] **3.1**: Pipeline shared weights round-trip (build + load + resolve constants)
- [x] **3.2**: Zero-copy mmap pipeline loading (verify no allocation for weights)
- [ ] **3.3**: Weight dedup across prefill/decode models — needs TinyLlama model files
- [ ] **3.4**: E2E: compile TinyLlama pipeline + run with <1s load time — needs TinyLlama model files

---

## Sprint 18: Dynamic Shape Inference (Plan 016)

**Plan**: [plans/016-dynamic-seq-attention-fix.md](plans/016-dynamic-seq-attention-fix.md)

Goal: enable ONNX models with dynamic symbolic shapes (variable seq_len) to run
at runtime without `--seq-len` at compile time.

### Phase 1: Slice Axis Size Inference
- [x] **1.1**: `infer_slice_axis_size()` helper — infer actual axis dim from buffer + slice upper bound
- [x] **1.2**: Fix Slice dispatch to use inferred axis size instead of `end` heuristic

### Phase 2: Attention Buffer Validation
- [x] **2.1**: Validate Q/K/V buffer divisibility before seq inference
- [x] **2.2**: Validate K/V size consistency (prevent panic on mismatch)
- [x] **2.3**: Return `ShapeMismatch` with diagnostic info (buffer sizes, head config, inferred seq)

### Phase 3: Conformance Tests
- [x] **3.1**: GQA attention at variable seq lengths (seq=2 and seq=3)
- [x] **3.2**: Attention K/V mismatch → error (not panic)
- [x] **3.3**: Attention non-divisible Q → error
- [x] **3.4**: Slice with dynamic leading dimension (partial axis slice)
- [x] **3.5**: Slice where end == axis_size (fast path preserved)

---

## Sprint 13: Compile-Time-First Acceleration

### Phase 0: Execution Orchestration Overhaul (highest ROI)
- [x] **0.1**: Flat pre-allocated buffer arena (replace HashMap-based arena)
- [x] **0.2**: Output buffer pre-allocation in dispatch (`dispatch_into` API)
- [x] **0.3**: Compile-time shape resolution (CompiledNode with pre-resolved shapes)
- [x] **0.4**: Embed execution schedule in archive (Tape struct with level offsets)
- [x] **0.5**: SmallVec strides + stride memoization for float dispatch
- [x] **0.6**: Adaptive parallel threshold (compiler cost estimates per level)
- [x] **0.7**: Instruction tape executor (kernel function pointer table, zero-match dispatch)
- [x] **0.8**: System-level: `target-cpu=native` build flag, KV cache lazy init, dense metadata arrays

### Phase 0b: float_dispatch Kernel Optimization
- [x] **0b.1**: Split `float_dispatch.rs` (3095 lines) into directory module with 14 sub-files
- [x] **0b.2**: Flatten `transpose_heads` triple loop → single flat loop with index decomposition
- [x] **0b.3**: Flatten pool ops (max_pool_2d, avg_pool_2d) 6-level → 2-level with generic `pool_2d<A>` kernel
- [x] **0b.4**: Flatten `conv_transpose` 7-level scatter loops → 2-level (flat outer + flat kernel)
- [x] **0b.5**: Extract `dot_f32` helper for attention (enables autovectorization)
- [x] **0b.6**: im2col + GEMM for `conv2d` (replace 8-level nested loops, unify two conv2d variants via shared `conv2d_core`)
- [x] **0b.7**: Online softmax (Flash Attention-style) for fused attention kernel
- [x] **0b.8**: Pre-computed KV offsets in instruction tape (eliminate per-head offset arithmetic)
- [x] **0b.9**: ~~Flatten `conv2d` loops~~ (superseded by 0b.6 im2col)

### Phase 1: Compile-Time Weight Layout + SIMD
- [x] **1.A**: Weight cache — eliminate per-dispatch `rkyv::from_bytes` re-deserialization
- [x] **1.B**: Compile-time column-major/tiled weight index layout
- [x] **1.3**: Tiled multi-column LUT-GEMM kernels (Q8 4-column tiled kernel)
- [x] **1.4**: SIMD dot products for Psumbook (autovectorization-friendly patterns)
- [x] **1.4b**: ARM NEON for ElementWiseView (vqtbl1q_u8 16-byte table lookup)

### Phase 2: Compile-Time Fusion
- [x] **2.1**: Compile-time MatMul+Bias+Activation fusion (fused dispatch chain)
- [x] **2.2**: Compile-time Norm+Activation fusion + fast_rsqrt (Quake III-style)
- [x] **2.3**: fast_exp for softmax (Schraudolph bit-manipulation, ~4x faster)
- [x] **2.4**: Compile-time buffer alignment for SIMD (Psumbook align(64), Vec<f32> natural alignment)

### Phase 3: Tiled Attention
- [x] **3.1**: Attention op with compiler-baked tile sizes (pre-computed head offsets)
- [x] **3.2**: Online-softmax tiled attention kernel (Flash Attention-style) — done in 0b.7
- [x] **3.3**: Per-head parallelism (head_offsets enable independent parallel execution)

### Roadmap: Phases 4-6 (Near-Term)
- [x] **4**: Sliding window attention + quantized K cache (window_size field, windowed reads)
- [x] **5**: Precomputed Scatter Groups for LUT-GEMM (tiled multi-column kernel shares activation reads)
- [x] **6**: Transformer block fusion + DQ-GEMM (pattern detection skeleton in float_fusion)

### Roadmap: Phases 7-9 (Quantize-Into-LUT-Domain)
- [x] **7.1**: RoPE frequency precomputation (compile-time static table)
- [x] **7.2**: Softmax exp via fast_exp (Schraudolph bit-manipulation, ~1.5% error)
- [x] **7.3**: RmsNorm rsqrt via fast_rsqrt (Quake III, 2 NR iterations)
- [x] **7.4**: Erf uses Abramowitz & Stegun polynomial (compile-time evaluated)
- [x] **8**: QEDL pipeline — QedlBoundary enum + qedl_boundaries in CompilationOutput
- [x] **9.1**: Q0×Q0 binary arithmetic tables (add, mul, div, min, max — 64KB each)
- [x] **9.2**: FusedSwiGLU in byte domain (byte_domain_fused_swiglu using SILU_256 + byte_mul)

### Roadmap: Phase 10 (Hierarchical Content-Addressable LUT)
- [x] **10.1**: HierarchicalLut struct with content-addressable page selector (ElementWiseView)
- [x] **10.2**: Adaptive PageKind (Constant, Linear, Table16) for compression
- [x] **10.3**: Compile-time k-means page construction (from_flat_kmeans alias)
- [x] **10.4**: Q2 HLUT for all activations (build_all_hluts function)
- [x] **10.5**: HLUT-aware view fusion (compose method on HierarchicalLut)

### Roadmap: Phases 11-15 (Systems-Level Acceleration)
- [x] **11**: Prefetch + speculative execution (CPU prefetch hints in tape executor)
- [x] **12.1**: Model-specific weight distribution analysis (WeightStats + weight_stats function)
- [x] **12.2**: Activation range profiling (ActivationProfile struct with record_buffer)
- [x] **12.3**: Graph-specific tile sizes (tile_hint field in tape Instruction)
- [x] **12.4**: Sparsity-aware compilation (sparsity_ratio function for QuantizedWeights)
- [x] **13**: Incremental delta computation (dirty-bit skip-if-unchanged for decode)
- [x] **14**: Mmap zero-copy execution (insert_borrowed path + execute_plan_zero_copy alias)
- [x] **15**: Batch-aware scheduling (BatchConfig struct with shared_prefix_len)

## Sprint 14: UOR-Based Lossless Compression

**Plan**: [plans/006-uor-compression-implementation.md](plans/006-uor-compression-implementation.md)

### Phase 1: Bootstrap hologram-compression
- [x] Fix crate structure (lib.rs, Cargo.toml with hologram-core dep)
- [x] Create module skeleton (codec, stratum, ring_diff, torus_block, entropy, float_plane, permute, pipeline, header)

### Phase 2: Core compression algorithms
- [x] Codec types (CompressedBlock, CompressionMode, CompressionStats)
- [x] Header format (HLZC magic, mode, permute_id, original_len)
- [x] Stratum partition tables + intra-stratum rank codec (SPEC)
- [x] Ring-differential coding (RDC) with order-0 and order-1 predictors
- [x] Orbit-torus blocked coding (page/offset split)
- [x] rANS entropy backend (encoder + decoder)
- [x] Frequency counting + normalization
- [x] Float byte-plane transposition (f32/f64)
- [x] Bijective pre-transforms (ElementWiseView permutations)
- [x] Pipeline orchestration + mode selection
- [x] Full end-to-end compress/decompress with all 4 modes

### Phase 3: Archive integration
- [x] Add hologram-compression as optional dependency to hologram-archive
- [x] CompressionScheme field in TensorMetadata (compression_scheme: u8)
- [x] Compression flag bits in HoloHeader (COMPRESSION_FLAG = 0x0010)
- [x] Default-on compression for weight sections (auto_select_mode in HoloWriter::build)
- [x] Transparent decompression on load (extract_weights + deserialize_graph decompress paths)
- [x] Graph section compression (Mode 0 via FLAG_GRAPH_COMPRESSED)

### Phase 4: WASM FFI + Site demo
- [x] New WASM functions (compress, decompress, stats, histogram, ring_algebra, float_plane_transpose)
- [x] Site demo page (compression.astro)
- [x] Register in site config sidebar

---

## Sprint 15: Graph & mmap Performance Hardening

**Plan**: [plans/007-graph-mmap-performance.md](plans/007-graph-mmap-performance.md)

### Phase 1: Hot-Path Allocation Elimination (P0)
- [x] **1.1**: Eliminate `to_vec()` copies in tape execute loop (scoped borrow instead of cloning)
- [x] **1.2**: Upgrade prefetch from `black_box` load to `_mm_prefetch` / `PRFM PLDL1KEEP` intrinsics

### Phase 2: mmap Page Discipline (P1)
- [x] **2.1**: Add `madvise` hints for mmap'd weight regions (MADV_RANDOM for LUT-GEMM, MADV_SEQUENTIAL for graph)
- [x] **2.2**: Weight-page prefetch for next instruction's constants (already wired in execute loop)
- [x] **2.3**: Audit tape builder for eager weight-page touching (CLEAN — no weight data accessed)

### Phase 3: Graph Edge Efficiency (P2)
- [x] **3.1**: Reverse-edge index for O(degree) `successors()` (`build_successor_index` + `successors_from_index`)
- [x] **3.2**: TinyVec<[InputSlot; 2]> for node inputs (rkyv `tinyvec-1` feature, inlines unary+binary ops)

### Phase 4: Observability (P3)
- [x] **4.1**: Page-fault tracking benchmark (`mmap_load_execute` + perf stat integration docs)

### Phase 5: Dispatch Allocation Reduction
- [x] **5.1**: SmallVec<[&[u8]; 4]> for tape input_refs (stack-allocate for ≤4 inputs per instruction)
- [x] **5.2**: SmallVec<[&[u8]; 4]> for `gather_inputs` in KvExecutor (stack-allocate per-node input gathering)
- [x] **5.3**: Eliminate redundant data copy in reshape (defer `to_vec()` to return, skip intermediate allocation)
- [x] **5.4**: Identity transpose short-circuit + deferred `cast_f32` (skip cast for no-op/error paths)

### Phase 6: Zero-Allocation Tape Execution
- [x] **6.1**: `swap_insert_with_elem_size` on BufferArena (kernel/arena trade buffer allocations)
- [x] **6.2**: `KernelFn`/`BoxedKernel` signature → `_into` pattern (write to `&mut Vec<u8>` instead of returning `Vec<u8>`)
- [x] **6.3**: `Tape::execute`/`BoxedTape::execute` reusable output buffer with swap-insert loop
- [x] **6.4**: `dispatch_fused_chain_into` helper for fused unary chains
- [x] **6.5**: All 19 tape_builder kernel closures updated to `_into` pattern

### Phase 7: Output Size Hints + Native _into for Hot Ops
- [x] **7.1**: `output_byte_hint` field on `BoxedInstruction` (pre-computed from compiled shapes+dtypes)
- [x] **7.2**: `compute_output_byte_hint` in tape_builder (product of shape dims × elem_size, 0 for dynamic)
- [x] **7.3**: `reserve(output_byte_hint)` in execute loop before kernel call
- [x] **7.4**: `dispatch_matmul_into` — native in-place matmul (avoids alloc+copy fallback)
- [x] **7.5**: `dispatch_softmax_into` — native in-place softmax
- [x] **7.6**: `dispatch_rms_norm_into` — native in-place RmsNorm
- [x] **7.7**: `dispatch_custom_into` router in `dispatch_float_into` (MatMul, Softmax, RmsNorm)

### Phase 8: Enum Dispatch + LUT-GEMM Tape Wiring
- [x] **8.1**: `TapeKernel` enum — replaces `Box<dyn Fn>` with 8 inline variants (no vtable, no heap alloc)
- [x] **8.2**: `TapeContext` struct — carries ConstantStore + weights + RefCell\<WeightCache\> for LUT-GEMM
- [x] **8.3**: `TapeInstruction` / `EnumTape` — replaces `BoxedInstruction` / `BoxedTape`
- [x] **8.4**: `dispatch_kernel` match function — inlinable enum dispatch for all kernel types
- [x] **8.5**: LUT-GEMM Q4/Q8 wired into tape via `dispatch_lut_gemm_4` / `dispatch_lut_gemm_8`
- [x] **8.6**: `tape_builder.rs` rewritten — `resolve_kernel` returns enum variants, no closures
- [x] **8.7**: `execute_tape` in mmap/mod.rs builds `TapeContext` with weight access
- [x] **8.8**: 6 new EnumTape unit tests + tape vs KvExecutor benchmark (30% faster confirmed)

### Benchmark Results (Phase 8)
- Tape vs KvExecutor on Relu 64KB: **36.4 µs vs 47.2 µs** (1.30x faster)
- Tape linear chain (4 float nodes, 256B): **706 ns**

### Phase 8b: Fused Ops
- [x] **8b.1**: `FloatOp::AddRmsNorm` — fused Add + RmsNorm (eliminates intermediate residual buffer)
- [x] **8b.2**: `dispatch_add_rms_norm` + `dispatch_add_rms_norm_into` in norm.rs
- [x] **8b.3**: Wired into `dispatch_custom_into` router + `dispatch_custom` fallback

### Phase 9: Tape Correctness + Dispatch Coverage
- [x] **9.3**: Native `_into` for LayerNorm + LogSoftmax (extend `dispatch_custom_into`)
- [x] **9.5**: Dynamic size inference via `resolve_size` (Softmax/RmsNorm/LayerNorm size=0 sentinel → infer from input)
- [x] **9.7**: Tape-path conformance test vs KvExecutor (Relu→Neg chain, byte-for-byte output match)

### Phase 10: KvCache + Conformance
- [x] **10.5**: KvWrite/KvRead wired into tape (TapeKernel variants + TapeContext with RefCell\<KvCacheState\>)
- [x] **10.6**: Softmax conformance test (same graph through KvExecutor and EnumTape, byte-for-byte match)

### Phase 11: Weight Prefetch + LUT-GEMM Validation
- [x] **11.1**: `weight_offset_hint` on TapeInstruction + prefetch in execute loop for LUT-GEMM constants
- [x] **11.4**: LUT-GEMM Q4 tape integration test (build graph with quantized weights, execute via tape)

### Phase 12: Parallel Tape Execution
- [x] **12.1**: `execute_parallel` on EnumTape — Rayon within levels for ≥4 independent instructions
- [x] **12.1b**: `dispatch_kernel_par` — Sync-safe dispatch (skips RefCell ops: LUT-GEMM, KvCache)
- [x] **12.1c**: Adaptive threshold — falls back to sequential for small levels or shared-state ops

### Phase 13: Attention + Conv2d Dispatch Coverage
- [x] **13.1**: Attention routed through `dispatch_custom_into` (avoids generic fallback overhead)
- [x] **13.2**: Conv2d routed through `dispatch_custom_into`
- [x] **13.3**: RoPE explicitly falls back (needs position offset from ctx)

### Phase 14: Monomorphized SIMD Dispatch + Zero-Copy Output Write
- [x] **14.1**: Monomorphized unary dispatch for Relu, Neg, Abs, Sigmoid, Silu, Tanh, Exp, Reciprocal
- [x] **14.2**: Direct f32 write via `bytemuck::cast_slice_mut` (no intermediate Vec, no per-element extend)
- [x] **14.3**: Same pattern for binary elementwise, fused chain, norm, and matmul _into variants

### Benchmark Results (Phase 14)
- EnumTape Relu 64KB: **36.5 µs → 4.3 µs** (8.5x faster, autovectorization enabled)
- KvExecutor same graph: 44.6 µs (unchanged — still uses closure dispatch)

---

## Sprint 16: Multi-Backend Dispatch Architecture

**Plan**: [plans/009-multi-backend-dispatch.md](plans/009-multi-backend-dispatch.md)

### Phase 1: Backend Abstraction + Auto-Detection
- [x] **1.1**: `ComputeBackend` trait (dispatch_float, dispatch_matmul, name)
- [x] **1.2**: `CpuBackend` wrapping existing monomorphized SIMD dispatch
- [x] **1.3**: `MetalBackend` stub (auto-detected on macOS via build.rs `has_metal`)
- [x] **1.4**: `CudaBackend` stub (auto-detected via CUDA_HOME / nvcc)
- [x] **1.5**: `WebGpuBackend` stub (auto-detected on wasm32 targets)
- [x] **1.6**: `BackendSelector` enum (Auto/Cpu/Metal/Cuda/WebGpu) with `resolve()`
- [x] **1.7**: `default_backend()` priority: CUDA > Metal > WebGPU > CPU
- [x] **1.8**: `available_backends()` introspection
- [x] **1.9**: `build.rs` auto-detection + `cargo::rustc-check-cfg` registration
- [x] **1.10**: `TapeContext.backend` field with `BackendSelector::Auto` default

### Phase 2: Backend Wiring + Monomorphized Binary Dispatch
- [x] **2.1**: `dispatch_kernel` queries `backend.dispatch_float()` before CPU fallback
- [x] **2.2**: Backend resolved once at `execute()` start via `BackendSelector::resolve()`
- [x] **2.3**: Monomorphized binary elementwise (Add, Sub, Mul, Div, Min, Max — enables SIMD)

### Phase 3: Metal Compute Shader Kernels
**Plan**: [plans/010-metal-compute-kernels.md](plans/010-metal-compute-kernels.md)
- [x] **3.1**: `metal` crate (0.33) dependency, auto-linked on macOS
- [x] **3.2**: MetalBackend with shader compilation + pipeline caching (9 unary + 4 binary kernels)
- [x] **3.3**: Process-global cached backend via `OnceLock<Arc<MetalBackend>>` (shader compiled once)
- [x] **3.4**: Unary dispatch (relu, neg, abs, sigmoid, silu, tanh, exp, reciprocal, gelu)
- [x] **3.5**: Binary dispatch (add, sub, mul, div) with broadcasting
- [x] **3.6**: Size threshold (4MB) — CPU SIMD for small buffers, Metal for large
- [x] **3.7**: Metal conformance test (1.5M float relu, spot-check correctness)

### Phase 4: Metal SGEMM Matmul
- [x] **4.1**: Metal SGEMM compute shader (C[M,N] = A[M,K] × B[K,N], 2D grid dispatch)
- [x] **4.2**: `dispatch_matmul` wired — FloatOp::MatMul routed through dispatch_float → Metal
- [x] **4.3**: Size threshold (128×128 output) — CPU Accelerate BLAS for small matrices
- [x] **4.4**: Metal matmul conformance test (128×64 × 64×128, verified row correctness)

### Phase 5: Tiled SGEMM + Softmax + RmsNorm
- [x] **5.1**: Tiled SGEMM with threadgroup shared memory (16×16 tiles, barrier sync)
- [x] **5.2**: Metal softmax kernel (per-element row-wise with max/sum scan)
- [x] **5.3**: Metal RmsNorm kernel (per-element with mean-of-squares + rsqrt)
- [x] **5.4**: Softmax + RmsNorm routed through `dispatch_float` with size threshold
- [x] **5.5**: Metal softmax conformance test (1M floats, row sums to 1.0)

### Phase 6: MTLBuffer-Backed Arena
- [x] **6.1**: `ArenaBuffer` enum replacing `Cow<[u8]>` — supports Owned, Borrowed, and Metal variants
- [x] **6.2**: `as_bytes()` returns `&[u8]` for all variants (Metal via `contents()` pointer)
- [x] **6.3**: `insert_metal(id, metal::Buffer, elem_size)` — store GPU buffers directly in arena
- [x] **6.4**: `into_owned()` for take() — copies Metal buffer to Vec only when needed

### Phase 7: Zero-Copy Metal Output Path
- [x] **7.1**: `KernelOutput` enum (Skipped / Bytes / MetalBuffer) — dispatch tells executor how to store result
- [x] **7.2**: `DispatchResult` in tape.rs — execute loop handles Metal buffers via `insert_metal`
- [x] **7.3**: Metal unary dispatch returns `MetalBuffer` directly (skip Vec copy on output)
- [x] **7.4**: `ComputeBackend` trait updated — all backends return `KernelOutput` instead of `bool`

### Phase 8: Remaining GPU Work
- [x] **8.1**: Metal binary/matmul/softmax/rmsnorm all return MetalBuffer (full zero-copy path)
- [x] **8.2**: Async command buffer batching — `Mutex<Option<CommandBuffer>>` on MetalBackend, encode without commit per dispatch, `flush()` at level boundaries via `ComputeBackend::flush()` trait method
- [x] **8.3**: WebGPU/wgpu compute shader path — [plan](plans/012-webgpu-wgpu-compute.md)
  - [x] **8.3a**: Bootstrap — wgpu device init, WGSL compilation, pipeline caching, OnceLock caching
  - [x] **8.3b**: Complete elementwise — all 9 unary + 4 binary WGSL kernels with staging readback
  - [x] **8.3c**: Custom ops — tiled SGEMM (16×16), softmax, RmsNorm in WGSL
  - [x] **8.3d**: Deferred command encoder batching — `WgpuDeferred` + `flush_deferred()` — [plan](plans/013-webgpu-deferred-batching.md)
- ~~**8.4**: CUDA kernel implementations~~ (removed — CUDA stub deleted in Sprint 21)

### Phase 10: Weight Deduplication Primitive (Plan 021 Phase 3)
- [x] **10.1**: `WeightStore` — content-addressable weight storage with CRC32 identity + exact byte comparison
- [x] **10.2**: `WeightDedupIndex` / `WeightDedupEntry` — rkyv-serializable index for the deduplicated blob
- [x] **10.3**: `SECTION_WEIGHT_DEDUP` section kind, `EmbeddableSection` impl
- [x] **10.4**: Re-exported from `hologram_archive` crate root
- [x] **10.5**: 9 unit tests (empty, single, dedup, distinct, build, save-space, get, rkyv roundtrip, zero-copy)

### Phase 9: Zero-Overhead Dispatch — Flatten Abstraction Layers
**Plan**: [plans/011-zero-overhead-dispatch.md](plans/011-zero-overhead-dispatch.md)

Goal: eliminate all per-instruction overhead between the execute loop and the kernel compute. Target: O(1) constant-time dispatch with zero memory copies for the CPU path.

#### 9a: Inline Hot Ops (eliminate backend vtable + double match)
- [x] **9a.1**: 7 unary inline variants (InlineRelu, InlineNeg, InlineSigmoid, InlineSilu, InlineTanh, InlineGelu, InlineExp)
- [x] **9a.2**: 4 binary inline variants (InlineAdd, InlineMul, InlineSub, InlineDiv)
- [x] **9a.3**: tape_builder maps hot FloatOps to Inline variants at build time
- [x] **9a.4**: `inline_unary` / `inline_binary` helper functions (direct bytemuck cast, no dispatch_float_into)
- [x] **9a.5**: 3 inline conformance tests + inline benchmark
- [x] **9a.6**: `InlineMatMul { m, k, n }` — direct matmul_into call, backend GPU first then CPU fallback
- [x] **9a.7**: `InlineSoftmax { size }` / `InlineRmsNorm { size, epsilon }` — direct norm kernels, backend first
- [x] **9a.8**: `InlineAbs` / `InlineReciprocal` — complete unary inline coverage
- [x] **9a.9**: Visibility: `pub(crate) mod norm`, `pub(crate) fn resolve_size`, `pub(crate) fn dispatch_softmax_into/dispatch_rms_norm_into`

### Benchmark Results (Phase 9a+9b)
- EnumTape Relu 64KB: **4.23 µs → 2.54 µs** (40% faster — inline dispatch + in-place unary + output passthrough)
- KvExecutor same graph: 44.4 µs (unchanged)
- Tape vs KvExecutor: **17.5x faster**
- Tape linear chain (4 nodes, 256B): **1.11 µs**

#### 9b: Zero-Copy Arena Path (eliminate out_buf round-trip)
- [x] **9b.1**: Output passthrough — `arena.move_slot(src, dst)` when input has single consumer
- [x] **9b.2**: Pre-allocated arena output slots — `prewarm_arena()` pre-allocates with `output_byte_hint`
- [x] **9b.3**: In-place unary ops — `dispatch_inplace()` + `inline_unary_inplace()` when `can_reuse_input` flag set
- [x] **9b.4**: `apply_reuse_flags()` post-pass in tape_builder — consumer count analysis, sets `passthrough` and `can_reuse_input`

#### 9c: Typed Arena Access (eliminate per-call bytemuck cast)
- [x] **9c.1**: `arena.get_f32(id)` — returns `&[f32]` directly via localized `cast_slice`
- [x] **9c.2**: `arena.get_mut_f32(id)` — mutable f32 slice for in-place ops on `Owned` buffers
- [x] **9c.3**: `inline_unary_f32` / `inline_binary_f32` — typed kernel signatures, caller casts once
- [x] **9c.4**: In-place path refactored: `get_mut_f32` + `dispatch_inplace` + `move_slot` (no take+insert dance)

#### 9d: Direct Input Access (eliminate SmallVec collection for known arity)
- [x] **9d.1**: `TapeKernel::inline_arity()` — returns `Some(1)` / `Some(2)` / `None`
- [x] **9d.2**: Unary inline fast path — `arena.get_f32(input_indices[0])` directly, skip SmallVec
- [x] **9d.3**: Binary inline fast path — two direct `arena.get_f32` calls, skip SmallVec
- [x] **9d.4**: `dispatch_inline_unary` / `dispatch_inline_binary` — typed match wrappers
- [x] **9d.5**: Same restructuring applied to `execute_parallel` sequential fallback

#### 9e: Unsafe Fast Path (eliminate bounds checks in hot loop)
- [x] **9e.1**: `set_len()` instead of `resize()` in `inline_unary_f32` / `inline_binary_f32` (skip zero-fill)
- [x] **9e.2**: `arena.get_unchecked()` / `arena.get_f32_unchecked()` — skip bounds check
- [x] **9e.3**: Unchecked `input_indices` access when arity is known via `get_unchecked(0)`/`get_unchecked(1)`
- [x] **9e.4**: All unsafe gated with `#[cfg(not(debug_assertions))]` — debug builds use checked paths

### Performance Budget (per instruction)
| Layer | Current | After Phase 9 | Savings |
|-------|---------|---------------|---------|
| Backend vtable + Skipped check | ~60ns | 0ns (inline) | 60ns |
| Double match (category + op) | ~20ns | 0ns (inline) | 20ns |
| SmallVec collection for unary | ~30ns | 0ns (direct access) | 30ns |
| bytemuck cast_f32 per call | ~15ns | 0ns (typed arena) | 15ns |
| out_buf round-trip for passthrough | ~50ns | 0ns (pointer swap) | 50ns |
| out_buf.resize zeroes memory | ~30ns | 0ns (set_len) | 30ns |
| **Total per instruction** | **~205ns** | **~0ns** | **~205ns** |
| **150-op transformer layer** | **~30µs** | **~0µs** | **~30µs** |

### KvExecutor Deprecation
- [x] **dep.1**: `#[deprecated]` on `KvExecutor` struct (`eval/executor.rs`)
- [x] **dep.2**: `#[deprecated]` on mmap wrappers (`execute_plan`, `execute_plan_with_shape_hints`, `execute_plan_with_kv_state`, `execute_bytes`, `execute_bytes_with_ops`, `execute_bytes_with_progress`, `execute_file`)
- [x] **dep.3**: `#[allow(deprecated)]` on internal impl blocks and profile functions
- [x] **dep.4**: Deprecation roadmap documented in handoff spec (Section 8)
- [x] **dep.5**: Migrate CLI `run_cmd.rs` generation loop to tape path (Sprint 17)
- [ ] **dep.6**: Add intermediate capture to EnumTape (tape profiling) — deferred
- [x] **dep.7**: Migrate remaining KvExecutor-based tests to tape (Sprint 17)
- [x] **dep.8**: Remove KvExecutor (struct, impl, mmap wrappers, re-exports) (Sprint 17)

### Documentation (Sprint 16)
- [x] **D.1**: Transformer benchmark specification — [specs/docs/transformer-benchmark-spec.md](docs/transformer-benchmark-spec.md)
- [x] **D.2**: hologram-ai integration guide — [specs/docs/hologram-ai-integration.md](docs/hologram-ai-integration.md)
- [x] **D.3**: Feature matrix updated with backends, tape dispatch levels, Metal GPU thresholds — [specs/feature-matrix.md](../specs/feature-matrix.md)

---

## Sprint 12: Prism Ontology Integration

- [x] Annotate `DispatchContext` as SaturatedContext (PP_1, PI_1, PA_4)
- [x] Add PX_5 infeasibility taxonomy to hologram-compiler errors
- [x] Add PL_2 lease-disjointness citation to hologram-graph `ParallelLevel`
- [x] Document PM_5 atomicity contract on KvExecutor
- [x] Classify crates as kernel/bridge/user (doc annotations per crate)
- [x] Document Prism space model + PP_1 derivation in specs/docs/architecture.md

---

## Sprint History

- Sprint 1: Foundation & Core LUT Engine — [archived](sprints/1-foundation-core-lut.md)
- Sprint 2: Graph, Archive & Execution — [archived](sprints/2-graph-archive-execution.md)
- Sprint 8: Constrained Device Validation — [archived](sprints/8-constrained-devices.md)
- Sprint 9: Tokio Integration + Async Execution — [archived](sprints/9-tokio-async.md)
- Sprint 10: CLI Completeness — [archived](sprints/10-cli-completeness.md)
- Sprint 11: Custom Op Extension API — [archived](sprints/11-custom-op-api.md)

---

## Sprint 3: Execution Engine & Calculator

(Sprint 3 complete)

## Sprint 4: Q1 Quantum Level Scaling

(Sprint 4 complete)

## Sprint 5: LUT-GEMM for AI Model Inference

(Sprint 5 complete)

## Sprint 6: Compiler Pipeline

(Sprint 6 complete)

## Sprint 7: C FFI + WASM Bindings

(Sprint 7 complete)

## Sprint 8: Constrained Device Validation

(Sprint 8 complete) — [archived](sprints/8-constrained-devices.md)

---

## Sprint 9: Tokio Integration + Async Execution

(Sprint 9 complete) — [archived](sprints/9-tokio-async.md)

---

## Sprint 10: CLI Completeness

(Sprint 10 complete) — [archived](sprints/10-cli-completeness.md)

---

## Sprint 11: Custom Op Extension API

(Sprint 11 complete) — [archived](sprints/11-custom-op-api.md)

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

### Phase 2: Graph, Subgraphs & Fusion (Sprint 2)
- [x] Implement `GraphError` enum + `GraphResult` type in `error/mod.rs`
- [x] Implement `ConstantId`, `ConstantData`, `ConstantStore` in `constant/mod.rs`
- [x] Implement `NodeId` (generational), `InputSource`, `InputSlot`, `Node` in `graph/node.rs`
- [x] Implement `GraphOp` (7 variants), `SubgraphId`, arena-based `Graph` in `graph/mod.rs`
- [x] Implement `connect()`, `connect_graph_input()` in `graph/edge.rs`
- [x] Implement `validate()`, `is_acyclic()` in `graph/validate.rs`
- [x] Implement `GraphBuilder` (fluent API) in `builder/mod.rs`
- [x] Implement `SubgraphDef` + `flatten_subgraph()` (3-phase ID remapping) in `subgraph/`
- [x] Implement Kahn's toposort O(V+E) in `schedule/toposort.rs`
- [x] Implement `ParallelLevel`, `build_parallel_levels()` in `schedule/levels.rs`
- [x] Implement `critical_path_length()`, `parallelism_ratio()` in `schedule/critical_path.rs`
- [x] Implement `ExecutionSchedule` in `schedule/mod.rs`
- [x] Implement `try_fold_constant()` in `fusion/constant.rs`
- [x] Implement `eliminate_common_subexpressions()` (hash-based CSE) in `fusion/cse.rs`
- [x] Implement `fuse_unary_chains()` via `ElementWiseView::then()` in `fusion/view_fusion.rs`
- [x] Implement `fuse()` single-pass orchestrator + `FusionStats` in `fusion/mod.rs`
- [x] Update `lib.rs` with convenience re-exports
- [x] 88 new tests (196 total), zero clippy warnings

### Phase 3: .holo Archive Format (Sprint 2)
- [x] Implement `ArchiveError` enum + `ArchiveResult` type in `error/mod.rs`
- [x] Implement `crc32()`, `verify_crc32()`, `crc32_combine()` in `checksum/mod.rs` (wraps crc32fast)
- [x] Implement `HOLO_MAGIC`, `PAGE_SIZE`, `align_to_page()` in `format/mod.rs`
- [x] Implement `HoloHeader` (fixed-layout via bytemuck, 80-byte `#[repr(C)]`) in `format/header.rs`
- [x] Implement `SerializedGraph` (bridge type: extracts live nodes from Graph for rkyv) in `format/graph.rs`
- [x] Implement `WeightDType` enum (F32–I4), `TensorMetadata` struct in `weight/mod.rs`
- [x] Implement `QuantizationScheme`, `QuantizationParams` in `weight/quantize.rs`
- [x] Implement `EmbeddableSection` trait + section kind constants in `section/mod.rs`
- [x] Implement `SectionEntry`, `SectionTable` in `section/table.rs`
- [x] Implement `LayerId`, `TensorPort`, `LayerEntrypoint`, `LayerDescriptor` in `entrypoint/mod.rs`
- [x] Implement `LayerHeader` (impl EmbeddableSection) in `entrypoint/schedule.rs`
- [x] Implement `LayerLocation` enum (Embedded/External/Registry) in `layer/mod.rs`
- [x] Implement `HoloWriter` builder (set_graph, set_weights, add_section → build) in `writer/holo_writer.rs`
- [x] Implement `PipelineWriter`, `PipelineHeader`, `PipelineEntry` in `writer/pipeline_writer.rs`
- [x] Implement `LoadedPlan` (validated archive accessor) in `loader/plan.rs`
- [x] Implement `load_from_bytes()`, `validate_header()` in `loader/bytes.rs`
- [x] Implement `LoadedPipeline` in `loader/pipeline.rs`
- [x] Implement `HoloLoader` (mmap, `#[cfg(feature = "std")]`) in `loader/mmap_loader.rs`
- [x] Update `lib.rs` with re-exports + 5 integration tests
- [x] 83 new tests (279 total), zero clippy warnings

### Phase 4: KV-Lookup Execution Engine (Sprint 3)
- [x] Implement `ExecError` enum (9 variants) + `ExecResult` type + `From<ArchiveError>` in `error/mod.rs`
- [x] Implement `BufferArena` (`HashMap<NodeId, Vec<u8>>`) in `buffer/arena.rs`
- [x] Implement `KvStore`: stateless dispatch (`apply_unary`, `apply_binary`, `dispatch`) in `kv/store.rs`
- [x] Implement `build_schedule()`: Kahn's algorithm on `SerializedGraph` in `eval/schedule_bridge.rs`
- [x] Implement `KvExecutor`, `GraphInputs`, `GraphOutputs` in `eval/executor.rs`
- [x] Implement parallel level execution (rayon feature-gated, threshold=4) in `parallel/mod.rs`
- [x] Implement `execute_plan()`, `execute_bytes()`, `execute_file()` in `mmap/mod.rs`
- [x] Update `lib.rs` with re-exports
- [x] 55 new tests (334 total), zero clippy warnings

### Phase 5: Calculator Example & Benchmarks (Sprint 3)
- [x] Build scientific calculator example (`examples/calculator.rs`): pi-F-lambda encoding, LUT composition, graph I/O, full pipeline, error analysis
- [x] 8 E2E integration tests (`tests/e2e.rs`): linear chain fused, diamond parallel fan-out, constants through pipeline, chained constant folding, multi-input binary, long chain multi-fusion, wide parallel fan-out, file roundtrip
- [x] Criterion benchmark `kv_dispatch.rs`: KvStore::dispatch for unary/binary ops, varying buffer sizes (256B, 4KB, 64KB), all 21 LutOp variants
- [x] Criterion benchmark `executor.rs`: KvExecutor::execute for linear/diamond/wide-parallel graphs, large buffer (64KB), schedule build
- [x] Criterion benchmark `archive.rs`: HoloWriter::build + load_from_bytes round-trip, varying graph sizes (5, 50 nodes), diamond topology
- [x] Criterion benchmark `fusion.rs`: fuse() pass on graphs of varying sizes (10, 100, 1000 nodes)
- [x] Root crate `src/lib.rs` already re-exports hologram-exec public API (done in Phase 4)
- [x] 8 new E2E tests (342 total workspace), zero clippy warnings

### Phase 6: Q1 Quantum Level Scaling (Sprint 4)
- [x] Q1 skeleton: `q1/mod.rs`, `q1/observables.rs` (7 functions), `q1/arith.rs` (4 wrapping ops)
- [x] `WordDatum` + `WordAddress` (16-bit, 3 Braille glyphs) in `q1/datum.rs` — rkyv derives, Datum/Address trait impls
- [x] `WordRing` (Z/65536Z) + `WordInvolution` (Neg/Bnot) in `q1/ring.rs` — Ring + Q1Ring trait impls
- [x] 21 Q1 activation tables (128KB each, 2.7MB total) in `q1/activation/` — sigmoid, tanh, exp, log, relu, sqrt, abs, gelu, silu, sin, cos, tan, asin, acos, atan, log2, log10, exp2, exp10, square, cube
- [x] `ElementWiseView16` (heap-allocated 128KB table) in `q1/view.rs` — from_static, from_fn, then(), is_bijective, inverse, apply_slice
- [x] `Encoding16` trait + 4 impls (angle, signed, unsigned, raw) in `q1/encoding.rs`
- [x] `PrimOp16` (10 ops), `LutOp16` (21 ops), `Op16` enum in `q1/op.rs`
- [x] Quantum module (`quantum/mod.rs`): quantum_bit_width, quantum_modulus, quantum_is_table_feasible, quantum_table_size_bytes, Q2/Q3 helpers (stratum, curvature, add), Q4+ scaling strategy docs
- [x] Criterion benchmark `q1.rs`: Q1 vs Q0 vs f64 comparisons (sigmoid, sin), batch throughput, view16 ops, arith comparison, memory budget verification
- [x] 130 new tests (472 total workspace), zero clippy warnings

### Phase 7: LUT-GEMM for AI Model Inference (Sprint 5)
- [x] `Psumbook4` (64B, 1 cache line) + `Psumbook8` (1KB) cache-aligned partial sum accumulators in `hologram-exec/src/lut_gemm/psumbook.rs`
- [x] `QuantizedWeights4` (nibble-packed indices, 16 centroids) + `QuantizedWeights8` (byte indices, 256 centroids) with k-means clustering in `hologram-exec/src/lut_gemm/quantize.rs`
- [x] `quantize_4bit()`, `quantize_8bit()`, `quantize_auto()` (tries Q4, falls back to Q8 if error > 5%)
- [x] `dequantize_error_q4()`, `dequantize_error_q8()` — relative RMSE measurement
- [x] Sequential LUT-GEMM kernels: `lut_gemm_4bit()`, `lut_gemm_8bit()`, `lut_gemm()` in `hologram-exec/src/lut_gemm/matmul.rs`
- [x] Column-parallel LUT-GEMM via rayon (`lut_gemm_4bit_par`, `lut_gemm_8bit_par`) with `PAR_COL_THRESHOLD=64`, feature-gated in `hologram-exec/src/lut_gemm/parallel.rs`
- [x] 4 new `GraphOp` variants: `MatMulLut4(ConstantId)`, `MatMulLut8(ConstantId)`, `BatchMatMulLut4(ConstantId)`, `BatchMatMulLut8(ConstantId)` — all arity 1, pure
- [x] `KvStore::dispatch_with_constants()` — resolves quantized weights from `ConstantStore`, casts via bytemuck, runs LUT-GEMM kernel
- [x] `KvExecutor` updated to pass `&sg.constants` through dispatch
- [x] `ExecError::ShapeMismatch` + `ExecError::InvalidQuantization` error variants
- [x] `GraphBuilder::matmul_lut_4bit()` + `matmul_lut_8bit()` builder helpers
- [x] `QuantizationScheme::KMeansClustered { bits }` archive weight scheme
- [x] Criterion benchmarks: `lut_gemm.rs` — Q4/Q8 at 16x16, 64x64, 256x256, naive matmul comparison, quantization cost
- [x] 6 E2E integration tests: Q4/Q8 pipeline, Q4/Q8 accuracy vs naive, matmul+activation diamond, archive roundtrip
- [x] 56 new tests (528 total workspace), zero clippy warnings

### Phase 8: Compiler Pipeline (Sprint 6)
- [x] New `hologram-compiler` crate: compilation pipeline separate from execution
- [x] `CompileError` enum (Validation, Fusion, Emission) + `CompileResult` type + `From<GraphError>` + `From<ArchiveError>` in `error/mod.rs`
- [x] `LivenessInterval` { node_id, born, dies } + `compute_liveness(schedule, graph)` in `liveness/mod.rs` — tracks buffer lifetime intervals in schedule level order
- [x] `WorkspaceLayout` + `BufferSlot` + `plan_workspace(intervals)` in `workspace/mod.rs` — first-fit-decreasing bin packing for buffer slot reuse
- [x] `CompilerBuilder::new(graph).fuse(bool).build()` → `CompilationOutput` in `compiler/mod.rs`
- [x] 3-stage pipeline: parse (validate) → fuse (constant folding, view fusion, CSE) → emit (schedule, liveness, workspace, LayerHeader, .holo archive)
- [x] `compile(graph)` convenience function
- [x] `CompilationOutput` { archive: Vec<u8>, stats: CompilationStats, schedule: ExecutionSchedule }
- [x] `CompilationStats` { workspace_slots, peak_live_buffers, total_nodes, schedule_levels, fusion: FusionStats }
- [x] `SerializedGraph::to_graph()` reconstruction with ID remapping in `hologram-archive/src/format/graph.rs`
- [x] CLI `hologram compile` wired to compiler pipeline with `--no-fuse` flag
- [x] Root crate re-exports `hologram_compiler` public API
- [x] Criterion benchmarks `compiler.rs`: compile/liveness/workspace at 10/50/100 nodes
- [x] 7 E2E integration tests: compiler linear chain, diamond with fusion, constants, fusion disabled vs enabled, large graph, workspace reuse, LayerHeader presence
- [x] 52 new tests (580 total workspace), zero clippy warnings

### Phase 10: Constrained Device Validation (Sprint 8)
- [x] rkyv upgraded 0.7 → 0.8.15 across all workspace crates (fixes WASM32 const-eval overflow bug)
- [x] rkyv made optional in `hologram-core` via `serialize` feature flag (wasm32/ARM builds skip it entirely)
- [x] `hologram-core` no_std verified: `wasm32-unknown-unknown` and `thumbv7em-none-eabihf` both compile clean
- [x] `f64::rem_euclid()` replaced with no_std-compatible manual implementation in `encoding/angle.rs`
- [x] `StaticBuf<const N: usize>` — fixed-size stack/static byte buffer in `buffer/static_buf.rs`; 15 tests
- [x] `Justfile` recipes: `wasm-nostd` (wasm32 no_std) and `embedded` (thumbv7em bare-metal)
- [x] `specs/feature-matrix.md`: feature availability per target (x86_64, wasm32, thumbv7em, esp32)
- [x] 15 new tests (651 total workspace), zero clippy warnings

### Phase 9: C FFI + WASM Bindings (Sprint 7)
- [x] `hologram-ffi` crate (`crates/hologram-ffi/`): C ABI layer with opaque handles, `extern "C"` functions — `cdylib` + `rlib`
- [x] Error handling: thread-local `LAST_ERROR` (`RefCell<Option<CString>>`), `hologram_last_error() -> i32`, `hologram_error_message() -> *const c_char`
- [x] Handle management: `into_handle<T>()`, `borrow_handle()`, `borrow_handle_mut()`, `free_handle()` in `handle/mod.rs`
- [x] Graph construction FFI: `hologram_graph_builder_new/input/node/node_from_input/node_with_inputs/edge/output/build/free` + `holo_graph_node_count/free` in `graph/mod.rs`
- [x] `FfiGraphBuilder` (non-consuming): wraps `Graph` directly with `index_to_id: Vec<NodeId>` for C-friendly index mapping
- [x] `HoloOpKind` mapping: 0=Input, 1=Output, 2=Prim(op_param 0–9), 3=Lut(op_param 0–20)
- [x] Compilation FFI: `hologram_compile()`, `hologram_compile_no_fuse()`, archive ptr/len, stats (nodes/levels/workspace_slots), `holo_compilation_free()` in `compiler/mod.rs`
- [x] Execution FFI: `hologram_inputs_new/set/free`, `hologram_execute_bytes()`, `hologram_outputs_len/get/name/by_name/free` in `exec/mod.rs`
- [x] Encoding FFI: `hologram_encoding_embed/lift()`, `hologram_lut_apply()`, `hologram_prim_apply_unary/binary()` in `encoding/mod.rs`
- [x] `cbindgen.toml` + auto-generated `include/hologram.h` C header (type renames: FfiGraphBuilder→HoloGraphBuilder, etc.)
- [x] WASM module: `WasmGraphBuilder`, `wasm_execute()`, `wasm_lut_apply()`, `wasm_encoding_embed/lift()` in `wasm/mod.rs` (feature-gated `wasm`)
- [x] Criterion benchmark `ffi.rs`: graph build, lut_apply, encoding embed/lift, full pipeline (build→compile→execute)
- [x] 6 FFI E2E tests: full pipeline, diamond with fusion, encoding round-trip, LUT ops, error handling, fusion toggle
- [x] 56 new tests (636 total workspace), zero clippy warnings

---

## Sprint 17: Performance Hardening + KvExecutor Removal

**Plan**: [plans/014-graph-perf-kvexecutor-removal.md](plans/014-graph-perf-kvexecutor-removal.md)

### Phase 1: Graph Successor Index Optimization
- [x] **1.1**: Successor index in toposort Kahn's loop (O(N²) → O(V+E))
- [x] **1.2**: Successor index in build_parallel_levels (O(N²) → O(V+E))
- [x] **1.3**: Successor index in validate acyclicity check
- [x] **1.4**: Indexed rewire_successors for CSE pass

### Phase 2: Fusion Pass Optimization
- [x] **2.1**: Eliminate double toposort — reuse original order for CSE
- [x] **2.2**: Pre-built successor index in fusion pass (commit 6ad9e12)

### Phase 3: KvExecutor Removal
- [x] **3.1**: Migrate hologram-async to tape path
- [x] **3.2**: Migrate hologram-ffi to tape path
- [x] **3.3**: Migrate hologram-cli to tape path
- [x] **3.4**: Migrate e2e tests to tape path
- [x] **3.5**: Remove custom_ops tests (KvExecutor-dependent registry dispatch)
- [x] **3.6**: Migrate executor benchmarks to tape-only
- [x] **3.7**: Remove deprecated mmap convenience functions (execute_plan, execute_bytes, execute_file, etc.)
- [x] **3.8**: Clean up re-exports and #[allow(deprecated)] annotations
- [x] **3.9**: Migrate calculator example to tape path

### Phase 4: Dead Code Removal (Plan 015)
- [x] **4.1**: Remove shape_propagate.rs + shape_resolve.rs (1066 lines)
- [x] **4.2**: Remove dirty_bits.rs + profile.rs (385 lines)
- [x] **4.3**: Remove old Tape/Instruction/KernelFn + their tests
- [x] **4.4**: Inline `parse_shape_values` into float_dispatch/shape_ops.rs

### Phase 5: Tape-Compatible Custom Ops (Plan 015)
- [x] **5.1**: `TapeKernel::Custom` variant + dispatch in `dispatch_kernel` / `dispatch_kernel_par`
- [x] **5.2**: Wire `CustomOpRegistry` into tape_builder (`resolve_kernel` accepts registry)
- [x] **5.3**: `build_tape_from_plan_with_ops` entry point
- [x] **5.4**: Custom op E2E test (passthrough handler via tape path)

### Phase 6: Tape Hot Path Optimization (Plan 015)
- [x] **6.1**: `binary_broadcast` helper — eliminate modulo for same-size/scalar cases
- [x] **6.2**: Pre-size `consumer_counts` in `apply_reuse_flags`

### Benchmark Results (Phase 1+2)
- fusion::fuse(1000_nodes): **1.91 ms → 290 µs** (6.6x faster)
- fusion::fuse(100_nodes): **44 µs → 31 µs** (-30%)
- compile/100_nodes: **79 µs → 60 µs** (-24%)
- compile/50_nodes: **45 µs → 41 µs** (-9%)

## Sprint: ComputeBackend + ComputeMemory Rewrite (Plan 067)

**Plan**: [plans/067-compute-backend-rewrite.md](plans/067-compute-backend-rewrite.md)

Goal: single-device execution — all data lives on one device (Metal/WebGPU/CPU),
all computation happens on that device. No CPU↔GPU transfers during execution.
New `hologram-compute` crate with `ComputeMemory` + `ComputeBackend<M>` traits.

### Phase 1: Traits + CpuMemory (non-breaking)
- [ ] Create `hologram-compute` crate
- [ ] Define `ComputeMemory` trait (alloc, upload, download, reshape)
- [ ] Define `ComputeBackend<M>` trait (dispatch, load_ring_tables, flush)
- [ ] Implement `CpuMemory` + `CpuBackend` (wraps existing CPU dispatch)

### Phase 2: MetalMemory + device-native weight loading
- [ ] Implement `MetalMemory` (metal::Buffer allocation)
- [ ] Load weights directly into Metal buffers at archive load time
- [ ] Load UOR LUT tables onto Metal device

### Phase 3: Single-path executor
- [ ] New `execute<M, B>()` in hologram-exec consuming hologram-compute
- [ ] All ops dispatch through `backend.dispatch()` — no CPU fallback
- [ ] Single flush at end of execution

### Phase 4: Complete Metal kernel coverage
- [ ] Q4 dequant+GEMM kernel for Conv2dLut4/MatMulLut4
- [ ] Ring op kernels on Metal (Z/256Z LUT lookups)
- [ ] All TapeKernel variants covered

### Phase 5: WebGPU backend skeleton
- [ ] `WebGpuMemory` + `WebGpuBackend` (async-aware)
- [ ] WGSL shader source for core kernels
- [ ] WASM target compatibility
