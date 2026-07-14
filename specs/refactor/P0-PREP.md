# P0 Preparation — Human-Gated Items

P1 (production code moves) is blocked until P0's human-owned gates clear (D23, D24, D29).
This doc holds ready-to-action artifacts for each. The agent-owned P0.5 spike and the
P1-preflight golden vectors (MG-5) + LICENSE files are already done on `chore/refactor`.

Status legend: ☐ not started · ◐ in progress · ☑ done.

## 1. Relicense consent (D24) — ☐

The workspace declares `MIT OR Apache-2.0` and now carries `LICENSE-MIT` + `LICENSE-APACHE`.
`holospaces` is currently `MIT` with an external contributor (Alex Flom). Before its code
is merged and published under the dual license, written consent is required.

**Draft request to send:**

> Subject: Relicense consent — holospaces → MIT OR Apache-2.0
>
> Hi Alex — we're consolidating holospaces into the main hologram repository, which is
> dual-licensed `MIT OR Apache-2.0` (the Rust-ecosystem standard: MIT keeps it permissive,
> Apache-2.0 adds an explicit patent grant). Your contributions to holospaces are currently
> MIT-only. Could you confirm, in writing (a reply here or a signed note on the PR), that
> you consent to your holospaces contributions being relicensed to `MIT OR Apache-2.0`?
> Nothing about the permissive nature changes; this only adds the Apache patent grant.
> Thanks!

**Exit:** consent recorded (link the reply/PR comment here). Publishing blocks without it.

## 2. Restructuring review (D29) — ☐

Bundle with the consent ask: a review of the parts of the spec suite that restructure
holospaces' work, so objections surface before code moves, not after.

Review checklist (share these sections):
- [ ] `01-crate-map.md` §"From `../holospaces`" — the Peer/Session/Manager hoist to
      `hologram-runtime`; emulators/κ-disk/boot → `spaces/holospaces`.
- [ ] `02-space-contract.md` §"What was hoisted out of holospaces (D7)".
- [ ] `06-migration.md` P2 (import) + P3 (hoist) — history-preserving subtree merge, CI
      absorption (CC catalog, QEMU oracles, Playwright), docs relocation to
      `specs/holospaces/`.

**Exit:** contributor acknowledges the restructuring plan (or files change requests).

## 3. crates.io name availability + ownership (D16) — ◐ (audited 2026-07-14)

Audited via `cargo search`. **One blocker: the facade name `hologram` is taken.**

### ⚠ BLOCKER — `hologram` is unavailable
`hologram = "0.1.4"` already exists on crates.io — an unrelated "Interpolation library
with multipurpose Radial Basis Function (RBF)." This is the **facade crate name**, the
heart of the "just add `hologram` with features" DX (D4). A user decision is required
before P3 publishes. Options:
1. **Acquire the name** — contact the `hologram` v0.1.4 owner about a transfer (it's a
   tiny, dormant niche crate; may be gettable). Best outcome — preserves D4 verbatim.
2. **Rename the facade** — **`uor-hologram`** (confirmed free on crates.io; ties the
   umbrella to the UOR-Foundation family). Keeps the `hologram::` *module* paths in code
   (`use uor_hologram as hologram;` or re-export), but changes the `Cargo.toml` line users
   write. Undercuts the headline one-word import slightly, but reads as intentional
   namespacing rather than an unavailable-name workaround.
3. **Scoped/prefixed publish** now, revisit acquisition later.
Recommendation: pursue (1) first; fall back to (2) with `uor-hologram` if unavailable.

**DECISION (2026-07-14):** pursue acquisition of `hologram`; if it can't be obtained,
publish the facade as **`uor-hologram`** (crate name), preserving the `hologram::` module
path via re-export so user code reads `use hologram::...`. Until resolved, the in-tree
facade package name stays `hologram` (`publish = false`-safe); the published `Cargo.toml`
`name =` is the only thing that changes at P3.

**Draft acquisition message (to the `hologram` v0.1.4 owner via their crates.io/repo contact):**

> Subject: Would you consider transferring the `hologram` crate name?
>
> Hi — I maintain the Hologram runtime project (github.com/Hologram-Technologies), a
> content-addressed execution runtime we're preparing to publish. The crates.io name
> `hologram` is the natural umbrella for our workspace. I see you hold `hologram = 0.1.4`
> (an RBF interpolation library). Would you be open to transferring the name — or adding
> us as an owner — perhaps with your library continuing under a more descriptive name
> (e.g. `rbf-interp`)? Happy to help with the migration and credit you. Thank you for
> considering it.

If declined or no response in a reasonable window, proceed with `uor-hologram`.

### Free (confirmed available)
All other target names are FREE:
- Core: hologram-types, hologram-ops, hologram-graph, hologram-archive, **hologram-compute**,
  hologram-exec, hologram-compiler, **hologram-space**, hologram-runtime, **hologram-net**,
  hologram-tck, hologram-ffi, hologram-cli, hologram-bench.
- Spaces: **holospaces**, holospaces-browser, holospaces-native, holospaces-bare.

Notes:
- `hologram-conformance` and `hologram-spike-sp3` are `publish = false` (not released).
- Retired names need no reservation: `hologram-host` (→ types), `hologram-backend`
  (→ compute), and every `hologram-substrate-*` / `hologram-store-*` / `hologram-runtime-*`
  / `hologram-net-*` / `hologram-bare-hal` (absorbed per `01-crate-map.md`).
- Still TODO: publish tokens + org ownership (Hologram-Technologies), and re-confirm the
  free names are still free at release time (someone could claim one in the interim —
  consider reserving the critical few with a `0.0.0` placeholder publish).

**Exit:** the `hologram` name decision is recorded; tokens/ownership settled; critical
names optionally reserved.

## 4. holospaces HEAD sync (D23, P0 proper) — ☐

In the `../holospaces` repo (still git-pinned): port from its pinned hologram rev to
hologram HEAD (absorb the breaking changes — kappa-leases, fused decode attention), get
its full V&V green, cut the bridge tag `hologram-ai` will pin. This is real engineering
that lives in the sibling repo; it makes the always-green gate real from P1. The agent can
do this work if pointed at the holospaces repo as a working directory.

**Exit:** holospaces V&V green against a hologram HEAD pin; bridge tag published;
`hologram-ai` switched onto it.

## When all four clear

P1 begins (per `06-migration.md`): the golden vectors (MG-5, done) + perf baselines are
the reference; then the renames/moves land, each keeping the enforced BDD scenarios green.
