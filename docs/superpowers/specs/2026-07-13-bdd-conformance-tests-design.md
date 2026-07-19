# BDD Conformance Tests — Design

Status: **Accepted** (brainstorming session 2026-07-13).
Owner spec inputs: `specs/refactor/00`–`07`, `CONFORMANCE.md`.
Reference model: [`afflom/UOR-Atlas-UTQC` `features/suites`](https://github.com/afflom/UOR-Atlas-UTQC/tree/main/features/suites).

## Purpose

Give the hologram **refactor** an executable, spec-first conformance layer in the
Gherkin + `cucumber` BDD style of the reference repo, cross-linked to and *informed by*
`CONFORMANCE.md`. Scenarios exist from day one, start `pending`, and turn green as the
refactor phases (P0–P6, see `specs/refactor/06-migration.md`) land. The suite is the
executable acceptance test for each phase and the mechanical enforcement of the honesty
rule: *no requirement is "done" until its scenario is green and CI-gated.*

Scope decision (2026-07-13): cover the **refactor laws + contract surface** (the new
normative surface introduced by `specs/refactor/*`), not a re-wrapping of the existing
green AS/MA/KC/… tensor invariants.

## Non-goals

- Re-expressing the existing tensor/exec invariant classes (AS, MA, KC, CA, RF, SG, FU,
  WS, WL, EL, CN, AD, PV, PA, NS, RP) as Gherkin — they already have passing Rust
  witnesses and stay as-is.
- Designing the full governance/attestation system (07 is requirements-only; the BDD
  scenarios encode the *boundary rules*, not a finished design).
- Any implementation of the refactor itself. This design adds the test scaffold only.

## Reference-repo model (what we are mirroring)

The reference repo uses Rust + the `cucumber` crate in a dedicated crate
(`crates/tqc-conformance` — "the BDD (cucumber) runner + the honesty meta-gate"), with
Gherkin `.feature` files under `features/suites/sN_topic/`. Scenarios carry tags like
`@row:… @stage:S2 @status:some-true @oracle:mtc-axioms`. A three-level honesty discipline
(`some-true` / `build` / `open`) is reported via `just report`, run via `just bdd`.

We adopt the **structure, tag discipline, and honesty meta-gate**, and remap the status
vocabulary to an enforcement axis (below) because the refactor is spec-first.

## Section 1 — Architecture & layout

```
features/                              # NEW — mirrors UOR-Atlas-UTQC/features
  README.md                            # how to run, the honesty gate, catalog linkage
  suites/
    s0_laws/                           # 00-overview laws (SPINE-1..6, κ-only, attenuation, async/sync…)
    s1_space_contract/                 # 02-space-contract + TCK battery
    s2_holo_format/                    # 03-holo-format (.holo v3, composition, views)
    s3_networks/                       # 04-networks (KappaSync, DHT, Network realization, tiers)
    s4_tooling/                        # 05-tooling (one binary, Client facade, FFI)
    s5_migration/                      # 06-migration (phased always-green gates)
    s6_governance/                     # 07-governance (R1–R4 boundary rules)
crates/hologram-conformance/           # NEW test-only crate = reference's `tqc-conformance`
  Cargo.toml                           # dev-deps: cucumber, tokio; the BDD runner + meta-gate
  tests/bdd.rs                         # cucumber entrypoint (points at ../../features)
  src/
    lib.rs                             # World, shared step context
    steps/                             # step definitions, one module per suite
    catalog.rs                         # CONFORMANCE.md parser (rows → {class,id,status})
    report.rs                          # status regeneration + bijection meta-gate
  tests/meta_gate.rs                   # the honesty gate (bijection + status agreement)
```

- **`hologram-conformance`** is the analog of `tqc-conformance`: the `cucumber` runner
  **plus the honesty meta-gate**. It is a **leaf-tier** crate (D22 — may depend on
  anything; nothing depends on it). It is added to the workspace `members` but **not**
  `default-members`, so it never enters the core build/test graph or the published crate
  set.
- Feature files live at repo root under `features/` (not inside the crate) so they read
  as first-class specs, exactly like the reference repo.
- Async: the `cucumber` runner is async (tokio), consistent with law 4
  (async contracts / sync compute) — the World models the async contract boundary; step
  bodies that touch tensor compute stay synchronous.

## Section 2 — Suite ↔ CONFORMANCE.md linkage (the core)

`CONFORMANCE.md` is a hand-maintained normative ledger whose rows are
`ID │ Statement │ Enforcement │ Witness │ Status`, grouped into invariant **classes**,
with a status legend (✅ enforced & passing · 🟡 partial · ⛔ gap). We **extend** the
catalog with new classes for the refactor surface, leaving existing classes untouched:

| New class | Scope | Spec |
|---|---|---|
| **LAW** | Repo-wide laws: SPINE-1..6, κ-only identity, capability attenuation, async-contract/sync-compute, one programmatic surface | 00 |
| **SP** | Space contract trait set + laws + TCK conformance battery; external-repo space parity (D21) | 02 |
| **HF** | `.holo` v3 container, capability-attenuated nesting, per-layer certificates | 03 |
| **NW** | Network κ-realization, KappaSync/DHT, public/restricted/private tiers | 04 |
| **TL** | Exactly one `hologram` binary, one public facade crate, FFI over Client | 05 |
| **MG** | Phased always-green migration gates (P0–P6), holospaces V&V at each boundary | 06 |
| **GV** | Governance R1–R4 boundary rules (traceability, audit, attestation, data governance) | 07 |

Linkage rules:

- **One catalog row ↔ one Gherkin scenario.** Each row (e.g. `LAW-2`, `GV-1`) has exactly
  one scenario whose tags name that row; the scenario **is** the row's `Witness`.
- The row's `Enforcement` cell reads "BDD scenario (+ underlying unit/integration test
  when one exists)". The `Witness` cell points at
  `features/suites/sN_.../file.feature::<Scenario name>`.
- **Bijection meta-gate** (`tests/meta_gate.rs`): parse `CONFORMANCE.md` and every
  `.feature` file; **fail** if — (a) a BDD-class row has no scenario; (b) a scenario names
  a class/id absent from the catalog; (c) a row's declared status disagrees with the
  scenario's actual run result. This makes the honesty rule mechanical.

## Section 3 — Scenario & tag conventions + honesty/status

Tags mirror the reference repo (`@row @stage @status @oracle`), remapped:

```gherkin
@class:GV @id:GV-1 @spec:07-governance §R1 @phase:P5 @status:pending
Feature: Traceability — every artifact traces to inputs by κ alone
  Scenario: a realization embeds its operand κs so references() yields full provenance
    Given an AppManifest realization built from known operand κs
    When I call references() on it
    Then the returned set equals the full provenance closure with no side tables
```

- `@class` / `@id` — the CONFORMANCE.md class and row id (drives the bijection gate).
- `@spec` — the owning refactor doc + section, for traceability back to requirements.
- `@phase` — the migration phase (P0–P6) that is expected to turn this scenario green.
- `@status` — enforcement axis, cross-walked to the CONFORMANCE.md legend:

  | `@status` | scenario behavior | catalog legend |
  |---|---|---|
  | `pending` | steps `skip` (cucumber pending) | ⛔ gap |
  | `partial` | some steps assert, rest skip | 🟡 partial |
  | `enforced` | all steps assert and pass | ✅ enforced & passing |

- **Teeth:** a scenario tagged `@status:enforced` that skips or fails breaks the build.
  You cannot claim ✅ without green steps.
- Step definitions begin as `pending` for unimplemented laws; each phase PR flips its
  scenarios to real assertions and updates the catalog row + `@status` together (the meta
  gate rejects a mismatch, so they cannot drift).

Rationale for `pending/partial/enforced` over the reference's `some-true/build/open`:
those describe *evidence provenance*; the refactor's useful axis is *enforcement state*.
The mapping to ✅/🟡/⛔ keeps a single source of truth in `CONFORMANCE.md`.

## Section 4 — Runner, Just targets, phased rollout

- **`just bdd`** — `cargo test -p hologram-conformance --test bdd` (the cucumber suite).
- **`just conformance-report`** — regenerate the CONFORMANCE.md status column for BDD
  classes from actual scenario results (analog of the reference repo's `just report`);
  fails on any drift (delegates to the meta-gate).
- **`just vv`** — add a `bdd` step so full V&V includes the conformance suite. CI gates on
  `just bdd` and `just conformance-report`.
- **Rollout is phase-aligned with `06-migration.md`:** land the full `.feature` tree now
  with every scenario `@status:pending`; each phase P0–P6 turns its scenarios `enforced`
  as the implementation arrives. The suite is the executable acceptance criterion for each
  phase boundary.

## Components & responsibilities (isolation view)

- **`features/suites/*.feature`** — the human-readable normative scenarios. Depend on
  nothing; define the acceptance surface. Changed when a refactor requirement changes.
- **`catalog.rs`** — parses `CONFORMANCE.md` into `{class, id, statement, status}`. Input:
  the markdown ledger. Output: a typed row set. No knowledge of cucumber.
- **`report.rs` + `meta_gate.rs`** — cross-check scenarios against catalog rows and
  regenerate the status column. Depend on `catalog.rs` and the parsed feature set.
- **`steps/`** — bind Given/When/Then to real assertions against the refactor
  implementation as it lands. Depend on the `hologram` facade (leaf-tier).
- **`World`** — per-scenario async context (space/session/client handles). The only
  async↔sync seam, per law 4.

## Error handling

- Malformed feature tags (unknown `@class`, missing `@id`) → meta-gate hard-fails with the
  offending file/scenario named.
- Catalog/scenario drift (row without scenario, scenario without row, status mismatch) →
  meta-gate hard-fails; `just conformance-report` refuses to write a divergent ledger.
- `pending` steps use cucumber's native skip (not panic) so a phase-incomplete suite is
  *skipped*, never *failing* — but an `enforced`-tagged skip is upgraded to a failure.

## Testing (of the harness itself)

- Unit tests for `catalog.rs` (parse a fixture ledger) and the tag parser.
- A meta-gate self-test: a deliberately drifted fixture (row without scenario) must fail.
- The meta-gate runs in CI on every PR, so the bijection can never silently rot.

## Open questions / deferred

- Exact per-row scenario text for each class is authored during `writing-plans` /
  implementation, one suite at a time, alongside the phase that enforces it.
- Whether `hologram-tck` (space contract battery) scenarios call the TCK directly or
  re-derive assertions — resolved when P1/P2 lands the contract crate.
```
