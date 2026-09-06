# Conformance BDD suites

Gherkin `.feature` suites for the hologram **refactor** (`specs/refactor/00`–`07`),
run by the `cucumber` crate in `crates/hologram-conformance`. Modeled on
`afflom/UOR-Atlas-UTQC`'s `features/suites`, informed by and cross-linked to the
root `CONFORMANCE.md` normative ledger.

## Layout

- `suites/s0_laws` … `suites/s6_governance` — one suite per refactor spec area.
- `suites/s7_readme` — the **RM** class: every fenced code block in the repo `README.md`
  is bound to exactly one row (`RM-N` ≡ the N-th fenced block, top-to-bottom). **30 rows are
  BDD scenarios** here, driven through the public facade / CLI / C ABI the README documents
  (step defs in `crates/hologram-conformance/tests/rm_steps/`). **5 rows are witnessed
  externally** — the SDK & browser surfaces the Rust `bdd` gate cannot run — bound to their own
  package tests (`sdk/python`, `sdk/typescript`, `spaces/holospaces-browser`) by the meta-gate's
  `check_witnessed_rows` audit, the same way the `CC`/`CS` classes cite cargo tests / scripts.
- Each scenario is tagged `@class:<C> @id:<C-N> @spec:<doc> @phase:<Pn> @status:<s>`.
- `@class`/`@id` bind the scenario to a `CONFORMANCE.md` row (classes LAW/SP/HF/NW/TL/MG/GV/RM).

## Status vocabulary (cross-walked to the CONFORMANCE.md legend)

| `@status` | scenario | catalog |
|---|---|---|
| `enforced` | all steps assert & pass | ✅ enforced |

## Running

- `just bdd` — run the suite + the honesty meta-gate.
- `just conformance-report` — verify the catalog ↔ scenario bijection (fails on drift).

## Honesty rule

The meta-gate (`crates/hologram-conformance/tests/meta_gate.rs`) statically enforces, for
every BDD-class row: exactly one scenario with the same `@id`; the row's status glyph
agrees with the scenario's `@status`; the row's `Witness` path + scenario name matches the
actual feature file; and each feature file declares exactly one scenario. This keeps the
ledger and the scenarios from drifting.

The static gate does not assert that an `enforced` scenario actually *passes* — that is
the runner's job. The `bdd` runner (`tests/bdd.rs`) uses `fail_on_skipped()`: every
undefined or skipped step is a build failure, without a tag-based exception. An enforced
row can only be green if its scenario has real, passing steps.

## Phased rollout

New scenarios must land with executable steps and `@status:enforced`; incomplete scenarios
are not registered. Promotion requires (1) executable step definitions, (2) the enforced
tag, and (3) the matching `CONFORMANCE.md` row at `✅`. The meta-gate rejects any mismatch.
