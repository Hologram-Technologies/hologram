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
| `pending` | steps skip (undefined) | ⛔ gap |
| `partial` | some steps assert | 🟡 partial |
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
the runner's job. The `bdd` runner (`tests/bdd.rs`) is wired with
`fail_on_skipped_with(@status:enforced)`: any scenario tagged `@status:enforced` whose
steps are undefined (or skipped) becomes a **build failure**, while `@status:pending`
scenarios are allowed to skip. So an `enforced` row can only be green if its scenario has
real, passing steps. **The rule — no requirement is "done" until its scenario is green and
CI-gated — is live for every scenario the moment it turns `enforced`.**

## Phased rollout

Scenarios land `pending` and turn `enforced` as the phase in their `@phase:` tag
implements the requirement. The teeth (`fail_on_skipped_with`) are already wired centrally,
so promoting a scenario is three steps: (1) add its step definitions in
`crates/hologram-conformance/tests/bdd.rs`, (2) flip its `@status` tag to `enforced`, and
(3) flip the matching `CONFORMANCE.md` row to `✅` (the meta-gate rejects any mismatch, so
these move together). **GV-1** (R1 traceability, witnessed against
`hologram-realizations::ContainerManifest::references()`) is the first enforced scenario
and the worked example of this path.
