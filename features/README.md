# Conformance BDD suites

Gherkin `.feature` suites for the hologram **refactor** (`specs/refactor/00`–`07`),
run by the `cucumber` crate in `crates/hologram-conformance`. Modeled on
`afflom/UOR-Atlas-UTQC`'s `features/suites`, informed by and cross-linked to the
root `CONFORMANCE.md` normative ledger.

## Layout

- `suites/s0_laws` … `suites/s6_governance` — one suite per refactor spec area.
- Each scenario is tagged `@class:<C> @id:<C-N> @spec:<doc> @phase:<Pn> @status:<s>`.
- `@class`/`@id` bind the scenario to a `CONFORMANCE.md` row (classes LAW/SP/HF/NW/TL/MG/GV).

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

What the static gate does **not** yet assert is that an `enforced` scenario actually
*passes* — scenarios are `pending` (skipped) until a phase implements their steps. The
run-result teeth arrive per phase: when a suite's phase lands, its steps are implemented,
its `@status` flips to `enforced`, its row flips to ✅, and `.fail_on_skipped()` is enabled
for that suite so an enforced-but-unimplemented scenario fails the build. **The rule — no
requirement is "done" until its scenario is green and CI-gated — is enforced from that
point on for each scenario as it turns `enforced`.**

## Phased rollout

Scenarios land `pending` and turn `enforced` as the phase in their `@phase:` tag
implements the requirement. When a phase implements a suite, add step definitions in
`crates/hologram-conformance/tests/bdd.rs`, flip the `@status` tag to `enforced`, update
the matching `CONFORMANCE.md` row to `✅`, and enable `.fail_on_skipped()` for that
suite's tag so an enforced-but-unimplemented scenario fails the build.
