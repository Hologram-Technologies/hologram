# hologram-conformance

> BDD (cucumber) conformance runner and static honesty meta-gate for the hologram refactor.

This crate is the workspace's conformance harness. Its library parses `CONFORMANCE.md` and the `features/suites` tree; its test binaries drive the shipping surfaces against those authorities so that a passing build is a passing conformance run (substrate-tripling discipline). It holds no domain types of its own — the realizations enter only through the step definitions' dev-dependencies.

## What it provides

- `ConformanceWorld` — the per-scenario cucumber `World`, the sole async↔sync seam (Law L4); state is stored as raw κ-label bytes so the library stays domain-type-free.
- `catalog` / `feature` / `report` — the catalog parser (`CONFORMANCE.md`), the feature parser (`features/suites`), and the bijection gate that fails the build when catalog rows and scenarios drift apart.
- `cc` — the CC/CS catalog support for the bijection audits (MG-7 binds every `CC` row to a witness test in `spaces/holospaces/tests`; MG-8 binds every `CS` row to a V1–V8 validator script).
- `SUITES_DIR` / `CONFORMANCE_MD` / `CC_TESTS_DIR` / `CS_SCRIPTS_DIR` — compile-time absolute paths into the repo-root authorities.
- `tests/bdd.rs` — the harness-free cucumber runner, discovering every `.feature` under `features/suites` and driving the real `hologram::Client` over `SpikeSpace` (LAW-3 / SP-3).
- `tests/meta_gate.rs` — the static honesty meta-gate: it fails the build if the catalog and scenarios are not in bijection.
- `tests/common/mod.rs` — `SpikeSpace`, the reference `hologram_space::Space` implementation absorbed from the former `hologram-spike-sp3` crate. Built from only hologram's public API (no sealed traits, no in-tree privilege), it witnesses LAW-3 (the open `Client<S: Space>` contract) and SP-3 / LAW-4 (composition: `Client` driving `compile → provision → run` over an outside space).

Part of the [hologram](../../README.md) workspace.
