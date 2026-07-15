# P2 — Import holospaces → `spaces/` (concrete plan)

Empirical map built 2026-07-15 from `../holospaces` (branch `chore/hologram-head-sync`,
pinned to hologram `22b0ce1` — **pre-refactor**, so it still consumes the *old* dissolved
crate names). P2 is therefore an **import + port**: bring holospaces into this repo under
`spaces/`, then repoint it from the git-pinned old crates onto the consolidated in-tree
contract.

## What comes over

| holospaces crate | LOC | Destination | Notes |
|---|---|---|---|
| `holospaces` | 31.6k | `spaces/holospaces` | the space impls (emulators, κ-disk, boot, peer/session) |
| `holospaces-node` | 691 | `spaces/holospaces` (bin) or `spaces/holospaces-node` | node binary |
| `holospaces-web` | 3.3k | `spaces/holospaces-browser` (or keep excluded) | wasm32 + web-sys; workspace-excluded today |
| `holospaces-emulator` | 167 | folds into `holospaces` | workspace-excluded today |

Peer/Session/Manager + `Client` **hoist to `hologram-runtime`** is **P3**, not P2 (D26).
P2 lands holospaces buildable; P3 does the contract hoist + first lockstep release.

## Old → new crate mapping (the port)

Source-import churn is small and mechanical — total ~76 references:

| Old (git-pinned, dissolved) | refs | New (in-tree) | How |
|---|---|---|---|
| `hologram_substrate_core` | 53 | `hologram_space` | absorbed (P1 move 3) |
| `hologram_realizations` | 6 | `hologram_space` | realizations module (move 3) |
| `hologram_store_mem` | 6 | `hologram_tck` | `mem` module (move 7) |
| `hologram_bare_hal` | 5 | `hologram_space` | `hal` module (move 1) |
| `hologram_net_http` | 3 | `hologram_net` | `http` module, feature `live` (move 5) |
| `hologram_net_bare` | 2 | `hologram_net` | `bare` module (move 5) |
| `hologram_runtime_bare` | 1 | `hologram_runtime` | feature `engine-wasmi` (move 6) |
| `hologram_substrate_tck` | (Cargo) | `hologram_tck` | (move 7) |
| `hologram_store_native` | (Cargo) | `hologram_store_native` | unchanged (P2-bound crate) |
| `hologram_runtime_wasmtime` | (Cargo) | `hologram_runtime` | feature `engine-wasmtime` (move 6) |
| `hologram-{exec,compiler,backend,graph,archive}` | — | unchanged | compute crates survive |

## Steps (each keeps the enforced BDD scenarios green)

1. **Import with history** — `git subtree add --prefix=spaces/holospaces ../holospaces
   chore/hologram-head-sync` (or a `git read-tree` merge). Preserves holospaces' commit
   history under `spaces/`. One-time, deliberate — brings another repo's history in.
2. **Repoint Cargo** — replace holospaces' `[workspace.dependencies]` git pins (all on
   `Hologram-Technologies/hologram` rev `22b0ce1`) with **path deps** on the consolidated
   crates + the right feature flags (`hologram-net` `live`/`tcp`, `hologram-runtime`
   `engine-wasmtime`/`engine-wasmi`). Fold holospaces' members into the root workspace (or
   keep a nested workspace — decide at import; root membership gives one lockfile + one RZ gate).
3. **Rewrite imports** — the ~76 `hologram_substrate_core::` → `hologram_space::` etc.
   (anchored seds, per the P1 move playbook).
4. **Green holospaces V&V** — its `justfile`/CI (CC catalog, QEMU oracles, Playwright) run
   against the new crates. Absorb its conformance suites into the honesty catalog where they
   overlap (MG-7 extraction proof).
5. **Docs** — relocate holospaces docs to `specs/holospaces/`.

## Risks / unknowns to confirm at execution

- **API drift** since `22b0ce1`: the consolidated crates changed shape (e.g. `hologram-net`
  is now feature-gated modules, `hologram-runtime` is engine-gated). Some call sites may need
  more than a rename — surface these by building `spaces/holospaces` and reading the errors.
- **`store-native`/`store-bare` → `spaces/`**: these two still live in `substrate/`. Once
  holospaces is in, decide whether they move under `spaces/holospaces-{native,bare}` or stay
  shared. (They're the P2 store move noted in SPRINT.)
- **Nested vs. flat workspace**: holospaces has its own `[workspace]`; folding into the root
  needs its members listed and its excludes (`holospaces-web`, `-emulator`) preserved.
- **wasm/browser**: `holospaces-web` (web-sys) stays target-excluded like `store-opfs`.

## Exit

`spaces/holospaces` builds against the in-tree consolidated crates; holospaces V&V green;
enforced BDD scenarios still green; RZ gate holds. Then P3 (hoist + first lockstep release).
