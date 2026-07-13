# 06 — Migration Plan: Phased, Always-Green

Decision: D17 (see `00-overview.md`). This document sequences the refactor; it does not
authorize starting it.

## Ground rules

1. **Always green.** Every phase lands with the full workspace CI passing AND the
   holospaces V&V suite passing (CC conformance catalog, QEMU differential oracles for
   RISC-V/AArch64/x86-64, Playwright browser tests, substrate TCK). "Keep holospaces
   functionality intact" is a per-phase gate, not an end-state hope.
2. **Moves and behavior changes never share a commit.** A phase that relocates code
   relocates it verbatim (plus mechanical path/name fixes); behavior changes get their
   own commits inside the phase.
3. **No long-lived divergent branch.** Phases land to `main` sequentially; each is
   independently shippable.
4. Each phase ends with its exit criteria checked and recorded (evidence, not assertion).
5. **κ-stability is sacred.** Crate moves/renames MUST NOT change any canonical byte
   form, realization IRI, wire frame, or `.holo` encoding — deployed stores (browser
   OPFS packs, redb stores, published archives) must keep resolving. Before P1, capture
   **golden vectors** (canonical bytes + κ for every realization kind, a v2 `.holo`, a
   SPINE-4 frame); every phase re-derives them bit-identically. A κ break is a format
   change and belongs to P4+ with explicit versioning, never to a refactor move.

## Phase P1 — In-repo restructure (this repo only)

- Rename `crates/hologram-backend` → `crates/hologram-compute` (D3).
- Merge `crates/hologram-host` into `crates/hologram-types` (D15).
- Create `crates/hologram-space` from `substrate/hologram-substrate-core` +
  `substrate/hologram-realizations` + `substrate/hologram-bare-hal`.
- Create `crates/hologram-tck` from `substrate/hologram-substrate-tck` +
  `substrate/hologram-store-mem` (as reference store).
- Move `substrate/hologram-runtime{,-wasmtime,-bare}` → `crates/hologram-runtime` with
  `engine-wasmtime` / `engine-wasmi` features.
- Create `crates/hologram-net` from the protocol/DHT cores of
  `substrate/hologram-net-{http,tcp,bare}`; park transport-specific code in a temporary
  `crates/hologram-net/transports/` module pending P2 (spaces don't exist yet).
- Merge `substrate/hologram-substrate-cli` into `crates/hologram-cli` as subcommands
  (store/net/node), resolving the duplicate binary name.
- Delete `substrate/`; park `substrate/hologram-efi` under a build-excluded location
  until P2 moves it to `spaces/holospaces-bare`.
- Facade features updated to the 01 matrix (minus `space-*` impls).

**Exit criteria**: `substrate/` gone; workspace builds all targets it built before
(including wasm32 + thumbv7em checks); all pre-existing tests + TCK pass; exactly one
binary named `hologram`; no crate named `hologram-backend` remains.

## Phase P2 — Import holospaces into `spaces/`

- Import `../holospaces` history (subtree or filtered merge — preserve history) into:
  `spaces/holospaces` (portable core), `spaces/holospaces-browser` (was holospaces-web),
  emulator codemodule build under `spaces/holospaces`.
- Create `spaces/holospaces-native` and `spaces/holospaces-bare` from the relocated
  substrate stores/transports (redb, bare) + engine selections; move net transports out
  of the P1 parking module into their spaces.
- Dedupe the two OPFS stores (substrate's `hologram-store-opfs` vs holospaces-web's
  `OpfsKappaStore`) into one in `holospaces-browser`.
- Replace all git-pinned `hologram-*` deps (rev `18f553d…`) with workspace path deps.
- Absorb holospaces CI: vv/ suites, CC catalog, QEMU oracles, Playwright — as distinct CI
  jobs so core-crate PRs and space PRs gate appropriately.
- Relocate holospaces docs (arc42/C4/OPM/15288) under `specs/holospaces/`; ADR numbering
  continues unbroken.
- Archive the `../holospaces` repo (read-only pointer to this repo).

**Exit criteria**: full V&V green in this repo's CI; every space crate passes the TCK on
its target; browser Pages deployment builds from this repo; no git deps between former
repos remain.

## Phase P3 — Hoist, unify, publish

- Hoist `Peer`, `Session`, `Manager`/`Operator`/`Roster`/`Configuration` from
  `spaces/holospaces` into `crates/hologram-runtime` (D7); spaces keep views only.
- Introduce `hologram::Client` (05); rebuild `hologram-ffi` over it (uniffi +
  wasm-bindgen + cbindgen); add the cross-language binding smoke suite.
- Complete the CLI subcommand tree over Client.
- Facade gains `space-browser`/`space-native`/`space-bare` features.
- Workspace lint hardening to the law-7 baseline (`missing_docs = "deny"`, etc.).
- **Cut the first lockstep crates.io release** (D16). Precondition (verify during P1,
  not here): all workspace crate names available/owned on crates.io, publish tokens and
  org ownership settled (see 01 §Publishing).
- Migrate `hologram-ai` (in its own repo) from git tags to the published facade —
  the proof the public API is sufficient. Fold what it needs but can't get through the
  facade back as facade fixes, not as new git pins. Named explicitly: its κ-addressing
  source moves from `holospaces::address` to the facade (`hologram::types` /
  `hologram::space`), or a holospaces git dep survives and the exit criterion fails.

**Exit criteria**: release published; hologram-ai CI green against it with zero git deps
on this repo; FFI smoke suite green for c/python/swift/ts; V&V still green.

## Phase P4 — `.holo` v3 (03-holo-format.md)

- AppManifest realization; layer model; entrypoints/exit codes; child refs +
  capability-attenuated `spawn_child` composition; portable-surface view slot + native
  override; v2 read-compat shim; `hologram app` CLI subcommands.
- TCK: app boot/compose/exit-propagation battery.

**Exit criteria**: a multi-layer demo app (wasm layer + tensor layer + rootfs layer +
portable view) boots on browser, native, and bare RAM-device spaces from the same `.holo`
bytes (same κ); v2 archives still load; release cut.

## Phase P5 — Networks, capability phase (04 §Phase A)

- Network realization (membership + policy + nesting); capability-gated
  fetch/announce/discover; `hologram network` subcommands; iroh transport pump for
  `holospaces-native`; TCK network battery.

**Exit criteria**: two-node private network demo (browser peer + native peer) with a
non-member refused at protocol layer; κ-only identity audit (no iroh types in any public
API or stored form); release cut.

## Phase P6 — Network payload encryption (04 §Phase B)

- Design doc first (per 04's four fixed requirements), then implementation.

**Exit criteria**: private-network confidentiality demo; dedup semantics documented and
tested; no_std participation proven on the bare space.

## Explicitly deferred beyond P6

Governance/attestation full design (07), design-system SDK projects, additional spaces
(ios, esp32) — enabled by, not part of, this refactor.
