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

## Phase P0 — Close the pin gap (in the holospaces repo; D23)

The always-green rule is vacuous while holospaces pins hologram at rev `18f553d…` —
main is ~104 commits ahead **including breaking changes** (`feat(exec)!` kappa-leases,
`feat(backend)!` fused decode attention). P0 makes the gate real before any move:

- In `../holospaces`, still git-pinned: port to current hologram HEAD; absorb the
  breaking API changes as ordinary dependency-bump commits (this is the port P2 was
  silently inheriting — done here, it's normal maintenance; done in P2, it poisons the
  tree-move).
- Full holospaces V&V green against HEAD.
- Cut the **bridge tag** hologram-ai will pin (it currently tracks `branch = "main"`,
  which freezes dead when P2 archives the repo).
- Obtain **written relicense consent** (D24) from holospaces' second contributor for
  MIT → MIT OR Apache-2.0.

- **Bundle the review (D29)**: obtain the relicense consent *and* a spec review of the
  holospaces-restructuring parts (02 §hoist, 01 mapping) from the contributor at the same
  time — objections surface before the move, not after.

**Exit criteria**: holospaces V&V green against a hologram HEAD pin; bridge tag
published and hologram-ai switched onto it; relicense consent recorded; restructuring
review acknowledged. From here, any hologram change that breaks holospaces is visible
immediately — ground rule 1 has teeth.

## Phase P0.5 — De-risk spike (throwaway; D28)

Before committing to the full move, prove the one structural bet the whole plan rests on:
that the **async contract world and the sync compute hot path compose under one
`Space` + `Client`**. On a throwaway branch, build the thinnest vertical slice —
`Client::builder().space(…).compile(src).open(κ).boot()` — and run it on **both** a
native and a wasm TCK target, resolving the Send-bound question (02 §trait shape)
empirically rather than by assertion.

**Exit criteria**: the slice compiles and runs on native + wasm; the Send-bound policy is
decided from evidence and written back into 02; the spike branch is **discarded** (it
informs P1's real work, it does not ship). If the slice *can't* be made to work, the
crate map (01) is revisited before any production move — this is the cheap place to learn
that, not P1.

## Phase P1 — In-repo restructure (this repo only)

- **First commit, before any move lands (preflight)**: capture the golden vectors
  (ground rule 5) **and the perf baselines** (D27 — roofline/kernel numbers from
  `hologram-bench`, the release gate's reference); run the crates.io name/ownership
  preflight (01 §Publishing); declare MSRV + edition in `workspace.package`; add
  LICENSE-MIT/LICENSE-APACHE + license fields (D24); audit per-crate
  `description`/`repository` metadata; pick and wire the release tool (01 §Publishing);
  enable cargo audit/deny in CI; **decide CI tiering** (fast core gate per-PR, heavy V&V
  on the merge queue — so "always green" survives QEMU + Playwright + binding builds);
  record the CI baseline being defended. The Send-bound question is answered by the P0.5
  spike, not re-run here.
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
- Delete `substrate/`; park target-specific and space-destined crates that have no home
  until P2 under a build-excluded `parked/` directory, moved verbatim: `hologram-efi`,
  `hologram-store-native`, `hologram-store-bare`, `hologram-store-opfs` (opfs keeps its
  workspace-excluded wasm32 build + Playwright verification running from the parked
  path). Parking is explicitly temporary — P2's exit criteria include `parked/` being
  empty and deleted.
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
- Dedupe the two OPFS stores. **DONE (2026-07-15)**, but the *other* direction from this
  sketch: the sync `OpfsKappaStore` `KappaStore` backend moved **into `crates/hologram-store-opfs`**
  (a backend belongs in a crate, not in the `holospaces-browser` space), joining the async
  file-per-κ + GC JS layer behind a default `js-api` feature; the space consumes it with
  `default-features = false`.
- Replace all git-pinned `hologram-*` deps (the P0 HEAD pin) with workspace path deps —
  after P0 this is a pure dependency-source swap, no API port.
- Absorb holospaces CI: vv/ suites, CC catalog, QEMU oracles, Playwright — as distinct CI
  jobs so core-crate PRs and space PRs gate appropriately (witnessed by `MG-7`). **CC absorption
  DONE + `MG-7` ENFORCED (2026-07-16).** The CC catalog is a non-BDD `CC` class in `CONFORMANCE.md`
  (**44/45 ✅**; CC-45 dogfood 🟡), each row witnessed by `spaces/holospaces/tests/cc*.rs`, bound by
  the artifact-free CC bijection audit (`cc_gate`). **vv/ artifact carriage: external, never
  committed** — import only the ~250K vv/ framework; the 170M `vv/artifacts/` is materialized by
  `scripts/vv-fetch.sh` from the pinned holospaces subtree (a `.gitignore` guard blocks it). Gated
  by the blocking `holospaces-vv-heavy` CI job (QEMU · e2fsprogs · Playwright). Three post-port
  browser-workbench suites (SCM/search/tasks) are quarantined non-gating, tracked.
- Relocate holospaces docs (arc42/C4/OPM/15288) under `specs/holospaces/`; ADR numbering
  continues unbroken. **DONE (2026-07-16, Phase G1)**: 104 tracked source files imported via
  `git archive` (the 1.6G tool downloads + the arc42-generator submodule content are gitignored /
  pointer-only). CS-* absorbed as a non-BDD `CS` class (CS-1..CS-6, validators V1–V8); `MG-8`
  shaped pending; the `docs-conformance` CI job (JDK 21 · Ruby 3 · Structurizr · cmark-gfm ·
  pandoc) is authored, non-blocking until observed green.
- Archive the `../holospaces` repo (read-only pointer to this repo).

**Exit criteria**: full V&V green in this repo's CI; every space crate passes the TCK on
its target; browser Pages deployment builds from this repo; no git deps between former
repos remain; `parked/` is empty and deleted.

## Phase P3 — Hoist, unify, publish

- Hoist `Peer`, `Session`, `Manager`/`Operator`/`Roster`/`Configuration` from
  `spaces/holospaces` into `crates/hologram-runtime` (D7); spaces keep views only.
- Generalize holospaces `projection.rs` (Workspace/Intent) into the contract's
  **surface capability** (02 §5) + TCK surface battery — P4's view layers depend on
  this existing first.
- **API shaping fixes the tracked law-2 violation** (02 §Sync): `KappaSync`'s
  `add_peer(&str)`/`add_gateway(&str)` become κ-shaped (PeerEndpoint κ + ephemeral
  transport hints) before anything publishes. Includes the Client method-table pass and
  freezing the `Space`/`Surface` trait signatures per 02's sketches.
- hologram-ai migration aid: publish a **mapping table** (v0.8.0 crate/feature →
  facade feature/module) with the release notes — its port crosses the P0-absorbed
  breaking changes plus the renames, not just a dependency-source swap.
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
on this repo; FFI smoke suite green for c/python/swift/ts; the Client method-table
naming review (05 §Principle) completed and the spec table updated to match what ships;
**extraction proof passes** — one in-tree space (e.g. `holospaces-native`) compiles and
passes the TCK against the published release using **version deps only** (the D21 smoke
test, kept green from here on); V&V still green.

## Phase P4 — `.holo` v3 (03-holo-format.md)

- AppManifest realization; layer model; entrypoints/exit codes; child refs +
  capability-attenuated `spawn_child` composition; portable-surface view slot + native
  override; v2 read-compat shim; `hologram app` CLI subcommands.
- TCK: app boot/compose/exit-propagation battery.

**Progress (2026-07-16/17):**
- **P4.1 done** — `AppManifest` realization in `hologram-space` (SPINE-2/3): closed `LayerKind`
  enum (wasm-codemodule / tensor-plan / rootfs-image / view; exit-semantics derived from kind),
  `Layer` descriptor, `primary: Option<u32>` so the degenerate tensor-only archive is valid,
  `validate()` (primary is exit-bearing; rootfs has arch; portable kinds don't) and `decode()`
  inverse; registered in `REGISTRY`. Native + wasm32 + thumbv7em green.
- **P4.2 done** — `.holo` v3 in `hologram-archive`: `FORMAT_VERSION` 2→3, `SectionKind::AppManifest`
  (discriminant 15, kinds 0–14 unchanged for κ-stability), writer `set_app_manifest` (opaque bytes),
  loader `app_manifest()` accessor, and the v2 read-shim (`MIN_READ_VERSION..=FORMAT_VERSION`).
  The inference pipeline (exec/ffi/runtime) round-trips v3 unchanged.
  - *Enforcement layering* (honesty note): the tensor-container reader (`LoadedPlan::into_plan`)
    does **not** require a manifest — it is the tensor-execution path, and a bare tensor archive is
    valid to it. The "a v3 archive without a manifest is invalid" invariant (03 §Encoding) is the
    **app loader's** to enforce; that loader lands in **P4.3**, and the compiler defaulting every
    archive to a single-tensor-plan manifest lands in **P4.4**. Until then the format *capability*
    exists without forcing every producer through it — no dishonest "v3 requires manifest" claim is
    made before the pieces that make it true are in place.

**Exit criteria**: a multi-layer demo app (wasm layer + tensor layer + rootfs layer +
portable view) boots on browser, native, and bare RAM-device spaces from the same `.holo`
bytes (same κ); v2 archives still load; release cut.

## Phase P5 — Networks, restricted tier (04 §Phase A)

- Network realization (membership + policy; `parent-network κ` field reserved, flat
  networks only); capability-gated fetch/announce/discover (**"restricted"** tier —
  "private" is reserved for P6 encryption, per 04's terminology ladder); `hologram
  network` subcommands; native transports for interop: iroh pump **plus WebRTC endpoint
  + WebSocket listener** so browser peers reach native peers directly; wire-version
  negotiation + bounded `resolve_closure` + codec fuzz targets (04 §Protocol
  hardening); the concrete bootstrap/signaling story (04 §hardening — owned, even if
  out-of-band); TCK network battery.

**Exit criteria**: two-node restricted network demo (browser peer + native peer,
over a shared transport per 04's interop rule) with a non-member refused at protocol
layer; κ-only identity audit (no iroh types in any public API or stored form); release
cut.

## Phase P6 — Network payload encryption (04 §Phase B)

- Design doc first (per 04's four fixed requirements), then implementation.

**Exit criteria**: private-network confidentiality demo; dedup semantics documented and
tested; no_std participation proven on the bare space.

## The refactor ends at P3 (D26)

**P0–P3 is the refactor**: the ecosystem consolidated, `substrate/` gone, one binary,
one facade, first lockstep release published, hologram-ai migrated onto it. At that point
the reorganization is *done and shipped* — it must be allowed to stabilize (real use,
bug-fix releases) before anything else begins.

**P4–P6 are a distinct follow-on effort**, not a continuation: `.holo` v3, networks, and
encryption are net-new feature development that happens to build on the reorganized tree.
They carry the opposite risk profile from a move ("build something new" vs "prove nothing
changed"), so they get their own go/no-go decision — and may warrant their own planning
session — after P3 has proven itself. Nothing in P0–P3 may foreclose them (the
format/network fields and seams are already reserved), but starting them is a separate
choice, not the default momentum of finishing P3.

## Explicitly deferred beyond P6

Governance/attestation full design (07), durability & replication policy design (04
§Open item — drafted alongside P6, implemented after), design-system SDK projects,
additional spaces (ios, esp32) — enabled by, not part of, this refactor.
