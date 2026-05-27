# Hologram Container Substrate — Architecture

> **Status:** defining document (implement *to* this). Decision record: [ADR-057](../adrs/057-hologram-container-substrate.md).
>
> **Scope:** the **deployment substrate** — Container Runtime, Storage Layer, Network Layer —
> that hosts isolated execution units (containers) over a κ-label graph across three substrates
> (browser, WASI/native, bare-metal). Normative source: *Hologram Container Specifications* and
> *Hologram Bare-Metal Substrate Specification*. This document reconciles those specs with the
> existing `hologram` workspace and fixes the architecture the implementation follows.

---

## 0. Two substrates, one foundation

The existing `hologram` workspace (v0.5.0) is the **compute substrate**: a UOR-native tensor
runtime (`hologram-types/-ops/-graph/-compiler/-exec/-backend/-archive/-host/-ffi`) that compiles a
graph to a `.holo` archive and executes it through a content-addressed buffer pool. It stays as-is.

This document defines a **second, orthogonal layer — the deployment substrate** — that hosts
containers, persists their state, and routes κ-labels between peers. The two layers are siblings
over the same foundation, not a stack:

```
            Applications (containers — incl. hologram-ai, and other platforms)
            Projections (UI surfaces; downstream, separate spec)
   ┌─────────────────────────── HOLOGRAM DEPLOYMENT SUBSTRATE (this doc) ──────────────────────────┐
   │   Container Runtime  │  Storage Layer (KappaStore)  │  Network Layer (KappaSync)               │
   │   ┌───────────────── browser ── WASI/native ── bare-metal backends ──────────────────────────┐ │
   └───┴──────────────────────────────────────────────────────────────────────────────────────────┘
   ┌─────────── HOLOGRAM COMPUTE SUBSTRATE (existing, optional dependency of *containers*) ──────────┐
   │   hologram-exec / -compiler / -backend / -ops / -graph / -archive  (Prism tensor runtime)       │
   └────────────────────────────────────────────────────────────────────────────────────────────────┘
   ┌────────────────────────────── UOR-ADDR + UOR FOUNDATION (κ-derivation, σ-pipeline) ─────────────┐
   └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

A container that only stores/fetches/routes κ-labels needs no tensor compute. A container that
computes (LLM inference, hashing, canonicalization) imports the compute substrate's `PrismModel`s.
And the deployment substrate itself **builds on hologram's optimal, already-V&V'd primitives where
it makes sense** — it does not firewall itself from them (§0.1).

### 0.1 Relationship to the hologram compute substrate (reuse, not firewall)

The container specs say *"Hologram does not link Prism."* We **reject that as written.** Two reasons:

1. It is already false: `uor-addr` (which mints and verifies every κ-label) depends transitively on
   `uor-prism` — the σ-axis hash *is* `prism::crypto`. Pretending otherwise would be dishonest.
2. **hologram is optimal in many ways** (zero-movement content-addressed pool, idempotent dedup,
   warm-start, cache-oblivious kernels, the BLAKE3 σ-axis validated byte-for-byte against the
   reference). The deployment substrate and the containers it hosts **should use hologram where it
   makes sense** rather than reimplement weaker versions.

The governing principle is therefore **reuse over reimplementation, subject to the performance
contract**:

- **Reuse hologram's validated primitives — but only the RZ-clean ones.** The σ-axis
  (`hologram-host::HologramHasher` = `prism::crypto::Blake3Hasher`, no_std, pulls no hologram compute
  crate) and witnessed composition (`uor-addr`'s `compose_*_blake3` + TC-05) are reused directly.
  **`hologram-archive` is NOT reused** — it depends on `hologram-backend` (the tensor kernel engine),
  so importing it would drag the compute engine into the store/route path (G-E1, RZ violation). The
  thin κ-label format helpers (`address_bytes`/`derive_label`) are **byte-identical** reimplementations
  over the *same* `HologramHasher` — the same σ-axis the compute substrate proved against the BLAKE3
  reference (AS), not a weaker parallel path.
- **Don't drag the heavy compute engine into the store/route path.** A pure storage/network node
  must not pull `hologram-exec`/`-backend`/`-archive` just to address bytes. Tensor compute is a
  *container* concern: containers that compute import `hologram-exec`; the runtime hosting them does
  not. The dependency is **selective** — `hologram-host` + `uor-addr` only.
- **Uphold hologram's performance contract.** Every substrate part is held to hologram's PV-class
  floors, re-expressed as the SP class (§4): zero-copy, idempotent-no-rewrite, bounded walks,
  no bottleneck. A substrate part that regresses the contract fails V&V.

> The load-bearing distinction is no longer "link Prism vs not" — it is **"reuse hologram's optimal
> κ-native primitives (yes, encouraged) vs. embed tensor compute in the host path (no — that's a
> container's job)."** The performance contract is the invariant that makes the reuse safe.

### 0.2 Code is κ-addressed: drivers, container bodies, and the engine are codemodule κ-labels

A load-bearing UOR consequence: **code is data, content-addressed like everything else.** uor-addr's
`codemodule` realization (CCMAS — Canonical Code-Module AST Serialization) addresses a code module's
AST to a κ-label (`uor_addr::codemodule::address_blake3` → `blake3:<64hex>`, hologram's native width).

So the substrate **does not hand-author drivers.** A device driver (NVMe, e1000, virtio-blk), a
container body, and the engine binary itself are all **codemodule κ-labels** — fetched, verified by
σ-axis re-derivation, and loaded through the *same* store → network → verify → instantiate path. The
`BlockDevice`/`NetworkInterface` HAL traits (§5) are the **interface** a loaded driver satisfies; the
driver **implementation** is a κ-addressed codemodule supplied at deployment, referenced by a
manifest exactly as a container's `code` operand is. "Writing a driver" is therefore not substrate
work — it is publishing a codemodule κ into the graph. *(Witnesses:
`hologram-realizations/tests/codemodule.rs` — a driver AST → deterministic blake3 κ, distinct per
code, recovered from a manifest's `references()`.)*

This is the same move as containers (§0): the runtime never special-cases driver code; it addresses
and loads it. The only thing the substrate ships is the *seam* (the HAL trait) and the loader.

**Importing drivers from authoritative sources.** Because a driver is a κ, the engine imports it
through the Network Layer (`get_with_fetch`) from any peer/gateway and **verifies on receipt** by
σ-axis re-derivation (SPINE-4). The content-addressed graph *is* the authority: a source cannot
forge a driver, because forged bytes do not re-derive to the requested κ. So the engine can pull
**arbitrary** drivers as needed, trustlessly — nvme, ahci, e1000, virtio, … — each fetched by its κ
and verified before it is cached or loaded. *(Witness: `hologram-runtime/tests/driver_import.rs` —
five arbitrary drivers imported from a source peer + verified; a forging source is refused.)*

---

## 1. The uor-native spine (non-negotiable invariants)

Every part of this substrate upholds these. They are the architecture; everything else is mechanism.
Each is restated as a conformance item in §7.

- **SPINE-1 — Canonical-bytes-or-nothing.** No object exists in the substrate except as canonical
  bytes addressed by a κ-label. There is no second naming surface — no UUID, no incrementing ID, no
  path, no hostname-as-identity. Container IDs, Capability Sets, snapshots, runtime state, error
  events, and channels are all κ-labels (spec §4.1, §4.5, §4.7, §7.2, §7.5, §4.4).
- **SPINE-2 — Realization-tagged.** Every canonical-form artifact carries its **realization IRI** in
  its bytes. Consumers verify the realization and refuse artifacts whose realization they do not
  implement (spec §10.9). Eight sibling realizations (§3.3).
- **SPINE-3 — Identity is composition; references are its inverse projection.** A composite
  artifact's κ-label *is* the **witnessed composition** of its operand κ-labels (uor-addr's
  categorical ops — `g2` product / `f4` quotient / `e6` filtration / `e7` augmentation / `e8`
  embedding, ADR-061; the repo's `derive_label_witnessed` / `compose_model`). Because the σ-axis is
  one-way (uor-addr exposes **no `decompose`** — verified), the canonical form **embeds its operand
  labels**, and each realization's `references(canonical_bytes)` is the *structural projection back
  to exactly those operands* — re-verifiable by re-deriving the parent from them. Reachability, GC,
  snapshot integrity, capability delegation, and the error log are all the one walk over this
  inverse projection (spec §5.3, §10.10). **Not** a byte-scan for label-shaped substrings.
- **SPINE-4 — Verify by re-derivation.** Bytes from any peer/gateway are accepted only after
  re-deriving the κ-label through the σ-axis and matching (spec §6.4, §10.3). Trustlessness is not a
  policy layer; it is the read path.
- **SPINE-5 — Append-only, eviction-tolerant.** The κ-label graph grows monotonically; there is no
  delete primitive at the container surface. Backends MAY evict *bytes* by reachability; they MAY
  NOT delete the *addressing relation*. A local `get → None` means "fetch," not "absent" (spec §5.2,
  §5.3, §10.5, §10.8).
- **SPINE-6 — No fallback, no arbitrary cap.** When a uor-native path exists, the substrate takes it
  and only it. No "fast path that bypasses addressing," no non-canonical serialization for
  "convenience," no hardcoded size/type/count ceiling that the κ-label graph does not itself imply.
  Limits are resource budgets (capability-scoped, §4.5), never structural shortcuts. **Arbitrary
  workloads are the requirement, not a goal** — a container is opaque Wasm + κ-addressed state.

---

## 2. Crate architecture

A new crate family under `substrate/`. **Dependency rule (reuse, bounded):** these build on
`uor-addr` (+ `uor-foundation`/`-sdk`), each other, and the hologram crates that provide an optimal,
externally-validated κ-native primitive — `hologram-host` (σ-axis) and `hologram-archive`
(addressing / composition / κ-store pattern). They **must not** pull the tensor compute engine
(`hologram-exec`/`-backend`/`-ops`/`-graph`/`-compiler`) into the pure store/route path; tensor
compute is a *container* dependency, not a host one. CI enforces the boundary two ways (§7, RZ): a
`cargo tree` check that the compute engine is absent from the store/route crates, **and** the SP
performance floors that hold the reused primitives to hologram's contract.

| Crate | Role | std? | Phase |
|---|---|---|---|
| `hologram-substrate-core` | Trait surfaces (`KappaStore`, `KappaSync`, `ContainerRuntime`), supporting types, `Capabilities`, `verify_kappa`, σ-axis registry, `Realization`+`references()` registry, `get_with_fetch`. | `#![no_std]`+`alloc`, executor-agnostic | 0 |
| `hologram-realizations` | The 8 canonical-form realizations + `references()` extractors + TC-05 witnesses. | `#![no_std]`+`alloc` | 0 |
| `hologram-store-mem` | `MemKappaStore` reference impl (also the conformance fixture). | `#![no_std]`+`alloc` | 0 |
| `hologram-store-native` | redb index + sharded blob store (spec §5.5). | std | 1 |
| `hologram-store-opfs` | OPFS/IndexedDB backend (spec §5.4). | wasm | 4 |
| `hologram-store-bare` | Block-device LBA backend (bare-metal §5). | `#![no_std]`+`alloc` | 0 (skeleton) |
| `hologram-net-http` | HTTP-CAS client + server (spec §6.3). | std | 2 |
| `hologram-net-libp2p` | rust-libp2p `KappaSync` (Kademlia + gossipsub). | std | 2 |
| `hologram-net-bare` | smoltcp + no_std libp2p fork (bare-metal §6). | `#![no_std]`+`alloc` | 0 (skeleton) |
| `hologram-runtime-wasmtime` | `ContainerRuntime` via Wasmtime (native). | std | 3 |
| `hologram-runtime-bare` | Wasmtime-no_std / interpreter (bare-metal §7). | `#![no_std]`+`alloc` | 0 (skeleton) |
| `hologram-bare-hal` | `BlockDevice`/`NetworkInterface` HAL traits (bare-metal §3.2.1). | `#![no_std]`+`alloc` | 0 |
| `hologram-substrate-cli` | `hologram serve/spawn/list/...` (spec §9.2). | std | 4 |

**no_std discipline** inherits the workspace pattern verified across the compute crates:
`#![cfg_attr(not(feature = "std"), no_std)]` + `extern crate alloc`, `default = ["std"]`, substrate
deps feature-gated. Every core/realization/HAL/bare crate compiles on `wasm32-unknown-unknown`,
`thumbv7em-none-eabi`, and (added) `x86_64-unknown-none`. The Justfile's `wasm`/`embedded` recipes
gain the new crates from day one so portability never silently regresses (§6).

**Async strategy (per approved decision).** Core declares the async traits via `async-trait`
(alloc-only, dyn-compatible, no_std-OK) and stays **executor-agnostic** — it names no runtime.
`KappaStore` is **sync** (bounded local work; matches the OPFS sync handle and the existing
`WarmStore`); `KappaSync` and `ContainerRuntime` are **async**. Native backends bring `tokio`;
bare-metal brings `embassy-executor`; browser brings `wasm-bindgen-futures` (spec §7.1).

---

## 3. Component surfaces (the contract)

These are the spec §8 surfaces verbatim in intent; `hologram-substrate-core` is the single source.

### 3.1 Addressing core

- `KappaLabel71` **is** `uor_addr::KappaLabel<71>` (BLAKE3, `blake3:<64 hex>`) — literally the same
  type the compute substrate uses for `ContentLabel`, not a parallel definition.
- `verify_kappa(bytes, kappa)` re-derives through the σ-axis and compares the digest, folding bytes
  through **`hologram_host::HologramHasher`** (= `prism::crypto::Blake3Hasher`, the path validated
  byte-for-byte against the BLAKE3 reference, root AS class) and formatting the 71-byte κ-label —
  byte-identical to `hologram-archive::address_bytes`, but without the `hologram-backend` dep (G-E1).
- Composition (container IDs, error-log ordered product) uses **`uor-addr`'s `compose_*_blake3`** +
  TC-05 witnesses directly, and a `derive_label`-equivalent fold over operand labels via the same
  `HologramHasher`.
- `SigmaAxis` registry — the axes uor-addr 0.2.0 *actually* ships: `blake3` (default, hologram
  ADR-052), `sha256`, `sha3-256`, `keccak256`, `sha512`. **`sha256d` is NOT in uor-addr** (it is
  prism-btc/Bitcoin-specific) and is excluded. `storage_put_axis`/`KappaStore::put` take an explicit
  axis; unknown axis ⇒ `UnknownAxis` (spec §5.1).
- **κ-label width is per-axis and lives in the type** (`uor_addr::KappaLabel<N>`: 71 blake3/sha256,
  73 sha3-256, 74 keccak256, 135 sha512). A single `KappaLabel<71>` therefore **cannot** be
  multi-axis. Resolution (confirmed, §9 G-B1): the substrate's **own** realization
  artifacts (manifests, capability sets, snapshots, runtime-state, error events, channels) are
  hologram-minted ⇒ **blake3, `KappaLabel<71>`**. *Stored content* keys (`KappaStore` keys) are
  axis-polymorphic — held in their on-the-wire `<axis>:<hex>` byte form / a `MAX_LABEL_BYTES`(=135)
  buffer — so foreign-axis content still stores and verifies.

### 3.2 Trait surfaces

```rust
pub trait KappaStore: Send + Sync {                       // sync (spec §8.1)
    fn put(&self, axis: &str, bytes: &[u8]) -> Result<KappaLabel71, StoreError>; // idempotent
    fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError>;     // Option ⇒ eviction-tolerant
    fn contains(&self, kappa: &KappaLabel71) -> bool;
    fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError>;                 // reachability root
    fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError>;
    fn iterate(&self) -> Box<dyn Iterator<Item = KappaLabel71> + '_>;
    fn pinned_roots(&self) -> Box<dyn Iterator<Item = KappaLabel71> + '_>;
    fn approximate_count(&self) -> usize;
    fn approximate_bytes(&self) -> u64;
}

#[async_trait] pub trait KappaSync: Send + Sync {          // async (spec §8.2)
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError>; // verifies on receipt
    async fn announce(&self, kappa: &KappaLabel71);
    async fn discover(&self, prefix: Option<&[u8]>, limit: usize) -> Box<dyn Iterator<Item = KappaLabel71>>;
    async fn add_peer(&self, multiaddr: &str) -> Result<(), SyncError>;
    async fn add_gateway(&self, url: &str) -> Result<(), SyncError>;
}

#[async_trait] pub trait ContainerRuntime: Send + Sync {   // async (spec §8.3)
    // caps is a Capability Set **κ-label** (SPINE-1), not a struct — the authority MUST live in the
    // graph to be auditable/revocable. `Capabilities` (§8.4) is only the decoded *view* of that
    // κ-label's capability-set realization. This corrects spec §8.3 (which passed the struct).
    async fn spawn(&self, container_id: &KappaLabel71, caps: &KappaLabel71) -> Result<ContainerHandle, RuntimeError>;
    async fn suspend(&self, h: ContainerHandle) -> Result<KappaLabel71, RuntimeError>; // → snapshot κ
    async fn resume(&self, snapshot: &KappaLabel71, caps: &KappaLabel71) -> Result<ContainerHandle, RuntimeError>;
    async fn terminate(&self, h: ContainerHandle) -> Result<(), RuntimeError>;
    fn list(&self) -> Vec<ContainerHandle>;
    fn info(&self, h: ContainerHandle) -> Option<ContainerInfo>;
}
```

> **Note (B5).** `KappaSync::fetch`/received bytes and the trust-boundary artifacts (manifests,
> snapshots, capability sets) use the **witnessed** derivation (TC-05, `AddressWitness::verify()`).
> The hot-path internal reuse key `derive_label` carries **no** witness — so "every κ-label is
> replayable" is false and is *not* claimed; only boundary-crossing artifacts are.

Supporting types (`Capabilities`, `ContainerHandle`, `ContainerInfo`, `ContainerState`, error enums,
`get_with_fetch`) per spec §8.0/§8.4. The container import/export surface (`hg_init/event/suspend/
resume/callback`; `storage_*`, `sync_*`, `clock`, `random`, `channels`, `spawn`, `diagnostics`) is
defined once and bound per substrate (WIT for WASI/native, direct imports for browser/bare-metal,
spec §4.4/§8.5).

### 3.3 Realizations (`hologram-realizations`)

Eight sibling realizations, each: a typed input shape → a canonicalize discipline whose canonical
form **embeds its operand κ-labels** → a κ-label that is the **witnessed composition** of those
operands (`derive_label_witnessed`/`compose_model`, *not* blake3-of-concatenation; A1/A2) → a
`references()` *structural projection* recovering exactly those operands (SPINE-3) → a TC-05 witness.
IRIs are normative (spec Appendix B): `container-manifest`, `capability-set`, `snapshot`, `runtime-state`,
`error-event`, `channel`, `http-cas-protocol`, `container-interface`. Plus the bare-metal sibling
formats (bare-metal §11): `bare-metal-storage-format`, `runtime-state-region`, `hardware-inventory`,
`crash-record`, `diagnostic-lba-record`, `intent-log-record`, `gc-mark-state`,
`hardware-abstraction-traits`, `boot-config`.

> **Decision:** realizations live here (a uor-addr *consumer*), not upstreamed into `uor-addr`, so
> the substrate is self-contained. Upstreaming is a later, non-blocking option.

### 3.4 Capability system

Capability Sets are κ-labels (SPINE-1) — passed to `spawn`/`resume` as κ-labels, not structs (B3).
Revocation is a Revoked Keyed ID on the set's κ-label; the runtime refuses subsequent imports (spec
§4.5, §10.12). All of this is graph operations over the SPINE-3 inverse projection — no separate ACL
store.

**Delegation containment is the foundation's `SubtypingLattice` relation, realized faithfully.**
The spec's *"E₈ filtration"* naming is wrong — the relation the foundation *defines* is the partial
order on `ConstrainedType`s where **more constraints = narrower = contained**
(`uor-foundation-0.5.2/src/user/type_.rs:293–309`):

- A Capability Set's authority maps to a constraint set; **delegation `C' ⊆ C` ⟺
  `constraints(C') ⊇ constraints(C)` ⟺ `grants(C') ⊆ grants(C)`** (every granted root/channel is a
  subset, every budget equal-or-tighter, every flag implied). This is the lattice's defining order.
- **Where it runs (G-A3, corrected):** uor-foundation 0.5.2 ships `TypeInclusion`/`SubtypingLattice`/
  `ConstrainedType` as **orphan-closure interfaces with no public constructor or containment checker**
  (only `Null*` stubs; you'd implement a `ConstrainedTypeResolver`). So the substrate realizes the
  lattice relation **directly** in `Capabilities::admits` — this is the UOR lattice *semantics*, not a
  non-UOR ACL fallback (SPINE-6). It is proven a genuine partial order (reflexive / antisymmetric on
  grant-equality / transitive) and rejects over-broad delegations (CR tests). When the foundation
  exposes a resolver, `admits` swaps to it without changing the relation.

The categorical ops (`e6` filtration / `e8` embedding) are *not* this relation and are not used for
containment.
*(Witnesses: `hologram-substrate-core::tests::cr_admits_is_{reflexive,transitive,antisymmetric_on_grants}`,
`cr_rejects_over_broad_delegations`.)*

---

## 4. Efficiency is the addressing, not a tax (PV parity)

Per the mandate that *containers and their parts uphold hologram efficiency and benchmarks*, the
substrate adopts the compute substrate's discipline ([BENCHMARKS.md](../../BENCHMARKS.md),
[CONFORMANCE.md](../../CONFORMANCE.md) PV class, `just perf`): content-addressing is the *mechanism*
of efficiency, and we measure it.

- **Zero-copy storage.** A value lives in one buffer; `get` returns a shared `Bytes`, not a copy.
  `put` is **idempotent** — identical (axis, bytes) ⇒ same κ-label, no second write (spec §10.2).
  This mirrors the compute substrate's single-buffer pool (CA class).
- **Dedup + memo by κ-label.** Equal content collapses to one κ-label network-wide; a `get`/`fetch`
  hit elides re-fetch and re-derivation. The hot cache (spec §5.4/§5.5) is the storage analog of the
  warm pool.
- **Warm-start parity.** A migrated/resumed container is *never cold*: its snapshot κ-label and the
  reachable cone are content-addressed, so residency checks elide redundant work (analogous to the
  WS class).
- **Bounded everything.** Reachability/GC walk is O(reachable · refs); recovery is O(log N); the
  intent log scan is O(1) (bare-metal §5.5). No unbounded scans on hot paths.

**New PV-style class `SP` (substrate performance)** is added to CONFORMANCE.md with criterion benches
under `just perf`: `put`/`get` throughput and zero-copy (no per-op alloc), idempotent-put no-write,
reachability-walk scaling, HTTP-CAS body-streaming with no intermediate buffer (spec §6.5). Floors
catch regressions; we do not chase micro-optimization beyond the floors (project V&V discipline).

---

## 5. Substrate-tripling (incl. bare-metal, first-class)

The conformance test (spec §3.4, §10.16): one `#![no_std]+alloc` core source compiles against all
three backends, and a container produces **byte-identical κ-labels for byte-identical input event
streams** on all three.

- **Above the backend boundary:** single source (core + realizations + HAL traits).
- **Below it:** per-substrate backends. Browser = OPFS + js-libp2p + Service-Worker CAS gateway;
  WASI/native = redb + rust-libp2p + axum; bare-metal = block-device LBA store + smoltcp + no_std
  libp2p/rustls/Wasmtime forks, booting from UEFI (`hologram.efi`), HAL traits `BlockDevice` /
  `NetworkInterface` (bare-metal §3.2.1). Bare-metal is built to the same trait surfaces from the
  start; the no_std fork strategy (libp2p/rustls/Wasmtime) is an explicit, tracked dependency.
- The single piece that is *not* substrate-portable is the `#[global_allocator]` binding at each
  binary's entry site (bare-metal §4.2).

---

## 6. Implementation phasing

Approved first slice: **all-three skeleton (Phase 0)** — every §8 trait + supporting types + the
`Realization`/`references()` registry + `verify_kappa` real, with **stub backends** for all three
substrates (incl. bare-metal HAL + `*-bare` crates) that compile on native + `wasm32-unknown-unknown`
+ `thumbv7em-none-eabi`. No real I/O behavior yet; `MemKappaStore` is the one working impl and the
conformance fixture.

1. **Phase 0 — skeleton** (this slice). Core, realizations (IRIs + canonical-form + `references()`),
   `MemKappaStore`, stub backends, HAL traits, Justfile/CI cross-target build. Locks the async/no_std
   model end-to-end.
2. **Phase 1 — Storage.** `hologram-store-native` (redb), reachability + GC, SP benches, conformance
   ST/§10.2/§10.5/§10.8.
3. **Phase 2 — Network.** HTTP-CAS first (client+server), then libp2p. NW/§10.3/§10.6.
4. **Phase 3 — Runtime.** Wasmtime backend, lifecycle, snapshot, capabilities/delegation/revocation.
   CR/§10.1/§10.4/§10.7/§10.11/§10.12.
5. **Phase 4 — CLI + browser + tripling proof** (§10.16).
6. **Phase 5 — bare-metal hardening** (UEFI boot, drivers, no_std forks) to the §10.13–§10.17 items.

---

## 7. Conformance classes (added to CONFORMANCE.md)

Each gets a class + IDs + normative statement + enforcement + witness + external grounding, in the
existing table form.

| Class | Covers | External authority |
|---|---|---|
| **SPINE** | §1 invariants SPINE-1..6 (canonical-only, realization-tagged, structural refs, verify-on-receipt, append-only, no-fallback) | the container specs as normative text |
| **ST** | KappaStore: idempotency (§10.2), append-only surface (§10.5), reachability eviction (§10.8) | spec §5, §8.1 |
| **NW** | KappaSync: verify-on-receipt (§10.3), HTTP-CAS wire-format byte-identity (§10.6) | spec §6.3/§6.4 |
| **CR** | ContainerRuntime: container-identity invariant (§10.1), capability enforcement (§10.4), delegation soundness (§10.7), subscription delivery (§10.11), revocation (§10.12) | spec §4 |
| **RZ** | realization-IRI tagging (§10.9), `references()` presence (§10.10), and the **bounded-reuse** rule — store/route crates reuse `hologram-host`/`-archive` but the tensor compute engine is absent from their `cargo tree` | spec §10.9/§10.10, Appendix B |
| **TR** | substrate-tripling byte-identity (§10.16) + no_std discipline (§10.14) + no-OS (§10.13) + crash safety (§10.15) + hardware probing (§10.17) | bare-metal §10 |
| **SP** | substrate performance floors (§4): zero-copy get, idempotent-put no-write, bounded reachability walk, streaming HTTP-CAS | criterion / `just perf` |

---

## 8. Decisions resolved (see ADR-057)

1. New layer = new crate family, **same workspace**, **reusing** hologram's optimal κ-native
   primitives (`hologram-host` σ-axis, `hologram-archive` addressing/composition/store pattern). The
   tensor compute engine (`hologram-exec`/`-backend`) is a *container* dependency, never pulled into
   the pure store/route host path. (Split to a sibling workspace later only if needed.)
2. The spec's "Hologram does not link Prism" is **rejected**. hologram is optimal and the substrate
   uses it where it makes sense; reuse over reimplementation, **bounded by the performance contract**
   (§0.1). Addressing stays uor-prism-grounded through `uor-addr` — the same identity layer the
   compute substrate proved.
3. Async per spec; core executor-agnostic.
4. Bare-metal first-class from Phase 0 (skeleton), hardened in Phase 5.
5. Realizations live in `hologram-realizations` (consumer), upstreaming deferred.
6. Efficiency held to PV-parity via a new SP class under `just perf`.

---

## 9. Grounding review — gaps, inconsistencies, assumptions

> **Do not assume the spec (or this doc) is correct.** Every item below was checked against the
> *actual* `uor-addr 0.2.0` / repo surfaces. Where the container spec diverges from hologram's proven
> UOR patterns, **hologram is the authority.** `[fixed]` = corrected inline above; `[decide]` =
> needs your call; `[track]` = carry into the relevant phase's red test.

**A. Non-UOR-native mechanisms (spec wrong; corrected toward hologram's patterns)**

| ID | Finding | Evidence | Status |
|---|---|---|---|
| G-A1 | `references()` is not a uor-addr primitive; the spec's byte-scan is foreign. Regrounded as the *inverse projection of a witnessed composition* (canonical form embeds operands). | no `references`/`decompose` in uor-addr (grep) | `[fixed]` SPINE-3, §3.3 |
| G-A2 | Identities specified as blake3(canonical concatenation); UOR-native is witnessed composition of operand labels (`derive_label_witnessed`/`compose_model`, which hologram already uses). | `hologram-archive` address.rs | `[fixed]` §3.3 |
| G-A3 | Spec's "E₈ filtration" is misnamed; the relation is the foundation's **`SubtypingLattice`** (more constraints = narrower = contained). **But** uor-foundation 0.5.2 ships it as an **orphan-closure interface with no public constructor/checker** (only `Null*` stubs; deeper read corrects the earlier "callable primitive" claim). So the substrate realizes the lattice relation faithfully in `Capabilities::admits` (grants(C')⊆grants(C)) — the UOR semantics, proven a partial order, **not** an ACL fallback. Swap to a `ConstrainedTypeResolver` when the foundation exposes one. | `uor-foundation-0.5.2/src/user/type_.rs:293–309, 1412 (NullTypeInclusion), 5355 (resolver)` | `[resolved]` §3.4 · CR tests pass |

**B. Type / API inconsistencies**

| ID | Finding | Status |
|---|---|---|
| G-B1 | `KappaLabel<71>` can't be multi-axis (width is per-axis: 71/73/74/135). **Confirmed:** substrate artifacts = blake3 `<71>` (ADR-052); stored-content keys axis-polymorphic (`<axis>:<hex>` / `MAX_LABEL_BYTES`=135 form). | `[resolved]` §3.1 |
| G-B2 | `sha256d` is not in uor-addr 0.2.0 — removed from the axis registry. | `[fixed]` §3.1 |
| G-B3 | `spawn(.., Capabilities)` struct violates SPINE-1 (authority not in graph). Changed to a Capability Set **κ-label**. | `[fixed]` §3.2 |
| G-B4 | `WarmStore::put(&mut self)` ≠ `KappaStore::put(&self)`; reuse needs interior mutability. | `[track]` Phase 0 |
| G-B5 | "Every κ-label is TC-05-replayable" is false — `derive_label` (hot path) is unwitnessed; only boundary artifacts are. | `[fixed]` §3.2 note |

**C. Spec-internal defects**

| ID | Finding | Status |
|---|---|---|
| G-C1 | UorTime is "since boot" (resets) yet bare-metal §4.5/§5.5 selects the runtime-state copy by "latest UorTime" — cannot order across reboots. Needs a reboot-monotonic generation counter. | `[track]` Phase 5; flag upstream |
| G-C2 | Browser `KappaStore` "sync" contradicts the kv-worker-behind-async-MessageChannel design. Must specify which side is sync and how the boundary is bridged (e.g. `SharedArrayBuffer`+`Atomics.wait`). | `[track]` Phase 4 |
| G-C3 | "Arbitrary workloads" is bounded on bare-metal (no native subprocess → Wasm-expressible only). Qualify the mandate. | `[track]` |
| G-C4 | Error-log ordered-product + republished runtime-state are unbounded chains with no compaction policy. | `[track]` Phase 1/3 |

**E. Dependency-graph findings (discovered during implementation)**

| ID | Finding | Evidence | Status |
|---|---|---|---|
| G-E1 | Reusing `hologram-archive` (planned in §0.1/§3.1) would pull `hologram-backend` (the tensor kernel engine) into the store/route path — an RZ violation. Reuse is narrowed to `hologram-host` (σ-axis) + `uor-addr` (composition); `address_bytes`/`derive_label` are byte-identical reimpls over the same `HologramHasher`. | `crates/hologram-archive/Cargo.toml` deps `hologram-backend` | `[fixed]` §0.1/§3.1 |

**D. Assumptions in this doc to validate, not trust**

| ID | Assumption | Status |
|---|---|---|
| G-D1 | async-trait `Send + Sync` bound vs embassy's typically `!Send` single-core futures (bare-metal) — may force a `?Send`/local variant. | `[track]` Phase 0 |
| G-D2 | `bytes::Bytes` needs atomics (ok on thumbv7em; not guaranteed on every bare-metal target). | `[track]` Phase 0 |
| G-D3 | redb is std-only — must never enter a no_std build. | `[track]` Phase 1 |
| G-D4 | Storage→realizations coupling: reachability needs a runtime IRI→extractor registry (static fn-pointer table on no_std) — make the dependency explicit. | `[track]` Phase 0/1 |

---

## 10. Worked examples (real-world, backed by passing tests)

> **Runnable** (`just examples`, in `substrate/hologram-runtime-wasmtime/examples/`): each is a
> narrated real-world use-case — `cas_artifact_cache` (content-addressed build/CI cache: dedup +
> reproducible-build κ + reachability GC), `event_bus` (IoT/message-bus pub-sub with durable
> offline catch-up), `least_privilege` (multi-tenant sandboxed plugins via capability containment,
> no escalation), `wasm_inference_container` (a real Wasm hologram-ai inference/transform container
> reading+writing the κ-graph through capability-gated host imports), and `live_migration`
> (checkpoint on one node → resume on another from the snapshot κ, session state intact).

These are grounded in the Phase-0 reference implementation
(`substrate/hologram-store-mem/tests/`); they run under `just vv-substrate`.

### 10.1 A hologram-ai LLM-inference container (lifecycle + trustless migration)

A hologram-ai inference container is opaque Wasm + κ-addressed state. Its whole lifecycle is
κ-label flow — no non-UOR identity anywhere:

1. **Provision (peer A).** The runtime model compiled to Wasm, the GGUF weights, and the params
   are leaves: `code = put(wasm)`, `weights = put(gguf)`, `params = put(json)`. The **Container ID
   is the manifest** — `ContainerManifest{code, weights, params}` whose canonical form *embeds*
   those three κ-labels; `container_id = put(manifest.canonicalize())` binds them (SPINE-1/3).
2. **Authority.** A `CapabilitySet` grants a model-data storage root + the `completions` channel,
   with bounded budgets. It is itself a κ-label (`caps_k`) — auditable and revocable, not a struct.
3. **Suspend → snapshot.** `Snapshot{container_id, prev:None, payload:<linear-mem+globals+cursor>}`;
   `snapshot_k = put(snapshot.canonicalize())`. The snapshot's `references()` resolve back to the
   Container ID (graph continuity).
4. **Pin + GC.** Pin `snapshot_k`/`container_id`/`caps_k`; `gc()` retains the entire reachable cone
   (manifest + its three operands) and reclaims only orphans — *no reachable κ is ever evicted*.
5. **Migrate to peer B (honest peer).** B doesn't have the snapshot locally; `get_with_fetch` pulls
   it, **re-derives the κ through the σ-axis (SPINE-4)**, and caches it. Resume proceeds from those
   verified bytes — byte-identical container on a different substrate (§5 tripling).
6. **Migrate via a malicious peer.** A peer that returns forged bytes is **rejected**: the forged
   content fails σ-axis re-derivation, nothing is cached. This is what makes the network trustless —
   not a policy layer, the read path itself.

*(Witness: `tests/worked_example.rs::llm_container_lifecycle_and_trustless_migration`.)*

### 10.2 A knowledge-graph container (append-only growth, dedup, reachability)

A knowledge-graph container publishes facts as κ-labels. Identical facts **dedup to one κ-label**
network-wide (idempotent `put`), so a re-asserted fact costs no storage and a re-fetch elides
(the efficiency *is* the addressing, §4). The graph grows monotonically (append-only, SPINE-5);
"forgetting" is local eviction of unreachable bytes, never deletion of the addressing relation —
a query for an evicted-but-reachable fact falls through to peers via `get_with_fetch`. Reachability
from the container's pinned roots (its `CapabilitySet`'s granted roots are reference edges) is the
GC boundary.

*(Witnesses: `tests/conformance.rs::st2_put_is_idempotent_no_duplicate_write`,
`st10_8_gc_retains_reachable_evicts_unreachable`,
`tests/worked_example.rs::capability_set_grants_are_reachability_roots`.)*

### 10.3 Why arbitrary workloads, on any substrate

The runtime never interprets a container's bytes — it stores, routes, and addresses κ-labels and
hosts Wasm. A container that does pure data work needs no compute substrate; one that computes
imports hologram's `PrismModel`s as a κ-addressed Wasm library. The *same* container code emits
*byte-identical* κ-labels on browser, native, and bare-metal (the σ-axis is the one BLAKE3 path,
validated against the reference), so a workload is portable by construction — bounded only by what
Wasm can express (on bare-metal, no native-subprocess escape hatch; G-C3).
