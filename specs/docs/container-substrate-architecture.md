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
  Limits are resource budgets (capability-scoped, §4.5), never structural shortcuts. **Workloads
  are arbitrary up to the host substrate's expressible-Wasm envelope:** on WASI/native and browser
  substrates, anything Wasm + the spec §4.4 import surface can express; on bare-metal there is no
  native-subprocess escape hatch (G-C3) — a container *is* Wasm + κ-addressed state, never a
  shell-out. The runtime refuses any container whose manifest declares an import outside the §4.4
  surface (`hologram.*` only); this is the structural enforcement of the workload bound.

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
| `hologram-realizations` | The 8 universal + 9 bare-metal sibling realizations (D1) + `ChainCompaction` barrier (B2) + `references()` extractors + TC-05 witnesses. | `#![no_std]`+`alloc` | 0 |
| `hologram-store-mem` | `MemKappaStore` reference impl (also the conformance fixture). | `#![no_std]`+`alloc` | 0 |
| `hologram-store-native` | redb index + **sharded blob store (spec §5.5)** + size-bounded LRU read-through cache (SP §4). | std | 1 |
| `hologram-store-opfs` | OPFS/IndexedDB backend (spec §5.4). | wasm | 4 |
| `hologram-store-bare` | **Merkle B-tree** of κ → extent over a raw `BlockDevice`; dual-buffered headers + CoW + persistent free-list (BT) + reboot-monotonic epoch (B1). | `#![no_std]`+`alloc` | 0 ✅ |
| `hologram-net-http` | HTTP-CAS client + server (spec §6.3). | std | 2 |
| `hologram-net-tcp` | **uor-native** TCP `KappaSync`: κ-XOR Kademlia DHT over raw TCP; peer identity = κ of `PeerEndpoint` realization (no PeerIds, no Multiaddrs — SPINE-1). Replaced the previous libp2p crate, whose PeerId / Multiaddr layer was a second non-κ naming surface. | std | 2 ✅ |
| `hologram-net-bare` | no_std [`KappaSync`] over the HAL `NetworkInterface` (bare-metal §6) — C2; frame-codec + verify-on-receipt. Same wire as `hologram-net-tcp` on std hosts; no libp2p layer. | `#![no_std]`+`alloc` | 0 ✅ |
| `hologram-runtime-wasmtime` | `ContainerRuntime` via Wasmtime (native). | std | 3 |
| `hologram-runtime-bare` | **wasmi** Wasm interpreter, no_std (bare-metal §7) — C1; implements the `ContainerEngine` seam symmetric to `runtime-wasmtime`. | `#![no_std]`+`alloc` | 0 ✅ |
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
- **Below it:** per-substrate backends. Browser = OPFS + uor-native TCP-over-WebTransport (when
  Wasmtime/`waitAsync` lands; currently OPFS-only via the bridge) + Service-Worker CAS gateway;
  WASI/native = redb + `hologram-net-tcp` (κ-XOR Kademlia, no libp2p) + axum; bare-metal =
  block-device LBA store + smoltcp + `hologram-net-bare` (HAL `NetworkInterface` frame codec, no
  libp2p) + Wasmi interpreter (`hologram-runtime-bare`), booting from UEFI (`hologram.efi`), HAL
  traits `BlockDevice` / `NetworkInterface` (bare-metal §3.2.1). Bare-metal is built to the same
  trait surfaces from the start.
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
3. **Phase 2 — Network.** HTTP-CAS first (client+server), then uor-native TCP (κ-XOR Kademlia, peer-identity-by-κ — no libp2p PeerId/Multiaddr second naming surface). NW/§10.3/§10.6.
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
| **DHT** | content discovery without a coordinator: κ-XOR Kademlia `PROVIDE`/`GET_PROVIDERS` over κ keys; peer identity = κ of `PeerEndpoint` (no PeerIds — SPINE-1); `announce(κ)` issues `Provide` to the K-closest peers; `fetch` walks K-closest then `GetProviders` (§11.1) | Kademlia paper (Maymounkov & Mazières) + substrate's own `hologram-net-tcp` wire spec |
| **FED** | hierarchical multi-source `KappaSync` over **hologram peers only** (local → uor-native TCP peer (§11.1) → HTTP-CAS peer), verify-on-receipt at every hop, `add_gateway` wires another hologram CAS-serving peer (§11.2) | each hop reuses NW class authority |
| **BT** | bare-metal store: **Merkle B-tree** of κ → extent (every page has its own κ; the store state is one root κ); CoW write-discipline; crash-atomic root flip (§11.3) | spec §5.2 + crash-safety §10.15 |
| **AR** | archival cold tier = **bare-metal hologram peer** participating in the federation chain (same `/cas/<κ>` + uor-native TCP transports as hot peers; durable across reboots via the §11.3 B-tree + §11.9 NIC driver-import); no external hosting (§11.4) | NW class authority + BM class (bare-metal substrate) |
| **OG** | OPFS reachability GC in real Chromium: mark from pins through `references()`, delete unreachable files (§11.5) | structural projection (SPINE-3) |
| **QC** | storage quota carries through suspend/resume: `storage_used` lives in the `Snapshot` payload's canonical bytes; the snapshot κ binds it (§11.6) | SPINE-1 (authority in graph) |
| **SC** | cross-container scheduling fairness: per-container weight (capability field) + deficit round-robin over UorTime ordering — **no wall-clock** (§11.7) | ADR-058 + DRR fair-queueing |
| **RV** | transitive revoke as a κ-graph walk: a `Delegation{parent_caps,child_caps}` realization is minted on `spawn_child`; revoking `caps_κ` walks its delegation cone via `references()` and revokes the entire subtree (§11.8) | SPINE-3 inverse projection |
| **NI** | `NetworkInterface` driver-import: codemodule κ-label → `WasmNetworkInterface` HAL binding → real packet I/O; mirrors `WasmBlockDevice` for HAL surface parity (§11.9) | uor-addr `codemodule` |

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
| G-B4 | `WarmStore::put(&mut self)` ≠ `KappaStore::put(&self)`; reuse needs interior mutability. | `[resolved]` every `KappaStore` impl uses interior mutability (`Mutex<Inner>`); no `&mut self` mismatch remains in the substrate. |
| G-B5 | "Every κ-label is TC-05-replayable" is false — `derive_label` (hot path) is unwitnessed; only boundary artifacts are. | `[fixed]` §3.2 note |

**C. Spec-internal defects**

| ID | Finding | Status |
|---|---|---|
| G-C1 | UorTime is "since boot" (resets) yet bare-metal §4.5/§5.5 selects the runtime-state copy by "latest UorTime" — cannot order across reboots. Needs a reboot-monotonic generation counter. | `[resolved]` B1 — `BareMetalKappaStore` header v4 persists `reboot_epoch`, bumped on every `open`. `RuntimeStateRegion` realization (D1) carries the pair `(reboot_epoch, generation)` — a total order over all writes across reboots. Witness: `store-bare/tests/reboot_epoch.rs`. |
| G-C2 | Browser `KappaStore` "sync" contradicts the kv-worker-behind-async-MessageChannel design. Must specify which side is sync and how the boundary is bridged (e.g. `SharedArrayBuffer`+`Atomics.wait`). | `[resolved]` B3 — `hologram-store-opfs::bridge` defines a `SharedArrayBuffer`-backed protocol. Main-thread `SyncOpfsBridge` writes a request, `Atomics::wait`s; paired Worker (`web/bridge-worker.mjs`) constructs a `BridgeWorker`, dispatches via the existing async `opfs_put`/`opfs_get`, writes the response, `Atomics::notify`s. Verify-on-receipt is preserved across the boundary. |
| G-C3 | "Arbitrary workloads" is bounded on bare-metal (no native subprocess → Wasm-expressible only). Qualify the mandate. | `[resolved]` SPINE-6 carries the qualification ("Wasm + §4.4 import surface only"); runtime refuses non-§4.4 imports at instantiate. |
| G-C4 | Error-log ordered-product + republished runtime-state are unbounded chains with no compaction policy. | `[resolved]` B2 — `Runtime::set_error_log_threshold` + `ChainCompaction` realization (zero operands; GC reclaims the old tail). Default depth = 128; `0` ⇒ unbounded (opt-in, SPINE-6). Witness: `runtime/tests/chain_compaction.rs`. |

**E. Dependency-graph findings (discovered during implementation)**

| ID | Finding | Evidence | Status |
|---|---|---|---|
| G-E1 | Reusing `hologram-archive` (planned in §0.1/§3.1) would pull `hologram-backend` (the tensor kernel engine) into the store/route path — an RZ violation. Reuse is narrowed to `hologram-host` (σ-axis) + `uor-addr` (composition); `address_bytes`/`derive_label` are byte-identical reimpls over the same `HologramHasher`. | `crates/hologram-archive/Cargo.toml` deps `hologram-backend` | `[fixed]` §0.1/§3.1 |

**D. Assumptions in this doc to validate, not trust**

| ID | Assumption | Status |
|---|---|---|
| G-D1 | async-trait `Send + Sync` bound vs embassy's typically `!Send` single-core futures (bare-metal) — may force a `?Send`/local variant. | `[resolved]` B4 — `LocalKappaSync` + `LocalContainerRuntime` are `#[async_trait(?Send)]` siblings of the multi-core traits; embassy executors implement these. Disjoint by design — std hosts don't silently degrade to `!Send`. Witness: `substrate-core::tests::local_kappa_sync_accepts_non_send_implementors`. |
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

---

## 11. Substrate completions — implementation depth

Phase 0 landed the contract; this section is the **implementation depth** of every surface that
Phase 0 left as a stub or a coordinator-bound design. Every entry here is uor-native — identity by
κ-label, relations by κ-graph composition, decisions by structural projection. No wall-clock, no
side-channel ACLs, no traditional ID maps.

### 11.1 Network discovery — κ-XOR Kademlia DHT, **uor-native**, no libp2p

`announce(κ)` and `discover(prefix, limit)` are implemented in `hologram-net-tcp` over a κ-XOR
Kademlia DHT layered on a raw TCP transport. The architecture rejected libp2p because its
PeerId (Ed25519-derived) + Multiaddr layer are a **second naming surface** alongside κ-labels —
SPINE-1 forbids that. The replacement keeps the Kademlia *algorithm* (XOR-over-content-keys is
uor-aligned by construction) and drops everything else:

- **Identity is κ.** A peer's identity is `address_bytes(PeerEndpoint.canonicalize())` —
  the κ of a [`PeerEndpoint`](../../substrate/hologram-realizations/src/lib.rs) realization
  carrying the transport address. There are no PeerIds on the wire.
- **Routing is κ-XOR.** 256 k-buckets indexed by the *decoded* 32-byte blake3 digest portion
  of κ; standard Kademlia `find_node` walk (α=3, K=20). `xor_distance` is the standard metric.
- **Provide / get_providers.** `announce(κ)` performs a κ-XOR walk to the K closest peers and
  sends them `DHT_PROVIDE(content_κ, our_endpoint_payload)` — they record the provider entry.
  `fetch(κ)` walks toward κ, calls `get_providers(κ)` on each closest hop, then dials each
  provider directly to fetch the bytes (verify-on-receipt — SPINE-4).
- **Wire format.** Length-prefixed `u32 LE len | u8 kind | payload`. Kinds are append-only
  (SPINE-5). `Kind::FetchReq / FetchResOk / FetchRes404 / Announce / Provide / FindNodeReq /
  FindNodeRes / GetProvidersReq / GetProvidersRes`. No Noise handshake — content integrity is
  provided by σ-axis verification at the application layer; transport encryption is a separate
  concern that can be added by wrapping `TcpStream` without changing the protocol.
- **Bootstrap.** `add_peer("host:port")` parses the address, computes its `PeerEndpoint` κ,
  inserts it in the routing table, and runs the Kademlia bootstrap step (find_node toward our
  own id). `add_peer` rejects Multiaddr-style strings fail-loud (SPINE-1).

This is genuinely coordinator-free content discovery, and the entire transport graph is now
κ-native: every routable identity is a κ, every wire-level lookup carries κ-labels, every
returned byte verifies by σ-axis re-derivation.

### 11.2 Federated multi-source — hierarchical `KappaSync`

The hologram network is **self-contained** — it is its own storage + transport, not a client of
external hosting. `FederatedKappaSync` chains **hologram peers** (the only kind there is) in
priority order, using the two intra-network transports the substrate defines:

1. **Local store** (zero-RTT, the existing `get_with_fetch` short-circuit).
2. **TCP peers** (`TcpKappaSync` from `hologram-net-tcp` — κ-XOR Kademlia DHT per §11.1, raw
   TCP framing, peer identity = κ of `PeerEndpoint`). Replaces the prior libp2p layer.
3. **HTTP-CAS peers** (`HttpKappaSync` — `add_gateway(url)` wires one here). A "gateway" in
   this context is itself a hologram node serving `/cas/<κ>` (spec §6.5) — not a bridge to
   anything else.

At every hop the bytes are re-derived through the σ-axis (SPINE-4); a forging peer is rejected, the
chain continues. `add_peer`/`add_gateway` route into the correct sub-sync by input shape
(`host:port` → TCP; URL → HTTP-CAS).

### 11.3 Bare-metal storage — **Merkle B-tree** of κ → extent

`hologram-store-bare`'s persistent layout is a **Merkle B-tree** (copy-on-write), not a
traditional LBA-pointer index:

- Every page (leaf or internal) is a κ-labeled record; children are referenced by κ-label, not LBA.
  The whole index *is* a κ-graph; the *store state* is one **root κ** held in the header sector.
- Data extents are bump-allocated sectors holding put-payloads; a leaf entry is
  `(κ_content, extent_lba, sectors)`.
- A bitmap of allocated sectors lives at a fixed early offset.
- Writes are copy-on-write: every modified page allocates fresh sectors, parent pointers update upward,
  and the **header sector (single-sector atomic write)** is the last write — flipping the root κ
  atomically commits the entire transaction. A torn write reverts to the previous root κ on reopen.
- GC walks the B-tree from pinned κ-roots, computes the reachable extent set, and frees the rest in
  the bitmap.

This subsumes the §5.2 deferred "B-tree + extent allocator" and is more uor-native than the spec
proposed: there is no side-channel naming — every node IS a κ.

### 11.4 Archival cold tier — **hologram bare-metal peers**

The hologram substrate does **not** delegate storage to external services (S3, IPFS, etc.) — it
**is** the storage network, end-to-end. The archival cold tier is what the **bare-metal
substrate** uniquely enables: a hologram node running directly on bare hardware (UEFI boot →
engine bring-up → `BareMetalKappaStore` over a raw `BlockDevice` + the Merkle B-tree of §11.3 +
the `NetworkInterface` driver-import of §11.9), serving the substrate's CAS at hardware capacity.
The whole stack is uor-native and self-contained:

- **The wire is the same.** A bare-metal archival peer speaks the same `/cas/<κ>` (spec §6.5) and
  uor-native TCP framing (§11.1) protocols as a hot RAM peer or a redb-backed warm peer. It is
  indistinguishable on the wire — it's just slower per-fetch but vastly larger and durable across
  reboots (TR class).
- **Why bare-metal makes archival possible.** A no-OS hologram node owns its block devices and
  NICs directly via codemodule-imported drivers (§11.9 + DU class), so it can be deployed on
  commodity disk hardware **without renting an OS-hosted service**. This is what closes the
  external-hosting gap: durability scales with disk count, not vendor contracts.
- **Replication is automatic.** Cache-on-fetch + `announce(κ)` (§11.1) means once N peers hold κ
  the network has factor-N durability without coordination. Archival peers favor this through
  their capability profile (ample `storage_quota_bytes`, `network_announce = true`).
- **Cold-tier latency** is achieved by ordering the federation chain (§11.2) hot RAM → warm redb →
  cold bare-metal, so an archival peer is queried only on hot/warm misses.

No new transport is required: a "cold peer" is the same `LibPeer`/`HttpKappaSync` interface — the
distinguishing element is the **substrate** it runs on (bare-metal, §3.2.1 HAL · §10 BM class · §11.3
B-tree · §11.9 NIC driver-import). Verification of this end-to-end role lives in the **AR** V&V
class — a federated fetch resolves through a bare-metal peer as the cold tail of the chain.

### 11.5 OPFS garbage collection

`hologram-store-opfs` implements `GarbageCollect::gc()` in the browser: list every file in the OPFS
root; mark from `pinned_roots()` through `references()` (the registry walk reused from native
stores); delete files not in the marked set. Verified end-to-end in real Chromium via the existing
Playwright harness (`opfs-test.mjs` extended with a GC scenario).

### 11.6 Quota carried across suspend/resume

The `Snapshot` realization payload now carries `storage_used: u64` alongside the linear-memory
image, globals, and cursor. The snapshot κ binds it (SPINE-1: authority in the graph). On resume,
the engine restores `storage_used` into `HostState`. A container cannot escape its quota by
suspending and resuming with a fresh ledger.

### 11.7 Cross-container scheduling — DRR over UorTime

`Capabilities` gains a `priority_weight: u32` field (0 = default = 1). The runtime's event delivery
runs **deficit round-robin** (DRR) keyed by `(container_κ, UorTime)`:

- Each container holds a deficit counter; the scheduler adds `priority_weight × quantum` per round.
- Eligible containers (with queued events) are served in `UorTime` order (ADR-058 — a monotonic
  per-engine progress counter, **not wall-clock**), pulling events while the deficit covers their
  cost (`cpu_time_per_event_ms`).
- A misbehaving container cannot starve others; the order is deterministic over uor-native quantities.

### 11.8 Transitive revoke — Delegation as a κ-graph edge

A new `Delegation{parent_caps: κ, child_caps: κ}` realization is minted on `spawn_child(parent, child_cid, child_caps)`
and put into the store. The runtime's revoke set is the reachable closure over the inverse projection:
`revoke(κ_p)` walks every Delegation whose `parent_caps == κ_p`, recursively, refusing future spawn/resume
for any descendant caps. Parent → child is now expressed in the κ-graph, recoverable by `references()` —
not in a side-channel `HashMap`.

### 11.9 `NetworkInterface` driver-import

A `WasmNetworkInterface` mirrors `WasmBlockDevice` (§3): a HAL `NetworkInterface` whose
`send`/`recv` route through an imported Wasm driver loaded by **codemodule κ-label** (uor-addr
`codemodule` — the same authority block-device drivers come from). The codemodule-κ → HAL binding is
the symmetric pattern across HAL surfaces; the V&V class **NI** asserts the *codemodule-κ → live
driver-backed device* path runs end-to-end (a hosted driver actually transports bytes through the
binding).

**RX waker bridge** (`register_rx_waker`). Production NICs are IRQ-driven, not poll-driven. The
driver imports `hologram.notify_rx()` from the host; when its IRQ fires (or the loopback test's
TX-then-RX path completes), the driver calls this import, which sets the RX-ready signal and
wakes any task registered via `NetworkInterface::register_rx_waker`. A lost-wakeup guard wakes the
task immediately if it registers after `notify_rx` has already fired.

---

## 12. Phase-2 completions — completeness audit

After Phase 1 (PR #25) landed every storage/network/runtime headline feature, a crate-by-crate
audit identified the remaining narrow areas and arbitrary defaults. Phase 2 closes them — every
addition is uor-native, every gap has an external-authority V&V test.

### 12.1 Multi-axis σ-axis registry (architecture §3.1 G-B1, V&V class **AS**)

All five axes uor-addr 0.2.0 ships are now first-class verification primitives — not just blake3:

- `verify_kappa` / `verify_kappa_axis` dispatch across `blake3` / `sha256` / `sha3-256` /
  `keccak256` / `sha512` via `prism::crypto`'s hashers.
- `address_bytes_axis(axis, bytes) -> Vec<u8>` returns the variable-width on-the-wire κ-label
  (71/73/74/135 bytes per axis).
- `KappaStore::put_axis` / `get_axis` / `contains_axis` accept any axis; the reference
  `MemKappaStore` opts in for all five (other backends keep the blake3-only hot path as
  declared in ADR-052).
- The TCK gains `axis_polymorphic_round_trip` so every backend that opts in is asserted to
  round-trip on all five axes.
- **External V&V**: the AS class differential-tests each axis against the upstream reference
  crates (`blake3`, `sha2::{Sha256,Sha512}`, `sha3::{Sha3_256,Keccak256}`) AND the FIPS 180-4 /
  FIPS 202 / Ethereum Keccak KAT vectors. A byte-level disagreement fails CI.

### 12.2 Container ABI completeness (spec §4.4, V&V class **CR-live**)

Every spec §4.4 host import is now wired in `runtime-wasmtime`:

- `sync_announce(kappa_ptr)` — buffers a `KappaSync::announce` intent. Drained by the network tick.
- `sync_fetch_request(kappa_ptr)` — buffers a `KappaSync::fetch` intent. The runtime fetches,
  verifies on receipt, caches locally; the next event sees the κ via `storage_get`. No
  sync-on-async deadlock — the intent-buffer pattern keeps the Wasm import synchronous.
- `spawn_child(cid_ptr, caps_ptr)` — buffers a child-spawn intent. Applied through the runtime's
  own `spawn_child`, which enforces delegation containment (`Capabilities::admits`).
- `diagnostics(class, code, ctx_ptr)` — mints an `ErrorEvent` realization threaded into the
  source container's error-log chain (SPINE-3 append-only).

The container ABI is now spec-complete; `ContainerIntents` carries the new buffers; `Runtime`
gains `with_sync(...)` for wiring the network layer and `process_pending_network()` for the
network event-loop tick.

### 12.3 Capability containment — 0=unbounded fix (architecture §3.4)

The naive `child ≤ parent` rule on `storage_quota_bytes` / `memory_max_bytes` /
`cpu_time_per_event_ms` silently widened authority: a child requesting unbounded (0) was accepted
under a bounded parent because `0 < N`. `Capabilities::admits` now applies a `budget_admits`
predicate: an unbounded parent admits any child; a bounded parent admits only a non-zero child
with `child ≤ parent`. This closes the silent-widening hole; the CR test battery asserts both
the admit and refuse directions.

### 12.4 Container entropy — ChaCha20 CSPRNG (spec §8.2, V&V class **EN**)

The `hologram.entropy(out_ptr, len)` import is now backed by **ChaCha20** (RFC 8439), seeded at
container instantiation from the host's `getrandom` (`rand_core::OsRng`). The previous
splitmix64 placeholder is gone — `getrandom` unavailability now fails loud
(`RuntimeError::InstantiationFailed("getrandom unavailable")`, SPINE-6 no-fallback). Independent
container instances observe independent streams (asserted by `entropy_import_is_cryptographic_rfc_8439_chacha20`).

### 12.5 NIC RX waker bridge (architecture §11.9)

`NetworkInterface::register_rx_waker` is no longer a no-op. The Wasm network driver imports
`hologram.notify_rx()`; calling it sets the RX-ready signal and wakes any registered task. A
lost-wakeup guard wakes immediately if registration races behind `notify_rx`. The NI V&V test
asserts both the wake-on-notify and the no-lost-wakeup-on-late-register paths.

### 12.6 Measured-boot driver κ (architecture §6.4, V&V class **BOOT**)

The UEFI binary previously verified κ against an embedded placeholder string — tautological.
`build.rs` now compiles a real Wasm block-device driver from a WAT source, computes its blake3
κ, and emits both to `$OUT_DIR`. The boot path `include_bytes!`-es the driver and
`include_str!`-es the expected κ; runtime re-derives κ and compares. Tampering with the embedded
bytes post-build is caught at boot. This is the **measured-boot** anchor for the substrate.

### 12.7 Bare-metal extent free-list (architecture §11.3, V&V class **BT**)

`hologram-store-bare`'s allocator previously bump-only; GC evictions leaked LBAs. v3 of the
header format adds `free_head_lba` + `free_head_digest`, persisting a chained page of free
extents. The allocator is now **best-fit** over the free list, with bump as the fallback. The
free list survives reboots; the BT class asserts post-GC reuse + reboot persistence of the
free-list state.

### 12.8 Native store — uor-native sharding + bounded read-through cache (spec §5.5, V&V class **SP**)

`hologram-store-native` is no longer "inline-all is correctness-equivalent." The §5.5 file-sharding
split is now the production path, and the read-through cache is no longer unbounded:

- **Sharding.** Content larger than `SHARD_THRESHOLD` (64 KiB) is split into `SHARD_SIZE` (64 KiB)
  pieces; each shard is itself content-addressed (`address_bytes(shard)`) and stored in the
  `INLINE` table. The top-level κ maps in the `SHARDED` table to a packed manifest of
  `(shard_κ, shard_size)` entries. Reassembly fetches each shard and concatenates. The user-facing
  κ is `address_bytes(whole content)` — no wire-visible change. Identical shards across distinct
  blobs **dedup automatically** by content-address — the uor-native property of which inline-all
  was only a degenerate case. (Witness:
  `hologram-store-native/tests/sharding_and_cache.rs::g2_*`.)
- **Bounded LRU.** The read-through Arc cache is now a **size-aware doubly-linked LRU** with a
  byte budget set per-store by `CacheConfig::cache_max_bytes` (default `256 MiB`; explicit
  override via `NativeKappaStore::open_with_config`). When the next `get` would push the total
  cached payload past the budget, the LRU evicts least-recently-used entries first. The
  persistent store is **unaffected** — eviction is local to the cache. A `cache_max_bytes = 0`
  is rejected at construction (the SP zero-copy floor requires a cache; fail-loud per SPINE-6).
  The cap is a *resource budget*, not a structural cap on what is storable — the persistent
  store grows freely; only resident bytes in RAM are bounded. (Witness:
  `hologram-store-native/tests/sharding_and_cache.rs::g1_*`.)
- **GC interaction.** Reachability walks reassemble sharded κs to extract their `references()`,
  exactly as the inline case. Eviction of an unreachable sharded κ removes its fragments —
  **unless a still-reachable sharded κ shares them by content-address**, in which case the
  shared shard stays. Cache entries for evicted κs are invalidated, no stale reads.
