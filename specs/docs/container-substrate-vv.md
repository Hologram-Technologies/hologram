# Hologram Container Substrate — Verification & Validation

> **Scope.** What the deployment substrate (Container Runtime, Storage Layer, Network Layer)
> verifies and how. Companion to [container-substrate-architecture.md](container-substrate-architecture.md)
> (the conceptual model) and [ADR-057](../adrs/057-hologram-container-substrate.md). This is a
> **documentation-driven, V&V-driven** plan: the conceptual model is established first (the
> architecture doc), then *evaluated* by the V&V defined here. Every part is validated against an
> **external authority** — never against the substrate itself.
>
> This mirrors the root [VERIFICATION.md](../../VERIFICATION.md) discipline (σ-axis vs the BLAKE3
> reference, kernels vs the ONNX spec, κ-labels vs uor-addr + TC-05 replay) and `prism-btc`
> (SHA vs FIPS-180-4, merkle vs rust-bitcoin, algebra vs Lean).

## Principle: import the authority; don't self-certify

A test that checks the substrate against bytes the substrate itself produced proves only internal
consistency. V&V here means conformance to an authority we did **not** author, **imported** into the
suite in one of three sanctioned forms (per part — see the open question below):

- **(A) Vendored known-answer vectors** — the spec's own published test vectors as fixed fixtures
  (e.g. BLAKE3 KATs, the WebAssembly spec testsuite, RFC example vectors).
- **(B) Linked reference implementation, differential** — run an independent implementation in-tree
  and assert byte-identical results (e.g. the `blake3` reference crate, a redb crash-model harness).
- **(C) Live interop against a reference peer** — exercise the wire protocol against an independent
  node (e.g. an authentic HTTP-CAS gateway implementation) and assert interoperation.

Re-derivation through the σ-axis (SPINE-4) is the universal cross-check underneath all three: any
bytes the substrate accepts are re-hashed to their κ-label, so even (C) reduces to an externally
defined hash equality.

## Per-part external authority (the V&V ground truth table)

| Substrate part | External authority | Import form | Witness / enforcement |
|---|---|---|---|
| σ-axis hash (κ-mint/verify, `verify_kappa`) | **BLAKE3** reference (KATs) + **FIPS 180-4** (sha256/512), **FIPS 202** (sha3/keccak) | A + B | byte-identity vs reference across chunk/subtree boundaries (inherits root **AS** class) |
| κ-labels / addressing | **uor-addr** (externally validated upstream) + **TC-05** replay | B | `AddressWitness::verify()` round-trip |
| container identity / composition (ordered + G₂) | **uor-addr** composition ADR-061 + TC-05; `prism-btc` ordered-product reference | B | order-sensitivity + witness replay |
| capability delegation / containment | the foundation's **`SubtypingLattice`** relation (`type_.rs:293–309`) — more constraints = narrower = contained. uor-foundation 0.5.2 exposes no public checker (orphan-closure, `Null*` stubs), so the relation is realized faithfully in `Capabilities::admits` (grants(C')⊆grants(C)), proven a partial order — the UOR semantics, not an ACL fallback (§9 G-A3). | B (semantics) | `core::tests::cr_admits_is_{reflexive,transitive,antisymmetric}`, `cr_rejects_over_broad_delegations`; swaps to a `ConstrainedTypeResolver` when shipped |
| canonical-form realizations (manifest, cap-set, snapshot, runtime-state, error-event, channel) | **RFC 8785 (JCS)** / **RFC 8949 (deterministic CBOR)** for the canonicalize discipline | A + B | canonical round-trip byte-identity + cross-encoder agreement; `references()` extracts exactly the embedded κ-labels |
| HTTP-CAS protocol | **RFC 9110/9112** (HTTP semantics/1.1), **RFC 9111** (`immutable`, `max-age`, ETag) | A + C | response **body** byte-identity across impls (§10.6); body re-derives to the κ-label (§10.3) |
| uor-native TCP transport (`hologram-net-tcp`) | substrate's own wire spec (architecture §11.1, append-only kinds) — libp2p removed because its PeerId + Multiaddr layer is a second non-κ naming surface (SPINE-1) | B | two-node fetch + DHT provider discovery + forged-response rejection: `net-tcp/tests/uor_native_dht.rs` (6 tests) |
| Kademlia DHT (`PROVIDE`/`GET_PROVIDERS`/`FIND_NODE`) | **Maymounkov & Mazières — "Kademlia: A Peer-to-peer Information System Based on the XOR Metric"** | B | `net-tcp/src/dht.rs` (k-bucket routing + κ-XOR distance) + the integration tests above |
| Scheduling fairness (DRR) | **Shreedhar & Varghese — "Efficient Fair Queuing using Deficit Round-Robin"** | B | weighted fairness assertion: counts of events delivered per round match weight ratios within a small tolerance |
| Wasm execution determinism | **WebAssembly Core spec** + official **spec testsuite**; **WASI Preview 2 / WIT** | A | same module ⇒ byte-identical output κ-labels (§10.1) |
| snapshot compression | **zstd** / **RFC 8878** | B | decompress(compress(x)) == x; snapshot κ unaffected |
| entropy CSPRNG | **RFC 8439 (ChaCha20)** + `rand_chacha` reference | A + B | KAT vectors |
| UorTime | **uor-foundation ADR-058** UorTime canonicalize discipline | B | 16-byte canonical encoding round-trip |
| native store index | **redb** ACID/crash model | B + fault-injection | power-loss simulation: no acknowledged-`put` κ-loss (§10.15) |
| browser store | **WHATWG File System Access API** (`FileSystemSyncAccessHandle`, OPFS) | C | spec-conformant idempotent put/get |
| bare-metal storage format (LBA, B-tree, extents, crash recovery) | self-defined format **+ the crash-safety property** | B + fault-injection | torn-write / power-loss harness: header/index/runtime-state invariants hold (§5.5, §10.15) |
| bare-metal boot + drivers | **UEFI 2.x**, **NVMe**, **AHCI** specs; **smoltcp** vs **RFC 791/793/4862/1191** | A + C | spec-conformant probe/IO; PMTUD/SLAAC behavior |
| substrate-tripling | the *other two substrates* (differential) — grounded because each re-derives via the external σ-axis | B/C | byte-identical κ-labels for identical input event streams across all three (§10.16) |
| efficiency (SP class) | measured baselines + budgets (no external "authority," but no self-graded pass) | benches | `just perf` floors: zero-copy get, idempotent-put no-write, bounded reachability walk |

## V&V axes (reproducible via `just vv` + new `just vv-substrate`)

1. **Architecture** — `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, `cargo test`.
2. **Bounded reuse (RZ)** — `cargo tree` gate proving the substrate **reuses** `hologram-host`/
   `-archive` (the optimal, validated κ-native primitives) but the **tensor compute engine**
   (`hologram-exec`/`-backend`) is **absent from the store/route crates** (ADR-057 §1/§2). Reuse is
   encouraged; embedding tensor compute in the host path is not.
3. **Correctness conformance** — the `conformance` targets checking each part against its authority
   above (classes SPINE / ST / NW / CR / RZ / TR).
4. **Replay (TC-05)** — every **boundary-crossing** κ-label (manifests, snapshots, capability sets,
   received bytes) re-certifies via `AddressWitness::verify()`. The hot-path internal reuse key
   `derive_label` is unwitnessed by design (architecture §9 G-B5) — replay is *not* claimed for it.
5. **Performance / no-bottleneck (SP)** — criterion benches with per-part floors; a regression fails.
6. **Portability (TR)** — every core/realization/HAL/`*-bare` crate builds on
   `wasm32-unknown-unknown`, `thumbv7em-none-eabi`, `x86_64-unknown-none`, `no_std`.
7. **Docs** — rustdoc, intra-doc links denied.

## The implementation discipline (documentation-driven, V&V-driven)

For every part, in order:

1. **Model** — its conceptual model is in the architecture doc (or an amendment to it) **before** code.
2. **Authority** — its external authority + import form is fixed in the table above **before** code.
3. **Witness** — a conformance test against that authority is written; it **fails** (red) initially.
4. **Implement** — to make the witness pass, taking only the uor-native path (no fallback, SPINE-6).
5. **Floor** — an SP bench guards its efficiency.
6. **Record** — the row's Status moves to VERIFIED with the witness path cited (the table below).

No part is "done" without an external witness. Gaps are tracked here, never hidden.

## Status (living — Phase 0 reference landed)

Witnesses live in `substrate/hologram-store-mem/tests/` and the crates' unit tests; run via
`just vv-substrate`. (13 integration + 7 unit tests passing; native + wasm32 + thumbv7em.)

| Class | Statement | Status |
|---|---|---|
| AS | σ-axis vs the independent **BLAKE3 reference crate** | ✅ `conformance.rs::as_sigma_axis_matches_independent_blake3_reference` |
| SPINE-4 | verify-by-re-derivation; tampered/forged bytes rejected | ✅ `conformance.rs::spine4_verify_rejects_tampered_bytes`, `worked_example.rs` (malicious peer) |
| SPINE-5/6 | append-only (no delete), absent⇒None, unknown-axis fails loud (no fallback) | ✅ `conformance.rs::st_get_absent_*`, `st_unknown_axis_fails_loud_no_fallback` |
| ST | KappaStore idempotency / pin-unpin / reachability eviction — **shared TCK across mem + native(redb)** | ✅ `tck::store_battery` (mem + native), `conformance.rs::st*`, `store-native::*gc*`, `*persists_across_reopen` |
| RZ | `references()` inverse-projection + registry dispatch + **dep-gate** (compute engine absent from mem/native/http) | ✅ `conformance.rs::rz_*`, realizations unit tests, `just vv-substrate` cargo-tree gate |
| SP | zero-copy get / idempotent-no-rewrite / bounded GC | ✅ `sp.rs::sp1..sp3` |
| TR | substrate-tripling: same source builds no_std on wasm32 + thumbv7em (core/realizations/store-mem/net-http/runtime); shared TCK passes identically on mem + native | ✅ `just wasm`/`just embedded`; `tck` (byte-identity-across-running-substrates test pending the Wasm runtime) |
| CLI | `hologram` node binary (spec §9.2): put/get/pin/unpin/gc/ls/inspect/verify/manifest/**caps**/**spawn**/**serve** — storage verbs via store-generic `run`; `spawn` runs a **real Wasm container** (Wasmtime + redb → snapshot κ); `serve` runs the **HTTP-CAS gateway** | ✅ `cli::tests::*` (5) + end-to-end binary smoke: provision→`caps`→`spawn`→snapshot κ (references the Container ID), and `serve`→`curl /cas/{κ}`→byte-identical (200) / absent→404 |
| RB | **resource budgets enforced** (spec §7.6 / §10): per-event **CPU** (Wasmtime fuel — a runaway loop traps, doesn't hang), **linear memory** (StoreLimits — `memory.grow` past the cap is refused), **storage quota** (per-container `storage_put` ledger — over-budget put refused). `0 = unbounded` (no arbitrary cap, SPINE-6) | ✅ `runtime-wasmtime/tests/resource_limits.rs` (3) |
| CH | **inter-container channels / pub-sub** (spec §4.4): publish adds a **Route** κ (endpoint=channel, target=payload); capability-gated `publish`/`subscribe` (§10.4); delivery via `hg_callback`; **subscriptions persistent across suspend/resume** (keyed by Container ID, §10.11) | ✅ `runtime::tests::ch_*` (3): delivery, gates, suspend/resume replay. **Cross-peer channel fanout** — see row F1. |
| NW | HTTP-CAS: protocol codec + server (200/404/400, §10.6 byte-identity) + **verify-on-receipt** + forgery rejection; **live HTTP/1.1 transport over `std::net`**; **multi-node** — 3 real nodes over TCP: peer fetch with 404→next-peer **fallback**, **cross-node discovery** (`GET /cas/?prefix=`, merged+deduped), eviction-tolerant `get_with_fetch` cache | ✅ `net-http::tests::*` (3) + `net-http::live::tests::*` (3, incl. `multi_node_fetch_fallback_and_cross_node_discovery`) |
| NW-tcp | **uor-native TCP transport** (architecture §11.1): peer identity = κ of `PeerEndpoint`; no PeerIds, no Multiaddrs, no second naming surface (SPINE-1). κ-fetch over a κ-XOR Kademlia DHT; verify-on-receipt at every hop (SPINE-4); forged-response rejection witnessed. IPv4 + IPv6 both first-class (distinct proto bytes ⇒ distinct identity κs). RPC timeout is **operator-configurable** via `TcpConfig` — no baked-in policy cap (SPINE-6). | ✅ `net-tcp::tests::nw_tcp_*` (8): peer identity κ, two-node fetch, DHT discovery without prior direct knowledge, forged-response refusal, `add_gateway` fail-loud, Multiaddr-style strings refused, IPv4/IPv6 distinct-κ + cross-decode refusal, RPC-timeout-configurable witness. Unit tests: wire codec + k-bucket routing (6). |
| CR-live | **real Wasm execution** via Wasmtime: `hg_*` exports + linear-memory snapshot/restore; full `Runtime` runs an actual container; **the §4.4 import surface is wired** — `log`, `storage_get`/`put`/`contains`/`pin`/`unpin`, `publish`/`subscribe`, `time_now`, `entropy` — with **capability-gated storage reads** (§10.4) and **capability-gated channel pub/sub applied by the runtime** (intent-buffer pattern). End-to-end: a Wasm container publishes → Route κ in graph → a Wasm subscriber's `hg_callback` records receipt | ✅ `runtime-wasmtime::tests::*` (4, incl. `wasm_container_publishes_and_subscriber_callback_records_receipt`). **`sync_announce` / `sync_fetch_request` / `spawn_child` / `diagnostics` imports** — see row CR-live (Phase 2) below. |
| HAL | bare-metal `BlockDevice`/`NetworkInterface` seam (spec §3.2.1) + RAM-disk impl; builds no_std on thumbv7em + wasm32 | ✅ `bare-hal::tests::*` (1) |
| BM | **bare-metal storage** (spec §5): `BareMetalKappaStore` **formats + persists** the κ-map on raw sectors over a `BlockDevice` (no filesystem); sync store drives async device via a no_std `block_on`. Unit (image serialize), integration (**shared TCK**), end-to-end (format + reachability GC + **reboot persistence**) | ✅ `store-bare::{unit,tests}::*` (4); no_std on thumbv7em |
| OPFS | **browser OPFS store** (spec §5.4): κ→bytes persisted in the Origin Private File System, keyed by the σ-axis κ-label; **verify-on-receipt** on read. Verified **in a real Chromium browser** via Playwright: put→κ (κ == `address(bytes)`, byte-identical to native/bare-metal — substrate-tripling), get round-trip, **persistence across reload**, absent→null | ✅ `scripts/opfs-browser-test.sh` / `just opfs-test` (Chromium: `OPFS-TEST: PASS`) |
| BOOT | **end-to-end UEFI boot** (bare-metal spec §3): `hologram.efi` (`x86_64-unknown-uefi`, no_std `#[entry]`, pure-Rust BLAKE3 σ-axis) **booted in QEMU/OVMF with no OS** — firmware → engine bring-up → put/get/σ-axis verify + reachability GC + reboot-persistence over a `BlockDevice` → prints `HOLOGRAM-BM: PASS` | ✅ `scripts/uefi-boot-test.sh` / `just uefi-boot` (boots real firmware, asserts PASS) |
| CM | **code is κ-addressed** — drivers, container bodies, the engine are **codemodule κ-labels** (uor-addr CCMAS), not hand-authored: a driver AST → deterministic blake3 κ, referenced by a manifest and loaded through the same store/fetch/verify path | ✅ `realizations/tests/codemodule.rs` (2) |
| DI | **driver import from authoritative sources**: the engine imports **arbitrary** drivers (nvme/ahci/e1000/virtio/…) by κ from a peer/gateway via `get_with_fetch` and **verifies on receipt** — the content-addressed graph *is* the authority, so a forging source cannot supply a driver. | ✅ `runtime/tests/driver_import.rs` (2): import 5 arbitrary drivers + cache; forging source rejected |
| DU | **imported driver USED by the device (end-to-end)**: a block-device driver is fetched by κ from a source → verified → instantiated as a `WasmBlockDevice` → `BareMetalKappaStore` runs over it, so **every sector the store reads/writes is executed by the imported driver's Wasm code**; a κ round-trips through the driver. The **booted** engine (QEMU/OVMF) verifies its driver κ before binding the device. | ✅ `runtime-wasmtime/tests/driver_backed_device.rs` (1, host: driver code performs store I/O) + `just uefi-boot` (`driver κ verified — binding device` → `PASS`) |
| CR | container identity (manifest) · lifecycle spawn/suspend/resume/terminate with snapshot-as-κ state continuity · **delegation containment enforced at `spawn_child`** (SubtypingLattice, partial order + over-broad rejection) · **revocation** refuses subsequent ops · cross-runtime migration | ✅ `core::tests::cr_*` (4) + `runtime::tests::cr_*` (5), hermetic against `MockEngine`. **Live Wasmtime engine impl** of the `ContainerEngine` seam: `WasmtimeEngine` in `hologram-runtime-wasmtime`, driven by `Runtime::new(WasmtimeEngine::new(), store)` across the existing CR/CR-live tests. **Bare-metal engine**: `BareMetalEngine` (wasmi) in `hologram-runtime-bare` — see row RTBARE. |
| SP | **hologram performance contract upheld** — every substrate part held to the PV-class floors (zero-copy, idempotent-no-rewrite, bounded walks, **size-bounded LRU under a byte budget**, uor-native shard dedup). SP-1..3 structural assertions: `store-mem/tests/sp.rs`. Native-backend additions: `store-native/tests/sharding_and_cache.rs` (G1-1..4 bounded LRU, G2-1..6 shard round-trip + dedup + GC). Criterion regression floors: `store-native/benches/sp_floors.rs` under `just perf`. | ✅ |
| TR | substrate-tripling byte-identity (same source emits byte-identical κ stream on mem + native + bare) + no_std builds + no-OS UEFI boot + crash-safety torn-write reverts + hardware probing (`BlockIO` handles enumerated at boot) | ✅ `runtime-wasmtime::tests::tripling::tr_same_container_emits_byte_identical_kappa_stream_on_mem_native_bare`; `just wasm`/`just embedded` (no_std builds); `just uefi-boot` (no-OS + probe + torn-write); `store-bare::tests::bt_torn_header_write_reverts_to_prior_root` |
| RTBARE | **`hologram-runtime-bare`** (C1) — no_std `ContainerEngine` over **wasmi**: instantiate, hg_init/event/suspend/resume/callback, linear-memory snapshot/restore, §4.4-only import gate (SPINE-6). The same `Runtime` orchestration drives this engine as drives Wasmtime — substrate-tripling at the runtime layer. | ✅ `hologram-runtime-bare::tests::*` (3) |
| NETBARE | **`hologram-net-bare`** (C2) — no_std `KappaSync` over the HAL `NetworkInterface`: frame codec (length-prefixed kind+payload, append-only kinds), verify-on-receipt on every `FETCH_RES_OK` (SPINE-4), forging responder refused. | ✅ `hologram-net-bare::tests::*` (3) including `bare_net_sync_rejects_forged_response` |
| B1 | **Reboot-monotonic epoch** (G-C1): `BareMetalKappaStore` header v4 persists `reboot_epoch`, bumped on every `open`; `RuntimeStateRegion` realization (D1) carries `(reboot_epoch, generation)` — total order over all writes across reboots. | ✅ `store-bare/tests/reboot_epoch.rs` (3) |
| B2 | **Chain compaction** (G-C4): `Runtime::set_error_log_threshold` + `ChainCompaction` realization (zero refs, breaks predecessor reachability). After threshold, old tail unreachable → reclaimed by GC. `0` ⇒ unbounded (opt-in, SPINE-6). | ✅ `runtime/tests/chain_compaction.rs` (2) |
| B3 | **Browser sync/async bridge** (G-C2): `hologram-store-opfs::bridge` — `SyncOpfsBridge` (main thread, `Atomics.wait` over a SAB) + `BridgeWorker` (Worker, `Atomics.waitAsync` + dispatch through the existing async `opfs_*` surface). All five opcodes (Put/Get/Contains/Delete/Iterate) wired through to OPFS — no stubs. Verify-on-receipt preserved across the boundary (main thread re-derives κ on the response bytes). | ✅ `cargo build --target wasm32-unknown-unknown --release` (bridge.rs compiles + exports `SyncOpfsBridge`/`BridgeWorker` + `bridge_state_offset`/`bridge_state_idle`); `web/bridge-worker.mjs` is the Worker glue. |
| B4 | **`?Send` async traits** (G-D1): `LocalKappaSync` + `LocalContainerRuntime` accept non-`Send` implementors (embassy futures). Disjoint from the std `Send+Sync` traits — explicit opt-in per executor model. | ✅ `substrate-core::tests::local_kappa_sync_accepts_non_send_implementors` |
| B5 | **Qualified workload bounds** (G-C3): runtime refuses any container that declares an import outside the spec §4.4 `hologram.*` host surface — the SPINE-1 "arbitrary workload" claim is structural, not aspirational. Symmetric in `runtime-wasmtime` (std) and `runtime-bare` (no_std). | ✅ `runtime-wasmtime::tests::workload_bounds::b5_container_with_non_hologram_import_is_refused_at_instantiation`; `runtime-bare::tests::bare_engine_refuses_non_hologram_imports` |
| D1 | **9 bare-metal sibling realizations** (architecture §3.3): `BareMetalStorageFormat`, `RuntimeStateRegion`, `HardwareInventory`, `CrashRecord`, `DiagnosticLbaRecord`, `IntentLogRecord`, `GcMarkState`, `HardwareAbstractionTraits`, `BootConfig`. All in `REGISTRY`; reachability walks them transparently. | ✅ `hologram-realizations::tests::*` (6 new tests + `registry_covers_all_nine_bare_metal_sibling_iris`) |
| F1 | **Cross-peer channel fanout** (CH-network): `publish` queues `KappaSync::announce(route_κ)`; remote peer's `Runtime::poll_channel_fanout(channel)` discovers, fetches (verify-on-receipt), delivers to local subscribers via the existing `pump`. | ✅ `runtime/tests/channel_fanout.rs::f1_cross_peer_channel_publish_fans_out_to_remote_subscriber` |
| E1 | **EFI hardware probing** (TR §10.17): boot path enumerates `BlockIO` (always) + `SimpleNetwork` (under `probe-nics` feature) handles, records the count, mints a `HardwareInventory` κ as the auditable graph node for the boot's bound devices. | ✅ `cargo build -p hologram-efi --target x86_64-unknown-uefi --release`; `just uefi-boot` prints the device counts |
| E2 | **NIC driver κ measured-boot anchor** (symmetric to §12.6 block-driver): `build.rs` assembles a real Wasm NIC driver, computes its blake3 κ, emits both. The boot path `include_bytes!`-es + `include_str!`-es + verifies on bring-up. Tampering with the embedded NIC driver post-build → FAIL. | ✅ `build.rs` emits `OUT_DIR/nic-driver.{wasm,kappa}`; `main.rs::verify_driver("nic", ...)` |
| DHT | **κ-XOR Kademlia content discovery** (no coordinator, uor-native): `announce(κ)` issues `DHT_PROVIDE(content_κ, our_endpoint)` to the K=20 closest peers in κ-XOR space; `fetch` walks K-closest via `FindNodeReq` then queries `GetProvidersReq`. The Kademlia *algorithm* is preserved — uor-aligned by construction (XOR over content keys is a structural relation, not a registry lookup) — but the libp2p PeerId/Multiaddr layer is gone (architecture §11.1) | ✅ `net-tcp::tests::nw_tcp_announce_then_get_providers_resolves_without_prior_knowledge` (3-node DHT walk; C resolves κ from A through B without direct knowledge of A) |
| FED | **Federated multi-source `KappaSync` over hologram peers only** — local → TCP DHT (§11.1) → HTTP-CAS peer, hierarchical fallback, verify-on-receipt at every hop; `add_gateway(url)` wires another hologram CAS peer (no stub). Test: hot miss → next peer hit → bytes verified; forging hop skipped (§11.2) | ✅ `substrate-core::tests::fed_*` (chain falls through; verify rejects forgery) |
| BT | **Merkle B-tree on bare-metal** (§11.3): every page is a κ-labeled record; the store root is one κ in the header sector; CoW write-discipline; torn-write reverts to previous root atomically. Tests: random put+get, GC reclaim, **reboot persistence under simulated crash mid-transaction** | ✅ `store-bare::tests::bt_*` (CoW round-trip, GC reclaim, torn-write reverts to prior root κ) |
| AR | **Archival cold tier = hologram bare-metal peer** (§11.4 + §11.3 + §11.9): a `BareMetalKappaStore`-backed node participates in the federation chain as the cold tail; same `/cas/<κ>` + uor-native TCP transports as any other peer; durability via the Merkle B-tree across reboots; NIC reached through a codemodule-imported driver. No external hosting — hologram IS the storage network. Test: three-tier chain (RAM-hot → redb-warm → bare-metal-cold), the cold-only κ falls all the way through and resolves through the bare-metal peer | ✅ `runtime-wasmtime::tests::archival_cold_tier_via_bare_metal_peer_in_federation` |
| OG | **OPFS garbage collection** (browser): mark from pins through `references()`, delete unreachable files. Test: put κ_a + κ_b, pin only κ_a, GC, κ_b absent in real Chromium (§11.5) | ✅ `opfs-test.mjs::OPFS-GC-TEST: PASS` (Playwright) |
| QC | **Quota carries through suspend/resume**: `Snapshot` payload includes `storage_used`; resume restores it; an over-budget put after resume is refused. Test: fill quota → suspend → resume → next put refused (§11.6) | ✅ `runtime-wasmtime::tests::qc_quota_carries_across_suspend_resume` |
| SC | **DRR scheduling over UorTime** — no wall-clock; `priority_weight` is a `Capabilities` field; misbehaving container cannot starve others. Test: 3 containers with weights `1/1/4`, the high-weight container is served ~4× as often, ordering deterministic across runs (§11.7) | ✅ `runtime::tests::sc_drr_fairness_over_uortime` |
| RV | **Transitive revoke via Delegation κ-graph**: `spawn_child` mints a `Delegation{parent_caps, child_caps}` realization; `revoke(κ_p)` walks the inverse projection (`references()`) and revokes the entire subtree. Test: A→B→C; revoke A; C's spawn/resume refused (§11.8) | ✅ `runtime::tests::rv_transitive_revoke_cascades_through_delegation` |
| NI | **`NetworkInterface` driver-import**: imported Wasm driver routes packets through the HAL; codemodule-κ → `WasmNetworkInterface` binding; end-to-end send/recv round-trip via the imported driver code. Mirrors DU for the network surface (§11.9). Includes the **RX waker bridge** — `hologram.notify_rx` from the driver fires the registered task; lost-wakeup guard wakes on late register | ✅ `runtime-wasmtime::tests::ni_imported_wasm_driver_routes_packets` + `ni_notify_rx_drives_register_rx_waker` |
| AS (axes) | **Multi-axis σ-axis** (§12.1) — `verify_kappa` / `address_bytes_axis` dispatch across blake3 / sha256 / sha3-256 / keccak256 / sha512; `KappaStore::put_axis` accepts every axis the reference backend opts in for. Differential KAT V&V against upstream `blake3`/`sha2`/`sha3` reference crates AND FIPS 180-4 / FIPS 202 / Ethereum-Keccak KAT vectors | ✅ `substrate-core::tests::as_*` (10 tests, all five axes, KAT + differential) |
| CR-live | **Container ABI completeness** (§12.2) — `sync_announce`, `sync_fetch_request`, `spawn_child`, `diagnostics` host imports wired in `runtime-wasmtime`; intents drained by `Runtime::process_pending_network` (network tick) or applied synchronously (spawn_child/diagnostics). The container can use the full §4.4 surface; the `ErrorEvent` chain is appended per spec §7.5 | ✅ `runtime-wasmtime::tests::container_abi_full::*` (5 tests, including sync-fetch resolves κ for next event) |
| CR-budget | **`Capabilities::admits` 0=unbounded fix** (§12.3) — the naive `child ≤ parent` rule silently widened authority when child=0 (unbounded) under bounded parent. The new `budget_admits` predicate refuses this; unbounded parent admits any child. Test: refuse unbounded-under-bounded; admit any-under-unbounded | ✅ `substrate-core::tests::cr_rejects_over_broad_delegations` + `cr_unbounded_parent_admits_any_child_budget` |
| EN | **ChaCha20 entropy** (§12.4) — `hologram.entropy` is now RFC 8439 ChaCha20 seeded from `OsRng`. Independent instances observe independent streams. `getrandom` unavailability fails loud (SPINE-6) | ✅ `runtime-wasmtime::tests::container_abi_full::entropy_import_is_cryptographic_rfc_8439_chacha20` |
| BOOT | **Measured-boot driver κ** (§12.6) — `build.rs` assembles the real driver Wasm + computes the expected blake3 κ; the boot path compares runtime-derived κ against the build-recorded anchor. Tampering with embedded bytes post-build → boot FAIL | ✅ `scripts/uefi-boot-test.sh` (QEMU/OVMF prints `driver κ verified (N bytes) — binding device`) |
| BT (alloc) | **Bare-metal extent free-list** (§12.7) — header v3 persists `free_head_lba` + `free_head_digest`; allocator is best-fit over the free list, bump fallback. GC eviction releases LBAs; subsequent puts reuse them. Free list survives reboots | ✅ `store-bare::tests::bt_free_list_reclaims_evicted_extents_across_reboots` |
| TCK | **Axis-polymorphic + zero-copy battery** — every backend runs `axis_polymorphic_round_trip` (opt-in per backend) and `zero_copy_get_returns_arc_handle` (consecutive `get`s share an `Arc`, the SP zero-copy floor) | ✅ `substrate-tck::*` (every backend re-asserts these via `store_battery`) |

## Import-form policy (resolved: hermetic-first)

The per-part **import form** (A vendored vectors / B linked reference / C live interop) is fixed in
the table per part. For **Wasm determinism** the policy is **hermetic-first**: the core
`just vv` gate uses vendored vectors + linked reference (A/B), offline and deterministic; live
interop (C) against upstream Wasmtime/`wasmi` runs in a separate, opt-in `just vv-interop`
lane. This keeps the core gate reproducible while still exercising the strongest authority on
demand. The TCP transport is hermetic-by-design — there is no third-party peer to interop with
(the substrate is its own wire spec, architecture §11.1).
