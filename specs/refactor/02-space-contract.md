# 02 — The Space Contract (`hologram-space`)

Decisions: D2, D3, D5, D7, D14 (see `00-overview.md`).

## Principle

A **space** is a place hologram executes — browser, native desktop, bare metal, iOS,
esp32. hologram defines the contract; a space implements it. **Every space implements
the identical surface**: no space-specific trait extensions, no optional methods gated by
platform. Platform differences live behind the traits (in the implementation), never in
them. The compile-time question "can hologram run here?" reduces to "does this crate
implement the contract and pass `hologram-tck`?"

```rust
// The embedder experience the contract must preserve:
use hologram::space::Space;

struct IosSpace;
impl Space for IosSpace {
    // storage, sync, runtime engine, HAL, surface — all of it, or it isn't a space
}
```

## The `Space` trait shape

`Space` is an **aggregate with associated types** — the one place a platform names its
concrete parts; everything downstream is generic over it:

```rust
trait Space {
    type Store: KappaStore;
    type Sync:  KappaSync;
    type Engine: ContainerEngine;
    type Surface: Surface;
    type Entropy: Entropy;
    type Clock: Clock;
    type Spawner: Spawner;

    fn store(&self) -> &Self::Store;
    // …accessor per part; construction/teardown lifecycle methods
}
```

Fixed now: (a) associated types, not `dyn` fields — `Client` is generic over `S: Space`
(monomorphized per platform; FFI/packaging crates instantiate concrete spaces, so no
object-safety requirement on `Space` itself); (b) the **Send-bound policy is maybe-Send**
— *resolved by the P0.5 spike* (2026-07-14) from the existing precedent, not deferred to
a P1 spike: the substrate already ships `KappaSync: Send + Sync` (native) and
`LocalKappaSync` under `#[async_trait(?Send)]` (wasm/bare, where futures are `!Send`), so
the async parts of `Space` use the same target-conditional bounds. Note the associated
types split by posture — `Store` is a **sync** trait (§Async posture), `Sync`/`Engine`
are async; (c) individual contract traits (`KappaStore`, …) remain independently usable
and dyn-capable where object safety allows — `Space` is composition, not a cage.

**Implemented shape (2026-07-15) — `Runtime`, not a bare `Engine`.** The trait above lists a
`type Engine: ContainerEngine` with the `ContainerRuntime` "composed above." Implementation
experience corrected this: a `Runtime<E, S>` **owns its store**, so composing it from a separate
`Engine` + `Store` would force Arc-sharing the store and the `Client` owning the composed runtime
(the lifecycle `Session` borrows the runtime). The clean shape the code prefers is for the space
to expose the **composed runtime** directly:

```rust
trait Space {
    type Store: KappaStore;      // sync (LAW-4)
    type Resolver: Resolver;     // async, maybe-Send — the resolve/boot seam
    type Runtime: ContainerRuntime; // the composed engine+store; Client::open drives a Session over it
    fn store(&self) -> &Self::Store;      // an impl typically delegates this to runtime().store()
    fn resolver(&self) -> &Self::Resolver;
    fn runtime(&self) -> &Self::Runtime;
}
```

The engine is named *inside* the `Runtime` (`Runtime<Engine, Store>`), so `type Engine` is an
implementation detail rather than a top-level associated type; `store()` delegates to
`runtime().store()` so there is one content store. `Sync`/`Surface`/`Entropy`/`Clock`/`Spawner`
remain the fuller contract to build out (each a space part a platform names); they are additive
and do not change this core. `Client::open(container_κ, caps_κ) -> Session` is wired over
`space.runtime()` (05-tooling.md).

## Contract contents

`hologram-space` unifies today's `substrate/hologram-substrate-core`,
`substrate/hologram-realizations`, and `substrate/hologram-bare-hal`. It contains **only**
traits, canonical forms, errors, and laws — zero platform code.

### 1. Storage — `KappaStore` (persistence lives in the contract)

Persistence is handled differently on every host (OPFS in the browser, redb natively, raw
sectors on bare metal, Core Data/FS on iOS) — therefore the *contract* owns the trait and
each space owns its implementation. Carried over from substrate-core unchanged in spirit:

- `put(bytes) -> κ` (idempotent — Law L3), `get(κ) -> Option<Bytes>`, `pin/unpin`, `gc`
  (reachability closure), wide-axis variants.
- `verify_kappa` (σ-axis re-derivation, SPINE-4) and `address_bytes` (κ-minting) as free
  functions — identical bytes yield identical κ on every space.

`KappaStore` is deliberately **local-only** — one space, one store, no peer awareness.
That is what keeps the trait implementable from esp32 flash to browser OPFS.

### 2. Sync — `KappaSync`

`fetch`, `announce`, `discover` with verify-on-receipt at every hop. Protocol semantics
live in `hologram-net` (see `04-networks.md`); the trait lives here because a space must
provide (or explicitly stub) its transport pump.

**Known law-2 violation to fix, tracked here**: today's trait carries
`add_peer(peer_addr: &str)` / `add_gateway(url: &str)` — string addresses, a second
naming surface. P1 moves the trait verbatim (κ-stability of code moves), and **P3's API
shaping replaces both with κ-shaped forms** (PeerEndpoint realization κ + ephemeral
transport hints supplied by the pump, never stored). The violation may not survive into
the first published release.

**Local by contract, distributed by composition.** `KappaSync` is the distribution seam:
`Peer::resolve(κ)` tries the local store, else fetches via sync, verifies, and may
persist; `resolve_closure(κ)` migrates whole object graphs (apps, snapshots, rosters)
between peers. Because content is immutable, append-only, and identically addressed
everywhere (same bytes → same κ), every local store is automatically a valid
replica/cache of the global content space — there is no consistency protocol to design,
only content exchange. Network-wide semantics (membership, policy, the "distributed
OPFS") are the composition of these two traits with the Network model in
`04-networks.md`; durability/replication policy is an explicit open item there.

### 3. Runtime — `ContainerRuntime` + `ContainerEngine`

- `ContainerRuntime`: `spawn(manifest, caps)`, `suspend -> snapshot κ`, `resume(snapshot,
  caps)`, `terminate` — the lifecycle every space drives.
- `ContainerEngine` (the seam engines implement): instantiate, init, event, suspend,
  resume, callback, snapshot_memory, restore_memory. It is a **synchronous** trait
  (`Send + Sync`) — the async lifecycle lives above it in `ContainerRuntime` (Send) /
  `LocalContainerRuntime` (`?Send`, for wasm/single-core executors), the maybe-Send pair
  matching LAW-4. Engines are shared implementation detail (D5).

**Engine coverage across targets (wasm / WASI / iOS / bare — the completeness question).**
The seam is what makes the runtime target-complete; `hologram-runtime` ships two reference
engines behind features that between them span every target, and a space may add a bespoke
one via the same trait:

| Target | Engine | Notes |
|--------|--------|-------|
| native desktop / server | `engine-wasmtime` (std, Cranelift JIT) | fastest; the default host engine |
| **browser (wasm32)** | `engine-wasmi` (no_std interpreter) | **verified to build for `wasm32-unknown-unknown`** — the interpreter runs *inside* the wasm sandbox |
| **iOS** | `engine-wasmi` | Apple forbids JIT; the interpreter needs none |
| bare-metal / esp32 | `engine-wasmi` | no_std, the C1 architecture path |

So `engine-wasmi` is the **portable universal engine** (browser + iOS + bare), and
`engine-wasmtime` is the native fast path. Two explicitly-noted extensions, both via the
seam, not new contract surface:
- **Fast browser engine**: the wasmi interpreter is correct but slow in the browser. A
  space MAY implement `ContainerEngine` over the browser's native `WebAssembly` API
  (JS-driven, near-native) — this lives in `holospaces-browser`, a *space-provided* engine
  (D5), not in `hologram-runtime`.
- **WASI**: hologram containers use hologram's own `hg_*` host-import ABI, not WASI. A WASI
  module is an *ingest* concern (wrap it like an OCI image, `03-holo-format.md`), or a
  future WASI-shim engine behind the same seam — not a gap in the runtime's trait surface.

### 4. HAL — `BlockDevice`, `NetworkInterface`, `Entropy`, `Clock`, `Spawner`

`BlockDevice` and `NetworkInterface` absorbed from `hologram-bare-hal` unchanged. The
κ-disk (in `spaces/holospaces`) implements `BlockDevice` over any `KappaStore` — Law L4:
no second storage medium.

**Three additions this refactor makes explicit** — today they are ambient per-platform
accidents (OsRng in wasmtime, JS-pumped time in the browser, no story on bare metal),
but every space needs them and hoisted space-agnostic code in `hologram-runtime` cannot
reach ambient platform APIs:

- **`Entropy`** — cryptographic randomness source (seeds the ChaCha20 machinery; key
  generation for operators/attestation). Browser: `crypto.getRandomValues`; native: OS
  RNG; bare: hardware RNG or explicit seed injection at provision.
- **`Clock`** — monotonic time (lifecycle timeouts, leases) and, where the platform has
  one, wall-clock for event payloads. Wall-clock is **never required** — a bare space
  without an RTC is conformant; consumers of wall-time must handle absence.
- **`Spawner`** — the executor seam: how this space polls the async contract traits
  (browser: microtask/worker; native: tokio; bare: the space's run loop). Hoisted
  runtime code spawns through this, never through a named executor.

All three are TCK-batteried like the rest of the contract.

### 5. Surface (new in this refactor; D10)

A minimal presentation/interaction capability generalizing holospaces' `projection.rs`
(Workspace/Intent): a κ-addressed projection of a running workload's state plus an intent
channel driving it (terminal I/O, file edits, framebuffer regions). Every space provides
a surface; a portable app view targets the surface and therefore runs on all spaces.
Spaces MAY additionally expose native view slots (see `03-holo-format.md` §views). The
surface is deliberately small — design systems plug in above it, in future projects.

**Implemented shape (2026-07-15) — κ in, κ out; no `Session` in the contract.** The seam
takes the running workload's **κ**, not a runtime `Session`: the contract crate `hologram-space`
must not depend on `hologram-runtime` (the RZ invariant), and `intent` returns the published
event's **κ** (not `()`), because an intent is content-addressed (Law L1 — identical intents
address to the same κ). The trait is **maybe-Send**, the same cfg-gated posture as `KappaSync`
(`Send + Sync` native / `?Send` on `wasm32`/bare, where a browser surface holds `!Send` DOM
handles). Lives in `hologram-space::surface`:

```rust
#[cfg(not(target_arch = "wasm32"))] #[async_trait] pub trait Surface: Send + Sync {
    async fn project(&self, workload: &KappaLabel71) -> Result<KappaLabel71, SurfaceError>;
    async fn intent(&self, workload: &KappaLabel71, intent: Intent)
        -> Result<KappaLabel71, SurfaceError>;
} // + a #[cfg(target_arch = "wasm32")] #[async_trait(?Send)] twin
pub enum Intent { TerminalInput(Vec<u8>), FileEdit { path, content }, FrameRegion { … } } // closed
pub enum SurfaceError { Headless, NotProjectable, Backend(&'static str) }
```

**Headless conformance**: a space with no display (esp32) implements Surface with the
null projection — `project` returns the canonical empty-projection κ, `intent` refuses
with a typed error. Realized as the reference `NullSurface` (`project` → `address_bytes(&[])`,
`intent` → `Err(SurfaceError::Headless)`); both reference spaces (SpikeSpace, TestSpace) use it.
The TCK surface battery has a headless profile; headless is a valid way to *pass*, not an
exemption. **This completes the 7/7 spec-02 `Space` parts.**

### 6. Realizations

The canonical forms move here from `substrate/hologram-realizations`: ContainerManifest,
CapabilitySet, Snapshot, RuntimeState, ErrorEvent, Channel, PeerEndpoint, Delegation,
ChainCompaction — plus new ones introduced by this refactor: **AppManifest** (03),
**Network** (04), and the hoisted **Holospace/Roster/Configuration** forms.

## Async posture (D14 — law; corrected by the P0.5 spike, 2026-07-14)

The original draft said "all I/O-shaped contract traits are async." The P0.5 de-risk
spike (D28) checked this against the working substrate and found it **wrong for storage**
— so the law is corrected here from evidence:

- **Storage is synchronous.** `KappaStore::put/get` return `Result` directly, not
  futures. This is correct and wasm-safe: the browser reaches persistent storage through
  `FileSystemSyncAccessHandle`, which is *synchronous inside a Web Worker* (the holospaces
  model) — so a sync store never blocks a forbidden thread. The reference `MemKappaStore`
  is `Send + Sync` via `spin::Mutex`, no `std`. Wrapping it in async would add a pointless
  layer over an already-correct sync API.
- **The tensor hot path is synchronous** (`hologram-exec`, `hologram-compute`) and
  allocation-free — unchanged.
- **Network sync and runtime lifecycle are asynchronous.** `KappaSync`
  (`fetch`/`announce`/`discover`) and `ContainerRuntime`/`ContainerEngine` wrap genuine
  I/O and event loops, so they are `async`.
- **The async↔sync seam is the network/boot boundary, not storage.** Async code enters
  the synchronous store + compute at that boundary; the deterministic kernel dispatch loop
  never sees a future.
- **Send-bound policy (resolved by the spike, from existing precedent)**: maybe-Send.
  The codebase already ships both `KappaSync: Send + Sync` (native/multi-thread executor)
  and `LocalKappaSync` under `#[async_trait(?Send)]` (bare-metal/wasm single-thread, where
  futures are `!Send`). The `Space`/`Client` async surface follows the same pattern —
  target-conditional Send bounds — rather than forcing one bound across both worlds.

## Conformance — `hologram-tck` defines "space"

- **TCK = Technology Compatibility Kit** (in the Java-TCK sense; the substrate crate
  called it a "Test Conformance Kit" — same meaning): the executable test battery that
  *is* the definition of conformance. Pass it and you are a valid space; there is no
  other certification.
- The TCK is the executable meaning of the contract: KappaStore battery (ST/SPINE
  invariants), sync verify-on-receipt battery, lifecycle battery (spawn/suspend/resume/
  terminate against the mock engine), HAL battery, surface battery.
- Ships the **reference in-memory store** (today's `substrate/hologram-store-mem`) as the
  oracle implementation every real store is differentially compared against.
- CI rule: every `spaces/holospaces-*` crate runs the full TCK on its target (browser via
  Playwright/wasm, native directly, bare via RAM block device). A space that does not
  pass the TCK does not merge.

## The space implementations

| Space | Store | Engine (selected) | Transport pumps | Views |
|-------|-------|-------------------|-----------------|-------|
| `holospaces-browser` | OPFS pack store | wasmi (`engine-wasmi`) | WebRTC data channel, WebSocket egress | wasm-bindgen Console/Workspace |
| `holospaces-native` | redb | wasmtime (`engine-wasmtime`) | TCP (κ-XOR DHT), **iroh** (QUIC/NAT traversal), WebRTC + WS listener (browser interop, 04) | CLI/desktop |
| `holospaces-bare` | raw-sector block store | wasmi | bare `NetworkInterface` pump | serial/framebuffer |
| future `holospaces-ios` | platform store | wasmi or platform | platform transport | Swift native |
| future `holospaces-esp32` | flash block store | wasmi | radio/eth pump | none/headless |

Naming pattern: **`holospaces-<host>`**. `spaces/holospaces` (no suffix) is the portable
core shared by all: system emulators (RISC-V / AArch64 / x86-64), κ-disk, OCI/devcontainer
boot provisioning, content-net glue (PacketLink/TransportEndpoint), projection machinery.

## External spaces (D21): the contract is open

Spaces may be created — and in-tree spaces later extracted — in **external repositories**.
In-tree residence is a convenience for CI co-gating, never a privilege. Enforcement rules:

1. **No sealed traits.** Nothing in the contract path (`Space`, `KappaStore`, `KappaSync`,
   `ContainerRuntime`, `ContainerEngine`, HAL, surface) uses the sealed-trait pattern or
   `pub(crate)` supertraits. If an external crate cannot write the impl, the contract is
   broken.
2. **Everything a space needs is published API.** A space builds against
   `hologram = { features = ["space", "runtime", "net"] }` (or the subcrates via the
   facade) — including the shared engines (`engine-wasmtime`, `engine-wasmi`) and the
   portable machinery in `spaces/holospaces` (which publishes like every other crate,
   D16). An in-tree space that reaches around the public surface is a bug the extraction
   proof (rule 5) will catch.
3. **TCK runs anywhere.** `hologram-tck` is consumable as a dev-dependency battery
   (`cargo test` in the external repo) and via `hologram space tck` against a space by
   name/path. Conformance certification must not depend on this repo's CI harness.
4. **No in-tree registry.** `Client` accepts any `impl Space` by value/generic — there is
   no compiled-in enumeration of blessed spaces. The facade's `space-browser`/`space-native`/
   `space-bare` features are re-export sugar for the in-tree impls, nothing more.
5. **Extraction proof obligation.** Moving any `spaces/holospaces-*` crate to its own
   repository must require only replacing workspace path deps with published version
   deps. This is checked in CI conceptually (no non-public API usage) and is the standing
   test that rules 1–4 hold.

Consequence: third parties can ship `holospaces-android`, `acme-holospaces-fpga`, etc.,
certify them with the public TCK, and never coordinate with this repo beyond following
lockstep releases.

## What was hoisted out of holospaces (D7)

`Peer`, `Session` (the Provisioned→Running→Suspended→Terminated lifecycle machine), and
the platform-manager model (`Manager`, `Operator`, `Roster`, `Configuration`,
control-plane-as-content per holospaces ADR-018) move to **`hologram-runtime`**. They are
space-agnostic workload-lifecycle management — exactly hologram's charter. Spaces keep
only their *views* of the manager (browser console UI, future iOS UI).

## Explicitly rejected alternatives (for the record)

- *holospaces defines the contracts, hologram implements below them* — rejected: inverts
  the working dependency direction and splits law-ownership from trait-ownership.
- *Spaces subsume compute backends* — rejected: environment and kernel-target are
  orthogonal axes; a browser space may host the wasm compute path, a native space the
  Metal path. Compute stays in `hologram-compute`.
- *Per-space trait extensions* — rejected by D5's clarification: uniform surface, TCK-
  enforced. Extension pressure must be answered by evolving the contract for everyone.
