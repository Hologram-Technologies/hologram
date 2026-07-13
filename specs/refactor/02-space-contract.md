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
object-safety requirement on `Space` itself); (b) the **Send-bound policy** is decided by
a P1 spike, because it decides whether one trait spans browser and native: wasm futures
are `!Send`, native wants `Send + 'static`. The candidate is maybe-Send (target-cfg'd
bounds à la `async_trait(?Send)` on wasm32) — the spike proves it against both TCK
targets before the trait freezes; (c) individual contract traits (`KappaStore`, …)
remain independently usable and dyn-capable where object safety allows — `Space` is
composition, not a cage.

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
  resume, callback, snapshot_memory, restore_memory. Engines are shared implementation
  detail (D5): `hologram-runtime` ships `engine-wasmtime` (std) and `engine-wasmi`
  (no_std) behind features; a space *selects* an engine, it does not implement one —
  unless a future space genuinely needs a bespoke engine, in which case it implements
  this same seam.

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

Shape (P3 freezes the exact signatures; this is the binding sketch):

```rust
trait Surface {
    async fn project(&self, session: &Session) -> Result<Kappa>; // state projection κ
    async fn intent(&self, session: &Session, intent: Intent) -> Result<()>;
}
// Intent: closed enum — TerminalInput, FileEdit, FrameRegion, … (exhaustive, like ops)
```

**Headless conformance**: a space with no display (esp32) implements Surface with the
null projection — `project` returns the canonical empty-projection κ, `intent` refuses
with a typed error. The TCK surface battery has a headless profile; headless is a valid
way to *pass*, not an exemption.

### 6. Realizations

The canonical forms move here from `substrate/hologram-realizations`: ContainerManifest,
CapabilitySet, Snapshot, RuntimeState, ErrorEvent, Channel, PeerEndpoint, Delegation,
ChainCompaction — plus new ones introduced by this refactor: **AppManifest** (03),
**Network** (04), and the hoisted **Holospace/Roster/Configuration** forms.

## Async posture (D14 — law)

- All I/O-shaped contract traits (`KappaStore`, `KappaSync`, `ContainerRuntime`,
  `ContainerEngine`, HAL) are **async**: native `async fn` in traits where object safety
  permits, `async-trait` where `dyn` dispatch is required.
- The tensor hot path (`hologram-exec`, `hologram-compute`) stays **synchronous** and
  allocation-free; it is reached from async code only at the session boundary.
- Rationale: blocking is illegal on the browser main thread; futures machinery is poison
  inside the deterministic kernel dispatch loop. The seam is explicit and singular.

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
