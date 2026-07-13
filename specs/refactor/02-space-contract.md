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

### 2. Sync — `KappaSync`

`fetch`, `announce`, `discover` with verify-on-receipt at every hop. Protocol semantics
live in `hologram-net` (see `04-networks.md`); the trait lives here because a space must
provide (or explicitly stub) its transport pump.

### 3. Runtime — `ContainerRuntime` + `ContainerEngine`

- `ContainerRuntime`: `spawn(manifest, caps)`, `suspend -> snapshot κ`, `resume(snapshot,
  caps)`, `terminate` — the lifecycle every space drives.
- `ContainerEngine` (the seam engines implement): instantiate, init, event, suspend,
  resume, callback, snapshot_memory, restore_memory. Engines are shared implementation
  detail (D5): `hologram-runtime` ships `engine-wasmtime` (std) and `engine-wasmi`
  (no_std) behind features; a space *selects* an engine, it does not implement one —
  unless a future space genuinely needs a bespoke engine, in which case it implements
  this same seam.

### 4. HAL — `BlockDevice`, `NetworkInterface`

Absorbed from `hologram-bare-hal` unchanged. The κ-disk (in `spaces/holospaces`)
implements `BlockDevice` over any `KappaStore` — Law L4: no second storage medium.

### 5. Surface (new in this refactor; D10)

A minimal presentation/interaction capability generalizing holospaces' `projection.rs`
(Workspace/Intent): a κ-addressed projection of a running workload's state plus an intent
channel driving it (terminal I/O, file edits, framebuffer regions). Every space provides
a surface; a portable app view targets the surface and therefore runs on all spaces.
Spaces MAY additionally expose native view slots (see `03-holo-format.md` §views). The
surface is deliberately small — design systems plug in above it, in future projects.

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
| `holospaces-native` | redb | wasmtime (`engine-wasmtime`) | TCP (κ-XOR DHT), **iroh** (QUIC/NAT traversal) | CLI/desktop |
| `holospaces-bare` | raw-sector block store | wasmi | bare `NetworkInterface` pump | serial/framebuffer |
| future `holospaces-ios` | platform store | wasmi or platform | platform transport | Swift native |
| future `holospaces-esp32` | flash block store | wasmi | radio/eth pump | none/headless |

Naming pattern: **`holospaces-<host>`**. `spaces/holospaces` (no suffix) is the portable
core shared by all: system emulators (RISC-V / AArch64 / x86-64), κ-disk, OCI/devcontainer
boot provisioning, content-net glue (PacketLink/TransportEndpoint), projection machinery.

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
