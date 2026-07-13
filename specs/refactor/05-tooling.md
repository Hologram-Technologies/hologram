# 05 — Tooling: Client Facade, CLI, FFI & SDKs

Decisions: D4, D6, D13 (see `00-overview.md`).

## Principle (law 6)

**One programmatic surface.** The `Client` type in the `hologram` facade crate is the
single place behavior is composed. The CLI is clap over Client. The C ABI is extern
functions over Client. Python/TypeScript/Swift SDKs are generated over those. Bindings
cannot drift because they contain no logic.

```rust
// The only programmatic entry — everything else wraps this.
let client = hologram::Client::builder()
    .space(NativeSpace::default())        // any impl of the space contract
    .build()?;

let app     = client.compile(source)?;            // → .holo (v3); pure compute, sync
let holo    = client.provision(app, caps).await?; // ingest → κ (store I/O — async, law 4)
let session = client.open(&holo).await?;          // resolve + Session (I/O — async)
session.boot().await?;
let snap    = session.suspend().await?;           // snapshot κ — migratable
```

`Client` surfaces, feature-gated exactly like the facade (01 §feature matrix):

| Area | Methods (indicative) | Backed by |
|------|----------------------|-----------|
| compile | `compile`, `compile_with_backward`, frontends | hologram-compiler |
| execute | `load`, `execute`, warm-start | hologram-exec |
| store | `put`, `get`, `pin`, `unpin`, `gc`, `verify` | space's KappaStore |
| net | `fetch`, `announce`, `discover`, `resolve_closure` | hologram-net + space transport |
| app | `provision`, `open`, `compose`, inspect manifest | hologram-archive + runtime |
| manager | `sign_in`, `roster`, `sync_from`, `configure`, `reconfigure` | hologram-runtime (hoisted Manager) |
| network | `create_network`, `join`, `delegate` | 04-networks realizations |

Async note (law 4): Client is async where the underlying contract is async; compile/exec
entry points are sync and internally bridge at the session boundary.

**The method table above is the public API in embryo — treated as such.** At the P3
release these names freeze into five languages' SDKs; renames after that are breaking
changes in every binding at once. Therefore: the table gets a dedicated naming review as
a P3 gate (before the release, after implementation experience), and until then every
Client method added in code must appear here first — spec leads, code follows.

## CLI (D13): one binary named `hologram`

Resolves today's conflict (two workspace binaries both named `hologram`:
`crates/hologram-cli` compute-side and `substrate/hologram-substrate-cli` node-side).
Both merge into `crates/hologram-cli`:

```
hologram
├─ compile   <source> [-o app.holo]        # was compute CLI
├─ run       <app.holo> [--space …]        # exec / boot per manifest
├─ app       inspect|compose|verify|pack   # .holo v3 tooling; pack --fat|--thin (03)
├─ store     put|get|pin|unpin|gc|ls|verify|manifest    # was substrate CLI
├─ net       fetch|announce|discover|peers
├─ network   create|join|delegate|show     # 04-networks (restricted/private tiers)
├─ space     list|tck                      # enumerate spaces, run conformance
├─ manager   roster|provision|open|configure|reconfigure
├─ node      serve                         # long-running peer (native space)
└─ bench …
```

Command logic stays store/space-generic and hermetically testable (a property today's
substrate CLI already has — preserve it). `anyhow` is permitted here and only here.

CLI defaults: the CLI runs over the **native space**; its local KappaStore lives in the
OS-appropriate data directory, overridable by flag/env. A store path is a *location*,
never an identity (law 2) — moving the directory changes nothing about content.

## FFI & SDKs (D6): hologram owns them, over Client

Target languages: **rust, c, python, typescript, swift** (kotlin later, same machinery).

| Language | Mechanism | Artifact |
|----------|-----------|----------|
| Rust | the facade itself | `hologram` crate |
| C | C-ABI cdylib + **cbindgen** header | `libhologram` + `hologram.h` |
| Python | **uniffi** over Client | `hologram` wheel |
| Swift | **uniffi** over Client | SwiftPM package |
| TypeScript (browser) | **wasm-bindgen** build of Client (wasm32) | npm package |
| TypeScript (node, optional) | **napi-rs** | npm package |

- The binding **surface** (types, methods, error enums exposed to foreign languages) is
  defined exactly once, in `crates/hologram-ffi` (cdylib + rlib, per-language features),
  replacing the current hand-rolled C ABI's direct crate composition with Client
  delegation.
- **SDK packaging composes surface + space, on the spaces/ side.** A shippable SDK needs
  a concrete space compiled in, and `hologram-ffi` (crates/) must not depend on
  `spaces/*` (dependency law — only the facade has that exception). So per-target
  packaging crates live with their space: the browser npm package is built from a thin
  packaging crate beside `holospaces-browser` (depends on hologram-ffi + the browser
  space — legal direction), the python wheel / SwiftPM package bundle the native space.
  Binding definitions never fork; packaging only selects the space.
- **FFI exposes Client over the platform's default space** (browser SDK → browser space,
  wheel/SwiftPM → native space). Bringing a *custom* space is a Rust-level affordance —
  foreign-language callers get a concrete, batteries-included Client.
- The wasm-bindgen glue inside `spaces/holospaces-browser` is **not** "the FFI" — it is
  that space's view/transport plumbing. The browser *SDK* (what an app developer imports
  from npm) is the packaging crate's wasm build described above.
- Binding conformance: one cross-language smoke suite (compile → provision → run → exit
  code) runs against every generated SDK **on every PR** — decided, not provisional. The
  CI cost (wasm + wheel + SwiftPM builds per PR) is accepted as the price of "a Client
  change that breaks a binding fails the same PR"; if minutes become a problem, the
  answer is caching/hardware, not demoting the gate to nightly.
- **FFI errors are typed, always.** `hologram-ffi` exposes only typed error enums
  (uniffi enums / C error codes) derived from Client's thiserror chains — no stringly
  errors, no anyhow anywhere in the FFI path. Consequence accepted deliberately: FFI
  error variants are **stable API** — adding one is minor, renaming/removing one is
  breaking in five languages. `anyhow` remains confined to `hologram-cli` alone.

## Runtime selection & platform plumbing (owners named)

- **Compute-target selection** (CPU/Metal/wgpu) is a **session/Client concern**, not a
  space concern (02 rejected spaces-subsume-compute): `Client` picks the best available
  kernel target for the platform by default; embedders override via builder config. The
  space never selects kernels.
- **Browser worker topology is the packaging crate's job.** The wasm compute pool is
  embedder-provided (worker registration, SharedArrayBuffer, COOP/COEP headers) — the
  browser SDK packaging crate owns spawning/registering workers and documents the
  required headers. App developers get a working pool by importing the npm package; they
  never hand-wire workers.
- **Observability ≠ audit.** Dev-facing diagnostics use the `tracing` ecosystem behind a
  facade-level subscriber seam (CLI pretty-prints, browser forwards to console, bare is
  silent by default). This is throwaway telemetry — the κ-chained audit trail
  (07 R2) is a separate, append-only thing; neither substitutes for the other.
- **Browser SDK size budget**: the npm package's wasm binary gets a CI-tracked size
  budget from P3 (regressions fail loudly; the number is set when the first real build
  exists, then defended).

## DX commitments

- `cargo add hologram` + one feature flag is a working embedder; `cargo install
  hologram-cli` (or platform installers) is a working operator.
- `hologram space tck` gives space implementors a one-command conformance verdict —
  the porting story for holospaces-ios/-esp32 is: implement the contract, run this.
- Errors follow law 7: typed thiserror chains through Client, rendered with context at
  the CLI/SDK edge; never a stringly-typed error across the FFI boundary (uniffi enums).
