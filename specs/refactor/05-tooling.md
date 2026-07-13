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

let app     = client.compile(source)?;            // → .holo (v3)
let holo    = client.provision(app, caps)?;       // ingest → κ
let session = client.open(&holo)?;                // Session (runtime lifecycle)
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

## CLI (D13): one binary named `hologram`

Resolves today's conflict (two workspace binaries both named `hologram`:
`crates/hologram-cli` compute-side and `substrate/hologram-substrate-cli` node-side).
Both merge into `crates/hologram-cli`:

```
hologram
├─ compile   <source> [-o app.holo]        # was compute CLI
├─ run       <app.holo> [--space …]        # exec / boot per manifest
├─ app       inspect|compose|verify        # .holo v3 tooling
├─ store     put|get|pin|unpin|gc|ls|verify|manifest    # was substrate CLI
├─ net       fetch|announce|discover|peers
├─ network   create|join|delegate|show     # 04-networks
├─ space     list|tck                      # enumerate spaces, run conformance
├─ manager   roster|provision|open|configure|reconfigure
├─ node      serve                         # long-running peer (native space)
└─ bench …
```

Command logic stays store/space-generic and hermetically testable (a property today's
substrate CLI already has — preserve it). `anyhow` is permitted here and only here.

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

- All live in `crates/hologram-ffi` (cdylib + rlib, per-language features), replacing the
  current hand-rolled C ABI's direct crate composition with Client delegation.
- The wasm-bindgen glue inside `spaces/holospaces-browser` is **not** "the FFI" — it is
  that space's view/transport plumbing. The browser *SDK* (what an app developer imports
  from npm) comes from hologram-ffi's wasm build.
- Binding conformance: one cross-language smoke suite (compile → provision → run → exit
  code) runs against every generated SDK in CI, so a Client change that breaks a binding
  fails the same PR.

## DX commitments

- `cargo add hologram` + one feature flag is a working embedder; `cargo install
  hologram-cli` (or platform installers) is a working operator.
- `hologram space tck` gives space implementors a one-command conformance verdict —
  the porting story for holospaces-ios/-esp32 is: implement the contract, run this.
- Errors follow law 7: typed thiserror chains through Client, rendered with context at
  the CLI/SDK edge; never a stringly-typed error across the FFI boundary (uniffi enums).
