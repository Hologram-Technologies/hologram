# 01 — Target Crate Map

Decisions: D1, D3, D4, D5, D7, D15, D16, D18 (see `00-overview.md`).

## Target layout

```
crates/
  hologram-types      # + absorbs hologram-host (type vocab, dtypes, shapes, κ re-exports)
  hologram-ops        # closed op catalog (64 ops, term emitters, reference evaluators)
  hologram-graph      # arena DAG IR + schedules
  hologram-archive    # .holo v3 codec: tensor plans + app manifests (03-holo-format.md)
  hologram-compute    # was hologram-backend: CPU/Metal/wgpu tensor kernels
  hologram-exec       # sync execution hot path (InferenceSession, BufferArena, warm-start)
  hologram-compiler   # graph → .holo (+ python/rust/typescript frontends)
  hologram-space      # THE CONTRACT (02-space-contract.md)
  hologram-runtime    # orchestration + Peer/Session/Manager model
                      #   features: engine-wasmtime, engine-wasmi
  hologram-net        # SPINE-4 protocol + κ-XOR DHT, no_std core (04-networks.md)
  hologram-tck        # conformance battery + reference in-memory store
  hologram-ffi        # C ABI + uniffi + wasm-bindgen/napi-rs over Client (05-tooling.md)
  hologram-cli        # the ONE `hologram` binary
  hologram-bench      # criterion benchmarks
  hologram            # facade + Client (feature-gated re-exports; only public entry)
spaces/
  holospaces          # portable core: system emulators (RISC-V/AArch64/x86-64), κ-disk,
                      #   boot/OCI/devcontainer provisioning, projection/surface
  holospaces-browser  # OPFS store, WebRTC/WebSocket transport pumps, wasm-bindgen views
  holospaces-native   # redb store, wasmtime engine selection, TCP + iroh transports
  holospaces-bare     # block-device store, wasmi engine selection, bare net (esp32 seed)
```

Dependency direction (law): `spaces/* → crates/*`, never the reverse. Within `crates/`,
the existing kernel → bridge → user layering is preserved.

## Per-crate charter (one axis of change each)

| Crate | Charter | Changes when… |
|-------|---------|---------------|
| hologram-types | Type vocabulary: DType, shapes, Digest/κ re-exports, host bounds/hash axis (absorbed from hologram-host) | the type vocabulary or hash-axis selection changes |
| hologram-ops | The closed op catalog: markers, IRIs, term emitters, reference evaluators | an op is added/changed in the catalog |
| hologram-graph | Graph IR, schedules, registries | the IR or scheduling model changes |
| hologram-archive | `.holo` v3 codec: sections, layers, manifests, certificates, footer | the archive format changes |
| hologram-compute | Kernel implementations per target (CPU SIMD/parallel, Metal, wgpu) | a kernel or target is added/optimized |
| hologram-exec | Content-addressed sync execution, buffer arena, κ-memo, warm-start | the execution strategy changes |
| hologram-compiler | Lowering, validation, caching, source frontends | compilation pipeline or a frontend changes |
| hologram-space | Space contract traits + realizations + HAL + SPINE laws | the contract every space implements changes |
| hologram-runtime | Container orchestration, lifecycle (Peer/Session), platform-manager model (Manager/Operator/Roster/Configuration), engines behind features | lifecycle/orchestration semantics change |
| hologram-net | uor-native wire protocol, framing, κ-XOR DHT logic (transport-free) | the network protocol changes |
| hologram-tck | The conformance definition of "a valid space"; reference mem store | conformance requirements change |
| hologram-ffi | Foreign-language bindings over Client | a binding surface changes |
| hologram-cli | Command-line shell over Client | a command changes |
| hologram-bench | Performance floors and regressions | benchmarks change |
| hologram | Facade: feature-gated re-exports + the `Client` type | the public API surface changes |
| holospaces | Space-agnostic emulation & provisioning machinery shared by all space impls | emulator/boot machinery changes |
| holospaces-{browser,native,bare} | One full space stack each: store + engine selection + transport + views | that platform changes |

## Source → target mapping

### From `crates/` (current)

| Current | Target | Notes |
|---------|--------|-------|
| crates/hologram-host | **merged into** crates/hologram-types | 83 LOC of type aliases + bounds; not worth a crate boundary |
| crates/hologram-types | crates/hologram-types | unchanged role |
| crates/hologram-ops | crates/hologram-ops | unchanged |
| crates/hologram-graph | crates/hologram-graph | unchanged |
| crates/hologram-archive | crates/hologram-archive | extended to v3 (03) |
| crates/hologram-backend | **renamed** crates/hologram-compute | the word "backend" is retired (D3) |
| crates/hologram-exec | crates/hologram-exec | unchanged role |
| crates/hologram-compiler | crates/hologram-compiler | unchanged role |
| crates/hologram-cli | **merged into** crates/hologram-cli (unified binary) | absorbs substrate CLI subcommands |
| crates/hologram-ffi | crates/hologram-ffi | rebuilt over Client (05) |
| crates/hologram-bench | crates/hologram-bench | unchanged |
| (root) hologram facade | crates/hologram | + gains `Client` |

### From `substrate/` (dissolved entirely)

| Current | Target | Notes |
|---------|--------|-------|
| substrate/hologram-substrate-core | crates/hologram-space | trait surfaces: KappaStore, KappaSync, ContainerRuntime, errors, verify_kappa |
| substrate/hologram-realizations | crates/hologram-space | canonical forms (ContainerManifest, CapabilitySet, Snapshot, …) |
| substrate/hologram-bare-hal | crates/hologram-space | BlockDevice, NetworkInterface HAL traits |
| substrate/hologram-substrate-tck | crates/hologram-tck | plus reference mem store |
| substrate/hologram-store-mem | crates/hologram-tck | becomes the reference/conformance store |
| substrate/hologram-store-native | spaces/holospaces-native | redb store |
| substrate/hologram-store-bare | spaces/holospaces-bare | block-device store |
| substrate/hologram-store-opfs | spaces/holospaces-browser | OPFS store (merges with holospaces-web's OpfsKappaStore — dedupe the two OPFS impls during P2) |
| substrate/hologram-runtime | crates/hologram-runtime | orchestration core |
| substrate/hologram-runtime-wasmtime | crates/hologram-runtime, feature `engine-wasmtime` | std-only feature |
| substrate/hologram-runtime-bare | crates/hologram-runtime, feature `engine-wasmi` | no_std-capable feature |
| substrate/hologram-net-http | crates/hologram-net (protocol) | `live` transport parts → holospaces-native |
| substrate/hologram-net-tcp | crates/hologram-net (DHT/wire) + spaces/holospaces-native (TCP transport) | protocol/transport split |
| substrate/hologram-net-bare | crates/hologram-net (shared frames) + spaces/holospaces-bare (pump) | |
| substrate/hologram-substrate-cli | **merged into** crates/hologram-cli | resolves the two-binaries-named-`hologram` conflict |
| substrate/hologram-efi | spaces/holospaces-bare (excluded target-specific build) | UEFI boot binary |

### From `../holospaces` (merged in as `spaces/`)

| Current | Target | Notes |
|---------|--------|-------|
| holospaces/crates/holospaces — Peer, Session (boot.rs), Manager, Operator/Roster (identity.rs), Configuration (config.rs) | **hoisted to** crates/hologram-runtime | D7: the model is space-agnostic workload-lifecycle management |
| holospaces/crates/holospaces — emulators (emulator.rs, aarch64.rs, x64.rs, devbus.rs), κ-disk (disk.rs), content_net glue (PacketLink, TransportEndpoint), boot/OCI/devcontainer/compose/dockerfile provisioning, projection.rs (surface) | spaces/holospaces | portable core, no_std + alloc |
| holospaces/crates/holospaces-web | spaces/holospaces-browser | OPFS store, webrtc.rs/wsnet.rs pumps, wasm-bindgen Console/Workspace |
| holospaces/crates/holospaces-emulator | spaces/holospaces (codemodule build target) | κ-addressed Wasm codemodule wrapper |
| holospaces vv/ (CC catalog, QEMU oracles, Playwright) | this repo's CI | absorbed in P2; must stay green (00 §success criteria) |
| holospaces docs (arc42, C4, OPM, 15288) | specs/ (namespaced, e.g. specs/holospaces/) | history preserved; ADR numbering continues |

All git-pinned `hologram-*` deps in holospaces (rev `18f553d…`) become workspace path deps.

## Facade feature matrix

`hologram` is the only crate users name. Features map 1:1 to internal crates:

```toml
[dependencies]
hologram = { version = "X.Y", default-features = false, features = ["space", "runtime"] }
```

| Feature | Re-exports | Pulls in |
|---------|-----------|----------|
| `types` (always on) | `hologram::types` | hologram-types |
| `ops` | `hologram::ops` | hologram-ops |
| `graph` | `hologram::graph` | hologram-graph |
| `archive` | `hologram::archive` | hologram-archive |
| `compute` | `hologram::compute` | hologram-compute |
| `exec` | `hologram::exec` | hologram-exec |
| `compiler` | `hologram::compiler` | hologram-compiler |
| `space` | `hologram::space` (Space contract, KappaStore, KappaSync, HAL, realizations) | hologram-space |
| `runtime` | `hologram::runtime` (Peer, Session, Manager) | hologram-runtime |
| `engine-wasmtime` / `engine-wasmi` | engine selection | hologram-runtime features |
| `net` | `hologram::net` | hologram-net |
| `tck` | `hologram::tck` | hologram-tck |
| `client` | `hologram::Client` | facade-level composition |
| `full` | everything above | |

Space implementation crates (`holospaces-*`) are also re-exported behind features
(`space-browser`, `space-native`, `space-bare`) so an embedder writes, e.g.:

```rust
use hologram::space::Space;
use hologram::spaces::native::NativeSpace;
```

## Publishing & versioning (D16)

- Every workspace crate publishes to crates.io.
- **Lockstep version** via `workspace.package.version`; one release = one version across
  all crates. `hologram-ai` and third parties depend on `hologram = "X.Y"` + features.
- First publishable release lands at the end of migration phase P3 (see `06-migration.md`).
- UOR crates (uor-foundation, uor-foundation-sdk, uor-prism, uor-prism-tensor, uor-addr)
  remain **external crates.io dependencies** (D18). hologram re-exports what its API needs
  (e.g. κ/Digest types through `hologram::types`); the long-term aim is that consumers
  never add a `uor-*` dependency directly.

## Workspace hygiene (law 7 applied)

- `[workspace.lints]`: `unsafe_code = "forbid"` outside hologram-compute SIMD and
  hologram-ffi (which get crate-level allowances with documented `# Safety` contracts);
  clippy pedantic baseline agreed during P1; `missing_docs = "warn"` → `"deny"` by P3.
- All error types via `thiserror` in libraries; `anyhow` only inside hologram-cli.
- `no_std + alloc` posture preserved where it exists today (types, ops, graph, archive,
  space, net, tck core, holospaces portable core); std-only code isolated behind features.
