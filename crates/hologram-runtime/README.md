# hologram-runtime

> Substrate-portable Container Runtime orchestration: lifecycle, snapshot-as-κ, and capability enforcement over a `ContainerEngine` seam and a `KappaStore`.

`hologram-runtime` holds the uor-native parts of the Container Runtime (spec §4): container
identity, the spawn / suspend / resume / terminate lifecycle, snapshots as κ-labels, and
**capability enforcement** (the `admits` containment at delegation). It runs over two seams — a
`ContainerEngine` (the Wasm instance; Wasmtime or interpreter is a backend) and a `KappaStore` — and
links nothing from the tensor compute engine (RZ).

The orchestration is engine-agnostic, so it is validated hermetically against a mock engine and then
drives a real Wasmtime engine unchanged (substrate-tripling at the runtime). It carries a channel bus
with persistent subscriptions, a DRR event scheduler for per-container fairness, error-log chains
compacted into `ChainCompaction` κ-labels past a tunable depth, and an optional `KappaSync` network
layer wired via `with_sync(...)` for container-driven fetch / announce.

## What it provides

- `Runtime<E, S>` — the substrate-portable runtime over an engine `E: ContainerEngine` and store `S: KappaStore`.
- `lifecycle::{Session, Phase, LifecycleError}` — the generic, space-agnostic session driving boot → suspend-to-κ-snapshot → resume → migrate → terminate.
- `ContainerEngine` / `ContainerIntents` / `HostContext` — re-exported from `hologram_space::engine`, the seam engine backends implement.
- `engine_wasmtime::{WasmtimeEngine, WasmBlockDevice, WasmNetworkInterface}` — the std Wasmtime backend (feature `engine-wasmtime`).
- `engine_wasmi` module — the `no_std` wasmi-interpreter backend (feature `engine-wasmi`).
- `FUEL_PER_MS`, `DEFAULT_ERROR_LOG_THRESHOLD` — the fuel↔CPU-time calibration and default error-log compaction depth.

## Features

- `std` (default) — enables `std` in `hologram-space`. Without it the crate is `no_std` + `alloc`.
- `engine-wasmtime` — the Wasmtime engine backend (spec §7.2 Option A); implies `std`, pulls `wasmtime` + ChaCha20 entropy (`rand_chacha` / `rand_core`).
- `engine-wasmi` — the bare-metal wasmi interpreter backend (architecture §2, C1); `no_std`, pulls `wasmi`.

## Targets & build notes

`no_std` + `alloc` when `std` is off, so the wasmi backend runs on bare-metal / embedded targets
(e.g. `thumbv7em` — the runtime uses a `spin::Mutex` counter rather than `AtomicU64` because Cortex-M
has no 64-bit atomics). `async-trait`'s boxed futures bring `Box` into `no_std` scope. The
`engine-wasmtime` examples (`cas_artifact_cache`, `event_bus`, `least_privilege`, `live_migration`,
`wasm_inference_container`) build only with that engine present.

Part of the [hologram](../../README.md) workspace.
