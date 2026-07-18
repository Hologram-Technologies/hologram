# hologram-emulator

> Deterministic RISC-V / x86-64 / aarch64 cores that boot an OS on the substrate.

The Hologram system emulator (ADR-009). Hoisted out of holospaces so system emulation is a
first-class hologram capability, it depends only on the `hologram-space` contract
(`KappaStore` / `MemKappaStore`). The core is `no_std` + `alloc` by default — the portable
build the browser and bare-metal peers compile — and the `std` feature (on by default) turns
on host-only surfaces the cores expose: native NAT egress/ingress over `std::net` and the
x86-64 boot trace.

## What it provides

- `Arch` — the target enum (RISC-V / x86-64 / aarch64) with OCI-arch, id, and label mappings.
- `emulator::Emulator` — the deterministic core: step/run, snapshot and restore, halt/trap handling.
- `machine::MachineSpec` — machine descriptors (device tree, `boot`, workspace and net boot variants) that stand up a bootable guest.
- The `codemodule` build (feature-gated) — the emulator core compiled to a hologram Wasm container, exporting the `hg_*` container ABI and importing only the `hologram` host ABI (`storage_put` / `storage_get`). It runs in a **batch** profile (a flat RISC-V image run to completion) or an **operating system** profile (a `HGOS` boot descriptor whose kernel and device-tree κ are read back from the substrate).

## Features

- `std` (default) — host-only surfaces: native NAT over `std::net` and the x86-64 boot trace.
- `cc44-trace` (off; dev-only) — streams the x86-64 guest serial console live to stderr and samples `rip`/TSC for CC-44 long-mode bring-up. Never enabled in CI or the default build.
- `codemodule` (off) — compiles the emulator to a `no_std` wasm32 `cdylib` container (pulls in `dlmalloc` for the guest heap). Absorbs the former `hologram-emulator-codemodule` crate; the module lives behind `cfg(all(feature = "codemodule", target_arch = "wasm32"))` so its `#[panic_handler]` / `#[global_allocator]` never enter a normal std lib link.

## Targets & build notes

`no_std` + `alloc` core; the default `crate-type` stays `lib` so ordinary builds produce no
cdylib. The Wasm container is built out-of-band by `scripts/build-emulator.sh`, which runs
`cargo rustc -p hologram-emulator --target wasm32-unknown-unknown --no-default-features
--features codemodule --release --crate-type cdylib`, emitting
`target/wasm32-unknown-unknown/release/hologram_emulator.wasm`.

Part of the [hologram](../../README.md) workspace.
