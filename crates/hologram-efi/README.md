# hologram-efi

> The bare-metal UEFI boot binary that brings up the engine over a block device.

`hologram.efi` (bare-metal spec §3) is a `no_std`, `no_main` UEFI application booted directly
by firmware. It re-derives its embedded driver κ-graph anchors (measured boot), probes the
hardware it is bound to, and exercises a `BareMetalKappaStore` end to end before shutting
down. It builds against `hologram-space` and `hologram-store` (the `bare` backend), with a
pure-Rust BLAKE3 so the σ-axis links into a PE/UEFI image.

## What it provides

The `hologram` binary (`src/main.rs`) runs one boot self-test and prints `HOLOGRAM-BM: PASS`
(or `FAIL`) to the UEFI console:

- **Measured-boot anchors** (arch §12.6 + E2) — the block-device and NIC driver κs are re-derived from the embedded bytes and compared to the κs `build.rs` recorded; a post-build tamper is caught here.
- **Hardware probing** (TR §10.17 + E1) — enumerates UEFI `BlockIO` (and, under `probe-nics`, `SimpleNetwork`) handles and mints a `HardwareInventory` κ summarizing what is bound.
- **Storage self-check** over `BareMetalKappaStore` — put → get → `verify_kappa` (SPINE-4), reachability GC (a pinned manifest keeps its operands, an orphan is evicted), and reboot persistence with a bumped `reboot_epoch` (G-C1).

`build.rs` compiles real WAT → Wasm drivers and records their BLAKE3 κ, so the boot-time
`verify_kappa` step is a meaningful measured-boot check rather than a tautology.

## Features

- `probe-nics` (off) — enumerate UEFI `SimpleNetwork` handles alongside block devices. Off by default because the protocol type is not exposed in every uefi-0.35 feature combination; a build with it absent still boots and simply reports zero NICs.

## Targets & build notes

Built for `x86_64-unknown-uefi` and excluded from the host workspace (a `no_std` PE target
with its own `panic = "abort"` profiles). Booted in QEMU/OVMF; the boot test asserts on the
`HOLOGRAM-BM: PASS` console line.

Part of the [hologram](../../README.md) workspace.
