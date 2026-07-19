# hologram-space

> The trait surface a host implements to become a place where hologram executes.

`hologram-space` defines the **space contract** (`specs/refactor/02-space-contract.md`): the
[`Space`] aggregate names a platform's concrete parts behind associated types, so everything
downstream is generic over `Space` and monomorphized per platform. A space is a *place* hologram
runs — native host, browser Worker, or bare-metal peer — and it wires together seven spec-02 parts:
a synchronous [`KappaStore`], the maybe-Send [`KappaSync`] network seam, the composed
[`ContainerRuntime`], the [`Entropy`] / [`Clock`] / [`Spawner`] HAL seams, and the maybe-Send
[`Surface`] presentation seam.

This crate also carries the portable σ-axis core absorbed from the former
`hologram-substrate-core` and `hologram-realizations` crates: κ-addressing (BLAKE3 content
addresses, 71-byte `blake3:<hex>` labels), verify-by-re-derivation, the container-engine seam, and
the canonical realization forms. Storage is synchronous (wasm-safe: the browser reaches persistent
storage through a synchronous OPFS handle inside a Worker); only the network/boot seam is async, and
its `Send` bound is *maybe-Send* — `Send` on native, `?Send` on `wasm32`/bare (LAW-4).

## What it provides

- `Space` — the aggregate trait naming a platform's `Store` / `Sync` / `Runtime` / `Entropy` / `Clock` / `Spawner` / `Surface`.
- `KappaStore`, `KappaSync`, `ContainerRuntime` — the storage, network, and lifecycle seams (re-exported from `substrate`).
- `address_bytes` / `verify_kappa` — σ-axis κ derivation and verify-on-receipt helpers.
- `engine::{ContainerEngine, ContainerIntents, HostContext}` — the Wasm-instance seam the runtime and its backends implement.
- `MemKappaStore` — the reference in-memory `KappaStore` (no_std + alloc, hashbrown + spin).
- Realization forms — `ContainerManifest`, `CapabilitySet`, `Snapshot`, and related canonical types.
- `hal::{BlockDevice, Clock, Entropy, Spawner, NetworkInterface, ManualClock, SeededEntropy, RamBlockDevice, NoopSpawner, …}` — the HAL seams and portable reference impls.
- `surface::{Surface, Intent, NullSurface, SurfaceError}` — the presentation / interaction seam.

## Features

- `std` (default) — enables `std` in `uor-addr` and `hologram-types`. Without it the crate is `no_std` + `alloc`.

## Targets & build notes

`no_std` + `alloc` when `std` is off; portable to `wasm32` and `thumbv7em` (Cortex-M). The core
carries only portable traits + signature/cipher bytes — the reference `SignatureVerifier`,
`PayloadCipher`, and `KeyWrapper` (ed25519-dalek / chacha20poly1305 / x25519-dalek) are
**dev-dependencies only**, so portability builds never pull curve25519 or a real cipher; a concrete
space provides its own (hardware / WebCrypto).

Part of the [hologram](../../README.md) workspace.
