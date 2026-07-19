# holospaces-browser

> The Hologram Platform Manager — holospaces' wasm32 browser peer, served from GitHub Pages.

holospaces-browser is a Space implementation: the wasm-bindgen surface over holospaces' Manager model, realizing the *Platform Manager* (arc42 chapter 5) and the *Browser* peer (arc42 chapter 7). Loading the κ-addressed bundle makes the browser a peer that *is* the substrate — there is no server (Law L1). It composes a full browser peer: an in-memory `MemKappaStore` (RAM as a cache, Law L3), an OPFS-backed κ-disk, and hologram's interpreter `ContainerEngine` (wasmi, which runs in wasm32 where a JIT cannot).

## What it provides

A `#[wasm_bindgen]` console surface over the Manager model, exposing:

- κ helpers — `kappa` (the content address every peer computes, Law L1) and `verify_kappa` (verify by re-derivation, Law L5 — what makes gateway-fetched content safe).
- The operator console — sign in, provision (both compute forms), view, resolve, the operator roster (R5), the browser `.holo` engine (CC-2), booting a userland container in-browser through the substrate runtime (CC-6), and importing/running a devcontainer with no Docker daemon and no cloud VM. The same holospace κ boots here as on a native or remote peer (Q6).
- `Workspace` — launch a holospace whose code is the system emulator to boot a real OS in the tab (CC-9) and drive it through the workspace projection (CC-11): a live terminal whose commands are canonical events advancing the holospace's κ snapshot, and an editor reading/editing environment content by κ.
- `WebRtcLink` (`webrtc`) — the CC-49 peer-to-peer WebRTC data channel carrying `BareNetSync` frames between browser peers, and `wsnet`, the CC-16 WebSocket egress transport tunnelling the wasm userspace TCP/IP NAT's streams out to a relay (ADR-014).

## Targets & build notes

wasm32-only (web-sys / wasm-bindgen). Excluded from the host workspace via its own empty `[workspace]` table, so it builds standalone regardless of where the checkout lives. Built via `scripts/browser-manager-test.sh` and the Pages deploy (wasm-pack + `wasm-opt -O3`, tuned for the interpreter hot path with `lto`, one codegen unit, and `panic = "abort"`). The wasm output keeps the name `holospaces_web` for compatibility with the existing web assets. It consumes the OPFS κ-disk backend from `hologram-store`'s `opfs` feature (`default-features = false`, so no duplicate `wasm-bindgen`).

Part of the [hologram](../../README.md) workspace.
