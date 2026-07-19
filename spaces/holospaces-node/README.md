# holospaces-node

> holospaces' bare-metal / edge peer — the egress exit and storage-sync node a browser tab routes through.

holospaces-node is a Space implementation: a first-class holospaces peer for low-powered devices. holospaces keeps a devcontainer's *compute* in the browser (the emulator boots a real OS in the tab), but a tab has no NIC and no durable storage. A flashed node supplies both — a device you own, not a bespoke external proxy — and OTA-updates itself from the holospaces GitHub Pages site. It is plain `std`, so it cross-compiles to small Linux SBCs.

## What it provides

Both a `holospaces-node` binary and a `holospaces_node` library:

- `egress` / `EgressServer` — the exit node: it forwards the browser guest's arbitrary TCP to the real internet (`apt`/`pip`/`npm`, `git` clone, an outbound socket), speaking the same `OPEN`/`DATA`/`CLOSE` egress framing the browser's `WsEgress` already uses (CC-16).
- `serve_connection` — serve one browser peer over its egress WebSocket: complete the `tungstenite` handshake, then shuttle guest frames to an `EgressServer` and frame the host's replies back, with bounded blocking I/O (one connection per node thread) suited to a low-powered device.
- `storage` — a persistent, file-backed `KappaStore` (`hologram-store` `native`) served over the substrate's HTTP-CAS protocol (`GET /cas/{κ}`), so operator content survives browser reloads and devices and is fetched trustlessly by verify-on-receipt (CC-20 / CC-38).
- `ota` — OTA from GitHub Pages: the node fetches its own κ-addressed updates over `ureq`, verifies them by re-derivation (Law L5 — a forged update is refused), and stages them for the next restart.

The `no_std` content-network core it shares with the browser (`holospaces::content_net`, CC-38) is what lets the smallest microcontroller variants participate too.

Part of the [hologram](../../README.md) workspace.
