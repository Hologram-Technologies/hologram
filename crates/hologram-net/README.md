# hologram-net

> The uor-native network layer (SPINE-4 `KappaSync`): bare frame protocol, HTTP-CAS gateway, and κ-XOR DHT over TCP or QUIC.

`hologram-net` consolidates the substrate's three network crates as feature-gated modules over the
one `KappaSync` seam. κ is the only identity everywhere (SPINE-1): no PeerIds, no Multiaddrs, no
second naming surface — every peer is the κ-label of its `PeerEndpoint` realization, every routed
key is a κ-label, and every fetched byte is σ-axis re-derived on receipt (SPINE-4,
verify-on-receipt). Per spec 04 the protocol core lives here; the per-space transport pumps move to
`spaces/` in P2.

## What it provides

- `bare` module — `BareNetSync`, a `no_std` `KappaSync` over the `NetworkInterface` HAL: `fetch` / `announce` / `discover` with a minimal length-prefixed frame codec (`u32 LE len | u8 kind | payload`) and a boot-populated peer table. No filesystem, no OS sockets. Always available.
- `http` module — the HTTP-CAS gateway protocol (spec §6.3): `cas_path` / `parse_cas_path`, `CasResponse`, mapping `GET /cas/{kappa}` to/from a `KappaStore`. Pure and hermetically testable. `http::live` (feature `live`) is the live HTTP/1.1 transport over `std::net` (`HttpKappaSync`), a thin `TcpListener` server + verifying client.
- `protocol` module — portable, transport-agnostic wire-protocol version negotiation (spec 04 §Protocol hardening).
- `tcp` module (feature `tcp`) — `KappaSync` over a κ-XOR Kademlia DHT on raw TCP + tokio, sharing the bare frame shape. std hosts only.
- `quic` module (feature `quic`) — `QuicPeer`, encrypted P2P over QUIC (quinn / rustls-ring), carrying the same frame protocol + wire-version handshake. std hosts only.

## Features

- `std` (default) — enables `std` in `hologram-space`. Without it the crate is `no_std` + `alloc` (the `bare` module).
- `live` — the live HTTP/1.1 CAS transport over `std::net` (`http::live`); implies `std`.
- `tcp` — the κ-XOR Kademlia DHT over TCP + tokio (`tcp`); implies `std`, pulls `tokio`.
- `quic` — encrypted P2P over QUIC (`quic`): quinn + rustls(ring) + a self-signed rcgen transport cert; implies `std` + tokio, pulls `quinn` / `rustls` / `rcgen`.

## Targets & build notes

The `bare` and `http` protocol layers are `no_std` + `alloc` and portable to `wasm32` / `thumbv7em`;
`live`, `tcp`, and `quic` are std-host transports. TLS gives QUIC an encrypted channel while content
integrity stays κ-verified. quinn is pinned to the ring crypto backend and carries **no blake3**, so
it does not conflict with the κ core's `blake3 1.5` pin — it is the dependency-light P2P substrate
iroh would layer relay / NAT-traversal onto (spec 04).

Part of the [hologram](../../README.md) workspace.
