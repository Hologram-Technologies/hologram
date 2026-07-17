//! # hologram-net — the uor-native network layer (SPINE-4 `KappaSync`)
//!
//! Consolidates the substrate's three network crates (P1) as feature-gated modules:
//!
//! - [`bare`] — `no_std` frame protocol over the `NetworkInterface` HAL (`BareNetSync`):
//!   `fetch` / `announce` / `discover` with verify-on-receipt. Always available.
//! - [`http`] — HTTP-CAS gateway protocol (spec §6.3). `http::live` (feature `live`) is
//!   the live HTTP/1.1 transport over `std::net`.
//! - `tcp` (feature `tcp`) — κ-XOR Kademlia DHT over TCP + tokio, for std hosts.
//!
//! κ is the only identity everywhere (SPINE-1): no PeerIds, no Multiaddrs. Per spec 04
//! the protocol core lives here; the per-space transport pumps move to `spaces/` in P2.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bare;
pub mod http;
/// Wire-protocol version negotiation (spec 04 §Protocol hardening) — portable, transport-agnostic.
pub mod protocol;
#[cfg(feature = "tcp")]
pub mod tcp;
