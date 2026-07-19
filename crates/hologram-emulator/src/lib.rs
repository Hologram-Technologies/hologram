//! Hologram system emulator — deterministic RISC-V / x86-64 / aarch64 cores that
//! boot an OS *on the substrate* (ADR-009). Hoisted out of holospaces so system
//! emulation is a first-class hologram capability; depends only on the hologram-space
//! contract (KappaStore / MemKappaStore).
//!
//! `no_std` + `alloc` by default (the portable emulator core the browser and
//! bare-metal peers compile). The `std` feature (on by default) turns on the
//! host-only surfaces the cores expose behind `#[cfg(feature = "std")]` — the
//! native NAT egress/ingress (`std::net`) and the x86-64 boot trace.
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unsafe_code)]

extern crate alloc;

pub mod emulator;
pub mod machine;

// The hologram Wasm **container codemodule** (ADR-009 execution surface), absorbed from the former
// `hologram-emulator-codemodule` crate. Compiled only for the `codemodule` wasm32 `cdylib` build
// (`scripts/build-emulator.sh`): its `#[panic_handler]` / `#[global_allocator]` must never enter a
// normal (std) lib link, so it is gated to `feature = "codemodule"` on `wasm32`.
#[cfg(all(feature = "codemodule", target_arch = "wasm32"))]
mod codemodule;

pub use emulator::Arch;
