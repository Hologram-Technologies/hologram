//! Hologram runtime executor (spec Part VIII).
//!
//! `no_std` + `alloc` by default (matching prism / uor-addr) so inference
//! runs in wasm and on embedded targets; the `std` feature and the `async`
//! feature (which implies `std`) add host-only amenities.

#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

pub mod buffer;
#[cfg(feature = "tiered-exec")]
pub mod coherence;
pub mod error;
pub mod prism_route;
pub mod session;
pub mod warm;
#[cfg(feature = "tiered-exec")]
pub use coherence::{DeviceOwner, LevelMigration, SlotCoherence, TierPolicy, TierReport};

pub use buffer::{BufferArena, InputBuffer, OutputBuffer, SlotSpan};
pub use error::ExecError;
pub use prism_route::AttestedExecution;
pub use session::{InferenceSession, SessionBackend};
#[cfg(feature = "std")]
pub use warm::FileWarmStore;
pub use warm::{fold_archive, MemWarmStore, WarmStore};
