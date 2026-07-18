//! # `hologram_store::opfs` — browser OPFS `KappaStore`
//!
//! The **browser OPFS store**: κ→bytes persisted in the Origin Private File System, keyed
//! by hologram's σ-axis κ-label — so a κ minted in the browser is byte-identical to one minted on
//! native/bare-metal (substrate-tripling). OPFS exposes two access regimes, and this crate owns a
//! store for each:
//!
//! - [`OpfsKappaStore`] ([`sync_store`]) — the **in-product** [`KappaStore`](hologram_space::KappaStore)
//!   backend: a single append-only OPFS pack file + an in-RAM offset index, driven synchronously
//!   through a `FileSystemSyncAccessHandle` inside a Worker (where the emulator's κ-disk runs).
//!   Always compiled; no JS bindings — this is what a `Space` consumes.
//! - The [`js_api`] layer + SAB [`bridge`] — an **async, file-per-κ** persistence + GC reference
//!   with `#[wasm_bindgen]` exports, verified end-to-end in a real Chromium via Playwright. Gated
//!   behind the default `js-api` feature (a consumer that only wants the backend takes
//!   `default-features = false` and pulls no `wasm-bindgen`).

/// The in-product synchronous OPFS `KappaStore` backend (Worker; pack file + offset index).
pub mod sync_store;
pub use sync_store::OpfsKappaStore;

/// The async, file-per-κ OPFS persistence + GC layer (`#[wasm_bindgen]` JS API).
#[cfg(feature = "js-api")]
pub mod js_api;
#[cfg(feature = "js-api")]
pub use js_api::*;

/// The main-thread↔Worker SharedArrayBuffer bridge that exposes a sync `KappaStore` on the
/// main thread by driving the [`js_api`] async functions in a paired Worker (architecture G-C2).
#[cfg(feature = "js-api")]
pub mod bridge;
