//! Root facade for the Hologram workspace.
//!
//! The implementation crates remain independently consumable, but applications
//! can depend on `hologram` and opt into the public surfaces they need:
//!
//! ```toml
//! [dependencies]
//! hologram = {
//!     git = "https://github.com/Hologram-Technologies/hologram",
//!     features = ["backend", "compiler", "exec"],
//! }
//! ```
//!
//! Use the `full` feature to expose every primary crate facade under `crates/`.

#![cfg_attr(not(feature = "std"), no_std)]

// The `client` facade composes over `alloc` collections (κ lists, output buffers) so it
// links on the no_std browser / bare-metal targets too.
#[cfg(feature = "client")]
extern crate alloc;

#[cfg(feature = "archive")]
pub mod archive {
    //! Facade for the `hologram-archive` crate.

    pub use hologram_archive::*;
}

#[cfg(feature = "backend")]
pub mod backend {
    //! Facade for the `hologram-compute` crate.

    pub use hologram_compute::*;
}

#[cfg(feature = "bench")]
pub mod bench {
    //! Facade for the `hologram-bench` crate.

    #[allow(unused_imports)]
    pub use hologram_bench::*;
}

#[cfg(feature = "cli")]
pub mod cli {
    //! Facade for the `hologram-cli` crate.

    pub use hologram_cli::*;
}

#[cfg(feature = "compiler")]
pub mod compiler {
    //! Facade for the `hologram-compiler` crate.

    pub use hologram_compiler::*;
}

#[cfg(feature = "exec")]
pub mod exec {
    //! Facade for the `hologram-exec` crate.

    pub use hologram_exec::*;
}

#[cfg(feature = "ffi")]
pub mod ffi {
    //! Facade for the `hologram-ffi` crate.

    pub use hologram_ffi::*;
}

#[cfg(feature = "graph")]
pub mod graph {
    //! Facade for the `hologram-graph` crate.

    pub use hologram_graph::*;
}

#[cfg(feature = "ops")]
pub mod ops {
    //! Facade for the `hologram-ops` crate.

    pub use hologram_ops::*;
}

#[cfg(feature = "types")]
pub mod types {
    //! Facade for the `hologram-types` crate.

    pub use hologram_types::*;
}

#[cfg(feature = "space")]
pub mod space {
    //! Facade for the `hologram-space` crate — the space contract (`Space`, `KappaStore`,
    //! `KappaSync`, HAL, realizations) that a [`Client`](crate::Client) composes over.

    pub use hologram_space::*;
}

// The `Client` facade (D4) — the single programmatic surface. Lifted to the crate root so
// callers write `hologram::Client`, per 05-tooling.md.
#[cfg(feature = "client")]
mod client;
#[cfg(feature = "client")]
pub use client::{BuildError, Client, ClientBuilder, Holo, RunError};
