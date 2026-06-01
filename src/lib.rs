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

#[cfg(feature = "archive")]
pub mod archive {
    //! Facade for the `hologram-archive` crate.

    pub use hologram_archive::*;
}

#[cfg(feature = "backend")]
pub mod backend {
    //! Facade for the `hologram-backend` crate.

    pub use hologram_backend::*;
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

#[cfg(feature = "host")]
pub mod host {
    //! Facade for the `hologram-host` crate.

    pub use hologram_host::*;
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
