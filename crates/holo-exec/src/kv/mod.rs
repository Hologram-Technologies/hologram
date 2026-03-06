//! KV-lookup dispatch: routes GraphOp to the correct O(1) kernel.

pub mod registry;
pub mod store;

pub use registry::{CustomHandler, CustomOpRegistry};
pub use store::KvStore;
