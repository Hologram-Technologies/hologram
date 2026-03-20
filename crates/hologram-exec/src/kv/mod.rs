//! KV-lookup dispatch: routes GraphOp to the correct O(1) kernel.

pub mod registry;
pub mod store;
pub mod weight_cache;

pub use registry::{CustomHandler, CustomOpRegistry};
pub use store::KvStore;
pub use weight_cache::WeightCache;
