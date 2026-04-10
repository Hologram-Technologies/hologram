//! KV-lookup dispatch: routes GraphOp to the correct O(1) kernel.

pub mod store;
pub mod weight_cache;

pub use store::KvStore;
pub use weight_cache::WeightCache;
