//! KV-lookup dispatch: routes GraphOp to the correct O(1) kernel.

pub mod store;

pub use store::KvStore;
