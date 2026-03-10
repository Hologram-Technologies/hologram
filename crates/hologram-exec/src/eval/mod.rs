//! Graph evaluation: schedule bridge and executor.

pub mod executor;
pub mod schedule_bridge;
pub mod shape_propagate;
pub mod shape_resolve;

pub use executor::{resolve_shape_spec, GraphInputs, GraphOutputs, KvExecutor};
pub use schedule_bridge::build_schedule;
