//! Graph evaluation: schedule bridge and executor.

pub mod executor;
pub mod schedule_bridge;

pub use executor::{GraphInputs, GraphOutputs, KvExecutor};
pub use schedule_bridge::build_schedule;
