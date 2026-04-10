//! Graph evaluation: schedule bridge and execution types.

pub mod executor;
pub mod schedule_bridge;

pub use executor::{GraphInputs, GraphOutputs};
pub use schedule_bridge::build_schedule;
