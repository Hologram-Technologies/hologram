//! Graph evaluation: schedule bridge and executor.

pub mod executor;
pub mod schedule_bridge;
pub mod shape_propagate;
pub mod shape_resolve;

pub use executor::{GraphInputs, GraphOutputs};
pub use schedule_bridge::build_schedule;
