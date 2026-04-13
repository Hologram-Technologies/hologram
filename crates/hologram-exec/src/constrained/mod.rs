//! Constrained execution profile for deterministic, bounded-memory tape execution.
//!
//! This module provides a restricted execution mode that enforces memory limits,
//! kernel allowlists, and weight residency policies. Workload-agnostic: supports
//! AI inference, numeric pipelines, rendering, and signal processing.

pub mod profile;
pub mod region_pack;
pub mod runner;
pub mod tape_subset;
pub mod weight_window;

pub use profile::{ConstrainedProfile, KernelAllowlist, KernelDiscriminant, WeightPolicy};
pub use region_pack::{PackedWeightSpan, RegionIndex};
pub use runner::{execute_constrained, ConstrainedRunner};
pub use tape_subset::validate_constrained_tape;
pub use weight_window::WeightWindow;
