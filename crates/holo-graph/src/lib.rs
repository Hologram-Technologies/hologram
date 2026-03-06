//! Expression graph, subgraph composition, fusion engine, and parallel scheduling.
//!
//! Provides a single unified `Graph` type (replacing v1's dual OperationGraph/CompileGraph),
//! subgraph templates with flatten/instantiation, single-pass fusion, and dependency-aware
//! parallel level scheduling.

pub mod builder;
pub mod constant;
pub mod error;
pub mod fusion;
pub mod graph;
pub mod schedule;
pub mod subgraph;
