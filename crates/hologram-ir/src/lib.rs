//! Expression graph, subgraph composition, structural analysis, and parallel scheduling.
//!
//! Provides a single unified `Graph` type, subgraph templates with
//! flatten/instantiation, and the `analysis/` module that hosts every
//! structural-finder pass: liveness, precision (ring-level inference), qedl
//! (domain-crossing detection), workspace (slot reuse planning), constant
//! folding, view detection (W8/W16 LUT collapse), float fusion (chains
//! and matmul/conv epilogue patterns), and CSE.
//!
//! Each analysis is a *finder*: it detects a structural pattern the source
//! graph already exhibits and emits the IR encoding of that finding.
//! Entry point: [`analysis::analyze`].

pub mod analysis;
pub mod builder;
pub mod constant;
pub mod error;
pub mod graph;
pub mod schedule;
pub mod subgraph;

// Convenience re-exports
pub use analysis::{analyze, StructuralFindings};
pub use builder::GraphBuilder;
pub use constant::{ConstantData, ConstantId, ConstantStore};
pub use error::{GraphError, GraphResult};
pub use graph::node::NodeId;
pub use graph::{Graph, GraphOp, SubgraphId};
pub use schedule::ExecutionSchedule;
pub use subgraph::SubgraphDef;
