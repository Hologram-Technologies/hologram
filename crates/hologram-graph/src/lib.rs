//! Hologram graph IR (spec Part VI).
//!
//! Arena-based DAG of nodes; each `Node` carries an `OpKind` (closed
//! catalog from `hologram-ops::OpKind`) plus its inputs and dtype/shape
//! metadata. The single `GraphOp` enum unifies all dispatch.

#![no_std]

extern crate alloc;

pub mod graph;
pub mod node;
pub mod schedule;
pub mod registry;
pub mod constant;
pub mod backward;

pub use graph::Graph;
pub use node::{Node, NodeId, GraphOp, InputSource, ConstantId, QuantAttrs};
pub use schedule::Schedule;
pub use registry::{ShapeRegistry, ShapeId, DTypeId, ShapeDescriptor};
pub use constant::ConstantStore;
pub use backward::{append_backward, BackwardError};
pub use hologram_ops::OpKind;
