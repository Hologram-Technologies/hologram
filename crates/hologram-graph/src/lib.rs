//! Hologram graph IR (spec Part VI).
//!
//! Arena-based DAG of nodes; each `Node` carries an `OpKind` (closed
//! catalog from `hologram-ops::OpKind`) plus its inputs and dtype/shape
//! metadata. The single `GraphOp` enum unifies all dispatch.

#![no_std]

extern crate alloc;

pub mod backward;
pub mod constant;
pub mod graph;
pub mod node;
pub mod registry;
pub mod schedule;

pub use backward::{append_backward, BackwardError};
pub use constant::ConstantStore;
pub use graph::Graph;
pub use hologram_ops::OpKind;
pub use node::{ConstantId, ConvAttrs, GraphOp, InputSource, LrnAttrs, Node, NodeId, QuantAttrs};
pub use registry::{DTypeId, ShapeDescriptor, ShapeId, ShapeRegistry};
pub use schedule::Schedule;
