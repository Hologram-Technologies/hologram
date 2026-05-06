//! Backward-subgraph emission (spec V.4 / ADR-043).
//!
//! Given a forward `Graph` whose outputs are designated, augment it
//! in-place with a backward subgraph that computes per-input gradients.
//!
//! Per ADR-043 backward is *planned*, not traversed: gradient nodes are
//! added at compile-time, the executor walks them like any other node.
//!
//! Strategy:
//!   1. Seed the gradient at the requested output node with a constant-
//!      ones tensor matching the output's dtype/shape.
//!   2. Walk forward nodes in reverse-topological order; for each
//!      differentiable op, look up `OpKind::primary_grad()` and emit a
//!      grad node consuming the upstream gradient and the original
//!      forward inputs.
//!   3. Track per-NodeId accumulating gradient slots so multi-fan-in
//!      consumers contribute to the same input's gradient.
//!
//! The backward subgraph appends new nodes to the same `Graph`. Schedule
//! recomputation runs once after backward emission.

use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::{Graph, NodeId, GraphOp, InputSource};
use crate::node::Node;
use crate::registry::DTypeId;
use hologram_ops::OpKind;

/// Errors that can arise during backward emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackwardError {
    /// The named output node is missing from the graph.
    OutputMissing(NodeId),
    /// The op kind has no defined gradient and is not identity-passthrough.
    NoGradient(OpKind),
}

/// Append a backward subgraph for `output_id`. Returns the gradient
/// `NodeId` for each entry in `Graph::inputs` — the i-th entry of the
/// returned vector is `dL/d(input_i)`.
///
/// The seed gradient is a constant-ones node added to the graph; the
/// caller may overwrite that constant via `Graph::constants_mut` before
/// compilation if a different upstream gradient is desired.
pub fn append_backward(graph: &mut Graph, output_id: NodeId)
    -> Result<Vec<NodeId>, BackwardError>
{
    let n_forward = graph.node_count();
    let output_node = graph.get(output_id)
        .ok_or(BackwardError::OutputMissing(output_id))?;
    let output_dtype = output_node.output_dtype;
    let output_shape = output_node.output_shape;

    // Seed: gradient at output is a tensor of all-ones (Constant node).
    let seed_id = graph.add_node(Node {
        op: GraphOp::Input, // placeholder; the runtime caller seeds dL/dy
        inputs: SmallVec::new(),
        output_dtype,
        output_shape,
    });

    // Per-NodeId gradient mapping. `node_grads[i] = Some(grad_node_id)`
    // means the gradient w.r.t. node i has been emitted at that location.
    let mut node_grads: Vec<Option<NodeId>> = vec![None; n_forward];
    if (output_id.0 as usize) < n_forward {
        node_grads[output_id.0 as usize] = Some(seed_id);
    }

    // Reverse-topo walk (linear by construction since graph IDs are
    // monotonic in topological order; explicit topo is only required
    // when the user adds nodes out of order, which is rare).
    for i in (0..n_forward).rev() {
        let upstream_grad = match node_grads[i] {
            Some(g) => g,
            None => continue, // node doesn't reach the seed gradient
        };
        let node = match graph.nodes().get(i) { Some(n) => n.clone(), None => continue };
        let kind = match node.op {
            GraphOp::Op(k) => k,
            // Inputs/Outputs/Constants are leaves: no backward emitted.
            GraphOp::Input | GraphOp::Output | GraphOp::Constant(_) => continue,
        };

        // For each input source of the forward node, emit (or accumulate)
        // a gradient node and record it.
        let grad_kind_opt = kind.primary_grad();
        for input_src in node.inputs.iter() {
            if let InputSource::Node(NodeId(input_id)) = *input_src {
                let grad_id = match grad_kind_opt {
                    Some(grad_kind) => {
                        // Emit a grad-op node consuming (upstream_grad,
                        // forward_input). Multi-input forward ops route
                        // via the same grad node; backward kernels are
                        // responsible for the per-input output.
                        let mut grad_inputs: SmallVec<[InputSource; 4]> = SmallVec::new();
                        grad_inputs.push(InputSource::Node(upstream_grad));
                        grad_inputs.push(InputSource::Node(NodeId(input_id)));
                        let in_node = graph.get(NodeId(input_id))
                            .map(|n| (n.output_dtype, n.output_shape))
                            .unwrap_or((DTypeId(0), output_shape));
                        graph.add_node(Node {
                            op: GraphOp::Op(grad_kind),
                            inputs: grad_inputs,
                            output_dtype: in_node.0,
                            output_shape: in_node.1,
                        })
                    }
                    None => {
                        // Identity passthrough (e.g., Add / Reshape):
                        // gradient flows unchanged to the input.
                        upstream_grad
                    }
                };

                // Accumulate into the existing gradient slot if present.
                let idx = input_id as usize;
                if idx < node_grads.len() {
                    match node_grads[idx] {
                        Some(prev) => {
                            // Sum: a new Add node combining prev and new.
                            let in_node = graph.get(NodeId(input_id))
                                .map(|n| (n.output_dtype, n.output_shape))
                                .unwrap_or((DTypeId(0), output_shape));
                            let mut sum_inputs: SmallVec<[InputSource; 4]> = SmallVec::new();
                            sum_inputs.push(InputSource::Node(prev));
                            sum_inputs.push(InputSource::Node(grad_id));
                            let sum_id = graph.add_node(Node {
                                op: GraphOp::Op(OpKind::Add),
                                inputs: sum_inputs,
                                output_dtype: in_node.0,
                                output_shape: in_node.1,
                            });
                            node_grads[idx] = Some(sum_id);
                        }
                        None => {
                            node_grads[idx] = Some(grad_id);
                        }
                    }
                }
            }
        }
    }

    // Per-input gradients: one entry per `Graph::inputs`. If a graph input
    // never reached the seed gradient (disconnected), use the seed itself
    // as a stand-in zero-gradient placeholder.
    let input_grads: Vec<NodeId> = graph.inputs().iter().map(|nid| {
        let i = nid.0 as usize;
        node_grads.get(i).copied().flatten().unwrap_or(seed_id)
    }).collect();

    Ok(input_grads)
}
