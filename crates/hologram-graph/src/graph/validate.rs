//! Graph validation: acyclicity and input reference checks.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use super::node::NodeId;
use super::Graph;
use crate::error::{GraphError, GraphResult};

/// Validate graph structure: acyclic, all input references valid.
pub fn validate(graph: &Graph) -> GraphResult<()> {
    validate_inputs(graph)?;
    if !is_acyclic(graph) {
        return Err(GraphError::CycleDetected);
    }
    Ok(())
}

/// Check that all node input references point to live nodes.
pub fn validate_inputs(graph: &Graph) -> GraphResult<()> {
    for node in graph.nodes() {
        for dep in node.dependencies() {
            if !graph.contains(dep) {
                return Err(GraphError::InvalidNode(dep));
            }
        }
    }
    Ok(())
}

/// Check that the graph is a DAG (no cycles). Uses Kahn's algorithm.
pub fn is_acyclic(graph: &Graph) -> bool {
    let ids: Vec<NodeId> = graph.node_ids();
    if ids.is_empty() {
        return true;
    }
    // Build in-degree map using Vec indexed by position in ids
    let total = ids.len();
    let id_to_pos: std::collections::HashMap<NodeId, usize> =
        ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut in_degree = alloc::vec![0usize; total];
    for node in graph.nodes() {
        if let Some(&pos) = id_to_pos.get(&node.id) {
            for dep in node.dependencies() {
                if id_to_pos.contains_key(&dep) {
                    in_degree[pos] += 1;
                }
            }
        }
    }
    let mut queue = VecDeque::new();
    for (pos, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(pos);
        }
    }
    let mut visited = 0usize;
    while let Some(pos) = queue.pop_front() {
        visited += 1;
        let id = ids[pos];
        for succ_id in graph.successors(id) {
            if let Some(&succ_pos) = id_to_pos.get(&succ_id) {
                in_degree[succ_pos] -= 1;
                if in_degree[succ_pos] == 0 {
                    queue.push_back(succ_pos);
                }
            }
        }
    }
    visited == total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOp;
    use hologram_core::op::LutOp;

    #[test]
    fn valid_dag() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);
        assert!(validate(&g).is_ok());
    }

    #[test]
    fn empty_graph_valid() {
        assert!(validate(&Graph::new()).is_ok());
    }

    #[test]
    fn single_node_valid() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        assert!(validate(&g).is_ok());
    }

    #[test]
    fn cycle_detected() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Lut(LutOp::Relu));
        let b = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        g.add_edge(a, b);
        g.add_edge(b, a);
        assert!(!is_acyclic(&g));
        assert_eq!(validate(&g), Err(GraphError::CycleDetected));
    }

    #[test]
    fn invalid_input_reference() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        g.add_edge(a, b);
        g.remove_node(a);
        assert!(validate_inputs(&g).is_err());
    }
}
