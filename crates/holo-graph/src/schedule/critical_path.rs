//! Critical path analysis for execution scheduling.

use std::collections::HashMap;

use crate::error::GraphResult;
use crate::graph::node::NodeId;
use crate::graph::Graph;
use crate::schedule::toposort;

/// Compute the critical path length (longest dependency chain). O(V + E).
pub fn critical_path_length(graph: &Graph) -> GraphResult<usize> {
    let order = toposort::toposort(graph)?;
    if order.is_empty() {
        return Ok(0);
    }
    let mut longest: HashMap<NodeId, usize> = HashMap::new();
    for &id in &order {
        let pred_max = graph
            .predecessors(id)
            .iter()
            .filter_map(|p| longest.get(p))
            .max()
            .copied()
            .unwrap_or(0);
        longest.insert(id, pred_max + 1);
    }
    Ok(longest.values().max().copied().unwrap_or(0))
}

/// Parallelism ratio: total_nodes / critical_path_length.
///
/// 1.0 = fully sequential. Higher = more parallelizable.
pub fn parallelism_ratio(graph: &Graph) -> GraphResult<f64> {
    let path_len = critical_path_length(graph)?;
    if path_len == 0 {
        return Ok(0.0);
    }
    Ok(graph.node_count() as f64 / path_len as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOp;
    use holo_core::op::LutOp;

    #[test]
    fn empty_graph() {
        assert_eq!(critical_path_length(&Graph::new()).unwrap(), 0);
        assert_eq!(parallelism_ratio(&Graph::new()).unwrap(), 0.0);
    }

    #[test]
    fn single_node() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        assert_eq!(critical_path_length(&g).unwrap(), 1);
        assert_eq!(parallelism_ratio(&g).unwrap(), 1.0);
    }

    #[test]
    fn linear_chain_4() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);
        g.add_edge(c, d);
        assert_eq!(critical_path_length(&g).unwrap(), 4);
        assert_eq!(parallelism_ratio(&g).unwrap(), 1.0);
    }

    #[test]
    fn parallel_fan_out() {
        // 0 → [1, 2, 3] → 4
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Lut(LutOp::Tanh));
        let e = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(a, c);
        g.add_edge(a, d);
        g.add_edge(b, e);
        g.add_edge(c, e);
        g.add_edge(d, e);
        assert_eq!(critical_path_length(&g).unwrap(), 3);
        // 5 nodes / 3 path = 1.666...
        assert!(parallelism_ratio(&g).unwrap() > 1.5);
    }
}
