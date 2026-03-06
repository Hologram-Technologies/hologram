//! Topological sort via Kahn's algorithm. O(V + E).

use std::collections::{HashMap, VecDeque};

use crate::error::{GraphError, GraphResult};
use crate::graph::node::NodeId;
use crate::graph::Graph;

/// Topological sort of graph nodes. Returns error on cycle.
pub fn toposort(graph: &Graph) -> GraphResult<Vec<NodeId>> {
    let ids = graph.node_ids();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let id_set: HashMap<NodeId, usize> =
        ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut in_degree = vec![0u32; ids.len()];
    for node in graph.nodes() {
        if let Some(&pos) = id_set.get(&node.id) {
            for dep in node.dependencies() {
                if id_set.contains_key(&dep) {
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
    let mut order = Vec::with_capacity(ids.len());
    while let Some(pos) = queue.pop_front() {
        order.push(ids[pos]);
        for succ_id in graph.successors(ids[pos]) {
            if let Some(&succ_pos) = id_set.get(&succ_id) {
                in_degree[succ_pos] -= 1;
                if in_degree[succ_pos] == 0 {
                    queue.push_back(succ_pos);
                }
            }
        }
    }
    if order.len() == ids.len() {
        Ok(order)
    } else {
        Err(GraphError::CycleDetected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOp;
    use holo_core::op::LutOp;

    #[test]
    fn linear_chain() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);
        let order = toposort(&g).unwrap();
        let pos_a = order.iter().position(|&id| id == a).unwrap();
        let pos_b = order.iter().position(|&id| id == b).unwrap();
        let pos_c = order.iter().position(|&id| id == c).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn diamond() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(a, c);
        g.add_edge(b, d);
        g.add_edge(c, d);
        let order = toposort(&g).unwrap();
        let pos = |id| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a) < pos(b));
        assert!(pos(a) < pos(c));
        assert!(pos(b) < pos(d));
        assert!(pos(c) < pos(d));
    }

    #[test]
    fn cycle_error() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Lut(LutOp::Relu));
        let b = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        g.add_edge(a, b);
        g.add_edge(b, a);
        assert!(toposort(&g).is_err());
    }

    #[test]
    fn empty_graph() {
        assert!(toposort(&Graph::new()).unwrap().is_empty());
    }

    #[test]
    fn single_node() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        assert_eq!(toposort(&g).unwrap().len(), 1);
    }

    #[test]
    fn disconnected_nodes() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        g.add_node(GraphOp::Input);
        g.add_node(GraphOp::Input);
        let order = toposort(&g).unwrap();
        assert_eq!(order.len(), 3);
    }
}
