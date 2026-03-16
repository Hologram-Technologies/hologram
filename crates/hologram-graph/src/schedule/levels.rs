//! Parallel execution levels via modified Kahn's algorithm. O(V + E).

use std::collections::HashMap;

use crate::error::{GraphError, GraphResult};
use crate::graph::node::NodeId;
use crate::graph::Graph;

/// A level of nodes that can execute in parallel.
///
/// **PL_2 (Lease Disjointness)**: each `ParallelLevel` is a *lease* in Prism terms — a disjoint
/// partition of the computation budget. The Kahn's-algorithm construction guarantees that no node
/// appears in more than one level, and that every node in a level has all its predecessors in
/// strictly earlier levels. This structural disjointness satisfies SR_9 (ContextLease fiber
/// disjointness) without requiring a runtime check.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ParallelLevel {
    /// Nodes in this level (no mutual dependencies).
    pub node_ids: Vec<NodeId>,
}

/// Build parallel execution levels. O(V + E).
///
/// Level N nodes have all dependencies in levels < N,
/// so all nodes within a level can execute concurrently.
pub fn build_parallel_levels(graph: &Graph) -> GraphResult<Vec<ParallelLevel>> {
    let ids = graph.node_ids();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let id_to_pos: HashMap<NodeId, usize> =
        ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let total = ids.len();
    let mut in_degree = vec![0u32; total];
    for node in graph.nodes() {
        if let Some(&pos) = id_to_pos.get(&node.id) {
            // Count each unique predecessor once, matching successors() which
            // also returns each successor at most once.
            let mut seen = std::collections::HashSet::new();
            for dep in node.dependencies() {
                if id_to_pos.contains_key(&dep) && seen.insert(dep) {
                    in_degree[pos] += 1;
                }
            }
        }
    }

    // Cursor-based level batching
    let mut ready: Vec<usize> = Vec::new();
    for (pos, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push(pos);
        }
    }

    let mut levels = Vec::new();
    let mut cursor = 0;
    let mut visited = 0usize;

    while cursor < ready.len() {
        let level_end = ready.len();
        let mut level_ids = Vec::new();
        for i in cursor..level_end {
            let pos = ready[i];
            let id = ids[pos];
            level_ids.push(id);
            visited += 1;
            for succ_id in graph.successors(id) {
                if let Some(&succ_pos) = id_to_pos.get(&succ_id) {
                    in_degree[succ_pos] -= 1;
                    if in_degree[succ_pos] == 0 {
                        ready.push(succ_pos);
                    }
                }
            }
        }
        levels.push(ParallelLevel {
            node_ids: level_ids,
        });
        cursor = level_end;
    }

    if visited == total {
        Ok(levels)
    } else {
        Err(GraphError::CycleDetected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOp;
    use hologram_core::op::{LutOp, PrimOp};

    #[test]
    fn empty() {
        let levels = build_parallel_levels(&Graph::new()).unwrap();
        assert!(levels.is_empty());
    }

    #[test]
    fn single_node() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        let levels = build_parallel_levels(&g).unwrap();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].node_ids.len(), 1);
    }

    #[test]
    fn linear_chain() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);
        g.add_edge(c, d);
        let levels = build_parallel_levels(&g).unwrap();
        assert_eq!(levels.len(), 4);
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
        let levels = build_parallel_levels(&g).unwrap();
        assert_eq!(levels.len(), 3);
        // Level 1 should have 3 parallel nodes
        assert_eq!(levels[1].node_ids.len(), 3);
    }

    #[test]
    fn diamond() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Prim(PrimOp::Add));
        g.add_edge(a, b);
        g.add_edge(a, c);
        g.add_edge(b, d);
        g.add_edge(c, d);
        let levels = build_parallel_levels(&g).unwrap();
        assert_eq!(levels.len(), 3);
        // Level 1: b and c in parallel
        assert_eq!(levels[1].node_ids.len(), 2);
    }
}
