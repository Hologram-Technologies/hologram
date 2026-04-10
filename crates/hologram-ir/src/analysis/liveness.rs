//! Liveness analysis for buffer lifetime tracking.
//!
//! Computes the lifetime interval `[born, dies]` for each node's output
//! buffer in terms of schedule level indices. Non-overlapping intervals
//! can share workspace buffer slots.
//!
//! # Architectural role
//!
//! This module is a **structure finder**, not a constructor. It computes a
//! property of the source graph (each node's lifetime in the schedule) that
//! the source already determines — it does not make optimization decisions
//! or reshape the graph. Per the SCS section 5 framing, liveness is
//! pre-existing structural content that the cross-compiler reads.
//!
//! # Performance
//!
//! O(N + E) per analysis pass: one HashMap lookup per node and one per
//! successor edge. Optionally parallelizable via the `parallel` feature.
//! **Perf: NEUTRAL** — pure compile-time work.

use std::collections::HashMap;

use crate::graph::node::NodeId;
use crate::graph::Graph;
use crate::schedule::ExecutionSchedule;

/// Lifetime interval of a node's output buffer.
///
/// `born` is the schedule level where the buffer is produced.
/// `dies` is the last schedule level where any consumer reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivenessInterval {
    /// The node that produces this buffer.
    pub node_id: NodeId,
    /// Level index where the buffer is first available.
    pub born: usize,
    /// Last level index where the buffer is consumed.
    pub dies: usize,
}

impl LivenessInterval {
    /// Duration of the interval (inclusive).
    #[must_use]
    pub fn duration(&self) -> usize {
        self.dies.saturating_sub(self.born) + 1
    }
}

/// Compute liveness intervals for all nodes in the schedule.
pub fn compute_liveness(schedule: &ExecutionSchedule, graph: &Graph) -> Vec<LivenessInterval> {
    if schedule.levels.is_empty() {
        return Vec::new();
    }
    let level_map = build_level_map(schedule);
    let max_level = schedule.levels.len() - 1;
    let last_use = build_last_use_map(graph, &level_map, max_level);

    build_intervals(&level_map, &last_use)
}

/// Map each node to the level it is scheduled in.
fn build_level_map(schedule: &ExecutionSchedule) -> HashMap<NodeId, usize> {
    let mut map = HashMap::new();
    for (level_idx, level) in schedule.levels.iter().enumerate() {
        for &node_id in &level.node_ids {
            map.insert(node_id, level_idx);
        }
    }
    map
}

/// For each node, find the last level where its output is consumed.
fn build_last_use_map(
    graph: &Graph,
    level_map: &HashMap<NodeId, usize>,
    max_level: usize,
) -> HashMap<NodeId, usize> {
    let ids = graph.node_ids();

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        ids.par_iter()
            .map(|&id| {
                let succs = graph.successors(id);
                let dies = compute_dies(&succs, level_map, max_level);
                (id, dies)
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut last_use = HashMap::new();
        for &id in &ids {
            let succs = graph.successors(id);
            let dies = compute_dies(&succs, level_map, max_level);
            last_use.insert(id, dies);
        }
        last_use
    }
}

/// Compute the `dies` level for a node given its successors.
fn compute_dies(
    successors: &[NodeId],
    level_map: &HashMap<NodeId, usize>,
    max_level: usize,
) -> usize {
    if successors.is_empty() {
        return max_level;
    }
    successors
        .iter()
        .filter_map(|s| level_map.get(s))
        .copied()
        .max()
        .unwrap_or(max_level)
}

/// Build intervals from the level map and last-use map.
fn build_intervals(
    level_map: &HashMap<NodeId, usize>,
    last_use: &HashMap<NodeId, usize>,
) -> Vec<LivenessInterval> {
    level_map
        .iter()
        .map(|(&node_id, &born)| {
            let dies = last_use.get(&node_id).copied().unwrap_or(born);
            LivenessInterval {
                node_id,
                born,
                dies,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOp;
    use hologram_core::op::{LutOp, PrimOp};

    fn find_interval(intervals: &[LivenessInterval], id: NodeId) -> &LivenessInterval {
        intervals.iter().find(|i| i.node_id == id).unwrap()
    }

    #[test]
    fn empty_schedule() {
        let g = Graph::new();
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);
        assert!(intervals.is_empty());
    }

    #[test]
    fn single_node() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);
        assert_eq!(intervals.len(), 1);
        let iv = find_interval(&intervals, a);
        assert_eq!(iv.born, 0);
        assert_eq!(iv.dies, 0); // max_level = 0, no successors
    }

    #[test]
    fn linear_chain() {
        // A(0) → B(1) → C(2)
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);

        let iv_a = find_interval(&intervals, a);
        assert_eq!(iv_a.born, 0);
        assert_eq!(iv_a.dies, 1); // consumed by B at level 1

        let iv_b = find_interval(&intervals, b);
        assert_eq!(iv_b.born, 1);
        assert_eq!(iv_b.dies, 2); // consumed by C at level 2

        let iv_c = find_interval(&intervals, c);
        assert_eq!(iv_c.born, 2);
        assert_eq!(iv_c.dies, 2); // sink: dies at max_level
    }

    #[test]
    fn diamond_graph() {
        // A(0) → [B(1), C(1)] → D(2)
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Prim(PrimOp::Add));
        g.add_edge(a, b);
        g.add_edge(a, c);
        g.add_edge(b, d);
        g.add_edge(c, d);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);

        let iv_a = find_interval(&intervals, a);
        assert_eq!(iv_a.born, 0);
        assert_eq!(iv_a.dies, 1); // consumed by B,C at level 1

        let iv_b = find_interval(&intervals, b);
        assert_eq!(iv_b.born, 1);
        assert_eq!(iv_b.dies, 2); // consumed by D at level 2

        let iv_d = find_interval(&intervals, d);
        assert_eq!(iv_d.born, 2);
        assert_eq!(iv_d.dies, 2); // sink
    }

    #[test]
    fn fan_out_extends_lifetime() {
        // A(0) → B(1) → D(2)
        // A(0) → C(1) → E(2) → F(3)
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Output);
        let e = g.add_node(GraphOp::Lut(LutOp::Tanh));
        let f = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(a, c);
        g.add_edge(b, d);
        g.add_edge(c, e);
        g.add_edge(e, f);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);

        // A is consumed at level 1 (by both B and C)
        let iv_a = find_interval(&intervals, a);
        assert_eq!(iv_a.born, 0);
        assert_eq!(iv_a.dies, 1);
    }

    #[test]
    fn sink_node_dies_at_max_level() {
        // A → B (B is a sink with no successors)
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);

        let iv_b = find_interval(&intervals, b);
        assert_eq!(iv_b.dies, 1); // max_level = 1
    }

    #[test]
    fn duration() {
        let iv = LivenessInterval {
            node_id: NodeId::new(0, 0),
            born: 1,
            dies: 3,
        };
        assert_eq!(iv.duration(), 3); // 3 - 1 + 1
    }

    #[test]
    fn same_level_duration() {
        let iv = LivenessInterval {
            node_id: NodeId::new(0, 0),
            born: 2,
            dies: 2,
        };
        assert_eq!(iv.duration(), 1);
    }

    #[test]
    fn fan_in_graph() {
        // A(0) and B(0) both feed C(1)
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Input);
        let c = g.add_node(GraphOp::Prim(PrimOp::Add));
        g.add_edge(a, c);
        g.add_edge(b, c);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);

        let iv_a = find_interval(&intervals, a);
        assert_eq!(iv_a.born, 0);
        assert_eq!(iv_a.dies, 1); // consumed by C at level 1

        let iv_b = find_interval(&intervals, b);
        assert_eq!(iv_b.born, 0);
        assert_eq!(iv_b.dies, 1);
    }

    #[test]
    fn three_level_chain_intervals() {
        // A → B → C → D (4 levels)
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);
        g.add_edge(c, d);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);

        // A: born=0, dies=1 (consumed by B)
        assert_eq!(find_interval(&intervals, a).dies, 1);
        // B: born=1, dies=2 (consumed by C)
        assert_eq!(find_interval(&intervals, b).dies, 2);
        // C: born=2, dies=3 (consumed by D)
        assert_eq!(find_interval(&intervals, c).dies, 3);
    }

    #[test]
    fn all_nodes_have_intervals() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(b, c);
        let sched = ExecutionSchedule::build(&g).unwrap();
        let intervals = compute_liveness(&sched, &g);
        assert_eq!(intervals.len(), 3);
    }
}
