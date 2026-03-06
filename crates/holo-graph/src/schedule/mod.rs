//! Execution scheduling: topological sort, parallel levels, critical path.

pub mod critical_path;
pub mod levels;
pub mod toposort;

use crate::error::GraphResult;
use crate::graph::Graph;
use levels::ParallelLevel;

/// Complete execution schedule for a graph.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct ExecutionSchedule {
    /// Execution levels (nodes within each level can run in parallel).
    pub levels: Vec<ParallelLevel>,
    /// Length of the critical path.
    pub critical_path: usize,
}

impl ExecutionSchedule {
    /// Build an execution schedule from a graph. O(V + E).
    pub fn build(graph: &Graph) -> GraphResult<Self> {
        let levels = levels::build_parallel_levels(graph)?;
        let cp = critical_path::critical_path_length(graph)?;
        Ok(Self {
            levels,
            critical_path: cp,
        })
    }

    /// Number of execution levels.
    #[must_use]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Parallelism ratio: total nodes / critical path.
    #[must_use]
    pub fn parallelism_ratio(&self) -> f64 {
        if self.critical_path == 0 {
            return 0.0;
        }
        let total: usize = self.levels.iter().map(|l| l.node_ids.len()).sum();
        total as f64 / self.critical_path as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOp;
    use holo_core::op::LutOp;

    #[test]
    fn schedule_diamond() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        let c = g.add_node(GraphOp::Lut(LutOp::Sigmoid));
        let d = g.add_node(GraphOp::Output);
        g.add_edge(a, b);
        g.add_edge(a, c);
        g.add_edge(b, d);
        g.add_edge(c, d);
        let sched = ExecutionSchedule::build(&g).unwrap();
        assert_eq!(sched.num_levels(), 3);
        assert_eq!(sched.critical_path, 3);
        assert!(sched.parallelism_ratio() > 1.0);
    }

    #[test]
    fn schedule_empty() {
        let sched = ExecutionSchedule::build(&Graph::new()).unwrap();
        assert_eq!(sched.num_levels(), 0);
        assert_eq!(sched.parallelism_ratio(), 0.0);
    }
}
