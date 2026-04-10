//! Build an `ExecutionSchedule` directly from a `SerializedGraph`.
//!
//! This avoids reconstructing a full arena `Graph` — we operate on the
//! dense node array using Kahn's cursor-based level batching. O(V + E).

use std::collections::HashMap;

use hologram_ir::graph::node::{InputSource, NodeId};
use hologram_ir::schedule::levels::ParallelLevel;
use hologram_ir::schedule::ExecutionSchedule;

use crate::error::{ExecError, ExecResult};

use hologram_archive::format::graph::SerializedGraph;

/// Build an `ExecutionSchedule` from a serialized graph.
///
/// Runs Kahn's algorithm on the dense node array to produce
/// parallel execution levels. Returns `CycleDetected` if the
/// graph contains a cycle.
pub fn build_schedule(sg: &SerializedGraph) -> ExecResult<ExecutionSchedule> {
    let nodes = &sg.nodes;
    if nodes.is_empty() {
        return Ok(ExecutionSchedule {
            levels: Vec::new(),
            critical_path: 0,
        });
    }

    // Position map: NodeId → index in the dense array
    let id_to_pos: HashMap<NodeId, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();

    let total = nodes.len();

    // Build successor lists and in-degrees from input edges.
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); total];
    let mut in_degree: Vec<u32> = vec![0; total];

    for (pos, node) in nodes.iter().enumerate() {
        for slot in &node.inputs {
            if let InputSource::Node(dep_id) = slot.source {
                if let Some(&dep_pos) = id_to_pos.get(&dep_id) {
                    successors[dep_pos].push(pos);
                    in_degree[pos] += 1;
                }
            }
        }
    }

    // Kahn's cursor-based level batching
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
            level_ids.push(nodes[pos].id);
            visited += 1;
            for &succ_pos in &successors[pos] {
                in_degree[succ_pos] -= 1;
                if in_degree[succ_pos] == 0 {
                    ready.push(succ_pos);
                }
            }
        }
        levels.push(ParallelLevel {
            node_ids: level_ids,
        });
        cursor = level_end;
    }

    if visited != total {
        return Err(ExecError::CycleDetected);
    }

    let critical_path = levels.len();

    Ok(ExecutionSchedule {
        levels,
        critical_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_ir::constant::ConstantStore;
    use hologram_ir::graph::node::{InputSlot, Node};
    use hologram_ir::graph::GraphOp;

    /// Helper to build a minimal SerializedGraph from nodes.
    fn sg(nodes: Vec<Node>) -> SerializedGraph {
        SerializedGraph {
            nodes,
            input_names: Vec::new(),
            output_names: Vec::new(),
            output_node_ids: Vec::new(),
            constants: ConstantStore::new(),
            constant_shapes: Vec::new(),
            node_shapes: Vec::new(),
            node_dtypes: Vec::new(),
        }
    }

    fn nid(n: u32) -> NodeId {
        NodeId::new(n, 0)
    }

    fn node(id: u32, op: GraphOp, inputs: Vec<InputSlot>) -> Node {
        Node {
            id: nid(id),
            op,
            inputs: inputs.into_iter().collect(),
            num_outputs: 1,
        }
    }

    #[test]
    fn empty_graph() {
        let sched = build_schedule(&sg(vec![])).unwrap();
        assert!(sched.levels.is_empty());
        assert_eq!(sched.critical_path, 0);
    }

    #[test]
    fn single_node() {
        let g = sg(vec![node(0, GraphOp::Input, vec![])]);
        let sched = build_schedule(&g).unwrap();
        assert_eq!(sched.levels.len(), 1);
        assert_eq!(sched.levels[0].node_ids, vec![nid(0)]);
    }

    #[test]
    fn linear_chain() {
        // 0 → 1 → 2
        let g = sg(vec![
            node(0, GraphOp::Input, vec![]),
            node(
                1,
                GraphOp::Lut(hologram_core::op::LutOp::Relu),
                vec![InputSlot::from_node(nid(0))],
            ),
            node(2, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
        ]);
        let sched = build_schedule(&g).unwrap();
        assert_eq!(sched.levels.len(), 3);
        assert_eq!(sched.critical_path, 3);
    }

    #[test]
    fn parallel_fan_out() {
        // 0 → [1, 2] → 3
        let g = sg(vec![
            node(0, GraphOp::Input, vec![]),
            node(
                1,
                GraphOp::Lut(hologram_core::op::LutOp::Relu),
                vec![InputSlot::from_node(nid(0))],
            ),
            node(
                2,
                GraphOp::Lut(hologram_core::op::LutOp::Sigmoid),
                vec![InputSlot::from_node(nid(0))],
            ),
            node(
                3,
                GraphOp::Prim(hologram_core::op::PrimOp::Add),
                vec![InputSlot::from_node(nid(1)), InputSlot::from_node(nid(2))],
            ),
        ]);
        let sched = build_schedule(&g).unwrap();
        assert_eq!(sched.levels.len(), 3);
        // Level 1: nodes 1 and 2 in parallel
        assert_eq!(sched.levels[1].node_ids.len(), 2);
    }

    #[test]
    fn graph_inputs_ignored_in_deps() {
        // Node with GraphInput source — no node dependency
        let g = sg(vec![node(
            0,
            GraphOp::Lut(hologram_core::op::LutOp::Relu),
            vec![InputSlot::from_graph_input(0)],
        )]);
        let sched = build_schedule(&g).unwrap();
        assert_eq!(sched.levels.len(), 1);
    }

    #[test]
    fn parallelism_ratio() {
        // 0 → [1, 2, 3] → 4  =>  3 levels, 5 nodes, ratio = 5/3
        let g = sg(vec![
            node(0, GraphOp::Input, vec![]),
            node(
                1,
                GraphOp::Lut(hologram_core::op::LutOp::Relu),
                vec![InputSlot::from_node(nid(0))],
            ),
            node(
                2,
                GraphOp::Lut(hologram_core::op::LutOp::Sigmoid),
                vec![InputSlot::from_node(nid(0))],
            ),
            node(
                3,
                GraphOp::Lut(hologram_core::op::LutOp::Tanh),
                vec![InputSlot::from_node(nid(0))],
            ),
            node(
                4,
                GraphOp::Output,
                vec![
                    InputSlot::from_node(nid(1)),
                    InputSlot::from_node(nid(2)),
                    InputSlot::from_node(nid(3)),
                ],
            ),
        ]);
        let sched = build_schedule(&g).unwrap();
        assert_eq!(sched.num_levels(), 3);
        let ratio = sched.parallelism_ratio();
        assert!(ratio > 1.5); // 5/3 ≈ 1.67
    }
}
