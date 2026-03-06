//! Workspace planning for buffer slot reuse.
//!
//! Assigns node output buffers to reusable slots via first-fit-decreasing
//! bin packing. Nodes with non-overlapping liveness intervals share slots.

use hologram_graph::graph::node::NodeId;

use crate::liveness::LivenessInterval;

/// A reusable workspace buffer slot.
#[derive(Debug, Clone)]
pub struct BufferSlot {
    /// Unique slot identifier.
    pub slot_id: u32,
    /// Nodes assigned to this slot (non-overlapping lifetimes).
    pub occupants: Vec<NodeId>,
}

/// Workspace layout mapping nodes to reusable buffer slots.
#[derive(Debug, Clone)]
pub struct WorkspaceLayout {
    /// All allocated slots.
    pub slots: Vec<BufferSlot>,
    /// Total number of slots needed.
    pub total_slots: usize,
    /// Node-to-slot assignments: (node_id, slot_id).
    pub assignments: Vec<(NodeId, u32)>,
}

/// Plan workspace buffer slot assignments from liveness intervals.
///
/// Uses first-fit-decreasing: sorts by lifetime duration (longest first),
/// then assigns each to the first compatible slot.
#[must_use]
pub fn plan_workspace(intervals: &[LivenessInterval]) -> WorkspaceLayout {
    if intervals.is_empty() {
        return empty_layout();
    }
    let sorted = sort_by_duration(intervals);
    assign_slots(&sorted, intervals)
}

/// Two intervals can share a slot if they don't overlap.
#[must_use]
fn can_share(a: &LivenessInterval, b: &LivenessInterval) -> bool {
    a.dies < b.born || b.dies < a.born
}

/// Return an empty workspace layout.
fn empty_layout() -> WorkspaceLayout {
    WorkspaceLayout {
        slots: Vec::new(),
        total_slots: 0,
        assignments: Vec::new(),
    }
}

/// Sort interval indices by duration (longest first).
fn sort_by_duration(intervals: &[LivenessInterval]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..intervals.len()).collect();
    indices.sort_by(|&a, &b| intervals[b].duration().cmp(&intervals[a].duration()));
    indices
}

/// Assign intervals to slots using first-fit.
fn assign_slots(sorted_indices: &[usize], intervals: &[LivenessInterval]) -> WorkspaceLayout {
    let mut slots: Vec<(u32, Vec<usize>)> = Vec::new();
    let mut assignments = Vec::new();

    for &idx in sorted_indices {
        let slot_id = find_or_create_slot(&mut slots, idx, intervals);
        let node_id = intervals[idx].node_id;
        assignments.push((node_id, slot_id));
    }

    build_layout(slots, intervals, assignments)
}

/// Find a compatible slot or create a new one.
fn find_or_create_slot(
    slots: &mut Vec<(u32, Vec<usize>)>,
    idx: usize,
    intervals: &[LivenessInterval],
) -> u32 {
    for (slot_id, occupants) in slots.iter_mut() {
        if occupants_compatible(occupants, idx, intervals) {
            occupants.push(idx);
            return *slot_id;
        }
    }
    let slot_id = slots.len() as u32;
    slots.push((slot_id, vec![idx]));
    slot_id
}

/// Check if an interval is compatible with all occupants.
fn occupants_compatible(
    occupants: &[usize],
    candidate: usize,
    intervals: &[LivenessInterval],
) -> bool {
    occupants
        .iter()
        .all(|&occ| can_share(&intervals[occ], &intervals[candidate]))
}

/// Convert internal representation to WorkspaceLayout.
fn build_layout(
    slots: Vec<(u32, Vec<usize>)>,
    intervals: &[LivenessInterval],
    assignments: Vec<(NodeId, u32)>,
) -> WorkspaceLayout {
    let total_slots = slots.len();
    let buffer_slots = slots
        .into_iter()
        .map(|(slot_id, occ_indices)| BufferSlot {
            slot_id,
            occupants: occ_indices.iter().map(|&i| intervals[i].node_id).collect(),
        })
        .collect();
    WorkspaceLayout {
        slots: buffer_slots,
        total_slots,
        assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(n: u32) -> NodeId {
        NodeId::new(n, 0)
    }

    fn iv(id: u32, born: usize, dies: usize) -> LivenessInterval {
        LivenessInterval {
            node_id: nid(id),
            born,
            dies,
        }
    }

    #[test]
    fn empty_intervals() {
        let layout = plan_workspace(&[]);
        assert_eq!(layout.total_slots, 0);
        assert!(layout.slots.is_empty());
        assert!(layout.assignments.is_empty());
    }

    #[test]
    fn single_interval() {
        let intervals = vec![iv(0, 0, 2)];
        let layout = plan_workspace(&intervals);
        assert_eq!(layout.total_slots, 1);
        assert_eq!(layout.assignments.len(), 1);
    }

    #[test]
    fn non_overlapping_share_slot() {
        // A: [0,1], B: [2,3] — non-overlapping, can share
        let intervals = vec![iv(0, 0, 1), iv(1, 2, 3)];
        let layout = plan_workspace(&intervals);
        assert_eq!(layout.total_slots, 1);
    }

    #[test]
    fn overlapping_separate_slots() {
        // A: [0,2], B: [1,3] — overlapping, need separate slots
        let intervals = vec![iv(0, 0, 2), iv(1, 1, 3)];
        let layout = plan_workspace(&intervals);
        assert_eq!(layout.total_slots, 2);
    }

    #[test]
    fn all_simultaneous_needs_n_slots() {
        // All born at 0, die at 0 — all simultaneous
        let intervals = vec![iv(0, 0, 0), iv(1, 0, 0), iv(2, 0, 0)];
        let layout = plan_workspace(&intervals);
        assert_eq!(layout.total_slots, 3);
    }

    #[test]
    fn sequential_chain_one_slot() {
        // A: [0,0], B: [1,1], C: [2,2] — sequential, 1 slot
        let intervals = vec![iv(0, 0, 0), iv(1, 1, 1), iv(2, 2, 2)];
        let layout = plan_workspace(&intervals);
        assert_eq!(layout.total_slots, 1);
    }

    #[test]
    fn all_nodes_assigned() {
        let intervals = vec![iv(0, 0, 2), iv(1, 1, 3), iv(2, 3, 5)];
        let layout = plan_workspace(&intervals);
        assert_eq!(layout.assignments.len(), 3);
    }

    #[test]
    fn can_share_non_overlapping() {
        assert!(can_share(&iv(0, 0, 1), &iv(1, 2, 3)));
        assert!(can_share(&iv(0, 2, 3), &iv(1, 0, 1)));
    }

    #[test]
    fn can_share_overlapping() {
        assert!(!can_share(&iv(0, 0, 2), &iv(1, 1, 3)));
        assert!(!can_share(&iv(0, 0, 2), &iv(1, 2, 3)));
    }

    #[test]
    fn diamond_pattern() {
        // A: [0,1], B: [1,2], C: [1,2], D: [2,2]
        let intervals = vec![iv(0, 0, 1), iv(1, 1, 2), iv(2, 1, 2), iv(3, 2, 2)];
        let layout = plan_workspace(&intervals);
        assert!(layout.total_slots <= 3);
        assert!(layout.total_slots >= 2);
    }

    #[test]
    fn slot_occupants_correct() {
        let intervals = vec![iv(0, 0, 0), iv(1, 1, 1)];
        let layout = plan_workspace(&intervals);
        assert_eq!(layout.total_slots, 1);
        assert_eq!(layout.slots[0].occupants.len(), 2);
    }

    #[test]
    fn assignments_reference_valid_slots() {
        let intervals = vec![iv(0, 0, 2), iv(1, 1, 3)];
        let layout = plan_workspace(&intervals);
        for (_, slot_id) in &layout.assignments {
            assert!((*slot_id as usize) < layout.total_slots);
        }
    }
}
