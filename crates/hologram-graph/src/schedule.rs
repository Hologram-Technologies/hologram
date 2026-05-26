//! Execution schedule (spec VI.3).

use crate::node::NodeId;
use alloc::vec::Vec;
use smallvec::SmallVec;

/// Levels of NodeIds executable in parallel.
#[derive(Debug, Default, Clone)]
pub struct Schedule {
    pub levels: Vec<SmallVec<[NodeId; 16]>>,
}

impl Schedule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &SmallVec<[NodeId; 16]>> {
        self.levels.iter()
    }
}
