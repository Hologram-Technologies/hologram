//! Inline constant store.

use alloc::vec::Vec;
use crate::node::ConstantId;
use crate::registry::{DTypeId, ShapeId};

#[derive(Debug, Clone)]
pub struct ConstantEntry {
    pub bytes: Vec<u8>,
    pub dtype: DTypeId,
    pub shape: ShapeId,
}

#[derive(Debug, Default, Clone)]
pub struct ConstantStore {
    entries: Vec<ConstantEntry>,
}

impl ConstantStore {
    pub fn new() -> Self { Self::default() }

    pub fn insert(&mut self, entry: ConstantEntry) -> ConstantId {
        let id = ConstantId(self.entries.len() as u32);
        self.entries.push(entry);
        id
    }

    pub fn get(&self, id: ConstantId) -> Option<&ConstantEntry> {
        self.entries.get(id.0 as usize)
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}
