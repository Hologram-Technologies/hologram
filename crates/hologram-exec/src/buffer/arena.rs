//! Arena-based buffer storage for graph execution intermediates.

use std::borrow::Cow;
use std::collections::HashMap;

use hologram_graph::graph::node::NodeId;

use crate::error::{ExecError, ExecResult};

/// Arena that stores output buffers keyed by `NodeId`.
///
/// Buffers are either borrowed (zero-copy from mmap'd weights or
/// inline constants) or owned (computed dispatch results). Reading
/// always returns `&[u8]` regardless of ownership.
pub struct BufferArena<'a> {
    buffers: HashMap<NodeId, Cow<'a, [u8]>>,
}

impl Default for BufferArena<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> BufferArena<'a> {
    /// Create an empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Create an arena with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buffers: HashMap::with_capacity(cap),
        }
    }

    /// Insert an owned buffer for the given node.
    pub fn insert(&mut self, id: NodeId, data: Vec<u8>) {
        self.buffers.insert(id, Cow::Owned(data));
    }

    /// Insert a borrowed buffer for the given node (zero-copy).
    pub fn insert_borrowed(&mut self, id: NodeId, data: &'a [u8]) {
        self.buffers.insert(id, Cow::Borrowed(data));
    }

    /// Get the buffer for the given node.
    pub fn get(&self, id: NodeId) -> ExecResult<&[u8]> {
        self.buffers
            .get(&id)
            .map(|v| v.as_ref())
            .ok_or(ExecError::BufferNotReady(id))
    }

    /// Whether a buffer exists for the given node.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.buffers.contains_key(&id)
    }

    /// Remove and return the buffer for the given node as owned bytes.
    pub fn take(&mut self, id: NodeId) -> ExecResult<Vec<u8>> {
        self.buffers
            .remove(&id)
            .map(|cow| cow.into_owned())
            .ok_or(ExecError::BufferNotReady(id))
    }

    /// Number of stored buffers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Whether the arena is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Remove all buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u32) -> NodeId {
        NodeId::new(n, 0)
    }

    #[test]
    fn new_is_empty() {
        let arena = BufferArena::new();
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn insert_and_get() {
        let mut arena = BufferArena::new();
        arena.insert(id(0), vec![1, 2, 3]);
        assert_eq!(arena.get(id(0)).unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn insert_borrowed_and_get() {
        let data = vec![4, 5, 6];
        let mut arena = BufferArena::new();
        arena.insert_borrowed(id(0), &data);
        assert_eq!(arena.get(id(0)).unwrap(), &[4, 5, 6]);
    }

    #[test]
    fn get_missing_returns_error() {
        let arena = BufferArena::new();
        assert!(arena.get(id(99)).is_err());
    }

    #[test]
    fn contains_check() {
        let mut arena = BufferArena::new();
        arena.insert(id(1), vec![42]);
        assert!(arena.contains(id(1)));
        assert!(!arena.contains(id(2)));
    }

    #[test]
    fn take_removes_buffer() {
        let mut arena = BufferArena::new();
        arena.insert(id(0), vec![10, 20]);
        let data = arena.take(id(0)).unwrap();
        assert_eq!(data, vec![10, 20]);
        assert!(arena.take(id(0)).is_err());
    }

    #[test]
    fn take_borrowed_clones() {
        let data = vec![10, 20];
        let mut arena = BufferArena::new();
        arena.insert_borrowed(id(0), &data);
        let taken = arena.take(id(0)).unwrap();
        assert_eq!(taken, vec![10, 20]);
    }

    #[test]
    fn with_capacity_works() {
        let arena = BufferArena::with_capacity(100);
        assert!(arena.is_empty());
    }

    #[test]
    fn clear_empties_arena() {
        let mut arena = BufferArena::new();
        arena.insert(id(0), vec![1]);
        arena.insert(id(1), vec![2]);
        assert_eq!(arena.len(), 2);
        arena.clear();
        assert!(arena.is_empty());
    }

    #[test]
    fn multiple_inserts() {
        let mut arena = BufferArena::new();
        for i in 0..10 {
            arena.insert(id(i), vec![i as u8]);
        }
        assert_eq!(arena.len(), 10);
        for i in 0..10 {
            assert_eq!(arena.get(id(i)).unwrap(), &[i as u8]);
        }
    }
}
