//! Arena-based buffer storage for graph execution intermediates.

use std::borrow::Cow;

use hologram_graph::graph::node::NodeId;

use crate::error::{ExecError, ExecResult};

/// Arena that stores output buffers keyed by `NodeId`.
///
/// Uses flat `Vec` indexing by `NodeId::index()` instead of `HashMap` for
/// O(1) lookup without hashing overhead. This is safe because node indices
/// are dense sequential integers assigned by the graph builder.
///
/// Buffers are either borrowed (zero-copy from mmap'd weights or
/// inline constants) or owned (computed dispatch results). Reading
/// always returns `&[u8]` regardless of ownership.
///
/// Each buffer also tracks its element size in bytes (4 for f32, 8 for i64,
/// 1 for bool/u8). This eliminates all hardcoded `/4` assumptions in shape
/// validation — the arena is the single source of truth for element sizes.
pub struct BufferArena<'a> {
    /// Flat buffer storage indexed by NodeId::index().
    buffers: Vec<Option<Cow<'a, [u8]>>>,
    /// Element size in bytes per node. 0 means "use default (4)".
    elem_sizes: Vec<u8>,
    /// Number of populated slots.
    count: usize,
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
            buffers: Vec::new(),
            elem_sizes: Vec::new(),
            count: 0,
        }
    }

    /// Create an arena with pre-allocated capacity for `cap` node slots.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let mut buffers = Vec::with_capacity(cap);
        buffers.resize_with(cap, || None);
        let elem_sizes = vec![0u8; cap];
        Self {
            buffers,
            elem_sizes,
            count: 0,
        }
    }

    /// Ensure the arena has room for the given index.
    #[inline]
    fn ensure_capacity(&mut self, idx: usize) {
        if idx >= self.buffers.len() {
            let new_len = (idx + 1).max(self.buffers.len() * 2);
            self.buffers.resize_with(new_len, || None);
            self.elem_sizes.resize(new_len, 0);
        }
    }

    /// Insert an owned buffer for the given node.
    pub fn insert(&mut self, id: NodeId, data: Vec<u8>) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        self.buffers[idx] = Some(Cow::Owned(data));
    }

    /// Insert an owned buffer with a known element size.
    pub fn insert_with_elem_size(&mut self, id: NodeId, data: Vec<u8>, elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        self.buffers[idx] = Some(Cow::Owned(data));
        self.elem_sizes[idx] = elem_size as u8;
    }

    /// Insert a borrowed buffer for the given node (zero-copy).
    pub fn insert_borrowed(&mut self, id: NodeId, data: &'a [u8]) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        self.buffers[idx] = Some(Cow::Borrowed(data));
    }

    /// Insert a borrowed buffer with a known element size.
    pub fn insert_borrowed_with_elem_size(&mut self, id: NodeId, data: &'a [u8], elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        self.buffers[idx] = Some(Cow::Borrowed(data));
        self.elem_sizes[idx] = elem_size as u8;
    }

    /// Set the element size for a node (without changing its buffer).
    pub fn set_elem_size(&mut self, id: NodeId, elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        self.elem_sizes[idx] = elem_size as u8;
    }

    /// Get the element size for a node. Returns 4 (f32) as the default.
    #[must_use]
    pub fn elem_size(&self, id: NodeId) -> usize {
        let idx = id.index() as usize;
        if idx < self.elem_sizes.len() {
            let es = self.elem_sizes[idx] as usize;
            if es > 0 {
                es
            } else {
                4
            }
        } else {
            4
        }
    }

    /// Get the element count for a node: `data.len() / elem_size`.
    pub fn elem_count(&self, id: NodeId) -> ExecResult<usize> {
        let data = self.get(id)?;
        let es = self.elem_size(id);
        Ok(data.len() / es)
    }

    /// Get the buffer for the given node.
    #[inline]
    pub fn get(&self, id: NodeId) -> ExecResult<&[u8]> {
        let idx = id.index() as usize;
        if idx < self.buffers.len() {
            if let Some(ref cow) = self.buffers[idx] {
                return Ok(cow.as_ref());
            }
        }
        Err(ExecError::BufferNotReady(id))
    }

    /// Whether a buffer exists for the given node.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        let idx = id.index() as usize;
        idx < self.buffers.len() && self.buffers[idx].is_some()
    }

    /// Remove and return the buffer for the given node as owned bytes.
    pub fn take(&mut self, id: NodeId) -> ExecResult<Vec<u8>> {
        let idx = id.index() as usize;
        if idx < self.buffers.len() {
            if let Some(cow) = self.buffers[idx].take() {
                self.count -= 1;
                return Ok(cow.into_owned());
            }
        }
        Err(ExecError::BufferNotReady(id))
    }

    /// Number of stored buffers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the arena is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Remove all buffers.
    pub fn clear(&mut self) {
        for slot in &mut self.buffers {
            *slot = None;
        }
        for es in &mut self.elem_sizes {
            *es = 0;
        }
        self.count = 0;
    }

    /// Snapshot all current buffers as owned copies.
    ///
    /// Returns `(data, elem_size)` for each node. This is non-destructive:
    /// buffers remain available in the arena after snapshotting.
    ///
    /// Intended for conformance testing / debugging only — clones all
    /// intermediate results. Feature-gated behind `profile`.
    #[cfg(feature = "profile")]
    pub fn snapshot(&self) -> std::collections::HashMap<NodeId, (Vec<u8>, usize)> {
        let mut map = std::collections::HashMap::new();
        for (idx, slot) in self.buffers.iter().enumerate() {
            if let Some(cow) = slot {
                let id = NodeId::new(idx as u32, 0);
                let es = if idx < self.elem_sizes.len() && self.elem_sizes[idx] > 0 {
                    self.elem_sizes[idx] as usize
                } else {
                    4
                };
                map.insert(id, (cow.to_vec(), es));
            }
        }
        map
    }
}

/// Running activation-range profile for a buffer.
///
/// Records min, max, mean and sample count from buffer data interpreted as f32.
/// Used for profiling activation ranges to guide quantization decisions.
#[derive(Debug, Clone, Copy)]
pub struct ActivationProfile {
    /// Minimum observed f32 value.
    pub min: f32,
    /// Maximum observed f32 value.
    pub max: f32,
    /// Running mean of observed f32 values.
    pub mean: f32,
    /// Total number of f32 samples recorded.
    pub n_samples: usize,
}

impl Default for ActivationProfile {
    fn default() -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            mean: 0.0,
            n_samples: 0,
        }
    }
}

impl ActivationProfile {
    /// Create a new empty profile.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update running statistics from a `&[u8]` buffer interpreted as f32.
    ///
    /// Bytes must be f32-aligned (length divisible by 4). Non-aligned
    /// trailing bytes are silently ignored.
    pub fn record_buffer(&mut self, buf: &[u8]) {
        let floats: &[f32] = match bytemuck::try_cast_slice(buf) {
            Ok(f) => f,
            Err(_) => return,
        };
        for &v in floats {
            if v < self.min {
                self.min = v;
            }
            if v > self.max {
                self.max = v;
            }
            // Incremental mean update: mean += (v - mean) / n
            self.n_samples += 1;
            self.mean += (v - self.mean) / self.n_samples as f32;
        }
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

    #[test]
    fn elem_size_default_is_f32() {
        let arena = BufferArena::new();
        assert_eq!(arena.elem_size(id(0)), 4);
    }

    #[test]
    fn elem_size_tracks_insertions() {
        let mut arena = BufferArena::new();
        // i64 data: 3 elements * 8 bytes = 24 bytes
        arena.insert_with_elem_size(id(0), vec![0u8; 24], 8);
        assert_eq!(arena.elem_size(id(0)), 8);
        assert_eq!(arena.elem_count(id(0)).unwrap(), 3);
    }

    #[test]
    fn set_elem_size_independent() {
        let mut arena = BufferArena::new();
        arena.insert(id(0), vec![0u8; 12]);
        // Default is f32 (4 bytes) → 3 elements
        assert_eq!(arena.elem_count(id(0)).unwrap(), 3);
        // Change to i32 — same 12 bytes, still 3 elements
        arena.set_elem_size(id(0), 4);
        assert_eq!(arena.elem_count(id(0)).unwrap(), 3);
        // Change to u8 — 12 bytes → 12 elements
        arena.set_elem_size(id(0), 1);
        assert_eq!(arena.elem_count(id(0)).unwrap(), 12);
    }
}
