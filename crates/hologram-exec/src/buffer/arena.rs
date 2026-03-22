//! Arena-based buffer storage for graph execution intermediates.

use hologram_graph::graph::node::NodeId;

use crate::error::{ExecError, ExecResult};

/// Buffer storage variant — supports CPU-owned, borrowed, and GPU-backed buffers.
///
/// On Apple Silicon (unified memory), `Metal` buffers are CPU-accessible via
/// `contents()` — the GPU and CPU share the same physical RAM. This enables
/// zero-copy between arena storage and Metal compute kernels.
enum ArenaBuffer<'a> {
    /// CPU-allocated owned buffer (computed dispatch results).
    Owned(Vec<u8>),
    /// Borrowed reference to external memory (mmap'd weights, constants).
    Borrowed(&'a [u8]),
    /// Metal GPU buffer (shared memory on Apple Silicon).
    /// CPU-readable via `contents()` pointer — zero-copy for both directions.
    #[cfg(has_metal)]
    Metal(metal::Buffer),
}

impl<'a> ArenaBuffer<'a> {
    /// Get a byte slice view of the buffer contents.
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            ArenaBuffer::Owned(v) => v,
            ArenaBuffer::Borrowed(s) => s,
            #[cfg(has_metal)]
            ArenaBuffer::Metal(buf) => {
                let ptr = buf.contents() as *const u8;
                unsafe { std::slice::from_raw_parts(ptr, buf.length() as usize) }
            }
        }
    }

    /// Convert to owned bytes (copies borrowed/Metal data).
    fn into_owned(self) -> Vec<u8> {
        match self {
            ArenaBuffer::Owned(v) => v,
            ArenaBuffer::Borrowed(s) => s.to_vec(),
            #[cfg(has_metal)]
            ArenaBuffer::Metal(buf) => {
                let ptr = buf.contents() as *const u8;
                let len = buf.length() as usize;
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            }
        }
    }
}

/// Arena that stores output buffers keyed by `NodeId`.
///
/// Uses flat `Vec` indexing by `NodeId::index()` instead of `HashMap` for
/// O(1) lookup without hashing overhead. This is safe because node indices
/// are dense sequential integers assigned by the graph builder.
///
/// Buffers can be owned (CPU `Vec<u8>`), borrowed (mmap'd `&[u8]`), or
/// Metal GPU buffers (shared memory on Apple Silicon). Reading always
/// returns `&[u8]` regardless of backing storage.
///
/// Each buffer also tracks its element size in bytes (4 for f32, 8 for i64,
/// 1 for bool/u8). This eliminates all hardcoded `/4` assumptions in shape
/// validation — the arena is the single source of truth for element sizes.
pub struct BufferArena<'a> {
    /// Flat buffer storage indexed by NodeId::index().
    buffers: Vec<Option<ArenaBuffer<'a>>>,
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
        self.buffers[idx] = Some(ArenaBuffer::Owned(data));
    }

    /// Insert an owned buffer with a known element size.
    pub fn insert_with_elem_size(&mut self, id: NodeId, data: Vec<u8>, elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        self.buffers[idx] = Some(ArenaBuffer::Owned(data));
        self.elem_sizes[idx] = elem_size as u8;
    }

    /// Swap-insert: take ownership of `buf`'s allocation and recycle the
    /// previously stored buffer back into `buf`.
    ///
    /// After warmup, this enables zero-allocation tape execution: the kernel
    /// writes into `buf`, the arena takes it, and `buf` receives the old
    /// occupant's allocation for the next instruction.
    pub fn swap_insert_with_elem_size(&mut self, id: NodeId, buf: &mut Vec<u8>, elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        // Take buf's data, give it to the arena.
        let new_data = std::mem::take(buf);
        let old = self.buffers[idx].replace(ArenaBuffer::Owned(new_data));
        // Recycle the old buffer's allocation into buf (if it was owned).
        if let Some(ArenaBuffer::Owned(mut old_vec)) = old {
            old_vec.clear();
            *buf = old_vec;
        }
        self.elem_sizes[idx] = elem_size as u8;
    }

    /// Insert a borrowed buffer for the given node (zero-copy).
    pub fn insert_borrowed(&mut self, id: NodeId, data: &'a [u8]) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        self.buffers[idx] = Some(ArenaBuffer::Borrowed(data));
    }

    /// Insert a borrowed buffer with a known element size.
    pub fn insert_borrowed_with_elem_size(&mut self, id: NodeId, data: &'a [u8], elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        self.buffers[idx] = Some(ArenaBuffer::Borrowed(data));
        self.elem_sizes[idx] = elem_size as u8;
    }

    /// Insert a Metal GPU buffer (zero-copy on Apple Silicon unified memory).
    #[cfg(has_metal)]
    pub fn insert_metal(&mut self, id: NodeId, buffer: metal::Buffer, elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        self.buffers[idx] = Some(ArenaBuffer::Metal(buffer));
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
            if let Some(ref buf) = self.buffers[idx] {
                return Ok(buf.as_bytes());
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

    /// Move a buffer from one slot to another without copying data.
    ///
    /// Used by Output passthrough: when the input has a single consumer,
    /// the buffer can be moved directly instead of copying through `out_buf`.
    pub fn move_slot(&mut self, src: NodeId, dst: NodeId) {
        let src_idx = src.index() as usize;
        let dst_idx = dst.index() as usize;
        self.ensure_capacity(dst_idx);
        if src_idx < self.buffers.len() {
            let buf = self.buffers[src_idx].take();
            if buf.is_some() {
                if dst_idx >= self.buffers.len() || self.buffers[dst_idx].is_none() {
                    self.count += 1;
                }
                self.buffers[dst_idx] = buf;
                let es = self.elem_sizes[src_idx];
                self.elem_sizes[dst_idx] = es;
                // src slot is now empty.
                self.count -= 1;
            }
        }
    }

    /// Remove and return the buffer for the given node as owned bytes.
    pub fn take(&mut self, id: NodeId) -> ExecResult<Vec<u8>> {
        let idx = id.index() as usize;
        if idx < self.buffers.len() {
            if let Some(buf) = self.buffers[idx].take() {
                self.count -= 1;
                return Ok(buf.into_owned());
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
            if let Some(buf) = slot {
                let id = NodeId::new(idx as u32, 0);
                let es = if idx < self.elem_sizes.len() && self.elem_sizes[idx] > 0 {
                    self.elem_sizes[idx] as usize
                } else {
                    4
                };
                map.insert(id, (buf.as_bytes().to_vec(), es));
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
