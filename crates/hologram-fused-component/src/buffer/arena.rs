//! Arena-based buffer storage for graph execution intermediates.

use hologram_ir::graph::node::NodeId;

use crate::error::{ExecError, ExecResult};

use super::mmap_buf::MmapBuffer;

/// Buffer storage variant — supports CPU-owned, borrowed, and GPU-backed buffers.
///
/// `Owned` uses `MmapBuffer` (anonymous mmap on Unix, Vec on WASM) so that
/// dropping a buffer returns pages to the OS immediately — no allocator
/// fragmentation. This is critical for vision models where Conv2d activations
/// at 512×512 can be 512MB each.
/// Size threshold below which outputs are stored as Vec (no mmap syscall).
/// Above this, mmap is used so pages return to OS on eviction (zero fragmentation).
/// 256 KB: below L2 cache; mmap overhead (~2-5 µs) exceeds memcpy cost at this size.
const MMAP_THRESHOLD: usize = 256 * 1024;

enum ArenaBuffer<'a> {
    /// CPU-allocated owned buffer (mmap anonymous pages on Unix).
    /// Pages returned to OS on drop via munmap — zero fragmentation.
    Owned(MmapBuffer),
    /// Small CPU-owned buffer stored as Vec (no mmap syscall).
    /// Used for outputs below `MMAP_THRESHOLD` to avoid mmap/munmap overhead
    /// that dominates execution time for small tensors.
    VecOwned(Vec<u8>),
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
            ArenaBuffer::Owned(m) => m.as_slice(),
            ArenaBuffer::VecOwned(v) => v.as_slice(),
            ArenaBuffer::Borrowed(s) => s,
            #[cfg(has_metal)]
            ArenaBuffer::Metal(buf) => {
                let ptr = buf.contents() as *const u8;
                unsafe { std::slice::from_raw_parts(ptr, buf.length() as usize) }
            }
        }
    }

    /// Convert to owned Vec<u8> (copies mmap/borrowed/Metal data).
    fn into_owned(self) -> Vec<u8> {
        match self {
            ArenaBuffer::Owned(m) => m.into_vec(),
            ArenaBuffer::VecOwned(v) => v,
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
    /// Per-buffer tensor metadata (shape, dtype). Parallel to `buffers`.
    /// `None` = no metadata available (legacy path, infer from buffer size).
    metas: Vec<Option<hologram_core::op::TensorMeta>>,
    /// Number of populated slots.
    count: usize,
    /// Free-list of recycled MmapBuffers. Evicted large buffers are pushed here
    /// instead of being dropped, so `swap_insert` can reuse them without syscalls.
    free_mmaps: Vec<MmapBuffer>,
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
            metas: Vec::new(),
            count: 0,
            free_mmaps: Vec::new(),
        }
    }

    /// Create an arena with pre-allocated capacity for `cap` node slots.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let mut buffers = Vec::with_capacity(cap);
        buffers.resize_with(cap, || None);
        let elem_sizes = vec![0u8; cap];
        let metas = vec![None; cap];
        Self {
            buffers,
            elem_sizes,
            metas,
            count: 0,
            free_mmaps: Vec::new(),
        }
    }

    /// Ensure the arena has room for the given index.
    #[inline]
    fn ensure_capacity(&mut self, idx: usize) {
        if idx >= self.buffers.len() {
            let new_len = (idx + 1).max(self.buffers.len() * 2);
            self.buffers.resize_with(new_len, || None);
            self.elem_sizes.resize(new_len, 0);
            self.metas.resize(new_len, None);
        }
    }

    /// Insert an owned buffer for the given node.
    /// Small buffers stored as Vec (no syscall); large as MmapBuffer (pages return to OS).
    pub fn insert(&mut self, id: NodeId, data: Vec<u8>) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        if data.len() < MMAP_THRESHOLD {
            self.buffers[idx] = Some(ArenaBuffer::VecOwned(data));
        } else {
            self.buffers[idx] = Some(ArenaBuffer::Owned(MmapBuffer::from_vec(data)));
        }
    }

    /// Insert an owned buffer with a known element size.
    pub fn insert_with_elem_size(&mut self, id: NodeId, data: Vec<u8>, elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        if data.len() < MMAP_THRESHOLD {
            self.buffers[idx] = Some(ArenaBuffer::VecOwned(data));
        } else {
            self.buffers[idx] = Some(ArenaBuffer::Owned(MmapBuffer::from_vec(data)));
        }
        self.elem_sizes[idx] = elem_size as u8;
    }

    /// Swap-insert: store `buf`'s data in the arena and drop the previous occupant.
    ///
    /// Small outputs (< MMAP_THRESHOLD): takes Vec ownership directly — no mmap
    /// syscall, no copy. This eliminates the mmap/munmap overhead that dominates
    /// execution time for small activation tensors.
    ///
    /// Large outputs (≥ MMAP_THRESHOLD): copies into mmap so pages return to OS
    /// on eviction via munmap — zero fragmentation for large activations.
    ///
    /// `buf` is left empty after the call so the caller can reuse it.
    pub fn swap_insert_with_elem_size(&mut self, id: NodeId, buf: &mut Vec<u8>, elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        let len = buf.len();
        if len < MMAP_THRESHOLD {
            // Small: take Vec ownership — O(1), no syscall.
            let data = std::mem::take(buf);
            self.buffers[idx] = Some(ArenaBuffer::VecOwned(data));
        } else {
            // Large: try to reuse a recycled MmapBuffer from the free-list.
            // Pick the first one that's large enough (avoids mmap/munmap syscall).
            let mmap = {
                let pos = self.free_mmaps.iter().position(|m| m.len() >= len);
                if let Some(pos) = pos {
                    let mut recycled = self.free_mmaps.swap_remove(pos);
                    recycled.as_mut_slice()[..len].copy_from_slice(buf);
                    recycled
                } else {
                    let mut fresh = MmapBuffer::new(len);
                    fresh.as_mut_slice().copy_from_slice(buf);
                    fresh
                }
            };
            buf.clear();
            if buf.capacity() > 64 * 1024 {
                buf.shrink_to(4096);
            }
            self.buffers[idx] = Some(ArenaBuffer::Owned(mmap));
        }
        self.elem_sizes[idx] = elem_size as u8;
        if idx < self.metas.len() {
            self.metas[idx] = Some(hologram_core::op::TensorMeta::infer_1d(len, elem_size));
        }
    }

    /// Swap-insert with a pre-allocated MmapBuffer (zero-copy into arena).
    ///
    /// Use when the output size is known upfront. The kernel writes directly
    /// into the MmapBuffer's slice, then this method moves it into the arena.
    pub fn swap_insert_mmap(&mut self, id: NodeId, mmap: MmapBuffer, elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        let len = mmap.len();
        self.buffers[idx] = Some(ArenaBuffer::Owned(mmap));
        self.elem_sizes[idx] = elem_size as u8;
        if idx < self.metas.len() {
            self.metas[idx] = Some(hologram_core::op::TensorMeta::infer_1d(len, elem_size));
        }
    }

    /// Insert with explicit tensor metadata.
    pub fn swap_insert_with_meta(
        &mut self,
        id: NodeId,
        buf: &mut Vec<u8>,
        meta: hologram_core::op::TensorMeta,
    ) {
        let elem_size = meta.dtype.byte_size();
        self.swap_insert_with_elem_size(id, buf, elem_size);
        let idx = id.index() as usize;
        if idx < self.metas.len() {
            self.metas[idx] = Some(meta);
        }
    }

    /// Set tensor metadata for a node (overwrites any inferred 1-D metadata).
    pub fn set_meta(&mut self, id: NodeId, meta: hologram_core::op::TensorMeta) {
        let idx = id.index() as usize;
        if idx < self.metas.len() {
            self.metas[idx] = Some(meta);
        }
    }

    /// Get tensor metadata for a node.
    pub fn get_meta(&self, id: NodeId) -> Option<&hologram_core::op::TensorMeta> {
        let idx = id.index() as usize;
        self.metas.get(idx).and_then(|m| m.as_ref())
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
    ///
    /// If the buffer is not aligned to `elem_size` bytes (e.g., f32 requires
    /// 4-byte alignment), copies to an owned aligned buffer instead.
    pub fn insert_borrowed_with_elem_size(&mut self, id: NodeId, data: &'a [u8], elem_size: usize) {
        let idx = id.index() as usize;
        self.ensure_capacity(idx);
        if self.buffers[idx].is_none() {
            self.count += 1;
        }
        // Ensure alignment: if the borrowed slice isn't aligned to elem_size,
        // copy to an owned Vec<u8> (which the allocator guarantees is aligned).
        if elem_size > 1 && !(data.as_ptr() as usize).is_multiple_of(elem_size) {
            self.buffers[idx] = Some(ArenaBuffer::VecOwned(data.to_vec()));
        } else {
            self.buffers[idx] = Some(ArenaBuffer::Borrowed(data));
        }
        self.elem_sizes[idx] = elem_size as u8;
        if idx < self.metas.len() {
            self.metas[idx] = Some(hologram_core::op::TensorMeta::infer_1d(
                data.len(),
                elem_size,
            ));
        }
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

    /// Get the buffer for the given node as a typed f32 slice.
    ///
    /// Single bytemuck cast localized at the arena layer — callers work
    /// with native `&[f32]` without per-call casts in hot kernel loops.
    #[inline]
    pub fn get_f32(&self, id: NodeId) -> ExecResult<&[f32]> {
        let bytes = self.get(id)?;
        if bytes.is_empty() {
            return Ok(&[]);
        }
        Ok(bytemuck::cast_slice(bytes))
    }

    /// Get a mutable f32 slice for in-place ops (only works on `Owned` buffers).
    ///
    /// Returns an error for `Borrowed` or `Metal` buffers since those
    /// cannot be modified in-place.
    #[inline]
    pub fn get_mut_f32(&mut self, id: NodeId) -> ExecResult<&mut [f32]> {
        let idx = id.index() as usize;
        if idx < self.buffers.len() {
            match self.buffers[idx] {
                Some(ArenaBuffer::Owned(ref mut m)) => {
                    return Ok(bytemuck::cast_slice_mut(m.as_mut_slice()));
                }
                Some(ArenaBuffer::VecOwned(ref mut v)) => {
                    return Ok(bytemuck::cast_slice_mut(v.as_mut_slice()));
                }
                _ => {}
            }
        }
        Err(ExecError::BufferNotReady(id))
    }

    /// Get buffer bytes without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `id.index()` is within the arena's capacity and
    /// the slot at that index is populated (`Some`). This is guaranteed when
    /// the tape builder has validated that all input indices reference nodes
    /// in the graph, and the arena has been seeded with all constants and inputs.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, id: NodeId) -> &[u8] {
        self.buffers
            .get_unchecked(id.index() as usize)
            .as_ref()
            .unwrap_unchecked()
            .as_bytes()
    }

    /// Typed f32 unchecked access — combines `get_unchecked` + `cast_slice`.
    ///
    /// # Safety
    /// Same requirements as [`get_unchecked`].
    #[inline(always)]
    pub unsafe fn get_f32_unchecked(&self, id: NodeId) -> &[f32] {
        let bytes = self.get_unchecked(id);
        // Empty slices from Vec::new() have dangling ptr (0x1) which
        // fails bytemuck alignment checks. Return empty &[f32] directly.
        if bytes.is_empty() {
            return &[];
        }
        bytemuck::cast_slice(bytes)
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
                // Propagate metadata from src to dst.
                if src_idx < self.metas.len() {
                    let meta = self.metas[src_idx].take();
                    if dst_idx < self.metas.len() {
                        self.metas[dst_idx] = meta;
                    }
                }
                // src slot is now empty.
                self.count -= 1;
            }
        }
    }

    /// In-place element-wise add: `lhs[i] += rhs[i % rhs_len]` with broadcast.
    ///
    /// Operates on two different arena slots without allocation. Uses pointer
    /// arithmetic to avoid the simultaneous `&mut`/`&` borrow conflict that
    /// would arise from calling `get_mut_f32` and `get_f32` on the same arena.
    ///
    /// # Safety
    /// Safe as long as `lhs_id != rhs_id` (no aliasing). Panics on same-slot.
    pub fn add_inplace(&mut self, lhs_id: NodeId, rhs_id: NodeId) -> bool {
        let lhs_idx = lhs_id.index() as usize;
        let rhs_idx = rhs_id.index() as usize;
        assert_ne!(
            lhs_idx, rhs_idx,
            "add_inplace: lhs and rhs must be different slots"
        );

        if lhs_idx >= self.buffers.len() || rhs_idx >= self.buffers.len() {
            return false;
        }

        // Get raw pointers to both buffers to bypass borrow checker.
        // Safe because indices are different — no aliasing.
        let (lhs_ptr, lhs_len) = {
            let buf = match &mut self.buffers[lhs_idx] {
                Some(ArenaBuffer::Owned(m)) => m.as_mut_slice(),
                Some(ArenaBuffer::VecOwned(v)) => v.as_mut_slice(),
                _ => return false,
            };
            (buf.as_mut_ptr(), buf.len())
        };

        let (rhs_ptr, rhs_len) = {
            let buf = match &self.buffers[rhs_idx] {
                Some(b) => b.as_bytes(),
                None => return false,
            };
            (buf.as_ptr(), buf.len())
        };

        if lhs_len % 4 != 0 || rhs_len % 4 != 0 || rhs_len == 0 {
            return false;
        }

        let lhs_floats = lhs_len / 4;
        let rhs_floats = rhs_len / 4;

        unsafe {
            let lhs = std::slice::from_raw_parts_mut(lhs_ptr as *mut f32, lhs_floats);
            let rhs = std::slice::from_raw_parts(rhs_ptr as *const f32, rhs_floats);
            for (i, v) in lhs.iter_mut().enumerate() {
                *v += rhs[i % rhs_floats];
            }
        }
        true
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

    /// Drop the buffer for a node, freeing its memory.
    ///
    /// Used by liveness-based eviction: once all consumers of a node
    /// have executed, the node's activation buffer is no longer needed.
    pub fn evict(&mut self, id: NodeId) {
        let idx = id.index() as usize;
        if idx < self.buffers.len() {
            if let Some(buf) = self.buffers[idx].take() {
                self.count -= 1;
                // Recycle large MmapBuffers to the free-list instead of dropping.
                // Avoids mmap/munmap syscalls on reuse.
                if let ArenaBuffer::Owned(mmap) = buf {
                    if mmap.len() >= MMAP_THRESHOLD {
                        self.free_mmaps.push(mmap);
                    }
                }
            }
        }
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

    #[test]
    fn add_inplace_basic() {
        let mut arena = BufferArena::new();
        let lhs: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec();
        let rhs: Vec<u8> = bytemuck::cast_slice(&[10.0f32, 20.0, 30.0, 40.0]).to_vec();
        arena.insert(id(0), lhs);
        arena.insert(id(1), rhs);

        assert!(arena.add_inplace(id(0), id(1)));

        let result: &[f32] = bytemuck::cast_slice(arena.get(id(0)).unwrap());
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn add_inplace_broadcast() {
        let mut arena = BufferArena::new();
        let lhs: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec();
        let rhs: Vec<u8> = bytemuck::cast_slice(&[100.0f32, 200.0]).to_vec();
        arena.insert(id(0), lhs);
        arena.insert(id(1), rhs);

        assert!(arena.add_inplace(id(0), id(1)));

        let result: &[f32] = bytemuck::cast_slice(arena.get(id(0)).unwrap());
        assert_eq!(result, &[101.0, 202.0, 103.0, 204.0]);
    }

    #[test]
    #[should_panic(expected = "add_inplace: lhs and rhs must be different slots")]
    fn add_inplace_same_slot_panics() {
        let mut arena = BufferArena::new();
        arena.insert(id(0), vec![0u8; 16]);
        arena.add_inplace(id(0), id(0));
    }

    #[test]
    fn add_inplace_returns_false_for_borrowed() {
        let arena_data = vec![0u8; 16];
        let mut arena = BufferArena::new();
        arena.insert_borrowed(id(0), &arena_data);
        arena.insert(id(1), vec![0u8; 16]);
        // Borrowed LHS can't be modified in-place.
        assert!(!arena.add_inplace(id(0), id(1)));
    }

    #[test]
    fn evict_recycles_mmap_buffers() {
        let mut arena = BufferArena::new();
        // Insert a large buffer (above MMAP_THRESHOLD = 256KB).
        let big = vec![42u8; 512 * 1024];
        arena.insert(id(0), big);
        assert_eq!(arena.free_mmaps.len(), 0);

        // Evict — should recycle to free-list.
        arena.evict(id(0));
        assert_eq!(arena.free_mmaps.len(), 1);
        assert!(arena.free_mmaps[0].len() >= 512 * 1024);
    }

    #[test]
    fn swap_insert_reuses_recycled_mmap() {
        let mut arena = BufferArena::new();
        // Insert and evict a 512KB buffer to populate the free-list.
        let big1 = vec![1u8; 512 * 1024];
        arena.insert(id(0), big1);
        arena.evict(id(0));
        assert_eq!(arena.free_mmaps.len(), 1);

        // Insert another buffer of same size — should reuse from free-list.
        let mut big2 = vec![2u8; 512 * 1024];
        arena.swap_insert_with_elem_size(id(1), &mut big2, 4);
        assert_eq!(arena.free_mmaps.len(), 0, "free-list should be drained");

        // Verify the data was written correctly.
        let data = arena.get(id(1)).unwrap();
        assert_eq!(data.len(), 512 * 1024);
        assert_eq!(data[0], 2);
    }

    #[test]
    fn small_evict_does_not_recycle() {
        let mut arena = BufferArena::new();
        // Insert a small buffer (below MMAP_THRESHOLD).
        arena.insert(id(0), vec![0u8; 1024]);
        arena.evict(id(0));
        // Small buffers are NOT recycled (they use VecOwned, not MmapBuffer).
        assert_eq!(arena.free_mmaps.len(), 0);
    }
}
