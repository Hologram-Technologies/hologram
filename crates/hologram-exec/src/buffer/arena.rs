//! Arena-based buffer storage for graph execution intermediates.

use hologram_graph::graph::node::NodeId;

// ── Inline f16 conversion (no `half` crate dependency) ──────────────────────

/// Convert f32 → f16 (IEEE 754 half-precision), returning the 16-bit pattern.
#[inline]
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf/NaN
        return (sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // Overflow → Inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16; // Too small → zero
        }
        // Denormalized
        let m = (mantissa | 0x800000) >> (1 - new_exp);
        return (sign | (m >> 13)) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}

/// Convert f16 (IEEE 754 half-precision bit pattern) → f32.
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Denormalized: normalize
        let mut m = mantissa;
        let mut e = 1u32;
        while m & 0x400 == 0 {
            m <<= 1;
            e += 1;
        }
        let m = (m & 0x3FF) << 13;
        let e = (127 - 15 + 1 - e) << 23;
        return f32::from_bits(sign | e | m);
    }
    if exp == 31 {
        let m = if mantissa != 0 { 0x400000 } else { 0 };
        return f32::from_bits(sign | 0x7F800000 | m); // Inf/NaN
    }

    let e = (exp + 127 - 15) << 23;
    let m = mantissa << 13;
    f32::from_bits(sign | e | m)
}

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

/// Threshold for F16 compression of activation buffers.
/// Buffers larger than this (in bytes) are stored as f16 to halve memory.
/// 512 KB — below this, the f16↔f32 conversion overhead isn't worth it.
const F16_COMPRESS_THRESHOLD: usize = 512 * 1024;

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
    /// F16-compressed activation buffer — stores f32 data as f16 to halve memory.
    /// The `f32_len` field tracks the original byte length (f32_len = f16_data.len() * 2).
    /// On read, data is expanded f16→f32 into the arena's scratch buffer.
    F16Compressed { data: MmapBuffer, f32_len: usize },
    /// Metal GPU buffer (shared memory on Apple Silicon).
    /// CPU-readable via `contents()` pointer — zero-copy for both directions.
    #[cfg(has_metal)]
    Metal(metal::Buffer),
}

impl<'a> ArenaBuffer<'a> {
    /// Get a byte slice view of the buffer contents.
    ///
    /// For `F16Compressed` buffers, returns the compressed f16 data.
    /// Callers that need f32 should use `BufferArena::get_f32_or_expand()`.
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            ArenaBuffer::Owned(m) => m.as_slice(),
            ArenaBuffer::VecOwned(v) => v.as_slice(),
            ArenaBuffer::Borrowed(s) => s,
            ArenaBuffer::F16Compressed { data, .. } => data.as_slice(),
            #[cfg(has_metal)]
            ArenaBuffer::Metal(buf) => {
                let ptr = buf.contents() as *const u8;
                unsafe { std::slice::from_raw_parts(ptr, buf.length() as usize) }
            }
        }
    }

    /// Whether this buffer is F16 compressed.
    #[allow(dead_code)]
    fn is_f16_compressed(&self) -> bool {
        matches!(self, ArenaBuffer::F16Compressed { .. })
    }

    /// Original f32 byte length (for F16Compressed, this is 2× the stored data).
    #[allow(dead_code)]
    fn f32_byte_len(&self) -> usize {
        match self {
            ArenaBuffer::F16Compressed { f32_len, .. } => *f32_len,
            other => other.as_bytes().len(),
        }
    }

    /// Convert to owned Vec<u8> (copies mmap/borrowed/Metal data).
    fn into_owned(self) -> Vec<u8> {
        match self {
            ArenaBuffer::Owned(m) => m.into_vec(),
            ArenaBuffer::VecOwned(v) => v,
            ArenaBuffer::Borrowed(s) => s.to_vec(),
            ArenaBuffer::F16Compressed { data, f32_len } => {
                // Expand f16→f32 on conversion.
                let f16_slice = data.as_slice();
                let n_floats = f32_len / 4;
                let mut out = vec![0u8; f32_len];
                let f32_out: &mut [f32] = bytemuck::cast_slice_mut(&mut out);
                let f16_data: &[u16] = bytemuck::cast_slice(&f16_slice[..n_floats * 2]);
                for (dst, &src) in f32_out.iter_mut().zip(f16_data.iter()) {
                    *dst = f16_bits_to_f32(src);
                }
                out
            }
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
            // Large: copy into mmap for OS page reclaim on eviction.
            let mut mmap = MmapBuffer::new(len);
            mmap.as_mut_slice().copy_from_slice(buf);
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

    /// Compress a buffer from f32 → f16 to halve its memory footprint.
    ///
    /// Only compresses Owned/VecOwned buffers that are > F16_COMPRESS_THRESHOLD
    /// and have elem_size == 4 (f32). Borrowed constants, small buffers, and
    /// already-compressed buffers are left unchanged.
    ///
    /// Call after a node's output is stored but before its consumer executes,
    /// when the node has a distant consumer (skip connection).
    pub fn compress_f16(&mut self, id: NodeId) {
        let idx = id.index() as usize;
        if idx >= self.buffers.len() {
            return;
        }
        // Only compress f32 buffers above threshold.
        if self.elem_sizes.get(idx).copied().unwrap_or(0) != 4 {
            return;
        }
        let src_bytes = match &self.buffers[idx] {
            Some(ArenaBuffer::Owned(m)) if m.len() >= F16_COMPRESS_THRESHOLD => m.as_slice(),
            Some(ArenaBuffer::VecOwned(v)) if v.len() >= F16_COMPRESS_THRESHOLD => v.as_slice(),
            _ => return,
        };
        let f32_len = src_bytes.len();
        let n_floats = f32_len / 4;
        let f16_byte_len = n_floats * 2;

        // Convert f32 → f16 into a new smaller buffer.
        let mut f16_buf = MmapBuffer::new(f16_byte_len);
        let src_f32: &[f32] = bytemuck::cast_slice(src_bytes);
        let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(f16_buf.as_mut_slice());
        for (dst, &src) in dst_u16.iter_mut().zip(src_f32.iter()) {
            *dst = f32_to_f16_bits(src);
        }

        self.buffers[idx] = Some(ArenaBuffer::F16Compressed {
            data: f16_buf,
            f32_len,
        });
    }

    /// Expand an F16-compressed buffer back to f32.
    ///
    /// Call before `get()` when a consumer needs the original f32 data.
    /// No-op if the buffer is already f32.
    pub fn expand_f32(&mut self, id: NodeId) {
        let idx = id.index() as usize;
        if idx >= self.buffers.len() {
            return;
        }
        if !matches!(self.buffers[idx], Some(ArenaBuffer::F16Compressed { .. })) {
            return;
        }
        let buf = self.buffers[idx].take().expect("checked Some");
        if let ArenaBuffer::F16Compressed { data, f32_len } = buf {
            let n_floats = f32_len / 4;
            let mut f32_buf = MmapBuffer::new(f32_len);
            let src_u16: &[u16] = bytemuck::cast_slice(data.as_slice());
            let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(f32_buf.as_mut_slice());
            for (dst, &src) in dst_f32.iter_mut().zip(src_u16[..n_floats].iter()) {
                *dst = f16_bits_to_f32(src);
            }
            self.buffers[idx] = Some(ArenaBuffer::Owned(f32_buf));
        }
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
        if idx < self.buffers.len() && self.buffers[idx].take().is_some() {
            self.count -= 1;
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
}
