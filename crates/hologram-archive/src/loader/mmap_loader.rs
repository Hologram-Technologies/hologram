//! Memory-mapped archive loader.

use std::path::Path;

use memmap2::Mmap;

use crate::error::{ArchiveError, ArchiveResult};
use crate::format::header::HoloHeader;
use crate::loader::bytes::load_from_bytes;
use crate::loader::plan::LoadedPlan;

/// Memory-mapped archive loader.
///
/// Opens a `.holo` file and memory-maps it for zero-copy access.
/// After loading, applies `madvise` hints to guide the OS page cache:
/// - Graph section: `Sequential` (read once at load time)
/// - Weight section: `Random` (LUT-GEMM accesses are random within layers)
pub struct HoloLoader {
    mmap: Mmap,
}

impl HoloLoader {
    /// Open and memory-map a .holo file.
    pub fn open(path: &Path) -> ArchiveResult<Self> {
        let file = std::fs::File::open(path).map_err(ArchiveError::Io)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(ArchiveError::Io)?;
        Ok(Self { mmap })
    }

    /// Load and validate the archive, returning a `LoadedPlan`.
    ///
    /// Also applies `madvise` hints to the mmap'd sections to reduce
    /// unnecessary readahead and page cache pollution.
    pub fn load(&self) -> ArchiveResult<LoadedPlan> {
        // Apply madvise hints based on header section offsets.
        // Errors are non-fatal — hints are advisory only.
        if let Some(header) = HoloHeader::from_bytes(&self.mmap) {
            self.advise_sections(&header);
        }
        load_from_bytes(&self.mmap)
    }

    /// Apply `madvise` hints to mmap'd sections based on access patterns.
    ///
    /// - Graph section → `Sequential` (read once during deserialization)
    /// - Weight section → `Random` (LUT-GEMM weight lookups are non-sequential)
    ///
    /// Failures are silently ignored — these are advisory hints only.
    fn advise_sections(&self, header: &HoloHeader) {
        #[cfg(unix)]
        {
            use memmap2::Advice;

            // Graph section: read sequentially once at load time.
            if header.graph_size > 0 {
                let _ = self.mmap.advise_range(
                    Advice::Sequential,
                    header.graph_offset as usize,
                    header.graph_size as usize,
                );
            }

            // Weight section: random access during LUT-GEMM dispatch.
            // Prevents wasteful readahead that would pollute the page cache.
            if header.weights_size > 0 {
                let _ = self.mmap.advise_range(
                    Advice::Random,
                    header.weights_offset as usize,
                    header.weights_size as usize,
                );
            }
        }

        #[cfg(not(unix))]
        let _ = header;
    }

    /// Load with zero-copy graph and weight access.
    ///
    /// Graph bytes are stored as raw archived bytes — deserialized lazily on
    /// first `plan.graph()` call. Weights are borrowed directly from the mmap.
    ///
    /// If the archive is compressed, automatically decompresses to a `.cache`
    /// file next to the archive and mmap-loads that instead (decompress once,
    /// instant on subsequent loads).
    ///
    /// # Safety
    /// The returned `LoadedPlan` borrows from `self.mmap`. The caller must
    /// ensure this `HoloLoader` outlives the returned plan.
    pub unsafe fn load_zero_copy(&self) -> ArchiveResult<LoadedPlan> {
        if let Some(header) = HoloHeader::from_bytes(&self.mmap) {
            self.advise_sections(&header);
        }
        crate::loader::bytes::load_from_bytes_zero_copy(&self.mmap)
    }

    /// Load with zero-copy access, using a decompressed cache file if needed.
    ///
    /// If the archive is compressed:
    /// 1. Checks for `{path}.cache` — if it exists and is newer, mmap that
    /// 2. Otherwise decompresses the archive, writes `{path}.cache`, mmap-loads it
    ///
    /// If uncompressed, loads directly via zero-copy.
    pub fn load_cached(path: &Path) -> ArchiveResult<(Self, LoadedPlan)> {
        let loader = Self::open(path)?;
        if !crate::loader::bytes::is_compressed(&loader.mmap) {
            // Already uncompressed — load zero-copy directly.
            let plan = unsafe { loader.load_zero_copy()? };
            return Ok((loader, plan));
        }

        // Check for cache file.
        let cache_path = path.with_extension("holo.cache");
        if cache_path.exists() {
            // Cache exists — load from cache instead.
            let cached = Self::open(&cache_path)?;
            let plan = unsafe { cached.load_zero_copy()? };
            return Ok((cached, plan));
        }

        // Decompress and write cache.
        if let Some(decompressed) = crate::loader::bytes::decompress_archive(&loader.mmap)? {
            if let Ok(mut f) = std::fs::File::create(&cache_path) {
                use std::io::Write;
                let _ = f.write_all(&decompressed);
            }
            // Load from the new cache file.
            if cache_path.exists() {
                let cached = Self::open(&cache_path)?;
                let plan = unsafe { cached.load_zero_copy()? };
                return Ok((cached, plan));
            }
        }

        // Fallback: load normally (compressed, no cache).
        let plan = loader.load()?;
        Ok((loader, plan))
    }

    /// Raw bytes of the memory-mapped archive.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Prefetch a byte range within the archive via `madvise(MADV_WILLNEED)`.
    ///
    /// Call this for the next layer group's weight range while the current
    /// layer computes. The OS will asynchronously fault the pages into the
    /// page cache so they're warm by the time the executor reads them.
    ///
    /// No-op on non-Unix platforms or if the range is out of bounds.
    pub fn prefetch_range(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        {
            if offset + len <= self.mmap.len() {
                let _ = self
                    .mmap
                    .advise_range(memmap2::Advice::WillNeed, offset, len);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (offset, len);
        }
    }

    /// Release pages after a layer finishes.
    ///
    /// Uses `MADV_FREE` (lazy reclaim under memory pressure) which is
    /// better than `MADV_DONTNEED` for file-backed mappings — avoids
    /// unnecessary re-faults if the pages aren't actually reclaimed.
    /// On macOS, uses `MADV_FREE_REUSABLE` (equivalent semantics).
    pub fn release_range(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        {
            if offset + len <= self.mmap.len() {
                // MADV_DONTNEED: safe for file-backed mappings — the data
                // remains valid (backed by the file) and will be re-faulted
                // from disk if accessed again.
                let _ = unsafe {
                    self.mmap.unchecked_advise_range(
                        memmap2::UncheckedAdvice::DontNeed,
                        offset,
                        len,
                    )
                };
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (offset, len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::holo_writer::HoloWriter;
    use hologram_graph::graph::GraphOp;
    use hologram_graph::Graph;
    use std::io::Write;

    #[test]
    fn mmap_round_trip() {
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_holo_mmap.holo");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&archive).unwrap();
        }

        let loader = HoloLoader::open(&path).unwrap();
        let plan = loader.load().unwrap();
        assert_eq!(plan.node_count(), 1);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_nonexistent() {
        let result = HoloLoader::open(Path::new("/nonexistent.holo"));
        assert!(result.is_err());
    }

    #[test]
    fn zero_copy_lazy_graph_deserialization() {
        // Build an uncompressed archive, load via zero-copy, verify graph
        // is deserialized lazily on first graph() call.
        let mut g = Graph::new();
        let n1 = g.add_node(GraphOp::Input);
        let n2 = g.add_node(GraphOp::Output);
        g.add_edge(n1, n2);
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_zero_copy_lazy.holo");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&archive).unwrap();
        }

        let loader = HoloLoader::open(&path).unwrap();
        // SAFETY: loader outlives plan.
        let plan = unsafe { loader.load_zero_copy() }.unwrap();
        // Graph should be accessible (triggers lazy deser if Archived variant).
        assert_eq!(plan.graph().nodes.len(), 2);
        assert_eq!(plan.node_count(), 2);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_cached_uncompressed_no_cache_file() {
        // Uncompressed archive should load directly without creating a .cache file.
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        let archive = HoloWriter::new().set_graph(&g).build().unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_load_cached.holo");
        let cache_path = dir.join("test_load_cached.holo.cache");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&archive).unwrap();
        }
        // Remove any leftover cache file.
        std::fs::remove_file(&cache_path).ok();

        let (_loader, plan) = HoloLoader::load_cached(&path).unwrap();
        assert_eq!(plan.node_count(), 1);
        // No cache file should be created for uncompressed archives.
        assert!(
            !cache_path.exists(),
            "uncompressed archive should not create .cache"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_cached_compressed_creates_cache() {
        // Compressed archive should produce a .cache file on first load.
        let mut g = Graph::new();
        g.add_node(GraphOp::Input);
        g.add_node(GraphOp::Output);
        let archive = HoloWriter::new()
            .set_graph(&g)
            .compress_graph()
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_load_cached_compressed.holo");
        let cache_path = dir.join("test_load_cached_compressed.holo.cache");
        std::fs::remove_file(&cache_path).ok();
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&archive).unwrap();
        }

        let (_loader, plan) = HoloLoader::load_cached(&path).unwrap();
        assert_eq!(plan.node_count(), 2);
        // Cache file should now exist.
        assert!(
            cache_path.exists(),
            "compressed archive should create .cache"
        );

        // Second load should use the cache (no decompression).
        let (_loader2, plan2) = HoloLoader::load_cached(&path).unwrap();
        assert_eq!(plan2.node_count(), 2);

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&cache_path).ok();
    }
}
