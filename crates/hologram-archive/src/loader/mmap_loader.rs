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

    /// Raw bytes of the memory-mapped archive.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
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
}
