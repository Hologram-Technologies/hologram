//! Memory-mapped archive loader.

use std::path::Path;

use memmap2::Mmap;

use crate::error::{ArchiveError, ArchiveResult};
use crate::loader::bytes::load_from_bytes;
use crate::loader::plan::LoadedPlan;

/// Memory-mapped archive loader.
///
/// Opens a `.holo` file and memory-maps it for zero-copy access.
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
    pub fn load(&self) -> ArchiveResult<LoadedPlan> {
        load_from_bytes(&self.mmap)
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
    use holo_graph::graph::GraphOp;
    use holo_graph::Graph;
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
