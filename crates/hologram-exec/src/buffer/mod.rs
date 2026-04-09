//! Buffer management for graph execution intermediates.

pub mod arena;
pub mod lent;
pub mod mmap_buf;
pub mod scatter_gather;
pub mod shape_map;

pub use arena::BufferArena;
pub use lent::{LentRegion, LentRegionMut, MmapLender};
pub use scatter_gather::ScatterGatherStream;
pub use shape_map::ShapeMap;
