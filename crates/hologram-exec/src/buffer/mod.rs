//! Buffer management for graph execution intermediates.

pub mod arena;
pub mod mmap_buf;
pub mod shape_map;

pub use arena::BufferArena;
pub use shape_map::ShapeMap;
