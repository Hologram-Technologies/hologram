//! Backend-side workspace abstraction.
//!
//! The runtime executor (`hologram-exec::BufferArena`) holds the actual
//! memory; this trait surfaces it to backend dispatch in a backend-agnostic
//! way. Per ADR-051 (workspace residency), GPU backends may keep their
//! storage device-resident; the `Workspace` trait handles that uniformly.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferRef {
    pub slot: u32,
    pub offset: u32,
    pub length: u32,
}

pub trait Workspace {
    /// Read-only view of a buffer slot.
    fn read(&self, buf: BufferRef) -> &[u8];
    /// Mutable view of a buffer slot.
    fn write(&mut self, buf: BufferRef) -> &mut [u8];
}
