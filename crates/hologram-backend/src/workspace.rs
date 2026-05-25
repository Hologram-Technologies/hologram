//! Backend-side workspace abstraction.
//!
//! The runtime executor (`hologram-exec::BufferArena`) holds the actual
//! memory; this trait surfaces it to backend dispatch in a backend-agnostic
//! way. Per ADR-051 (workspace residency), GPU backends may keep their
//! storage device-resident; the `Workspace` trait handles that uniformly.

use smallvec::SmallVec;

/// Read-slice set returned by [`Workspace::split_borrow`]. Inline-stacked
/// for up to 4 operands (every kernel reads ≤ 4), so the disjoint borrow on
/// the per-kernel hot path allocates nothing (the zero-cost contract).
pub type SplitReads<'a> = SmallVec<[&'a [u8]; 4]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferRef {
    pub slot: u32,
    /// Byte offset within the slot. `u64` so tensors/workspaces beyond
    /// 4 GiB don't overflow (ADR-060: no fixed byte ceiling).
    pub offset: u64,
    /// Byte length. `u64` for the same reason; `0` means "the slot's full
    /// extent" (kernels compute their own byte count from shape + dtype).
    pub length: u64,
}

pub trait Workspace {
    /// Read-only view of a buffer slot.
    fn read(&self, buf: BufferRef) -> &[u8];
    /// Mutable view of a buffer slot.
    fn write(&mut self, buf: BufferRef) -> &mut [u8];

    /// Zero-copy disjoint borrow: return `&[u8]` slices for each read
    /// buffer plus an `&mut [u8]` slice for the single write buffer,
    /// all backed by the same workspace storage. Required for every
    /// `Workspace` consumed by CPU compute — the kernels never fall
    /// back to a `read.to_vec() + write` clone path, so any
    /// `Workspace` impl that runs through `hologram_backend::cpu` MUST
    /// supply this. GPU-resident workspaces that bridge to CPU
    /// fallback paths maintain a host-shadow `BufferArena` and
    /// delegate; pure GPU workspaces that never call CPU kernels can
    /// ignore the requirement.
    ///
    /// Returns `None` only when the requested ranges overlap or are
    /// out-of-range (a runtime bug). CPU kernels treat `None` as a
    /// programming error and propagate `SlotOutOfRange`.
    ///
    /// Callers are responsible for supplying disjoint buffers; the
    /// schedule's per-level independence (spec VIII.2) guarantees
    /// slot-level disjointness in the executor's call stream.
    fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(SplitReads<'a>, &'a mut [u8])>;
}
