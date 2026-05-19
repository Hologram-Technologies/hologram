//! Bounded-residency weight window for constrained execution.
//!
//! Tracks which constants are "resident" (actively referenced) and enforces
//! a hard memory cap. Evicts least-recently-used entries when the cap would
//! be exceeded by a new load.

use hologram_graph::constant::{ConstantId, ConstantStore};
use smallvec::SmallVec;

use crate::error::{ExecError, ExecResult};

/// A single resident weight entry.
#[derive(Debug, Clone)]
struct ResidentWeight {
    id: ConstantId,
    byte_size: usize,
}

/// Bounded-residency weight manager.
///
/// Tracks which weight constants are currently "resident" (their byte ranges
/// are being accessed) and enforces a strict memory cap. When a new weight
/// would exceed the cap, the oldest resident entry is evicted first.
///
/// This does not own weight data — data lives in the mmap'd archive slice.
/// The window only tracks bookkeeping (which IDs are resident and their sizes).
#[derive(Debug)]
pub struct WeightWindow {
    max_bytes: usize,
    current_bytes: usize,
    resident: SmallVec<[ResidentWeight; 8]>,
}

impl WeightWindow {
    /// Create a new weight window with the given byte cap.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current_bytes: 0,
            resident: SmallVec::new(),
        }
    }

    /// Current resident weight memory in bytes.
    #[inline]
    #[must_use]
    pub fn current_usage(&self) -> usize {
        self.current_bytes
    }

    /// Maximum allowed weight memory in bytes.
    #[inline]
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Ensure the given constants are resident, evicting LRU entries if needed.
    ///
    /// For each required constant not already resident, computes its byte size
    /// from the `ConstantStore` and adds it. If the total would exceed `max_bytes`,
    /// evicts the oldest resident entries until there's room.
    ///
    /// Returns `Err` if a single constant exceeds the entire window cap.
    pub fn ensure(&mut self, required: &[ConstantId], constants: &ConstantStore) -> ExecResult<()> {
        for &cid in required {
            // Skip if already resident.
            if self.resident.iter().any(|r| r.id == cid) {
                continue;
            }

            let byte_size = self.constant_byte_size(cid, constants)?;

            // Single constant exceeds entire window — cannot fit.
            if byte_size > self.max_bytes {
                return Err(ExecError::ConstrainedViolation(format!(
                    "constant {:?} is {} bytes, exceeds weight window cap of {} bytes",
                    cid, byte_size, self.max_bytes
                )));
            }

            // Evict oldest entries until there's room.
            while self.current_bytes + byte_size > self.max_bytes && !self.resident.is_empty() {
                let evicted = self.resident.remove(0);
                self.current_bytes -= evicted.byte_size;
            }

            self.resident.push(ResidentWeight { id: cid, byte_size });
            self.current_bytes += byte_size;
        }

        Ok(())
    }

    /// Explicitly evict the given constants (post-op cleanup).
    pub fn evict(&mut self, released: &[ConstantId]) {
        for &cid in released {
            if let Some(pos) = self.resident.iter().position(|r| r.id == cid) {
                let removed = self.resident.remove(pos);
                self.current_bytes -= removed.byte_size;
            }
        }
    }

    /// Look up the byte size of a constant.
    fn constant_byte_size(&self, cid: ConstantId, constants: &ConstantStore) -> ExecResult<usize> {
        let data = constants
            .get(cid)
            .ok_or(ExecError::ConstantNotFound(cid.raw()))?;
        Ok(data.byte_size() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_graph::constant::ConstantData;

    fn make_store(sizes: &[u64]) -> ConstantStore {
        let mut store = ConstantStore::new();
        for &size in sizes {
            store.insert(ConstantData::Deferred {
                byte_size: size,
                source_id: 0,
            });
        }
        store
    }

    #[test]
    fn empty_window() {
        let ww = WeightWindow::new(1024);
        assert_eq!(ww.current_usage(), 0);
        assert_eq!(ww.max_bytes(), 1024);
    }

    #[test]
    fn ensure_loads_constants() {
        let store = make_store(&[100, 200, 300]);
        let mut ww = WeightWindow::new(1024);
        ww.ensure(&[ConstantId::new(0), ConstantId::new(1)], &store)
            .unwrap();
        assert_eq!(ww.current_usage(), 300); // 100 + 200
    }

    #[test]
    fn ensure_deduplicates() {
        let store = make_store(&[100]);
        let mut ww = WeightWindow::new(1024);
        ww.ensure(&[ConstantId::new(0), ConstantId::new(0)], &store)
            .unwrap();
        assert_eq!(ww.current_usage(), 100);
    }

    #[test]
    fn eviction_on_overflow() {
        let store = make_store(&[400, 400, 400]);
        let mut ww = WeightWindow::new(800);
        // Load first two: 400 + 400 = 800 (at cap)
        ww.ensure(&[ConstantId::new(0), ConstantId::new(1)], &store)
            .unwrap();
        assert_eq!(ww.current_usage(), 800);
        // Load third: must evict first to make room
        ww.ensure(&[ConstantId::new(2)], &store).unwrap();
        assert_eq!(ww.current_usage(), 800); // 400 + 400
                                             // First constant should have been evicted
        assert!(!ww.resident.iter().any(|r| r.id == ConstantId::new(0)));
    }

    #[test]
    fn single_constant_too_large() {
        let store = make_store(&[2000]);
        let mut ww = WeightWindow::new(1000);
        let err = ww.ensure(&[ConstantId::new(0)], &store).unwrap_err();
        assert!(matches!(err, ExecError::ConstrainedViolation(_)));
    }

    #[test]
    fn explicit_evict() {
        let store = make_store(&[100, 200]);
        let mut ww = WeightWindow::new(1024);
        ww.ensure(&[ConstantId::new(0), ConstantId::new(1)], &store)
            .unwrap();
        assert_eq!(ww.current_usage(), 300);
        ww.evict(&[ConstantId::new(0)]);
        assert_eq!(ww.current_usage(), 200);
    }

    #[test]
    fn evict_nonexistent_is_noop() {
        let store = make_store(&[100]);
        let mut ww = WeightWindow::new(1024);
        ww.ensure(&[ConstantId::new(0)], &store).unwrap();
        ww.evict(&[ConstantId::new(99)]); // doesn't exist
        assert_eq!(ww.current_usage(), 100);
    }

    #[test]
    fn constant_not_found() {
        let store = make_store(&[]);
        let mut ww = WeightWindow::new(1024);
        let err = ww.ensure(&[ConstantId::new(0)], &store).unwrap_err();
        assert!(matches!(err, ExecError::ConstantNotFound(0)));
    }
}
