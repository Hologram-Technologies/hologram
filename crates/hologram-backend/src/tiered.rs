//! Tiered execution support (PM_7 backend integration).
//!
//! Provides unified-memory detection and migration helpers for backends
//! participating in the `HybridExecutor` pipeline.

use hologram_types::MemoryTier;

/// Information about the memory architecture of the active device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryArchitecture {
    /// True if CPU and device share physical memory (e.g. Apple Silicon).
    /// When true, migrations are no-ops — both see the same bytes.
    pub unified: bool,
}

impl MemoryArchitecture {
    /// Detect the memory architecture of the current system.
    ///
    /// On macOS with Metal, queries the device for unified memory support.
    /// On all other platforms, assumes discrete (non-unified) memory.
    pub fn detect() -> Self {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            // Apple Silicon devices always have unified memory.
            // Intel Macs with discrete GPUs do not.
            if let Some(device) = metal::Device::system_default() {
                return Self {
                    unified: device.has_unified_memory(),
                };
            }
        }

        // Non-Metal platforms: assume discrete memory.
        Self { unified: false }
    }

    /// Constant for CPU-only execution (always "unified" trivially).
    pub const CPU_ONLY: Self = Self { unified: true };
}

/// Decode tier assignments from raw archive bytes into `MemoryTier` values.
///
/// Each byte in the input is a `MemoryTier` discriminant (0–3).
/// Unknown discriminants default to `CpuMain` (backward compatibility).
pub fn decode_tier_assignments(bytes: &[u8]) -> Vec<MemoryTier> {
    bytes
        .iter()
        .map(|&b| MemoryTier::from_u8(b).unwrap_or(MemoryTier::CpuMain))
        .collect()
}

/// Returns `true` if the given tier assignments contain any `Device` tier.
/// If not, the hybrid executor can skip all migration logic and use the
/// simple `Executor::run_levels` path instead.
#[inline]
pub fn has_device_tier(tiers: &[MemoryTier]) -> bool {
    tiers.iter().any(|t| !t.is_cpu())
}

/// Trait for backends that can perform slot migrations between CPU and
/// device memory. Implemented by GPU backends to provide the actual
/// staging-buffer transfers; CPU-only backends get a blanket no-op.
pub trait MigrationBackend {
    /// Upload slot data from the CPU-side arena to device memory.
    fn upload_slot(&self, slot: u32, data: &[u8]) -> Result<(), crate::BackendError>;

    /// Download slot data from device memory back to the CPU-side arena.
    fn download_slot(&self, slot: u32, out: &mut [u8]) -> Result<(), crate::BackendError>;

    /// Returns `true` if migrations are no-ops (unified memory).
    fn is_unified(&self) -> bool {
        false
    }
}

/// Blanket no-op migration for CPU backends. All data is already in the
/// shared arena, so there's nothing to transfer.
pub struct NoOpMigration;

impl MigrationBackend for NoOpMigration {
    #[inline]
    fn upload_slot(&self, _slot: u32, _data: &[u8]) -> Result<(), crate::BackendError> {
        Ok(())
    }

    #[inline]
    fn download_slot(&self, _slot: u32, _out: &mut [u8]) -> Result<(), crate::BackendError> {
        Ok(())
    }

    #[inline]
    fn is_unified(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_known_discriminants() {
        let raw = vec![0, 1, 2, 3, 0];
        let tiers = decode_tier_assignments(&raw);
        assert_eq!(
            tiers,
            vec![
                MemoryTier::CpuL1,
                MemoryTier::CpuL2,
                MemoryTier::CpuMain,
                MemoryTier::Device,
                MemoryTier::CpuL1,
            ]
        );
    }

    #[test]
    fn decode_unknown_defaults_to_cpu_main() {
        let raw = vec![255, 4, 99];
        let tiers = decode_tier_assignments(&raw);
        assert!(tiers.iter().all(|t| *t == MemoryTier::CpuMain));
    }

    #[test]
    fn has_device_tier_detection() {
        assert!(!has_device_tier(&[MemoryTier::CpuL1, MemoryTier::CpuL2]));
        assert!(has_device_tier(&[MemoryTier::CpuL1, MemoryTier::Device]));
    }

    #[test]
    fn cpu_only_is_unified() {
        // A CPU-only architecture is trivially unified (one address space), so
        // coherence migrations are always no-ops.
        let arch = MemoryArchitecture::CPU_ONLY;
        assert!(arch.unified);
    }
}
