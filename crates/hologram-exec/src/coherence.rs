//! Software coherence layer for unified memory emulation (PM_7).
//!
//! Tracks per-slot device ownership and precomputes migration schedules
//! at level boundaries. On unified-memory hardware (Apple Silicon / Metal),
//! migrations are no-ops — the coherence metadata exists but triggers no
//! data movement.
//!
//! # Invariants
//!
//! - PL_2 (Lease Disjointness): within a `ParallelLevel`, each slot has
//!   exactly one writer. Ownership conflicts are impossible intra-level.
//! - Ownership transfers happen only at level boundaries.
//! - The migration schedule is precomputed at session load (no runtime
//!   fault handling).

extern crate alloc;
use alloc::vec::Vec;
use hologram_types::MemoryTier;

/// Which device currently holds the authoritative copy of a buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DeviceOwner {
    /// CPU holds the authoritative copy.
    Cpu = 0,
    /// Accelerator/GPU holds the authoritative copy.
    Device = 1,
    /// Both hold valid copies (read-only shared state).
    Shared = 2,
}

/// Per-slot coherence metadata, parallel to `BufferArena::slots`.
#[derive(Debug, Clone, Copy)]
pub struct SlotCoherence {
    /// Current owner of the authoritative data.
    pub owner: DeviceOwner,
    /// Memory tier assigned at compile time (from PM_7).
    pub tier: MemoryTier,
    /// Generation counter — incremented on every write.
    /// Detects stale copies after ownership transfer.
    pub generation: u32,
}

impl SlotCoherence {
    /// Default state: CPU-owned, main-memory tier, generation zero.
    #[inline]
    pub const fn new(tier: MemoryTier) -> Self {
        Self {
            owner: DeviceOwner::Cpu,
            tier,
            generation: 0,
        }
    }
}

/// Precomputed migration descriptor for a single execution level.
///
/// Built once at session load from the exec plan + tier assignments.
/// Each entry lists which slots must be transferred between CPU and
/// Device before the level's kernels execute.
#[derive(Debug, Clone, Default)]
pub struct LevelMigration {
    /// Slots whose data must be uploaded CPU → Device before this level.
    pub cpu_to_device: Vec<u32>,
    /// Slots whose data must be downloaded Device → CPU before this level.
    pub device_to_cpu: Vec<u32>,
}

impl LevelMigration {
    /// Returns `true` if no migrations are needed for this level.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cpu_to_device.is_empty() && self.device_to_cpu.is_empty()
    }
}

/// Aggregate tier statistics emitted during session load (PM_7).
#[cfg(feature = "tiered-exec")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierReport {
    /// Number of calls assigned to CpuL1.
    pub cpu_l1_calls: u32,
    /// Number of calls assigned to CpuL2.
    pub cpu_l2_calls: u32,
    /// Number of calls assigned to CpuMain.
    pub cpu_main_calls: u32,
    /// Number of calls assigned to Device.
    pub device_calls: u32,
    /// Sum of all `cpu_to_device` + `device_to_cpu` slot entries across levels.
    pub total_migration_slots: u32,
    /// Count of levels that have at least one migration.
    pub levels_with_migrations: u32,
}

/// Build a [`TierReport`] from the loaded tier assignments and migration schedule.
#[cfg(feature = "tiered-exec")]
pub fn build_report(tiers: &[MemoryTier], migrations: &[LevelMigration]) -> TierReport {
    let mut cpu_l1_calls: u32 = 0;
    let mut cpu_l2_calls: u32 = 0;
    let mut cpu_main_calls: u32 = 0;
    let mut device_calls: u32 = 0;

    for tier in tiers {
        match tier {
            MemoryTier::CpuL1 => cpu_l1_calls += 1,
            MemoryTier::CpuL2 => cpu_l2_calls += 1,
            MemoryTier::CpuMain => cpu_main_calls += 1,
            MemoryTier::Device => device_calls += 1,
        }
    }

    let mut total_migration_slots: u32 = 0;
    let mut levels_with_migrations: u32 = 0;

    for m in migrations {
        let count = (m.cpu_to_device.len() + m.device_to_cpu.len()) as u32;
        total_migration_slots += count;
        if count > 0 {
            levels_with_migrations += 1;
        }
    }

    TierReport {
        cpu_l1_calls,
        cpu_l2_calls,
        cpu_main_calls,
        device_calls,
        total_migration_slots,
        levels_with_migrations,
    }
}

/// Runtime tier override policy for an inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierPolicy {
    /// Use archive-embedded tier assignments (default).
    Compiled,
    /// Override all calls to CPU execution.
    ForceAllCpu,
    /// Override all calls to Device execution.
    ForceAllDevice,
}

impl TierPolicy {
    /// Apply this policy to a compile-time tier, returning the effective tier.
    #[inline]
    pub const fn apply(&self, compiled: MemoryTier) -> MemoryTier {
        match self {
            Self::Compiled => compiled,
            Self::ForceAllCpu => MemoryTier::CpuMain,
            Self::ForceAllDevice => MemoryTier::Device,
        }
    }
}

/// Build the migration schedule from tier assignments and the exec plan.
///
/// For each level, determines which slots need to move between CPU and
/// Device based on the tier assignments of the calls in that level vs.
/// the current ownership state (initially all CPU).
///
/// # Arguments
/// - `levels`: per-level kernel-call indices (from `ExecPlan` section)
/// - `tiers`: per-call tier assignment (from `TierAssignments` section)
/// - `call_outputs`: per-call output slot index
/// - `call_inputs`: per-call input slot indices (flattened with offsets)
/// - `slot_count`: total number of buffer slots
pub fn build_migration_schedule(
    levels: &[Vec<u32>],
    tiers: &[MemoryTier],
    call_outputs: &[u32],
    call_inputs: &[Vec<u32>],
    slot_count: usize,
) -> Vec<LevelMigration> {
    let mut owners = alloc::vec![DeviceOwner::Cpu; slot_count];
    let mut schedule = Vec::with_capacity(levels.len());

    for level in levels {
        let mut migration = LevelMigration::default();

        // Determine which slots this level reads and which device needs them.
        for &call_idx in level {
            let idx = call_idx as usize;
            let tier = tiers.get(idx).copied().unwrap_or(MemoryTier::CpuMain);
            let needs_device = !tier.is_cpu();

            // Check inputs: do they need migration?
            if let Some(inputs) = call_inputs.get(idx) {
                for &slot in inputs {
                    let s = slot as usize;
                    if s >= owners.len() {
                        continue;
                    }
                    if needs_device && owners[s] == DeviceOwner::Cpu {
                        migration.cpu_to_device.push(slot);
                        owners[s] = DeviceOwner::Shared;
                    } else if !needs_device && owners[s] == DeviceOwner::Device {
                        migration.device_to_cpu.push(slot);
                        owners[s] = DeviceOwner::Shared;
                    }
                }
            }
        }

        // After dispatch: update ownership for written slots.
        for &call_idx in level {
            let idx = call_idx as usize;
            let tier = tiers.get(idx).copied().unwrap_or(MemoryTier::CpuMain);
            if let Some(&out_slot) = call_outputs.get(idx) {
                let s = out_slot as usize;
                if s < owners.len() {
                    owners[s] = if tier.is_cpu() {
                        DeviceOwner::Cpu
                    } else {
                        DeviceOwner::Device
                    };
                }
            }
        }

        schedule.push(migration);
    }

    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_schedule_for_cpu_only() {
        // All calls are CPU-tier → no migrations ever.
        let levels = vec![vec![0, 1], vec![2]];
        let tiers = vec![MemoryTier::CpuL1, MemoryTier::CpuL2, MemoryTier::CpuMain];
        let outputs = vec![0, 1, 2];
        let inputs = vec![vec![], vec![0], vec![1]];

        let schedule = build_migration_schedule(&levels, &tiers, &outputs, &inputs, 3);

        assert_eq!(schedule.len(), 2);
        assert!(schedule[0].is_empty());
        assert!(schedule[1].is_empty());
    }

    #[test]
    fn migration_when_gpu_reads_cpu_slot() {
        // Level 0: CPU writes slot 0
        // Level 1: GPU reads slot 0 → needs cpu_to_device migration
        let levels = vec![vec![0], vec![1]];
        let tiers = vec![MemoryTier::CpuL1, MemoryTier::Device];
        let outputs = vec![0, 1];
        let inputs = vec![vec![], vec![0]];

        let schedule = build_migration_schedule(&levels, &tiers, &outputs, &inputs, 2);

        assert!(schedule[0].is_empty());
        assert_eq!(schedule[1].cpu_to_device, vec![0]);
        assert!(schedule[1].device_to_cpu.is_empty());
    }

    #[test]
    fn tier_policy_overrides() {
        assert_eq!(
            TierPolicy::Compiled.apply(MemoryTier::Device),
            MemoryTier::Device
        );
        assert_eq!(
            TierPolicy::ForceAllCpu.apply(MemoryTier::Device),
            MemoryTier::CpuMain
        );
        assert_eq!(
            TierPolicy::ForceAllDevice.apply(MemoryTier::CpuL1),
            MemoryTier::Device
        );
    }

    #[cfg(feature = "tiered-exec")]
    #[test]
    fn build_report_counts_tiers_and_migrations() {
        use super::{build_report, LevelMigration, TierReport};

        let tiers = vec![
            MemoryTier::CpuL1,
            MemoryTier::CpuL2,
            MemoryTier::CpuMain,
            MemoryTier::Device,
            MemoryTier::Device,
            MemoryTier::CpuL1,
        ];

        let migrations = vec![
            LevelMigration {
                cpu_to_device: vec![0],
                device_to_cpu: vec![],
            },
            LevelMigration::default(), // empty — no migrations
            LevelMigration {
                cpu_to_device: vec![1, 2],
                device_to_cpu: vec![3],
            },
        ];

        let report = build_report(&tiers, &migrations);

        assert_eq!(
            report,
            TierReport {
                cpu_l1_calls: 2,
                cpu_l2_calls: 1,
                cpu_main_calls: 1,
                device_calls: 2,
                total_migration_slots: 4,  // 1 + 0 + 3
                levels_with_migrations: 2, // levels 0 and 2
            }
        );
    }
}
