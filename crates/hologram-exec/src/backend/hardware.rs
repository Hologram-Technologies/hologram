//! Runtime hardware capability detection and per-op threshold tuning.
//!
//! `HardwareCaps::detect()` queries the system's GPU at runtime and caches
//! the result. `OpThresholds` translates capabilities into per-op-category
//! minimum byte thresholds for GPU dispatch — below these thresholds, CPU
//! SIMD is faster due to kernel launch overhead.
//!
//! Instead of one blanket threshold for all ops, each op category gets
//! a threshold tuned to its arithmetic intensity and the hardware's
//! characteristics.

use std::sync::OnceLock;

/// GPU family for threshold tuning heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFamily {
    /// No GPU available (CPU-only build or headless).
    None,
    /// Apple Silicon M1 / A14+.
    AppleM1,
    /// Apple Silicon M2 / A15+.
    AppleM2,
    /// Apple Silicon M3 / A17+.
    AppleM3,
    /// Apple Silicon M4.
    AppleM4,
    /// Generic Apple GPU (family detected but generation unknown).
    AppleGeneric,
    /// WebGPU via wgpu (cross-platform — Vulkan, DX12, browser WebGPU).
    /// Conservative thresholds since hardware varies widely.
    Wgpu,
}

/// Detected hardware capabilities of the current system.
#[derive(Debug, Clone)]
pub struct HardwareCaps {
    /// GPU and CPU share the same physical memory (Apple Silicon).
    pub unified_memory: bool,
    /// GPU supports native f16 arithmetic.
    pub f16_support: bool,
    /// Number of GPU compute units (0 = unknown or no GPU).
    pub compute_units: u32,
    /// Maximum single buffer size in bytes (0 = unknown).
    pub max_buffer_length: u64,
    /// GPU family for heuristic tuning.
    pub gpu_family: GpuFamily,
}

impl HardwareCaps {
    /// Detect hardware capabilities at runtime (cached on first call).
    pub fn detect() -> &'static Self {
        static CAPS: OnceLock<HardwareCaps> = OnceLock::new();
        CAPS.get_or_init(Self::detect_inner)
    }

    #[cfg(has_metal)]
    fn detect_inner() -> Self {
        use metal::Device;
        match Device::system_default() {
            Some(device) => {
                let unified_memory = device.has_unified_memory();
                let max_buffer_length = device.max_buffer_length();
                let gpu_family = Self::detect_apple_family(&device);
                Self {
                    unified_memory,
                    f16_support: true, // All Apple Silicon supports f16.
                    compute_units: 0,  // Metal API doesn't expose this directly.
                    max_buffer_length,
                    gpu_family,
                }
            }
            None => Self::cpu_only(),
        }
    }

    #[cfg(has_metal)]
    fn detect_apple_family(device: &metal::Device) -> GpuFamily {
        // Probe GPU families from newest to oldest.
        // MTLGPUFamily values: Apple9=1009, Apple8=1008, Apple7=1007, Apple6=1006, ...
        // M4 = Apple9 (family 1009), M3 = Apple8 (family 1008),
        // M2 = Apple8 (family 1008), M1 = Apple7 (family 1007).
        //
        // The `metal` crate exposes `supports_family()` but the enum coverage
        // varies by crate version. Use a conservative approach: check from
        // newest to oldest, default to AppleGeneric if any family matches.
        if device.supports_family(metal::MTLGPUFamily::Apple9) {
            GpuFamily::AppleM4
        } else if device.supports_family(metal::MTLGPUFamily::Apple8) {
            // M2 and M3 both report Apple8; distinguish by max_buffer_length.
            // M3 supports 128GB buffers, M2 caps at ~96GB (varies by SKU).
            // This is an approximation; exact detection would need IOKit queries.
            if device.max_buffer_length() > 100 * 1024 * 1024 * 1024 {
                GpuFamily::AppleM3
            } else {
                GpuFamily::AppleM2
            }
        } else if device.supports_family(metal::MTLGPUFamily::Apple7) {
            GpuFamily::AppleM1
        } else {
            GpuFamily::AppleGeneric
        }
    }

    #[cfg(not(has_metal))]
    fn detect_inner() -> Self {
        #[cfg(has_webgpu)]
        {
            Self {
                unified_memory: false,
                f16_support: false, // Varies by WebGPU adapter; assume no.
                compute_units: 0,
                max_buffer_length: 0,
                gpu_family: GpuFamily::Wgpu,
            }
        }
        #[cfg(not(has_webgpu))]
        {
            Self::cpu_only()
        }
    }

    fn cpu_only() -> Self {
        Self {
            unified_memory: false,
            f16_support: false,
            compute_units: 0,
            max_buffer_length: 0,
            gpu_family: GpuFamily::None,
        }
    }
}

/// Per-op-category minimum byte thresholds for GPU dispatch.
///
/// Below these thresholds, the CPU backend is faster due to GPU kernel
/// launch overhead (~10-50µs per command buffer commit on Metal).
/// Thresholds are tuned per GPU family based on arithmetic intensity:
/// - Elementwise ops (low intensity): high threshold (GPU overhead dominates)
/// - MatMul (high intensity): low threshold (GPU wins early)
/// - Conv2d: medium threshold (depends on kernel size)
#[derive(Debug, Clone)]
pub struct OpThresholds {
    /// Minimum input bytes for elementwise ops (relu, sigmoid, add, mul, ...).
    pub elementwise_min_bytes: usize,
    /// Minimum output elements (M×N) for matmul dispatch.
    pub matmul_min_elements: usize,
    /// Minimum input bytes for softmax.
    pub softmax_min_bytes: usize,
    /// Minimum input bytes for RMS norm.
    pub norm_min_bytes: usize,
}

impl OpThresholds {
    /// Default conservative thresholds (matches legacy 4MB behavior).
    pub const DEFAULT: Self = Self {
        elementwise_min_bytes: 4 * 1024 * 1024, // 4MB = 1M floats
        matmul_min_elements: 128 * 128,         // 16K elements
        softmax_min_bytes: 4 * 1024 * 1024,
        norm_min_bytes: 4 * 1024 * 1024,
    };
}

impl From<&HardwareCaps> for OpThresholds {
    fn from(caps: &HardwareCaps) -> Self {
        match caps.gpu_family {
            GpuFamily::AppleM4 => Self {
                // M4: fastest GPU, lowest crossover points.
                elementwise_min_bytes: 1024 * 1024, // 1MB
                matmul_min_elements: 64 * 64,       // 4K elements
                softmax_min_bytes: 512 * 1024,      // 512KB
                norm_min_bytes: 512 * 1024,
            },
            GpuFamily::AppleM3 => Self {
                elementwise_min_bytes: 2 * 1024 * 1024, // 2MB
                matmul_min_elements: 64 * 64,
                softmax_min_bytes: 1024 * 1024,
                norm_min_bytes: 1024 * 1024,
            },
            GpuFamily::AppleM2 => Self {
                elementwise_min_bytes: 2 * 1024 * 1024,
                matmul_min_elements: 96 * 96,
                softmax_min_bytes: 1024 * 1024,
                norm_min_bytes: 1024 * 1024,
            },
            GpuFamily::AppleM1 | GpuFamily::AppleGeneric => Self {
                // M1: original Apple Silicon, slightly higher thresholds.
                elementwise_min_bytes: 4 * 1024 * 1024,
                matmul_min_elements: 128 * 128,
                softmax_min_bytes: 2 * 1024 * 1024,
                norm_min_bytes: 2 * 1024 * 1024,
            },
            GpuFamily::Wgpu => Self {
                // WebGPU: staging buffer readback adds overhead.
                // Conservative — same as legacy for elementwise, slightly
                // better for matmul since arithmetic intensity compensates.
                elementwise_min_bytes: 4 * 1024 * 1024,
                matmul_min_elements: 128 * 128,
                softmax_min_bytes: 4 * 1024 * 1024,
                norm_min_bytes: 4 * 1024 * 1024,
            },
            GpuFamily::None => Self::DEFAULT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_valid_caps() {
        let caps = HardwareCaps::detect();
        // On any platform, detect should not panic.
        let _family = caps.gpu_family;
    }

    #[test]
    fn thresholds_from_caps() {
        let caps = HardwareCaps {
            unified_memory: true,
            f16_support: true,
            compute_units: 10,
            max_buffer_length: 16 * 1024 * 1024 * 1024,
            gpu_family: GpuFamily::AppleM3,
        };
        let thresholds = OpThresholds::from(&caps);
        assert_eq!(thresholds.elementwise_min_bytes, 2 * 1024 * 1024);
        assert_eq!(thresholds.matmul_min_elements, 64 * 64);
    }

    #[test]
    fn cpu_only_gets_defaults() {
        let caps = HardwareCaps {
            unified_memory: false,
            f16_support: false,
            compute_units: 0,
            max_buffer_length: 0,
            gpu_family: GpuFamily::None,
        };
        let thresholds = OpThresholds::from(&caps);
        assert_eq!(
            thresholds.elementwise_min_bytes,
            OpThresholds::DEFAULT.elementwise_min_bytes
        );
        assert_eq!(
            thresholds.matmul_min_elements,
            OpThresholds::DEFAULT.matmul_min_elements
        );
    }

    #[test]
    fn m4_has_lowest_thresholds() {
        let m4 = OpThresholds::from(&HardwareCaps {
            unified_memory: true,
            f16_support: true,
            compute_units: 0,
            max_buffer_length: 0,
            gpu_family: GpuFamily::AppleM4,
        });
        let m1 = OpThresholds::from(&HardwareCaps {
            unified_memory: true,
            f16_support: true,
            compute_units: 0,
            max_buffer_length: 0,
            gpu_family: GpuFamily::AppleM1,
        });
        assert!(m4.elementwise_min_bytes < m1.elementwise_min_bytes);
        assert!(m4.matmul_min_elements < m1.matmul_min_elements);
    }

    #[test]
    fn detect_is_cached() {
        let caps1 = HardwareCaps::detect();
        let caps2 = HardwareCaps::detect();
        // Same pointer — OnceLock returns the same instance.
        assert!(std::ptr::eq(caps1, caps2));
    }
}
