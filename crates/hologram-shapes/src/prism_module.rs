//! The [`PrismModule`] trait — every substrate carrying a hologram-declared
//! shape implements this.
//!
//! # Architectural role
//!
//! `PrismModule` is the **interface** between the cross-compiler and the
//! Prism module layer. The compiler uses it to:
//!
//! 1. Look up which shape a module carries (`shape()`).
//! 2. Validate an archive's conformance shape section against a module's
//!    declared shape (`load()`).
//! 3. Execute queries against a loaded module (`execute()`).
//! 4. Filter modules by substrate requirements during routing
//!    (`substrate_requirements()`).
//! 5. Pick the highest-directness module among the survivors
//!    (`expected_directness_ratio()`).
//! 6. Verify the module against its shape via the conformance test suite
//!    (`primitive_operations()`, `execute_traced()`).
//!
//! # Performance principle
//!
//! The trait is **never used in inner inference loops**. All trait calls
//! happen at:
//!
//! - Module construction time (`new()` — backend resolution).
//! - Archive load time (`load()` — once per loaded model).
//! - Per-query entry (`execute()` — once per top-level inference call,
//!   *not* once per kernel).
//! - Conformance test time (the `*_traced` and `primitive_operations`
//!   accessors).
//!
//! Inside `execute()`, the implementation drops down to its own monomorphised
//! tape dispatch with zero further trait indirection. **Perf: NEUTRAL** —
//! the trait adds one indirect call per top-level inference, not per kernel.

use core::any::Any;
use core::fmt;

use crate::shape::{Shape, ShapeId};

/// Substrate class — what physical layer a Prism module targets. Used by the
/// compiler's routing algorithm to filter modules by deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SubstrateClass {
    /// x86_64 CPU with optional SIMD (AVX2 / SSE4.2).
    X86_64,
    /// ARM64 CPU with optional NEON.
    Arm64,
    /// WebAssembly (no JIT, no native SIMD).
    Wasm,
    /// Apple Metal GPU.
    MetalGpu,
    /// WebGPU (browser or wgpu-native).
    WebGpu,
}

impl fmt::Display for SubstrateClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::X86_64 => "x86_64",
            Self::Arm64 => "arm64",
            Self::Wasm => "wasm",
            Self::MetalGpu => "metal_gpu",
            Self::WebGpu => "webgpu",
        };
        f.write_str(s)
    }
}

/// What a Prism module needs from the physical substrate.
///
/// The compiler's routing algorithm queries a candidate module's
/// `substrate_requirements()` and filters out modules whose requirements are
/// not satisfied by the deployment.
#[derive(Debug, Clone)]
pub struct SubstrateRequirements {
    /// Physical substrate class this module targets.
    pub substrate_class: SubstrateClass,
    /// Required CPU feature flags (e.g., `"avx2"`, `"neon"`, `"sse4.1"`).
    /// Empty means scalar fallback works.
    pub cpu_features: &'static [&'static str],
    /// Minimum available memory in bytes.
    pub min_memory_bytes: u64,
    /// Maximum number of fused operations the module can execute as a single
    /// transition.
    pub max_fusion_depth: u32,
}

/// Error returned by [`PrismModule::load`] when an archive cannot be loaded.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LoadError {
    /// The archive's conformance shape section does not match this module's
    /// declared shape.
    ShapeMismatch {
        /// Shape declared by the loading module.
        expected: ShapeId,
        /// Shape declared by the archive.
        found: ShapeId,
    },
    /// The archive predates the conformance shape requirement (no shape
    /// section present). Per the no-backwards-compat principle, hologram
    /// rejects such archives — recompile from source.
    MissingShapeSection,
    /// The archive bytes are corrupt or do not parse as a `.holo` archive.
    CorruptArchive {
        /// Diagnostic message.
        message: &'static str,
    },
    /// The archive declares CPU features or memory requirements the
    /// deployment cannot satisfy.
    UnsupportedSubstrate {
        /// What was missing or unsupported.
        reason: &'static str,
    },
    /// I/O failure during load.
    Io {
        /// Diagnostic message.
        message: &'static str,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, found } => write!(
                f,
                "archive declares shape {} but loader expects shape {}",
                found, expected
            ),
            Self::MissingShapeSection => f.write_str(
                "archive has no conformance shape section (predates v0.5.0); \
                 recompile from source",
            ),
            Self::CorruptArchive { message } => write!(f, "corrupt archive: {}", message),
            Self::UnsupportedSubstrate { reason } => {
                write!(f, "substrate requirements not satisfied: {}", reason)
            }
            Self::Io { message } => write!(f, "i/o error: {}", message),
        }
    }
}

/// Error returned by [`PrismModule::execute`] when a query cannot be answered.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ExecError {
    /// The query references an operation outside the module's declared shape.
    UnknownOperation {
        /// Diagnostic message.
        message: &'static str,
    },
    /// Shape mismatch between query inputs and the loaded model.
    ShapeMismatch {
        /// Diagnostic message.
        message: &'static str,
    },
    /// Backend reported a hardware fault.
    BackendFault {
        /// Diagnostic message.
        message: &'static str,
    },
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOperation { message } => write!(f, "unknown operation: {}", message),
            Self::ShapeMismatch { message } => write!(f, "input shape mismatch: {}", message),
            Self::BackendFault { message } => write!(f, "backend fault: {}", message),
        }
    }
}

/// Trace data produced by [`PrismModule::execute_traced`]. Used by the
/// conformance test suite to compute the directness ratio and to verify
/// primitivity.
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    /// Total number of substrate operations executed for this query.
    pub total_substrate_operations: u64,
    /// Number of those operations that are irreducible in `T_F` (i.e., that
    /// correspond directly to a primitive of the module's declared shape).
    pub irreducible_operations: u64,
    /// Number of arena rest-state inspections performed (used by the
    /// state-space identity test).
    pub arena_inspections: u64,
}

impl ExecutionTrace {
    /// Compute the directness ratio for this trace: `T / D` where `T` is the
    /// number of irreducible operations and `D` is the total. A ratio of 1.0
    /// means the substrate answered the query through irreducible operations
    /// only — the maximum possible directness.
    pub fn directness_ratio(&self) -> f64 {
        if self.total_substrate_operations == 0 {
            1.0
        } else {
            self.irreducible_operations as f64 / self.total_substrate_operations as f64
        }
    }
}

/// The trait every Prism module implements. The compiler dispatches against
/// this trait, and the conformance test infrastructure verifies a module
/// against its declared shape via this interface.
///
/// # Performance contract
///
/// **None of these methods is called in an inner inference loop.** Each call
/// happens at:
///
/// - Module construction time: building the module instance (zero or one call).
/// - Archive load time: `load()` (one call per loaded model).
/// - Per-query entry: `execute()` (one call per top-level inference, not per
///   kernel — the kernel dispatch lives inside the implementation's own
///   monomorphised tape executor).
/// - Conformance test time: `primitive_operations()`, `execute_traced()`.
///
/// Implementations must keep `execute()`'s overhead bounded by O(1) trait
/// dispatch + the module's own native query handling. **No allocation in the
/// inner loop**, no per-kernel virtual dispatch.
///
/// # Example
///
/// A minimal module that carries `F_PRISM_STRICT` and accepts a unit
/// query:
///
/// ```
/// use core::any::Any;
/// use hologram_shapes::prism_module::{
///     ExecError, LoadError, PrismModule, SubstrateClass, SubstrateRequirements,
/// };
/// use hologram_shapes::shape::{Shape, F_PRISM_STRICT};
///
/// struct DummyModule;
///
/// impl PrismModule for DummyModule {
///     type Compiled = ();
///     type Query = ();
///     type Output = ();
///
///     fn shape(&self) -> &'static Shape { F_PRISM_STRICT }
///     fn name(&self) -> &'static str { "dummy" }
///     fn load(&self, _: &[u8]) -> Result<(), LoadError> { Ok(()) }
///     fn execute(&self, _: &(), _: &()) -> Result<(), ExecError> { Ok(()) }
///     fn substrate_requirements(&self) -> SubstrateRequirements {
///         SubstrateRequirements {
///             substrate_class: SubstrateClass::X86_64,
///             cpu_features: &[],
///             min_memory_bytes: 0,
///             max_fusion_depth: 0,
///         }
///     }
///     fn expected_directness_ratio(&self) -> f64 { 1.0 }
///     fn as_any(&self) -> &dyn Any { self }
/// }
///
/// let m = DummyModule;
/// assert_eq!(m.name(), "dummy");
/// assert_eq!(m.shape_id(), F_PRISM_STRICT.id);
/// ```
pub trait PrismModule: Send + Sync {
    /// The runtime-ready compiled form. Its state space must be in bijection
    /// with `Val(F)` for the module's declared shape (the state-space
    /// identity condition of the SCS carrying criterion).
    type Compiled: Send + Sync;

    /// The query type this module accepts.
    type Query;

    /// The output type this module produces.
    type Output;

    /// The conformance shape this module carries.
    fn shape(&self) -> &'static Shape;

    /// Convenience accessor for the shape's ID.
    fn shape_id(&self) -> ShapeId {
        self.shape().id
    }

    /// Human-readable name (e.g., `"hologram-fused-component"`).
    fn name(&self) -> &'static str;

    /// Load a `.holo` archive into the module's runtime-ready form.
    ///
    /// Validates the archive's conformance shape section against
    /// [`shape()`](Self::shape) before producing the compiled form. Returns
    /// [`LoadError::ShapeMismatch`] if the archive targets a different shape.
    ///
    /// **Perf:** one call per loaded model. The body may allocate the
    /// compiled form, mmap weights, prewarm caches, etc. — none of this
    /// happens at inference time.
    fn load(&self, archive_data: &[u8]) -> Result<Self::Compiled, LoadError>;

    /// Execute a query against the compiled form. Cost is `O(d)` per Prism
    /// section 5, where `d` is the runtime's hyperdimensional dimension. The
    /// constant depends on query complexity but is independent of the source
    /// artifact's parameter count `N`.
    ///
    /// **Perf:** one trait-dispatch call per top-level query. The body drops
    /// into the implementation's own native dispatch (e.g., the tape
    /// executor) with no further indirection.
    fn execute(
        &self,
        compiled: &Self::Compiled,
        query: &Self::Query,
    ) -> Result<Self::Output, ExecError>;

    /// Reports what this module needs from the physical substrate. Used by
    /// the compiler's routing algorithm to filter eligible modules.
    fn substrate_requirements(&self) -> SubstrateRequirements;

    /// Expected directness ratio for a representative query distribution.
    /// Used by the routing algorithm as the tiebreaker among modules whose
    /// substrate requirements are all satisfied.
    fn expected_directness_ratio(&self) -> f64;

    /// The set of primitive operation IRIs this module realises. Must match
    /// the shape's `primitives` slice exactly (transition fidelity).
    ///
    /// The default implementation returns the shape's primitives slice
    /// directly — implementations should override only if they need to
    /// dynamically advertise a subset (e.g., when feature flags disable
    /// some primitives).
    fn primitive_operations(&self) -> &'static [&'static str] {
        self.shape().primitives
    }

    /// Optional instrumented execution. The default implementation falls
    /// back to [`execute()`](Self::execute) and returns an empty trace; the
    /// state-space identity and primitivity tests in
    /// [`crate::conformance_tests`] use this method's trace data to verify
    /// the carrying criterion's three conditions.
    ///
    /// Implementations should override only when conformance test coverage
    /// against this module is needed.
    fn execute_traced(
        &self,
        compiled: &Self::Compiled,
        query: &Self::Query,
    ) -> Result<(Self::Output, ExecutionTrace), ExecError> {
        let output = self.execute(compiled, query)?;
        Ok((output, ExecutionTrace::default()))
    }

    /// Downcast to a concrete type. Useful when test infrastructure needs
    /// module-specific introspection (e.g., arena inspection for the
    /// state-space identity test).
    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{F_PRISM_FUSED_COMPONENT, F_PRISM_STRICT};

    /// A trivial implementation used to verify the trait shape.
    struct DummyModule;
    struct DummyCompiled;

    impl PrismModule for DummyModule {
        type Compiled = DummyCompiled;
        type Query = ();
        type Output = ();

        fn shape(&self) -> &'static Shape {
            F_PRISM_STRICT
        }

        fn name(&self) -> &'static str {
            "dummy"
        }

        fn load(&self, _archive_data: &[u8]) -> Result<Self::Compiled, LoadError> {
            Ok(DummyCompiled)
        }

        fn execute(
            &self,
            _compiled: &Self::Compiled,
            _query: &Self::Query,
        ) -> Result<Self::Output, ExecError> {
            Ok(())
        }

        fn substrate_requirements(&self) -> SubstrateRequirements {
            SubstrateRequirements {
                substrate_class: SubstrateClass::X86_64,
                cpu_features: &[],
                min_memory_bytes: 0,
                max_fusion_depth: 0,
            }
        }

        fn expected_directness_ratio(&self) -> f64 {
            1.0
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn dummy_module_compiles_against_strict_shape() {
        let m = DummyModule;
        assert_eq!(m.shape().name, "F_prism_strict");
        assert_eq!(m.shape_id(), F_PRISM_STRICT.id);
        assert_eq!(m.name(), "dummy");
        assert_eq!(m.expected_directness_ratio(), 1.0);
        assert_eq!(m.primitive_operations().len(), 5);
    }

    #[test]
    fn shape_mismatch_error_displays_both_ids() {
        let err = LoadError::ShapeMismatch {
            expected: F_PRISM_STRICT.id,
            found: F_PRISM_FUSED_COMPONENT.id,
        };
        let s = alloc::format!("{}", err);
        assert!(s.contains("declares shape"));
        assert!(s.contains("expects shape"));
    }

    #[test]
    fn execution_trace_directness_ratio_zero_total() {
        let t = ExecutionTrace::default();
        assert_eq!(t.directness_ratio(), 1.0);
    }

    #[test]
    fn execution_trace_directness_ratio_partial() {
        let t = ExecutionTrace {
            total_substrate_operations: 10,
            irreducible_operations: 7,
            arena_inspections: 0,
        };
        assert!((t.directness_ratio() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn substrate_class_display() {
        assert_eq!(alloc::format!("{}", SubstrateClass::X86_64), "x86_64");
        assert_eq!(alloc::format!("{}", SubstrateClass::Arm64), "arm64");
        assert_eq!(alloc::format!("{}", SubstrateClass::Wasm), "wasm");
        assert_eq!(alloc::format!("{}", SubstrateClass::MetalGpu), "metal_gpu");
        assert_eq!(alloc::format!("{}", SubstrateClass::WebGpu), "webgpu");
    }

    /// Trait must be object-safe so the compiler's `PrismModuleRegistry` can
    /// hold `Box<dyn PrismModule>`. Note: PrismModule has associated types
    /// (Compiled/Query/Output), so it is *not* object-safe directly. The
    /// registry will use a wrapper trait that erases the associated types.
    /// This test asserts the trait can at least be used as a generic bound.
    #[test]
    fn trait_is_a_generic_bound() {
        fn _accepts<M: PrismModule>(_m: &M) {}
        _accepts(&DummyModule);
    }
}
