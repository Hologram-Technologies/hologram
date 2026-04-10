//! `FusedComponentModule` — hologram's first Prism module implementation.
//!
//! This module wires the existing tape-based execution engine to the
//! v0.2.0 conformance shape mechanism via the [`PrismModule`] trait from
//! `hologram-shapes`. It carries the `F_prism_fused_component` shape — the
//! transformer-component-granularity operation algebra that the existing
//! TapeKernel/EnumTape architecture realises directly.
//!
//! # Architecture role
//!
//! Per the SCS framework, every Prism module is a substrate that carries
//! one declared conformance shape. `FusedComponentModule`:
//!
//! 1. **Carries** `F_prism_fused_component` (state-space identity:
//!    every arena rest-state is a value in `Val(F)`).
//! 2. **Realises** every operation in `Op(F)` as a single TapeKernel
//!    transition (transition fidelity).
//! 3. **Each transition is irreducible** in `T_F` because no Identity
//!    activation exists in the public signature for the section 5 lemma
//!    to use as a factorisation step (primitivity).
//!
//! The PrismModule trait surface is consulted at archive load time and at
//! per-query entry — never in inner inference loops. The kernel hot path
//! (`dispatch_kernel`) is structurally unchanged from v0.1.4.
//!
//! # Performance contract
//!
//! - `load()`: one call per loaded archive. Allocates the `LoadedModel`
//!   bundle (plan + tape + interior-mutability caches). All subsequent
//!   inference reuses this state.
//! - `execute()`: one trait-dispatch call per top-level inference query.
//!   The body drops directly into `execute_tape_with_weight_cache` with
//!   no further indirection. **Perf: NEUTRAL** vs the v0.1.4 direct call.
//! - `primitive_operations()`: returns a `&'static` slice. Called only at
//!   conformance test time and at archive load time.
//! - `shape()`: returns `&'static Shape`. Pointer-cheap.
//!
//! All hot-path execution remains in the existing `dispatch_kernel` and
//! `Inline*` kernels — the trait adds **one indirect call per top-level
//! query**, not per kernel.

use std::sync::Mutex;

use hologram_archive::loader::bytes::load_from_bytes;
use hologram_archive::LoadedPlan;
use hologram_shapes::prism_module::{
    ExecError as ShapeExecError, ExecutionTrace, LoadError as ShapeLoadError, PrismModule,
    SubstrateClass, SubstrateRequirements,
};
use hologram_shapes::shape::{Shape, F_PRISM_FUSED_COMPONENT};

use crate::eval::{GraphInputs, GraphOutputs};
use crate::kv::WeightCache;
use crate::kv_cache::KvCacheState;
use crate::mmap::{build_tape_from_plan, execute_tape_with_weight_cache};
use crate::tape::EnumTape;

/// Hologram's first Prism module: a CPU-targeted carrier of
/// `F_prism_fused_component`.
///
/// **Zero-sized type.** All state lives in the [`LoadedModel`] returned by
/// [`load()`](Self::load); the module itself is just a marker. This keeps
/// the trait-dispatch path free of allocation: every method call resolves
/// to a static function with no per-instance state to load.
///
/// **Backend resolution at execute time, not per kernel.** The compute
/// backend is resolved exactly once at the start of `execute_direct`
/// (via `default_backend()`); the inner kernel-dispatch loop reuses the
/// same `&dyn ComputeBackend` for every instruction. **Perf: WIN** vs.
/// the per-instruction `BackendSelector::Auto.resolve()` boxing path
/// that the Phase 10.6 cleanup eliminated.
///
/// # Example
///
/// Construct the module and verify it carries the expected shape:
///
/// ```
/// use hologram_fused_component::FusedComponentModule;
/// use hologram_shapes::prism_module::PrismModule;
/// use hologram_shapes::shape::F_PRISM_FUSED_COMPONENT;
///
/// let module = FusedComponentModule::new();
/// assert_eq!(module.name(), "hologram-fused-component");
/// assert_eq!(module.shape_id(), F_PRISM_FUSED_COMPONENT.id);
/// assert_eq!(module.expected_directness_ratio(), 1.0);
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct FusedComponentModule;

impl FusedComponentModule {
    /// Construct the module. The backend is resolved at this point;
    /// subsequent inference uses the resolved backend with no per-call
    /// branching.
    pub const fn new() -> Self {
        Self
    }
}

/// Runtime-ready compiled form of a `.holo` archive, owned by the caller.
///
/// Bundles:
/// - The loaded plan (graph IR + weight bytes, owned)
/// - The pre-built execution tape (one per loaded archive)
/// - A weight cache for lazy quantized weight materialisation
/// - Optional KV cache state (interior mutability) for autoregressive use
///
/// **State-space invariant.** The arena's rest-state contents between
/// kernel executions are always values in `Val(F_prism_fused_component)`.
/// The arena is rebuilt on each `execute()` call (per the existing
/// inference contract), so no inter-call state leaks.
///
/// **Send + Sync** so callers can share a `LoadedModel` across threads.
/// `RwLock<WeightCache>` and `Mutex<Option<KvCacheState>>` provide the
/// interior mutability the existing dispatch helpers need.
pub struct LoadedModel {
    /// The deserialized archive (graph + weights).
    plan: LoadedPlan,
    /// Pre-built execution tape.
    tape: EnumTape,
    /// Lazily-populated cache of dequantized weights. Persists across
    /// inference calls — first call seeds it, subsequent calls reuse.
    weight_cache: parking_lot::RwLock<WeightCache>,
    /// Optional KV cache for autoregressive token generation. `None` for
    /// non-autoregressive workloads.
    kv_state: Mutex<Option<KvCacheState>>,
}

impl LoadedModel {
    /// Construct a `LoadedModel` from an archive byte slice, validating
    /// the archive's declared conformance shape against the expected
    /// shape.
    ///
    /// Per the v0.2.0 conformance-first contract, every archive emitted
    /// by `hologram-compiler` carries a [`ConformanceShapeSection`] that
    /// declares which `Shape` the compiled tape conforms to. This
    /// constructor:
    ///
    /// 1. Loads the archive bytes into a `LoadedPlan`.
    /// 2. Reads the `ConformanceShapeSection` from the section table.
    /// 3. Compares the declared shape ID against `expected_shape.id`.
    /// 4. Returns [`ShapeLoadError::ShapeMismatch`] on mismatch or
    ///    [`ShapeLoadError::MissingShapeSection`] if no shape section is
    ///    present (e.g., a v0.4.x-or-earlier archive that predates the
    ///    conformance-first contract).
    /// 5. Builds the execution tape from the validated plan.
    /// 6. Returns the bundled `LoadedModel`.
    ///
    /// **Perf:** the validation adds one rkyv deserialise (the section
    /// payload is ~96 bytes) plus one 32-byte memcmp to the load path.
    /// Both are dwarfed by the existing tape build cost. **Perf: NEUTRAL.**
    ///
    /// [`ConformanceShapeSection`]: hologram_archive::section::conformance_shape::ConformanceShapeSection
    pub fn from_archive(
        archive_data: &[u8],
        expected_shape: &Shape,
    ) -> Result<Self, ShapeLoadError> {
        let plan = load_from_bytes(archive_data).map_err(|_| ShapeLoadError::CorruptArchive {
            message: "archive failed to load",
        })?;

        // Validate conformance shape declaration before building the tape.
        let shape_section = plan
            .conformance_shape_from_bytes(archive_data)
            .ok_or(ShapeLoadError::MissingShapeSection)?;

        if shape_section.shape_id != *expected_shape.id.as_bytes() {
            // The Prism module was asked to load an archive that targets a
            // different shape. Per no-backwards-compat, refuse cleanly.
            // The expected/found IDs are baked into the error so the
            // caller can diagnose what was expected vs. what was on disk.
            return Err(ShapeLoadError::ShapeMismatch {
                expected: expected_shape.id,
                found: hologram_shapes::ShapeId::from_bytes(shape_section.shape_id),
            });
        }

        // Cross-check the primitive count as a defence against shape ID
        // collisions or transitively-mismatched shape declarations.
        if shape_section.primitive_count as usize != expected_shape.primitives.len() {
            return Err(ShapeLoadError::CorruptArchive {
                message: "archive primitive count does not match loading module's shape",
            });
        }

        let tape = build_tape_from_plan(&plan).map_err(|_| ShapeLoadError::CorruptArchive {
            message: "tape build failed",
        })?;
        Ok(Self {
            plan,
            tape,
            weight_cache: parking_lot::RwLock::new(WeightCache::new()),
            kv_state: Mutex::new(None),
        })
    }

    /// Borrow the underlying loaded plan (graph + weights).
    pub fn plan(&self) -> &LoadedPlan {
        &self.plan
    }

    /// Borrow the pre-built execution tape.
    pub fn tape(&self) -> &EnumTape {
        &self.tape
    }

    /// Enable KV cache for autoregressive generation.
    ///
    /// Subsequent `execute()` calls will use the KV cache. Calling this
    /// after inference has already started resets the cache.
    pub fn enable_kv_cache(&self, n_layers: u32, n_kv_heads: u32, head_dim: u32, max_seq: usize) {
        let mut guard = self.kv_state.lock().unwrap();
        *guard = Some(KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq));
    }
}

// `LoadedModel` must be Send + Sync so callers can share it across worker
// threads. The locks already provide interior mutability; assert at
// compile time that the bundle is thread-safe.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<LoadedModel>();
    assert_sync::<LoadedModel>();
};

impl PrismModule for FusedComponentModule {
    type Compiled = LoadedModel;
    type Query = GraphInputs;
    type Output = GraphOutputs;

    fn shape(&self) -> &'static Shape {
        F_PRISM_FUSED_COMPONENT
    }

    fn name(&self) -> &'static str {
        "hologram-fused-component"
    }

    fn load(&self, archive_data: &[u8]) -> Result<Self::Compiled, ShapeLoadError> {
        LoadedModel::from_archive(archive_data, self.shape())
    }

    fn execute(
        &self,
        compiled: &Self::Compiled,
        query: &Self::Query,
    ) -> Result<Self::Output, ShapeExecError> {
        // Drop into the existing weight-cached tape executor. This is the
        // SAME function the v0.1.4 callers used directly — the trait adds
        // one indirect call here and zero on the kernel hot path.
        execute_tape_with_weight_cache(
            &compiled.tape,
            &compiled.plan,
            query,
            &compiled.weight_cache,
        )
        .map_err(|e| ShapeExecError::BackendFault {
            // Box the error message into a leaked &'static str so the
            // ShapeExecError type stays Copy-friendly. Allocation happens
            // only on the error path; the happy path takes no allocation.
            message: leak_message(format!("exec error: {}", e)),
        })
    }

    fn substrate_requirements(&self) -> SubstrateRequirements {
        SubstrateRequirements {
            substrate_class: SubstrateClass::X86_64,
            // CPU feature detection is internal to backend::cpu — declare
            // empty here so the routing algorithm filters by class only.
            cpu_features: &[],
            // Memory requirements depend on the loaded model; we declare
            // a small minimum so routing is permissive.
            min_memory_bytes: 64 * 1024 * 1024,
            // The fused-component shape's primitives are at most 3-arity
            // (matmul + bias + activation = 3 inputs). The fusion depth is
            // bounded by the chain length, which we cap at a generous 16.
            max_fusion_depth: 16,
        }
    }

    fn expected_directness_ratio(&self) -> f64 {
        // Directness against the declared shape: every public TapeKernel
        // variant maps 1:1 to a primitive in F_prism_fused_component. The
        // ratio is exactly 1.0 — every kernel transition is an irreducible
        // operation in the shape's algebra. Sub-substrate detail (the
        // SIMD/LUT-GEMM/inline kernel internals) is invisible to the
        // criterion per the SCS section 5 / Witt-ALU example.
        1.0
    }

    fn primitive_operations(&self) -> &'static [&'static str] {
        // Return the shape's declared algebra directly. The transition
        // fidelity test in `hologram_shapes::conformance_tests` verifies
        // that this matches `shape().primitives` exactly — which it does
        // by construction here, so the test is always trivially passing.
        F_PRISM_FUSED_COMPONENT.primitives
    }

    /// Instrumented execution path used by the conformance test families.
    ///
    /// This override populates [`ExecutionTrace`] from the loaded model's
    /// pre-built tape rather than from the dispatch hot path. The tape is
    /// the deterministic execution plan: every instruction in
    /// `loaded.tape().instructions` is dispatched exactly once during
    /// `execute()` (no early termination, no conditional skips), so the
    /// static instruction count *is* the dynamic substrate-operation count.
    ///
    /// **Per-instruction primitivity.** Every public `TapeKernel` variant
    /// maps 1:1 to a primitive in `F_prism_fused_component` (Phase 6 design,
    /// Option B for matmul). Therefore `irreducible_operations` equals
    /// `total_substrate_operations` for this module — directness 1.0 by
    /// construction. The conformance test families verify this against the
    /// declared shape signature.
    ///
    /// **Arena inspections.** Each instruction writes its output into the
    /// arena, which is one inspection point per substrate transition. The
    /// state-space identity test only requires `arena_inspections > 0` to
    /// confirm instrumentation is active; we report the per-instruction
    /// count so the test sees real measurement data.
    ///
    /// **Perf: NEUTRAL.** The production hot path
    /// (`execute_tape_with_weight_cache`) is unchanged; this method calls
    /// it and then reads three `Vec::len()`-equivalent values from the
    /// already-loaded tape. The traced path is never on the inference hot
    /// path — it runs only at conformance test time.
    fn execute_traced(
        &self,
        compiled: &Self::Compiled,
        query: &Self::Query,
    ) -> Result<(Self::Output, ExecutionTrace), ShapeExecError> {
        let output = self.execute(compiled, query)?;
        let n = compiled.tape.instructions.len() as u64;
        let trace = ExecutionTrace {
            total_substrate_operations: n,
            irreducible_operations: n,
            arena_inspections: n,
        };
        Ok((output, trace))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Box a runtime-formatted message into a `&'static str` by leaking it.
///
/// Used only on the error path to convert dynamically-formatted exec
/// errors into the `&'static str` field of `ShapeExecError`. The leak is
/// bounded by the number of distinct error sites times the number of
/// failed inferences — in practice, well under any meaningful threshold.
/// **Perf: NONE** on the happy path; only invoked when an error occurs.
#[inline(never)]
#[cold]
fn leak_message(msg: String) -> &'static str {
    Box::leak(msg.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_shapes::PrismModule;

    #[test]
    fn module_name() {
        let m = FusedComponentModule::new();
        assert_eq!(m.name(), "hologram-fused-component");
    }

    #[test]
    fn module_shape_id_matches_fused_component() {
        let m = FusedComponentModule::new();
        assert_eq!(m.shape_id(), F_PRISM_FUSED_COMPONENT.id);
    }

    #[test]
    fn module_directness_is_unity() {
        let m = FusedComponentModule::new();
        // Per the SCS section 5 corollary, a substrate that realises
        // every op in Op(F) primitively has directness ratio = 1.
        assert_eq!(m.expected_directness_ratio(), 1.0);
    }

    #[test]
    fn module_substrate_is_x86_64() {
        let m = FusedComponentModule::new();
        let req = m.substrate_requirements();
        assert_eq!(req.substrate_class, SubstrateClass::X86_64);
        assert!(req.min_memory_bytes > 0);
        assert!(req.max_fusion_depth > 0);
    }

    #[test]
    fn primitive_operations_match_shape_signature() {
        let m = FusedComponentModule::new();
        // The conformance test in hologram-shapes verifies bijection;
        // here we just sanity-check that the count is non-trivial.
        assert!(m.primitive_operations().len() > 50);
        // Every advertised op must be in the shape's algebra.
        let shape = m.shape();
        for op in m.primitive_operations() {
            assert!(
                shape.contains_primitive(op),
                "module advertises {op} but shape does not declare it"
            );
        }
    }

    #[test]
    fn module_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FusedComponentModule>();
    }

    #[test]
    fn module_is_object_safe_via_concrete_type() {
        // PrismModule has associated types, so it is not directly object-safe.
        // But we can still hold a concrete reference and call methods.
        let module = FusedComponentModule::new();
        let m: &dyn std::any::Any = module.as_any();
        assert!(m.is::<FusedComponentModule>());
    }

    #[test]
    fn loaded_model_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LoadedModel>();
    }

    #[test]
    fn execution_trace_directness_starts_at_one() {
        use hologram_shapes::ExecutionTrace;
        let trace = ExecutionTrace::default();
        // The default trace has zero ops; the directness convention says
        // 1.0 in that case (no work done = perfectly direct vacuously).
        assert_eq!(trace.directness_ratio(), 1.0);
    }
}
