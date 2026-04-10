//! Parameterised conformance test infrastructure for the SCS carrying
//! criterion.
//!
//! Three test families correspond to the three carrying conditions:
//!
//! 1. **State-space identity** ([`test_state_space_identity`]) — every
//!    accessible state of the substrate is in `Val(F)`, and every value of
//!    `Val(F)` is representable as a state of the substrate.
//! 2. **Transition fidelity** ([`test_transition_fidelity`]) — every legal
//!    substrate transition corresponds to an operation in `Op(F)`, and every
//!    operation in `Op(F)` is realisable as a substrate transition.
//! 3. **Primitivity** ([`test_primitivity`]) — each substrate transition
//!    realises its corresponding operation as a single transition rather
//!    than as a composition through intermediate `Val(F)` states.
//!
//! Each test takes a [`PrismModule`] and verifies one carrying condition
//! against the module's declared shape. They are parameterised — the same
//! test functions verify any module declaring any shape.
//!
//! # Performance
//!
//! These tests run only at conformance check time (in `cargo test`, in CI,
//! or in a manual diagnostic run). They are **never** invoked at production
//! inference time. Their cost does not affect runtime perf at all.
//!
//! Even within tests, the per-test cost is bounded by the number of declared
//! primitives in the shape (~60 for `F_prism_fused_component`) times the
//! per-operation test fixture cost. Each test runs in well under a second on
//! commodity CPU.
//!
//! # Limits of automated checking
//!
//! The state-space identity and primitivity tests are *indirect* — they
//! verify properties of arena rest-states and execution traces that the
//! substrate exposes via `execute_traced`. A module that does not implement
//! `execute_traced` (relying on the default implementation) skips the
//! deeper inspection. The transition fidelity test is direct: it compares
//! the shape's `primitives` list with the module's `primitive_operations()`
//! return value.
//!
//! For the strongest guarantees, an implementation that wishes to be
//! conformance-verified should override `execute_traced` to return a
//! complete [`ExecutionTrace`].

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::prism_module::PrismModule;
use crate::shape::Shape;

/// Outcome of a single conformance test family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceReport {
    /// The shape under test.
    pub shape_name: &'static str,
    /// The module under test.
    pub module_name: &'static str,
    /// Test family (state-space identity / transition fidelity / primitivity).
    pub family: &'static str,
    /// Pass/fail.
    pub passed: bool,
    /// List of human-readable failure messages, empty if `passed`.
    pub failures: Vec<String>,
    /// Number of operations checked.
    pub operations_checked: usize,
}

impl ConformanceReport {
    /// Construct a passing report.
    fn passing(
        shape: &'static Shape,
        module_name: &'static str,
        family: &'static str,
        ops: usize,
    ) -> Self {
        Self {
            shape_name: shape.name,
            module_name,
            family,
            passed: true,
            failures: Vec::new(),
            operations_checked: ops,
        }
    }

    /// Construct a failing report.
    fn failing(
        shape: &'static Shape,
        module_name: &'static str,
        family: &'static str,
        failures: Vec<String>,
        ops: usize,
    ) -> Self {
        Self {
            shape_name: shape.name,
            module_name,
            family,
            passed: false,
            failures,
            operations_checked: ops,
        }
    }
}

/// Combined report for the full three-condition conformance check.
///
/// Note: not `Eq` because `directness_ratio: Option<f64>` carries an `f64`,
/// which has `PartialEq` but not `Eq` (NaN != NaN).
#[derive(Debug, Clone, PartialEq)]
pub struct FullConformanceReport {
    /// State-space identity result.
    pub state_space_identity: ConformanceReport,
    /// Transition fidelity result.
    pub transition_fidelity: ConformanceReport,
    /// Primitivity result.
    pub primitivity: ConformanceReport,
    /// Aggregate directness ratio measured during the run, if available.
    pub directness_ratio: Option<f64>,
}

impl FullConformanceReport {
    /// Whether all three conditions held.
    pub fn passed(&self) -> bool {
        self.state_space_identity.passed
            && self.transition_fidelity.passed
            && self.primitivity.passed
    }
}

/// Run the **transition fidelity** test against a module.
///
/// Verifies that the module's `primitive_operations()` list matches the
/// shape's declared `primitives` exactly. There must be no extras and no
/// missing operations.
///
/// This is the *direct* test of transition fidelity: it does not need to
/// execute any operations to detect mismatches.
///
/// **Perf:** O(N) in the number of declared primitives. N ~ 60 for
/// `F_prism_fused_component`. Runs in microseconds.
pub fn test_transition_fidelity<M: PrismModule>(module: &M) -> ConformanceReport {
    let shape = module.shape();
    let module_ops = module.primitive_operations();

    let shape_set: BTreeSet<&str> = shape.primitives.iter().copied().collect();
    let module_set: BTreeSet<&str> = module_ops.iter().copied().collect();

    let mut failures = Vec::new();
    for declared in &shape_set {
        if !module_set.contains(declared) {
            failures.push(format!(
                "module is missing declared primitive: {}",
                declared
            ));
        }
    }
    for advertised in &module_set {
        if !shape_set.contains(advertised) {
            failures.push(format!(
                "module advertises operation not in shape's algebra: {}",
                advertised
            ));
        }
    }

    if failures.is_empty() {
        ConformanceReport::passing(shape, module.name(), "transition_fidelity", shape_set.len())
    } else {
        ConformanceReport::failing(
            shape,
            module.name(),
            "transition_fidelity",
            failures,
            shape_set.len(),
        )
    }
}

/// Run the **state-space identity** test against a module.
///
/// Indirect: verifies that during instrumented execution, no arena
/// inspection reveals a state outside `Val(F)`. This requires the module to
/// override `execute_traced` and to populate the `arena_inspections` field
/// of [`ExecutionTrace`]. The default implementation reports zero
/// inspections; this test then degrades to a *coverage warning* rather than
/// a hard failure.
///
/// **Perf:** O(K) per query where K is the number of operations in the test
/// query. Runs only in conformance test mode.
pub fn test_state_space_identity<M, F>(
    module: &M,
    archive: &[u8],
    query_factory: F,
) -> ConformanceReport
where
    M: PrismModule,
    F: Fn() -> M::Query,
{
    let shape = module.shape();
    let mut failures = Vec::new();

    // The module's runtime-ready compiled form is needed to inspect arena
    // rest-state. The caller supplies the archive bytes; for unit tests with
    // dummy modules, an empty slice is fine (the module's load() returns
    // Ok). For real Prism modules, the caller must supply a compiled
    // archive (`hologram_compiler::compile(...).archive`) or load() will
    // refuse it as an empty buffer.
    let compiled = match module.load(archive) {
        Ok(c) => c,
        Err(e) => {
            // Empty archive will be rejected by most real modules. Treat
            // load failure here as a "no test data" outcome rather than a
            // pass — flag it so the user knows to provide a real archive.
            failures.push(format!(
                "state-space identity check requires loadable archive: {}",
                e
            ));
            return ConformanceReport::failing(
                shape,
                module.name(),
                "state_space_identity",
                failures,
                0,
            );
        }
    };

    let query = query_factory();
    let trace = match module.execute_traced(&compiled, &query) {
        Ok((_output, trace)) => trace,
        Err(e) => {
            failures.push(format!("execute_traced failed: {}", e));
            return ConformanceReport::failing(
                shape,
                module.name(),
                "state_space_identity",
                failures,
                0,
            );
        }
    };

    if trace.arena_inspections == 0 {
        // The module does not implement instrumented arena inspection.
        // Degrade to a warning by emitting a single failure that flags the
        // missing instrumentation rather than asserting state-space
        // identity passed silently.
        failures.push(
            "module does not implement arena inspection in execute_traced — \
             cannot verify state-space identity directly. Override \
             execute_traced and populate trace.arena_inspections to enable \
             this check."
                .to_string(),
        );
    }

    if failures.is_empty() {
        ConformanceReport::passing(
            shape,
            module.name(),
            "state_space_identity",
            trace.arena_inspections as usize,
        )
    } else {
        ConformanceReport::failing(
            shape,
            module.name(),
            "state_space_identity",
            failures,
            trace.arena_inspections as usize,
        )
    }
}

/// Run the **primitivity** test against a module.
///
/// Indirect: verifies that the module's [`ExecutionTrace`] reports
/// `irreducible_operations == total_substrate_operations` — i.e., the
/// module realises every operation in its declared algebra primitively.
///
/// A module that has any `irreducible_operations < total_substrate_operations`
/// is realising some operation through a factorisation, which the
/// section 5 corollary forbids for any substrate that carries `F`.
///
/// **Perf:** O(K) per query, with K small. Runs only in conformance test
/// mode.
pub fn test_primitivity<M, F>(module: &M, archive: &[u8], query_factory: F) -> ConformanceReport
where
    M: PrismModule,
    F: Fn() -> M::Query,
{
    let shape = module.shape();
    let mut failures = Vec::new();

    let compiled = match module.load(archive) {
        Ok(c) => c,
        Err(e) => {
            failures.push(format!(
                "primitivity check requires loadable archive: {}",
                e
            ));
            return ConformanceReport::failing(shape, module.name(), "primitivity", failures, 0);
        }
    };

    let query = query_factory();
    let (_output, trace) = match module.execute_traced(&compiled, &query) {
        Ok(r) => r,
        Err(e) => {
            failures.push(format!("execute_traced failed: {}", e));
            return ConformanceReport::failing(shape, module.name(), "primitivity", failures, 0);
        }
    };

    if trace.total_substrate_operations == 0 {
        failures.push(
            "module did not record any substrate operations in trace — \
             override execute_traced and populate trace fields to enable \
             primitivity check."
                .to_string(),
        );
    } else if trace.irreducible_operations < trace.total_substrate_operations {
        let nonprimitive = trace.total_substrate_operations - trace.irreducible_operations;
        failures.push(format!(
            "{} of {} substrate operations were not primitive (directness = {:.4})",
            nonprimitive,
            trace.total_substrate_operations,
            trace.directness_ratio()
        ));
    }

    if failures.is_empty() {
        ConformanceReport::passing(
            shape,
            module.name(),
            "primitivity",
            trace.total_substrate_operations as usize,
        )
    } else {
        ConformanceReport::failing(
            shape,
            module.name(),
            "primitivity",
            failures,
            trace.total_substrate_operations as usize,
        )
    }
}

/// Run all three conformance test families against a module and produce a
/// combined report.
///
/// The query factory is invoked once per test family (three times total).
/// Provide a representative query that exercises a meaningful subset of the
/// shape's algebra.
pub fn test_full_conformance<M, F>(
    module: &M,
    archive: &[u8],
    query_factory: F,
) -> FullConformanceReport
where
    M: PrismModule,
    F: Fn() -> M::Query,
{
    let transition_fidelity = test_transition_fidelity(module);
    let state_space_identity = test_state_space_identity(module, archive, &query_factory);
    let primitivity = test_primitivity(module, archive, &query_factory);

    // Aggregate directness ratio: re-run the query once more to get a fresh
    // trace independent of the per-test traces above.
    let directness_ratio = match module.load(archive) {
        Ok(compiled) => match module.execute_traced(&compiled, &query_factory()) {
            Ok((_, trace)) if trace.total_substrate_operations > 0 => {
                Some(trace.directness_ratio())
            }
            _ => None,
        },
        Err(_) => None,
    };

    FullConformanceReport {
        state_space_identity,
        transition_fidelity,
        primitivity,
        directness_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prism_module::{
        ExecError, ExecutionTrace, LoadError, PrismModule, SubstrateClass, SubstrateRequirements,
    };
    use crate::shape::{Shape, F_PRISM_FUSED_COMPONENT, F_PRISM_STRICT};
    use core::any::Any;

    /// A minimal but conformance-faithful test module.
    struct StrictTestModule;
    struct StrictCompiled;

    impl PrismModule for StrictTestModule {
        type Compiled = StrictCompiled;
        type Query = ();
        type Output = ();

        fn shape(&self) -> &'static Shape {
            F_PRISM_STRICT
        }

        fn name(&self) -> &'static str {
            "strict_test_module"
        }

        fn load(&self, _archive_data: &[u8]) -> Result<Self::Compiled, LoadError> {
            Ok(StrictCompiled)
        }

        fn execute(
            &self,
            _compiled: &Self::Compiled,
            _query: &Self::Query,
        ) -> Result<Self::Output, ExecError> {
            Ok(())
        }

        fn execute_traced(
            &self,
            _compiled: &Self::Compiled,
            _query: &Self::Query,
        ) -> Result<(Self::Output, ExecutionTrace), ExecError> {
            // Pretend we executed 5 ops, all primitive.
            Ok((
                (),
                ExecutionTrace {
                    total_substrate_operations: 5,
                    irreducible_operations: 5,
                    arena_inspections: 5,
                },
            ))
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

    /// A module that lies about its primitives (advertises an unknown op).
    struct LyingModule;

    impl PrismModule for LyingModule {
        type Compiled = ();
        type Query = ();
        type Output = ();

        fn shape(&self) -> &'static Shape {
            F_PRISM_STRICT
        }
        fn name(&self) -> &'static str {
            "lying_module"
        }
        fn primitive_operations(&self) -> &'static [&'static str] {
            &[
                "https://hologram.uor.foundation/op/bind",
                // missing the other four
                "https://hologram.uor.foundation/op/spurious_op", // not in shape
            ]
        }
        fn load(&self, _: &[u8]) -> Result<Self::Compiled, LoadError> {
            Ok(())
        }
        fn execute(&self, _: &Self::Compiled, _: &Self::Query) -> Result<Self::Output, ExecError> {
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
            0.0
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A module that admits non-primitive operations.
    struct NonPrimitiveModule;
    impl PrismModule for NonPrimitiveModule {
        type Compiled = ();
        type Query = ();
        type Output = ();

        fn shape(&self) -> &'static Shape {
            F_PRISM_STRICT
        }
        fn name(&self) -> &'static str {
            "non_primitive"
        }
        fn load(&self, _: &[u8]) -> Result<Self::Compiled, LoadError> {
            Ok(())
        }
        fn execute(&self, _: &Self::Compiled, _: &Self::Query) -> Result<Self::Output, ExecError> {
            Ok(())
        }
        fn execute_traced(
            &self,
            _: &Self::Compiled,
            _: &Self::Query,
        ) -> Result<(Self::Output, ExecutionTrace), ExecError> {
            // 10 ops, only 6 primitive — directness 0.6.
            Ok((
                (),
                ExecutionTrace {
                    total_substrate_operations: 10,
                    irreducible_operations: 6,
                    arena_inspections: 10,
                },
            ))
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
            0.6
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn transition_fidelity_passes_for_strict_module() {
        let m = StrictTestModule;
        let r = test_transition_fidelity(&m);
        assert!(r.passed, "failures: {:?}", r.failures);
        assert_eq!(r.operations_checked, 5);
        assert_eq!(r.family, "transition_fidelity");
    }

    #[test]
    fn transition_fidelity_fails_for_lying_module() {
        let m = LyingModule;
        let r = test_transition_fidelity(&m);
        assert!(!r.passed);
        // Should report both missing primitives and spurious ones.
        assert!(r.failures.iter().any(|f| f.contains("missing")));
        assert!(r.failures.iter().any(|f| f.contains("not in shape")));
    }

    #[test]
    fn primitivity_passes_for_strict_module() {
        let m = StrictTestModule;
        let r = test_primitivity(&m, &[], || ());
        assert!(r.passed, "failures: {:?}", r.failures);
    }

    #[test]
    fn primitivity_fails_for_non_primitive_module() {
        let m = NonPrimitiveModule;
        let r = test_primitivity(&m, &[], || ());
        assert!(!r.passed);
        assert!(r.failures[0].contains("not primitive"));
        assert!(r.failures[0].contains("0.6000"));
    }

    #[test]
    fn state_space_identity_passes_for_strict_module() {
        let m = StrictTestModule;
        let r = test_state_space_identity(&m, &[], || ());
        assert!(r.passed, "failures: {:?}", r.failures);
    }

    #[test]
    fn full_conformance_aggregates_all_three() {
        let m = StrictTestModule;
        let r = test_full_conformance(&m, &[], || ());
        assert!(r.passed());
        assert!(r.directness_ratio.is_some());
        assert!((r.directness_ratio.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fused_component_shape_can_be_referenced() {
        // Smoke test: verifies the fused-component shape constant is at least
        // syntactically usable.
        let s = F_PRISM_FUSED_COMPONENT;
        assert!(s.primitive_count() > 50);
    }
}
