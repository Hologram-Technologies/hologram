//! Cascade engine — 7-stage evaluation loop with certificate memoization.
//!
//! The engine runs the cascade pipeline as a tight loop over a function
//! pointer table indexed by `CascadeStage` discriminant. Each stage handler
//! returns a `Transition` that determines the next action.
//!
//! At Stage 0 (Init), the engine checks the `CertificateStore` for a cached
//! certificate matching `(unit_address, quantum_level)`. On cache hit, the
//! cascade skips to Extract via `Transition::Skip(Extract)`.

use hologram_core::term::{Binding, HoloCompileUnit, TermArena, TermId, TermKind};

use crate::certificate::{Certificate, CertificateStore};
use crate::stage::{CascadeStage, CascadeState, HaltReason, Transition};

/// Recursively evaluate a term to an i64 value, if possible.
///
/// Handles literals, unary/binary PrimOp applications, variable lookups
/// (via binding resolution), and passthrough nodes. Returns `None` for
/// terms that cannot be statically evaluated (LutApp, FloatApp, etc.).
///
/// Depth-limited to 64 to prevent stack overflow on pathological inputs.
fn eval_term(
    arena: &TermArena,
    bindings: &[Binding],
    binding_count: u8,
    id: TermId,
    depth: u8,
) -> Option<i64> {
    if depth > 64 {
        return None;
    }
    let node = arena.get(id);
    match &node.kind {
        TermKind::IntLit(v) => Some(*v),
        TermKind::BrailleLit(v) => Some(*v as i64),
        TermKind::QuantumLit { value, .. } => Some(*value as i64),

        TermKind::UnaryApp { op, arg } => {
            let val = eval_term(arena, bindings, binding_count, *arg, depth + 1)?;
            Some(op.apply_unary(val as u8) as i64)
        }

        TermKind::BinaryApp { op, lhs, rhs } => {
            let l = eval_term(arena, bindings, binding_count, *lhs, depth + 1)?;
            let r = eval_term(arena, bindings, binding_count, *rhs, depth + 1)?;
            Some(op.apply_binary(l as u8, r as u8) as i64)
        }

        TermKind::Var(var_id) => {
            for i in 0..binding_count as usize {
                if bindings[i].var == *var_id {
                    return eval_term(arena, bindings, binding_count, bindings[i].rhs, depth + 1);
                }
            }
            None
        }

        TermKind::Passthrough(inner) => {
            eval_term(arena, bindings, binding_count, *inner, depth + 1)
        }

        // LutApp, FloatApp, RingUnaryApp, RingBinaryApp, Constant,
        // GraphInput, GraphOutput, FusedViewRef — cannot evaluate statically.
        _ => None,
    }
}

/// Result of a successful cascade evaluation.
#[derive(Debug)]
pub struct CascadeResult {
    /// Final state at convergence.
    pub state: CascadeState,
    /// Whether the result was served from the certificate cache.
    pub cache_hit: bool,
}

/// Stage handler function signature.
type StageHandler = fn(&mut CascadeState, &HoloCompileUnit, &CertificateStore) -> Transition;

/// 7-entry function pointer table indexed by `CascadeStage as usize`.
const HANDLERS: [StageHandler; CascadeStage::COUNT] = [
    handle_init,
    handle_declare,
    handle_factorize,
    handle_resolve,
    handle_attest,
    handle_extract,
    handle_converge,
];

/// Run the full cascade pipeline for a CompileUnit.
///
/// The unit must have passed preflight (shape validation, budget solvency,
/// unit address computation) before being submitted to the cascade.
pub fn run_cascade(
    unit: &HoloCompileUnit,
    store: &mut CertificateStore,
) -> Result<CascadeResult, HaltReason> {
    let state = CascadeState::from_unit(
        unit.unit_address,
        unit.quantum_level,
        unit.thermodynamic_budget,
    );

    run_cascade_loop(state, unit, store)
}

// ── Stage Handlers ───────────────────────────────────────────────────────────

/// Stage 0 — Init (Ω⁰): Check certificate cache, initialize graph from CompileUnit.
fn handle_init(
    state: &mut CascadeState,
    _unit: &HoloCompileUnit,
    store: &CertificateStore,
) -> Transition {
    // Memoization: check if this (address, level) pair has been evaluated before.
    // On cache hit, skip stages 1-4 entirely. Single match arm for branch predictor.
    if matches!(
        store.get(&state.unit_address, state.quantum_level),
        Some(cert) if cert.converged
    ) {
        return Transition::Skip(CascadeStage::Extract);
    }

    // Initialize the graph from the CompileUnit via term lowering.
    // term_lower lives in hologram-compiler (bridge layer). To avoid a
    // circular dependency (cascade is kernel, compiler is bridge), the
    // graph is expected to be pre-lowered and passed in via the state.
    // The `run_cascade_with_graph()` entry point accepts a pre-lowered graph.
    // For the standard path, `run_cascade()` expects the caller to have
    // lowered the term before submission.
    state.budget_consumed = 0.0;
    Transition::Advance
}

/// Stage 1 — Declare (Ω¹): Select resolver, promote ring levels.
///
/// Calls `precision::promote_prim_ring_levels()` to analyze output distributions
/// and promote Prim nodes to RingPrimUnary/RingPrimBinary where needed.
/// Cost: O(n) where n = graph node count.
fn handle_declare(
    state: &mut CascadeState,
    unit: &HoloCompileUnit,
    _store: &CertificateStore,
) -> Transition {
    if let Some(ref mut graph) = state.graph {
        let promoted = crate::precision::promote_prim_ring_levels(graph);
        state.ring_prims_promoted = promoted;
        let cost = (promoted.max(1) as f64) * core::f64::consts::LN_2;
        state.budget_consumed += cost;
    } else {
        state.budget_consumed += core::f64::consts::LN_2;
    }

    // Register effect declarations from the compile unit
    for i in 0..unit.effect_decl_count as usize {
        let decl = &unit.effect_decls[i];
        let _ = state.effect_store.register(
            "",
            &decl.target_fibers[..decl.fiber_count as usize],
            decl.budget_delta,
            decl.commutes,
        );
    }

    // Register dispatch declarations from the compile unit
    for i in 0..unit.dispatch_decl_count as usize {
        let decl = &unit.dispatch_decls[i];
        let _ = state
            .dispatch_registry
            .register(&[], "", decl.resolver_id, decl.priority);
    }

    if state.budget_exceeded() {
        return Transition::Halt(HaltReason::BudgetExhausted {
            consumed: state.budget_consumed,
            allocated: state.budget_allocated,
        });
    }

    Transition::Advance
}

/// Stage 2 — Factorize (Ω²): CSE, constant folding, view fusion.
///
/// Calls `hologram_graph::fuse()` to run all fusion passes.
/// Respects `state.skip_fusion` to bypass when disabled.
/// Captures `FusionStats` into `state.fusion_stats`.
fn handle_factorize(
    state: &mut CascadeState,
    _unit: &HoloCompileUnit,
    _store: &CertificateStore,
) -> Transition {
    if let Some(ref mut graph) = state.graph {
        if !state.skip_fusion {
            match hologram_graph::fuse(graph) {
                Ok(stats) => {
                    state.fusion_stats = stats;
                }
                Err(e) => {
                    return Transition::Halt(HaltReason::StageFailure {
                        stage: CascadeStage::Factorize,
                        message: format!("fusion failed: {}", e),
                    });
                }
            }
        }
        let cost = (graph.node_count() as f64) * core::f64::consts::LN_2;
        state.budget_consumed += cost;
    } else {
        let cost = core::f64::consts::LN_2;
        state.budget_consumed += cost;
    }

    if state.budget_exceeded() {
        return Transition::Halt(HaltReason::BudgetExhausted {
            consumed: state.budget_consumed,
            allocated: state.budget_allocated,
        });
    }

    Transition::Advance
}

/// Stage 3 — Resolve (Ω³): Build execution schedule via Kahn's algorithm.
fn handle_resolve(
    state: &mut CascadeState,
    _unit: &HoloCompileUnit,
    _store: &CertificateStore,
) -> Transition {
    if let Some(ref graph) = state.graph {
        match hologram_graph::ExecutionSchedule::build(graph) {
            Ok(schedule) => {
                state.schedule = Some(schedule);
            }
            Err(e) => {
                return Transition::Halt(HaltReason::Contradiction(format!(
                    "scheduling failed (cycle or invalid graph): {}",
                    e
                )));
            }
        }
        let cost = core::f64::consts::LN_2;
        state.budget_consumed += cost;
    } else {
        let cost = core::f64::consts::LN_2;
        state.budget_consumed += cost;
    }

    if state.budget_exceeded() {
        return Transition::Halt(HaltReason::BudgetExhausted {
            consumed: state.budget_consumed,
            allocated: state.budget_allocated,
        });
    }

    Transition::Advance
}

/// Stage 4 — Attest (Ω⁴): Compute liveness intervals, plan workspace, QEDL boundaries,
/// verify assertions, check for contradictions.
fn handle_attest(
    state: &mut CascadeState,
    unit: &HoloCompileUnit,
    _store: &CertificateStore,
) -> Transition {
    // Check assertions from the CompileUnit.
    for i in 0..unit.assertion_count as usize {
        let assertion = &unit.assertions[i];

        // Recursively evaluate both sides of the assertion.
        if let (Some(lhs_val), Some(rhs_val)) = (
            eval_term(
                &unit.arena,
                &*unit.bindings,
                unit.binding_count,
                assertion.lhs,
                0,
            ),
            eval_term(
                &unit.arena,
                &*unit.bindings,
                unit.binding_count,
                assertion.rhs,
                0,
            ),
        ) {
            if lhs_val != rhs_val {
                return Transition::Halt(HaltReason::Contradiction(format!(
                    "assertion {} failed: lhs={} rhs={}",
                    i, lhs_val, rhs_val
                )));
            }
        }
        // Non-evaluable assertions (contain LutApp, FloatApp, etc.) are skipped —
        // they require runtime execution and cannot be checked at compile time.

        state.budget_consumed += core::f64::consts::LN_2;
        if state.budget_exceeded() {
            return Transition::Halt(HaltReason::BudgetExhausted {
                consumed: state.budget_consumed,
                allocated: state.budget_allocated,
            });
        }
    }

    if let (Some(ref graph), Some(ref schedule)) = (&state.graph, &state.schedule) {
        let intervals = crate::liveness::compute_liveness(schedule, graph);
        let layout = crate::workspace::plan_workspace(&intervals);

        // Compute QEDL domain-crossing boundaries from schedule topo order.
        let topo_order: Vec<hologram_graph::graph::node::NodeId> = schedule
            .levels
            .iter()
            .flat_map(|level| level.node_ids.iter().copied())
            .collect();
        let qedl = crate::qedl::insert_qedl_boundaries(graph, &topo_order);

        let cost = (intervals.len().max(1) as f64) * core::f64::consts::LN_2;
        state.budget_consumed += cost;

        state.liveness_intervals = Some(intervals);
        state.workspace_layout = Some(layout);
        state.qedl_boundaries = Some(qedl);

        // Validate grounding at QEDL Dequantize boundaries
        if let Some(ref boundaries) = state.qedl_boundaries {
            for (_, boundary, _) in boundaries {
                if matches!(boundary, crate::qedl::QedlBoundary::Dequantize) {
                    let level = hologram_core::op::RingLevel::from(state.quantum_level);
                    let bw = level.byte_width() as usize;
                    let test = vec![0u8; bw];
                    if hologram_core::ring::grounding::ground_at_level(level, &test).is_none() {
                        return Transition::Halt(HaltReason::StageFailure {
                            stage: CascadeStage::Attest,
                            message: "grounding not available at declared quantum level".into(),
                        });
                    }
                    break; // Only need to validate once per level
                }
            }
        }
    } else {
        state.budget_consumed += core::f64::consts::LN_2;
    }

    if state.budget_exceeded() {
        return Transition::Halt(HaltReason::BudgetExhausted {
            consumed: state.budget_consumed,
            allocated: state.budget_allocated,
        });
    }

    Transition::Advance
}

/// Stage 5 — Extract (Ω⁵): Build execution tape from graph + schedule.
/// Re-entry point on certificate cache hit (graph/schedule may be absent).
fn handle_extract(
    state: &mut CascadeState,
    _unit: &HoloCompileUnit,
    _store: &CertificateStore,
) -> Transition {
    if let (Some(ref graph), Some(ref schedule)) = (&state.graph, &state.schedule) {
        let sg = hologram_archive::format::graph::SerializedGraph::from_graph(graph);
        match crate::tape_builder::build_tape(&sg, schedule, None) {
            Ok(tape) => {
                state.tape = Some(tape);
                state.serialized_graph = Some(sg);
            }
            Err(e) => {
                return Transition::Halt(HaltReason::StageFailure {
                    stage: CascadeStage::Extract,
                    message: format!("tape build failed: {}", e),
                });
            }
        }
        let cost = (graph.node_count() as f64) * core::f64::consts::LN_2;
        state.budget_consumed += cost;
    } else {
        // Cache-hit path: no graph/schedule available, just advance.
        state.budget_consumed += core::f64::consts::LN_2;
    }

    if state.budget_exceeded() {
        return Transition::Halt(HaltReason::BudgetExhausted {
            consumed: state.budget_consumed,
            allocated: state.budget_allocated,
        });
    }

    Transition::Advance
}

/// Stage 6 — Converge (π): Emit archive with LayerHeader + CompileUnitMeta, write certificate.
fn handle_converge(
    state: &mut CascadeState,
    unit: &HoloCompileUnit,
    _store: &CertificateStore,
) -> Transition {
    // Produce .holo archive bytes if graph is available.
    if let Some(ref graph) = state.graph {
        let layer_header = build_layer_header(graph, &state.schedule);
        let unit_meta = hologram_archive::section::compile_unit_meta::CompileUnitMeta {
            unit_address: state.unit_address,
            quantum_level: state.quantum_level.index() as u8,
            budget: state.budget_allocated,
            domain_count: unit.target_domain_count,
            term_count: unit.arena.len(),
            binding_count: unit.binding_count,
            assertion_count: unit.assertion_count,
        };
        let mut writer = hologram_archive::HoloWriter::new().set_graph(graph);
        writer = writer.add_section(&layer_header);
        writer = writer.add_section(&unit_meta);
        match writer.build() {
            Ok(bytes) => {
                state.archive_bytes = Some(bytes);
            }
            Err(e) => {
                return Transition::Halt(HaltReason::StageFailure {
                    stage: CascadeStage::Converge,
                    message: format!("archive emission failed: {}", e),
                });
            }
        }
    }

    Transition::Converged
}

/// Build a LayerHeader describing the graph as a single layer.
fn build_layer_header(
    graph: &hologram_graph::Graph,
    schedule: &Option<hologram_graph::ExecutionSchedule>,
) -> hologram_archive::LayerHeader {
    use hologram_archive::entrypoint::{LayerDescriptor, LayerEntrypoint, LayerId, TensorPort};
    use hologram_archive::weight::WeightDType;

    let inputs: Vec<TensorPort> = graph
        .inputs()
        .iter()
        .map(|name| TensorPort {
            name: name.clone(),
            shape: vec![1],
            dtype: WeightDType::U8,
        })
        .collect();

    let outputs: Vec<TensorPort> = graph
        .outputs()
        .iter()
        .map(|(name, _)| TensorPort {
            name: name.clone(),
            shape: vec![1],
            dtype: WeightDType::U8,
        })
        .collect();

    let descriptor = LayerDescriptor {
        id: LayerId(0),
        name: "main".into(),
        entrypoint: LayerEntrypoint::Graph,
        inputs,
        outputs,
        group: 0,
        plan_offset: 0,
        plan_size: 0,
    };

    let sched_levels = if let Some(ref s) = schedule {
        vec![vec![LayerId(0); s.num_levels()]]
    } else {
        vec![]
    };

    hologram_archive::LayerHeader {
        layers: vec![descriptor],
        schedule: sched_levels,
    }
}

/// Run the cascade with a pre-lowered graph.
///
/// This is the primary entry point when the caller has already lowered
/// a CompileUnit to a Graph (e.g., via `term_lower::lower_to_graph`).
pub fn run_cascade_with_graph(
    unit: &HoloCompileUnit,
    graph: hologram_graph::Graph,
    store: &mut CertificateStore,
) -> Result<CascadeResult, HaltReason> {
    run_cascade_with_graph_opts(unit, graph, store, false)
}

/// Run the cascade with a pre-lowered graph and configuration options.
///
/// `skip_fusion`: if true, the Factorize stage skips fusion passes.
pub fn run_cascade_with_graph_opts(
    unit: &HoloCompileUnit,
    graph: hologram_graph::Graph,
    store: &mut CertificateStore,
    skip_fusion: bool,
) -> Result<CascadeResult, HaltReason> {
    let mut state = CascadeState::from_unit(
        unit.unit_address,
        unit.quantum_level,
        unit.thermodynamic_budget,
    );
    state.graph = Some(graph);
    state.skip_fusion = skip_fusion;

    run_cascade_loop(state, unit, store)
}

/// Core cascade loop shared by all entry points.
fn run_cascade_loop(
    mut state: CascadeState,
    unit: &HoloCompileUnit,
    store: &mut CertificateStore,
) -> Result<CascadeResult, HaltReason> {
    let mut cache_hit = false;

    loop {
        let handler = HANDLERS[state.stage as usize];
        match handler(&mut state, unit, store) {
            Transition::Advance => {
                state.stage = state.stage.next();
            }
            Transition::Skip(target) => {
                cache_hit = target == CascadeStage::Extract && state.stage == CascadeStage::Init;
                state.stage = target;
            }
            Transition::Converged => {
                store.insert(Certificate {
                    unit_address: state.unit_address,
                    quantum_level: state.quantum_level,
                    budget_consumed: state.budget_consumed,
                    converged: true,
                });
                return Ok(CascadeResult { state, cache_hit });
            }
            Transition::Halt(reason) => {
                return Err(reason);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::term::{TermArena, TermKind};
    use uor_foundation::enums::VerificationDomain;
    use uor_foundation::QuantumLevel;

    fn make_unit(budget: f64) -> HoloCompileUnit {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(42));
        let mut unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::Q0,
            budget,
            &[VerificationDomain::Algebraic],
        );
        let hash = *blake3::hash(b"test").as_bytes();
        unit.unit_address = hash;
        unit.address = hologram_core::term::HoloAddress::from_hash(hash);
        unit
    }

    #[test]
    fn cascade_converges() {
        let unit = make_unit(100.0);
        let mut store = CertificateStore::new(16);

        let result = run_cascade(&unit, &mut store).unwrap();
        assert_eq!(result.state.stage, CascadeStage::Converge);
        assert!(!result.cache_hit);
        assert!(result.state.budget_consumed > 0.0);
    }

    #[test]
    fn cascade_writes_certificate() {
        let unit = make_unit(100.0);
        let mut store = CertificateStore::new(16);

        run_cascade(&unit, &mut store).unwrap();

        let cert = store.get(&unit.unit_address, QuantumLevel::Q0);
        assert!(cert.is_some());
        assert!(cert.unwrap().converged);
    }

    #[test]
    fn cascade_cache_hit_skips_stages() {
        let unit = make_unit(100.0);
        let mut store = CertificateStore::new(16);

        let result1 = run_cascade(&unit, &mut store).unwrap();
        assert!(!result1.cache_hit);

        let result2 = run_cascade(&unit, &mut store).unwrap();
        assert!(result2.cache_hit);
        assert_eq!(result2.state.stage, CascadeStage::Converge);
    }

    #[test]
    fn cascade_budget_exhaustion() {
        let unit = make_unit(0.5);
        let mut store = CertificateStore::new(16);

        let result = run_cascade(&unit, &mut store);
        assert!(result.is_err());
        match result.unwrap_err() {
            HaltReason::BudgetExhausted { .. } => {}
            other => panic!("expected BudgetExhausted, got {:?}", other),
        }
    }

    #[test]
    fn cascade_stage_transitions() {
        let unit = make_unit(1000.0);
        let mut store = CertificateStore::new(16);

        let result = run_cascade(&unit, &mut store).unwrap();
        assert_eq!(result.state.stage, CascadeStage::Converge);
    }

    #[test]
    fn different_addresses_no_cache_hit() {
        let unit1 = make_unit(100.0);
        let mut unit2 = make_unit(100.0);
        let hash2 = *blake3::hash(b"different").as_bytes();
        unit2.unit_address = hash2;
        unit2.address = hologram_core::term::HoloAddress::from_hash(hash2);

        let mut store = CertificateStore::new(16);

        run_cascade(&unit1, &mut store).unwrap();
        let result = run_cascade(&unit2, &mut store).unwrap();
        assert!(!result.cache_hit);
    }

    #[test]
    fn different_levels_no_cache_hit() {
        let unit_q0 = make_unit(100.0);
        let mut unit_q1 = make_unit(100.0);
        unit_q1.quantum_level = QuantumLevel::Q1;
        unit_q1.unit_address = unit_q0.unit_address;
        unit_q1.address = hologram_core::term::HoloAddress::from_hash(unit_q0.unit_address);

        let mut store = CertificateStore::new(16);

        run_cascade(&unit_q0, &mut store).unwrap();
        let result = run_cascade(&unit_q1, &mut store).unwrap();
        assert!(!result.cache_hit);
    }

    #[test]
    fn cascade_with_graph_runs_fusion() {
        use hologram_core::op::PrimOp;
        use hologram_graph::{GraphBuilder, GraphOp};

        let unit = make_unit(1000.0);
        let graph = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Prim(PrimOp::Neg), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("result", 2)
            .input("x")
            .build();

        let mut store = CertificateStore::new(16);
        let result = run_cascade_with_graph(&unit, graph, &mut store).unwrap();

        assert_eq!(result.state.stage, CascadeStage::Converge);
        assert!(result.state.archive_bytes.is_some());
        assert!(result.state.schedule.is_some());
        assert!(result.state.tape.is_some());
        assert!(result.state.serialized_graph.is_some());
    }

    #[test]
    fn cascade_without_graph_no_tape() {
        let unit = make_unit(1000.0);
        let mut store = CertificateStore::new(16);
        let result = run_cascade(&unit, &mut store).unwrap();
        assert!(result.state.tape.is_none());
        assert!(result.state.serialized_graph.is_none());
    }

    #[test]
    fn cascade_with_graph_populates_all_state_fields() {
        use hologram_core::op::PrimOp;
        use hologram_graph::{GraphBuilder, GraphOp};

        let unit = make_unit(1000.0);
        let graph = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Prim(PrimOp::Neg), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("result", 2)
            .input("x")
            .build();

        let mut store = CertificateStore::new(16);
        let result = run_cascade_with_graph(&unit, graph, &mut store).unwrap();

        assert!(
            result
                .state
                .liveness_intervals
                .as_ref()
                .map_or(false, |v| !v.is_empty()),
            "liveness_intervals should be populated"
        );
        assert!(
            result.state.workspace_layout.is_some(),
            "workspace_layout should be populated"
        );
        let _ = &result.state.qedl_boundaries;
    }
}
