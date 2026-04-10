//! Compiler pipeline: structure-finder compilation to `.holo` archive.
//!
//! Per Prism section 4 of the SCS framework, the compiler is a
//! *structure-finder*, not a constructor. The pipeline runs as a flat
//! sequence of finder passes:
//!
//! 1. **Precision** — observable-guided ring-level inference
//!    ([`hologram_ir::analysis::precision::promote_prim_ring_levels`])
//! 2. **Pattern detection** — fusion opportunity detection
//!    ([`hologram_ir::analysis::analyze`])
//! 3. **Schedule** — topological-level scheduling
//!    ([`hologram_ir::ExecutionSchedule::build`])
//! 4. **Liveness + workspace** — buffer lifetime intervals and slot reuse
//!    ([`hologram_ir::analysis::liveness::compute_liveness`],
//!    [`hologram_ir::analysis::workspace::plan_workspace`])
//! 5. **QEDL boundaries** — domain-crossing detection
//!    ([`hologram_ir::analysis::qedl::insert_qedl_boundaries`])
//! 6. **Tape build** — emission via [`hologram_fused_component::tape_builder::build_tape`]
//! 7. **Archive emission** — `LayerHeader` + `CompileUnitMeta` via
//!    [`hologram_archive::HoloWriter`]
//!
//! Public surface:
//!
//! - [`Compiler::compile`] — explicit compile method taking a [`SourceInput`].
//! - [`compile`] — convenience wrapper that calls
//!   `Compiler::default().compile(SourceInput::Graph(graph))`.
//! - [`crate::compile_from_source`] — convenience wrapper that calls
//!   `Compiler::default().compile(SourceInput::TermSource { ... })`.
//!
//! # Pipeline shape
//!
//! The compiler is a structure-finder, not a state machine. It sequences
//! the analyses in [`hologram_ir::analysis`] directly: no state
//! machine, no certificate store, no budget-tracking field updates, no
//! fusion knob, and no per-stage metadata in the output. Each pass reads
//! preexisting structural content from the source graph and emits a
//! plan-shaped artifact.

use hologram_ir::analysis::{liveness, precision, qedl, workspace, StructuralFindings};
use hologram_ir::graph::node::NodeId;
use hologram_ir::graph::Graph;
use hologram_ir::schedule::ExecutionSchedule;

use hologram_core::op::RingLevel;
use hologram_core::term::{HoloAddress, HoloCompileUnit, TermArena, TermKind};
use hologram_foundation::enums::VerificationDomain;
use hologram_foundation::WittLevel;

use crate::error::{CompileError, CompileResult};

/// Re-export the QEDL boundary type from the analysis layer for consumers
/// that pattern-match against compilation output.
pub use hologram_ir::analysis::qedl::{EncodingId, QedlBoundary};

/// Statistics from the compilation process.
///
/// Reframed from v0.1.4's `CompilationStats`. Each field describes a
/// structural finding the compiler made about the source graph, not an
/// optimisation decision the compiler made about the output.
#[derive(Debug, Clone, Default)]
pub struct CompilationStats {
    /// Number of workspace buffer slots (after reuse).
    pub workspace_slots: usize,
    /// Peak number of simultaneously live buffers.
    pub peak_live_buffers: usize,
    /// Total nodes in the graph.
    pub total_nodes: usize,
    /// Number of execution schedule levels.
    pub schedule_levels: usize,
    /// Findings from the structural analysis pass.
    pub findings: StructuralFindings,
    /// Number of Prim nodes promoted to RingPrimUnary/Binary by the precision pass.
    pub ring_prims_promoted: usize,
}

/// Output of the compilation pipeline.
#[derive(Debug, Clone)]
pub struct CompilationOutput {
    /// The compiled `.holo` archive bytes.
    pub archive: Vec<u8>,
    /// Compilation statistics.
    pub stats: CompilationStats,
    /// The execution schedule.
    pub schedule: ExecutionSchedule,
    /// QEDL pipeline boundaries: nodes where quantize/dequantize transitions occur.
    pub qedl_boundaries: Vec<(NodeId, QedlBoundary, EncodingId)>,
    /// Content-addressed identifier of the compiled unit (BLAKE3 of graph structure).
    pub unit_address: [u8; 32],
}

/// Input source for the compiler.
///
/// The `Term` variant carries a full `HoloCompileUnit` (~2 KB) — boxing
/// would add indirection for a payload that is consumed exactly once, so
/// we allow the size difference here.
///
/// # Examples
///
/// Compile a raw graph (the simplest path):
///
/// ```
/// use hologram_compiler::{Compiler, SourceInput};
/// use hologram_ir::Graph;
///
/// let _ = Compiler::default().compile(SourceInput::Graph(Graph::new()));
/// ```
///
/// Compile from UOR term-language source text:
///
/// ```
/// use hologram_compiler::{Compiler, SourceInput};
/// use hologram_foundation::WittLevel;
/// use hologram_foundation::enums::VerificationDomain;
///
/// let _ = Compiler::default().compile(SourceInput::TermSource {
///     source: "42".to_owned(),
///     level: WittLevel::W8,
///     budget: 100.0,
///     domains: vec![VerificationDomain::Algebraic],
/// });
/// ```
#[allow(clippy::large_enum_variant)]
pub enum SourceInput {
    /// Raw graph — will be wrapped in a synthetic CompileUnit.
    Graph(Graph),
    /// Pre-built CompileUnit + pre-lowered Graph.
    Term { unit: HoloCompileUnit, graph: Graph },
    /// UOR term language source text. The compiler parses, runs
    /// preflight, and lowers to a Graph internally.
    TermSource {
        source: String,
        level: WittLevel,
        budget: f64,
        domains: Vec<VerificationDomain>,
    },
}

/// Minimal Prism module registry.
///
/// # ⚠️ Scaffold only
///
/// **This type currently holds no module instances.** The routing
/// function in [`Compiler::compile`] always picks `FusedComponentModule`,
/// regardless of what is "registered" here. The struct exists so that
/// adding a second `PrismModule` is a non-breaking change to
/// [`Compiler::new`]'s signature — when that day comes, this becomes a
/// real `Vec<Box<dyn PrismModuleHandle>>` (the type-erased handle trait
/// does not exist yet either) plus a `route(source: &SourceInput) -> &dyn
/// PrismModuleHandle` method that picks the maximum-directness module
/// whose `substrate_requirements()` are satisfied by the declared
/// `SubstrateClass`.
///
/// **Do not** treat this type as a real registry today. Constructing it
/// via [`PrismModuleRegistry::single_fused_component`] documents intent;
/// it does not change behaviour because there is only one module.
///
/// **Perf: NEUTRAL.** The registry is consulted once per `compile`
/// call, never inside the kernel hot path.
#[derive(Debug, Default, Clone, Copy)]
pub struct PrismModuleRegistry {
    /// Number of registered Prism modules. Currently fixed at 1.
    /// When a second module lands, this becomes a real `Vec<...>` of
    /// type-erased handles plus a routing function.
    _registered: usize,
}

impl PrismModuleRegistry {
    /// Construct a registry containing only `FusedComponentModule`.
    /// This is the default for the current single-module deployment.
    #[must_use]
    pub const fn single_fused_component() -> Self {
        Self { _registered: 1 }
    }
}

/// The structure-finder compiler.
///
/// Constructed once per compilation campaign and reused across many
/// `compile()` calls. Holds a [`PrismModuleRegistry`] (the set of Prism
/// modules the compiler can route to) and a target [`SubstrateClass`]
/// (the deployment substrate, used by routing to filter modules whose
/// `substrate_requirements()` are satisfied).
///
/// The current single-module setup means routing is trivial: every
/// `compile` call resolves to `FusedComponentModule`'s
/// `F_prism_fused_component` shape, and the compiler emits an archive
/// declaring that shape via the conformance-shape section. When a
/// second Prism module lands, the routing function will select the
/// maximum-directness module whose `substrate_requirements()` match the
/// declared substrate.
///
/// **Perf:** the trait surface is consulted at compile time only.
#[derive(Debug, Clone, Copy)]
pub struct Compiler {
    registry: PrismModuleRegistry,
    substrate: hologram_shapes::prism_module::SubstrateClass,
}

impl Default for Compiler {
    /// Default `Compiler` for the current single-module deployment:
    /// `FusedComponentModule` targeting x86_64.
    fn default() -> Self {
        Self {
            registry: PrismModuleRegistry::single_fused_component(),
            substrate: hologram_shapes::prism_module::SubstrateClass::X86_64,
        }
    }
}

impl Compiler {
    /// Construct a compiler with an explicit registry and substrate.
    #[must_use]
    pub const fn new(
        registry: PrismModuleRegistry,
        substrate: hologram_shapes::prism_module::SubstrateClass,
    ) -> Self {
        Self {
            registry,
            substrate,
        }
    }

    /// Compile a `SourceInput` into a `.holo` archive.
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_compiler::{Compiler, SourceInput};
    /// use hologram_ir::Graph;
    ///
    /// let compiler = Compiler::default();
    /// let out = compiler.compile(SourceInput::Graph(Graph::new())).unwrap();
    /// assert!(!out.archive.is_empty());
    /// ```
    pub fn compile(&self, source: SourceInput) -> CompileResult<CompilationOutput> {
        // FIXME(phase-12): real routing. Currently the registry has
        // exactly one entry and the substrate is unused — every compile
        // resolves to FusedComponentModule. The discard below documents
        // that the fields are intentionally unread until a second
        // PrismModule lands.
        let _ = (self.registry, self.substrate);

        match source {
            SourceInput::Graph(graph) => {
                let unit = unit_from_graph(&graph);
                compile_via_finder_pipeline(unit, graph)
            }
            SourceInput::Term { unit, graph } => compile_via_finder_pipeline(unit, graph),
            SourceInput::TermSource {
                source,
                level,
                budget,
                domains,
            } => {
                let parsed = crate::term_parser::parse(&source)
                    .map_err(|e| CompileError::Validation(e.to_string()))?;

                let mut unit =
                    HoloCompileUnit::new(parsed.arena, parsed.root, level, budget, &domains);
                unit.bindings = parsed.bindings;
                unit.binding_count = parsed.binding_count;
                unit.assertions = parsed.assertions;
                unit.assertion_count = parsed.assertion_count;
                unit.type_decls = parsed.type_decls;
                unit.type_decl_count = parsed.type_decl_count;

                crate::preflight::run_preflight(&mut unit)
                    .map_err(|e| CompileError::Validation(e.to_string()))?;

                let graph = crate::term_lower::lower_to_graph(&unit)?;
                compile_via_finder_pipeline(unit, graph)
            }
        }
    }
}

/// Compile a graph into a `.holo` archive with default settings.
///
/// Convenience wrapper around `Compiler::default().compile(SourceInput::Graph(graph))`.
pub fn compile(graph: Graph) -> CompileResult<CompilationOutput> {
    Compiler::default().compile(SourceInput::Graph(graph))
}

/// Compile from a `uor!` macro expansion (enforcement `TermArena`).
///
/// The `uor!` macro parses EBNF surface syntax at compile time and produces a
/// typed `enforcement::TermArena`. This function converts it to hologram's
/// arena, runs preflight (including enforcement validation), lowers to a Graph,
/// and runs the structure-finder pipeline.
pub fn compile_uor_arena<const CAP: usize>(
    enforcement_arena: &hologram_foundation::enforcement::TermArena<CAP>,
    root_index: u32,
    level: RingLevel,
    budget: f64,
    domains: &[VerificationDomain],
) -> CompileResult<CompilationOutput> {
    // Phase 2 bridge: enforcement → hologram term arena
    let (arena, root) = hologram_core::term::enforcement_bridge::convert_enforcement_arena(
        enforcement_arena,
        root_index,
    )
    .map_err(|e| CompileError::Validation(format!("enforcement bridge: {e:?}")))?;

    // Build CompileUnit
    let mut unit = HoloCompileUnit::new(arena, root, level.into(), budget, domains);

    // Phase 3 preflight (includes enforcement validation)
    crate::preflight::run_preflight(&mut unit)
        .map_err(|e| CompileError::Validation(e.to_string()))?;

    // Lower to graph
    let graph = crate::term_lower::lower_to_graph(&unit)?;

    compile_via_finder_pipeline(unit, graph)
}

/// Build a `HoloCompileUnit` from a `Graph`.
///
/// This is the primary integration point for consumers like hologram-ai
/// that build graphs via `GraphBuilder`. The unit address is computed from
/// the graph structure (BLAKE3 hash of node ops and edges) — content-addressed
/// identity is a structural property of the graph, not a constructor decision.
///
/// Default parameters:
/// - Quantum level: `Q0`
/// - Budget: `f64::MAX` (unconstrained for graph-based compilation)
/// - Domains: `[Algebraic]`
pub fn unit_from_graph(graph: &Graph) -> HoloCompileUnit {
    unit_from_graph_with(
        graph,
        WittLevel::W8,
        f64::MAX,
        &[VerificationDomain::Algebraic],
    )
}

/// Build a `HoloCompileUnit` from a `Graph` with explicit parameters.
pub fn unit_from_graph_with(
    graph: &Graph,
    level: WittLevel,
    budget: f64,
    domains: &[VerificationDomain],
) -> HoloCompileUnit {
    let mut arena = TermArena::new();
    let root = arena.alloc(TermKind::IntLit(0));
    let mut unit = HoloCompileUnit::new(arena, root, level, budget, domains);

    // Compute content-addressed hash from graph structure.
    // Uses Hash trait on GraphOp (zero-allocation) instead of Debug formatting.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = blake3::Hasher::new();
    hasher.update(&(graph.node_count() as u64).to_le_bytes());
    for node in graph.nodes() {
        let mut h = DefaultHasher::new();
        node.op.hash(&mut h);
        hasher.update(&h.finish().to_le_bytes());
        for dep in node.dependencies() {
            hasher.update(&dep.index().to_le_bytes());
        }
    }
    let hash = *hasher.finalize().as_bytes();
    unit.unit_address = hash;
    unit.address = HoloAddress::from_hash(hash);
    unit
}

/// Core compilation pipeline — sequential structure-finder passes.
///
/// Each pass is a finder that reads structural content from the source
/// graph; the pipeline produces a `.holo` archive carrying the layer
/// header, compile-unit metadata, and conformance shape declaration.
fn compile_via_finder_pipeline(
    unit: HoloCompileUnit,
    graph: Graph,
) -> CompileResult<CompilationOutput> {
    let unit_address = unit.unit_address;
    let mut graph = graph;

    // 1. Precision: observable-guided ring-level inference (mutates graph).
    let ring_prims_promoted = precision::promote_prim_ring_levels(&mut graph);

    // 2. Structural pattern detection: fusion / view detection / cse / constants.
    let findings = hologram_ir::analysis::analyze(&mut graph)
        .map_err(|e| CompileError::Validation(format!("analysis pass: {}", e)))?;

    // 2b. Compile-time assertion verification.
    verify_assertions(&unit)?;

    // 3. Build the topological execution schedule.
    let schedule = ExecutionSchedule::build(&graph).map_err(|e| {
        CompileError::Validation(format!("scheduling failed (cycle or invalid graph): {}", e))
    })?;

    // 4. Liveness intervals + workspace layout.
    let intervals = liveness::compute_liveness(&schedule, &graph);
    let layout = workspace::plan_workspace(&intervals);

    // 5. QEDL domain-crossing boundaries.
    let topo_order: Vec<NodeId> = schedule
        .levels
        .iter()
        .flat_map(|level| level.node_ids.iter().copied())
        .collect();
    let qedl_boundaries = qedl::insert_qedl_boundaries(&graph, &topo_order);

    // 5b. Validate grounding at QEDL Dequantize boundaries.
    //
    // The check needs a wire-format `RingLevel` to query the grounding
    // table. The unit's `WittLevel` may be any bit width; only the four
    // spec-named levels (W8/W16/W24/W32) have grounding tables. Reject
    // explicitly here rather than silently falling back.
    for (_, boundary, _) in &qedl_boundaries {
        if matches!(boundary, QedlBoundary::Dequantize) {
            let level = RingLevel::from_witt_level(unit.witt_level).ok_or_else(|| {
                CompileError::Validation(format!(
                    "grounding requires a spec-named WittLevel (W8/W16/W24/W32); \
                     unit declared {} bits",
                    unit.witt_level.witt_length()
                ))
            })?;
            let bw = level.byte_width() as usize;
            let test = vec![0u8; bw];
            if hologram_core::ring::grounding::ground_at_level(level, &test).is_none() {
                return Err(CompileError::Validation(
                    "grounding not available at declared quantum level".into(),
                ));
            }
            break;
        }
    }

    // 6. Build the execution tape.
    let serialized_graph = hologram_archive::format::graph::SerializedGraph::from_graph(&graph);
    let _tape = hologram_fused_component::tape_builder::build_tape(&serialized_graph, &schedule)
        .map_err(|e| CompileError::Validation(format!("tape build failed: {}", e)))?;

    // 7. Build LayerHeader + CompileUnitMeta + ConformanceShapeSection
    //    and emit the archive.
    //
    // Per the v0.2.0 conformance-first contract, every archive carries a
    // declaration of which `Shape` its compiled tape conforms to. The
    // loading `PrismModule` validates this declaration before any
    // execution proceeds.
    let layer_header = build_layer_header(&graph, &schedule);
    let unit_meta = hologram_archive::section::compile_unit_meta::CompileUnitMeta {
        unit_address,
        // Store the bit width directly. Phase 10 dropped the v0.1.4
        // quantum-index packing in favour of carrying the actual Witt
        // length on the wire.
        witt_length: unit.witt_level.witt_length(),
        budget: unit.thermodynamic_budget,
        domain_count: unit.target_domain_count,
        term_count: unit.arena.len(),
        binding_count: unit.binding_count,
        assertion_count: unit.assertion_count,
    };
    let shape_section = build_conformance_shape_section(&unit);
    let archive = hologram_archive::HoloWriter::new()
        .set_graph(&graph)
        .add_section(&layer_header)
        .add_section(&unit_meta)
        .add_section(&shape_section)
        .build()
        .map_err(|e| CompileError::Validation(format!("archive emission failed: {}", e)))?;

    // Assemble statistics from finder outputs.
    let total_nodes = graph.node_count();
    let peak_live_buffers = schedule
        .levels
        .iter()
        .map(|l| l.node_ids.len())
        .max()
        .unwrap_or(0);
    let stats = CompilationStats {
        workspace_slots: layout.total_slots,
        peak_live_buffers,
        total_nodes,
        schedule_levels: schedule.num_levels(),
        findings,
        ring_prims_promoted,
    };

    Ok(CompilationOutput {
        archive,
        stats,
        schedule,
        qedl_boundaries,
        unit_address,
    })
}

/// Compile-time assertion verification. Bails out as soon as an
/// evaluable assertion fails.
fn verify_assertions(unit: &HoloCompileUnit) -> CompileResult<()> {
    use hologram_core::term::TermKind;

    fn eval(
        arena: &TermArena,
        bindings: &[hologram_core::term::Binding],
        binding_count: u8,
        id: hologram_core::term::TermId,
        depth: u8,
    ) -> Option<i64> {
        if depth > 64 {
            return None;
        }
        let node = arena.get(id);
        match &node.kind {
            TermKind::IntLit(v) => Some(*v),
            TermKind::QuantumLit { value, .. } => Some(*value as i64),
            TermKind::UnaryApp { op, arg } => {
                let val = eval(arena, bindings, binding_count, *arg, depth + 1)?;
                Some(op.apply_unary(val as u8) as i64)
            }
            TermKind::BinaryApp { op, lhs, rhs } => {
                let l = eval(arena, bindings, binding_count, *lhs, depth + 1)?;
                let r = eval(arena, bindings, binding_count, *rhs, depth + 1)?;
                Some(op.apply_binary(l as u8, r as u8) as i64)
            }
            TermKind::Var(var_id) => {
                for i in 0..binding_count as usize {
                    if bindings[i].var == *var_id {
                        return eval(arena, bindings, binding_count, bindings[i].rhs, depth + 1);
                    }
                }
                None
            }
            TermKind::Passthrough(inner) => eval(arena, bindings, binding_count, *inner, depth + 1),
            _ => None,
        }
    }

    for i in 0..unit.assertion_count as usize {
        let a = &unit.assertions[i];
        if let (Some(lhs), Some(rhs)) = (
            eval(&unit.arena, &*unit.bindings, unit.binding_count, a.lhs, 0),
            eval(&unit.arena, &*unit.bindings, unit.binding_count, a.rhs, 0),
        ) {
            if lhs != rhs {
                return Err(CompileError::Validation(format!(
                    "assertion {} failed: lhs={} rhs={}",
                    i, lhs, rhs
                )));
            }
        }
    }
    Ok(())
}

/// Build a `LayerHeader` describing the graph as a single layer.
fn build_layer_header(
    graph: &hologram_ir::Graph,
    schedule: &ExecutionSchedule,
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

    let sched_levels = vec![vec![LayerId(0); schedule.num_levels()]];

    hologram_archive::LayerHeader {
        layers: vec![descriptor],
        schedule: sched_levels,
    }
}

/// Build the conformance shape section that the archive will declare.
///
/// Per the v0.2.0 conformance-first contract, every archive emitted by
/// `hologram-compiler` carries a declaration of which `Shape` the
/// compiled tape conforms to. The loading `PrismModule` validates this
/// declaration before any execution proceeds.
///
/// **Currently always declares `F_prism_fused_component`.** This is the
/// only shape hologram has implemented a Prism module for. When
/// additional Prism modules are added (e.g., `prism-composition-strict`
/// for `F_prism_strict`), the routing algorithm will pick a shape per
/// source artifact and this helper will become a per-routing-decision
/// lookup. For now, the choice is static.
///
/// **Witt-length range:** the compiler reports the unit's quantum level
/// as both the min and max because the current pipeline targets a single
/// Witt level per archive. Mixed-level archives (W8 nodes alongside W32
/// nodes via `Lift`/`Project` boundaries) would report a wider range.
///
/// **Perf: COMPILE-TIME** — runs once per compilation, never at runtime.
fn build_conformance_shape_section(
    unit: &HoloCompileUnit,
) -> hologram_archive::section::conformance_shape::ConformanceShapeSection {
    use hologram_shapes::shape::F_PRISM_FUSED_COMPONENT;

    let shape = F_PRISM_FUSED_COMPONENT;
    let bits = unit.witt_level.witt_length();
    hologram_archive::section::conformance_shape::ConformanceShapeSection::new(
        *shape.id.as_bytes(),
        shape.target_class,
        shape.name,
        shape.primitives.len() as u32,
        bits,
        bits,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_archive::loader::bytes::load_from_bytes;
    use hologram_archive::section::SECTION_LAYER_HEADER;
    use hologram_core::op::{LutOp, PrimOp};
    use hologram_ir::builder::GraphBuilder;
    use hologram_ir::constant::ConstantData;
    use hologram_ir::graph::GraphOp;

    fn linear_chain() -> Graph {
        GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .build()
    }

    fn diamond_graph() -> Graph {
        GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
            .node_with_inputs(GraphOp::Output, &[3])
            .build()
    }

    #[test]
    fn compile_empty_graph() {
        let g = Graph::new();
        let out = compile(g).unwrap();
        assert_eq!(out.stats.total_nodes, 0);
        assert_eq!(out.stats.schedule_levels, 0);
        assert!(!out.archive.is_empty());
    }

    #[test]
    fn compile_linear_chain() {
        let out = compile(linear_chain()).unwrap();
        assert_eq!(out.stats.total_nodes, 3);
        assert_eq!(out.stats.schedule_levels, 3);
        assert!(out.stats.workspace_slots <= 3);
    }

    #[test]
    fn compile_diamond() {
        let out = compile(diamond_graph()).unwrap();
        assert_eq!(out.stats.total_nodes, 5);
        assert!(out.stats.schedule_levels >= 3);
    }

    #[test]
    fn compile_finds_view_fusion_opportunities() {
        // Sigmoid → Relu → Output: the chain folds into a single FusedView.
        // No more `fuse(true|false)` knob — fusion always runs because it
        // is a structural finding, not a user-controlled optimisation.
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build();
        let out = Compiler::default().compile(SourceInput::Graph(g)).unwrap();
        assert!(out.stats.findings.views_fused >= 1);
    }

    #[test]
    fn compile_with_constants() {
        let g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![42]))
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .build();
        let out = compile(g).unwrap();
        assert!(!out.archive.is_empty());
    }

    #[test]
    fn archive_is_loadable() {
        let out = compile(linear_chain()).unwrap();
        let plan = load_from_bytes(&out.archive).unwrap();
        assert!(plan.header().is_valid_magic());
    }

    #[test]
    fn archive_has_layer_header() {
        let out = compile(linear_chain()).unwrap();
        let plan = load_from_bytes(&out.archive).unwrap();
        assert!(plan.sections().find(SECTION_LAYER_HEADER).is_some());
    }

    #[test]
    fn workspace_stats_correct() {
        let out = compile(linear_chain()).unwrap();
        assert!(out.stats.workspace_slots > 0);
        assert!(out.stats.peak_live_buffers > 0);
    }

    #[test]
    fn schedule_in_output() {
        let out = compile(linear_chain()).unwrap();
        assert_eq!(out.schedule.num_levels(), 3);
    }

    #[test]
    fn compile_wide_parallel() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Exp), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .node_with_inputs(GraphOp::Output, &[3])
            .node_with_inputs(GraphOp::Output, &[4])
            .build();
        let out = compile(g).unwrap();
        assert!(out.stats.peak_live_buffers >= 4);
    }

    #[test]
    fn findings_propagated() {
        let g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![10]))
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .build();
        let out = compile(g).unwrap();
        // Relu(10) = 10, should fold
        assert!(out.stats.findings.constants_folded >= 1);
    }

    #[test]
    fn compile_invalid_graph_fails() {
        let mut g = Graph::new();
        let a = g.add_node(GraphOp::Input);
        let b = g.add_node(GraphOp::Lut(LutOp::Relu));
        g.add_edge(a, b);
        g.add_edge(b, a); // cycle
        let result = compile(g);
        assert!(result.is_err());
    }

    #[test]
    fn compile_graph_with_named_io() {
        let mut g = Graph::new();
        g.add_input("x");
        let inp = g.add_node(GraphOp::Input);
        let relu = g.add_node(GraphOp::Lut(LutOp::Relu));
        let out = g.add_node(GraphOp::Output);
        g.add_edge(inp, relu);
        g.add_edge(relu, out);
        g.add_output("y", out);
        let output = compile(g).unwrap();
        assert!(!output.archive.is_empty());
    }

    #[test]
    fn compile_then_load_node_count() {
        let g = diamond_graph();
        let node_count = g.node_count();
        let out = compile(g).unwrap();
        let plan = load_from_bytes(&out.archive).unwrap();
        assert_eq!(plan.node_count(), node_count);
    }

    #[test]
    fn unit_address_populated() {
        let out = compile(linear_chain()).unwrap();
        assert_ne!(out.unit_address, [0u8; 32]);
    }

    #[test]
    fn from_unit_produces_valid_archive() {
        let graph = linear_chain();
        let unit = unit_from_graph(&graph);
        let out = Compiler::default()
            .compile(SourceInput::Term { unit, graph })
            .unwrap();
        assert!(!out.archive.is_empty());
        assert_eq!(out.stats.total_nodes, 3);
        assert_eq!(out.stats.schedule_levels, 3);
        assert_ne!(out.unit_address, [0u8; 32]);
    }

    #[test]
    fn unit_from_graph_deterministic() {
        let g1 = linear_chain();
        let g2 = linear_chain();
        let u1 = unit_from_graph(&g1);
        let u2 = unit_from_graph(&g2);
        assert_eq!(
            u1.unit_address, u2.unit_address,
            "identical graphs must produce identical unit addresses"
        );
    }

    #[test]
    fn archive_has_compile_unit_meta() {
        use hologram_archive::section::SECTION_COMPILE_UNIT_META;
        let out = compile(linear_chain()).unwrap();
        let plan = load_from_bytes(&out.archive).unwrap();
        assert!(
            plan.sections().find(SECTION_COMPILE_UNIT_META).is_some(),
            "compiled archive must contain CompileUnitMeta section"
        );
    }
}
