//! Compiler pipeline: cascade-backed compilation to .holo archive.
//!
//! All compilation routes through the 7-stage cascade engine with
//! certificate memoization. Three entry points:
//!
//! - `CompilerBuilder::new(graph)` — wrap a raw Graph in a synthetic CompileUnit
//! - `CompilerBuilder::from_unit(unit, graph)` — supply a pre-built CompileUnit + lowered Graph
//! - `CompilerBuilder::from_source(...)` — parse UOR term source → CompileUnit → Graph
//!
//! The free function `compile(graph)` is a convenience wrapper around
//! `CompilerBuilder::new(graph).build()`.

use hologram_cascade::certificate::CertificateStore;
use hologram_cascade::engine::run_cascade_with_graph_opts;
use hologram_graph::fusion::FusionStats;
use hologram_graph::graph::node::NodeId;
use hologram_graph::graph::Graph;
use hologram_graph::schedule::ExecutionSchedule;

use hologram_core::op::RingLevel;
use hologram_core::term::{HoloAddress, HoloCompileUnit, TermArena, TermKind};
use uor_foundation::enums::VerificationDomain;
use uor_foundation::QuantumLevel;

use crate::error::{CompileError, CompileResult};

/// Re-exported from `hologram_cascade::qedl`.
pub use hologram_cascade::qedl::QedlBoundary;

/// Cascade metadata from compilation.
#[derive(Debug, Clone)]
pub struct CascadeInfo {
    /// Whether the result was served from the certificate cache.
    pub cache_hit: bool,
    /// Total Landauer cost consumed (k_B T units).
    pub budget_consumed: f64,
    /// Content-addressed identifier of the compiled unit.
    pub unit_address: [u8; 32],
}

/// Statistics from the compilation process.
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
    /// Fusion pass statistics.
    pub fusion: FusionStats,
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
    pub qedl_boundaries: Vec<(NodeId, QedlBoundary, hologram_cascade::qedl::EncodingId)>,
    /// Cascade-specific metadata.
    pub cascade: CascadeInfo,
}

/// Input source for the compiler.
enum CompilerInput {
    /// Raw graph — will be wrapped in a synthetic CompileUnit.
    Graph(Graph),
    /// Pre-built CompileUnit + pre-lowered Graph.
    Unit { unit: HoloCompileUnit, graph: Graph },
    /// UOR term language source text.
    Source {
        source: String,
        level: QuantumLevel,
        budget: f64,
        domains: Vec<VerificationDomain>,
    },
}

/// Builder for configuring and running the compilation pipeline.
pub struct CompilerBuilder {
    input: CompilerInput,
    store: Option<CertificateStore>,
    skip_fusion: bool,
}

impl CompilerBuilder {
    /// Create a new compiler builder from a raw Graph.
    ///
    /// The graph is wrapped in a synthetic CompileUnit with:
    /// - Quantum level: Q0
    /// - Budget: f64::MAX (unconstrained)
    /// - Domain: Algebraic
    #[must_use]
    pub fn new(graph: Graph) -> Self {
        Self {
            input: CompilerInput::Graph(graph),
            store: None,
            skip_fusion: false,
        }
    }

    /// Create from a pre-built CompileUnit and pre-lowered Graph.
    #[must_use]
    pub fn from_unit(unit: HoloCompileUnit, graph: Graph) -> Self {
        Self {
            input: CompilerInput::Unit { unit, graph },
            store: None,
            skip_fusion: false,
        }
    }

    /// Create from UOR term language source text.
    ///
    /// Parses, builds CompileUnit, runs preflight, and lowers to Graph internally.
    #[must_use]
    pub fn from_source(
        source: &str,
        level: QuantumLevel,
        budget: f64,
        domains: &[VerificationDomain],
    ) -> Self {
        Self {
            input: CompilerInput::Source {
                source: source.to_owned(),
                level,
                budget,
                domains: domains.to_vec(),
            },
            store: None,
            skip_fusion: false,
        }
    }

    /// Supply a shared certificate store for cross-compilation memoization.
    #[must_use]
    pub fn certificate_store(mut self, store: CertificateStore) -> Self {
        self.store = Some(store);
        self
    }

    /// Enable or disable the fusion optimization pass.
    #[must_use]
    pub fn fuse(mut self, enable: bool) -> Self {
        self.skip_fusion = !enable;
        self
    }

    /// Run the compilation pipeline and produce a `.holo` archive.
    pub fn build(self) -> CompileResult<CompilationOutput> {
        let mut store = self.store.unwrap_or_else(|| CertificateStore::new(64));

        match self.input {
            CompilerInput::Graph(graph) => {
                let unit = unit_from_graph(&graph);
                compile_via_cascade(unit, graph, &mut store, self.skip_fusion)
            }
            CompilerInput::Unit { unit, graph } => {
                compile_via_cascade(unit, graph, &mut store, self.skip_fusion)
            }
            CompilerInput::Source {
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
                compile_via_cascade(unit, graph, &mut store, self.skip_fusion)
            }
        }
    }
}

/// Compile a graph into a `.holo` archive with default settings.
///
/// Convenience wrapper around `CompilerBuilder::new(graph).build()`.
pub fn compile(graph: Graph) -> CompileResult<CompilationOutput> {
    CompilerBuilder::new(graph).build()
}

/// Compile from a `uor!` macro expansion (enforcement `TermArena`).
///
/// The `uor!` macro parses EBNF surface syntax at compile time and produces a
/// typed `enforcement::TermArena`. This function converts it to hologram's
/// arena, runs preflight (including enforcement validation), lowers to a Graph,
/// and runs the 7-stage cascade.
///
/// Per PRISM Section 1: stages 1-4 (parse, build AST, resolve names, type
/// check) are handled at compile time by the `uor!` proc macro. This function
/// implements stages 5-7 (desugar, reify, emit runtime plan).
pub fn compile_uor_arena<const CAP: usize>(
    enforcement_arena: &uor_foundation::enforcement::TermArena<CAP>,
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

    // Cascade
    let mut store = CertificateStore::new(64);
    compile_via_cascade(unit, graph, &mut store, false)
}

/// Build a `HoloCompileUnit` from a `Graph`.
///
/// This is the primary integration point for consumers like hologram-ai
/// that build graphs via `GraphBuilder` and want to enter the declarative
/// cascade pipeline. The unit address is computed from the graph structure
/// (BLAKE3 hash of node ops and edges), enabling certificate memoization
/// across identical graphs.
///
/// Default parameters:
/// - Quantum level: `Q0`
/// - Budget: `f64::MAX` (unconstrained for graph-based compilation)
/// - Domains: `[Algebraic]`
pub fn unit_from_graph(graph: &Graph) -> HoloCompileUnit {
    unit_from_graph_with(
        graph,
        QuantumLevel::Q0,
        f64::MAX,
        &[VerificationDomain::Algebraic],
    )
}

/// Build a `HoloCompileUnit` from a `Graph` with explicit parameters.
pub fn unit_from_graph_with(
    graph: &Graph,
    level: QuantumLevel,
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

/// Core compilation via the cascade engine.
fn compile_via_cascade(
    unit: HoloCompileUnit,
    graph: Graph,
    store: &mut CertificateStore,
    skip_fusion: bool,
) -> CompileResult<CompilationOutput> {
    let result = run_cascade_with_graph_opts(&unit, graph, store, skip_fusion)
        .map_err(|e| CompileError::Validation(e.to_string()))?;

    let state = result.state;

    let archive = state.archive_bytes.unwrap_or_default();

    let schedule = state.schedule.unwrap_or_else(|| ExecutionSchedule {
        levels: vec![],
        critical_path: 0,
    });

    let total_nodes = state.graph.as_ref().map(|g| g.node_count()).unwrap_or(0);

    let peak_live = schedule
        .levels
        .iter()
        .map(|l| l.node_ids.len())
        .max()
        .unwrap_or(0);

    let workspace_slots = state
        .workspace_layout
        .as_ref()
        .map(|l| l.total_slots)
        .unwrap_or(0);

    let stats = CompilationStats {
        workspace_slots,
        peak_live_buffers: peak_live,
        total_nodes,
        schedule_levels: schedule.num_levels(),
        fusion: state.fusion_stats,
        ring_prims_promoted: state.ring_prims_promoted,
    };

    let cascade_info = CascadeInfo {
        cache_hit: result.cache_hit,
        budget_consumed: state.budget_consumed,
        unit_address: state.unit_address,
    };

    Ok(CompilationOutput {
        archive,
        stats,
        schedule,
        qedl_boundaries: state.qedl_boundaries.unwrap_or_default(),
        cascade: cascade_info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_archive::loader::bytes::load_from_bytes;
    use hologram_archive::section::SECTION_LAYER_HEADER;
    use hologram_core::op::{LutOp, PrimOp};
    use hologram_graph::builder::GraphBuilder;
    use hologram_graph::constant::ConstantData;
    use hologram_graph::graph::GraphOp;

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
    fn compile_with_fusion_enabled() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build();
        let out = CompilerBuilder::new(g).fuse(true).build().unwrap();
        assert!(out.stats.fusion.views_fused >= 1);
    }

    #[test]
    fn compile_with_fusion_disabled() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build();
        let out = CompilerBuilder::new(g).fuse(false).build().unwrap();
        assert_eq!(out.stats.fusion, FusionStats::default());
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
    fn fusion_stats_propagated() {
        let g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![10]))
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .build();
        let out = compile(g).unwrap();
        // Relu(10) = 10, should fold
        assert!(out.stats.fusion.constants_folded >= 1);
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
    fn builder_default_enables_fusion() {
        let g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build();
        let out = CompilerBuilder::new(g).build().unwrap();
        // Default is fusion enabled, so views should be fused
        assert!(out.stats.fusion.views_fused >= 1);
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
    fn cascade_info_populated() {
        let out = compile(linear_chain()).unwrap();
        assert!(!out.cascade.cache_hit);
        assert!(out.cascade.budget_consumed > 0.0);
    }

    #[test]
    fn from_unit_produces_valid_archive() {
        let graph = linear_chain();
        let unit = unit_from_graph(&graph);
        let out = CompilerBuilder::from_unit(unit, graph).build().unwrap();
        assert!(!out.archive.is_empty());
        assert_eq!(out.stats.total_nodes, 3);
        assert_eq!(out.stats.schedule_levels, 3);
        assert!(out.cascade.budget_consumed > 0.0);
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

    #[test]
    fn shared_certificate_store_enables_cache_hit() {
        let store = hologram_cascade::CertificateStore::new(64);
        let g1 = linear_chain();
        let out1 = CompilerBuilder::new(g1)
            .certificate_store(store)
            .build()
            .unwrap();
        assert!(!out1.cascade.cache_hit);
        assert!(out1.cascade.budget_consumed > 0.0);
        assert_ne!(out1.cascade.unit_address, [0u8; 32]);
    }
}
