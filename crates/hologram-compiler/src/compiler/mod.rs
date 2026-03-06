//! Compiler pipeline: Graph → optimized .holo archive.
//!
//! Three stages: parse (validate), fuse (optimize), emit (schedule + archive).

use hologram_archive::entrypoint::schedule::LayerHeader;
use hologram_archive::entrypoint::{LayerDescriptor, LayerEntrypoint, LayerId, TensorPort};
use hologram_archive::weight::WeightDType;
use hologram_archive::writer::holo_writer::HoloWriter;
use hologram_graph::fusion::{self, FusionStats};
use hologram_graph::graph::validate;
use hologram_graph::graph::Graph;
use hologram_graph::schedule::ExecutionSchedule;

use crate::error::{CompileError, CompileResult};
use crate::liveness;
use crate::workspace;

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
}

/// Builder for configuring and running the compilation pipeline.
pub struct CompilerBuilder {
    graph: Graph,
    enable_fusion: bool,
}

impl CompilerBuilder {
    /// Create a new compiler builder with the given graph.
    #[must_use]
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            enable_fusion: true,
        }
    }

    /// Enable or disable the fusion optimization pass.
    #[must_use]
    pub fn fuse(mut self, enable: bool) -> Self {
        self.enable_fusion = enable;
        self
    }

    /// Run the compilation pipeline and produce a `.holo` archive.
    pub fn build(self) -> CompileResult<CompilationOutput> {
        compile_impl(self.graph, self.enable_fusion)
    }
}

/// Compile a graph into a `.holo` archive with default settings.
pub fn compile(graph: Graph) -> CompileResult<CompilationOutput> {
    CompilerBuilder::new(graph).build()
}

/// Internal compilation implementation.
fn compile_impl(mut graph: Graph, enable_fusion: bool) -> CompileResult<CompilationOutput> {
    parse_stage(&graph)?;
    let fusion_stats = fuse_stage(&mut graph, enable_fusion)?;
    emit_stage(&graph, fusion_stats)
}

/// Stage 1: validate graph structure.
fn parse_stage(graph: &Graph) -> CompileResult<()> {
    validate::validate(graph).map_err(CompileError::from)
}

/// Stage 2: run fusion optimization pass if enabled.
fn fuse_stage(graph: &mut Graph, enable: bool) -> CompileResult<FusionStats> {
    if !enable {
        return Ok(FusionStats::default());
    }
    fusion::fuse(graph).map_err(CompileError::from)
}

/// Stage 3: schedule, liveness, workspace, emit archive.
fn emit_stage(graph: &Graph, fusion_stats: FusionStats) -> CompileResult<CompilationOutput> {
    let schedule = build_schedule(graph)?;
    let intervals = liveness::compute_liveness(&schedule, graph);
    let layout = workspace::plan_workspace(&intervals);
    let stats = build_stats(graph, &schedule, &layout, fusion_stats);
    let layer_header = build_layer_header(graph, &schedule);
    let archive = write_archive(graph, &layer_header)?;

    Ok(CompilationOutput {
        archive,
        stats,
        schedule,
    })
}

/// Build execution schedule from graph.
fn build_schedule(graph: &Graph) -> CompileResult<ExecutionSchedule> {
    ExecutionSchedule::build(graph).map_err(CompileError::from)
}

/// Compute compilation statistics.
fn build_stats(
    graph: &Graph,
    schedule: &ExecutionSchedule,
    layout: &workspace::WorkspaceLayout,
    fusion: FusionStats,
) -> CompilationStats {
    CompilationStats {
        workspace_slots: layout.total_slots,
        peak_live_buffers: compute_peak_live(schedule),
        total_nodes: graph.node_count(),
        schedule_levels: schedule.num_levels(),
        fusion,
    }
}

/// Compute peak simultaneously live buffers across levels.
fn compute_peak_live(schedule: &ExecutionSchedule) -> usize {
    schedule
        .levels
        .iter()
        .map(|l| l.node_ids.len())
        .max()
        .unwrap_or(0)
}

/// Build a LayerHeader describing the graph as a single layer.
fn build_layer_header(graph: &Graph, schedule: &ExecutionSchedule) -> LayerHeader {
    let descriptor = build_layer_descriptor(graph);
    let sched_levels = build_schedule_levels(schedule);
    LayerHeader {
        layers: vec![descriptor],
        schedule: sched_levels,
    }
}

/// Build the layer descriptor for the main graph.
fn build_layer_descriptor(graph: &Graph) -> LayerDescriptor {
    LayerDescriptor {
        id: LayerId(0),
        name: "main".into(),
        entrypoint: LayerEntrypoint::Graph,
        inputs: build_input_ports(graph),
        outputs: build_output_ports(graph),
        group: 0,
        plan_offset: 0,
        plan_size: 0,
    }
}

/// Build input tensor ports from graph inputs.
fn build_input_ports(graph: &Graph) -> Vec<TensorPort> {
    graph
        .inputs()
        .iter()
        .map(|name| TensorPort {
            name: name.clone(),
            shape: vec![1],
            dtype: WeightDType::U8,
        })
        .collect()
}

/// Build output tensor ports from graph outputs.
fn build_output_ports(graph: &Graph) -> Vec<TensorPort> {
    graph
        .outputs()
        .iter()
        .map(|(name, _)| TensorPort {
            name: name.clone(),
            shape: vec![1],
            dtype: WeightDType::U8,
        })
        .collect()
}

/// Convert schedule levels to LayerId groups.
fn build_schedule_levels(schedule: &ExecutionSchedule) -> Vec<Vec<LayerId>> {
    vec![vec![LayerId(0); schedule.num_levels()]]
}

/// Write the .holo archive.
fn write_archive(graph: &Graph, layer_header: &LayerHeader) -> CompileResult<Vec<u8>> {
    HoloWriter::new()
        .set_graph(graph)
        .add_section(layer_header)
        .build()
        .map_err(CompileError::from)
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
}
