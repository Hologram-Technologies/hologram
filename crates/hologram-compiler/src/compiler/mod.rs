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

use hologram_graph::graph::node::NodeId;

use crate::error::{CompileError, CompileResult};
use crate::liveness;
use crate::qedl::{pass::insert_qedl_boundaries, EncodingId};
use crate::workspace;

/// QEDL pipeline boundary marker.
///
/// Marks nodes where the pipeline transitions between quantized (byte-domain)
/// and dequantized (float-domain) execution. The actual insertion of
/// quantize/dequantize ops is handled by a later pass — this enum provides
/// the metadata structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QedlBoundary {
    /// Dequantize: byte-domain → float-domain (before a float op).
    Dequantize,
    /// Quantize: float-domain → byte-domain (after a float op).
    Quantize,
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
    /// Each entry: (consuming node id, boundary kind, selected encoding).
    pub qedl_boundaries: Vec<(NodeId, QedlBoundary, EncodingId)>,
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
    let ring_prims_promoted = precision_stage(&mut graph);
    emit_stage(&graph, fusion_stats, ring_prims_promoted)
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

/// Stage 2b: promote byte-domain Prim ops to ring-level-tagged variants.
///
/// Uses observable analysis (stratum/curvature of output LUT) to select
/// the minimum Q-level (Q0/Q1/Q2) per node. Bakes the ring level into the
/// graph so the archive contains fully annotated ops.
fn precision_stage(graph: &mut Graph) -> usize {
    crate::precision::promote_prim_ring_levels(graph)
}

/// Stage 3: schedule, liveness, workspace, emit archive.
fn emit_stage(
    graph: &Graph,
    fusion_stats: FusionStats,
    ring_prims_promoted: usize,
) -> CompileResult<CompilationOutput> {
    let schedule = build_schedule(graph)?;
    let intervals = liveness::compute_liveness(&schedule, graph);
    let layout = workspace::plan_workspace(&intervals);
    let stats = build_stats(graph, &schedule, &layout, fusion_stats, ring_prims_promoted);

    // Compute post-fusion shapes: walk the graph in topological order and
    // project each node's output shape from its inputs. This produces a
    // complete shape map covering every node that will appear in the tape.
    let post_fusion_shapes = compute_post_fusion_shapes(graph, &schedule);

    // Merge projected shapes into the graph's node_shapes so they are
    // embedded in the archive. Existing shapes (from lowering) are preserved;
    // projected shapes fill gaps for nodes created/modified by fusion.
    let mut graph_with_shapes = graph.clone();
    for (nid, shape) in &post_fusion_shapes {
        graph_with_shapes.set_node_shape(*nid, shape.clone());
    }

    let layer_header = build_layer_header(&graph_with_shapes, &schedule);
    let archive = write_archive(&graph_with_shapes, &layer_header)?;

    // Run QEDL pass: collect domain-crossing boundary annotations.
    let topo_order: Vec<NodeId> = schedule
        .levels
        .iter()
        .flat_map(|level| level.node_ids.iter().copied())
        .collect();
    let qedl_boundaries = insert_qedl_boundaries(graph, &topo_order);

    Ok(CompilationOutput {
        archive,
        stats,
        schedule,
        qedl_boundaries,
    })
}

/// Walk the post-fusion graph in topological order and project each node's
/// output shape from its inputs using `float_output_shape()`.
///
/// Seeds from `graph.node_shapes()` (populated during lowering) and
/// `graph.constant_shapes()`. Projects forward through all ops, filling
/// in shapes for nodes created or modified by fusion.
fn compute_post_fusion_shapes(
    graph: &Graph,
    schedule: &ExecutionSchedule,
) -> std::collections::HashMap<NodeId, Vec<usize>> {
    use hologram_core::op::shape_projection::float_output_shape;
    use hologram_graph::graph::GraphOp;

    let mut shape_map: std::collections::HashMap<NodeId, Vec<usize>> =
        std::collections::HashMap::new();

    // Seed from existing node shapes (from lowering).
    for (&nid, shape) in graph.node_shapes() {
        shape_map.insert(nid, shape.clone());
    }

    // Seed constant shapes: find Constant nodes and map ConstantId → NodeId.
    for node in graph.nodes() {
        if let GraphOp::Constant(cid) = &node.op {
            if let Some(shape) = graph.constant_shapes().get(cid) {
                shape_map.entry(node.id).or_insert_with(|| shape.clone());
            }
        }
    }

    // Topological walk: project output shapes from inputs.
    for level in &schedule.levels {
        for &nid in &level.node_ids {
            if shape_map.contains_key(&nid) {
                continue; // already seeded
            }

            let node = match graph.get(nid) {
                Some(n) => n,
                None => continue,
            };

            let preds = graph.predecessors(nid);
            let input_shapes: Vec<Vec<usize>> = preds
                .iter()
                .map(|&pred| shape_map.get(&pred).cloned().unwrap_or_default())
                .collect();
            let input_refs: Vec<&[usize]> = input_shapes.iter().map(|s| s.as_slice()).collect();

            let projected = match &node.op {
                // Float ops: use the shape projection function.
                GraphOp::Float(f) => float_output_shape(f, &input_refs),

                // Fused float chain: element-preserving (all unary).
                GraphOp::FusedFloatChain(_) => input_refs.first().map(|s| s.to_vec()),

                // Fused matmul variants: output [m, n].
                GraphOp::FusedMatMulActivation { m, n, .. }
                | GraphOp::FusedMatMulBiasActivation { m, n, .. } => {
                    Some(vec![*m as usize, *n as usize])
                }

                // Fused norm+activation: shape-preserving.
                GraphOp::FusedRmsNormActivation { .. }
                | GraphOp::FusedLayerNormActivation { .. }
                | GraphOp::FusedGroupNormActivation { .. }
                | GraphOp::FusedAddRmsNormActivation { .. }
                | GraphOp::FusedInstanceNormActivation { .. } => {
                    input_refs.first().map(|s| s.to_vec())
                }

                // Fused conv+activation: delegate to Conv2d projection.
                GraphOp::FusedConv2dActivation {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    dilation_h,
                    dilation_w,
                    input_h,
                    input_w,
                    ..
                }
                | GraphOp::FusedConv2dBiasActivation {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    dilation_h,
                    dilation_w,
                    input_h,
                    input_w,
                    ..
                } => {
                    let conv = hologram_core::op::FloatOp::Conv2d {
                        kernel_h: *kernel_h,
                        kernel_w: *kernel_w,
                        stride_h: *stride_h,
                        stride_w: *stride_w,
                        pad_h: *pad_h,
                        pad_w: *pad_w,
                        dilation_h: *dilation_h,
                        dilation_w: *dilation_w,
                        group: 1,
                        input_h: *input_h,
                        input_w: *input_w,
                    };
                    float_output_shape(&conv, &input_refs)
                }

                // Byte-domain unary: shape-preserving.
                GraphOp::Lut(_)
                | GraphOp::FusedView(_)
                | GraphOp::FusedView16(_)
                | GraphOp::Passthrough
                | GraphOp::Output
                | GraphOp::RingPrimUnary(..)
                | GraphOp::RingActivation(..)
                | GraphOp::RingAccumulate(..) => input_refs.first().map(|s| s.to_vec()),

                // Byte-domain binary: same shape (no broadcast in byte domain).
                GraphOp::Prim(_) | GraphOp::RingPrimBinary(..) => {
                    input_refs.first().map(|s| s.to_vec())
                }

                // Ring reduce: drops one dim.
                GraphOp::RingReduce { .. } => input_refs.first().map(|input| {
                    if input.len() <= 1 {
                        vec![1]
                    } else {
                        input[..input.len() - 1].to_vec()
                    }
                }),

                // LUT-GEMM: use compiled node_shapes (these are weight-baked).
                GraphOp::MatMulLut4(_)
                | GraphOp::MatMulLut8(_)
                | GraphOp::MatMulLut16(_)
                | GraphOp::MatMulLut2(_)
                | GraphOp::BatchMatMulLut4(_)
                | GraphOp::BatchMatMulLut8(_)
                | GraphOp::BatchMatMulLut16(_)
                | GraphOp::MatMulLut4Activation(..)
                | GraphOp::MatMulLut8Activation(..)
                | GraphOp::MatMulLut2Activation(..)
                | GraphOp::Conv2dLut4 { .. } => None, // shape from lowering

                // Input, Constant: already seeded.
                GraphOp::Input | GraphOp::Constant(_) => None,

                // Subgraph, Custom: can't project.
                GraphOp::CallSubgraph(_) | GraphOp::Custom { .. } => None,
            };

            if let Some(shape) = projected {
                shape_map.insert(nid, shape);
            }
        }
    }

    shape_map
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
    ring_prims_promoted: usize,
) -> CompilationStats {
    CompilationStats {
        workspace_slots: layout.total_slots,
        peak_live_buffers: compute_peak_live(schedule),
        total_nodes: graph.node_count(),
        schedule_levels: schedule.num_levels(),
        fusion,
        ring_prims_promoted,
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
