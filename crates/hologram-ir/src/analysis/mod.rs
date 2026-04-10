//! Structural analysis passes — the cross-compiler's **finder** layer.
//!
//! Per Prism section 4 of the SCS framework, the compiler is a
//! *structure-finder*, not a constructor. The modules in this directory all
//! follow that pattern: they read structural properties that the source
//! graph already determines, and they produce findings (intervals, levels,
//! boundaries, layouts, fusion opportunities) that the cross-compiler emits
//! into the archive's characterization report.
//!
//! Two groups of analyses live here:
//!
//! ## Schedule-shape analyses
//!
//! - [`liveness`] — buffer lifetime intervals from schedule level structure
//! - [`precision`] — observable-guided ring-level inference (W8/W16/W24)
//! - [`qedl`] — domain-crossing detection between byte and float subgraphs
//! - [`workspace`] — buffer slot reuse via first-fit-decreasing bin-packing
//!
//! ## Pattern-detection analyses
//!
//! - [`constant_folding`] — finds subgraphs whose value is determined by
//!   constant inputs
//! - [`view_detection`] — finds elementwise-op chains realisable as a
//!   single 256-byte LUT
//! - [`q1_view_detection`] — finds W16-domain chains realisable as a
//!   single 65,536-entry LUT
//! - [`float_fusion`] — finds float-op chains and matmul/conv epilogue
//!   patterns realisable as fused IR ops
//! - [`cse`] — finds redundant subexpressions
//!
//! Each pass detects a structural opportunity that the source graph
//! already exhibits; the IR's `Fused*` variants are the *encoding* of
//! those findings. The dispatcher [`analyze`] runs all five
//! pattern-detection passes (and CSE) in a single topological walk.
//!
//! # Performance principle
//!
//! Every pass in this module runs at **compile time**, not at inference
//! time. They are all marked **Perf: COMPILE-TIME** in the conformance-first
//! refactor plan: their work moves out of runtime entirely.
//!
//! The output (fused TapeKernel variants, ring-level annotations, workspace
//! layouts) is what makes runtime fast. Deleting any of these passes would
//! force the runtime to do their work per query, which is the constructor
//! pattern the v0.2.0 refactor exists to eliminate.

pub mod constant_folding;
pub mod cse;
pub mod float_fusion;
pub mod liveness;
pub mod precision;
pub mod q1_view_detection;
pub mod qedl;
pub mod view_detection;
pub mod workspace;

use crate::error::GraphResult;
use crate::graph::Graph;
use crate::schedule::toposort;

/// Statistics from one full analysis pass over a graph.
///
/// Each counter is a *finding count* — how many of each structural
/// pattern the analysis detected.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StructuralFindings {
    /// Number of subgraphs found to be constant-determinable.
    pub constants_folded: usize,
    /// Number of byte-domain unary chains realised as Q0 view-fused LUTs.
    pub views_fused: usize,
    /// Number of Q1-domain unary chains realised as 65,536-entry LUTs.
    pub q1_views_fused: usize,
    /// Number of float-domain unary chains realised as `FusedFloatChain`.
    pub float_chains_fused: usize,
    /// Number of matmul/conv + activation patterns found as fused IR ops.
    pub matmul_activations_fused: usize,
    /// Number of duplicate computations found by CSE.
    pub cse_eliminated: usize,
}

impl StructuralFindings {
    /// Total number of nodes the analyses replaced or removed.
    #[must_use]
    pub fn total_removed(&self) -> usize {
        self.constants_folded
            + self.views_fused
            + self.q1_views_fused
            + self.float_chains_fused
            + self.matmul_activations_fused
            + self.cse_eliminated
    }
}

/// Run the full structural-finder pass over a graph in a single topological
/// walk.
///
/// This function does not optimise: it detects structural patterns the
/// source graph already exhibits and replaces matching subgraphs with the
/// IR's encoding of those findings.
///
/// **Perf: COMPILE-TIME** — runs once per compilation, never at runtime.
/// The output is what makes runtime fast (fused IR ops dispatch directly
/// to fused TapeKernel variants).
pub fn analyze(graph: &mut Graph) -> GraphResult<StructuralFindings> {
    let succ_index = graph.build_successor_index();
    let order = toposort::toposort_with_index(graph, &succ_index)?;
    let mut stats = StructuralFindings::default();

    for &id in &order {
        if graph.get(id).is_none() {
            continue;
        }

        // 1. Constant folding — finds subgraphs determined by constants.
        if constant_folding::try_fold_constant(graph, id) {
            stats.constants_folded += 1;
            continue;
        }

        // 2a. Q0 view detection (backward chain walk).
        while view_detection::try_fuse_unary_backward(graph, id, &succ_index) {
            stats.views_fused += 1;
        }

        // 2b. Q1 view detection (backward chain walk).
        while q1_view_detection::try_fuse_q1_unary_backward(graph, id, &succ_index) {
            stats.q1_views_fused += 1;
        }

        // 3. MatMul + bias + activation (3-node → 1-node, highest value).
        if float_fusion::try_fuse_matmul_bias_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
            continue; // MatMul consumed — skip 2-node fusion.
        }

        // 3b. Transpose elimination (inverse transpose pairs → Passthrough).
        if float_fusion::try_eliminate_inverse_transpose(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1; // reuse counter for structural opts
            continue;
        }

        // 3c. Conv2d + bias + activation (3-node, same priority as matmul bias fusion).
        if float_fusion::try_fuse_conv2d_bias_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
            continue;
        }

        // 4. MatMul + activation epilogue (forward: matmul absorbs successor).
        if float_fusion::try_fuse_matmul_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
        }
        // 4b. Conv2d + activation epilogue.
        if float_fusion::try_fuse_conv2d_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
        }
        // 4c. Silu + Mul → FusedSwiGLU.
        if float_fusion::try_fuse_swiglu(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
        }
        if float_fusion::try_fuse_lut_gemm_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
        }
        if float_fusion::try_fuse_norm_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
        }

        // 5. Float chain detection (f32-domain backward chain walk).
        while float_fusion::try_fuse_float_unary(graph, id, &succ_index) {
            stats.float_chains_fused += 1;
        }
    }

    // 6. CSE — reuses original topo order. Removed nodes are skipped via
    //    graph.get(id).is_none() inside CSE. Topo invariant holds because
    //    these passes only remove nodes, never add new dependencies.
    stats.cse_eliminated = cse::eliminate_common_subexpressions(graph, &order, &succ_index);

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::constant::ConstantData;
    use crate::graph::GraphOp;
    use hologram_core::op::{LutOp, PrimOp};

    #[test]
    fn analyze_empty() {
        let mut g = Graph::new();
        let stats = analyze(&mut g).unwrap();
        assert_eq!(stats, StructuralFindings::default());
    }

    #[test]
    fn analyze_constant_propagation() {
        // const(10) → Relu → Output
        // Relu(10) = 10, so Relu should be folded.
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![10])) // 0: Constant
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1: Relu
            .node_with_inputs(GraphOp::Output, &[1]) // 2: Output
            .build();
        let stats = analyze(&mut g).unwrap();
        assert_eq!(stats.constants_folded, 1);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn analyze_chain_and_cse() {
        // Input → Sigmoid → Relu → Output
        //       ↘ Sigmoid → Relu → Output
        // View detection: each Sigmoid→Relu chain becomes FusedView.
        // CSE: the two identical FusedViews share a result.
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 1
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1]) // 2
            .node_with_inputs(GraphOp::Output, &[2]) // 3
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 4
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[4]) // 5
            .node_with_inputs(GraphOp::Output, &[5]) // 6
            .build();
        let stats = analyze(&mut g).unwrap();
        assert!(stats.views_fused >= 2);
        assert!(stats.cse_eliminated >= 1);
    }

    #[test]
    fn analyze_all_three() {
        // const(5) + const(3) → Add → Relu → Output
        // Step 1: Add(5,3) folds to const(8)
        // Step 2: Relu(const(8)) folds to const(8) (relu(8)=8)
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![5])) // 0
            .constant(ConstantData::Bytes(vec![3])) // 1
            .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[0, 1]) // 2
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[2]) // 3
            .node_with_inputs(GraphOp::Output, &[3]) // 4
            .build();
        let stats = analyze(&mut g).unwrap();
        // Add(5,3)=8 folds, then Relu(8)=8 folds
        assert!(stats.constants_folded >= 2);
    }

    #[test]
    fn findings_total() {
        let stats = StructuralFindings {
            constants_folded: 3,
            views_fused: 2,
            q1_views_fused: 1,
            float_chains_fused: 1,
            matmul_activations_fused: 1,
            cse_eliminated: 1,
        };
        assert_eq!(stats.total_removed(), 9);
    }

    #[test]
    fn analyze_no_ops_on_pure_io() {
        // Input → Output (nothing to optimize)
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Output, &[0])
            .build();
        let stats = analyze(&mut g).unwrap();
        assert_eq!(stats.total_removed(), 0);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn analyze_matmul_activation_via_full_pass() {
        use hologram_core::op::FloatOp;
        // Input0, Input1 → MatMul → Silu → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0
            .node(GraphOp::Input) // 1
            .node_with_inputs(
                GraphOp::Float(FloatOp::MatMul { m: 4, k: 8, n: 16 }),
                &[0, 1],
            ) // 2
            .node_with_inputs(GraphOp::Float(FloatOp::Silu), &[2]) // 3
            .node_with_inputs(GraphOp::Output, &[3]) // 4
            .build();

        let stats = analyze(&mut g).unwrap();
        assert_eq!(stats.matmul_activations_fused, 1);
        assert_eq!(g.node_count(), 4); // Input0, Input1, FusedMatMulActivation, Output

        // Verify the fused node exists with correct parameters.
        let has_fused = g.node_ids().into_iter().any(|id| {
            matches!(
                g.get(id).unwrap().op,
                GraphOp::FusedMatMulActivation {
                    m: 4,
                    k: 8,
                    n: 16,
                    activation: FloatOp::Silu,
                }
            )
        });
        assert!(has_fused, "should contain FusedMatMulActivation with Silu");
    }

    #[test]
    fn analyze_matmul_bias_activation_via_full_pass() {
        use hologram_core::op::FloatOp;
        // Input0, Input1(weight) → MatMul → Add(bias_constant) → Relu → Output
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input) // 0: activation input
            .node(GraphOp::Input) // 1: weight
            .node_with_inputs(
                GraphOp::Float(FloatOp::MatMul { m: 1, k: 64, n: 32 }),
                &[0, 1],
            ) // 2: MatMul
            .constant(ConstantData::Bytes(vec![0u8; 128])) // 3: bias constant (32 f32s)
            .node_with_inputs(GraphOp::Float(FloatOp::Add), &[2, 3]) // 4: Add(matmul, bias)
            .node_with_inputs(GraphOp::Float(FloatOp::Relu), &[4]) // 5: Relu
            .node_with_inputs(GraphOp::Output, &[5]) // 6: Output
            .build();

        let stats = analyze(&mut g).unwrap();
        assert_eq!(
            stats.matmul_activations_fused, 1,
            "should fuse MatMul+Bias+Activation"
        );
        // Should have: Input0, Input1, bias_constant, FusedMatMulBiasActivation, Output
        // MatMul and Add removed (2 nodes gone).
        let has_fused = g.node_ids().into_iter().any(|id| {
            matches!(
                g.get(id).unwrap().op,
                GraphOp::FusedMatMulBiasActivation { .. }
            )
        });
        assert!(has_fused, "should contain FusedMatMulBiasActivation");
    }
}
