//! Single-pass fusion engine.
//!
//! One topological walk interleaving five optimizations:
//! 1. Constant folding — evaluate ops on constant inputs at compile time
//! 2. View fusion — collapse byte-domain unary chains into a single 256-byte LUT
//! 3. MatMul+activation epilogue fusion — absorb successor activation into matmul
//! 4. Float chain fusion — collapse f32-domain unary chains into FusedFloatChain
//! 5. CSE — deduplicate nodes with identical (op, sorted predecessors)
//!
//! Why single-pass works: topo order ensures predecessors are processed first.
//! Constant folding propagates forward. View fusion looks backward (chain
//! already stable). CSE sees final form after folding/fusion.

pub mod constant;
pub mod cse;
pub mod float_fusion;
pub mod q1_view_fusion;
pub mod view_fusion;

use crate::error::GraphResult;
use crate::graph::Graph;
use crate::schedule::toposort;

/// Statistics from a fusion pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusionStats {
    /// Number of nodes constant-folded.
    pub constants_folded: usize,
    /// Number of unary chains fused into FusedViews.
    pub views_fused: usize,
    /// Number of Q1 unary chains fused into FusedView16.
    pub q1_views_fused: usize,
    /// Number of float element-wise chains fused into FusedFloatChain.
    pub float_chains_fused: usize,
    /// Number of MatMul+activation pairs fused (epilogue fusion).
    pub matmul_activations_fused: usize,
    /// Number of duplicate nodes eliminated by CSE.
    pub cse_eliminated: usize,
}

impl FusionStats {
    /// Total number of nodes removed by all optimizations.
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

/// Run the single-pass fusion engine on the graph.
///
/// Applies constant folding, view fusion, and CSE in a single topological walk.
/// Returns statistics about optimizations applied.
pub fn fuse(graph: &mut Graph) -> GraphResult<FusionStats> {
    let order = toposort::toposort(graph)?;
    let mut stats = FusionStats::default();
    let succ_index = graph.build_successor_index();

    for &id in &order {
        if graph.get(id).is_none() {
            continue;
        }

        // 1. Constant folding
        if constant::try_fold_constant(graph, id) {
            stats.constants_folded += 1;
            continue;
        }

        // 2a. Q0 view fusion (backward chain walk)
        while view_fusion::try_fuse_unary_backward(graph, id, &succ_index) {
            stats.views_fused += 1;
        }

        // 2b. Q1 view fusion (backward chain walk)
        while q1_view_fusion::try_fuse_q1_unary_backward(graph, id, &succ_index) {
            stats.q1_views_fused += 1;
        }

        // 3. MatMul + bias + activation (3-node → 1-node, highest value)
        if float_fusion::try_fuse_matmul_bias_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
            continue; // MatMul consumed — skip 2-node fusion.
        }

        // 4. MatMul + activation epilogue fusion (forward: matmul absorbs successor)
        if float_fusion::try_fuse_matmul_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
        }
        if float_fusion::try_fuse_lut_gemm_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
        }
        if float_fusion::try_fuse_norm_activation(graph, id, &succ_index) {
            stats.matmul_activations_fused += 1;
        }

        // 5. Float chain fusion (f32-domain backward chain walk)
        while float_fusion::try_fuse_float_unary(graph, id, &succ_index) {
            stats.float_chains_fused += 1;
        }
    }

    // 5. CSE — reuses original topo order. Removed nodes are skipped via
    //    graph.get(id).is_none() inside CSE. Topo invariant holds because
    //    fusion only removes nodes, never adds new dependencies.
    stats.cse_eliminated = cse::eliminate_common_subexpressions(graph, &order);

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
    fn fuse_empty() {
        let mut g = Graph::new();
        let stats = fuse(&mut g).unwrap();
        assert_eq!(stats, FusionStats::default());
    }

    #[test]
    fn fuse_constant_propagation() {
        // const(10) → Relu → Output
        // Relu(10) = 10, so Relu should be folded.
        let mut g = GraphBuilder::new()
            .constant(ConstantData::Bytes(vec![10])) // 0: Constant
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1: Relu
            .node_with_inputs(GraphOp::Output, &[1]) // 2: Output
            .build();
        let stats = fuse(&mut g).unwrap();
        assert_eq!(stats.constants_folded, 1);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn fuse_chain_and_cse() {
        // Input → Sigmoid → Relu → Output
        //       ↘ Sigmoid → Relu → Output
        // View fusion: each Sigmoid→Relu chain becomes FusedView.
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
        let stats = fuse(&mut g).unwrap();
        assert!(stats.views_fused >= 2);
        assert!(stats.cse_eliminated >= 1);
    }

    #[test]
    fn fuse_all_three() {
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
        let stats = fuse(&mut g).unwrap();
        // Add(5,3)=8 folds, then Relu(8)=8 folds
        assert!(stats.constants_folded >= 2);
    }

    #[test]
    fn fuse_stats_total() {
        let stats = FusionStats {
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
    fn fuse_no_ops_on_pure_io() {
        // Input → Output (nothing to optimize)
        let mut g = GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Output, &[0])
            .build();
        let stats = fuse(&mut g).unwrap();
        assert_eq!(stats.total_removed(), 0);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn fuse_matmul_activation_via_full_pass() {
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

        let stats = fuse(&mut g).unwrap();
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
    fn fuse_matmul_bias_activation_via_full_pass() {
        use crate::constant::ConstantData;
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

        let stats = fuse(&mut g).unwrap();
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
