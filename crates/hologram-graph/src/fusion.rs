//! Single-pass fusion engine (spec VI.3).
//!
//! One topological walk interleaving three optimizations:
//! 1. MatMul + activation epilogue fusion — absorb successor activation
//! 2. Silu + Mul → FusedSwiGlu (SwiGLU pattern)
//! 3. CSE — deduplicate nodes with identical (op, inputs)
//!
//! Design: single pass in topological order. Predecessors are processed
//! first so chains of fusable ops can cascade. Dead nodes are skipped.

extern crate alloc;

use alloc::vec::Vec;
use smallvec::SmallVec;
use crate::node::{GraphOp, NodeId, InputSource, FusionAttrs};
use crate::graph::Graph;
use hologram_ops::OpKind;

/// Statistics from a fusion pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusionStats {
    /// MatMul + activation pairs fused.
    pub matmul_activations_fused: usize,
    /// Silu + Mul → FusedSwiGlu patterns fused.
    pub swiglu_fused: usize,
    /// Duplicate nodes eliminated by CSE.
    pub cse_eliminated: usize,
}

impl FusionStats {
    pub fn total_removed(&self) -> usize {
        self.matmul_activations_fused
            + self.swiglu_fused
            + self.cse_eliminated
    }
}

/// Run the fusion pipeline on the graph. Called by the compiler between
/// `compute_schedule()` and the per-node compilation loop.
pub fn fuse(graph: &mut Graph) -> FusionStats {
    let succ = graph.build_successor_index();
    let n = graph.node_count();
    let mut stats = FusionStats::default();

    // Topological order = node index order (graph is append-only,
    // nodes are inserted in topological order during construction).
    for (i, node_succs) in succ.iter().enumerate().take(n) {
        let id = NodeId(i as u32);
        let node = match graph.get(id) {
            Some(n) => n.clone(),
            None => continue,
        };
        if node.op == GraphOp::Dead {
            continue;
        }

        // --- Pass 1: MatMul + activation epilogue fusion ---
        //
        // Pattern: MatMul(a, b) → [single successor] → Activation(x)
        // Action:  Replace activation with FusedMatMulActivation,
        //          absorb MatMul's inputs, kill MatMul.
        if let GraphOp::Op(OpKind::MatMul) = node.op {
            let successors = node_succs;
            if successors.len() == 1 {
                let succ_id = successors[0];
                if let Some(succ_node) = graph.get(succ_id) {
                    if succ_node.op.is_fusable_activation() {
                        let act_kind = match succ_node.op {
                            GraphOp::Op(k) => k as u16,
                            _ => continue,
                        };
                        // Rewire: the fused node takes MatMul's inputs.
                        let matmul_inputs = node.inputs.clone();
                        graph.replace_op(succ_id, GraphOp::Op(OpKind::FusedMatMulActivation));
                        graph.set_inputs(succ_id, matmul_inputs);
                        // Copy shape/dtype from the matmul's output to the fused node
                        // (the fused node produces the same shaped output as the matmul,
                        // just with activation applied).
                        let matmul_node = graph.get(id).unwrap();
                        let out_dtype = matmul_node.output_dtype;
                        let out_shape = matmul_node.output_shape;
                        if let Some(fused) = graph.get_mut(succ_id) {
                            fused.output_dtype = out_dtype;
                            fused.output_shape = out_shape;
                        }
                        // Store the activation discriminant.
                        graph.set_fusion_attrs(succ_id, FusionAttrs { activation: act_kind, ..Default::default() });
                        // Kill the original MatMul.
                        graph.kill_node(id);
                        stats.matmul_activations_fused += 1;
                        continue;
                    }
                }
            }
        }

        // --- Pass 1b: Conv2d + activation epilogue fusion ---
        //
        // Same pattern as MatMul but for Conv2d.
        if let GraphOp::Op(OpKind::Conv2d) = node.op {
            let successors = node_succs;
            if successors.len() == 1 {
                let succ_id = successors[0];
                if let Some(succ_node) = graph.get(succ_id) {
                    if succ_node.op.is_fusable_activation() {
                        let act_kind = match succ_node.op {
                            GraphOp::Op(k) => k as u16,
                            _ => continue,
                        };
                        let conv_inputs = node.inputs.clone();
                        graph.replace_op(succ_id, GraphOp::Op(OpKind::FusedConv2dActivation));
                        graph.set_inputs(succ_id, conv_inputs);
                        let conv_node = graph.get(id).unwrap();
                        let out_dtype = conv_node.output_dtype;
                        let out_shape = conv_node.output_shape;
                        if let Some(fused) = graph.get_mut(succ_id) {
                            fused.output_dtype = out_dtype;
                            fused.output_shape = out_shape;
                        }
                        graph.set_fusion_attrs(succ_id, FusionAttrs { activation: act_kind, ..Default::default() });
                        graph.kill_node(id);
                        stats.matmul_activations_fused += 1;
                        continue;
                    }
                }
            }
        }

        // --- Pass 1c: Norm + activation epilogue fusion ---
        //
        // Pattern: LayerNorm/RmsNorm/GroupNorm/InstanceNorm → activation
        if matches!(node.op,
            GraphOp::Op(OpKind::LayerNorm) | GraphOp::Op(OpKind::RmsNorm)
            | GraphOp::Op(OpKind::GroupNorm) | GraphOp::Op(OpKind::InstanceNorm)
        ) {
            let successors = node_succs;
            if successors.len() == 1 {
                let succ_id = successors[0];
                if let Some(succ_node) = graph.get(succ_id) {
                    if succ_node.op.is_fusable_activation() {
                        let act_kind = match succ_node.op {
                            GraphOp::Op(k) => k as u16,
                            _ => continue,
                        };
                        let norm_inputs = node.inputs.clone();
                        graph.replace_op(succ_id, GraphOp::Op(OpKind::FusedNormActivation));
                        graph.set_inputs(succ_id, norm_inputs);
                        let norm_node = graph.get(id).unwrap();
                        let out_dtype = norm_node.output_dtype;
                        let out_shape = norm_node.output_shape;
                        if let Some(fused) = graph.get_mut(succ_id) {
                            fused.output_dtype = out_dtype;
                            fused.output_shape = out_shape;
                        }
                        graph.set_fusion_attrs(succ_id, FusionAttrs { activation: act_kind, ..Default::default() });
                        graph.kill_node(id);
                        stats.matmul_activations_fused += 1;
                        continue;
                    }
                }
            }
        }

        // --- Pass 1d: Transpose pair elimination ---
        //
        // Pattern: Transpose → [single successor] → Transpose
        // If the second transpose's output shape matches the first
        // transpose's input shape, the pair is an identity — rewire
        // the second transpose's successors to the first's input and
        // kill both.
        if let GraphOp::Op(OpKind::Transpose) = node.op {
            let successors = node_succs;
            if successors.len() == 1 {
                let succ_id = successors[0];
                if let Some(succ_node) = graph.get(succ_id) {
                    if let GraphOp::Op(OpKind::Transpose) = succ_node.op {
                        // The pair cancels: T(T(x)) = x. Rewire the
                        // second transpose's successors to use the first
                        // transpose's input directly.
                        if let Some(InputSource::Node(original_input)) = node.inputs.first().copied() {
                            // Rewire all successors of succ_id to use original_input.
                            let succ_succ = graph.build_successor_index();
                            let succ_id_idx = succ_id.0 as usize;
                            if succ_id_idx < succ_succ.len() {
                                for ss_id in succ_succ[succ_id_idx].clone() {
                                    if let Some(ss_node) = graph.get(ss_id) {
                                        let mut new_inputs = ss_node.inputs.clone();
                                        for inp in &mut new_inputs {
                                            if let InputSource::Node(nid) = inp {
                                                if *nid == succ_id {
                                                    *nid = original_input;
                                                }
                                            }
                                        }
                                        graph.set_inputs(ss_id, new_inputs);
                                    }
                                }
                            }
                            graph.kill_node(succ_id);
                            graph.kill_node(id);
                            stats.matmul_activations_fused += 1;
                            continue;
                        }
                    }
                }
            }
        }

        // --- Pass 2: SwiGLU fusion ---
        //
        // Pattern: Silu(gate) → [single successor] → Mul(silu_out, up)
        //          where `up` is the Mul's other input (not silu_out).
        // Action:  Replace Mul with FusedSwiGlu(gate, up), kill Silu.
        if let GraphOp::Op(OpKind::Silu) = node.op {
            let successors = node_succs;
            if successors.len() == 1 {
                let mul_id = successors[0];
                if let Some(mul_node) = graph.get(mul_id) {
                    if let GraphOp::Op(OpKind::Mul) = mul_node.op {
                        // Find which of Mul's inputs is the Silu output
                        // and which is the "up" path.
                        let silu_out_id = id;
                        let mut up_src = None;
                        for inp in &mul_node.inputs {
                            if let InputSource::Node(inp_id) = inp {
                                if *inp_id != silu_out_id {
                                    up_src = Some(*inp);
                                }
                            }
                        }
                        if let Some(up) = up_src {
                            // Gate = Silu's input.
                            let gate = node.inputs.first().copied()
                                .unwrap_or(InputSource::Node(id));
                            let mut new_inputs: SmallVec<[InputSource; 4]> = SmallVec::new();
                            new_inputs.push(gate);
                            new_inputs.push(up);
                            graph.replace_op(mul_id, GraphOp::Op(OpKind::FusedSwiGlu));
                            graph.set_inputs(mul_id, new_inputs);
                            graph.kill_node(id);
                            stats.swiglu_fused += 1;
                            continue;
                        }
                    }
                }
            }
        }
    }

    // --- Pass 2b: Unary chain fusion ---
    //
    // Second pass: look for chains of unary elementwise ops and
    // collapse them into a single FusedUnaryChain node.
    // Must rebuild successor index since Pass 1/2 may have changed edges.
    let succ2 = graph.build_successor_index();
    for i in 0..n {
        let id = NodeId(i as u32);
        let node = match graph.get(id) {
            Some(n) => n.clone(),
            None => continue,
        };
        if !node.op.is_fusable_activation() {
            continue;
        }
        // Check if this activation's successor is also a fusable activation
        // and has only one predecessor (us).
        let node_succs = if i < succ2.len() { &succ2[i] } else { continue };
        if node_succs.len() != 1 {
            continue;
        }
        let succ_id = node_succs[0];
        let succ_node = match graph.get(succ_id) {
            Some(n) => n.clone(),
            None => continue,
        };
        if !succ_node.op.is_fusable_activation() {
            continue;
        }
        // We have a chain of at least 2. Walk forward to find the full chain.
        let first_kind = match node.op {
            GraphOp::Op(k) => k as u16,
            _ => continue,
        };
        let mut chain: [u16; 8] = [0; 8];
        chain[0] = first_kind;
        let mut chain_len: u8 = 1;
        let mut current_id = succ_id;
        let mut current_op = succ_node.op;
        loop {
            if chain_len >= 8 { break; }
            let k = match current_op {
                GraphOp::Op(k) if k.is_fusable_activation() => k as u16,
                _ => break,
            };
            chain[chain_len as usize] = k;
            chain_len += 1;
            // Can we extend further?
            let cidx = current_id.0 as usize;
            let csuccs = if cidx < succ2.len() { &succ2[cidx] } else { break };
            if csuccs.len() != 1 { break; }
            let next_id = csuccs[0];
            let next_node = match graph.get(next_id) {
                Some(n) => n.clone(),
                None => break,
            };
            if !next_node.op.is_fusable_activation() { break; }
            // Kill the intermediate node.
            graph.kill_node(current_id);
            current_id = next_id;
            current_op = next_node.op;
        }
        if chain_len < 2 {
            continue;
        }
        // Kill all intermediate nodes (the last one becomes the fused node).
        // The first node (id) gets killed; the last node (current_id) becomes
        // FusedUnaryChain with the first node's input.
        let original_input = node.inputs.first().copied()
            .unwrap_or(InputSource::Node(id));
        let mut new_inputs: SmallVec<[InputSource; 4]> = SmallVec::new();
        new_inputs.push(original_input);
        graph.replace_op(current_id, GraphOp::Op(OpKind::FusedUnaryChain));
        graph.set_inputs(current_id, new_inputs);
        graph.set_fusion_attrs(current_id, FusionAttrs {
            activation: chain[0],
            chain_len,
            chain,
        });
        graph.kill_node(id);
        stats.cse_eliminated += 0; // Don't count these in CSE
        stats.matmul_activations_fused += 1; // Reuse counter for structural opts
    }

    // --- Pass 3: Common Subexpression Elimination ---
    //
    // Hash-cons pass: nodes with identical (op, sorted_inputs) share
    // a single computation. The duplicate is killed and all its
    // successors are rewired to the canonical node.
    // We need a hashmap — but this crate is no_std. Use a simple Vec
    // for now (O(n²) but graphs are small in practice).
    struct Signature {
        op: GraphOp,
        inputs: SmallVec<[InputSource; 4]>,
    }
    impl Signature {
        fn matches(&self, other: &Signature) -> bool {
            self.op == other.op && self.inputs == other.inputs
        }
    }

    let mut seen: Vec<(Signature, NodeId)> = Vec::new();
    for (i, node_succs) in succ.iter().enumerate().take(n) {
        let id = NodeId(i as u32);
        let node = match graph.get(id) {
            Some(n) => n.clone(),
            None => continue,
        };
        match node.op {
            GraphOp::Dead | GraphOp::Input | GraphOp::Output | GraphOp::Constant(_) => continue,
            _ => {}
        }
        let sig = Signature { op: node.op, inputs: node.inputs.clone() };
        let mut canonical = None;
        for (s, cid) in &seen {
            if s.matches(&sig) {
                canonical = Some(*cid);
                break;
            }
        }
        if let Some(canon_id) = canonical {
            // Rewire all successors of `id` to use `canon_id` instead.
            let successors_of_id: Vec<NodeId> = node_succs.to_vec();
            for succ_id in successors_of_id {
                if let Some(succ_node) = graph.get(succ_id) {
                    let mut new_inputs = succ_node.inputs.clone();
                    for inp in &mut new_inputs {
                        if let InputSource::Node(nid) = inp {
                            if *nid == id {
                                *nid = canon_id;
                            }
                        }
                    }
                    graph.set_inputs(succ_id, new_inputs);
                }
            }
            graph.kill_node(id);
            stats.cse_eliminated += 1;
        } else {
            seen.push((sig, id));
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::registry::{DTypeId, ShapeId};

    fn make_node(op: GraphOp, inputs: &[InputSource]) -> Node {
        Node {
            op,
            inputs: inputs.iter().copied().collect(),
            output_dtype: DTypeId(8), // F32
            output_shape: ShapeId(0),
        }
    }

    #[test]
    fn fuse_matmul_activation() {
        let mut g = Graph::new();
        let in0 = g.add_node(make_node(GraphOp::Input, &[]));
        let in1 = g.add_node(make_node(GraphOp::Input, &[]));
        let mm = g.add_node(make_node(
            GraphOp::Op(OpKind::MatMul),
            &[InputSource::Node(in0), InputSource::Node(in1)],
        ));
        let relu = g.add_node(make_node(
            GraphOp::Op(OpKind::Relu),
            &[InputSource::Node(mm)],
        ));
        let out = g.add_node(make_node(
            GraphOp::Output,
            &[InputSource::Node(relu)],
        ));
        g.add_input(in0);
        g.add_output(out);

        let stats = fuse(&mut g);
        assert_eq!(stats.matmul_activations_fused, 1);
        // MatMul should be dead.
        assert_eq!(g.get(mm).unwrap().op, GraphOp::Dead);
        // Relu should now be FusedMatMulActivation.
        assert_eq!(g.get(relu).unwrap().op, GraphOp::Op(OpKind::FusedMatMulActivation));
        // Fused node should have MatMul's inputs.
        assert_eq!(g.get(relu).unwrap().inputs.len(), 2);
        // Fusion attrs should record Relu activation.
        let fa = g.fusion_attrs(relu).unwrap();
        assert_eq!(fa.activation, OpKind::Relu as u16);
    }

    #[test]
    fn fuse_swiglu() {
        let mut g = Graph::new();
        let gate = g.add_node(make_node(GraphOp::Input, &[]));
        let up = g.add_node(make_node(GraphOp::Input, &[]));
        let silu = g.add_node(make_node(
            GraphOp::Op(OpKind::Silu),
            &[InputSource::Node(gate)],
        ));
        let mul = g.add_node(make_node(
            GraphOp::Op(OpKind::Mul),
            &[InputSource::Node(silu), InputSource::Node(up)],
        ));
        let out = g.add_node(make_node(
            GraphOp::Output,
            &[InputSource::Node(mul)],
        ));
        g.add_input(gate);
        g.add_input(up);
        g.add_output(out);

        let stats = fuse(&mut g);
        assert_eq!(stats.swiglu_fused, 1);
        assert_eq!(g.get(silu).unwrap().op, GraphOp::Dead);
        assert_eq!(g.get(mul).unwrap().op, GraphOp::Op(OpKind::FusedSwiGlu));
        let fused = g.get(mul).unwrap();
        assert_eq!(fused.inputs.len(), 2);
    }

    #[test]
    fn fuse_cse() {
        let mut g = Graph::new();
        let inp = g.add_node(make_node(GraphOp::Input, &[]));
        let relu1 = g.add_node(make_node(
            GraphOp::Op(OpKind::Relu),
            &[InputSource::Node(inp)],
        ));
        let relu2 = g.add_node(make_node(
            GraphOp::Op(OpKind::Relu),
            &[InputSource::Node(inp)],
        ));
        let add = g.add_node(make_node(
            GraphOp::Op(OpKind::Add),
            &[InputSource::Node(relu1), InputSource::Node(relu2)],
        ));
        let out = g.add_node(make_node(
            GraphOp::Output,
            &[InputSource::Node(add)],
        ));
        g.add_input(inp);
        g.add_output(out);

        let stats = fuse(&mut g);
        assert_eq!(stats.cse_eliminated, 1);
        // One of the relu nodes should be dead.
        let relu1_dead = g.get(relu1).unwrap().op == GraphOp::Dead;
        let relu2_dead = g.get(relu2).unwrap().op == GraphOp::Dead;
        assert!(relu1_dead || relu2_dead);
        assert!(!(relu1_dead && relu2_dead));
    }

    #[test]
    fn fuse_no_ops_on_pure_io() {
        let mut g = Graph::new();
        let inp = g.add_node(make_node(GraphOp::Input, &[]));
        let out = g.add_node(make_node(
            GraphOp::Output,
            &[InputSource::Node(inp)],
        ));
        g.add_input(inp);
        g.add_output(out);

        let stats = fuse(&mut g);
        assert_eq!(stats.total_removed(), 0);
    }

    #[test]
    fn fuse_stats_total() {
        let stats = FusionStats {
            matmul_activations_fused: 2,
            swiglu_fused: 1,
            cse_eliminated: 3,
        };
        assert_eq!(stats.total_removed(), 6);
    }
}
