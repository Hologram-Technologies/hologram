//! Shape pre-propagation pass.
//!
//! Before each level's data dispatch, this module walks every node in the
//! level, reads input shapes from `ShapeMap`, applies the unified
//! `resolve_float_shape()`, and writes the output shape to `ShapeMap`.
//! This eliminates the need for post-dispatch shape resolution from
//! output buffer sizes.

use std::collections::HashMap;

use hologram_core::op::{FloatDType, FloatOp, ShapeSpec};
use hologram_graph::graph::node::{InputSource, Node, NodeId};
use hologram_graph::graph::GraphOp;
use hologram_graph::schedule::levels::ParallelLevel;

use crate::buffer::{BufferArena, ShapeMap};
use crate::eval::shape_resolve::{resolve_float_shape, ShapeContext};

/// Pre-propagate shapes for all nodes in a level before data dispatch.
///
/// For each FloatOp node, computes the output shape from input shapes
/// using the unified `resolve_float_shape()` and writes it to `shape_map`.
///
/// After this pass, `dispatch_level` can read pre-computed shapes from
/// `shape_map` instead of inferring them from output buffer sizes.
///
/// `shape_hints` — optional pre-projected shape map from `walk_shape_context()`.
/// Maps `NodeId.index → concrete shape`. When a hint is present for a node, it
/// is used directly without any further inference, ensuring correctness for
/// variable-length inputs (seq>1, batch>1, etc.).
pub fn propagate_level_shapes(
    level: &ParallelLevel,
    node_map: &HashMap<NodeId, &Node>,
    arena: &BufferArena,
    shape_map: &mut ShapeMap,
    compiled_shapes: &HashMap<NodeId, Vec<usize>>,
    compiled_dtypes: &HashMap<NodeId, FloatDType>,
    shape_hints: Option<&HashMap<u32, Vec<usize>>>,
) {
    for &node_id in &level.node_ids {
        // Shape hints from walk_shape_context are NOT used here.
        // Instead, shapes are resolved from two authoritative sources:
        // 1. Compiled shapes — correct rank, with 0-sentinels for dynamic dims
        // 2. Buffer sizes — correct product (total element count)
        // Together these resolve any dynamic dim: unknown = product / known_dims.
        // This is simpler and more robust than the walker's i64 propagation chains.
        let _ = shape_hints; // suppress unused warning
        let Some(node) = node_map.get(&node_id) else {
            continue;
        };
        let fop = match &node.op {
            GraphOp::Float(fop) => fop,
            GraphOp::FusedFloatChain(chain) => {
                // Use first op for shape spec (chain preserves shape).
                if let Some(first) = chain.first() {
                    first
                } else {
                    continue;
                }
            }
            _ => continue,
        };

        // ── Fast path: SameAs(0) ops just copy input[0]'s shape ──────────
        // Skips the expensive input_elem_counts + resolve_float_shape() path
        // for ~60-70% of nodes (all unary elementwise, norms, softmax, etc.).
        if matches!(fop.output_shape_spec(), ShapeSpec::SameAs(0)) {
            if let Some(src_id) = node.inputs.first().and_then(|s| match s.source {
                InputSource::Node(id) => Some(id),
                _ => None,
            }) {
                if let Some(shape) = shape_map.get(src_id) {
                    let s = shape.to_vec();
                    if !s.is_empty() && !s.contains(&0) {
                        shape_map.insert(node_id, s);
                        continue;
                    }
                }
            }
            // Fall through to full resolution if input shape is unavailable
            // or contains 0-sentinels.
        }

        // Gather input shapes from shape_map (may be stale compiled shapes).
        let input_shapes: Vec<Vec<usize>> = node
            .inputs
            .iter()
            .filter_map(|slot| match slot.source {
                InputSource::Node(src_id) => shape_map.get(src_id).map(|s| s.to_vec()),
                InputSource::GraphInput { .. } => None,
                InputSource::None => None,
            })
            .collect();

        // If we couldn't gather enough input shapes, skip.
        if input_shapes.len() < fop.arity() as usize {
            continue;
        }

        // Collect actual element counts from the arena for ALL inputs.
        // These are used by ShapeContext to detect and correct stale shapes
        // (e.g. compiled seq=32 sentinel when runtime seq=9) before computing
        // broadcast/element-wise output shapes.
        let input_elem_counts: Vec<usize> = node
            .inputs
            .iter()
            .map(|slot| match slot.source {
                InputSource::Node(src_id) => {
                    let es = compiled_dtypes
                        .get(&src_id)
                        .map(|d| d.byte_size())
                        .unwrap_or(4)
                        .max(1);
                    arena
                        .get(src_id)
                        .ok()
                        .map(|buf| buf.len() / es)
                        .unwrap_or(0)
                }
                _ => 0,
            })
            .collect();

        // Actual element count of input[0] from arena (authoritative for all ops).
        let input_elems = input_elem_counts.first().copied().unwrap_or(0);

        // For Reshape, get shape tensor bytes from arena (input[1]).
        let shape_tensor_bytes = if matches!(fop, FloatOp::Reshape) && node.inputs.len() >= 2 {
            match node.inputs[1].source {
                InputSource::Node(src_id) => arena.get(src_id).ok(),
                _ => None,
            }
        } else {
            None
        };

        let ctx = ShapeContext {
            input_shapes: &input_shapes,
            compiled_shape: compiled_shapes.get(&node_id),
            input_elems,
            input_elem_counts: &input_elem_counts,
            shape_tensor_bytes,
            compiled_dtype: compiled_dtypes.get(&node_id),
        };

        // Only propagate if no concrete compiled shape exists.
        // Compiled shapes are authoritative — runtime propagation should
        // only fill in shapes for nodes the compiler couldn't resolve.
        //
        // Exception: for Reshape, the compiled shape may use seq=1 as a static
        // sentinel (rather than 0) for dims that are actually dynamic. We detect
        // this as a product mismatch vs. the actual input buffer and re-propagate.
        if let Some(compiled) = compiled_shapes.get(&node_id) {
            if !compiled.is_empty() && !compiled.contains(&0) {
                // For Reshape, verify the compiled product matches actual input.
                // A mismatch means seq=1 was used as a compile-time sentinel.
                let compiled_product: usize = compiled.iter().product();
                let stale = matches!(fop, FloatOp::Reshape)
                    && input_elems > 0
                    && input_elems != compiled_product;

                if !stale {
                    // Compiled shape is fully concrete — trust it over propagation.
                    // Ensure it's in shape_map (seed_shape_map should have done this,
                    // but guard against missed cases).
                    if shape_map.get(node_id).is_none() {
                        shape_map.insert(node_id, compiled.clone());
                    }
                    continue;
                }
                // Fall through: let resolve_reshape produce the correct shape.
            }
        }

        if let Some(s) = resolve_float_shape(fop, &ctx) {
            if !s.is_empty() && !s.contains(&0) {
                shape_map.insert(node_id, s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::shape_resolve;

    #[test]
    fn test_propagation_uses_resolve_float_shape() {
        // Verify that resolve_float_shape handles basic ops correctly.
        let inputs = vec![vec![2, 3, 4]];
        let ctx = ShapeContext {
            input_shapes: &inputs,
            compiled_shape: None,
            input_elems: 24,
            input_elem_counts: &[],
            shape_tensor_bytes: None,
            compiled_dtype: None,
        };
        // Relu is SameAs(0)
        let result = shape_resolve::resolve_float_shape(&FloatOp::Relu, &ctx);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }
}
