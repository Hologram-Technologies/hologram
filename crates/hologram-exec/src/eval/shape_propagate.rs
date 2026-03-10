//! Shape pre-propagation pass.
//!
//! Before each level's data dispatch, this module walks every node in the
//! level, reads input shapes from `ShapeMap`, applies the unified
//! `resolve_float_shape()`, and writes the output shape to `ShapeMap`.
//! This eliminates the need for post-dispatch shape resolution from
//! output buffer sizes.

use std::collections::HashMap;

use hologram_core::op::{FloatDType, FloatOp};
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
pub fn propagate_level_shapes(
    level: &ParallelLevel,
    node_map: &HashMap<NodeId, &Node>,
    arena: &BufferArena,
    shape_map: &mut ShapeMap,
    compiled_shapes: &HashMap<NodeId, Vec<usize>>,
    compiled_dtypes: &HashMap<NodeId, FloatDType>,
) {
    for &node_id in &level.node_ids {
        let Some(node) = node_map.get(&node_id) else {
            continue;
        };
        let GraphOp::Float(fop) = &node.op else {
            continue;
        };

        // Gather input shapes from shape_map.
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

        let input_elems = input_shapes
            .first()
            .map(|s| s.iter().product::<usize>())
            .unwrap_or(0);

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
            shape_tensor_bytes,
            compiled_dtype: compiled_dtypes.get(&node_id),
        };

        // Only propagate if no concrete compiled shape exists.
        // Compiled shapes are authoritative — runtime propagation should
        // only fill in shapes for nodes the compiler couldn't resolve.
        if let Some(compiled) = compiled_shapes.get(&node_id) {
            if !compiled.is_empty() && !compiled.contains(&0) {
                // Compiled shape is fully concrete — trust it over propagation.
                // Ensure it's in shape_map (seed_shape_map should have done this,
                // but guard against missed cases).
                if shape_map.get(node_id).is_none() {
                    shape_map.insert(node_id, compiled.clone());
                }
                continue;
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
            shape_tensor_bytes: None,
            compiled_dtype: None,
        };
        // Relu is SameAs(0)
        let result = shape_resolve::resolve_float_shape(&FloatOp::Relu, &ctx);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }
}
