//! Term-to-Graph lowering — translates a validated CompileUnit into
//! a [`Graph`] for the existing compilation pipeline.
//!
//! O(n) — one pass over the arena, one graph node per term node.
//! The `node_map` array avoids re-lowering shared subexpressions.

use hologram_core::op::RingLevel;
use hologram_core::term::{Binding, HoloCompileUnit, TermArena, TermId, TermKind, VarId};
use hologram_graph::graph::edge;
use hologram_graph::{ConstantData, Graph, GraphOp};

use crate::error::{CompileError, CompileResult};

/// Lower a validated CompileUnit into a Graph.
///
/// Each term node maps to a graph node:
/// - `IntLit(v)` / `BrailleLit(v)` / `QuantumLit` → `Constant` node
/// - `UnaryApp { op, arg }` → `Prim(op)` with one edge
/// - `BinaryApp { op, lhs, rhs }` → `Prim(op)` with two edges
/// - `Var(id)` → reference to the node produced by the corresponding binding
///
/// The resulting graph feeds directly into `compile()`.
pub fn lower_to_graph(unit: &HoloCompileUnit) -> CompileResult<Graph> {
    let arena = &unit.arena;
    let bindings = &unit.bindings[..unit.binding_count as usize];
    let mut graph = Graph::new();

    // TermId → NodeId mapping. None = not yet lowered.
    let mut node_map = vec![None; arena.len() as usize];

    // Process bindings first: they define variables.
    // After this loop, node_map[binding.rhs.0] holds the graph NodeId for each binding.
    for binding in bindings {
        let _node_id = lower_term(arena, bindings, binding.rhs, &mut graph, &mut node_map)?;
    }

    // Lower the root term.
    let root_node = lower_term(arena, bindings, unit.root_term, &mut graph, &mut node_map)?;

    // Create output node.
    let output_id = graph.add_node(GraphOp::Output);
    edge::connect(&mut graph, root_node, output_id, 0);
    graph.add_output("result", output_id);

    Ok(graph)
}

/// Recursively lower a single term, memoizing in `node_map`.
///
/// Variable resolution: when a `Var(VarId)` is encountered, the function
/// walks the bindings slice to find the binding whose `var` matches,
/// then returns the already-lowered graph node for that binding's `rhs`.
/// O(k) per variable where k = number of bindings (k <= 64).
fn lower_term(
    arena: &TermArena,
    bindings: &[Binding],
    id: TermId,
    graph: &mut Graph,
    node_map: &mut [Option<hologram_graph::graph::node::NodeId>],
) -> CompileResult<hologram_graph::graph::node::NodeId> {
    let idx = id.0 as usize;

    // Memoized: already lowered.
    if let Some(nid) = node_map.get(idx).copied().flatten() {
        return Ok(nid);
    }

    let node = arena.get(id);
    let nid = match node.kind {
        TermKind::IntLit(v) => {
            let bytes = encode_literal(v, RingLevel::Q0);
            let cid = graph.add_constant(ConstantData::Bytes(bytes));
            graph.add_node(GraphOp::Constant(cid))
        }
        TermKind::BrailleLit(v) => {
            let cid = graph.add_constant(ConstantData::Bytes(vec![v]));
            graph.add_node(GraphOp::Constant(cid))
        }
        TermKind::QuantumLit { level, value } => {
            let bytes = encode_literal(value as i64, level);
            let cid = graph.add_constant(ConstantData::Bytes(bytes));
            graph.add_node(GraphOp::Constant(cid))
        }
        TermKind::UnaryApp { op, arg } => {
            let arg_nid = lower_term(arena, bindings, arg, graph, node_map)?;
            let nid = graph.add_node(GraphOp::Prim(op));
            edge::connect(graph, arg_nid, nid, 0);
            nid
        }
        TermKind::BinaryApp { op, lhs, rhs } => {
            let lhs_nid = lower_term(arena, bindings, lhs, graph, node_map)?;
            let rhs_nid = lower_term(arena, bindings, rhs, graph, node_map)?;
            let nid = graph.add_node(GraphOp::Prim(op));
            edge::connect(graph, lhs_nid, nid, 0);
            edge::connect(graph, rhs_nid, nid, 1);
            nid
        }
        TermKind::LutApp { op, arg } => {
            let arg_nid = lower_term(arena, bindings, arg, graph, node_map)?;
            let nid = graph.add_node(GraphOp::Lut(op));
            edge::connect(graph, arg_nid, nid, 0);
            nid
        }
        TermKind::FloatApp { op, arg0, arg1 } => {
            let float_op = *arena.get_float_op(op);
            let a0 = lower_term(arena, bindings, arg0, graph, node_map)?;
            let nid = graph.add_node(GraphOp::Float(float_op));
            edge::connect(graph, a0, nid, 0);
            if arg1.0 != u32::MAX {
                let a1 = lower_term(arena, bindings, arg1, graph, node_map)?;
                edge::connect(graph, a1, nid, 1);
            }
            nid
        }
        TermKind::RingUnaryApp { op, level, arg } => {
            let arg_nid = lower_term(arena, bindings, arg, graph, node_map)?;
            let nid = graph.add_node(GraphOp::RingPrimUnary(op, level));
            edge::connect(graph, arg_nid, nid, 0);
            nid
        }
        TermKind::RingBinaryApp { op, level, lhs, rhs } => {
            let lhs_nid = lower_term(arena, bindings, lhs, graph, node_map)?;
            let rhs_nid = lower_term(arena, bindings, rhs, graph, node_map)?;
            let nid = graph.add_node(GraphOp::RingPrimBinary(op, level));
            edge::connect(graph, lhs_nid, nid, 0);
            edge::connect(graph, rhs_nid, nid, 1);
            nid
        }
        TermKind::Constant(cref) => {
            let cid = hologram_graph::ConstantId::new(cref.0);
            graph.add_node(GraphOp::Constant(cid))
        }
        TermKind::GraphInput(_) => {
            graph.add_node(GraphOp::Input)
        }
        TermKind::GraphOutput(inner) => {
            let inner_nid = lower_term(arena, bindings, inner, graph, node_map)?;
            let nid = graph.add_node(GraphOp::Output);
            edge::connect(graph, inner_nid, nid, 0);
            nid
        }
        TermKind::FusedViewRef(vref) => {
            let view = *arena.get_view(vref);
            graph.add_node(GraphOp::FusedView(view))
        }
        TermKind::Passthrough(inner) => {
            let inner_nid = lower_term(arena, bindings, inner, graph, node_map)?;
            let nid = graph.add_node(GraphOp::Passthrough);
            edge::connect(graph, inner_nid, nid, 0);
            nid
        }
        TermKind::Var(vid) => {
            return resolve_var(arena, bindings, vid, graph, node_map);
        }
    };

    if idx < node_map.len() {
        node_map[idx] = Some(nid);
    }

    Ok(nid)
}

/// Resolve a variable reference by finding its binding and returning the
/// already-lowered graph node for the binding's rhs term.
fn resolve_var(
    arena: &TermArena,
    bindings: &[Binding],
    vid: VarId,
    graph: &mut Graph,
    node_map: &mut [Option<hologram_graph::graph::node::NodeId>],
) -> CompileResult<hologram_graph::graph::node::NodeId> {
    for binding in bindings {
        if binding.var == vid {
            return lower_term(arena, bindings, binding.rhs, graph, node_map);
        }
    }
    Err(CompileError::Validation(format!(
        "unbound variable VarId({})",
        vid.0
    )))
}

/// Encode an integer literal as bytes at the given quantum level.
fn encode_literal(value: i64, level: RingLevel) -> Vec<u8> {
    match level {
        RingLevel::Q0 => vec![value as u8],
        RingLevel::Q1 => (value as u16).to_le_bytes().to_vec(),
        RingLevel::Q2 => {
            let v = value as u32;
            vec![v as u8, (v >> 8) as u8, (v >> 16) as u8]
        }
        RingLevel::Q3 => (value as u32).to_le_bytes().to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::PrimOp;
    use hologram_core::term::{HoloCompileUnit, TermArena, TermKind};
    use uor_foundation::enums::VerificationDomain;

    #[test]
    fn lower_integer_literal() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(42));
        let unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Should have: 1 Constant node + 1 Output node = 2 nodes
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn lower_unary_application() {
        let mut arena = TermArena::new();
        let lit = arena.alloc(TermKind::IntLit(42));
        let root = arena.alloc(TermKind::UnaryApp {
            op: PrimOp::Neg,
            arg: lit,
        });
        let unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Constant(42) → Neg → Output = 3 nodes
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn lower_binary_application() {
        let mut arena = TermArena::new();
        let a = arena.alloc(TermKind::IntLit(1));
        let b = arena.alloc(TermKind::IntLit(2));
        let root = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: a,
            rhs: b,
        });
        let unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Constant(1) + Constant(2) → Add → Output = 4 nodes
        assert_eq!(graph.node_count(), 4);
    }

    #[test]
    fn lower_nested_expression() {
        // add(mul(2, 3), neg(1))
        let mut arena = TermArena::new();
        let two = arena.alloc(TermKind::IntLit(2));
        let three = arena.alloc(TermKind::IntLit(3));
        let one = arena.alloc(TermKind::IntLit(1));
        let mul = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Mul,
            lhs: two,
            rhs: three,
        });
        let neg = arena.alloc(TermKind::UnaryApp {
            op: PrimOp::Neg,
            arg: one,
        });
        let root = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: mul,
            rhs: neg,
        });
        let unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // 3 constants + Mul + Neg + Add + Output = 7 nodes
        assert_eq!(graph.node_count(), 7);
    }

    #[test]
    fn lower_quantum_literal() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::QuantumLit {
            level: RingLevel::Q1,
            value: 1000,
        });
        let unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q1,
            12.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        assert_eq!(graph.node_count(), 2); // Constant + Output
    }

    #[test]
    fn lower_variable_resolution() {
        // "let x : Q0 = 42 ; neg(x)" → Constant(42) → Neg → Output = 3 nodes
        let unit = {
            let mut arena = TermArena::new();
            let lit = arena.alloc(TermKind::IntLit(42));
            let var_node = arena.alloc(TermKind::Var(hologram_core::term::VarId(0)));
            let root = arena.alloc(TermKind::UnaryApp {
                op: PrimOp::Neg,
                arg: var_node,
            });

            let mut unit = HoloCompileUnit::new(
                arena,
                root,
                RingLevel::Q0,
                6.0,
                &[VerificationDomain::Algebraic],
            );
            unit.bindings[0] = hologram_core::term::Binding {
                var: hologram_core::term::VarId(0),
                ty: hologram_core::term::TypeId::UNCONSTRAINED,
                rhs: lit,
            };
            unit.binding_count = 1;
            unit
        };

        let graph = lower_to_graph(&unit).unwrap();
        // Constant(42) + Neg + Output = 3 nodes
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn lower_multiple_variables() {
        // "let a : Q0 = 1 ; let b : Q0 = 2 ; add(a, b)"
        let unit = {
            let mut arena = TermArena::new();
            let lit_a = arena.alloc(TermKind::IntLit(1));
            let lit_b = arena.alloc(TermKind::IntLit(2));
            let var_a = arena.alloc(TermKind::Var(hologram_core::term::VarId(0)));
            let var_b = arena.alloc(TermKind::Var(hologram_core::term::VarId(1)));
            let root = arena.alloc(TermKind::BinaryApp {
                op: PrimOp::Add,
                lhs: var_a,
                rhs: var_b,
            });

            let mut unit = HoloCompileUnit::new(
                arena,
                root,
                RingLevel::Q0,
                6.0,
                &[VerificationDomain::Algebraic],
            );
            unit.bindings[0] = hologram_core::term::Binding {
                var: hologram_core::term::VarId(0),
                ty: hologram_core::term::TypeId::UNCONSTRAINED,
                rhs: lit_a,
            };
            unit.bindings[1] = hologram_core::term::Binding {
                var: hologram_core::term::VarId(1),
                ty: hologram_core::term::TypeId::UNCONSTRAINED,
                rhs: lit_b,
            };
            unit.binding_count = 2;
            unit
        };

        let graph = lower_to_graph(&unit).unwrap();
        // Constant(1) + Constant(2) + Add + Output = 4 nodes
        assert_eq!(graph.node_count(), 4);
    }

    #[test]
    fn lower_unbound_variable_errors() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::Var(hologram_core::term::VarId(99)));
        let unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );
        assert!(lower_to_graph(&unit).is_err());
    }

    #[test]
    fn lower_lut_application() {
        let mut arena = TermArena::new();
        let lit = arena.alloc(TermKind::IntLit(42));
        let root = arena.alloc(TermKind::LutApp {
            op: hologram_core::op::LutOp::Sigmoid,
            arg: lit,
        });
        let unit = HoloCompileUnit::new(
            arena,
            root,
            RingLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Constant(42) → Lut(Sigmoid) → Output = 3 nodes
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn encode_literal_q0() {
        assert_eq!(encode_literal(42, RingLevel::Q0), vec![42u8]);
    }

    #[test]
    fn encode_literal_q1() {
        let bytes = encode_literal(1000, RingLevel::Q1);
        assert_eq!(bytes, 1000u16.to_le_bytes().to_vec());
    }

    #[test]
    fn encode_literal_q3() {
        let bytes = encode_literal(0x12345678, RingLevel::Q3);
        assert_eq!(bytes, 0x12345678u32.to_le_bytes().to_vec());
    }
}
