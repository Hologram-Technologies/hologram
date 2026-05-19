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

    // ADR-053: v3 archives require shape coverage on every
    // non-Input/non-Output/non-Constant node and on every referenced
    // constant. The term lowering produces scalar Q0/Q1/Q2/Q3 values
    // — every node yields a single ring element, shape `[1]`. Populate
    // here in one pass rather than at each construction site.
    populate_scalar_shapes(&mut graph);

    Ok(graph)
}

/// Walk every node and constant in the graph and assign a default
/// `[1]` shape where one is missing. Used by the term-lowering path
/// where every value is a scalar ring element.
fn populate_scalar_shapes(graph: &mut Graph) {
    use hologram_graph::graph::node::NodeId;

    let node_ids: Vec<NodeId> = graph.nodes().map(|n| n.id).collect();
    let constant_ids: Vec<_> = node_ids
        .iter()
        .filter_map(|&id| match graph.get(id).map(|n| &n.op) {
            Some(GraphOp::Constant(cid)) => Some(*cid),
            _ => None,
        })
        .collect();

    for id in node_ids {
        let needs_shape = match graph.get(id).map(|n| &n.op) {
            Some(GraphOp::Input | GraphOp::Output | GraphOp::Constant(_)) => false,
            Some(_) => graph.node_shapes().get(&id).is_none(),
            None => false,
        };
        if needs_shape {
            graph.set_node_shape(id, vec![1]);
        }
    }
    for cid in constant_ids {
        if !graph.constant_shapes().contains_key(&cid) {
            graph.set_constant_shape(cid, vec![1]);
        }
    }
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
            // Constant fold: if operand is a literal, evaluate at compile time.
            if let TermKind::IntLit(a) = arena.get(arg).kind {
                let result = op.apply_unary_u64(a as u64, 1);
                let bytes = encode_literal(result as i64, RingLevel::Q0);
                let cid = graph.add_constant(ConstantData::Bytes(bytes));
                graph.add_node(GraphOp::Constant(cid))
            } else {
                let arg_nid = lower_term(arena, bindings, arg, graph, node_map)?;
                let nid = graph.add_node(GraphOp::Prim(op));
                edge::connect(graph, arg_nid, nid, 0);
                nid
            }
        }
        TermKind::BinaryApp { op, lhs, rhs } => {
            // Constant fold: if both operands are literals, evaluate at compile time.
            let lhs_kind = arena.get(lhs).kind;
            let rhs_kind = arena.get(rhs).kind;
            if let (TermKind::IntLit(a), TermKind::IntLit(b)) = (lhs_kind, rhs_kind) {
                let result = op.apply_binary_u64(a as u64, b as u64, 1);
                let bytes = encode_literal(result as i64, RingLevel::Q0);
                let cid = graph.add_constant(ConstantData::Bytes(bytes));
                graph.add_node(GraphOp::Constant(cid))
            } else if let (
                TermKind::QuantumLit {
                    level: la,
                    value: va,
                },
                TermKind::QuantumLit {
                    level: lb,
                    value: vb,
                },
            ) = (lhs_kind, rhs_kind)
            {
                if la == lb {
                    let bw = la.byte_width();
                    let result = op.apply_binary_u64(va as u64, vb as u64, bw);
                    let bytes = encode_literal(result as i64, la);
                    let cid = graph.add_constant(ConstantData::Bytes(bytes));
                    graph.add_node(GraphOp::Constant(cid))
                } else {
                    // Different levels: can't fold, lower normally
                    let lhs_nid = lower_term(arena, bindings, lhs, graph, node_map)?;
                    let rhs_nid = lower_term(arena, bindings, rhs, graph, node_map)?;
                    let nid = graph.add_node(GraphOp::Prim(op));
                    edge::connect(graph, lhs_nid, nid, 0);
                    edge::connect(graph, rhs_nid, nid, 1);
                    nid
                }
            } else {
                let lhs_nid = lower_term(arena, bindings, lhs, graph, node_map)?;
                let rhs_nid = lower_term(arena, bindings, rhs, graph, node_map)?;
                let nid = graph.add_node(GraphOp::Prim(op));
                edge::connect(graph, lhs_nid, nid, 0);
                edge::connect(graph, rhs_nid, nid, 1);
                nid
            }
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
            // Prefer canonical `Compute(SemanticOp)` when the op is
            // covered by the canonical layer; falls back to legacy
            // `Float(FloatOp)` otherwise. ADR-047 Sprint 37 Phase 3.3.
            let nid = graph.add_node(GraphOp::from_float(float_op));
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
        TermKind::RingBinaryApp {
            op,
            level,
            lhs,
            rhs,
        } => {
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
        TermKind::GraphInput(_) => graph.add_node(GraphOp::Input),
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
    use uor_foundation::QuantumLevel;

    #[test]
    fn lower_integer_literal() {
        let mut arena = TermArena::new();
        let root = arena.alloc(TermKind::IntLit(42));
        let unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::Q0,
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
            QuantumLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Constant-folded: neg(42) → Constant(214) + Output = 2 nodes
        assert_eq!(graph.node_count(), 2);
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
            QuantumLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Constant-folded: add(1, 2) → Constant(3) + Output = 2 nodes
        assert_eq!(graph.node_count(), 2);
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
            QuantumLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Leaf-level constant fold: mul(2,3)→Constant(6), neg(1)→Constant(255)
        // Outer add sees BinaryApp/UnaryApp in arena (not IntLit) → not folded
        // Constant(6) + Constant(255) + Prim(Add) + Output = 4 nodes
        assert_eq!(graph.node_count(), 4);
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
            QuantumLevel::Q1,
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
                QuantumLevel::Q0,
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
                QuantumLevel::Q0,
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
            QuantumLevel::Q0,
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
            QuantumLevel::Q0,
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

    #[test]
    fn fold_binary_literals() {
        // add(3, 5) should constant-fold to a single Constant(8) node
        let mut arena = TermArena::new();
        let a = arena.alloc(TermKind::IntLit(3));
        let b = arena.alloc(TermKind::IntLit(5));
        let root = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: a,
            rhs: b,
        });
        let unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Folded: 1 Constant + 1 Output = 2 nodes (not 2 Constant + 1 Prim + 1 Output = 4)
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn fold_unary_literal() {
        // neg(1) should constant-fold to Constant(255) at Q0
        let mut arena = TermArena::new();
        let a = arena.alloc(TermKind::IntLit(1));
        let root = arena.alloc(TermKind::UnaryApp {
            op: PrimOp::Neg,
            arg: a,
        });
        let unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );

        let graph = lower_to_graph(&unit).unwrap();
        // Folded: 1 Constant + 1 Output = 2 nodes (not 1 Constant + 1 Prim + 1 Output = 3)
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn no_fold_when_not_both_literals() {
        // add(3, x) should NOT fold — x is a variable
        let mut arena = TermArena::new();
        let lit = arena.alloc(TermKind::IntLit(3));
        let var = arena.alloc(TermKind::Var(hologram_core::term::VarId(0)));
        let root = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: lit,
            rhs: var,
        });
        let mut unit = HoloCompileUnit::new(
            arena,
            root,
            QuantumLevel::Q0,
            6.0,
            &[VerificationDomain::Algebraic],
        );
        unit.bindings[0] = hologram_core::term::Binding {
            var: hologram_core::term::VarId(0),
            ty: hologram_core::term::TypeId::UNCONSTRAINED,
            rhs: hologram_core::term::TermId(0), // points to lit
        };
        unit.binding_count = 1;

        let graph = lower_to_graph(&unit).unwrap();
        // Not folded: Constant(3) + Constant(3 via binding) + Prim(Add) + Output = 4 nodes
        assert!(graph.node_count() >= 3);
    }
}
