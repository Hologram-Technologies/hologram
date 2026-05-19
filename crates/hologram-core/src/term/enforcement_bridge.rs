//! Bridge between `uor_foundation::enforcement::Term` (PRISM surface AST)
//! and hologram's `TermKind` (lowered IR).
//!
//! Per PRISM Section 1, `enforcement::Term` is the source language AST produced
//! by the `uor!` macro at compile time. Hologram's `TermKind` is the lowered IR
//! consumed by the cascade pipeline. This module provides:
//!
//! - **Forward bridge**: `enforcement::Term` -> `TermKind` via [`lower_enforcement_node`]
//! - **Reverse bridge**: `TermKind` -> `enforcement::Term` via [`to_enforcement_term`]
//! - **Arena conversion**: full arena conversion via [`convert_enforcement_arena`]
//!
//! All conversions are `#[inline]`, zero-allocation per node, and `#![no_std]`-compatible.

extern crate alloc;

use crate::op::{PrimOp, RingLevel};
use crate::term::{Assertion, Binding, TermArena, TermId, TermKind, TypeId, VarId};

use uor_foundation::enforcement::{Term as EnfTerm, TermArena as EnfArena, TermList};
use uor_foundation::{PrimitiveOp, QuantumLevel};

/// Error during enforcement term lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerError {
    /// Quantum level beyond Q3 (unsupported by hologram).
    UnsupportedLevel(u32),
    /// Index exceeds u16::MAX (hologram VarId/TypeId limit).
    IndexOverflow(u32),
    /// Operator arity not 1 or 2.
    UnsupportedArity(u32),
    /// Recurse/Unfold without bounded depth.
    UnboundedRecursion,
    /// Destination arena is full.
    ArenaFull,
    /// Referenced node not found in source arena.
    MissingNode(u32),
    /// Referenced node not yet lowered (forward reference).
    NotYetLowered(u32),
}

// ── Forward bridge: enforcement::Term → TermKind ────────────────────────────

/// Lower a single enforcement::Term node into the hologram TermArena.
///
/// `node_map[i]` caches the hologram `TermId` for enforcement node `i`.
/// Returns the hologram `TermId` for the lowered node.
///
/// O(1) per node (excluding Match/Recurse/Unfold which are O(arms) or O(depth)).
/// Zero allocation: writes directly into the pre-existing `TermArena`.
#[inline]
pub fn lower_enforcement_node<const CAP: usize>(
    index: u32,
    src: &EnfArena<CAP>,
    dst: &mut TermArena,
    node_map: &mut [Option<TermId>],
) -> Result<TermId, LowerError> {
    // Memoization: return cached result if already lowered
    if let Some(cached) = node_map[index as usize] {
        return Ok(cached);
    }

    let term = src.get(index).ok_or(LowerError::MissingNode(index))?;
    let kind = match term {
        EnfTerm::Literal { value, level } => {
            let rl = RingLevel::from_quantum(*level)
                .ok_or(LowerError::UnsupportedLevel(level.index()))?;
            TermKind::QuantumLit {
                level: rl,
                value: *value as u32,
            }
        }

        EnfTerm::Variable { name_index } => {
            if *name_index > u16::MAX as u32 {
                return Err(LowerError::IndexOverflow(*name_index));
            }
            TermKind::Var(VarId(*name_index as u16))
        }

        EnfTerm::Application { operator, args } => {
            let op = prim_op_from_foundation(*operator);
            match args.len {
                1 => {
                    let arg = resolve_mapped(args.start, node_map)?;
                    TermKind::UnaryApp { op, arg }
                }
                2 => {
                    let lhs = resolve_mapped(args.start, node_map)?;
                    let rhs = resolve_mapped(args.start + 1, node_map)?;
                    TermKind::BinaryApp { op, lhs, rhs }
                }
                n => return Err(LowerError::UnsupportedArity(n)),
            }
        }

        EnfTerm::Lift {
            operand_index,
            target,
        } => {
            let rl = RingLevel::from_quantum(*target)
                .ok_or(LowerError::UnsupportedLevel(target.index()))?;
            let arg = resolve_mapped(*operand_index, node_map)?;
            TermKind::RingUnaryApp {
                op: PrimOp::Succ,
                level: rl,
                arg,
            }
        }

        EnfTerm::Project {
            operand_index,
            target,
        } => {
            let rl = RingLevel::from_quantum(*target)
                .ok_or(LowerError::UnsupportedLevel(target.index()))?;
            let arg = resolve_mapped(*operand_index, node_map)?;
            TermKind::RingUnaryApp {
                op: PrimOp::Pred,
                level: rl,
                arg,
            }
        }

        EnfTerm::Match {
            scrutinee_index,
            arms,
        } => {
            // Ring-algebraic conditional selection chain.
            // Arms are consecutive (pattern, body) pairs.
            let scrutinee = resolve_mapped(*scrutinee_index, node_map)?;
            if arms.len == 0 || arms.len % 2 != 0 {
                TermKind::Passthrough(scrutinee)
            } else {
                // Last body is the default accumulator
                let last_body_idx = arms.start + arms.len - 1;
                let mut acc = lower_enforcement_node(last_body_idx, src, dst, node_map)?;
                let num_arms = arms.len / 2;
                // Process arms right-to-left: each arm selects body when
                // scrutinee == pattern via XOR-to-zero + mask
                for i in (0..num_arms).rev() {
                    let pat_idx = arms.start + i * 2;
                    let body_idx = arms.start + i * 2 + 1;
                    let pat = lower_enforcement_node(pat_idx, src, dst, node_map)?;
                    let body = lower_enforcement_node(body_idx, src, dst, node_map)?;
                    let xor = dst.alloc(TermKind::BinaryApp {
                        op: PrimOp::Xor,
                        lhs: scrutinee,
                        rhs: pat,
                    });
                    let mask = dst.alloc(TermKind::UnaryApp {
                        op: PrimOp::Bnot,
                        arg: xor,
                    });
                    let selected = dst.alloc(TermKind::BinaryApp {
                        op: PrimOp::And,
                        lhs: body,
                        rhs: mask,
                    });
                    let kept = dst.alloc(TermKind::BinaryApp {
                        op: PrimOp::And,
                        lhs: acc,
                        rhs: xor,
                    });
                    acc = dst.alloc(TermKind::BinaryApp {
                        op: PrimOp::Or,
                        lhs: selected,
                        rhs: kept,
                    });
                }
                node_map[index as usize] = Some(acc);
                return Ok(acc);
            }
        }

        EnfTerm::Recurse {
            measure_index,
            base_index,
            step_index,
        } => {
            // Bounded unrolling: measure must be a literal <= 256.
            let measure_term = src
                .get(*measure_index)
                .ok_or(LowerError::MissingNode(*measure_index))?;
            let depth = match measure_term {
                EnfTerm::Literal { value, .. } if *value <= 256 => *value as usize,
                _ => return Err(LowerError::UnboundedRecursion),
            };
            let mut acc = lower_enforcement_node(*base_index, src, dst, node_map)?;
            let step_node = src
                .get(*step_index)
                .ok_or(LowerError::MissingNode(*step_index))?;
            let step_op = match step_node {
                EnfTerm::Application { operator, .. } => prim_op_from_foundation(*operator),
                _ => PrimOp::Succ,
            };
            for _ in 0..depth {
                acc = dst.alloc(TermKind::UnaryApp {
                    op: step_op,
                    arg: acc,
                });
            }
            node_map[index as usize] = Some(acc);
            return Ok(acc);
        }

        EnfTerm::Unfold {
            seed_index,
            step_index,
        } => {
            // Bounded stream: fixed truncation depth of 8.
            let mut acc = lower_enforcement_node(*seed_index, src, dst, node_map)?;
            let step_node = src
                .get(*step_index)
                .ok_or(LowerError::MissingNode(*step_index))?;
            let step_op = match step_node {
                EnfTerm::Application { operator, .. } => prim_op_from_foundation(*operator),
                _ => PrimOp::Succ,
            };
            for _ in 0..8 {
                acc = dst.alloc(TermKind::UnaryApp {
                    op: step_op,
                    arg: acc,
                });
            }
            node_map[index as usize] = Some(acc);
            return Ok(acc);
        }

        EnfTerm::Try {
            body_index,
            handler_index,
        } => {
            // Lower body directly; record handler in arena side table.
            let body = lower_enforcement_node(*body_index, src, dst, node_map)?;
            let handler = lower_enforcement_node(*handler_index, src, dst, node_map)?;
            dst.register_error_handler(body, handler);
            node_map[index as usize] = Some(body);
            return Ok(body);
        }
    };

    let id = dst.alloc(kind);
    node_map[index as usize] = Some(id);
    Ok(id)
}

/// Convert an entire enforcement `TermArena` into a hologram `TermArena`.
///
/// O(n) where n = `src.len()`. The `node_map` is heap-allocated as a `Vec`
/// (the only allocation besides the destination arena).
pub fn convert_enforcement_arena<const CAP: usize>(
    src: &EnfArena<CAP>,
    root_index: u32,
) -> Result<(TermArena, TermId), LowerError> {
    let len = src.len() as usize;
    let mut dst = TermArena::with_capacity(len);
    let mut node_map = alloc::vec![None; len];

    // Walk arena bottom-up (enforcement arenas are append-only: children < parents)
    for i in 0..len as u32 {
        lower_enforcement_node(i, src, &mut dst, &mut node_map)?;
    }

    let root = node_map[root_index as usize].ok_or(LowerError::MissingNode(root_index))?;
    Ok((dst, root))
}

// ── Reverse bridge: TermKind → enforcement::Term ────────────────────────────

/// Convert a hologram `TermKind` node to an `enforcement::Term`.
///
/// Returns `None` for compiler-IR-only variants that have no surface equivalent
/// (FloatApp, LutApp, FusedViewRef, Constant, GraphInput, GraphOutput, Passthrough).
///
/// O(1) per node. Zero allocation.
#[inline]
pub fn to_enforcement_term(kind: &TermKind, level: QuantumLevel) -> Option<EnfTerm> {
    match *kind {
        TermKind::IntLit(v) => Some(EnfTerm::Literal {
            value: v as u64,
            level,
        }),
        TermKind::BrailleLit(v) => Some(EnfTerm::Literal {
            value: v as u64,
            level: QuantumLevel::Q0,
        }),
        TermKind::QuantumLit {
            level: rl,
            value: v,
        } => Some(EnfTerm::Literal {
            value: v as u64,
            level: rl.into(),
        }),
        TermKind::UnaryApp { op, arg } => Some(EnfTerm::Application {
            operator: op.to_foundation(),
            args: TermList {
                start: arg.0,
                len: 1,
            },
        }),
        TermKind::BinaryApp { op, lhs, .. } => Some(EnfTerm::Application {
            operator: op.to_foundation(),
            args: TermList {
                start: lhs.0,
                len: 2,
            },
        }),
        TermKind::Var(vid) => Some(EnfTerm::Variable {
            name_index: vid.0 as u32,
        }),
        TermKind::RingUnaryApp { op, level: rl, arg } => match op {
            PrimOp::Succ => Some(EnfTerm::Lift {
                operand_index: arg.0,
                target: rl.into(),
            }),
            PrimOp::Pred => Some(EnfTerm::Project {
                operand_index: arg.0,
                target: rl.into(),
            }),
            _ => Some(EnfTerm::Application {
                operator: op.to_foundation(),
                args: TermList {
                    start: arg.0,
                    len: 1,
                },
            }),
        },
        // Compiler-IR variants have no surface equivalent
        TermKind::FloatApp { .. }
        | TermKind::LutApp { .. }
        | TermKind::RingBinaryApp { .. }
        | TermKind::Constant(_)
        | TermKind::GraphInput(_)
        | TermKind::GraphOutput(_)
        | TermKind::FusedViewRef(_)
        | TermKind::Passthrough(_) => None,
    }
}

/// Convert a hologram `TermArena` to a `Vec<enforcement::Term>` for builder validation.
///
/// Compiler-IR nodes that have no surface equivalent are mapped to
/// `Term::Literal { value: 0, level }` as inert placeholders.
///
/// O(n) where n = arena.len(). Allocates one `Vec<Term>`.
pub fn arena_to_enforcement_terms(
    arena: &TermArena,
    level: QuantumLevel,
) -> alloc::vec::Vec<EnfTerm> {
    let mut terms = alloc::vec::Vec::with_capacity(arena.len() as usize);
    for i in 0..arena.len() {
        let node = arena.get(TermId(i));
        let term =
            to_enforcement_term(&node.kind, level).unwrap_or(EnfTerm::Literal { value: 0, level });
        terms.push(term);
    }
    terms
}

// ── Binding / Assertion bridge ──────────────────────────────────────────────

/// Convert an `enforcement::Binding` to a hologram `Binding`.
///
/// Requires `node_map` from a prior arena conversion.
#[inline]
pub fn convert_binding(
    enf: &uor_foundation::enforcement::Binding,
    node_map: &[Option<TermId>],
) -> Result<Binding, LowerError> {
    if enf.name_index > u16::MAX as u32 {
        return Err(LowerError::IndexOverflow(enf.name_index));
    }
    let rhs = resolve_mapped(enf.value_index, node_map)?;
    let ty = if enf.type_index <= u16::MAX as u32 {
        TypeId(enf.type_index as u16)
    } else {
        TypeId::UNCONSTRAINED
    };
    Ok(Binding {
        var: VarId(enf.name_index as u16),
        ty,
        rhs,
    })
}

/// Convert an `enforcement::Assertion` to a hologram `Assertion`.
#[inline]
pub fn convert_assertion(
    enf: &uor_foundation::enforcement::Assertion,
    node_map: &[Option<TermId>],
) -> Result<Assertion, LowerError> {
    let lhs = resolve_mapped(enf.lhs_index, node_map)?;
    let rhs = resolve_mapped(enf.rhs_index, node_map)?;
    Ok(Assertion {
        lhs,
        rhs,
        canonical: false,
    })
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Look up a node_map entry, returning `LowerError::NotYetLowered` if missing.
#[inline]
fn resolve_mapped(index: u32, node_map: &[Option<TermId>]) -> Result<TermId, LowerError> {
    node_map
        .get(index as usize)
        .copied()
        .flatten()
        .ok_or(LowerError::NotYetLowered(index))
}

/// Map `PrimitiveOp` (foundation) to `PrimOp` (hologram).
/// Uses the existing `PrimOp::from_foundation()` method.
#[inline]
fn prim_op_from_foundation(op: PrimitiveOp) -> PrimOp {
    PrimOp::from_foundation(op)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforcement_literal_converts() {
        let mut src = EnfArena::<4>::new();
        let i = src
            .push(EnfTerm::Literal {
                value: 42,
                level: QuantumLevel::Q0,
            })
            .unwrap();

        let (arena, root) = convert_enforcement_arena(&src, i).unwrap();
        assert_eq!(arena.len(), 1);
        let node = arena.get(root);
        assert!(matches!(
            node.kind,
            TermKind::QuantumLit {
                level: RingLevel::Q0,
                value: 42
            }
        ));
    }

    #[test]
    fn enforcement_add_converts() {
        let mut src = EnfArena::<4>::new();
        let i3 = src
            .push(EnfTerm::Literal {
                value: 3,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        // i5: slot consumed implicitly via `len: 2` in the Application below.
        let _i5 = src
            .push(EnfTerm::Literal {
                value: 5,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let root = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Add,
                args: TermList { start: i3, len: 2 },
            })
            .unwrap();

        let (arena, holo_root) = convert_enforcement_arena(&src, root).unwrap();
        assert_eq!(arena.len(), 3);
        let node = arena.get(holo_root);
        assert!(matches!(
            node.kind,
            TermKind::BinaryApp {
                op: PrimOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn enforcement_variable_converts() {
        let mut src = EnfArena::<4>::new();
        let i = src.push(EnfTerm::Variable { name_index: 7 }).unwrap();

        let (arena, root) = convert_enforcement_arena(&src, i).unwrap();
        let node = arena.get(root);
        assert!(matches!(node.kind, TermKind::Var(VarId(7))));
    }

    #[test]
    fn enforcement_lift_converts() {
        let mut src = EnfArena::<4>::new();
        let lit = src
            .push(EnfTerm::Literal {
                value: 1,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let lift = src
            .push(EnfTerm::Lift {
                operand_index: lit,
                target: QuantumLevel::Q3,
            })
            .unwrap();

        let (arena, root) = convert_enforcement_arena(&src, lift).unwrap();
        let node = arena.get(root);
        assert!(matches!(
            node.kind,
            TermKind::RingUnaryApp {
                op: PrimOp::Succ,
                level: RingLevel::Q3,
                ..
            }
        ));
    }

    #[test]
    fn enforcement_unary_neg_converts() {
        let mut src = EnfArena::<4>::new();
        let lit = src
            .push(EnfTerm::Literal {
                value: 10,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let neg = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Neg,
                args: TermList { start: lit, len: 1 },
            })
            .unwrap();

        let (arena, root) = convert_enforcement_arena(&src, neg).unwrap();
        let node = arena.get(root);
        assert!(matches!(
            node.kind,
            TermKind::UnaryApp {
                op: PrimOp::Neg,
                ..
            }
        ));
    }

    #[test]
    fn enforcement_unsupported_level_errors() {
        let mut src = EnfArena::<4>::new();
        let i = src
            .push(EnfTerm::Literal {
                value: 1,
                level: QuantumLevel::new(10), // Beyond Q3
            })
            .unwrap();

        let result = convert_enforcement_arena(&src, i);
        assert!(matches!(result, Err(LowerError::UnsupportedLevel(10))));
    }

    #[test]
    fn enforcement_unsupported_arity_errors() {
        let mut src = EnfArena::<4>::new();
        let _i = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Add,
                args: TermList { start: 0, len: 0 }, // arity 0
            })
            .unwrap();

        let result = convert_enforcement_arena(&src, 0);
        assert!(matches!(result, Err(LowerError::UnsupportedArity(0))));
    }

    #[test]
    fn reverse_bridge_literal_roundtrip() {
        let kind = TermKind::QuantumLit {
            level: RingLevel::Q1,
            value: 100,
        };
        let enf = to_enforcement_term(&kind, QuantumLevel::Q1).unwrap();
        assert!(matches!(
            enf,
            EnfTerm::Literal {
                value: 100,
                level
            } if level == QuantumLevel::Q1
        ));
    }

    #[test]
    fn reverse_bridge_ir_returns_none() {
        let kind = TermKind::FloatApp {
            op: crate::term::FloatOpRef(0),
            arg0: TermId(0),
            arg1: TermId(1),
        };
        assert!(to_enforcement_term(&kind, QuantumLevel::Q0).is_none());
    }

    #[test]
    fn arena_roundtrip_preserves_structure() {
        // Build enforcement arena: neg(add(1, 2))
        let mut src = EnfArena::<8>::new();
        let i1 = src
            .push(EnfTerm::Literal {
                value: 1,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        // i2: slot consumed implicitly via `len: 2` in the Application below.
        let _i2 = src
            .push(EnfTerm::Literal {
                value: 2,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let add = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Add,
                args: TermList { start: i1, len: 2 },
            })
            .unwrap();
        let neg = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Neg,
                args: TermList { start: add, len: 1 },
            })
            .unwrap();

        // Forward: enforcement → hologram
        let (arena, root) = convert_enforcement_arena(&src, neg).unwrap();
        assert_eq!(arena.len(), 4);

        // Verify structure
        let root_node = arena.get(root);
        assert!(matches!(
            root_node.kind,
            TermKind::UnaryApp {
                op: PrimOp::Neg,
                ..
            }
        ));

        // Reverse: hologram → enforcement terms
        let terms = arena_to_enforcement_terms(&arena, QuantumLevel::Q0);
        assert_eq!(terms.len(), 4);
    }

    #[test]
    fn conversion_performance() {
        // Performance contract: 100K arena conversions (4 nodes each) < 200ms
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            let mut src = EnfArena::<4>::new();
            let i1 = src
                .push(EnfTerm::Literal {
                    value: 1,
                    level: QuantumLevel::Q0,
                })
                .unwrap();
            // i2: slot consumed implicitly via `len: 2` in the Application below.
            let _i2 = src
                .push(EnfTerm::Literal {
                    value: 2,
                    level: QuantumLevel::Q0,
                })
                .unwrap();
            let root = src
                .push(EnfTerm::Application {
                    operator: PrimitiveOp::Add,
                    args: TermList { start: i1, len: 2 },
                })
                .unwrap();
            let _ = convert_enforcement_arena(&src, root);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "100K conversions took {}ms (target < 200ms)",
            elapsed.as_millis()
        );
    }

    #[test]
    fn enforcement_match_two_arms() {
        // match x { 0 => 10; 1 => 20; }
        let mut src = EnfArena::<16>::new();
        let scrutinee = src
            .push(EnfTerm::Literal {
                value: 0,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let pat0 = src
            .push(EnfTerm::Literal {
                value: 0,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        // body0 / pat1 / body1: arms consumed implicitly via `len: 4` on the Match below.
        let _body0 = src
            .push(EnfTerm::Literal {
                value: 10,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let _pat1 = src
            .push(EnfTerm::Literal {
                value: 1,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let _body1 = src
            .push(EnfTerm::Literal {
                value: 20,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let root = src
            .push(EnfTerm::Match {
                scrutinee_index: scrutinee,
                arms: TermList {
                    start: pat0,
                    len: 4,
                },
            })
            .unwrap();

        let (arena, holo_root) = convert_enforcement_arena(&src, root).unwrap();
        // Should produce conditional chain nodes, not a Passthrough
        let node = arena.get(holo_root);
        assert!(
            !matches!(node.kind, TermKind::Passthrough(_)),
            "match should not produce Passthrough"
        );
        // Chain should contain OR as final combiner
        assert!(matches!(
            node.kind,
            TermKind::BinaryApp { op: PrimOp::Or, .. }
        ));
    }

    #[test]
    fn enforcement_recurse_depth_3() {
        // recurse(3, 0, succ) → succ(succ(succ(0)))
        let mut src = EnfArena::<8>::new();
        let measure = src
            .push(EnfTerm::Literal {
                value: 3,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let base = src
            .push(EnfTerm::Literal {
                value: 0,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let step = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Succ,
                args: TermList {
                    start: base,
                    len: 1,
                },
            })
            .unwrap();
        let root = src
            .push(EnfTerm::Recurse {
                measure_index: measure,
                base_index: base,
                step_index: step,
            })
            .unwrap();

        let (arena, holo_root) = convert_enforcement_arena(&src, root).unwrap();
        // Should have 1 literal + 3 UnaryApp(Succ) = 4 nodes
        let node = arena.get(holo_root);
        assert!(matches!(
            node.kind,
            TermKind::UnaryApp {
                op: PrimOp::Succ,
                ..
            }
        ));
    }

    #[test]
    fn enforcement_recurse_unbounded_errors() {
        // recurse(x_variable, 0, succ) should fail: non-literal measure
        let mut src = EnfArena::<8>::new();
        let measure = src.push(EnfTerm::Variable { name_index: 0 }).unwrap();
        let base = src
            .push(EnfTerm::Literal {
                value: 0,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let step = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Succ,
                args: TermList {
                    start: base,
                    len: 1,
                },
            })
            .unwrap();
        let root = src
            .push(EnfTerm::Recurse {
                measure_index: measure,
                base_index: base,
                step_index: step,
            })
            .unwrap();

        let result = convert_enforcement_arena(&src, root);
        assert!(matches!(result, Err(LowerError::UnboundedRecursion)));
    }

    #[test]
    fn enforcement_unfold_8_steps() {
        // unfold(0, succ) → 9 nodes (1 literal + 8 UnaryApp)
        let mut src = EnfArena::<16>::new();
        let seed = src
            .push(EnfTerm::Literal {
                value: 0,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let step = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Succ,
                args: TermList {
                    start: seed,
                    len: 1,
                },
            })
            .unwrap();
        let root = src
            .push(EnfTerm::Unfold {
                seed_index: seed,
                step_index: step,
            })
            .unwrap();

        let (arena, holo_root) = convert_enforcement_arena(&src, root).unwrap();
        let node = arena.get(holo_root);
        assert!(matches!(
            node.kind,
            TermKind::UnaryApp {
                op: PrimOp::Succ,
                ..
            }
        ));
        // Count nodes: seed literal (1) + 8 UnaryApp = 9 total
        // (plus the original enforcement nodes lowered)
        assert!(arena.len() >= 9);
    }

    #[test]
    fn enforcement_try_records_handler() {
        // try { neg(1) } catch { 0 }
        let mut src = EnfArena::<8>::new();
        let lit1 = src
            .push(EnfTerm::Literal {
                value: 1,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let body = src
            .push(EnfTerm::Application {
                operator: PrimitiveOp::Neg,
                args: TermList {
                    start: lit1,
                    len: 1,
                },
            })
            .unwrap();
        let handler = src
            .push(EnfTerm::Literal {
                value: 0,
                level: QuantumLevel::Q0,
            })
            .unwrap();
        let root = src
            .push(EnfTerm::Try {
                body_index: body,
                handler_index: handler,
            })
            .unwrap();

        let (arena, holo_root) = convert_enforcement_arena(&src, root).unwrap();
        // Body should be the result (not Passthrough)
        let node = arena.get(holo_root);
        assert!(
            !matches!(node.kind, TermKind::Passthrough(_)),
            "try should return body directly, not Passthrough"
        );
        // Handler should be recorded in the error handler table
        let handler_id = arena.error_handler_for(holo_root);
        assert!(
            handler_id.is_some(),
            "error handler should be registered for try body"
        );
    }
}
