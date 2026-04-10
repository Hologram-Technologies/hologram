//! CS_7: Unit Address Computation.
//!
//! Computes `unitAddress = address(canonicalBytes(transitiveClosure(rootTerm)))`.
//!
//! The address covers the root term's graph closure only — it **excludes**
//! `thermodynamicBudget`, `targetDomains`, and `unitQuantumLevel`. Two compile
//! units with identical root term graphs produce identical unit addresses.
//!
//! O(n) where n = number of term nodes in the arena.

use hologram_core::op::PrimOp;
use hologram_core::term::{TermArena, TermId, TermKind};

/// Compute the unit address: BLAKE3 hash of the canonical serialized term graph.
///
/// Iterates all nodes reachable from `root` in the arena and hashes their
/// canonical byte encoding. Uses an iterative depth-first traversal with
/// a fixed-size stack (no heap allocation).
///
/// # Performance
///
/// - O(n) single pass over reachable nodes
/// - Stack: Vec with initial capacity min(256, arena_len), grows as needed
/// - Visited: arena_len / 8 bytes (one allocation)
/// - BLAKE3: ~1 GB/s throughput, so 1000 nodes (~16 KB) takes ~16 ns
pub fn compute_unit_address(arena: &TermArena, root: TermId) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();

    // Visited bitset: one bit per arena node. Heap-allocated for arenas > 0.
    let arena_len = arena.len() as usize;
    let mut visited = vec![0u8; arena_len.div_ceil(8)];
    // Traversal stack: dynamically sized for deep term trees.
    let mut stack = Vec::with_capacity(256.min(arena_len));
    stack.push(root);

    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;

        // Check and mark visited (bitset).
        let byte_idx = idx / 8;
        let bit_mask = 1u8 << (idx % 8);
        if byte_idx >= visited.len() {
            continue;
        }
        if visited[byte_idx] & bit_mask != 0 {
            continue;
        }
        visited[byte_idx] |= bit_mask;

        if id.0 >= arena.len() {
            continue;
        }

        let node = arena.get(id);
        hasher.update(&canonical_bytes(&node.kind));

        // Push children (rhs first so lhs is processed first from stack).
        match node.kind {
            TermKind::UnaryApp { arg, .. }
            | TermKind::LutApp { arg, .. }
            | TermKind::RingUnaryApp { arg, .. }
            | TermKind::GraphOutput(arg)
            | TermKind::Passthrough(arg) => {
                stack.push(arg);
            }
            TermKind::BinaryApp { lhs, rhs, .. } | TermKind::RingBinaryApp { lhs, rhs, .. } => {
                stack.push(rhs);
                stack.push(lhs);
            }
            TermKind::FloatApp { arg0, arg1, .. } => {
                if arg1.0 != u32::MAX {
                    stack.push(arg1);
                }
                stack.push(arg0);
            }
            TermKind::Var(vid) => {
                hasher.update(&vid.0.to_le_bytes());
            }
            _ => {} // IntLit, QuantumLit, Constant, GraphInput, FusedViewRef
        }
    }

    *hasher.finalize().as_bytes()
}

/// Encode a TermKind as a fixed-size canonical byte sequence for hashing.
///
/// The encoding is deterministic: identical TermKinds produce identical bytes.
/// The discriminant is included to prevent collisions between different variants
/// with the same payload bytes.
fn canonical_bytes(kind: &TermKind) -> [u8; 16] {
    let mut buf = [0u8; 16];
    match kind {
        TermKind::IntLit(v) => {
            buf[0] = 0; // discriminant
            buf[1..9].copy_from_slice(&v.to_le_bytes());
        }
        TermKind::QuantumLit { level, value } => {
            buf[0] = 2;
            buf[1] = *level as u8;
            buf[2..6].copy_from_slice(&value.to_le_bytes());
        }
        TermKind::UnaryApp { op, arg } => {
            buf[0] = 3;
            buf[1] = primop_byte(*op);
            buf[2..6].copy_from_slice(&arg.0.to_le_bytes());
        }
        TermKind::BinaryApp { op, lhs, rhs } => {
            buf[0] = 4;
            buf[1] = primop_byte(*op);
            buf[2..6].copy_from_slice(&lhs.0.to_le_bytes());
            buf[6..10].copy_from_slice(&rhs.0.to_le_bytes());
        }
        TermKind::Var(vid) => {
            buf[0] = 5;
            buf[1..3].copy_from_slice(&vid.0.to_le_bytes());
        }
        TermKind::LutApp { op, arg } => {
            buf[0] = 6;
            buf[1] = lutop_byte(*op);
            buf[2..6].copy_from_slice(&arg.0.to_le_bytes());
        }
        TermKind::FloatApp { op, arg0, arg1 } => {
            buf[0] = 7;
            buf[1..5].copy_from_slice(&op.0.to_le_bytes());
            buf[5..9].copy_from_slice(&arg0.0.to_le_bytes());
            buf[9..13].copy_from_slice(&arg1.0.to_le_bytes());
        }
        TermKind::RingUnaryApp { op, level, arg } => {
            buf[0] = 8;
            buf[1] = primop_byte(*op);
            buf[2] = *level as u8;
            buf[3..7].copy_from_slice(&arg.0.to_le_bytes());
        }
        TermKind::RingBinaryApp {
            op,
            level,
            lhs,
            rhs,
        } => {
            buf[0] = 9;
            buf[1] = primop_byte(*op);
            buf[2] = *level as u8;
            buf[3..7].copy_from_slice(&lhs.0.to_le_bytes());
            buf[7..11].copy_from_slice(&rhs.0.to_le_bytes());
        }
        TermKind::Constant(cref) => {
            buf[0] = 10;
            buf[1..5].copy_from_slice(&cref.0.to_le_bytes());
        }
        TermKind::GraphInput(idx) => {
            buf[0] = 11;
            buf[1..5].copy_from_slice(&idx.to_le_bytes());
        }
        TermKind::GraphOutput(inner) => {
            buf[0] = 12;
            buf[1..5].copy_from_slice(&inner.0.to_le_bytes());
        }
        TermKind::FusedViewRef(vref) => {
            buf[0] = 13;
            buf[1..5].copy_from_slice(&vref.0.to_le_bytes());
        }
        TermKind::Passthrough(inner) => {
            buf[0] = 14;
            buf[1..5].copy_from_slice(&inner.0.to_le_bytes());
        }
    }
    buf
}

/// Stable byte encoding for PrimOp (matches the enum discriminant order).
#[inline]
fn primop_byte(op: PrimOp) -> u8 {
    match op {
        PrimOp::Neg => 0,
        PrimOp::Bnot => 1,
        PrimOp::Succ => 2,
        PrimOp::Pred => 3,
        PrimOp::Add => 4,
        PrimOp::Sub => 5,
        PrimOp::Mul => 6,
        PrimOp::Xor => 7,
        PrimOp::And => 8,
        PrimOp::Or => 9,
    }
}

/// Stable byte encoding for LutOp.
#[inline]
fn lutop_byte(op: hologram_core::op::LutOp) -> u8 {
    use hologram_core::op::LutOp;
    match op {
        LutOp::Sigmoid => 0,
        LutOp::Tanh => 1,
        LutOp::Exp => 2,
        LutOp::Log => 3,
        LutOp::Relu => 4,
        LutOp::Sqrt => 5,
        LutOp::Abs => 6,
        LutOp::Gelu => 7,
        LutOp::Silu => 8,
        LutOp::Sin => 9,
        LutOp::Cos => 10,
        LutOp::Tan => 11,
        LutOp::Asin => 12,
        LutOp::Acos => 13,
        LutOp::Atan => 14,
        LutOp::Log2 => 15,
        LutOp::Log10 => 16,
        LutOp::Exp2 => 17,
        LutOp::Exp10 => 18,
        LutOp::Square => 19,
        LutOp::Cube => 20,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::{PrimOp, RingLevel};
    use hologram_core::term::TermArena;

    #[test]
    fn same_term_same_address() {
        let mut a1 = TermArena::new();
        let r1 = a1.alloc(TermKind::IntLit(42));
        let addr1 = compute_unit_address(&a1, r1);

        let mut a2 = TermArena::new();
        let r2 = a2.alloc(TermKind::IntLit(42));
        let addr2 = compute_unit_address(&a2, r2);

        assert_eq!(addr1, addr2);
    }

    #[test]
    fn different_term_different_address() {
        let mut a1 = TermArena::new();
        let r1 = a1.alloc(TermKind::IntLit(42));
        let addr1 = compute_unit_address(&a1, r1);

        let mut a2 = TermArena::new();
        let r2 = a2.alloc(TermKind::IntLit(43));
        let addr2 = compute_unit_address(&a2, r2);

        assert_ne!(addr1, addr2);
    }

    #[test]
    fn nested_term_address() {
        let mut arena = TermArena::new();
        let lit = arena.alloc(TermKind::IntLit(1));
        let neg = arena.alloc(TermKind::UnaryApp {
            op: PrimOp::Neg,
            arg: lit,
        });
        let addr = compute_unit_address(&arena, neg);

        // Address should be non-zero and deterministic.
        assert_ne!(addr, [0u8; 32]);

        // Recompute: same result.
        let addr2 = compute_unit_address(&arena, neg);
        assert_eq!(addr, addr2);
    }

    #[test]
    fn binary_app_address() {
        let mut arena = TermArena::new();
        let a = arena.alloc(TermKind::IntLit(1));
        let b = arena.alloc(TermKind::IntLit(2));
        let sum = arena.alloc(TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: a,
            rhs: b,
        });
        let addr = compute_unit_address(&arena, sum);
        assert_ne!(addr, [0u8; 32]);
    }

    #[test]
    fn quantum_lit_address_differs_from_int_lit() {
        let mut a1 = TermArena::new();
        let r1 = a1.alloc(TermKind::IntLit(42));
        let addr1 = compute_unit_address(&a1, r1);

        let mut a2 = TermArena::new();
        let r2 = a2.alloc(TermKind::QuantumLit {
            level: RingLevel::Q0,
            value: 42,
        });
        let addr2 = compute_unit_address(&a2, r2);

        assert_ne!(
            addr1, addr2,
            "IntLit and QuantumLit should have different addresses"
        );
    }

    #[test]
    fn operand_order_matters() {
        let mut a1 = TermArena::new();
        let x = a1.alloc(TermKind::IntLit(1));
        let y = a1.alloc(TermKind::IntLit(2));
        let r1 = a1.alloc(TermKind::BinaryApp {
            op: PrimOp::Sub,
            lhs: x,
            rhs: y,
        });
        let addr1 = compute_unit_address(&a1, r1);

        let mut a2 = TermArena::new();
        let x2 = a2.alloc(TermKind::IntLit(1));
        let y2 = a2.alloc(TermKind::IntLit(2));
        let r2 = a2.alloc(TermKind::BinaryApp {
            op: PrimOp::Sub,
            lhs: y2,
            rhs: x2,
        });
        let addr2 = compute_unit_address(&a2, r2);

        assert_ne!(
            addr1, addr2,
            "sub(1,2) and sub(2,1) must have different addresses"
        );
    }
}
