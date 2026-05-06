//! Spec VII.3 reference evaluator round-trip:
//! a Term tree built by `direct::*` op markers evaluates to the
//! corresponding scalar PrimitiveOp result.

use hologram_ops::{ScalarEvaluatorU64, ReferenceEvaluator};
use hologram_ops::direct::{NegOp, AddOp, MulOp, AndOp, OrOp, XorOp, SuccOp, PredOp};
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::enforcement::{TermArena, Term, TermList};

fn make_arena_2_args(a: u64, b: u64) -> (TermArena<8>, u32, u32) {
    let mut arena: TermArena<8> = TermArena::new();
    let av = arena.push(Term::Variable { name_index: 0 }).unwrap();
    let bv = arena.push(Term::Variable { name_index: 1 }).unwrap();
    let _ = (a, b);
    let _ = bv;
    (arena, av, bv)
}

fn eval_unary(prim: PrimitiveOp, x: u64) -> u64 {
    let mut arena: TermArena<8> = TermArena::new();
    let v = arena.push(Term::Variable { name_index: 0 }).unwrap();
    let _root = arena.push(Term::Application {
        operator: prim,
        args: TermList { start: v, len: 1 },
    }).unwrap();
    let bindings = [x];
    ScalarEvaluatorU64::evaluate(&arena, _root, &bindings).unwrap()
}

fn eval_binary(prim: PrimitiveOp, a: u64, b: u64) -> u64 {
    let (mut arena, av, _bv) = make_arena_2_args(a, b);
    let _root = arena.push(Term::Application {
        operator: prim,
        args: TermList { start: av, len: 2 },
    }).unwrap();
    let bindings = [a, b];
    ScalarEvaluatorU64::evaluate(&arena, _root, &bindings).unwrap()
}

#[test]
fn neg_matches_wrapping_neg() {
    let _ = NegOp::PRIMITIVE;
    assert_eq!(eval_unary(PrimitiveOp::Neg, 7), 7u64.wrapping_neg());
    assert_eq!(eval_unary(PrimitiveOp::Neg, 0), 0);
}

#[test]
fn succ_pred_round_trip() {
    let _ = SuccOp::PRIMITIVE;
    let _ = PredOp::PRIMITIVE;
    for x in [0u64, 1, 100, u64::MAX] {
        let s = eval_unary(PrimitiveOp::Succ, x);
        let p = eval_unary(PrimitiveOp::Pred, s);
        assert_eq!(p, x);
    }
}

#[test]
fn add_is_commutative() {
    let _ = AddOp::PRIMITIVE;
    for (a, b) in [(0u64, 0), (1, 2), (123, 456), (u64::MAX, 1)] {
        assert_eq!(eval_binary(PrimitiveOp::Add, a, b), eval_binary(PrimitiveOp::Add, b, a));
    }
}

#[test]
fn mul_matches_wrapping_mul() {
    let _ = MulOp::PRIMITIVE;
    assert_eq!(eval_binary(PrimitiveOp::Mul, 3, 7), 21);
    assert_eq!(eval_binary(PrimitiveOp::Mul, 0xFF_FF_FF_FF, 2), 0xFF_FF_FF_FF_u64.wrapping_mul(2));
}

#[test]
fn and_or_xor_match_native() {
    let _ = AndOp::PRIMITIVE;
    let _ = OrOp::PRIMITIVE;
    let _ = XorOp::PRIMITIVE;
    let a = 0b1100u64;
    let b = 0b1010u64;
    assert_eq!(eval_binary(PrimitiveOp::And, a, b), a & b);
    assert_eq!(eval_binary(PrimitiveOp::Or,  a, b), a | b);
    assert_eq!(eval_binary(PrimitiveOp::Xor, a, b), a ^ b);
}

#[test]
fn lift_truncates_to_witt_level() {
    let mut arena: TermArena<8> = TermArena::new();
    let v = arena.push(Term::Variable { name_index: 0 }).unwrap();
    let lift = arena.push(Term::Lift { operand_index: v, target: WittLevel::W8 }).unwrap();
    let bindings = [0xFFFF_FFFF_FFFF_FFFFu64];
    let out = ScalarEvaluatorU64::evaluate(&arena, lift, &bindings).unwrap();
    assert_eq!(out, 0xFF);
}
