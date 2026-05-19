//! Reference evaluator tests for Match / Try / Recurse / Unfold.

use hologram_ops::{EvalError, ReferenceEvaluator, ScalarEvaluatorU64};
use prism::vocabulary::WittLevel;
use uor_foundation::enforcement::{Term, TermArena, TermList};
use uor_foundation::pipeline::literal_u64;

/// Construct a `Term::Literal` from a `u64` at the given Witt level.
/// Mirrors the helper in `hologram_ops::emit::push_literal` so tests
/// stay decoupled from the upstream `TermValue` wire format.
fn lit(v: u64, level: WittLevel) -> Term {
    literal_u64(v, level)
}

#[test]
fn match_picks_first_matching_arm() {
    // scrutinee == 5; arms = [literal 3, literal 5, literal 99]; should pick 5
    // (which equals scrutinee). The actual semantics: pick first arm whose
    // value equals the scrutinee, default to last arm.
    let mut arena: TermArena<8> = TermArena::new();
    let lit5 = arena.push(lit(5, WittLevel::W8)).unwrap();
    let lit3 = arena.push(lit(3, WittLevel::W8)).unwrap();
    let lit5b = arena.push(lit(5, WittLevel::W8)).unwrap();
    let lit99 = arena.push(lit(99, WittLevel::W8)).unwrap();
    let _ = (lit3, lit5b, lit99);
    let arms_start = lit3;
    let m = arena
        .push(Term::Match {
            scrutinee_index: lit5,
            arms: TermList {
                start: arms_start,
                len: 3,
            },
        })
        .unwrap();
    let v = ScalarEvaluatorU64::evaluate(&arena, m, &[]).unwrap();
    assert_eq!(v, 5);
}

#[test]
fn match_default_when_no_arm_matches() {
    let mut arena: TermArena<8> = TermArena::new();
    let scrut = arena.push(lit(100, WittLevel::W8)).unwrap();
    let arm0 = arena.push(lit(1, WittLevel::W8)).unwrap();
    let _arm1 = arena.push(lit(2, WittLevel::W8)).unwrap();
    let _arm2 = arena.push(lit(42, WittLevel::W8)).unwrap();
    let m = arena
        .push(Term::Match {
            scrutinee_index: scrut,
            arms: TermList {
                start: arm0,
                len: 3,
            },
        })
        .unwrap();
    let v = ScalarEvaluatorU64::evaluate(&arena, m, &[]).unwrap();
    assert_eq!(v, 42); // default = last arm
}

#[test]
fn recurse_descends_to_zero() {
    // measure starts at 5, base = 0, step = 1; recurse runs 5 times
    // accumulating step values: 0 + 1 + 1 + 1 + 1 + 1 = 5.
    let mut arena: TermArena<8> = TermArena::new();
    let measure = arena.push(lit(5, WittLevel::W8)).unwrap();
    let base = arena.push(lit(0, WittLevel::W8)).unwrap();
    let step = arena.push(lit(1, WittLevel::W8)).unwrap();
    let r = arena
        .push(Term::Recurse {
            measure_index: measure,
            base_index: base,
            step_index: step,
        })
        .unwrap();
    let v = ScalarEvaluatorU64::evaluate(&arena, r, &[]).unwrap();
    assert_eq!(v, 5);
}

#[test]
fn try_falls_back_on_error() {
    // body: divide by zero (we'll fake by using an unknown variable)
    // handler: literal 7
    let mut arena: TermArena<8> = TermArena::new();
    let body = arena.push(Term::Variable { name_index: 99 }).unwrap();
    let handler = arena.push(lit(7, WittLevel::W8)).unwrap();
    let t = arena
        .push(Term::Try {
            body_index: body,
            handler_index: handler,
        })
        .unwrap();
    let v = ScalarEvaluatorU64::evaluate(&arena, t, &[]).unwrap();
    assert_eq!(v, 7); // body errored; handler took over
}

#[test]
fn try_succeeds_when_body_works() {
    let mut arena: TermArena<8> = TermArena::new();
    let body = arena.push(lit(42, WittLevel::W8)).unwrap();
    let handler = arena.push(lit(7, WittLevel::W8)).unwrap();
    let t = arena
        .push(Term::Try {
            body_index: body,
            handler_index: handler,
        })
        .unwrap();
    let v = ScalarEvaluatorU64::evaluate(&arena, t, &[]).unwrap();
    assert_eq!(v, 42);
}

#[test]
fn recurse_invalid_measure_terminates_via_safety_ceiling() {
    // Even with a large measure, the safety ceiling MAX_RECURSE_ITERATIONS
    // ensures the evaluator returns rather than looping forever.
    let mut arena: TermArena<8> = TermArena::new();
    let measure = arena.push(lit(u64::MAX, WittLevel::new(64))).unwrap();
    let base = arena.push(lit(0, WittLevel::new(64))).unwrap();
    let step = arena.push(lit(1, WittLevel::new(64))).unwrap();
    let r = arena
        .push(Term::Recurse {
            measure_index: measure,
            base_index: base,
            step_index: step,
        })
        .unwrap();
    let result = ScalarEvaluatorU64::evaluate(&arena, r, &[]);
    // Either Ok with a bounded result, or a NonTerminating error — both
    // are acceptable. The key property: it terminates.
    match result {
        Ok(_) | Err(EvalError::NonTerminating) => {}
        Err(other) => panic!("unexpected error: {:?}", other),
    }
}
