//! Integration tests for the declarative compilation path:
//! parse → preflight → lower → compile → archive.

use hologram_compiler::compile_from_source;
use hologram_core::op::RingLevel;
use uor_foundation::enums::VerificationDomain;

#[test]
fn end_to_end_integer_literal() {
    let result = compile_from_source("42", RingLevel::Q0, 100.0, &[VerificationDomain::Algebraic]);
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
    let output = result.unwrap();
    assert!(!output.archive.is_empty());
}

#[test]
fn end_to_end_unary_application() {
    let result =
        compile_from_source("neg(42)", RingLevel::Q0, 100.0, &[VerificationDomain::Algebraic]);
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
    let output = result.unwrap();
    assert!(!output.archive.is_empty());
    assert!(output.stats.total_nodes > 0);
}

#[test]
fn end_to_end_binary_application() {
    let result = compile_from_source(
        "add(1, 2)",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}

#[test]
fn end_to_end_nested_expression() {
    let result = compile_from_source(
        "add(mul(2, 3), neg(1))",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}

#[test]
fn end_to_end_with_binding() {
    let result = compile_from_source(
        "let x : Q0 = 42 ; neg(x)",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}

#[test]
fn end_to_end_with_assertion() {
    let result = compile_from_source(
        "let x : Q0 = 0 ; assert neg(x) = x ; x",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}

#[test]
fn end_to_end_assertion_contradiction() {
    // neg(42) = 214 in Q0 ring (wrapping_neg), which != 42
    let result = compile_from_source(
        "let x : Q0 = 42 ; assert neg(x) = x ; x",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_err(), "should reject contradictory assertion");
}

#[test]
fn end_to_end_q1_quantum_literal() {
    let result = compile_from_source(
        "1000@Q1",
        RingLevel::Q1,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}

#[test]
fn end_to_end_rejects_insufficient_budget() {
    let result = compile_from_source(
        "42",
        RingLevel::Q0,
        1.0, // below Q0 minimum of 5.545
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_err());
}

#[test]
fn end_to_end_rejects_out_of_range_literal() {
    let result = compile_from_source(
        "256", // > 255, out of Q0 range
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_err());
}

#[test]
fn end_to_end_with_comments() {
    let result = compile_from_source(
        "-- this is a comment\nadd(1, 2) -- inline",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}

#[test]
fn end_to_end_lut_sigmoid() {
    let result = compile_from_source(
        "sigmoid(42)",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}

#[test]
fn end_to_end_lut_chain() {
    let result = compile_from_source(
        "relu(sigmoid(42))",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}

#[test]
fn end_to_end_lut_with_prim() {
    let result = compile_from_source(
        "sigmoid(neg(42))",
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_ok(), "compile_from_source failed: {:?}", result.err());
}
