//! Integration tests for the declarative compilation path:
//! parse → preflight → lower → compile → archive.

use hologram_compiler::compile_from_source;
use hologram_foundation::enums::VerificationDomain;
use hologram_foundation::WittLevel;

#[test]
fn end_to_end_integer_literal() {
    let result = compile_from_source("42", WittLevel::W8, 100.0, &[VerificationDomain::Algebraic]);
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert!(!output.archive.is_empty());
}

#[test]
fn end_to_end_unary_application() {
    let result = compile_from_source(
        "neg(42)",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert!(!output.archive.is_empty());
    assert!(output.stats.total_nodes > 0);
}

#[test]
fn end_to_end_binary_application() {
    let result = compile_from_source(
        "add(1, 2)",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}

#[test]
fn end_to_end_nested_expression() {
    let result = compile_from_source(
        "add(mul(2, 3), neg(1))",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}

#[test]
fn end_to_end_with_binding() {
    let result = compile_from_source(
        "let x : W8 = 42 ; neg(x)",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}

#[test]
fn end_to_end_with_assertion() {
    let result = compile_from_source(
        "let x : W8 = 0 ; assert neg(x) = x ; x",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}

#[test]
fn end_to_end_assertion_contradiction() {
    // neg(42) = 214 in the W8 ring (wrapping_neg), which != 42
    let result = compile_from_source(
        "let x : W8 = 42 ; assert neg(x) = x ; x",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_err(), "should reject contradictory assertion");
}

#[test]
fn end_to_end_w16_witt_literal() {
    let result = compile_from_source(
        "1000@W16",
        WittLevel::W16,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}

#[test]
fn end_to_end_legacy_q_syntax_still_parses() {
    // Phase 12 added W8/W16/W24/W32 spellings to the term-language
    // syntax but kept Q0/Q1/Q2/Q3 working as a compatibility shim. Both
    // forms must still produce a valid compile.
    let q_form = compile_from_source(
        "let x : Q0 = 42 ; neg(x)",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        q_form.is_ok(),
        "legacy Q0 syntax must still parse: {:?}",
        q_form.err()
    );

    let w_form = compile_from_source(
        "let x : W8 = 42 ; neg(x)",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(w_form.is_ok(), "W8 syntax must parse: {:?}", w_form.err());

    // Both forms must compile to the same archive bytes (same TypeId
    // encoding, same RingLevel, same downstream pipeline).
    let q_archive = q_form.unwrap().archive;
    let w_archive = w_form.unwrap().archive;
    assert_eq!(
        q_archive, w_archive,
        "Q0 and W8 must produce byte-identical archives"
    );

    // And the legacy form for the W16 quantum literal works too.
    let q1_lit = compile_from_source(
        "1000@Q1",
        WittLevel::W16,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(q1_lit.is_ok(), "legacy @Q1 literal must parse");
}

#[test]
fn end_to_end_rejects_insufficient_budget() {
    let result = compile_from_source(
        "42",
        WittLevel::W8,
        1.0, // below Q0 minimum of 5.545
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_err());
}

#[test]
fn end_to_end_rejects_out_of_range_literal() {
    let result = compile_from_source(
        "256", // > 255, out of Q0 range
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(result.is_err());
}

#[test]
fn end_to_end_with_comments() {
    let result = compile_from_source(
        "-- this is a comment\nadd(1, 2) -- inline",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}

#[test]
fn end_to_end_lut_sigmoid() {
    let result = compile_from_source(
        "sigmoid(42)",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}

#[test]
fn end_to_end_lut_chain() {
    let result = compile_from_source(
        "relu(sigmoid(42))",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}

#[test]
fn end_to_end_lut_with_prim() {
    let result = compile_from_source(
        "sigmoid(neg(42))",
        WittLevel::W8,
        100.0,
        &[VerificationDomain::Algebraic],
    );
    assert!(
        result.is_ok(),
        "compile_from_source failed: {:?}",
        result.err()
    );
}
