//! Parser for the UOR term language.
//!
//! Provides a zero-copy lexer and recursive descent parser that produces
//! an arena-allocated term graph. The parser handles the core subset of
//! the EBNF grammar: literals, applications (unary/binary with the 10
//! PrimOps), variables, let-bindings, assertions, and type declarations.
//!
//! # Example
//!
//! ```ignore
//! use hologram_compiler::term_parser;
//!
//! let unit = term_parser::parse("let x : Q0 = 42 ; neg(x)")?;
//! assert_eq!(unit.binding_count, 1);
//! ```

pub mod error;
pub mod lexer;
pub mod parser;

pub use error::ParseError;

use hologram_core::term::{Assertion, Binding, TermArena, TermId, TypeDecl};

/// The output of a successful parse: an arena-allocated term graph with
/// bindings, assertions, and type declarations.
#[derive(Debug)]
pub struct ParsedUnit {
    /// Arena holding all term nodes.
    pub arena: TermArena,
    /// The root term (last expression or last binding's rhs).
    pub root: TermId,
    /// Let-bindings in declaration order.
    pub bindings: Box<[Binding; 64]>,
    pub binding_count: u8,
    /// Assertions to verify.
    pub assertions: Box<[Assertion; 32]>,
    pub assertion_count: u8,
    /// Type declarations.
    pub type_decls: Box<[TypeDecl; 16]>,
    pub type_decl_count: u8,
}

/// Parse UOR term language source into a [`ParsedUnit`].
///
/// O(n) where n = source length. Zero-copy lexing, direct arena allocation.
pub fn parse(source: &str) -> Result<ParsedUnit, ParseError> {
    parser::Parser::new(source).parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::PrimOp;
    use hologram_core::term::TermKind;

    #[test]
    fn parse_integer_literal() {
        let unit = parse("42").unwrap();
        assert_eq!(unit.arena.get(unit.root).kind, TermKind::IntLit(42));
    }

    #[test]
    fn parse_unary_application() {
        let unit = parse("neg(42)").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::UnaryApp { op, arg } => {
                assert_eq!(op, PrimOp::Neg);
                assert_eq!(unit.arena.get(arg).kind, TermKind::IntLit(42));
            }
            other => panic!("expected UnaryApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_binary_application() {
        let unit = parse("add(1, 2)").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::BinaryApp { op, lhs, rhs } => {
                assert_eq!(op, PrimOp::Add);
                assert_eq!(unit.arena.get(lhs).kind, TermKind::IntLit(1));
                assert_eq!(unit.arena.get(rhs).kind, TermKind::IntLit(2));
            }
            other => panic!("expected BinaryApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_nested_application() {
        let unit = parse("neg(neg(42))").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::UnaryApp { op, arg } => {
                assert_eq!(op, PrimOp::Neg);
                match unit.arena.get(arg).kind {
                    TermKind::UnaryApp {
                        op: inner_op,
                        arg: inner_arg,
                    } => {
                        assert_eq!(inner_op, PrimOp::Neg);
                        assert_eq!(unit.arena.get(inner_arg).kind, TermKind::IntLit(42));
                    }
                    other => panic!("expected inner UnaryApp, got {:?}", other),
                }
            }
            other => panic!("expected UnaryApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_let_binding_with_variable() {
        let unit = parse("let x : Q0 = 42 ; neg(x)").unwrap();
        assert_eq!(unit.binding_count, 1);
        // Root should be neg(x)
        match unit.arena.get(unit.root).kind {
            TermKind::UnaryApp { op, arg } => {
                assert_eq!(op, PrimOp::Neg);
                assert!(matches!(unit.arena.get(arg).kind, TermKind::Var(_)));
            }
            other => panic!("expected UnaryApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_assertion() {
        let unit = parse("let x : Q0 = 42 ; assert neg(neg(x)) = x ;").unwrap();
        assert_eq!(unit.binding_count, 1);
        assert_eq!(unit.assertion_count, 1);
        assert!(!unit.assertions[0].canonical);
    }

    #[test]
    fn parse_quantum_tagged_literal() {
        let unit = parse("42@Q1").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::QuantumLit { level, value } => {
                assert_eq!(level, hologram_core::op::RingLevel::Q1);
                assert_eq!(value, 42);
            }
            other => panic!("expected QuantumLit, got {:?}", other),
        }
    }

    #[test]
    fn parse_type_declaration() {
        let unit = parse("type MyT { residue : 0 ; }").unwrap();
        assert_eq!(unit.type_decl_count, 1);
        assert_eq!(
            unit.type_decls[0].constraint,
            hologram_core::term::ConstraintKind::Residue
        );
    }

    #[test]
    fn parse_all_unary_ops() {
        for (src, expected_op) in [
            ("neg(0)", PrimOp::Neg),
            ("bnot(0)", PrimOp::Bnot),
            ("succ(0)", PrimOp::Succ),
            ("pred(0)", PrimOp::Pred),
        ] {
            let unit = parse(src).unwrap();
            match unit.arena.get(unit.root).kind {
                TermKind::UnaryApp { op, .. } => assert_eq!(op, expected_op, "failed for {}", src),
                other => panic!("expected UnaryApp for {}, got {:?}", src, other),
            }
        }
    }

    #[test]
    fn parse_all_binary_ops() {
        for (src, expected_op) in [
            ("add(0, 1)", PrimOp::Add),
            ("sub(0, 1)", PrimOp::Sub),
            ("mul(0, 1)", PrimOp::Mul),
            ("xor(0, 1)", PrimOp::Xor),
            ("and(0, 1)", PrimOp::And),
            ("or(0, 1)", PrimOp::Or),
        ] {
            let unit = parse(src).unwrap();
            match unit.arena.get(unit.root).kind {
                TermKind::BinaryApp { op, .. } => assert_eq!(op, expected_op, "failed for {}", src),
                other => panic!("expected BinaryApp for {}, got {:?}", src, other),
            }
        }
    }

    #[test]
    fn parse_with_comments() {
        let unit = parse("-- compute negation\nneg(42) -- result").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::UnaryApp { op, .. } => assert_eq!(op, PrimOp::Neg),
            other => panic!("expected UnaryApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_error_unmatched_paren() {
        let result = parse("neg(");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.expected, "term (literal, application, or variable)");
    }

    #[test]
    fn parse_error_missing_comma() {
        let result = parse("add(1 2)");
        assert!(result.is_err());
    }

    #[test]
    fn parse_complex_expression() {
        // add(mul(2, 3), neg(succ(1)))
        let unit = parse("add(mul(2, 3), neg(succ(1)))").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::BinaryApp { op, .. } => assert_eq!(op, PrimOp::Add),
            other => panic!("expected BinaryApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_multiple_bindings() {
        let src = "let a : Q0 = 1 ; let b : Q0 = 2 ; add(a, b)";
        let unit = parse(src).unwrap();
        assert_eq!(unit.binding_count, 2);
        match unit.arena.get(unit.root).kind {
            TermKind::BinaryApp { op, .. } => assert_eq!(op, PrimOp::Add),
            other => panic!("expected BinaryApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_lut_sigmoid() {
        let unit = parse("sigmoid(42)").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::LutApp { op, arg } => {
                assert_eq!(op, hologram_core::op::LutOp::Sigmoid);
                assert_eq!(unit.arena.get(arg).kind, TermKind::IntLit(42));
            }
            other => panic!("expected LutApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_all_lut_ops() {
        use hologram_core::op::LutOp;
        for (src, expected_op) in [
            ("sigmoid(0)", LutOp::Sigmoid),
            ("tanh(0)", LutOp::Tanh),
            ("exp(0)", LutOp::Exp),
            ("log(0)", LutOp::Log),
            ("relu(0)", LutOp::Relu),
            ("sqrt(0)", LutOp::Sqrt),
            ("abs(0)", LutOp::Abs),
            ("gelu(0)", LutOp::Gelu),
            ("silu(0)", LutOp::Silu),
            ("sin(0)", LutOp::Sin),
            ("cos(0)", LutOp::Cos),
            ("tan(0)", LutOp::Tan),
            ("asin(0)", LutOp::Asin),
            ("acos(0)", LutOp::Acos),
            ("atan(0)", LutOp::Atan),
            ("log2(0)", LutOp::Log2),
            ("log10(0)", LutOp::Log10),
            ("exp2(0)", LutOp::Exp2),
            ("exp10(0)", LutOp::Exp10),
            ("square(0)", LutOp::Square),
            ("cube(0)", LutOp::Cube),
        ] {
            let unit = parse(src).unwrap();
            match unit.arena.get(unit.root).kind {
                TermKind::LutApp { op, .. } => assert_eq!(op, expected_op, "failed for {}", src),
                other => panic!("expected LutApp for {}, got {:?}", src, other),
            }
        }
    }

    #[test]
    fn parse_nested_lut_and_prim() {
        // sigmoid(neg(42))
        let unit = parse("sigmoid(neg(42))").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::LutApp { op, arg } => {
                assert_eq!(op, hologram_core::op::LutOp::Sigmoid);
                match unit.arena.get(arg).kind {
                    TermKind::UnaryApp { op: inner_op, .. } => assert_eq!(inner_op, PrimOp::Neg),
                    other => panic!("expected UnaryApp, got {:?}", other),
                }
            }
            other => panic!("expected LutApp, got {:?}", other),
        }
    }

    #[test]
    fn parse_lut_chain() {
        // relu(sigmoid(42))
        let unit = parse("relu(sigmoid(42))").unwrap();
        match unit.arena.get(unit.root).kind {
            TermKind::LutApp { op, .. } => assert_eq!(op, hologram_core::op::LutOp::Relu),
            other => panic!("expected LutApp, got {:?}", other),
        }
    }
}
