//! Recursive descent parser for the UOR term language.
//!
//! O(n) where n = input length. LL(1) grammar — each production is determined
//! by a single lookahead token. Allocates directly into a [`TermArena`].

use hologram_core::op::{LutOp, PrimOp, RingLevel};
use hologram_core::term::{
    Assertion, Binding, ConstraintKind, TermArena, TermId, TermKind, TypeDecl, TypeId, VarId,
};

use super::error::ParseError;
use super::lexer::{Lexer, Token};
use super::ParsedUnit;

/// Maximum number of named variables in scope.
const MAX_VARS: usize = 256;

/// Parser state.
pub struct Parser<'src> {
    lexer: Lexer<'src>,
    arena: TermArena,
    /// Variable name → VarId mapping. Linear scan is fine for < 256 entries.
    var_names: [&'src str; MAX_VARS],
    var_count: u16,
    bindings: Box<[Binding; 64]>,
    binding_count: u8,
    assertions: Box<[Assertion; 32]>,
    assertion_count: u8,
    type_decls: Box<[TypeDecl; 16]>,
    type_decl_count: u8,
}

impl<'src> Parser<'src> {
    pub fn new(src: &'src str) -> Self {
        Self {
            lexer: Lexer::new(src),
            arena: TermArena::with_capacity(256),
            var_names: [""; MAX_VARS],
            var_count: 0,
            bindings: Box::new(
                [Binding {
                    var: VarId(0),
                    ty: TypeId::UNCONSTRAINED,
                    rhs: TermId(0),
                }; 64],
            ),
            binding_count: 0,
            assertions: Box::new(
                [Assertion {
                    lhs: TermId(0),
                    rhs: TermId(0),
                    canonical: false,
                }; 32],
            ),
            assertion_count: 0,
            type_decls: Box::new(
                [TypeDecl {
                    name_id: VarId(0),
                    constraint: ConstraintKind::Residue,
                    value: TermId(0),
                }; 16],
            ),
            type_decl_count: 0,
        }
    }

    /// Parse the full source and return a [`ParsedUnit`].
    pub fn parse(mut self) -> Result<ParsedUnit, ParseError> {
        // Parse statements until EOF. The last expression is the root term.
        let mut last_term: Option<TermId> = None;

        loop {
            let tok = self.lexer.peek()?;
            match tok {
                Token::Eof => break,
                Token::Let => {
                    self.parse_let_binding()?;
                }
                Token::Assert => {
                    self.parse_assertion()?;
                }
                Token::Type => {
                    self.parse_type_decl()?;
                }
                _ => {
                    let term = self.parse_term()?;
                    last_term = Some(term);
                    // Consume optional trailing semicolon
                    if self.lexer.peek()? == Token::Semi {
                        self.lexer.next_token()?;
                    }
                }
            }
        }

        let root = last_term.unwrap_or_else(|| {
            // If no standalone expression, use the last binding's rhs
            if self.binding_count > 0 {
                self.bindings[self.binding_count as usize - 1].rhs
            } else {
                self.arena.alloc(TermKind::IntLit(0))
            }
        });

        Ok(ParsedUnit {
            arena: self.arena,
            root,
            bindings: self.bindings,
            binding_count: self.binding_count,
            assertions: self.assertions,
            assertion_count: self.assertion_count,
            type_decls: self.type_decls,
            type_decl_count: self.type_decl_count,
        })
    }

    // ── Statement parsers ────────────────────────────────────────────────────

    /// `"let" identifier ":" quantum-level "=" term ";"`
    fn parse_let_binding(&mut self) -> Result<(), ParseError> {
        self.expect(Token::Let)?;
        let name = self.expect_ident()?;
        self.expect(Token::Colon)?;
        let ty = self.parse_type_annotation()?;
        self.expect(Token::Eq)?;
        let rhs = self.parse_term()?;
        self.expect(Token::Semi)?;

        let var_id = self.intern_var(name);
        let idx = self.binding_count as usize;
        if idx < 64 {
            self.bindings[idx] = Binding {
                var: var_id,
                ty,
                rhs,
            };
            self.binding_count += 1;
        }
        Ok(())
    }

    /// `"assert" term ("=" | "≡") term ";"`
    fn parse_assertion(&mut self) -> Result<(), ParseError> {
        self.expect(Token::Assert)?;
        let lhs = self.parse_term()?;

        let tok = self.lexer.next_token()?;
        let canonical = match tok {
            Token::Eq => false,
            Token::Equiv => true,
            _ => {
                return Err(ParseError {
                    offset: self.lexer.offset(),
                    expected: "'=' or '≡'",
                    found: format!("{:?}", tok),
                });
            }
        };

        let rhs = self.parse_term()?;
        self.expect(Token::Semi)?;

        let idx = self.assertion_count as usize;
        if idx < 32 {
            self.assertions[idx] = Assertion {
                lhs,
                rhs,
                canonical,
            };
            self.assertion_count += 1;
        }
        Ok(())
    }

    /// `"type" identifier "{" { constraint-kind ":" term ";" } "}"`
    fn parse_type_decl(&mut self) -> Result<(), ParseError> {
        self.expect(Token::Type)?;
        let name = self.expect_ident()?;
        let name_id = self.intern_var(name);
        self.expect(Token::LBrace)?;

        loop {
            let tok = self.lexer.peek()?;
            if tok == Token::RBrace {
                self.lexer.next_token()?;
                break;
            }
            let constraint = self.parse_constraint_kind()?;
            self.expect(Token::Colon)?;
            let value = self.parse_term()?;
            self.expect(Token::Semi)?;

            let idx = self.type_decl_count as usize;
            if idx < 16 {
                self.type_decls[idx] = TypeDecl {
                    name_id,
                    constraint,
                    value,
                };
                self.type_decl_count += 1;
            }
        }

        Ok(())
    }

    // ── Term parser ──────────────────────────────────────────────────────────

    /// Parse a term: `literal | application | variable`.
    fn parse_term(&mut self) -> Result<TermId, ParseError> {
        let tok = self.lexer.peek()?;
        match tok {
            // Unary operations
            Token::Neg | Token::Bnot | Token::Succ | Token::Pred => self.parse_unary_application(),
            // Binary operations
            Token::Add | Token::Sub | Token::Mul | Token::Xor | Token::And | Token::Or => {
                self.parse_binary_application()
            }
            // LUT activation functions
            Token::Sigmoid
            | Token::Tanh
            | Token::Exp
            | Token::Log
            | Token::Relu
            | Token::Sqrt
            | Token::Abs
            | Token::Gelu
            | Token::Silu
            | Token::Sin
            | Token::Cos
            | Token::Tan
            | Token::Asin
            | Token::Acos
            | Token::Atan
            | Token::Log2
            | Token::Log10
            | Token::Exp2
            | Token::Exp10
            | Token::Square
            | Token::Cube => self.parse_lut_application(),
            // Integer literal (possibly Witt-level-tagged)
            Token::Int(_) => self.parse_literal(),
            // Variable
            Token::Ident(_) => {
                let name = self.expect_ident()?;
                let var_id = self.lookup_var(name)?;
                Ok(self.arena.alloc(TermKind::Var(var_id)))
            }
            _ => Err(ParseError {
                offset: self.lexer.offset(),
                expected: "term (literal, application, or variable)",
                found: format!("{:?}", tok),
            }),
        }
    }

    /// Parse a literal, possibly quantum-tagged: `integer ["@" quantum-level]`.
    fn parse_literal(&mut self) -> Result<TermId, ParseError> {
        let tok = self.lexer.next_token()?;
        let value = match tok {
            Token::Int(v) => v,
            _ => unreachable!(),
        };

        // Check for Witt-level tag: `42@W8` (or legacy `42@Q0`).
        if self.lexer.peek()? == Token::At {
            self.lexer.next_token()?; // consume @
            let level = self.parse_witt_level()?;
            return Ok(self.arena.alloc(TermKind::QuantumLit {
                level,
                value: value as u32,
            }));
        }

        Ok(self.arena.alloc(TermKind::IntLit(value)))
    }

    /// `lut-op "(" term ")"`.
    fn parse_lut_application(&mut self) -> Result<TermId, ParseError> {
        let op = self.parse_lut_op()?;
        self.expect(Token::LParen)?;
        let arg = self.parse_term()?;
        self.expect(Token::RParen)?;
        Ok(self.arena.alloc(TermKind::LutApp { op, arg }))
    }

    /// `unary-op "(" term ")"`.
    fn parse_unary_application(&mut self) -> Result<TermId, ParseError> {
        let op = self.parse_unary_op()?;
        self.expect(Token::LParen)?;
        let arg = self.parse_term()?;
        self.expect(Token::RParen)?;
        Ok(self.arena.alloc(TermKind::UnaryApp { op, arg }))
    }

    /// `binary-op "(" term "," term ")"`.
    fn parse_binary_application(&mut self) -> Result<TermId, ParseError> {
        let op = self.parse_binary_op()?;
        self.expect(Token::LParen)?;
        let lhs = self.parse_term()?;
        self.expect(Token::Comma)?;
        let rhs = self.parse_term()?;
        self.expect(Token::RParen)?;
        Ok(self.arena.alloc(TermKind::BinaryApp { op, lhs, rhs }))
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn parse_unary_op(&mut self) -> Result<PrimOp, ParseError> {
        let tok = self.lexer.next_token()?;
        match tok {
            Token::Neg => Ok(PrimOp::Neg),
            Token::Bnot => Ok(PrimOp::Bnot),
            Token::Succ => Ok(PrimOp::Succ),
            Token::Pred => Ok(PrimOp::Pred),
            _ => Err(ParseError {
                offset: self.lexer.offset(),
                expected: "unary operator (neg, bnot, succ, pred)",
                found: format!("{:?}", tok),
            }),
        }
    }

    fn parse_lut_op(&mut self) -> Result<LutOp, ParseError> {
        let tok = self.lexer.next_token()?;
        match tok {
            Token::Sigmoid => Ok(LutOp::Sigmoid),
            Token::Tanh => Ok(LutOp::Tanh),
            Token::Exp => Ok(LutOp::Exp),
            Token::Log => Ok(LutOp::Log),
            Token::Relu => Ok(LutOp::Relu),
            Token::Sqrt => Ok(LutOp::Sqrt),
            Token::Abs => Ok(LutOp::Abs),
            Token::Gelu => Ok(LutOp::Gelu),
            Token::Silu => Ok(LutOp::Silu),
            Token::Sin => Ok(LutOp::Sin),
            Token::Cos => Ok(LutOp::Cos),
            Token::Tan => Ok(LutOp::Tan),
            Token::Asin => Ok(LutOp::Asin),
            Token::Acos => Ok(LutOp::Acos),
            Token::Atan => Ok(LutOp::Atan),
            Token::Log2 => Ok(LutOp::Log2),
            Token::Log10 => Ok(LutOp::Log10),
            Token::Exp2 => Ok(LutOp::Exp2),
            Token::Exp10 => Ok(LutOp::Exp10),
            Token::Square => Ok(LutOp::Square),
            Token::Cube => Ok(LutOp::Cube),
            _ => Err(ParseError {
                offset: self.lexer.offset(),
                expected: "LUT activation function",
                found: format!("{:?}", tok),
            }),
        }
    }

    fn parse_binary_op(&mut self) -> Result<PrimOp, ParseError> {
        let tok = self.lexer.next_token()?;
        match tok {
            Token::Add => Ok(PrimOp::Add),
            Token::Sub => Ok(PrimOp::Sub),
            Token::Mul => Ok(PrimOp::Mul),
            Token::Xor => Ok(PrimOp::Xor),
            Token::And => Ok(PrimOp::And),
            Token::Or => Ok(PrimOp::Or),
            _ => Err(ParseError {
                offset: self.lexer.offset(),
                expected: "binary operator (add, sub, mul, xor, and, or)",
                found: format!("{:?}", tok),
            }),
        }
    }

    /// Parse a Witt-level annotation. Accepts both the preferred
    /// `W8`/`W16`/`W24`/`W32` spelling and the legacy v0.1.4
    /// `Q0`/`Q1`/`Q2`/`Q3` spelling. Both map to the same `RingLevel`.
    fn parse_witt_level(&mut self) -> Result<RingLevel, ParseError> {
        let tok = self.lexer.next_token()?;
        match tok {
            Token::Q0 | Token::W8 => Ok(RingLevel::Q0),
            Token::Q1 | Token::W16 => Ok(RingLevel::Q1),
            Token::Q2 | Token::W24 => Ok(RingLevel::Q2),
            Token::Q3 | Token::W32 => Ok(RingLevel::Q3),
            _ => Err(ParseError {
                offset: self.lexer.offset(),
                expected: "Witt level (W8/W16/W24/W32 or legacy Q0/Q1/Q2/Q3)",
                found: format!("{:?}", tok),
            }),
        }
    }

    fn parse_type_annotation(&mut self) -> Result<TypeId, ParseError> {
        let tok = self.lexer.peek()?;
        match tok {
            Token::Q0
            | Token::Q1
            | Token::Q2
            | Token::Q3
            | Token::W8
            | Token::W16
            | Token::W24
            | Token::W32 => {
                self.lexer.next_token()?;
                // TypeId encoding: 0=W8, 1=W16, 2=W24, 3=W32. Both Q*
                // and W* spellings map to the same encoding.
                let id = match tok {
                    Token::Q0 | Token::W8 => 0,
                    Token::Q1 | Token::W16 => 1,
                    Token::Q2 | Token::W24 => 2,
                    Token::Q3 | Token::W32 => 3,
                    _ => unreachable!(),
                };
                Ok(TypeId(id))
            }
            Token::Ident(_) => {
                let name = self.expect_ident()?;
                let var_id = self.lookup_var(name)?;
                Ok(TypeId(var_id.0))
            }
            _ => Err(ParseError {
                offset: self.lexer.offset(),
                expected: "type annotation (W8/W16/W24/W32, legacy Q0/Q1/Q2/Q3, or type name)",
                found: format!("{:?}", tok),
            }),
        }
    }

    fn parse_constraint_kind(&mut self) -> Result<ConstraintKind, ParseError> {
        let tok = self.lexer.next_token()?;
        match tok {
            Token::Residue => Ok(ConstraintKind::Residue),
            Token::Carry => Ok(ConstraintKind::Carry),
            Token::Hamming => Ok(ConstraintKind::Hamming),
            Token::Depth => Ok(ConstraintKind::Depth),
            Token::Fiber => Ok(ConstraintKind::Fiber),
            Token::Affine => Ok(ConstraintKind::Affine),
            _ => Err(ParseError {
                offset: self.lexer.offset(),
                expected: "constraint kind (residue, carry, hamming, depth, fiber, affine)",
                found: format!("{:?}", tok),
            }),
        }
    }

    fn expect(&mut self, expected: Token<'_>) -> Result<(), ParseError> {
        let tok = self.lexer.next_token()?;
        if std::mem::discriminant(&tok) == std::mem::discriminant(&expected) {
            Ok(())
        } else {
            Err(ParseError {
                offset: self.lexer.offset(),
                expected: match expected {
                    Token::LParen => "'('",
                    Token::RParen => "')'",
                    Token::LBrace => "'{'",
                    Token::RBrace => "'}'",
                    Token::Comma => "','",
                    Token::Colon => "':'",
                    Token::Semi => "';'",
                    Token::At => "'@'",
                    Token::Eq => "'='",
                    Token::Let => "'let'",
                    Token::Assert => "'assert'",
                    Token::Type => "'type'",
                    _ => "expected token",
                },
                found: format!("{:?}", tok),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<&'src str, ParseError> {
        let tok = self.lexer.next_token()?;
        if let Token::Ident(name) = tok {
            Ok(name)
        } else {
            Err(ParseError {
                offset: self.lexer.offset(),
                expected: "identifier",
                found: format!("{:?}", tok),
            })
        }
    }

    fn intern_var(&mut self, name: &'src str) -> VarId {
        // Check if already interned
        for i in 0..self.var_count as usize {
            if self.var_names[i] == name {
                return VarId(i as u16);
            }
        }
        // Intern new
        let id = self.var_count;
        if (id as usize) < MAX_VARS {
            self.var_names[id as usize] = name;
            self.var_count += 1;
        }
        VarId(id)
    }

    fn lookup_var(&self, name: &str) -> Result<VarId, ParseError> {
        for i in 0..self.var_count as usize {
            if self.var_names[i] == name {
                return Ok(VarId(i as u16));
            }
        }
        Err(ParseError {
            offset: self.lexer.offset(),
            expected: "bound variable",
            found: name.to_string(),
        })
    }
}
