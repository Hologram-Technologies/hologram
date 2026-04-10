//! Zero-copy lexer for the UOR term language.
//!
//! Tokens hold `&str` slices into the input — no string allocation during lexing.
//! The lexer is O(n) where n = input length, with single-character lookahead.

use super::error::ParseError;

/// Token produced by the lexer. All variants are `Copy`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Token<'src> {
    /// Integer literal.
    Int(i64),
    /// Identifier (variable name, type name).
    Ident(&'src str),

    // ── Keywords (PrimOp operations) ──
    Neg,
    Bnot,
    Succ,
    Pred,
    Add,
    Sub,
    Mul,
    Xor,
    And,
    Or,

    // ── Keywords (LutOp activations) ──
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Relu,
    Sqrt,
    Abs,
    Gelu,
    Silu,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Log2,
    Log10,
    Exp2,
    Exp10,
    Square,
    Cube,

    // ── Keywords (statements) ──
    Let,
    Assert,
    Type,

    // ── Keywords (constraints) ──
    Residue,
    Carry,
    Hamming,
    Depth,
    Fiber,
    Affine,

    // ── Witt levels ──
    //
    // Each Witt level has two source-language spellings:
    //   `W8`/`W16`/`W24`/`W32` — preferred, matches `WittLevel::W*`
    //   `Q0`/`Q1`/`Q2`/`Q3`   — legacy v0.1.4 quantum-index spelling, retained
    //                           for source compatibility with existing scripts
    //
    // Both spellings produce the same downstream `RingLevel` value; the
    // distinction is purely lexical so error messages can echo what the
    // user wrote. Q* and W* tokens are interchangeable everywhere a
    // ring-level annotation is accepted.
    Q0,
    Q1,
    Q2,
    Q3,
    W8,
    W16,
    W24,
    W32,

    // ── Punctuation ──
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semi,
    At,
    /// `=` (strict ring equality in assertions, or assignment in let)
    Eq,
    /// `≡` (canonical-form equivalence in assertions)
    Equiv,

    Eof,
}

/// Lexer state: current position in the source string.
pub struct Lexer<'src> {
    src: &'src str,
    pos: usize,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer over the given source.
    pub fn new(src: &'src str) -> Self {
        Self { src, pos: 0 }
    }

    /// Current byte offset in the source.
    pub fn offset(&self) -> usize {
        self.pos
    }

    /// Peek at the next token without consuming it.
    pub fn peek(&self) -> Result<Token<'src>, ParseError> {
        let mut clone = Self {
            src: self.src,
            pos: self.pos,
        };
        clone.next_token()
    }

    /// Consume and return the next token.
    pub fn next_token(&mut self) -> Result<Token<'src>, ParseError> {
        self.skip_whitespace_and_comments();

        if self.pos >= self.src.len() {
            return Ok(Token::Eof);
        }

        let start = self.pos;
        let bytes = self.src.as_bytes();
        let b = bytes[self.pos];

        // Single-character tokens
        match b {
            b'(' => {
                self.pos += 1;
                return Ok(Token::LParen);
            }
            b')' => {
                self.pos += 1;
                return Ok(Token::RParen);
            }
            b'{' => {
                self.pos += 1;
                return Ok(Token::LBrace);
            }
            b'}' => {
                self.pos += 1;
                return Ok(Token::RBrace);
            }
            b',' => {
                self.pos += 1;
                return Ok(Token::Comma);
            }
            b':' => {
                self.pos += 1;
                return Ok(Token::Colon);
            }
            b';' => {
                self.pos += 1;
                return Ok(Token::Semi);
            }
            b'@' => {
                self.pos += 1;
                return Ok(Token::At);
            }
            b'=' => {
                self.pos += 1;
                return Ok(Token::Eq);
            }
            _ => {}
        }

        // UTF-8 multi-byte: check for ≡ (U+2261, bytes E2 89 A1)
        if b == 0xE2
            && self.pos + 2 < self.src.len()
            && bytes[self.pos + 1] == 0x89
            && bytes[self.pos + 2] == 0xA1
        {
            self.pos += 3;
            return Ok(Token::Equiv);
        }

        // Integer literal
        if b.is_ascii_digit() {
            return self.lex_integer(start);
        }

        // Negative integer literal
        if b == b'-' && self.pos + 1 < self.src.len() && bytes[self.pos + 1].is_ascii_digit() {
            return self.lex_integer(start);
        }

        // Identifier or keyword
        if b.is_ascii_alphabetic() || b == b'_' {
            return self.lex_ident(start);
        }

        Err(ParseError {
            offset: start,
            expected: "token",
            found: self.snippet(start, 20),
        })
    }

    fn lex_integer(&mut self, start: usize) -> Result<Token<'src>, ParseError> {
        let bytes = self.src.as_bytes();
        if bytes[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let slice = &self.src[start..self.pos];
        let value = slice.parse::<i64>().map_err(|_| ParseError {
            offset: start,
            expected: "integer",
            found: slice.to_string(),
        })?;
        Ok(Token::Int(value))
    }

    fn lex_ident(&mut self, start: usize) -> Result<Token<'src>, ParseError> {
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len()
            && (bytes[self.pos].is_ascii_alphanumeric() || bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        let word = &self.src[start..self.pos];
        let tok = match word {
            "neg" => Token::Neg,
            "bnot" => Token::Bnot,
            "succ" => Token::Succ,
            "pred" => Token::Pred,
            "add" => Token::Add,
            "sub" => Token::Sub,
            "mul" => Token::Mul,
            "xor" => Token::Xor,
            "and" => Token::And,
            "or" => Token::Or,
            // LutOp activations
            "sigmoid" => Token::Sigmoid,
            "tanh" => Token::Tanh,
            "exp" => Token::Exp,
            "log" => Token::Log,
            "relu" => Token::Relu,
            "sqrt" => Token::Sqrt,
            "abs" => Token::Abs,
            "gelu" => Token::Gelu,
            "silu" => Token::Silu,
            "sin" => Token::Sin,
            "cos" => Token::Cos,
            "tan" => Token::Tan,
            "asin" => Token::Asin,
            "acos" => Token::Acos,
            "atan" => Token::Atan,
            "log2" => Token::Log2,
            "log10" => Token::Log10,
            "exp2" => Token::Exp2,
            "exp10" => Token::Exp10,
            "square" => Token::Square,
            "cube" => Token::Cube,
            "let" => Token::Let,
            "assert" => Token::Assert,
            "type" => Token::Type,
            "residue" => Token::Residue,
            "carry" => Token::Carry,
            "hamming" => Token::Hamming,
            "depth" => Token::Depth,
            "fiber" => Token::Fiber,
            "affine" => Token::Affine,
            "Q0" => Token::Q0,
            "Q1" => Token::Q1,
            "Q2" => Token::Q2,
            "Q3" => Token::Q3,
            "W8" => Token::W8,
            "W16" => Token::W16,
            "W24" => Token::W24,
            "W32" => Token::W32,
            _ => Token::Ident(word),
        };
        Ok(tok)
    }

    fn skip_whitespace_and_comments(&mut self) {
        let bytes = self.src.as_bytes();
        loop {
            // Skip whitespace
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            // Skip line comments: --
            if self.pos + 1 < bytes.len() && bytes[self.pos] == b'-' && bytes[self.pos + 1] == b'-'
            {
                while self.pos < bytes.len() && bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            // Skip block comments: (* ... *)
            if self.pos + 1 < bytes.len() && bytes[self.pos] == b'(' && bytes[self.pos + 1] == b'*'
            {
                self.pos += 2;
                while self.pos + 1 < bytes.len() {
                    if bytes[self.pos] == b'*' && bytes[self.pos + 1] == b')' {
                        self.pos += 2;
                        break;
                    }
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
    }

    fn snippet(&self, start: usize, max_len: usize) -> String {
        let end = (start + max_len).min(self.src.len());
        self.src[start..end].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_all(src: &str) -> Vec<Token<'_>> {
        let mut lexer = Lexer::new(src);
        let mut tokens = Vec::new();
        loop {
            let tok = lexer.next_token().unwrap();
            if tok == Token::Eof {
                break;
            }
            tokens.push(tok);
        }
        tokens
    }

    #[test]
    fn lex_simple_unary() {
        let tokens = lex_all("neg(42)");
        assert_eq!(
            tokens,
            vec![Token::Neg, Token::LParen, Token::Int(42), Token::RParen]
        );
    }

    #[test]
    fn lex_binary_application() {
        let tokens = lex_all("add(1, 2)");
        assert_eq!(
            tokens,
            vec![
                Token::Add,
                Token::LParen,
                Token::Int(1),
                Token::Comma,
                Token::Int(2),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn lex_let_binding() {
        let tokens = lex_all("let x : Q0 = 42 ;");
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Ident("x"),
                Token::Colon,
                Token::Q0,
                Token::Eq,
                Token::Int(42),
                Token::Semi,
            ]
        );
    }

    #[test]
    fn lex_assertion() {
        let tokens = lex_all("assert neg(neg(x)) = x ;");
        assert_eq!(
            tokens,
            vec![
                Token::Assert,
                Token::Neg,
                Token::LParen,
                Token::Neg,
                Token::LParen,
                Token::Ident("x"),
                Token::RParen,
                Token::RParen,
                Token::Eq,
                Token::Ident("x"),
                Token::Semi,
            ]
        );
    }

    #[test]
    fn lex_quantum_literal() {
        let tokens = lex_all("42@Q1");
        assert_eq!(tokens, vec![Token::Int(42), Token::At, Token::Q1]);
    }

    #[test]
    fn lex_witt_literal() {
        // The preferred W8/W16/W24/W32 spelling lexes to the W tokens.
        let tokens = lex_all("42@W16");
        assert_eq!(tokens, vec![Token::Int(42), Token::At, Token::W16]);
    }

    #[test]
    fn lex_all_witt_levels() {
        let tokens = lex_all("W8 W16 W24 W32");
        assert_eq!(tokens, vec![Token::W8, Token::W16, Token::W24, Token::W32]);
    }

    #[test]
    fn lex_let_with_witt_annotation() {
        // Both Q0 and W8 are accepted in let-bindings.
        let tokens = lex_all("let x : W8 = 42 ;");
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Ident("x"),
                Token::Colon,
                Token::W8,
                Token::Eq,
                Token::Int(42),
                Token::Semi,
            ]
        );
    }

    #[test]
    fn lex_all_ops() {
        let tokens = lex_all("neg bnot succ pred add sub mul xor and or");
        assert_eq!(
            tokens,
            vec![
                Token::Neg,
                Token::Bnot,
                Token::Succ,
                Token::Pred,
                Token::Add,
                Token::Sub,
                Token::Mul,
                Token::Xor,
                Token::And,
                Token::Or,
            ]
        );
    }

    #[test]
    fn lex_line_comment() {
        let tokens = lex_all("add(1, -- this is a comment\n2)");
        assert_eq!(
            tokens,
            vec![
                Token::Add,
                Token::LParen,
                Token::Int(1),
                Token::Comma,
                Token::Int(2),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn lex_block_comment() {
        let tokens = lex_all("add(1, (* block *) 2)");
        assert_eq!(
            tokens,
            vec![
                Token::Add,
                Token::LParen,
                Token::Int(1),
                Token::Comma,
                Token::Int(2),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn lex_negative_integer() {
        let tokens = lex_all("-42");
        assert_eq!(tokens, vec![Token::Int(-42)]);
    }

    #[test]
    fn lex_constraint_keywords() {
        let tokens = lex_all("residue carry hamming depth fiber affine");
        assert_eq!(
            tokens,
            vec![
                Token::Residue,
                Token::Carry,
                Token::Hamming,
                Token::Depth,
                Token::Fiber,
                Token::Affine,
            ]
        );
    }

    #[test]
    fn lex_type_decl() {
        let tokens = lex_all("type MyType { residue : 0 ; }");
        assert_eq!(
            tokens,
            vec![
                Token::Type,
                Token::Ident("MyType"),
                Token::LBrace,
                Token::Residue,
                Token::Colon,
                Token::Int(0),
                Token::Semi,
                Token::RBrace,
            ]
        );
    }

    #[test]
    fn lex_empty_input() {
        let tokens = lex_all("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn lex_whitespace_only() {
        let tokens = lex_all("   \n\t  ");
        assert!(tokens.is_empty());
    }
}
