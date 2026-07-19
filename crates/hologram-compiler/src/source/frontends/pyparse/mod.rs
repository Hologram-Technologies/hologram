//! Minimal, self-contained parser for the restricted Python subset accepted by
//! the compiler's Python frontend.
//!
//! This intentionally re-implements only the tiny slice of the `rustpython-parser`
//! surface that [`super::python`] relies on, so that the LGPL-3.0 transitive
//! dependency (`malachite`, via `rustpython-parser`) can be dropped. The AST
//! types, field names, methods, and `.range()`/`.as_str()` accessors below mirror
//! the rustpython API used at the call site; behaviour is unchanged.
//!
//! The supported grammar:
//!   - Module: list of statements.
//!   - `def NAME(ARGS):` NEWLINE INDENT <body> DEDENT.
//!   - Simple statements: `TARGET = EXPR` (Assign), bare `EXPR` (Expr), `pass`.
//!   - Every other statement (`if`, `for`, `while`, `return`, `class`, ...) is
//!     parsed into an opaque [`Stmt::Other`] carrying the range of its first
//!     token, plus any following more-indented block consumed as opaque; the
//!     frontend rejects these via its catch-all match arm.
//!   - Expressions: Name, Str, Int, Float, `True`/`False`/`None`, `[..]` List,
//!     `(..)` Tuple / parenthesised expr, Call `f(a, kw=v)`, Attribute `a.b.c`,
//!     and prefix unary `+`/`-`.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

mod lexer;

use lexer::{Lexer, Token, TokenKind};

/// Byte offset into the source string. Mirrors `rustpython_parser::text_size::TextSize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextSize(u32);

impl TextSize {
    fn new(value: usize) -> Self {
        Self(value as u32)
    }
}

impl From<TextSize> for usize {
    fn from(value: TextSize) -> Self {
        value.0 as usize
    }
}

/// Half-open byte range into the source string. Mirrors
/// `rustpython_parser::text_size::TextRange`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextRange {
    start: TextSize,
    end: TextSize,
}

impl TextRange {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start: TextSize::new(start),
            end: TextSize::new(end),
        }
    }

    pub fn start(&self) -> TextSize {
        self.start
    }

    pub fn end(&self) -> TextSize {
        self.end
    }
}

/// Provides `.range()` on every AST node. Mirrors `rustpython_parser::ast::Ranged`.
pub trait Ranged {
    fn range(&self) -> TextRange;
}

/// Parse error exposing a byte `offset`, mirroring `rustpython_parser::ParseError`.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub offset: TextSize,
}

impl ParseError {
    fn at(offset: usize) -> Self {
        Self {
            offset: TextSize::new(offset),
        }
    }
}

/// Identifier wrapper exposing `.as_str()`, mirroring rustpython's `Identifier`.
#[derive(Debug, Clone)]
pub struct Identifier(String);

impl Identifier {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A parsed suite: a flat list of module-level statements.
pub type Suite = Vec<Stmt>;

/// Statement AST. Only the variants used by the frontend are modelled richly;
/// everything else lands in [`Stmt::Other`].
#[derive(Debug)]
pub enum Stmt {
    FunctionDef(StmtFunctionDef),
    Assign(StmtAssign),
    Expr(StmtExpr),
    Pass(StmtPass),
    /// Any statement the restricted frontend does not model (`if`, `for`,
    /// `return`, `class`, ...). Carries the range of its first token.
    Other(StmtOther),
}

impl Ranged for Stmt {
    fn range(&self) -> TextRange {
        match self {
            Stmt::FunctionDef(node) => node.range,
            Stmt::Assign(node) => node.range,
            Stmt::Expr(node) => node.range,
            Stmt::Pass(node) => node.range,
            Stmt::Other(node) => node.range,
        }
    }
}

#[derive(Debug)]
pub struct StmtFunctionDef {
    pub name: Identifier,
    pub args: Arguments,
    pub body: Vec<Stmt>,
    range: TextRange,
}

#[derive(Debug)]
pub struct StmtAssign {
    pub targets: Vec<Expr>,
    pub value: Expr,
    range: TextRange,
}

impl Ranged for StmtAssign {
    fn range(&self) -> TextRange {
        self.range
    }
}

#[derive(Debug)]
pub struct StmtExpr {
    pub value: Expr,
    range: TextRange,
}

#[derive(Debug)]
pub struct StmtPass {
    range: TextRange,
}

#[derive(Debug)]
pub struct StmtOther {
    range: TextRange,
}

/// Function arguments. Mirrors rustpython's `Arguments`: the frontend only reads
/// `.args` and `.posonlyargs`, each element being an [`ArgWithDefault`].
#[derive(Debug)]
pub struct Arguments {
    pub posonlyargs: Vec<ArgWithDefault>,
    pub args: Vec<ArgWithDefault>,
}

/// Mirrors rustpython's `ArgWithDefault`, whose `.def.arg` is the identifier.
#[derive(Debug)]
pub struct ArgWithDefault {
    pub def: Arg,
}

#[derive(Debug)]
pub struct Arg {
    pub arg: Identifier,
}

/// Keyword argument in a call. Mirrors rustpython's `Keyword`.
#[derive(Debug)]
pub struct Keyword {
    pub arg: Option<Identifier>,
    pub value: Expr,
    range: TextRange,
}

impl Ranged for Keyword {
    fn range(&self) -> TextRange {
        self.range
    }
}

/// Expression AST. Only the variants used by the frontend are modelled.
#[derive(Debug)]
pub enum Expr {
    Call(ExprCall),
    Name(ExprName),
    Attribute(ExprAttribute),
    Constant(ExprConstant),
    UnaryOp(ExprUnaryOp),
    List(ExprList),
    Tuple(ExprTuple),
}

impl Ranged for Expr {
    fn range(&self) -> TextRange {
        match self {
            Expr::Call(node) => node.range,
            Expr::Name(node) => node.range,
            Expr::Attribute(node) => node.range,
            Expr::Constant(node) => node.range,
            Expr::UnaryOp(node) => node.range,
            Expr::List(node) => node.range,
            Expr::Tuple(node) => node.range,
        }
    }
}

#[derive(Debug)]
pub struct ExprCall {
    pub func: Box<Expr>,
    pub args: Vec<Expr>,
    pub keywords: Vec<Keyword>,
    range: TextRange,
}

impl Ranged for ExprCall {
    fn range(&self) -> TextRange {
        self.range
    }
}

#[derive(Debug)]
pub struct ExprName {
    pub id: Identifier,
    range: TextRange,
}

#[derive(Debug)]
pub struct ExprAttribute {
    pub value: Box<Expr>,
    pub attr: Identifier,
    range: TextRange,
}

#[derive(Debug)]
pub struct ExprConstant {
    pub value: Constant,
    range: TextRange,
}

#[derive(Debug)]
pub struct ExprUnaryOp {
    pub op: UnaryOp,
    pub operand: Box<Expr>,
    range: TextRange,
}

#[derive(Debug)]
pub struct ExprList {
    pub elts: Vec<Expr>,
    range: TextRange,
}

#[derive(Debug)]
pub struct ExprTuple {
    pub elts: Vec<Expr>,
    range: TextRange,
}

/// Literal constant. `Int` stores the original text (no bignum needed; the
/// frontend only ever calls `.to_string()` on it).
#[derive(Debug, Clone)]
pub enum Constant {
    Str(String),
    Int(String),
    Float(f64),
    Bool(bool),
    None,
}

/// Unary operators. Only `+`/`-` are produced by this parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    USub,
    UAdd,
}

/// Entry point trait mirroring `rustpython_parser::Parse`.
pub trait Parse: Sized {
    fn parse(text: &str, _source_path: &str) -> Result<Self, ParseError>;
}

impl Parse for Suite {
    fn parse(text: &str, _source_path: &str) -> Result<Self, ParseError> {
        Parser::new(text)?.parse_module()
    }
}

/// Recursive-descent parser over a token stream produced by the [`Lexer`].
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(text: &str) -> Result<Self, ParseError> {
        let tokens = Lexer::new(text).tokenize()?;
        Ok(Self { tokens, pos: 0 })
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn advance(&mut self) -> Token {
        let token = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        token
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Token, ParseError> {
        if self.peek_kind() == kind {
            Ok(self.advance())
        } else {
            Err(ParseError::at(self.peek().start))
        }
    }

    /// Consume any leading NEWLINE tokens (blank lines between statements).
    fn skip_newlines(&mut self) {
        while matches!(self.peek_kind(), TokenKind::Newline) {
            self.advance();
        }
    }

    fn parse_module(&mut self) -> Result<Suite, ParseError> {
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek_kind(), TokenKind::Eof) {
                break;
            }
            stmts.push(self.parse_statement()?);
        }
        Ok(stmts)
    }

    /// Parse one statement at the current indentation level.
    fn parse_statement(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_kind() {
            TokenKind::Keyword(word) if word == "def" => self.parse_function_def(),
            TokenKind::Keyword(word) if word == "pass" => {
                let token = self.advance();
                let range = TextRange::new(token.start, token.end);
                self.end_simple_statement()?;
                Ok(Stmt::Pass(StmtPass { range }))
            }
            TokenKind::Keyword(_) => self.parse_other_statement(),
            _ => self.parse_expr_or_assign(),
        }
    }

    /// `def NAME(ARGS): NEWLINE INDENT <body> DEDENT`.
    fn parse_function_def(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek().start;
        self.advance(); // `def`
        let name_token = self.expect_name()?;
        let name = Identifier(name_token.0);
        self.expect(&TokenKind::LParen)?;
        let args = self.parse_arg_list()?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Colon)?;
        let (body, end) = self.parse_block()?;
        Ok(Stmt::FunctionDef(StmtFunctionDef {
            name,
            args,
            body,
            range: TextRange::new(start, end),
        }))
    }

    /// Parse a comma-separated parameter list (names only; defaults/annotations
    /// are consumed but not modelled beyond the identifier).
    fn parse_arg_list(&mut self) -> Result<Arguments, ParseError> {
        let mut args = Vec::new();
        loop {
            if matches!(self.peek_kind(), TokenKind::RParen) {
                break;
            }
            let name_token = self.expect_name()?;
            args.push(ArgWithDefault {
                def: Arg {
                    arg: Identifier(name_token.0),
                },
            });
            // Skip an optional annotation `: type` and default `= expr`.
            if matches!(self.peek_kind(), TokenKind::Colon) {
                self.advance();
                self.parse_expr()?;
            }
            if matches!(self.peek_kind(), TokenKind::Equals) {
                self.advance();
                self.parse_expr()?;
            }
            match self.peek_kind() {
                TokenKind::Comma => {
                    self.advance();
                }
                _ => break,
            }
        }
        Ok(Arguments {
            posonlyargs: Vec::new(),
            args,
        })
    }

    /// Parse an indented block, returning its statements and end offset. If no
    /// INDENT follows, the block is empty.
    fn parse_block(&mut self) -> Result<(Vec<Stmt>, usize), ParseError> {
        self.skip_newlines();
        if !matches!(self.peek_kind(), TokenKind::Indent) {
            // Empty / inline block; nothing to consume.
            return Ok((Vec::new(), self.previous_end()));
        }
        self.advance(); // INDENT
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            match self.peek_kind() {
                TokenKind::Dedent | TokenKind::Eof => break,
                _ => stmts.push(self.parse_statement()?),
            }
        }
        let end = self.previous_end();
        if matches!(self.peek_kind(), TokenKind::Dedent) {
            self.advance();
        }
        Ok((stmts, end))
    }

    /// Any compound/unsupported statement: capture the first token's start,
    /// consume the rest of the header line, then swallow a trailing block.
    fn parse_other_statement(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek().start;
        // Consume the header line up to (but not including) NEWLINE.
        let mut had_colon = false;
        loop {
            match self.peek_kind() {
                TokenKind::Newline | TokenKind::Eof => break,
                TokenKind::Colon => {
                    had_colon = true;
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }
        let mut end = self.previous_end();
        if matches!(self.peek_kind(), TokenKind::Newline) {
            self.advance();
        }
        // If the header ended with a colon, consume any indented body opaquely.
        if had_colon {
            let block_end = self.consume_opaque_block();
            if block_end > end {
                end = block_end;
            }
        }
        Ok(Stmt::Other(StmtOther {
            range: TextRange::new(start, end),
        }))
    }

    /// Swallow an indented block without building statements from it. Returns the
    /// end offset of the consumed block.
    fn consume_opaque_block(&mut self) -> usize {
        self.skip_newlines();
        if !matches!(self.peek_kind(), TokenKind::Indent) {
            return self.previous_end();
        }
        let mut depth = 0usize;
        loop {
            match self.peek_kind() {
                TokenKind::Indent => {
                    depth += 1;
                    self.advance();
                }
                TokenKind::Dedent => {
                    self.advance();
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                TokenKind::Eof => break,
                _ => {
                    self.advance();
                }
            }
        }
        self.previous_end()
    }

    /// `TARGET = EXPR` (Assign) or a bare `EXPR` (Expr).
    fn parse_expr_or_assign(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek().start;
        let first = self.parse_expr()?;
        if matches!(self.peek_kind(), TokenKind::Equals) {
            self.advance();
            let value = self.parse_expr()?;
            let end = value.range().end.into();
            self.end_simple_statement()?;
            Ok(Stmt::Assign(StmtAssign {
                targets: alloc::vec![first],
                value,
                range: TextRange::new(start, end),
            }))
        } else {
            let end = first.range().end.into();
            self.end_simple_statement()?;
            Ok(Stmt::Expr(StmtExpr {
                value: first,
                range: TextRange::new(start, end),
            }))
        }
    }

    /// After a simple statement, require NEWLINE or EOF.
    fn end_simple_statement(&mut self) -> Result<(), ParseError> {
        match self.peek_kind() {
            TokenKind::Newline => {
                self.advance();
                Ok(())
            }
            TokenKind::Eof | TokenKind::Dedent => Ok(()),
            _ => Err(ParseError::at(self.peek().start)),
        }
    }

    // ----- Expressions -----

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_unary()
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.peek_kind() {
            TokenKind::Plus | TokenKind::Minus => {
                let op_token = self.advance();
                let op = if matches!(op_token.kind, TokenKind::Minus) {
                    UnaryOp::USub
                } else {
                    UnaryOp::UAdd
                };
                let operand = self.parse_unary()?;
                let end = operand.range().end.into();
                Ok(Expr::UnaryOp(ExprUnaryOp {
                    op,
                    operand: Box::new(operand),
                    range: TextRange::new(op_token.start, end),
                }))
            }
            _ => self.parse_postfix(),
        }
    }

    /// Primary followed by attribute access `.attr` and calls `(...)`.
    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek_kind() {
                TokenKind::Dot => {
                    self.advance();
                    let attr_token = self.expect_name()?;
                    let start = expr.range().start.into();
                    let end = attr_token.2;
                    expr = Expr::Attribute(ExprAttribute {
                        value: Box::new(expr),
                        attr: Identifier(attr_token.0),
                        range: TextRange::new(start, end),
                    });
                }
                TokenKind::LParen => {
                    let start = expr.range().start.into();
                    self.advance();
                    let (args, keywords) = self.parse_call_args()?;
                    let close = self.expect(&TokenKind::RParen)?;
                    expr = Expr::Call(ExprCall {
                        func: Box::new(expr),
                        args,
                        keywords,
                        range: TextRange::new(start, close.end),
                    });
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    /// Parse call arguments: positional `expr` and keyword `name=expr`.
    fn parse_call_args(&mut self) -> Result<(Vec<Expr>, Vec<Keyword>), ParseError> {
        let mut args = Vec::new();
        let mut keywords = Vec::new();
        loop {
            if matches!(self.peek_kind(), TokenKind::RParen) {
                break;
            }
            // Detect `name = value` keyword argument.
            if let TokenKind::Name(name) = self.peek_kind().clone() {
                if matches!(self.tokens[self.pos + 1].kind, TokenKind::Equals) {
                    let name_token = self.advance(); // name
                    self.advance(); // `=`
                    let value = self.parse_expr()?;
                    let end = value.range().end.into();
                    keywords.push(Keyword {
                        arg: Some(Identifier(name)),
                        value,
                        range: TextRange::new(name_token.start, end),
                    });
                    if matches!(self.peek_kind(), TokenKind::Comma) {
                        self.advance();
                        continue;
                    }
                    break;
                }
            }
            let value = self.parse_expr()?;
            args.push(value);
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.advance();
                continue;
            }
            break;
        }
        Ok((args, keywords))
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let token = self.peek().clone();
        match &token.kind {
            TokenKind::Name(name) => {
                self.advance();
                Ok(Expr::Name(ExprName {
                    id: Identifier(name.clone()),
                    range: TextRange::new(token.start, token.end),
                }))
            }
            TokenKind::Keyword(word) if word == "True" || word == "False" => {
                self.advance();
                Ok(Expr::Constant(ExprConstant {
                    value: Constant::Bool(word == "True"),
                    range: TextRange::new(token.start, token.end),
                }))
            }
            TokenKind::Keyword(word) if word == "None" => {
                self.advance();
                Ok(Expr::Constant(ExprConstant {
                    value: Constant::None,
                    range: TextRange::new(token.start, token.end),
                }))
            }
            TokenKind::Str(value) => {
                self.advance();
                Ok(Expr::Constant(ExprConstant {
                    value: Constant::Str(value.clone()),
                    range: TextRange::new(token.start, token.end),
                }))
            }
            TokenKind::Int(text) => {
                self.advance();
                Ok(Expr::Constant(ExprConstant {
                    value: Constant::Int(text.clone()),
                    range: TextRange::new(token.start, token.end),
                }))
            }
            TokenKind::Float(value) => {
                self.advance();
                Ok(Expr::Constant(ExprConstant {
                    value: Constant::Float(*value),
                    range: TextRange::new(token.start, token.end),
                }))
            }
            TokenKind::LBracket => self.parse_list(),
            TokenKind::LParen => self.parse_paren_or_tuple(),
            _ => Err(ParseError::at(token.start)),
        }
    }

    fn parse_list(&mut self) -> Result<Expr, ParseError> {
        let open = self.advance(); // `[`
        let mut elts = Vec::new();
        loop {
            if matches!(self.peek_kind(), TokenKind::RBracket) {
                break;
            }
            elts.push(self.parse_expr()?);
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.advance();
                continue;
            }
            break;
        }
        let close = self.expect(&TokenKind::RBracket)?;
        Ok(Expr::List(ExprList {
            elts,
            range: TextRange::new(open.start, close.end),
        }))
    }

    /// `(expr)` -> the inner expr; `(a, b, ...)` -> a Tuple.
    fn parse_paren_or_tuple(&mut self) -> Result<Expr, ParseError> {
        let open = self.advance(); // `(`
        let mut elts = Vec::new();
        let mut had_comma = false;
        loop {
            if matches!(self.peek_kind(), TokenKind::RParen) {
                break;
            }
            elts.push(self.parse_expr()?);
            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.advance();
                had_comma = true;
                continue;
            }
            break;
        }
        let close = self.expect(&TokenKind::RParen)?;
        if !had_comma && elts.len() == 1 {
            // Parenthesised single expression, not a tuple.
            Ok(elts.into_iter().next().unwrap())
        } else {
            Ok(Expr::Tuple(ExprTuple {
                elts,
                range: TextRange::new(open.start, close.end),
            }))
        }
    }

    // ----- Helpers -----

    /// Expect an identifier (or a soft keyword usable as a name). Returns the
    /// `(text, start, end)` triple.
    fn expect_name(&mut self) -> Result<(String, usize, usize), ParseError> {
        match self.peek_kind().clone() {
            TokenKind::Name(name) => {
                let token = self.advance();
                Ok((name, token.start, token.end))
            }
            _ => Err(ParseError::at(self.peek().start)),
        }
    }

    /// End offset of the most recently consumed token.
    fn previous_end(&self) -> usize {
        if self.pos == 0 {
            0
        } else {
            self.tokens[self.pos - 1].end
        }
    }
}
