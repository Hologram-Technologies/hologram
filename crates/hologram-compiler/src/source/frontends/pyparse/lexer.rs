//! Byte-offset-tracking lexer for the restricted Python subset.
//!
//! Emits identifiers, keywords, int/float literals, string literals (single &
//! double quoted, including triple-quoted), the punctuation `( ) [ ] , . : =`
//! and the unary sign tokens `+ -`, plus `NEWLINE` and `INDENT`/`DEDENT`
//! computed from an indentation stack. `#` line comments and blank lines are
//! ignored. Line continuations inside brackets suppress `NEWLINE`.

use alloc::string::String;
use alloc::vec::Vec;

use super::ParseError;

/// Keywords recognised as [`TokenKind::Keyword`]. Everything else is a name.
/// `True`/`False`/`None` are kept as keywords and handled by the parser as
/// constants; the frontend never uses them as identifiers.
const KEYWORDS: &[&str] = &[
    "def", "pass", "if", "elif", "else", "for", "while", "with", "try", "except", "finally",
    "return", "class", "import", "from", "as", "in", "is", "and", "or", "not", "lambda", "yield",
    "global", "nonlocal", "assert", "del", "raise", "break", "continue", "async", "await", "True",
    "False", "None",
];

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Name(String),
    Keyword(String),
    Int(String),
    Float(f64),
    Str(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,
    Colon,
    Equals,
    Plus,
    Minus,
    Newline,
    Indent,
    Dedent,
    Eof,
}

/// A token carrying its byte range `[start, end)` into the source.
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub start: usize,
    pub end: usize,
}

pub struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    /// Indentation column stack; always starts with 0.
    indents: Vec<usize>,
    /// Bracket nesting depth; newlines are suppressed while > 0.
    bracket_depth: usize,
    /// True at the start of a logical line (need to measure indentation).
    at_line_start: bool,
    tokens: Vec<Token>,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            indents: Vec::from([0]),
            bracket_depth: 0,
            at_line_start: true,
            tokens: Vec::new(),
        }
    }

    fn push(&mut self, kind: TokenKind, start: usize, end: usize) {
        self.tokens.push(Token { kind, start, end });
    }

    fn peek_byte(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn byte_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    /// Tokenize the entire source, appending a trailing EOF.
    pub fn tokenize(mut self) -> Result<Vec<Token>, ParseError> {
        while self.pos < self.bytes.len() {
            if self.at_line_start && self.bracket_depth == 0 && self.handle_line_start()? {
                continue;
            }
            let byte = self.bytes[self.pos];
            match byte {
                b' ' | b'\t' | b'\r' => {
                    self.pos += 1;
                }
                b'\n' => {
                    self.emit_newline();
                }
                b'\\' if self.byte_at(1) == Some(b'\n') => {
                    // Explicit line continuation.
                    self.pos += 2;
                }
                b'#' => {
                    self.skip_comment();
                }
                b'(' => self.punct(TokenKind::LParen),
                b')' => self.punct(TokenKind::RParen),
                b'[' => self.punct(TokenKind::LBracket),
                b']' => self.punct(TokenKind::RBracket),
                b',' => self.punct(TokenKind::Comma),
                b'.' if !self.next_is_digit() => self.punct(TokenKind::Dot),
                b':' => self.punct(TokenKind::Colon),
                b'=' => self.punct(TokenKind::Equals),
                b'+' => self.punct(TokenKind::Plus),
                b'-' => self.punct(TokenKind::Minus),
                b'"' | b'\'' => self.lex_string()?,
                b'0'..=b'9' => self.lex_number(),
                b'.' => self.lex_number(),
                _ if is_ident_start(byte) => self.lex_ident(),
                _ => {
                    return Err(ParseError {
                        offset: super::TextSize::new(self.pos),
                    })
                }
            }
        }
        self.finish()
    }

    /// Emit a NEWLINE (if the logical line had content) and mark line start.
    fn emit_newline(&mut self) {
        let start = self.pos;
        self.pos += 1;
        if self.line_has_content() {
            self.push(TokenKind::Newline, start, self.pos);
        }
        self.at_line_start = true;
    }

    /// True if the current logical line has already emitted a real token.
    fn line_has_content(&self) -> bool {
        matches!(
            self.tokens.last().map(|token| &token.kind),
            Some(kind) if !matches!(kind, TokenKind::Newline | TokenKind::Indent | TokenKind::Dedent)
        )
    }

    /// Measure indentation at the start of a logical line and emit
    /// INDENT/DEDENT tokens. Returns true if the line was blank/comment-only and
    /// should be skipped.
    fn handle_line_start(&mut self) -> Result<bool, ParseError> {
        let mut col = 0usize;
        let mut cursor = self.pos;
        while let Some(byte) = self.bytes.get(cursor).copied() {
            match byte {
                b' ' => {
                    col += 1;
                    cursor += 1;
                }
                b'\t' => {
                    // Tabs advance to the next multiple of 8, matching CPython.
                    col += 8 - (col % 8);
                    cursor += 1;
                }
                _ => break,
            }
        }
        // Blank line or comment-only line: skip without emitting indentation.
        match self.bytes.get(cursor).copied() {
            None => {
                self.pos = cursor;
                self.at_line_start = false;
                return Ok(false);
            }
            Some(b'\n') => {
                self.pos = cursor + 1;
                return Ok(true);
            }
            Some(b'\r') => {
                self.pos = cursor + 1;
                return Ok(true);
            }
            Some(b'#') => {
                self.pos = cursor;
                self.skip_comment();
                // After the comment we are at a newline or EOF; loop again.
                return Ok(true);
            }
            _ => {}
        }

        self.pos = cursor;
        self.at_line_start = false;

        let current = *self.indents.last().unwrap();
        if col > current {
            self.indents.push(col);
            self.push(TokenKind::Indent, cursor, cursor);
        } else if col < current {
            while *self.indents.last().unwrap() > col {
                self.indents.pop();
                self.push(TokenKind::Dedent, cursor, cursor);
            }
            if *self.indents.last().unwrap() != col {
                // Inconsistent dedent.
                return Err(ParseError {
                    offset: super::TextSize::new(cursor),
                });
            }
        }
        Ok(false)
    }

    fn skip_comment(&mut self) {
        while let Some(byte) = self.peek_byte() {
            if byte == b'\n' {
                break;
            }
            self.pos += 1;
        }
    }

    fn punct(&mut self, kind: TokenKind) {
        let start = self.pos;
        self.pos += 1;
        match &kind {
            TokenKind::LParen | TokenKind::LBracket => self.bracket_depth += 1,
            TokenKind::RParen | TokenKind::RBracket => {
                self.bracket_depth = self.bracket_depth.saturating_sub(1);
            }
            _ => {}
        }
        self.push(kind, start, self.pos);
    }

    fn next_is_digit(&self) -> bool {
        matches!(self.byte_at(1), Some(b'0'..=b'9'))
    }

    fn lex_ident(&mut self) {
        let start = self.pos;
        while let Some(byte) = self.peek_byte() {
            if is_ident_continue(byte) {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = &self.src[start..self.pos];
        let kind = if KEYWORDS.contains(&text) {
            TokenKind::Keyword(String::from(text))
        } else {
            TokenKind::Name(String::from(text))
        };
        self.push(kind, start, self.pos);
    }

    fn lex_number(&mut self) {
        let start = self.pos;
        let mut is_float = false;
        // Integer/float digits, underscores, a decimal point, and exponents.
        while let Some(byte) = self.peek_byte() {
            match byte {
                b'0'..=b'9' | b'_' => self.pos += 1,
                b'.' => {
                    is_float = true;
                    self.pos += 1;
                }
                b'e' | b'E' => {
                    is_float = true;
                    self.pos += 1;
                    if matches!(self.peek_byte(), Some(b'+' | b'-')) {
                        self.pos += 1;
                    }
                }
                b'x' | b'X' | b'o' | b'O' | b'b' | b'B' | b'a'..=b'f' | b'A'..=b'F'
                    if self.pos > start =>
                {
                    // Hex/oct/bin digits and prefixes; treated as integer text.
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let text: String = self.src[start..self.pos]
            .chars()
            .filter(|&c| c != '_')
            .collect();
        if is_float {
            let value = text.parse::<f64>().unwrap_or(0.0);
            self.push(TokenKind::Float(value), start, self.pos);
        } else {
            self.push(TokenKind::Int(text), start, self.pos);
        }
    }

    fn lex_string(&mut self) -> Result<(), ParseError> {
        let start = self.pos;
        let quote = self.bytes[self.pos];
        let triple = self.byte_at(1) == Some(quote) && self.byte_at(2) == Some(quote);
        let mut value = String::new();
        if triple {
            self.pos += 3;
            loop {
                match self.peek_byte() {
                    None => {
                        return Err(ParseError {
                            offset: super::TextSize::new(start),
                        });
                    }
                    Some(b)
                        if b == quote
                            && self.byte_at(1) == Some(quote)
                            && self.byte_at(2) == Some(quote) =>
                    {
                        self.pos += 3;
                        break;
                    }
                    Some(b'\\') => self.push_escaped(&mut value),
                    Some(_) => self.push_char(&mut value),
                }
            }
        } else {
            self.pos += 1;
            loop {
                match self.peek_byte() {
                    None | Some(b'\n') => {
                        return Err(ParseError {
                            offset: super::TextSize::new(start),
                        });
                    }
                    Some(b) if b == quote => {
                        self.pos += 1;
                        break;
                    }
                    Some(b'\\') => self.push_escaped(&mut value),
                    Some(_) => self.push_char(&mut value),
                }
            }
        }
        self.push(TokenKind::Str(value), start, self.pos);
        Ok(())
    }

    /// Consume a backslash escape and append the resolved character.
    fn push_escaped(&mut self, value: &mut String) {
        self.pos += 1; // backslash
        match self.peek_byte() {
            Some(b'n') => {
                value.push('\n');
                self.pos += 1;
            }
            Some(b't') => {
                value.push('\t');
                self.pos += 1;
            }
            Some(b'r') => {
                value.push('\r');
                self.pos += 1;
            }
            Some(b'\\') => {
                value.push('\\');
                self.pos += 1;
            }
            Some(b'\'') => {
                value.push('\'');
                self.pos += 1;
            }
            Some(b'"') => {
                value.push('"');
                self.pos += 1;
            }
            Some(b'0') => {
                value.push('\0');
                self.pos += 1;
            }
            Some(b'\n') => {
                // Line continuation inside a string.
                self.pos += 1;
            }
            Some(_) => {
                // Unknown escape: keep the backslash literally then the char.
                value.push('\\');
                self.push_char(value);
            }
            None => {}
        }
    }

    /// Append the next UTF-8 character, advancing `pos` by its byte length.
    fn push_char(&mut self, value: &mut String) {
        let rest = &self.src[self.pos..];
        if let Some(ch) = rest.chars().next() {
            value.push(ch);
            self.pos += ch.len_utf8();
        } else {
            self.pos += 1;
        }
    }

    /// Emit a trailing NEWLINE, close open indentation, and append EOF.
    fn finish(mut self) -> Result<Vec<Token>, ParseError> {
        let end = self.bytes.len();
        if self.line_has_content() {
            self.push(TokenKind::Newline, end, end);
        }
        while *self.indents.last().unwrap() > 0 {
            self.indents.pop();
            self.push(TokenKind::Dedent, end, end);
        }
        self.push(TokenKind::Eof, end, end);
        Ok(self.tokens)
    }
}

fn is_ident_start(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic() || byte >= 0x80
}

fn is_ident_continue(byte: u8) -> bool {
    is_ident_start(byte) || byte.is_ascii_digit()
}
