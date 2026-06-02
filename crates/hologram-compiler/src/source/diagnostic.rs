//! Source parser diagnostics.

use alloc::string::{String, ToString};

use crate::error::CompileError;

/// Source parse diagnostic with a stable error kind and source position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDiagnostic {
    /// 1-based source line.
    pub line: usize,
    /// 1-based source column.
    pub column: usize,
    /// Stable diagnostic kind.
    pub kind: &'static str,
    /// Rejected source token or fragment.
    pub rejected: String,
}

impl SourceDiagnostic {
    /// Build a source diagnostic.
    pub fn new(line: usize, column: usize, kind: &'static str, rejected: String) -> Self {
        Self {
            line,
            column,
            kind,
            rejected,
        }
    }

    /// Build a diagnostic for a whole input when no precise line exists.
    pub fn global(kind: &'static str) -> Self {
        Self::new(1, 1, kind, String::new())
    }

    /// Convert back into the existing compatibility error shape.
    pub fn into_compile_error(self) -> CompileError {
        CompileError::SourceParse(self.kind)
    }
}

/// Build a diagnostic from a parser remainder.
pub(crate) fn from_remainder(
    line: usize,
    base_column: usize,
    parsed: &str,
    remainder: &str,
    kind: &'static str,
) -> SourceDiagnostic {
    let offset = parsed.len().saturating_sub(remainder.len());
    let leading = remainder.len() - remainder.trim_start().len();
    let column = base_column + offset + leading;
    SourceDiagnostic::new(line, column, kind, rejected_fragment(remainder))
}

/// Build a diagnostic for a line-level semantic parse error.
pub(crate) fn from_line(
    line: usize,
    base_column: usize,
    parsed: &str,
    kind: &'static str,
) -> SourceDiagnostic {
    SourceDiagnostic::new(line, base_column, kind, rejected_fragment(parsed))
}

/// Return the stable source-parse kind for an existing compile error.
pub(crate) fn compile_error_kind(err: &CompileError) -> &'static str {
    match err {
        CompileError::SourceParse(kind) => kind,
        _ => "source: parse failed",
    }
}

fn rejected_fragment(remainder: &str) -> String {
    let trimmed = remainder.trim_start();
    if trimmed.is_empty() {
        return "<eol>".to_string();
    }
    trimmed
        .split_whitespace()
        .next()
        .unwrap_or(trimmed)
        .to_string()
}
