//! Source frontends (spec VII.6).
//!
//! The compatibility path still accepts the original line-oriented grammar:
//!
//! ```text
//!   input  <name> [:<shape>]              — graph input port (f32)
//!   const  <name> :<shape> = v0,v1,...    — f32 constant tensor
//!   op     <op_name> <input...> [:<shape>] [as=<alias>]
//!   output <name>                         — graph output port
//! ```
//!
//! New frontends should parse into [`SourceProgram`] and reuse the shared
//! `SourceProgram -> Graph` lowerer rather than allocating graph nodes directly.

mod attrs;
mod diagnostic;
mod document;
mod frontend;
mod frontends;
mod ir;
mod lower;
mod op_table;

pub use attrs::op_attr_names;
pub use diagnostic::SourceDiagnostic;
pub use document::{SourceDocument, SourceGraph, SourceParseOptions};
pub use frontend::{SourceFrontend, SourceFrontendInfo};
pub use frontends::{HologramFrontend, PythonFrontend, RustFrontend, TypeScriptFrontend};
pub use ir::{
    SourceAttrs, SourceBinding, SourceConst, SourceExpr, SourceExternalConst, SourceExternalTensor,
    SourceExternalTensorLocation, SourceInput, SourceItem, SourceOpCall, SourceOutput,
    SourceProgram, SourceSpan, SourceSymbol, SourceTensorLiteral, SourceType,
};

use crate::error::CompileError;
use hologram_graph::Graph;

/// Source language accepted by the compiler frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLanguage {
    /// Native Hologram source language.
    Hologram,
    /// Restricted Python builder frontend.
    Python,
    /// Restricted TypeScript builder frontend.
    TypeScript,
    /// Restricted Rust builder frontend.
    Rust,
}

/// Parse native Hologram source and lower it directly to a graph.
pub fn parse(source: &str) -> Result<Graph, CompileError> {
    let program = parse_ir(source, SourceLanguage::Hologram)?;
    lower::lower_program(program)
}

/// Parse source text into a document containing one or more graph regions.
pub fn parse_document(
    source: &str,
    language: SourceLanguage,
) -> Result<SourceDocument, CompileError> {
    frontends::parse_document(source, language)
}

/// Parse source text into a document with a source-position diagnostic.
pub fn parse_document_diagnostic(
    source: &str,
    language: SourceLanguage,
) -> Result<SourceDocument, SourceDiagnostic> {
    frontends::parse_document_diagnostic(source, language)
}

/// Parse source text into the common source IR.
pub fn parse_ir(source: &str, language: SourceLanguage) -> Result<SourceProgram, CompileError> {
    parse_ir_with_options(source, language, &SourceParseOptions::default())
}

/// Parse source text into source IR with graph-selection options.
pub fn parse_ir_with_options(
    source: &str,
    language: SourceLanguage,
    options: &SourceParseOptions,
) -> Result<SourceProgram, CompileError> {
    frontends::parse_ir_with_options(source, language, options)
}

/// Parse source text into source IR with a source-position diagnostic.
pub fn parse_ir_diagnostic(
    source: &str,
    language: SourceLanguage,
) -> Result<SourceProgram, SourceDiagnostic> {
    parse_ir_diagnostic_with_options(source, language, &SourceParseOptions::default())
}

/// Parse source text into source IR with options and a source-position diagnostic.
pub fn parse_ir_diagnostic_with_options(
    source: &str,
    language: SourceLanguage,
    options: &SourceParseOptions,
) -> Result<SourceProgram, SourceDiagnostic> {
    frontends::parse_ir_diagnostic_with_options(source, language, options)
}

/// Lower a source program into the graph IR.
pub fn lower_ir(program: &SourceProgram) -> Result<Graph, CompileError> {
    lower::lower_ir(program)
}

/// Resolve a source language from an explicit name or optional extension.
pub fn resolve_source_language(
    explicit: Option<&str>,
    extension: Option<&str>,
) -> Result<SourceLanguage, CompileError> {
    match explicit {
        Some(language) if !language.eq_ignore_ascii_case("auto") => {
            source_language_from_name(language)
                .ok_or(CompileError::SourceParse("unknown source language"))
        }
        _ => Ok(source_language_from_optional_extension(extension)),
    }
}

/// Resolve a source language from a frontend name or alias.
pub fn source_language_from_name(name: &str) -> Option<SourceLanguage> {
    frontends::language_from_name(name)
}

/// Resolve a source language from a frontend filename extension.
pub fn source_language_from_extension(extension: &str) -> Option<SourceLanguage> {
    frontends::language_from_extension(extension)
}

fn source_language_from_optional_extension(extension: Option<&str>) -> SourceLanguage {
    extension
        .and_then(source_language_from_extension)
        .unwrap_or(SourceLanguage::Hologram)
}
