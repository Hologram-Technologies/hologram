//! Source frontend adapter boundary.

use crate::error::CompileError;
use crate::source::{
    diagnostic, SourceDiagnostic, SourceDocument, SourceLanguage, SourceParseOptions, SourceProgram,
};

/// Static metadata for one source frontend.
#[derive(Debug, Clone, Copy)]
pub struct SourceFrontendInfo {
    language: SourceLanguage,
    names: &'static [&'static str],
    extensions: &'static [&'static str],
}

impl SourceFrontendInfo {
    /// Construct frontend metadata.
    pub const fn new(
        language: SourceLanguage,
        names: &'static [&'static str],
        extensions: &'static [&'static str],
    ) -> Self {
        Self {
            language,
            names,
            extensions,
        }
    }

    /// Source language handled by this frontend.
    pub const fn language(self) -> SourceLanguage {
        self.language
    }

    /// Accepted CLI names and aliases.
    pub const fn names(self) -> &'static [&'static str] {
        self.names
    }

    /// Accepted filename extensions without a leading dot.
    pub const fn extensions(self) -> &'static [&'static str] {
        self.extensions
    }

    /// Return whether `name` is one of this frontend's aliases.
    pub fn matches_name(self, name: &str) -> bool {
        self.names
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(name))
    }

    /// Return whether `extension` is one of this frontend's extensions.
    pub fn matches_extension(self, extension: &str) -> bool {
        let extension = extension.strip_prefix('.').unwrap_or(extension);
        self.extensions
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(extension))
    }
}

/// Parser adapter that extracts graph regions from one source language.
pub trait SourceFrontend: Default {
    /// Static language names, aliases, and extensions for this frontend.
    const INFO: SourceFrontendInfo;

    /// Parse source text into a document of graph regions.
    fn parse_document(&self, source: &str) -> Result<SourceDocument, CompileError>;

    /// Parse source text into a document with source-position diagnostics.
    fn parse_document_diagnostic(&self, source: &str) -> Result<SourceDocument, SourceDiagnostic> {
        self.parse_document(source)
            .map_err(|err| SourceDiagnostic::global(diagnostic::compile_error_kind(&err)))
    }

    /// Parse source text into the common source IR.
    fn parse_ir(&self, source: &str) -> Result<SourceProgram, CompileError> {
        self.parse_document(source)?
            .select(&SourceParseOptions::default())
    }

    /// Parse source text into source IR with a source-position diagnostic.
    fn parse_ir_diagnostic(&self, source: &str) -> Result<SourceProgram, SourceDiagnostic> {
        self.parse_document_diagnostic(source)?
            .select_diagnostic(&SourceParseOptions::default())
    }
}
