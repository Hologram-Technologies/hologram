//! Language-specific source frontends.

mod hologram;
#[cfg(feature = "frontend-python")]
mod pyparse;
mod python;
mod rust;
mod typescript;

use crate::error::CompileError;
use crate::source::frontend::{SourceFrontend, SourceFrontendInfo};
use crate::source::{
    SourceDiagnostic, SourceDocument, SourceLanguage, SourceParseOptions, SourceProgram,
};
pub use hologram::HologramFrontend;
use hologram_graph::Graph;
pub use python::PythonFrontend;
pub use rust::RustFrontend;
pub use typescript::TypeScriptFrontend;

struct RegisteredFrontend {
    info: SourceFrontendInfo,
    parse_document: fn(&str) -> Result<SourceDocument, CompileError>,
    parse_document_diagnostic: fn(&str) -> Result<SourceDocument, SourceDiagnostic>,
}

impl RegisteredFrontend {
    const fn new(
        info: SourceFrontendInfo,
        parse_document: fn(&str) -> Result<SourceDocument, CompileError>,
        parse_document_diagnostic: fn(&str) -> Result<SourceDocument, SourceDiagnostic>,
    ) -> Self {
        Self {
            info,
            parse_document,
            parse_document_diagnostic,
        }
    }

    fn parse_document(&self, source: &str) -> Result<SourceDocument, CompileError> {
        (self.parse_document)(source)
    }

    fn parse_document_diagnostic(&self, source: &str) -> Result<SourceDocument, SourceDiagnostic> {
        (self.parse_document_diagnostic)(source)
    }
}

const FRONTENDS: &[RegisteredFrontend] = &[
    RegisteredFrontend::new(
        HologramFrontend::INFO,
        parse_document_with::<HologramFrontend>,
        parse_document_diagnostic_with::<HologramFrontend>,
    ),
    RegisteredFrontend::new(
        PythonFrontend::INFO,
        parse_document_with::<PythonFrontend>,
        parse_document_diagnostic_with::<PythonFrontend>,
    ),
    RegisteredFrontend::new(
        TypeScriptFrontend::INFO,
        parse_document_with::<TypeScriptFrontend>,
        parse_document_diagnostic_with::<TypeScriptFrontend>,
    ),
    RegisteredFrontend::new(
        RustFrontend::INFO,
        parse_document_with::<RustFrontend>,
        parse_document_diagnostic_with::<RustFrontend>,
    ),
];

/// Parse source text into a document with the selected language frontend.
pub fn parse_document(
    source: &str,
    language: SourceLanguage,
) -> Result<SourceDocument, CompileError> {
    frontend_for_language(language)
        .ok_or(CompileError::SourceParse("source language unsupported"))?
        .parse_document(source)
}

pub(crate) fn looks_like_hologram_v2(source: &str) -> bool {
    hologram::looks_like_v2(source)
}

pub(crate) fn parse_legacy_hologram_graph(source: &str) -> Result<Graph, CompileError> {
    hologram::parse_legacy_graph(source)
}

/// Parse source text into a document with source-position diagnostics.
pub fn parse_document_diagnostic(
    source: &str,
    language: SourceLanguage,
) -> Result<SourceDocument, SourceDiagnostic> {
    frontend_for_language(language)
        .ok_or_else(|| SourceDiagnostic::global("source language unsupported"))?
        .parse_document_diagnostic(source)
}

/// Parse source text with the selected language frontend and options.
pub fn parse_ir_with_options(
    source: &str,
    language: SourceLanguage,
    options: &SourceParseOptions,
) -> Result<SourceProgram, CompileError> {
    parse_document(source, language)?.select(options)
}

/// Parse source text with the selected language frontend, options, and diagnostics.
pub fn parse_ir_diagnostic_with_options(
    source: &str,
    language: SourceLanguage,
    options: &SourceParseOptions,
) -> Result<SourceProgram, SourceDiagnostic> {
    parse_document_diagnostic(source, language)?.select_diagnostic(options)
}

/// Resolve a frontend by CLI language name or alias.
pub fn language_from_name(name: &str) -> Option<SourceLanguage> {
    FRONTENDS
        .iter()
        .find(|frontend| frontend.info.matches_name(name))
        .map(|frontend| frontend.info.language())
}

/// Resolve a frontend by filename extension.
pub fn language_from_extension(extension: &str) -> Option<SourceLanguage> {
    FRONTENDS
        .iter()
        .find(|frontend| frontend.info.matches_extension(extension))
        .map(|frontend| frontend.info.language())
}

fn frontend_for_language(language: SourceLanguage) -> Option<&'static RegisteredFrontend> {
    FRONTENDS
        .iter()
        .find(|frontend| frontend.info.language() == language)
}

fn parse_document_with<F>(source: &str) -> Result<SourceDocument, CompileError>
where
    F: SourceFrontend,
{
    F::default().parse_document(source)
}

fn parse_document_diagnostic_with<F>(source: &str) -> Result<SourceDocument, SourceDiagnostic>
where
    F: SourceFrontend,
{
    F::default().parse_document_diagnostic(source)
}
