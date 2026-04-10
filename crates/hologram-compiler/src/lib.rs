//! Structure-finder compilation to `.holo` archive.
//!
//! Per Prism section 4 of the SCS framework, the compiler is a
//! *structure-finder*, not a constructor. The pipeline runs as a flat
//! sequence of finder passes (see [`compiler`]).
//!
//! Entry points:
//!
//! - [`compile(graph)`](compile) — compile a raw Graph (wraps in synthetic CompileUnit)
//! - [`compile_from_source(source, ...)`](compile_from_source) — compile from UOR term language text
//! - [`Compiler::compile`](crate::compiler::Compiler::compile) — explicit
//!   compile method taking a [`SourceInput`](crate::compiler::SourceInput)
//!
//! Additionally provides:
//! - **[`term_parser`]**: UOR term language → arena-allocated term graph
//! - **[`preflight`]**: CompileUnit admission (shape validation, budget solvency, unit address)
//! - **[`term_lower`]**: CompileUnit → Graph lowering

pub mod compiler;
pub mod error;
pub mod preflight;
pub mod term_lower;
pub mod term_parser;

pub use compiler::{
    compile, unit_from_graph, unit_from_graph_with, CompilationOutput, CompilationStats, Compiler,
    PrismModuleRegistry, SourceInput,
};
pub use error::{CompileError, CompileResult};

use hologram_foundation::enums::VerificationDomain;
use hologram_foundation::WittLevel;

/// Compile from UOR term language source text.
///
/// Convenience wrapper around
/// `Compiler::default().compile(SourceInput::TermSource { ... })`.
pub fn compile_from_source(
    source: &str,
    witt_level: WittLevel,
    thermodynamic_budget: f64,
    target_domains: &[VerificationDomain],
) -> CompileResult<CompilationOutput> {
    Compiler::default().compile(SourceInput::TermSource {
        source: source.to_owned(),
        level: witt_level,
        budget: thermodynamic_budget,
        domains: target_domains.to_vec(),
    })
}
