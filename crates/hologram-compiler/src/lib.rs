//! Compilation pipeline: cascade-backed compilation to .holo archive.
//!
//! All compilation routes through the 7-stage cascade engine with
//! certificate memoization. Entry points:
//!
//! - `compile(graph)` — compile a raw Graph (wraps in synthetic CompileUnit)
//! - `compile_from_source(source, ...)` — compile from UOR term language text
//! - `CompilerBuilder` — configurable compilation with shared certificate stores
//!
//! Additionally provides:
//! - **term_parser**: UOR term language → arena-allocated term graph
//! - **preflight**: CompileUnit admission (shape validation, CS_6, CS_7)
//! - **term_lower**: CompileUnit → Graph lowering

pub mod compiler;
pub mod error;
pub mod preflight;
pub mod term_lower;
pub mod term_parser;

pub use compiler::{
    compile, unit_from_graph, unit_from_graph_with, CascadeInfo, CompilationOutput,
    CompilationStats, CompilerBuilder,
};
pub use error::{CompileError, CompileResult};
pub use hologram_cascade::certificate::CertificateStore;

use uor_foundation::enums::VerificationDomain;
use uor_foundation::WittLevel as QuantumLevel;

/// Compile from UOR term language source text.
///
/// Convenience wrapper around `CompilerBuilder::from_source(...).build()`.
pub fn compile_from_source(
    source: &str,
    quantum_level: QuantumLevel,
    thermodynamic_budget: f64,
    target_domains: &[VerificationDomain],
) -> CompileResult<CompilationOutput> {
    CompilerBuilder::from_source(source, quantum_level, thermodynamic_budget, target_domains)
        .build()
}
