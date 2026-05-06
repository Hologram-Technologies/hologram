//! Hologram compiler (spec Part VII).
//!
//! Per-node CompileUnit pipeline:
//!   1. Lookup op marker for node.op_kind.
//!   2. Resolve concrete shape/dtype/host-bounds generics.
//!   3. Emit Term tree into TermArena.
//!   4. Build CompileUnit via CompileUnitBuilder.
//!   5. Validate (CompileUnitBuilder::validate).
//!   6. Run completeness (pipeline::run_tower_completeness).
//!   7. Cache by ContentFingerprint<32>.
//!   8. Lower to backend KernelCall.
//!   9. Emit (kernel_call, certificate, fingerprint) into archive.

pub mod compiler;
pub mod cache;
pub mod lower;
pub mod pipeline;
pub mod source;
pub mod error;

pub use compiler::{Compiler, BackendKind, CompilationOutput, CompilationStats};
pub use cache::{CertificateCache, CachedCertificate};
pub use error::CompileError;

/// Convenience: parse UOR source -> Graph -> compile.
pub fn compile_from_source(
    source: &str,
    level: uor_foundation::WittLevel,
    target: BackendKind,
) -> Result<CompilationOutput, CompileError> {
    let graph = source::parse(source)?;
    Compiler::new(graph, target, level).compile()
}

/// Convenience: compile a pre-built graph.
pub fn compile(
    graph: hologram_graph::Graph,
    target: BackendKind,
    level: uor_foundation::WittLevel,
) -> Result<CompilationOutput, CompileError> {
    Compiler::new(graph, target, level).compile()
}

/// Compile a forward graph with an attached backward subgraph
/// (spec V.4 / ADR-043). Backward is *planned* — gradient nodes are
/// appended to `graph` ahead of time, then the augmented graph is
/// compiled normally. Returns the gradient `NodeId`s alongside the
/// compiled archive.
pub fn compile_with_backward(
    mut graph: hologram_graph::Graph,
    output_id: hologram_graph::NodeId,
    target: BackendKind,
    level: uor_foundation::WittLevel,
) -> Result<(CompilationOutput, alloc::vec::Vec<hologram_graph::NodeId>), CompileError> {
    let input_grads = hologram_graph::append_backward(&mut graph, output_id)
        .map_err(|_| CompileError::CompletenessFailure)?;
    let output = Compiler::new(graph, target, level).compile()?;
    Ok((output, input_grads))
}

extern crate alloc;
