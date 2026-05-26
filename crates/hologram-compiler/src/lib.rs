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
//!
//! `no_std` + `alloc` by default (matching prism / uor-addr) so the whole
//! compile pipeline runs in wasm and on embedded targets; the `std` feature
//! only adds `tracing` diagnostics.

#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

pub mod cache;
pub mod compiler;
pub mod error;
pub mod lower;
pub mod pipeline;
pub mod source;

pub use cache::{CachedCertificate, CertificateCache};
pub use compiler::{BackendKind, CompilationOutput, CompilationStats, Compiler};
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
    use hologram_graph::node::Node;
    use hologram_graph::{GraphOp, InputSource};
    use smallvec::SmallVec;

    // Desugar composites first so backward differentiates their primitive
    // pipelines (e.g. SwiGLU → MatMul·Silu·MatMul·Mul), reusing those VJPs
    // rather than needing a separate composite gradient.
    graph.desugar_composites();
    let input_grads = hologram_graph::append_backward(&mut graph, output_id)
        .map_err(|_| CompileError::CompletenessFailure)?;
    // Gradients are outputs of the backward graph: expose each as a graph
    // output port so it is materialized and retained. The elision pass roots
    // dead-node elimination at the output set, so a gradient that fed nothing
    // else would otherwise be (correctly) pruned as unreachable.
    for &g in &input_grads {
        let (dt, sh) = graph
            .get(g)
            .map(|n| (n.output_dtype, n.output_shape))
            .unwrap_or((hologram_graph::registry::DTypeId(0), hologram_graph::registry::ShapeId(0)));
        let out = graph.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(g)]),
            output_dtype: dt,
            output_shape: sh,
        });
        graph.add_output(out);
    }
    let output = Compiler::new(graph, target, level).compile()?;
    Ok((output, input_grads))
}
