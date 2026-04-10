//! Async wrapper for the hologram compilation pipeline.

use hologram_compiler::{compile, CompilationOutput, CompileResult};
use hologram_ir::Graph;
use tokio::task::JoinHandle;

/// Async wrapper around the hologram compiler.
///
/// Runs the compilation pipeline on a blocking thread via
/// `tokio::task::spawn_blocking`, returning a `JoinHandle` the caller
/// can `.await` from any async context.
pub struct AsyncCompiler {
    graph: Graph,
}

impl AsyncCompiler {
    /// Create a new async compiler for the given graph.
    #[must_use]
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    /// Spawn compilation on a blocking thread and return a `JoinHandle`.
    pub fn compile(self) -> JoinHandle<CompileResult<CompilationOutput>> {
        tokio::task::spawn_blocking(move || compile(self.graph))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_ir::builder::GraphBuilder;
    use hologram_ir::graph::GraphOp;

    fn simple_graph() -> Graph {
        GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Output, &[0])
            .output("y", 1)
            .build()
    }

    #[tokio::test]
    async fn compile_round_trip() {
        let output = AsyncCompiler::new(simple_graph())
            .compile()
            .await
            .unwrap()
            .unwrap();
        assert!(!output.archive.is_empty());
        assert_eq!(output.stats.total_nodes, 2);
    }

    #[tokio::test]
    async fn compile_multiple_concurrent() {
        let (a, b) = tokio::join!(
            AsyncCompiler::new(simple_graph()).compile(),
            AsyncCompiler::new(simple_graph()).compile(),
        );
        assert!(a.unwrap().is_ok());
        assert!(b.unwrap().is_ok());
    }

    #[tokio::test]
    async fn compile_empty_graph() {
        use hologram_ir::Graph;
        let output = AsyncCompiler::new(Graph::new())
            .compile()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(output.stats.total_nodes, 0);
    }

    #[tokio::test]
    async fn compile_fusion_runs_unconditionally() {
        use hologram_core::op::LutOp;
        // Fusion always runs in v0.2.0 — no knob.
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("y", 2)
            .build();
        let output = AsyncCompiler::new(g).compile().await.unwrap().unwrap();
        assert_eq!(output.stats.total_nodes, 3);
    }
}
