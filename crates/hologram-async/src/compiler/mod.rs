//! Async wrapper for the hologram compilation pipeline.

use hologram_compiler::{CompilationOutput, CompileResult, CompilerBuilder};
use hologram_graph::Graph;
use tokio::task::JoinHandle;

/// Async wrapper around `CompilerBuilder`.
///
/// Runs the compilation pipeline on a blocking thread via
/// `tokio::task::spawn_blocking`, returning a `JoinHandle` the caller
/// can `.await` from any async context.
pub struct AsyncCompiler {
    graph: Graph,
    enable_fusion: bool,
}

impl AsyncCompiler {
    /// Create a new async compiler for the given graph.
    #[must_use]
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            enable_fusion: true,
        }
    }

    /// Enable or disable the fusion optimization pass (default: enabled).
    #[must_use]
    pub fn fuse(mut self, enable: bool) -> Self {
        self.enable_fusion = enable;
        self
    }

    /// Spawn compilation on a blocking thread and return a `JoinHandle`.
    pub fn compile(self) -> JoinHandle<CompileResult<CompilationOutput>> {
        tokio::task::spawn_blocking(move || {
            CompilerBuilder::new(self.graph)
                .fuse(self.enable_fusion)
                .build()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_graph::builder::GraphBuilder;
    use hologram_graph::graph::GraphOp;

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
    async fn compile_no_fuse() {
        let output = AsyncCompiler::new(simple_graph())
            .fuse(false)
            .compile()
            .await
            .unwrap()
            .unwrap();
        assert!(!output.archive.is_empty());
        assert_eq!(output.stats.fusion.constants_folded, 0);
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
        use hologram_graph::Graph;
        let output = AsyncCompiler::new(Graph::new())
            .compile()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(output.stats.total_nodes, 0);
    }

    #[tokio::test]
    async fn compile_fuse_enabled_by_default() {
        use hologram_core::op::LutOp;
        // Build a graph where fusion can fold constants
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            // ADR-053: Relu (idx 1) requires shape coverage for v3 archives.
            .set_node_shape(1, vec![3])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("y", 2)
            .build();
        let output = AsyncCompiler::new(g).compile().await.unwrap().unwrap();
        assert_eq!(output.stats.total_nodes, 3);
    }
}
