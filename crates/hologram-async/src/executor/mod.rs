//! Async wrapper for archive execution.

use hologram_exec::{execute_bytes, ExecResult, GraphInputs, GraphOutputs};
use tokio::task::JoinHandle;

/// Async wrapper for `execute_bytes`.
///
/// Runs the executor on a blocking thread so callers can `.await` it
/// from an async context without stalling the Tokio executor.
pub struct AsyncExecutor;

impl AsyncExecutor {
    /// Execute a `.holo` archive on a blocking thread.
    pub fn execute(archive: Vec<u8>, inputs: GraphInputs) -> JoinHandle<ExecResult<GraphOutputs>> {
        tokio::task::spawn_blocking(move || execute_bytes(&archive, &inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_archive::writer::holo_writer::HoloWriter;
    use hologram_core::op::LutOp;
    use hologram_graph::builder::GraphBuilder;
    use hologram_graph::graph::GraphOp;

    fn relu_archive() -> Vec<u8> {
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("y", 2)
            .build();
        HoloWriter::new().set_graph(&g).build().unwrap()
    }

    fn passthrough_archive() -> Vec<u8> {
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Output, &[0])
            .output("y", 1)
            .build();
        HoloWriter::new().set_graph(&g).build().unwrap()
    }

    #[tokio::test]
    async fn execute_passthrough() {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![10, 20, 30]);
        let result = AsyncExecutor::execute(passthrough_archive(), inputs)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[10, 20, 30]);
    }

    #[tokio::test]
    async fn execute_relu() {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![0, 128, 255]);
        let result = AsyncExecutor::execute(relu_archive(), inputs)
            .await
            .unwrap()
            .unwrap();
        let y = result.by_name("y").unwrap();
        assert_eq!(y[0], LutOp::Relu.apply(0));
        assert_eq!(y[1], LutOp::Relu.apply(128));
        assert_eq!(y[2], LutOp::Relu.apply(255));
    }

    #[tokio::test]
    async fn execute_concurrent() {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![42]);
        let (a, b) = tokio::join!(
            AsyncExecutor::execute(passthrough_archive(), inputs.clone()),
            AsyncExecutor::execute(passthrough_archive(), inputs.clone()),
        );
        assert_eq!(a.unwrap().unwrap().by_name("y").unwrap(), &[42]);
        assert_eq!(b.unwrap().unwrap().by_name("y").unwrap(), &[42]);
    }

    #[tokio::test]
    async fn execute_invalid_archive_errors() {
        let inputs = GraphInputs::new();
        let result = AsyncExecutor::execute(vec![0u8; 64], inputs).await.unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_missing_input_errors() {
        // relu_archive expects input 0 — omit it
        let result = AsyncExecutor::execute(relu_archive(), GraphInputs::new())
            .await
            .unwrap();
        assert!(result.is_err());
    }
}
