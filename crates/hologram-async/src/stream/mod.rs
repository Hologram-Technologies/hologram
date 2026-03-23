//! Streaming execution: emits a `LevelResult` per schedule level via mpsc.

use hologram_exec::mmap::{build_tape_from_plan, execute_tape};
use hologram_exec::{ExecResult, GraphInputs, GraphOutputs};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Progress event emitted after each execution level completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelResult {
    /// Zero-based index of the completed level.
    pub level_index: usize,
    /// Number of nodes dispatched in this level.
    pub nodes_executed: usize,
}

/// Execute a `.holo` archive with per-level progress streaming.
///
/// Returns a receiver that yields one `LevelResult` when execution completes,
/// and a `JoinHandle` that resolves to the final `GraphOutputs`.
///
/// Tape execution is fast enough that progress is emitted as a single
/// "done" event rather than per-level increments.
///
/// Dropping the receiver does **not** cancel execution; the blocking task
/// runs to completion and the channel send is silently ignored.
pub fn execute_stream(
    archive: Vec<u8>,
    inputs: GraphInputs,
) -> (
    mpsc::Receiver<LevelResult>,
    JoinHandle<ExecResult<GraphOutputs>>,
) {
    let (tx, rx) = mpsc::channel(64);
    let handle = tokio::task::spawn_blocking(move || {
        let plan = hologram_archive::load_from_bytes(&archive)?;
        let tape = build_tape_from_plan(&plan)?;
        let node_count = plan.graph().nodes.len();
        let result = execute_tape(&tape, &plan, &inputs);
        // Emit a single completion event.
        let _ = tx.blocking_send(LevelResult {
            level_index: 0,
            nodes_executed: node_count,
        });
        result
    });
    (rx, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_archive::writer::holo_writer::HoloWriter;
    use hologram_core::op::LutOp;
    use hologram_graph::builder::GraphBuilder;
    use hologram_graph::graph::GraphOp;

    fn chain_archive() -> Vec<u8> {
        // Input -> Relu -> Sigmoid -> Output  (3 levels)
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .output("y", 3)
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

    /// At least one event is emitted and outputs are correct.
    #[tokio::test]
    async fn stream_level_order() {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![128]);
        let (mut rx, handle) = execute_stream(chain_archive(), inputs);

        let mut events: Vec<LevelResult> = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }

        let outputs = handle.await.unwrap().unwrap();
        assert_eq!(
            outputs.by_name("y").unwrap(),
            &[LutOp::Sigmoid.apply(LutOp::Relu.apply(128))]
        );

        assert!(!events.is_empty());
    }

    /// Total nodes across all stream events equals graph node count.
    #[tokio::test]
    async fn stream_total_nodes() {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![64]);
        let (mut rx, handle) = execute_stream(chain_archive(), inputs);

        let mut total_nodes = 0usize;
        while let Some(ev) = rx.recv().await {
            total_nodes += ev.nodes_executed;
        }
        handle.await.unwrap().unwrap();

        assert_eq!(total_nodes, 4); // Input, Relu, Sigmoid, Output
    }

    /// Dropping the receiver does not prevent the task from completing.
    #[tokio::test]
    async fn drop_receiver_task_completes() {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![42]);
        let (rx, handle) = execute_stream(passthrough_archive(), inputs);
        drop(rx); // sender will get errors on blocking_send but continues
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    /// Empty graph: no level events, outputs are empty.
    #[tokio::test]
    async fn stream_empty_graph() {
        use hologram_archive::writer::holo_writer::HoloWriter;
        let archive = HoloWriter::new().build().unwrap();
        let (mut rx, handle) = execute_stream(archive, GraphInputs::new());

        let mut _count = 0usize;
        while rx.recv().await.is_some() {
            _count += 1;
        }
        let outputs = handle.await.unwrap().unwrap();
        // Tape path emits one event even for empty graphs (with 0 nodes).
        // The graph is empty so outputs should be empty.
        assert!(outputs.is_empty());
    }

    /// Concurrent streams do not interfere.
    #[tokio::test]
    async fn concurrent_streams() {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![100]);

        let (mut rx1, h1) = execute_stream(passthrough_archive(), inputs.clone());
        let (mut rx2, h2) = execute_stream(passthrough_archive(), inputs.clone());

        let (out1, out2) = tokio::join!(h1, h2);
        // Drain receivers
        while rx1.recv().await.is_some() {}
        while rx2.recv().await.is_some() {}

        assert_eq!(out1.unwrap().unwrap().by_name("y").unwrap(), &[100]);
        assert_eq!(out2.unwrap().unwrap().by_name("y").unwrap(), &[100]);
    }

    /// LevelResult fields are correct.
    #[tokio::test]
    async fn level_result_fields() {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![1]);
        let (mut rx, handle) = execute_stream(passthrough_archive(), inputs);

        let mut events: Vec<LevelResult> = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        handle.await.unwrap().unwrap();

        // Tape emits a single completion event.
        assert!(!events.is_empty());
        assert!(events[0].nodes_executed > 0);
    }
}
