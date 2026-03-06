//! `hologram inspect` — print metadata from a `.holo` archive.

use crate::error::CliError;
use clap::Args;
use holo_archive::load_from_bytes;
use holo_exec::build_schedule;
use std::path::PathBuf;

/// Arguments for the inspect subcommand.
#[derive(Args)]
pub struct InspectArgs {
    /// Path to the `.holo` file to inspect.
    pub file: PathBuf,
}

/// Execute the inspect command.
pub async fn execute(args: InspectArgs) -> Result<(), CliError> {
    let data = std::fs::read(&args.file)?;
    let plan = load_from_bytes(&data)?;
    let sg = plan.graph();
    let schedule = build_schedule(sg)?;
    print_info(&args.file, data.len(), sg, schedule.levels.len());
    Ok(())
}

/// Print archive metadata.
fn print_info(
    path: &std::path::Path,
    file_size: usize,
    sg: &holo_archive::format::graph::SerializedGraph,
    level_count: usize,
) {
    println!("file:    {:?}", path);
    println!("size:    {} bytes", file_size);
    println!("nodes:   {}", sg.node_count());
    println!("inputs:  [{}]", sg.input_names.join(", "));
    println!("outputs: [{}]", sg.output_names.join(", "));
    println!("levels:  {}", level_count);
}

#[cfg(test)]
mod tests {
    use holo_archive::writer::holo_writer::HoloWriter;
    use holo_core::op::LutOp;
    use holo_graph::builder::GraphBuilder;
    use holo_graph::graph::GraphOp;

    /// Build a small chain archive for testing.
    fn chain_archive() -> Vec<u8> {
        let g = GraphBuilder::new()
            .input("x")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("y", 2)
            .build();
        HoloWriter::new().set_graph(&g).build().unwrap()
    }

    #[test]
    fn load_and_inspect_succeeds() {
        let data = chain_archive();
        let plan = holo_archive::load_from_bytes(&data).unwrap();
        let sg = plan.graph();
        assert_eq!(sg.node_count(), 3);
        assert_eq!(sg.input_names, vec!["x"]);
        assert_eq!(sg.output_names, vec!["y"]);
    }

    #[test]
    fn schedule_levels_nonzero() {
        let data = chain_archive();
        let plan = holo_archive::load_from_bytes(&data).unwrap();
        let schedule = holo_exec::build_schedule(plan.graph()).unwrap();
        assert!(!schedule.levels.is_empty());
    }

    #[test]
    fn empty_archive_zero_nodes() {
        let data = HoloWriter::new().build().unwrap();
        let plan = holo_archive::load_from_bytes(&data).unwrap();
        assert_eq!(plan.node_count(), 0);
    }

    #[test]
    fn inspect_input_output_names() {
        let g = GraphBuilder::new()
            .input("a")
            .input("b")
            .node_from_graph_input(GraphOp::Input, 0)
            .node_from_graph_input(GraphOp::Input, 1)
            .node_with_inputs(GraphOp::Output, &[0])
            .node_with_inputs(GraphOp::Output, &[1])
            .output("p", 2)
            .output("q", 3)
            .build();
        let data = HoloWriter::new().set_graph(&g).build().unwrap();
        let plan = holo_archive::load_from_bytes(&data).unwrap();
        let sg = plan.graph();
        assert_eq!(sg.input_names, vec!["a", "b"]);
        assert_eq!(sg.output_names, vec!["p", "q"]);
    }
}
