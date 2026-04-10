//! `hologram inspect` — print metadata from a `.holo` archive.

mod graph;
mod json;
mod layout;
mod schedule;
mod sections;
mod summary;
mod weights;

use crate::error::CliError;
use clap::Args;
use hologram_archive::load_from_bytes;
use hologram_fused_component::build_schedule;
use std::path::PathBuf;

/// Inspection depth.
#[derive(Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum)]
pub enum DetailLevel {
    /// File size, format version, node count, I/O names, schedule overview.
    Summary,
    /// All nodes with operations, edges, and constant references.
    Graph,
    /// Parallel levels, nodes per level, critical path.
    Schedule,
    /// Section table entries with kind, size, offset, checksum.
    Sections,
    /// Weight tensor metadata from the weight index section.
    Weights,
    /// Visual byte-map of the archive layout.
    Layout,
    /// Everything combined.
    Full,
    /// Machine-readable JSON.
    Json,
}

/// Arguments for the inspect subcommand.
#[derive(Args)]
pub struct InspectArgs {
    /// Path to the `.holo` file to inspect.
    pub file: PathBuf,
    /// Levels of detail to display (may be repeated).
    #[arg(long, value_enum, default_values_t = [DetailLevel::Summary])]
    pub detail: Vec<DetailLevel>,
}

/// Execute the inspect command.
pub async fn execute(args: InspectArgs) -> Result<(), CliError> {
    let data = std::fs::read(&args.file)?;
    let plan = load_from_bytes(&data)?;
    let schedule = build_schedule(plan.graph())?;
    dispatch(&args, &data, &plan, &schedule);
    Ok(())
}

/// Expand meta-levels and deduplicate.
fn resolve_levels(raw: &[DetailLevel]) -> Vec<DetailLevel> {
    use DetailLevel::*;
    let mut out = Vec::new();
    for &level in raw {
        match level {
            Full => {
                for &l in &[Summary, Graph, Schedule, Sections, Weights, Layout] {
                    if !out.contains(&l) {
                        out.push(l);
                    }
                }
            }
            other => {
                if !out.contains(&other) {
                    out.push(other);
                }
            }
        }
    }
    out
}

/// Dispatch to the requested detail level printers.
fn dispatch(
    args: &InspectArgs,
    data: &[u8],
    plan: &hologram_archive::LoadedPlan,
    schedule: &hologram_ir::ExecutionSchedule,
) {
    let levels = resolve_levels(&args.detail);
    if levels.contains(&DetailLevel::Json) {
        json::print(args, data, plan, schedule);
        return;
    }
    let mut first = true;
    for level in &levels {
        if !first {
            println!();
        }
        first = false;
        print_level(*level, args, data, plan, schedule);
    }
}

/// Print a single detail level.
fn print_level(
    level: DetailLevel,
    args: &InspectArgs,
    data: &[u8],
    plan: &hologram_archive::LoadedPlan,
    schedule: &hologram_ir::ExecutionSchedule,
) {
    match level {
        DetailLevel::Summary => summary::print(args, data, plan, schedule),
        DetailLevel::Graph => graph::print(plan),
        DetailLevel::Schedule => schedule::print(plan, schedule),
        DetailLevel::Sections => sections::print(plan),
        DetailLevel::Weights => weights::print(plan),
        DetailLevel::Layout => layout::print(data, plan),
        DetailLevel::Full | DetailLevel::Json => {}
    }
}

#[cfg(test)]
mod tests {
    use hologram_archive::writer::holo_writer::HoloWriter;
    use hologram_core::op::LutOp;
    use hologram_ir::builder::GraphBuilder;
    use hologram_ir::graph::GraphOp;

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
        let plan = hologram_archive::load_from_bytes(&data).unwrap();
        let sg = plan.graph();
        assert_eq!(sg.node_count(), 3);
        assert_eq!(sg.input_names, vec!["x"]);
        assert_eq!(sg.output_names, vec!["y"]);
    }

    #[test]
    fn schedule_levels_nonzero() {
        let data = chain_archive();
        let plan = hologram_archive::load_from_bytes(&data).unwrap();
        let schedule = hologram_fused_component::build_schedule(plan.graph()).unwrap();
        assert!(!schedule.levels.is_empty());
    }

    #[test]
    fn empty_archive_zero_nodes() {
        let data = HoloWriter::new().build().unwrap();
        let plan = hologram_archive::load_from_bytes(&data).unwrap();
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
        let plan = hologram_archive::load_from_bytes(&data).unwrap();
        let sg = plan.graph();
        assert_eq!(sg.input_names, vec!["a", "b"]);
        assert_eq!(sg.output_names, vec!["p", "q"]);
    }

    #[test]
    fn detail_default_is_summary() {
        let args = super::InspectArgs {
            file: "test.holo".into(),
            detail: vec![],
        };
        let levels = super::resolve_levels(&args.detail);
        assert!(levels.is_empty());
    }

    #[test]
    fn full_expands_to_all_levels() {
        let levels = super::resolve_levels(&[super::DetailLevel::Full]);
        assert_eq!(levels.len(), 6);
        assert!(levels.contains(&super::DetailLevel::Summary));
        assert!(levels.contains(&super::DetailLevel::Graph));
        assert!(levels.contains(&super::DetailLevel::Schedule));
        assert!(levels.contains(&super::DetailLevel::Sections));
        assert!(levels.contains(&super::DetailLevel::Weights));
        assert!(levels.contains(&super::DetailLevel::Layout));
    }

    #[test]
    fn duplicate_levels_deduped() {
        let levels = super::resolve_levels(&[
            super::DetailLevel::Graph,
            super::DetailLevel::Graph,
            super::DetailLevel::Summary,
        ]);
        assert_eq!(levels.len(), 2);
    }

    #[test]
    fn json_output_parses() {
        let data = chain_archive();
        let plan = hologram_archive::load_from_bytes(&data).unwrap();
        let schedule = hologram_fused_component::build_schedule(plan.graph()).unwrap();
        let args = super::InspectArgs {
            file: "test.holo".into(),
            detail: vec![super::DetailLevel::Json],
        };
        let val = super::json::build(&args, &data, &plan, &schedule);
        assert!(val["graph"]["node_count"].as_u64().unwrap() == 3);
        assert_eq!(val["graph"]["inputs"][0], "x");
        assert_eq!(val["graph"]["outputs"][0], "y");
        assert!(val["schedule"]["num_levels"].as_u64().unwrap() > 0);
    }
}
