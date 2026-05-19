//! `hologram compile` — compile a serialized graph to `.holo` archive.

use crate::error::CliError;
use clap::Args;
use std::path::PathBuf;

/// Arguments for the compile subcommand.
#[derive(Args)]
pub struct CompileArgs {
    /// Input file (rkyv-serialized graph).
    pub input: PathBuf,
    /// Output `.holo` file path.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Disable fusion optimization pass.
    #[arg(long)]
    pub no_fuse: bool,
}

/// Execute the compile command.
pub async fn execute(args: CompileArgs) -> Result<(), CliError> {
    let output_path = resolve_output(&args);
    let graph = load_graph(&args.input)?;
    let result = run_compiler(graph, !args.no_fuse)?;
    write_archive(&output_path, &result.archive)?;
    print_stats(&args.input, &output_path, &result.stats);
    Ok(())
}

/// Resolve output path, defaulting to input with `.holo` extension.
fn resolve_output(args: &CompileArgs) -> PathBuf {
    args.output
        .clone()
        .unwrap_or_else(|| args.input.with_extension("holo"))
}

/// Load and deserialize a graph from an rkyv file.
fn load_graph(path: &PathBuf) -> Result<hologram_graph::Graph, CliError> {
    let data = std::fs::read(path)?;
    let sg = deserialize_graph(&data)?;
    Ok(reconstruct_graph(&sg))
}

/// Deserialize a SerializedGraph from bytes.
fn deserialize_graph(
    data: &[u8],
) -> Result<hologram_archive::format::graph::SerializedGraph, CliError> {
    rkyv::from_bytes::<hologram_archive::format::graph::SerializedGraph, rkyv::rancor::Error>(data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e}")))
        .map_err(CliError::from)
}

/// Reconstruct a live Graph from a SerializedGraph.
fn reconstruct_graph(
    sg: &hologram_archive::format::graph::SerializedGraph,
) -> hologram_graph::Graph {
    sg.to_graph()
}

/// Run the compiler pipeline.
fn run_compiler(
    graph: hologram_graph::Graph,
    enable_fusion: bool,
) -> Result<hologram_compiler::CompilationOutput, CliError> {
    hologram_compiler::CompilerBuilder::new(graph)
        .fuse(enable_fusion)
        .build()
        .map_err(CliError::from)
}

/// Write archive bytes to disk.
fn write_archive(path: &PathBuf, data: &[u8]) -> Result<(), CliError> {
    std::fs::write(path, data).map_err(CliError::from)
}

/// Print compilation statistics.
fn print_stats(input: &PathBuf, output: &PathBuf, stats: &hologram_compiler::CompilationStats) {
    println!("Compiled {:?} -> {:?}", input, output);
    println!("  nodes: {}", stats.total_nodes);
    println!("  levels: {}", stats.schedule_levels);
    println!("  workspace slots: {}", stats.workspace_slots);
    println!(
        "  fusion: {} folded, {} fused, {} CSE",
        stats.fusion.constants_folded, stats.fusion.views_fused, stats.fusion.cse_eliminated,
    );
}
