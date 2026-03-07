//! `--detail summary` output.

use super::InspectArgs;
use crate::fmt::format_bytes;
use hologram_archive::LoadedPlan;
use hologram_graph::ExecutionSchedule;

/// Print the summary detail level.
pub fn print(args: &InspectArgs, data: &[u8], plan: &LoadedPlan, schedule: &ExecutionSchedule) {
    print_archive(args, data.len() as u64, plan);
    println!();
    print_graph_summary(plan);
    println!();
    print_schedule_summary(plan, schedule);
}

/// Print archive-level metadata.
fn print_archive(args: &InspectArgs, size: u64, plan: &LoadedPlan) {
    let h = plan.header();
    println!("Archive:       {:?}", args.file);
    println!("File size:     {} ({})", format_bytes(size), size);
    println!("Format:        HOLO v{}", h.version);
}

/// Print graph I/O summary.
fn print_graph_summary(plan: &LoadedPlan) {
    let sg = plan.graph();
    let h = plan.header();
    println!("Graph:");
    println!("  Nodes:       {}", sg.node_count());
    print_io_names("Inputs", &sg.input_names);
    print_io_names("Outputs", &sg.output_names);
    println!("  Graph size:  {}", format_bytes(h.graph_size));
    println!("  Weights:     {}", format_bytes(h.weights_size));
}

/// Print named I/O ports with indices.
fn print_io_names(label: &str, names: &[String]) {
    if names.is_empty() {
        println!("  {label}:      (none)");
        return;
    }
    let items: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{n} (index {i})"))
        .collect();
    println!("  {label}:     {}", items.join(", "));
}

/// Print schedule summary.
fn print_schedule_summary(plan: &LoadedPlan, schedule: &ExecutionSchedule) {
    let h = plan.header();
    println!("Schedule:");
    println!("  Levels:        {}", schedule.levels.len());
    println!("  Critical path: {}", schedule.critical_path);
    println!("  Parallelism:   {:.2}x", schedule.parallelism_ratio());
    println!("  Sections:      {} entries", h.section_count);
}
