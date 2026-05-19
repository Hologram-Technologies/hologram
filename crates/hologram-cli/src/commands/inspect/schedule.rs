//! `--detail schedule` output.

use hologram_archive::LoadedPlan;
use hologram_graph::ExecutionSchedule;

/// Print the full execution schedule.
pub fn print(plan: &LoadedPlan, schedule: &ExecutionSchedule) {
    let total: usize = plan.graph().node_count();
    let levels = &schedule.levels;
    println!(
        "Execution Schedule ({} levels, {} nodes):",
        levels.len(),
        total
    );
    for (i, level) in levels.iter().enumerate() {
        print_level(i, level);
    }
    println!();
    print_stats(schedule);
}

/// Print a single level's node list.
fn print_level(idx: usize, level: &hologram_graph::schedule::levels::ParallelLevel) {
    let ids: Vec<String> = level
        .node_ids
        .iter()
        .map(|n| n.index().to_string())
        .collect();
    let count = level.node_ids.len();
    let noun = if count == 1 { "node" } else { "nodes" };
    println!("  Level {idx}:  [{}]  ({count} {noun})", ids.join(", "));
}

/// Print schedule statistics.
fn print_stats(schedule: &ExecutionSchedule) {
    let max = schedule
        .levels
        .iter()
        .map(|l| l.node_ids.len())
        .max()
        .unwrap_or(0);
    let max_idx = schedule
        .levels
        .iter()
        .position(|l| l.node_ids.len() == max)
        .unwrap_or(0);
    println!("  Critical path:    {} levels", schedule.critical_path);
    println!("  Max parallelism:  {max} nodes (level {max_idx})");
    println!("  Avg parallelism:  {:.2}x", schedule.parallelism_ratio());
}
