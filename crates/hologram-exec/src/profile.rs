//! Lightweight execution profiling for hologram-exec.
//!
//! Gated behind the `profile` feature flag — zero overhead when disabled.
//! Collects per-op-type timing, per-level timing, and shape propagation cost.
//!
//! # Usage
//!
//! ```bash
//! cargo run --features profile -p hologram -- run model.holo --prompt "Hello"
//! ```
//!
//! The profile summary is printed to stderr at the end of execution.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Accumulated timing data for a single op type or phase.
#[derive(Debug, Clone, Default)]
pub struct OpTiming {
    pub total: Duration,
    pub count: usize,
    pub total_bytes: usize,
}

/// Execution profile accumulator.
///
/// Collects wall-clock timing per op type, per level, and for shape propagation.
/// All methods are `#[inline]` so the compiler can eliminate them when not used.
#[derive(Debug, Clone)]
pub struct PerfProfile {
    /// Per-op-type timing (e.g., "MatMul" → total time + call count).
    pub ops: HashMap<&'static str, OpTiming>,
    /// Per-level: (shape_propagation_time, dispatch_time, node_count).
    pub levels: Vec<(Duration, Duration, usize)>,
    /// Total execution wall time.
    pub total: Duration,
    /// Reusable timer for nested measurements.
    timer: Instant,
}

impl PerfProfile {
    /// Create a new profile accumulator.
    #[inline]
    pub fn new() -> Self {
        Self {
            ops: HashMap::with_capacity(64),
            levels: Vec::with_capacity(256),
            total: Duration::ZERO,
            timer: Instant::now(),
        }
    }

    /// Start the total execution timer.
    #[inline]
    pub fn start_total(&mut self) {
        self.timer = Instant::now();
    }

    /// Stop the total execution timer.
    #[inline]
    pub fn stop_total(&mut self) {
        self.total = self.timer.elapsed();
    }

    /// Record timing for a single op dispatch.
    #[inline]
    pub fn record_op(&mut self, name: &'static str, elapsed: Duration, output_bytes: usize) {
        let entry = self.ops.entry(name).or_default();
        entry.total += elapsed;
        entry.count += 1;
        entry.total_bytes += output_bytes;
    }

    /// Record timing for a complete level.
    #[inline]
    pub fn record_level(
        &mut self,
        shape_time: Duration,
        dispatch_time: Duration,
        node_count: usize,
    ) {
        self.levels.push((shape_time, dispatch_time, node_count));
    }

    /// Print a summary table to stderr.
    pub fn print_summary(&self) {
        eprintln!("\n{self}");
    }
}

impl Default for PerfProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PerfProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════════"
        )?;
        writeln!(f, "  EXECUTION PROFILE")?;
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════════"
        )?;
        writeln!(
            f,
            "  Total wall time: {:.3}ms",
            self.total.as_secs_f64() * 1000.0
        )?;
        writeln!(f)?;

        // ── Per-op table sorted by total time (descending) ──
        let mut ops: Vec<_> = self.ops.iter().collect();
        ops.sort_by(|a, b| b.1.total.cmp(&a.1.total));

        let total_op_time: Duration = ops.iter().map(|(_, t)| t.total).sum();

        writeln!(f, "  OP TIMING (sorted by total time)")?;
        writeln!(
            f,
            "  ─────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "  {:20} {:>8} {:>10} {:>10} {:>8} {:>8}",
            "Op", "Calls", "Total(ms)", "Avg(µs)", "Out(MB)", "Pct(%)"
        )?;
        writeln!(
            f,
            "  ─────────────────────────────────────────────────────────────"
        )?;

        for (name, timing) in &ops {
            let total_ms = timing.total.as_secs_f64() * 1000.0;
            let avg_us = if timing.count > 0 {
                timing.total.as_secs_f64() * 1_000_000.0 / timing.count as f64
            } else {
                0.0
            };
            let out_mb = timing.total_bytes as f64 / (1024.0 * 1024.0);
            let pct = if total_op_time.as_nanos() > 0 {
                timing.total.as_secs_f64() / total_op_time.as_secs_f64() * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "  {:20} {:>8} {:>10.3} {:>10.1} {:>8.2} {:>7.1}%",
                name, timing.count, total_ms, avg_us, out_mb, pct
            )?;
        }

        writeln!(
            f,
            "  ─────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "  {:20} {:>8} {:>10.3}",
            "TOTAL",
            ops.iter().map(|(_, t)| t.count).sum::<usize>(),
            total_op_time.as_secs_f64() * 1000.0,
        )?;

        // ── Level summary ──
        if !self.levels.is_empty() {
            writeln!(f)?;
            writeln!(f, "  LEVEL TIMING (top 10 by dispatch time)")?;
            writeln!(
                f,
                "  ─────────────────────────────────────────────────────────────"
            )?;
            writeln!(
                f,
                "  {:>6} {:>8} {:>12} {:>12}",
                "Level", "Nodes", "Shape(ms)", "Dispatch(ms)"
            )?;
            writeln!(
                f,
                "  ─────────────────────────────────────────────────────────────"
            )?;

            let mut indexed: Vec<_> = self.levels.iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));

            for (i, (shape_t, dispatch_t, nodes)) in indexed.iter().take(10) {
                writeln!(
                    f,
                    "  {:>6} {:>8} {:>12.3} {:>12.3}",
                    i,
                    nodes,
                    shape_t.as_secs_f64() * 1000.0,
                    dispatch_t.as_secs_f64() * 1000.0,
                )?;
            }

            let total_shape: Duration = self.levels.iter().map(|(s, _, _)| *s).sum();
            let total_dispatch: Duration = self.levels.iter().map(|(_, d, _)| *d).sum();
            writeln!(
                f,
                "  ─────────────────────────────────────────────────────────────"
            )?;
            writeln!(
                f,
                "  {:>6} {:>8} {:>12.3} {:>12.3}",
                "ALL",
                self.levels.iter().map(|(_, _, n)| n).sum::<usize>(),
                total_shape.as_secs_f64() * 1000.0,
                total_dispatch.as_secs_f64() * 1000.0,
            )?;
            writeln!(
                f,
                "  Shape propagation overhead: {:.1}% of dispatch time",
                if total_dispatch.as_nanos() > 0 {
                    total_shape.as_secs_f64() / total_dispatch.as_secs_f64() * 100.0
                } else {
                    0.0
                }
            )?;
        }

        writeln!(
            f,
            "═══════════════════════════════════════════════════════════════"
        )?;
        Ok(())
    }
}

/// Get the op name as a `&'static str` for profiling.
///
/// Returns a short, human-readable name for each `GraphOp` variant.
pub fn op_name(op: &hologram_graph::graph::GraphOp) -> &'static str {
    use hologram_graph::graph::GraphOp;
    match op {
        GraphOp::Constant(_) => "Constant",
        GraphOp::Float(fop) => fop.short_name(),
        _ => "Other",
    }
}
