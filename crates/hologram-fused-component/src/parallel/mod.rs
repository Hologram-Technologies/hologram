//! Parallel level execution (feature-gated on `parallel`/`rayon`).
//!
//! Executes all nodes within a `ParallelLevel` concurrently when the
//! level is large enough. Below the adaptive threshold, sequential
//! execution avoids rayon overhead.

use hologram_ir::graph::node::NodeId;
use hologram_ir::schedule::levels::ParallelLevel;

use crate::error::ExecResult;

/// Minimum number of nodes in a level to use parallel execution.
///
/// Below this, the rayon thread-pool overhead exceeds the parallelism gain.
/// This is the baseline threshold; `should_parallelize` may apply additional
/// heuristics based on estimated work per node.
pub const PARALLEL_THRESHOLD: usize = 4;

/// Estimated byte-count below which a node's dispatch is "cheap" —
/// rayon overhead dominates for small buffers even with many nodes.
const SMALL_BUFFER_BYTES: usize = 256;

/// Execute a level's nodes with adaptive parallelism.
///
/// Uses `estimated_bytes` (total output bytes for the level) to decide
/// whether parallelism is worthwhile. Small levels with tiny buffers
/// run sequentially even above `PARALLEL_THRESHOLD` node count.
pub fn execute_level<F>(level: &ParallelLevel, f: F) -> ExecResult<Vec<(NodeId, Vec<u8>)>>
where
    F: Fn(NodeId) -> ExecResult<Vec<u8>> + Sync,
{
    if should_parallelize(level, None) {
        execute_level_parallel(level, f)
    } else {
        execute_level_sequential(level, f)
    }
}

/// Execute a level with an explicit work estimate for adaptive thresholding.
///
/// `estimated_bytes_per_node`: if provided, used to decide whether the work
/// per node justifies thread-pool overhead.
pub fn execute_level_adaptive<F>(
    level: &ParallelLevel,
    estimated_bytes_per_node: Option<usize>,
    f: F,
) -> ExecResult<Vec<(NodeId, Vec<u8>)>>
where
    F: Fn(NodeId) -> ExecResult<Vec<u8>> + Sync,
{
    if should_parallelize(level, estimated_bytes_per_node) {
        execute_level_parallel(level, f)
    } else {
        execute_level_sequential(level, f)
    }
}

/// Sequential fallback.
fn execute_level_sequential<F>(level: &ParallelLevel, f: F) -> ExecResult<Vec<(NodeId, Vec<u8>)>>
where
    F: Fn(NodeId) -> ExecResult<Vec<u8>>,
{
    let mut results = Vec::with_capacity(level.node_ids.len());
    for &id in &level.node_ids {
        let output = f(id)?;
        results.push((id, output));
    }
    Ok(results)
}

/// Whether to use parallel execution for this level.
///
/// Adaptive: considers both node count and estimated work per node.
/// Small buffers (e.g. shape ops producing 8 bytes) aren't worth parallelizing
/// even in large levels.
pub fn should_parallelize(level: &ParallelLevel, estimated_bytes_per_node: Option<usize>) -> bool {
    if !cfg!(feature = "parallel") {
        return false;
    }
    let n = level.node_ids.len();
    if n < PARALLEL_THRESHOLD {
        return false;
    }
    // If we have a work estimate, require meaningful work per node.
    if let Some(bytes) = estimated_bytes_per_node {
        if bytes < SMALL_BUFFER_BYTES {
            // Tiny buffers: raise threshold to avoid thread-pool overhead.
            return n >= PARALLEL_THRESHOLD * 4;
        }
    }
    true
}

/// Parallel execution via rayon (only compiled with the `parallel` feature).
#[cfg(feature = "parallel")]
fn execute_level_parallel<F>(level: &ParallelLevel, f: F) -> ExecResult<Vec<(NodeId, Vec<u8>)>>
where
    F: Fn(NodeId) -> ExecResult<Vec<u8>> + Sync,
{
    use rayon::prelude::*;

    level
        .node_ids
        .par_iter()
        .map(|&id| {
            let output = f(id)?;
            Ok((id, output))
        })
        .collect()
}

/// Sequential fallback when rayon is not available.
#[cfg(not(feature = "parallel"))]
fn execute_level_parallel<F>(level: &ParallelLevel, f: F) -> ExecResult<Vec<(NodeId, Vec<u8>)>>
where
    F: Fn(NodeId) -> ExecResult<Vec<u8>>,
{
    execute_level_sequential(level, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(n: u32) -> NodeId {
        NodeId::new(n, 0)
    }

    #[test]
    fn sequential_execution() {
        let level = ParallelLevel {
            node_ids: vec![nid(0), nid(1), nid(2)],
        };
        let results = execute_level(&level, |id| Ok(vec![id.index() as u8])).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (nid(0), vec![0]));
        assert_eq!(results[1], (nid(1), vec![1]));
        assert_eq!(results[2], (nid(2), vec![2]));
    }

    #[test]
    fn empty_level() {
        let level = ParallelLevel { node_ids: vec![] };
        let results = execute_level(&level, |_| Ok(vec![])).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn error_propagation() {
        use crate::error::ExecError;
        let level = ParallelLevel {
            node_ids: vec![nid(0), nid(1)],
        };
        let result = execute_level(&level, |id| {
            if id.index() == 1 {
                Err(ExecError::NodeNotFound(id))
            } else {
                Ok(vec![42])
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn threshold_check() {
        let small = ParallelLevel {
            node_ids: vec![nid(0), nid(1)],
        };
        assert!(!should_parallelize(&small, None));

        let large = ParallelLevel {
            node_ids: (0..10).map(nid).collect(),
        };
        // Only true with `parallel` feature
        if cfg!(feature = "parallel") {
            assert!(should_parallelize(&large, None));
        } else {
            assert!(!should_parallelize(&large, None));
        }
    }

    #[test]
    fn adaptive_skips_tiny_buffers() {
        let large = ParallelLevel {
            node_ids: (0..10).map(nid).collect(),
        };
        // Large level but tiny buffers — should NOT parallelize (unless very many nodes)
        assert!(!should_parallelize(&large, Some(8)));
        // Large level with meaningful work — should parallelize (if feature enabled)
        if cfg!(feature = "parallel") {
            assert!(should_parallelize(&large, Some(4096)));
        }
    }

    #[test]
    fn adaptive_tiny_with_many_nodes() {
        // Even tiny buffers should parallelize with enough nodes (4x threshold)
        let huge = ParallelLevel {
            node_ids: (0..20).map(nid).collect(),
        };
        if cfg!(feature = "parallel") {
            assert!(should_parallelize(&huge, Some(8)));
        }
    }

    #[test]
    fn large_level_produces_correct_results() {
        let level = ParallelLevel {
            node_ids: (0..8).map(nid).collect(),
        };
        let results = execute_level(&level, |id| Ok(vec![id.index() as u8 * 2])).unwrap();
        assert_eq!(results.len(), 8);
        for (i, (id, data)) in results.iter().enumerate() {
            assert_eq!(id.index(), i as u32);
            assert_eq!(data, &[i as u8 * 2]);
        }
    }
}
