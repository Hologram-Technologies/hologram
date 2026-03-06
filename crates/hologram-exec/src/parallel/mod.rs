//! Parallel level execution (feature-gated on `parallel`/`rayon`).
//!
//! Executes all nodes within a `ParallelLevel` concurrently when the
//! level is large enough. Below `PARALLEL_THRESHOLD`, sequential
//! execution avoids rayon overhead.

use hologram_graph::graph::node::NodeId;
use hologram_graph::schedule::levels::ParallelLevel;

use crate::error::ExecResult;

/// Minimum number of nodes in a level to use parallel execution.
pub const PARALLEL_THRESHOLD: usize = 4;

/// Execute a level's nodes, calling `f(node_id)` for each.
///
/// When the `parallel` feature is enabled and the level has at least
/// `PARALLEL_THRESHOLD` nodes, uses rayon's `par_iter`. Otherwise
/// executes sequentially.
pub fn execute_level<F>(level: &ParallelLevel, f: F) -> ExecResult<Vec<(NodeId, Vec<u8>)>>
where
    F: Fn(NodeId) -> ExecResult<Vec<u8>> + Sync,
{
    if should_parallelize(level) {
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
fn should_parallelize(level: &ParallelLevel) -> bool {
    cfg!(feature = "parallel") && level.node_ids.len() >= PARALLEL_THRESHOLD
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
        assert!(!should_parallelize(&small));

        let large = ParallelLevel {
            node_ids: (0..10).map(nid).collect(),
        };
        // Only true with `parallel` feature
        if cfg!(feature = "parallel") {
            assert!(should_parallelize(&large));
        } else {
            assert!(!should_parallelize(&large));
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
