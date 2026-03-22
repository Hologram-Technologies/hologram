//! KvExecutor: level-by-level graph execution using KV dispatch.
//!
//! The executor walks through each `ParallelLevel` in the schedule,
//! gathers inputs from the arena, dispatches via `KvStore`, and
//! stores outputs back. All mutation is between levels, never during.

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use hologram_archive::format::graph::SerializedGraph;
use hologram_core::op::FloatDType;
use hologram_graph::constant::{ConstantData, ConstantId};
use hologram_graph::graph::node::{InputSlot, InputSource, Node, NodeId};
use hologram_graph::graph::GraphOp;
use hologram_graph::schedule::levels::ParallelLevel;
use hologram_graph::schedule::ExecutionSchedule;

use crate::buffer::{BufferArena, ShapeMap};
use crate::error::{ExecError, ExecResult};
use crate::eval::shape_resolve;
use crate::float_dispatch;
use crate::kv::{CustomOpRegistry, KvStore};
use hologram_core::op::{FloatOp, OpCategory, ShapeDim, ShapeSpec};

/// Runtime context passed to dispatch during execution.
///
/// Carries execution-time state that cannot be baked into the compiled graph
/// (e.g., the current token position for RoPE during KV cache decode).
/// Non-KV execution passes `None` — zero overhead.
pub struct ExecutionContext {
    /// Position offset for positional encodings (RoPE).
    /// Set from `KvCacheState::write_pos()` at the start of each call.
    /// 0 during prefill, N during decode (N = tokens already cached).
    pub position_offset: u32,
}

/// Configuration for batch-aware scheduling.
///
/// Supports continuous batching: multiple sequences share a KV cache prefix
/// and diverge at `shared_prefix_len`. The scheduler can overlap prefill of
/// new sequences with decode of in-flight sequences, amortizing attention
/// cost over the shared prefix region.
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Number of sequences in the batch.
    pub batch_size: usize,
    /// Number of tokens in the shared KV cache prefix.
    /// Tokens `0..shared_prefix_len` are computed once and reused.
    pub shared_prefix_len: usize,
}

/// Pre-resolved per-node metadata, indexed by `NodeId::index()`.
/// Eliminates HashMap lookups in the hot dispatch path.
struct CompiledNodeTable<'a> {
    nodes: Vec<Option<&'a Node>>,
    shapes: Vec<Option<Vec<usize>>>,
    dtypes: Vec<Option<FloatDType>>,
}

impl<'a> CompiledNodeTable<'a> {
    fn from_graph(sg: &'a SerializedGraph) -> Self {
        let max_idx = sg
            .nodes
            .iter()
            .map(|n| n.id.index() as usize)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let mut nodes = vec![None; max_idx];
        for n in &sg.nodes {
            let idx = n.id.index() as usize;
            nodes[idx] = Some(n);
        }

        let mut shapes = vec![None; max_idx];
        for (id, shape) in &sg.node_shapes {
            let idx = id.index() as usize;
            if idx < max_idx {
                shapes[idx] = Some(shape.clone());
            }
        }

        let mut dtypes = vec![None; max_idx];
        for (id, dtype) in &sg.node_dtypes {
            let idx = id.index() as usize;
            if idx < max_idx {
                dtypes[idx] = Some(*dtype);
            }
        }

        Self {
            nodes,
            shapes,
            dtypes,
        }
    }

    #[inline]
    fn get_node(&self, id: NodeId) -> Option<&'a Node> {
        let idx = id.index() as usize;
        if idx < self.nodes.len() {
            self.nodes[idx]
        } else {
            None
        }
    }

    #[inline]
    fn get_compiled_shape(&self, id: NodeId) -> Option<&Vec<usize>> {
        let idx = id.index() as usize;
        if idx < self.shapes.len() {
            self.shapes[idx].as_ref()
        } else {
            None
        }
    }

    #[inline]
    fn get_compiled_dtype(&self, id: NodeId) -> Option<&FloatDType> {
        let idx = id.index() as usize;
        if idx < self.dtypes.len() {
            self.dtypes[idx].as_ref()
        } else {
            None
        }
    }
}

/// Immutable graph-wide context shared across level dispatch and shape propagation.
/// (PP_1, PI_1, PA_4 — see Prism ontology)
struct DispatchContext<'a> {
    table: CompiledNodeTable<'a>,
    // HashMap views for callers that need HashMap interface (shape_propagate).
    // These are built once at construction and shared via reference.
    node_map: HashMap<NodeId, &'a Node>,
    compiled_shapes: HashMap<NodeId, Vec<usize>>,
    compiled_dtypes: HashMap<NodeId, FloatDType>,
    inputs: &'a GraphInputs,
    constants: &'a hologram_graph::constant::ConstantStore,
    registry: Option<&'a CustomOpRegistry>,
    weights: &'a [u8],
}

/// Named graph inputs: maps input index to byte data and optional shape.
#[derive(Debug, Clone, Default)]
pub struct GraphInputs {
    inputs: HashMap<u32, Vec<u8>>,
    shapes: HashMap<u32, Vec<usize>>,
}

impl GraphInputs {
    /// Create empty inputs.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    /// Set data for graph input at `index`.
    pub fn set(&mut self, index: u32, data: Vec<u8>) {
        self.inputs.insert(index, data);
    }

    /// Set data with an explicit N-D shape for graph input at `index`.
    pub fn set_with_shape(&mut self, index: u32, data: Vec<u8>, shape: Vec<usize>) {
        self.inputs.insert(index, data);
        self.shapes.insert(index, shape);
    }

    /// Get data for graph input at `index`.
    pub fn get(&self, index: u32) -> Option<&[u8]> {
        self.inputs.get(&index).map(|v| v.as_slice())
    }

    /// Get the shape for graph input at `index`, if set.
    pub fn shape(&self, index: u32) -> Option<&[usize]> {
        self.shapes.get(&index).map(|v| v.as_slice())
    }

    /// Create from a list of (index, data) pairs.
    #[must_use]
    pub fn from_pairs(pairs: Vec<(u32, Vec<u8>)>) -> Self {
        Self {
            inputs: pairs.into_iter().collect(),
            shapes: HashMap::new(),
        }
    }
}

/// Named graph outputs: list of (name, data) pairs.
#[derive(Debug, Clone, Default)]
pub struct GraphOutputs {
    outputs: Vec<(String, Vec<u8>)>,
}

impl GraphOutputs {
    /// Number of outputs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Whether there are no outputs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Get output by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<(&str, &[u8])> {
        self.outputs
            .get(index)
            .map(|(name, data)| (name.as_str(), data.as_slice()))
    }

    /// Get output by name.
    #[must_use]
    pub fn by_name(&self, name: &str) -> Option<&[u8]> {
        self.outputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// Consume into inner vec.
    #[must_use]
    pub fn into_inner(self) -> Vec<(String, Vec<u8>)> {
        self.outputs
    }

    /// Create from named output pairs.
    #[must_use]
    pub fn from_named(outputs: Vec<(String, Vec<u8>)>) -> Self {
        Self { outputs }
    }
}

/// Stateless graph executor using KV-lookup dispatch.
/// PM_5 atomicity: execution is atomic per level — partial level results are never visible.
pub struct KvExecutor;

impl KvExecutor {
    /// Execute a serialized graph according to its schedule.
    ///
    /// **PM_5 (Transaction Atomicity)**: `sg` and `schedule` are never mutated. On error, the
    /// caller's `SerializedGraph` and `ExecutionSchedule` remain valid and unchanged (PA_4 base
    /// binding preservation makes rollback free). It is safe to retry with different `GraphInputs`
    /// after a failure.
    pub fn execute(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
    ) -> ExecResult<GraphOutputs> {
        Self::execute_with_progress(sg, schedule, inputs, |_, _| {})
    }

    /// Execute with a per-level progress callback.
    ///
    /// `on_level(level_index, nodes_executed)` is called after each level completes.
    pub fn execute_with_progress<F>(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        on_level: F,
    ) -> ExecResult<GraphOutputs>
    where
        F: FnMut(usize, usize),
    {
        Self::execute_core(sg, schedule, inputs, None, &[], on_level)
    }

    /// Execute with a custom op registry (no progress callback).
    pub fn execute_with_registry(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        registry: &CustomOpRegistry,
    ) -> ExecResult<GraphOutputs> {
        Self::execute_core(sg, schedule, inputs, Some(registry), &[], |_, _| {})
    }

    /// Execute with archive weight data (no custom ops or progress).
    ///
    /// Primary method for archives containing `ConstantData::Deferred`
    /// references that resolve from the weight blob.
    pub fn execute_with_plan(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        weights: &[u8],
    ) -> ExecResult<GraphOutputs> {
        Self::execute_core(sg, schedule, inputs, None, weights, |_, _| {})
    }

    /// Execute with pre-projected shape hints from `walk_shape_context()`.
    ///
    /// `shape_hints` maps `NodeId.index() → concrete shape` for every node
    /// whose shape was projected by the `ShapeContextGraph` runtime walker.
    /// These hints are applied before any compiled-shape or inferred-shape
    /// resolution, ensuring correctness for variable-length inputs (seq>1).
    pub fn execute_with_shape_hints(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        weights: &[u8],
        shape_hints: &HashMap<u32, Vec<usize>>,
    ) -> ExecResult<GraphOutputs> {
        Self::execute_core_with_hints(
            sg,
            schedule,
            inputs,
            None,
            weights,
            |_, _| {},
            Some(shape_hints),
        )
    }

    /// Execute with shape hints AND a mutable KV cache state.
    ///
    /// `kv_state` is mutated during execution: `FloatOp::KvWrite` nodes
    /// append K/V data to the cache, and `FloatOp::KvRead` nodes read
    /// the cached data back. The cache persists across calls for
    /// autoregressive generation.
    pub fn execute_with_kv_state(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        weights: &[u8],
        shape_hints: &HashMap<u32, Vec<usize>>,
        kv_state: &mut crate::kv_cache::KvCacheState,
    ) -> ExecResult<GraphOutputs> {
        Self::execute_core_with_kv(
            sg,
            schedule,
            inputs,
            None,
            weights,
            |_, _| {},
            Some(shape_hints),
            Some(kv_state),
        )
    }

    /// Execute with a custom op registry and archive weight data.
    ///
    /// `weights` is the raw weight blob from the `.holo` archive.
    /// `ConstantData::Deferred` nodes are resolved by slicing into this blob
    /// using `source_id` as the byte offset.
    pub fn execute_with_weights(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        registry: &CustomOpRegistry,
        weights: &[u8],
    ) -> ExecResult<GraphOutputs> {
        Self::execute_core(sg, schedule, inputs, Some(registry), weights, |_, _| {})
    }

    pub(crate) fn execute_core<F>(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        registry: Option<&CustomOpRegistry>,
        weights: &[u8],
        on_level: F,
    ) -> ExecResult<GraphOutputs>
    where
        F: FnMut(usize, usize),
    {
        Self::execute_core_with_hints(sg, schedule, inputs, registry, weights, on_level, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_core_with_kv<F>(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        registry: Option<&CustomOpRegistry>,
        weights: &[u8],
        mut on_level: F,
        shape_hints: Option<&HashMap<u32, Vec<usize>>>,
        mut kv_state: Option<&mut crate::kv_cache::KvCacheState>,
    ) -> ExecResult<GraphOutputs>
    where
        F: FnMut(usize, usize),
    {
        use hologram_core::op::FloatOp;

        let compiled_dtypes = sg.node_dtypes_map();
        let mut arena = seed_arena(sg, weights, &compiled_dtypes)?;
        let mut shape_map = seed_shape_map(sg, &arena, inputs, &compiled_dtypes);

        let dctx = DispatchContext {
            table: CompiledNodeTable::from_graph(sg),
            node_map: build_node_map(sg),
            compiled_shapes: sg.node_shapes_map(),
            compiled_dtypes,
            inputs,
            constants: &sg.constants,
            registry,
            weights,
        };

        let max_level_size = schedule
            .levels
            .iter()
            .map(|l| l.node_ids.len())
            .max()
            .unwrap_or(0);
        let mut results_buf: Vec<(NodeId, Vec<u8>, Vec<usize>)> =
            Vec::with_capacity(max_level_size);

        // Track seq_len for KV cache advance (inferred from first KvWrite input).
        let mut kv_seq_len: Option<usize> = None;
        // Node IDs handled by KV cache — skip in dispatch_level.
        let mut kv_handled: HashSet<NodeId> = HashSet::new();

        // ExecutionContext carries position offset for RoPE during KV cache decode.
        let exec_ctx = kv_state.as_deref().map(|kv| ExecutionContext {
            position_offset: kv.write_pos() as u32,
        });

        for (i, level) in schedule.levels.iter().enumerate() {
            super::shape_propagate::propagate_level_shapes(
                level,
                &dctx.node_map,
                &arena,
                &mut shape_map,
                &dctx.compiled_shapes,
                &dctx.compiled_dtypes,
                shape_hints,
            );

            // Check for KvWrite/KvRead nodes in this level and handle them
            // before the normal dispatch loop.
            if let Some(kv) = kv_state.as_deref_mut() {
                for &node_id in &level.node_ids {
                    let Some(node) = dctx.node_map.get(&node_id) else {
                        continue;
                    };
                    if let GraphOp::Float(FloatOp::KvWrite {
                        layer,
                        is_key,
                        n_kv_heads,
                        head_dim,
                    }) = &node.op
                    {
                        // Gather input and copy to release arena borrow.
                        let input_data = {
                            let refs = gather_inputs(node, &arena, dctx.inputs)?;
                            refs.first().map(|d| d.to_vec()).unwrap_or_default()
                        };
                        if !input_data.is_empty() {
                            let floats: &[f32] = if input_data.len() % 4 == 0 {
                                bytemuck::cast_slice(&input_data)
                            } else {
                                &[]
                            };

                            let nkv = *n_kv_heads as usize;
                            let hd = *head_dim as usize;
                            let stride = nkv * hd;
                            let seq = if stride > 0 { floats.len() / stride } else { 1 };

                            // KV data arrives from ONNX attention in heads-first layout:
                            //   [heads, seq, dim]
                            // The cache stores in seq-first layout:
                            //   [seq, heads, dim]
                            // Transpose before writing.
                            let seq_first = transpose_heads_to_seq_first(floats, nkv, seq, hd);

                            if *is_key {
                                kv.write_layer(*layer, &seq_first, &[]);
                            } else {
                                kv.write_layer(*layer, &[], &seq_first);
                            }

                            if kv_seq_len.is_none() && *is_key {
                                kv_seq_len = Some(seq.max(1));
                            }

                            // Output for downstream attention node.
                            if kv.write_pos() == 0 {
                                // Prefill: pass through original heads-first data.
                                arena.insert(node_id, input_data);
                            } else {
                                // Decode: read full cache (seq-first) and transpose
                                // back to heads-first for the attention kernel.
                                let total_seq = kv.write_pos() + seq;
                                let full = if *is_key {
                                    kv.read_k_through(*layer, seq)
                                } else {
                                    kv.read_v_through(*layer, seq)
                                };
                                let heads_first =
                                    transpose_seq_first_to_heads(full, nkv, total_seq, hd);
                                arena.insert(
                                    node_id,
                                    bytemuck::cast_slice::<f32, u8>(&heads_first).to_vec(),
                                );
                                shape_map.insert(node_id, vec![nkv, total_seq, hd]);
                            }
                            kv_handled.insert(node_id);
                        }
                    }
                }
            }

            // If KV cache handled some nodes, create a filtered level
            // that excludes them (they're already in the arena).
            let effective_level = if kv_handled.is_empty() {
                level.clone()
            } else {
                let filtered_ids: Vec<NodeId> = level
                    .node_ids
                    .iter()
                    .copied()
                    .filter(|id| !kv_handled.contains(id))
                    .collect();
                ParallelLevel {
                    node_ids: filtered_ids,
                }
            };

            let count = dispatch_level(
                &effective_level,
                &dctx,
                &mut arena,
                &mut shape_map,
                &mut results_buf,
                exec_ctx.as_ref(),
                #[cfg(feature = "profile")]
                &mut { crate::profile::PerfProfile::new() },
            )?;

            on_level(i, count);
        }

        // Advance KV cache position after all layers have been written.
        if let (Some(kv), Some(seq)) = (kv_state, kv_seq_len) {
            kv.advance(seq);
        }

        extract_named_outputs(sg, &mut arena)
    }

    pub(crate) fn execute_core_with_hints<F>(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        registry: Option<&CustomOpRegistry>,
        weights: &[u8],
        mut on_level: F,
        shape_hints: Option<&HashMap<u32, Vec<usize>>>,
    ) -> ExecResult<GraphOutputs>
    where
        F: FnMut(usize, usize),
    {
        let compiled_dtypes = sg.node_dtypes_map();
        let mut arena = seed_arena(sg, weights, &compiled_dtypes)?;
        let mut shape_map = seed_shape_map(sg, &arena, inputs, &compiled_dtypes);

        let dctx = DispatchContext {
            table: CompiledNodeTable::from_graph(sg),
            node_map: build_node_map(sg),
            compiled_shapes: sg.node_shapes_map(),
            compiled_dtypes,
            inputs,
            constants: &sg.constants,
            registry,
            weights,
        };

        #[cfg(feature = "profile")]
        let mut prof = crate::profile::PerfProfile::new();
        #[cfg(feature = "profile")]
        prof.start_total();

        let max_level_size = schedule
            .levels
            .iter()
            .map(|l| l.node_ids.len())
            .max()
            .unwrap_or(0);
        let mut results_buf: Vec<(NodeId, Vec<u8>, Vec<usize>)> =
            Vec::with_capacity(max_level_size);

        for (i, level) in schedule.levels.iter().enumerate() {
            // Pre-propagate shapes for this level before data dispatch.
            #[cfg(feature = "profile")]
            let shape_start = std::time::Instant::now();

            super::shape_propagate::propagate_level_shapes(
                level,
                &dctx.node_map,
                &arena,
                &mut shape_map,
                &dctx.compiled_shapes,
                &dctx.compiled_dtypes,
                shape_hints,
            );

            #[cfg(feature = "profile")]
            let shape_elapsed = shape_start.elapsed();
            #[cfg(feature = "profile")]
            let dispatch_start = std::time::Instant::now();

            let count = dispatch_level(
                level,
                &dctx,
                &mut arena,
                &mut shape_map,
                &mut results_buf,
                None, // No execution context for non-KV path
                #[cfg(feature = "profile")]
                &mut prof,
            )?;

            #[cfg(feature = "profile")]
            prof.record_level(shape_elapsed, dispatch_start.elapsed(), count);

            on_level(i, count);
        }

        #[cfg(feature = "profile")]
        {
            prof.stop_total();
            prof.print_summary();
        }

        extract_named_outputs(sg, &mut arena)
    }

    /// Execute and capture all intermediate node buffers + shapes.
    ///
    /// Returns both the normal `GraphOutputs` and a snapshot of every
    /// intermediate buffer in the arena. Intended for conformance testing
    /// only — clones all intermediate data.
    #[cfg(feature = "profile")]
    pub fn execute_with_intermediates(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        weights: &[u8],
    ) -> ExecResult<IntermediateCapture> {
        let compiled_dtypes = sg.node_dtypes_map();
        let mut arena = seed_arena(sg, weights, &compiled_dtypes)?;
        let mut shape_map = seed_shape_map(sg, &arena, inputs, &compiled_dtypes);

        let dctx = DispatchContext {
            node_map: build_node_map(sg),
            compiled_shapes: sg.node_shapes_map(),
            compiled_dtypes,
            inputs,
            constants: &sg.constants,
            registry: None,
            weights,
        };

        let max_level_size = schedule
            .levels
            .iter()
            .map(|l| l.node_ids.len())
            .max()
            .unwrap_or(0);
        let mut results_buf: Vec<(NodeId, Vec<u8>, Vec<usize>)> =
            Vec::with_capacity(max_level_size);

        let mut prof = crate::profile::PerfProfile::new();
        prof.start_total();

        for (_i, level) in schedule.levels.iter().enumerate() {
            let shape_start = std::time::Instant::now();
            super::shape_propagate::propagate_level_shapes(
                level,
                &dctx.node_map,
                &arena,
                &mut shape_map,
                &dctx.compiled_shapes,
                &dctx.compiled_dtypes,
                None,
            );
            let shape_elapsed = shape_start.elapsed();
            let dispatch_start = std::time::Instant::now();

            let count = dispatch_level(
                level,
                &dctx,
                &mut arena,
                &mut shape_map,
                &mut results_buf,
                None,
                &mut prof,
            )?;
            prof.record_level(shape_elapsed, dispatch_start.elapsed(), count);
        }

        prof.stop_total();
        prof.print_summary();

        // Snapshot all intermediates before extracting outputs.
        let node_buffers = arena.snapshot();
        let node_shapes = shape_map.snapshot();

        let outputs = extract_named_outputs(sg, &mut arena)?;

        Ok(IntermediateCapture {
            node_buffers,
            node_shapes,
            outputs,
        })
    }

    /// Execute with shape hints and capture all intermediate node buffers.
    ///
    /// Combines [`execute_with_shape_hints`] (correct shapes at seq>1) with
    /// [`execute_with_intermediates`] (capture all node buffers for debugging).
    /// This is the correct way to do node-by-node conformance testing for
    /// models with dynamic shapes (0-sentinel parameters).
    #[cfg(feature = "profile")]
    pub fn execute_with_intermediates_and_shape_hints(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
        weights: &[u8],
        shape_hints: &HashMap<u32, Vec<usize>>,
    ) -> ExecResult<IntermediateCapture> {
        let compiled_dtypes = sg.node_dtypes_map();
        let mut arena = seed_arena(sg, weights, &compiled_dtypes)?;
        let mut shape_map = seed_shape_map(sg, &arena, inputs, &compiled_dtypes);

        let dctx = DispatchContext {
            node_map: build_node_map(sg),
            compiled_shapes: sg.node_shapes_map(),
            compiled_dtypes,
            inputs,
            constants: &sg.constants,
            registry: None,
            weights,
        };

        let max_level_size = schedule
            .levels
            .iter()
            .map(|l| l.node_ids.len())
            .max()
            .unwrap_or(0);
        let mut results_buf: Vec<(NodeId, Vec<u8>, Vec<usize>)> =
            Vec::with_capacity(max_level_size);

        let mut prof = crate::profile::PerfProfile::new();
        prof.start_total();

        for (_i, level) in schedule.levels.iter().enumerate() {
            let shape_start = std::time::Instant::now();
            super::shape_propagate::propagate_level_shapes(
                level,
                &dctx.node_map,
                &arena,
                &mut shape_map,
                &dctx.compiled_shapes,
                &dctx.compiled_dtypes,
                Some(shape_hints),
            );
            let shape_elapsed = shape_start.elapsed();
            let dispatch_start = std::time::Instant::now();

            let count = dispatch_level(
                level,
                &dctx,
                &mut arena,
                &mut shape_map,
                &mut results_buf,
                None,
                &mut prof,
            )?;
            prof.record_level(shape_elapsed, dispatch_start.elapsed(), count);
        }

        prof.stop_total();
        prof.print_summary();

        // Snapshot all intermediates before extracting outputs.
        let node_buffers = arena.snapshot();
        let node_shapes = shape_map.snapshot();

        let outputs = extract_named_outputs(sg, &mut arena)?;

        Ok(IntermediateCapture {
            node_buffers,
            node_shapes,
            outputs,
        })
    }
}

/// Captured intermediate state from graph execution.
///
/// Contains all node buffers and shapes at the end of execution,
/// plus the normal graph outputs. Used for conformance testing.
#[cfg(feature = "profile")]
pub struct IntermediateCapture {
    /// All node buffers: `NodeId → (data_bytes, elem_size)`.
    pub node_buffers: std::collections::HashMap<NodeId, (Vec<u8>, usize)>,
    /// All node shapes: `NodeId → shape_dims`.
    pub node_shapes: std::collections::HashMap<NodeId, Vec<usize>>,
    /// Normal graph outputs.
    pub outputs: GraphOutputs,
}

/// Build a `NodeId → &Node` lookup map for the graph.
/// Transpose KV data from heads-first `[heads, seq, dim]` to seq-first `[seq, heads, dim]`.
///
/// Used when writing ONNX attention output (heads-first) to the KV cache (seq-first).
fn transpose_heads_to_seq_first(
    data: &[f32],
    n_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total = n_heads * seq * head_dim;
    if data.len() < total || seq == 0 || n_heads == 0 || head_dim == 0 {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; total];
    for h in 0..n_heads {
        for s in 0..seq {
            let src = (h * seq + s) * head_dim;
            let dst = (s * n_heads + h) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    out
}

/// Transpose KV data from seq-first `[seq, heads, dim]` to heads-first `[heads, seq, dim]`.
///
/// Used when reading from the KV cache (seq-first) for the attention kernel (heads-first).
fn transpose_seq_first_to_heads(
    data: &[f32],
    n_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total = n_heads * seq * head_dim;
    if data.len() < total || seq == 0 || n_heads == 0 || head_dim == 0 {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; total];
    for s in 0..seq {
        for h in 0..n_heads {
            let src = (s * n_heads + h) * head_dim;
            let dst = (h * seq + s) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    out
}

fn build_node_map(sg: &SerializedGraph) -> HashMap<NodeId, &Node> {
    sg.nodes.iter().map(|n| (n.id, n)).collect()
}

/// Initialize the arena and seed all constant nodes.
///
/// Also seeds elem_sizes from `compiled_dtypes` for every node that has
/// a known dtype, and marks graph Input nodes as I64 (the standard dtype
/// for token IDs and attention masks in LLM models).
fn seed_arena<'a>(
    sg: &'a SerializedGraph,
    weights: &'a [u8],
    compiled_dtypes: &HashMap<NodeId, FloatDType>,
) -> ExecResult<BufferArena<'a>> {
    let mut arena = BufferArena::with_capacity(sg.nodes.len());
    for node in &sg.nodes {
        match &node.op {
            GraphOp::Constant(cid) => {
                let data = resolve_constant_ref(&sg.constants, *cid, weights)?;
                let es = compiled_dtypes
                    .get(&node.id)
                    .map(|d| d.byte_size())
                    .unwrap_or(4);
                arena.insert_borrowed_with_elem_size(node.id, data, es);
            }
            GraphOp::Input => {
                // Input nodes: use compiled dtype if available, else I64
                // (standard for LLM token IDs / attention masks).
                let es = compiled_dtypes
                    .get(&node.id)
                    .map(|d| d.byte_size())
                    .unwrap_or(8); // I64 default for inputs
                arena.set_elem_size(node.id, es);
            }
            _ => {
                // Seed elem_size from compiled_dtypes if available.
                if let Some(dtype) = compiled_dtypes.get(&node.id) {
                    arena.set_elem_size(node.id, dtype.byte_size());
                }
            }
        }
    }
    Ok(arena)
}

/// Initialize shape map with compiled shapes, constant shapes, and graph input shapes.
///
/// Priority: compiled node shapes > caller-provided input shapes > inferred 1-D shapes.
/// Compiled shapes may contain 0-sentinel dimensions (symbolic at compile time);
/// these are resolved at dispatch time from actual buffer sizes.
fn seed_shape_map(
    sg: &SerializedGraph,
    arena: &BufferArena,
    inputs: &GraphInputs,
    compiled_dtypes: &HashMap<NodeId, FloatDType>,
) -> ShapeMap {
    let mut sm = ShapeMap::new();

    // 1. Seed compiled node shapes (from lowering).
    let compiled = sg.node_shapes_map();
    for (node_id, shape) in &compiled {
        sm.insert(*node_id, shape.clone());
    }

    // 2. Seed constant shapes from the constant_shapes table.
    let const_shapes = sg.constant_shapes_map();

    for node in &sg.nodes {
        match &node.op {
            GraphOp::Constant(cid) => {
                // Prefer N-D constant shape from the table, else 1-D from buffer.
                if let Some(shape) = const_shapes.get(cid) {
                    if !compiled.contains_key(&node.id) {
                        sm.insert(node.id, shape.clone());
                    }
                } else if !compiled.contains_key(&node.id) {
                    if let Ok(buf) = arena.get(node.id) {
                        let elem_size = compiled_dtypes
                            .get(&node.id)
                            .map(|d| d.byte_size())
                            .unwrap_or(4);
                        sm.insert(
                            node.id,
                            ShapeMap::infer_1d_with_elem_size(buf.len(), elem_size),
                        );
                    }
                }
            }
            GraphOp::Input => {
                // Caller-provided runtime shape overrides compiled shape
                // (which may have 0-sentinels for batch/seq dims).
                if let Some(slot_idx) = find_input_slot_index(node) {
                    if let Some(shape) = inputs.shape(slot_idx) {
                        sm.insert(node.id, shape.to_vec());
                    }
                }
            }
            _ => {}
        }
    }
    sm
}

/// Find the graph input slot index that an Input node reads from.
fn find_input_slot_index(node: &Node) -> Option<u32> {
    node.inputs.iter().find_map(|slot| match slot.source {
        InputSource::GraphInput { index } => Some(index),
        _ => None,
    })
}

/// Resolve `size=0` sentinels in FloatOps using the input shape's last dimension.
///
/// When the compiler can't determine the last-dim size (symbolic shapes like seq_len),
/// it emits size=0. At runtime we resolve this from the actual input shape.
/// If the shape-based size doesn't divide the input, falls back to the full input length.
fn resolve_dynamic_sizes(
    op: &GraphOp,
    input_shapes: &[Vec<usize>],
    input_refs: &[&[u8]],
    elem_size: usize,
) -> Option<GraphOp> {
    let resolve = |input_idx: usize| -> u32 {
        let shape = input_shapes.get(input_idx);
        let shape_rank = shape.map(|s| s.len()).unwrap_or(0);
        let shape_last = shape.and_then(|s| s.last().copied()).unwrap_or(0);
        // Validate: the last dim must divide the actual buffer size.
        let buf_floats = input_refs
            .get(input_idx)
            .map(|b| b.len() / elem_size)
            .unwrap_or(0);
        // Guard: a 1-D shape `[total_elems]` means shape tracking fell back to
        // flat — `last()` is the total element count, not the row width. Using it
        // as a Softmax/RmsNorm `size` would normalize across the entire tensor
        // instead of per-row. Only trust `shape_last` when rank >= 2.
        if shape_rank >= 2
            && shape_last > 0
            && buf_floats > 0
            && buf_floats.is_multiple_of(shape_last)
        {
            shape_last as u32
        } else if buf_floats > 0 && shape_rank < 2 {
            // 1-D fallback: don't override — return 0 so the compiled size is kept.
            0
        } else if buf_floats > 0 {
            buf_floats as u32
        } else {
            0
        }
    };

    match op {
        GraphOp::Float(fop) => {
            let resolved = match fop {
                FloatOp::Softmax { size } => {
                    // Resolve the last-dim size from the actual input shape.
                    // This handles both the size=0 sentinel (unknown at compile time)
                    // and the stale size=1 case (when dynamic seq_len was concretized
                    // to 1 at compile time but the runtime tensor has seq > 1).
                    let actual = resolve(0);
                    if actual > 0 && actual != *size {
                        FloatOp::Softmax { size: actual }
                    } else {
                        return None;
                    }
                }
                FloatOp::LogSoftmax { size } => {
                    let actual = resolve(0);
                    if actual > 0 && actual != *size {
                        FloatOp::LogSoftmax { size: actual }
                    } else {
                        return None;
                    }
                }
                FloatOp::RmsNorm { size: 0, epsilon } => FloatOp::RmsNorm {
                    size: resolve(0),
                    epsilon: *epsilon,
                },
                FloatOp::LayerNorm { size: 0, epsilon } => FloatOp::LayerNorm {
                    size: resolve(0),
                    epsilon: *epsilon,
                },
                FloatOp::ReduceSum { size: 0 } => FloatOp::ReduceSum { size: resolve(0) },
                FloatOp::ReduceMean { size: 0 } => FloatOp::ReduceMean { size: resolve(0) },
                FloatOp::ReduceMax { size: 0 } => FloatOp::ReduceMax { size: resolve(0) },
                FloatOp::ReduceMin { size: 0 } => FloatOp::ReduceMin { size: resolve(0) },
                FloatOp::ReduceProd { size: 0 } => FloatOp::ReduceProd { size: resolve(0) },
                FloatOp::InstanceNorm { size: 0, epsilon } => FloatOp::InstanceNorm {
                    size: resolve(0),
                    epsilon: *epsilon,
                },
                // Embed: infer embedding dim from weight table shape[-1] when dim=0.
                FloatOp::Embed { dim: 0, quant } => {
                    let dim = input_shapes
                        .get(1)
                        .and_then(|s| s.last())
                        .copied()
                        .unwrap_or(0) as u32;
                    if dim > 0 {
                        FloatOp::Embed { dim, quant: *quant }
                    } else {
                        return None;
                    }
                }
                // Gather: infer row width from weight table shape[-1] when dim=0.
                FloatOp::Gather { dim: 0, dtype } => {
                    let dim = input_shapes
                        .get(1)
                        .and_then(|s| s.last())
                        .copied()
                        .unwrap_or(0) as u32;
                    if dim > 0 {
                        FloatOp::Gather { dim, dtype: *dtype }
                    } else {
                        return None;
                    }
                }
                // Concat: infer row sizes from input shapes when both are 0-sentinels.
                FloatOp::Concat {
                    size_a: 0,
                    size_b: 0,
                    dtype,
                } => {
                    let size_a = input_shapes
                        .first()
                        .and_then(|s| s.last())
                        .copied()
                        .unwrap_or(0) as u32;
                    let size_b = input_shapes
                        .get(1)
                        .and_then(|s| s.last())
                        .copied()
                        .unwrap_or(0) as u32;
                    if size_a > 0 && size_b > 0 {
                        FloatOp::Concat {
                            size_a,
                            size_b,
                            dtype: *dtype,
                        }
                    } else {
                        return None;
                    }
                }
                // Attention: infer head_dim from Q shape when head_dim=0.
                FloatOp::Attention {
                    head_dim: 0,
                    num_q_heads,
                    num_kv_heads,
                    scale,
                    causal,
                    heads_first,
                    ..
                } if *num_q_heads > 0 => {
                    let q_last = input_shapes
                        .first()
                        .and_then(|s| s.last())
                        .copied()
                        .unwrap_or(0);
                    let head_dim = if q_last > 0 {
                        (q_last / *num_q_heads as usize) as u32
                    } else {
                        0
                    };
                    if head_dim > 0 {
                        FloatOp::Attention {
                            head_dim,
                            num_q_heads: *num_q_heads,
                            num_kv_heads: *num_kv_heads,
                            scale: *scale,
                            causal: *causal,
                            heads_first: *heads_first,
                            qk_norm: false,
                            rope: false,
                            rope_base: 0,
                        }
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };
            Some(GraphOp::Float(resolved))
        }
        _ => None,
    }
}

/// Resolve 0-sentinel dimensions in a compiled shape using actual result size.
///
/// Compiled shapes may have 0 for dimensions that were symbolic at compile time
/// (e.g. batch_size, seq_len). At runtime we resolve these from the actual output
/// byte count. Strategy:
/// - Count the number of 0-dims and the product of known dims
/// - If there's exactly one 0-dim, compute it as `total_elements / known_product`
/// - If there are multiple 0-dims, try to inherit from input shapes
/// - If resolution fails, return the shape with 0s unresolved (better than wrong inference)
fn resolve_compiled_shape(
    compiled: &[usize],
    result_bytes: usize,
    input_shapes: &[Vec<usize>],
    elem_size: usize,
) -> Vec<usize> {
    let zero_count = compiled.iter().filter(|&&d| d == 0).count();
    if zero_count == 0 {
        return compiled.to_vec();
    }

    let elem_size = elem_size.max(1);
    let total_elems = result_bytes / elem_size;

    let known_product: usize = compiled.iter().filter(|&&d| d > 0).product();
    let known_product = known_product.max(1);

    let mut resolved = compiled.to_vec();

    if zero_count == 1 {
        // Single unknown dim: compute from total elements.
        let unknown_dim = if known_product > 0 && total_elems > 0 {
            total_elems / known_product
        } else {
            0
        };
        for d in &mut resolved {
            if *d == 0 {
                *d = unknown_dim;
                break;
            }
        }
    } else {
        // Multiple unknown dims: try to inherit from matching input shape dims.
        for (i, d) in resolved.iter_mut().enumerate() {
            if *d == 0 {
                for in_shape in input_shapes {
                    if let Some(&in_dim) = in_shape.get(i) {
                        if in_dim > 0 {
                            *d = in_dim;
                            break;
                        }
                    }
                }
            }
        }
        // If still have 0s and exactly one remains, resolve from total.
        let remaining_zeros = resolved.iter().filter(|&&d| d == 0).count();
        if remaining_zeros == 1 {
            let kp: usize = resolved.iter().filter(|&&d| d > 0).product();
            let kp = kp.max(1);
            let unknown = if kp > 0 { total_elems / kp } else { 0 };
            for d in &mut resolved {
                if *d == 0 {
                    *d = unknown;
                    break;
                }
            }
        } else if remaining_zeros == 2 {
            // Two equal unknowns (e.g. seq×seq in attention scores [batch, heads, seq, seq]).
            // If their product is a perfect square, both resolve to sqrt.
            let kp: usize = resolved
                .iter()
                .filter(|&&d| d > 0)
                .product::<usize>()
                .max(1);
            if total_elems > 0 && total_elems.is_multiple_of(kp) {
                let rem = total_elems / kp;
                let isqrt = (rem as f64).sqrt() as usize;
                if isqrt > 0 && isqrt * isqrt == rem {
                    for d in &mut resolved {
                        if *d == 0 {
                            *d = isqrt;
                        }
                    }
                }
            }
        }
    }
    resolved
}

/// Resolve a `ShapeSpec` against actual runtime input shapes and output buffer size.
///
/// Legacy resolver kept for compatibility. New code should use
/// `shape_resolve::resolve_float_shape()` instead.
pub fn resolve_shape_spec(
    spec: &ShapeSpec,
    input_shapes: &[Vec<usize>],
    result_bytes: usize,
    elem_size: usize,
) -> Vec<usize> {
    let elem_size = elem_size.max(1);
    let total_elems = result_bytes / elem_size;

    match spec {
        ShapeSpec::SameAs(i) => input_shapes
            .get(*i as usize)
            .cloned()
            .unwrap_or_else(|| vec![total_elems]),

        ShapeSpec::Broadcast(a, b) => {
            let sa = input_shapes.get(*a as usize);
            let sb = input_shapes.get(*b as usize);
            match (sa, sb) {
                (Some(a_shape), Some(b_shape)) if b_shape.len() > a_shape.len() => b_shape.clone(),
                (Some(a_shape), _) => a_shape.clone(),
                (_, Some(b_shape)) => b_shape.clone(),
                _ => vec![total_elems],
            }
        }

        ShapeSpec::DropLastDim(i) => {
            if let Some(s) = input_shapes.get(*i as usize) {
                if s.len() > 1 {
                    s[..s.len() - 1].to_vec()
                } else {
                    vec![1]
                }
            } else {
                vec![total_elems]
            }
        }

        ShapeSpec::Dims(dims) => {
            let mut shape = Vec::with_capacity(dims.len());
            let mut known_product = 1usize;
            let mut inferred_idx = None;

            for (i, dim) in dims.iter().enumerate() {
                match dim {
                    ShapeDim::Fixed(v) => {
                        let v = *v as usize;
                        shape.push(v);
                        known_product = known_product.saturating_mul(v.max(1));
                    }
                    ShapeDim::FromInput { input, axis } => {
                        let v = input_shapes
                            .get(*input as usize)
                            .and_then(|s| {
                                let idx = if *axis < 0 {
                                    s.len().wrapping_add(*axis as usize)
                                } else {
                                    *axis as usize
                                };
                                s.get(idx).copied()
                            })
                            .unwrap_or(1);
                        shape.push(v);
                        known_product = known_product.saturating_mul(v.max(1));
                    }
                    ShapeDim::Inferred => {
                        shape.push(0); // placeholder
                        inferred_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = inferred_idx {
                shape[idx] = if known_product > 0 {
                    total_elems / known_product
                } else {
                    total_elems
                };
            }
            shape
        }

        ShapeSpec::BroadcastAll => {
            // Use highest-rank input shape as the output shape.
            input_shapes
                .iter()
                .max_by_key(|s| s.len())
                .cloned()
                .unwrap_or_else(|| vec![total_elems])
        }

        ShapeSpec::Custom => {
            // Caller must handle Custom separately.
            vec![total_elems]
        }
    }
}

/// Execute all non-constant nodes in a single level; returns the count dispatched.
fn dispatch_level(
    level: &ParallelLevel,
    dctx: &DispatchContext<'_>,
    arena: &mut BufferArena,
    shape_map: &mut ShapeMap,
    results: &mut Vec<(NodeId, Vec<u8>, Vec<usize>)>,
    ctx: Option<&ExecutionContext>,
    #[cfg(feature = "profile")] prof: &mut crate::profile::PerfProfile,
) -> ExecResult<usize> {
    results.clear();

    for &node_id in &level.node_ids {
        // Use flat table for O(1) lookup without hashing in the hot dispatch path.
        let node = dctx
            .table
            .get_node(node_id)
            .ok_or(ExecError::NodeNotFound(node_id))?;

        if matches!(node.op, GraphOp::Constant(_)) {
            continue;
        }

        let input_refs = gather_inputs(node, arena, dctx.inputs)?;

        // ── Fast path: skip input_shapes for ops that don't need them ────
        // Unary elementwise/bytebool ops never use shapes in dispatch.
        // Binary elementwise ops only need shapes when broadcasting
        // (different-sized inputs). This eliminates per-input HashMap
        // lookups for the majority of nodes.
        let op_category = match &node.op {
            GraphOp::Float(fop) => Some(fop.category()),
            _ => None,
        };
        let skip_shapes = match op_category {
            Some(
                OpCategory::UnaryElementwise | OpCategory::UnaryByteBool | OpCategory::UnaryToU8,
            ) => true,
            Some(
                OpCategory::BinaryElementwise
                | OpCategory::BinaryCompare
                | OpCategory::BinaryByteBool,
            ) if input_refs.len() >= 2 && input_refs[0].len() == input_refs[1].len() => true,
            _ => false,
        };

        // Gather input shapes for shape-aware ops.
        // When compiled shapes contain 0-sentinels (dynamic dims like seq_len),
        // resolve them from the actual input buffer sizes.
        let input_shapes: Vec<Vec<usize>> = if skip_shapes {
            Vec::new()
        } else {
            node.inputs
                .iter()
                .zip(input_refs.iter())
                .map(|(slot, buf)| {
                    let dep_id = match slot.source {
                        InputSource::Node(id) => Some(id),
                        _ => None,
                    };
                    let es = dep_id
                        .and_then(|id| dctx.compiled_dtypes.get(&id))
                        .map(|d| d.byte_size())
                        .unwrap_or(4);
                    let raw = dep_id
                        .and_then(|id| shape_map.get(id).map(|s| s.to_vec()))
                        .unwrap_or_else(|| ShapeMap::infer_1d_with_elem_size(buf.len(), es));
                    // Resolve 0-sentinel dims from actual buffer size.
                    if raw.contains(&0) {
                        let resolved = resolve_compiled_shape(&raw, buf.len(), &[], es);
                        if !resolved.contains(&0) {
                            return resolved;
                        }
                    }
                    // Validate that the tracked shape actually matches the buffer.
                    // If not (e.g. shape was incorrectly corrected by correct_stale_shape),
                    // try the compiled shape with 0-sentinels resolved — which preserves
                    // the true dimensionality (n_heads, head_dim) from the compiler.
                    let n_elems = if es > 0 { buf.len() / es } else { 0 };
                    let raw_prod: usize = raw.iter().product();
                    if n_elems > 0 && raw_prod != n_elems {
                        if let Some(compiled) = dep_id.and_then(|id| dctx.compiled_shapes.get(&id))
                        {
                            let resolved = resolve_compiled_shape(compiled, buf.len(), &[], es);
                            let resolved_prod: usize = resolved.iter().product();
                            if !resolved.contains(&0) && resolved_prod == n_elems {
                                return resolved;
                            }
                        }
                    }
                    raw
                })
                .collect()
        };

        // Handle shape-aware ops (Reshape, Transpose, Shape) specially.
        #[cfg(feature = "profile")]
        let op_start = std::time::Instant::now();
        #[cfg(feature = "profile")]
        let op_name = crate::profile::op_name(&node.op);

        let (result, out_shape) = match &node.op {
            GraphOp::Float(FloatOp::Shape { dtype, start, end }) => {
                // Return (a slice of) the input's logical shape as an i64 tensor.
                // `start`/`end` implement ONNX opset-15 Shape attributes — they
                // restrict the output to `dims[start..end]`.  Defaults: start=0,
                // end=i64::MAX (all dims).  Negative values count from the rank end.
                let in_shape = &input_shapes[0];
                let elem_size = dtype.byte_size().max(1);
                let in_bytes = input_refs[0].len();

                // Resolve input shape for the Shape op's i64 output.
                // The walker's shape_map may have wrong RANK (e.g., 5-dim instead
                // of 4-dim) due to broken i64 propagation chains in Reshape targets.
                // Prefer the compiled shape's rank (authoritative) and resolve
                // 0-sentinel dims from the actual buffer size.
                let total_elems = in_bytes / elem_size;
                let input_nid = node.inputs.first().and_then(|s| match s.source {
                    InputSource::Node(id) => Some(id),
                    _ => None,
                });
                let compiled_input_shape = input_nid.and_then(|id| dctx.compiled_shapes.get(&id));

                let mut shape_i64: Vec<i64> = if let Some(cs) = compiled_input_shape {
                    // Compiled shape has authoritative rank. Resolve 0-sentinels.
                    let resolved = resolve_compiled_shape(cs, in_bytes, &input_shapes, elem_size);
                    if !resolved.contains(&0) && resolved.iter().product::<usize>() == total_elems {
                        resolved.iter().map(|&d| d as i64).collect()
                    } else if !in_shape.is_empty()
                        && !in_shape.contains(&0)
                        && in_shape.len() == cs.len()
                        && in_shape.iter().product::<usize>() == total_elems
                    {
                        // Walker shape has matching rank and product — use it.
                        in_shape.iter().map(|&d| d as i64).collect()
                    } else {
                        // Can't resolve — fall back to flat.
                        vec![total_elems as i64]
                    }
                } else if !in_shape.is_empty()
                    && !in_shape.contains(&0)
                    && in_shape.iter().product::<usize>() == total_elems
                {
                    // No compiled shape — trust the walker.
                    in_shape.iter().map(|&d| d as i64).collect()
                } else {
                    // Tracked shape has 0-sentinels or is inconsistent.
                    // Try compiled shape, resolving sentinels.
                    let resolved = compiled_input_shape
                        .map(|cs| resolve_compiled_shape(cs, in_bytes, &input_shapes, elem_size))
                        .filter(|s| !s.contains(&0) && s.iter().product::<usize>() == total_elems);
                    if let Some(good) = resolved {
                        good.iter().map(|&d| d as i64).collect()
                    } else if !in_shape.is_empty() && in_shape.len() > 1 && in_shape.contains(&0) {
                        // Shape has 0-sentinels for dynamic dims but rank is known.
                        // Resolve what we can from the actual buffer size: fill 0-sentinel
                        // dims by dividing total_elems by the product of known dims.
                        // This preserves the tensor's rank so downstream Gather/Slice
                        // ops on the shape output don't fail with wrong element count.
                        let known_product: usize = in_shape
                            .iter()
                            .filter(|&&d| d > 0)
                            .product::<usize>()
                            .max(1);
                        let inferred = if total_elems > 0 && total_elems >= known_product {
                            total_elems / known_product
                        } else {
                            0
                        };
                        in_shape
                            .iter()
                            .map(|&d| if d == 0 { inferred as i64 } else { d as i64 })
                            .collect()
                    } else {
                        // Last resort: flatten to 1-D with correct element count.
                        vec![total_elems as i64]
                    }
                };

                // Apply start/end slicing (ONNX opset 15 Shape attributes).
                let rank = shape_i64.len() as i64;
                let s = if *start < 0 {
                    (rank + start).max(0) as usize
                } else {
                    (*start as usize).min(shape_i64.len())
                };
                let e = if *end == i64::MAX {
                    shape_i64.len()
                } else if *end < 0 {
                    (rank + end).max(0) as usize
                } else {
                    (*end as usize).min(shape_i64.len())
                };
                shape_i64 = shape_i64[s..e].to_vec();

                let data: Vec<u8> = bytemuck::cast_slice(&shape_i64).to_vec();
                let out_shape = vec![shape_i64.len()];
                (data, out_shape)
            }
            GraphOp::Float(FloatOp::Concat { dtype, .. }) if input_refs.len() > 2 => {
                // N-input concat: concatenate all input bytes.
                let mut data = Vec::new();
                for inp in &input_refs {
                    data.extend_from_slice(inp);
                }
                let elem_size = dtype.byte_size().max(1);
                let out_shape = vec![data.len() / elem_size];
                (data, out_shape)
            }
            GraphOp::Float(FloatOp::Reshape) => {
                // Reshape preserves dtype — elem_size comes from the input's arena
                // tracked size, NOT from compiled_dtypes (which may mis-assign dtypes
                // from adjacent nodes like IsNaN that share a node-ID range).
                let elem_size = node
                    .inputs
                    .first()
                    .and_then(|s| match s.source {
                        InputSource::Node(id) => Some(arena.elem_size(id)),
                        _ => None,
                    })
                    .filter(|&es| es > 0)
                    .unwrap_or(4);
                let n_elems = input_refs[0].len() / elem_size;

                // Shape tensor (input 1) is the authoritative target shape for ONNX Reshape.
                // Parse it before consulting shape_map — propagation may have stale values.
                let tensor_shape = if input_refs.len() >= 2 && !input_refs[1].is_empty() {
                    shape_resolve::parse_shape_values(input_refs[1], n_elems)
                } else {
                    None
                };

                // Priority for output shape:
                // 1. **Shape tensor** with exact match — authoritative runtime truth
                //    (computed by the graph's Shape→Gather→Concat chain at runtime).
                //    This takes priority over compiled shapes because compiled shapes
                //    may have 0-sentinels or stale concretized values.
                // 2. Compiled shape with exact match (no 0-sentinels)
                // 3. Shape map (from walker, validated against compiled fixed dims)
                // 4. Shape tensor with expansion (broadcast/GQA key repeat)
                // 5. Flat fallback
                let (data, shape) = {
                    let tensor_exact = tensor_shape
                        .as_ref()
                        .filter(|s| s.iter().product::<usize>() == n_elems);
                    let cs_exact = dctx
                        .compiled_shapes
                        .get(&node_id)
                        .filter(|s| !s.contains(&0) && s.iter().product::<usize>() == n_elems);
                    let reshape_cs = dctx.compiled_shapes.get(&node_id);
                    let sm_exact = shape_map.get(node_id).filter(|s| {
                        if s.contains(&0) || s.iter().product::<usize>() != n_elems {
                            return false;
                        }
                        // Validate fixed dims against compiled shape.
                        if let Some(cs) = reshape_cs {
                            if cs.len() != s.len() {
                                return false;
                            }
                            for (ci, si) in cs.iter().zip(s.iter()) {
                                if *ci > 0 && *ci != *si {
                                    return false;
                                }
                            }
                        }
                        true
                    });

                    if let Some(s) = tensor_exact {
                        (input_refs[0].to_vec(), s.to_vec())
                    } else if let Some(s) = cs_exact {
                        (input_refs[0].to_vec(), s.to_vec())
                    } else if let Some(s) = sm_exact {
                        (input_refs[0].to_vec(), s.to_vec())
                    } else {
                        // Try resolving 0-sentinels in shape_map via actual element count.
                        let sm_resolved = shape_map
                            .get(node_id)
                            .filter(|s| s.contains(&0))
                            .map(|s| {
                                resolve_compiled_shape(
                                    s,
                                    input_refs[0].len(),
                                    &input_shapes,
                                    elem_size,
                                )
                            })
                            .filter(|s| !s.contains(&0) && s.iter().product::<usize>() == n_elems);

                        if let Some(s) = sm_resolved {
                            (input_refs[0].to_vec(), s)
                        } else if let Some(ts) = tensor_shape.as_ref() {
                            // Broadcast expansion (e.g. GQA key repeat): tensor requests more elements.
                            let tp: usize = ts.iter().product();
                            if tp > n_elems && n_elems > 0 && tp.is_multiple_of(n_elems) {
                                match float_dispatch::dispatch_reshape_with_shape(&input_refs) {
                                    Ok((d, _)) => (d, ts.clone()),
                                    Err(_) => (input_refs[0].to_vec(), ts.clone()),
                                }
                            } else {
                                (input_refs[0].to_vec(), vec![n_elems])
                            }
                        } else {
                            (input_refs[0].to_vec(), vec![n_elems])
                        }
                    }
                };
                (data, shape)
            }
            GraphOp::Float(FloatOp::Slice {
                axis_from_end,
                start,
                end,
            }) => {
                let start = *start as usize;
                let end = *end as usize;
                let axis_from_end = *axis_from_end as usize;
                let in_shape = &input_shapes[0];
                let data = input_refs[0];

                // Determine the axis and dim size.
                let ndim = in_shape.len().max(1);
                let axis = ndim.saturating_sub(axis_from_end);
                let axis_size = in_shape.get(axis).copied().unwrap_or(1);
                // end=0 is a 0-sentinel (compiled from dynamic seq_len=0);
                // treat it as "full axis" so the slice is not erroneously empty.
                let actual_end = if end == 0 {
                    axis_size
                } else {
                    end.min(axis_size)
                };

                if start >= actual_end || start >= axis_size {
                    // Empty slice — return empty data with zero shape.
                    let mut out_shape = in_shape.to_vec();
                    if axis < out_shape.len() {
                        out_shape[axis] = 0;
                    }
                    (vec![], out_shape)
                } else {
                    let slice_len = actual_end - start;
                    // Compute strides for the slice.
                    let pre: usize = in_shape[..axis].iter().product::<usize>().max(1);
                    let post: usize = in_shape[axis + 1..].iter().product::<usize>().max(1);
                    let elem_size = dctx
                        .compiled_dtypes
                        .get(&node_id)
                        .or_else(|| {
                            node.inputs.first().and_then(|s| match s.source {
                                InputSource::Node(id) => dctx.compiled_dtypes.get(&id),
                                _ => None,
                            })
                        })
                        .map(|d| d.byte_size())
                        .unwrap_or(4)
                        .max(1);

                    let chunk = post * elem_size; // bytes per element along axis
                    let src_stride = axis_size * chunk;
                    let dst_stride = slice_len * chunk;
                    let out_bytes = pre * dst_stride;

                    let mut out = vec![0u8; out_bytes];
                    for i in 0..pre {
                        let src_off = i * src_stride + start * chunk;
                        let dst_off = i * dst_stride;
                        out[dst_off..dst_off + dst_stride]
                            .copy_from_slice(&data[src_off..src_off + dst_stride]);
                    }

                    let mut out_shape = in_shape.to_vec();
                    if axis < out_shape.len() {
                        out_shape[axis] = slice_len;
                    }
                    (out, out_shape)
                }
            }
            GraphOp::Float(FloatOp::MatMul { m, k, n }) => {
                // Shape is pre-computed by shape_resolve. Use input shapes
                // from shape_map (already resolved by pre-propagation).
                let a_es = input_elem_size(&node.inputs, 0, arena);
                let b_es = input_elem_size(&node.inputs, 1, arena);
                let a_elems = input_refs[0].len() / a_es;
                let b_elems = input_refs[1].len() / b_es;

                // Try batched matmul when A is ≥3-D (batch × M × K).
                // B may be ≥3-D (per-head weight) or 2-D (shared projection
                // weight): dispatch_batched_matmul handles broadcast via
                // b_batch_count, so 2-D B is safe for any batch size in A.
                let batched = if input_shapes.len() >= 2 {
                    let a_prod = input_shapes[0].iter().product::<usize>();
                    let b_prod = input_shapes[1].iter().product::<usize>();
                    let a_ok = input_shapes[0].len() >= 3
                        && !input_shapes[0].contains(&0)
                        && a_prod == a_elems;
                    let b_ok = input_shapes[1].len() >= 2
                        && !input_shapes[1].contains(&0)
                        && b_prod == b_elems;
                    if a_ok && b_ok {
                        Some((input_shapes[0].clone(), input_shapes[1].clone()))
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((a_shape, b_shape)) = batched {
                    float_dispatch::dispatch_batched_matmul(&input_refs, &a_shape, &b_shape)
                        .map_err(|e| ExecError::ShapeMismatch {
                            expected: format!("node {node_id:?} (batched MatMul)"),
                            actual: e.to_string(),
                        })?
                } else {
                    // 2D matmul fallback.
                    let m_hint = *m as usize;
                    let k_hint = *k as usize;
                    let n_hint = *n as usize;
                    let result =
                        float_dispatch::dispatch_matmul(&input_refs, m_hint, k_hint, n_hint)
                            .map_err(|e| ExecError::ShapeMismatch {
                                expected: format!(
                                    "node {node_id:?} (MatMul m={m_hint} k={k_hint} n={n_hint})"
                                ),
                                actual: e.to_string(),
                            })?;
                    // Read pre-computed shape from shape_map, fall back to input-based inference.
                    let mm_elem_size = dctx
                        .compiled_dtypes
                        .get(&node_id)
                        .map(|d| d.byte_size())
                        .unwrap_or(4);
                    let shape = shape_map
                        .get(node_id)
                        .filter(|s| !s.contains(&0))
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|| vec![result.len() / mm_elem_size]);
                    (result, shape)
                }
            }
            GraphOp::Float(FloatOp::Transpose { perm, ndim }) => {
                let nd = *ndim as usize;
                let p = &perm[..nd];
                // Input shape should be resolved in shape_map by pre-propagation.
                let in_shape = input_shapes[0].clone();
                // If input shape has fewer dims than perm expects,
                // pad to match ndim dims.
                let effective_shape = if in_shape.len() < nd {
                    let total: usize = in_shape.iter().copied().fold(1usize, usize::saturating_mul);
                    let mut s = vec![1usize; nd];
                    if nd > 0 {
                        s[0] = total;
                    }
                    s
                } else {
                    in_shape
                };
                let (data, shape) =
                    float_dispatch::dispatch_transpose(input_refs[0], p, &effective_shape)
                        .map_err(|e| ExecError::ShapeMismatch {
                            expected: format!("node {node_id:?} (Transpose)"),
                            actual: e.to_string(),
                        })?;
                (data, shape)
            }
            GraphOp::Float(FloatOp::Where) => {
                // Where(condition, true_val, false_val).
                // The condition may be Bool (1 byte/elem) or f32-encoded booleans.
                // Use elem_size from arena to interpret correctly.
                let cond_es = input_elem_size(&node.inputs, 0, arena);
                let cond_bytes = input_refs[0];
                let x = input_refs.get(1).copied().unwrap_or(&[]);
                let y = input_refs.get(2).copied().unwrap_or(&[]);

                // Convert condition to per-element booleans based on actual dtype.
                let cond_bools: Vec<u8> = if cond_es == 1 {
                    // Already u8 booleans — use directly.
                    cond_bytes.iter().map(|&v| (v != 0) as u8).collect()
                } else if cond_es == 4 {
                    // f32-encoded booleans.
                    bytemuck::cast_slice::<u8, f32>(cond_bytes)
                        .iter()
                        .map(|&v| (v != 0.0) as u8)
                        .collect()
                } else if cond_es == 8 {
                    // i64-encoded booleans.
                    bytemuck::cast_slice::<u8, i64>(cond_bytes)
                        .iter()
                        .map(|&v| (v != 0) as u8)
                        .collect()
                } else {
                    cond_bytes.iter().map(|&v| (v != 0) as u8).collect()
                };

                let x_f32 = bytemuck::cast_slice::<u8, f32>(x);
                let y_f32 = bytemuck::cast_slice::<u8, f32>(y);
                let n = cond_bools.len().max(x_f32.len()).max(y_f32.len());
                let out: Vec<f32> = (0..n)
                    .map(|i| {
                        if cond_bools[i % cond_bools.len()] != 0 {
                            x_f32[i % x_f32.len()]
                        } else {
                            y_f32[i % y_f32.len()]
                        }
                    })
                    .collect();
                let result: Vec<u8> = bytemuck::cast_slice(&out).to_vec();
                let shape = shape_map
                    .get(node_id)
                    .filter(|s| !s.contains(&0))
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| vec![out.len()]);
                (result, shape)
            }
            // Cast I64→F32: detect when input data is already f32 (from hologram's
            // f32-domain lowering). If the input's arena elem_size is 4 (f32) but
            // the Cast says from=I64 (8 bytes), the data was already converted.
            // Pass through as-is to avoid reinterpreting f32 bytes as i64.
            //
            // TEMPORARILY DISABLED — investigating hang.
            // GraphOp::Float(FloatOp::Cast { from: FloatDType::I64, to: FloatDType::F32 })
            // | GraphOp::Float(FloatOp::Cast { from: FloatDType::I32, to: FloatDType::F32 }) => { ... }
            _ => {
                // Resolve size=0 sentinel in FloatOps using the input's last shape dim.
                let input0_es = input_elem_size(&node.inputs, 0, arena);
                let resolved_op =
                    resolve_dynamic_sizes(&node.op, &input_shapes, &input_refs, input0_es);
                let dispatch_op = resolved_op.as_ref().unwrap_or(&node.op);
                let result = KvStore::dispatch_with_shapes(
                    dispatch_op,
                    &input_refs,
                    dctx.constants,
                    dctx.registry,
                    dctx.weights,
                    &input_shapes,
                    ctx,
                )
                .map_err(|e| ExecError::ShapeMismatch {
                    expected: format!("node {node_id:?} ({:?})", node.op),
                    actual: e.to_string(),
                })?;

                // Shape: prefer pre-computed from shape_map (set by pre-propagation pass).
                // Use arena-tracked elem_size (bootstrapped from output_dtype).
                let node_elem_size = compute_output_elem_size(&node_id, dctx, arena);
                let n_result_elems = if node_elem_size > 0 {
                    result.len() / node_elem_size
                } else {
                    0
                };

                let sm_val = shape_map.get(node_id).map(|s| s.to_vec());
                // Validate walker shape hints against compiled shapes.
                // The compiled shape has correct fixed dims (e.g., [32, 0, 64])
                // where 0 = dynamic seq. Reject hints where a non-zero compiled
                // dim doesn't match (e.g., hint [64, 1, 64] vs compiled [32, 0, 64]
                // — dim 0 is 64≠32). This catches broken walk_shape_context
                // projections that produce wrong shapes with correct products.
                let compiled_shape = dctx.table.get_compiled_shape(node_id);
                let shape = sm_val
                    .as_ref()
                    .filter(|s| {
                        if s.contains(&0) {
                            return false;
                        }
                        if n_result_elems > 0 && s.iter().product::<usize>() != n_result_elems {
                            return false;
                        }
                        // Check fixed dims match compiled shape.
                        if let Some(cs) = compiled_shape {
                            if cs.len() != s.len() {
                                return false;
                            }
                            for (ci, si) in cs.iter().zip(s.iter()) {
                                if *ci > 0 && *ci != *si {
                                    return false; // Fixed dim mismatch
                                }
                            }
                        }
                        true
                    })
                    .cloned()
                    .unwrap_or_else(|| {
                        // Pre-propagated shape is absent or stale (product mismatch).
                        //
                        // For shape-preserving ops (ShapeSpec::SameAs — Softmax,
                        // LayerNorm, elementwise unary, etc.) output shape == input
                        // shape.  When the compiled shape is stale (concretized at
                        // seq=1) use input_shapes[0] as the authoritative output shape.
                        //
                        // Limiting to SameAs ops is intentional: for Gemm/Reduce/Embed
                        // the output rank/dims differ from the input, so using
                        // resolve_float_shape would produce wrong dimensionality
                        // (e.g. [2,2048] for a [1,2,2048] output) that cascades into
                        // downstream batched MatMul with wrong strides.
                        if let GraphOp::Float(fop) = dispatch_op {
                            if matches!(fop.output_shape_spec(), ShapeSpec::SameAs(_)) {
                                if let Some(in_shape) = input_shapes.first() {
                                    let prod: usize = in_shape.iter().product();
                                    if !in_shape.is_empty()
                                        && !in_shape.contains(&0)
                                        && (n_result_elems == 0 || prod == n_result_elems)
                                    {
                                        return in_shape.clone();
                                    }
                                }
                            }
                        }
                        // Try to resolve 0-sentinel compiled shape from result buffer.
                        sm_val
                            .as_ref()
                            .map(|cs| {
                                resolve_compiled_shape(
                                    cs,
                                    result.len(),
                                    &input_shapes,
                                    node_elem_size,
                                )
                            })
                            .filter(|s| !s.contains(&0))
                            .unwrap_or_else(|| {
                                ShapeMap::infer_1d_with_elem_size(result.len(), node_elem_size)
                            })
                    });

                (result, shape)
            }
        };

        #[cfg(feature = "profile")]
        prof.record_op(op_name, op_start.elapsed(), result.len());

        results.push((node_id, result, out_shape));
    }

    let dispatched = results.len();
    for (id, data, mut shape) in results.drain(..) {
        // Compute output elem_size from the op's declared output_dtype.
        // Input elem_sizes come from the arena (already tracked for all
        // upstream nodes), making this fully self-bootstrapping.
        let elem_size = compute_output_elem_size(&id, dctx, arena);
        let actual_elems = data.len() / elem_size;
        let shape_prod: usize = shape.iter().product();

        if actual_elems > 0 && shape_prod > 0 && actual_elems != shape_prod {
            // Try to preserve dimensionality by scaling the seq-sentinel dim
            // (a compile-time 1 at a non-batch position that maps to runtime seq_len).
            shape = correct_shape_for_actual_elems(&shape, actual_elems)
                .unwrap_or_else(|| vec![actual_elems]);
        }

        arena.insert_with_elem_size(id, data, elem_size);
        shape_map.insert(id, shape);
    }
    Ok(dispatched)
}

/// Extract the named output buffers from the arena in declaration order.
fn extract_named_outputs(
    sg: &SerializedGraph,
    arena: &mut BufferArena,
) -> ExecResult<GraphOutputs> {
    let mut outputs = Vec::with_capacity(sg.output_names.len());
    for (i, name) in sg.output_names.iter().enumerate() {
        let node_id = sg.output_node_ids[i];
        outputs.push((name.clone(), arena.take(node_id)?));
    }
    Ok(GraphOutputs { outputs })
}

/// Get the element size of a node's input at the given slot index.
///
/// Reads from the arena (which tracks elem_size per node). Falls back to
/// 8 for graph-level inputs (I64 token IDs) and 4 (f32) otherwise.
fn input_elem_size(inputs: &[InputSlot], idx: usize, arena: &BufferArena<'_>) -> usize {
    inputs
        .get(idx)
        .map(|slot| match &slot.source {
            InputSource::Node(src_id) => arena.elem_size(*src_id),
            InputSource::GraphInput { .. } => 8, // I64
            InputSource::None => 4,
        })
        .unwrap_or(4)
}

/// Attempt to correct a shape whose product doesn't match `actual_elems`.
///
/// Handles the common case where a compile-time "1" sentinel (frozen seq_len)
/// needs to be scaled up to match the actual runtime buffer size.
/// Returns `None` if no clean single-dim correction exists.
fn correct_shape_for_actual_elems(shape: &[usize], actual_elems: usize) -> Option<Vec<usize>> {
    use crate::eval::shape_resolve::correct_stale_shape;
    let shape_prod: usize = shape.iter().product();
    if shape_prod == 0 || actual_elems == 0 || shape_prod == actual_elems {
        return None;
    }
    let corrected = correct_stale_shape(shape, actual_elems);
    let new_prod: usize = corrected.iter().product();
    if new_prod == actual_elems && corrected != shape {
        Some(corrected)
    } else {
        None
    }
}

/// Compute the output element size for a node using `FloatOp::output_dtype()`.
///
/// Reads input elem_sizes from the arena (which tracks them per-node),
/// making this self-bootstrapping — no external dtype map needed.
/// Falls back to compiled_dtypes, then 4 (f32 default).
fn compute_output_elem_size(
    node_id: &NodeId,
    dctx: &DispatchContext<'_>,
    arena: &BufferArena<'_>,
) -> usize {
    let node = match dctx.table.get_node(*node_id) {
        Some(n) => n,
        None => {
            return dctx
                .table
                .get_compiled_dtype(*node_id)
                .map(|d| d.byte_size())
                .unwrap_or(4);
        }
    };
    match &node.op {
        GraphOp::Float(fop) => {
            // Build input dtype list from arena elem_sizes.
            let input_dtypes: Vec<FloatDType> = node
                .inputs
                .iter()
                .map(|slot| {
                    let es = match &slot.source {
                        InputSource::Node(src_id) => arena.elem_size(*src_id),
                        InputSource::GraphInput { .. } => 8, // I64 for graph inputs
                        InputSource::None => 4,
                    };
                    FloatDType::from_byte_size(es)
                })
                .collect();
            fop.output_dtype(&input_dtypes).byte_size()
        }
        _ => dctx
            .compiled_dtypes
            .get(node_id)
            .map(|d| d.byte_size())
            .unwrap_or(4),
    }
}

/// Gather input buffers for a node as borrowed slices (zero-copy).
///
/// Returns a `SmallVec<[&[u8]; 4]>` to avoid heap allocation for ops
/// with ≤4 inputs (covers unary, binary, ternary, and most variadic ops).
fn gather_inputs<'a>(
    node: &Node,
    arena: &'a BufferArena,
    graph_inputs: &'a GraphInputs,
) -> ExecResult<SmallVec<[&'a [u8]; 4]>> {
    let mut bufs = SmallVec::with_capacity(node.inputs.len());
    for (slot_idx, slot) in node.inputs.iter().enumerate() {
        match slot.source {
            InputSource::Node(dep_id) => {
                bufs.push(arena.get(dep_id)?);
            }
            InputSource::GraphInput { index } => {
                let data = graph_inputs
                    .get(index)
                    .ok_or(ExecError::MissingGraphInput(index))?;
                bufs.push(data);
            }
            InputSource::None => {
                return Err(ExecError::MissingInput {
                    node: node.id,
                    slot: slot_idx,
                });
            }
        }
    }
    Ok(bufs)
}

/// Resolve a constant ID to a borrowed byte slice (zero-copy).
///
/// `Bytes` constants borrow from the `ConstantStore`'s inline data.
/// `Deferred` constants borrow from the `weights` blob using
/// `source_id` as the byte offset and `byte_size` as the length.
fn resolve_constant_ref<'a>(
    store: &'a hologram_graph::constant::ConstantStore,
    cid: ConstantId,
    weights: &'a [u8],
) -> ExecResult<&'a [u8]> {
    let data = store
        .get(cid)
        .ok_or(ExecError::ConstantNotFound(cid.raw()))?;
    match data {
        ConstantData::Bytes(bytes) => Ok(bytes),
        ConstantData::Deferred {
            byte_size,
            source_id,
        } => {
            let start = *source_id as usize;
            let end = start + *byte_size as usize;
            if end > weights.len() {
                return Err(ExecError::ConstantNotFound(cid.raw()));
            }
            Ok(&weights[start..end])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::build_schedule;
    use hologram_core::op::{LutOp, PrimOp};
    use hologram_graph::constant::ConstantStore;
    use hologram_graph::graph::node::InputSlot;

    fn nid(n: u32) -> NodeId {
        NodeId::new(n, 0)
    }

    fn node(id: u32, op: GraphOp, inputs: Vec<InputSlot>) -> Node {
        Node {
            id: nid(id),
            op,
            inputs: inputs.into_iter().collect(),
            num_outputs: 1,
        }
    }

    fn sg_with_io(
        nodes: Vec<Node>,
        input_names: Vec<&str>,
        output_names: Vec<&str>,
        output_ids: Vec<NodeId>,
    ) -> SerializedGraph {
        SerializedGraph {
            nodes,
            input_names: input_names.into_iter().map(String::from).collect(),
            output_names: output_names.into_iter().map(String::from).collect(),
            output_node_ids: output_ids,
            constants: ConstantStore::new(),
            constant_shapes: Vec::new(),
            node_shapes: Vec::new(),
            node_dtypes: Vec::new(),
        }
    }

    #[test]
    fn passthrough() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(1)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![42, 43, 44]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.by_name("y").unwrap(), &[42, 43, 44]);
    }

    #[test]
    fn relu_pipeline() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(2)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![0, 128, 255]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        let y = result.by_name("y").unwrap();
        assert_eq!(y[0], LutOp::Relu.apply(0));
        assert_eq!(y[1], LutOp::Relu.apply(128));
        assert_eq!(y[2], LutOp::Relu.apply(255));
    }

    #[test]
    fn diamond_graph() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    2,
                    GraphOp::Lut(LutOp::Sigmoid),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    3,
                    GraphOp::Prim(PrimOp::Add),
                    vec![InputSlot::from_node(nid(1)), InputSlot::from_node(nid(2))],
                ),
                node(4, GraphOp::Output, vec![InputSlot::from_node(nid(3))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(4)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![128]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        let y = result.by_name("y").unwrap();
        let expected = LutOp::Relu
            .apply(128)
            .wrapping_add(LutOp::Sigmoid.apply(128));
        assert_eq!(y[0], expected);
    }

    #[test]
    fn two_inputs() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(1, GraphOp::Input, vec![InputSlot::from_graph_input(1)]),
                node(
                    2,
                    GraphOp::Prim(PrimOp::Add),
                    vec![InputSlot::from_node(nid(0)), InputSlot::from_node(nid(1))],
                ),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(2))]),
            ],
            vec!["a", "b"],
            vec!["sum"],
            vec![nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let inputs = GraphInputs::from_pairs(vec![(0, vec![10, 20]), (1, vec![5, 100])]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.by_name("sum").unwrap(), &[15, 120]);
    }

    #[test]
    fn multiple_outputs() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
            ],
            vec!["x"],
            vec!["raw", "activated"],
            vec![nid(2), nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![100]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.by_name("raw").unwrap(), &[100]);
        assert_eq!(
            result.by_name("activated").unwrap(),
            &[LutOp::Relu.apply(100)]
        );
    }

    #[test]
    fn constant_node() {
        let mut constants = ConstantStore::new();
        let cid = constants.insert(ConstantData::Bytes(vec![7, 8, 9]));
        let sg = SerializedGraph {
            nodes: vec![
                node(0, GraphOp::Constant(cid), vec![]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            input_names: Vec::new(),
            output_names: vec!["out".into()],
            output_node_ids: vec![nid(1)],
            constants,
            constant_shapes: Vec::new(),
            node_shapes: Vec::new(),
            node_dtypes: Vec::new(),
        };
        let sched = build_schedule(&sg).unwrap();
        let result = KvExecutor::execute(&sg, &sched, &GraphInputs::new()).unwrap();
        assert_eq!(result.by_name("out").unwrap(), &[7, 8, 9]);
    }

    #[test]
    fn missing_graph_input() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(1)],
        );
        let sched = build_schedule(&sg).unwrap();
        assert!(KvExecutor::execute(&sg, &sched, &GraphInputs::new()).is_err());
    }

    #[test]
    fn chain_of_unary() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    2,
                    GraphOp::Lut(LutOp::Sigmoid),
                    vec![InputSlot::from_node(nid(1))],
                ),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(2))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![128]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        let y = result.by_name("y").unwrap();
        assert_eq!(y[0], LutOp::Sigmoid.apply(LutOp::Relu.apply(128)));
    }

    #[test]
    fn empty_graph() {
        let sg = sg_with_io(vec![], vec![], vec![], vec![]);
        let sched = build_schedule(&sg).unwrap();
        assert!(KvExecutor::execute(&sg, &sched, &GraphInputs::new())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn output_by_index() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(1)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![1, 2, 3]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        let (name, data) = result.get(0).unwrap();
        assert_eq!(name, "y");
        assert_eq!(data, &[1, 2, 3]);
    }

    #[test]
    fn fused_view_execution() {
        use hologram_core::view::ElementWiseView;
        let view = ElementWiseView::new(|x| x.wrapping_mul(3));
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::FusedView(view),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(2)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![1, 2, 3, 10]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap(), &[3, 6, 9, 30]);
    }

    #[test]
    fn xor_self_via_neg() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Prim(PrimOp::Neg),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    2,
                    GraphOp::Prim(PrimOp::Xor),
                    vec![InputSlot::from_node(nid(0)), InputSlot::from_node(nid(1))],
                ),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(2))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![10]);
        let result = KvExecutor::execute(&sg, &sched, &inputs).unwrap();
        assert_eq!(result.by_name("y").unwrap()[0], 10u8 ^ 10u8.wrapping_neg());
    }

    /// Progress callback fires once per level with sequential indices.
    #[test]
    fn progress_callback_fires_per_level() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Output, vec![InputSlot::from_node(nid(1))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(2)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![128]);

        let mut events: Vec<(usize, usize)> = Vec::new();
        let result = KvExecutor::execute_with_progress(&sg, &sched, &inputs, |idx, count| {
            events.push((idx, count));
        })
        .unwrap();

        assert_eq!(result.by_name("y").unwrap(), &[LutOp::Relu.apply(128)]);
        assert!(!events.is_empty());
        for (i, (idx, _)) in events.iter().enumerate() {
            assert_eq!(*idx, i);
        }
    }

    /// Total dispatched node count across all levels equals graph size.
    #[test]
    fn progress_callback_total_count() {
        let sg = sg_with_io(
            vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Lut(LutOp::Relu),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(
                    2,
                    GraphOp::Lut(LutOp::Sigmoid),
                    vec![InputSlot::from_node(nid(1))],
                ),
                node(3, GraphOp::Output, vec![InputSlot::from_node(nid(2))]),
            ],
            vec!["x"],
            vec!["y"],
            vec![nid(3)],
        );
        let sched = build_schedule(&sg).unwrap();
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![64]);

        let mut total = 0usize;
        KvExecutor::execute_with_progress(&sg, &sched, &inputs, |_, count| total += count).unwrap();

        assert_eq!(total, 4); // Input, Relu, Sigmoid, Output
    }

    #[test]
    fn deferred_constant_with_weights() {
        use hologram_graph::constant::ConstantData;
        let mut constants = ConstantStore::new();
        let cid = constants.insert(ConstantData::Deferred {
            byte_size: 3,
            source_id: 0,
        });
        let sg = SerializedGraph {
            nodes: vec![
                node(0, GraphOp::Constant(cid), vec![]),
                node(1, GraphOp::Output, vec![InputSlot::from_node(nid(0))]),
            ],
            input_names: Vec::new(),
            output_names: vec!["out".into()],
            output_node_ids: vec![nid(1)],
            constants,
            constant_shapes: Vec::new(),
            node_shapes: Vec::new(),
            node_dtypes: Vec::new(),
        };
        let sched = build_schedule(&sg).unwrap();
        let weights = vec![10, 20, 30, 99];
        let result =
            KvExecutor::execute_with_plan(&sg, &sched, &GraphInputs::new(), &weights).unwrap();
        assert_eq!(result.by_name("out").unwrap(), &[10, 20, 30]);
    }

    /// Shape op with 0-sentinel dims preserves rank instead of flattening to 1-D.
    ///
    /// Regression test for TinyLlama NodeId(498): when the input shape has a
    /// 0-sentinel for a dynamic dim (e.g. `[1, 4, 0, 8]`), the Shape op must
    /// output 4 i64 values (preserving the rank), not 1 value (flat element count).
    /// Otherwise, a downstream Gather on the shape tensor fails with
    /// "expected i64 index < 1, got index = 2".
    #[test]
    fn shape_op_preserves_rank_with_zero_sentinel() {
        // Graph: Input(0) → Shape(1) → Gather(2, index_const) → Output(3)
        //
        // Input: f32 data with compiled shape [1, 4, 0, 8] (seq=0 sentinel).
        // Shape output should be 4 i64 values, not 1.
        // Gather picks element at index 2 → the resolved seq value.

        let mut constants = ConstantStore::new();
        // Index constant: i64 scalar = 2
        let index_data: Vec<u8> = 2i64.to_le_bytes().to_vec();
        let index_cid = constants.insert(ConstantData::Bytes(index_data));

        let sg = SerializedGraph {
            nodes: vec![
                node(0, GraphOp::Input, vec![InputSlot::from_graph_input(0)]),
                node(
                    1,
                    GraphOp::Float(FloatOp::Shape {
                        dtype: FloatDType::F32,
                        start: 0,
                        end: i64::MAX,
                    }),
                    vec![InputSlot::from_node(nid(0))],
                ),
                node(2, GraphOp::Constant(index_cid), vec![]),
                node(
                    3,
                    GraphOp::Float(FloatOp::Gather {
                        dim: 1,
                        dtype: FloatDType::I64,
                    }),
                    // Gather inputs: (indices, data) — swapped from ONNX order
                    vec![
                        InputSlot::from_node(nid(2)), // indices = constant [2]
                        InputSlot::from_node(nid(1)), // data = Shape output
                    ],
                ),
                node(4, GraphOp::Output, vec![InputSlot::from_node(nid(3))]),
            ],
            input_names: vec!["X".into()],
            output_names: vec!["dim_val".into()],
            output_node_ids: vec![nid(4)],
            constants,
            constant_shapes: Vec::new(),
            // Compiled shape for node 0: [1, 4, 0, 8] (0-sentinel for seq)
            node_shapes: vec![(nid(0), vec![1, 4, 0, 8])],
            // Node 0 is F32, node 1 Shape output is I64
            node_dtypes: vec![
                (nid(0), FloatDType::F32),
                (nid(1), FloatDType::I64),
                (nid(2), FloatDType::I64),
                (nid(3), FloatDType::I64),
            ],
        };

        let sched = build_schedule(&sg).expect("schedule");

        // Provide actual input: [1, 4, 3, 8] (seq=3) → 96 f32 elements → 384 bytes
        let mut inputs = GraphInputs::new();
        let data: Vec<f32> = (0..96).map(|i| i as f32).collect();
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        inputs.set_with_shape(0, bytes, vec![1, 4, 3, 8]);

        let result = KvExecutor::execute(&sg, &sched, &inputs).expect(
            "Shape op with 0-sentinel should preserve rank, not flatten to 1-D. \
             If this fails with 'expected i64 index < 1', the Shape handler is \
             falling back to [total_elems] instead of [1, 4, 3, 8].",
        );

        let out = result.by_name("dim_val").expect("output not found");
        // Gather picked element at index 2 from Shape output [1, 4, 3, 8] → value 3
        let val = i64::from_le_bytes(out.try_into().expect("expected 8 bytes"));
        assert_eq!(
            val, 3,
            "Gather(Shape([1,4,3,8]), idx=2) should be 3 (the seq dim), got {val}. \
             Shape op likely flattened to [96] instead of preserving rank."
        );
    }
}
