//! `Compiler` (spec VII.1, VII.2).
//!
//! Per-node pipeline:
//!   1. Lookup op marker for `node.op_kind`.
//!   2. Resolve concrete shape/dtype/host-bounds generics.
//!   3. Emit Term tree into TermArena.
//!   4. Build CompileUnit via CompileUnitBuilder.
//!   5. Validate.
//!   6. Run completeness via pipeline::run_tower_completeness.
//!   7. Cache by ContentFingerprint<32>.
//!   8. Lower to backend KernelCall.
//!   9. Emit (kernel_call, certificate, fingerprint) into archive.

use alloc::boxed::Box;
use alloc::vec::Vec;

use hologram_archive::certificate_codec::{self, CertificateRecord};
use hologram_archive::constant_codec::ConstantEntry;
use hologram_archive::{HoloWriter, PortDescriptor, WeightStore};
use hologram_backend::{BufferRef, KernelCall, MAX_RANK};
use hologram_graph::{Graph, GraphOp};
use hologram_host::HologramHasher;
use hologram_ops::{HoloArena, HoloTerm};
use prism::operation::Term;
use prism::vocabulary::{Hasher, VerificationDomain, WittLevel};
// `Binding` is foundation-only (not in prism::operation's curated
// surface as of 0.1.3); reach it through prism's substrate re-export.
use crate::cache::{CachedCertificate, CertificateCache};
use crate::error::CompileError;
use crate::lower::{self, LoweredNode};
use crate::pipeline::{self as compile_pipeline, PerNodeUnit};
use prism::uor_foundation::enforcement::Binding;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Avx2,
    Avx512,
    Neon,
    Metal,
    Wgpu,
}

impl BackendKind {
    pub const fn name(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Avx2 => "avx2",
            BackendKind::Avx512 => "avx512",
            BackendKind::Neon => "neon",
            BackendKind::Metal => "metal",
            BackendKind::Wgpu => "wgpu",
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CompilationStats {
    pub total_nodes: u32,
    pub schedule_levels: u32,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub validated_units: u32,
}

pub struct CompilationOutput {
    pub archive: Vec<u8>,
    pub stats: CompilationStats,
}

pub struct Compiler {
    graph: Graph,
    target: BackendKind,
    level: WittLevel,
    /// Per-compile certificate cache (spec VII.4). Populated as nodes lower
    /// and consulted before recomputing identical (op, type, backend) triples.
    pub cache: CertificateCache,
}

impl Compiler {
    pub fn new(graph: Graph, target: BackendKind, level: WittLevel) -> Self {
        Self {
            graph,
            target,
            level,
            cache: CertificateCache::new(),
        }
    }

    pub fn compile(mut self) -> Result<CompilationOutput, CompileError> {
        // Path B: desugar composite ops (e.g. Clip → Min∘Max) into their
        // primitive-op pipelines before scheduling, so the rest of the pipeline
        // sees only primitives + irreducible structured kernels.
        self.graph.desugar_composites();

        // Algebraic elision: drop computation UOR's algebra proves
        // unnecessary (identity elements, involutions, dead nodes) so it is
        // never scheduled, dispatched, or addressed. Runs after desugaring so
        // it also simplifies the primitive pipelines composites expand into.
        self.graph.elide_invariants();

        // Schedule.
        self.graph.compute_schedule();

        let mut stats = CompilationStats {
            total_nodes: self.graph.node_count() as u32,
            schedule_levels: self
                .graph
                .schedule()
                .map(|s| s.level_count() as u32)
                .unwrap_or(0),
            cache_hits: 0,
            cache_misses: 0,
            validated_units: 0,
        };

        let mut kernel_calls: Vec<KernelCall> = Vec::with_capacity(self.graph.node_count());
        let mut certificate_records: Vec<CertificateRecord> =
            Vec::with_capacity(self.graph.node_count());

        // Per-level kernel-call indices (spec VIII.2). Each entry holds the
        // call positions in `kernel_calls` that belong to that schedule
        // level; the runtime executor walks them in order, parallelizing
        // within a level where the backend supports it.
        let mut exec_plan: Vec<Vec<u32>> = Vec::new();

        // Per-node element counts derived from the graph's shape registry.
        // `byte_lengths[i]` = element_count * bytes_per_element(dtype) — the
        // size in bytes the workspace slot must hold for node i.
        // u64 with no `.min(u32::MAX)` cap — element counts and byte
        // lengths must not be ceilinged at 4 GiB (ADR-060).
        let element_counts: Vec<u64> = self
            .graph
            .nodes()
            .iter()
            .map(|n| {
                self.graph
                    .shape_registry()
                    .get(n.output_shape)
                    .map(|s| s.total_elements())
                    .unwrap_or(0)
            })
            .collect();
        let byte_lengths: Vec<u64> = self
            .graph
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let bytes_per = bytes_per_element(n.output_dtype.0) as u64;
                element_counts[i].saturating_mul(bytes_per)
            })
            .collect();

        // Emit kernel calls in schedule (topological) order so the executor's
        // sequential walk respects data dependencies even when graph nodes
        // were inserted out of order. Build a `traversal_levels: Vec<Vec<u32>>`
        // grouped by schedule level so we can record kernel-call indices
        // per level for the runtime exec plan.
        let traversal_levels: Vec<Vec<u32>> = match self.graph.schedule() {
            Some(sched) => sched
                .levels
                .iter()
                .map(|level| level.iter().map(|hologram_graph::NodeId(id)| *id).collect())
                .collect(),
            None => vec![(0..self.graph.node_count() as u32).collect()],
        };

        for level_nodes in &traversal_levels {
            let mut level_calls: Vec<u32> = Vec::with_capacity(level_nodes.len());

            for &idx_u32 in level_nodes {
                let idx = idx_u32 as usize;
                let node = match self.graph.nodes().get(idx) {
                    Some(n) => n,
                    None => continue,
                };
                let kind = match node.op {
                    GraphOp::Op(k) => k,
                    GraphOp::Input | GraphOp::Output | GraphOp::Constant(_) => continue,
                };

                // Steps 3-5: emit Term tree and validate the per-node CompileUnit.
                // Materialize a contiguous &[Term] from the arena's `Option<Term>` slots.
                let arena = build_node_arena(kind, self.level)?;
                let term_vec: Vec<HoloTerm> = arena.as_slice().iter().filter_map(|t| *t).collect();
                // Per-arity binding table: each binding references the
                // corresponding `Term::Variable` pushed at the start of the
                // arena (indices 0..arity). The surface strings are
                // ahead-of-time `'static` so binding entries are usable in
                // the upstream `BindingsTable`. Spec O-3.
                let arity = kind.primary_arity() as usize;
                let bindings: &[Binding] = &VAR_BINDINGS[..arity.min(VAR_BINDINGS.len())];
                let domains: &[VerificationDomain] = &[VerificationDomain::Algebraic];
                // build_unit returns `Validated<CompileUnit>`; we only need
                // its side-effect (validation `?` propagates shape errors).
                // The per-node certificate captured below is the persistent
                // archive artifact; the unit itself does not survive.
                compile_pipeline::build_unit(&PerNodeUnit {
                    root_term: &term_vec,
                    bindings,
                    witt_level: self.level,
                    budget: 1,
                    target_domains: domains,
                })?;
                stats.validated_units += 1;

                // Step 6: run completeness against the result type at the active level.
                // Upstream rejects (residual / non-trivial constraints) surface as
                // a per-node CompileError; layout-only ops are exempted because
                // their Term tree is a single Variable (no algebraic content).
                let completeness = compile_pipeline::run_completeness(self.level);
                let cert_record = match (&completeness, kind.is_layout_only()) {
                    (Ok(v), _) => CertificateRecord::from_validated(v),
                    (Err(_), true) => CertificateRecord {
                        witt_bits: self.level.witt_length() as u16,
                        width_bytes: 0,
                        fingerprint: [0u8; 32],
                    },
                    (Err(_), false) => return Err(CompileError::CompletenessFailure),
                };

                // Step 7: cache lookup / insert. The cache is keyed on
                // (op_kind, level, backend); the value is the per-type
                // certificate record. The kernel call itself is *per-node*
                // (different slot wiring and shape parameters per graph node)
                // and is therefore always re-lowered — the cache hit path
                // saves only the validation/completeness work.
                let fingerprint = compute_fingerprint(kind, self.level, self.target);
                if self.cache.get_raw(&fingerprint).is_some() {
                    stats.cache_hits += 1;
                } else {
                    stats.cache_misses += 1;
                    self.cache.insert_raw(
                        fingerprint,
                        CachedCertificate {
                            record: cert_record,
                        },
                    );
                }

                // Step 8: lower to KernelCall using per-node shape-derived sizing.
                let element_count = element_counts.get(idx).copied().unwrap_or(0);
                let byte_len = byte_lengths.get(idx).copied().unwrap_or(0);
                let dtype = node.output_dtype.0;
                // Quantization params (spec X-5). The compiler reads any
                // `set_quant_attrs(NodeId, _)` the graph builder attached and
                // threads them into the lowered node; the K::Dequantize arm
                // consumes them directly.
                let quant_attrs = self
                    .graph
                    .quant_attrs(hologram_graph::NodeId(idx as u32))
                    .unwrap_or_default();
                let lowered = LoweredNode {
                    kind,
                    inputs: collect_buffers(&self.graph, node, &byte_lengths),
                    output: BufferRef {
                        slot: idx as u32,
                        offset: 0,
                        length: byte_len,
                    },
                    element_count,
                    witt_bits: self.level.witt_length() as u16,
                    dtype,
                    shape: lower::ShapeArgs::from_graph(
                        &self.graph,
                        hologram_graph::NodeId(idx as u32),
                        node,
                    ),
                    quant: {
                        // Per-channel (axis ≥ 0): derive channel count + inner
                        // stride from the dequantize input shape; per-tensor
                        // otherwise (channels = 0).
                        let (channels, inner) = if quant_attrs.axis >= 0 {
                            quant_channel_dims(&self.graph, node, quant_attrs.axis as usize)
                                .unwrap_or((0, 0))
                        } else {
                            (0, 0)
                        };
                        lower::QuantParams {
                            quant_dtype: quant_attrs.quant_dtype,
                            scale_bits: quant_attrs.scale_bits,
                            zero_point: quant_attrs.zero_point,
                            channels,
                            inner,
                        }
                    },
                };
                let mut kernel_call = lower::lower(&lowered)?;
                // Slice = `ProjectField`: point the input BufferRef at the
                // sliced sub-region [byte_offset, byte_offset+byte_len) computed
                // from the starts/ends index constants. The copy kernel then
                // reads exactly that field, and the executor turns it into a
                // zero-movement view. Only the axis-0 contiguous, unit-step case
                // is realized; anything else is rejected (no silent-wrong).
                if matches!(kind, hologram_graph::OpKind::Slice) {
                    if let KernelCall::Slice(lc) = &mut kernel_call {
                        let (off, len) = slice_view_bytes(&self.graph, node)
                            .ok_or(CompileError::CompletenessFailure)?;
                        lc.input.offset = off;
                        lc.input.length = len;
                    }
                }
                // Transpose: fill the permutation + input dims from the perm
                // operand (or the default reverse) and the input shape.
                if matches!(kind, hologram_graph::OpKind::Transpose) {
                    if let KernelCall::Transpose(tc) = &mut kernel_call {
                        let (rank, dims, perm) = transpose_plan(&self.graph, node)
                            .ok_or(CompileError::CompletenessFailure)?;
                        tc.rank = rank;
                        tc.dims = dims;
                        tc.perm = perm;
                    }
                }
                // LRN: batch/channels/inner from the rank-4 input shape;
                // size/α/β/bias from the node's LrnAttrs (defaults otherwise).
                if matches!(kind, hologram_graph::OpKind::Lrn) {
                    if let KernelCall::Lrn(lc) = &mut kernel_call {
                        let (batch, channels, inner) =
                            lrn_dims(&self.graph, node).ok_or(CompileError::CompletenessFailure)?;
                        let a = self
                            .graph
                            .lrn_attrs(hologram_graph::NodeId(idx as u32))
                            .unwrap_or_default();
                        lc.batch = batch;
                        lc.channels = channels;
                        lc.inner = inner;
                        lc.size = a.size;
                        lc.alpha_bits = a.alpha_bits;
                        lc.beta_bits = a.beta_bits;
                        lc.bias_bits = a.bias_bits;
                    }
                }
                // RoPE: head_dim = the input's last dimension.
                if matches!(kind, hologram_graph::OpKind::RotaryEmbedding) {
                    if let KernelCall::RotaryEmbedding(rc) = &mut kernel_call {
                        rc.head_dim = rope_head_dim(&self.graph, node)
                            .ok_or(CompileError::CompletenessFailure)?;
                    }
                }
                // GroupNorm/InstanceNorm: fill `num_groups`. GroupNorm reads the
                // node's `NormAttrs` (ONNX `num_groups`, default 1); InstanceNorm
                // is the per-channel case, so `num_groups = channels`. `channels`
                // and per-sample `feature` were derived in `ShapeArgs::from_graph`.
                if matches!(kind, hologram_graph::OpKind::GroupNorm) {
                    if let KernelCall::GroupNorm(nc) = &mut kernel_call {
                        nc.num_groups = self
                            .graph
                            .norm_attrs(hologram_graph::NodeId(idx as u32))
                            .unwrap_or_default()
                            .num_groups
                            .max(1);
                    }
                }
                if matches!(kind, hologram_graph::OpKind::InstanceNorm) {
                    if let KernelCall::InstanceNorm(nc) = &mut kernel_call {
                        nc.num_groups = nc.channels.max(1);
                    }
                }
                // Expand: in_dims (input shape) + out_dims (output shape) for
                // the broadcast gather.
                if matches!(kind, hologram_graph::OpKind::Expand) {
                    if let KernelCall::Expand(ec) = &mut kernel_call {
                        let (rank, in_dims, out_dims) = expand_plan(&self.graph, node)
                            .ok_or(CompileError::CompletenessFailure)?;
                        ec.rank = rank;
                        ec.in_dims = in_dims;
                        ec.out_dims = out_dims;
                    }
                }
                // Reduce: the kernel folds over its *input* elements, so
                // `element_count` is the input count (not the reduced output
                // count). `rank`/`dims` come from the input shape; `axes_mask`/
                // `keepdims` from the node's `ReduceAttrs` (absent ⇒ reduce all
                // axes — full reduction to a scalar, the prior behavior).
                if matches!(
                    kind,
                    hologram_graph::OpKind::ReduceSum
                        | hologram_graph::OpKind::ReduceMean
                        | hologram_graph::OpKind::ReduceProd
                        | hologram_graph::OpKind::ReduceMin
                        | hologram_graph::OpKind::ReduceMax
                ) {
                    if let (
                        Some(in_count),
                        KernelCall::ReduceSum(rc)
                        | KernelCall::ReduceMean(rc)
                        | KernelCall::ReduceProd(rc)
                        | KernelCall::ReduceMin(rc)
                        | KernelCall::ReduceMax(rc),
                    ) = (input_element_count(&self.graph, node), &mut kernel_call)
                    {
                        rc.element_count = in_count;
                        let attrs = self.graph.reduce_attrs(hologram_graph::NodeId(idx as u32));
                        if let Some((rank, dims)) = reduce_input_dims(&self.graph, node) {
                            rc.rank = rank;
                            rc.dims = dims;
                            // Absent attrs ⇒ reduce all axes (mask 0 is the
                            // kernel's "full reduction" sentinel).
                            rc.axes_mask = attrs.map(|a| a.axes_mask).unwrap_or(0);
                            rc.keepdims = attrs.map(|a| a.keepdims).unwrap_or(false);
                        }
                    }
                }
                // Resize: same in/out dims (no broadcast constraint) — the
                // kernel maps each output index to the nearest input index.
                if matches!(kind, hologram_graph::OpKind::Resize) {
                    if let KernelCall::Resize(ec) = &mut kernel_call {
                        let (rank, in_dims, out_dims) = reindex_dims(&self.graph, node)
                            .ok_or(CompileError::CompletenessFailure)?;
                        ec.rank = rank;
                        ec.in_dims = in_dims;
                        ec.out_dims = out_dims;
                    }
                }
                // Gather: flatten the data shape to [outer, axis_dim, inner]
                // around the GatherAttrs axis, and count the indices — the
                // kernel's indexed-copy geometry. `idx_dtype` is the indices
                // operand's dtype (i32/i64).
                if matches!(kind, hologram_graph::OpKind::Gather) {
                    if let KernelCall::Gather(gc) = &mut kernel_call {
                        let (outer, axis_dim, inner, num_indices, idx_dtype) =
                            gather_plan(&self.graph, node, hologram_graph::NodeId(idx as u32))
                                .ok_or(CompileError::CompletenessFailure)?;
                        gc.outer = outer;
                        gc.axis_dim = axis_dim;
                        gc.inner = inner;
                        gc.num_indices = num_indices;
                        gc.idx_dtype = idx_dtype;
                    }
                }
                // Cast: the destination dtype is the node's output dtype (set at
                // lowering); fill the source dtype from the input operand so the
                // kernel knows both ends of the numeric conversion.
                if matches!(kind, hologram_graph::OpKind::Cast) {
                    if let KernelCall::Cast(cc) = &mut kernel_call {
                        cc.src_dtype = operand_dtype(&self.graph, node, 0)
                            .ok_or(CompileError::CompletenessFailure)?;
                    }
                }
                // Pad = placement into a zeroed buffer: write the data into the
                // output's interior [lo, lo+data) (axis-0). The fresh output
                // buffer is zeroed, so the pad regions remain zero.
                if matches!(kind, hologram_graph::OpKind::Pad) {
                    if let KernelCall::Pad(lc) = &mut kernel_call {
                        let (lo_off, data_len, data_count) = pad_view_bytes(&self.graph, node)
                            .ok_or(CompileError::CompletenessFailure)?;
                        lc.input = BufferRef {
                            slot: lc.input.slot,
                            offset: 0,
                            length: data_len,
                        };
                        lc.output.offset = lo_off;
                        lc.output.length = data_len;
                        lc.element_count = data_count;
                    }
                }
                level_calls.push(kernel_calls.len() as u32);
                kernel_calls.push(kernel_call);
                certificate_records.push(cert_record);
            }

            if !level_calls.is_empty() {
                exec_plan.push(level_calls);
            }
        }

        // Warm-start (WS class) is *not* emitted here. A κ-label is a
        // deterministic function of the compiled graph, so the runtime derives
        // the constant-only-cone lattice itself at load (post-fusion, always
        // matching execution) — baking the labels would be redundant. The
        // archive carries only the cone's *materialized results*, and those
        // require running kernels, so they are baked by the post-compile fold
        // pass (`hologram_exec::fold_archive`, run by the CLI), not the
        // (execution-free) compiler.
        let node_count = self.graph.node_count() as u32;

        // ── Weight-layout monomorphism (UOR-native, zero runtime copy) ──
        //
        // A matmul's B operand is its weight. When that weight is a *constant*
        // (known at compile time) consumed by this matmul alone, pre-pack it
        // into the panel layout the cache-oblivious leaf streams contiguously
        // (`hologram_backend::layout`, the shared single source of truth for
        // the layout). The packing is a compile-time data-
        // representation transform baked into the archive — part of the single
        // monomorphism the ONNX model compiles to — so at runtime the kernel
        // reads B with no strided gather and **no copy**. f32 only; the packed
        // weight is content-addressed by its (packed) bytes like any constant.
        const DTYPE_F32: u8 = 8;
        let n_const = self.graph.constants().len();
        let mut packed_consts: Vec<Option<Vec<u8>>> = vec![None; n_const];
        {
            // Census of slot reads/writes across all calls: a constant the
            // matmul uniquely consumes (count == 1) packs unambiguously.
            let mut uses: hashbrown::HashMap<u32, u32> = hashbrown::HashMap::new();
            for call in kernel_calls.iter() {
                for bref in hologram_backend::buffers(call) {
                    if bref.slot != u32::MAX {
                        *uses.entry(bref.slot).or_insert(0) += 1;
                    }
                }
            }
            for call in kernel_calls.iter_mut() {
                if let KernelCall::MatMul(mm) = call {
                    if mm.dtype != DTYPE_F32 || mm.b_packed || mm.b.slot < node_count {
                        continue;
                    }
                    let cid = (mm.b.slot - node_count) as usize;
                    if cid >= n_const || uses.get(&mm.b.slot) != Some(&1) {
                        continue;
                    }
                    let (k, n) = (mm.k as usize, mm.n as usize);
                    let entry = match self
                        .graph
                        .constants()
                        .get(hologram_graph::ConstantId(cid as u32))
                    {
                        Some(e) => e,
                        None => continue,
                    };
                    if entry.bytes.len() != k * n * 4 {
                        continue; // shape/dtype guard
                    }
                    // f32 weight (elem = 4); shared layout = single source of truth.
                    let pbytes =
                        hologram_backend::layout::pack_b_panels_bytes(&entry.bytes, k, n, 4);
                    mm.b.length = pbytes.len() as u64;
                    mm.b_packed = true;
                    packed_consts[cid] = Some(pbytes);
                }
            }
        }
        // Body a constant emits: its packed layout if packed, else its bytes.
        let const_body = |i: usize| -> Vec<u8> {
            packed_consts[i].clone().unwrap_or_else(|| {
                self.graph
                    .constants()
                    .get(hologram_graph::ConstantId(i as u32))
                    .map(|e| e.bytes.clone())
                    .unwrap_or_default()
            })
        };

        // Step 9: emit archive.
        let mut writer = HoloWriter::new();
        writer.set_kernel_calls(kernel_calls);
        if let Some(s) = self.graph.schedule() {
            writer.set_schedule(s.clone());
        }
        if !exec_plan.is_empty() {
            writer.set_exec_plan(exec_plan);
        }
        // Weight dedup (spec X.3 + X-7 trillion-param scale).
        //
        // Every constant body is BLAKE3-keyed once and stored in the
        // archive's `Weights` section. The `Constants` section then
        // emits *references* — a slot/dtype paired with the
        // fingerprint — instead of inlining the body a second time.
        // Identical weight bodies share storage at the archive level
        // (one body, N references), and at session load each slot
        // resolves its body via a single WeightStore lookup.
        //
        // Inline bodies are reserved for genuinely small literals (a
        // few KB). The 4 KiB threshold below distinguishes "constant"
        // (inline) from "weight" (referenced).
        const INLINE_THRESHOLD_BYTES: usize = 4096;
        let mut weights = WeightStore::new();
        let mut const_fingerprints: Vec<Option<[u8; 32]>> =
            vec![None; self.graph.constants().len()];
        for (i, slot) in const_fingerprints.iter_mut().enumerate() {
            let body = const_body(i);
            if body.len() > INLINE_THRESHOLD_BYTES {
                let fp = weights.insert(body);
                *slot = Some(fp.0);
            }
        }
        writer.set_weights(weights);
        writer.set_shape_registry(self.graph.shape_registry().clone());
        if !certificate_records.is_empty() {
            writer.set_certificates(certificate_codec::encode(&certificate_records));
        }

        // Emit input/output port descriptors so the runtime can map caller
        // tensors into the workspace's slot numbering.
        //
        // For an Input node, slot = node_id (the executor writes input bytes
        // there before kernel dispatch).
        //
        // For an Output node, the data actually lives in the slot of the
        // node that produced its first input (Output nodes don't run a
        // kernel of their own). Aliasing the port to the producer's slot
        // means the runtime reads the actual computed bytes.
        let inputs: Vec<PortDescriptor> = self
            .graph
            .inputs()
            .iter()
            .copied()
            .enumerate()
            .map(|(port_i, id)| {
                let idx = id.0 as usize;
                let n = self.graph.nodes().get(idx);
                PortDescriptor {
                    name: self.graph.input_name(port_i).into(),
                    slot: idx as u32,
                    element_count: element_counts.get(idx).copied().unwrap_or(0),
                    dtype: n.map(|n| n.output_dtype.0).unwrap_or(0),
                    shape: n.map(|n| shape_dims(&self.graph, n)).unwrap_or_default(),
                }
            })
            .collect();

        let outputs: Vec<PortDescriptor> = self
            .graph
            .outputs()
            .iter()
            .copied()
            .enumerate()
            .map(|(port_i, id)| {
                use hologram_graph::{InputSource, NodeId};
                let idx = id.0 as usize;
                let n = self.graph.nodes().get(idx);
                // An Output node runs no kernel of its own; its port must alias
                // the slot where its first input's data actually lives. Resolve
                // by source kind — a non-`Node` source (a const-folded constant,
                // or a direct graph-input passthrough) lives in a *different*
                // slot than the Output node's own index. The previous code only
                // handled `Node` and fell back to the Output node's own (never
                // written) slot, so an Output sourced from a Constant aliased an
                // unwritten slot → `WorkspaceExhausted` at execute.
                let resolved = n.and_then(|n| n.inputs.first()).and_then(|src| match *src {
                    InputSource::Node(NodeId(p)) => {
                        let p = p as usize;
                        Some((
                            p as u32,
                            element_counts.get(p).copied().unwrap_or(0),
                            self.graph.nodes().get(p).map(|x| x.output_dtype.0),
                        ))
                    }
                    // Inline constant operand: its bytes are pre-filled into
                    // slot `node_count + cid` (see the constants emission below).
                    InputSource::Constant(cid) => {
                        let entry = self.graph.constants().get(cid)?;
                        let dt = entry.dtype.0;
                        let ec = (entry.bytes.len() / bytes_per_element(dt).max(1)) as u64;
                        Some((node_count + cid.0, ec, Some(dt)))
                    }
                    // Direct graph-input passthrough: alias the input node's slot
                    // (= its node index), which the runtime fills with the bound
                    // input bytes before dispatch.
                    InputSource::GraphInput(g) => {
                        let in_idx = self.graph.inputs().get(g as usize)?.0 as usize;
                        Some((
                            in_idx as u32,
                            element_counts.get(in_idx).copied().unwrap_or(0),
                            self.graph.nodes().get(in_idx).map(|x| x.output_dtype.0),
                        ))
                    }
                });
                let (slot, element_count, dtype) = resolved.unwrap_or((
                    idx as u32,
                    element_counts.get(idx).copied().unwrap_or(0),
                    n.map(|n| n.output_dtype.0),
                ));
                // Output shape: the Output node carries the result shape on its
                // own `output_shape`, so read it directly (it equals the
                // producer's shape).
                let shape = n.map(|n| shape_dims(&self.graph, n)).unwrap_or_default();
                PortDescriptor {
                    name: self.graph.output_name(port_i).into(),
                    slot,
                    element_count,
                    dtype: dtype.or_else(|| n.map(|n| n.output_dtype.0)).unwrap_or(0),
                    shape,
                }
            })
            .collect();
        writer.set_inputs(inputs);
        writer.set_outputs(outputs);

        // Open producer metadata (tokenizer, generation config, …): embed each
        // as an archive Extension section, carried opaquely to the runtime.
        for (key, bytes) in self.graph.extensions() {
            writer.add_extension(key.clone(), bytes.clone());
        }

        // Emit constants: each entry pre-fills a workspace slot with the
        // constant's bytes at session-load time. Small bodies are
        // inlined; larger bodies become references into the Weights
        // pool (see weight dedup above).
        let mut constants: Vec<ConstantEntry> = (0..self.graph.constants().len())
            .filter_map(|i| {
                let id = hologram_graph::ConstantId(i as u32);
                let entry = self.graph.constants().get(id)?;
                let slot = node_count + (i as u32);
                let dtype = entry.dtype.0;
                Some(if let Some(fp) = const_fingerprints[i] {
                    ConstantEntry::reference(slot, dtype, fp)
                } else {
                    // Inline the (possibly packed) body — the weight-layout
                    // monomorphism stores packed bytes for packed constants.
                    ConstantEntry::inline(slot, dtype, const_body(i))
                })
            })
            .collect();

        // A `GraphOp::Constant` node is referenced as `InputSource::Node(ni)`,
        // so its *own* slot (its node index) must also be pre-filled — not just
        // the `node_count + cid` slot that backs inline `InputSource::Constant`
        // operands. Bind each constant node's slot to its body. (Used by the
        // backward pass, which materializes identity-element / zero tensors as
        // constant nodes.)
        for (ni, node) in self.graph.nodes().iter().enumerate() {
            if let hologram_graph::GraphOp::Constant(cid) = node.op {
                let i = cid.0 as usize;
                let dtype = self
                    .graph
                    .constants()
                    .get(cid)
                    .map(|e| e.dtype.0)
                    .unwrap_or(0);
                let entry = if let Some(Some(fp)) = const_fingerprints.get(i).copied() {
                    ConstantEntry::reference(ni as u32, dtype, fp)
                } else {
                    ConstantEntry::inline(ni as u32, dtype, const_body(i))
                };
                constants.push(entry);
            }
        }
        if !constants.is_empty() {
            writer.set_constants(constants);
        }

        let archive = writer.finish().map_err(CompileError::Archive)?;

        Ok(CompilationOutput { archive, stats })
    }
}

/// FNV-1a 64-bit hash (`const`-evaluable). The foundation's `Binding`
/// declares `content_address` as an FNV-1a content address (and uses the
/// same incremental mix in `primitive_session_binding_signature`); this is
/// the canonical derivation, so each binding's address is the FNV-1a hash
/// of its surface identifier rather than a hand-picked constant.
const fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET_BASIS;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

/// Compile-time binding table for the per-node Term arena (spec O-3).
/// Each entry maps `name_index = i` to `value_index = i`, which is the
/// arena slot of the corresponding `Term::Variable` pushed first.
/// `surface` is a static identifier (`"v0"`, `"v1"`, `"v2"`) for tooling.
/// `content_address` is the binding's FNV-1a content address (over its
/// surface identifier), used by upstream's `BindingsTable` for
/// cross-binding deduplication.
const VAR_BINDINGS: &[Binding] = &[
    Binding {
        name_index: 0,
        type_index: 0,
        value_index: 0,
        surface: "v0",
        content_address: fnv1a_64(b"v0"),
    },
    Binding {
        name_index: 1,
        type_index: 0,
        value_index: 1,
        surface: "v1",
        content_address: fnv1a_64(b"v1"),
    },
    Binding {
        name_index: 2,
        type_index: 0,
        value_index: 2,
        surface: "v2",
        content_address: fnv1a_64(b"v2"),
    },
];

/// Emit a per-node Term arena (spec V.3 / VII.2 step 3).
///
/// Pushes one `Term::Variable` per argument contiguously, then dispatches
/// to the op marker's `emit_term` via `hologram_ops::emit_op_term` —
/// the Term tree IS the formal specification (spec invariant I-9).
fn build_node_arena(
    kind: hologram_graph::OpKind,
    level: WittLevel,
) -> Result<Box<HoloArena<128>>, CompileError> {
    // Heap-allocate the arena: `HoloArena<128>` is a 128-slot array of
    // `Option<Term>` and each `Term::Literal` carries an inline
    // `TermValue` byte buffer (`HOLOGRAM_INLINE_BYTES`). On-stack
    // instantiation in a deep compile loop overflows the default thread
    // stack; boxing keeps the arena on the heap while preserving the
    // value-type ownership story.
    let mut arena: Box<HoloArena<128>> = Box::new(HoloArena::new());
    let arity = kind.primary_arity();

    let args_start = arena
        .push(Term::Variable { name_index: 0 })
        .ok_or(CompileError::ArenaOverflow("variable 0"))?;
    for i in 1..arity {
        arena
            .push(Term::Variable {
                name_index: i as u32,
            })
            .ok_or(CompileError::ArenaOverflow("variable"))?;
    }

    hologram_ops::emit_op_term(kind, &mut arena, level, args_start)
        .ok_or(CompileError::ArenaOverflow("op emitter"))?;

    Ok(arena)
}

/// Compute a content fingerprint over (op_iri, witt_level, backend_kind).
fn compute_fingerprint(
    kind: hologram_graph::OpKind,
    level: WittLevel,
    target: BackendKind,
) -> [u8; 32] {
    let h = HologramHasher::initial();
    let h = h.fold_bytes(kind.name().as_bytes());
    let h = h.fold_bytes(&level.witt_length().to_le_bytes());
    let h = h.fold_bytes(target.name().as_bytes());
    h.finalize()
}

fn collect_buffers(
    graph: &Graph,
    node: &hologram_graph::Node,
    byte_lengths: &[u64],
) -> Vec<BufferRef> {
    use hologram_graph::InputSource;
    node.inputs
        .iter()
        .map(|src| match *src {
            InputSource::Node(hologram_graph::NodeId(id)) => BufferRef {
                slot: id,
                offset: 0,
                length: byte_lengths.get(id as usize).copied().unwrap_or(0),
            },
            InputSource::Constant(hologram_graph::ConstantId(id)) => BufferRef {
                slot: graph.node_count() as u32 + id,
                offset: 0,
                length: graph
                    .constants()
                    .get(hologram_graph::ConstantId(id))
                    .map(|e| e.bytes.len() as u64)
                    .unwrap_or(0),
            },
            InputSource::GraphInput(idx) => {
                let id = graph
                    .inputs()
                    .get(idx as usize)
                    .map(|hologram_graph::NodeId(i)| *i)
                    .unwrap_or(0);
                BufferRef {
                    slot: id,
                    offset: 0,
                    length: byte_lengths.get(id as usize).copied().unwrap_or(0),
                }
            }
        })
        .collect()
}

/// Bytes per element for a dtype-id (mirrors `hologram_backend::cpu::dtype`
/// constants). Centralised here so the compiler doesn't depend on the CPU
/// backend module path.
const fn bytes_per_element(dtype: u8) -> usize {
    match dtype {
        0..=2 => 1,     // BOOL, U8, I8
        6 | 7 => 2,     // F16, BF16
        4 | 8 => 4,     // I32, F32
        3 | 5 | 9 => 8, // U64, I64, F64
        _ => 1,
    }
}

/// Compute the Slice = `ProjectField` sub-region as `(byte_offset, byte_len)`
/// for the axis-0, unit-step case: `Slice(data, starts, ends)` with the index
/// bounds as i64 (ONNX) constants. The byte field is
/// `[start·inner·elem, end·inner·elem)` where `inner` is the product of the
/// non-leading dims. Returns `None` for any shape the contiguous view can't
/// represent (wrong arity, non-constant or out-of-range bounds, missing
/// shape) — the caller rejects rather than emit a silent-wrong slice.
fn slice_view_bytes(graph: &Graph, node: &hologram_graph::Node) -> Option<(u64, u64)> {
    use hologram_graph::{InputSource, NodeId};
    // data, starts, ends — the axis-0 contiguous form.
    if node.inputs.len() != 3 {
        return None;
    }
    let reg = graph.shape_registry();
    let data_shape = match node.inputs[0] {
        InputSource::Node(NodeId(id)) => graph
            .nodes()
            .get(id as usize)
            .and_then(|n| reg.get(n.output_shape).cloned()),
        InputSource::Constant(cid) => graph
            .constants()
            .get(cid)
            .and_then(|e| reg.get(e.shape).cloned()),
        InputSource::GraphInput(idx) => graph
            .inputs()
            .get(idx as usize)
            .and_then(|&NodeId(i)| graph.nodes().get(i as usize))
            .and_then(|n| reg.get(n.output_shape).cloned()),
    }?;
    let d0 = data_shape.dim(0)? as i64;
    let inner: u64 = (1..data_shape.rank as usize)
        .map(|i| data_shape.dim(i).unwrap_or(1))
        .product();
    let read_i64 = |src: InputSource| -> Option<i64> {
        match src {
            InputSource::Constant(cid) => {
                let e = graph.constants().get(cid)?;
                e.bytes
                    .get(0..8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            }
            _ => None,
        }
    };
    let start = read_i64(node.inputs[1])?.clamp(0, d0);
    let end = read_i64(node.inputs[2])?.clamp(0, d0);
    if end < start {
        return None;
    }
    let elem = bytes_per_element(node.output_dtype.0) as u64;
    let offset = start as u64 * inner * elem;
    let len = (end - start) as u64 * inner * elem;
    Some((offset, len))
}

/// Compute the axis-0 Pad placement as `(lo_byte_offset, data_byte_len,
/// data_count)`: data is copied into the zeroed output at byte offset
/// `lo·inner·elem`. `Pad(data, pads, ...)` with `pads` an i64 [2·rank] ONNX
/// tensor (`[begin_0..begin_{r-1}, end_0..end_{r-1}]`). Returns `None` unless
/// only axis-0 is padded (every inner begin/end is 0) — anything else is a
/// non-contiguous pad that this offset-placement form cannot represent.
fn pad_view_bytes(graph: &Graph, node: &hologram_graph::Node) -> Option<(u64, u64, u64)> {
    use hologram_graph::{InputSource, NodeId};
    if node.inputs.len() < 2 {
        return None;
    }
    let reg = graph.shape_registry();
    let data_shape = match node.inputs[0] {
        InputSource::Node(NodeId(id)) => graph
            .nodes()
            .get(id as usize)
            .and_then(|n| reg.get(n.output_shape).cloned()),
        InputSource::Constant(cid) => graph
            .constants()
            .get(cid)
            .and_then(|e| reg.get(e.shape).cloned()),
        InputSource::GraphInput(idx) => graph
            .inputs()
            .get(idx as usize)
            .and_then(|&NodeId(i)| graph.nodes().get(i as usize))
            .and_then(|n| reg.get(n.output_shape).cloned()),
    }?;
    let rank = data_shape.rank as usize;
    let inner: u64 = (1..rank).map(|i| data_shape.dim(i).unwrap_or(1)).product();
    let data_count: u64 = (0..rank).map(|i| data_shape.dim(i).unwrap_or(1)).product();
    let pads = match node.inputs[1] {
        InputSource::Constant(cid) => &graph.constants().get(cid)?.bytes,
        _ => return None,
    };
    // i64 [2·rank]; require every inner (axis ≥ 1) begin and end to be zero.
    if rank == 0 || pads.len() < 8 {
        return None;
    }
    let pad_at = |i: usize| -> Option<i64> {
        pads.get(i * 8..i * 8 + 8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
    };
    for axis in 1..rank {
        if pad_at(axis).unwrap_or(0) != 0 || pad_at(rank + axis).unwrap_or(0) != 0 {
            return None; // inner-axis pad — not a contiguous placement
        }
    }
    let lo = pad_at(0).unwrap_or(0).max(0) as u64;
    let elem = bytes_per_element(node.output_dtype.0) as u64;
    Some((lo * inner * elem, data_count * elem, data_count))
}

/// LRN dims `(batch, channels, inner)` from a rank-4 `[N, C, H, W]` input
/// (inner = H·W); also accepts rank-2 `[N, C]` (inner = 1). `None` otherwise.
fn lrn_dims(graph: &Graph, node: &hologram_graph::Node) -> Option<(u32, u32, u32)> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let s = match node.inputs.first().copied()? {
        InputSource::Node(NodeId(id)) => graph
            .nodes()
            .get(id as usize)
            .and_then(|n| reg.get(n.output_shape).cloned()),
        InputSource::Constant(cid) => graph
            .constants()
            .get(cid)
            .and_then(|e| reg.get(e.shape).cloned()),
        InputSource::GraphInput(idx) => graph
            .inputs()
            .get(idx as usize)
            .and_then(|&NodeId(i)| graph.nodes().get(i as usize))
            .and_then(|n| reg.get(n.output_shape).cloned()),
    }?;
    let rank = s.rank as usize;
    if rank < 2 {
        return None;
    }
    let batch = s.dim(0)? as u32;
    let channels = s.dim(1)? as u32;
    let inner: u64 = (2..rank).map(|i| s.dim(i).unwrap_or(1)).product();
    Some((batch, channels, inner as u32))
}

/// `(rank, in_dims, out_dims)` from the input and output shapes (same rank, no
/// broadcast constraint) — used by Resize's nearest-neighbor gather.
fn reindex_dims(
    graph: &Graph,
    node: &hologram_graph::Node,
) -> Option<(u8, [u32; MAX_RANK], [u32; MAX_RANK])> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let in_shape = match node.inputs.first().copied()? {
        InputSource::Node(NodeId(id)) => graph
            .nodes()
            .get(id as usize)
            .and_then(|n| reg.get(n.output_shape).cloned()),
        InputSource::Constant(cid) => graph
            .constants()
            .get(cid)
            .and_then(|e| reg.get(e.shape).cloned()),
        InputSource::GraphInput(idx) => graph
            .inputs()
            .get(idx as usize)
            .and_then(|&NodeId(i)| graph.nodes().get(i as usize))
            .and_then(|n| reg.get(n.output_shape).cloned()),
    }?;
    let out_shape = reg.get(node.output_shape).cloned()?;
    let rank = out_shape.rank as usize;
    if rank == 0 || rank > MAX_RANK || in_shape.rank as usize != rank {
        return None;
    }
    let mut in_dims = [0u32; MAX_RANK];
    let mut out_dims = [0u32; MAX_RANK];
    for i in 0..rank {
        in_dims[i] = in_shape.dim(i)? as u32;
        out_dims[i] = out_shape.dim(i)? as u32;
    }
    Some((rank as u8, in_dims, out_dims))
}

/// RoPE head dimension = the input tensor's last dim (the rotated axis).
fn rope_head_dim(graph: &Graph, node: &hologram_graph::Node) -> Option<u32> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let shape = match node.inputs.first().copied()? {
        InputSource::Node(NodeId(id)) => graph
            .nodes()
            .get(id as usize)
            .and_then(|n| reg.get(n.output_shape).cloned()),
        InputSource::Constant(cid) => graph
            .constants()
            .get(cid)
            .and_then(|e| reg.get(e.shape).cloned()),
        InputSource::GraphInput(idx) => graph
            .inputs()
            .get(idx as usize)
            .and_then(|&NodeId(i)| graph.nodes().get(i as usize))
            .and_then(|n| reg.get(n.output_shape).cloned()),
    }?;
    let rank = shape.rank as usize;
    if rank == 0 {
        return None;
    }
    Some(shape.dim(rank - 1)? as u32)
}

/// Resolve an Expand's `(rank, in_dims, out_dims)` from the input shape and the
/// node's broadcast output shape (same rank; each input dim equals the output
/// dim or is 1). `None` for rank 0/>8 or an incompatible (non-broadcast) shape.
/// Total element count of a node's first input (the number of elements a
/// full reduction folds over).
fn input_element_count(graph: &Graph, node: &hologram_graph::Node) -> Option<u64> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let shape = match node.inputs.first().copied()? {
        InputSource::Node(NodeId(id)) => graph.nodes().get(id as usize)?.output_shape,
        InputSource::Constant(cid) => graph.constants().get(cid)?.shape,
        InputSource::GraphInput(idx) => {
            let &NodeId(i) = graph.inputs().get(idx as usize)?;
            graph.nodes().get(i as usize)?.output_shape
        }
    };
    reg.get(shape).map(|d| d.total_elements())
}

/// `(channels, inner)` for per-channel dequantization along `axis` of the
/// dequantize node's input shape: `channels = dim[axis]`, `inner = ∏ dims
/// after axis` (so element `i`'s channel is `(i / inner) % channels`).
fn quant_channel_dims(
    graph: &Graph,
    node: &hologram_graph::Node,
    axis: usize,
) -> Option<(u32, u32)> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let shape = match node.inputs.first().copied()? {
        InputSource::Node(NodeId(id)) => graph.nodes().get(id as usize)?.output_shape,
        InputSource::Constant(cid) => graph.constants().get(cid)?.shape,
        InputSource::GraphInput(idx) => {
            let &NodeId(i) = graph.inputs().get(idx as usize)?;
            graph.nodes().get(i as usize)?.output_shape
        }
    };
    let d = reg.get(shape)?;
    let rank = d.rank as usize;
    if axis >= rank {
        return None;
    }
    let channels = d.dim(axis)? as u32;
    let inner: u64 = ((axis + 1)..rank).map(|i| d.dim(i).unwrap_or(1)).product();
    Some((channels, inner.min(u32::MAX as u64) as u32))
}

/// `(rank, dims[..rank])` of a reduce node's input shape (row-major, ≤ rank 8),
/// for filling `ReduceCall`'s axis-reduction geometry.
fn reduce_input_dims(graph: &Graph, node: &hologram_graph::Node) -> Option<(u8, [u32; MAX_RANK])> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let shape = match node.inputs.first().copied()? {
        InputSource::Node(NodeId(id)) => graph.nodes().get(id as usize)?.output_shape,
        InputSource::Constant(cid) => graph.constants().get(cid)?.shape,
        InputSource::GraphInput(idx) => {
            let &NodeId(i) = graph.inputs().get(idx as usize)?;
            graph.nodes().get(i as usize)?.output_shape
        }
    };
    let d = reg.get(shape)?;
    if d.rank as usize > MAX_RANK {
        return None;
    }
    let mut dims = [0u32; MAX_RANK];
    for (i, slot) in dims.iter_mut().enumerate().take(d.rank as usize) {
        *slot = d.dim(i).unwrap_or(0).min(u32::MAX as u64) as u32;
    }
    Some((d.rank, dims))
}

fn expand_plan(
    graph: &Graph,
    node: &hologram_graph::Node,
) -> Option<(u8, [u32; MAX_RANK], [u32; MAX_RANK])> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let in_shape = match node.inputs.first().copied()? {
        InputSource::Node(NodeId(id)) => graph
            .nodes()
            .get(id as usize)
            .and_then(|n| reg.get(n.output_shape).cloned()),
        InputSource::Constant(cid) => graph
            .constants()
            .get(cid)
            .and_then(|e| reg.get(e.shape).cloned()),
        InputSource::GraphInput(idx) => graph
            .inputs()
            .get(idx as usize)
            .and_then(|&NodeId(i)| graph.nodes().get(i as usize))
            .and_then(|n| reg.get(n.output_shape).cloned()),
    }?;
    let out_shape = reg.get(node.output_shape).cloned()?;
    let rank = out_shape.rank as usize;
    if rank == 0 || rank > MAX_RANK || in_shape.rank as usize != rank {
        return None;
    }
    let mut in_dims = [0u32; MAX_RANK];
    let mut out_dims = [0u32; MAX_RANK];
    for i in 0..rank {
        let id = in_shape.dim(i)? as u32;
        let od = out_shape.dim(i)? as u32;
        if id != od && id != 1 {
            return None; // not a valid broadcast
        }
        in_dims[i] = id;
        out_dims[i] = od;
    }
    Some((rank as u8, in_dims, out_dims))
}

/// Resolve a Transpose's `(rank, input_dims, perm)` from the data shape and the
/// optional perm operand (an i64 [rank] constant); absent perm defaults to the
/// full axis reversal (ONNX). `None` for rank 0 / >8 or an out-of-range perm.
fn transpose_plan(
    graph: &Graph,
    node: &hologram_graph::Node,
) -> Option<(u8, [u32; MAX_RANK], [u8; MAX_RANK])> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let data_shape = match node.inputs.first().copied()? {
        InputSource::Node(NodeId(id)) => graph
            .nodes()
            .get(id as usize)
            .and_then(|n| reg.get(n.output_shape).cloned()),
        InputSource::Constant(cid) => graph
            .constants()
            .get(cid)
            .and_then(|e| reg.get(e.shape).cloned()),
        InputSource::GraphInput(idx) => graph
            .inputs()
            .get(idx as usize)
            .and_then(|&NodeId(i)| graph.nodes().get(i as usize))
            .and_then(|n| reg.get(n.output_shape).cloned()),
    }?;
    let rank = data_shape.rank as usize;
    if rank == 0 || rank > MAX_RANK {
        return None;
    }
    let mut dims = [0u32; MAX_RANK];
    for (i, d) in dims.iter_mut().enumerate().take(rank) {
        *d = data_shape.dim(i)? as u32;
    }
    let mut perm = [0u8; MAX_RANK];
    match node.inputs.get(1).copied() {
        Some(InputSource::Constant(cid)) => {
            let bytes = &graph.constants().get(cid)?.bytes;
            for (i, p) in perm.iter_mut().enumerate().take(rank) {
                let v = i64::from_le_bytes(bytes.get(i * 8..i * 8 + 8)?.try_into().ok()?);
                if v < 0 || v as usize >= rank {
                    return None;
                }
                *p = v as u8;
            }
        }
        Some(_) => return None, // non-constant perm
        None => {
            // Default: reverse all axes.
            for (i, p) in perm.iter_mut().enumerate().take(rank) {
                *p = (rank - 1 - i) as u8;
            }
        }
    }
    Some((rank as u8, dims, perm))
}

/// Resolve a Gather's indexed-copy geometry from the data + indices shapes and
/// the node's `GatherAttrs` axis. Returns `(outer, axis_dim, inner,
/// num_indices, idx_dtype)`: the data tensor is flattened to
/// `[outer, axis_dim, inner]` (the product of the dims before `axis`, the
/// gathered axis, and the product of the dims after, in elements), and
/// `num_indices` is the product of the indices shape. `idx_dtype` is the
/// indices operand's dtype (`i32`/`i64`). `None` for a missing shape, an
/// out-of-range axis, or an overflowing product (so a malformed Gather fails
/// loud rather than gathering garbage).
/// The row-major shape (dims) of a node's `output_shape`, for the port
/// descriptors. Empty if the shape isn't registered (the flat element count
/// stays authoritative).
fn shape_dims(graph: &Graph, node: &hologram_graph::Node) -> alloc::vec::Vec<u64> {
    match graph.shape_registry().get(node.output_shape) {
        Some(d) => (0..d.rank as usize).filter_map(|i| d.dim(i)).collect(),
        None => alloc::vec::Vec::new(),
    }
}

/// The dtype of a node's `idx`-th input operand (resolving through node /
/// constant / graph-input sources). Used to fill a `Cast`'s source dtype and a
/// `Gather`'s index dtype from the operands the op consumes.
fn operand_dtype(graph: &Graph, node: &hologram_graph::Node, idx: usize) -> Option<u8> {
    use hologram_graph::{InputSource, NodeId};
    match node.inputs.get(idx).copied()? {
        InputSource::Node(NodeId(id)) => graph.nodes().get(id as usize).map(|n| n.output_dtype.0),
        InputSource::Constant(cid) => graph.constants().get(cid).map(|e| e.dtype.0),
        InputSource::GraphInput(g) => {
            let &NodeId(j) = graph.inputs().get(g as usize)?;
            graph.nodes().get(j as usize).map(|n| n.output_dtype.0)
        }
    }
}

fn gather_plan(
    graph: &Graph,
    node: &hologram_graph::Node,
    node_id: hologram_graph::NodeId,
) -> Option<(u64, u64, u64, u64, u8)> {
    use hologram_graph::{InputSource, NodeId};
    let reg = graph.shape_registry();
    let operand = |i: usize| -> Option<(hologram_graph::ShapeDescriptor, u8)> {
        match node.inputs.get(i).copied()? {
            InputSource::Node(NodeId(id)) => {
                let n = graph.nodes().get(id as usize)?;
                Some((reg.get(n.output_shape).cloned()?, n.output_dtype.0))
            }
            InputSource::Constant(cid) => {
                let e = graph.constants().get(cid)?;
                Some((reg.get(e.shape).cloned()?, e.dtype.0))
            }
            InputSource::GraphInput(g) => {
                let &NodeId(j) = graph.inputs().get(g as usize)?;
                let n = graph.nodes().get(j as usize)?;
                Some((reg.get(n.output_shape).cloned()?, n.output_dtype.0))
            }
        }
    };
    let (data, _) = operand(0)?;
    let (indices, idx_dtype) = operand(1)?;
    let rank = data.rank as usize;
    if rank == 0 {
        return None;
    }
    // Normalize the axis (ONNX permits a negative axis counting from the end).
    let axis_raw = graph.gather_attrs(node_id).map(|a| a.axis).unwrap_or(0);
    let axis = if axis_raw < 0 {
        axis_raw + rank as i32
    } else {
        axis_raw
    };
    if axis < 0 || axis as usize >= rank {
        return None;
    }
    let axis = axis as usize;
    let mut outer: u64 = 1;
    for i in 0..axis {
        outer = outer.checked_mul(data.dim(i)?)?;
    }
    let axis_dim = data.dim(axis)?;
    let mut inner: u64 = 1;
    for i in (axis + 1)..rank {
        inner = inner.checked_mul(data.dim(i)?)?;
    }
    let mut num_indices: u64 = 1;
    for i in 0..indices.rank as usize {
        num_indices = num_indices.checked_mul(indices.dim(i)?)?;
    }
    Some((outer, axis_dim, inner, num_indices, idx_dtype))
}
