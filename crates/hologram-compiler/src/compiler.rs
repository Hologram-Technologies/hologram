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
use hologram_ops::{HoloArena, HoloTerm};
use hologram_types::HologramHasher;
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
        // A weight slot's `weight_layout` describes the layout of the bytes as
        // they are *bound*. A constant's bytes are already in the graph, in
        // `[k,n]`; nothing will transpose them behind the graph's back. Reject
        // the false declaration here rather than let the loader read `[k,n]`
        // bytes as `[n,k]`.
        validate_weight_layout_declarations(&self.graph)?;

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
                // Honours sub-byte packing (i4 → ceil(n/2), e8cb → ceil(n/8)).
                // An unrecognized tag sizes to 0 so a downstream bounds check
                // fails loudly rather than under-allocating a live buffer.
                n.output_dtype
                    .storage_bytes_u64(element_counts[i])
                    .unwrap_or(0)
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

        // KvCacheWrite trailing placement: the executor may realize an
        // output-only cache write as an in-place move on the resident cache
        // (κ move semantics — the old cache label is consumed). That is sound
        // only if every other reader of the cache has already run, so a
        // KvCacheWrite whose result feeds nothing but graph outputs is
        // deferred to a trailing schedule level after all other work. A write
        // whose result IS consumed in-graph keeps its dependency-ordered
        // place (the runtime then takes the honest-copy path — the load-time
        // steal analysis in hologram-exec re-derives eligibility from the
        // plan itself, so a hand-built archive cannot spoof it).
        let traversal_levels: Vec<Vec<u32>> = {
            let mut levels = traversal_levels;
            let nodes = self.graph.nodes();
            let mut deferred: Vec<u32> = Vec::new();
            let consumed_by_compute = |id: u32| {
                nodes.iter().any(|n| {
                    !matches!(n.op, GraphOp::Output)
                        && n.inputs.iter().any(|src| {
                            matches!(*src, hologram_graph::InputSource::Node(hologram_graph::NodeId(i)) if i == id)
                        })
                })
            };
            for level in &mut levels {
                level.retain(|&id| {
                    let is_kv_write = nodes.get(id as usize).is_some_and(|n| {
                        matches!(n.op, GraphOp::Op(hologram_graph::OpKind::KvCacheWrite))
                    });
                    if is_kv_write && !consumed_by_compute(id) {
                        deferred.push(id);
                        false
                    } else {
                        true
                    }
                });
            }
            if !deferred.is_empty() {
                levels.push(deferred);
            }
            levels
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
                    )?,
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
                            // Weight-slot declaration: carried through so the
                            // *load-time* fusion can build the fused decode call
                            // for a weight whose bytes arrive after compile.
                            weight_layout: quant_attrs.weight_layout,
                            act_quant: quant_attrs.act_quant,
                            codebook: codebook_ref(&self.graph, node),
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

        const DTYPE_F32: u8 = 8;
        let n_const = self.graph.constants().len();
        // Constant bodies rewritten by the compile-time layout passes below:
        // the body a constant emits is its derived layout if one exists, else
        // its original bytes.
        let mut packed_consts: Vec<Option<Vec<u8>>> = vec![None; n_const];

        // ── Decode-shape int8 fusion (compile-time, weight-layout
        //    monomorphism for quantized weights) ──
        //
        // A constant symmetric per-channel i8 weight uniquely consumed by a
        // `Dequantize → MatMul(B)` chain at decode shapes (small static m)
        // fuses to one `MatMulDequant` **in the archive**, with the constant
        // transposed to output-major `[n,k]` and per-token dynamic activation
        // quantization (W8A8). The transposed bytes are derived content under
        // the constant's own κ — exactly like the f32 panel packing below —
        // and the fused call is the only reader of that layout. Dynamic
        // quantized weights keep the load-time fusion path (W8A32, `[k,n]`).
        let (fused_calls, fused_plan) = fuse_const_i8_decode(
            &self.graph,
            kernel_calls,
            exec_plan,
            node_count,
            &mut packed_consts,
        );
        let mut kernel_calls = fused_calls;
        let exec_plan = fused_plan;

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
            // A **weightless** constant carries a κ, not bytes: the body arrives
            // at materialization through a `WeightProvider`. Emit it by
            // reference under the declared κ and put *nothing* in the Weights
            // section — that is the whole point, the archive holds no weight
            // bytes and dedupes across models. `load_paged` resolves it; a
            // fully-resident `load` fails loud rather than pinning an empty body.
            if let Some(kappa) = self
                .graph
                .constants()
                .external(hologram_graph::ConstantId(i as u32))
            {
                *slot = Some(kappa);
                continue;
            }
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
                        // Element count comes from the declared shape, not from
                        // `bytes.len() / width`: the sub-byte tiers (i4, e8cb)
                        // pack several elements per byte, so byte length alone
                        // under-counts them (i4 by 2×, e8cb by 8×).
                        let ec = self
                            .graph
                            .shape_registry()
                            .get(entry.shape)?
                            .total_elements();
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

/// Resolve the codebook operand a vector-quantized `Dequantize` decodes against:
/// the node's **4th input**, which must be a constant. Returned as a `BufferRef`
/// into the constant slot space (`node_count + cid`), or `NO_CODEBOOK` when the
/// node declares none.
///
/// Length is *not* validated here — the fusions and the backend each enforce the
/// full `E8CB_MAX_ENTRIES × group_dim` index space, because the backend must not
/// trust an archive it did not produce.
fn codebook_ref(graph: &Graph, node: &hologram_graph::Node) -> BufferRef {
    use hologram_graph::InputSource;
    let node_count = graph.node_count() as u32;
    match node.inputs.get(3) {
        Some(InputSource::Constant(cid)) => {
            let len = graph
                .constants()
                .get(*cid)
                .map(|e| e.bytes.len() as u64)
                .unwrap_or(0);
            BufferRef {
                slot: node_count + cid.0,
                offset: 0,
                length: len,
            }
        }
        _ => hologram_backend::DequantizeCall::NO_CODEBOOK,
    }
}

/// Bytes per element for a fixed-width dtype. Delegates to the canonical
/// [`hologram_graph::registry::DTypeId`] (re-exported from `hologram-types`);
/// `None` for the sub-byte tiers (`i4`, `e8cb`) — whose storage is not
/// `n × width` — and for any unrecognized tag. Callers that size a buffer use
/// `DTypeId::storage_bytes_u64`; callers that require a whole-byte element
/// (slice/pad placement) propagate the `None` as "not representable".
const fn bytes_per_element(dtype: u8) -> Option<usize> {
    hologram_graph::registry::DTypeId(dtype).bytes_per_element()
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
    let elem = bytes_per_element(node.output_dtype.0)? as u64;
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
    let elem = bytes_per_element(node.output_dtype.0)? as u64;
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

/// A weight slot's `weight_layout` is a statement about the bytes as they will
/// be **bound**, not a request. `OUTPUT_MAJOR` means "when this weight arrives
/// it will already be `[n,k]`", and the only kernel that reads `[n,k]` is the
/// fused output-major decode GEMV. Every other path — the W8A32 dequant loop,
/// the standalone Dequantize kernel — reads `[k,n]`.
///
/// So a declaration that no output-major kernel can serve has **no correct
/// execution at all**: taking any other path would transpose the weight by
/// accident and return a plausible, wrong answer. Every precondition is static,
/// so this is a compile error rather than a load-time surprise:
///
/// - the weight is **not a graph constant** (a constant's bytes are already
///   here, in `[k,n]`; the claim would simply be false — to put a constant on
///   the fused path set `act_quant` alone and let the compiler transpose it);
/// - `act_quant = W8A8_TOKEN_SYM` (there is no output-major W8A32 kernel);
/// - the quant tier is registered and output-major-fusable, with `k` a whole
///   number of its groups and a byte-aligned column span;
/// - `k` is inside the exact-i32 accumulation bound;
/// - scales are per-output-column (`axis == 1` over a rank-2 `[k,n]` weight);
/// - a vector-quantized tier brings its codebook, and a scalar tier does not.
///
/// `hologram-exec`'s loader re-checks the same predicate and refuses rather than
/// falling back — defence in depth for archives this compiler did not produce.
fn validate_weight_layout_declarations(graph: &Graph) -> Result<(), CompileError> {
    use hologram_backend::{mm_act_quant, quant_tier::quant_tier};
    use hologram_graph::{DTypeId, GraphOp, InputSource, NodeId, OpKind};

    for (idx, node) in graph.nodes().iter().enumerate() {
        if !matches!(node.op, GraphOp::Op(OpKind::Dequantize)) {
            continue;
        }
        let Some(attrs) = graph.quant_attrs(NodeId(idx as u32)) else {
            continue;
        };
        if attrs.weight_layout == hologram_types::weight_layout::ROW_MAJOR {
            continue;
        }
        // A constant *with bytes* carries them in `[k,n]`, here, now — declaring
        // `OUTPUT_MAJOR` for it is a false statement about the graph's own bytes,
        // and the compiler transposes such a weight itself (`fuse_const_i8_decode`)
        // once `act_quant` opts in.
        //
        // A constant with **no** bytes is a different animal: it is a κ naming
        // content that arrives at materialization — a weightless compile. Its bytes
        // are not `[k,n]`, because there are none. It is a load-time-bound weight
        // that happens to be addressed through the constant table, and it is exactly
        // the case `QuantAttrs::weight_layout` exists to serve. So ask whether the
        // constant *has bytes*, not whether it is a constant — the same question
        // `fuse_const_i8_decode` asks before it transposes anything.
        if let Some(InputSource::Constant(cid)) = node.inputs.first() {
            if graph
                .constants()
                .get(*cid)
                .is_some_and(|e| !e.bytes.is_empty())
            {
                return Err(CompileError::GraphValidation(
                    "Dequantize over a graph constant with bytes declared \
                     weight_layout = OUTPUT_MAJOR; those bytes are [k,n] and the \
                     declaration describes bound bytes. Set act_quant = W8A8_TOKEN_SYM \
                     to opt the constant into the fused output-major decode path — the \
                     compiler transposes it for you. (A zero-byte constant is a \
                     weightless κ binding and may declare OUTPUT_MAJOR.)",
                ));
            }
        }
        if attrs.act_quant != hologram_types::act_quant::W8A8_TOKEN_SYM {
            return Err(CompileError::GraphValidation(
                "weight_layout = OUTPUT_MAJOR requires act_quant = W8A8_TOKEN_SYM;                  there is no output-major W8A32 kernel, and no other kernel can read                  the [n,k] bytes this weight promises to bind.",
            ));
        }
        let Some(tier) = quant_tier(DTypeId(attrs.quant_dtype)) else {
            return Err(CompileError::GraphValidation(
                "weight_layout = OUTPUT_MAJOR on an unregistered quant tier;                  no output-major kernel can decode it.",
            ));
        };
        if !tier.omajor_fusable {
            return Err(CompileError::GraphValidation(
                "weight_layout = OUTPUT_MAJOR on a quant tier with no output-major GEMV.",
            ));
        }
        // The weight is `[k, n]` logically, whatever its bound byte order.
        let Some(shape) = graph.shape_registry().get(node.output_shape) else {
            continue;
        };
        if shape.rank != 2 {
            return Err(CompileError::GraphValidation(
                "weight_layout = OUTPUT_MAJOR on a non-rank-2 weight.",
            ));
        }
        let (Some(k), Some(_n)) = (shape.dim(0), shape.dim(1)) else {
            continue;
        };
        let k = k as usize;
        if !tier.omajor_k_ok(k) {
            return Err(CompileError::GraphValidation(
                "weight_layout = OUTPUT_MAJOR but k is not a whole number of the tier's                  groups with a byte-aligned column span; the output-major GEMV cannot                  address it.",
            ));
        }
        if k > mm_act_quant::K_MAX {
            return Err(CompileError::GraphValidation(
                "weight_layout = OUTPUT_MAJOR but k exceeds the exact-i32 accumulation                  bound; the output-major GEMV would overflow, so it declines and no                  other kernel can read [n,k].",
            ));
        }
        if attrs.axis != 1 {
            return Err(CompileError::GraphValidation(
                "weight_layout = OUTPUT_MAJOR requires per-output-column scales                  (axis = 1); per-tensor and group-wise scales have no output-major GEMV.",
            ));
        }
        // The output-major integer GEMV decodes a **symmetric** weight: it
        // computes `Σ q·w` and has no term for a per-column zero point. When the
        // zero points are a graph constant their values are static, so reject
        // here rather than at execute time. (A zero-point bound at load is not
        // knowable now; `matmul_dequant` re-checks and fails loud.)
        if let Some(InputSource::Constant(zp)) = node.inputs.get(2) {
            if let Some(entry) = graph.constants().get(*zp) {
                if entry.bytes.iter().any(|&b| b != 0) {
                    return Err(CompileError::GraphValidation(
                        "weight_layout = OUTPUT_MAJOR with a non-zero zero-point; the \
                         output-major integer GEMV decodes symmetric weights only. \
                         Asymmetric weights are correct on the generic dequant path — \
                         leave act_quant = W8A32 and weight_layout = ROW_MAJOR.",
                    ));
                }
            }
        }
        // A VQ tier decodes indices through the model's own codebook: the node's
        // 4th input. A scalar tier must not carry one.
        let has_codebook = node.inputs.len() >= 4;
        if tier.needs_codebook != has_codebook {
            return Err(CompileError::GraphValidation(
                "weight_layout = OUTPUT_MAJOR: a vector-quantized tier must supply its                  codebook as the Dequantize node's 4th input, and a scalar tier must not.",
            ));
        }
    }
    Ok(())
}

/// Compile-time `Dequantize(const) → MatMul(B)` fusion at decode shapes.
///
/// Pattern (all conditions required, checked structurally):
/// - the Dequantize reads a **constant** i8 or packed-i4 weight (i4: even
///   `k`, whole packed bytes — the LUT tier, half the streamed bytes) that
///   no other call reads;
/// - scale/zero-point are per-channel over the matmul's output columns
///   (`channels == n`, `inner == 1`) and the zero-point constant is all-zero
///   (symmetric);
/// - the dequant output is a private single-consumer intermediate (not a
///   port) feeding a plain (`!b_packed`) f32 MatMul's B operand;
/// - `m` is decode-small (≤ 4, static per archive) and `k` is inside the
///   exact-i32 accumulation bound.
///
/// The match emits one fused `MatMulDequant { bq_omajor, act_quant: W8A8 }`
/// in the MatMul's place, drops the Dequantize, renumbers the exec plan, and
/// rewrites the constant body to the output-major `[n,k]` transpose — derived
/// content under the constant's own κ, the quantized analog of the f32 panel
/// packing. The fused call is the transposed layout's only reader, so the
/// `[k,n]` interpretation can never leak. Everything else falls through to
/// the load-time fusion (W8A32 over `[k,n]`), which no-ops on already-fused
/// archives.
fn fuse_const_i8_decode(
    graph: &Graph,
    calls: Vec<KernelCall>,
    plan: Vec<Vec<u32>>,
    node_count: u32,
    packed_consts: &mut [Option<Vec<u8>>],
) -> (Vec<KernelCall>, Vec<Vec<u32>>) {
    use hologram_backend::quant_tier::quant_tier;
    use hologram_backend::{buffers, mm_act_quant, MatMulDequantCall};
    use hologram_graph::{ConstantId, DTypeId, InputSource, NodeId};
    // The one dtype vocabulary — no locally re-declared tags.
    const DTYPE_F32: u8 = DTypeId::F32.raw();
    // The decode-shape bound lives with the kernels it gates
    // (`decode_gate::OMAJOR_W8A8_MAX_M`), not as a literal here.
    use hologram_backend::kernel_call::decode_gate::OMAJOR_W8A8_MAX_M as M_GATE;

    // Census: total references per slot, producer/reader counts, reader index.
    let n_calls = calls.len();
    let mut uses: hashbrown::HashMap<u32, u32> = hashbrown::HashMap::new();
    let mut prod_count: hashbrown::HashMap<u32, u32> = hashbrown::HashMap::new();
    let mut read_count: hashbrown::HashMap<u32, u32> = hashbrown::HashMap::new();
    let mut read_idx: hashbrown::HashMap<u32, usize> = hashbrown::HashMap::new();
    for (ci, call) in calls.iter().enumerate() {
        let bufs = buffers(call);
        for b in &bufs {
            if b.slot != u32::MAX {
                *uses.entry(b.slot).or_insert(0) += 1;
            }
        }
        if let Some((out, ins)) = bufs.split_last() {
            for r in ins {
                if r.slot != u32::MAX {
                    *read_count.entry(r.slot).or_insert(0) += 1;
                    read_idx.insert(r.slot, ci);
                }
            }
            if out.slot != u32::MAX {
                *prod_count.entry(out.slot).or_insert(0) += 1;
            }
        }
    }
    // Slots an output port can alias (resolved exactly as the port-descriptor
    // emission resolves them): neither the dequant intermediate nor the weight
    // constant may be externally visible.
    let mut port_slots: hashbrown::HashSet<u32> = hashbrown::HashSet::new();
    for &id in graph.outputs() {
        port_slots.insert(id.0);
        if let Some(node) = graph.nodes().get(id.0 as usize) {
            match node.inputs.first() {
                Some(InputSource::Node(NodeId(p))) => {
                    port_slots.insert(*p);
                }
                Some(InputSource::Constant(cid)) => {
                    port_slots.insert(node_count + cid.0);
                }
                Some(InputSource::GraphInput(g)) => {
                    if let Some(NodeId(p)) = graph.inputs().get(*g as usize) {
                        port_slots.insert(*p);
                    }
                }
                None => {}
            }
        }
    }

    let mut absorbed = vec![false; n_calls];
    let mut fused: Vec<Option<KernelCall>> = (0..n_calls).map(|_| None).collect();
    for i in 0..n_calls {
        let dq = match &calls[i] {
            KernelCall::Dequantize(c) => *c,
            _ => continue,
        };
        // Which weight encoding is this, and does a fused output-major decode
        // GEMV exist for it? Both answers come from the tier registry, so a new
        // tier is registered once rather than added to a condition here.
        let Some(tier) = quant_tier(DTypeId(dq.quant_dtype)) else {
            continue;
        };
        // W8A8 quantizes the activation, so it **changes the computed value**.
        // It is opt-in per weight slot, and a compile-time-constant weight is no
        // different in that respect from one bound at load time: the graph must
        // say so. Without this the compiler would silently re-key and re-value
        // every constant symmetric per-channel weight used at a decode shape —
        // an invisible numerics upgrade to an existing path, which is precisely
        // what a byte-exact consumer cannot absorb.
        //
        // Only `act_quant` is consulted. `weight_layout` describes how the
        // weight's bytes *arrive*, and a constant's bytes are in the graph, in
        // `[k,n]`. The compiler owns them, so it transposes them itself below
        // and sets `bq_omajor` on the call it emits. A constant that claimed
        // `OUTPUT_MAJOR` would be asserting something false about its own bytes;
        // `validate_weight_layout_declarations` rejects that at compile time.
        if dq.act_quant != hologram_types::act_quant::W8A8_TOKEN_SYM {
            continue;
        }
        if !tier.omajor_fusable || !dq.per_channel() || dq.inner != 1 {
            continue;
        }
        // Constant weight, read by this dequant alone, not port-aliased.
        let wslot = dq.input.slot;
        if wslot < node_count || port_slots.contains(&wslot) {
            continue;
        }
        let cid = (wslot - node_count) as usize;
        if cid >= packed_consts.len()
            || packed_consts[cid].is_some()
            || uses.get(&wslot) != Some(&1)
        {
            continue;
        }
        // Private single-consumer dequant output feeding a MatMul's B.
        let s = dq.output.slot;
        if s == u32::MAX || port_slots.contains(&s) {
            continue;
        }
        if prod_count.get(&s) != Some(&1) || read_count.get(&s) != Some(&1) {
            continue;
        }
        let j = match read_idx.get(&s) {
            Some(&j) if j != i && !absorbed[j] && fused[j].is_none() => j,
            _ => continue,
        };
        let mm = match &calls[j] {
            KernelCall::MatMul(c) if c.dtype == DTYPE_F32 && !c.b_packed && c.b.slot == s => *c,
            _ => continue,
        };
        let (k, n) = (mm.k as usize, mm.n as usize);
        if mm.m == 0 || mm.m > M_GATE || k > mm_act_quant::K_MAX || dq.channels != mm.n {
            continue;
        }
        // Symmetric: the per-channel zero-point constant must be all-zero.
        let zp_slot = dq.zero_points.slot;
        if zp_slot < node_count {
            continue;
        }
        let zp_ok = graph
            .constants()
            .get(ConstantId(zp_slot - node_count))
            .map(|e| e.bytes.len() == n * 4 && e.bytes.iter().all(|&b| b == 0))
            .unwrap_or(false);
        if !zp_ok {
            continue;
        }
        // `k` must be a whole number of the tier's groups, and the constant must
        // be exactly the tier's `[k,n]` storage. Both are tier data — the i4
        // nibble packing and the e8cb 8-element grouping are no longer special
        // cases here.
        if !tier.omajor_k_ok(k) {
            continue;
        }
        let Some(want_len) = tier.weight_bytes(k, n) else {
            continue;
        };
        let entry = match graph.constants().get(ConstantId(cid as u32)) {
            Some(e) if e.bytes.len() == want_len => e,
            _ => continue, // shape/dtype guard
        };
        // Derive the output-major layout: one transpose of the tier's
        // `[k/group_dim, n]` unit grid into `[n, k/group_dim]`, so each output's
        // units are contiguous. Baked into the archive; zero runtime copy.
        let Some(t) = tier.omajor_repack(&entry.bytes, k, n) else {
            continue;
        };
        // A vector-quantized tier decodes its indices through the model's own
        // codebook, supplied as the Dequantize node's 4th input. Bind it as a
        // read operand (it folds into the fused call's κ, so two models with
        // different codebooks address differently). Without it there is nothing
        // to decode against, so the fusion is declined rather than guessed.
        let codebook = if tier.needs_codebook {
            // Carried on the lowered call (the Dequantize node's 4th input). The
            // full 256-entry index space is required so the kernel dereferences
            // any `u8` index in range without a per-call bounds scan.
            let want = hologram_graph::DTypeId::E8CB_MAX_ENTRIES * tier.group_dim as usize;
            if !dq.has_codebook() || dq.codebook.length as usize != want {
                continue;
            }
            dq.codebook
        } else {
            MatMulDequantCall::NO_CODEBOOK
        };
        packed_consts[cid] = Some(t);
        fused[j] = Some(KernelCall::MatMulDequant(MatMulDequantCall {
            a: mm.a,
            bq: dq.input,
            scales: dq.scales,
            zero_points: dq.zero_points,
            output: mm.output,
            m: mm.m,
            k: mm.k,
            n: mm.n,
            channels: dq.channels,
            inner: dq.inner,
            quant_dtype: dq.quant_dtype,
            dtype: mm.dtype,
            scale_bits: dq.scale_bits,
            zero_point: dq.zero_point,
            bq_omajor: true,
            act_quant: mm_act_quant::W8A8_TOKEN_SYM,
            // The epilogue (act/residual) is absorbed by the load-time
            // epilogue pass over the archive-carried fused call.
            act: 0,
            residual: MatMulDequantCall::NO_RESIDUAL,
            codebook,
        }));
        absorbed[i] = true; // drop the standalone dequant
    }
    if !absorbed.iter().any(|&a| a) {
        return (calls, plan);
    }
    // Rebuild the call list and renumber the per-level exec plan.
    let mut new_calls: Vec<KernelCall> = Vec::with_capacity(n_calls);
    let mut remap = vec![u32::MAX; n_calls];
    for i in 0..n_calls {
        if absorbed[i] {
            continue;
        }
        remap[i] = new_calls.len() as u32;
        new_calls.push(fused[i].take().unwrap_or(calls[i]));
    }
    let mut new_plan: Vec<Vec<u32>> = Vec::with_capacity(plan.len());
    for level in &plan {
        let lvl: Vec<u32> = level
            .iter()
            .filter_map(|&ci| {
                let ci = ci as usize;
                (ci < n_calls && !absorbed[ci]).then(|| remap[ci])
            })
            .collect();
        if !lvl.is_empty() {
            new_plan.push(lvl);
        }
    }
    (new_calls, new_plan)
}
