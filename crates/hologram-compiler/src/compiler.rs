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

use hologram_archive::{HoloWriter, WeightStore, PortDescriptor};
use hologram_archive::certificate_codec::{self, CertificateRecord};
use hologram_archive::constant_codec::ConstantEntry;
use hologram_backend::{KernelCall, BufferRef};
use hologram_graph::{Graph, GraphOp};
use hologram_host::HologramHasher;
use prism::operation::{Term, TermArena};
use prism::vocabulary::{Hasher, VerificationDomain, WittLevel};
// `Binding` is foundation-only (not in prism::operation's curated
// surface as of 0.1.3); reach it through prism's substrate re-export.
use prism::uor_foundation::enforcement::Binding;
use crate::cache::{CertificateCache, CachedCertificate};
use crate::error::CompileError;
use crate::lower::{self, LoweredNode};
use crate::pipeline::{self as compile_pipeline, PerNodeUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu, Avx2, Avx512, Neon, Metal, Wgpu,
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
        Self { graph, target, level, cache: CertificateCache::new() }
    }

    pub fn compile(mut self) -> Result<CompilationOutput, CompileError> {
        // Schedule.
        self.graph.compute_schedule();

        let mut stats = CompilationStats {
            total_nodes: self.graph.node_count() as u32,
            schedule_levels: self.graph.schedule().map(|s| s.level_count() as u32).unwrap_or(0),
            cache_hits: 0,
            cache_misses: 0,
            validated_units: 0,
        };

        let mut kernel_calls: Vec<KernelCall> = Vec::with_capacity(self.graph.node_count());
        let mut certificate_records: Vec<CertificateRecord> = Vec::with_capacity(self.graph.node_count());

        // Per-level kernel-call indices (spec VIII.2). Each entry holds the
        // call positions in `kernel_calls` that belong to that schedule
        // level; the runtime executor walks them in order, parallelizing
        // within a level where the backend supports it.
        let mut exec_plan: Vec<Vec<u32>> = Vec::new();

        // Per-node element counts derived from the graph's shape registry.
        // `byte_lengths[i]` = element_count * bytes_per_element(dtype) — the
        // size in bytes the workspace slot must hold for node i.
        let element_counts: Vec<u32> = self.graph.nodes().iter().map(|n| {
            self.graph.shape_registry()
                .get(n.output_shape)
                .map(|s| s.total_elements().min(u32::MAX as u64) as u32)
                .unwrap_or(0)
        }).collect();
        let byte_lengths: Vec<u32> = self.graph.nodes().iter().enumerate().map(|(i, n)| {
            let elements = element_counts[i] as u64;
            let bytes_per = bytes_per_element(n.output_dtype.0) as u64;
            (elements * bytes_per).min(u32::MAX as u64) as u32
        }).collect();

        // Emit kernel calls in schedule (topological) order so the executor's
        // sequential walk respects data dependencies even when graph nodes
        // were inserted out of order. Build a `traversal_levels: Vec<Vec<u32>>`
        // grouped by schedule level so we can record kernel-call indices
        // per level for the runtime exec plan.
        let traversal_levels: Vec<Vec<u32>> = match self.graph.schedule() {
            Some(sched) => sched.levels.iter()
                .map(|level| level.iter().map(|hologram_graph::NodeId(id)| *id).collect())
                .collect(),
            None => vec![(0..self.graph.node_count() as u32).collect()],
        };

        for level_nodes in &traversal_levels {
            let mut level_calls: Vec<u32> = Vec::with_capacity(level_nodes.len());

        for &idx_u32 in level_nodes {
            let idx = idx_u32 as usize;
            let node = match self.graph.nodes().get(idx) { Some(n) => n, None => continue };
            let kind = match node.op {
                GraphOp::Op(k) => k,
                GraphOp::Input | GraphOp::Output | GraphOp::Constant(_) => continue,
            };

            // Steps 3-5: emit Term tree and validate the per-node CompileUnit.
            // Materialize a contiguous &[Term] from the arena's `Option<Term>` slots.
            let arena = build_node_arena(kind, self.level)?;
            let term_vec: Vec<Term> = arena.as_slice().iter().filter_map(|t| *t).collect();
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
                result_type_iri: "https://hologram.uor.foundation/type/tensor",
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
                self.cache.insert_raw(fingerprint, CachedCertificate { record: cert_record });
            }

            // Step 8: lower to KernelCall using per-node shape-derived sizing.
            let element_count = element_counts.get(idx).copied().unwrap_or(0);
            let byte_len = byte_lengths.get(idx).copied().unwrap_or(0);
            let dtype = node.output_dtype.0;
            // Quantization params (spec X-5). The compiler reads any
            // `set_quant_attrs(NodeId, _)` the graph builder attached and
            // threads them into the lowered node; the K::Dequantize arm
            // consumes them directly.
            let quant_attrs = self.graph.quant_attrs(hologram_graph::NodeId(idx as u32))
                .unwrap_or_default();
            let lowered = LoweredNode {
                kind,
                inputs: collect_buffers(&self.graph, node, &byte_lengths),
                output: BufferRef { slot: idx as u32, offset: 0, length: byte_len },
                element_count,
                witt_bits: self.level.witt_length() as u16,
                dtype,
                shape: lower::ShapeArgs::from_graph(&self.graph, hologram_graph::NodeId(idx as u32), node),
                quant: lower::QuantParams {
                    quant_dtype: quant_attrs.quant_dtype,
                    scale_bits: quant_attrs.scale_bits,
                    zero_point: quant_attrs.zero_point,
                },
            };
            let kernel_call = lower::lower(&lowered)?;
            level_calls.push(kernel_calls.len() as u32);
            kernel_calls.push(kernel_call);
            certificate_records.push(cert_record);
        }

            if !level_calls.is_empty() {
                exec_plan.push(level_calls);
            }
        }

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
            if let Some(entry) = self.graph.constants().get(hologram_graph::ConstantId(i as u32)) {
                if entry.bytes.len() > INLINE_THRESHOLD_BYTES {
                    let fp = weights.insert(entry.bytes.clone());
                    *slot = Some(fp.0);
                }
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
        let inputs: Vec<PortDescriptor> = self.graph.inputs().iter().copied().map(|id| {
            let idx = id.0 as usize;
            let n = self.graph.nodes().get(idx);
            PortDescriptor {
                slot: idx as u32,
                element_count: element_counts.get(idx).copied().unwrap_or(0),
                dtype: n.map(|n| n.output_dtype.0).unwrap_or(0),
            }
        }).collect();

        let outputs: Vec<PortDescriptor> = self.graph.outputs().iter().copied().map(|id| {
            let idx = id.0 as usize;
            let n = self.graph.nodes().get(idx);
            // Resolve the output port's data slot to the producer node's
            // slot via the Output node's first input source.
            let producer_idx = n.and_then(|n| n.inputs.first()).and_then(|src| match *src {
                hologram_graph::InputSource::Node(hologram_graph::NodeId(p)) => Some(p as usize),
                _ => None,
            }).unwrap_or(idx);
            PortDescriptor {
                slot: producer_idx as u32,
                element_count: element_counts.get(producer_idx).copied().unwrap_or(0),
                dtype: self.graph.nodes().get(producer_idx)
                    .map(|p| p.output_dtype.0)
                    .or_else(|| n.map(|n| n.output_dtype.0))
                    .unwrap_or(0),
            }
        }).collect();
        writer.set_inputs(inputs);
        writer.set_outputs(outputs);

        // Emit constants: each entry pre-fills a workspace slot with the
        // constant's bytes at session-load time. Small bodies are
        // inlined; larger bodies become references into the Weights
        // pool (see weight dedup above).
        let node_count = self.graph.node_count() as u32;
        let constants: Vec<ConstantEntry> = (0..self.graph.constants().len())
            .filter_map(|i| {
                let id = hologram_graph::ConstantId(i as u32);
                let entry = self.graph.constants().get(id)?;
                let slot = node_count + (i as u32);
                let dtype = entry.dtype.0;
                Some(if let Some(fp) = const_fingerprints[i] {
                    ConstantEntry::reference(slot, dtype, fp)
                } else {
                    ConstantEntry::inline(slot, dtype, entry.bytes.clone())
                })
            })
            .collect();
        if !constants.is_empty() {
            writer.set_constants(constants);
        }

        let archive = writer.finish().map_err(CompileError::Archive)?;

        Ok(CompilationOutput { archive, stats })
    }
}

/// Compile-time binding table for the per-node Term arena (spec O-3).
/// Each entry maps `name_index = i` to `value_index = i`, which is the
/// arena slot of the corresponding `Term::Variable` pushed first.
/// `surface` is a static identifier (`"v0"`, `"v1"`, `"v2"`) for tooling.
/// `content_address` is the binding's FNV-1a fingerprint, used by
/// upstream's `BindingsTable` for cross-binding deduplication.
const VAR_BINDINGS: &[Binding] = &[
    Binding {
        name_index: 0, type_index: 0, value_index: 0,
        surface: "v0", content_address: 0xCBF2_9CE4_8422_2325,
    },
    Binding {
        name_index: 1, type_index: 0, value_index: 1,
        surface: "v1", content_address: 0xCBF2_9CE4_8422_2326,
    },
    Binding {
        name_index: 2, type_index: 0, value_index: 2,
        surface: "v2", content_address: 0xCBF2_9CE4_8422_2327,
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
) -> Result<Box<TermArena<128>>, CompileError> {
    // Heap-allocate the arena: `TermArena<128>` is a 128-slot array of
    // `Option<Term>` and each `Term::Literal` carries a 4 KiB
    // `TermValue` byte buffer (`DefaultHostBounds::TERM_VALUE_MAX_BYTES`
    // in `uor-foundation 0.4.15`). On-stack instantiation in a deep
    // compile loop overflows the default thread stack; boxing keeps the
    // arena on the heap while preserving the value-type ownership story.
    let mut arena: Box<TermArena<128>> = Box::new(TermArena::new());
    let arity = kind.primary_arity();

    let args_start = arena.push(Term::Variable { name_index: 0 })
        .ok_or(CompileError::ArenaOverflow("variable 0"))?;
    for i in 1..arity {
        arena.push(Term::Variable { name_index: i as u32 })
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

fn collect_buffers(graph: &Graph, node: &hologram_graph::Node, byte_lengths: &[u32]) -> Vec<BufferRef> {
    use hologram_graph::InputSource;
    node.inputs.iter().map(|src| match *src {
        InputSource::Node(hologram_graph::NodeId(id)) => BufferRef {
            slot: id,
            offset: 0,
            length: byte_lengths.get(id as usize).copied().unwrap_or(0),
        },
        InputSource::Constant(hologram_graph::ConstantId(id)) => BufferRef {
            slot: graph.node_count() as u32 + id,
            offset: 0,
            length: graph.constants().get(hologram_graph::ConstantId(id))
                .map(|e| e.bytes.len() as u32).unwrap_or(0),
        },
        InputSource::GraphInput(idx) => {
            let id = graph.inputs().get(idx as usize)
                .map(|hologram_graph::NodeId(i)| *i)
                .unwrap_or(0);
            BufferRef {
                slot: id,
                offset: 0,
                length: byte_lengths.get(id as usize).copied().unwrap_or(0),
            }
        }
    }).collect()
}

/// Bytes per element for a dtype-id (mirrors `hologram_backend::cpu::dtype`
/// constants). Centralised here so the compiler doesn't depend on the CPU
/// backend module path.
const fn bytes_per_element(dtype: u8) -> usize {
    match dtype {
        0..=2 => 1,            // BOOL, U8, I8
        6 | 7 => 2,            // F16, BF16
        4 | 8 => 4,            // I32, F32
        3 | 5 | 9 => 8,        // U64, I64, F64
        _ => 1,
    }
}
