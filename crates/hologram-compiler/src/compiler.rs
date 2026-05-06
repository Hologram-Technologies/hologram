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
use hologram_backend::{KernelCall, BufferRef};
use hologram_graph::{Graph, GraphOp};
use hologram_host::HologramHasher;
use uor_foundation::WittLevel;
use uor_foundation::enforcement::{Hasher, Term, TermArena, TermList, Binding};
use uor_foundation::enums::VerificationDomain;
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

        // Per-node element counts derived from the graph's shape registry.
        let element_counts: Vec<u32> = self.graph.nodes().iter().map(|n| {
            self.graph.shape_registry()
                .get(n.output_shape)
                .map(|s| s.total_elements().min(u32::MAX as u64) as u32)
                .unwrap_or(0)
        }).collect();

        // Emit kernel calls in schedule (topological) order so the executor's
        // sequential walk respects data dependencies even when graph nodes
        // were inserted out of order.
        let traversal: Vec<u32> = match self.graph.schedule() {
            Some(sched) => sched.levels.iter()
                .flat_map(|level| level.iter().copied())
                .map(|hologram_graph::NodeId(id)| id)
                .collect(),
            None => (0..self.graph.node_count() as u32).collect(),
        };

        for &idx_u32 in &traversal {
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
            let bindings: &[Binding] = &[];
            let domains: &[VerificationDomain] = &[VerificationDomain::Algebraic];
            let _validated_unit = compile_pipeline::build_unit(&PerNodeUnit {
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

            // Step 7: cache lookup / insert.
            let fingerprint = compute_fingerprint(kind, self.level, self.target);
            let cached = self.cache.get_raw(&fingerprint);

            // Step 8: lower to KernelCall using shape-derived sizing.
            let element_count = element_counts.get(idx).copied().unwrap_or(0);
            let dtype = node.output_dtype.0;
            let lowered = LoweredNode {
                kind,
                inputs: collect_buffers(&self.graph, node),
                output: BufferRef { slot: idx as u32, offset: 0, length: element_count },
                element_count,
                witt_bits: self.level.witt_length() as u16,
                dtype,
            };
            let kernel_call = if let Some(c) = cached {
                stats.cache_hits += 1;
                c.kernel_call
            } else {
                stats.cache_misses += 1;
                let call = lower::lower(&lowered)?;
                self.cache.insert_raw(fingerprint, CachedCertificate {
                    record: cert_record,
                    kernel_call: call,
                });
                call
            };
            kernel_calls.push(kernel_call);
            certificate_records.push(cert_record);
        }

        // Step 9: emit archive.
        let mut writer = HoloWriter::new();
        writer.set_kernel_calls(kernel_calls);
        if let Some(s) = self.graph.schedule() {
            writer.set_schedule(s.clone());
        }
        writer.set_weights(WeightStore::new());
        writer.set_shape_registry(self.graph.shape_registry().clone());
        if !certificate_records.is_empty() {
            writer.set_certificates(certificate_codec::encode(&certificate_records));
        }

        // Emit input/output port descriptors so the runtime can map caller
        // tensors into the workspace's slot numbering.
        let port_for = |id: hologram_graph::NodeId| -> PortDescriptor {
            let idx = id.0 as usize;
            let n = self.graph.nodes().get(idx);
            let element_count = element_counts.get(idx).copied().unwrap_or(0);
            PortDescriptor {
                slot: idx as u32,
                element_count,
                dtype: n.map(|n| n.output_dtype.0).unwrap_or(0),
            }
        };
        let inputs: Vec<PortDescriptor> = self.graph.inputs().iter().copied().map(port_for).collect();
        let outputs: Vec<PortDescriptor> = self.graph.outputs().iter().copied().map(port_for).collect();
        writer.set_inputs(inputs);
        writer.set_outputs(outputs);

        let archive = writer.finish().map_err(CompileError::Archive)?;

        Ok(CompilationOutput { archive, stats })
    }
}

/// Emit a per-node Term arena. Returns a fixed-CAP arena populated with the
/// op's canonical decomposition (spec V.3).
fn build_node_arena(
    kind: hologram_graph::OpKind,
    _level: WittLevel,
) -> Result<TermArena<128>, CompileError> {
    let mut arena: TermArena<128> = TermArena::new();
    let arity = kind.primary_arity();

    // Push one Variable term per argument (contiguously), so that
    // `TermList { start: args_start, len: arity }` resolves correctly.
    let args_start = arena.push(Term::Variable { name_index: 0 })
        .ok_or(CompileError::ArenaOverflow("variable 0"))?;
    for i in 1..arity {
        arena.push(Term::Variable { name_index: i as u32 })
            .ok_or(CompileError::ArenaOverflow("variable"))?;
    }

    if kind.is_layout_only() {
        // Layout ops emit a single Variable referencing the remapped binding.
        return Ok(arena);
    }

    arena.push(Term::Application {
        operator: kind.primary_primitive(),
        args: TermList { start: args_start, len: arity as u32 },
    }).ok_or(CompileError::ArenaOverflow("application"))?;

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

#[allow(dead_code)]
fn target_domains() -> &'static [VerificationDomain] {
    &[VerificationDomain::Algebraic]
}

fn collect_buffers(_graph: &Graph, node: &hologram_graph::Node) -> Vec<BufferRef> {
    node.inputs.iter().enumerate().map(|(i, _)| BufferRef {
        slot: i as u32, offset: 0, length: 0,
    }).collect()
}
