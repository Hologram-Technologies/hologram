//! `InferenceSession` (spec VIII.1).

use alloc::vec::Vec;

use hashbrown::HashMap;
use smallvec::SmallVec;

use crate::buffer::{BufferArena, InputBuffer, OutputBuffer};
use crate::error::ExecError;
use hologram_archive::{
    address_bytes, constant_codec, decode_exec_plan, decode_ports, decode_weights, decoder,
    derive_label, derive_label_witnessed, format::SectionKind, warm_codec, ContentLabel,
    HoloLoader, PortDescriptor, WarmEntry, WeightFingerprint,
};
use hologram_backend::{
    buffers, Backend, KernelCall, MatMulActivationCall, MatMulAddActivationCall, MatMulAddCall,
    MatMulCall,
};

/// f32 dtype tag (matches `port_bytes_per_element` / the backend's
/// `DTYPE_F32`). Content-addressed fusion only fires for f32 matmuls.
const DTYPE_F32: u8 = 8;

/// Max distinct input-label sets the graph-level memo retains. A re-run
/// whose input ports content-address to a key already present returns its
/// cached outputs without touching the graph (O(1) in graph size) — the
/// content-addressing fast path for redundant execution (repeated prompt,
/// replayed request). Best-effort past the cap.
const GRAPH_MEMO_CAP: usize = 1024;

/// Precomputed per-node content-addressing metadata (built once at load).
/// The runtime walk derives each node's output κ-label from its op
/// signature and the current labels of its operand slots, so an identical
/// computation — whether a whole-graph replay or a shared sub-graph
/// (common prefix / CSE within one run) — is recognized and its kernel
/// dispatch is elided. Keeping this off the hot path (no per-op
/// re-derivation of signatures or operand lists) is what makes the
/// addressing overhead `O(operands)`, not `O(tensor)`.
struct NodeMeta {
    opcode: u16,
    params: SmallVec<[u8; 32]>,
    /// Operand slots in deterministic kernel order (the ordered-composition
    /// order); `u32::MAX` sentinels (e.g. an absent norm residual) excluded.
    inputs: SmallVec<[u32; 4]>,
    output: u32,
    /// Commutative op (`f(a,b)=f(b,a)`): operand labels are canonicalized
    /// (sorted) before addressing, so `a∘b` and `b∘a` share one κ-label.
    commutative: bool,
}

pub struct InferenceSession<B: SessionBackend> {
    /// Compiled kernel calls in topological order (compiler emits them
    /// per `compute_schedule` levels, flattened).
    kernel_calls: Vec<KernelCall>,
    /// Per-level kernel-call indices (spec VIII.2). Each entry holds
    /// indices into `kernel_calls`; the executor walks levels in order,
    /// parallelizing within a level when the backend permits.
    exec_plan: Vec<Vec<u32>>,
    inputs: Vec<PortDescriptor>,
    outputs: Vec<PortDescriptor>,
    /// The content-addressed buffer pool — the single execution substrate.
    /// Constants are pinned; intermediates/outputs/inputs are transient and
    /// byte-bounded. A value lives in exactly one buffer and is referred to
    /// by binding a slot to it: **zero runtime movement** (no copy to/from a
    /// separate store, no copy-back on reuse).
    pool: BufferArena,
    backend: B,
    /// Per-slot byte size (padded to 64), precomputed at load; a node's
    /// output buffer is allocated at its slot's size.
    slot_sizes: Vec<usize>,
    /// `(slot, label)` for each model constant, so the walk re-binds the
    /// pinned constant buffers by label each run after `rebind_reset`.
    const_bindings: Vec<(u32, ContentLabel)>,
    /// Graph-level memo: input-port κ-labels → output-port κ-labels. A
    /// re-execution whose inputs content-address to a present key returns
    /// the cached output *addresses* without walking the graph or moving
    /// any tensor bytes — the zero-cost reuse path (TC-01). Output values
    /// live in `pool`, resolvable by label.
    graph_memo: HashMap<SmallVec<[ContentLabel; 4]>, SmallVec<[ContentLabel; 4]>>,
    /// Per-input-port `(last bytes, label)` cache. Content-addressing a
    /// leaf is `O(bytes)` (BLAKE3); re-hashing an unchanged input every
    /// execute is the dominant cost on the reuse path. A byte-equality
    /// check against the previous input is far cheaper than re-hashing, so
    /// a repeated input reuses its κ-label without re-running the σ-axis.
    input_cache: Vec<Option<(Vec<u8>, ContentLabel)>>,
    /// Per-node content-addressing metadata, parallel to `kernel_calls`.
    node_meta: Vec<NodeMeta>,
    /// Initial per-slot κ-labels: model-constant slots addressed by their
    /// content (a constant is a leaf). Cloned at the start of each walk and
    /// extended with the boundary-input labels. `None` = not yet addressed.
    slot_label_init: Vec<Option<ContentLabel>>,
    /// `is_output_slot[s]` ⇒ slot `s` backs a graph output port. The node
    /// that writes such a slot additionally mints the **witnessed**
    /// (TC-05-replayable) output address (`derive_label_witnessed`) — the
    /// boundary address a caller receives. Interior nodes use only the
    /// cheap reuse key, so the per-prism-pipeline cost is paid once per
    /// output port, not per node.
    is_output_slot: Vec<bool>,
    /// Kernels dispatched in the most recent compute walk (a memo miss).
    /// With per-node addressing, a shared sub-graph is *not* dispatched, so
    /// `last_dispatched < kernel_count` whenever sub-graph reuse fires.
    last_dispatched: usize,
    /// Kernels elided in the most recent walk because their output κ-label
    /// was already resident (sub-graph / common-subexpression reuse).
    last_skipped: usize,
    /// Reused per-walk scratch (slot→label, output-port→witnessed-label) so
    /// a compute miss allocates nothing beyond the first run — the
    /// zero-cost-contract walk has no per-execute heap growth.
    slot_label_scratch: Vec<Option<ContentLabel>>,
    out_witnessed_scratch: Vec<Option<ContentLabel>>,
    /// Archive's canonical 32-byte content fingerprint (spec X.1).
    /// Routed through `prism::pipeline::run` as a W256 `Term::Literal`
    /// in `execute_attested` so the `Grounded<Digest<32>>` attestation
    /// anchors to *this* session's content, not a static dummy term.
    archive_fingerprint: [u8; 32],
    /// Per-call memory tier assignments (PM_7). Loaded from the archive's
    /// `TierAssignments` section. Empty if the section is absent.
    #[cfg(feature = "tiered-exec")]
    tiers: Vec<hologram_types::MemoryTier>,
    /// Precomputed per-level migration schedule.
    #[cfg(feature = "tiered-exec")]
    migrations: Vec<crate::coherence::LevelMigration>,
    /// Runtime tier override policy.
    #[cfg(feature = "tiered-exec")]
    tier_policy: crate::coherence::TierPolicy,
}

/// Backend bounds required for `InferenceSession` execute. Without the
/// `parallel` feature, plain `Backend<WS = BufferArena>` suffices. With
/// the feature on, the backend must be `Clone + Send + Sync` so that
/// per-thread copies can dispatch concurrently against disjoint slots
/// in the same schedule level.
#[cfg(not(feature = "parallel"))]
pub trait SessionBackend: Backend<WS = BufferArena> {}
#[cfg(not(feature = "parallel"))]
impl<B: Backend<WS = BufferArena>> SessionBackend for B {}

#[cfg(feature = "parallel")]
pub trait SessionBackend: Backend<WS = BufferArena> + Clone + Send + Sync {}
#[cfg(feature = "parallel")]
impl<B: Backend<WS = BufferArena> + Clone + Send + Sync> SessionBackend for B {}

impl<B: SessionBackend> InferenceSession<B> {
    /// Load and prepare an `.holo` archive for execution.
    pub fn load(bytes: &[u8], backend: B) -> Result<Self, ExecError> {
        let loader = HoloLoader::from_bytes(bytes)?;
        let archive_fingerprint = loader.fingerprint();
        let plan = loader.into_plan()?;
        let calls_section = plan.section(SectionKind::KernelCalls)?;
        let kernel_calls = decoder::decode_calls(calls_section).map_err(ExecError::Archive)?;

        let inputs = plan
            .section(SectionKind::Inputs)
            .ok()
            .map(decode_ports)
            .transpose()
            .map_err(ExecError::Archive)?
            .unwrap_or_default();
        let outputs = plan
            .section(SectionKind::Outputs)
            .ok()
            .map(decode_ports)
            .transpose()
            .map_err(ExecError::Archive)?
            .unwrap_or_default();

        // Decode the per-level kernel-call schedule (spec VIII.2). If the
        // archive omits an `ExecPlan`, fall back to a single level holding
        // every call (sequential execution).
        let exec_plan: Vec<Vec<u32>> = plan
            .section(SectionKind::ExecPlan)
            .ok()
            .map(decode_exec_plan)
            .transpose()
            .map_err(ExecError::Archive)?
            .unwrap_or_else(|| vec![(0..kernel_calls.len() as u32).collect()]);

        // Content-addressed fusion (the UOR-native execution pass): collapse
        // `matmul → elementwise-activation` sub-graphs into one fused op so
        // the activation's intermediate is never separately materialized or
        // addressed — the fused op carries a single κ-derivation. Runs once
        // at load over the decoded schedule; a no-op when nothing fuses.
        let (kernel_calls, exec_plan) = fuse_matmul_epilogue(kernel_calls, exec_plan, &outputs);
        // Fuse `dequantize → matmul` so a quantized weight is dequantized inside
        // the matmul (transient panel) rather than materializing the dense f32
        // weight in the pool. Constant quantized weights are folded at warm-start
        // (no runtime dequant), so a runtime dequant feeding a matmul is dynamic
        // — fusing it is a pure win.
        let (kernel_calls, exec_plan) = fuse_dequant_matmul(kernel_calls, exec_plan, &outputs);
        // Fuse `Expand → elementwise-binary` so the broadcast operand is read
        // with stride-0 indexing in place — the materialized broadcast tensor is
        // elided (the zero-movement realization of Expand for its dominant
        // consumer, e.g. the norm-VJP `Expand → Mul` and bias/scale broadcasts).
        let (kernel_calls, exec_plan) = fuse_expand_binary(kernel_calls, exec_plan, &outputs);

        // Constants are pre-fill payloads that the runtime writes into
        // designated workspace slots before any kernel dispatches.
        // Each entry is either inline bytes or a content-addressed
        // reference into the `Weights` section (spec X.3 + X-7).
        let constant_entries: Vec<constant_codec::ConstantEntry> = plan
            .section(SectionKind::Constants)
            .ok()
            .map(constant_codec::decode)
            .transpose()
            .map_err(ExecError::Archive)?
            .unwrap_or_default();

        // Decode the WeightStore so constant references can resolve.
        // Missing section is fine — only inline-only graphs hit that path.
        let weight_store = plan
            .section(SectionKind::Weights)
            .ok()
            .map(decode_weights)
            .transpose()
            .map_err(ExecError::Archive)?;

        // Provision workspace with **per-slot** byte sizes (spec VIII.3).
        //
        // Earlier revisions sized every slot at the maximum byte count
        // across all references. That makes total memory `slot_count *
        // max_size`, which scales catastrophically when one tensor is
        // GB-sized and the rest are KB-sized. The corrected layout
        // computes a per-slot size from the largest *referencing* call
        // (kernel BufferRef.length, port byte count, or constant body),
        // and lays slots out at cumulative offsets — total memory is
        // `Σ size_i` rather than `n · max_size_i`. This is a hard
        // requirement for trillion-parameter / UHD streaming workloads
        // (spec X-7).
        let mut slot_count: usize = 0;
        let bump = |sc: &mut usize, slot: u32| {
            let need = (slot as usize).saturating_add(1);
            if need > *sc {
                *sc = need;
            }
        };
        for b in kernel_calls.iter().flat_map(buffers) {
            if b.slot != u32::MAX {
                bump(&mut slot_count, b.slot);
            }
        }
        for p in inputs.iter().chain(outputs.iter()) {
            bump(&mut slot_count, p.slot);
        }
        for e in constant_entries.iter() {
            bump(&mut slot_count, e.slot);
        }

        // Byte sizes are u64 throughout (ADR-060: no 4 GiB ceiling).
        let mut sizes: Vec<u64> = vec![0u64; slot_count];
        for b in kernel_calls.iter().flat_map(buffers) {
            if b.slot != u32::MAX {
                let s = &mut sizes[b.slot as usize];
                if b.length > *s {
                    *s = b.length;
                }
            }
        }
        for p in inputs.iter().chain(outputs.iter()) {
            let bytes_per = port_bytes_per_element(p.dtype) as u64;
            let n = p.element_count.saturating_mul(bytes_per);
            let s = &mut sizes[p.slot as usize];
            if n > *s {
                *s = n;
            }
        }
        for e in constant_entries.iter() {
            // Inline bodies report their length directly; references
            // resolve through the WeightStore for sizing.
            let n: u64 = if e.by_reference {
                weight_store
                    .as_ref()
                    .and_then(|s| s.get(WeightFingerprint(e.fingerprint)))
                    .map(|b| b.len() as u64)
                    .unwrap_or(0)
            } else {
                e.bytes.len() as u64
            };
            let s = &mut sizes[e.slot as usize];
            if n > *s {
                *s = n;
            }
        }
        // Floor each slot at 64 bytes so kernels that compute their own
        // strides always have headroom.
        for s in sizes.iter_mut() {
            if *s < 64 {
                *s = 64;
            }
        }
        // Round each slot to a 64-byte boundary. The arena's backing
        // storage is 64-byte aligned (see `BufferArena::AlignedBytes`),
        // and rounding individual slot lengths up to multiples of 64
        // keeps the cumulative `offset` of every slot 64-byte aligned —
        // which in turn lets `bytemuck::cast_slice::<u8, f32>` succeed
        // zero-copy on every slot. Without this, mid-arena slots can
        // sit at odd 4-byte boundaries and force the elementwise
        // fallback path. 64 bytes is the AVX-512 / cache-line width.
        for s in sizes.iter_mut() {
            *s = s.next_multiple_of(64);
        }

        // Per-slot byte sizes (padded) drive value-buffer allocation.
        let slot_sizes: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();

        // The content-addressed pool is the single substrate. Each model
        // constant is **pinned** by its content κ-label (a leaf — its label
        // is its content; identical weights dedupe to one buffer). The walk
        // re-binds these pinned buffers by label each run via
        // `const_bindings`; their κ-label also seeds `slot_label_init` so a
        // weight-consuming op is addressable. No fixed byte arena, no
        // second copy of any weight.
        let mut pool = BufferArena::new();
        let mut slot_label_init: Vec<Option<ContentLabel>> = vec![None; slot_count];
        let mut const_bindings: Vec<(u32, ContentLabel)> = Vec::new();
        for entry in &constant_entries {
            let body: &[u8] = if entry.by_reference {
                weight_store
                    .as_ref()
                    .and_then(|s| s.get(WeightFingerprint(entry.fingerprint)))
                    .unwrap_or(&[])
            } else {
                &entry.bytes
            };
            let label = address_bytes(body);
            pool.pin_bytes(label, body);
            slot_label_init[entry.slot as usize] = Some(label);
            const_bindings.push((entry.slot, label));
        }

        // Warm-start fold (WS-2). If the archive carries materialized
        // constant-only-cone results (baked by `crate::warm::fold_archive`),
        // pin each under its κ-label so it is resident. The **existing**
        // residency check in the node walk then elides that cone node on the
        // very first run — the cache is never cold, with no walk changes and
        // no second code path. Keyed purely by κ-label, so it matches whatever
        // label the (post-fusion) walk derives. A plain `compile()` bakes no
        // results; the lattice of labels is derived on demand (`cone_lattice`),
        // never stored redundantly in the archive.
        if let Ok(section) = plan.section(SectionKind::WarmStart) {
            for entry in warm_codec::decode(section).map_err(ExecError::Archive)? {
                if !entry.result.is_empty() {
                    pool.pin_bytes(entry.label, &entry.result);
                }
            }
        }

        // Precompute per-node addressing metadata: op signature + operand
        // and output slots (operands in deterministic kernel order, the
        // ordered-composition order). Built once so the walk never
        // re-derives signatures on the hot path.
        let node_meta: Vec<NodeMeta> = kernel_calls
            .iter()
            .map(|call| {
                let sig = call.op_signature();
                let refs = buffers(call);
                let (output, ins) = refs.split_last().expect("every kernel has an output");
                let inputs: SmallVec<[u32; 4]> = ins
                    .iter()
                    .map(|b| b.slot)
                    .filter(|&s| s != u32::MAX)
                    .collect();
                NodeMeta {
                    opcode: sig.opcode,
                    params: SmallVec::from_slice(sig.params()),
                    inputs,
                    output: output.slot,
                    commutative: call.is_commutative(),
                }
            })
            .collect();

        let mut is_output_slot = vec![false; slot_count];
        for p in &outputs {
            if (p.slot as usize) < is_output_slot.len() {
                is_output_slot[p.slot as usize] = true;
            }
        }

        let inputs_len = inputs.len();

        // PM_7: load tier assignments and build migration schedule.
        #[cfg(feature = "tiered-exec")]
        let (tiers, migrations) = {
            let tier_bytes: Vec<u8> = plan
                .section(SectionKind::TierAssignments)
                .ok()
                .map(|b| b.to_vec())
                .unwrap_or_default();
            let tiers = hologram_backend::tiered::decode_tier_assignments(&tier_bytes);
            // Build migration schedule from tier assignments + exec plan.
            let call_outputs: Vec<u32> = kernel_calls
                .iter()
                .map(|c| {
                    let bufs = hologram_backend::buffers(c);
                    bufs.last().map(|b| b.slot).unwrap_or(u32::MAX)
                })
                .collect();
            let call_inputs: Vec<Vec<u32>> = kernel_calls
                .iter()
                .map(|c| {
                    let bufs = hologram_backend::buffers(c);
                    if bufs.len() > 1 {
                        bufs[..bufs.len() - 1]
                            .iter()
                            .filter(|b| b.slot != u32::MAX)
                            .map(|b| b.slot)
                            .collect()
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            let migs = crate::coherence::build_migration_schedule(
                &exec_plan,
                &tiers,
                &call_outputs,
                &call_inputs,
                slot_count,
            );
            let report = crate::coherence::build_report(&tiers, &migs);
            tracing::info!(
                cpu_l1 = report.cpu_l1_calls,
                cpu_l2 = report.cpu_l2_calls,
                cpu_main = report.cpu_main_calls,
                device = report.device_calls,
                migration_slots = report.total_migration_slots,
                migration_levels = report.levels_with_migrations,
                "PM_7 tier report"
            );
            (tiers, migs)
        };

        Ok(Self {
            kernel_calls,
            exec_plan,
            inputs,
            outputs,
            pool,
            backend,
            slot_sizes,
            const_bindings,
            graph_memo: HashMap::new(),
            input_cache: vec![None; inputs_len],
            node_meta,
            slot_label_init,
            is_output_slot,
            last_dispatched: 0,
            last_skipped: 0,
            slot_label_scratch: Vec::new(),
            out_witnessed_scratch: Vec::new(),
            archive_fingerprint,
            #[cfg(feature = "tiered-exec")]
            tiers,
            #[cfg(feature = "tiered-exec")]
            migrations,
            #[cfg(feature = "tiered-exec")]
            tier_policy: crate::coherence::TierPolicy::Compiled,
        })
    }

    /// Execute one inference pass from raw input bytes, returning raw
    /// output bytes. This is the byte↔address boundary: inputs are
    /// content-addressed (the σ-axis hashes each distinct input *once* —
    /// the per-port `input_cache` reuses the κ-label for an unchanged
    /// input, so identical bytes are never rehashed) and outputs are
    /// resolved back to bytes. Inside, everything operates on addresses
    /// via [`Self::execute_addressed`].
    ///
    /// Callers driving a pipeline (e.g. autoregressive decode) should
    /// prefer [`Self::execute_addressed`], which never touches raw bytes
    /// and so never hashes.
    pub fn execute(&mut self, inputs: &[InputBuffer]) -> Result<Vec<OutputBuffer>, ExecError> {
        if inputs.len() != self.inputs.len() {
            return Err(ExecError::InputMismatch);
        }
        // Address each input once (the per-port cache reuses the κ-label for
        // an unchanged input, so identical bytes are never rehashed).
        let mut key: SmallVec<[ContentLabel; 4]> = SmallVec::with_capacity(self.inputs.len());
        for (i, (port, buf)) in self.inputs.iter().zip(inputs.iter()).enumerate() {
            let n_bytes = (port.element_count as usize)
                .saturating_mul(port_bytes_per_element(port.dtype))
                .min(buf.bytes.len());
            let region = &buf.bytes[..n_bytes];
            let label = match &self.input_cache[i] {
                Some((prev, lbl)) if prev.as_slice() == region => *lbl,
                _ => {
                    let lbl = address_bytes(region);
                    self.input_cache[i] = Some((region.to_vec(), lbl));
                    lbl
                }
            };
            key.push(label);
        }

        // Whole-graph memo hit: outputs already addressed and resident — no
        // walk, no movement.
        let cached = self.graph_memo.get(&key).cloned();
        if let Some(labels) = cached {
            if labels.iter().all(|l| self.pool.resident(l)) {
                return self.collect_outputs(&labels);
            }
        }

        // Miss: bind constants + inputs into the pool, then walk.
        self.pool.rebind_reset(self.slot_sizes.len());
        for &(slot, label) in &self.const_bindings {
            self.pool.bind_resident(slot as usize, &label);
        }
        for (port, (buf, label)) in self.inputs.iter().zip(inputs.iter().zip(key.iter())) {
            let n_bytes = (port.element_count as usize)
                .saturating_mul(port_bytes_per_element(port.dtype))
                .min(buf.bytes.len());
            self.pool.store_unbound(*label, &buf.bytes[..n_bytes]);
            self.pool.bind_resident(port.slot as usize, label);
        }
        let labels = self.compute_and_label(key)?;
        self.collect_outputs(&labels)
    }

    /// Resolve output-port κ-labels to caller byte buffers (the only
    /// address→byte copy, at the boundary).
    fn collect_outputs(&self, labels: &[ContentLabel]) -> Result<Vec<OutputBuffer>, ExecError> {
        let mut out = Vec::with_capacity(self.outputs.len());
        for (port, label) in self.outputs.iter().zip(labels.iter()) {
            let n_bytes = (port.element_count as usize) * port_bytes_per_element(port.dtype);
            let full = self
                .pool
                .resolve(label)
                .ok_or(ExecError::WorkspaceExhausted)?;
            out.push(OutputBuffer {
                bytes: full.iter().take(n_bytes).copied().collect(),
            });
        }
        Ok(out)
    }

    /// Execute on content addresses: inputs are given by κ-label (from
    /// [`Self::intern_input`] or a previous call's output), outputs are
    /// returned as κ-labels. **Nothing is rehashed** — an already-addressed
    /// value flows by its 71-byte label. On a graph-memo hit this returns
    /// the cached output addresses immediately, without materializing
    /// inputs, walking the graph, or copying any tensor bytes (TC-01
    /// zero-cost reuse). This is the surface a content-addressed pipeline
    /// composes on: feed one call's output labels straight into the next.
    pub fn execute_addressed(
        &mut self,
        input_labels: &[ContentLabel],
    ) -> Result<Vec<ContentLabel>, ExecError> {
        if input_labels.len() != self.inputs.len() {
            return Err(ExecError::InputMismatch);
        }
        let key: SmallVec<[ContentLabel; 4]> = input_labels.iter().copied().collect();
        // Hit only counts if the cached output addresses are still
        // resolvable; otherwise fall through and recompute.
        let cached = self.graph_memo.get(&key).cloned();
        if let Some(labels) = cached {
            if labels.iter().all(|l| self.pool.resident(l)) {
                return Ok(labels.into_vec());
            }
        }
        // Miss: the addressed inputs are already resident (interned, or a
        // prior call's outputs); bind constants + inputs by label — **no
        // hashing, no copy** — then walk.
        self.pool.rebind_reset(self.slot_sizes.len());
        for &(slot, label) in &self.const_bindings {
            self.pool.bind_resident(slot as usize, &label);
        }
        for (port, label) in self.inputs.iter().zip(input_labels.iter()) {
            if !self.pool.bind_resident(port.slot as usize, label) {
                return Err(ExecError::InputMismatch);
            }
        }
        Ok(self.compute_and_label(key)?.into_vec())
    }

    /// Intern raw bytes into a content address — the byte→address
    /// boundary. The σ-axis hashes the bytes *once*; thereafter the value
    /// is referred to by its κ-label and never rehashed. Use the returned
    /// label as an input to [`Self::execute_addressed`].
    pub fn intern_input(&mut self, bytes: &[u8]) -> ContentLabel {
        let label = address_bytes(bytes);
        self.pool.store_unbound(label, bytes);
        label
    }

    /// Resolve a content address to its bytes — the address→byte boundary
    /// for reading an output returned by [`Self::execute_addressed`].
    #[must_use]
    pub fn resolve(&self, label: &ContentLabel) -> Option<&[u8]> {
        self.pool.resolve(label)
    }

    /// Compute a memo miss by walking the schedule with **per-node content
    /// addressing** — the sub-graph reuse path. Inputs must already be
    /// resident in their slots; `key` holds the boundary-input labels in
    /// input-port order.
    ///
    /// Each node's *reuse key* is the cheap, order-sensitive derivation
    /// [`derive_label`] of its operand slots' current labels with its op
    /// signature (opcode ‖ scalar params) — `O(operands)` regardless of
    /// tensor size, paid on the hot path for every node with no measurable
    /// overhead (TC-01). If that key is already resident in the pool, the
    /// computation is identical to one already performed (a shared
    /// sub-graph), so the kernel is **elided** and `out_slot` is *bound* to
    /// the existing buffer — **no copy**. Otherwise a fresh output buffer is
    /// bound, the kernel writes it once, and it is retained by label — again
    /// **no copy**. Reuse and retention are pointer-level; nothing moves.
    ///
    /// A node that writes a **graph output port** mints the witnessed
    /// (TC-05-replayable) boundary address via [`derive_label_witnessed`]
    /// (CA-3) and retains its buffer under that label — the prism-pipeline
    /// grounding cost is paid once per output port, not per node. The
    /// input→output mapping (`key`) is recorded for the O(1) whole-graph hit.
    fn compute_and_label(
        &mut self,
        key: SmallVec<[ContentLabel; 4]>,
    ) -> Result<SmallVec<[ContentLabel; 4]>, ExecError> {
        self.last_dispatched = 0;
        self.last_skipped = 0;
        // Seed slot labels: constants (from init) + boundary inputs. Reuse
        // the scratch allocation across runs (no per-execute heap growth).
        let mut slot_label = core::mem::take(&mut self.slot_label_scratch);
        slot_label.clear();
        slot_label.extend_from_slice(&self.slot_label_init);
        for (port, lbl) in self.inputs.iter().zip(key.iter()) {
            slot_label[port.slot as usize] = Some(*lbl);
        }
        // Witnessed boundary address per output slot, minted at its producer.
        let mut out_witnessed = core::mem::take(&mut self.out_witnessed_scratch);
        out_witnessed.clear();
        out_witnessed.resize(slot_label.len(), None);

        for li in 0..self.exec_plan.len() {
            for ni in 0..self.exec_plan[li].len() {
                let ci = self.exec_plan[li][ni] as usize;
                if ci >= self.kernel_calls.len() {
                    return Err(ExecError::Backend);
                }
                let out_slot = self.node_meta[ci].output as usize;

                // Gather operand labels and compute the cheap reuse key.
                let mut in_labels: SmallVec<[ContentLabel; 4]> = SmallVec::new();
                let mut addressable = true;
                for &s in &self.node_meta[ci].inputs {
                    match slot_label[s as usize] {
                        Some(l) => in_labels.push(l),
                        None => {
                            addressable = false;
                            break;
                        }
                    }
                }
                // Algebraic canonicalization: a commutative op's value is
                // independent of operand order, so sort the operand labels to
                // a canonical order before addressing. `a∘b` and `b∘a` then
                // derive the *same* κ-label and reuse each other's result
                // (and one boundary address) — UOR's algebra eliding compute.
                if addressable && self.node_meta[ci].commutative {
                    in_labels.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
                }
                let is_out = self.is_output_slot[out_slot];
                let label = if addressable {
                    // Reshape is a UOR addressing op, not compute: a row-major
                    // buffer's bytes are unchanged by a logical shape change, so
                    // its content κ-label *is* its input's. Propagating that
                    // label makes the resident-reuse path below bind the output
                    // slot to the input's buffer — zero dispatch, zero movement,
                    // zero copy. (Re-addressing, not a memcpy.)
                    if matches!(self.kernel_calls[ci], KernelCall::Reshape(_)) {
                        in_labels.first().copied()
                    } else {
                        let meta = &self.node_meta[ci];
                        Some(derive_label(meta.opcode, &meta.params, &in_labels))
                    }
                } else {
                    None
                };

                // Interior sub-graph reuse: the value is resident → bind the
                // output slot to its buffer (no dispatch, no copy).
                if let Some(label) = label {
                    if !is_out && self.pool.resident(&label) {
                        self.pool.bind_resident(out_slot, &label);
                        slot_label[out_slot] = Some(label);
                        self.last_skipped += 1;
                        continue;
                    }
                }

                // Slice = `ProjectField`: an interior slice is a zero-movement
                // view onto its input's sub-region (the compiler set the input
                // BufferRef to `[offset, offset+len)`). Bind the view — no
                // dispatch, no copy. (Output-port slices fall through to the
                // copy kernel so they retain their own buffer correctly.)
                if !is_out {
                    if let KernelCall::Slice(lc) = &self.kernel_calls[ci] {
                        let (data_slot, off, len) = (
                            lc.input.slot as usize,
                            lc.input.offset as usize,
                            lc.input.length as usize,
                        );
                        self.pool.bind_view(out_slot, data_slot, off, len);
                        slot_label[out_slot] = label;
                        self.last_skipped += 1;
                        continue;
                    }
                }

                // Miss / novel: bind a fresh output buffer and dispatch the
                // kernel straight into it.
                let size = self.slot_sizes.get(out_slot).copied().unwrap_or(64);
                self.pool.bind_fresh(out_slot, size);
                let call = self.kernel_calls[ci];
                self.backend
                    .dispatch(&call, &mut self.pool)
                    .map_err(|_| ExecError::Backend)?;
                self.last_dispatched += 1;

                match (label, is_out) {
                    (Some(label), false) => {
                        // Retain interior result by its cheap reuse key.
                        self.pool.retain(out_slot, label);
                        slot_label[out_slot] = Some(label);
                    }
                    (Some(label), true) => {
                        // Output port: retain under the witnessed boundary
                        // address (CA-3); `slot_label` keeps the cheap label
                        // for any downstream derivation.
                        let meta = &self.node_meta[ci];
                        let witnessed =
                            derive_label_witnessed(meta.opcode, &meta.params, &in_labels)
                                .map_err(|_| ExecError::Backend)?
                                .address;
                        self.pool.retain(out_slot, witnessed);
                        out_witnessed[out_slot] = Some(witnessed);
                        slot_label[out_slot] = Some(label);
                    }
                    (None, _) => {
                        slot_label[out_slot] = None;
                    }
                }
            }
        }

        // Collect output-port labels; ensure each is resolvable in the pool.
        let mut out_labels: SmallVec<[ContentLabel; 4]> =
            SmallVec::with_capacity(self.outputs.len());
        for j in 0..self.outputs.len() {
            let slot = self.outputs[j].slot as usize;
            let label = if let Some(l) = out_witnessed[slot] {
                l
            } else {
                // Fallback for an un-addressable output: address its logical
                // bytes (the only place a result is hashed) and retain.
                let n_bytes = (self.outputs[j].element_count as usize)
                    * port_bytes_per_element(self.outputs[j].dtype);
                let l = {
                    let full = self
                        .pool
                        .read_slot(slot)
                        .ok_or(ExecError::WorkspaceExhausted)?;
                    address_bytes(&full[..n_bytes.min(full.len())])
                };
                self.pool.retain(slot, l);
                l
            };
            out_labels.push(label);
        }
        // Return the scratch allocations to the session for the next run.
        self.slot_label_scratch = slot_label;
        self.out_witnessed_scratch = out_witnessed;
        if self.graph_memo.len() < GRAPH_MEMO_CAP {
            self.graph_memo.insert(key, out_labels.clone());
        }
        Ok(out_labels)
    }

    /// Kernels dispatched in the most recent compute walk (a graph-memo
    /// miss). Sub-graph reuse drops this below [`Self::kernel_count`].
    #[inline]
    pub fn last_dispatched(&self) -> usize {
        self.last_dispatched
    }

    /// Kernels elided in the most recent walk because their output κ-label
    /// was already resident — the count of reused sub-graph nodes.
    #[inline]
    pub fn last_skipped(&self) -> usize {
        self.last_skipped
    }

    pub fn kernel_count(&self) -> usize {
        self.kernel_calls.len()
    }

    /// Number of fused `matmul → activation` ops in the loaded schedule
    /// (content-addressed fusion). Each one elides an activation's
    /// intermediate from materialization and addressing. Zero when no
    /// fusable sub-graph was present.
    pub fn fused_count(&self) -> usize {
        self.kernel_calls
            .iter()
            .filter(|c| matches!(c, KernelCall::MatMulActivation(_)))
            .count()
    }
    /// Number of fused `matmul → add(residual)` ops in the loaded schedule —
    /// the transformer skip connection collapsed into the matmul epilogue,
    /// eliding both the matmul intermediate and the separate bandwidth-bound
    /// add pass.
    pub fn residual_fused_count(&self) -> usize {
        self.kernel_calls
            .iter()
            .filter(|c| matches!(c, KernelCall::MatMulAdd(_)))
            .count()
    }
    /// Number of fused `matmul → add → activation` ops in the loaded schedule —
    /// the MLP epilogue `act(A·B + bias)` collapsed into one op, eliding the
    /// matmul product, the post-add sum, *and* the activation intermediate.
    pub fn add_activation_fused_count(&self) -> usize {
        self.kernel_calls
            .iter()
            .filter(|c| matches!(c, KernelCall::MatMulAddActivation(_)))
            .count()
    }
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
    pub fn schedule_levels(&self) -> usize {
        self.exec_plan.len()
    }

    /// Per-port descriptors (slot id, element count, dtype tag) for the
    /// archive's inputs / outputs. Callers use these to size caller-side
    /// buffers when wiring through the FFI / async bridges.
    pub fn input_ports(&self) -> &[PortDescriptor] {
        &self.inputs
    }
    pub fn output_ports(&self) -> &[PortDescriptor] {
        &self.outputs
    }

    /// Byte length of the `i`-th declared input port. Returns 0 when
    /// `i >= input_count()`. Callers size their per-port input buffers from
    /// this (the archive's declared shape × dtype) rather than guessing a
    /// fixed capacity.
    pub fn input_byte_len(&self, i: usize) -> usize {
        self.inputs
            .get(i)
            .map(|p| (p.element_count as usize) * port_bytes_per_element(p.dtype))
            .unwrap_or(0)
    }

    /// Byte length of the `i`-th declared output port. Returns 0 when
    /// `i >= output_count()` so callers can pre-size buffers with a
    /// single bounded probe.
    pub fn output_byte_len(&self, i: usize) -> usize {
        self.outputs
            .get(i)
            .map(|p| (p.element_count as usize) * port_bytes_per_element(p.dtype))
            .unwrap_or(0)
    }

    /// The archive's canonical 32-byte content fingerprint (spec X.1).
    /// Used by `execute_attested` to anchor the prism attestation to
    /// this session's content.
    #[inline]
    pub fn archive_fingerprint(&self) -> [u8; 32] {
        self.archive_fingerprint
    }

    /// Set the runtime tier override policy (PM_7).
    #[cfg(feature = "tiered-exec")]
    pub fn set_tier_policy(&mut self, policy: crate::coherence::TierPolicy) {
        self.tier_policy = policy;
    }

    /// Get the current tier policy.
    #[cfg(feature = "tiered-exec")]
    pub fn tier_policy(&self) -> crate::coherence::TierPolicy {
        self.tier_policy
    }

    /// Number of distinct content-addressed values resident in the
    /// session's store. Grows as novel values are produced; stays flat
    /// across a re-execution whose every value is already addressed (all
    /// memo hits). Exposed for observability of the content-addressed
    /// execution substrate.
    #[inline]
    pub fn content_store_len(&self) -> usize {
        self.pool.resident_len()
    }

    /// The warm-start lattice (WS class): `(slot, κ-label)` of every
    /// constant-only-cone node — nodes whose transitive inputs are all
    /// constants. **Derived on demand** from the (post-fusion) kernel calls
    /// and the constants' leaf labels — a κ-label is a deterministic function
    /// of the compiled graph, so this is exactly what the runtime walk would
    /// mint, with no redundant copy baked into the archive and no per-load
    /// cost when warm-start is unused. Empty for an all-input graph. These are
    /// the keys the persisted store (WS-3) resolves and the V&V checks.
    #[must_use]
    pub fn warm_lattice(&self) -> Vec<(u32, ContentLabel)> {
        self.cone_lattice()
    }

    /// Derive the constant-only-cone lattice over the *post-fusion* kernel
    /// calls. Because it runs post-fusion, a fused constant-only
    /// `matmul→activation` is addressed as the single op the walk dispatches,
    /// so its result warms correctly (no pre-/post-fusion mismatch).
    fn cone_lattice(&self) -> Vec<(u32, ContentLabel)> {
        let input_slots: Vec<u32> = self.inputs.iter().map(|p| p.slot).collect();
        hologram_archive::derive_cone_lattice(
            &self.kernel_calls,
            &self.const_bindings,
            &input_slots,
        )
        .into_iter()
        .map(|e| (e.slot, e.label))
        .collect()
    }

    /// **Materialize the constant-only cone (WS-2, the fold).** Run every
    /// cone node — its inputs are all constants/weights, so it needs no graph
    /// input — and return a [`WarmEntry`] per node carrying its lattice
    /// κ-label and its computed result bytes. Baking these into the archive's
    /// `WarmStart` section (see [`crate::warm::fold_archive`]) lets a later
    /// load pin them as resident, so the **existing** residency check in the
    /// node walk elides the whole cone on the first run.
    ///
    /// Runs the cone through the *same* kernels and pool as a normal execute
    /// (no second compute path), so the materialized bytes are byte-identical
    /// to what the cold walk would produce. The lattice is derived
    /// **post-fusion** (see `cone_lattice`), so a fused constant-only
    /// `matmul→activation` is addressed as the single op the walk dispatches
    /// and is materialized correctly — there is no pre-/post-fusion mismatch.
    pub fn materialize_cone(&mut self) -> Result<Vec<WarmEntry>, ExecError> {
        let lattice = self.cone_lattice();
        if lattice.is_empty() {
            return Ok(Vec::new());
        }
        let cone: HashMap<u32, ContentLabel> = lattice.iter().copied().collect();
        // Bind constants; the cone reads only those (and earlier cone nodes).
        self.pool.rebind_reset(self.slot_sizes.len());
        for &(slot, label) in &self.const_bindings {
            self.pool.bind_resident(slot as usize, &label);
        }
        let mut out: Vec<WarmEntry> = Vec::with_capacity(lattice.len());
        for li in 0..self.exec_plan.len() {
            for ni in 0..self.exec_plan[li].len() {
                let ci = self.exec_plan[li][ni] as usize;
                if ci >= self.kernel_calls.len() {
                    return Err(ExecError::Backend);
                }
                let out_slot = self.node_meta[ci].output;
                let label = match cone.get(&out_slot) {
                    Some(l) => *l,
                    None => continue, // not a cone node (input-dependent)
                };
                let size = self
                    .slot_sizes
                    .get(out_slot as usize)
                    .copied()
                    .unwrap_or(64);
                self.pool.bind_fresh(out_slot as usize, size);
                let call = self.kernel_calls[ci];
                self.backend
                    .dispatch(&call, &mut self.pool)
                    .map_err(|_| ExecError::Backend)?;
                self.pool.retain(out_slot as usize, label);
                // Harvest the *logical* output bytes (the call's output
                // BufferRef length), so a pinned result resolves to exactly
                // what a dispatch would produce.
                let logical = buffers(&call)
                    .last()
                    .map(|b| b.length as usize)
                    .unwrap_or(0);
                let full = self
                    .pool
                    .read_slot(out_slot as usize)
                    .ok_or(ExecError::WorkspaceExhausted)?;
                let take = logical.min(full.len());
                out.push(WarmEntry {
                    slot: out_slot,
                    label,
                    result: full[..take].to_vec(),
                });
            }
        }
        Ok(out)
    }

    /// **Persist the materialized cone to a κ-store (WS-3).** Materializes the
    /// constant-only cone (see [`Self::materialize_cone`]) and writes each
    /// result under its lattice κ-label, so a later process can warm from it.
    /// Returns the number of cone nodes persisted.
    pub fn persist_cone(
        &mut self,
        store: &mut dyn crate::warm::WarmStore,
    ) -> Result<usize, ExecError> {
        let entries = self.materialize_cone()?;
        for e in &entries {
            store.put(&e.label, &e.result);
        }
        Ok(entries.len())
    }

    /// **Warm the pool from a persisted κ-store (WS-3).** For each
    /// constant-only-cone node, if the store holds a result under its lattice
    /// κ-label, pin it resident — so the **existing** residency check in the
    /// node walk elides that node on the first run, exactly as a baked fold
    /// (WS-2) would, but sourced from a prior process. A label the store
    /// lacks is simply left cold (the runtime recomputes it), so a missing or
    /// foreign store is always safe. Returns the number of cone nodes warmed.
    pub fn warm_from_store(&mut self, store: &dyn crate::warm::WarmStore) -> usize {
        let mut warmed = 0usize;
        for (_slot, label) in self.cone_lattice() {
            if !self.pool.resident(&label) {
                if let Some(bytes) = store.get(&label) {
                    self.pool.pin_bytes(label, &bytes);
                    warmed += 1;
                }
            }
        }
        warmed
    }
}

impl<B: SessionBackend> InferenceSession<B> {
    pub fn workspace(&self) -> &BufferArena {
        &self.pool
    }
    pub fn workspace_mut(&mut self) -> &mut BufferArena {
        &mut self.pool
    }
}

/// Bytes-per-element for a port descriptor's dtype tag (mirrors
/// `hologram_backend::cpu::dtype` constants but kept local to avoid an
/// upward dependency from exec on the backend's internal module).
const fn port_bytes_per_element(dtype: u8) -> usize {
    match dtype {
        0..=2 => 1,     // BOOL, U8, I8
        6 | 7 => 2,     // F16, BF16
        4 | 8 => 4,     // I32, F32
        3 | 5 | 9 => 8, // U64, I64, F64
        _ => 1,
    }
}

/// Content-addressed fusion pass (run once at load). Collapses every
/// `matmul → elementwise-unary-activation` pair into a single fused
/// [`KernelCall::MatMulActivation`], provided the fusion is *safe*:
///
/// * the matmul's output slot is produced only by that matmul and read by
///   exactly one consumer — the activation (so the intermediate has no
///   other observer), and
/// * that output slot is not a graph output port.
///
/// The fused op writes directly to the activation's output slot, so the
/// matmul's intermediate is never materialized as a distinct addressed
/// value — the executor addresses the fused node as one κ-derivation. The
/// schedule is rebuilt with the activation's level entry dropped; the fused
/// op stays at the matmul's (earlier) level, which preserves all
/// dependencies (its result is ready no later than before).
/// Fuse a `Dequantize → MatMul` pair — the dequant produces the matmul's B
/// operand and has no other consumer — into one [`KernelCall::MatMulDequant`],
/// so the dense f32 weight is never materialized in the pool (the quantized
/// source is dequantized into a transient panel inside the matmul kernel).
/// Constant quantized weights are folded at warm-start (no runtime dequant), so
/// a *runtime* dequant feeding a matmul is a dynamic operand — fusing it elides
/// a full dense tensor with no recompute regression. Runs at load.
/// Fuse `Expand → elementwise-binary` (`Add`/`Sub`/`Mul`) — the Expand produces
/// one binary operand and has no other consumer — into one
/// [`KernelCall::BroadcastBinary`], reading the pre-Expand operand with stride-0
/// broadcast indexing so the materialized broadcast tensor is elided. Float
/// only (the byte ring isn't fused). Runs at load.
/// Expand→Binary fusion removed — `BroadcastBinaryCall` no longer exists in main.
fn fuse_expand_binary(
    calls: Vec<KernelCall>,
    plan: Vec<Vec<u32>>,
    _outputs: &[PortDescriptor],
) -> (Vec<KernelCall>, Vec<Vec<u32>>) {
    (calls, plan)
}

/// Dequant→MatMul fusion removed — `MatMulDequantCall` no longer exists in main.
fn fuse_dequant_matmul(
    calls: Vec<KernelCall>,
    plan: Vec<Vec<u32>>,
    _outputs: &[PortDescriptor],
) -> (Vec<KernelCall>, Vec<Vec<u32>>) {
    (calls, plan)
}

fn fuse_matmul_epilogue(
    calls: Vec<KernelCall>,
    plan: Vec<Vec<u32>>,
    outputs: &[PortDescriptor],
) -> (Vec<KernelCall>, Vec<Vec<u32>>) {
    use hashbrown::HashSet;
    let n = calls.len();

    // Per-slot producer/reader census (excluding the u32::MAX sentinel).
    let mut prod_count: HashMap<u32, u32> = HashMap::new();
    let mut read_count: HashMap<u32, u32> = HashMap::new();
    let mut read_idx: HashMap<u32, usize> = HashMap::new();
    for (i, call) in calls.iter().enumerate() {
        let refs = buffers(call);
        if let Some((out, ins)) = refs.split_last() {
            for r in ins {
                if r.slot != u32::MAX {
                    *read_count.entry(r.slot).or_insert(0) += 1;
                    read_idx.insert(r.slot, i);
                }
            }
            if out.slot != u32::MAX {
                *prod_count.entry(out.slot).or_insert(0) += 1;
            }
        }
    }
    let out_slots: HashSet<u32> = outputs.iter().map(|p| p.slot).collect();
    // Schedule level at which each slot is produced — so a residual operand is
    // only fused when it is ready no later than the matmul that absorbs it.
    let mut slot_level: HashMap<u32, usize> = HashMap::new();
    for (li, level) in plan.iter().enumerate() {
        for &ci in level {
            if let Some((out, _)) = buffers(&calls[ci as usize]).split_last() {
                if out.slot != u32::MAX {
                    slot_level.insert(out.slot, li);
                }
            }
        }
    }

    // Decide fusions: fused[i] replaces matmul i; absorbed[j] drops the
    // consumer (activation or residual-add) j.
    let mut absorbed = vec![false; n];
    let mut fused: Vec<Option<KernelCall>> = (0..n).map(|_| None).collect();
    for i in 0..n {
        let mm = match &calls[i] {
            KernelCall::MatMul(c) if c.dtype == DTYPE_F32 => *c,
            _ => continue,
        };
        let s = mm.output.slot;
        if s == u32::MAX || out_slots.contains(&s) {
            continue;
        }
        if prod_count.get(&s) != Some(&1) || read_count.get(&s) != Some(&1) {
            continue;
        }
        let j = match read_idx.get(&s) {
            Some(&j) if j != i && !absorbed[j] => j,
            _ => continue,
        };
        let jrefs = buffers(&calls[j]);
        let (jout, jins) = match jrefs.split_last() {
            Some(v) => v,
            None => continue,
        };

        if let Some(act) = calls[j].fused_activation() {
            // matmul → elementwise-activation: the activation reads exactly s.
            if jins.len() != 1 || jins[0].slot != s {
                continue;
            }
            let fused_mm = MatMulCall {
                output: *jout,
                ..mm
            };
            fused[i] = Some(KernelCall::MatMulActivation(MatMulActivationCall {
                mm: fused_mm,
                act,
            }));
            absorbed[j] = true;
        } else if matches!(&calls[j], KernelCall::Add(_)) {
            // matmul → add(matmul_out, residual): the transformer residual.
            // One add operand is s; the other is the residual, which must be
            // ready no later than the matmul's level (else moving the read
            // earlier would observe a not-yet-computed tensor).
            if jins.len() != 2 {
                continue;
            }
            let residual = if jins[0].slot == s {
                jins[1]
            } else if jins[1].slot == s {
                jins[0]
            } else {
                continue;
            };
            let ready = match (slot_level.get(&residual.slot), slot_level.get(&s)) {
                (None, _) => true, // graph input / constant — resident at load
                (Some(rl), Some(ml)) => rl < ml,
                _ => true,
            };
            if !ready {
                continue;
            }

            // Three-op chain `matmul → add → activation` (the MLP epilogue
            // `act(A·B + bias)`): if the add's own output has a single observer
            // that is an elementwise activation, absorb it too so neither the
            // matmul product nor the post-add sum is ever materialized.
            let add_out = jout.slot;
            let act_consumer = (add_out != u32::MAX
                && !out_slots.contains(&add_out)
                && prod_count.get(&add_out) == Some(&1)
                && read_count.get(&add_out) == Some(&1))
            .then(|| read_idx.get(&add_out).copied())
            .flatten()
            .filter(|&k| k != j && !absorbed[k]);
            if let Some(k) = act_consumer {
                if let Some(act) = calls[k].fused_activation() {
                    let krefs = buffers(&calls[k]);
                    if let Some((kout, kins)) = krefs.split_last() {
                        if kins.len() == 1 && kins[0].slot == add_out {
                            let fused_mm = MatMulCall {
                                output: *kout,
                                ..mm
                            };
                            fused[i] =
                                Some(KernelCall::MatMulAddActivation(MatMulAddActivationCall {
                                    mm: fused_mm,
                                    residual,
                                    act,
                                }));
                            absorbed[j] = true;
                            absorbed[k] = true;
                            continue;
                        }
                    }
                }
            }

            let fused_mm = MatMulCall {
                output: *jout,
                ..mm
            };
            fused[i] = Some(KernelCall::MatMulAdd(MatMulAddCall {
                mm: fused_mm,
                residual,
            }));
            absorbed[j] = true;
        }
    }
    if !absorbed.iter().any(|&a| a) {
        return (calls, plan);
    }

    // Rebuild calls + old→new index remap (absorbed activations dropped).
    let mut new_calls: Vec<KernelCall> = Vec::with_capacity(n);
    let mut remap = vec![u32::MAX; n];
    for i in 0..n {
        if absorbed[i] {
            continue;
        }
        remap[i] = new_calls.len() as u32;
        new_calls.push(fused[i].take().unwrap_or(calls[i]));
    }
    // Rebuild schedule: remap surviving indices, drop absorbed, drop empties.
    let mut new_plan: Vec<Vec<u32>> = Vec::with_capacity(plan.len());
    for level in &plan {
        let lvl: Vec<u32> = level
            .iter()
            .filter_map(|&ci| {
                let ci = ci as usize;
                (ci < n && !absorbed[ci]).then(|| remap[ci])
            })
            .collect();
        if !lvl.is_empty() {
            new_plan.push(lvl);
        }
    }
    (new_calls, new_plan)
}
