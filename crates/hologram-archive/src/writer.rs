//! Archive writer (spec X.2).

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::ArchiveError;
use crate::format::{SectionKind, SectionRef, FORMAT_VERSION, MAGIC};
use crate::weight::WeightStore;
use hologram_compute::KernelCall;
use hologram_graph::{Schedule, ShapeRegistry};

/// A graph input/output port's identity: which workspace slot the runtime
/// fills (input) or reads (output), its semantic `name`, dtype, and full
/// `shape`. The name and shape let a caller map model inputs
/// (`input_ids`/`attention_mask`/`pixel_values`/…) to ports and know their
/// dimensions — multi-input models can't be driven positionally alone. Both
/// ONNX and GGUF carry these, and `.holo` now preserves them.
#[derive(Debug, Clone, Default)]
pub struct PortDescriptor {
    /// Semantic port name (e.g. `"input_ids"`). Empty string ⇒ unnamed.
    pub name: String,
    pub slot: u32,
    /// Total element count = product of `shape` (authoritative for buffer
    /// sizing). `u64` so tensors with more than 4.29 B elements don't overflow
    /// (ADR-060: no fixed ceiling).
    pub element_count: u64,
    pub dtype: u8,
    /// Full row-major shape. Empty ⇒ rank unknown (a scalar or a producer whose
    /// shape wasn't registered); `element_count` remains authoritative.
    pub shape: Vec<u64>,
}

#[derive(Default)]
pub struct HoloWriter {
    /// `.holo` v3 application manifest: the opaque canonical bytes of an
    /// `AppManifest` realization. Empty ⇒ a bare tensor archive (no manifest
    /// section emitted). See [`HoloWriter::set_app_manifest`].
    app_manifest: Vec<u8>,
    kernel_calls: Vec<KernelCall>,
    schedule: Option<Schedule>,
    weights: WeightStore,
    shape_registry: Option<ShapeRegistry>,
    certificates: Vec<u8>,
    trace: Vec<u8>,
    metadata: Vec<u8>,
    inputs: Vec<PortDescriptor>,
    outputs: Vec<PortDescriptor>,
    constants: Vec<crate::constant_codec::ConstantEntry>,
    /// Per-level kernel-call indices (spec VIII.2).
    exec_plan: Vec<Vec<u32>>,
    /// Open producer-defined metadata sections (`key`, `bytes`); one
    /// `SectionKind::Extension` section each. See [`HoloWriter::add_extension`].
    extensions: Vec<(String, Vec<u8>)>,
    /// κ-addressed content blobs embedded in a **fat** `.holo` (`κ bytes`, `content`); one
    /// `SectionKind::ContentBlob` section each. See [`HoloWriter::add_content_blob`].
    content_blobs: Vec<(Vec<u8>, Vec<u8>)>,
}

impl HoloWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the `.holo` v3 application manifest — the canonical bytes of an
    /// `AppManifest` realization (`hologram-space`). The archive layer stores it
    /// opaquely; the app loader resolves its closure. Omit it for a bare tensor
    /// archive (no manifest section is emitted).
    pub fn set_app_manifest(&mut self, bytes: Vec<u8>) {
        self.app_manifest = bytes;
    }
    pub fn set_kernel_calls(&mut self, calls: Vec<KernelCall>) {
        self.kernel_calls = calls;
    }
    pub fn set_schedule(&mut self, sched: Schedule) {
        self.schedule = Some(sched);
    }
    pub fn set_weights(&mut self, weights: WeightStore) {
        self.weights = weights;
    }
    pub fn set_shape_registry(&mut self, registry: ShapeRegistry) {
        self.shape_registry = Some(registry);
    }
    pub fn set_certificates(&mut self, bytes: Vec<u8>) {
        self.certificates = bytes;
    }
    pub fn set_trace(&mut self, bytes: Vec<u8>) {
        self.trace = bytes;
    }
    pub fn set_metadata(&mut self, bytes: Vec<u8>) {
        self.metadata = bytes;
    }
    pub fn set_inputs(&mut self, ports: Vec<PortDescriptor>) {
        self.inputs = ports;
    }
    pub fn set_outputs(&mut self, ports: Vec<PortDescriptor>) {
        self.outputs = ports;
    }
    pub fn set_constants(&mut self, entries: Vec<crate::constant_codec::ConstantEntry>) {
        self.constants = entries;
    }
    pub fn set_exec_plan(&mut self, levels: Vec<Vec<u32>>) {
        self.exec_plan = levels;
    }
    /// Embed a κ-addressed content blob — a **fat** `.holo` carries one per layer/closure κ so it
    /// resolves without the store (spec 03 §Fat and thin). `kappa` is the 71-byte κ-label; `content`
    /// is its bytes. Repeatable.
    pub fn add_content_blob(&mut self, kappa: impl Into<Vec<u8>>, content: impl Into<Vec<u8>>) {
        self.content_blobs.push((kappa.into(), content.into()));
    }
    /// Attach an open producer-defined metadata section under `key` (tokenizer,
    /// generation config, class labels, …). Repeatable; the runtime stores it
    /// opaquely and a consumer fetches it by key.
    pub fn add_extension(&mut self, key: impl Into<String>, bytes: Vec<u8>) {
        self.extensions.push((key.into(), bytes));
    }

    /// Serialize the archive into an in-memory buffer.
    /// Body layout: header || section_table || section_payloads || footer.
    pub fn finish(self) -> Result<Vec<u8>, ArchiveError> {
        // Build payload sections in a stable order.
        let mut payloads: Vec<(SectionKind, Vec<u8>)> = Vec::new();

        // AppManifest first (the application root) when present — a thin archive
        // is manifest + certificates + footer. Opaque bytes; omitted for a bare
        // tensor archive.
        if !self.app_manifest.is_empty() {
            payloads.push((SectionKind::AppManifest, self.app_manifest.clone()));
        }

        // KernelCalls — encoded via `kernel_codec` (one tagged variant per
        // OpKind, total round-trip with `decoder::decode_calls`).
        let calls_bytes = encode_kernel_calls(&self.kernel_calls);
        payloads.push((SectionKind::KernelCalls, calls_bytes));

        if let Some(sched) = &self.schedule {
            payloads.push((SectionKind::Schedule, encode_schedule(sched)));
        }
        payloads.push((SectionKind::Weights, encode_weights(&self.weights)));
        if let Some(reg) = &self.shape_registry {
            payloads.push((SectionKind::ShapeRegistry, encode_shape_registry(reg)));
        }
        if !self.certificates.is_empty() {
            payloads.push((SectionKind::Certificates, self.certificates.clone()));
        }
        if !self.trace.is_empty() {
            payloads.push((SectionKind::Trace, self.trace.clone()));
        }
        if !self.metadata.is_empty() {
            payloads.push((SectionKind::Metadata, self.metadata.clone()));
        }
        if !self.inputs.is_empty() {
            payloads.push((SectionKind::Inputs, encode_ports(&self.inputs)));
        }
        if !self.outputs.is_empty() {
            payloads.push((SectionKind::Outputs, encode_ports(&self.outputs)));
        }
        if !self.constants.is_empty() {
            payloads.push((
                SectionKind::Constants,
                crate::constant_codec::encode(&self.constants),
            ));
        }
        if !self.exec_plan.is_empty() {
            payloads.push((SectionKind::ExecPlan, encode_exec_plan(&self.exec_plan)));
        }
        // Open producer metadata: one Extension section per key.
        for (key, bytes) in &self.extensions {
            payloads.push((SectionKind::Extension, encode_extension(key, bytes)));
        }
        // Embedded content blobs (a fat archive): one ContentBlob per (κ, content) — `κ71 ‖ content`.
        for (kappa, content) in &self.content_blobs {
            let mut blob = Vec::with_capacity(kappa.len() + content.len());
            blob.extend_from_slice(kappa);
            blob.extend_from_slice(content);
            payloads.push((SectionKind::ContentBlob, blob));
        }

        Ok(Self::assemble(payloads))
    }

    /// Frame a set of raw `(kind, bytes)` sections into a complete `.holo`: header || section table
    /// || payloads || 32-byte BLAKE3 footer (spec X.1). The low-level primitive the fat/thin tooling
    /// uses to re-package an archive from manipulated sections without re-encoding their content.
    #[must_use]
    pub fn assemble(payloads: Vec<(SectionKind, Vec<u8>)>) -> Vec<u8> {
        let header_size = 4 + 2 + 2 + 2; // magic + version + flags + section_count
        let section_entry_size = 1 + 7 + 8 + 8; // kind(u8) + pad(7) + offset(u64) + length(u64)
        let table_size = section_entry_size * payloads.len();
        let mut payload_offset = (header_size + table_size) as u64;
        let mut sections: Vec<SectionRef> = Vec::with_capacity(payloads.len());
        for (kind, body) in &payloads {
            sections.push(SectionRef {
                kind: *kind,
                offset: payload_offset,
                length: body.len() as u64,
            });
            payload_offset += body.len() as u64;
        }

        // Total size = header + table + payloads + footer (32 bytes).
        let mut out: Vec<u8> = Vec::with_capacity(payload_offset as usize + 32);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&(payloads.len() as u16).to_le_bytes());

        for s in &sections {
            out.push(s.kind as u8);
            out.extend_from_slice(&[0u8; 7]); // pad
            out.extend_from_slice(&s.offset.to_le_bytes());
            out.extend_from_slice(&s.length.to_le_bytes());
        }
        for (_, body) in &payloads {
            out.extend_from_slice(body);
        }

        // Footer: 32-byte content fingerprint over all preceding bytes,
        // computed through hologram's canonical `Hasher<32>` selection
        // (`prism::crypto::Blake3Hasher` per wiki ADR-031).
        use prism::vocabulary::Hasher;
        let footer: [u8; 32] = hologram_types::HologramHasher::initial()
            .fold_bytes(&out)
            .finalize();
        out.extend_from_slice(&footer);
        out
    }
}

fn encode_kernel_calls(calls: &[KernelCall]) -> Vec<u8> {
    crate::kernel_codec::encode_calls(calls)
}

/// Extension section wire format: `key_len(u16) key(utf8) bytes(..)`. The
/// matching reader is `HoloLoader::extensions` (zero-copy borrow).
fn encode_extension(key: &str, bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + key.len() + bytes.len());
    out.extend_from_slice(&(key.len() as u16).to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(bytes);
    out
}

/// Per-port wire format (FORMAT_VERSION ≥ 2):
/// `name_len(u16) name(utf8) slot(u32) element_count(u64) dtype(u8)
///  rank(u8) dims(u64 × rank)`.
fn encode_ports(ports: &[PortDescriptor]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + ports.len() * 24);
    out.extend_from_slice(&(ports.len() as u32).to_le_bytes());
    for p in ports {
        let name = p.name.as_bytes();
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&p.slot.to_le_bytes());
        out.extend_from_slice(&p.element_count.to_le_bytes());
        out.push(p.dtype);
        out.push(p.shape.len() as u8);
        for &d in &p.shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }
    out
}

/// Decode a `PortDescriptor` slice from a section payload (FORMAT_VERSION ≥ 2).
pub fn decode_ports(bytes: &[u8]) -> Result<Vec<PortDescriptor>, ArchiveError> {
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated {
            needed: 4,
            actual: bytes.len(),
        });
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut cursor = 4usize;
    let need = |cur: usize, n: usize, total: usize| -> Result<(), ArchiveError> {
        if cur + n > total {
            Err(ArchiveError::Truncated {
                needed: cur + n,
                actual: total,
            })
        } else {
            Ok(())
        }
    };
    for _ in 0..count {
        need(cursor, 2, bytes.len())?;
        let name_len = u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()) as usize;
        cursor += 2;
        need(cursor, name_len, bytes.len())?;
        let name = core::str::from_utf8(&bytes[cursor..cursor + name_len])
            .map_err(|_| ArchiveError::Io("port name is not valid UTF-8"))?
            .into();
        cursor += name_len;
        need(cursor, 13, bytes.len())?;
        let slot = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let element_count = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let dtype = bytes[cursor];
        cursor += 1;
        let rank = bytes[cursor] as usize;
        cursor += 1;
        need(cursor, rank * 8, bytes.len())?;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(u64::from_le_bytes(
                bytes[cursor..cursor + 8].try_into().unwrap(),
            ));
            cursor += 8;
        }
        out.push(PortDescriptor {
            name,
            slot,
            element_count,
            dtype,
            shape,
        });
    }
    Ok(out)
}

/// Encode per-level kernel-call indices (spec VIII.2). Same wire shape
/// as `encode_schedule` but the indices are kernel-call positions
/// (not graph NodeIds).
fn encode_exec_plan(levels: &[Vec<u32>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(levels.len() as u32).to_le_bytes());
    for level in levels {
        out.extend_from_slice(&(level.len() as u32).to_le_bytes());
        for &idx in level {
            out.extend_from_slice(&idx.to_le_bytes());
        }
    }
    out
}

/// Decode per-level kernel-call indices.
pub fn decode_exec_plan(bytes: &[u8]) -> Result<Vec<Vec<u32>>, ArchiveError> {
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated {
            needed: 4,
            actual: bytes.len(),
        });
    }
    let level_count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut levels = Vec::with_capacity(level_count);
    let mut cursor = 4usize;
    for _ in 0..level_count {
        if cursor + 4 > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed: cursor + 4,
                actual: bytes.len(),
            });
        }
        let n = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let needed = cursor + n * 4;
        if needed > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed,
                actual: bytes.len(),
            });
        }
        let mut level = Vec::with_capacity(n);
        for _ in 0..n {
            level.push(u32::from_le_bytes(
                bytes[cursor..cursor + 4].try_into().unwrap(),
            ));
            cursor += 4;
        }
        levels.push(level);
    }
    Ok(levels)
}

fn encode_schedule(sched: &Schedule) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(sched.levels.len() as u32).to_le_bytes());
    for level in &sched.levels {
        out.extend_from_slice(&(level.len() as u32).to_le_bytes());
        for hologram_graph::NodeId(id) in level {
            out.extend_from_slice(&id.to_le_bytes());
        }
    }
    out
}

fn encode_weights(weights: &WeightStore) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(weights.len() as u32).to_le_bytes());
    for (fp, bytes) in weights.entries() {
        out.extend_from_slice(&fp.0);
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Decode the `Weights` section back into a `WeightStore`. Mirrors
/// `encode_weights`. Used by the runtime to resolve content-addressed
/// constant references at session load (spec X.3).
pub fn decode_weights(bytes: &[u8]) -> Result<crate::weight::WeightStore, ArchiveError> {
    use crate::weight::{WeightFingerprint, WeightStore};
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated {
            needed: 4,
            actual: bytes.len(),
        });
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut store = WeightStore::new();
    let mut cur = 4usize;
    for _ in 0..count {
        if cur + 32 + 8 > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed: cur + 40,
                actual: bytes.len(),
            });
        }
        let mut fp = [0u8; 32];
        fp.copy_from_slice(&bytes[cur..cur + 32]);
        cur += 32;
        let len = u64::from_le_bytes(bytes[cur..cur + 8].try_into().unwrap()) as usize;
        cur += 8;
        if cur + len > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed: cur + len,
                actual: bytes.len(),
            });
        }
        let body = bytes[cur..cur + len].to_vec();
        cur += len;
        let inserted = store.insert(body);
        // Sanity: the decoder's inserted fingerprint matches the
        // archived one. Mismatches indicate archive corruption.
        if inserted != WeightFingerprint(fp) {
            return Err(ArchiveError::ChecksumMismatch);
        }
    }
    Ok(store)
}

fn encode_shape_registry(reg: &ShapeRegistry) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(reg.len() as u32).to_le_bytes());
    for i in 0..reg.len() {
        if let Some(d) = reg.get(hologram_graph::ShapeId(i as u32)) {
            out.push(d.rank);
            for k in 0..8 {
                out.extend_from_slice(&d.dims[k].to_le_bytes());
            }
            // overflow length
            let overflow = d
                .dims_overflow
                .as_ref()
                .map(|v| v.len() as u32)
                .unwrap_or(0);
            out.extend_from_slice(&overflow.to_le_bytes());
            if let Some(over) = &d.dims_overflow {
                for &x in over.iter() {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
    out
}
