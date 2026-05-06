//! Archive writer (spec X.2).

use std::io::Write;
use hologram_backend::KernelCall;
use hologram_graph::{Schedule, ShapeRegistry};
use crate::format::{MAGIC, FORMAT_VERSION, SectionKind, SectionRef};
use crate::weight::WeightStore;
use crate::error::ArchiveError;

/// Single input/output port descriptor: which workspace slot the runtime
/// fills (input) or reads (output), and how many bytes it carries.
#[derive(Debug, Clone, Copy)]
pub struct PortDescriptor {
    pub slot: u32,
    pub element_count: u32,
    pub dtype: u8,
}

#[derive(Default)]
pub struct HoloWriter {
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
}

impl HoloWriter {
    pub fn new() -> Self { Self::default() }

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

    /// Serialize the archive into an in-memory buffer.
    /// Body layout: header || section_table || section_payloads || footer.
    pub fn finish(self) -> Result<Vec<u8>, ArchiveError> {
        // Build payload sections in a stable order.
        let mut payloads: Vec<(SectionKind, Vec<u8>)> = Vec::new();

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
            payloads.push((SectionKind::Constants, crate::constant_codec::encode(&self.constants)));
        }
        if !self.exec_plan.is_empty() {
            payloads.push((SectionKind::ExecPlan, encode_exec_plan(&self.exec_plan)));
        }

        // Compute layout.
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

        // Footer: BLAKE3 over all preceding bytes.
        let footer: [u8; 32] = blake3::hash(&out).into();
        out.write_all(&footer).map_err(|_| ArchiveError::Io("footer write"))?;

        Ok(out)
    }
}

fn encode_kernel_calls(calls: &[KernelCall]) -> Vec<u8> {
    crate::kernel_codec::encode_calls(calls)
}

fn encode_ports(ports: &[PortDescriptor]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + ports.len() * 9);
    out.extend_from_slice(&(ports.len() as u32).to_le_bytes());
    for p in ports {
        out.extend_from_slice(&p.slot.to_le_bytes());
        out.extend_from_slice(&p.element_count.to_le_bytes());
        out.push(p.dtype);
    }
    out
}

/// Decode a `PortDescriptor` slice from a section payload.
pub fn decode_ports(bytes: &[u8]) -> Result<Vec<PortDescriptor>, ArchiveError> {
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated { needed: 4, actual: bytes.len() });
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut cursor = 4usize;
    for _ in 0..count {
        if cursor + 9 > bytes.len() {
            return Err(ArchiveError::Truncated { needed: cursor + 9, actual: bytes.len() });
        }
        let slot = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let element_count = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let dtype = bytes[cursor];
        cursor += 1;
        out.push(PortDescriptor { slot, element_count, dtype });
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
        return Err(ArchiveError::Truncated { needed: 4, actual: bytes.len() });
    }
    let level_count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut levels = Vec::with_capacity(level_count);
    let mut cursor = 4usize;
    for _ in 0..level_count {
        if cursor + 4 > bytes.len() {
            return Err(ArchiveError::Truncated { needed: cursor + 4, actual: bytes.len() });
        }
        let n = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let needed = cursor + n * 4;
        if needed > bytes.len() {
            return Err(ArchiveError::Truncated { needed, actual: bytes.len() });
        }
        let mut level = Vec::with_capacity(n);
        for _ in 0..n {
            level.push(u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()));
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
    use crate::weight::{WeightStore, WeightFingerprint};
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated { needed: 4, actual: bytes.len() });
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut store = WeightStore::new();
    let mut cur = 4usize;
    for _ in 0..count {
        if cur + 32 + 8 > bytes.len() {
            return Err(ArchiveError::Truncated { needed: cur + 40, actual: bytes.len() });
        }
        let mut fp = [0u8; 32];
        fp.copy_from_slice(&bytes[cur..cur + 32]);
        cur += 32;
        let len = u64::from_le_bytes(bytes[cur..cur + 8].try_into().unwrap()) as usize;
        cur += 8;
        if cur + len > bytes.len() {
            return Err(ArchiveError::Truncated { needed: cur + len, actual: bytes.len() });
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
            let overflow = d.dims_overflow.as_ref().map(|v| v.len() as u32).unwrap_or(0);
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
