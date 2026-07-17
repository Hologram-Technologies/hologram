//! # hologram-realizations
//!
//! The Hologram deployment-substrate canonical-form realizations (spec Appendix B). Each is
//! **IRI-tagged** (SPINE-2), **embeds its operand κ-labels** in a uniform layout, and exposes
//! [`references`](crate::Realization::references) as the *inverse projection*
//! recovering exactly those operands (SPINE-3). Identity is the leaf κ-label of the
//! operand-embedding canonical form (architecture §3.3 / G-A2: the witnessed-composition binding
//! is a tracked upgrade — uor-addr ships only commutative `compose_g2_product_blake3`, and the
//! ordered PrismModel lives behind the compute engine, excluded by RZ).

use crate::{address_bytes, Capabilities, KappaLabel, KappaLabel71, RealizationError, References};
use alloc::string::String;
use alloc::vec::Vec;

// ───────────────────── uniform operand-embedding layout (SPINE-2/3) ─────────────────────

const KAPPA71: usize = 71;

/// Encode a realization's canonical form: `IRI ‖ 0x00 ‖ u32(n_refs) ‖ n_refs×κ71 ‖ u32(len) ‖ payload`.
/// The embedded κ-labels are the artifact's reachability edges; the payload carries axis-typed
/// scalar/opaque content that does not itself name κ-labels.
fn encode(iri: &str, refs: &[KappaLabel71], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(iri.len() + 1 + 8 + refs.len() * KAPPA71 + payload.len());
    out.extend_from_slice(iri.as_bytes());
    out.push(0);
    out.extend_from_slice(&(refs.len() as u32).to_le_bytes());
    for r in refs {
        out.extend_from_slice(r.as_array());
    }
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Generic inverse projection (SPINE-3): validate the leading IRI, then recover exactly the
/// embedded operand κ-labels. Used by every realization's `references()`.
fn extract_refs(iri: &str, bytes: &[u8]) -> Result<References, RealizationError> {
    let nul = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or(RealizationError::Malformed)?;
    if &bytes[..nul] != iri.as_bytes() {
        return Err(RealizationError::WrongIri);
    }
    let mut cur = nul + 1;
    let n = read_u32(bytes, &mut cur)? as usize;
    let mut refs = Vec::with_capacity(n);
    for _ in 0..n {
        let end = cur
            .checked_add(KAPPA71)
            .ok_or(RealizationError::Truncated)?;
        let arr: [u8; KAPPA71] = bytes
            .get(cur..end)
            .ok_or(RealizationError::Truncated)?
            .try_into()
            .map_err(|_| RealizationError::Truncated)?;
        refs.push(KappaLabel::from_bytes(&arr).map_err(|_| RealizationError::Malformed)?);
        cur = end;
    }
    Ok(refs)
}

fn read_u32(bytes: &[u8], cur: &mut usize) -> Result<u32, RealizationError> {
    let end = cur.checked_add(4).ok_or(RealizationError::Truncated)?;
    let arr: [u8; 4] = bytes
        .get(*cur..end)
        .ok_or(RealizationError::Truncated)?
        .try_into()
        .map_err(|_| RealizationError::Truncated)?;
    *cur = end;
    Ok(u32::from_le_bytes(arr))
}

fn read_u64(bytes: &[u8], cur: &mut usize) -> Result<u64, RealizationError> {
    let end = cur.checked_add(8).ok_or(RealizationError::Truncated)?;
    let arr: [u8; 8] = bytes
        .get(*cur..end)
        .ok_or(RealizationError::Truncated)?
        .try_into()
        .map_err(|_| RealizationError::Truncated)?;
    *cur = end;
    Ok(u64::from_le_bytes(arr))
}

/// Read a length-prefixed (`u32 LE`) UTF-8 string from a realization payload. Used by the
/// [`AppManifest`] codec for per-layer entrypoints and kind-specific tags (arch / surface).
fn read_str(bytes: &[u8], cur: &mut usize) -> Result<String, RealizationError> {
    let len = read_u32(bytes, cur)? as usize;
    let end = cur.checked_add(len).ok_or(RealizationError::Truncated)?;
    let slice = bytes.get(*cur..end).ok_or(RealizationError::Truncated)?;
    *cur = end;
    String::from_utf8(slice.to_vec()).map_err(|_| RealizationError::Malformed)
}

/// The opaque payload a realization's canonical form carries after its embedded operand κ-labels
/// (the inverse of `encode`'s payload region) — used to recover a snapshot's linear-memory digest
/// or a capability set's scalar budgets.
pub fn payload_of(iri: &str, bytes: &[u8]) -> Result<alloc::vec::Vec<u8>, RealizationError> {
    let nul = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or(RealizationError::Malformed)?;
    if &bytes[..nul] != iri.as_bytes() {
        return Err(RealizationError::WrongIri);
    }
    let mut cur = nul + 1;
    let n = read_u32(bytes, &mut cur)? as usize;
    cur = cur
        .checked_add(n * KAPPA71)
        .ok_or(RealizationError::Truncated)?;
    let len = read_u32(bytes, &mut cur)? as usize;
    let end = cur.checked_add(len).ok_or(RealizationError::Truncated)?;
    Ok(bytes
        .get(cur..end)
        .ok_or(RealizationError::Truncated)?
        .to_vec())
}

/// Macro: a realization whose `references()` is the generic inverse projection over its IRI.
macro_rules! realization {
    ($ty:ty, $iri:literal) => {
        impl crate::Realization for $ty {
            const IRI: crate::RealizationId = $iri;
            fn canonicalize(&self) -> Vec<u8> {
                let (refs, payload) = self.parts();
                encode($iri, &refs, &payload)
            }
            fn references(canonical_bytes: &[u8]) -> Result<References, RealizationError> {
                extract_refs($iri, canonical_bytes)
            }
        }
        impl $ty {
            /// The realization's leaf κ-label (content address of the operand-embedding form).
            pub fn kappa(&self) -> KappaLabel71 {
                use crate::Realization;
                address_bytes(&self.canonicalize())
            }
        }
    };
}

// ───────────────────────────── the realizations (spec Appendix B) ─────────────────────────────

/// `https://hologram.foundation/realization/container-manifest` — the Container ID *is* the
/// manifest (spec §4.1). Operands: code module, initial state, instantiation parameters.
pub struct ContainerManifest {
    pub code: KappaLabel71,
    pub initial_state: KappaLabel71,
    pub parameters: KappaLabel71,
}
impl ContainerManifest {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (
            alloc::vec![self.code, self.initial_state, self.parameters],
            Vec::new(),
        )
    }
}
realization!(
    ContainerManifest,
    "https://hologram.foundation/realization/container-manifest"
);

/// `https://hologram.foundation/realization/capability-set` — a container's authority (spec §4.5).
/// Operands are the κ-labels it grants access to (storage roots, publish/subscribe channels) —
/// the reachability roots; payload carries the scalar budgets/flags. **Delegation containment**
/// (TypeInclusion/SubtypingLattice, §3.4) is a runtime/Phase-3 concern and is *not* faked here.
pub struct CapabilitySet {
    pub caps: Capabilities,
}
impl CapabilitySet {
    pub fn new(caps: Capabilities) -> Self {
        Self { caps }
    }
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let c = &self.caps;
        let mut refs = Vec::new();
        refs.extend_from_slice(&c.storage_roots);
        refs.extend_from_slice(&c.publish_channels);
        refs.extend_from_slice(&c.subscribe_channels);
        // Payload encodes the group counts (so the refs split back into the three sets) + budgets.
        let mut p = Vec::new();
        p.extend_from_slice(&(c.storage_roots.len() as u32).to_le_bytes());
        p.extend_from_slice(&(c.publish_channels.len() as u32).to_le_bytes());
        p.extend_from_slice(&(c.subscribe_channels.len() as u32).to_le_bytes());
        p.extend_from_slice(&c.storage_quota_bytes.to_le_bytes());
        p.extend_from_slice(&c.memory_max_bytes.to_le_bytes());
        p.extend_from_slice(&c.cpu_time_per_event_ms.to_le_bytes());
        p.extend_from_slice(&c.priority_weight.to_le_bytes());
        p.push((c.network_fetch as u8) | ((c.network_announce as u8) << 1));
        (refs, p)
    }
    /// Decode a capability-set canonical form back to its [`Capabilities`] view — the inverse of
    /// `canonicalize`. The runtime decodes the spawn'd caps κ-label to enforce `admits` (CR).
    pub fn to_capabilities(bytes: &[u8]) -> Result<Capabilities, RealizationError> {
        let refs = <Self as crate::Realization>::references(bytes)?;
        let payload = payload_of(
            "https://hologram.foundation/realization/capability-set",
            bytes,
        )?;
        let mut cur = 0usize;
        let ns = read_u32(&payload, &mut cur)? as usize;
        let np = read_u32(&payload, &mut cur)? as usize;
        let nsub = read_u32(&payload, &mut cur)? as usize;
        if ns + np + nsub != refs.len() {
            return Err(RealizationError::Malformed);
        }
        let storage_quota_bytes = read_u64(&payload, &mut cur)?;
        let memory_max_bytes = read_u64(&payload, &mut cur)?;
        let cpu_time_per_event_ms = read_u64(&payload, &mut cur)?;
        let priority_weight = read_u32(&payload, &mut cur)?;
        let flags = *payload.get(cur).ok_or(RealizationError::Truncated)?;
        Ok(Capabilities {
            storage_roots: refs[..ns].to_vec(),
            publish_channels: refs[ns..ns + np].to_vec(),
            subscribe_channels: refs[ns + np..].to_vec(),
            storage_quota_bytes,
            memory_max_bytes,
            cpu_time_per_event_ms,
            priority_weight,
            network_fetch: flags & 1 != 0,
            network_announce: flags & 2 != 0,
        })
    }
}
realization!(
    CapabilitySet,
    "https://hologram.foundation/realization/capability-set"
);

/// `https://hologram.foundation/realization/snapshot` — suspended container state (spec §4.7,
/// arch §11.6). Operands: the Container ID and the prior snapshot (the suspend/resume chain).
/// Payload: the **storage-quota ledger** (`storage_used`, 8 bytes LE) followed by the opaque
/// linear-memory / globals / cursor digest. Carrying `storage_used` here is what keeps the
/// container's storage quota honest across suspend/resume (arch §11.6): the κ binds the ledger.
pub struct Snapshot {
    pub container_id: KappaLabel71,
    pub previous: Option<KappaLabel71>,
    pub storage_used: u64,
    pub state_payload: Vec<u8>,
}
impl Snapshot {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = alloc::vec![self.container_id];
        if let Some(p) = self.previous {
            refs.push(p);
        }
        let mut payload = Vec::with_capacity(8 + self.state_payload.len());
        payload.extend_from_slice(&self.storage_used.to_le_bytes());
        payload.extend_from_slice(&self.state_payload);
        (refs, payload)
    }
}

impl Snapshot {
    /// Parse a Snapshot's payload (as returned by [`payload_of`]) into `(storage_used, mem_bytes)`.
    /// Returns `RealizationError::Truncated` if the payload is shorter than the 8-byte ledger.
    pub fn parse_payload(bytes: &[u8]) -> Result<(u64, &[u8]), RealizationError> {
        if bytes.len() < 8 {
            return Err(RealizationError::Truncated);
        }
        let arr: [u8; 8] = bytes[..8].try_into().unwrap();
        Ok((u64::from_le_bytes(arr), &bytes[8..]))
    }
}
realization!(Snapshot, "https://hologram.foundation/realization/snapshot");

/// `https://hologram.foundation/realization/runtime-state` — the runtime's durable root (spec §7.2).
/// Operands: peer keyed ID, error-log root, active container IDs. Payload: scalar counters.
pub struct RuntimeState {
    pub peer_keyed_id: KappaLabel71,
    pub error_log_root: KappaLabel71,
    pub active_containers: Vec<KappaLabel71>,
    pub counters_payload: Vec<u8>,
}
impl RuntimeState {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = alloc::vec![self.peer_keyed_id, self.error_log_root];
        refs.extend_from_slice(&self.active_containers);
        (refs, self.counters_payload.clone())
    }
}
realization!(
    RuntimeState,
    "https://hologram.foundation/realization/runtime-state"
);

/// `https://hologram.foundation/realization/error-event` — an error occurrence (spec §7.5).
/// Operands: the source Container ID, the predecessor error-event (the append-only log chain),
/// and an optional payload-context κ. Payload: classification + code.
pub struct ErrorEvent {
    pub source: KappaLabel71,
    pub predecessor: Option<KappaLabel71>,
    pub context: Option<KappaLabel71>,
    /// `[classification:u8][code:u32]`.
    pub class_code_payload: Vec<u8>,
}
impl ErrorEvent {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = alloc::vec![self.source];
        if let Some(p) = self.predecessor {
            refs.push(p);
        }
        if let Some(c) = self.context {
            refs.push(c);
        }
        (refs, self.class_code_payload.clone())
    }
}
realization!(
    ErrorEvent,
    "https://hologram.foundation/realization/error-event"
);

/// `https://hologram.foundation/realization/channel` — a channel declaration (spec §4.4).
/// Operand: the payload type-shape κ (if any). Payload: name + retention + peer-scope.
pub struct Channel {
    pub type_shape: Option<KappaLabel71>,
    pub decl_payload: Vec<u8>,
}
impl Channel {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let refs = self.type_shape.map(|t| alloc::vec![t]).unwrap_or_default();
        (refs, self.decl_payload.clone())
    }
}
realization!(Channel, "https://hologram.foundation/realization/channel");

/// `https://hologram.foundation/realization/route` — the keyed-ID / data-bus primitive (spec §4.4):
/// publishing a payload κ to a channel κ is structurally adding a Route whose `endpoint` is the
/// channel and whose `target` is the published κ. Channels are append-only by construction — a
/// channel's message history is the set of Routes with that endpoint. Operands: [endpoint, target].
pub struct Route {
    pub endpoint: KappaLabel71,
    pub target: KappaLabel71,
}
impl Route {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (alloc::vec![self.endpoint, self.target], Vec::new())
    }
}
realization!(Route, "https://hologram.foundation/realization/route");

/// `https://hologram.foundation/realization/peer-endpoint` — a **peer's transport address**,
/// content-addressed (architecture §11.1 / NW-tcp). A peer's identity κ on the network is the
/// κ of this realization; the `transport_payload` carries the wire form `proto:u8 | port:u16
/// LE | host_bytes` so a receiver can dial back without needing PeerIds or Multiaddrs (the
/// non-uor-native naming surfaces the substrate replaced libp2p with). No operands — the
/// endpoint is a leaf identity.
///
/// **Wire format**: `proto:u8 | port:u16 LE | host_bytes`. Proto byte distinguishes:
///   - `0` = TCPv4, host is 4 bytes (total 7 bytes)
///   - `1` = TCPv6, host is 16 bytes (total 19 bytes)
///
/// Both protos are first-class — neither silently falls back to the other (SPINE-6).
pub struct PeerEndpoint {
    /// `proto:u8 | port:u16 LE | host_bytes (4 or 16 depending on proto)`.
    pub transport_payload: Vec<u8>,
}
impl PeerEndpoint {
    /// Wire byte for TCPv4.
    pub const PROTO_TCP4: u8 = 0;
    /// Wire byte for TCPv6.
    pub const PROTO_TCP6: u8 = 1;

    /// Build a TCPv4 endpoint payload from `(host, port)`.
    pub fn tcp4(host: [u8; 4], port: u16) -> Self {
        let mut payload = Vec::with_capacity(1 + 2 + 4);
        payload.push(Self::PROTO_TCP4);
        payload.extend_from_slice(&port.to_le_bytes());
        payload.extend_from_slice(&host);
        Self {
            transport_payload: payload,
        }
    }
    /// Build a TCPv6 endpoint payload from `(host, port)`.
    pub fn tcp6(host: [u8; 16], port: u16) -> Self {
        let mut payload = Vec::with_capacity(1 + 2 + 16);
        payload.push(Self::PROTO_TCP6);
        payload.extend_from_slice(&port.to_le_bytes());
        payload.extend_from_slice(&host);
        Self {
            transport_payload: payload,
        }
    }
    /// Parse a TCPv4 endpoint back to `(host, port)`. Returns `None` for other protos / malformed
    /// payloads — the caller is expected to dispatch on `[0]` first if it accepts both protos.
    pub fn parse_tcp4(bytes: &[u8]) -> Option<([u8; 4], u16)> {
        if bytes.len() != 1 + 2 + 4 || bytes[0] != Self::PROTO_TCP4 {
            return None;
        }
        let port = u16::from_le_bytes([bytes[1], bytes[2]]);
        let mut host = [0u8; 4];
        host.copy_from_slice(&bytes[3..7]);
        Some((host, port))
    }
    /// Parse a TCPv6 endpoint back to `(host, port)`.
    pub fn parse_tcp6(bytes: &[u8]) -> Option<([u8; 16], u16)> {
        if bytes.len() != 1 + 2 + 16 || bytes[0] != Self::PROTO_TCP6 {
            return None;
        }
        let port = u16::from_le_bytes([bytes[1], bytes[2]]);
        let mut host = [0u8; 16];
        host.copy_from_slice(&bytes[3..19]);
        Some((host, port))
    }
    /// Total on-the-wire payload size given the proto byte (`proto:u8 | port:u16 | host`).
    /// Returns `None` for unknown protos.
    pub fn payload_size_for_proto(proto: u8) -> Option<usize> {
        match proto {
            Self::PROTO_TCP4 => Some(1 + 2 + 4),
            Self::PROTO_TCP6 => Some(1 + 2 + 16),
            _ => None,
        }
    }
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (Vec::new(), self.transport_payload.clone())
    }
}
realization!(
    PeerEndpoint,
    "https://hologram.foundation/realization/peer-endpoint"
);

/// `https://hologram.foundation/realization/chain-compaction` — a **chain-compaction barrier**
/// (architecture §9 G-C4 → §11). Used to bound the unbounded predecessor chains of `ErrorEvent`
/// (and any other realization that chains by `predecessor`). Operands: **none** — by design,
/// a compaction barrier *breaks* the predecessor pointer chain so the older tail becomes
/// unreachable from any pinned root and the storage backend's reachability GC reclaims it
/// (SPINE-5). Payload: `fold_count:u32 LE | boundary:KappaLabel71` — a content-bound summary
/// of the folded segment; the `boundary` κ is `address_bytes(head_κ || fold_count)` so an
/// auditor can prove "exactly K events ending at this boundary were compacted here" but
/// cannot recover the individual entries.
pub struct ChainCompaction {
    pub fold_count: u32,
    pub boundary: KappaLabel71,
}
impl ChainCompaction {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        // The boundary κ is part of the *payload* (content-bound metadata), not the operand
        // set, because reachability via `references()` must stop at the compaction barrier —
        // that's the entire point of the bound.
        let mut payload = Vec::with_capacity(4 + 71);
        payload.extend_from_slice(&self.fold_count.to_le_bytes());
        payload.extend_from_slice(self.boundary.as_array());
        (Vec::new(), payload)
    }
}
realization!(
    ChainCompaction,
    "https://hologram.foundation/realization/chain-compaction"
);

/// `https://hologram.foundation/realization/delegation` — a parent → child **capability-delegation
/// edge** in the κ-graph (arch §11.8). Minted by the runtime on `spawn_child(parent, child_caps)`
/// so the parent-child relation is *expressed in the κ-graph* rather than a side-channel map. The
/// transitive-revoke walk recovers the delegation cone by the inverse projection over Delegation
/// references: `revoke(κ_p)` ⇒ revoke every `child_caps` reachable from `parent_caps == κ_p`.
/// Operands: [parent_caps, child_caps].
pub struct Delegation {
    pub parent_caps: KappaLabel71,
    pub child_caps: KappaLabel71,
}
impl Delegation {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (alloc::vec![self.parent_caps, self.child_caps], Vec::new())
    }
}
realization!(
    Delegation,
    "https://hologram.foundation/realization/delegation"
);

// ─────────────────── bare-metal sibling realizations (arch §3.3) ───────────────────
// Each adopts the same operand-embedding canonical form as the universal realizations above,
// so reachability walks, GC, and verify-on-receipt work uniformly across substrates.

/// `https://hologram.foundation/realization/bare-metal-storage-format` — the on-disk layout
/// descriptor (bare-metal §5). Operands: the boot-config κ (the substrate's measured-boot
/// anchor). Payload: format version, sector size, block count, header A/B LBAs, pinned-list
/// head LBA, free-extent head LBA. Pinned in the on-disk header so the format is **self-
/// describing** — a fresh peer can recover the substrate by reading this realization first.
pub struct BareMetalStorageFormat {
    pub boot_config: KappaLabel71,
    /// `version:u32 LE | sector_size:u32 LE | block_count:u64 LE | header_a:u64 LE |
    ///  header_b:u64 LE | pinned_head:u64 LE | free_head:u64 LE`.
    pub layout_payload: Vec<u8>,
}
impl BareMetalStorageFormat {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (alloc::vec![self.boot_config], self.layout_payload.clone())
    }
}
realization!(
    BareMetalStorageFormat,
    "https://hologram.foundation/realization/bare-metal-storage-format"
);

/// `https://hologram.foundation/realization/runtime-state-region` — a *physical region* holding
/// a [`RuntimeState`] copy on disk (bare-metal §4.5). Operand: the RuntimeState κ that this
/// region currently materializes. Payload: `region_lba:u64 LE | region_sectors:u32 LE |
/// generation:u64 LE | reboot_epoch:u64 LE`. The pair (`generation`, `reboot_epoch`) is the
/// **reboot-monotonic ordering** that resolves G-C1: two regions are compared by `reboot_epoch`
/// first (which is persisted and incremented on each boot) and `generation` within an epoch,
/// not by UorTime (which resets across reboots).
pub struct RuntimeStateRegion {
    pub state: KappaLabel71,
    /// `region_lba | region_sectors | generation | reboot_epoch` LE.
    pub region_payload: Vec<u8>,
}
impl RuntimeStateRegion {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (alloc::vec![self.state], self.region_payload.clone())
    }
    /// Decode `(region_lba, region_sectors, generation, reboot_epoch)` from the region payload.
    pub fn decode(bytes: &[u8]) -> Result<(u64, u32, u64, u64), RealizationError> {
        let p = payload_of(
            "https://hologram.foundation/realization/runtime-state-region",
            bytes,
        )?;
        let mut cur = 0usize;
        let lba = read_u64(&p, &mut cur)?;
        let sectors = read_u32(&p, &mut cur)?;
        let gen_ = read_u64(&p, &mut cur)?;
        let epoch = read_u64(&p, &mut cur)?;
        Ok((lba, sectors, gen_, epoch))
    }
}
realization!(
    RuntimeStateRegion,
    "https://hologram.foundation/realization/runtime-state-region"
);

/// `https://hologram.foundation/realization/hardware-inventory` — the set of HAL devices a
/// bare-metal node has bound, recorded as a κ-graph node at boot (TR class §10.17). Operands:
/// one κ per bound device (each device is itself a codemodule κ — its driver). Payload:
/// `n_block:u32 LE | n_nic:u32 LE` describing how to partition the operand list.
pub struct HardwareInventory {
    /// Block-device driver κs.
    pub block_devices: Vec<KappaLabel71>,
    /// NIC driver κs.
    pub nics: Vec<KappaLabel71>,
}
impl HardwareInventory {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = Vec::with_capacity(self.block_devices.len() + self.nics.len());
        refs.extend_from_slice(&self.block_devices);
        refs.extend_from_slice(&self.nics);
        let mut p = Vec::with_capacity(8);
        p.extend_from_slice(&(self.block_devices.len() as u32).to_le_bytes());
        p.extend_from_slice(&(self.nics.len() as u32).to_le_bytes());
        (refs, p)
    }
}
realization!(
    HardwareInventory,
    "https://hologram.foundation/realization/hardware-inventory"
);

/// `https://hologram.foundation/realization/crash-record` — a captured fault (bare-metal §7.5).
/// Operand: the source RuntimeState κ at crash time; an optional predecessor crash κ chains the
/// post-mortem log. Payload: `(class:u8, code:u32, reboot_epoch:u64, message_bytes)`.
pub struct CrashRecord {
    pub source_state: KappaLabel71,
    pub predecessor: Option<KappaLabel71>,
    pub crash_payload: Vec<u8>,
}
impl CrashRecord {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = alloc::vec![self.source_state];
        if let Some(p) = self.predecessor {
            refs.push(p);
        }
        (refs, self.crash_payload.clone())
    }
}
realization!(
    CrashRecord,
    "https://hologram.foundation/realization/crash-record"
);

/// `https://hologram.foundation/realization/diagnostic-lba-record` — one diagnostic-log line
/// written to the dedicated diagnostic LBA region (bare-metal §7.6). Operand: the source
/// RuntimeState κ that emitted it; optional predecessor κ chains the diagnostic stream.
/// Payload: `severity:u8 | code:u32 | reboot_epoch:u64 | message`. Append-only by SPINE-5.
pub struct DiagnosticLbaRecord {
    pub source_state: KappaLabel71,
    pub predecessor: Option<KappaLabel71>,
    pub diag_payload: Vec<u8>,
}
impl DiagnosticLbaRecord {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = alloc::vec![self.source_state];
        if let Some(p) = self.predecessor {
            refs.push(p);
        }
        (refs, self.diag_payload.clone())
    }
}
realization!(
    DiagnosticLbaRecord,
    "https://hologram.foundation/realization/diagnostic-lba-record"
);

/// `https://hologram.foundation/realization/intent-log-record` — an intent-log entry used
/// during crash recovery (bare-metal §5.5). Operands: the affected resource κs (typically the
/// content κ being put / pinned). Payload: `opcode:u8 | params...`. The intent log makes the
/// crash recovery O(1) — the recover path replays only un-committed intents.
pub struct IntentLogRecord {
    pub affected: Vec<KappaLabel71>,
    pub intent_payload: Vec<u8>,
}
impl IntentLogRecord {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (self.affected.clone(), self.intent_payload.clone())
    }
}
realization!(
    IntentLogRecord,
    "https://hologram.foundation/realization/intent-log-record"
);

/// `https://hologram.foundation/realization/gc-mark-state` — a checkpoint of an in-progress
/// reachability-GC mark phase (bare-metal §5.5). Operands: the frontier κs still to walk.
/// Payload: `phase:u8 | marked_count:u64 | evicted_count:u64` (counters). The realization
/// makes long-running GC resumable across reboots — pick up the mark phase from this κ.
pub struct GcMarkState {
    pub frontier: Vec<KappaLabel71>,
    pub gc_payload: Vec<u8>,
}
impl GcMarkState {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (self.frontier.clone(), self.gc_payload.clone())
    }
}
realization!(
    GcMarkState,
    "https://hologram.foundation/realization/gc-mark-state"
);

/// `https://hologram.foundation/realization/hardware-abstraction-traits` — a κ-addressed
/// declaration of the HAL trait surface a bare-metal substrate honors (BlockDevice,
/// NetworkInterface). Operands: codemodule κs for each trait's reference implementation (so a
/// fresh peer can locate, verify, and load the reference adapters by κ alone). Payload:
/// `n_trait_impls:u32 | trait_names...` (length-prefixed names).
pub struct HardwareAbstractionTraits {
    pub trait_impls: Vec<KappaLabel71>,
    pub traits_payload: Vec<u8>,
}
impl HardwareAbstractionTraits {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        (self.trait_impls.clone(), self.traits_payload.clone())
    }
}
realization!(
    HardwareAbstractionTraits,
    "https://hologram.foundation/realization/hardware-abstraction-traits"
);

/// `https://hologram.foundation/realization/boot-config` — the **measured-boot** anchor of a
/// bare-metal node (arch §12.6 generalized). Operands: the block-device driver κ, the NIC
/// driver κ, the hardware-abstraction-traits κ, and an optional initial RuntimeState κ.
/// Payload: `policy_bits:u32 LE | initial_reboot_epoch:u64 LE`. The container substrate verifies
/// every driver κ recorded here against the σ-axis at bring-up; a tampered binary fails to boot.
pub struct BootConfig {
    pub block_driver: KappaLabel71,
    pub nic_driver: KappaLabel71,
    pub hal_traits: KappaLabel71,
    pub initial_state: Option<KappaLabel71>,
    /// `policy_bits:u32 LE | initial_reboot_epoch:u64 LE`.
    pub boot_payload: Vec<u8>,
}
impl BootConfig {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = alloc::vec![self.block_driver, self.nic_driver, self.hal_traits];
        if let Some(s) = self.initial_state {
            refs.push(s);
        }
        (refs, self.boot_payload.clone())
    }
}
realization!(
    BootConfig,
    "https://hologram.foundation/realization/boot-config"
);

// ─────────────────── .holo v3 application container (P4, spec 03) ───────────────────
// One format: an application is a manifest naming an ordered list of κ-referenced layers plus
// the child apps it composes (D9). A tensor-only archive is the degenerate single-layer case.

/// A `.holo` v3 layer's kind (spec 03 §v3 structure) — a **closed** enum, extended only by a
/// format-version bump (exhaustive matching, no catch-all). The kind alone fixes whether a layer
/// bears an exit code: an application is "a binary with an exit code", so only the code-bearing
/// kinds may serve as a manifest's `primary` (an app's exit code cannot be undefined).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum LayerKind {
    /// A compiled wasm code module, booted via the engine seam; exit-code bearing (`_start`).
    WasmCodemodule = 0,
    /// A compiled tensor plan, run as an `InferenceSession`; no exit code. The degenerate
    /// single-layer archive (a v2-style compiled graph) is exactly one of these.
    TensorPlan = 1,
    /// A rootfs image, booted in the emulator + κ-disk; exit-code bearing; ISA fixed at provision
    /// (its `aux` tag carries the mandatory `arch`).
    RootfsImage = 2,
    /// A UI view (D10), attached when its surface is ready; no exit code. Its `aux` tag carries the
    /// `surface` (`portable`, `native(ios)`, …).
    View = 3,
}

impl LayerKind {
    /// Whether this kind produces an exit code — a manifest's `primary` MUST be such a layer
    /// (spec 03 §Encoding decisions).
    pub fn has_exit_semantics(self) -> bool {
        matches!(self, LayerKind::WasmCodemodule | LayerKind::RootfsImage)
    }
    fn from_u8(b: u8) -> Result<Self, RealizationError> {
        match b {
            0 => Ok(LayerKind::WasmCodemodule),
            1 => Ok(LayerKind::TensorPlan),
            2 => Ok(LayerKind::RootfsImage),
            3 => Ok(LayerKind::View),
            _ => Err(RealizationError::Malformed),
        }
    }
}

/// One layer of a `.holo` v3 application: a κ-referenced payload plus its boot descriptor. The
/// `entry` is the layer's entrypoint (like `main`); `aux` is the kind-specific tag — the **arch**
/// for a rootfs-image (mandatory, ISA fixed at provision) or the **surface** for a view, empty for
/// the portable code/tensor kinds.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Layer {
    pub kind: LayerKind,
    /// The layer's payload κ (a codemodule, tensor plan, rootfs image, or view bundle). Dedup
    /// spans layers: a model shared by two layers is stored once (Law L3).
    pub content: KappaLabel71,
    /// Entrypoint name (e.g. `_start`, a session id, `boot`).
    pub entry: String,
    /// Kind-specific tag: arch for rootfs-image, surface for view, empty otherwise.
    pub aux: String,
}

impl Layer {
    /// A portable wasm code-module layer (exit-bearing).
    pub fn wasm(content: KappaLabel71, entry: impl Into<String>) -> Self {
        Self {
            kind: LayerKind::WasmCodemodule,
            content,
            entry: entry.into(),
            aux: String::new(),
        }
    }
    /// A tensor-plan layer (no exit code) — the degenerate single-layer archive is one of these.
    pub fn tensor(content: KappaLabel71, entry: impl Into<String>) -> Self {
        Self {
            kind: LayerKind::TensorPlan,
            content,
            entry: entry.into(),
            aux: String::new(),
        }
    }
    /// A rootfs-image layer for a fixed `arch` (exit-bearing; ISA fixed at provision).
    pub fn rootfs(
        content: KappaLabel71,
        entry: impl Into<String>,
        arch: impl Into<String>,
    ) -> Self {
        Self {
            kind: LayerKind::RootfsImage,
            content,
            entry: entry.into(),
            aux: arch.into(),
        }
    }
    /// A view layer for a `surface` (D10; no exit code — attached when the surface is ready).
    pub fn view(content: KappaLabel71, surface: impl Into<String>) -> Self {
        Self {
            kind: LayerKind::View,
            content,
            entry: String::new(),
            aux: surface.into(),
        }
    }
}

/// Why an [`AppManifest`] is not loadable (spec 03 §Encoding decisions — validated at load, before
/// any layer boots). Distinct from [`RealizationError`] (which is about malformed *bytes*): these
/// are well-formed manifests that violate the format's execution invariants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ManifestError {
    /// A manifest has no layers (an application has at least one).
    NoLayers,
    /// `primary` does not index any layer.
    PrimaryOutOfRange,
    /// `primary` indexes a layer with no exit semantics (tensor-plan or view).
    PrimaryNotExitBearing,
    /// A rootfs-image layer is missing its mandatory `arch` tag (ISA fixed at provision).
    RootfsMissingArch,
    /// A portable layer (wasm-codemodule / tensor-plan) carries an `arch`/`aux` tag it must not.
    PortableLayerHasArch,
    /// A view layer is missing its `surface` tag.
    ViewMissingSurface,
}

/// `https://hologram.foundation/realization/app-manifest` — a `.holo` v3 application (spec 03).
/// **One format**: the manifest embeds every layer κ, every child `(app κ, caps κ)`, and the
/// required-capabilities κ as operands, so [`references`](crate::Realization::references) yields the
/// whole application's reachability closure — migrating an app between peers is
/// `resolve_closure(app κ)`, the same operation as migrating any content. A tensor-only archive is
/// the degenerate single-layer case (one `TensorPlan` layer, `primary == None`, no children).
pub struct AppManifest {
    /// Index of the layer whose exit code IS the application's exit code — must be exit-bearing.
    /// `None` for a non-executable archive (a degenerate tensor-only / library artifact has no
    /// app exit code; "running" it is opening a session).
    pub primary: Option<u32>,
    /// The `CapabilitySet` κ the app needs; provision checks `granted ⊇ requires` and fails fast.
    /// The grant, not this request, is what the runtime enforces — `requires` is a declaration.
    pub requires: KappaLabel71,
    /// Ordered layers; boot order = index order.
    pub layers: Vec<Layer>,
    /// Composed child apps (D9): each `(app κ, delegated caps κ)`. The delegated set must be a
    /// subset of the parent's effective set (attenuation only — enforced at `spawn_child`, not
    /// here; amplification is unrepresentable).
    pub children: Vec<(KappaLabel71, KappaLabel71)>,
}

impl AppManifest {
    /// Sentinel encoding `primary == None` in the canonical payload (a manifest with 2³²−1 layers
    /// is not representable, so the top u32 is free as a "no primary" marker).
    const NO_PRIMARY: u32 = u32::MAX;

    /// The degenerate single-layer archive (spec 03 §Degenerate case): one tensor-plan layer, no
    /// primary, no children — what the compiler emits for a compile-only tensor graph by default.
    pub fn single_tensor_plan(
        content: KappaLabel71,
        entry: impl Into<String>,
        requires: KappaLabel71,
    ) -> Self {
        Self {
            primary: None,
            requires,
            layers: alloc::vec![Layer::tensor(content, entry)],
            children: Vec::new(),
        }
    }

    /// Validate the manifest's execution invariants (spec 03 §Encoding decisions). Called at load,
    /// before any layer boots — see [`ManifestError`] for the rejection reasons.
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.layers.is_empty() {
            return Err(ManifestError::NoLayers);
        }
        if let Some(i) = self.primary {
            let layer = self
                .layers
                .get(i as usize)
                .ok_or(ManifestError::PrimaryOutOfRange)?;
            if !layer.kind.has_exit_semantics() {
                return Err(ManifestError::PrimaryNotExitBearing);
            }
        }
        for layer in &self.layers {
            match layer.kind {
                LayerKind::RootfsImage => {
                    if layer.aux.is_empty() {
                        return Err(ManifestError::RootfsMissingArch);
                    }
                }
                LayerKind::View => {
                    if layer.aux.is_empty() {
                        return Err(ManifestError::ViewMissingSurface);
                    }
                }
                LayerKind::WasmCodemodule | LayerKind::TensorPlan => {
                    if !layer.aux.is_empty() {
                        return Err(ManifestError::PortableLayerHasArch);
                    }
                }
            }
        }
        Ok(())
    }

    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = Vec::with_capacity(1 + self.layers.len() + 2 * self.children.len());
        refs.push(self.requires);
        for l in &self.layers {
            refs.push(l.content);
        }
        for (app, caps) in &self.children {
            refs.push(*app);
            refs.push(*caps);
        }
        let mut p = Vec::new();
        p.extend_from_slice(&self.primary.unwrap_or(Self::NO_PRIMARY).to_le_bytes());
        p.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for l in &self.layers {
            p.push(l.kind as u8);
            p.extend_from_slice(&(l.entry.len() as u32).to_le_bytes());
            p.extend_from_slice(l.entry.as_bytes());
            p.extend_from_slice(&(l.aux.len() as u32).to_le_bytes());
            p.extend_from_slice(l.aux.as_bytes());
        }
        p.extend_from_slice(&(self.children.len() as u32).to_le_bytes());
        (refs, p)
    }

    /// Decode a canonical app-manifest form back into its structured view — the inverse of
    /// `canonicalize`. Recovers `primary`, the ordered layers (kinds + entrypoints + tags, with
    /// each layer's κ re-bound from the operand list), and the composed children.
    pub fn decode(bytes: &[u8]) -> Result<AppManifest, RealizationError> {
        let refs = <Self as crate::Realization>::references(bytes)?;
        let payload = payload_of(
            "https://hologram.foundation/realization/app-manifest",
            bytes,
        )?;
        let mut cur = 0usize;
        let primary_raw = read_u32(&payload, &mut cur)?;
        let primary = (primary_raw != Self::NO_PRIMARY).then_some(primary_raw);
        let n_layers = read_u32(&payload, &mut cur)? as usize;
        // refs = [requires, layer κ × n_layers, (app κ, caps κ) × n_children].
        let requires = *refs.first().ok_or(RealizationError::Malformed)?;
        let layer_refs = refs
            .get(1..1 + n_layers)
            .ok_or(RealizationError::Malformed)?;
        let mut layers = Vec::with_capacity(n_layers);
        for content in layer_refs {
            let kind = LayerKind::from_u8(*payload.get(cur).ok_or(RealizationError::Truncated)?)?;
            cur += 1;
            let entry = read_str(&payload, &mut cur)?;
            let aux = read_str(&payload, &mut cur)?;
            layers.push(Layer {
                kind,
                content: *content,
                entry,
                aux,
            });
        }
        let n_children = read_u32(&payload, &mut cur)? as usize;
        let child_refs = refs
            .get(1 + n_layers..)
            .ok_or(RealizationError::Malformed)?;
        if child_refs.len() != 2 * n_children {
            return Err(RealizationError::Malformed);
        }
        let children = child_refs
            .chunks_exact(2)
            .map(|c| (c[0], c[1]))
            .collect::<Vec<_>>();
        Ok(AppManifest {
            primary,
            requires,
            layers,
            children,
        })
    }
}
realization!(
    AppManifest,
    "https://hologram.foundation/realization/app-manifest"
);

// ─────────────────── networks (P5, spec 04) ───────────────────
// A Network is the VPC analogue — itself a κ-addressed realization. It is created by *publishing*
// this realization (no server, no RPC); peers resolve and apply it. Tiers gate capability at the
// protocol boundary.

/// A network's tier (spec 04 §Tiers) — the capability gate applied at the **protocol boundary**. A
/// closed enum (exhaustive matching, no catch-all).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NetworkTier {
    /// Open policy — anyone may fetch/announce/discover.
    Public = 0,
    /// Capability-gated — fetch/announce/discover require a capability proof derived from
    /// membership; non-members are refused at the protocol boundary.
    Restricted = 1,
    /// Restricted **plus** payload encryption (ships in P6). The word "private" is reserved until
    /// encryption exists — capability-wise it gates exactly like `Restricted`.
    Private = 2,
}

impl NetworkTier {
    fn from_u8(b: u8) -> Result<Self, RealizationError> {
        match b {
            0 => Ok(NetworkTier::Public),
            1 => Ok(NetworkTier::Restricted),
            2 => Ok(NetworkTier::Private),
            _ => Err(RealizationError::Malformed),
        }
    }

    /// The **protocol-boundary** capability gate (spec 04 §Tiers / NW-2): whether a peer may
    /// perform `op`, decided *only* from the tier and whether the peer is a member — never from the
    /// payload, store state, or any business data. That the gate's inputs are exactly
    /// `(tier, is_member)` is what makes the check structurally a boundary check: it cannot live in
    /// business logic because it is given none. Public admits anyone; restricted/private require
    /// membership. (Private additionally encrypts the payload — a P6 concern, not this capability gate.)
    #[must_use]
    pub fn admits(self, _op: NetworkOp, peer_is_member: bool) -> bool {
        match self {
            NetworkTier::Public => true,
            NetworkTier::Restricted | NetworkTier::Private => peer_is_member,
        }
    }
}

/// A network operation gated at the protocol boundary (spec 04): the three verbs a peer performs
/// against a network — the surface [`NetworkTier::admits`] arbitrates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NetworkOp {
    /// Publish content into the network's shared store.
    Store,
    /// Resolve content by κ from the network.
    Fetch,
    /// Advertise possession of a κ to the network.
    Announce,
}

/// `https://hologram.foundation/realization/network` — a network (the VPC analogue, spec 04),
/// itself κ-addressed. Operands: the **membership** set (operator / peer-endpoint κs), the
/// **policy** CapabilitySet κ (admission + fetch/announce/discover rights + quotas), and an
/// optional **parent-network** κ. Payload: the membership count (so the operands split back) + the
/// tier byte. `references()` recovers exactly those operand κs — no side tables (NW-1).
///
/// **Nesting is reserved, not implemented** (spec 04 §Nesting): `parent` may be carried, but P5 is
/// flat networks only — subnet policy-composition semantics get their own design.
pub struct Network {
    /// Operator / peer-endpoint κs that constitute the network's membership.
    pub membership: Vec<KappaLabel71>,
    /// The policy CapabilitySet κ (admission + fetch/announce/discover rights + quotas).
    pub policy: KappaLabel71,
    /// Optional parent-network κ — reserved for nesting; flat networks only in P5.
    pub parent: Option<KappaLabel71>,
    /// The capability tier gating the protocol boundary.
    pub tier: NetworkTier,
}

impl Network {
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = Vec::with_capacity(self.membership.len() + 2);
        refs.extend_from_slice(&self.membership);
        refs.push(self.policy);
        if let Some(p) = self.parent {
            refs.push(p);
        }
        // Payload: membership count (so `references()` splits back into membership / policy /
        // parent) + the tier byte.
        let mut p = Vec::with_capacity(5);
        p.extend_from_slice(&(self.membership.len() as u32).to_le_bytes());
        p.push(self.tier as u8);
        (refs, p)
    }

    /// Decode a canonical network form back into its structured view — the inverse of
    /// `canonicalize`. Recovers the membership set, policy κ, optional parent κ, and tier.
    pub fn decode(bytes: &[u8]) -> Result<Network, RealizationError> {
        let refs = <Self as crate::Realization>::references(bytes)?;
        let payload = payload_of("https://hologram.foundation/realization/network", bytes)?;
        let mut cur = 0usize;
        let n_membership = read_u32(&payload, &mut cur)? as usize;
        let tier = NetworkTier::from_u8(*payload.get(cur).ok_or(RealizationError::Truncated)?)?;
        // refs = [membership × n, policy, parent?].
        let membership = refs
            .get(..n_membership)
            .ok_or(RealizationError::Malformed)?
            .to_vec();
        let policy = *refs.get(n_membership).ok_or(RealizationError::Malformed)?;
        let parent = refs.get(n_membership + 1).copied();
        Ok(Network {
            membership,
            policy,
            parent,
            tier,
        })
    }
}
realization!(Network, "https://hologram.foundation/realization/network");

// ─────────────────── attestation keys (P6, spec 07 R3) ───────────────────

/// `https://hologram.foundation/realization/attestation-key` — a signing key **bound to a
/// κ-addressed identity as published content** (spec 07 R3). The key's identity IS its κ (the
/// address of this realization) — self-sovereign key material published as content, exactly like
/// Operator identity, never a second identity surface smuggled in through certificates (law 2
/// applies to attestation too). Operands: **none** — a key is a leaf identity (like `PeerEndpoint`).
/// Payload: `scheme:u8 ‖ public_key_bytes`, so the key is self-describing.
///
/// **Rotation** publishes a *new* key as new content (a new κ); old attestations stay verifiable
/// against the key κ that made them. **Revocation** is an append-only event, never deletion — you
/// cannot remove a key from a κ-store, only publish its revocation and require verifiers to check
/// the chain.
pub struct AttestationKey {
    /// Signature-scheme id (e.g. ed25519), kept in the payload so the key is self-describing.
    pub scheme: u8,
    /// The public key material — published as content; its κ is the key's one identity.
    pub public_key: Vec<u8>,
}

impl AttestationKey {
    pub fn new(scheme: u8, public_key: Vec<u8>) -> Self {
        Self { scheme, public_key }
    }
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut payload = Vec::with_capacity(1 + self.public_key.len());
        payload.push(self.scheme);
        payload.extend_from_slice(&self.public_key);
        (Vec::new(), payload) // no operands — a key is a leaf identity
    }
    /// Recover `(scheme, public_key)` from a canonical attestation-key form.
    pub fn decode(bytes: &[u8]) -> Result<(u8, Vec<u8>), RealizationError> {
        let payload = payload_of(
            "https://hologram.foundation/realization/attestation-key",
            bytes,
        )?;
        let scheme = *payload.first().ok_or(RealizationError::Truncated)?;
        Ok((scheme, payload[1..].to_vec()))
    }
}
realization!(
    AttestationKey,
    "https://hologram.foundation/realization/attestation-key"
);

// ─────────────────── audit trail (P6, spec 07 R2) ───────────────────

/// A container lifecycle transition (spec 07 R2) — the events the audit seam records. A closed
/// enum (exhaustive matching, no catch-all): `boot` spawns; `suspend`/`resume`/`terminate` follow.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum LifecycleTransition {
    /// `boot` — the container spawned and is running.
    Spawn = 0,
    /// `suspend` — captured to a κ snapshot.
    Suspend = 1,
    /// `resume` — restarted from a snapshot.
    Resume = 2,
    /// `terminate` — ended, not resumable.
    Terminate = 3,
}

impl LifecycleTransition {
    fn from_u8(b: u8) -> Result<Self, RealizationError> {
        match b {
            0 => Ok(LifecycleTransition::Spawn),
            1 => Ok(LifecycleTransition::Suspend),
            2 => Ok(LifecycleTransition::Resume),
            3 => Ok(LifecycleTransition::Terminate),
            _ => Err(RealizationError::Malformed),
        }
    }
}

/// `https://hologram.foundation/realization/audit-event` — one lifecycle-transition record in the
/// append-only **audit κ-chain** (spec 07 R2). Operands: the subject container κ + the predecessor
/// audit-event κ (the chain link; absent at the head). Payload: the transition byte. This is the
/// **one seam** every lifecycle transition emits through — SPINE-5 makes the chain tamper-evident
/// for free, and because it is κ-content the audit trail needs no separate access-control mechanism.
pub struct AuditEvent {
    /// The container whose lifecycle this records.
    pub subject: KappaLabel71,
    /// The prior audit-event κ (the append-only chain link); `None` at the chain head.
    pub predecessor: Option<KappaLabel71>,
    /// Which transition occurred.
    pub transition: LifecycleTransition,
}

impl AuditEvent {
    /// The audit seam: mint the event recording `transition` on `subject`, linked to the prior
    /// audit-event κ (`predecessor`). Every lifecycle transition passes through here — no bypass.
    #[must_use]
    pub fn record(
        transition: LifecycleTransition,
        subject: KappaLabel71,
        predecessor: Option<KappaLabel71>,
    ) -> Self {
        Self {
            subject,
            predecessor,
            transition,
        }
    }
    fn parts(&self) -> (Vec<KappaLabel71>, Vec<u8>) {
        let mut refs = alloc::vec![self.subject];
        if let Some(p) = self.predecessor {
            refs.push(p);
        }
        (refs, alloc::vec![self.transition as u8])
    }
    /// Recover the recorded transition from a canonical audit-event form.
    pub fn transition_of(bytes: &[u8]) -> Result<LifecycleTransition, RealizationError> {
        let p = payload_of("https://hologram.foundation/realization/audit-event", bytes)?;
        LifecycleTransition::from_u8(*p.first().ok_or(RealizationError::Truncated)?)
    }
}
realization!(
    AuditEvent,
    "https://hologram.foundation/realization/audit-event"
);

// ───────────────────────────── registry (G-D4) ─────────────────────────────

use crate::{Realization, RealizationId, RefExtractor};

/// The IRI → reference-extractor table the storage backend borrows for reachability walks
/// (spec §5.3). One row per realization above. Static (no_std-friendly).
pub static REGISTRY: &[(RealizationId, RefExtractor)] = &[
    (
        ContainerManifest::IRI,
        <ContainerManifest as Realization>::references,
    ),
    (
        CapabilitySet::IRI,
        <CapabilitySet as Realization>::references,
    ),
    (Snapshot::IRI, <Snapshot as Realization>::references),
    (RuntimeState::IRI, <RuntimeState as Realization>::references),
    (ErrorEvent::IRI, <ErrorEvent as Realization>::references),
    (Channel::IRI, <Channel as Realization>::references),
    (Route::IRI, <Route as Realization>::references),
    (Delegation::IRI, <Delegation as Realization>::references),
    (
        ChainCompaction::IRI,
        <ChainCompaction as Realization>::references,
    ),
    (PeerEndpoint::IRI, <PeerEndpoint as Realization>::references),
    // Bare-metal sibling realizations (arch §3.3, D1).
    (
        BareMetalStorageFormat::IRI,
        <BareMetalStorageFormat as Realization>::references,
    ),
    (
        RuntimeStateRegion::IRI,
        <RuntimeStateRegion as Realization>::references,
    ),
    (
        HardwareInventory::IRI,
        <HardwareInventory as Realization>::references,
    ),
    (CrashRecord::IRI, <CrashRecord as Realization>::references),
    (
        DiagnosticLbaRecord::IRI,
        <DiagnosticLbaRecord as Realization>::references,
    ),
    (
        IntentLogRecord::IRI,
        <IntentLogRecord as Realization>::references,
    ),
    (GcMarkState::IRI, <GcMarkState as Realization>::references),
    (
        HardwareAbstractionTraits::IRI,
        <HardwareAbstractionTraits as Realization>::references,
    ),
    (BootConfig::IRI, <BootConfig as Realization>::references),
    // .holo v3 application container (P4, spec 03).
    (AppManifest::IRI, <AppManifest as Realization>::references),
    // Networks (P5, spec 04).
    (Network::IRI, <Network as Realization>::references),
    // Attestation keys (P6, spec 07 R3).
    (
        AttestationKey::IRI,
        <AttestationKey as Realization>::references,
    ),
    // Audit trail (P6, spec 07 R2).
    (AuditEvent::IRI, <AuditEvent as Realization>::references),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{references, KappaStore, Realization};

    fn k(seed: &[u8]) -> KappaLabel71 {
        address_bytes(seed)
    }

    #[test]
    fn manifest_references_are_exactly_the_embedded_operands() {
        let m = ContainerManifest {
            code: k(b"code"),
            initial_state: k(b"state"),
            parameters: k(b"params"),
        };
        let bytes = m.canonicalize();
        let refs = ContainerManifest::references(&bytes).unwrap();
        assert_eq!(refs, alloc::vec![k(b"code"), k(b"state"), k(b"params")]);
    }

    #[test]
    fn registry_dispatch_recovers_references_without_knowing_the_type() {
        let s = Snapshot {
            container_id: k(b"cid"),
            previous: Some(k(b"prev")),
            storage_used: 0,
            state_payload: alloc::vec![1, 2, 3],
        };
        let bytes = s.canonicalize();
        // The store only sees bytes + the registry; it recovers edges via the embedded IRI.
        let refs = references(&bytes, REGISTRY).unwrap();
        assert_eq!(refs, alloc::vec![k(b"cid"), k(b"prev")]);
    }

    #[test]
    fn wrong_iri_is_rejected() {
        let c = Channel {
            type_shape: None,
            decl_payload: alloc::vec![9],
        };
        let mut bytes = c.canonicalize();
        bytes[0] = b'X';
        assert!(Channel::references(&bytes).is_err());
    }

    #[test]
    fn bare_metal_storage_format_round_trips_through_registry() {
        let f = BareMetalStorageFormat {
            boot_config: k(b"boot"),
            layout_payload: alloc::vec![1, 2, 3, 4],
        };
        let bytes = f.canonicalize();
        let refs = references(&bytes, REGISTRY).unwrap();
        assert_eq!(refs, alloc::vec![k(b"boot")]);
    }

    #[test]
    fn runtime_state_region_decodes_reboot_epoch_and_generation() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&42u64.to_le_bytes()); // region_lba
        payload.extend_from_slice(&8u32.to_le_bytes()); // sectors
        payload.extend_from_slice(&17u64.to_le_bytes()); // generation
        payload.extend_from_slice(&3u64.to_le_bytes()); // reboot_epoch
        let r = RuntimeStateRegion {
            state: k(b"state"),
            region_payload: payload,
        };
        let bytes = r.canonicalize();
        let (lba, sectors, gen_, epoch) = RuntimeStateRegion::decode(&bytes).unwrap();
        assert_eq!((lba, sectors, gen_, epoch), (42, 8, 17, 3));
    }

    #[test]
    fn hardware_inventory_partitions_operands_correctly() {
        let inv = HardwareInventory {
            block_devices: alloc::vec![k(b"blk0"), k(b"blk1")],
            nics: alloc::vec![k(b"nic0")],
        };
        let bytes = inv.canonicalize();
        let refs = HardwareInventory::references(&bytes).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0], k(b"blk0"));
        assert_eq!(refs[2], k(b"nic0"));
    }

    #[test]
    fn crash_diagnostic_and_intent_chain_via_references() {
        let cr = CrashRecord {
            source_state: k(b"st"),
            predecessor: Some(k(b"prev")),
            crash_payload: alloc::vec![0xff],
        };
        assert_eq!(
            CrashRecord::references(&cr.canonicalize()).unwrap(),
            alloc::vec![k(b"st"), k(b"prev")]
        );
        let dr = DiagnosticLbaRecord {
            source_state: k(b"st"),
            predecessor: None,
            diag_payload: alloc::vec![1],
        };
        assert_eq!(
            DiagnosticLbaRecord::references(&dr.canonicalize()).unwrap(),
            alloc::vec![k(b"st")]
        );
        let il = IntentLogRecord {
            affected: alloc::vec![k(b"a"), k(b"b")],
            intent_payload: alloc::vec![0x10, 0, 0, 0],
        };
        assert_eq!(
            IntentLogRecord::references(&il.canonicalize()).unwrap(),
            alloc::vec![k(b"a"), k(b"b")]
        );
    }

    #[test]
    fn boot_config_canonicalizes_with_optional_initial_state() {
        let bc = BootConfig {
            block_driver: k(b"blk-drv"),
            nic_driver: k(b"nic-drv"),
            hal_traits: k(b"hal"),
            initial_state: Some(k(b"rs0")),
            boot_payload: alloc::vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        };
        let refs = BootConfig::references(&bc.canonicalize()).unwrap();
        assert_eq!(refs.len(), 4);
        assert_eq!(refs[0], k(b"blk-drv"));
        assert_eq!(refs[1], k(b"nic-drv"));
        assert_eq!(refs[2], k(b"hal"));
        assert_eq!(refs[3], k(b"rs0"));
    }

    #[test]
    fn registry_covers_all_nine_bare_metal_sibling_iris() {
        let nine = [
            BareMetalStorageFormat::IRI,
            RuntimeStateRegion::IRI,
            HardwareInventory::IRI,
            CrashRecord::IRI,
            DiagnosticLbaRecord::IRI,
            IntentLogRecord::IRI,
            GcMarkState::IRI,
            HardwareAbstractionTraits::IRI,
            BootConfig::IRI,
        ];
        for iri in &nine {
            assert!(
                REGISTRY.iter().any(|(reg_iri, _)| reg_iri == iri),
                "{iri} missing from REGISTRY"
            );
        }
    }

    #[test]
    fn identity_is_deterministic_and_binds_operands() {
        let a = ContainerManifest {
            code: k(b"c"),
            initial_state: k(b"s"),
            parameters: k(b"p"),
        };
        let b = ContainerManifest {
            code: k(b"c"),
            initial_state: k(b"s"),
            parameters: k(b"p"),
        };
        let c = ContainerManifest {
            code: k(b"c"),
            initial_state: k(b"DIFFERENT"),
            parameters: k(b"p"),
        };
        assert_eq!(a.kappa(), b.kappa());
        assert_ne!(a.kappa(), c.kappa());
    }

    // ── .holo v3 AppManifest (P4) ──

    fn full_app() -> AppManifest {
        AppManifest {
            primary: Some(0),
            requires: k(b"caps"),
            layers: alloc::vec![
                Layer::wasm(k(b"wasm"), "_start"),
                Layer::tensor(k(b"plan"), "sess"),
                Layer::rootfs(k(b"rootfs"), "boot", "riscv64"),
                Layer::view(k(b"view"), "portable"),
            ],
            children: alloc::vec![(k(b"child-app"), k(b"child-caps"))],
        }
    }

    #[test]
    fn app_manifest_references_are_the_whole_reachability_closure() {
        let m = full_app();
        // references() = [requires, 4×layer κ, child app κ, child caps κ] — the full closure a
        // `resolve_closure(app κ)` fetches to migrate the app.
        let refs = AppManifest::references(&m.canonicalize()).unwrap();
        assert_eq!(
            refs,
            alloc::vec![
                k(b"caps"),
                k(b"wasm"),
                k(b"plan"),
                k(b"rootfs"),
                k(b"view"),
                k(b"child-app"),
                k(b"child-caps"),
            ]
        );
    }

    #[test]
    fn app_manifest_round_trips_through_decode() {
        let m = full_app();
        let decoded = AppManifest::decode(&m.canonicalize()).unwrap();
        assert_eq!(decoded.primary, Some(0));
        assert_eq!(decoded.requires, k(b"caps"));
        assert_eq!(decoded.layers, m.layers);
        assert_eq!(
            decoded.children,
            alloc::vec![(k(b"child-app"), k(b"child-caps"))]
        );
    }

    #[test]
    fn app_manifest_dispatches_through_registry() {
        let m = full_app();
        let refs = references(&m.canonicalize(), REGISTRY).unwrap();
        assert_eq!(refs.len(), 7);
        assert!(REGISTRY.iter().any(|(iri, _)| *iri == AppManifest::IRI));
    }

    #[test]
    fn degenerate_tensor_only_archive_is_valid_with_no_primary() {
        // Spec §Degenerate case: a v2-style compiled graph is one tensor-plan layer, no exit code.
        let m = AppManifest::single_tensor_plan(k(b"graph"), "sess", k(b"caps"));
        assert_eq!(m.primary, None);
        m.validate().unwrap();
        let decoded = AppManifest::decode(&m.canonicalize()).unwrap();
        assert_eq!(decoded.primary, None);
        assert_eq!(decoded.layers.len(), 1);
        assert_eq!(decoded.layers[0].kind, LayerKind::TensorPlan);
    }

    #[test]
    fn primary_must_index_an_exit_bearing_layer() {
        // A wasm primary is fine; a tensor-plan / view primary is rejected (undefined exit code).
        assert!(full_app().validate().is_ok());
        let bad = AppManifest {
            primary: Some(1), // the tensor-plan layer
            ..full_app()
        };
        assert_eq!(bad.validate(), Err(ManifestError::PrimaryNotExitBearing));
        let oor = AppManifest {
            primary: Some(99),
            ..full_app()
        };
        assert_eq!(oor.validate(), Err(ManifestError::PrimaryOutOfRange));
    }

    #[test]
    fn rootfs_requires_arch_and_portable_kinds_reject_it() {
        let no_arch = AppManifest {
            layers: alloc::vec![
                Layer::wasm(k(b"w"), "_start"),
                Layer {
                    kind: LayerKind::RootfsImage,
                    content: k(b"r"),
                    entry: "boot".into(),
                    aux: String::new(), // missing arch
                },
            ],
            ..full_app()
        };
        assert_eq!(no_arch.validate(), Err(ManifestError::RootfsMissingArch));
        let wasm_with_arch = AppManifest {
            layers: alloc::vec![Layer {
                kind: LayerKind::WasmCodemodule,
                content: k(b"w"),
                entry: "_start".into(),
                aux: "riscv64".into(), // portable kind must not carry arch
            }],
            ..full_app()
        };
        assert_eq!(
            wasm_with_arch.validate(),
            Err(ManifestError::PortableLayerHasArch)
        );
    }

    #[test]
    fn app_identity_binds_every_operand_and_the_layer_order() {
        let base = full_app();
        // Reordering layers changes identity (boot order = manifest order is part of the app).
        let reordered = AppManifest {
            layers: alloc::vec![
                Layer::tensor(k(b"plan"), "sess"),
                Layer::wasm(k(b"wasm"), "_start"),
                Layer::rootfs(k(b"rootfs"), "boot", "riscv64"),
                Layer::view(k(b"view"), "portable"),
            ],
            primary: Some(1),
            ..full_app()
        };
        assert_ne!(base.kappa(), reordered.kappa());
        // A different child caps κ changes identity (composition is bound).
        let reattenuated = AppManifest {
            children: alloc::vec![(k(b"child-app"), k(b"OTHER-caps"))],
            ..full_app()
        };
        assert_ne!(base.kappa(), reattenuated.kappa());
    }

    // ── resolve_closure — the app-loader reachability primitive (P4.3) ──

    #[test]
    fn resolve_closure_walks_the_whole_app_from_its_kappa() {
        let store = crate::MemKappaStore::new();
        let put = |b: &[u8]| store.put("blake3", b).unwrap();
        // Opaque layer contents + caps + child edges (leaves — not realizations).
        let wasm = put(b"rc-wasm-content");
        let plan = put(b"rc-plan-content");
        let caps = put(b"rc-caps-content");
        let child_app = put(b"rc-child-app");
        let child_caps = put(b"rc-child-caps");
        let manifest = AppManifest {
            primary: Some(0),
            requires: caps,
            layers: alloc::vec![Layer::wasm(wasm, "_start"), Layer::tensor(plan, "sess")],
            children: alloc::vec![(child_app, child_caps)],
        };
        let app = put(&manifest.canonicalize());
        let closure = crate::resolve_closure(app, &store, REGISTRY).unwrap();
        // A fat store resolves the whole app locally — no missing edges.
        assert!(closure.is_complete());
        for k in [app, caps, wasm, plan, child_app, child_caps] {
            assert!(
                closure.reachable.contains(&k),
                "closure must reach every operand κ"
            );
        }
        assert_eq!(
            closure.reachable.len(),
            6,
            "exactly the app + its 5 operand edges"
        );
    }

    // ── Network realization + tier gate (P5, spec 04) ──

    #[test]
    fn network_references_are_exactly_membership_and_policy() {
        let net = Network {
            membership: alloc::vec![k(b"op-a"), k(b"op-b")],
            policy: k(b"policy"),
            parent: None,
            tier: NetworkTier::Restricted,
        };
        // NW-1: references() yields exactly [membership..., policy] — no side tables.
        let refs = Network::references(&net.canonicalize()).unwrap();
        assert_eq!(refs, alloc::vec![k(b"op-a"), k(b"op-b"), k(b"policy")]);
    }

    #[test]
    fn network_round_trips_through_decode_including_parent_and_tier() {
        let net = Network {
            membership: alloc::vec![k(b"m0")],
            policy: k(b"pol"),
            parent: Some(k(b"parent-net")),
            tier: NetworkTier::Private,
        };
        let decoded = Network::decode(&net.canonicalize()).unwrap();
        assert_eq!(decoded.membership, alloc::vec![k(b"m0")]);
        assert_eq!(decoded.policy, k(b"pol"));
        assert_eq!(decoded.parent, Some(k(b"parent-net")));
        assert_eq!(decoded.tier, NetworkTier::Private);
        // Dispatches through the registry like every other realization.
        let refs = references(&net.canonicalize(), REGISTRY).unwrap();
        assert_eq!(refs.len(), 3); // m0, pol, parent-net
    }

    #[test]
    fn tiers_gate_capability_at_the_boundary() {
        // NW-2: the gate is decided from (tier, is_member) alone — a boundary check, never business
        // logic. Public admits anyone; restricted/private require membership; for every op.
        for op in [NetworkOp::Store, NetworkOp::Fetch, NetworkOp::Announce] {
            assert!(
                NetworkTier::Public.admits(op, false),
                "public admits a non-member"
            );
            assert!(
                !NetworkTier::Restricted.admits(op, false),
                "restricted refuses a non-member"
            );
            assert!(
                NetworkTier::Restricted.admits(op, true),
                "restricted admits a member"
            );
            // Private gates capability exactly like restricted (encryption is the P6 add-on).
            assert_eq!(
                NetworkTier::Private.admits(op, false),
                NetworkTier::Restricted.admits(op, false)
            );
        }
    }

    // ── AttestationKey (P6 GV-3) + per-capability boundary (P6 GV-4) ──

    #[test]
    fn attestation_key_identity_is_its_kappa_single_surface() {
        // GV-3: the signing key is published as content; its identity IS its κ — never a second
        // identity surface. Same key content ⇒ same κ (one identity); a different key ⇒ different κ.
        let key = AttestationKey::new(0, alloc::vec![1, 2, 3, 4]);
        let bytes = key.canonicalize();
        let kappa = key.kappa();
        // The identity is verifiable content (SPINE-1): its canonical form re-derives to its κ.
        assert!(crate::verify_kappa(&bytes, &kappa).unwrap());
        // Deterministic single surface: republishing the same key yields the same κ.
        assert_eq!(
            AttestationKey::new(0, alloc::vec![1, 2, 3, 4]).kappa(),
            kappa
        );
        // A rotated key is new content with a new κ (old attestations still name the old κ).
        assert_ne!(
            AttestationKey::new(0, alloc::vec![9, 9, 9, 9]).kappa(),
            kappa
        );
        // A key is a leaf identity — no operands (references empty).
        assert!(AttestationKey::references(&bytes).unwrap().is_empty());
        assert_eq!(
            AttestationKey::decode(&bytes).unwrap(),
            (0, alloc::vec![1, 2, 3, 4])
        );
    }

    #[test]
    fn capability_admits_network_op_at_the_boundary_per_capability() {
        use crate::Capabilities;
        // GV-4: a capability policy with a per-capability quota; the check is decided from the
        // capability alone (boundary), and the budget is this capability's own (not global).
        let policy = Capabilities {
            storage_roots: alloc::vec![],
            storage_quota_bytes: 1000,
            network_fetch: true,
            network_announce: false,
            publish_channels: alloc::vec![],
            subscribe_channels: alloc::vec![],
            memory_max_bytes: 0,
            cpu_time_per_event_ms: 0,
            priority_weight: 0,
        };
        assert!(
            policy.admits_network_op(NetworkOp::Fetch, 0),
            "fetch granted"
        );
        assert!(
            !policy.admits_network_op(NetworkOp::Announce, 0),
            "announce not granted"
        );
        assert!(
            policy.admits_network_op(NetworkOp::Store, 500),
            "store within quota"
        );
        assert!(
            !policy.admits_network_op(NetworkOp::Store, 2000),
            "store over quota refused"
        );
        // Per-capability accounting: a second capability's quota is independent, not global.
        let other = Capabilities {
            storage_quota_bytes: 5000,
            ..policy.clone()
        };
        assert!(
            other.admits_network_op(NetworkOp::Store, 2000),
            "the other cap has its own budget"
        );
    }

    #[test]
    fn audit_events_form_a_kappa_chain_over_every_transition() {
        // GV-2: every lifecycle transition emits through the one seam (AuditEvent::record), threading
        // an append-only κ-chain. All four variants map through it — no path bypasses it.
        let subject = k(b"gv2-container");
        let transitions = [
            LifecycleTransition::Spawn,
            LifecycleTransition::Suspend,
            LifecycleTransition::Resume,
            LifecycleTransition::Terminate,
        ];
        let mut head: Option<KappaLabel71> = None;
        let mut chain = alloc::vec![];
        for t in transitions {
            let event = AuditEvent::record(t, subject, head);
            let bytes = event.canonicalize();
            // Pointable at the κ-chain: references() recovers the subject (+ predecessor link).
            let refs = AuditEvent::references(&bytes).unwrap();
            assert_eq!(refs[0], subject);
            if let Some(prev) = head {
                assert_eq!(refs[1], prev, "each event links to its predecessor");
            } else {
                assert_eq!(refs.len(), 1, "the head has no predecessor");
            }
            assert_eq!(AuditEvent::transition_of(&bytes).unwrap(), t);
            let kappa = event.kappa();
            chain.push(kappa);
            head = Some(kappa);
        }
        // Four transitions ⇒ four distinct linked events (total coverage, no bypass).
        assert_eq!(chain.len(), 4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(
                    chain[i], chain[j],
                    "each transition mints a distinct audit κ"
                );
            }
        }
    }

    #[test]
    fn thin_closure_reports_the_missing_layer_edge() {
        let store = crate::MemKappaStore::new();
        let put = |b: &[u8]| store.put("blake3", b).unwrap();
        let caps = put(b"rc-thin-caps");
        // The layer κ is named by the manifest but its bytes are never stored (a thin archive).
        let absent_layer = address_bytes(b"rc-absent-layer");
        let manifest = AppManifest {
            primary: None,
            requires: caps,
            layers: alloc::vec![Layer::tensor(absent_layer, "sess")],
            children: Vec::new(),
        };
        let app = put(&manifest.canonicalize());
        let closure = crate::resolve_closure(app, &store, REGISTRY).unwrap();
        assert!(
            !closure.is_complete(),
            "a thin closure has an unresolved edge"
        );
        assert_eq!(closure.missing, alloc::vec![absent_layer]);
        // The edge is still *reachable* (named in the manifest) — just not present locally.
        assert!(closure.reachable.contains(&absent_layer));
    }
}
