#![cfg_attr(not(feature = "std"), no_std)]
//! # hologram-realizations
//!
//! The Hologram deployment-substrate canonical-form realizations (spec Appendix B). Each is
//! **IRI-tagged** (SPINE-2), **embeds its operand κ-labels** in a uniform layout, and exposes
//! [`references`](hologram_substrate_core::Realization::references) as the *inverse projection*
//! recovering exactly those operands (SPINE-3). Identity is the leaf κ-label of the
//! operand-embedding canonical form (architecture §3.3 / G-A2: the witnessed-composition binding
//! is a tracked upgrade — uor-addr ships only commutative `compose_g2_product_blake3`, and the
//! ordered PrismModel lives behind the compute engine, excluded by RZ).

extern crate alloc;

use alloc::vec::Vec;
use hologram_substrate_core::{
    address_bytes, Capabilities, KappaLabel, KappaLabel71, RealizationError, References,
};

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
        impl hologram_substrate_core::Realization for $ty {
            const IRI: hologram_substrate_core::RealizationId = $iri;
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
                use hologram_substrate_core::Realization;
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
        let refs = <Self as hologram_substrate_core::Realization>::references(bytes)?;
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

// ───────────────────────────── registry (G-D4) ─────────────────────────────

use hologram_substrate_core::{Realization, RealizationId, RefExtractor};

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
];

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_substrate_core::{references, Realization};

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
}
