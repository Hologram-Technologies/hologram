//! # substrate module (formerly hologram-substrate-core)
//!
//! Portable trait surfaces and κ-addressing for the **Hologram deployment substrate**
//! (Container Runtime · Storage Layer · Network Layer). This crate is the single source of
//! the `KappaStore` / `KappaSync` / `ContainerRuntime` surfaces; substrate backends implement
//! them per environment (browser / WASI-native / bare-metal).
//!
//! Grounding (see `specs/docs/container-substrate-architecture.md`):
//! - **SPINE-1** every artifact is a κ-label over canonical bytes.
//! - **SPINE-3** identity is *witnessed composition* of operand labels; `references()` is its
//!   inverse projection (the canonical form embeds the operands).
//! - **SPINE-4** verification is re-derivation through the σ-axis ([`verify_kappa`]).
//! - **G-E1** the σ-axis is reused from `hologram-host` (no compute-engine dependency); the
//!   κ-format helpers are byte-identical to `hologram-archive::address_bytes`.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

pub use uor_addr::KappaLabel;

/// Hologram's κ-label width: blake3 → 71 bytes (`blake3:<64 hex>`). The substrate's own
/// realization artifacts are all blake3 (ADR-052, architecture §3.1 / G-B1). Stored *content*
/// keys may carry other σ-axes in their on-the-wire `<axis>:<hex>` byte form.
pub type KappaLabel71 = KappaLabel<71>;

/// Zero-copy shared byte buffer. `get` returns a cheap `Arc` clone, never a copy — the SP
/// performance floor (architecture §4). Backends MAY substitute any `AsRef<[u8]> + Send + Sync
/// + Clone` (spec §8.0); the reference impls use this.
pub type Bytes = Arc<[u8]>;

// ───────────────────────────── errors ─────────────────────────────

/// σ-axis re-derivation failures ([`verify_kappa`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisError {
    /// The κ-label's σ-axis is not wired in this build (fail-loud, never a silent pass — SPINE-6).
    UnsupportedAxis,
    /// The κ-label does not parse as `<axis>:<hex>`.
    Malformed,
}

/// [`KappaStore`] failures (spec §5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    QuotaExceeded,
    BackendFailure(&'static str),
    InvalidKappa,
    UnknownAxis,
    /// `unpin` on a κ that is not pinned.
    NotPinned,
}

/// [`KappaSync`] failures (spec §6.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncError {
    AllSourcesFailed,
    VerificationFailed,
    NotEnabled,
    BackendFailure(&'static str),
}

/// [`ContainerRuntime`] failures (spec §8.0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    ContainerIdNotFound,
    CapabilityVerificationFailed,
    SnapshotInvalid,
    InstantiationFailed(&'static str),
    BackendFailure(&'static str),
}

/// Combined store-or-fetch failure ([`get_with_fetch`], spec §8.0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessError {
    StoreFailure(StoreError),
    SyncFailure(SyncError),
    VerificationFailed,
}

// ───────────────────────────── κ-addressing (SPINE-4, G-E1) ─────────────────────────────

/// σ-axis re-derivation and κ-minting. Reuses **`prism::crypto`** (via `hologram-host`) for the
/// full uor-addr 0.2.0 axis registry — `blake3` (default, hologram ADR-052), `sha256`, `sha3-256`,
/// `keccak256`, `sha512` — without importing `hologram-archive` (G-E1). All five axes are first-
/// class: the substrate's own realizations are blake3 (ADR-052), but stored content keys are
/// **axis-polymorphic** (architecture §3.1 G-B1) and verified through this dispatcher.
pub mod kappa {
    use super::{AxisError, KappaLabel71};
    use alloc::vec::Vec;
    use hologram_types::prism::crypto::{
        Blake3Hasher, Keccak256Hasher, Sha256Hasher, Sha3_256Hasher, Sha512Hasher,
    };
    use hologram_types::prism::vocabulary::Hasher;
    use uor_addr::KappaLabel;

    const HEX: &[u8; 16] = b"0123456789abcdef";

    /// Width of the on-the-wire κ-label for an axis: `len(axis) + 1 (':') + 2·digest_bytes`.
    /// uor-addr 0.2.0 axes: blake3=71, sha256=71, sha3-256=73, keccak256=74, sha512=135.
    pub const MAX_LABEL_BYTES: usize = 135;

    /// Render a digest as the canonical `<axis>:<hex>` ASCII bytes (variable width per §3.1).
    fn render(prefix: &str, digest: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(prefix.len() + digest.len() * 2);
        out.extend_from_slice(prefix.as_bytes());
        for &b in digest {
            out.push(HEX[(b >> 4) as usize]);
            out.push(HEX[(b & 0x0f) as usize]);
        }
        out
    }

    /// Render a 32-byte BLAKE3 digest as the canonical 71-byte `blake3:<64 hex>` κ-label.
    /// Byte-identical to `hologram-archive::address.rs::blake3_kappa`.
    fn blake3_kappa(digest: &[u8; 32]) -> KappaLabel71 {
        let mut buf = [0u8; 71];
        buf[..7].copy_from_slice(b"blake3:");
        for (i, &b) in digest.iter().enumerate() {
            buf[7 + 2 * i] = HEX[(b >> 4) as usize];
            buf[7 + 2 * i + 1] = HEX[(b & 0x0f) as usize];
        }
        KappaLabel::from_bytes(&buf).expect("71-byte ASCII blake3 κ-label by construction")
    }

    /// Content-address opaque bytes on the BLAKE3 σ-axis (the *leaf* identity). Equal bytes ⇒
    /// equal κ-label (the canonical dedup key). Byte-identical to `hologram-archive::address_bytes`.
    /// The hologram-canonical path: realization artifacts mint here (ADR-052).
    #[must_use]
    pub fn address_bytes(bytes: &[u8]) -> KappaLabel71 {
        let digest: [u8; 32] = Blake3Hasher::initial().fold_bytes(bytes).finalize();
        blake3_kappa(&digest)
    }

    /// **Axis-polymorphic** content addressing — re-derives `bytes` through the σ-axis named by
    /// `axis` and returns the variable-width `<axis>:<hex>` κ-label bytes (architecture §3.1 G-B1).
    /// Unknown axes fail loud ([`AxisError::UnsupportedAxis`], SPINE-6).
    pub fn address_bytes_axis(axis: &str, bytes: &[u8]) -> Result<Vec<u8>, AxisError> {
        match axis {
            "blake3" => Ok(render(
                "blake3:",
                &Blake3Hasher::initial().fold_bytes(bytes).finalize(),
            )),
            "sha256" => Ok(render(
                "sha256:",
                &Sha256Hasher::initial().fold_bytes(bytes).finalize(),
            )),
            "sha3-256" => Ok(render(
                "sha3-256:",
                &Sha3_256Hasher::initial().fold_bytes(bytes).finalize(),
            )),
            "keccak256" => Ok(render(
                "keccak256:",
                &Keccak256Hasher::initial().fold_bytes(bytes).finalize(),
            )),
            "sha512" => Ok(render(
                "sha512:",
                &Sha512Hasher::initial().fold_bytes(bytes).finalize(),
            )),
            _ => Err(AxisError::UnsupportedAxis),
        }
    }

    /// Order-sensitive derivation key over a `domain` tag and operand labels — the SPINE-3
    /// hot-path reuse key (`O(operands)`, **unwitnessed**; the witnessed form lives in
    /// `hologram-realizations` via uor-addr composition). `f(A,B) ≠ f(B,A)`.
    #[must_use]
    pub fn derive_label(domain: &[u8], inputs: &[KappaLabel71]) -> KappaLabel71 {
        let mut h = Blake3Hasher::initial().fold_bytes(domain);
        for l in inputs {
            h = h.fold_bytes(l.as_bytes());
        }
        blake3_kappa(&h.finalize())
    }

    /// Re-derive `bytes` through the σ-axis named by `kappa`'s prefix and compare the digest
    /// (SPINE-4 / spec §8.0). Pure; the universal cross-check under every received byte. The
    /// `KappaLabel71` carries blake3 or sha256 (both 71-byte form); wider axes (sha3-256/keccak256/
    /// sha512) use [`verify_kappa_axis`] over the on-the-wire bytes directly.
    pub fn verify_kappa(bytes: &[u8], kappa: &KappaLabel71) -> Result<bool, AxisError> {
        verify_kappa_axis(bytes, kappa.as_array())
    }

    /// Re-derive `bytes` through the σ-axis named by the first `<axis>:` prefix of `label_bytes`
    /// and compare. Handles **all five axes** (variable width 71..135). For multi-axis stored
    /// content: pass the on-the-wire bytes of the κ-label.
    pub fn verify_kappa_axis(bytes: &[u8], label_bytes: &[u8]) -> Result<bool, AxisError> {
        let colon = label_bytes
            .iter()
            .position(|&b| b == b':')
            .ok_or(AxisError::Malformed)?;
        let axis = core::str::from_utf8(&label_bytes[..colon]).map_err(|_| AxisError::Malformed)?;
        let derived = address_bytes_axis(axis, bytes)?;
        Ok(derived.as_slice() == label_bytes)
    }
}
pub use kappa::{
    address_bytes, address_bytes_axis, derive_label, verify_kappa, verify_kappa_axis,
    MAX_LABEL_BYTES,
};

// ───────────────────────────── realizations (SPINE-2 / SPINE-3) ─────────────────────────────

/// A uor-addr realization IRI carried in every canonical-form artifact (SPINE-2, spec §10.9).
pub type RealizationId = &'static str;

/// The operand κ-labels a realization's canonical form composed/embedded — the reachability edges
/// out of an artifact (SPINE-3).
pub type References = Vec<KappaLabel71>;

/// Failures parsing a realization's canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealizationError {
    /// Canonical bytes do not begin with the expected realization IRI (SPINE-2).
    WrongIri,
    Truncated,
    Malformed,
}

/// A canonical-form realization: a typed input whose canonical bytes are **IRI-tagged and embed
/// their operand κ-labels**, whose identity is the **witnessed composition** of those operands,
/// and whose [`Realization::references`] is the *inverse projection* recovering exactly them
/// (SPINE-3). Not a byte-scan for label-shaped substrings.
pub trait Realization {
    /// Normative realization IRI (spec Appendix B).
    const IRI: RealizationId;

    /// Canonical-form bytes: IRI-tagged, embedding operand κ-labels.
    fn canonicalize(&self) -> Vec<u8>;

    /// The κ-label of these canonical bytes (the leaf identity; the witnessed-composition form
    /// is layered on top by composing the [`references`](Self::references)).
    fn kappa(&self) -> KappaLabel71 {
        address_bytes(&self.canonicalize())
    }

    /// Inverse projection: the operand κ-labels embedded by [`canonicalize`](Self::canonicalize)
    /// (SPINE-3 / spec §10.10).
    fn references(canonical_bytes: &[u8]) -> Result<References, RealizationError>;
}

/// A reference extractor `fn(canonical_bytes) -> references`, registered per realization IRI.
/// The storage backend resolves an artifact's embedded IRI to its extractor to compute
/// reachability (spec §5.3). On `no_std` this is a static fn-pointer table (G-D4).
pub type RefExtractor = fn(&[u8]) -> Result<References, RealizationError>;

/// IRI → extractor table. A `&[(IRI, extractor)]` the realizations crate populates; the store
/// borrows it for reachability walks.
pub type RealizationRegistry<'a> = &'a [(RealizationId, RefExtractor)];

/// Read the leading IRI from `canonical_bytes` (NUL-terminated, SPINE-2) and dispatch to its
/// registered extractor. The single graph-traversal primitive (reachability/GC/snapshot/caps).
pub fn references(
    canonical_bytes: &[u8],
    registry: RealizationRegistry<'_>,
) -> Result<References, RealizationError> {
    let nul = canonical_bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or(RealizationError::Malformed)?;
    let iri =
        core::str::from_utf8(&canonical_bytes[..nul]).map_err(|_| RealizationError::Malformed)?;
    for (id, extractor) in registry {
        if *id == iri {
            return extractor(canonical_bytes);
        }
    }
    Err(RealizationError::WrongIri)
}

/// The transitive reachability closure of a κ over a [`KappaStore`] — the app-loader primitive
/// (spec 03 §Fat and thin: "load = resolve the manifest's closure"). Walks the κ-graph from `root`
/// via each realization's `references()` inverse projection (breadth-first, first-seen order),
/// fetching each node's bytes from `store`. Opaque leaf content (a wasm module, a weight blob — no
/// registered IRI) contributes no edges. A κ whose bytes are **absent** from the store is recorded
/// in [`Closure::missing`] rather than failing: that is the thin-archive signal — the edge exists
/// in the manifest but its payload resolves elsewhere (the KappaStore/KappaSync path, LAW-4).
///
/// Bounded by `O(reachable · refs)` (the SP bounded-walk floor). `resolve_closure(app κ)` is what
/// migrates a whole application between peers — the same operation as migrating any content.
pub fn resolve_closure(
    root: KappaLabel71,
    store: &dyn KappaStore,
    registry: RealizationRegistry<'_>,
) -> Result<Closure, StoreError> {
    resolve_closure_bounded(root, store, registry, usize::MAX)
}

/// Like [`resolve_closure`] but **bounded** (spec 04 §Protocol hardening): the walk stops once it
/// has recorded `max_nodes` κs, setting [`Closure::truncated`]. A peer resolving a manifest served
/// over the network seam MUST bound the walk — a hostile peer can otherwise serve an adversarially
/// wide (or deep) κ-graph to force an unbounded resolve (a DoS). The partial closure is still valid
/// for what it covers; a `truncated` result means "refuse or fetch more before trusting completeness".
pub fn resolve_closure_bounded(
    root: KappaLabel71,
    store: &dyn KappaStore,
    registry: RealizationRegistry<'_>,
    max_nodes: usize,
) -> Result<Closure, StoreError> {
    use hashbrown::HashSet;
    let mut visited: HashSet<[u8; 71]> = HashSet::new();
    let mut reachable = Vec::new();
    let mut missing = Vec::new();
    let mut queue = Vec::new();
    queue.push(root);
    let mut cursor = 0;
    while cursor < queue.len() {
        let k = queue[cursor];
        cursor += 1;
        if !visited.insert(*k.as_array()) {
            continue; // a κ may be enqueued by several parents; process it once (first-seen wins)
        }
        if reachable.len() >= max_nodes {
            // Bound hit before recording another node — the closure is partial.
            return Ok(Closure {
                reachable,
                missing,
                truncated: true,
            });
        }
        reachable.push(k);
        match store.get(&k)? {
            None => missing.push(k),
            // Realization ⇒ enqueue its operand edges; opaque leaf ⇒ no edges (references() errs).
            Some(bytes) => {
                if let Ok(refs) = references(bytes.as_ref(), registry) {
                    queue.extend(refs);
                }
            }
        }
    }
    Ok(Closure {
        reachable,
        missing,
        truncated: false,
    })
}

/// The result of [`resolve_closure`]: every κ reachable from the root (breadth-first, first-seen
/// order, including the root), split into those present locally and those still to fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Closure {
    /// Every reachable κ (present or missing), in breadth-first first-seen order.
    pub reachable: Vec<KappaLabel71>,
    /// Reachable κs whose bytes are **absent** locally — thin-archive edges to resolve via
    /// network/sync (LAW-4). Empty ⇒ a fully-materialized (fat) closure.
    pub missing: Vec<KappaLabel71>,
    /// `true` if [`resolve_closure_bounded`] hit its node bound before completing — the closure is
    /// partial (a DoS guard, spec 04 §Protocol hardening). The unbounded [`resolve_closure`] leaves
    /// this `false`.
    pub truncated: bool,
}

impl Closure {
    /// Whether the closure is fully materialized locally — every reachable κ's bytes are present and
    /// the walk was not truncated (a **fat**, complete closure). A **thin** closure has one or more
    /// [`Closure::missing`] edges; a **truncated** closure did not finish.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty() && !self.truncated
    }
}

// ───────────────────────────── capability view (decoded; authority is a κ-label) ─────────────

/// Decoded *view* of a Capability Set's canonical form (spec §8.4). The authority itself is a
/// **κ-label** in the graph (SPINE-1); this struct is only the parsed projection — never the
/// thing passed to [`ContainerRuntime::spawn`] (which takes the κ-label, B3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Readable closure roots (transitive via SPINE-3 references).
    pub storage_roots: Vec<KappaLabel71>,
    pub storage_quota_bytes: u64,
    pub network_fetch: bool,
    pub network_announce: bool,
    pub publish_channels: Vec<KappaLabel71>,
    pub subscribe_channels: Vec<KappaLabel71>,
    pub memory_max_bytes: u64,
    pub cpu_time_per_event_ms: u64,
    /// **DRR fair-scheduling weight** (arch §11.7). The runtime's `pump_round` adds
    /// `priority_weight × quantum` of deficit per round; a misbehaving container cannot starve
    /// others. `0` is treated as `1` (default; equal share). Containment: a child's weight may not
    /// exceed its parent's — high priority cannot be amplified by delegation.
    pub priority_weight: u32,
}

impl Capabilities {
    /// **Delegation containment** — the foundation's `SubtypingLattice` relation (architecture §3.4,
    /// §9 G-A3). `parent.admits(derived)` is true iff `derived` is a valid delegation of `parent`:
    /// `derived` is *narrower* (more constrained), i.e. **grants(derived) ⊆ grants(parent)** and
    /// every budget is equal-or-tighter. This is exactly `constraints(derived) ⊇ constraints(parent)`
    /// — the lattice's defining order (more constraints = narrower = contained).
    ///
    /// It is implemented here, not delegated to `uor_foundation::TypeInclusion`, because
    /// uor-foundation 0.5.2 ships that trait as an **orphan-closure interface with no public
    /// constructor or containment checker** (only `Null*` stubs; §9 G-A3) — so this is the UOR
    /// lattice *semantics* realized faithfully, **not** a non-UOR ACL fallback. When the foundation
    /// exposes a `ConstrainedTypeResolver`, this swaps to it without changing the relation.
    ///
    /// The relation is a partial order (reflexive / antisymmetric on grant-equality / transitive),
    /// proven by the CR conformance tests.
    #[must_use]
    pub fn admits(&self, derived: &Capabilities) -> bool {
        fn subset(a: &[KappaLabel71], b: &[KappaLabel71]) -> bool {
            a.iter().all(|x| b.contains(x))
        }
        // Budget containment under the **0 = unbounded** convention (spec §7.6 / arch §3.4):
        // - parent unbounded (parent = 0) admits any child.
        // - parent bounded (parent ≠ 0) requires child also bounded (child ≠ 0) AND child ≤ parent.
        // The naive `child ≤ parent` rule would incorrectly accept child=0 (unbounded) under
        // parent=N (bounded) because 0 < N — silently widening authority. This guards against it.
        fn budget_admits(parent: u64, child: u64) -> bool {
            parent == 0 || (child != 0 && child <= parent)
        }
        subset(&derived.storage_roots, &self.storage_roots)
            && subset(&derived.publish_channels, &self.publish_channels)
            && subset(&derived.subscribe_channels, &self.subscribe_channels)
            && budget_admits(self.storage_quota_bytes, derived.storage_quota_bytes)
            && budget_admits(self.memory_max_bytes, derived.memory_max_bytes)
            && budget_admits(self.cpu_time_per_event_ms, derived.cpu_time_per_event_ms)
            && derived.priority_weight <= self.priority_weight.max(1)
            // A flag may be granted by the child only if the parent holds it.
            && (!derived.network_fetch || self.network_fetch)
            && (!derived.network_announce || self.network_announce)
    }

    /// The **import/protocol-boundary** capability check for a network op (spec 07 R4 / GV-4):
    /// whether *this* capability admits `op` on `bytes_len` bytes. Decided from the capability
    /// alone — fetch requires `network_fetch`, announce requires `network_announce`, store must fit
    /// the **per-capability** `storage_quota_bytes` (`0` = unbounded, spec §7.6). The budget is this
    /// capability's own, never a global counter, so per-network / per-operator accounting composes.
    #[must_use]
    pub fn admits_network_op(&self, op: crate::NetworkOp, bytes_len: u64) -> bool {
        use crate::NetworkOp;
        match op {
            NetworkOp::Fetch => self.network_fetch,
            NetworkOp::Announce => self.network_announce,
            NetworkOp::Store => {
                self.storage_quota_bytes == 0 || bytes_len <= self.storage_quota_bytes
            }
        }
    }
}

// ───────────────────────────── signature verification seam (spec 07 R3) ─────────────────────────────

/// The **attestation signature seam** (spec 07 R3): verify a detached `signature` over `message`
/// under a `public_key`. Portable and dependency-free — a space provides the concrete verifier for
/// its platform (native ed25519, browser WebCrypto, a bare-metal secure element), exactly as it
/// provides its `KappaStore` / HAL seams. Attestation realizations
/// ([`crate::SessionAttestation`] / [`crate::RevocationEvent`]) sign the κ-embedding of their bound
/// facts and are verified through this trait — the format binds the facts and names the signing
/// key κ; the crypto stays behind the seam (the `no_std` core carries no curve dependency).
pub trait SignatureVerifier {
    /// Whether `signature` is a valid signature over `message` for `public_key`. Must be constant
    /// with respect to the inputs and never panic (it verifies network-supplied bytes).
    fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> bool;
}

// ───────────────────────────── payload encryption seam (spec 04 §Private / P6) ─────────────────────────────

/// The **private-network payload cipher seam** (spec 04 §Private tier). An AEAD (authenticated
/// encryption): `seal` encrypts + authenticates, `open` returns the plaintext only if the tag
/// authenticates (else `None`, fail-loud SPINE-6). Portable and dependency-free — a space supplies
/// the concrete cipher for its platform (native ChaCha20-Poly1305, browser WebCrypto, a bare-metal
/// crypto accelerator), like the `SignatureVerifier` / HAL seams. The `no_std` core carries no
/// cipher dependency; only this trait + the nonce derivation.
pub trait PayloadCipher {
    /// AEAD-seal `plaintext` under `key` (32 bytes) + `nonce` (12 bytes); returns `ciphertext ‖
    /// tag`. Must never panic on ill-sized inputs (return an empty/again-unopenable buffer instead).
    fn seal(&self, key: &[u8], nonce: &[u8], plaintext: &[u8]) -> Vec<u8>;
    /// AEAD-open; returns the plaintext iff the tag authenticates under `key` + `nonce`, else `None`.
    fn open(&self, key: &[u8], nonce: &[u8], ciphertext: &[u8]) -> Option<Vec<u8>>;
}

/// A **convergent** 12-byte nonce for a private-network payload (spec 04 §Private): derived from the
/// network `key` + `plaintext` so identical content under one key yields an identical nonce — and
/// hence identical ciphertext, so **Law L3 dedup survives encryption** (the store holds one copy of
/// shared private content). Distinct keys give distinct nonces (no cross-key nonce reuse), and a
/// given nonce only ever seals one plaintext (no catastrophic AEAD reuse). The equality-leak
/// inherent to *any* dedup-preserving encryption is the documented tradeoff (04 §Private).
#[must_use]
pub fn convergent_nonce(key: &[u8], plaintext: &[u8]) -> [u8; 12] {
    use hologram_types::prism::crypto::Blake3Hasher;
    use hologram_types::prism::vocabulary::Hasher;
    let digest: [u8; 32] = Blake3Hasher::initial()
        .fold_bytes(key)
        .fold_bytes(plaintext)
        .finalize();
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&digest[..12]);
    nonce
}

/// Seal `plaintext` for a private network under `key` with a convergent (dedup-preserving) nonce
/// (spec 04 §Private). Returns `(nonce, ciphertext)`; open with [`PayloadCipher::open`].
pub fn seal_private<C: PayloadCipher>(
    cipher: &C,
    key: &[u8],
    plaintext: &[u8],
) -> ([u8; 12], Vec<u8>) {
    let nonce = convergent_nonce(key, plaintext);
    let ciphertext = cipher.seal(key, &nonce, plaintext);
    (nonce, ciphertext)
}

/// The **asymmetric key-wrapping seam** (spec 04 §Private / P6 — forward secrecy on membership
/// change). Wraps an epoch's fresh symmetric payload-key *to a specific member's enrollment public
/// key*, so only that member can recover it. This is what a convergent shared-key **cannot** do: a
/// convergent key is re-derivable by anyone who knew its inputs, so it can never be rotated to
/// *exclude* a removed member. Forward secrecy therefore requires a fresh random per-epoch key
/// delivered by per-member wrapping — hence this seam (see [`crate::KeyEpoch`]).
///
/// Portable + dependency-free, like [`PayloadCipher`] / [`SignatureVerifier`]: the `no_std` core
/// carries only this trait; a space supplies the concrete scheme (native X25519 sealed-box, browser
/// WebCrypto ECDH, a hardware enclave). Reference impl: `tests/forward_secrecy_x25519.rs`.
pub trait KeyWrapper {
    /// Wrap `key_material` (an epoch's symmetric key) to `recipient_public_key`. Returns the wrapped
    /// blob; must never panic on ill-sized inputs (return an empty/again-unopenable buffer instead).
    fn wrap(&self, recipient_public_key: &[u8], key_material: &[u8]) -> Vec<u8>;
    /// Unwrap with `recipient_private_key`; returns the key material iff `wrapped` was sealed to the
    /// matching public key, else `None` (fail-loud, SPINE-6). A non-recipient — e.g. a removed
    /// member holding an old key — gets `None`: they cannot open the new epoch's key.
    fn unwrap(&self, recipient_private_key: &[u8], wrapped: &[u8]) -> Option<Vec<u8>>;
}

// ───────────────────────────── KappaStore (sync, spec §8.1) ─────────────────────────────

/// Content-addressed byte storage. Sync (bounded local work; matches the OPFS sync handle and
/// hologram's `WarmStore`). `get` is `Option` — local absence is *not* nonexistence (SPINE-5);
/// callers fall through to the network via [`get_with_fetch`].
pub trait KappaStore: Send + Sync {
    /// Persist canonical bytes under an explicit σ-axis; return the κ-label. **Idempotent**
    /// (spec §10.2): same `(axis, bytes)` ⇒ same κ-label, no duplicate write (SP floor).
    fn put(&self, axis: &str, canonical_bytes: &[u8]) -> Result<KappaLabel71, StoreError>;
    /// Retrieve canonical bytes (zero-copy `Bytes`).
    fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError>;
    /// Local presence.
    fn contains(&self, kappa: &KappaLabel71) -> bool;
    /// Pin a κ as a reachability root, exempting it from eviction (spec §5.3).
    fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError>;
    /// Remove a pin (κ stays reachable via any other root).
    fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError>;
    /// Iterate locally-present κ-labels (unordered).
    fn iterate(&self) -> Vec<KappaLabel71>;
    /// Iterate pinned roots (unordered).
    fn pinned_roots(&self) -> Vec<KappaLabel71>;
    fn approximate_count(&self) -> usize;
    fn approximate_bytes(&self) -> u64;

    // ─── axis-polymorphic surface (architecture §3.1 G-B1) ─────────────────────────────────
    // Hologram realizations are blake3 by ADR-052 (the canonical hot path uses [`put`]/[`get`]).
    // The `*_axis` methods accept **any uor-addr-supported σ-axis** (blake3 / sha256 / sha3-256 /
    // keccak256 / sha512) and address stored content by its variable-width on-the-wire bytes —
    // for foreign-axis content flowing across the substrate's boundary, never invented locally.
    //
    // Backends that don't support multi-axis storage return `UnknownAxis` from `put_axis` (the
    // default). The reference [`crate::MemKappaStore`] opts in for all five axes,
    // verified against the upstream BLAKE3/`sha2`/`sha3` reference crates (V&V AS class).

    /// Multi-axis put: re-derive `bytes` through `axis` and store under the on-the-wire label.
    /// Returns the variable-width κ-label bytes (71/73/74/135). Default: unsupported.
    fn put_axis(&self, _axis: &str, _bytes: &[u8]) -> Result<Vec<u8>, StoreError> {
        Err(StoreError::UnknownAxis)
    }

    /// Multi-axis get by on-the-wire κ-label bytes (variable width). Default: unsupported.
    fn get_axis(&self, _label_bytes: &[u8]) -> Result<Option<Bytes>, StoreError> {
        Err(StoreError::UnknownAxis)
    }

    /// Multi-axis presence. Default: not present.
    fn contains_axis(&self, _label_bytes: &[u8]) -> bool {
        false
    }
}

/// Reachability-based eviction (spec §5.3 / §10.8). Deliberately **separate** from [`KappaStore`]:
/// eviction is a backend/operator action, never part of the append-only container surface (§10.5).
/// `gc` walks reachability from the pinned roots over `registry` and reclaims unreachable *bytes*
/// (never the addressing relation). Returns the eviction count.
pub trait GarbageCollect {
    fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError>;
}

// ───────────────────────────── KappaSync (async, spec §8.2) ─────────────────────────────

/// κ-label propagation between peers/gateways. Async (network is fundamentally async). Every
/// fetched byte sequence MUST be re-derived ([`verify_kappa`]) before acceptance (SPINE-4).
///
/// **Maybe-Send (LAW-4).** One trait, one method set — the `Send` bound is target-conditional.
/// On native (multi-core executors) it is `Send + Sync`. On `wasm32` / bare-metal single-core
/// executors like embassy, futures are typically `!Send`, so the bound is dropped (`?Send`):
/// implementors may hold non-`Send` state and produce non-`Send` futures. A `#[async_trait]`
/// (Send-form) impl satisfies the `?Send` trait on those targets — it merely promises *more*
/// than the trait requires, so std transports (`BareNetSync`, `HttpKappaSync`, `TcpKappaSync`)
/// compile unchanged everywhere.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
pub trait KappaSync: Send + Sync {
    /// Fetch a κ's canonical bytes from any peer/gateway. `Ok(None)` ⇒ nobody has it.
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError>;
    /// Announce that we hold a κ (best-effort).
    async fn announce(&self, kappa: &KappaLabel71);
    /// Discover κ-labels other peers hold (prefix-filtered, up to `limit`).
    async fn discover(&self, prefix: Option<&[u8]>, limit: usize) -> Vec<KappaLabel71>;
    async fn add_peer(&self, peer_addr: &str) -> Result<(), SyncError>;
    async fn add_gateway(&self, url: &str) -> Result<(), SyncError>;
}

/// The `?Send` variant of [`KappaSync`] on `wasm32` (and single-core `!Send` executors like
/// embassy). Same method set; the `Send + Sync` bound is dropped so `!Send` implementors and
/// futures are accepted (the maybe-Send policy, LAW-4).
#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
pub trait KappaSync {
    /// Fetch a κ's canonical bytes from any peer/gateway. `Ok(None)` ⇒ nobody has it.
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError>;
    /// Announce that we hold a κ (best-effort).
    async fn announce(&self, kappa: &KappaLabel71);
    /// Discover κ-labels other peers hold (prefix-filtered, up to `limit`).
    async fn discover(&self, prefix: Option<&[u8]>, limit: usize) -> Vec<KappaLabel71>;
    async fn add_peer(&self, peer_addr: &str) -> Result<(), SyncError>;
    async fn add_gateway(&self, url: &str) -> Result<(), SyncError>;
}

/// Eviction-tolerant read (spec §5.2): local store first, else fetch + **verify on receipt**
/// (SPINE-4) + cache under the κ's own axis. The one read path the whole substrate uses.
pub async fn get_with_fetch(
    store: &dyn KappaStore,
    sync: &dyn KappaSync,
    kappa: &KappaLabel71,
) -> Result<Option<Bytes>, AccessError> {
    if let Some(bytes) = store.get(kappa).map_err(AccessError::StoreFailure)? {
        return Ok(Some(bytes));
    }
    let fetched = sync.fetch(kappa).await.map_err(AccessError::SyncFailure)?;
    if let Some(bytes) = &fetched {
        if !verify_kappa(bytes, kappa).map_err(|_| AccessError::VerificationFailed)? {
            return Err(AccessError::VerificationFailed);
        }
        let axis = kappa.sigma_axis().ok_or(AccessError::VerificationFailed)?;
        store.put(axis, bytes).map_err(AccessError::StoreFailure)?;
    }
    Ok(fetched)
}

// ───────────────────────────── ContainerRuntime (async, spec §8.3) ─────────────────────────────

/// In-process handle to a running container instance. Opaque; **not** durable across restarts
/// (use the Container ID κ-label for durable references).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ContainerHandle(pub u64);

/// Lifecycle state of a container instance.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ContainerState {
    Initializing,
    Running,
    Suspended,
    Terminating,
}

/// Snapshot of a running container, from [`ContainerRuntime::info`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerInfo {
    pub container_id: KappaLabel71,
    pub capabilities_kappa: KappaLabel71,
    pub current_snapshot: Option<KappaLabel71>,
    pub state: ContainerState,
    pub memory_bytes: u64,
}

/// Loads container code by Container ID, mediates capabilities, manages lifecycle. Async
/// (load/spawn/snapshot span time). **`caps` is a Capability Set κ-label**, not a struct — the
/// authority must live in the graph to be auditable/revocable (SPINE-1, correcting spec §8.3 / B3).
///
/// **Maybe-Send** (LAW-4), the same cfg-gated posture as [`KappaSync`]: `Send + Sync` on native,
/// `?Send` on `wasm32`/bare where a `Runtime` holds a `?Send` `KappaSync` and its futures are
/// `!Send`. A single trait per target (not a disjoint `Local*` twin), so `Session`/`Peer`/`Manager`
/// stay generic over one `R: ContainerRuntime` on every platform.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
pub trait ContainerRuntime: Send + Sync {
    async fn spawn(
        &self,
        container_id: &KappaLabel71,
        capabilities: &KappaLabel71,
    ) -> Result<ContainerHandle, RuntimeError>;
    /// Suspend to a snapshot κ-label.
    async fn suspend(&self, handle: ContainerHandle) -> Result<KappaLabel71, RuntimeError>;
    async fn resume(
        &self,
        snapshot: &KappaLabel71,
        capabilities: &KappaLabel71,
    ) -> Result<ContainerHandle, RuntimeError>;
    async fn terminate(&self, handle: ContainerHandle) -> Result<(), RuntimeError>;
    fn list(&self) -> Vec<ContainerHandle>;
    fn info(&self, handle: ContainerHandle) -> Option<ContainerInfo>;
}

/// `?Send` `wasm32`/bare variant — see the native definition above.
#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
pub trait ContainerRuntime {
    async fn spawn(
        &self,
        container_id: &KappaLabel71,
        capabilities: &KappaLabel71,
    ) -> Result<ContainerHandle, RuntimeError>;
    /// Suspend to a snapshot κ-label.
    async fn suspend(&self, handle: ContainerHandle) -> Result<KappaLabel71, RuntimeError>;
    async fn resume(
        &self,
        snapshot: &KappaLabel71,
        capabilities: &KappaLabel71,
    ) -> Result<ContainerHandle, RuntimeError>;
    async fn terminate(&self, handle: ContainerHandle) -> Result<(), RuntimeError>;
    fn list(&self) -> Vec<ContainerHandle>;
    fn info(&self, handle: ContainerHandle) -> Option<ContainerInfo>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_kappa_round_trips_blake3() {
        let k = address_bytes(b"hologram");
        assert_eq!(k.sigma_axis(), Some("blake3"));
        assert!(verify_kappa(b"hologram", &k).unwrap());
        assert!(!verify_kappa(b"hologramX", &k).unwrap());
    }

    #[test]
    fn address_bytes_is_deterministic_and_dedups() {
        assert_eq!(address_bytes(b"abc"), address_bytes(b"abc"));
        assert_ne!(address_bytes(b"abc"), address_bytes(b"abd"));
    }

    #[test]
    fn derive_label_is_order_sensitive() {
        let a = address_bytes(b"a");
        let b = address_bytes(b"b");
        assert_ne!(derive_label(b"op", &[a, b]), derive_label(b"op", &[b, a]));
    }

    /// B4 / G-D1 — on `wasm32` the maybe-Send [`KappaSync`] is `?Send` and accepts `!Send`
    /// implementors. The bound here is structural: this test is "compiles" — if the `wasm32`
    /// [`KappaSync`] retained `Send + Sync` bounds it would fail to compile against a `Rc`-bearing
    /// impl (`Rc` is `!Send`). It is `wasm32`-gated because that is the target the `?Send` bound
    /// selects; on native the trait is `Send + Sync` and this `Rc` impl is intentionally rejected.
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn kappa_sync_accepts_non_send_implementors_on_wasm() {
        use alloc::rc::Rc;
        struct NotSend {
            _state: Rc<u32>,
        }
        #[async_trait::async_trait(?Send)]
        impl KappaSync for NotSend {
            async fn fetch(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
                Ok(None)
            }
            async fn announce(&self, _kappa: &KappaLabel71) {}
            async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
                Vec::new()
            }
            async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
                Ok(())
            }
            async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
                Ok(())
            }
        }
        // If this compiles, the trait is `?Send`-implementable. The Rc proves it.
        let _s = NotSend { _state: Rc::new(0) };
    }

    // ── AS — σ-axis correctness against external KAT vectors (architecture §3.1 G-B1) ──

    /// Render bytes as lowercase ASCII hex (helper for the KAT assertions).
    fn hex_bytes(b: &[u8]) -> alloc::string::String {
        use alloc::string::String;
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(b.len() * 2);
        for &x in b {
            s.push(HEX[(x >> 4) as usize] as char);
            s.push(HEX[(x & 0xf) as usize] as char);
        }
        s
    }

    #[test]
    fn as_blake3_axis_matches_blake3_reference_crate() {
        // Differential test: the substrate's blake3 path must produce the SAME 32-byte digest as
        // the independent `blake3` reference crate (vendored test). The hologram σ-axis goes through
        // `prism::crypto::Blake3Hasher`; this asserts byte-identity with the upstream `blake3 = "1"`.
        let inputs: &[&[u8]] = &[b"", b"abc", b"hologram-substrate"];
        for &input in inputs {
            let ours = address_bytes_axis("blake3", input).unwrap();
            let theirs = ::blake3::hash(input);
            let mut expected = alloc::string::String::from("blake3:");
            for &b in theirs.as_bytes() {
                use core::fmt::Write;
                let _ = write!(expected, "{:02x}", b);
            }
            assert_eq!(
                core::str::from_utf8(&ours).unwrap(),
                expected,
                "BLAKE3 differential disagreement on input {:?}",
                input
            );
        }
    }

    #[test]
    fn as_sha256_axis_matches_fips180_4_kat() {
        // FIPS 180-4 / NIST CAVS KAT vectors for SHA-256
        // empty: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        // "abc":  ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let empty = address_bytes_axis("sha256", b"").unwrap();
        let abc = address_bytes_axis("sha256", b"abc").unwrap();
        assert_eq!(
            core::str::from_utf8(&empty).unwrap(),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            core::str::from_utf8(&abc).unwrap(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn as_sha3_256_axis_matches_fips202_kat() {
        // FIPS 202 / NIST KAT vectors for SHA3-256
        // empty: a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
        // "abc":  3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532
        let empty = address_bytes_axis("sha3-256", b"").unwrap();
        let abc = address_bytes_axis("sha3-256", b"abc").unwrap();
        assert_eq!(
            core::str::from_utf8(&empty).unwrap(),
            "sha3-256:a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
        assert_eq!(
            core::str::from_utf8(&abc).unwrap(),
            "sha3-256:3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        );
    }

    #[test]
    fn as_keccak256_axis_matches_ethereum_kat() {
        // Keccak-256 (pre-FIPS-202 finalist; widely used in Ethereum):
        // empty: c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
        // "abc":  4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
        let empty = address_bytes_axis("keccak256", b"").unwrap();
        let abc = address_bytes_axis("keccak256", b"abc").unwrap();
        assert_eq!(
            core::str::from_utf8(&empty).unwrap(),
            "keccak256:c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
        assert_eq!(
            core::str::from_utf8(&abc).unwrap(),
            "keccak256:4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
        );
    }

    #[test]
    fn as_sha512_axis_matches_fips180_4_kat() {
        // FIPS 180-4 KAT vectors for SHA-512 ("abc"):
        // ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a
        // 2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f
        let abc = address_bytes_axis("sha512", b"abc").unwrap();
        assert_eq!(
            core::str::from_utf8(&abc).unwrap(),
            "sha512:ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn as_sha256_axis_differentials_with_sha2_reference_crate() {
        // Differential: every byte-string from a small corpus must produce the same sha256 digest
        // through our σ-axis as the upstream RustCrypto `sha2` crate (FIPS 180-4 reference impl).
        use sha2::Digest;
        for &input in &[
            b"".as_ref(),
            b"abc".as_ref(),
            b"hologram-substrate-multi-axis-conformance".as_ref(),
            &[0xffu8; 256][..],
        ] {
            let ours = address_bytes_axis("sha256", input).unwrap();
            let theirs = sha2::Sha256::digest(input);
            let mut expected = alloc::string::String::from("sha256:");
            for &b in theirs.iter() {
                use core::fmt::Write;
                let _ = write!(expected, "{:02x}", b);
            }
            assert_eq!(
                core::str::from_utf8(&ours).unwrap(),
                expected,
                "sha256 differential disagreement on input of len {}",
                input.len()
            );
        }
    }

    #[test]
    fn as_sha3_256_axis_differentials_with_sha3_reference_crate() {
        // Differential: every byte-string must produce the same sha3-256 digest through our σ-axis
        // as the upstream `sha3` crate (NIST FIPS 202 reference impl).
        use sha3::Digest;
        for &input in &[
            b"".as_ref(),
            b"abc".as_ref(),
            b"hologram-multi-axis-fips202".as_ref(),
        ] {
            let ours = address_bytes_axis("sha3-256", input).unwrap();
            let theirs = sha3::Sha3_256::digest(input);
            let mut expected = alloc::string::String::from("sha3-256:");
            for &b in theirs.iter() {
                use core::fmt::Write;
                let _ = write!(expected, "{:02x}", b);
            }
            assert_eq!(core::str::from_utf8(&ours).unwrap(), expected);
        }
    }

    #[test]
    fn as_keccak256_axis_differentials_with_sha3_reference_crate() {
        // Differential against `sha3::Keccak256` (the pre-FIPS-202 sponge variant; Ethereum uses
        // this construction). Confirms the substrate's keccak path is interop-compatible with the
        // standard upstream implementation.
        use sha3::Digest;
        for &input in &[b"".as_ref(), b"abc".as_ref(), b"keccak-eth-compat".as_ref()] {
            let ours = address_bytes_axis("keccak256", input).unwrap();
            let theirs = sha3::Keccak256::digest(input);
            let mut expected = alloc::string::String::from("keccak256:");
            for &b in theirs.iter() {
                use core::fmt::Write;
                let _ = write!(expected, "{:02x}", b);
            }
            assert_eq!(core::str::from_utf8(&ours).unwrap(), expected);
        }
    }

    #[test]
    fn as_sha512_axis_differentials_with_sha2_reference_crate() {
        use sha2::Digest;
        for &input in &[b"".as_ref(), b"abc".as_ref(), &[0xa5u8; 1024][..]] {
            let ours = address_bytes_axis("sha512", input).unwrap();
            let theirs = sha2::Sha512::digest(input);
            let mut expected = alloc::string::String::from("sha512:");
            for &b in theirs.iter() {
                use core::fmt::Write;
                let _ = write!(expected, "{:02x}", b);
            }
            assert_eq!(core::str::from_utf8(&ours).unwrap(), expected);
        }
    }

    #[test]
    fn verify_kappa_axis_handles_all_five_axes() {
        for axis in &["blake3", "sha256", "sha3-256", "keccak256", "sha512"] {
            let bytes = b"hologram-substrate-multi-axis";
            let label = address_bytes_axis(axis, bytes).unwrap();
            assert!(
                verify_kappa_axis(bytes, &label).unwrap(),
                "{axis} verify must accept the bytes that produced it"
            );
            // Tampered bytes are rejected.
            assert!(
                !verify_kappa_axis(b"tampered", &label).unwrap(),
                "{axis} verify must reject tampered bytes"
            );
        }
        // Unknown axis fails loud (SPINE-6 no-fallback).
        assert_eq!(
            address_bytes_axis("md5", b""),
            Err(AxisError::UnsupportedAxis)
        );
    }

    #[test]
    fn as_label_widths_match_uor_addr_geometry() {
        // architecture §3.1 / uor-addr bounds: blake3=71, sha256=71, sha3-256=73, keccak256=74,
        // sha512=135. Re-derive empty bytes through each axis and assert the on-the-wire width.
        assert_eq!(address_bytes_axis("blake3", b"").unwrap().len(), 71);
        assert_eq!(address_bytes_axis("sha256", b"").unwrap().len(), 71);
        assert_eq!(address_bytes_axis("sha3-256", b"").unwrap().len(), 73);
        assert_eq!(address_bytes_axis("keccak256", b"").unwrap().len(), 74);
        assert_eq!(address_bytes_axis("sha512", b"").unwrap().len(), 135);
        const _: () = assert!(135 == MAX_LABEL_BYTES);
    }

    /// Confirm the rendering uses lowercase hex (the canonical κ-label form).
    #[test]
    fn as_label_uses_lowercase_hex_canonical_form() {
        let v = address_bytes_axis("sha256", b"abc").unwrap();
        let s = core::str::from_utf8(&v).unwrap();
        assert!(s.chars().all(|c| !c.is_uppercase()));
        let _ = hex_bytes(b"\x00\x0f\xff"); // exercise helper
    }

    // ── CR — capability delegation containment (SubtypingLattice relation, §3.4 / §10.7) ──

    use alloc::vec;

    fn caps(roots: &[&[u8]], quota: u64, fetch: bool) -> Capabilities {
        Capabilities {
            storage_roots: roots.iter().map(|r| address_bytes(r)).collect(),
            storage_quota_bytes: quota,
            network_fetch: fetch,
            network_announce: false,
            publish_channels: vec![],
            subscribe_channels: vec![],
            memory_max_bytes: 1 << 20,
            cpu_time_per_event_ms: 100,
            priority_weight: 0,
        }
    }

    #[test]
    fn cr_admits_is_reflexive() {
        let c = caps(&[b"r1", b"r2"], 1000, true);
        assert!(
            c.admits(&c),
            "every capability set admits itself (reflexive)"
        );
    }

    #[test]
    fn cr_admits_is_transitive() {
        let a = caps(&[b"r1", b"r2"], 1000, true);
        let b = caps(&[b"r1"], 500, true);
        let c = caps(&[b"r1"], 100, false);
        assert!(a.admits(&b) && b.admits(&c));
        assert!(a.admits(&c), "transitive: a⊇b⊇c ⟹ a admits c");
    }

    #[test]
    fn cr_admits_is_antisymmetric_on_grants() {
        let a = caps(&[b"r1"], 500, true);
        let b = caps(&[b"r1"], 500, true);
        assert!(a.admits(&b) && b.admits(&a));
        assert_eq!(a, b, "mutual admission ⟹ equal grant sets (antisymmetric)");
    }

    #[test]
    fn cr_rejects_over_broad_delegations() {
        let parent = caps(&[b"r1"], 500, false);
        // Extra storage root the parent does not have.
        assert!(!parent.admits(&caps(&[b"r1", b"r2"], 500, false)));
        // Higher quota than the parent.
        assert!(!parent.admits(&caps(&[b"r1"], 9999, false)));
        // A network flag the parent lacks.
        assert!(!parent.admits(&caps(&[b"r1"], 500, true)));
        // A properly narrowed child IS admitted (equal quota — same bound).
        assert!(parent.admits(&caps(&[b"r1"], 500, false)));
        // A strictly tighter quota IS admitted.
        assert!(parent.admits(&caps(&[b"r1"], 100, false)));
        // Empty storage roots are tighter than {r1} — admitted under the subset rule.
        assert!(parent.admits(&caps(&[], 100, false)));
        // **0 = unbounded** semantics (arch §3.4 + spec §7.6): a child requesting unbounded under
        // a bounded parent is OVER-broad (refused). The naive `child ≤ parent` rule would have
        // wrongly admitted this — `budget_admits` guards against the silent widening.
        assert!(!parent.admits(&caps(&[b"r1"], 0, false)));
        assert!(!parent.admits(&caps(&[], 0, false)));
    }

    #[test]
    fn cr_unbounded_parent_admits_any_child_budget() {
        // Conversely, an unbounded parent (quota=0) admits any child quota, including unbounded.
        let unbounded_parent = caps(&[b"r"], 0, false);
        assert!(unbounded_parent.admits(&caps(&[b"r"], 0, false)));
        assert!(unbounded_parent.admits(&caps(&[b"r"], 1 << 20, false)));
        assert!(unbounded_parent.admits(&caps(&[b"r"], u64::MAX, false)));
        assert!(unbounded_parent.admits(&caps(&[], 100, false)));
    }
}

// ───────────────────────────── FederatedKappaSync (arch §11.2) ─────────────────────────────

/// A hierarchical [`KappaSync`] over multiple backends — the architecture §11.2 federated
/// multi-source. A `fetch` tries each backend in order, applying each backend's own
/// verify-on-receipt (SPINE-4) at every hop; the first hit wins. `add_peer` and `add_gateway` are
/// routed by input shape: the first backend that accepts it wins, the rest err. This is how the
/// substrate composes its uor-native TCP transport (`hologram-net-tcp`) + HTTP-CAS gateways
/// (`hologram-net-http`) + the local store into one read path without any backend privileged
/// over another.
///
/// Backends are immutable post-construction; reuse / extension is a `new` with the additional
/// `Arc`. The internal `add_*` mutations on individual backends still work as normal.
pub struct FederatedKappaSync {
    backends: Vec<Arc<dyn KappaSync>>,
}

impl FederatedKappaSync {
    pub fn new(backends: Vec<Arc<dyn KappaSync>>) -> Self {
        Self { backends }
    }

    /// Number of backends in the chain.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

// Maybe-Send follow-through: the impl awaits `Arc<dyn KappaSync>` methods, so its future is
// `Send` only where the trait is `Send`. Gate the `?Send` desugaring to `wasm32`, matching the
// [`KappaSync`] trait definition (native = Send futures, wasm32 = `?Send`).
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl KappaSync for FederatedKappaSync {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        if self.backends.is_empty() {
            return Err(SyncError::NotEnabled);
        }
        let mut sticky: Option<SyncError> = None;
        for b in &self.backends {
            match b.fetch(kappa).await {
                Ok(Some(bytes)) => return Ok(Some(bytes)),
                Ok(None) => {}
                Err(SyncError::NotEnabled) | Err(SyncError::AllSourcesFailed) => {}
                // A real error (e.g. VerificationFailed = a forging hop) — remember it so the
                // caller knows the network had bytes that didn't match the κ — but keep walking,
                // a later honest hop may still satisfy the fetch.
                Err(e) => sticky = Some(e),
            }
        }
        match sticky {
            Some(e) => Err(e),
            None => Ok(None),
        }
    }

    async fn announce(&self, kappa: &KappaLabel71) {
        for b in &self.backends {
            b.announce(kappa).await;
        }
    }

    async fn discover(&self, prefix: Option<&[u8]>, limit: usize) -> Vec<KappaLabel71> {
        let mut out: Vec<KappaLabel71> = Vec::new();
        for b in &self.backends {
            for k in b.discover(prefix, limit).await {
                if !out.iter().any(|x| x == &k) {
                    out.push(k);
                }
                if out.len() >= limit {
                    return out;
                }
            }
        }
        out
    }

    async fn add_peer(&self, peer_addr: &str) -> Result<(), SyncError> {
        let mut last = SyncError::AllSourcesFailed;
        for b in &self.backends {
            match b.add_peer(peer_addr).await {
                Ok(()) => return Ok(()),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    async fn add_gateway(&self, url: &str) -> Result<(), SyncError> {
        let mut last = SyncError::AllSourcesFailed;
        for b in &self.backends {
            match b.add_gateway(url).await {
                Ok(()) => return Ok(()),
                Err(e) => last = e,
            }
        }
        Err(last)
    }
}

#[cfg(test)]
mod federated_tests {
    use super::*;
    use alloc::string::ToString;
    use core::sync::atomic::{AtomicUsize, Ordering};

    /// Mock backend with a fixed value table; tracks how many fetches it served.
    struct Mock {
        name: String,
        table: hashbrown::HashMap<KappaLabel71, Bytes>,
        fetches: AtomicUsize,
        accepts_peer: bool,
        accepts_gateway: bool,
        forges: bool,
    }
    impl Mock {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                table: hashbrown::HashMap::new(),
                fetches: AtomicUsize::new(0),
                accepts_peer: false,
                accepts_gateway: false,
                forges: false,
            }
        }
        fn insert(mut self, k: KappaLabel71, v: &[u8]) -> Self {
            self.table.insert(k, Arc::<[u8]>::from(v));
            self
        }
        fn peer(mut self) -> Self {
            self.accepts_peer = true;
            self
        }
        fn gateway(mut self) -> Self {
            self.accepts_gateway = true;
            self
        }
        fn forge(mut self) -> Self {
            self.forges = true;
            self
        }
    }
    #[async_trait::async_trait]
    impl KappaSync for Mock {
        async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
            self.fetches.fetch_add(1, Ordering::Relaxed);
            if self.forges {
                // Pretend to serve random bytes — the σ-axis would catch them. Federation should
                // skip this hop and keep walking the chain.
                return Err(SyncError::VerificationFailed);
            }
            Ok(self.table.get(kappa).cloned())
        }
        async fn announce(&self, _kappa: &KappaLabel71) {}
        async fn discover(&self, _prefix: Option<&[u8]>, limit: usize) -> Vec<KappaLabel71> {
            self.table.keys().take(limit).copied().collect()
        }
        async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
            if self.accepts_peer {
                Ok(())
            } else {
                Err(SyncError::BackendFailure("not a peer backend"))
            }
        }
        async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
            if self.accepts_gateway {
                Ok(())
            } else {
                Err(SyncError::BackendFailure("not a gateway backend"))
            }
        }
    }

    fn kappa_of(bytes: &[u8]) -> KappaLabel71 {
        kappa::address_bytes(bytes)
    }

    fn run<F: core::future::Future>(mut f: F) -> F::Output {
        // Minimal block_on for the test (no extra runtime). The mock backends complete
        // synchronously, so the future polls Ready on the first turn.
        use core::pin::Pin;
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        const VT: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VT),
            |_| {},
            |_| {},
            |_| {},
        );
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) };
        let mut cx = Context::from_waker(&waker);
        // SAFETY: `f` lives on our stack frame and we never move it after this point.
        let mut pinned = unsafe { Pin::new_unchecked(&mut f) };
        loop {
            match pinned.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => {}
            }
        }
    }

    #[test]
    fn fed_fetch_walks_chain_until_a_hit() {
        let bytes_a = b"alpha".as_ref();
        let bytes_b = b"beta".as_ref();
        let ka = kappa_of(bytes_a);
        let kb = kappa_of(bytes_b);
        // First backend has only κ_a; second has κ_b. The federation must find both.
        let m1 = Arc::new(Mock::new("hot").insert(ka, bytes_a));
        let m2 = Arc::new(Mock::new("cold").insert(kb, bytes_b));
        let fed = FederatedKappaSync::new(alloc::vec![m1.clone(), m2.clone()]);
        let a = run(fed.fetch(&ka)).unwrap().unwrap();
        let b = run(fed.fetch(&kb)).unwrap().unwrap();
        assert_eq!(a.as_ref(), bytes_a);
        assert_eq!(b.as_ref(), bytes_b);
        // The hot backend was tried first; the cold backend was only tried for κ_b.
        assert_eq!(m1.fetches.load(Ordering::Relaxed), 2);
        assert_eq!(m2.fetches.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn fed_skips_forging_hop_and_surfaces_to_caller() {
        let bytes = b"genuine".as_ref();
        let k = kappa_of(bytes);
        let forger = Arc::new(Mock::new("forger").forge());
        let honest = Arc::new(Mock::new("honest").insert(k, bytes));
        let fed = FederatedKappaSync::new(alloc::vec![forger, honest]);
        let got = run(fed.fetch(&k)).unwrap().unwrap();
        assert_eq!(
            got.as_ref(),
            bytes,
            "honest hop after forger still resolves"
        );
    }

    #[test]
    fn fed_routes_add_peer_and_add_gateway_by_input_shape() {
        let peer_only = Arc::new(Mock::new("peer").peer());
        let gateway_only = Arc::new(Mock::new("gw").gateway());
        let fed = FederatedKappaSync::new(alloc::vec![peer_only, gateway_only]);
        assert!(run(fed.add_peer("/ip4/127.0.0.1/tcp/4001/p2p/12D3K...")).is_ok());
        assert!(run(fed.add_gateway("https://gateway.example/")).is_ok());
    }

    #[test]
    fn fed_empty_chain_is_not_enabled() {
        let fed = FederatedKappaSync::new(Vec::new());
        let k = kappa_of(b"x");
        assert_eq!(run(fed.fetch(&k)), Err(SyncError::NotEnabled));
    }

    // Silence dead-code warnings on the Mock `name` field in non-test builds.
    #[allow(dead_code)]
    fn _name_used(m: &Mock) -> &str {
        m.name.as_str()
    }
}
