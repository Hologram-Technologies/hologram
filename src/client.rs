//! **`Client` (D4)** — the single programmatic surface (05-tooling.md §law-6).
//!
//! `Client<S: Space>` composes the compiler + exec hot path over any space's `KappaStore`
//! and network `KappaSync` seam. The CLI, C ABI, and SDKs all wrap this one type — behavior
//! lives in exactly one place, so bindings cannot drift.
//!
//! This is the kept realization of the P0.5 SP-3 spike (D28): **compile** (sync compute) →
//! **provision** (sync storage, law 4) → **resolve / run** (the async network seam calling
//! straight into the sync compute hot path — the only async↔sync transition, LAW-4). It also
//! drives the container lifecycle: **`open`** returns a [`Session`] over the space's
//! [`runtime`](hologram_space::Space::runtime) (`boot`/`suspend`/`resume`/`terminate`).

use alloc::vec::Vec;

use hologram_archive::{HoloLoader, HoloWriter, SectionKind};
use hologram_compiler::{compile, BackendKind, CompileError};
use hologram_compute::CpuBackend;
use hologram_exec::{BufferArena, ExecError, InferenceSession, InputBuffer};
use hologram_graph::Graph;
use hologram_runtime::Session;
use hologram_space::{
    resolve_closure, verify_kappa, AppManifest, Bytes, Closure, GarbageCollect, KappaLabel71,
    KappaStore, KappaSync, LayerKind, MemKappaStore, Realization, Space, StoreError, SyncError,
    REGISTRY,
};
use prism::vocabulary::WittLevel;

/// A compiled `.holo` application (v3 archive bytes), before it is provisioned to a store.
#[derive(Clone, Debug)]
pub struct Holo {
    bytes: Vec<u8>,
}

impl Holo {
    /// Wrap already-compiled archive bytes as a `Holo`.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
    /// The `.holo` archive bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Consume into the raw archive bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// The single programmatic surface (D4): generic over any [`Space`], so it monomorphizes
/// per platform. One place composes behavior; the CLI / FFI / SDKs are thin wrappers.
pub struct Client<S: Space> {
    space: S,
    target: BackendKind,
    level: WittLevel,
}

/// Builder for a [`Client`] (05-tooling.md: `Client::builder().space(..).build()`).
pub struct ClientBuilder<S> {
    space: Option<S>,
    target: BackendKind,
    level: WittLevel,
}

impl<S> Default for ClientBuilder<S> {
    fn default() -> Self {
        Self {
            space: None,
            target: BackendKind::Cpu,
            level: WittLevel::W32,
        }
    }
}

impl<S: Space> ClientBuilder<S> {
    /// Set the space-contract implementation the client composes over.
    #[must_use]
    pub fn space(mut self, space: S) -> Self {
        self.space = Some(space);
        self
    }
    /// Set the compile backend target (default: CPU).
    #[must_use]
    pub fn target(mut self, target: BackendKind) -> Self {
        self.target = target;
        self
    }
    /// Set the Witt precision level (default: W32).
    #[must_use]
    pub fn level(mut self, level: WittLevel) -> Self {
        self.level = level;
        self
    }
    /// Build the client.
    ///
    /// # Errors
    ///
    /// [`BuildError::NoSpace`] if no space was set.
    pub fn build(self) -> Result<Client<S>, BuildError> {
        Ok(Client {
            space: self.space.ok_or(BuildError::NoSpace)?,
            target: self.target,
            level: self.level,
        })
    }
}

impl<S: Space> Client<S> {
    /// A builder for a client (05-tooling.md entry point).
    #[must_use]
    pub fn builder() -> ClientBuilder<S> {
        ClientBuilder::default()
    }

    /// Compose a client directly over a space (CPU / W32 defaults).
    pub fn new(space: S) -> Self {
        Self {
            space,
            target: BackendKind::Cpu,
            level: WittLevel::W32,
        }
    }

    /// The composed space.
    pub fn space(&self) -> &S {
        &self.space
    }

    /// The space's content-addressed store.
    pub fn store(&self) -> &S::Store {
        self.space.store()
    }

    /// **Compile** a graph to a `.holo` — synchronous (pure compute hot path, law 4).
    ///
    /// # Errors
    ///
    /// [`CompileError`] on an invalid graph or an unsupported op/target.
    pub fn compile(&self, graph: Graph) -> Result<Holo, CompileError> {
        let out = compile(graph, self.target, self.level)?;
        Ok(Holo { bytes: out.archive })
    }

    /// **Provision** a `.holo` into the space's store, returning its κ — synchronous
    /// storage (law 4).
    ///
    /// # Errors
    ///
    /// [`StoreError`] if the store rejects the write.
    pub fn provision(&self, holo: &Holo) -> Result<KappaLabel71, StoreError> {
        self.space.store().put("blake3", holo.as_bytes())
    }

    /// **Open** a lifecycle [`Session`] for a provisioned container: its container-manifest κ
    /// under a capability-set κ, driven over the space's [`runtime`](hologram_space::Space::runtime).
    /// The returned session is in the `Provisioned` phase; the caller drives
    /// `boot`/`suspend`/`resume`/`terminate` (05-tooling.md). Because a snapshot is content
    /// (a κ), a session suspended on one space can be resumed on another.
    pub fn open(&self, container: &KappaLabel71, caps: &KappaLabel71) -> Session<'_, S::Runtime> {
        Session::provision(self.space.runtime(), *container, *caps)
    }

    /// Fetch a κ's bytes from the local store.
    ///
    /// # Errors
    ///
    /// [`StoreError`] on a store failure.
    pub fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        self.space.store().get(kappa)
    }

    /// Pin a κ as a GC reachability root.
    ///
    /// # Errors
    ///
    /// [`StoreError`] on a store failure.
    pub fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        self.space.store().pin(kappa)
    }

    /// Remove a pin.
    ///
    /// # Errors
    ///
    /// [`StoreError`] on a store failure.
    pub fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        self.space.store().unpin(kappa)
    }

    /// List locally-present κ-labels.
    pub fn ls(&self) -> Vec<KappaLabel71> {
        self.space.store().iterate()
    }

    /// Re-derive `bytes` through the σ-axis and check they match `kappa` (SPINE-4).
    pub fn verify(&self, bytes: &[u8], kappa: &KappaLabel71) -> bool {
        verify_kappa(bytes, kappa).unwrap_or(false)
    }

    /// **Inspect** a `.holo` v3 application without running it (spec 03): decode its manifest and
    /// check every layer's certificate. A layer's certificate is its κ-identity **bound into the
    /// app's committed identity** — the manifest κ is the content address of the canonical bytes
    /// that embed every layer κ, so removing, swapping, or altering a layer changes the app κ. The
    /// check is **thin**: it needs only the manifest, never the payload (fat) profile, and returns
    /// one verdict per layer — certificates travel with the manifest and inspection never strips
    /// them. (Deep content authenticity — re-deriving each payload — is the fat verify-on-receipt
    /// path, `resolve_closure` + [`verify`](Self::verify), not this inspection.)
    ///
    /// # Errors
    ///
    /// [`InspectError`] if the bytes are not a loadable `.holo`, carry no manifest (a bare tensor
    /// container), or the manifest is malformed.
    pub fn inspect(&self, holo: &Holo) -> Result<AppInspection, InspectError> {
        let plan = HoloLoader::from_bytes(holo.as_bytes())
            .map_err(|_| InspectError::NotLoadable)?
            .into_plan()
            .map_err(|_| InspectError::NotLoadable)?;
        let manifest_bytes = plan.app_manifest().ok_or(InspectError::NotAnApplication)?;
        let manifest =
            AppManifest::decode(manifest_bytes).map_err(|_| InspectError::MalformedManifest)?;
        // The reachability closure of the manifest bytes IS the set of committed operand κs; a
        // layer's certificate verifies iff its κ is bound into that closure (and hence the app κ).
        let refs = <AppManifest as Realization>::references(manifest_bytes)
            .map_err(|_| InspectError::MalformedManifest)?;
        let layers = manifest
            .layers
            .iter()
            .map(|l| LayerCertVerdict {
                layer: l.content,
                kind: l.kind,
                verified: refs.contains(&l.content),
            })
            .collect();
        Ok(AppInspection {
            app: manifest.kappa(),
            layers,
        })
    }

    /// Whether `holo` is a **fat** archive (spec 03 §Fat and thin): its manifest's closure resolves
    /// entirely from the archive's own embedded content blobs — no store, no network. A **thin**
    /// archive returns `false`; its layers resolve through the store/sync at load.
    pub fn is_fat(&self, holo: &Holo) -> bool {
        Self::archive_closure(holo).is_some_and(|c| c.is_complete())
    }

    /// Resolve the manifest's closure over a scratch store seeded **only** from the archive's own
    /// content blobs — the basis for [`is_fat`](Self::is_fat). `None` if there is no manifest.
    fn archive_closure(holo: &Holo) -> Option<Closure> {
        let plan = HoloLoader::from_bytes(holo.as_bytes())
            .ok()?
            .into_plan()
            .ok()?;
        let manifest_bytes = plan.app_manifest()?;
        let scratch = MemKappaStore::new();
        let manifest_kappa = scratch.put("blake3", manifest_bytes).ok()?;
        // Each blob's content re-addresses to its own κ (content-addressed), so the layer κs the
        // manifest names resolve iff their bytes were embedded.
        for (_kappa, content) in plan.content_blobs().ok()? {
            let _ = scratch.put("blake3", content);
        }
        resolve_closure(manifest_kappa, &scratch, REGISTRY).ok()
    }

    /// Convert `holo` to a **thin** archive (spec 03 §Fat and thin): manifest + certificates only,
    /// dropping embedded content — layers resolve through the store/sync at load. The manifest κ (the
    /// app's identity) is **unchanged** — fat↔thin is packaging, never identity. Idempotent.
    ///
    /// # Errors
    ///
    /// [`InspectError`] if `holo` is not a loadable `.holo` or carries no manifest.
    pub fn thin(&self, holo: &Holo) -> Result<Holo, InspectError> {
        let plan = HoloLoader::from_bytes(holo.as_bytes())
            .map_err(|_| InspectError::NotLoadable)?
            .into_plan()
            .map_err(|_| InspectError::NotLoadable)?;
        let manifest = plan
            .app_manifest()
            .ok_or(InspectError::NotAnApplication)?
            .to_vec();
        let mut sections: Vec<(SectionKind, Vec<u8>)> = Vec::new();
        sections.push((SectionKind::AppManifest, manifest));
        if let Ok(certs) = plan.section(SectionKind::Certificates) {
            sections.push((SectionKind::Certificates, certs.to_vec()));
        }
        Ok(Holo::from_bytes(HoloWriter::assemble(sections)))
    }

    /// Convert `holo` to a **fat** archive (spec 03 §Fat and thin): manifest + certificates + a
    /// content blob for every layer/closure κ resolvable from the store, so the file is
    /// self-contained. The manifest κ is **unchanged**. κs absent from the store stay unresolved (the
    /// result is as fat as the store allows — re-run once missing content is synced).
    ///
    /// # Errors
    ///
    /// [`InspectError`] if `holo` is not a loadable `.holo`, carries no manifest, or the store fails.
    pub fn fat(&self, holo: &Holo) -> Result<Holo, InspectError> {
        let plan = HoloLoader::from_bytes(holo.as_bytes())
            .map_err(|_| InspectError::NotLoadable)?
            .into_plan()
            .map_err(|_| InspectError::NotLoadable)?;
        let manifest = plan
            .app_manifest()
            .ok_or(InspectError::NotAnApplication)?
            .to_vec();
        // Seed the manifest so the closure walk can start from its κ, then resolve over the store.
        let manifest_kappa = self
            .space
            .store()
            .put("blake3", &manifest)
            .map_err(|_| InspectError::NotLoadable)?;
        let closure = resolve_closure(manifest_kappa, self.space.store(), REGISTRY)
            .map_err(|_| InspectError::NotLoadable)?;
        let mut sections: Vec<(SectionKind, Vec<u8>)> = Vec::new();
        sections.push((SectionKind::AppManifest, manifest));
        if let Ok(certs) = plan.section(SectionKind::Certificates) {
            sections.push((SectionKind::Certificates, certs.to_vec()));
        }
        for kappa in &closure.reachable {
            if let Ok(Some(content)) = self.space.store().get(kappa) {
                let mut blob = kappa.as_array().to_vec();
                blob.extend_from_slice(content.as_ref());
                sections.push((SectionKind::ContentBlob, blob));
            }
        }
        Ok(Holo::from_bytes(HoloWriter::assemble(sections)))
    }

    /// **Resolve** a κ: the space's network [`KappaSync`] seam first (verify-on-receipt, Law L5),
    /// else the local store — the **async** network seam (law 4).
    ///
    /// # Errors
    ///
    /// [`SyncError`] on a network-sync failure, or on a local-store failure (surfaced as
    /// [`SyncError::BackendFailure`] so the async seam carries one unified error).
    pub async fn resolve(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        match self.space.sync().fetch(kappa).await? {
            Some(bytes) => Ok(Some(bytes)),
            None => self
                .space
                .store()
                .get(kappa)
                .map_err(|_| SyncError::BackendFailure("local-store get")),
        }
    }

    /// **Run** a provisioned `.holo` by κ: resolve it (async network seam), then execute the
    /// synchronous compute hot path over the CPU backend, returning the raw output buffers.
    /// This is the composition proof — an `async fn` awaiting the resolver, then calling
    /// straight into synchronous load + execute (the only async↔sync transition, LAW-4).
    ///
    /// # Errors
    ///
    /// [`RunError`] if the κ cannot be resolved, or on a load / execute failure.
    pub async fn run(
        &self,
        kappa: &KappaLabel71,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, RunError> {
        let holo = self
            .resolve(kappa)
            .await
            .map_err(RunError::Resolve)?
            .ok_or(RunError::NotFound)?;
        let backend: CpuBackend<BufferArena> = CpuBackend::new();
        let mut session = InferenceSession::load(&holo, backend).map_err(RunError::Exec)?;
        let ibufs: Vec<InputBuffer> = inputs.iter().map(|b| InputBuffer { bytes: b }).collect();
        let outputs = session.execute(&ibufs).map_err(RunError::Exec)?;
        Ok(outputs.iter().map(|o| o.bytes.to_vec()).collect())
    }
}

impl<S: Space> Client<S>
where
    S::Store: GarbageCollect,
{
    /// Walk reachability from the pinned roots; evict unreachable content. Returns the
    /// number of evicted entries. Available when the space's store supports GC.
    ///
    /// # Errors
    ///
    /// [`StoreError`] on a store failure.
    pub fn gc(&self) -> Result<usize, StoreError> {
        self.space.store().gc(REGISTRY)
    }
}

/// The result of [`Client::inspect`] — a `.holo` v3 application's κ-identity plus a per-layer
/// certificate verdict. Every layer of the manifest appears here, in boot order (certificates are
/// never stripped by inspection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppInspection {
    /// The application's κ-identity (the manifest's content address).
    pub app: KappaLabel71,
    /// One verdict per layer, in manifest (boot) order.
    pub layers: Vec<LayerCertVerdict>,
}

impl AppInspection {
    /// Whether every layer's certificate verified — the whole application's provenance is intact.
    #[must_use]
    pub fn all_verified(&self) -> bool {
        self.layers.iter().all(|l| l.verified)
    }
}

/// One layer's per-layer certificate verdict from [`Client::inspect`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerCertVerdict {
    /// The layer's content κ — its identity and its certificate.
    pub layer: KappaLabel71,
    /// The layer's kind (wasm-codemodule / tensor-plan / rootfs-image / view).
    pub kind: LayerKind,
    /// Whether the certificate verifies — the layer κ is bound into the app's committed identity.
    pub verified: bool,
}

/// Why [`Client::inspect`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectError {
    /// The bytes are not a loadable `.holo` archive.
    NotLoadable,
    /// The archive carries no AppManifest section — it is a bare tensor container, not an app.
    NotAnApplication,
    /// The AppManifest section is malformed.
    MalformedManifest,
}

/// Why [`ClientBuilder::build`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    /// No space was set on the builder.
    NoSpace,
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BuildError::NoSpace => f.write_str("Client::builder(): no space was set"),
        }
    }
}

/// Why [`Client::run`] failed.
#[derive(Debug)]
pub enum RunError {
    /// Resolving the κ over the network seam / local store failed.
    Resolve(SyncError),
    /// The κ resolved to nothing locally or over the network.
    NotFound,
    /// Loading or executing the compiled workload failed.
    Exec(ExecError),
}

impl core::fmt::Display for RunError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RunError::Resolve(e) => write!(f, "resolve failed: {e:?}"),
            RunError::NotFound => f.write_str("workload κ not found locally or over the network"),
            RunError::Exec(e) => write!(f, "execute failed: {e:?}"),
        }
    }
}
