#![cfg_attr(not(feature = "std"), no_std)]
//! # hologram-runtime
//!
//! The **substrate-portable Container Runtime orchestration** (spec §4). The uor-native parts of
//! the runtime — container identity, the spawn/suspend/resume/terminate lifecycle, snapshots as
//! κ-labels, and **capability enforcement** (the `admits` containment at delegation) — live here,
//! over two seams: a [`ContainerEngine`] (the Wasm instance; Wasmtime/interpreter is a backend) and
//! a `KappaStore`. None of this links the tensor compute engine (RZ).
//!
//! The orchestration is engine-agnostic, so it is validated hermetically against a mock engine; the
//! same orchestration drives a real Wasmtime engine unchanged (substrate-tripling at the runtime).

extern crate alloc;

use alloc::boxed::Box; // async-trait emits `Box` unqualified; bring it into no_std scope.
use alloc::sync::Arc;
use alloc::vec::Vec;

use hashbrown::{HashMap, HashSet};
use hologram_realizations::{
    CapabilitySet, ChainCompaction, ContainerManifest, Delegation, ErrorEvent, Snapshot,
};
use hologram_substrate_core::{
    Capabilities, ContainerHandle, ContainerInfo, ContainerState, KappaLabel71, KappaStore,
    KappaSync, Realization, RealizationRegistry, RuntimeError,
};
use spin::Mutex;

/// Host-side context handed to a [`ContainerEngine`] at instantiation so a container's import
/// surface (`storage_get` / `storage_put` / `log`, spec §4.4) can reach the node's store —
/// **capability-gated** by the container's storage roots at the import boundary (§10.4). The engine
/// wires these into its host-function bindings; the mock engine ignores them.
pub struct HostContext {
    pub store: Arc<dyn KappaStore>,
    pub storage_roots: Vec<KappaLabel71>,
    pub registry: RealizationRegistry<'static>,
    /// Max Wasm linear memory in bytes; **0 = unbounded** (no arbitrary cap, SPINE-6).
    pub memory_max_bytes: u64,
    /// Wasm fuel budget per event/call; **0 = unbounded**. The deterministic analog of
    /// `cpu_time_per_event_ms` (§7.5 fuel checkpoint) — see [`FUEL_PER_MS`].
    pub cpu_fuel_per_event: u64,
    /// Total bytes this container's `storage_put` may persist; **0 = unbounded** (§7.6).
    pub storage_quota_bytes: u64,
}

/// Calibration of the deterministic fuel ↔ CPU-time proxy: Wasm fuel is an instruction count, not
/// wall-clock, so `cpu_time_per_event_ms` is converted to a fuel budget at this rate (§7.5). It is a
/// calibration, not an arbitrary ceiling — `0 ms` stays unbounded.
pub const FUEL_PER_MS: u64 = 1_000_000;

/// The Wasm-instance seam (spec §4.2/§4.4). A backend (Wasmtime, an interpreter, or the test mock)
/// implements it; the runtime orchestration above is identical across all of them.
pub trait ContainerEngine: Send + Sync {
    type Instance: Send;

    /// Instantiate a container from its Wasm code bytes, wiring its host import surface against
    /// `ctx` (storage/log, capability-gated). `ctx` is ignored by engines that host import-free
    /// containers (e.g. the mock).
    fn instantiate(&self, code: &[u8], ctx: &HostContext) -> Result<Self::Instance, RuntimeError>;
    /// `hg_init` — called once with the initial-state bytes. Returns a status code (0 = ok).
    fn init(&self, inst: &mut Self::Instance, initial_state: &[u8]) -> u32;
    /// `hg_event` — dispatch one event (its κ-label bytes).
    fn event(&self, inst: &mut Self::Instance, event_kappa: &[u8]) -> u32;
    /// `hg_suspend` — flush before snapshot.
    fn suspend(&self, inst: &mut Self::Instance) -> u32;
    /// `hg_resume` — after linear memory is restored.
    fn resume(&self, inst: &mut Self::Instance) -> u32;
    /// `hg_callback` — deliver a subscription/continuation payload (spec §4.4). `payload_kappa` is
    /// the published κ-label's bytes.
    fn callback(&self, inst: &mut Self::Instance, callback_id: u32, payload_kappa: &[u8]) -> u32;
    /// Drain (and clear) the `publish`/`subscribe` intents a container issued via its import surface
    /// during the last call. The runtime applies them **with capability enforcement** (§10.4), so
    /// the engine itself does no gating. Engines whose containers cannot issue intents return empty.
    fn drain_intents(&self, _inst: &mut Self::Instance) -> ContainerIntents {
        ContainerIntents::default()
    }
    /// Serialize linear memory + globals for the snapshot payload (spec §4.7).
    fn snapshot_memory(&self, inst: &Self::Instance) -> Vec<u8>;
    /// Restore linear memory + globals from a snapshot payload.
    fn restore_memory(&self, inst: &mut Self::Instance, mem: &[u8]);
    /// Current linear-memory size (for `ContainerInfo`/budgets).
    fn memory_bytes(&self, inst: &Self::Instance) -> u64;
    /// Bytes already charged against this container's storage quota (arch §11.6). Carried through
    /// the `Snapshot` payload so the ledger survives suspend/resume; default 0 for engines that
    /// don't enforce a storage quota.
    fn storage_used(&self, _inst: &Self::Instance) -> u64 {
        0
    }
    /// Restore the storage-quota ledger from a resumed snapshot (arch §11.6).
    fn restore_storage_used(&self, _inst: &mut Self::Instance, _used: u64) {}
}

struct Entry<I> {
    inst: I,
    info: ContainerInfo,
    caps: Capabilities,
}

/// Side effects a container requested through its import surface during the last call. The runtime
/// applies them **with capability enforcement** (§10.4) after the engine returns, so the engine
/// itself does no gating. The intent-buffer pattern keeps the container ABI synchronous (Wasm host
/// functions are sync) while async network/spawn work runs after.
#[derive(Default)]
pub struct ContainerIntents {
    /// `(channel κ, payload κ)` pairs the container published (`publish` import).
    pub published: Vec<(KappaLabel71, KappaLabel71)>,
    /// `(channel κ, callback_id)` pairs the container subscribed (`subscribe` import).
    pub subscribed: Vec<(KappaLabel71, u32)>,
    /// κ-labels the container asked to **announce** to the network (`sync_announce` import,
    /// spec §4.4 + arch §11.1). Applied by the runtime via `KappaSync::announce` after the call.
    pub announces: Vec<KappaLabel71>,
    /// κ-labels the container asked to **fetch** from the network (`sync_fetch_request` import).
    /// Applied via `KappaSync::fetch` + verify-on-receipt + local-store cache; the resolved bytes
    /// become visible to the next event via `storage_get` (no sync-on-async deadlock).
    pub fetches: Vec<KappaLabel71>,
    /// `(child container-id κ, child capability-set κ)` pairs the container requested to spawn as
    /// a child (`spawn_child` import). Applied via the runtime's `spawn_child` (admits check).
    pub child_spawns: Vec<(KappaLabel71, KappaLabel71)>,
    /// Structured diagnostic events the container raised (`diagnostics` import, spec §7.5). Each is
    /// `(classification, code, optional context κ)`. The runtime mints an `ErrorEvent` realization
    /// per intent and threads it into the source container's error-log chain.
    pub diagnostics: Vec<(u8, u32, Option<KappaLabel71>)>,
}

/// A persistent subscription (spec §10.11). Keyed by **Container ID** (not the ephemeral handle) so
/// it survives suspend/resume; `cursor` is the per-channel last-delivered position.
struct Sub {
    container_id: [u8; 71],
    channel: [u8; 71],
    callback_id: u32,
    cursor: usize,
}

/// Per-container event queue + deficit counter for the DRR scheduler (arch §11.7). Idle queues
/// don't accumulate deficit — fairness is over containers with pending work, not over absolute time.
#[derive(Default)]
struct EventQueue {
    pending: Vec<Vec<u8>>,
    deficit: u64,
}

/// The substrate-portable runtime over an engine `E` and a store `S`.
pub struct Runtime<E: ContainerEngine, S: KappaStore> {
    engine: E,
    store: Arc<S>,
    table: Mutex<HashMap<u64, Entry<E::Instance>>>,
    revoked: Mutex<HashSet<[u8; 71]>>,
    // Channel bus (spec §4.4): per-channel ordered published payloads + persistent subscriptions.
    channels: Mutex<HashMap<[u8; 71], Vec<KappaLabel71>>>,
    subs: Mutex<Vec<Sub>>,
    // Per-container DRR queues — populated by `enqueue_event`, drained by `pump_round` (arch §11.7).
    queues: Mutex<HashMap<u64, EventQueue>>,
    // A spin Mutex counter, not AtomicU64 — Cortex-M (thumbv7em) has no 64-bit atomics (G-D1).
    next: Mutex<u64>,
    /// Optional network layer (spec §6) — wired by `with_sync(...)` to enable `sync_announce` /
    /// `sync_fetch_request` container imports + auto-fetch on event delivery. `None` ⇒ no-network
    /// runtime (the existing hermetic semantics).
    sync: Option<Arc<dyn KappaSync>>,
    /// Per-container error-log chain heads (Container ID → most-recent ErrorEvent κ). Threaded
    /// through `predecessor` so the append-only history is recoverable (SPINE-3 / spec §7.5).
    error_log_heads: Mutex<HashMap<[u8; 71], KappaLabel71>>,
    /// Per-container error-log chain *depth* since the last compaction (arch §9 G-C4 → §11). When
    /// the depth would exceed `error_log_threshold`, the next `emit_diagnostic` folds the chain
    /// into a `ChainCompaction` κ (no operands → GC reclaims the old tail) and resets to 0.
    error_log_depth: Mutex<HashMap<[u8; 71], u32>>,
    /// Chain-compaction threshold (depth at which the next event's predecessor becomes a
    /// `ChainCompaction` barrier rather than the prior head). `0 ⇒ unbounded` (SPINE-6, opt-in).
    /// Default `DEFAULT_ERROR_LOG_THRESHOLD = 128`.
    error_log_threshold: Mutex<u32>,
    /// Async network intents queued by `deliver_event`. The caller drives the actual `KappaSync`
    /// calls via [`Runtime::process_pending_network`] (a tick of the network event loop).
    pending_network: Mutex<Vec<NetworkIntent>>,
}

/// Default error-log chain depth before the next emit folds the tail into a `ChainCompaction`
/// barrier. Operators tune via [`Runtime::set_error_log_threshold`]; `0` disables compaction
/// (the architecture's `[track]`-resolved policy for G-C4 — unbounded only by explicit opt-in).
pub const DEFAULT_ERROR_LOG_THRESHOLD: u32 = 128;

/// `ErrorEvent` classification space (the `class:u8` field of the payload). The architecture
/// reserves the low byte; runtime-emitted denials use the `0x10..=0x1F` band so an auditor can
/// tell a container's own `diagnostics(...)` (default 0x01) from a runtime-minted capability
/// refusal at-a-glance. Append-only — never renumber an existing class (SPINE-5).
pub mod diag_class {
    /// Container-emitted diagnostic via the `diagnostics(class, code, ctx)` host import.
    pub const CONTAINER_EMITTED: u8 = 0x01;
    /// Runtime refused a `publish` intent (channel not in `publish_channels`).
    pub const PUBLISH_DENIED: u8 = 0x10;
    /// Runtime refused a `subscribe` intent (channel not in `subscribe_channels`).
    pub const SUBSCRIBE_DENIED: u8 = 0x11;
    /// Runtime refused a `spawn_child` intent (caps not admitted under parent — `admits` failed).
    pub const SPAWN_CHILD_DENIED: u8 = 0x12;
}

/// A network side-effect the container requested through its `sync_*` imports (spec §4.4). The
/// runtime queues these synchronously during `deliver_event` and applies them via
/// [`Runtime::process_pending_network`] on the network event loop.
#[derive(Clone, Debug)]
pub enum NetworkIntent {
    /// `sync_announce` — best-effort `KappaSync::announce`.
    Announce(KappaLabel71),
    /// `sync_fetch_request` — `KappaSync::fetch` + verify-on-receipt + local-store cache. The
    /// resolved bytes become visible to the next event via `storage_get`.
    Fetch(KappaLabel71),
}

fn be(_e: hologram_substrate_core::StoreError) -> RuntimeError {
    RuntimeError::BackendFailure("store")
}

impl<E: ContainerEngine + 'static, S: KappaStore + 'static> Runtime<E, S> {
    pub fn new(engine: E, store: S) -> Self {
        Self {
            engine,
            store: Arc::new(store),
            table: Mutex::new(HashMap::new()),
            revoked: Mutex::new(HashSet::new()),
            channels: Mutex::new(HashMap::new()),
            subs: Mutex::new(Vec::new()),
            queues: Mutex::new(HashMap::new()),
            next: Mutex::new(1),
            sync: None,
            error_log_heads: Mutex::new(HashMap::new()),
            error_log_depth: Mutex::new(HashMap::new()),
            error_log_threshold: Mutex::new(DEFAULT_ERROR_LOG_THRESHOLD),
            pending_network: Mutex::new(Vec::new()),
        }
    }

    /// Set the error-log chain-compaction threshold (arch §9 G-C4 → §11): when the depth of a
    /// container's error-log chain would exceed `threshold`, the next `emit_diagnostic` folds
    /// the tail into a `ChainCompaction` κ. `0` ⇒ unbounded (opt-in, SPINE-6).
    pub fn set_error_log_threshold(&self, threshold: u32) {
        *self.error_log_threshold.lock() = threshold;
    }

    /// Wire a network layer (a [`KappaSync`]) so containers' `sync_announce` / `sync_fetch_request`
    /// imports become live and the runtime auto-fetches event payload κs that aren't local. Calling
    /// twice replaces the prior sync (the last writer wins, by design — federation chains compose
    /// inside the supplied sync).
    pub fn with_sync(mut self, sync: Arc<dyn KappaSync>) -> Self {
        self.sync = Some(sync);
        self
    }

    /// Publish a payload κ to a channel κ (spec §4.4) — capability-gated on the publisher's
    /// `publish_channels`. Structurally adds a Route κ (endpoint=channel, target=payload) to the
    /// graph, appends to the channel's ordered history, **announces the Route κ to the network
    /// layer (F1, cross-peer fanout) if a sync is wired**, and delivers to current subscribers.
    /// Returns the Route κ-label.
    pub fn publish(
        &self,
        publisher: ContainerHandle,
        channel: &KappaLabel71,
        payload: &KappaLabel71,
    ) -> Result<KappaLabel71, RuntimeError> {
        {
            let table = self.table.lock();
            let e = table
                .get(&publisher.0)
                .ok_or(RuntimeError::ContainerIdNotFound)?;
            if self
                .revoked
                .lock()
                .contains(e.info.capabilities_kappa.as_array())
            {
                return Err(RuntimeError::CapabilityVerificationFailed);
            }
            if !e.caps.publish_channels.contains(channel) {
                return Err(RuntimeError::CapabilityVerificationFailed); // §10.4
            }
        }
        self.channels
            .lock()
            .entry(*channel.as_array())
            .or_default()
            .push(*payload);
        let route = hologram_realizations::Route {
            endpoint: *channel,
            target: *payload,
        };
        let route_k = self
            .store
            .put("blake3", &route.canonicalize())
            .map_err(be)?;
        // F1 — cross-peer fanout: queue an `announce(route_k)` intent if a network layer is wired.
        // The next `process_pending_network` tick announces over `KappaSync::announce`; peers
        // subscribed to this channel pull the Route via `poll_channel_fanout`.
        if self.sync.is_some() {
            self.pending_network
                .lock()
                .push(NetworkIntent::Announce(route_k));
        }
        self.pump();
        Ok(route_k)
    }

    /// Poll the network for new Route κs targeting `channel`, deliver them to local subscribers.
    /// This is the **subscriber side** of cross-peer channel fanout (F1) — the symmetric counterpart
    /// to `publish`'s `announce`. Returns the number of remote Routes newly delivered locally.
    ///
    /// Implementation: discover the κs the network currently advertises (DHT/peer set),
    /// `get_with_fetch` each, verify on receipt (σ-axis re-derivation, SPINE-4), and append any
    /// whose `Route.endpoint == channel` to the local channel history. The existing `pump()`
    /// then walks the per-subscription cursors and delivers to running containers.
    pub async fn poll_channel_fanout(&self, channel: &KappaLabel71) -> Result<usize, RuntimeError> {
        self.poll_channel_fanout_with_limit(channel, usize::MAX)
            .await
    }

    /// Same as [`poll_channel_fanout`] with an explicit `limit` on the number of candidate κs
    /// to consider in one pass. `usize::MAX` ⇒ no cap (the structural cap is whatever the
    /// underlying `KappaSync::discover` returns). The caller picks the bound; this method
    /// imposes none of its own (SPINE-6).
    pub async fn poll_channel_fanout_with_limit(
        &self,
        channel: &KappaLabel71,
        limit: usize,
    ) -> Result<usize, RuntimeError> {
        let Some(sync) = &self.sync else {
            return Ok(0);
        };
        // The DHT prefix hint is the full channel κ — peers may match on any prefix length, so
        // we hand them the whole thing and let them decide. (No magic-number prefix slice.)
        let prefix_bytes = channel.as_array();
        let candidates = sync.discover(Some(prefix_bytes), limit).await;
        let mut delivered = 0usize;
        for k in candidates {
            // Fetch + verify-on-receipt + cache locally.
            let fetched =
                hologram_substrate_core::get_with_fetch(self.store.as_ref(), sync.as_ref(), &k)
                    .await;
            let Ok(Some(bytes)) = fetched else { continue };
            // Try parsing as a Route; deliver only if its endpoint matches `channel`.
            if let Ok(refs) = hologram_realizations::Route::references(bytes.as_ref()) {
                if refs.len() == 2 && refs[0] == *channel {
                    let target = refs[1];
                    let mut channels = self.channels.lock();
                    let log = channels.entry(*channel.as_array()).or_default();
                    if !log.contains(&target) {
                        log.push(target);
                        delivered += 1;
                    }
                }
            }
        }
        self.pump();
        Ok(delivered)
    }

    /// Subscribe a container to a channel κ (capability-gated on `subscribe_channels`). The
    /// subscription is persistent (keyed by Container ID, §10.11); the container receives every κ
    /// published to the channel via `hg_callback`, including the backlog and any published during a
    /// suspension (replayed on resume).
    pub fn subscribe(
        &self,
        subscriber: ContainerHandle,
        channel: &KappaLabel71,
        callback_id: u32,
    ) -> Result<(), RuntimeError> {
        let container_id = {
            let table = self.table.lock();
            let e = table
                .get(&subscriber.0)
                .ok_or(RuntimeError::ContainerIdNotFound)?;
            if !e.caps.subscribe_channels.contains(channel) {
                return Err(RuntimeError::CapabilityVerificationFailed); // §10.4
            }
            *e.info.container_id.as_array()
        };
        self.subs.lock().push(Sub {
            container_id,
            channel: *channel.as_array(),
            callback_id,
            cursor: 0,
        });
        self.pump();
        Ok(())
    }

    /// Deliver any pending channel messages to **running** subscriber instances via `hg_callback`,
    /// advancing per-subscription cursors. Suspended containers accumulate (cursor frozen) and
    /// receive the backlog on resume (§10.11).
    fn pump(&self) {
        let mut deliveries: Vec<(u64, u32, KappaLabel71)> = Vec::new();
        {
            let channels = self.channels.lock();
            let mut subs = self.subs.lock();
            let table = self.table.lock();
            for sub in subs.iter_mut() {
                let running = table.iter().find(|(_, e)| {
                    e.info.container_id.as_array() == &sub.container_id
                        && e.info.state == ContainerState::Running
                });
                if let Some((h, _)) = running {
                    if let Some(log) = channels.get(&sub.channel) {
                        while sub.cursor < log.len() {
                            deliveries.push((*h, sub.callback_id, log[sub.cursor]));
                            sub.cursor += 1;
                        }
                    }
                }
            }
        }
        let mut table = self.table.lock();
        for (h, cb, payload) in deliveries {
            if let Some(e) = table.get_mut(&h) {
                self.engine.callback(&mut e.inst, cb, payload.as_array());
            }
        }
    }

    fn next_handle(&self) -> u64 {
        let mut n = self.next.lock();
        let h = *n;
        *n += 1;
        h
    }

    /// Expose the runtime's store as a shared `Arc` so external infrastructure (e.g. a
    /// cross-peer KappaSync adapter, a serving HTTP-CAS endpoint) can read the same store the
    /// runtime is writing to.
    pub fn store_arc(&self) -> Arc<S> {
        self.store.clone()
    }

    pub fn store(&self) -> &S {
        self.store.as_ref()
    }

    /// Resolve + decode a capability-set κ-label to its [`Capabilities`] view, refusing a revoked
    /// set (spec §4.5 / §10.12).
    fn resolve_caps(&self, caps_kappa: &KappaLabel71) -> Result<Capabilities, RuntimeError> {
        if self.revoked.lock().contains(caps_kappa.as_array()) {
            return Err(RuntimeError::CapabilityVerificationFailed);
        }
        let bytes = self
            .store
            .get(caps_kappa)
            .map_err(be)?
            .ok_or(RuntimeError::CapabilityVerificationFailed)?;
        CapabilitySet::to_capabilities(bytes.as_ref())
            .map_err(|_| RuntimeError::CapabilityVerificationFailed)
    }

    fn instantiate_from(
        &self,
        container_id: &KappaLabel71,
        caps_kappa: &KappaLabel71,
        caps: Capabilities,
        restore: Option<(&[u8], u64)>,
        current_snapshot: Option<KappaLabel71>,
    ) -> Result<ContainerHandle, RuntimeError> {
        let manifest = self
            .store
            .get(container_id)
            .map_err(be)?
            .ok_or(RuntimeError::ContainerIdNotFound)?;
        let refs = ContainerManifest::references(manifest.as_ref())
            .map_err(|_| RuntimeError::SnapshotInvalid)?;
        let code_k = refs.first().ok_or(RuntimeError::ContainerIdNotFound)?;
        let code = self
            .store
            .get(code_k)
            .map_err(be)?
            .ok_or(RuntimeError::InstantiationFailed("code not present"))?;

        let ctx = HostContext {
            store: self.store.clone(),
            storage_roots: caps.storage_roots.clone(),
            registry: hologram_realizations::REGISTRY,
            memory_max_bytes: caps.memory_max_bytes,
            cpu_fuel_per_event: caps.cpu_time_per_event_ms.saturating_mul(FUEL_PER_MS),
            storage_quota_bytes: caps.storage_quota_bytes,
        };
        let mut inst = self.engine.instantiate(code.as_ref(), &ctx)?;
        match restore {
            Some((mem, storage_used)) => {
                self.engine.restore_memory(&mut inst, mem);
                self.engine.restore_storage_used(&mut inst, storage_used);
                self.engine.resume(&mut inst);
            }
            None => {
                // initial state is the manifest's second operand, if present locally.
                let state = refs.get(1).and_then(|k| self.store.get(k).ok().flatten());
                self.engine.init(&mut inst, state.as_deref().unwrap_or(&[]));
            }
        }

        let handle = ContainerHandle(self.next_handle());
        let info = ContainerInfo {
            container_id: *container_id,
            capabilities_kappa: *caps_kappa,
            current_snapshot,
            state: ContainerState::Running,
            memory_bytes: self.engine.memory_bytes(&inst),
        };
        self.table
            .lock()
            .insert(handle.0, Entry { inst, info, caps });
        Ok(handle)
    }

    /// Deliver an event to a running container (`hg_event`). Refuses if its capability set has been
    /// revoked (spec §10.12).
    pub fn deliver_event(
        &self,
        handle: ContainerHandle,
        event_kappa: &[u8],
    ) -> Result<u32, RuntimeError> {
        let mut table = self.table.lock();
        let entry = table
            .get_mut(&handle.0)
            .ok_or(RuntimeError::ContainerIdNotFound)?;
        if self
            .revoked
            .lock()
            .contains(entry.info.capabilities_kappa.as_array())
        {
            return Err(RuntimeError::CapabilityVerificationFailed);
        }
        let code = self.engine.event(&mut entry.inst, event_kappa);
        entry.info.memory_bytes = self.engine.memory_bytes(&entry.inst);
        let intents = self.engine.drain_intents(&mut entry.inst);
        let source_container_id = entry.info.container_id;
        drop(table);
        // Apply the container's channel intents with capability enforcement (§10.4). A denied
        // intent is **not** observable to the container itself (capability hygiene — leaking a
        // denial would let the container probe for channels it can't access), but it IS minted
        // as a runtime ErrorEvent so the substrate operator sees it in the audit trail. The
        // diagnostic's `class` distinguishes container-emitted (0x01) from runtime denials
        // (0x10..0x12). This closes the prior silent-drop hole (SPINE-6 audit-trail rule).
        for (channel, payload) in intents.published {
            if self.publish(handle, &channel, &payload).is_err() {
                let _ = self.emit_diagnostic(
                    &source_container_id,
                    diag_class::PUBLISH_DENIED,
                    0,
                    Some(channel),
                );
                let _ = payload; // payload is observed only on success
            }
        }
        for (channel, callback_id) in intents.subscribed {
            if self.subscribe(handle, &channel, callback_id).is_err() {
                let _ = self.emit_diagnostic(
                    &source_container_id,
                    diag_class::SUBSCRIBE_DENIED,
                    callback_id,
                    Some(channel),
                );
            }
        }
        // Apply spawn_child intents with delegation containment (admits check) inside spawn_child.
        for (cid, caps) in intents.child_spawns {
            if self.spawn_child(handle, &cid, &caps).is_err() {
                let _ = self.emit_diagnostic(
                    &source_container_id,
                    diag_class::SPAWN_CHILD_DENIED,
                    0,
                    Some(cid),
                );
            }
        }
        // Apply diagnostics: mint an ErrorEvent realization, thread the predecessor (SPINE-3 chain),
        // put it in the store. The Container ID identifies the source; the chain head moves forward.
        for (class, code_field, ctx) in intents.diagnostics {
            let _ = self.emit_diagnostic(&source_container_id, class, code_field, ctx);
        }
        // Queue async network intents — the network event loop drives them via
        // `process_pending_network`. The container's storage_get on the next event sees fetched
        // bytes; no synchronous-on-async deadlock.
        {
            let mut q = self.pending_network.lock();
            for k in intents.announces {
                q.push(NetworkIntent::Announce(k));
            }
            for k in intents.fetches {
                q.push(NetworkIntent::Fetch(k));
            }
        }
        Ok(code)
    }

    /// Mint an [`ErrorEvent`] realization for a diagnostic the container raised (spec §7.5), thread
    /// it into the source container's append-only error log (predecessor = prior head), put into
    /// the store. Returns the new chain head.
    pub fn emit_diagnostic(
        &self,
        source_container_id: &KappaLabel71,
        classification: u8,
        code: u32,
        context: Option<KappaLabel71>,
    ) -> Result<KappaLabel71, RuntimeError> {
        let mut heads = self.error_log_heads.lock();
        let mut depths = self.error_log_depth.lock();
        let threshold = *self.error_log_threshold.lock();
        let prior_head = heads.get(source_container_id.as_array()).copied();
        let current_depth = *depths.get(source_container_id.as_array()).unwrap_or(&0);

        // Chain-compaction barrier (G-C4): when the depth has reached the threshold, the next
        // event's predecessor becomes a `ChainCompaction` κ that *breaks* the predecessor chain.
        // The old tail is no longer reachable from any pinned root, so the store's reachability
        // GC will reclaim it (SPINE-5). The barrier's payload is content-bound to the boundary
        // (head κ + depth) — an auditor sees the count and the boundary κ but not the entries.
        let (predecessor, new_depth) =
            if threshold > 0 && current_depth >= threshold && prior_head.is_some() {
                let mut summary_input: Vec<u8> = Vec::with_capacity(71 + 4);
                if let Some(h) = &prior_head {
                    summary_input.extend_from_slice(h.as_array());
                }
                summary_input.extend_from_slice(&current_depth.to_le_bytes());
                let barrier = ChainCompaction {
                    fold_count: current_depth,
                    // The on-the-wire 71-byte κ-label form is the content-bound summary — same
                    // determinism as a raw blake3 digest, no hex-decode round-trip required.
                    boundary: hologram_substrate_core::address_bytes(&summary_input),
                };
                let barrier_kappa = self
                    .store
                    .put("blake3", &barrier.canonicalize())
                    .map_err(be)?;
                (Some(barrier_kappa), 1u32)
            } else {
                (prior_head, current_depth.saturating_add(1))
            };

        let mut payload = Vec::with_capacity(5);
        payload.push(classification);
        payload.extend_from_slice(&code.to_le_bytes());
        let event = ErrorEvent {
            source: *source_container_id,
            predecessor,
            context,
            class_code_payload: payload,
        };
        let k = self
            .store
            .put("blake3", &event.canonicalize())
            .map_err(be)?;
        heads.insert(*source_container_id.as_array(), k);
        depths.insert(*source_container_id.as_array(), new_depth);
        Ok(k)
    }

    /// Drive one tick of the network event loop: apply every queued `sync_announce` and
    /// `sync_fetch_request` intent (spec §6 / arch §11.1). Idempotent — pending stay queued until a
    /// `sync` is wired via [`with_sync`]; once wired, every intent is applied. The next event a
    /// container processes can `storage_get` newly-fetched κs from the local store.
    pub async fn process_pending_network(&self) -> usize {
        let Some(sync) = &self.sync else {
            return 0;
        };
        let intents: Vec<NetworkIntent> = {
            let mut q = self.pending_network.lock();
            core::mem::take(&mut *q)
        };
        let count = intents.len();
        for intent in intents {
            match intent {
                NetworkIntent::Announce(k) => sync.announce(&k).await,
                NetworkIntent::Fetch(k) => {
                    // `get_with_fetch` performs verify-on-receipt + caches the bytes locally.
                    let _ = hologram_substrate_core::get_with_fetch(
                        self.store.as_ref(),
                        sync.as_ref(),
                        &k,
                    )
                    .await;
                }
            }
        }
        count
    }

    /// Test introspection: how many network intents are queued (post-deliver_event, pre-pump).
    pub fn pending_network_count(&self) -> usize {
        self.pending_network.lock().len()
    }

    /// Spawn a **child** container with a derived capability set — the enforcement point for
    /// delegation (spec §4.5 / §10.7). The runtime refuses unless the parent **admits** the
    /// derived set (`grants(child) ⊆ grants(parent)`, the SubtypingLattice relation).
    pub fn spawn_child(
        &self,
        parent: ContainerHandle,
        child_container_id: &KappaLabel71,
        derived_caps_kappa: &KappaLabel71,
    ) -> Result<ContainerHandle, RuntimeError> {
        let (parent_caps, parent_caps_kappa) = {
            let table = self.table.lock();
            let entry = table
                .get(&parent.0)
                .ok_or(RuntimeError::ContainerIdNotFound)?;
            (entry.caps.clone(), entry.info.capabilities_kappa)
        };
        let derived = self.resolve_caps(derived_caps_kappa)?;
        if !parent_caps.admits(&derived) {
            return Err(RuntimeError::CapabilityVerificationFailed);
        }
        // Express the parent → child delegation as a κ-graph edge (arch §11.8). The Delegation κ
        // lives in the store; revoke walks the inverse projection to cascade through descendants.
        let delegation = Delegation {
            parent_caps: parent_caps_kappa,
            child_caps: *derived_caps_kappa,
        };
        let _ = self
            .store
            .put("blake3", &delegation.canonicalize())
            .map_err(be)?;
        self.instantiate_from(child_container_id, derived_caps_kappa, derived, None, None)
    }

    /// Compute the **transitive delegation cone** of `root_caps` — every `child_caps` reachable
    /// from `root_caps` via `Delegation{parent,child}` edges in the κ-graph. The κ-graph IS the
    /// authority (no side-channel map); on each call we recover the cone by walking the store's
    /// Delegation realizations (arch §11.8).
    fn delegation_cone(&self, root_caps: &KappaLabel71) -> HashSet<[u8; 71]> {
        let mut cone: HashSet<[u8; 71]> = HashSet::new();
        cone.insert(*root_caps.as_array());
        // Snapshot the Delegation edges currently in the store.
        let mut edges: Vec<([u8; 71], [u8; 71])> = Vec::new();
        for k in self.store.iterate() {
            if let Ok(Some(b)) = self.store.get(&k) {
                let bytes = b.as_ref();
                if bytes.starts_with(Delegation::IRI.as_bytes()) {
                    if let Ok(refs) = <Delegation as Realization>::references(bytes) {
                        if refs.len() == 2 {
                            edges.push((*refs[0].as_array(), *refs[1].as_array()));
                        }
                    }
                }
            }
        }
        // Fixpoint: repeatedly add child_caps for edges whose parent is in the cone.
        let mut changed = true;
        while changed {
            changed = false;
            for (p, c) in &edges {
                if cone.contains(p) && !cone.contains(c) {
                    cone.insert(*c);
                    changed = true;
                }
            }
        }
        cone
    }

    /// Enqueue an event for fair DRR delivery via [`pump_round`] (arch §11.7) — the multi-tenant
    /// scheduling path. The direct [`deliver_event`] still works for single-container immediate
    /// delivery; this is the entry point when many containers compete for cycles.
    pub fn enqueue_event(&self, h: ContainerHandle, payload: Vec<u8>) -> Result<(), RuntimeError> {
        let table = self.table.lock();
        if !table.contains_key(&h.0) {
            return Err(RuntimeError::ContainerIdNotFound);
        }
        drop(table);
        self.queues
            .lock()
            .entry(h.0)
            .or_default()
            .pending
            .push(payload);
        Ok(())
    }

    /// Run one **deficit round-robin** scheduling round (arch §11.7). For each container with
    /// queued events (visited in handle / UorTime order — ADR-058, **no wall-clock**), add
    /// `priority_weight × quantum` deficit and dispatch events while the deficit covers their cost
    /// (one unit per event). Idle queues don't accumulate deficit, so a periodically-quiet
    /// container can't burst. Returns the `(handle, event_return_code)` tuples actually delivered
    /// this round.
    pub fn pump_round(&self, quantum: u32) -> Vec<(ContainerHandle, u32)> {
        let mut delivered = Vec::new();
        // Snapshot handles in UorTime order (handles are minted monotonically; sorting is
        // deterministic given that ordering).
        let mut handles: Vec<u64> = self.table.lock().keys().copied().collect();
        handles.sort_unstable();

        for h in handles {
            // Skip revoked containers — no work for a denied principal.
            let (weight, revoked) = {
                let table = self.table.lock();
                let Some(e) = table.get(&h) else {
                    continue;
                };
                let r = self
                    .revoked
                    .lock()
                    .contains(e.info.capabilities_kappa.as_array());
                (e.caps.priority_weight.max(1) as u64, r)
            };
            if revoked {
                continue;
            }

            // Pull pending events under deficit budget.
            let mut to_deliver: Vec<Vec<u8>> = Vec::new();
            {
                let mut queues = self.queues.lock();
                let q = queues.entry(h).or_default();
                if q.pending.is_empty() {
                    q.deficit = 0; // idle: reset (no burst on next ready)
                    continue;
                }
                q.deficit = q.deficit.saturating_add(weight * quantum as u64);
                while q.deficit > 0 && !q.pending.is_empty() {
                    q.deficit -= 1;
                    to_deliver.push(q.pending.remove(0));
                }
            }
            for payload in to_deliver {
                if let Ok(code) = self.deliver_event(ContainerHandle(h), &payload) {
                    delivered.push((ContainerHandle(h), code));
                }
            }
        }
        delivered
    }

    /// Revoke a capability set (spec §10.12 + arch §11.8): subsequent imports/events from the
    /// revoked caps **and from every descendant minted under it via `spawn_child`** are refused.
    /// The κ-labels it already produced remain in the store (append-only); only future operations
    /// authorized by the revoked cone are denied.
    pub fn revoke(&self, caps_kappa: &KappaLabel71) {
        let cone = self.delegation_cone(caps_kappa);
        let mut revoked = self.revoked.lock();
        revoked.extend(cone);
    }
}

#[async_trait::async_trait]
impl<E: ContainerEngine + 'static, S: KappaStore + 'static>
    hologram_substrate_core::ContainerRuntime for Runtime<E, S>
{
    async fn spawn(
        &self,
        container_id: &KappaLabel71,
        capabilities: &KappaLabel71,
    ) -> Result<ContainerHandle, RuntimeError> {
        let caps = self.resolve_caps(capabilities)?;
        self.instantiate_from(container_id, capabilities, caps, None, None)
    }

    async fn suspend(&self, handle: ContainerHandle) -> Result<KappaLabel71, RuntimeError> {
        let mut table = self.table.lock();
        let entry = table
            .get_mut(&handle.0)
            .ok_or(RuntimeError::ContainerIdNotFound)?;
        self.engine.suspend(&mut entry.inst);
        let mem = self.engine.snapshot_memory(&entry.inst);
        let storage_used = self.engine.storage_used(&entry.inst);
        let snapshot = Snapshot {
            container_id: entry.info.container_id,
            previous: entry.info.current_snapshot,
            storage_used,
            state_payload: mem,
        };
        let snapshot_k = self
            .store
            .put("blake3", &snapshot.canonicalize())
            .map_err(be)?;
        entry.info.current_snapshot = Some(snapshot_k);
        entry.info.state = ContainerState::Suspended;
        Ok(snapshot_k)
    }

    async fn resume(
        &self,
        snapshot: &KappaLabel71,
        capabilities: &KappaLabel71,
    ) -> Result<ContainerHandle, RuntimeError> {
        let caps = self.resolve_caps(capabilities)?;
        let snap_bytes = self
            .store
            .get(snapshot)
            .map_err(be)?
            .ok_or(RuntimeError::SnapshotInvalid)?;
        let refs =
            Snapshot::references(snap_bytes.as_ref()).map_err(|_| RuntimeError::SnapshotInvalid)?;
        let container_id = *refs.first().ok_or(RuntimeError::SnapshotInvalid)?;
        let payload = hologram_realizations::payload_of(Snapshot::IRI, snap_bytes.as_ref())
            .map_err(|_| RuntimeError::SnapshotInvalid)?;
        let (storage_used, mem) =
            Snapshot::parse_payload(&payload).map_err(|_| RuntimeError::SnapshotInvalid)?;
        let mem = mem.to_vec();
        let handle = self.instantiate_from(
            &container_id,
            capabilities,
            caps,
            Some((&mem, storage_used)),
            Some(*snapshot),
        )?;
        // Replay any channel messages published while this container was suspended (§10.11).
        self.pump();
        Ok(handle)
    }

    async fn terminate(&self, handle: ContainerHandle) -> Result<(), RuntimeError> {
        self.table.lock().remove(&handle.0);
        Ok(())
    }

    fn list(&self) -> Vec<ContainerHandle> {
        self.table
            .lock()
            .keys()
            .map(|k| ContainerHandle(*k))
            .collect()
    }

    fn info(&self, handle: ContainerHandle) -> Option<ContainerInfo> {
        self.table.lock().get(&handle.0).map(|e| e.info.clone())
    }
}

// ───────────────────────────── test mock engine ─────────────────────────────

/// A deterministic in-process [`ContainerEngine`] for hermetic V&V: linear memory is a `Vec<u8>`,
/// `init` seeds it, `event` mutates it, snapshot/restore copy it. No Wasm — the runtime
/// orchestration is exactly what's under test.
#[derive(Default)]
pub struct MockEngine;

impl<S: KappaStore + 'static> Runtime<MockEngine, S> {
    /// Test helper (mock engine only): the `(callback_id, payload_kappa_bytes)` delivered to a
    /// container instance via `hg_callback` — lets a test observe channel delivery.
    pub fn delivered_callbacks(&self, handle: ContainerHandle) -> Vec<(u32, Vec<u8>)> {
        self.table
            .lock()
            .get(&handle.0)
            .map(|e| e.inst.callbacks.clone())
            .unwrap_or_default()
    }
}

/// A mock container instance: its linear memory + a call log + received callbacks.
pub struct MockInstance {
    pub memory: Vec<u8>,
    pub log: Vec<&'static str>,
    /// `(callback_id, payload_kappa_bytes)` delivered via [`ContainerEngine::callback`] — lets a
    /// test observe channel delivery without a real Wasm `hg_callback`.
    pub callbacks: Vec<(u32, Vec<u8>)>,
}

impl ContainerEngine for MockEngine {
    type Instance = MockInstance;
    fn instantiate(&self, _code: &[u8], _ctx: &HostContext) -> Result<MockInstance, RuntimeError> {
        Ok(MockInstance {
            memory: Vec::new(),
            log: Vec::new(),
            callbacks: Vec::new(),
        })
    }
    fn init(&self, inst: &mut MockInstance, initial_state: &[u8]) -> u32 {
        inst.memory = initial_state.to_vec();
        inst.log.push("init");
        0
    }
    fn event(&self, inst: &mut MockInstance, event_kappa: &[u8]) -> u32 {
        inst.memory.extend_from_slice(event_kappa);
        inst.log.push("event");
        0
    }
    fn suspend(&self, inst: &mut MockInstance) -> u32 {
        inst.log.push("suspend");
        0
    }
    fn resume(&self, inst: &mut MockInstance) -> u32 {
        inst.log.push("resume");
        0
    }
    fn callback(&self, inst: &mut MockInstance, callback_id: u32, payload_kappa: &[u8]) -> u32 {
        inst.callbacks.push((callback_id, payload_kappa.to_vec()));
        0
    }
    fn snapshot_memory(&self, inst: &MockInstance) -> Vec<u8> {
        inst.memory.clone()
    }
    fn restore_memory(&self, inst: &mut MockInstance, mem: &[u8]) {
        inst.memory = mem.to_vec();
    }
    fn memory_bytes(&self, inst: &MockInstance) -> u64 {
        inst.memory.len() as u64
    }
}
