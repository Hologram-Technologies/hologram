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
use hologram_realizations::{CapabilitySet, ContainerManifest, Snapshot};
use hologram_substrate_core::{
    Capabilities, ContainerHandle, ContainerInfo, ContainerState, KappaLabel71, KappaStore,
    Realization, RealizationRegistry, RuntimeError,
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
}

struct Entry<I> {
    inst: I,
    info: ContainerInfo,
    caps: Capabilities,
}

/// Channel `publish`/`subscribe` calls a container made through its import surface, buffered by the
/// engine and applied (capability-gated) by the runtime after the call.
#[derive(Default)]
pub struct ContainerIntents {
    /// `(channel κ, payload κ)` pairs the container published.
    pub published: Vec<(KappaLabel71, KappaLabel71)>,
    /// `(channel κ, callback_id)` pairs the container subscribed.
    pub subscribed: Vec<(KappaLabel71, u32)>,
}

/// A persistent subscription (spec §10.11). Keyed by **Container ID** (not the ephemeral handle) so
/// it survives suspend/resume; `cursor` is the per-channel last-delivered position.
struct Sub {
    container_id: [u8; 71],
    channel: [u8; 71],
    callback_id: u32,
    cursor: usize,
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
    // A spin Mutex counter, not AtomicU64 — Cortex-M (thumbv7em) has no 64-bit atomics (G-D1).
    next: Mutex<u64>,
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
            next: Mutex::new(1),
        }
    }

    /// Publish a payload κ to a channel κ (spec §4.4) — capability-gated on the publisher's
    /// `publish_channels`. Structurally adds a Route κ (endpoint=channel, target=payload) to the
    /// graph, appends to the channel's ordered history, and delivers to current subscribers.
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
        self.pump();
        Ok(route_k)
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
        restore: Option<&[u8]>,
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
            Some(mem) => {
                self.engine.restore_memory(&mut inst, mem);
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
        drop(table);
        // Apply the container's channel intents with capability enforcement (§10.4); an
        // unauthorized publish/subscribe is silently dropped (the gate returns Err).
        for (channel, payload) in intents.published {
            let _ = self.publish(handle, &channel, &payload);
        }
        for (channel, callback_id) in intents.subscribed {
            let _ = self.subscribe(handle, &channel, callback_id);
        }
        Ok(code)
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
        let parent_caps = {
            let table = self.table.lock();
            table
                .get(&parent.0)
                .ok_or(RuntimeError::ContainerIdNotFound)?
                .caps
                .clone()
        };
        let derived = self.resolve_caps(derived_caps_kappa)?;
        if !parent_caps.admits(&derived) {
            return Err(RuntimeError::CapabilityVerificationFailed);
        }
        self.instantiate_from(child_container_id, derived_caps_kappa, derived, None, None)
    }

    /// Revoke a capability set (spec §10.12): subsequent imports/events from any container holding
    /// it are refused. The κ-labels it already produced remain (append-only).
    pub fn revoke(&self, caps_kappa: &KappaLabel71) {
        self.revoked.lock().insert(*caps_kappa.as_array());
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
        let snapshot = Snapshot {
            container_id: entry.info.container_id,
            previous: entry.info.current_snapshot,
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
        let mem = hologram_realizations::payload_of(Snapshot::IRI, snap_bytes.as_ref())
            .map_err(|_| RuntimeError::SnapshotInvalid)?;
        let handle = self.instantiate_from(
            &container_id,
            capabilities,
            caps,
            Some(&mem),
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
