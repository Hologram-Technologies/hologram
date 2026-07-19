//! The **container-engine seam** (spec §4.2/§4.4): the Wasm-instance trait a backend
//! (Wasmtime, an interpreter, or a test mock) implements, plus the two supporting
//! structs the runtime hands across it — [`HostContext`] (the host import surface at
//! instantiation) and [`ContainerIntents`] (the side effects a container requested).
//!
//! This is part of the space **contract**: the runtime orchestration is generic over
//! `ContainerEngine`, and every backend implements it identically. It links nothing from
//! the tensor compute engine (RZ) and stays `no_std + alloc`.

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::{KappaLabel71, KappaStore, RealizationRegistry, RuntimeError};

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
    /// `cpu_time_per_event_ms` (§7.5 fuel checkpoint) — see `hologram_runtime::FUEL_PER_MS`.
    pub cpu_fuel_per_event: u64,
    /// Total bytes this container's `storage_put` may persist; **0 = unbounded** (§7.6).
    pub storage_quota_bytes: u64,
}

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
