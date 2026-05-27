//! # hologram-runtime-wasmtime
//!
//! The production [`hologram_runtime::ContainerEngine`] backend (spec §7.2
//! Option A): real Wasm execution via Wasmtime + Cranelift, with linear-memory snapshot/restore
//! **and the host import surface** wired through a `Linker` (spec §4.4), **capability-gated at the
//! import boundary** (§10.4). Plugging this into `hologram_runtime::Runtime` makes
//! `spawn`/`suspend`/`resume` execute an actual container that can read and write the κ-graph — the
//! same orchestration validated against the mock engine, now over real Wasm with real I/O.
//!
//! **Container ABI (this engine):** the module exports `memory` and `hg_init(ptr,len)->i32`,
//! `hg_event(ptr,len)->i32`, `hg_suspend()->i32`, `hg_resume()->i32`. The engine writes input bytes
//! at offset 0 and calls the export with `(0, len)`. Imports (module `"hologram"`):
//! - `log(level, ptr, len)` — record a message.
//! - `storage_put(ptr, len, out_ptr) -> i32` — put `mem[ptr..ptr+len]`; write the 71-byte κ-label
//!   to `mem[out_ptr..]`; return 0 (or -1).
//! - `storage_get(kappa_ptr, out_ptr, out_cap) -> i32` — read the 71-byte κ at `kappa_ptr`; **only
//!   if it is reachable from the container's storage roots** (§4.5/§10.4); copy up to `out_cap`
//!   bytes to `out_ptr`; return the length, or -1 (absent or capability-denied).

use std::collections::HashSet;
use std::sync::Arc;

use hologram_runtime::{ContainerEngine, ContainerIntents, HostContext};
use hologram_substrate_core::{
    references, KappaLabel, KappaLabel71, KappaStore, RealizationRegistry, RuntimeError,
};
use wasmtime::{Caller, Engine, Extern, Instance, Linker, Memory, Module, Store, TypedFunc};

pub mod block;
pub use block::WasmBlockDevice;

const PAGE: usize = 64 * 1024;

/// Per-instance host state backing the import surface.
struct HostState {
    store: Arc<dyn KappaStore>,
    roots: Vec<KappaLabel71>,
    registry: RealizationRegistry<'static>,
    log: Vec<(u32, Vec<u8>)>,
    /// Buffered channel intents (applied, capability-gated, by the runtime after the call).
    published: Vec<(KappaLabel71, KappaLabel71)>,
    subscribed: Vec<(KappaLabel71, u32)>,
    /// UorTime is computational (ADR-058): a monotonic per-engine progress counter, not wall-clock.
    rewrite_steps: u64,
    /// Entropy stream state (splitmix64; a production backend uses a hardware CSPRNG, spec §8.2).
    rng: u64,
    /// Resource accounting (§7.6): linear-memory limiter, storage-quota ledger, per-event fuel.
    limits: wasmtime::StoreLimits,
    storage_quota: u64, // 0 = unbounded
    storage_used: u64,
}

fn read_kappa(mem: &Memory, caller: &Caller<'_, HostState>, ptr: i32) -> Option<KappaLabel71> {
    mem.data(caller)
        .get(ptr as usize..ptr as usize + 71)
        .and_then(|s| <[u8; 71]>::try_from(s).ok())
        .and_then(|a| KappaLabel::from_bytes(&a).ok())
}

/// Is `target` reachable from `roots` via `references()` — the capability read-closure (§4.5).
fn in_closure(
    store: &dyn KappaStore,
    registry: RealizationRegistry<'_>,
    roots: &[KappaLabel71],
    target: &KappaLabel71,
) -> bool {
    let mut seen: HashSet<[u8; 71]> = HashSet::new();
    let mut frontier: Vec<KappaLabel71> = roots.to_vec();
    while let Some(k) = frontier.pop() {
        if &k == target {
            return true;
        }
        if !seen.insert(*k.as_array()) {
            continue;
        }
        if let Ok(Some(b)) = store.get(&k) {
            if let Ok(refs) = references(b.as_ref(), registry) {
                frontier.extend(refs);
            }
        }
    }
    false
}

fn mem_of(caller: &mut Caller<'_, HostState>) -> Option<Memory> {
    match caller.get_export("memory") {
        Some(Extern::Memory(m)) => Some(m),
        _ => None,
    }
}

/// A Wasmtime-backed container engine.
pub struct WasmtimeEngine {
    engine: Engine,
}

impl Default for WasmtimeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmtimeEngine {
    pub fn new() -> Self {
        // Fuel metering on (§7.5/§7.6 CPU bound); memory bound via per-instance StoreLimits.
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config).expect("wasmtime config");
        Self { engine }
    }

    fn linker(&self) -> Result<Linker<HostState>, RuntimeError> {
        let mut linker = Linker::new(&self.engine);
        let ifail = |_| RuntimeError::InstantiationFailed("linker");

        linker
            .func_wrap(
                "hologram",
                "log",
                |mut caller: Caller<'_, HostState>, level: i32, ptr: i32, len: i32| {
                    if let Some(mem) = mem_of(&mut caller) {
                        if let Some(s) = mem
                            .data(&caller)
                            .get(ptr as usize..ptr as usize + len as usize)
                        {
                            let v = s.to_vec();
                            caller.data_mut().log.push((level as u32, v));
                        }
                    }
                },
            )
            .map_err(ifail)?;

        linker
            .func_wrap(
                "hologram",
                "storage_put",
                |mut caller: Caller<'_, HostState>, ptr: i32, len: i32, out_ptr: i32| -> i32 {
                    let Some(mem) = mem_of(&mut caller) else {
                        return -1;
                    };
                    let Some(input) = mem
                        .data(&caller)
                        .get(ptr as usize..ptr as usize + len as usize)
                        .map(|s| s.to_vec())
                    else {
                        return -1;
                    };
                    // Storage quota (§7.6): refuse if this put would exceed the container's budget
                    // (0 = unbounded). Idempotent re-puts of already-stored bytes don't re-charge.
                    {
                        let s = caller.data_mut();
                        let kappa = hologram_substrate_core::address_bytes(&input);
                        let already = s.store.contains(&kappa);
                        if s.storage_quota != 0
                            && !already
                            && s.storage_used + input.len() as u64 > s.storage_quota
                        {
                            return -1; // QuotaExceeded
                        }
                        if !already {
                            s.storage_used += input.len() as u64;
                        }
                    }
                    let store = caller.data().store.clone();
                    let kappa = match store.put("blake3", &input) {
                        Ok(k) => k,
                        Err(_) => return -1,
                    };
                    let dst = mem.data_mut(&mut caller);
                    match dst.get_mut(out_ptr as usize..out_ptr as usize + 71) {
                        Some(slot) => {
                            slot.copy_from_slice(kappa.as_array());
                            0
                        }
                        None => -1,
                    }
                },
            )
            .map_err(ifail)?;

        linker
            .func_wrap(
                "hologram",
                "storage_get",
                |mut caller: Caller<'_, HostState>,
                 kappa_ptr: i32,
                 out_ptr: i32,
                 out_cap: i32|
                 -> i32 {
                    let Some(mem) = mem_of(&mut caller) else {
                        return -1;
                    };
                    let Some(karr) = mem
                        .data(&caller)
                        .get(kappa_ptr as usize..kappa_ptr as usize + 71)
                        .and_then(|s| <[u8; 71]>::try_from(s).ok())
                    else {
                        return -1;
                    };
                    let Ok(kappa) = KappaLabel::from_bytes(&karr) else {
                        return -1;
                    };
                    let store = caller.data().store.clone();
                    let roots = caller.data().roots.clone();
                    let registry = caller.data().registry;
                    // Capability gate (§10.4): refuse a κ outside the container's read-closure.
                    if !in_closure(store.as_ref(), registry, &roots, &kappa) {
                        return -1;
                    }
                    let bytes = match store.get(&kappa) {
                        Ok(Some(b)) => b,
                        _ => return -1,
                    };
                    let n = core::cmp::min(bytes.len(), out_cap as usize);
                    let dst = mem.data_mut(&mut caller);
                    match dst.get_mut(out_ptr as usize..out_ptr as usize + n) {
                        Some(slot) => {
                            slot.copy_from_slice(&bytes[..n]);
                            n as i32
                        }
                        None => -1,
                    }
                },
            )
            .map_err(ifail)?;

        // storage_contains(kappa_ptr) -> i32 (1 present / 0 absent / -1 malformed)
        linker
            .func_wrap(
                "hologram",
                "storage_contains",
                |mut caller: Caller<'_, HostState>, kappa_ptr: i32| -> i32 {
                    let Some(mem) = mem_of(&mut caller) else {
                        return -1;
                    };
                    let Some(k) = read_kappa(&mem, &caller, kappa_ptr) else {
                        return -1;
                    };
                    i32::from(caller.data().store.contains(&k))
                },
            )
            .map_err(ifail)?;

        // storage_pin / storage_unpin (kappa_ptr) -> i32 (0 ok / -1 err)
        linker
            .func_wrap(
                "hologram",
                "storage_pin",
                |mut caller: Caller<'_, HostState>, kappa_ptr: i32| -> i32 {
                    let Some(mem) = mem_of(&mut caller) else {
                        return -1;
                    };
                    let Some(k) = read_kappa(&mem, &caller, kappa_ptr) else {
                        return -1;
                    };
                    if caller.data().store.pin(&k).is_ok() {
                        0
                    } else {
                        -1
                    }
                },
            )
            .map_err(ifail)?;
        linker
            .func_wrap(
                "hologram",
                "storage_unpin",
                |mut caller: Caller<'_, HostState>, kappa_ptr: i32| -> i32 {
                    let Some(mem) = mem_of(&mut caller) else {
                        return -1;
                    };
                    let Some(k) = read_kappa(&mem, &caller, kappa_ptr) else {
                        return -1;
                    };
                    if caller.data().store.unpin(&k).is_ok() {
                        0
                    } else {
                        -1
                    }
                },
            )
            .map_err(ifail)?;

        // publish(channel_ptr, kappa_ptr) — buffer the intent; the runtime applies it gated (§10.4).
        linker
            .func_wrap(
                "hologram",
                "publish",
                |mut caller: Caller<'_, HostState>, channel_ptr: i32, kappa_ptr: i32| {
                    let Some(mem) = mem_of(&mut caller) else {
                        return;
                    };
                    let (Some(ch), Some(payload)) = (
                        read_kappa(&mem, &caller, channel_ptr),
                        read_kappa(&mem, &caller, kappa_ptr),
                    ) else {
                        return;
                    };
                    caller.data_mut().published.push((ch, payload));
                },
            )
            .map_err(ifail)?;

        // subscribe(channel_ptr, callback_id) — buffer the intent.
        linker
            .func_wrap(
                "hologram",
                "subscribe",
                |mut caller: Caller<'_, HostState>, channel_ptr: i32, callback_id: i32| {
                    let Some(mem) = mem_of(&mut caller) else {
                        return;
                    };
                    let Some(ch) = read_kappa(&mem, &caller, channel_ptr) else {
                        return;
                    };
                    caller.data_mut().subscribed.push((ch, callback_id as u32));
                },
            )
            .map_err(ifail)?;

        // time_now(out_ptr) — write 16-byte canonical UorTime [landauer-nats f64 ‖ rewrite-steps u64]
        // (ADR-058 computational time; monotonic per-engine progress, never wall-clock).
        linker
            .func_wrap(
                "hologram",
                "time_now",
                |mut caller: Caller<'_, HostState>, out_ptr: i32| {
                    let steps = {
                        let s = caller.data_mut();
                        s.rewrite_steps += 1;
                        s.rewrite_steps
                    };
                    let Some(mem) = mem_of(&mut caller) else {
                        return;
                    };
                    let mut buf = [0u8; 16];
                    buf[..8].copy_from_slice(&(steps as f64).to_le_bytes());
                    buf[8..].copy_from_slice(&steps.to_le_bytes());
                    if let Some(slot) = mem
                        .data_mut(&mut caller)
                        .get_mut(out_ptr as usize..out_ptr as usize + 16)
                    {
                        slot.copy_from_slice(&buf);
                    }
                },
            )
            .map_err(ifail)?;

        // entropy(out_ptr, len) — fill `len` bytes from the instance's stream.
        linker
            .func_wrap(
                "hologram",
                "entropy",
                |mut caller: Caller<'_, HostState>, out_ptr: i32, len: i32| {
                    let mut bytes = Vec::with_capacity(len as usize);
                    {
                        let s = caller.data_mut();
                        for _ in 0..len {
                            // splitmix64
                            s.rng = s.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
                            let mut z = s.rng;
                            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                            bytes.push((z ^ (z >> 31)) as u8);
                        }
                    }
                    let Some(mem) = mem_of(&mut caller) else {
                        return;
                    };
                    if let Some(slot) = mem
                        .data_mut(&mut caller)
                        .get_mut(out_ptr as usize..out_ptr as usize + len as usize)
                    {
                        slot.copy_from_slice(&bytes);
                    }
                },
            )
            .map_err(ifail)?;

        Ok(linker)
    }
}

/// A live Wasm container instance: its `Store` (host state), the module instance, and `memory`.
pub struct WasmInstance {
    store: Store<HostState>,
    instance: Instance,
    memory: Memory,
    /// Per-event fuel budget (§7.6); 0 = unbounded. Reset before each lifecycle call so a single
    /// event that exceeds it traps (fuel-exhausted `PipelineFailure`, §7.5) instead of running away.
    cpu_fuel: u64,
}

impl WasmInstance {
    fn ensure_capacity(&mut self, need: usize) {
        let have = self.memory.data_size(&self.store);
        if need > have {
            let extra_pages = (need - have).div_ceil(PAGE) as u64;
            let _ = self.memory.grow(&mut self.store, extra_pages);
        }
    }

    /// Refuel for one lifecycle call (CPU bound, §7.6). Fuel metering is always on, so an
    /// "unbounded" budget (0) is realized as `u64::MAX` rather than leaving the store starved.
    fn refuel(&mut self) {
        let fuel = if self.cpu_fuel > 0 {
            self.cpu_fuel
        } else {
            u64::MAX
        };
        let _ = self.store.set_fuel(fuel);
    }

    fn call_io(&mut self, name: &str, bytes: &[u8]) -> u32 {
        if !bytes.is_empty() {
            self.ensure_capacity(bytes.len());
            self.memory.data_mut(&mut self.store)[..bytes.len()].copy_from_slice(bytes);
        }
        self.refuel();
        let f: Result<TypedFunc<(i32, i32), i32>, _> =
            self.instance.get_typed_func(&mut self.store, name);
        match f {
            Ok(f) => f
                .call(&mut self.store, (0, bytes.len() as i32))
                .map(|c| c as u32)
                .unwrap_or(1),
            Err(_) => 1,
        }
    }

    fn call_unit(&mut self, name: &str) -> u32 {
        self.refuel();
        let f: Result<TypedFunc<(), i32>, _> = self.instance.get_typed_func(&mut self.store, name);
        match f {
            Ok(f) => f.call(&mut self.store, ()).map(|c| c as u32).unwrap_or(1),
            Err(_) => 0,
        }
    }

    /// Logs the container emitted via the `log` import (for diagnostics/tests).
    pub fn logs(&self) -> &[(u32, Vec<u8>)] {
        &self.store.data().log
    }
}

impl ContainerEngine for WasmtimeEngine {
    type Instance = WasmInstance;

    fn instantiate(&self, code: &[u8], ctx: &HostContext) -> Result<WasmInstance, RuntimeError> {
        let module = Module::new(&self.engine, code)
            .map_err(|_| RuntimeError::InstantiationFailed("invalid wasm module"))?;
        let host = HostState {
            store: ctx.store.clone(),
            roots: ctx.storage_roots.clone(),
            registry: ctx.registry,
            log: Vec::new(),
            published: Vec::new(),
            subscribed: Vec::new(),
            rewrite_steps: 0,
            // Seed the entropy stream from the container's identity-ish context (deterministic per
            // instance here; a production backend seeds from a hardware RNG, spec §8.2).
            rng: ctx.storage_roots.len() as u64 ^ 0xD1B5_4A32_D192_ED03,
            // Memory bound (§7.6): cap linear-memory growth at memory_max_bytes; 0 = unbounded.
            limits: if ctx.memory_max_bytes > 0 {
                wasmtime::StoreLimitsBuilder::new()
                    .memory_size(ctx.memory_max_bytes as usize)
                    .build()
            } else {
                wasmtime::StoreLimits::default()
            },
            storage_quota: ctx.storage_quota_bytes,
            storage_used: 0,
        };
        let mut store = Store::new(&self.engine, host);
        store.limiter(|s| &mut s.limits);
        // Fuel metering is always on, so seed fuel for instantiation (a start function may run);
        // unbounded (0) → u64::MAX. `refuel` resets per-event fuel before each lifecycle call.
        let _ = store.set_fuel(if ctx.cpu_fuel_per_event > 0 {
            ctx.cpu_fuel_per_event
        } else {
            u64::MAX
        });
        let linker = self.linker()?;
        let instance = linker.instantiate(&mut store, &module).map_err(|_| {
            RuntimeError::InstantiationFailed("instantiation trapped (memory cap or trap)")
        })?;
        let memory =
            instance
                .get_memory(&mut store, "memory")
                .ok_or(RuntimeError::InstantiationFailed(
                    "module exports no `memory`",
                ))?;
        Ok(WasmInstance {
            store,
            instance,
            memory,
            cpu_fuel: ctx.cpu_fuel_per_event,
        })
    }

    fn init(&self, inst: &mut WasmInstance, initial_state: &[u8]) -> u32 {
        inst.call_io("hg_init", initial_state)
    }
    fn event(&self, inst: &mut WasmInstance, event_kappa: &[u8]) -> u32 {
        inst.call_io("hg_event", event_kappa)
    }
    fn suspend(&self, inst: &mut WasmInstance) -> u32 {
        inst.call_unit("hg_suspend")
    }
    fn resume(&self, inst: &mut WasmInstance) -> u32 {
        inst.call_unit("hg_resume")
    }
    fn callback(&self, inst: &mut WasmInstance, callback_id: u32, payload_kappa: &[u8]) -> u32 {
        if !payload_kappa.is_empty() {
            inst.ensure_capacity(payload_kappa.len());
            inst.memory.data_mut(&mut inst.store)[..payload_kappa.len()]
                .copy_from_slice(payload_kappa);
        }
        inst.refuel();
        let f: Result<TypedFunc<(i32, i32, i32), i32>, _> =
            inst.instance.get_typed_func(&mut inst.store, "hg_callback");
        match f {
            Ok(f) => f
                .call(
                    &mut inst.store,
                    (callback_id as i32, 0, payload_kappa.len() as i32),
                )
                .map(|c| c as u32)
                .unwrap_or(1),
            Err(_) => 0,
        }
    }
    fn snapshot_memory(&self, inst: &WasmInstance) -> Vec<u8> {
        inst.memory.data(&inst.store).to_vec()
    }
    fn restore_memory(&self, inst: &mut WasmInstance, mem: &[u8]) {
        inst.ensure_capacity(mem.len());
        inst.memory.data_mut(&mut inst.store)[..mem.len()].copy_from_slice(mem);
    }
    fn memory_bytes(&self, inst: &WasmInstance) -> u64 {
        inst.memory.data_size(&inst.store) as u64
    }
    fn drain_intents(&self, inst: &mut WasmInstance) -> ContainerIntents {
        let s = inst.store.data_mut();
        ContainerIntents {
            published: core::mem::take(&mut s.published),
            subscribed: core::mem::take(&mut s.subscribed),
        }
    }
}
