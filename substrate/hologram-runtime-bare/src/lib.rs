#![cfg_attr(not(feature = "std"), no_std)]
//! # hologram-runtime-bare
//!
//! The bare-metal [`ContainerEngine`] (architecture §2 / C1). Symmetric to
//! `hologram-runtime-wasmtime` but no_std: a real **wasmi**-based Wasm interpreter that the
//! bare-metal substrate uses to run containers without an OS-hosted compiler. The `Runtime`
//! orchestration in `hologram-runtime` is engine-agnostic, so the same orchestration drives
//! Wasmtime on std hosts and `BareMetalEngine` on bare-metal.
//!
//! ## What this crate provides
//! - [`BareMetalEngine`] — a [`ContainerEngine`] implementation over [`wasmi`].
//! - `hg_init` / `hg_event` / `hg_suspend` / `hg_resume` / `hg_callback` exports are looked up by
//!   name on instantiation; calls trap into the interpreter.
//! - Linear-memory snapshot/restore for the canonical `Snapshot` payload (state continuity).
//!
//! ## Host imports
//! The bare-metal substrate's `hologram.*` host imports are bound by the **hosting binary**
//! (e.g. `hologram-efi`) against its imported NIC / BlockDevice drivers. At the engine seam
//! here, the [`wasmi::Linker`] is empty — a container that *declares* any `hologram.*` imports
//! will be refused by `instantiate_and_start` with a link error (fail-loud per SPINE-6: no
//! silent no-op imports that would let a container *think* it called storage_put and discover
//! later that nothing happened). This is symmetric with the workload-bound gate (B5) which
//! refuses imports outside `hologram.*` entirely.
//!
//! ## What this crate does NOT provide
//! - WASI imports — the substrate's container ABI is the `hologram.*` import surface only
//!   (SPINE-6 / G-C3; see `hologram-runtime-wasmtime/tests/workload_bounds.rs`).
//! - JIT — bare-metal is interpreted by design; the architecture trades peak throughput for
//!   no_std + audit-by-construction.

extern crate alloc;

use alloc::vec::Vec;

use hologram_runtime::{ContainerEngine, ContainerIntents, HostContext};
use hologram_substrate_core::RuntimeError;
use spin::Mutex;
use wasmi::{Engine as WasmiEngine, Func, Instance, Linker, Memory, Module, Store, TypedFunc};

/// The no_std-friendly [`ContainerEngine`]. Holds a single `wasmi::Engine` (reusable across
/// containers) — the per-container `Store` is allocated in [`instantiate`].
pub struct BareMetalEngine {
    inner: WasmiEngine,
}

impl Default for BareMetalEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BareMetalEngine {
    /// A new bare-metal engine. `wasmi`'s default config is no_std-safe; we do not enable any
    /// features that depend on the host.
    pub fn new() -> Self {
        Self {
            inner: WasmiEngine::default(),
        }
    }
}

/// Per-container state. The interpreter `Store` holds a `()` host data slot — `BareMetalEngine`
/// does not expose host imports (the bare-metal substrate's import surface is added in the
/// `Linker` here when the substrate wires its host functions; this crate provides the seam, not
/// the surface, mirroring the std runtime's separation of orchestration from imports).
pub struct WasmiInstance {
    store: Store<()>,
    memory: Memory,
    hg_init: Option<TypedFunc<(i32, i32), i32>>,
    hg_event: Option<TypedFunc<(i32, i32), i32>>,
    hg_suspend: Option<TypedFunc<(), i32>>,
    hg_resume: Option<TypedFunc<(), i32>>,
    hg_callback: Option<TypedFunc<(i32, i32, i32), i32>>,
    /// Bytes already attributed to this container's storage quota (mirrored from `HostContext`).
    storage_used: u64,
    /// Intents the container raised through its import surface during the last call. The bare-
    /// metal substrate's import surface (when wired) appends here; the runtime drains via
    /// `drain_intents`. The minimal seam keeps this empty (no imports yet).
    intents: Mutex<ContainerIntents>,
    /// A scratch region for events that need it (publish-payload bytes, fetched κs).
    event_scratch: Vec<u8>,
}

fn ifail(_: impl core::fmt::Debug) -> RuntimeError {
    RuntimeError::InstantiationFailed("bare-metal Wasm instantiation")
}

impl ContainerEngine for BareMetalEngine {
    type Instance = WasmiInstance;

    fn instantiate(&self, code: &[u8], _ctx: &HostContext) -> Result<Self::Instance, RuntimeError> {
        // SPINE-6 / G-C3 workload-bounds guard, symmetric with the wasmtime engine: refuse any
        // import outside the `hologram.*` host surface. Even with no imports wired here, the
        // refusal must be structural.
        let module = Module::new(&self.inner, code).map_err(ifail)?;
        for imp in module.imports() {
            if imp.module() != "hologram" {
                return Err(RuntimeError::InstantiationFailed(
                    "bare-metal: container declares an import outside the spec §4.4 host surface",
                ));
            }
        }
        let mut store = Store::new(&self.inner, ());
        // The bare-metal substrate's `hologram.*` host-import surface is supplied by the hosting
        // binary at link time (each NIC/BlockDevice driver is itself a codemodule κ; the binary
        // wires the §4.4 imports against its bound devices). At the engine-seam level here, the
        // Linker is empty — so a container that **declares** any `hologram.*` imports will be
        // refused by `instantiate_and_start` with a link error. That's fail-loud (SPINE-6):
        // either the host binary has wired the matching import, or instantiation refuses. There
        // is no silent no-op import that would let a container *think* it called storage_put
        // and discover later that nothing happened.
        let linker: Linker<()> = Linker::new(&self.inner);
        let instance: Instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(ifail)?;
        let memory =
            instance
                .get_memory(&store, "memory")
                .ok_or(RuntimeError::InstantiationFailed(
                    "bare-metal: module exports no `memory`",
                ))?;
        let lookup = |s: &mut Store<()>, name: &str| -> Option<Func> {
            instance.get_export(&*s, name).and_then(|e| e.into_func())
        };
        let hg_init =
            lookup(&mut store, "hg_init").and_then(|f| f.typed::<(i32, i32), i32>(&store).ok());
        let hg_event =
            lookup(&mut store, "hg_event").and_then(|f| f.typed::<(i32, i32), i32>(&store).ok());
        let hg_suspend =
            lookup(&mut store, "hg_suspend").and_then(|f| f.typed::<(), i32>(&store).ok());
        let hg_resume =
            lookup(&mut store, "hg_resume").and_then(|f| f.typed::<(), i32>(&store).ok());
        let hg_callback = lookup(&mut store, "hg_callback")
            .and_then(|f| f.typed::<(i32, i32, i32), i32>(&store).ok());
        Ok(WasmiInstance {
            store,
            memory,
            hg_init,
            hg_event,
            hg_suspend,
            hg_resume,
            hg_callback,
            storage_used: 0,
            intents: Mutex::new(ContainerIntents::default()),
            event_scratch: Vec::new(),
        })
    }

    fn init(&self, inst: &mut Self::Instance, initial_state: &[u8]) -> u32 {
        // Stage initial state in the scratch region; pass (ptr, len) to hg_init.
        let off = stage_in_memory(&mut inst.memory, &mut inst.store, initial_state);
        match (&inst.hg_init, off) {
            (Some(f), Some((ptr, len))) => f
                .call(&mut inst.store, (ptr, len))
                .map(|v| v as u32)
                .unwrap_or(u32::MAX),
            (Some(f), None) => f
                .call(&mut inst.store, (0, 0))
                .map(|v| v as u32)
                .unwrap_or(u32::MAX),
            _ => 0, // no hg_init exported is OK (some containers have no init)
        }
    }

    fn event(&self, inst: &mut Self::Instance, event_kappa: &[u8]) -> u32 {
        inst.event_scratch.clear();
        inst.event_scratch.extend_from_slice(event_kappa);
        let off = stage_in_memory(&mut inst.memory, &mut inst.store, &inst.event_scratch);
        match (&inst.hg_event, off) {
            (Some(f), Some((ptr, len))) => f
                .call(&mut inst.store, (ptr, len))
                .map(|v| v as u32)
                .unwrap_or(u32::MAX),
            _ => 0,
        }
    }

    fn suspend(&self, inst: &mut Self::Instance) -> u32 {
        // Distinguish "no hg_suspend exported" (= 0 = ok) from "trap on call" (= u32::MAX).
        // SPINE-6: a trap is fail-loud; the caller sees a non-zero status, not silent success.
        match &inst.hg_suspend {
            Some(f) => match f.call(&mut inst.store, ()) {
                Ok(v) => v as u32,
                Err(_) => u32::MAX,
            },
            None => 0,
        }
    }

    fn resume(&self, inst: &mut Self::Instance) -> u32 {
        match &inst.hg_resume {
            Some(f) => match f.call(&mut inst.store, ()) {
                Ok(v) => v as u32,
                Err(_) => u32::MAX,
            },
            None => 0,
        }
    }

    fn callback(&self, inst: &mut Self::Instance, callback_id: u32, payload_kappa: &[u8]) -> u32 {
        let off = stage_in_memory(&mut inst.memory, &mut inst.store, payload_kappa);
        match (&inst.hg_callback, off) {
            (Some(f), Some((ptr, len))) => f
                .call(&mut inst.store, (callback_id as i32, ptr, len))
                .map(|v| v as u32)
                .unwrap_or(u32::MAX),
            _ => 0,
        }
    }

    fn drain_intents(&self, inst: &mut Self::Instance) -> ContainerIntents {
        core::mem::take(&mut *inst.intents.lock())
    }

    fn snapshot_memory(&self, inst: &Self::Instance) -> Vec<u8> {
        let data = inst.memory.data(&inst.store);
        let mut out = Vec::with_capacity(data.len());
        out.extend_from_slice(data);
        out
    }

    fn restore_memory(&self, inst: &mut Self::Instance, mem: &[u8]) {
        let dst = inst.memory.data_mut(&mut inst.store);
        let n = mem.len().min(dst.len());
        dst[..n].copy_from_slice(&mem[..n]);
    }

    fn memory_bytes(&self, inst: &Self::Instance) -> u64 {
        inst.memory.data(&inst.store).len() as u64
    }

    fn storage_used(&self, inst: &Self::Instance) -> u64 {
        inst.storage_used
    }

    fn restore_storage_used(&self, inst: &mut Self::Instance, used: u64) {
        inst.storage_used = used;
    }
}

/// Stage `bytes` at offset `0x1000` in `memory` (above a container's typical globals section);
/// return `(ptr, len)` for the call. Returns `None` if memory is too small to hold the staged
/// bytes — the structural cap is the container's own linear-memory size, not a policy constant.
fn stage_in_memory(memory: &mut Memory, store: &mut Store<()>, bytes: &[u8]) -> Option<(i32, i32)> {
    const SCRATCH: usize = 0x1000;
    let data = memory.data_mut(&mut *store);
    if SCRATCH + bytes.len() > data.len() {
        return None;
    }
    data[SCRATCH..SCRATCH + bytes.len()].copy_from_slice(bytes);
    Some((SCRATCH as i32, bytes.len() as i32))
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use hologram_realizations::REGISTRY;
    use hologram_runtime::Runtime;
    use hologram_store_mem::MemKappaStore;
    use hologram_substrate_core::{Capabilities, ContainerRuntime, KappaStore, Realization};

    /// Minimal counter container — same shape as the wasmtime test. Proves the bare-metal engine
    /// runs real Wasm modules end-to-end.
    const COUNTER_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "hg_init")    (param i32 i32) (result i32) (i32.const 0))
      (func (export "hg_suspend") (result i32) (i32.const 0))
      (func (export "hg_resume")  (result i32) (i32.const 0))
      (func (export "hg_event") (param i32 i32) (result i32)
        (i32.store8 (i32.const 0)
          (i32.add (i32.load8_u (i32.const 0)) (i32.const 1)))
        (i32.const 0)))
    "#;

    fn wasm() -> Vec<u8> {
        wat::parse_str(COUNTER_WAT).expect("valid wat")
    }

    fn ctx() -> HostContext {
        use alloc::sync::Arc;
        HostContext {
            store: Arc::new(MemKappaStore::new()),
            storage_roots: alloc::vec![],
            registry: REGISTRY,
            memory_max_bytes: 0,
            cpu_fuel_per_event: 0,
            storage_quota_bytes: 0,
        }
    }

    #[test]
    fn bare_engine_executes_wasm_and_snapshots_memory() {
        let engine = BareMetalEngine::new();
        let mut inst = engine.instantiate(&wasm(), &ctx()).unwrap();
        assert_eq!(engine.init(&mut inst, &[]), 0);
        for _ in 0..3 {
            assert_eq!(engine.event(&mut inst, &[]), 0);
        }
        let snap = engine.snapshot_memory(&inst);
        assert_eq!(snap[0], 3, "hg_event incremented the counter three times");

        // Restore into a fresh instance — state continuity over real Wasm memory.
        let mut inst2 = engine.instantiate(&wasm(), &ctx()).unwrap();
        engine.restore_memory(&mut inst2, &snap);
        assert_eq!(engine.snapshot_memory(&inst2)[0], 3);
        engine.event(&mut inst2, &[]);
        assert_eq!(engine.snapshot_memory(&inst2)[0], 4);
    }

    #[test]
    fn bare_engine_drives_full_lifecycle_through_runtime() {
        pollster::block_on(async {
            let store = MemKappaStore::new();
            let code = store.put("blake3", &wasm()).unwrap();
            let empty = store.put("blake3", b"").unwrap();
            let cid = store
                .put(
                    "blake3",
                    &hologram_realizations::ContainerManifest {
                        code,
                        initial_state: empty,
                        parameters: empty,
                    }
                    .canonicalize(),
                )
                .unwrap();
            let caps = store
                .put(
                    "blake3",
                    &hologram_realizations::CapabilitySet::new(Capabilities {
                        storage_roots: alloc::vec![],
                        storage_quota_bytes: 0,
                        network_fetch: false,
                        network_announce: false,
                        publish_channels: alloc::vec![],
                        subscribe_channels: alloc::vec![],
                        memory_max_bytes: 1 << 20,
                        cpu_time_per_event_ms: 100,
                        priority_weight: 0,
                    })
                    .canonicalize(),
                )
                .unwrap();
            let rt = Runtime::new(BareMetalEngine::new(), store);
            let h = rt.spawn(&cid, &caps).await.unwrap();
            rt.deliver_event(h, &[]).unwrap();
            rt.deliver_event(h, &[]).unwrap();
            let snap = rt.suspend(h).await.unwrap();
            // The snapshot's references chain back to the Container ID (graph continuity).
            let bytes = rt.store().get(&snap).unwrap().unwrap();
            assert_eq!(
                hologram_realizations::Snapshot::references(bytes.as_ref()).unwrap()[0],
                cid
            );
        });
    }

    #[test]
    fn bare_engine_refuses_non_hologram_imports() {
        let wat = r#"
        (module
          (import "wasi_snapshot_preview1" "fd_write"
            (func $wasi (param i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "hg_event") (param i32 i32) (result i32) (i32.const 0)))
        "#;
        let bytes = wat::parse_str(wat).unwrap();
        let engine = BareMetalEngine::new();
        let result = engine.instantiate(&bytes, &ctx());
        match result {
            Err(RuntimeError::InstantiationFailed(reason)) => {
                assert!(
                    reason.contains("spec §4.4 host surface"),
                    "wrong refusal: {reason}",
                );
            }
            Err(other) => panic!("wrong error: {other:?}"),
            Ok(_) => panic!("expected refusal, got Ok"),
        }
    }
}
