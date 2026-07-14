//! G-C3 / SPINE-6 witness — **qualified workload bounds**. A container is opaque Wasm + the
//! spec §4.4 `hologram.*` host import surface. Any import outside `hologram.*` (a WASI import, an
//! `env` shim, an unknown side-channel) would be an unmediated escape hatch — on bare-metal in
//! particular, there is no native-subprocess to host it. The runtime refuses such modules at
//! instantiation; this test pins the refusal.

use hologram_realizations::ContainerManifest;
use hologram_runtime::Runtime;
use hologram_runtime_wasmtime::WasmtimeEngine;
use hologram_space::{Capabilities, ContainerRuntime, KappaStore, Realization};
use hologram_store_mem::MemKappaStore;

/// A container declaring a WASI import — the kind of escape hatch the substrate must refuse.
const WASI_BACKDOOR_WAT: &str = r#"
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $wasi_fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "hg_init")    (param i32 i32) (result i32) (i32.const 0))
  (func (export "hg_suspend") (result i32) (i32.const 0))
  (func (export "hg_resume")  (result i32) (i32.const 0))
  (func (export "hg_event") (param i32 i32) (result i32)
    (drop (call $wasi_fd_write (i32.const 1) (i32.const 0) (i32.const 0) (i32.const 0)))
    (i32.const 0)))
"#;

#[test]
fn b5_container_with_non_hologram_import_is_refused_at_instantiation() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let code = store
            .put("blake3", &wat::parse_str(WASI_BACKDOOR_WAT).unwrap())
            .unwrap();
        let empty = store.put("blake3", b"").unwrap();
        let cid = store
            .put(
                "blake3",
                &ContainerManifest {
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
                    storage_roots: vec![],
                    storage_quota_bytes: 0,
                    network_fetch: false,
                    network_announce: false,
                    publish_channels: vec![],
                    subscribe_channels: vec![],
                    memory_max_bytes: 1 << 20,
                    cpu_time_per_event_ms: 100,
                    priority_weight: 0,
                })
                .canonicalize(),
            )
            .unwrap();

        let rt = Runtime::new(WasmtimeEngine::new(), store);
        let err = rt
            .spawn(&cid, &caps)
            .await
            .expect_err("spawn must refuse a container declaring an import outside `hologram.*`");
        // Specifically the workload-bound refusal, not some other instantiation failure.
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("spec §4.4 host surface"),
            "unexpected refusal reason: {msg}",
        );
    });
}
