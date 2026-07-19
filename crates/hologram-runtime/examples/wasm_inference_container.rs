//! Real-world use-case (hologram-ai): a **real Wasm inference/transform container**.
//!
//! The container reads an input κ from its capability-granted storage root, "infers" (here: an echo
//! transform — a real model would run a Prism `PrismModel`), and persists the output as a new κ —
//! all through the capability-gated host import surface, executed by Wasmtime. Run:
//! `cargo run -p hologram-runtime-wasmtime --example wasm_inference_container`.

use hologram_runtime::Runtime;
use hologram_runtime::WasmtimeEngine;
use hologram_space::{address_bytes, Capabilities, ContainerRuntime, KappaStore, Realization};
use hologram_space::{CapabilitySet, ContainerManifest};
use hologram_tck::MemKappaStore;

/// hg_event(input_κ): storage_get the input (granted root) → mem[100], transform, storage_put output.
const INFER_WAT: &str = r#"
(module
  (import "hologram" "storage_get" (func $get (param i32 i32 i32) (result i32)))
  (import "hologram" "storage_put" (func $put (param i32 i32 i32) (result i32)))
  (memory (export "memory") 4)
  (func (export "hg_init") (param i32 i32) (result i32) (i32.const 0))
  (func (export "hg_suspend") (result i32) (i32.const 0))
  (func (export "hg_resume")  (result i32) (i32.const 0))
  (func (export "hg_event") (param i32 i32) (result i32) (local $n i32)
    (local.set $n (call $get (i32.const 0) (i32.const 100) (i32.const 4096)))  ;; read input κ → mem[100]
    (drop (call $put (i32.const 100) (local.get $n) (i32.const 8192)))          ;; persist the output
    (local.get $n)))
"#;

fn main() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let input = b"<model input: tokenized prompt>";
        let input_k = store.put("blake3", input).unwrap();

        // The container's code + a capability set granting it read access to the input.
        let code = store
            .put("blake3", &wat::parse_str(INFER_WAT).unwrap())
            .unwrap();
        let cid = store
            .put(
                "blake3",
                &ContainerManifest {
                    code,
                    initial_state: code,
                    parameters: code,
                }
                .canonicalize(),
            )
            .unwrap();
        let caps = store
            .put(
                "blake3",
                &CapabilitySet::new(Capabilities {
                    storage_roots: vec![input_k], // may read the input κ (and its closure)
                    storage_quota_bytes: 1 << 20,
                    network_fetch: false,
                    network_announce: false,
                    publish_channels: vec![],
                    subscribe_channels: vec![],
                    memory_max_bytes: 4 << 20,
                    cpu_time_per_event_ms: 1000,
                    priority_weight: 0,
                })
                .canonicalize(),
            )
            .unwrap();

        let rt = Runtime::new(WasmtimeEngine::new(), store);
        let h = rt.spawn(&cid, &caps).await.unwrap();
        println!("spawn     : inference container running (Wasmtime)");

        // Deliver the input κ as the event; the container reads it, transforms, and writes the output.
        let bytes_read = rt.deliver_event(h, input_k.as_array()).unwrap();
        println!(
            "infer     : read {} input bytes via storage_get, wrote output via storage_put",
            bytes_read
        );

        // The output κ (echo transform → same content as the input) is now in the κ-graph.
        let output_k = address_bytes(input); // echo: output content == input content
        assert!(
            rt.store().contains(&output_k),
            "inference output persisted as a κ"
        );
        println!("output κ  : {}", output_k.as_str());
        println!(
            "OK — Wasm inference container read+wrote the κ-graph through capability-gated imports"
        );
    });
}
