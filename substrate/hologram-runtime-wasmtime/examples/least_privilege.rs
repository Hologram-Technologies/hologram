//! Real-world use-case: **multi-tenant least privilege** (sandboxed plugins).
//!
//! A tenant supervisor holding two data roots spawns a plugin child restricted to ONE root; an
//! attempt to grant the plugin a root the supervisor does **not** hold is refused by capability
//! containment (the SubtypingLattice `admits` relation). Run: `cargo run -p hologram-runtime-wasmtime
//! --example least_privilege`.

use hologram_realizations::{CapabilitySet, ContainerManifest};
use hologram_runtime::{MockEngine, Runtime};
use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::{
    Capabilities, ContainerRuntime, KappaLabel71, KappaStore, Realization,
};

fn caps(roots: Vec<KappaLabel71>, quota: u64) -> Capabilities {
    Capabilities {
        storage_roots: roots,
        storage_quota_bytes: quota,
        network_fetch: false,
        network_announce: false,
        publish_channels: vec![],
        subscribe_channels: vec![],
        memory_max_bytes: 0,
        cpu_time_per_event_ms: 0,
        priority_weight: 0,
    }
}

fn main() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let tenant_data = store.put("blake3", b"tenant://data").unwrap();
        let secrets = store.put("blake3", b"other-tenant://secrets").unwrap(); // NOT granted to the supervisor

        // Supervisor: may read its own tenant's data only, quota 1 MiB.
        let code = store.put("blake3", b"supervisor").unwrap();
        let sup_cid = store
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
        let sup_caps = store
            .put(
                "blake3",
                &CapabilitySet::new(caps(vec![tenant_data], 1 << 20)).canonicalize(),
            )
            .unwrap();

        // A plugin container body.
        let pcode = store.put("blake3", b"untrusted-plugin").unwrap();
        let plugin_cid = store
            .put(
                "blake3",
                &ContainerManifest {
                    code: pcode,
                    initial_state: pcode,
                    parameters: pcode,
                }
                .canonicalize(),
            )
            .unwrap();

        let rt = Runtime::new(MockEngine, store);
        let supervisor = rt.spawn(&sup_cid, &sup_caps).await.unwrap();

        // Narrowed delegation: data only, smaller quota → admitted.
        let narrow = rt
            .store()
            .put(
                "blake3",
                &CapabilitySet::new(caps(vec![tenant_data], 4096)).canonicalize(),
            )
            .unwrap();
        assert!(rt.spawn_child(supervisor, &plugin_cid, &narrow).is_ok());
        println!("admit     : plugin sandboxed to tenant://data (quota 4 KiB) — least privilege");

        // Privilege escalation attempts — all refused (cannot exceed the supervisor's authority).
        let grab_secrets = rt
            .store()
            .put(
                "blake3",
                &CapabilitySet::new(caps(vec![tenant_data, secrets], 4096)).canonicalize(),
            )
            .unwrap();
        let grab_quota = rt
            .store()
            .put(
                "blake3",
                &CapabilitySet::new(caps(vec![tenant_data], 1 << 30)).canonicalize(),
            )
            .unwrap();
        assert!(rt
            .spawn_child(supervisor, &plugin_cid, &grab_secrets)
            .is_err());
        assert!(rt
            .spawn_child(supervisor, &plugin_cid, &grab_quota)
            .is_err());
        println!(
            "refuse    : plugin cannot reach tenant://secrets or raise its quota (containment)"
        );

        println!("OK — capability delegation enforces least privilege (no escalation)");
    });
}
