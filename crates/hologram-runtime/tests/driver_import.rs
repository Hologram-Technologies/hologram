//! Driver import V&V: a **driver is a codemodule κ-label** (uor-addr CCMAS), imported from an
//! **authoritative source** (a peer/gateway) and **verified on receipt** (SPINE-4). The
//! content-addressed graph is the authority — a forged source cannot serve a driver whose bytes
//! re-derive to the requested κ — so the engine imports **arbitrary** drivers trustlessly.

use async_trait::async_trait;
use hologram_space::{
    address_bytes, get_with_fetch, Bytes, KappaLabel71, KappaStore, KappaSync, SyncError,
};
use hologram_tck::MemKappaStore;
use std::collections::HashMap;
use uor_addr::codemodule::CodeModuleValue;

/// The serialized bytes of a "driver" code module (its CCMAS AST). The engine stores/transports it
/// like any artifact; its content address is `address_bytes` of these canonical bytes.
fn driver_bytes(name: &str) -> Vec<u8> {
    let body = CodeModuleValue::atom("0");
    let ret = CodeModuleValue::atom("DeviceError");
    let read = CodeModuleValue::function("read", &[], &ret, &body);
    let write = CodeModuleValue::function("write", &[], &ret, &body);
    CodeModuleValue::module(name, &[read, write])
        .tagged_bytes()
        .to_vec()
}

/// An authoritative source peer that serves driver codemodules from its store. `forge=true` models
/// a malicious source that returns wrong bytes for every request.
struct SourcePeer {
    blobs: HashMap<[u8; 71], Vec<u8>>,
    forge: bool,
}

#[async_trait]
impl KappaSync for SourcePeer {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        if self.forge {
            return Ok(Some(Bytes::from(b"not-the-driver-you-asked-for".to_vec())));
        }
        Ok(self
            .blobs
            .get(kappa.as_array())
            .map(|v| Bytes::from(v.clone())))
    }
    async fn announce(&self, _kappa: &KappaLabel71) {}
    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        Vec::new()
    }
    async fn add_peer(&self, _m: &str) -> Result<(), SyncError> {
        Ok(())
    }
    async fn add_gateway(&self, _u: &str) -> Result<(), SyncError> {
        Ok(())
    }
}

#[test]
fn import_arbitrary_drivers_from_an_authoritative_source_and_verify() {
    pollster::block_on(async {
        // The authoritative source publishes a set of arbitrary drivers (codemodule κ-labels).
        let names = ["nvme", "ahci", "e1000", "virtio-net", "virtio-blk"];
        let mut blobs = HashMap::new();
        let mut driver_kappas = Vec::new();
        for n in names {
            let bytes = driver_bytes(n);
            let k = address_bytes(&bytes); // the driver's content address
            blobs.insert(*k.as_array(), bytes);
            driver_kappas.push(k);
        }
        let source = SourcePeer {
            blobs,
            forge: false,
        };

        // A fresh engine node has none of them locally; it imports each by κ, verified on receipt.
        let local = MemKappaStore::new();
        for k in &driver_kappas {
            assert!(!local.contains(k));
            let imported = get_with_fetch(&local, &source, k).await.unwrap();
            assert!(imported.is_some(), "driver imported from the source");
            assert!(
                local.contains(k),
                "verified driver cached locally for loading"
            );
        }
    });
}

#[test]
fn a_forging_source_cannot_supply_a_driver() {
    pollster::block_on(async {
        let k = address_bytes(&driver_bytes("nvme"));
        let evil = SourcePeer {
            blobs: HashMap::new(),
            forge: true,
        };
        let local = MemKappaStore::new();
        // The forged bytes don't re-derive to the requested κ → rejected (§6.4); nothing cached.
        assert!(get_with_fetch(&local, &evil, &k).await.is_err());
        assert!(!local.contains(&k));
    });
}
