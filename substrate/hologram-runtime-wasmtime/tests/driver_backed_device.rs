//! End-to-end: a driver is **imported from an authoritative source, verified, and then USED by the
//! device**. The driver is an executable Wasm block-device module; the engine fetches it by κ
//! (verify-on-receipt), instantiates it as a `WasmBlockDevice`, and runs `BareMetalKappaStore` over
//! it — so every sector the store reads/writes is executed by the imported driver's code.

use async_trait::async_trait;
use hologram_runtime_wasmtime::WasmBlockDevice;
use hologram_space::{
    address_bytes, get_with_fetch, Bytes, KappaLabel71, KappaStore, KappaSync, SyncError,
};
use hologram_store_bare::BareMetalKappaStore;
use hologram_store_mem::MemKappaStore;
use std::collections::HashMap;

/// An executable block-device driver in Wasm: holds the "disk" in linear memory at `$DISK`, and
/// `read`/`write` move whole sectors between the disk and the host scratch pointer via `memory.copy`.
const BLOCK_DRIVER_WAT: &str = r#"
(module
  (memory (export "memory") 8)                      ;; 512 KiB
  (global $DISK i32 (i32.const 0x20000))            ;; disk region at 128 KiB
  (func (export "sector_size")  (result i32) (i32.const 512))
  (func (export "sector_count") (result i64) (i64.const 64))
  (func (export "read")  (param $lba i64) (param $sectors i32) (param $ptr i32) (result i32)
    (memory.copy (local.get $ptr)
                 (i32.add (global.get $DISK) (i32.mul (i32.wrap_i64 (local.get $lba)) (i32.const 512)))
                 (i32.mul (local.get $sectors) (i32.const 512)))
    (i32.const 0))
  (func (export "write") (param $lba i64) (param $sectors i32) (param $ptr i32) (result i32)
    (memory.copy (i32.add (global.get $DISK) (i32.mul (i32.wrap_i64 (local.get $lba)) (i32.const 512)))
                 (local.get $ptr)
                 (i32.mul (local.get $sectors) (i32.const 512)))
    (i32.const 0))
  (func (export "flush") (result i32) (i32.const 0)))
"#;

/// An authoritative source serving the driver by κ (verify-on-receipt makes it trustless).
struct DriverSource {
    blobs: HashMap<[u8; 71], Vec<u8>>,
}
#[async_trait]
impl KappaSync for DriverSource {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        Ok(self
            .blobs
            .get(kappa.as_array())
            .map(|v| Bytes::from(v.clone())))
    }
    async fn announce(&self, _k: &KappaLabel71) {}
    async fn discover(&self, _p: Option<&[u8]>, _l: usize) -> Vec<KappaLabel71> {
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
fn imported_driver_backs_the_store_device_end_to_end() {
    pollster::block_on(async {
        // 1) An authoritative source publishes the block-device driver (Wasm) under its κ.
        let driver_wasm = wat::parse_str(BLOCK_DRIVER_WAT).expect("valid wat");
        let driver_k = address_bytes(&driver_wasm);
        let source = DriverSource {
            blobs: HashMap::from([(*driver_k.as_array(), driver_wasm.clone())]),
        };

        // 2) A node imports the driver by κ, verified on receipt (a forged driver would be rejected).
        let local = MemKappaStore::new();
        let imported = get_with_fetch(&local, &source, &driver_k)
            .await
            .unwrap()
            .expect("driver imported");
        assert_eq!(imported.as_ref(), driver_wasm.as_slice());

        // 3) Instantiate the verified driver as the block device.
        let device = WasmBlockDevice::from_code(imported.as_ref()).expect("driver instantiates");

        // 4) Run the bare-metal store over the driver-backed device: every sector read/written here
        //    is executed by the IMPORTED driver's Wasm code.
        let store = BareMetalKappaStore::open(device).expect("store over driver device");
        let payload = b"data-stored-through-an-imported-wasm-driver";
        let k = store.put("blake3", payload).expect("put via driver");
        assert_eq!(
            store.get(&k).unwrap().unwrap().as_ref(),
            payload,
            "round-trip through the driver"
        );

        // A second κ to exercise multiple sectors through the driver.
        let k2 = store.put("blake3", &[7u8; 2048]).unwrap();
        assert_eq!(store.get(&k2).unwrap().unwrap().len(), 2048);
    });
}
