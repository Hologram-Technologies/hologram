#![cfg(feature = "engine-wasmtime")]

//! NI class (arch §11.9): a `NetworkInterface` driver is **imported from an authoritative source,
//! verified, and then USED by the device**. The driver is an executable Wasm network-driver module;
//! the engine fetches it by κ (verify-on-receipt), instantiates it as a `WasmNetworkInterface`, and
//! exercises transmit + receive — so every byte that traverses the HAL is moved by the imported
//! driver's code. Symmetric to the DU class for block devices (`driver_backed_device.rs`).

use async_trait::async_trait;
use hologram_runtime::WasmNetworkInterface;
use hologram_space::NetworkInterface;
use hologram_space::{address_bytes, get_with_fetch, Bytes, KappaLabel71, KappaSync, SyncError};
use hologram_tck::MemKappaStore;
use std::collections::HashMap;

/// An executable network-driver in Wasm: a tiny loopback NIC that buffers transmitted frames into
/// linear memory at `$QUEUE`, then drains them on `receive`. Demonstrates the HAL contract
/// end-to-end (mac/mtu/transmit/receive) including the **`notify_rx`** waker — after each
/// transmit, the driver calls `hologram.notify_rx()` to indicate RX-ready (IRQ surrogate).
const NET_DRIVER_WAT: &str = r#"
(module
  (import "hologram" "notify_rx" (func $notify_rx))
  (memory (export "memory") 4)                  ;; 256 KiB
  (global $QUEUE_HEAD (mut i32) (i32.const 0))  ;; bytes pending in the loopback queue (frame len)
  (global $QUEUE i32 (i32.const 0x10000))       ;; queue buffer at 64 KiB

  ;; mac_address(out_ptr): write the 6-byte MAC at `out_ptr` (DE:AD:BE:EF:00:01)
  (func (export "mac_address") (param $out i32)
    (i32.store8 (i32.add (local.get $out) (i32.const 0)) (i32.const 0xDE))
    (i32.store8 (i32.add (local.get $out) (i32.const 1)) (i32.const 0xAD))
    (i32.store8 (i32.add (local.get $out) (i32.const 2)) (i32.const 0xBE))
    (i32.store8 (i32.add (local.get $out) (i32.const 3)) (i32.const 0xEF))
    (i32.store8 (i32.add (local.get $out) (i32.const 4)) (i32.const 0x00))
    (i32.store8 (i32.add (local.get $out) (i32.const 5)) (i32.const 0x01)))

  (func (export "mtu") (result i32) (i32.const 1500))

  ;; transmit(ptr, len) -> i32: copy the frame into $QUEUE; signal RX-ready (loopback echo);
  ;; return bytes written. -1 = backpressure (queue already holds a frame).
  (func (export "transmit") (param $ptr i32) (param $len i32) (result i32)
    (if (i32.ne (global.get $QUEUE_HEAD) (i32.const 0))
      (then (return (i32.const -1))))
    (memory.copy (global.get $QUEUE) (local.get $ptr) (local.get $len))
    (global.set $QUEUE_HEAD (local.get $len))
    ;; Fire the RX-ready signal — the host wakes any task registered via register_rx_waker.
    (call $notify_rx)
    (local.get $len))

  ;; receive(ptr, cap) -> i32: drain the queue into `[ptr, ptr+len]`; return len (0 = nothing).
  (func (export "receive") (param $ptr i32) (param $cap i32) (result i32) (local $n i32)
    (local.set $n (global.get $QUEUE_HEAD))
    (if (i32.eqz (local.get $n)) (then (return (i32.const 0))))
    (if (i32.gt_u (local.get $n) (local.get $cap))
      (then (local.set $n (local.get $cap))))
    (memory.copy (local.get $ptr) (global.get $QUEUE) (local.get $n))
    (global.set $QUEUE_HEAD (i32.const 0))
    (local.get $n)))
"#;

/// An authoritative source serving the driver by κ.
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
fn ni_imported_wasm_driver_routes_packets() {
    pollster::block_on(async {
        // 1) Authoritative source publishes the network-driver under its κ.
        let driver_wasm = wat::parse_str(NET_DRIVER_WAT).expect("valid wat");
        let driver_k = address_bytes(&driver_wasm);
        let source = DriverSource {
            blobs: HashMap::from([(*driver_k.as_array(), driver_wasm.clone())]),
        };

        // 2) Node imports the driver by κ, verified on receipt (a forging source is rejected).
        let local = MemKappaStore::new();
        let imported = get_with_fetch(&local, &source, &driver_k)
            .await
            .unwrap()
            .expect("driver imported");
        assert_eq!(imported.as_ref(), driver_wasm.as_slice());

        // 3) Instantiate the verified driver as the network interface.
        let nic = WasmNetworkInterface::from_code(imported.as_ref()).expect("driver instantiates");

        // 4) The HAL contract round-trips through the driver's code.
        assert_eq!(nic.mac_address(), [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01]);
        assert_eq!(nic.mtu(), 1500);

        let frame = b"\x00\x01\x02\x03\x04\x05\x06\x07hello-from-an-imported-driver";
        let written = nic.transmit(frame).expect("transmit");
        assert_eq!(written, frame.len(), "driver echoes its own write back");

        // Backpressure surfaces correctly when the loopback queue is full.
        assert!(
            matches!(
                nic.transmit(b"another"),
                Err(hologram_space::NicError::Backpressure)
            ),
            "queue full → backpressure"
        );

        // Drain the queue via receive — bytes round-trip through the imported driver's memory.
        let mut buf = vec![0u8; nic.mtu() as usize];
        let n = nic.receive(&mut buf).expect("receive");
        assert_eq!(n, frame.len());
        assert_eq!(&buf[..n], &frame[..]);

        // After draining, receive returns 0 (no frame buffered).
        assert_eq!(nic.receive(&mut buf).unwrap(), 0);
    });
}

#[test]
fn ni_notify_rx_drives_register_rx_waker() {
    // arch §11.9 RX waker bridge: the driver's `hologram.notify_rx` host import sets the RX-ready
    // signal and wakes any task registered via `register_rx_waker`. This replaces the prior no-op
    // and is what closes the production async-I/O loop on real NICs (IRQ → notify → task wake).
    use core::sync::atomic::{AtomicBool, Ordering};
    use core::task::{Context, RawWaker, RawWakerVTable, Waker};

    let driver_wasm = wat::parse_str(NET_DRIVER_WAT).expect("valid wat");
    let nic = WasmNetworkInterface::from_code(&driver_wasm).expect("driver instantiates");

    // Build a waker that flips an atomic when fired (we don't need a full executor).
    static WOKEN: AtomicBool = AtomicBool::new(false);
    fn raw_waker() -> RawWaker {
        fn clone(_: *const ()) -> RawWaker {
            raw_waker()
        }
        fn wake(_: *const ()) {
            WOKEN.store(true, Ordering::SeqCst);
        }
        fn wake_by_ref(_: *const ()) {
            WOKEN.store(true, Ordering::SeqCst);
        }
        fn drop_(_: *const ()) {}
        RawWaker::new(
            core::ptr::null(),
            &RawWakerVTable::new(clone, wake, wake_by_ref, drop_),
        )
    }
    let waker = unsafe { Waker::from_raw(raw_waker()) };
    let _ = Context::from_waker(&waker); // ensure the API shape is valid

    // No signal yet → register defers until notify.
    WOKEN.store(false, Ordering::SeqCst);
    nic.register_rx_waker(waker.clone());
    assert!(
        !WOKEN.load(Ordering::SeqCst),
        "waker not fired before notify_rx"
    );
    assert!(!nic.rx_pending());

    // Transmit a frame — the driver's loopback path calls hologram.notify_rx, which fires the waker.
    nic.transmit(b"loopback-frame").unwrap();
    assert!(
        WOKEN.load(Ordering::SeqCst),
        "notify_rx must wake the registered task"
    );
    assert!(nic.rx_pending());

    // After draining, the pending bit clears. A late register after pending is set wakes immediately.
    let mut buf = vec![0u8; nic.mtu() as usize];
    let _ = nic.receive(&mut buf).unwrap();
    assert!(!nic.rx_pending());

    // Lost-wakeup guard: if `notify_rx` fired before `register_rx_waker`, registration still wakes.
    nic.transmit(b"another").unwrap();
    let _ = nic.receive(&mut buf).unwrap(); // drains
                                            // Re-arm scenario: driver calls notify_rx by transmitting again BEFORE we register.
    nic.transmit(b"third").unwrap();
    assert!(nic.rx_pending());
    WOKEN.store(false, Ordering::SeqCst);
    nic.register_rx_waker(waker.clone());
    assert!(
        WOKEN.load(Ordering::SeqCst),
        "registering after a pending notify must wake immediately (no lost wakeup)"
    );
}
