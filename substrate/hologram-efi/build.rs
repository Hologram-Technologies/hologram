//! Build-time driver assembly for the bare-metal boot binary (arch §11.9 + DI/DU classes).
//!
//! Compiles a real Wasm block-device driver from a WAT source, computes its blake3 κ-label, and
//! emits both to `$OUT_DIR`. The boot binary `include_bytes!`-es the driver and `include_str!`-es
//! the expected κ — turning `verify_kappa` into a real measured-boot check: if the embedded
//! driver bytes are tampered post-build, the runtime re-derives a different κ and the boot fails.
//!
//! This replaces the prior placeholder string with a genuine codemodule κ-graph anchor.

use std::env;
use std::fs;
use std::path::PathBuf;

/// The block-device driver bytes the boot binary brings up. Same shape as the test fixture in
/// `runtime-wasmtime/tests/driver_backed_device.rs` (sector_size/sector_count/read/write/flush) —
/// proving the substrate's codemodule-κ → device path is the same for tests and for boot.
const BLOCK_DRIVER_WAT: &str = r#"
(module
  (memory (export "memory") 8)
  (global $DISK i32 (i32.const 0x20000))
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

/// The NIC driver (E2). Same shape as the test fixture in
/// `runtime-wasmtime/tests/driver_backed_nic.rs` — `mac_address`/`mtu`/`transmit`/`receive` plus
/// the `hologram.notify_rx` host import (production RX-ready signal). Build records its blake3 κ
/// in `OUT_DIR/nic-driver.kappa`; the boot binary verifies on bring-up (the measured-boot anchor
/// for the network surface, symmetric to the block-device driver §12.6).
const NIC_DRIVER_WAT: &str = r#"
(module
  (import "hologram" "notify_rx" (func $notify_rx))
  (memory (export "memory") 4)
  (global $QHEAD (mut i32) (i32.const 0))
  (global $QUEUE i32 (i32.const 0x10000))
  (func (export "mac_address") (param $out i32)
    (i32.store8 (i32.add (local.get $out) (i32.const 0)) (i32.const 0x02))
    (i32.store8 (i32.add (local.get $out) (i32.const 1)) (i32.const 0x00))
    (i32.store8 (i32.add (local.get $out) (i32.const 2)) (i32.const 0xC0))
    (i32.store8 (i32.add (local.get $out) (i32.const 3)) (i32.const 0xFF))
    (i32.store8 (i32.add (local.get $out) (i32.const 4)) (i32.const 0xEE))
    (i32.store8 (i32.add (local.get $out) (i32.const 5)) (i32.const 0x01)))
  (func (export "mtu") (result i32) (i32.const 1500))
  (func (export "transmit") (param $ptr i32) (param $len i32) (result i32)
    (if (i32.ne (global.get $QHEAD) (i32.const 0))
      (then (return (i32.const -1))))
    (memory.copy (global.get $QUEUE) (local.get $ptr) (local.get $len))
    (global.set $QHEAD (local.get $len))
    (call $notify_rx)
    (local.get $len))
  (func (export "receive") (param $ptr i32) (param $cap i32) (result i32) (local $n i32)
    (local.set $n (global.get $QHEAD))
    (if (i32.eqz (local.get $n)) (then (return (i32.const 0))))
    (if (i32.gt_u (local.get $n) (local.get $cap))
      (then (local.set $n (local.get $cap))))
    (memory.copy (local.get $ptr) (global.get $QUEUE) (local.get $n))
    (global.set $QHEAD (i32.const 0))
    (local.get $n)))
"#;

/// Write `bytes` to `out_dir/name` and the canonical κ-label form of `blake3(bytes)` to
/// `out_dir/<stem>.kappa` (`blake3:<64 hex>`).
fn emit_with_kappa(out_dir: &PathBuf, name: &str, stem: &str, bytes: &[u8]) {
    let digest = blake3::hash(bytes);
    let mut kappa = String::with_capacity(71);
    kappa.push_str("blake3:");
    for &b in digest.as_bytes() {
        let hi = b >> 4;
        let lo = b & 0xf;
        kappa.push(char::from_digit(hi as u32, 16).unwrap());
        kappa.push(char::from_digit(lo as u32, 16).unwrap());
    }
    fs::write(out_dir.join(name), bytes).expect("write driver bytes");
    fs::write(out_dir.join(format!("{stem}.kappa")), &kappa).expect("write driver kappa");
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // Block-device driver — §12.6 measured-boot anchor.
    let block_wasm = wat::parse_str(BLOCK_DRIVER_WAT).expect("block WAT parse");
    emit_with_kappa(&out_dir, "driver.wasm", "driver", &block_wasm);

    // NIC driver — E2, symmetric measured-boot anchor on the network surface.
    let nic_wasm = wat::parse_str(NIC_DRIVER_WAT).expect("nic WAT parse");
    emit_with_kappa(&out_dir, "nic-driver.wasm", "nic-driver", &nic_wasm);

    println!("cargo:rerun-if-changed=build.rs");
}
