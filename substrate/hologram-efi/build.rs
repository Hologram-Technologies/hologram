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

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let wasm = wat::parse_str(BLOCK_DRIVER_WAT).expect("WAT parse");
    let digest = blake3::hash(&wasm);
    // The canonical κ-label form `blake3:<64 hex>` — what the runtime's `address_bytes` produces.
    let mut kappa = String::with_capacity(71);
    kappa.push_str("blake3:");
    for &b in digest.as_bytes() {
        let hi = b >> 4;
        let lo = b & 0xf;
        kappa.push(char::from_digit(hi as u32, 16).unwrap());
        kappa.push(char::from_digit(lo as u32, 16).unwrap());
    }
    fs::write(out_dir.join("driver.wasm"), &wasm).expect("write driver.wasm");
    fs::write(out_dir.join("driver.kappa"), &kappa).expect("write driver.kappa");
    println!("cargo:rerun-if-changed=build.rs");
}
