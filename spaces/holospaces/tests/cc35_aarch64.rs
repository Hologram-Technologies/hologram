//! `CC-35` — the system emulator executes the AArch64 (A64) integer ISA
//! correctly (ADR-021, arc42 ch.10 conformance catalog).
//!
//! The implementation under test is the [`aarch64`](holospaces::emulator::aarch64)
//! integer core. The authority is the **Arm Architecture Reference Manual** (ARM
//! DDI 0487) for the A64 base instruction set + `PSTATE.NZCV`, with
//! `qemu-system-aarch64`/`qemu-aarch64` as the differential oracle. These
//! witnesses run **real, toolchain-assembled** A64 binaries
//! (`vv/artifacts/cc35/*.bin`, built from the committed `.s` sources by
//! `vv/artifacts/cc35/build.sh`): each is a self-checking battery that, run on
//! the core at its reset PC, writes `PASS\n` and exits `0` exactly when every
//! Arm-ARM-defined result holds — the same stdout + status `qemu-aarch64`
//! produces for the same machine code (`vv/suites/cc35-aarch64-core.sh`).

use holospaces::emulator::aarch64::{Cpu, Halt};
use std::path::Path;

/// Directory holding the committed A64 batteries (`vv/artifacts/cc35/*.bin`,
/// built from the `.s` sources by `vv/artifacts/cc35/build.sh`). Read at
/// runtime (not `include_bytes!`) so the crate's test binary still compiles in a
/// checkout that has not built the `vv/` fixture tree; the individual battery
/// tests then skip with a note rather than failing to build.
fn cc35_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vv/artifacts/cc35")
}

/// Load a battery `.bin` if present; `None` when the fixture tree is absent.
fn battery(name: &str) -> Option<Vec<u8>> {
    std::fs::read(cc35_dir().join(name)).ok()
}

/// Load a committed A64 battery, run it on the core, and return its
/// `(console, exit_status)`. The battery is position-independent, so any reset
/// base works; 16 MiB of RAM gives the stack frame headroom.
fn run_battery(image: &[u8]) -> (Vec<u8>, u64) {
    const BASE: u64 = 0x4000_0000;
    let mut cpu = Cpu::new(BASE, 16 * 1024 * 1024);
    cpu.load_image(image);
    match cpu.run(10_000_000) {
        Halt::Exit(status) => (cpu.console().to_vec(), status),
        other => panic!("battery did not exit cleanly: {other:?}"),
    }
}

/// The data-processing battery: every A64 data-processing group's result equals
/// the Arm-ARM-defined value.
#[test]
fn the_a64_data_processing_battery_passes() {
    let Some(image) = battery("arith.bin") else {
        eprintln!("cc35: arith.bin fixture absent (vv/artifacts/cc35 not built) — skipping");
        return;
    };
    let (console, status) = run_battery(&image);
    assert_eq!(console, b"PASS\n", "arith battery verdict");
    assert_eq!(status, 0, "arith battery exit status");
}

/// The load/store battery: its cases round-trip through memory correctly (the
/// "full addressing-mode + extension family" is the fixture's scope; this test
/// runs the fixture and asserts its `PASS` verdict).
#[test]
fn the_a64_load_store_battery_passes() {
    let Some(image) = battery("memory.bin") else {
        eprintln!("cc35: memory.bin fixture absent (vv/artifacts/cc35 not built) — skipping");
        return;
    };
    let (console, status) = run_battery(&image);
    assert_eq!(console, b"PASS\n", "memory battery verdict");
    assert_eq!(status, 0, "memory battery exit status");
}

/// The control-flow battery: branches + `NZCV` condition codes drive real loops,
/// a subroutine call, and the bit-test branches to the Arm-ARM-defined result
/// (`sum(1..=100) == 5050`).
#[test]
fn the_a64_control_flow_battery_passes() {
    let Some(image) = battery("control.bin") else {
        eprintln!("cc35: control.bin fixture absent (vv/artifacts/cc35 not built) — skipping");
        return;
    };
    let (console, status) = run_battery(&image);
    assert_eq!(console, b"PASS\n", "control battery verdict");
    assert_eq!(status, 0, "control battery exit status");
}

/// The SIMD&FP battery: the Advanced-SIMD (NEON) data-processing forms +
/// scalar floating-point (FADD/FSUB/FMUL/FDIV, the general↔SIMD `FMOV`, the
/// int↔fp conversions, `FCMP`) each equal their Arm-ARM-defined result.
#[test]
fn the_a64_simd_fp_battery_passes() {
    let Some(image) = battery("simd.bin") else {
        eprintln!("cc35: simd.bin fixture absent (vv/artifacts/cc35 not built) — skipping");
        return;
    };
    let (console, status) = run_battery(&image);
    assert_eq!(console, b"PASS\n", "simd/fp battery verdict");
    assert_eq!(status, 0, "simd/fp battery exit status");
}
