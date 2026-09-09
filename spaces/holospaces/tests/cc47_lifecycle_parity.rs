//! CC-47 — lifecycle snapshot parity on the AArch64 and x86-64 cores.
//!
//! These are executable (not ignored) suspend/drop/restore witnesses.  Each
//! machine has live CPU state, modified RAM, and a non-empty κ-disk.  The
//! snapshot itself is persisted and recovered by κ before the original machine
//! is dropped, then execution continues in lock-step from the recovered bytes.

use hologram_space::{KappaStore, MemKappaStore};
use holospaces::emulator::{aarch64, x64};

const SNAPSHOT_AXIS: &str = "blake3";

fn migrate(snapshot: &[u8]) -> Vec<u8> {
    let store = MemKappaStore::new();
    let kappa = store
        .put(SNAPSHOT_AXIS, snapshot)
        .expect("snapshot is valid canonical content");
    store
        .get(&kappa)
        .expect("κ lookup succeeds")
        .expect("snapshot is present")
        .to_vec()
}

#[test]
fn aarch64_suspend_drop_restore_preserves_cpu_ram_disk_and_continuation() {
    // movz x0,#1; add x0,x0,#1; add x0,x0,#1; hlt #0
    let image = [
        0x20, 0x00, 0x80, 0xd2, 0x00, 0x04, 0x00, 0x91, 0x00, 0x04, 0x00, 0x91, 0x00, 0x00, 0x40,
        0xd4,
    ];
    let mut machine = aarch64::Cpu::new(0x4000, 4096);
    machine.load_image(&image);
    machine.vv_ram_write(0x4100, b"aarch64-live-ram");
    machine.attach_disk(vec![0xa5; 1024]);
    machine.attach_workspace(&[("state.txt", b"aarch64-workspace")]);
    assert_eq!(machine.run(1), aarch64::Halt::OutOfBudget);
    assert_eq!(machine.xreg(0), 1);

    let suspended = machine.snapshot();
    let migrated = migrate(&suspended);
    drop(machine);

    let mut resumed = aarch64::Cpu::restore(&migrated).expect("AArch64 snapshot restores");
    assert_eq!(resumed.snapshot(), suspended, "restore is byte-identical");
    assert_eq!(resumed.vv_ram_read(0x4100, 16), b"aarch64-live-ram");
    assert_eq!(
        resumed.workspace_file("state.txt"),
        Some(b"aarch64-workspace".as_slice())
    );

    // Two independent restores are the uninterrupted-vs-resumed continuation
    // oracle: one step produces identical state and the following step advances.
    let mut control = aarch64::Cpu::restore(&suspended).expect("control restores");
    assert_eq!(resumed.run(1), aarch64::Halt::OutOfBudget);
    assert_eq!(control.run(1), aarch64::Halt::OutOfBudget);
    assert_eq!(resumed.snapshot(), control.snapshot());
    assert_eq!(resumed.xreg(0), 2);
    assert_eq!(resumed.run(1), aarch64::Halt::OutOfBudget);
    assert_eq!(resumed.xreg(0), 3);

    let mut different_disk = aarch64::Cpu::new(0x4000, 4096);
    different_disk.load_image(&image);
    different_disk.vv_ram_write(0x4100, b"aarch64-live-ram");
    different_disk.attach_disk(vec![0x5a; 1024]);
    different_disk.attach_workspace(&[("state.txt", b"aarch64-workspace")]);
    assert_eq!(different_disk.run(1), aarch64::Halt::OutOfBudget);
    assert_ne!(
        different_disk.snapshot(),
        suspended,
        "κ-disk bytes are in the snapshot"
    );
}

#[test]
fn x64_suspend_drop_restore_preserves_cpu_ram_disk_and_continuation() {
    // mov eax,1; add eax,1; add eax,1; hlt
    let image = [0xb8, 1, 0, 0, 0, 0x83, 0xc0, 1, 0x83, 0xc0, 1, 0xf4];
    let mut machine = x64::Cpu::new(4096);
    machine.load_at(0, &image);
    machine.vv_ram_write(0x100, b"x86-64-live-ram");
    machine.attach_disk(vec![0xa5; 1024]);
    machine.attach_workspace(&[("state.txt", b"x86-64-workspace")]);
    assert_eq!(machine.run(1), x64::Halt::OutOfBudget);
    assert_eq!(machine.reg(0), 1);

    let suspended = machine.snapshot();
    let migrated = migrate(&suspended);
    drop(machine);

    let mut resumed = x64::Cpu::restore(&migrated).expect("x86-64 snapshot restores");
    assert_eq!(resumed.snapshot(), suspended, "restore is byte-identical");
    assert_eq!(resumed.vv_ram_read(0x100, 15), b"x86-64-live-ram");
    assert_eq!(
        resumed.workspace_file("state.txt"),
        Some(b"x86-64-workspace".as_slice())
    );

    let mut control = x64::Cpu::restore(&suspended).expect("control restores");
    assert_eq!(resumed.run(1), x64::Halt::OutOfBudget);
    assert_eq!(control.run(1), x64::Halt::OutOfBudget);
    assert_eq!(resumed.snapshot(), control.snapshot());
    assert_eq!(resumed.reg(0), 2);
    assert_eq!(resumed.run(1), x64::Halt::OutOfBudget);
    assert_eq!(resumed.reg(0), 3);

    let mut different_disk = x64::Cpu::new(4096);
    different_disk.load_at(0, &image);
    different_disk.vv_ram_write(0x100, b"x86-64-live-ram");
    different_disk.attach_disk(vec![0x5a; 1024]);
    different_disk.attach_workspace(&[("state.txt", b"x86-64-workspace")]);
    assert_eq!(different_disk.run(1), x64::Halt::OutOfBudget);
    assert_ne!(
        different_disk.snapshot(),
        suspended,
        "κ-disk bytes are in the snapshot"
    );
}

#[test]
fn architecture_and_framing_are_strict() {
    let a64 = aarch64::Cpu::new(0x4000, 4096).snapshot();
    let x64 = x64::Cpu::new(4096).snapshot();
    assert!(x64::Cpu::restore(&a64).is_err());
    assert!(aarch64::Cpu::restore(&x64).is_err());
    assert!(aarch64::Cpu::restore(&a64[..a64.len() - 1]).is_err());
    let mut trailing = x64;
    trailing.push(0);
    assert!(x64::Cpu::restore(&trailing).is_err());
}
