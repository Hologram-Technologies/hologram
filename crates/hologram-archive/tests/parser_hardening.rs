//! Parser hardening for the `.holo` **section parser** (spec `refactor/03` §Parser hardening —
//! a standing requirement). The loader parses network-supplied bytes, so the section-table parse
//! and every section access must never panic on hostile input: forged `u64` offsets/lengths are
//! bounds-checked (and cannot overflow `usize` on 32-bit targets), and truncated tables error
//! cleanly. This deterministic mutation suite drives the full parse chain
//! (`from_bytes_unchecked → into_plan → section/app_manifest/extensions`) over truncations, byte
//! mutations, and pseudo-random noise, asserting it never panics. CI-permanent.

use hologram_archive::{HoloLoader, HoloWriter, SectionKind};
use std::panic::{catch_unwind, AssertUnwindSafe};

struct Rng(u64);
impl Rng {
    fn byte(&mut self) -> u8 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x & 0xff) as u8
    }
}

/// Drive the whole loader chain over `bytes`, discarding every result — the point is that no path
/// panics, whatever the input.
fn drive(bytes: &[u8]) {
    let Ok(loader) = HoloLoader::from_bytes_unchecked(bytes) else {
        return;
    };
    let Ok(plan) = loader.into_plan() else {
        return;
    };
    let _ = plan.sections();
    let _ = plan.app_manifest();
    let _ = plan.extensions();
    let _ = plan.content_blobs();
    for kind in [
        SectionKind::KernelCalls,
        SectionKind::Weights,
        SectionKind::Certificates,
        SectionKind::Extension,
        SectionKind::AppManifest,
    ] {
        let _ = plan.section(kind);
    }
}

fn assert_panic_free(seed: &[u8]) {
    let run = |b: &[u8]| catch_unwind(AssertUnwindSafe(|| drive(b))).is_ok();
    for n in 0..=seed.len() {
        assert!(
            run(&seed[..n]),
            "loader panicked on truncated prefix of len {n}"
        );
    }
    for i in 0..seed.len() {
        for delta in [0x01u8, 0x3f, 0x80, 0xff] {
            let mut m = seed.to_vec();
            m[i] = m[i].wrapping_add(delta);
            assert!(run(&m), "loader panicked on byte mutation at offset {i}");
        }
    }
    let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ seed.len() as u64);
    for len in [0usize, 1, 40, 128, 1024] {
        for _ in 0..64 {
            let buf: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
            assert!(run(&buf), "loader panicked on random input of len {len}");
        }
    }
}

fn full_archive() -> Vec<u8> {
    let mut w = HoloWriter::new();
    w.set_app_manifest(b"IRI:app-manifest\x00...manifest bytes...".to_vec());
    w.add_extension("tokenizer.json", b"{\"m\":\"bpe\"}".to_vec());
    w.add_extension("gen_config", vec![1, 2, 3, 4, 5]);
    w.finish().unwrap()
}

#[test]
fn section_parser_never_panics() {
    assert_panic_free(&full_archive());
}

#[test]
fn forged_section_offset_and_length_error_not_overflow() {
    // The section table is `header(10) ‖ [kind:u8 pad:7 offset:u64 length:u64] × count`. Forge the
    // first entry's offset and length to u64::MAX — on a 32-bit `usize` `start + length` would
    // overflow; the checked bound must instead reject it cleanly (no panic, no OOB).
    let mut bytes = full_archive();
    let first_entry = 10; // after magic+version+flags+count
    let offset_at = first_entry + 8; // skip kind(1) + pad(7)
    bytes[offset_at..offset_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    bytes[offset_at + 8..offset_at + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    // Whatever kind sits first, accessing it must not panic — drive the whole chain.
    assert!(
        catch_unwind(AssertUnwindSafe(|| drive(&bytes))).is_ok(),
        "a forged u64 offset/length must be rejected, never overflow usize"
    );
}
