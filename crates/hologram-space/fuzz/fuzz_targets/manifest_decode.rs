#![no_main]
//! Coverage-guided fuzz of the `.holo` v3 **manifest decoder** (spec 03 §Parser hardening). The
//! loader parses network-supplied bytes, so `AppManifest::decode` / `references` must never panic
//! on hostile input — every length is bounds-checked, no unbounded allocation. Complements the
//! deterministic in-tree mutation suite (`tests/parser_hardening.rs`).
use hologram_space::{AppManifest, Realization};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Must return Ok/Err, never panic/OOM, on any bytes.
    let _ = AppManifest::decode(data);
    let _ = <AppManifest as Realization>::references(data);
});
