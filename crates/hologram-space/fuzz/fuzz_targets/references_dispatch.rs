#![no_main]
//! Coverage-guided fuzz of the **generic registry dispatch** `references(bytes, REGISTRY)` (spec 03
//! §Parser hardening) — the store's single network entry point, which parses arbitrary IRI-tagged
//! bytes into a realization's operand κs. Must never panic, whatever the input.
use hologram_space::{references, REGISTRY};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = references(data, REGISTRY);
});
