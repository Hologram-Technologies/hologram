//! The **CS bijection audit** against the real `CONFORMANCE.md` + the ported holospaces V1–V8
//! validator scripts. Fails the build if any `CS` catalog row cites a validator script that isn't
//! present. Text-only (no docs toolchain) — it enforces the honesty of the CS catalog (MG-8)
//! offline, independent of the `docs-conformance` CI job that actually runs the validators.
use hologram_conformance::{catalog, cc, CONFORMANCE_MD, CS_SCRIPTS_DIR};
use std::path::Path;

#[test]
fn cs_catalog_binds_to_present_validator_scripts() {
    let md = std::fs::read_to_string(CONFORMANCE_MD).expect("read CONFORMANCE.md");
    let rows = catalog::parse_catalog(&md);

    if let Err(violations) = cc::check_cs_bijection(&rows, Path::new(CS_SCRIPTS_DIR)) {
        panic!(
            "CS bijection audit failed (every CS row must cite a present validator script):\n  - {}",
            violations.join("\n  - ")
        );
    }
}
