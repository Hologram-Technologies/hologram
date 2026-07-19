//! The **CC bijection audit** against the real `CONFORMANCE.md` + the ported holospaces `cc*.rs`
//! witness tree. Fails the build if any `CC` catalog row cites a witness test that isn't present
//! (a renamed/removed/typo'd witness, or a claimed component with no test). Artifact-free — parses
//! text only — so it enforces the honesty of the CC catalog (MG-7) offline, independent of the
//! heavy QEMU/browser CI tier that actually runs the boots.
use hologram_conformance::{catalog, cc, CC_TESTS_DIR, CONFORMANCE_MD};
use std::path::Path;

#[test]
fn cc_catalog_binds_to_present_witness_tests() {
    let md = std::fs::read_to_string(CONFORMANCE_MD).expect("read CONFORMANCE.md");
    let rows = catalog::parse_catalog(&md);
    let witnesses =
        cc::collect_cc_witnesses(Path::new(CC_TESTS_DIR)).expect("walk the holospaces cc tests");

    if let Err(violations) = cc::check_cc_bijection(&rows, &witnesses) {
        panic!(
            "CC bijection audit failed (every CC row must cite a present witness test):\n  - {}",
            violations.join("\n  - ")
        );
    }
}
