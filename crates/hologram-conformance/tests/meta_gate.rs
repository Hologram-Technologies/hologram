//! Runs the honesty meta-gate against the real CONFORMANCE.md + features tree.
//! Fails the build if the catalog and scenarios drift out of bijection.
use hologram_conformance::{catalog, feature, report, CONFORMANCE_MD, SUITES_DIR};
use std::path::Path;

#[test]
fn catalog_and_scenarios_are_in_bijection() {
    let md = std::fs::read_to_string(CONFORMANCE_MD).expect("read CONFORMANCE.md");
    let rows = catalog::parse_catalog(&md);
    let scenarios = feature::parse_features(Path::new(SUITES_DIR)).expect("parse features");

    if let Err(violations) = report::check_bijection(&rows, &scenarios) {
        panic!(
            "conformance honesty meta-gate failed:\n  - {}",
            violations.join("\n  - ")
        );
    }
}
