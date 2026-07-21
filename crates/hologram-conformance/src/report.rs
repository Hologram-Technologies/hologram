//! The honesty meta-gate: a *static* check that the CONFORMANCE.md catalog and the
//! Gherkin scenarios are in bijection for the eight BDD classes. It verifies, for
//! every BDD-class row: exactly one scenario with the same `@id`; the row's status
//! glyph agrees with the scenario's `@status` tag; the row's `Witness` path+scenario
//! name matches the actual file; and the file declares exactly one scenario.
//!
//! An RM row may instead be **witnessed externally** — bound to an SDK / browser package
//! test the Rust `bdd` gate cannot run (the CC/CS pattern). Those rows carry a non-`.feature`
//! `Witness` and are bound by [`check_witnessed_rows`] rather than to a Gherkin scenario.
use crate::catalog::CatalogRow;
use crate::feature::{status_tag_to_glyph, ScenarioRef};
use std::path::Path;

/// The refactor classes whose rows are witnessed by BDD scenarios, plus `RM` — the
/// README public-surface suite (`features/suites/s7_readme`), which binds every fenced
/// code block in `README.md` to exactly one scenario.
pub const BDD_CLASSES: &[&str] = &["LAW", "SP", "HF", "NW", "TL", "MG", "GV", "RM"];

fn is_bdd(class: &str) -> bool {
    BDD_CLASSES.contains(&class)
}

/// A BDD-class row witnessed by an external artifact (an SDK / browser package test) rather than a
/// Gherkin scenario: its `Witness` is a `file::marker` path that is not a `.feature`. Bound by
/// [`check_witnessed_rows`] (the CC/CS pattern), so [`check_bijection`] requires no scenario for it.
fn externally_witnessed(row: &CatalogRow) -> bool {
    is_bdd(&row.class)
        && row
            .witness
            .as_deref()
            .is_some_and(|w| !w.contains(".feature::"))
}

/// Check row↔scenario bijection, status agreement, witness binding, and
/// one-scenario-per-file for the BDD classes only. `Err(violations)` when off.
pub fn check_bijection(rows: &[CatalogRow], scenarios: &[ScenarioRef]) -> Result<(), Vec<String>> {
    let mut violations = Vec::new();

    // Every BDD-class catalog row has exactly one scenario, with matching status
    // and matching witness path/name.
    for row in rows.iter().filter(|r| is_bdd(&r.class)) {
        // Externally-witnessed rows (SDK / browser package tests) are bound by
        // `check_witnessed_rows`, not to a Gherkin scenario — skip the scenario requirement.
        if externally_witnessed(row) {
            continue;
        }
        let matches: Vec<&ScenarioRef> = scenarios.iter().filter(|s| s.id == row.id).collect();
        match matches.as_slice() {
            [] => violations.push(format!("catalog row {} has no scenario", row.id)),
            [s] => {
                let scenario_status = status_tag_to_glyph(&s.status_tag)
                    .and_then(crate::catalog::Status::from_legend);
                if scenario_status != Some(row.status) {
                    violations.push(format!(
                        "row {} status disagrees with scenario @status:{}",
                        row.id, s.status_tag
                    ));
                }
                if let Some(w) = &row.witness {
                    let actual = format!("{}::{}", s.rel_path, s.scenario);
                    if *w != actual {
                        violations.push(format!(
                            "row {} witness `{}` does not match scenario `{}`",
                            row.id, w, actual
                        ));
                    }
                }
            }
            many => violations.push(format!(
                "catalog row {} has {} scenarios (want 1)",
                row.id,
                many.len()
            )),
        }
    }

    // Every BDD-class scenario names a real row, has a valid status, and its file
    // declares exactly one scenario.
    for s in scenarios.iter().filter(|s| is_bdd(&s.class)) {
        if !rows.iter().any(|r| r.id == s.id) {
            violations.push(format!(
                "scenario {} ({}) names nonexistent catalog row {}",
                s.name, s.file, s.id
            ));
        }
        if status_tag_to_glyph(&s.status_tag).is_none() {
            violations.push(format!(
                "scenario {} has invalid @status:{}",
                s.name, s.status_tag
            ));
        }
        if s.scenario_count != 1 {
            violations.push(format!(
                "feature {} declares {} scenarios (want exactly 1)",
                s.rel_path, s.scenario_count
            ));
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Bind every externally-witnessed BDD-class row (see `externally_witnessed`) to a present
/// witness: the `file::marker` path must resolve to a real file under `repo_root` whose contents
/// name the cited `marker`. This is the honest analogue of the CC/CS bijection audits for the RM
/// rows the Rust `bdd` gate cannot run — the SDK / browser surfaces verified by their own package
/// tests. `Err(violations)` if any witness is malformed, absent, or does not name its marker.
pub fn check_witnessed_rows(rows: &[CatalogRow], repo_root: &Path) -> Result<(), Vec<String>> {
    let mut violations = Vec::new();
    for row in rows.iter().filter(|r| externally_witnessed(r)) {
        let witness = row.witness.as_deref().unwrap_or_default();
        let Some((path, marker)) = witness.split_once("::") else {
            violations.push(format!(
                "row {} witness `{witness}` is not a `file::marker` path",
                row.id
            ));
            continue;
        };
        match std::fs::read_to_string(repo_root.join(path)) {
            Ok(body) if body.contains(marker) => {}
            Ok(_) => violations.push(format!(
                "row {} witness `{witness}` — `{path}` does not name `{marker}`",
                row.id
            )),
            Err(_) => violations.push(format!(
                "row {} witness `{witness}` — file `{path}` is absent",
                row.id
            )),
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Status;

    fn row(id: &str, status: Status) -> CatalogRow {
        CatalogRow {
            class: id.split_once('-').unwrap().0.to_string(),
            id: id.into(),
            status,
            witness: None,
        }
    }
    fn scn(id: &str, status_tag: &str) -> ScenarioRef {
        ScenarioRef {
            class: id.split_once('-').unwrap().0.to_string(),
            id: id.into(),
            status_tag: status_tag.into(),
            file: format!("{id}.feature"),
            name: id.into(),
            scenario: id.into(),
            rel_path: format!("s0/{id}.feature"),
            scenario_count: 1,
        }
    }

    #[test]
    fn clean_when_in_bijection() {
        let rows = vec![row("GV-1", Status::Gap)];
        let scns = vec![scn("GV-1", "pending")];
        assert!(check_bijection(&rows, &scns).is_ok());
    }

    #[test]
    fn flags_row_without_scenario() {
        let rows = vec![row("GV-1", Status::Gap)];
        let err = check_bijection(&rows, &[]).unwrap_err();
        assert!(err.iter().any(|v| v.contains("has no scenario")));
    }

    #[test]
    fn flags_status_disagreement() {
        let rows = vec![row("GV-1", Status::Enforced)]; // ✅
        let scns = vec![scn("GV-1", "pending")]; // ⛔
        let err = check_bijection(&rows, &scns).unwrap_err();
        assert!(err.iter().any(|v| v.contains("status disagrees")));
    }

    #[test]
    fn flags_witness_mismatch() {
        let rows = vec![CatalogRow {
            class: "GV".into(),
            id: "GV-1".into(),
            status: Status::Gap,
            witness: Some("s6/wrong.feature::wrong name".into()),
        }];
        let scns = vec![scn("GV-1", "pending")];
        let err = check_bijection(&rows, &scns).unwrap_err();
        assert!(err.iter().any(|v| v.contains("witness")));
    }

    #[test]
    fn flags_multiple_scenarios_in_file() {
        let rows = vec![row("GV-1", Status::Gap)];
        let mut s = scn("GV-1", "pending");
        s.scenario_count = 2;
        let scns = vec![s];
        let err = check_bijection(&rows, &scns).unwrap_err();
        assert!(err.iter().any(|v| v.contains("want exactly 1")));
    }

    #[test]
    fn ignores_non_bdd_classes() {
        let rows = vec![row("KC-1", Status::Enforced)]; // not a BDD class
        assert!(check_bijection(&rows, &[]).is_ok());
    }

    fn witnessed_row(id: &str, witness: &str) -> CatalogRow {
        CatalogRow {
            class: id.split_once('-').unwrap().0.to_string(),
            id: id.into(),
            status: Status::Partial,
            witness: Some(witness.into()),
        }
    }

    #[test]
    fn externally_witnessed_row_needs_no_scenario() {
        // An RM row witnessed by an SDK package test (not a `.feature`) is bound by
        // `check_witnessed_rows`, so the scenario bijection demands no Gherkin scenario.
        let rows = vec![witnessed_row("RM-20", "sdk/python/tests/x.py::test_thing")];
        assert!(check_bijection(&rows, &[]).is_ok());
    }

    #[test]
    fn feature_witnessed_rm_row_still_needs_a_scenario() {
        // A `.feature` witness is a BDD scenario row — still policed by the bijection.
        let rows = vec![witnessed_row("RM-3", "s7_readme/x.feature::runs")];
        let err = check_bijection(&rows, &[]).unwrap_err();
        assert!(err.iter().any(|v| v.contains("has no scenario")));
    }

    #[test]
    fn witnessed_rows_bind_present_and_flag_absent() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        // Self-referential witness: this file names `check_witnessed_rows`.
        let present = vec![witnessed_row(
            "RM-20",
            "src/report.rs::check_witnessed_rows",
        )];
        assert!(check_witnessed_rows(&present, root).is_ok());
        // Absent file → violation.
        let absent = vec![witnessed_row("RM-21", "src/does-not-exist.rs::x")];
        assert!(check_witnessed_rows(&absent, root).unwrap_err()[0].contains("is absent"));
        // Present file, marker not named → violation. (Cite `Cargo.toml`, not this file, so the
        // marker literal below can't accidentally satisfy `contains` from the test source itself.)
        let bad = vec![witnessed_row(
            "RM-22",
            "Cargo.toml::marker_not_in_that_file",
        )];
        assert!(check_witnessed_rows(&bad, root).unwrap_err()[0].contains("does not name"));
    }
}
