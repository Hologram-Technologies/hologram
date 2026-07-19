//! The honesty meta-gate: a *static* check that the CONFORMANCE.md catalog and the
//! Gherkin scenarios are in bijection for the seven BDD classes. It verifies, for
//! every BDD-class row: exactly one scenario with the same `@id`; the row's status
//! glyph agrees with the scenario's `@status` tag; the row's `Witness` path+scenario
//! name matches the actual file; and the file declares exactly one scenario.
use crate::catalog::CatalogRow;
use crate::feature::{status_tag_to_glyph, ScenarioRef};

/// The refactor classes whose rows are witnessed by BDD scenarios.
pub const BDD_CLASSES: &[&str] = &["LAW", "SP", "HF", "NW", "TL", "MG", "GV"];

fn is_bdd(class: &str) -> bool {
    BDD_CLASSES.contains(&class)
}

/// Check row↔scenario bijection, status agreement, witness binding, and
/// one-scenario-per-file for the BDD classes only. `Err(violations)` when off.
pub fn check_bijection(rows: &[CatalogRow], scenarios: &[ScenarioRef]) -> Result<(), Vec<String>> {
    let mut violations = Vec::new();

    // Every BDD-class catalog row has exactly one scenario, with matching status
    // and matching witness path/name.
    for row in rows.iter().filter(|r| is_bdd(&r.class)) {
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
}
