//! The **CC bijection audit** — the honesty backbone for the holospaces V&V absorption (MG-7).
//!
//! Analogous to the BDD meta-gate ([`crate::report::check_bijection`]) but for the non-BDD `CC`
//! class: it binds every `CC` catalog row to a **present, named** witness test in the ported
//! holospaces space (`spaces/holospaces/tests/cc*.rs`), WITHOUT compiling or running those tests
//! and WITHOUT the 170M `vv/artifacts/` tree. Parsing two text inputs (the ledger + the `.rs`
//! sources) is all it needs — which is what makes MG-7 enforceable offline, before the heavy
//! QEMU/browser CI tier exists. It proves the catalog is honest (no row points at a witness that
//! isn't there); the CI tiers prove the witnesses actually pass.

use crate::catalog::CatalogRow;
use std::collections::BTreeSet;
use std::path::Path;

/// A witness test discovered in the holospaces space: the `cc*.rs` file name (e.g.
/// `cc1_kappa_kat.rs`) and a `#[test]`/`#[tokio::test]` fn declared in it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CcWitness {
    /// The test file's base name (e.g. `cc1_kappa_kat.rs`).
    pub file: String,
    /// A test fn name declared under a `#[test]`/`#[tokio::test]` attribute in that file.
    pub test_fn: String,
}

/// Check that every `CC` catalog row cites a witness of the form
/// `…/FILE.rs::FN` that exists in `witnesses`. `Err(violations)` on drift.
///
/// This is the load-bearing check behind MG-7: the catalog cannot claim a component conforms
/// without a real, named witness test present for it.
pub fn check_cc_bijection(
    rows: &[CatalogRow],
    witnesses: &BTreeSet<CcWitness>,
) -> Result<(), Vec<String>> {
    let mut violations = Vec::new();
    let mut cited = 0usize;

    for row in rows.iter().filter(|r| r.class == "CC") {
        cited += 1;
        let Some(w) = &row.witness else {
            violations.push(format!("CC row {} cites no witness test", row.id));
            continue;
        };
        let Some((path, test_fn)) = w.rsplit_once("::") else {
            violations.push(format!(
                "CC row {} witness `{w}` is not of the form `path::fn`",
                row.id
            ));
            continue;
        };
        let file = path.rsplit('/').next().unwrap_or(path);
        let needle = CcWitness {
            file: file.to_string(),
            test_fn: test_fn.to_string(),
        };
        if !witnesses.contains(&needle) {
            violations.push(format!(
                "CC row {} witness `{w}` — no `#[test] fn {test_fn}` found in {file}",
                row.id
            ));
        }
    }

    if cited == 0 {
        violations.push("no CC rows found in the ledger — the CC catalog is missing".to_string());
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Walk `tests_dir` for `cc*.rs` files and collect every `#[test]`/`#[tokio::test]` fn as a
/// [`CcWitness`]. A light source scan — no compilation — mirroring how [`crate::feature`] walks
/// the feature tree. `#[ignore]` (and other attributes) between the test attribute and the `fn`
/// are tolerated: the attribute arms, the next `fn` name is recorded.
pub fn collect_cc_witnesses(tests_dir: &Path) -> std::io::Result<BTreeSet<CcWitness>> {
    let mut out = BTreeSet::new();
    for entry in std::fs::read_dir(tests_dir)? {
        let path = entry?.path();
        let file = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) if n.starts_with("cc") && n.ends_with(".rs") => n.to_string(),
            _ => continue,
        };
        let src = std::fs::read_to_string(&path)?;
        let mut armed = false;
        for line in src.lines() {
            let t = line.trim_start();
            if t.starts_with("#[test]") || t.starts_with("#[tokio::test]") {
                armed = true;
                continue;
            }
            if armed {
                if let Some(name) = fn_name(t) {
                    out.insert(CcWitness {
                        file: file.clone(),
                        test_fn: name,
                    });
                    armed = false;
                }
                // else (e.g. `#[ignore]`, a doc line): stay armed until the fn.
            }
        }
    }
    Ok(out)
}

/// Extract the fn name from a `fn NAME`/`async fn NAME`/`pub fn NAME` line, else `None`.
fn fn_name(line: &str) -> Option<String> {
    let rest = line.strip_prefix("pub ").unwrap_or(line);
    let rest = rest.strip_prefix("async ").unwrap_or(rest);
    let rest = rest.strip_prefix("fn ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Check that every `CS` (specification-conformance) row cites a validator script that exists in
/// `scripts_dir` (`specs/holospaces/scripts/`). CS witnesses are the V1–V8 shell validators, not
/// Rust tests, so this binds by file presence — the honesty check that the docs V&V is really
/// absorbed (MG-8), independent of the docs toolchain. `Err(violations)` on drift.
pub fn check_cs_bijection(rows: &[CatalogRow], scripts_dir: &Path) -> Result<(), Vec<String>> {
    let mut violations = Vec::new();
    let mut cited = 0usize;
    for row in rows.iter().filter(|r| r.class == "CS") {
        cited += 1;
        let Some(w) = &row.witness else {
            violations.push(format!("CS row {} cites no validator script", row.id));
            continue;
        };
        let file = w.rsplit('/').next().unwrap_or(w);
        if !scripts_dir.join(file).exists() {
            violations.push(format!(
                "CS row {} witness `{w}` — validator script {file} not found in the docs scripts",
                row.id
            ));
        }
    }
    if cited == 0 {
        violations.push("no CS rows found in the ledger — the CS catalog is missing".to_string());
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
    use crate::catalog::{CatalogRow, Status};

    fn cc_row(id: &str, witness: Option<&str>) -> CatalogRow {
        CatalogRow {
            class: "CC".into(),
            id: id.into(),
            status: Status::Partial,
            witness: witness.map(str::to_string),
        }
    }
    fn wit(file: &str, f: &str) -> CcWitness {
        CcWitness {
            file: file.into(),
            test_fn: f.into(),
        }
    }

    #[test]
    fn clean_when_every_row_binds() {
        let rows = vec![cc_row(
            "CC-1",
            Some("spaces/holospaces/tests/cc1_kappa_kat.rs::kappa_digest_equals_reference"),
        )];
        let w: BTreeSet<_> = [wit("cc1_kappa_kat.rs", "kappa_digest_equals_reference")]
            .into_iter()
            .collect();
        assert!(check_cc_bijection(&rows, &w).is_ok());
    }

    #[test]
    fn flags_missing_witness_fn() {
        let rows = vec![cc_row(
            "CC-1",
            Some("spaces/holospaces/tests/cc1_kappa_kat.rs::renamed_away"),
        )];
        let w: BTreeSet<_> = [wit("cc1_kappa_kat.rs", "the_real_fn")]
            .into_iter()
            .collect();
        let err = check_cc_bijection(&rows, &w).unwrap_err();
        assert!(err
            .iter()
            .any(|v| v.contains("no `#[test] fn renamed_away`")));
    }

    #[test]
    fn flags_row_without_witness() {
        let rows = vec![cc_row("CC-1", None)];
        let err = check_cc_bijection(&rows, &BTreeSet::new()).unwrap_err();
        assert!(err.iter().any(|v| v.contains("cites no witness")));
    }

    #[test]
    fn flags_empty_catalog() {
        let err = check_cc_bijection(&[], &BTreeSet::new()).unwrap_err();
        assert!(err.iter().any(|v| v.contains("CC catalog is missing")));
    }

    #[test]
    fn fn_name_parses_forms() {
        assert_eq!(fn_name("fn a_test() {").as_deref(), Some("a_test"));
        assert_eq!(fn_name("async fn b() {").as_deref(), Some("b"));
        assert_eq!(fn_name("pub async fn c() {").as_deref(), Some("c"));
        assert_eq!(fn_name("let x = 1;"), None);
    }
}
