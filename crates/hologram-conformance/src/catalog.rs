//! Parser for the `CONFORMANCE.md` normative ledger.
//!
//! A row is a markdown table line whose first cell is a bold row id
//! (`| **LAW-1** | … | ✅ |`). We extract the id, its class prefix, and the
//! trailing status glyph.

/// Enforcement status, single-sourced from the CONFORMANCE.md legend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Enforced,
    Partial,
    Gap,
}

impl Status {
    /// Map a legend glyph to a status. Returns `None` for anything else.
    pub fn from_legend(glyph: &str) -> Option<Status> {
        match glyph.trim() {
            "✅" => Some(Status::Enforced),
            "🟡" => Some(Status::Partial),
            "⛔" => Some(Status::Gap),
            _ => None,
        }
    }
}

/// One catalog row: class prefix, full id, declared status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogRow {
    pub class: String,
    pub id: String,
    pub status: Status,
}

/// Parse every id-bearing table row from a CONFORMANCE.md body.
pub fn parse_catalog(md: &str) -> Vec<CatalogRow> {
    let mut rows = Vec::new();
    for line in md.lines() {
        let line = line.trim();
        if !line.starts_with("| **") {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // cells[0] is empty (leading pipe); cells[1] is the id cell.
        let Some(id_cell) = cells.get(1) else { continue };
        let Some(id) = id_cell.strip_prefix("**").and_then(|s| s.strip_suffix("**")) else {
            continue;
        };
        // id looks like LAW-1 / KC-6 / GV-3 — split on the first '-'.
        let Some((class, _)) = id.split_once('-') else { continue };
        // The last non-empty cell holds the status glyph.
        let Some(glyph) = cells.iter().rev().find(|c| !c.is_empty()) else { continue };
        let Some(status) = Status::from_legend(glyph) else { continue };
        rows.push(CatalogRow {
            class: class.to_string(),
            id: id.to_string(),
            status,
        });
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "\
## LAW — repo laws
| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **LAW-1** | canonical bytes or nothing | BDD scenario | s0_laws/spine.feature | ⛔ |
| **LAW-2** | κ-only identity | BDD scenario | s0_laws/identity.feature | 🟡 |
| **KC-1** | matmul conforms | test | conformance.rs | ✅ |
";

    #[test]
    fn parses_ids_classes_and_status() {
        let rows = parse_catalog(FIXTURE);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], CatalogRow { class: "LAW".into(), id: "LAW-1".into(), status: Status::Gap });
        assert_eq!(rows[1].status, Status::Partial);
        assert_eq!(rows[2], CatalogRow { class: "KC".into(), id: "KC-1".into(), status: Status::Enforced });
    }

    #[test]
    fn ignores_non_row_lines() {
        assert!(parse_catalog("just prose\n| header | only |\n").is_empty());
    }
}
