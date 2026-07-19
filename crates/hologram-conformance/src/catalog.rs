//! Parser for the `CONFORMANCE.md` normative ledger.
//!
//! A row is a markdown table line whose first cell is a bold row id
//! (`| **LAW-1** | … | ✅ |`). We extract the id, its class prefix, the trailing
//! status glyph, and (when present) the backtick-wrapped `Witness` path of the
//! form `sN_.../file.feature::Scenario name`.

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

/// One catalog row: class prefix, full id, declared status, and the witness
/// path (`file.feature::Scenario name`) when the row cites a BDD scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogRow {
    pub class: String,
    pub id: String,
    pub status: Status,
    pub witness: Option<String>,
}

/// Extract a `sN_.../file.feature::Scenario name` witness from a table cell,
/// stripping any surrounding backticks/whitespace. `None` if the cell holds no
/// feature-scenario witness of that form.
fn extract_witness(cell: &str) -> Option<String> {
    let token = cell.trim().trim_matches('`').trim();
    // A witness is a `file::item` path — a Gherkin scenario (`.feature::…`, the BDD classes), a
    // Rust test (`.rs::…`, the non-BDD AS/KC/CC classes), or a validator script (`…/scripts/…`
    // ending `.sh`, the CS class). Each gate only consults the form it owns.
    let is_script = token.contains("/scripts/") && token.ends_with(".sh");
    if token.contains(".feature::") || token.contains(".rs::") || is_script {
        Some(token.to_string())
    } else {
        None
    }
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
        let Some(id_cell) = cells.get(1) else {
            continue;
        };
        let Some(id) = id_cell
            .strip_prefix("**")
            .and_then(|s| s.strip_suffix("**"))
        else {
            continue;
        };
        // id looks like LAW-1 / KC-6 / GV-3 — split on the first '-'.
        let Some((class, _)) = id.split_once('-') else {
            continue;
        };
        // Status is the rightmost cell that parses as a legend glyph. Scanning from the
        // right (rather than assuming the last non-empty cell) tolerates any trailing
        // columns a future ledger schema might add to the right of Status.
        let Some(status) = cells.iter().rev().find_map(|c| Status::from_legend(c)) else {
            continue;
        };
        let witness = cells.iter().find_map(|c| extract_witness(c));
        rows.push(CatalogRow {
            class: class.to_string(),
            id: id.to_string(),
            status,
            witness,
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
        assert_eq!(
            rows[0],
            CatalogRow {
                class: "LAW".into(),
                id: "LAW-1".into(),
                status: Status::Gap,
                witness: None,
            }
        );
        assert_eq!(rows[1].status, Status::Partial);
        assert_eq!(
            rows[2],
            CatalogRow {
                class: "KC".into(),
                id: "KC-1".into(),
                status: Status::Enforced,
                witness: None,
            }
        );
    }

    #[test]
    fn ignores_non_row_lines() {
        assert!(parse_catalog("just prose\n| header | only |\n").is_empty());
    }

    #[test]
    fn finds_status_when_a_column_sits_right_of_it() {
        let md = "| **GV-1** | traceability | BDD scenario | trace.feature | ⛔ | see note |\n";
        let rows = parse_catalog(md);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "GV-1");
        assert_eq!(rows[0].status, Status::Gap);
    }

    #[test]
    fn extracts_backtick_wrapped_witness() {
        let md = "| **GV-1** | traceability | BDD scenario | `s6_governance/trace.feature::references yields full provenance` | ⛔ |\n";
        let rows = parse_catalog(md);
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].witness.as_deref(),
            Some("s6_governance/trace.feature::references yields full provenance")
        );
    }
}
