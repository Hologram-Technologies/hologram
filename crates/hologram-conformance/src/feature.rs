//! Parser for Gherkin `.feature` files: extracts the `@class:/@id:/@status:` tags
//! that bind a Feature to a CONFORMANCE.md row, plus the suite-relative path, the
//! first `Scenario:` name, and the scenario count (to enforce one per file).
use std::path::{Path, PathBuf};

/// A scenario's binding to a catalog row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioRef {
    pub class: String,
    pub id: String,
    pub status_tag: String,
    /// Bare filename, e.g. `traceability.feature`.
    pub file: String,
    /// The `Feature:` title.
    pub name: String,
    /// The first `Scenario:` title — what the CONFORMANCE.md Witness path cites.
    pub scenario: String,
    /// Suite-relative path, e.g. `s6_governance/traceability.feature`.
    pub rel_path: String,
    /// Number of `Scenario:` lines in the file (must be exactly 1).
    pub scenario_count: usize,
}

/// Map a `@status:` tag to the CONFORMANCE.md legend glyph.
pub fn status_tag_to_glyph(tag: &str) -> Option<&'static str> {
    match tag {
        "pending" => Some("⛔"),
        "partial" => Some("🟡"),
        "enforced" => Some("✅"),
        _ => None,
    }
}

/// Extract the value of `@key:value` from a whitespace-split tag line.
fn tag_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_whitespace().find_map(|t| t.strip_prefix(key))
}

/// Parse one feature file's tags, Feature title, first Scenario title, and
/// scenario count into a `ScenarioRef`. `rel_path` is the suite-relative path.
fn parse_one(rel_path: &str, body: &str) -> Option<ScenarioRef> {
    let mut class = None;
    let mut id = None;
    let mut status = None;
    let mut name = None;
    let mut scenario = None;
    let mut scenario_count = 0usize;
    for line in body.lines() {
        let line = line.trim();
        if line.starts_with('@') {
            class = class.or_else(|| tag_value(line, "@class:").map(str::to_string));
            id = id.or_else(|| tag_value(line, "@id:").map(str::to_string));
            status = status.or_else(|| tag_value(line, "@status:").map(str::to_string));
        } else if let Some(rest) = line.strip_prefix("Feature:") {
            name = name.or_else(|| Some(rest.trim().to_string()));
        } else if let Some(rest) = line.strip_prefix("Scenario:") {
            scenario_count += 1;
            scenario = scenario.or_else(|| Some(rest.trim().to_string()));
        }
    }
    let file = rel_path.rsplit('/').next().unwrap_or(rel_path).to_string();
    Some(ScenarioRef {
        class: class?,
        id: id?,
        status_tag: status?,
        file,
        name: name?,
        scenario: scenario?,
        rel_path: rel_path.to_string(),
        scenario_count,
    })
}

/// Walk `root` recursively and parse every `*.feature` file.
pub fn parse_features(root: &Path) -> std::io::Result<Vec<ScenarioRef>> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "feature") {
                let body = std::fs::read_to_string(&path)?;
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                if let Some(sref) = parse_one(&rel, &body) {
                    out.push(sref);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tag_block() {
        let body = "@class:GV @id:GV-1 @spec:07-governance @phase:P5 @status:pending\n\
                    Feature: Traceability by κ\n  Scenario: references yields full provenance\n    Given y\n";
        let sref = parse_one("s6_governance/trace.feature", body).unwrap();
        assert_eq!(sref.class, "GV");
        assert_eq!(sref.id, "GV-1");
        assert_eq!(sref.status_tag, "pending");
        assert_eq!(sref.file, "trace.feature");
        assert_eq!(sref.name, "Traceability by κ");
        assert_eq!(sref.scenario, "references yields full provenance");
        assert_eq!(sref.rel_path, "s6_governance/trace.feature");
        assert_eq!(sref.scenario_count, 1);
    }

    #[test]
    fn counts_multiple_scenarios() {
        let body = "@class:GV @id:GV-9 @status:pending\n\
                    Feature: F\n  Scenario: a\n    Given x\n  Scenario: b\n    Given y\n";
        let sref = parse_one("s6_governance/f.feature", body).unwrap();
        assert_eq!(sref.scenario_count, 2);
        assert_eq!(sref.scenario, "a");
    }

    #[test]
    fn status_glyph_mapping() {
        assert_eq!(status_tag_to_glyph("enforced"), Some("✅"));
        assert_eq!(status_tag_to_glyph("partial"), Some("🟡"));
        assert_eq!(status_tag_to_glyph("pending"), Some("⛔"));
        assert_eq!(status_tag_to_glyph("bogus"), None);
    }
}
