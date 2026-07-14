//! Parser for Gherkin `.feature` files: extracts the `@class:/@id:/@status:` tags
//! that bind a Feature to a CONFORMANCE.md row.
use std::path::{Path, PathBuf};

/// A scenario's binding to a catalog row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioRef {
    pub class: String,
    pub id: String,
    pub status_tag: String,
    pub file: String,
    pub name: String,
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

/// Parse one feature file's leading tag block + Feature line into a `ScenarioRef`.
fn parse_one(path: &Path, body: &str) -> Option<ScenarioRef> {
    let mut class = None;
    let mut id = None;
    let mut status = None;
    let mut name = None;
    for line in body.lines() {
        let line = line.trim();
        if line.starts_with('@') {
            class = class.or_else(|| tag_value(line, "@class:").map(str::to_string));
            id = id.or_else(|| tag_value(line, "@id:").map(str::to_string));
            status = status.or_else(|| tag_value(line, "@status:").map(str::to_string));
        } else if let Some(rest) = line.strip_prefix("Feature:") {
            name = Some(rest.trim().to_string());
            break;
        }
    }
    Some(ScenarioRef {
        class: class?,
        id: id?,
        status_tag: status?,
        file: path.file_name()?.to_string_lossy().into_owned(),
        name: name?,
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
                if let Some(sref) = parse_one(&path, &body) {
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
                    Feature: Traceability by κ\n  Scenario: x\n    Given y\n";
        let sref = parse_one(Path::new("/x/trace.feature"), body).unwrap();
        assert_eq!(sref.class, "GV");
        assert_eq!(sref.id, "GV-1");
        assert_eq!(sref.status_tag, "pending");
        assert_eq!(sref.file, "trace.feature");
        assert_eq!(sref.name, "Traceability by κ");
    }

    #[test]
    fn status_glyph_mapping() {
        assert_eq!(status_tag_to_glyph("enforced"), Some("✅"));
        assert_eq!(status_tag_to_glyph("pending"), Some("⛔"));
        assert_eq!(status_tag_to_glyph("bogus"), None);
    }
}
