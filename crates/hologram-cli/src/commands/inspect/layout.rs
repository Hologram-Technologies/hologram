//! `--detail layout` output — visual byte-map of the `.holo` archive.

use crate::fmt::format_bytes;
use hologram_archive::section::{
    SECTION_CUSTOM_BASE, SECTION_LAYER_HEADER, SECTION_PIPELINE, SECTION_WEIGHT_INDEX,
};
use hologram_archive::LoadedPlan;

/// Total width of the diagram bar (characters).
const BAR_WIDTH: usize = 60;

/// A contiguous region in the archive.
struct Region {
    label: String,
    offset: u64,
    size: u64,
}

/// Print the visual archive layout.
pub fn print(data: &[u8], plan: &LoadedPlan) {
    let regions = collect_regions(plan);
    let total = data.len() as u64;
    print_header(total);
    print_bar(&regions, total);
    println!();
    print_table(&regions, total);
    print_gaps(&regions, total);
}

/// Collect all known regions from the header and section table.
fn collect_regions(plan: &LoadedPlan) -> Vec<Region> {
    let h = plan.header();
    let mut regions = vec![
        Region {
            label: "Header".into(),
            offset: 0,
            size: h.graph_offset,
        },
        Region {
            label: "Graph".into(),
            offset: h.graph_offset,
            size: h.graph_size,
        },
    ];
    for entry in &plan.sections().entries {
        regions.push(Region {
            label: section_label(entry.kind),
            offset: entry.offset,
            size: entry.size,
        });
    }
    if h.section_table_size > 0 {
        regions.push(Region {
            label: "Section Table".into(),
            offset: h.section_table_offset,
            size: h.section_table_size,
        });
    }
    if h.weights_size > 0 {
        regions.push(Region {
            label: "Weights".into(),
            offset: h.weights_offset,
            size: h.weights_size,
        });
    }
    regions.sort_by_key(|r| r.offset);
    regions
}

/// Map section kind to a display label.
fn section_label(kind: u32) -> String {
    match kind {
        SECTION_WEIGHT_INDEX => "Sec: weight_index".into(),
        SECTION_LAYER_HEADER => "Sec: layer_header".into(),
        SECTION_PIPELINE => "Sec: pipeline".into(),
        k if k >= SECTION_CUSTOM_BASE => format!("Sec: custom({})", k),
        k => format!("Sec: kind({})", k),
    }
}

/// Print the title line.
fn print_header(total: u64) {
    println!("Archive Layout ({}, {} bytes):", format_bytes(total), total);
}

/// Print a proportional ASCII bar.
fn print_bar(regions: &[Region], total: u64) {
    if total == 0 {
        println!("  (empty archive)");
        return;
    }
    let mut bar = vec![b'.'; BAR_WIDTH];
    let chars = b"HGWSTI???????";
    for region in regions {
        let ch = region_char(&region.label, chars);
        let start = ((region.offset as f64 / total as f64) * BAR_WIDTH as f64) as usize;
        let width = ((region.size as f64 / total as f64) * BAR_WIDTH as f64).ceil() as usize;
        let end = (start + width).min(BAR_WIDTH);
        for cell in &mut bar[start..end] {
            *cell = ch;
        }
    }
    let s = String::from_utf8_lossy(&bar);
    println!("  |{s}|");
    print_legend(regions, chars);
}

/// Pick a character for the region.
fn region_char(label: &str, _chars: &[u8]) -> u8 {
    match label {
        "Header" => b'H',
        "Graph" => b'G',
        "Weights" => b'W',
        "Section Table" => b'T',
        _ if label.starts_with("Sec:") => b'S',
        _ => b'?',
    }
}

/// Print a compact legend.
fn print_legend(regions: &[Region], chars: &[u8]) {
    let mut seen = Vec::new();
    for r in regions {
        let ch = region_char(&r.label, chars) as char;
        let key = (ch, r.label.clone());
        if !seen.iter().any(|(c, _): &(char, String)| *c == ch) {
            seen.push(key);
        }
    }
    let items: Vec<String> = seen
        .iter()
        .map(|(ch, label)| format!("{ch}={label}"))
        .collect();
    println!("   {}  .=padding/gap", items.join("  "));
}

/// Print the region offset/size table.
fn print_table(regions: &[Region], total: u64) {
    let max_label = regions.iter().map(|r| r.label.len()).max().unwrap_or(0);
    let w = max_label.max(6);
    println!(
        "  {:<w$}  {:>10}  {:>10}  {:>10}  {:>5}",
        "Region", "Offset", "End", "Size", "%"
    );
    println!(
        "  {:-<w$}  {:->10}  {:->10}  {:->10}  {:->5}",
        "", "", "", "", ""
    );
    for r in regions {
        let end = r.offset + r.size;
        let pct = if total > 0 {
            (r.size as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<w$}  {:>10}  {:>10}  {:>10}  {:>4.1}%",
            r.label,
            r.offset,
            end,
            format_bytes(r.size),
            pct,
        );
    }
}

/// Detect and print gaps (alignment padding) between regions.
fn print_gaps(regions: &[Region], total: u64) {
    let mut gaps = Vec::new();
    for pair in regions.windows(2) {
        let end = pair[0].offset + pair[0].size;
        let next = pair[1].offset;
        if next > end {
            gaps.push((end, next - end));
        }
    }
    if let Some(last) = regions.last() {
        let end = last.offset + last.size;
        if end < total {
            gaps.push((end, total - end));
        }
    }
    if gaps.is_empty() {
        println!("\n  No alignment gaps.");
    } else {
        let gap_total: u64 = gaps.iter().map(|(_, s)| s).sum();
        println!(
            "\n  Alignment Gaps ({}, {:.1}% of file):",
            format_bytes(gap_total),
            (gap_total as f64 / total as f64) * 100.0
        );
        for (offset, size) in &gaps {
            println!("    offset={:<10}  size={}", offset, format_bytes(*size));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_chars_are_distinct() {
        let chars = b"HGWSTI???????";
        assert_eq!(region_char("Header", chars), b'H');
        assert_eq!(region_char("Graph", chars), b'G');
        assert_eq!(region_char("Weights", chars), b'W');
        assert_eq!(region_char("Section Table", chars), b'T');
        assert_eq!(region_char("Sec: weight_index", chars), b'S');
    }

    #[test]
    fn section_label_known_kinds() {
        assert_eq!(section_label(1), "Sec: weight_index");
        assert_eq!(section_label(2), "Sec: layer_header");
        assert_eq!(section_label(3), "Sec: pipeline");
    }

    #[test]
    fn section_label_custom() {
        assert!(section_label(0x1000).starts_with("Sec: custom"));
    }

    #[test]
    fn collect_regions_sorted() {
        use hologram_archive::writer::holo_writer::HoloWriter;
        let data = HoloWriter::new().build().unwrap();
        let plan = hologram_archive::load_from_bytes(&data).unwrap();
        let regions = collect_regions(&plan);
        for pair in regions.windows(2) {
            assert!(pair[0].offset <= pair[1].offset);
        }
    }
}
