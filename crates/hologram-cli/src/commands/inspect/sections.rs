//! `--detail sections` output.

use crate::fmt::format_bytes;
use hologram_archive::section::{
    SECTION_CUSTOM_BASE, SECTION_LAYER_HEADER, SECTION_PIPELINE, SECTION_WEIGHT_INDEX,
};
use hologram_archive::LoadedPlan;

/// Print the section table.
pub fn print(plan: &LoadedPlan) {
    let entries = &plan.sections().entries;
    println!("Section Table ({} entries):", entries.len());
    for entry in entries {
        print_entry(entry);
    }
    if entries.is_empty() {
        println!("  (none)");
    }
}

/// Print a single section entry.
fn print_entry(entry: &hologram_archive::section::table::SectionEntry) {
    println!(
        "  Kind {} ({})  offset={}  size={}  checksum={:#010x}",
        entry.kind,
        section_kind_name(entry.kind),
        entry.offset,
        format_bytes(entry.size),
        entry.checksum,
    );
}

/// Map a section kind to a human-readable name.
fn section_kind_name(kind: u32) -> &'static str {
    match kind {
        SECTION_WEIGHT_INDEX => "weight_index",
        SECTION_LAYER_HEADER => "layer_header",
        SECTION_PIPELINE => "pipeline",
        k if k >= SECTION_CUSTOM_BASE => "custom",
        _ => "unknown",
    }
}
