//! `--detail sections` output.

use crate::fmt::format_bytes;
use hologram_archive::section::{
    SECTION_COMPILE_UNIT_META, SECTION_CUSTOM_BASE, SECTION_LAYER_HEADER, SECTION_PIPELINE,
    SECTION_WEIGHT_DEDUP, SECTION_WEIGHT_INDEX,
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
    let hex: String = entry.checksum.iter().map(|b| format!("{b:02x}")).collect();
    println!(
        "  Kind {} ({})  offset={}  size={}  checksum={}",
        entry.kind,
        section_kind_name(entry.kind),
        entry.offset,
        format_bytes(entry.size),
        hex,
    );
}

/// Map a section kind to a human-readable name.
fn section_kind_name(kind: u32) -> &'static str {
    match kind {
        SECTION_WEIGHT_INDEX => "weight_index",
        SECTION_LAYER_HEADER => "layer_header",
        SECTION_PIPELINE => "pipeline",
        SECTION_WEIGHT_DEDUP => "weight_dedup",
        SECTION_COMPILE_UNIT_META => "compile_unit_meta",
        k if k >= SECTION_CUSTOM_BASE => "custom",
        _ => "unknown",
    }
}
