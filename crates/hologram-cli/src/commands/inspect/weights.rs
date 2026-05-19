//! `--detail weights` output.

use crate::fmt::format_bytes;
use hologram_archive::section::SECTION_WEIGHT_INDEX;
use hologram_archive::weight::TensorMetadata;
use hologram_archive::LoadedPlan;

/// Print weight tensor metadata.
pub fn print(plan: &LoadedPlan) {
    let entry = plan.sections().find(SECTION_WEIGHT_INDEX);
    let Some(entry) = entry else {
        println!("Weights: no weight index section");
        return;
    };
    let raw = plan.weights();
    print_tensors(raw, entry);
}

/// Deserialize and print tensor entries from the weight index.
fn print_tensors(raw: &[u8], entry: &hologram_archive::section::table::SectionEntry) {
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    if end > raw.len() {
        println!("Weights: section data out of bounds");
        return;
    }
    let slice = &raw[start..end];
    let Ok(tensors) = deserialize_tensors(slice) else {
        println!("Weights: failed to deserialize weight index");
        return;
    };
    let total: u64 = tensors.iter().map(|t| t.size).sum();
    print_header(total, tensors.len());
    for tensor in &tensors {
        print_tensor(tensor);
    }
}

/// Try to deserialize the tensor metadata vec.
fn deserialize_tensors(bytes: &[u8]) -> Result<Vec<TensorMetadata>, rkyv::rancor::Error> {
    rkyv::from_bytes::<Vec<TensorMetadata>, rkyv::rancor::Error>(bytes)
}

/// Print the weights header line.
fn print_header(total: u64, count: usize) {
    let noun = if count == 1 { "tensor" } else { "tensors" };
    println!("Weights ({} total, {} {noun}):", format_bytes(total), count);
}

/// Print a single tensor line.
fn print_tensor(t: &TensorMetadata) {
    let shape: Vec<String> = t.shape.iter().map(ToString::to_string).collect();
    println!(
        "  {:20}  [{:>12}]  {:4}  offset={:<8}  size={}",
        format!("{:?}", t.name),
        shape.join(", "),
        t.dtype.name(),
        t.offset,
        format_bytes(t.size),
    );
}
