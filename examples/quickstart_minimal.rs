//! Quickstart 1 — minimal: apply a single LUT operation to a byte buffer.
//!
//! This example is referenced by README.md "Quick start" and is built by
//! `cargo build --examples` so the README's embedded code stays in sync
//! with the actual public API. Any drift breaks this build.
//!
//! Run with: `cargo run --example quickstart_minimal`

use hologram_core::op::LutOp;
use hologram_core::view::ElementWiseView;

fn main() {
    // Build a 256-byte lookup table for the sigmoid LUT op. Materialising
    // the table once amortises the function call across every input.
    let op = LutOp::Sigmoid;
    let view = ElementWiseView::new(|x| op.apply(x));

    // Apply the LUT to some input bytes. This is the simplest possible
    // hologram operation: one O(1) array index per element, no graph,
    // no compiler, no executor.
    let input: Vec<u8> = (0u8..16).collect();
    let output: Vec<u8> = input.iter().map(|&b| view.apply(b)).collect();

    println!("input  = {input:?}");
    println!("output = {output:?}");
    println!("(byte-domain sigmoid via 256-entry LUT, O(1) per element)");
}
