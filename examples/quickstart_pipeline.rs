//! Quickstart 2 — full pipeline: build a graph, compile through the
//! v0.2.0 `Compiler` API, load via the `PrismModule` trait, execute, and
//! verify the output.
//!
//! This example is referenced by README.md "Quick start" and is built by
//! `cargo build --examples` so the README's embedded code stays in sync
//! with the actual public API. Any drift breaks this build.
//!
//! Run with: `cargo run --example quickstart_pipeline`

use hologram_compiler::{Compiler, SourceInput};
use hologram_core::op::LutOp;
use hologram_fused_component::{FusedComponentModule, GraphInputs};
use hologram_ir::builder::GraphBuilder;
use hologram_ir::graph::GraphOp;
use hologram_shapes::prism_module::PrismModule;

fn main() {
    // 1. Build a tiny graph: input → sigmoid → relu → output.
    let graph = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
        .node_with_inputs(GraphOp::Output, &[2])
        .output("y", 3)
        .build();

    // 2. Compile via the v0.2.0 structure-finder Compiler. Every archive
    //    carries a ConformanceShapeSection declaring which Shape it
    //    realises — the loader (next step) verifies that section.
    let compiler = Compiler::default();
    let output = compiler
        .compile(SourceInput::Graph(graph))
        .expect("compilation failed");
    println!("compiled: {} archive bytes", output.archive.len());

    // 3. Construct the Prism module that carries F_prism_fused_component.
    //    `load()` validates the archive's conformance shape against this
    //    module's expected shape and returns a runtime-ready LoadedModel.
    let module = FusedComponentModule::new();
    let loaded = module.load(&output.archive).expect("load failed");

    // 4. Execute the loaded model with a single-byte input.
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![100u8]);
    let outputs = module.execute(&loaded, &inputs).expect("execute failed");

    // 5. Read the output by name.
    let y = outputs.by_name("y").expect("output 'y' missing");
    println!("output 'y' = {y:?}");
    println!("(input 100 → sigmoid → relu → {y:?})");
}
