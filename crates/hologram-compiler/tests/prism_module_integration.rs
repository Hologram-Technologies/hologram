//! End-to-end integration test for the Prism module pipeline.
//!
//! Verifies that the v0.2.0 conformance scaffolding (`PrismModule` trait,
//! `F_prism_fused_component` shape, conformance test families) actually
//! validates the rebuilt `hologram-exec` Prism module.
//!
//! This test exercises the keystone wiring of Phase 6:
//!
//! 1. Build a small test graph via `hologram_ir::GraphBuilder`.
//! 2. Compile it to a `.holo` archive via `hologram_compiler::compile`.
//! 3. Load the archive into `FusedComponentModule` via the `PrismModule`
//!    trait's `load()` method.
//! 4. Execute the loaded module via the trait's `execute()` method.
//! 5. Run the conformance test families against the module.
//!
//! Per the conformance-first principle, this test is what ties the abstract
//! `Shape` declaration in `hologram-shapes` to the concrete tape executor
//! in `hologram-fused-component`. Failure here means the trait
//! surface doesn't compose.

use hologram_compiler::compile;
use hologram_core::op::LutOp;
use hologram_fused_component::{FusedComponentModule, GraphInputs};
use hologram_ir::builder::GraphBuilder;
use hologram_ir::graph::GraphOp;
use hologram_shapes::conformance_tests::{test_full_conformance, test_transition_fidelity};
use hologram_shapes::prism_module::PrismModule;
use hologram_shapes::shape::F_PRISM_FUSED_COMPONENT;

/// Build a simple linear chain: Input → Sigmoid → Relu → Output.
fn linear_chain_graph() -> hologram_ir::Graph {
    GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
        .node_with_inputs(GraphOp::Output, &[2])
        .output("y", 3)
        .build()
}

#[test]
fn prism_module_loads_compiled_archive() {
    // 1. Compile a graph to a .holo archive.
    let graph = linear_chain_graph();
    let output = compile(graph).expect("compilation must succeed");
    assert!(!output.archive.is_empty(), "archive bytes must be produced");

    // 2. Load the archive into the Prism module via the trait surface.
    let module = FusedComponentModule::new();
    let loaded = module
        .load(&output.archive)
        .expect("PrismModule::load must accept compiler-emitted archive");

    // 3. Verify the loaded model has the expected structure. The analysis
    // pass may fuse Sigmoid→Relu into a single FusedView, so the post-fold
    // node count is lower than the pre-compile count.
    assert!(
        !loaded.tape().instructions.is_empty(),
        "loaded tape must have instructions"
    );
    assert!(
        !loaded.plan().graph().nodes.is_empty(),
        "loaded plan must include at least the I/O nodes"
    );
}

#[test]
fn prism_module_executes_compiled_archive() {
    // Compile a graph and execute it via the Prism module trait.
    let graph = linear_chain_graph();
    let output = compile(graph).expect("compilation must succeed");

    let module = FusedComponentModule::new();
    let loaded = module.load(&output.archive).expect("load must succeed");

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![100u8]);

    let outputs = module
        .execute(&loaded, &inputs)
        .expect("execute must succeed via PrismModule trait");

    let result = outputs.by_name("y").expect("output 'y' must be present");
    // Sigmoid(100) → Relu(...) — the exact value is not the point; we
    // just verify execution completes and produces output of the right
    // shape.
    assert_eq!(result.len(), 1, "single-byte chain produces single output");
}

#[test]
fn prism_module_shape_is_fused_component() {
    let module = FusedComponentModule::new();
    assert_eq!(module.shape().id, F_PRISM_FUSED_COMPONENT.id);
    assert_eq!(module.shape().name, "F_prism_fused_component");
    assert_eq!(module.name(), "hologram-fused-component");
}

#[test]
fn transition_fidelity_passes_for_fused_component_module() {
    // Per the carrying criterion's transition fidelity condition: the
    // module's primitive_operations() must match the shape's declared
    // algebra exactly. By construction (FusedComponentModule returns
    // F_PRISM_FUSED_COMPONENT.primitives directly), this is trivially
    // true — but the test confirms the wiring is correct.
    let module = FusedComponentModule::new();
    let report = test_transition_fidelity(&module);
    assert!(
        report.passed,
        "transition fidelity must hold for FusedComponentModule. Failures: {:?}",
        report.failures
    );
    // The shape declares ~60 primitives.
    assert!(
        report.operations_checked >= 50,
        "expected ~60 primitives, got {}",
        report.operations_checked
    );
}

#[test]
fn full_conformance_report_for_default_module() {
    // Phase 10.1: with `execute_traced` overridden to populate the trace
    // from the loaded model's tape, all three test families produce real
    // measurements when handed a real compiled archive. The integration
    // test compiles a representative graph, hands the archive bytes to
    // the conformance helpers, and asserts every family passes.
    let archive = compile(linear_chain_graph()).unwrap().archive;

    let module = FusedComponentModule::new();
    let input_factory = || {
        let mut inputs = GraphInputs::new();
        inputs.set(0, vec![100u8]);
        inputs
    };
    let report = test_full_conformance(&module, &archive, input_factory);

    assert!(
        report.transition_fidelity.passed,
        "transition fidelity must hold: {:?}",
        report.transition_fidelity.failures
    );
    assert!(
        report.state_space_identity.passed,
        "state-space identity must hold: {:?}",
        report.state_space_identity.failures
    );
    assert!(
        report.primitivity.passed,
        "primitivity must hold: {:?}",
        report.primitivity.failures
    );
    let directness = report
        .directness_ratio
        .expect("directness ratio must be measurable");
    assert!(
        (directness - 1.0).abs() < 1e-9,
        "FusedComponentModule must achieve directness 1.0; got {directness}"
    );
}

#[test]
fn compiled_archive_carries_conformance_shape_section() {
    // Per the v0.2.0 conformance-first contract, every archive emitted
    // by hologram-compiler must carry a ConformanceShapeSection.
    use hologram_archive::loader::bytes::load_from_bytes;
    use hologram_archive::section::SECTION_CONFORMANCE_SHAPE;

    let archive = compile(linear_chain_graph()).unwrap().archive;
    let plan = load_from_bytes(&archive).expect("archive must be loadable");

    // The section table must contain the conformance shape entry.
    let entry = plan.sections().find(SECTION_CONFORMANCE_SHAPE);
    assert!(
        entry.is_some(),
        "compiled archive must declare its conformance shape"
    );

    // Read the section payload via the loader helper.
    let section = plan
        .conformance_shape_from_bytes(&archive)
        .expect("conformance shape section must parse");
    assert_eq!(section.shape_id, *F_PRISM_FUSED_COMPONENT.id.as_bytes());
    assert_eq!(section.shape_name, "F_prism_fused_component");
    assert_eq!(
        section.primitive_count as usize,
        F_PRISM_FUSED_COMPONENT.primitives.len()
    );
}

#[test]
fn prism_module_load_rejects_archive_with_wrong_shape_id() {
    // Synthesise a corrupted archive: take a real compiled archive and
    // overwrite the shape ID in the conformance section so it no longer
    // matches F_prism_fused_component. The PrismModule::load() path must
    // refuse it with ShapeMismatch.
    use hologram_archive::loader::bytes::load_from_bytes;
    use hologram_archive::section::conformance_shape::ConformanceShapeSection;
    use hologram_archive::section::{EmbeddableSection, SECTION_CONFORMANCE_SHAPE};
    use hologram_shapes::prism_module::LoadError;

    let mut archive = compile(linear_chain_graph()).unwrap().archive;

    // Locate the conformance shape section in the archive.
    let plan = load_from_bytes(&archive).unwrap();
    let entry = plan
        .sections()
        .find(SECTION_CONFORMANCE_SHAPE)
        .expect("section present")
        .clone();

    // Build a replacement section with a deliberately-wrong shape ID
    // (a single byte flipped is enough to change the BLAKE3-derived ID).
    let mut wrong_id = *F_PRISM_FUSED_COMPONENT.id.as_bytes();
    wrong_id[0] ^= 0xFF;
    let wrong_section = ConformanceShapeSection::new(
        wrong_id,
        F_PRISM_FUSED_COMPONENT.target_class,
        F_PRISM_FUSED_COMPONENT.name,
        F_PRISM_FUSED_COMPONENT.primitives.len() as u32,
        8,
        8,
    );
    let wrong_bytes = wrong_section.to_bytes();
    assert_eq!(
        wrong_bytes.len() as u64,
        entry.size,
        "replacement section must be the same size for in-place patching"
    );

    // Patch the archive bytes in place.
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    archive[start..end].copy_from_slice(&wrong_bytes);

    // Now ask the Prism module to load the patched archive.
    let module = FusedComponentModule::new();
    let result = module.load(&archive);

    match result {
        Err(LoadError::ShapeMismatch { expected, found }) => {
            assert_eq!(expected, F_PRISM_FUSED_COMPONENT.id);
            assert_ne!(found, F_PRISM_FUSED_COMPONENT.id);
        }
        Err(other) => panic!("expected ShapeMismatch, got {:?}", other),
        Ok(_) => panic!("expected load to fail on shape mismatch"),
    }
}

#[test]
fn module_archive_round_trip_preserves_outputs() {
    // Compile twice with the same graph → load via PrismModule → execute
    // → outputs must be byte-identical. This is a cross-check that the
    // compile/load/execute pipeline is deterministic at the trait surface.
    let g1 = linear_chain_graph();
    let g2 = linear_chain_graph();

    let archive1 = compile(g1).unwrap().archive;
    let archive2 = compile(g2).unwrap().archive;

    let module = FusedComponentModule::new();
    let m1 = module.load(&archive1).unwrap();
    let m2 = module.load(&archive2).unwrap();

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![42u8]);

    let out1 = module.execute(&m1, &inputs).unwrap();
    let out2 = module.execute(&m2, &inputs).unwrap();

    let r1 = out1.by_name("y").unwrap();
    let r2 = out2.by_name("y").unwrap();
    assert_eq!(r1, r2, "deterministic compile/execute pipeline");
}
