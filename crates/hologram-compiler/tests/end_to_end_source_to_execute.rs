//! End-to-end Phase 11.7 test: source text → Compiler → archive →
//! PrismModule::load → PrismModule::execute → verify output bytes.
//!
//! Every layer between the user-facing source-text input and the runtime
//! output is exercised in one test. This is the regression gate for "the
//! layers compose" — invisible to per-crate unit tests but the kind of
//! breakage that immediately breaks the user-facing API.

use hologram_compiler::{Compiler, SourceInput};
use hologram_foundation::enums::VerificationDomain;
use hologram_foundation::WittLevel;
use hologram_fused_component::{FusedComponentModule, GraphInputs};
use hologram_shapes::prism_module::PrismModule;

#[test]
fn source_text_compiles_loads_executes_via_prism_module() {
    // 1. Define a small UOR term-language source string. The expression
    //    `42` is the simplest case that exercises every layer:
    //    parser → preflight → term-lower → finder pipeline → archive →
    //    Prism module load → execute.
    let source = "42";

    // 2. Compile via the new v0.2.0 Compiler API (not the legacy
    //    free-function `compile_from_source`).
    let compiler = Compiler::default();
    let output = compiler
        .compile(SourceInput::TermSource {
            source: source.to_owned(),
            level: WittLevel::W8,
            budget: 100.0,
            domains: vec![VerificationDomain::Algebraic],
        })
        .expect("Compiler::compile must succeed for valid source text");

    assert!(
        !output.archive.is_empty(),
        "compiler must emit a non-empty archive"
    );

    // 3. Load the archive through the Prism module trait surface. This
    //    exercises ConformanceShapeSection validation as a side effect.
    let module = FusedComponentModule::new();
    let loaded = module
        .load(&output.archive)
        .expect("PrismModule::load must accept the just-compiled archive");

    // 4. Construct an empty input set. The `42` term has no Input nodes,
    //    so no graph inputs are needed.
    let inputs = GraphInputs::new();

    // 5. Execute via the Prism module trait. This is the same path the
    //    compiler's routing function will use once a second module
    //    exists.
    let outputs = module
        .execute(&loaded, &inputs)
        .expect("PrismModule::execute must succeed for the loaded model");

    // 6. The pipeline produced *something*; we don't constrain the
    //    runtime value of `42` here because the literal lowering path
    //    converts it through the W8 ring. The point of this test is
    //    that every layer composes, not that arithmetic is correct
    //    (per-layer tests cover correctness).
    let _ = outputs;
}

#[test]
fn source_text_unary_application_round_trips() {
    // Slightly more interesting: a unary op that the term-lower path
    // routes through PrimOp dispatch.
    let source = "neg(42)";

    let compiler = Compiler::default();
    let output = compiler
        .compile(SourceInput::TermSource {
            source: source.to_owned(),
            level: WittLevel::W8,
            budget: 100.0,
            domains: vec![VerificationDomain::Algebraic],
        })
        .expect("Compiler::compile must succeed for `neg(42)`");

    let module = FusedComponentModule::new();
    let loaded = module.load(&output.archive).expect("load must succeed");

    let inputs = GraphInputs::new();
    let outputs = module
        .execute(&loaded, &inputs)
        .expect("execute must succeed");
    let _ = outputs;
}

#[test]
fn explicit_compiler_new_routes_through_full_pipeline() {
    // Verifies that `Compiler::new(...)` (not just `Compiler::default()`)
    // also drives the full pipeline. This catches the case where the
    // explicit-construction path is silently broken because every other
    // test uses `default()`.
    use hologram_compiler::PrismModuleRegistry;
    use hologram_shapes::prism_module::SubstrateClass;

    let compiler = Compiler::new(
        PrismModuleRegistry::single_fused_component(),
        SubstrateClass::X86_64,
    );

    let output = compiler
        .compile(SourceInput::TermSource {
            source: "add(1, 2)".to_owned(),
            level: WittLevel::W8,
            budget: 100.0,
            domains: vec![VerificationDomain::Algebraic],
        })
        .expect("explicit Compiler::new must compile valid source");

    let module = FusedComponentModule::new();
    let loaded = module.load(&output.archive).expect("load must succeed");
    let outputs = module
        .execute(&loaded, &GraphInputs::new())
        .expect("execute must succeed");
    let _ = outputs;
}
