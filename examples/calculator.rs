//! Scientific calculator example demonstrating the full hologram pipeline.
//!
//! Shows:
//! 1. Pi-F-lambda encoding: f64 → byte → LUT → byte → f64
//! 2. LUT composition via view fusion (e.g. sin(cos(x)))
//! 3. Graph I/O with named inputs/outputs
//! 4. Full pipeline: build → fuse → serialize → load → execute
//! 5. Error analysis: LUT approximation vs f64 reference

use hologram_core::encoding::{AngleEncoding, Encoding, SignedEncoding, UnsignedEncoding};
use hologram_core::op::{LutOp, PrimOp};
use hologram_core::view::ElementWiseView;
use hologram_graph::builder::GraphBuilder;
use hologram_graph::fusion;
use hologram_graph::graph::GraphOp;

use hologram_archive::HoloWriter;

use hologram_exec::{execute_bytes, GraphInputs};

fn main() {
    println!("=== Hologram Scientific Calculator ===\n");

    demo_pi_f_lambda();
    demo_lut_composition();
    demo_graph_io();
    demo_full_pipeline();
}

// ---------------------------------------------------------------------------
// Demo 1: Pi-F-Lambda encoding round-trip
// ---------------------------------------------------------------------------

fn demo_pi_f_lambda() {
    println!("--- Demo 1: Pi-F-Lambda Encoding ---");
    println!("  f64 → embed(pi) → LUT[F] → lift(lambda) → f64\n");

    let angle = AngleEncoding;
    let signed = SignedEncoding;
    let unsigned = UnsignedEncoding;

    // Test sin via angle encoding
    let test_values = [0.0_f64, 0.5, 1.0, 2.0, std::f64::consts::PI, 5.0];
    println!(
        "  {:>10} {:>10} {:>10} {:>10}",
        "input", "f64_sin", "lut_sin", "error"
    );
    for &x in &test_values {
        let byte_in = angle.embed(x);
        let byte_out = LutOp::Sin.apply(byte_in);
        let lut_result = signed.lift(byte_out);
        let f64_result = x.sin();
        let error = (lut_result - f64_result).abs();
        println!(
            "  {:>10.4} {:>10.6} {:>10.6} {:>10.6}",
            x, f64_result, lut_result, error
        );
    }

    // Test sqrt via unsigned encoding
    println!();
    println!(
        "  {:>10} {:>10} {:>10} {:>10}",
        "input", "f64_sqrt", "lut_sqrt", "error"
    );
    let sqrt_values = [0.0_f64, 0.1, 0.25, 0.5, 0.75, 1.0];
    for &x in &sqrt_values {
        let byte_in = unsigned.embed(x);
        let byte_out = LutOp::Sqrt.apply(byte_in);
        let lut_result = unsigned.lift(byte_out);
        let f64_result = x.sqrt();
        let error = (lut_result - f64_result).abs();
        println!(
            "  {:>10.4} {:>10.6} {:>10.6} {:>10.6}",
            x, f64_result, lut_result, error
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Demo 2: LUT composition via ElementWiseView
// ---------------------------------------------------------------------------

fn demo_lut_composition() {
    println!("--- Demo 2: LUT Composition (sin(cos(x))) ---");

    // Build composed view: cos → sin
    let cos_view = ElementWiseView::from_table(*LutOp::Cos.table());
    let sin_view = ElementWiseView::from_table(*LutOp::Sin.table());
    let composed = cos_view.then(&sin_view);

    // Compare composed single-lookup vs chained lookups
    println!(
        "  {:>6} {:>12} {:>12} {:>6}",
        "byte", "chained", "composed", "match"
    );
    let test_bytes: [u8; 8] = [0, 32, 64, 96, 128, 160, 192, 224];
    for &b in &test_bytes {
        let chained = LutOp::Sin.apply(LutOp::Cos.apply(b));
        let fused = composed.apply(b);
        println!(
            "  {:>6} {:>12} {:>12} {:>6}",
            b,
            chained,
            fused,
            if chained == fused { "yes" } else { "NO" }
        );
    }

    // Also show exp2(log(x)) ≈ identity
    println!("\n  exp2(log(x)) ≈ identity?");
    let log_view = ElementWiseView::from_table(*LutOp::Log.table());
    let exp2_view = ElementWiseView::from_table(*LutOp::Exp2.table());
    let roundtrip = log_view.then(&exp2_view);
    let mut matches = 0;
    for b in 0u8..=255 {
        if roundtrip.apply(b) == b {
            matches += 1;
        }
    }
    println!("  {matches}/256 byte values round-trip exactly");
    println!();
}

// ---------------------------------------------------------------------------
// Demo 3: Graph I/O with named inputs/outputs
// ---------------------------------------------------------------------------

fn demo_graph_io() {
    println!("--- Demo 3: Graph with Named I/O ---");

    // Build: x → relu(x), sigmoid(x), abs(x)
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 2
        .node_with_inputs(GraphOp::Lut(LutOp::Abs), &[0]) // 3
        .node_with_inputs(GraphOp::Output, &[1]) // 4
        .node_with_inputs(GraphOp::Output, &[2]) // 5
        .node_with_inputs(GraphOp::Output, &[3]) // 6
        .output("relu", 4)
        .output("sigmoid", 5)
        .output("abs", 6)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    let test_input: Vec<u8> = (0..=255).step_by(32).collect();
    let mut inputs = GraphInputs::new();
    inputs.set(0, test_input.clone());

    let result = execute_bytes(&archive, &inputs).unwrap();

    println!("  {:>6} {:>8} {:>8} {:>8}", "x", "relu", "sigmoid", "abs");
    let relu_out = result.by_name("relu").unwrap();
    let sigmoid_out = result.by_name("sigmoid").unwrap();
    let abs_out = result.by_name("abs").unwrap();
    for (i, &x) in test_input.iter().enumerate() {
        println!(
            "  {:>6} {:>8} {:>8} {:>8}",
            x, relu_out[i], sigmoid_out[i], abs_out[i]
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Demo 4: Full pipeline — build → fuse → serialize → load → execute
// ---------------------------------------------------------------------------

fn demo_full_pipeline() {
    println!("--- Demo 4: Full Pipeline (build → fuse → serialize → load → execute) ---");

    // Build graph: x → sin → cos → output
    // Fusion should collapse sin→cos into a single FusedView.
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_with_inputs(GraphOp::Lut(LutOp::Sin), &[0]) // 1
        .node_with_inputs(GraphOp::Lut(LutOp::Cos), &[1]) // 2
        .node_with_inputs(GraphOp::Output, &[2]) // 3
        .output("y", 3)
        .build();

    println!("  Nodes before fusion: {}", g.node_count());

    // Fuse
    let stats = fusion::fuse(&mut g).unwrap();
    println!(
        "  Fusion: {} views fused, {} constants folded, {} CSE eliminated",
        stats.views_fused, stats.constants_folded, stats.cse_eliminated
    );
    println!("  Nodes after fusion:  {}", g.node_count());

    // Serialize to .holo bytes
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    println!("  Archive size: {} bytes", archive.len());

    // Load and execute
    let test_data: Vec<u8> = (0..=255).collect();
    let mut inputs = GraphInputs::new();
    inputs.set(0, test_data.clone());

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    // Verify: LUT result should match direct composition
    let sin_view = ElementWiseView::from_table(*LutOp::Sin.table());
    let cos_view = ElementWiseView::from_table(*LutOp::Cos.table());
    let expected_view = sin_view.then(&cos_view);

    let mut mismatches = 0;
    for b in 0u8..=255 {
        if output[b as usize] != expected_view.apply(b) {
            mismatches += 1;
        }
    }
    println!(
        "  Execution matches direct composition: {}/256",
        256 - mismatches
    );

    // Error analysis vs f64 reference
    println!("\n  Error analysis (angle encoding → cos(sin(x)) vs f64):");
    let angle = AngleEncoding;
    let signed = SignedEncoding;
    let mut max_error = 0.0_f64;
    let mut sum_error = 0.0_f64;
    let n = 256;
    for b in 0u8..=255 {
        let x = angle.lift(b);
        let f64_result = x.sin().cos();
        let lut_result = signed.lift(output[b as usize]);
        let error = (lut_result - f64_result).abs();
        if error > max_error {
            max_error = error;
        }
        sum_error += error;
    }
    println!("  Max error:  {max_error:.6}");
    println!("  Mean error: {:.6}", sum_error / n as f64);

    // Show a few sample values
    println!(
        "\n  {:>6} {:>10} {:>10} {:>10} {:>10}",
        "byte", "angle", "f64", "lut", "error"
    );
    for &b in &[0u8, 32, 64, 96, 128, 160, 192, 224] {
        let x = angle.lift(b);
        let f64_val = x.sin().cos();
        let lut_val = signed.lift(output[b as usize]);
        let err = (lut_val - f64_val).abs();
        println!(
            "  {:>6} {:>10.4} {:>10.6} {:>10.6} {:>10.6}",
            b, x, f64_val, lut_val, err
        );
    }

    // Also demonstrate add (binary op)
    println!("\n  Binary op demo: add(x, x) mod 256");
    let mut g2 = GraphBuilder::new()
        .input("a")
        .input("b")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_from_graph_input(GraphOp::Input, 1) // 1
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[0, 1]) // 2
        .node_with_inputs(GraphOp::Output, &[2]) // 3
        .output("sum", 3)
        .build();
    let _ = fusion::fuse(&mut g2).unwrap();
    let archive2 = HoloWriter::new().set_graph(&g2).build().unwrap();

    let mut inputs2 = GraphInputs::new();
    inputs2.set(0, vec![10, 100, 200, 250]);
    inputs2.set(1, vec![5, 50, 100, 200]);
    let result2 = execute_bytes(&archive2, &inputs2).unwrap();
    let sum = result2.by_name("sum").unwrap();

    println!("  {:>6} {:>6} {:>6} {:>10}", "a", "b", "sum", "expected");
    let a_vals = [10u8, 100, 200, 250];
    let b_vals = [5u8, 50, 100, 200];
    for i in 0..4 {
        let expected = a_vals[i].wrapping_add(b_vals[i]);
        println!(
            "  {:>6} {:>6} {:>6} {:>10}",
            a_vals[i], b_vals[i], sum[i], expected
        );
    }
    println!();
}
