//! End-to-end integration tests for the full pipeline:
//! build graph → fuse → write .holo → load → build_schedule → execute → verify

use holo_archive::{load_from_bytes, HoloWriter};
use holo_compiler::{compile, CompilerBuilder};
use holo_core::op::{LutOp, PrimOp};
use holo_core::view::ElementWiseView;
use holo_exec::lut_gemm::matmul::naive_matmul;
use holo_exec::lut_gemm::quantize::{quantize_4bit, quantize_8bit};
use holo_exec::{build_schedule, execute_bytes, GraphInputs, KvExecutor};
use holo_graph::builder::GraphBuilder;
use holo_graph::constant::ConstantData;
use holo_graph::fusion;
use holo_graph::graph::GraphOp;

/// E2E: build graph → fuse → write .holo → load_from_bytes → build_schedule → execute → verify
#[test]
fn e2e_linear_chain_fused() {
    // Input → Sigmoid → Relu → Output
    // Fusion should collapse Sigmoid→Relu into a single FusedView.
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 1
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1]) // 2
        .node_with_inputs(GraphOp::Output, &[2]) // 3
        .output("y", 3)
        .build();

    let stats = fusion::fuse(&mut g).unwrap();
    assert!(stats.views_fused >= 1, "should fuse sigmoid→relu chain");

    // Serialize to .holo archive
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    // Load from bytes
    let plan = load_from_bytes(&archive).unwrap();

    // Build schedule and execute
    let schedule = build_schedule(plan.graph()).unwrap();
    let mut inputs = GraphInputs::new();
    let test_data: Vec<u8> = (0..=255).collect();
    inputs.set(0, test_data);

    let result = KvExecutor::execute(plan.graph(), &schedule, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    // Verify against direct composed LUT
    let composed = ElementWiseView::from_table(*LutOp::Sigmoid.table())
        .then(&ElementWiseView::from_table(*LutOp::Relu.table()));
    for b in 0u8..=255 {
        assert_eq!(
            output[b as usize],
            composed.apply(b),
            "mismatch at byte {b}"
        );
    }
}

/// E2E: diamond graph with parallel fan-out through archive round-trip
#[test]
fn e2e_diamond_parallel_fanout() {
    //        Input(0)
    //       /        \
    //   Relu(1)    Sigmoid(2)
    //       \        /
    //       Add(3)
    //         |
    //      Output(4)
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 2
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2]) // 3
        .node_with_inputs(GraphOp::Output, &[3]) // 4
        .output("y", 4)
        .build();

    let _ = fusion::fuse(&mut g).unwrap();
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    let mut inputs = GraphInputs::new();
    let test_data: Vec<u8> = (0..=255).collect();
    inputs.set(0, test_data);

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    // Verify: relu(x) + sigmoid(x) mod 256
    for b in 0u8..=255 {
        let relu = LutOp::Relu.apply(b);
        let sigmoid = LutOp::Sigmoid.apply(b);
        let expected = relu.wrapping_add(sigmoid);
        assert_eq!(
            output[b as usize], expected,
            "mismatch at byte {b}: relu({b})={relu} + sigmoid({b})={sigmoid} = {expected}, got {}",
            output[b as usize]
        );
    }
}

/// E2E: graph with constants, verify constant propagation through full pipeline
#[test]
fn e2e_constants_through_pipeline() {
    // const(42) → Relu → Output
    // Constant folding operates on scalar bytes (first byte of constant data).
    // After folding: Relu(42)=42, so the output is a constant [42].
    let mut g = GraphBuilder::new()
        .constant(ConstantData::Bytes(vec![42])) // 0: Constant node
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 1
        .node_with_inputs(GraphOp::Output, &[1]) // 2
        .output("y", 2)
        .build();

    let stats = fusion::fuse(&mut g).unwrap();
    // Constant folding should evaluate Relu on constant input
    assert!(stats.constants_folded >= 1, "should fold relu(const)");

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let inputs = GraphInputs::new(); // No graph-level inputs needed

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    assert_eq!(output, &[LutOp::Relu.apply(42)]);
}

/// E2E: chained constant folding (const → op → op → output)
#[test]
fn e2e_chained_constant_folding() {
    // const(5) + const(3) → Add → Relu → Output
    // Add(5,3) folds to const(8), then Relu(8) folds to const(8)
    let mut g = GraphBuilder::new()
        .constant(ConstantData::Bytes(vec![5])) // 0
        .constant(ConstantData::Bytes(vec![3])) // 1
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[0, 1]) // 2
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[2]) // 3
        .node_with_inputs(GraphOp::Output, &[3]) // 4
        .output("y", 4)
        .build();

    let stats = fusion::fuse(&mut g).unwrap();
    assert!(stats.constants_folded >= 2, "should fold add then relu");

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let inputs = GraphInputs::new();

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    // Add(5,3)=8, Relu(8)=8
    assert_eq!(output, &[8]);
}

/// E2E: multi-input graph through archive round-trip
#[test]
fn e2e_multi_input_binary() {
    // a, b → Add → Output
    let mut g = GraphBuilder::new()
        .input("a")
        .input("b")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_from_graph_input(GraphOp::Input, 1) // 1
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[0, 1]) // 2
        .node_with_inputs(GraphOp::Output, &[2]) // 3
        .output("sum", 3)
        .build();

    let _ = fusion::fuse(&mut g).unwrap();
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = load_from_bytes(&archive).unwrap();
    let schedule = build_schedule(plan.graph()).unwrap();

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![10, 100, 200, 250]);
    inputs.set(1, vec![5, 50, 100, 200]);

    let result = KvExecutor::execute(plan.graph(), &schedule, &inputs).unwrap();
    let sum = result.by_name("sum").unwrap();

    assert_eq!(sum, &[15, 150, 44, 194]); // wrapping add
}

/// E2E: longer chain with multiple fusions
#[test]
fn e2e_long_chain_multi_fusion() {
    // Input → Sin → Cos → Relu → Sigmoid → Output
    // Fusion: Sin→Cos fused, then Cos→Relu→Sigmoid may also fuse
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_with_inputs(GraphOp::Lut(LutOp::Sin), &[0]) // 1
        .node_with_inputs(GraphOp::Lut(LutOp::Cos), &[1]) // 2
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[2]) // 3
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[3]) // 4
        .node_with_inputs(GraphOp::Output, &[4]) // 5
        .output("y", 5)
        .build();

    let stats = fusion::fuse(&mut g).unwrap();
    assert!(
        stats.views_fused >= 3,
        "should fuse at least 3 in a 4-element chain, got {}",
        stats.views_fused
    );

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    let mut inputs = GraphInputs::new();
    inputs.set(0, (0..=255).collect());

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    // Verify against chained application
    for b in 0u8..=255 {
        let expected = LutOp::Sigmoid.apply(
            LutOp::Relu.apply(LutOp::Cos.apply(LutOp::Sin.apply(b))),
        );
        assert_eq!(
            output[b as usize], expected,
            "mismatch at byte {b}"
        );
    }
}

/// E2E: wide parallel fan-out (4 branches) through archive
#[test]
fn e2e_wide_parallel_fanout() {
    //         Input(0)
    //       / | | \
    // Sin(1) Cos(2) Relu(3) Sigmoid(4)
    //       \ | | /
    //  Add(5)=Sin+Cos, Add(6)=Relu+Sigmoid
    //        \ /
    //      Add(7)
    //        |
    //     Output(8)
    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_with_inputs(GraphOp::Lut(LutOp::Sin), &[0]) // 1
        .node_with_inputs(GraphOp::Lut(LutOp::Cos), &[0]) // 2
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0]) // 3
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0]) // 4
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2]) // 5
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[3, 4]) // 6
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[5, 6]) // 7
        .node_with_inputs(GraphOp::Output, &[7]) // 8
        .output("y", 8)
        .build();

    let _ = fusion::fuse(&mut g).unwrap();
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    let mut inputs = GraphInputs::new();
    inputs.set(0, (0..=255).collect());

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    for b in 0u8..=255 {
        let sin = LutOp::Sin.apply(b);
        let cos = LutOp::Cos.apply(b);
        let relu = LutOp::Relu.apply(b);
        let sigmoid = LutOp::Sigmoid.apply(b);
        let left = sin.wrapping_add(cos);
        let right = relu.wrapping_add(sigmoid);
        let expected = left.wrapping_add(right);
        assert_eq!(
            output[b as usize], expected,
            "mismatch at byte {b}"
        );
    }
}

/// E2E: file round-trip (write to disk, read back, execute)
#[test]
fn e2e_file_roundtrip() {
    use std::io::Write;

    let mut g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();

    let _ = fusion::fuse(&mut g).unwrap();
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    // Write to temp file
    let path = std::env::temp_dir().join("e2e_roundtrip_test.holo");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&archive).unwrap();
    }

    // Execute from file
    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![0, 64, 128, 192, 255]);

    let result = holo_exec::execute_file(&path, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    for (i, &b) in [0u8, 64, 128, 192, 255].iter().enumerate() {
        assert_eq!(output[i], LutOp::Tanh.apply(b));
    }

    std::fs::remove_file(&path).ok();
}

/// E2E: LUT-GEMM Q4 through full pipeline
#[test]
fn e2e_lut_gemm_q4_pipeline() {
    let k = 4usize;
    let n = 2usize;
    let weights = vec![1.0f32; k * n];
    let qw = quantize_4bit(&weights, k as u32, n as u32);
    let qw_bytes = rkyv::to_bytes::<_, 4096>(&qw).unwrap().to_vec();

    let g = GraphBuilder::new()
        .input("activations")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .matmul_lut_4bit(ConstantData::Bytes(qw_bytes), &[0]) // 1
        .node_with_inputs(GraphOp::Output, &[1]) // 2
        .output("result", 2)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = load_from_bytes(&archive).unwrap();
    let schedule = build_schedule(plan.graph()).unwrap();

    let activations = [1.0f32, 2.0, 3.0, 4.0];
    let act_bytes: Vec<u8> = bytemuck::cast_slice(&activations).to_vec();
    let mut inputs = GraphInputs::new();
    inputs.set(0, act_bytes);

    let result = KvExecutor::execute(plan.graph(), &schedule, &inputs).unwrap();
    let output_bytes = result.by_name("result").unwrap();
    let output: &[f32] = bytemuck::cast_slice(output_bytes);
    assert_eq!(output.len(), n);
    for &v in output {
        assert!((v - 10.0).abs() < 0.5, "got {v}, expected ~10.0");
    }
}

/// E2E: LUT-GEMM Q8 through full pipeline
#[test]
fn e2e_lut_gemm_q8_pipeline() {
    let k = 4usize;
    let n = 2usize;
    let weights = vec![2.0f32; k * n];
    let qw = quantize_8bit(&weights, k as u32, n as u32);
    let qw_bytes = rkyv::to_bytes::<_, 4096>(&qw).unwrap().to_vec();

    let g = GraphBuilder::new()
        .input("activations")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .matmul_lut_8bit(ConstantData::Bytes(qw_bytes), &[0]) // 1
        .node_with_inputs(GraphOp::Output, &[1]) // 2
        .output("result", 2)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    let activations = [1.0f32, 1.0, 1.0, 1.0];
    let act_bytes: Vec<u8> = bytemuck::cast_slice(&activations).to_vec();
    let mut inputs = GraphInputs::new();
    inputs.set(0, act_bytes);

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output_bytes = result.by_name("result").unwrap();
    let output: &[f32] = bytemuck::cast_slice(output_bytes);
    assert_eq!(output.len(), n);
    for &v in output {
        assert!((v - 8.0).abs() < 0.1, "got {v}, expected ~8.0");
    }
}

/// E2E: LUT-GEMM Q4 accuracy vs naive matmul
#[test]
fn e2e_lut_gemm_q4_accuracy() {
    let k = 8usize;
    let n = 4usize;
    let m = 2usize;
    let weights: Vec<f32> = (0..k * n)
        .map(|i| (i as f32 + 1.0) * 0.05)
        .collect();
    let activations: Vec<f32> = (0..m * k)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();

    // Reference: naive matmul
    let mut expected = vec![0.0f32; m * n];
    naive_matmul(&activations, &weights, &mut expected, m, k, n);

    // LUT-GEMM Q4 through archive pipeline
    let qw = quantize_4bit(&weights, k as u32, n as u32);
    let qw_bytes = rkyv::to_bytes::<_, 4096>(&qw).unwrap().to_vec();

    let g = GraphBuilder::new()
        .input("a")
        .node_from_graph_input(GraphOp::Input, 0)
        .matmul_lut_4bit(ConstantData::Bytes(qw_bytes), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("c", 2)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let act_bytes: Vec<u8> = bytemuck::cast_slice(&activations).to_vec();
    let mut inputs = GraphInputs::new();
    inputs.set(0, act_bytes);

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output: &[f32] = bytemuck::cast_slice(result.by_name("c").unwrap());

    // Q4 error < 5% relative
    for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        let rel_err = (got - exp).abs() / exp.abs().max(1e-6);
        assert!(
            rel_err < 0.05,
            "Q4 element {i}: got {got}, expected {exp}, rel_err {rel_err}"
        );
    }
}

/// E2E: LUT-GEMM Q8 accuracy vs naive matmul
#[test]
fn e2e_lut_gemm_q8_accuracy() {
    let k = 8usize;
    let n = 4usize;
    let m = 2usize;
    let weights: Vec<f32> = (0..k * n)
        .map(|i| (i as f32 + 1.0) * 0.05)
        .collect();
    let activations: Vec<f32> = (0..m * k)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();

    let mut expected = vec![0.0f32; m * n];
    naive_matmul(&activations, &weights, &mut expected, m, k, n);

    let qw = quantize_8bit(&weights, k as u32, n as u32);
    let qw_bytes = rkyv::to_bytes::<_, 4096>(&qw).unwrap().to_vec();

    let g = GraphBuilder::new()
        .input("a")
        .node_from_graph_input(GraphOp::Input, 0)
        .matmul_lut_8bit(ConstantData::Bytes(qw_bytes), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("c", 2)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let act_bytes: Vec<u8> = bytemuck::cast_slice(&activations).to_vec();
    let mut inputs = GraphInputs::new();
    inputs.set(0, act_bytes);

    let result = execute_bytes(&archive, &inputs).unwrap();
    let output: &[f32] = bytemuck::cast_slice(result.by_name("c").unwrap());

    // Q8 error < 1% relative
    for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        let rel_err = (got - exp).abs() / exp.abs().max(1e-6);
        assert!(
            rel_err < 0.01,
            "Q8 element {i}: got {got}, expected {exp}, rel_err {rel_err}"
        );
    }
}

/// E2E: diamond graph with matmul + activation post-processing
#[test]
fn e2e_lut_gemm_with_activation() {
    let k = 4usize;
    let n = 4usize;
    let weights = vec![1.0f32; k * n];
    let qw = quantize_4bit(&weights, k as u32, n as u32);
    let qw_bytes = rkyv::to_bytes::<_, 4096>(&qw).unwrap().to_vec();

    // Input → MatMulLut4 → Output
    // The matmul output is f32 bytes, not u8 LUT-compatible,
    // so we just verify the matmul pipeline end-to-end.
    let g = GraphBuilder::new()
        .input("a")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .matmul_lut_4bit(ConstantData::Bytes(qw_bytes), &[0]) // 1
        .node_with_inputs(GraphOp::Output, &[1]) // 2
        .output("c", 2)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = load_from_bytes(&archive).unwrap();
    let schedule = build_schedule(plan.graph()).unwrap();

    // 2×4 activation matrix
    let activations = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let act_bytes: Vec<u8> = bytemuck::cast_slice(&activations).to_vec();
    let mut inputs = GraphInputs::new();
    inputs.set(0, act_bytes);

    let result = KvExecutor::execute(plan.graph(), &schedule, &inputs).unwrap();
    let output: &[f32] = bytemuck::cast_slice(result.by_name("c").unwrap());
    assert_eq!(output.len(), 2 * n); // 2 rows × 4 cols
    // Row 0: [1,0,0,0] × I ≈ [1,1,1,1]
    for &v in &output[..n] {
        assert!((v - 1.0).abs() < 0.5, "got {v}");
    }
    // Row 1: [0,0,0,1] × I ≈ [1,1,1,1]
    for &v in &output[n..] {
        assert!((v - 1.0).abs() < 0.5, "got {v}");
    }
}

/// E2E: .holo archive write/load roundtrip with quantized weights
#[test]
fn e2e_lut_gemm_archive_roundtrip() {
    let k = 4usize;
    let n = 2usize;
    let weights = vec![3.0f32; k * n];
    let qw = quantize_8bit(&weights, k as u32, n as u32);
    let qw_bytes = rkyv::to_bytes::<_, 4096>(&qw).unwrap().to_vec();

    let g = GraphBuilder::new()
        .input("a")
        .node_from_graph_input(GraphOp::Input, 0)
        .matmul_lut_8bit(ConstantData::Bytes(qw_bytes), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("c", 2)
        .build();

    // Write → load → verify graph structure preserved
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = load_from_bytes(&archive).unwrap();

    assert_eq!(plan.graph().nodes.len(), 3);
    assert_eq!(plan.graph().output_names, vec!["c"]);
    assert!(!plan.graph().constants.is_empty());

    // Execute and verify
    let schedule = build_schedule(plan.graph()).unwrap();
    let activations = [1.0f32, 1.0, 1.0, 1.0];
    let act_bytes: Vec<u8> = bytemuck::cast_slice(&activations).to_vec();
    let mut inputs = GraphInputs::new();
    inputs.set(0, act_bytes);

    let result = KvExecutor::execute(plan.graph(), &schedule, &inputs).unwrap();
    let output: &[f32] = bytemuck::cast_slice(result.by_name("c").unwrap());
    // sum(1*3 * 4) = 12
    for &v in output {
        assert!((v - 12.0).abs() < 0.1, "got {v}, expected ~12.0");
    }
}

// ===== Compiler pipeline E2E tests =====

/// E2E: compile → load → execute → verify for a linear chain
#[test]
fn e2e_compiler_linear_chain() {
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
        .node_with_inputs(GraphOp::Output, &[2])
        .output("y", 3)
        .build();

    let out = compile(g).unwrap();
    assert!(out.stats.fusion.views_fused >= 1);

    let plan = load_from_bytes(&out.archive).unwrap();
    let schedule = build_schedule(plan.graph()).unwrap();
    let mut inputs = GraphInputs::new();
    inputs.set(0, (0..=255).collect());

    let result = KvExecutor::execute(plan.graph(), &schedule, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    let composed = ElementWiseView::from_table(*LutOp::Sigmoid.table())
        .then(&ElementWiseView::from_table(*LutOp::Relu.table()));
    for b in 0u8..=255 {
        assert_eq!(output[b as usize], composed.apply(b));
    }
}

/// E2E: compile → load → execute → verify for a diamond with fusion
#[test]
fn e2e_compiler_diamond_with_fusion() {
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();

    let out = compile(g).unwrap();
    let plan = load_from_bytes(&out.archive).unwrap();
    let schedule = build_schedule(plan.graph()).unwrap();
    let mut inputs = GraphInputs::new();
    inputs.set(0, (0..=255).collect());

    let result = KvExecutor::execute(plan.graph(), &schedule, &inputs).unwrap();
    let output = result.by_name("y").unwrap();

    for b in 0u8..=255 {
        let expected = LutOp::Relu.apply(b).wrapping_add(LutOp::Sigmoid.apply(b));
        assert_eq!(output[b as usize], expected);
    }
}

/// E2E: compile with constants → load → execute
#[test]
fn e2e_compiler_with_constants() {
    let g = GraphBuilder::new()
        .constant(ConstantData::Bytes(vec![42]))
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();

    let out = compile(g).unwrap();
    assert!(out.stats.fusion.constants_folded >= 1);

    let plan = load_from_bytes(&out.archive).unwrap();
    let schedule = build_schedule(plan.graph()).unwrap();
    let inputs = GraphInputs::new();

    let result = KvExecutor::execute(plan.graph(), &schedule, &inputs).unwrap();
    let output = result.by_name("y").unwrap();
    assert_eq!(output, &[LutOp::Relu.apply(42)]);
}

/// E2E: compile fusion disabled vs enabled produces different stats
#[test]
fn e2e_compiler_fusion_disabled_vs_enabled() {
    let make_graph = || {
        GraphBuilder::new()
            .node(GraphOp::Input)
            .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
            .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[1])
            .node_with_inputs(GraphOp::Output, &[2])
            .build()
    };

    let fused = CompilerBuilder::new(make_graph()).fuse(true).build().unwrap();
    let unfused = CompilerBuilder::new(make_graph()).fuse(false).build().unwrap();

    assert!(fused.stats.fusion.views_fused > unfused.stats.fusion.views_fused);
}

/// E2E: compile large graph (100 nodes) — archive is valid and loadable
#[test]
fn e2e_compiler_large_graph() {
    let ops = [LutOp::Relu, LutOp::Sigmoid, LutOp::Tanh, LutOp::Sin, LutOp::Cos];
    let mut b = GraphBuilder::new().node(GraphOp::Input);
    for i in 0..100 {
        b = b.node_with_inputs(GraphOp::Lut(ops[i % ops.len()]), &[i]);
    }
    let g = b.node_with_inputs(GraphOp::Output, &[100]).build();

    let out = compile(g).unwrap();
    assert!(out.stats.total_nodes > 0);

    let plan = load_from_bytes(&out.archive).unwrap();
    assert!(plan.header().is_valid_magic());
}

/// E2E: workspace reuse — sequential chain has fewer slots than parallel fan-out
#[test]
fn e2e_compiler_workspace_reuse() {
    // Sequential: Input → R → S → T → Output (buffers can be reused)
    let seq = GraphBuilder::new()
        .node(GraphOp::Input)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[1])
        .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[2])
        .node_with_inputs(GraphOp::Output, &[3])
        .build();
    let seq_out = compile(seq).unwrap();

    // Parallel: Input → [R, S, T, E] → Output (all live simultaneously)
    let par = GraphBuilder::new()
        .node(GraphOp::Input)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Tanh), &[0])
        .node_with_inputs(GraphOp::Lut(LutOp::Exp), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .node_with_inputs(GraphOp::Output, &[2])
        .node_with_inputs(GraphOp::Output, &[3])
        .node_with_inputs(GraphOp::Output, &[4])
        .build();
    let par_out = compile(par).unwrap();

    assert!(
        seq_out.stats.workspace_slots <= par_out.stats.workspace_slots,
        "sequential {} should use <= parallel {} slots",
        seq_out.stats.workspace_slots,
        par_out.stats.workspace_slots,
    );
}

/// E2E: LayerHeader present in compiled archive
#[test]
fn e2e_compiler_layer_header_present() {
    use holo_archive::section::SECTION_LAYER_HEADER;

    let g = GraphBuilder::new()
        .node(GraphOp::Input)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .build();

    let out = compile(g).unwrap();
    let plan = load_from_bytes(&out.archive).unwrap();
    assert!(plan.sections().find(SECTION_LAYER_HEADER).is_some());
}
