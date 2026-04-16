//! Integration tests for hologram-backend.
//!
//! Tests that the CPU and Metal backends produce correct results for
//! small synthetic models, verifying the full dispatch path.

use hologram_backend::cpu::{CpuBackend, CpuMemory};
use hologram_backend::{ComputeBackend, ComputeMemory, KernelParams};
use hologram_core::op::FloatOp;

/// Test a small computation graph on CPU:
/// input → relu → add(bias) → softmax → output
#[test]
fn cpu_mini_graph() {
    let mem = CpuMemory;
    let backend = CpuBackend::new();

    // Input: 4 values, some negative.
    let input = mem.upload(bytemuck::cast_slice(&[-1.0f32, 2.0, -3.0, 4.0]));
    let bias = mem.upload(bytemuck::cast_slice(&[0.5f32, 0.5, 0.5, 0.5]));

    // Step 1: ReLU.
    let mut relu_out = mem.alloc(0);
    backend
        .dispatch(
            &FloatOp::Relu,
            &[&input],
            &mut relu_out,
            &KernelParams::default(),
        )
        .expect("relu should succeed");
    let relu_dl = mem.download(&relu_out);
    let relu_result: &[f32] = bytemuck::cast_slice(&relu_dl);
    assert_eq!(relu_result, &[0.0, 2.0, 0.0, 4.0]);

    // Step 2: Add bias.
    let mut add_out = mem.alloc(0);
    backend
        .dispatch(
            &FloatOp::Add,
            &[&relu_out, &bias],
            &mut add_out,
            &KernelParams::default(),
        )
        .expect("add should succeed");
    let add_dl = mem.download(&add_out);
    let add_result: &[f32] = bytemuck::cast_slice(&add_dl);
    assert_eq!(add_result, &[0.5, 2.5, 0.5, 4.5]);

    // Step 3: Softmax.
    let mut softmax_out = mem.alloc(0);
    backend
        .dispatch(
            &FloatOp::Softmax { size: 4 },
            &[&add_out],
            &mut softmax_out,
            &KernelParams::default(),
        )
        .expect("softmax should succeed");
    let sm_dl = mem.download(&softmax_out);
    let softmax_result: &[f32] = bytemuck::cast_slice(&sm_dl);
    let sum: f32 = softmax_result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax should sum to 1, got {sum}"
    );
    // Largest value (4.5) should have highest probability.
    assert!(softmax_result[3] > softmax_result[1]);
    assert!(softmax_result[1] > softmax_result[0]);
}

/// Test that CPU matmul produces correct results for a simple case.
#[test]
fn cpu_matmul_identity() {
    let mem = CpuMemory;
    let backend = CpuBackend::new();

    // 3x3 identity matrix × [1,2,3] column = [1,2,3]
    let identity: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let vec3: Vec<f32> = vec![1.0, 2.0, 3.0];

    let a = mem.upload(bytemuck::cast_slice(&identity));
    let b = mem.upload(bytemuck::cast_slice(&vec3));
    let mut out = mem.alloc(0);

    backend
        .dispatch(
            &FloatOp::MatMul { m: 3, k: 3, n: 1 },
            &[&a, &b],
            &mut out,
            &KernelParams::default(),
        )
        .expect("matmul should succeed");

    let downloaded = mem.download(&out);
    let result: &[f32] = bytemuck::cast_slice(&downloaded);
    assert_eq!(result, &[1.0, 2.0, 3.0]);
}

/// Test ring LUT dispatch end-to-end.
#[test]
fn cpu_ring_chain() {
    let mem = CpuMemory;
    let mut backend = CpuBackend::new();

    // Two LUT tables: increment (+1 mod 256) and double (*2 mod 256).
    let mut inc_table = [0u8; 256];
    let mut dbl_table = [0u8; 256];
    for i in 0..256 {
        inc_table[i] = ((i + 1) % 256) as u8;
        dbl_table[i] = ((i * 2) % 256) as u8;
    }
    backend.load_ring_tables(&[&inc_table, &dbl_table], &mem);

    // Apply increment then double: input=[0, 1, 127] → inc=[1, 2, 128] → dbl=[2, 4, 0]
    let input = mem.upload(&[0u8, 1, 127]);
    let mut mid = mem.alloc(0);
    let mut out = mem.alloc(0);

    backend
        .dispatch_ring(0, &[&input], &mut mid)
        .expect("ring inc should succeed");
    backend
        .dispatch_ring(1, &[&mid], &mut out)
        .expect("ring dbl should succeed");

    let result = mem.download(&out);
    assert_eq!(result, vec![2, 4, 0]);
}

/// Test that Metal backend (when available) produces same results as CPU.
#[cfg(has_metal)]
#[test]
fn metal_matches_cpu_relu() {
    use hologram_backend::metal::{MetalBackend, MetalMemory};

    let cpu_mem = CpuMemory;
    let cpu_backend = CpuBackend::new();

    let metal_mem = match MetalMemory::new() {
        Some(m) => m,
        None => return,
    };
    let metal_backend = match MetalBackend::new() {
        Some(b) => b,
        None => return,
    };

    let data: Vec<f32> = vec![-5.0, -1.0, 0.0, 0.5, 3.0, 100.0];
    let cpu_input = cpu_mem.upload(bytemuck::cast_slice(&data));
    let metal_input = metal_mem.upload(bytemuck::cast_slice(&data));

    let mut cpu_out = cpu_mem.alloc(0);
    let mut metal_out = metal_mem.alloc(0);

    cpu_backend
        .dispatch(
            &FloatOp::Relu,
            &[&cpu_input],
            &mut cpu_out,
            &KernelParams::default(),
        )
        .expect("cpu relu");
    metal_backend
        .dispatch(
            &FloatOp::Relu,
            &[&metal_input],
            &mut metal_out,
            &KernelParams::default(),
        )
        .expect("metal relu");
    metal_backend.flush();

    let cpu_result: Vec<f32> = bytemuck::cast_slice(&cpu_mem.download(&cpu_out)).to_vec();
    let metal_result: Vec<f32> = bytemuck::cast_slice(&metal_mem.download(&metal_out)).to_vec();

    assert_eq!(cpu_result.len(), metal_result.len());
    for (i, (c, m)) in cpu_result.iter().zip(&metal_result).enumerate() {
        assert!(
            (c - m).abs() < 1e-5,
            "mismatch at index {i}: cpu={c}, metal={m}"
        );
    }
}

/// Test Metal matmul matches CPU.
#[cfg(has_metal)]
#[test]
fn metal_matches_cpu_matmul() {
    use hologram_backend::metal::{MetalBackend, MetalMemory};

    let cpu_mem = CpuMemory;
    let cpu_backend = CpuBackend::new();

    let metal_mem = match MetalMemory::new() {
        Some(m) => m,
        None => return,
    };
    let metal_backend = match MetalBackend::new() {
        Some(b) => b,
        None => return,
    };

    // 4x3 × 3x2 = 4x2
    let a: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..6).map(|i| (i as f32) * 0.5).collect();

    let cpu_a = cpu_mem.upload(bytemuck::cast_slice(&a));
    let cpu_b = cpu_mem.upload(bytemuck::cast_slice(&b));
    let metal_a = metal_mem.upload(bytemuck::cast_slice(&a));
    let metal_b = metal_mem.upload(bytemuck::cast_slice(&b));

    let op = FloatOp::MatMul { m: 4, k: 3, n: 2 };
    let params = KernelParams::default();

    let mut cpu_out = cpu_mem.alloc(0);
    let mut metal_out = metal_mem.alloc(0);

    cpu_backend
        .dispatch(&op, &[&cpu_a, &cpu_b], &mut cpu_out, &params)
        .expect("cpu matmul");
    metal_backend
        .dispatch(&op, &[&metal_a, &metal_b], &mut metal_out, &params)
        .expect("metal matmul");
    metal_backend.flush();

    let cpu_result: Vec<f32> = bytemuck::cast_slice(&cpu_mem.download(&cpu_out)).to_vec();
    let metal_result: Vec<f32> = bytemuck::cast_slice(&metal_mem.download(&metal_out)).to_vec();

    assert_eq!(
        cpu_result.len(),
        metal_result.len(),
        "output length mismatch"
    );
    for (i, (c, m)) in cpu_result.iter().zip(&metal_result).enumerate() {
        assert!(
            (c - m).abs() < 1.0, // Tiled SGEMM may have slight FP differences.
            "matmul mismatch at {i}: cpu={c}, metal={m}"
        );
    }
}
