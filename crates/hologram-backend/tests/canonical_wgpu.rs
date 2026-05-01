//! Conformance tests for the canonical WebGPU backend.
//!
//! These tests are `#[ignore]`d by default — they require a working
//! Vulkan/Metal/DX12 driver on the host. Run explicitly with:
//!
//! ```bash
//! cargo test -p hologram-backend --features webgpu --test canonical_wgpu -- --ignored
//! ```

#![cfg(feature = "webgpu")]

use hologram_backend::canonical::WgpuBackend;
use hologram_transform::{
    check_forward, check_forward_then_backward, AddCall, AddGradCall, AddRmsNormCall,
    AddRmsNormGradCall, AddressTable, BinaryCall, CompiledPlan, ConcatCall, Conv2dCall,
    ConvTransposeCall, CpuBackend, GlobalAvgPoolCall, GroupNormCall, InstanceNormGradCall,
    KernelCall, LayerNormGradCall, MatMulCall, MatMulGradACall, MatMulGradBCall, NormFullCall,
    NormScaleCall, Pool2dCall, Pool2dKind, ReduceCall, ReduceKind, ReshapeCall, RmsNormGradCall,
    SliceCall, SlotSpan, SoftmaxCall, SoftmaxGradCall, SoftmaxGradKind, SubGradCall, Tolerance,
    UnaryCall, UnaryKind, WorkspaceLayout,
};

const N: usize = 64;

fn empty_address_table() -> AddressTable {
    AddressTable {
        spans: Box::new([]),
        grads: Box::new([]),
    }
}

fn binary_plan(forward: KernelCall) -> CompiledPlan {
    CompiledPlan {
        forward: Box::new([forward]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 3 * N,
        },
    }
}

fn unary_plan(forward: KernelCall) -> CompiledPlan {
    CompiledPlan {
        forward: Box::new([forward]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 2 * N,
        },
    }
}

fn run_binary(plan: &CompiledPlan, gpu: &mut WgpuBackend, seed_b: impl Fn(usize) -> f32) {
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        plan,
        &mut cpu,
        gpu,
        |buf| {
            let a: Vec<f32> = (0..N).map(|i| 0.1 * i as f32 - 1.0).collect();
            let b: Vec<f32> = (0..N).map(&seed_b).collect();
            buf.write_span(SlotSpan { offset: 0, len: N }, &a);
            buf.write_span(SlotSpan { offset: N, len: N }, &b);
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu binary diverged: {:?}", res);
}

fn run_unary(plan: &CompiledPlan, gpu: &mut WgpuBackend, seed: impl Fn(usize) -> f32) {
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        plan,
        &mut cpu,
        gpu,
        |buf| {
            let xs: Vec<f32> = (0..N).map(&seed).collect();
            buf.write_span(SlotSpan { offset: 0, len: N }, &xs);
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu unary diverged: {:?}", res);
}

/// ADR-051 step 3: every binary arm goes through the resident path
/// (no per-call upload/download). This test runs Add + the full binary
/// family via `dispatch_resident` against a `WgpuWorkspace` and
/// asserts the output matches the CPU reference exactly. If the
/// resident bind-group setup is broken, this test catches it before
/// the slow-path fallback hides the regression.
#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_binary_resident_path_matches_cpu_reference() {
    use hologram_transform::{BackendWorkspace, CanonicalBackend};

    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let mut cpu = CpuBackend::new();
    let bc = BinaryCall {
        a: SlotSpan { offset: 0, len: N },
        b: SlotSpan { offset: N, len: N },
        c: SlotSpan {
            offset: 2 * N,
            len: N,
        },
    };

    let cases: &[(&str, KernelCall, fn(usize) -> f32)] = &[
        (
            "Add",
            KernelCall::Add(AddCall {
                a: bc.a,
                b: bc.b,
                c: bc.c,
            }),
            |i| 0.07 * i as f32 + 0.3,
        ),
        ("Sub", KernelCall::Sub(bc), |i| 0.07 * i as f32 + 0.3),
        ("Mul", KernelCall::Mul(bc), |i| 0.07 * i as f32 + 0.3),
        ("Min", KernelCall::Min(bc), |i| 0.07 * i as f32 + 0.3),
        ("Max", KernelCall::Max(bc), |i| 0.07 * i as f32 + 0.3),
        ("Equal", KernelCall::Equal(bc), |i| 0.07 * i as f32 + 0.3),
        ("Less", KernelCall::Less(bc), |i| 0.07 * i as f32 + 0.3),
        ("Greater", KernelCall::Greater(bc), |i| {
            0.07 * i as f32 + 0.3
        }),
        ("And", KernelCall::And(bc), |i| 0.07 * i as f32 + 0.3),
        ("Or", KernelCall::Or(bc), |i| 0.07 * i as f32 + 0.3),
        ("Xor", KernelCall::Xor(bc), |i| 0.07 * i as f32 + 0.3),
        ("Div", KernelCall::Div(bc), |i| 0.07 * i as f32 + 1.0),
        ("Mod", KernelCall::Mod(bc), |i| 0.07 * i as f32 + 1.0),
        ("Pow", KernelCall::Pow(bc), |_| 2.0),
    ];

    for (name, call, seed_b) in cases {
        eprintln!("checking resident binary op: {name}");
        let a: Vec<f32> = (0..N).map(|i| 0.1 * i as f32 - 1.0).collect();
        let b: Vec<f32> = (0..N).map(seed_b).collect();

        // Resident path on GPU.
        let mut ws = gpu.alloc_workspace(3 * N).expect("alloc workspace");
        ws.write_span(SlotSpan { offset: 0, len: N }, &a)
            .expect("write a");
        ws.write_span(SlotSpan { offset: N, len: N }, &b)
            .expect("write b");
        gpu.dispatch_resident(&mut ws, call)
            .expect("resident dispatch");
        let gpu_c = ws
            .read_span(SlotSpan {
                offset: 2 * N,
                len: N,
            })
            .expect("read c");

        // CPU reference for the same call.
        let mut storage = vec![0.0_f32; 3 * N];
        storage[..N].copy_from_slice(&a);
        storage[N..2 * N].copy_from_slice(&b);
        cpu.dispatch(&mut storage, call).expect("cpu dispatch");
        let cpu_c = &storage[2 * N..3 * N];

        for (i, (&g, &c)) in gpu_c.iter().zip(cpu_c.iter()).enumerate() {
            // Bitwise-equal for finite-only ops; float tolerance for arithmetic.
            if !c.is_finite() && !g.is_finite() {
                continue;
            }
            assert!(
                (g - c).abs() <= 1e-5_f32.max(c.abs() * 1e-5),
                "{name} resident-vs-cpu diverged at index {i}: gpu={g} cpu={c}"
            );
        }
    }
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_add_matches_cpu_reference() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let plan = binary_plan(KernelCall::Add(AddCall {
        a: SlotSpan { offset: 0, len: N },
        b: SlotSpan { offset: N, len: N },
        c: SlotSpan {
            offset: 2 * N,
            len: N,
        },
    }));
    run_binary(&plan, &mut gpu, |i| -0.05 * i as f32 + 1.0);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_binary_family_matches_cpu_reference() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let bc = BinaryCall {
        a: SlotSpan { offset: 0, len: N },
        b: SlotSpan { offset: N, len: N },
        c: SlotSpan {
            offset: 2 * N,
            len: N,
        },
    };
    let cases: &[(&str, KernelCall)] = &[
        ("Sub", KernelCall::Sub(bc)),
        ("Mul", KernelCall::Mul(bc)),
        ("Min", KernelCall::Min(bc)),
        ("Max", KernelCall::Max(bc)),
        ("Equal", KernelCall::Equal(bc)),
        ("Less", KernelCall::Less(bc)),
        ("LessOrEqual", KernelCall::LessOrEqual(bc)),
        ("Greater", KernelCall::Greater(bc)),
        ("GreaterOrEqual", KernelCall::GreaterOrEqual(bc)),
        ("And", KernelCall::And(bc)),
        ("Or", KernelCall::Or(bc)),
        ("Xor", KernelCall::Xor(bc)),
    ];
    for (name, call) in cases {
        let plan = binary_plan(*call);
        eprintln!("checking binary op: {name}");
        run_binary(&plan, &mut gpu, |i| 0.07 * i as f32 + 0.3);
    }
    // Div / Mod: avoid division by zero in the seed.
    let plan = binary_plan(KernelCall::Div(bc));
    eprintln!("checking binary op: Div");
    run_binary(&plan, &mut gpu, |i| 0.07 * i as f32 + 1.0);
    let plan = binary_plan(KernelCall::Mod(bc));
    eprintln!("checking binary op: Mod");
    run_binary(&plan, &mut gpu, |i| 0.07 * i as f32 + 1.0);
    // Pow: positive base avoids NaN territory; integer-ish exponent.
    let plan = binary_plan(KernelCall::Pow(bc));
    eprintln!("checking binary op: Pow");
    run_binary(&plan, &mut gpu, |_| 2.0);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_reshape_copies_input_to_output() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let plan = unary_plan(KernelCall::Reshape(ReshapeCall {
        input: SlotSpan { offset: 0, len: N },
        output: SlotSpan { offset: N, len: N },
    }));
    run_unary(&plan, &mut gpu, |i| 0.1 * i as f32 - 1.0);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_unary_family_matches_cpu_reference() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let uc = UnaryCall {
        input: SlotSpan { offset: 0, len: N },
        output: SlotSpan { offset: N, len: N },
    };
    // Kinds whose canonical CPU reference is well-defined for the
    // seed below (no NaN/Inf collisions). Kinds requiring positive
    // input (Log, Sqrt) get a positive seed.
    type Seed = fn(usize) -> f32;
    let cases: &[(UnaryKind, Seed)] = &[
        (UnaryKind::Neg, neg_seed),
        (UnaryKind::Relu, neg_seed),
        (UnaryKind::Sigmoid, neg_seed),
        (UnaryKind::Tanh, neg_seed),
        (UnaryKind::Exp, neg_seed),
        (UnaryKind::Abs, neg_seed),
        (UnaryKind::Sin, neg_seed),
        (UnaryKind::Cos, neg_seed),
        (UnaryKind::Sign, neg_seed),
        (UnaryKind::Silu, neg_seed),
        (UnaryKind::Floor, neg_seed),
        (UnaryKind::Ceil, neg_seed),
        (UnaryKind::Round, neg_seed),
        (UnaryKind::Log, pos_seed),
        (UnaryKind::Sqrt, pos_seed),
        (UnaryKind::Reciprocal, pos_seed),
        (UnaryKind::Gelu, neg_seed),
        (UnaryKind::Erf, neg_seed),
        (UnaryKind::Not, ternary_seed),
        (UnaryKind::IsNaN, neg_seed),
    ];
    for (kind, seed) in cases {
        let plan = unary_plan(KernelCall::Unary(uc, *kind));
        eprintln!("checking unary kind: {kind:?}");
        run_unary(&plan, &mut gpu, *seed);
    }
}

fn neg_seed(i: usize) -> f32 {
    0.1 * i as f32 - 3.0
}

fn pos_seed(i: usize) -> f32 {
    0.1 * i as f32 + 0.5
}

/// Mix of zero and nonzero values — exercises `Not` truthiness.
fn ternary_seed(i: usize) -> f32 {
    match i % 3 {
        0 => 0.0,
        1 => 1.5,
        _ => -2.5,
    }
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_slice_last_axis_matches_cpu_reference() {
    // Input is 4 rows of length 8; slice [2..6) per row → 4×4 output.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let in_n = 32;
    let out_n = 16;
    let plan = CompiledPlan {
        forward: Box::new([KernelCall::Slice(SliceCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            output: SlotSpan {
                offset: in_n,
                len: out_n,
            },
            axis_size: 8,
            start: 2,
            end: 6,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: in_n + out_n,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let xs: Vec<f32> = (0..in_n).map(|i| 0.1 * i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &xs,
            );
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu Slice diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_concat_last_axis_matches_cpu_reference() {
    // 3 rows, A row=4, B row=2 → concat into 3×6 output.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let a_n = 12;
    let b_n = 6;
    let out_n = 18;
    let plan = CompiledPlan {
        forward: Box::new([KernelCall::Concat(ConcatCall {
            a: SlotSpan {
                offset: 0,
                len: a_n,
            },
            b: SlotSpan {
                offset: a_n,
                len: b_n,
            },
            output: SlotSpan {
                offset: a_n + b_n,
                len: out_n,
            },
            size_a: 4,
            size_b: 2,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: a_n + b_n + out_n,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let a_data: Vec<f32> = (0..a_n).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..b_n).map(|i| 100.0 + i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: a_n,
                },
                &a_data,
            );
            buf.write_span(
                SlotSpan {
                    offset: a_n,
                    len: b_n,
                },
                &b_data,
            );
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu Concat diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_add_grad_accumulates_into_both_grads() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::AddGrad(AddGradCall {
            dc: SlotSpan { offset: 0, len: N },
            da: SlotSpan { offset: N, len: N },
            db: SlotSpan {
                offset: 2 * N,
                len: N,
            },
        })]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 3 * N,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = hologram_transform::check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let dc_seed: Vec<f32> = (0..N).map(|i| 0.05 * i as f32 + 0.1).collect();
            let da_seed: Vec<f32> = (0..N).map(|i| -0.02 * i as f32).collect();
            let db_seed: Vec<f32> = (0..N).map(|i| 0.03 * i as f32 + 1.0).collect();
            buf.write_span(SlotSpan { offset: 0, len: N }, &dc_seed);
            buf.write_span(SlotSpan { offset: N, len: N }, &da_seed);
            buf.write_span(
                SlotSpan {
                    offset: 2 * N,
                    len: N,
                },
                &db_seed,
            );
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu AddGrad diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_mul_grad_via_cpu_fallback_matches_reference() {
    use hologram_transform::MulGradCall;
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    // Workspace: a, b, dc, da, db.
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::MulGrad(MulGradCall {
            a: SlotSpan { offset: 0, len: N },
            b: SlotSpan { offset: N, len: N },
            dc: SlotSpan {
                offset: 2 * N,
                len: N,
            },
            da: SlotSpan {
                offset: 3 * N,
                len: N,
            },
            db: SlotSpan {
                offset: 4 * N,
                len: N,
            },
        })]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 5 * N,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = hologram_transform::check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let a: Vec<f32> = (0..N).map(|i| 0.1 * i as f32).collect();
            let b: Vec<f32> = (0..N).map(|i| -0.2 * i as f32 + 0.3).collect();
            let dc: Vec<f32> = (0..N).map(|i| 0.5 - 0.01 * i as f32).collect();
            buf.write_span(SlotSpan { offset: 0, len: N }, &a);
            buf.write_span(SlotSpan { offset: N, len: N }, &b);
            buf.write_span(
                SlotSpan {
                    offset: 2 * N,
                    len: N,
                },
                &dc,
            );
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu MulGrad fallback diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_unary_grad_via_cpu_fallback_matches_reference() {
    // UnaryGrad with kind=Sigmoid: dA[i] += dC[i] * out[i] * (1 - out[i]).
    use hologram_transform::{UnaryGradCall, UnaryGradKind};
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    // Workspace: source(forward output), dc, da.
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::UnaryGrad(
            UnaryGradCall {
                source: SlotSpan { offset: 0, len: N },
                dc: SlotSpan { offset: N, len: N },
                da: SlotSpan {
                    offset: 2 * N,
                    len: N,
                },
            },
            UnaryGradKind::Sigmoid,
        )]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 3 * N,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = hologram_transform::check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            // Plausible sigmoid output ∈ (0, 1).
            let src: Vec<f32> = (0..N).map(|i| 0.4 + 0.005 * i as f32).collect();
            let dc: Vec<f32> = (0..N).map(|i| 0.5 - 0.01 * i as f32).collect();
            buf.write_span(SlotSpan { offset: 0, len: N }, &src);
            buf.write_span(SlotSpan { offset: N, len: N }, &dc);
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(
        res.is_match(),
        "wgpu UnaryGrad fallback diverged: {:?}",
        res
    );
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_neg_grad_subtracts_dc_from_da() {
    use hologram_transform::NegGradCall;
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::NegGrad(NegGradCall {
            dc: SlotSpan { offset: 0, len: N },
            da: SlotSpan { offset: N, len: N },
        })]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 2 * N,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = hologram_transform::check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let dc_seed: Vec<f32> = (0..N).map(|i| 0.05 * i as f32 + 0.1).collect();
            let da_seed: Vec<f32> = (0..N).map(|i| -0.02 * i as f32).collect();
            buf.write_span(SlotSpan { offset: 0, len: N }, &dc_seed);
            buf.write_span(SlotSpan { offset: N, len: N }, &da_seed);
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu NegGrad diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_sub_grad_subtracts_from_db_and_adds_to_da() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::SubGrad(SubGradCall {
            dc: SlotSpan { offset: 0, len: N },
            da: SlotSpan { offset: N, len: N },
            db: SlotSpan {
                offset: 2 * N,
                len: N,
            },
        })]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 3 * N,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = hologram_transform::check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let dc_seed: Vec<f32> = (0..N).map(|i| 0.05 * i as f32 + 0.1).collect();
            buf.write_span(SlotSpan { offset: 0, len: N }, &dc_seed);
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu SubGrad diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_reduce_family_matches_cpu_reference() {
    // 8 rows of size 8 → 8 outputs.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let in_n = 64;
    let out_n = 8;
    let row_size = 8;
    let kinds = [
        ReduceKind::Sum,
        ReduceKind::Mean,
        ReduceKind::Max,
        ReduceKind::Min,
        ReduceKind::Prod,
    ];
    for kind in kinds {
        let plan = CompiledPlan {
            forward: Box::new([KernelCall::Reduce(
                ReduceCall {
                    input: SlotSpan {
                        offset: 0,
                        len: in_n,
                    },
                    output: SlotSpan {
                        offset: in_n,
                        len: out_n,
                    },
                    size: row_size,
                },
                kind,
            )]),
            backward: Box::new([]),
            address_table: empty_address_table(),
            workspace: WorkspaceLayout {
                total_elements: in_n + out_n,
            },
        };
        let mut cpu = CpuBackend::new();
        let res = check_forward(
            &plan,
            &mut cpu,
            &mut gpu,
            |buf| {
                // Avoid extreme magnitudes / zeros for Prod (which would
                // produce 0 across whole rows): seed in [0.5, 1.5].
                let xs: Vec<f32> = (0..in_n).map(|i| 0.5 + (i as f32) * 0.012).collect();
                buf.write_span(
                    SlotSpan {
                        offset: 0,
                        len: in_n,
                    },
                    &xs,
                );
            },
            // `Prod` over 8 elements ≈ 0.5^8…1.5^8 — magnitudes up to
            // ~25, so absolute diffs accumulate. LOOSE keeps headroom
            // for the order-of-multiplication divergence; the test
            // mainly proves the dispatch path is wired correctly.
            Tolerance::LOOSE,
        )
        .expect("conformance run");
        assert!(res.is_match(), "wgpu Reduce {:?} diverged: {:?}", kind, res);
    }
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_softmax_matches_cpu_reference() {
    // 8 rows × 8 elements. Output same shape as input.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let in_n = 64;
    let row_size = 8;
    for (label, op) in &[("Softmax", "softmax"), ("LogSoftmax", "log_softmax")] {
        let kernel = if *op == "softmax" {
            KernelCall::Softmax(SoftmaxCall {
                input: SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                output: SlotSpan {
                    offset: in_n,
                    len: in_n,
                },
                size: row_size,
            })
        } else {
            KernelCall::LogSoftmax(SoftmaxCall {
                input: SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                output: SlotSpan {
                    offset: in_n,
                    len: in_n,
                },
                size: row_size,
            })
        };
        let plan = CompiledPlan {
            forward: Box::new([kernel]),
            backward: Box::new([]),
            address_table: empty_address_table(),
            workspace: WorkspaceLayout {
                total_elements: 2 * in_n,
            },
        };
        let mut cpu = CpuBackend::new();
        let res = check_forward(
            &plan,
            &mut cpu,
            &mut gpu,
            |buf| {
                // Mix of magnitudes per row to exercise the max-subtract
                // numerical-stability trick.
                let xs: Vec<f32> = (0..in_n)
                    .map(|i| {
                        ((i % row_size) as f32) - (row_size as f32 - 1.0) / 2.0 + 0.1 * i as f32
                    })
                    .collect();
                buf.write_span(
                    SlotSpan {
                        offset: 0,
                        len: in_n,
                    },
                    &xs,
                );
            },
            // Softmax output ∈ [0, 1] makes TIGHT realistic; LogSoftmax
            // can drift slightly because of the inner `log(sum)`.
            Tolerance::new(1e-5, 1e-5),
        )
        .expect("conformance run");
        assert!(res.is_match(), "wgpu {label} diverged: {:?}", res);
    }
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_norm_family_matches_cpu_reference() {
    // 4 rows × 8 elements. Weight + (optional) bias are length-8.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let row_size = 8_u32;
    let rows = 4_usize;
    let in_n = (row_size as usize) * rows;
    let eps_bits = 1.0e-5_f32.to_bits();

    // RmsNorm and InstanceNorm: workspace [input | weight | output].
    for (label, build) in &[
        ("RmsNorm", KernelCall::RmsNorm as fn(_) -> _),
        ("InstanceNorm", KernelCall::InstanceNorm as fn(_) -> _),
    ] {
        let plan = CompiledPlan {
            forward: Box::new([build(NormScaleCall {
                input: SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                weight: SlotSpan {
                    offset: in_n,
                    len: row_size as usize,
                },
                output: SlotSpan {
                    offset: in_n + row_size as usize,
                    len: in_n,
                },
                size: row_size,
                epsilon: eps_bits,
            })]),
            backward: Box::new([]),
            address_table: empty_address_table(),
            workspace: WorkspaceLayout {
                total_elements: 2 * in_n + row_size as usize,
            },
        };
        let mut cpu = CpuBackend::new();
        let res = check_forward(
            &plan,
            &mut cpu,
            &mut gpu,
            |buf| seed_norm_inputs(buf, in_n, row_size as usize),
            Tolerance::new(1e-5, 1e-5),
        )
        .expect("conformance run");
        assert!(res.is_match(), "wgpu {label} diverged: {:?}", res);
    }

    // LayerNorm: workspace [input | weight | bias | output].
    let plan = CompiledPlan {
        forward: Box::new([KernelCall::LayerNorm(NormFullCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            weight: SlotSpan {
                offset: in_n,
                len: row_size as usize,
            },
            bias: SlotSpan {
                offset: in_n + row_size as usize,
                len: row_size as usize,
            },
            output: SlotSpan {
                offset: in_n + 2 * row_size as usize,
                len: in_n,
            },
            size: row_size,
            epsilon: eps_bits,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 2 * in_n + 2 * row_size as usize,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            seed_norm_inputs(buf, in_n, row_size as usize);
            // Bias.
            let b: Vec<f32> = (0..row_size as usize).map(|i| 0.05 * i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: in_n + row_size as usize,
                    len: row_size as usize,
                },
                &b,
            );
        },
        Tolerance::new(1e-5, 1e-5),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu LayerNorm diverged: {:?}", res);

    // AddRmsNorm: workspace [residual | input | weight | output].
    let plan = CompiledPlan {
        forward: Box::new([KernelCall::AddRmsNorm(AddRmsNormCall {
            residual: SlotSpan {
                offset: 0,
                len: in_n,
            },
            input: SlotSpan {
                offset: in_n,
                len: in_n,
            },
            weight: SlotSpan {
                offset: 2 * in_n,
                len: row_size as usize,
            },
            output: SlotSpan {
                offset: 2 * in_n + row_size as usize,
                len: in_n,
            },
            size: row_size,
            epsilon: eps_bits,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 3 * in_n + row_size as usize,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let res_data: Vec<f32> = (0..in_n).map(|i| 0.05 * i as f32 - 0.5).collect();
            let in_data: Vec<f32> = (0..in_n).map(|i| -0.03 * i as f32 + 0.2).collect();
            let w_data: Vec<f32> = (0..row_size as usize)
                .map(|i| 1.0 + 0.1 * i as f32)
                .collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &res_data,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n,
                    len: in_n,
                },
                &in_data,
            );
            buf.write_span(
                SlotSpan {
                    offset: 2 * in_n,
                    len: row_size as usize,
                },
                &w_data,
            );
        },
        Tolerance::new(1e-5, 1e-5),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu AddRmsNorm diverged: {:?}", res);
}

fn seed_norm_inputs(buf: &mut hologram_transform::BufferSet, in_n: usize, row_size: usize) {
    let xs: Vec<f32> = (0..in_n).map(|i| 0.05 * i as f32 - 1.0).collect();
    let ws: Vec<f32> = (0..row_size).map(|i| 1.0 + 0.1 * i as f32).collect();
    buf.write_span(
        SlotSpan {
            offset: 0,
            len: in_n,
        },
        &xs,
    );
    buf.write_span(
        SlotSpan {
            offset: in_n,
            len: row_size,
        },
        &ws,
    );
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_matmul_matches_cpu_reference() {
    // 4×3 × 3×5 → 4×5. Sized so workgroup (64) covers in one dispatch.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let m = 4_usize;
    let k = 3_usize;
    let n = 5_usize;
    let plan = CompiledPlan {
        forward: Box::new([KernelCall::MatMul(MatMulCall {
            a: SlotSpan {
                offset: 0,
                len: m * k,
            },
            b: SlotSpan {
                offset: m * k,
                len: k * n,
            },
            c: SlotSpan {
                offset: m * k + k * n,
                len: m * n,
            },
            m,
            k,
            n,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: m * k + k * n + m * n,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let a: Vec<f32> = (0..m * k).map(|i| 0.1 * i as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| -0.2 * i as f32 + 0.5).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: m * k,
                },
                &a,
            );
            buf.write_span(
                SlotSpan {
                    offset: m * k,
                    len: k * n,
                },
                &b,
            );
        },
        Tolerance::new(1e-5, 1e-5),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu MatMul diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_matmul_grad_a_accumulates_correctly() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let m = 4_usize;
    let k = 3_usize;
    let n = 5_usize;
    // Workspace: dc | b | da_seed.
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::MatMulGradA(MatMulGradACall {
            dc: SlotSpan {
                offset: 0,
                len: m * n,
            },
            b: SlotSpan {
                offset: m * n,
                len: k * n,
            },
            da: SlotSpan {
                offset: m * n + k * n,
                len: m * k,
            },
            m,
            k,
            n,
        })]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: m * n + k * n + m * k,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = hologram_transform::check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let dc: Vec<f32> = (0..m * n).map(|i| 0.05 * i as f32 + 0.1).collect();
            let b: Vec<f32> = (0..k * n).map(|i| -0.03 * i as f32 + 0.7).collect();
            let da_seed: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: m * n,
                },
                &dc,
            );
            buf.write_span(
                SlotSpan {
                    offset: m * n,
                    len: k * n,
                },
                &b,
            );
            buf.write_span(
                SlotSpan {
                    offset: m * n + k * n,
                    len: m * k,
                },
                &da_seed,
            );
        },
        Tolerance::new(1e-5, 1e-5),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu MatMulGradA diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_matmul_grad_b_accumulates_correctly() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let m = 4_usize;
    let k = 3_usize;
    let n = 5_usize;
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::MatMulGradB(MatMulGradBCall {
            a: SlotSpan {
                offset: 0,
                len: m * k,
            },
            dc: SlotSpan {
                offset: m * k,
                len: m * n,
            },
            db: SlotSpan {
                offset: m * k + m * n,
                len: k * n,
            },
            m,
            k,
            n,
        })]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: m * k + m * n + k * n,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = hologram_transform::check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let a: Vec<f32> = (0..m * k).map(|i| 0.1 * i as f32).collect();
            let dc: Vec<f32> = (0..m * n).map(|i| -0.04 * i as f32 + 0.3).collect();
            let db_seed: Vec<f32> = (0..k * n).map(|i| 0.02 * i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: m * k,
                },
                &a,
            );
            buf.write_span(
                SlotSpan {
                    offset: m * k,
                    len: m * n,
                },
                &dc,
            );
            buf.write_span(
                SlotSpan {
                    offset: m * k + m * n,
                    len: k * n,
                },
                &db_seed,
            );
        },
        Tolerance::new(1e-5, 1e-5),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu MatMulGradB diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_pool2d_family_matches_cpu_reference() {
    // 1 batch × 2 channels × 4×4 → kernel 2×2 stride 2 → 1×2×2×2 output.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let n = 1_u32;
    let c = 2_u32;
    let h_in = 4_u32;
    let w_in = 4_u32;
    let h_out = 2_u32;
    let w_out = 2_u32;
    let in_n = (n * c * h_in * w_in) as usize;
    let out_n = (n * c * h_out * w_out) as usize;

    for kind in [Pool2dKind::Max, Pool2dKind::Avg] {
        let plan = CompiledPlan {
            forward: Box::new([KernelCall::Pool2d(
                Pool2dCall {
                    input: SlotSpan {
                        offset: 0,
                        len: in_n,
                    },
                    output: SlotSpan {
                        offset: in_n,
                        len: out_n,
                    },
                    n,
                    c,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    kernel_h: 2,
                    kernel_w: 2,
                    stride_h: 2,
                    stride_w: 2,
                    pad_h: 0,
                    pad_w: 0,
                },
                kind,
            )]),
            backward: Box::new([]),
            address_table: empty_address_table(),
            workspace: WorkspaceLayout {
                total_elements: in_n + out_n,
            },
        };
        let mut cpu = CpuBackend::new();
        let res = check_forward(
            &plan,
            &mut cpu,
            &mut gpu,
            |buf| {
                let xs: Vec<f32> = (0..in_n).map(|i| 0.1 * i as f32 - 1.0).collect();
                buf.write_span(
                    SlotSpan {
                        offset: 0,
                        len: in_n,
                    },
                    &xs,
                );
            },
            Tolerance::TIGHT,
        )
        .expect("conformance run");
        assert!(res.is_match(), "wgpu Pool2d {:?} diverged: {:?}", kind, res);
    }
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_global_avg_pool_matches_cpu_reference() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let n = 1_u32;
    let c = 3_u32;
    let h = 4_u32;
    let w = 4_u32;
    let in_n = (n * c * h * w) as usize;
    let out_n = (n * c) as usize;
    let plan = CompiledPlan {
        forward: Box::new([KernelCall::GlobalAvgPool(GlobalAvgPoolCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            output: SlotSpan {
                offset: in_n,
                len: out_n,
            },
            n,
            c,
            h,
            w,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: in_n + out_n,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let xs: Vec<f32> = (0..in_n).map(|i| 0.05 * i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &xs,
            );
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu GlobalAvgPool diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_conv2d_matches_cpu_reference() {
    // 1×2×4×4 input, 3 output channels with 3×3 kernel, padding 1,
    // stride 1 → 1×3×4×4 output. Bias present.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let n = 1_u32;
    let c_in = 2_u32;
    let c_out = 3_u32;
    let h_in = 4_u32;
    let w_in = 4_u32;
    let h_out = 4_u32;
    let w_out = 4_u32;
    let kernel_h = 3_u32;
    let kernel_w = 3_u32;
    let in_n = (n * c_in * h_in * w_in) as usize;
    let weight_n = (c_out * c_in * kernel_h * kernel_w) as usize;
    let bias_n = c_out as usize;
    let out_n = (n * c_out * h_out * w_out) as usize;

    let plan = CompiledPlan {
        forward: Box::new([KernelCall::Conv2d(Conv2dCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            weight: SlotSpan {
                offset: in_n,
                len: weight_n,
            },
            bias: SlotSpan {
                offset: in_n + weight_n,
                len: bias_n,
            },
            output: SlotSpan {
                offset: in_n + weight_n + bias_n,
                len: out_n,
            },
            n,
            c_in,
            c_out,
            h_in,
            w_in,
            h_out,
            w_out,
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: in_n + weight_n + bias_n + out_n,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let xs: Vec<f32> = (0..in_n).map(|i| 0.05 * i as f32 - 0.5).collect();
            let ws: Vec<f32> = (0..weight_n)
                .map(|i| ((i as f32) * 0.07).cos() * 0.3)
                .collect();
            let bs: Vec<f32> = (0..bias_n).map(|i| 0.1 * i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &xs,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n,
                    len: weight_n,
                },
                &ws,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n + weight_n,
                    len: bias_n,
                },
                &bs,
            );
        },
        Tolerance::new(5e-5, 5e-5),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu Conv2d diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_conv_transpose_2d_matches_cpu_reference() {
    // 1×2×3×3 input, 3 output channels, 3×3 kernel, stride 2 → 1×3×7×7
    // (the upsampling case that exercises stride-aware index inversion).
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let n = 1_u32;
    let c_in = 2_u32;
    let c_out = 3_u32;
    let h_in = 3_u32;
    let w_in = 3_u32;
    let h_out = 7_u32;
    let w_out = 7_u32;
    let kernel_h = 3_u32;
    let kernel_w = 3_u32;
    let in_n = (n * c_in * h_in * w_in) as usize;
    // ConvTranspose weight is [C_in, C_out/group, kH, kW].
    let weight_n = (c_in * c_out * kernel_h * kernel_w) as usize;
    let bias_n = c_out as usize;
    let out_n = (n * c_out * h_out * w_out) as usize;

    let plan = CompiledPlan {
        forward: Box::new([KernelCall::ConvTranspose2d(ConvTransposeCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            weight: SlotSpan {
                offset: in_n,
                len: weight_n,
            },
            bias: SlotSpan {
                offset: in_n + weight_n,
                len: bias_n,
            },
            output: SlotSpan {
                offset: in_n + weight_n + bias_n,
                len: out_n,
            },
            n,
            c_in,
            c_out,
            h_in,
            w_in,
            h_out,
            w_out,
            kernel_h,
            kernel_w,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: in_n + weight_n + bias_n + out_n,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let xs: Vec<f32> = (0..in_n).map(|i| 0.1 * i as f32 - 0.5).collect();
            let ws: Vec<f32> = (0..weight_n)
                .map(|i| ((i as f32) * 0.07).sin() * 0.4)
                .collect();
            let bs: Vec<f32> = (0..bias_n).map(|i| 0.05 * i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &xs,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n,
                    len: weight_n,
                },
                &ws,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n + weight_n,
                    len: bias_n,
                },
                &bs,
            );
        },
        Tolerance::new(5e-5, 5e-5),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu ConvTranspose2d diverged: {:?}", res);
}

/// Confirms the dispatch arm is exhaustive: every canonical
#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_rms_norm_grad_matches_cpu_reference() {
    norm_grad_test(/* layer = */ false);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_instance_norm_grad_matches_cpu_reference() {
    norm_grad_test(/* layer = */ true);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_fused_swiglu_matches_cpu_reference() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let plan = binary_plan(KernelCall::FusedSwiGlu(BinaryCall {
        a: SlotSpan { offset: 0, len: N },
        b: SlotSpan { offset: N, len: N },
        c: SlotSpan {
            offset: 2 * N,
            len: N,
        },
    }));
    run_binary(&plan, &mut gpu, |i| 0.07 * i as f32 + 0.2);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_group_norm_matches_cpu_reference() {
    // 16-element input with 4 groups → group_elements = 4.
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let in_n = 16_usize;
    let groups = 4_u32;
    let group_elements = in_n / groups as usize;
    let eps_bits = 1.0e-5_f32.to_bits();
    let plan = CompiledPlan {
        forward: Box::new([KernelCall::GroupNorm(GroupNormCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            weight: SlotSpan {
                offset: in_n,
                len: group_elements,
            },
            bias: SlotSpan {
                offset: in_n + group_elements,
                len: group_elements,
            },
            output: SlotSpan {
                offset: in_n + 2 * group_elements,
                len: in_n,
            },
            num_groups: groups,
            epsilon: eps_bits,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 2 * in_n + 2 * group_elements,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let xs: Vec<f32> = (0..in_n).map(|i| 0.1 * i as f32 - 0.5).collect();
            let ws: Vec<f32> = (0..group_elements).map(|i| 1.0 + 0.1 * i as f32).collect();
            let bs: Vec<f32> = (0..group_elements).map(|i| 0.05 * i as f32).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &xs,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n,
                    len: group_elements,
                },
                &ws,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n + group_elements,
                    len: group_elements,
                },
                &bs,
            );
        },
        Tolerance::new(1e-5, 1e-5),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu GroupNorm diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_softmax_grad_family_matches_cpu_reference() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let size = 8;
    let rows = 4;
    let n = size * rows;
    for kind in [SoftmaxGradKind::Softmax, SoftmaxGradKind::LogSoftmax] {
        let plan = CompiledPlan {
            forward: Box::new([]),
            backward: Box::new([KernelCall::SoftmaxGrad(
                SoftmaxGradCall {
                    output: SlotSpan { offset: 0, len: n },
                    dc: SlotSpan { offset: n, len: n },
                    da: SlotSpan {
                        offset: 2 * n,
                        len: n,
                    },
                    size,
                },
                kind,
            )]),
            address_table: empty_address_table(),
            workspace: WorkspaceLayout {
                total_elements: 3 * n,
            },
        };
        let mut cpu = CpuBackend::new();
        let res = check_forward_then_backward(
            &plan,
            &mut cpu,
            &mut gpu,
            |buf| {
                // For Softmax the "output" is row-normalised
                // probabilities ∈ (0, 1). For LogSoftmax it's the row's
                // log-probabilities (negative). Use a simple seed that
                // works for both — exp(o) only matters for log-softmax,
                // and any small finite o keeps things in range.
                let out: Vec<f32> = (0..n).map(|i| -0.05 * i as f32 + 0.4).collect();
                let dc: Vec<f32> = (0..n).map(|i| 0.03 * i as f32 - 0.1).collect();
                buf.write_span(SlotSpan { offset: 0, len: n }, &out);
                buf.write_span(SlotSpan { offset: n, len: n }, &dc);
            },
            Tolerance::new(1e-5, 1e-5),
        )
        .expect("conformance run");
        assert!(
            res.is_match(),
            "wgpu SoftmaxGrad {:?} diverged: {:?}",
            kind,
            res
        );
    }
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_add_rms_norm_grad_matches_cpu_reference() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let size = 8_u32;
    let rows = 4_usize;
    let in_n = (size as usize) * rows;
    let eps_bits = 1.0e-5_f32.to_bits();
    // Workspace: residual | input | weight | dy | d_residual | d_input | dw.
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::AddRmsNormGrad(AddRmsNormGradCall {
            residual: SlotSpan {
                offset: 0,
                len: in_n,
            },
            input: SlotSpan {
                offset: in_n,
                len: in_n,
            },
            weight: SlotSpan {
                offset: 2 * in_n,
                len: size as usize,
            },
            dy: SlotSpan {
                offset: 2 * in_n + size as usize,
                len: in_n,
            },
            d_residual: SlotSpan {
                offset: 3 * in_n + size as usize,
                len: in_n,
            },
            d_input: SlotSpan {
                offset: 4 * in_n + size as usize,
                len: in_n,
            },
            dw: SlotSpan {
                offset: 5 * in_n + size as usize,
                len: size as usize,
            },
            size,
            epsilon: eps_bits,
        })]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 5 * in_n + 2 * size as usize,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let res_data: Vec<f32> = (0..in_n).map(|i| 0.03 * i as f32 - 0.2).collect();
            let in_data: Vec<f32> = (0..in_n).map(|i| -0.02 * i as f32 + 0.4).collect();
            let ws: Vec<f32> = (0..size as usize).map(|i| 1.0 + 0.1 * i as f32).collect();
            let dy: Vec<f32> = (0..in_n).map(|i| 0.02 * i as f32 + 0.1).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &res_data,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n,
                    len: in_n,
                },
                &in_data,
            );
            buf.write_span(
                SlotSpan {
                    offset: 2 * in_n,
                    len: size as usize,
                },
                &ws,
            );
            buf.write_span(
                SlotSpan {
                    offset: 2 * in_n + size as usize,
                    len: in_n,
                },
                &dy,
            );
        },
        Tolerance::new(1e-4, 1e-4),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu AddRmsNormGrad diverged: {:?}", res);
}

#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_layer_norm_grad_matches_cpu_reference() {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let size = 8_u32;
    let rows = 4_usize;
    let in_n = (size as usize) * rows;
    let eps_bits = 1.0e-5_f32.to_bits();
    // Workspace: input | weight | dy | dx | dw | db.
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([KernelCall::LayerNormGrad(LayerNormGradCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            weight: SlotSpan {
                offset: in_n,
                len: size as usize,
            },
            dy: SlotSpan {
                offset: in_n + size as usize,
                len: in_n,
            },
            dx: SlotSpan {
                offset: 2 * in_n + size as usize,
                len: in_n,
            },
            dw: SlotSpan {
                offset: 3 * in_n + size as usize,
                len: size as usize,
            },
            db: SlotSpan {
                offset: 3 * in_n + 2 * size as usize,
                len: size as usize,
            },
            size,
            epsilon: eps_bits,
        })]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 3 * in_n + 3 * size as usize,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let xs: Vec<f32> = (0..in_n).map(|i| 0.05 * i as f32 - 0.5).collect();
            let ws: Vec<f32> = (0..size as usize).map(|i| 1.0 + 0.1 * i as f32).collect();
            let dy: Vec<f32> = (0..in_n).map(|i| 0.02 * i as f32 + 0.1).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &xs,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n,
                    len: size as usize,
                },
                &ws,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n + size as usize,
                    len: in_n,
                },
                &dy,
            );
        },
        Tolerance::new(1e-4, 1e-4),
    )
    .expect("conformance run");
    assert!(res.is_match(), "wgpu LayerNormGrad diverged: {:?}", res);
}

fn norm_grad_test(instance: bool) {
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let size = 8_u32;
    let rows = 4_usize;
    let in_n = (size as usize) * rows;
    let eps_bits = 1.0e-5_f32.to_bits();
    // Workspace: input | weight | dy | dx | dw.
    let plan_call = if instance {
        KernelCall::InstanceNormGrad(InstanceNormGradCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            weight: SlotSpan {
                offset: in_n,
                len: size as usize,
            },
            dy: SlotSpan {
                offset: in_n + size as usize,
                len: in_n,
            },
            dx: SlotSpan {
                offset: 2 * in_n + size as usize,
                len: in_n,
            },
            dw: SlotSpan {
                offset: 3 * in_n + size as usize,
                len: size as usize,
            },
            size,
            epsilon: eps_bits,
        })
    } else {
        KernelCall::RmsNormGrad(RmsNormGradCall {
            input: SlotSpan {
                offset: 0,
                len: in_n,
            },
            weight: SlotSpan {
                offset: in_n,
                len: size as usize,
            },
            dy: SlotSpan {
                offset: in_n + size as usize,
                len: in_n,
            },
            dx: SlotSpan {
                offset: 2 * in_n + size as usize,
                len: in_n,
            },
            dw: SlotSpan {
                offset: 3 * in_n + size as usize,
                len: size as usize,
            },
            size,
            epsilon: eps_bits,
        })
    };
    let plan = CompiledPlan {
        forward: Box::new([]),
        backward: Box::new([plan_call]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 3 * in_n + 2 * size as usize,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward_then_backward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let xs: Vec<f32> = (0..in_n).map(|i| 0.05 * i as f32 - 0.5).collect();
            let ws: Vec<f32> = (0..size as usize).map(|i| 1.0 + 0.1 * i as f32).collect();
            let dy: Vec<f32> = (0..in_n).map(|i| 0.02 * i as f32 + 0.1).collect();
            buf.write_span(
                SlotSpan {
                    offset: 0,
                    len: in_n,
                },
                &xs,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n,
                    len: size as usize,
                },
                &ws,
            );
            buf.write_span(
                SlotSpan {
                    offset: in_n + size as usize,
                    len: in_n,
                },
                &dy,
            );
        },
        Tolerance::new(1e-4, 1e-4),
    )
    .expect("conformance run");
    let label = if instance {
        "InstanceNormGrad"
    } else {
        "RmsNormGrad"
    };
    assert!(res.is_match(), "wgpu {label} diverged: {:?}", res);
}

/// `KernelCall` variant either runs a real WGSL shader or routes
/// through `host_cpu_fallback`. Adding a new variant without
/// updating the match fails the build (no `_` arm) — this test just
/// verifies a representative still-fallback variant returns success
/// with correct values.
#[test]
#[ignore = "requires a working wgpu adapter (Vulkan/Metal/DX12)"]
fn wgpu_attention_via_cpu_fallback_matches_reference() {
    use hologram_transform::AttentionCall;
    let mut gpu = WgpuBackend::new().expect("wgpu init");
    let head_dim = 4_usize;
    let seq = 3_usize;
    let n = seq * head_dim;
    let plan = CompiledPlan {
        forward: Box::new([KernelCall::Attention(AttentionCall {
            q: SlotSpan { offset: 0, len: n },
            k: SlotSpan { offset: n, len: n },
            v: SlotSpan {
                offset: 2 * n,
                len: n,
            },
            output: SlotSpan {
                offset: 3 * n,
                len: n,
            },
            scratch: SlotSpan::empty(0),
            batch: 1,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: head_dim as u32,
            seq_q: seq as u32,
            seq_kv: seq as u32,
            scale_bits: 1.0_f32.to_bits(),
            causal: false,
        })]),
        backward: Box::new([]),
        address_table: empty_address_table(),
        workspace: WorkspaceLayout {
            total_elements: 4 * n,
        },
    };
    let mut cpu = CpuBackend::new();
    let res = check_forward(
        &plan,
        &mut cpu,
        &mut gpu,
        |buf| {
            let q: Vec<f32> = (0..n).map(|i| 0.05 * i as f32).collect();
            let k: Vec<f32> = (0..n).map(|i| -0.03 * i as f32 + 0.4).collect();
            let v: Vec<f32> = (0..n).map(|i| 0.07 * i as f32 - 0.5).collect();
            buf.write_span(SlotSpan { offset: 0, len: n }, &q);
            buf.write_span(SlotSpan { offset: n, len: n }, &k);
            buf.write_span(
                SlotSpan {
                    offset: 2 * n,
                    len: n,
                },
                &v,
            );
        },
        Tolerance::TIGHT,
    )
    .expect("conformance run");
    assert!(
        res.is_match(),
        "wgpu Attention fallback diverged: {:?}",
        res
    );
}
