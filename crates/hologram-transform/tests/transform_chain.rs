//! End-to-end tests: chain → plan → execute, for ADD and MatMul.

use hologram_ops::{
    ConcatAttrs, Conv2dAttrs, GroupNormAttrs, NormAttrs, SliceAttrs, TransposeAttrs,
};
use hologram_transform::{
    compile, AddInputs, AddRmsNormInputs, AddressRef, BufferSet, Conv2dInputs, Executor,
    KernelCall, MatMulInputs, NormFullInputs, NormScaleInputs, SemanticOp, TensorId,
    TransformChain, UnaryInputs,
};

fn build_add_chain(grad: bool) -> TransformChain {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[3], grad);
    let bb = b.add_tensor(&[3], grad);
    let c = b.add_tensor(&[3], grad);
    b.push_add(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    b.build()
}

fn build_matmul_chain() -> TransformChain {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[2, 3], true);
    let bb = b.add_tensor(&[3, 2], true);
    let c = b.add_tensor(&[2, 2], true);
    b.push_matmul(MatMulInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    })
    .expect("valid matmul shapes");
    b.build()
}

#[test]
fn build_chain_creates_tensors_and_node() {
    let chain = build_add_chain(true);
    assert_eq!(chain.tensors.len(), 3);
    assert_eq!(chain.nodes.len(), 1);
}

#[test]
fn resolves_address_refs_to_disjoint_spans() {
    let plan = compile(&build_add_chain(true)).unwrap();
    let t0 = plan.address_table.span(hologram_transform::TensorId(0));
    let t1 = plan.address_table.span(hologram_transform::TensorId(1));
    let t2 = plan.address_table.span(hologram_transform::TensorId(2));
    assert_eq!(t0.offset, 0);
    assert_eq!(t1.offset, 3);
    assert_eq!(t2.offset, 6);
    assert_eq!(plan.workspace_elements(), 18); // 3 values + 3 grads
}

#[test]
fn add_chain_compiles_forward_and_backward() {
    let plan = compile(&build_add_chain(true)).unwrap();
    assert_eq!(plan.forward_len(), 1);
    assert_eq!(plan.backward_len(), 1);
    assert!(matches!(plan.forward[0], KernelCall::Add(_)));
    assert!(matches!(plan.backward[0], KernelCall::AddGrad(_)));
}

#[test]
fn matmul_chain_compiles_forward_and_backward() {
    let plan = compile(&build_matmul_chain()).unwrap();
    assert_eq!(plan.forward_len(), 1);
    assert_eq!(plan.backward_len(), 2);
    assert!(matches!(plan.forward[0], KernelCall::MatMul(_)));
    assert!(matches!(plan.backward[0], KernelCall::MatMulGradA(_)));
    assert!(matches!(plan.backward[1], KernelCall::MatMulGradB(_)));
}

#[test]
fn execute_add_forward_produces_elementwise_sum() {
    let plan = compile(&build_add_chain(true)).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, hologram_transform::TensorId(0), &[1.0, 2.0, 3.0]);
    buf.write_tensor(&plan, hologram_transform::TensorId(1), &[10.0, 20.0, 30.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    let c = buf.read_tensor(&plan, hologram_transform::TensorId(2));
    assert_eq!(c, &[11.0, 22.0, 33.0]);
}

#[test]
fn execute_add_backward_accumulates_into_both_grads() {
    let plan = compile(&build_add_chain(true)).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_grad(&plan, hologram_transform::TensorId(2), &[1.0, 1.0, 1.0]);
    Executor::run_backward(&plan, &mut buf).unwrap();
    let da = buf.read_grad(&plan, hologram_transform::TensorId(0));
    let db = buf.read_grad(&plan, hologram_transform::TensorId(1));
    assert_eq!(da, &[1.0, 1.0, 1.0]);
    assert_eq!(db, &[1.0, 1.0, 1.0]);
}

#[test]
fn execute_matmul_forward_matches_reference() {
    let plan = compile(&build_matmul_chain()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]; C = [[4,5],[10,11]]
    buf.write_tensor(
        &plan,
        hologram_transform::TensorId(0),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    buf.write_tensor(
        &plan,
        hologram_transform::TensorId(1),
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    );
    Executor::run_forward(&plan, &mut buf).unwrap();
    let c = buf.read_tensor(&plan, hologram_transform::TensorId(2));
    assert_eq!(c, &[4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn execute_matmul_backward_writes_both_grads() {
    let plan = compile(&build_matmul_chain()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    // A = [[1,2,3],[4,5,6]]; B = [[7,8],[9,10],[11,12]]; dC = [[1,1],[1,1]]
    buf.write_tensor(
        &plan,
        hologram_transform::TensorId(0),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    buf.write_tensor(
        &plan,
        hologram_transform::TensorId(1),
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    );
    buf.write_grad(
        &plan,
        hologram_transform::TensorId(2),
        &[1.0, 1.0, 1.0, 1.0],
    );
    Executor::run_backward(&plan, &mut buf).unwrap();
    // dA[i,p] = sum_j dC[i,j] * B[p,j]; with dC=ones, dA[i,p] = B[p,0]+B[p,1].
    // Row sums of B = [15,19,23]. So dA = [[15,19,23],[15,19,23]].
    let da = buf.read_grad(&plan, hologram_transform::TensorId(0));
    assert_eq!(da, &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
    // dB[p,j] = sum_i A[i,p] * dC[i,j]; with dC=ones, dB[p,j] = A[0,p]+A[1,p].
    // Col sums of A = [5, 7, 9]; expand to [k=3, n=2] => [[5,5],[7,7],[9,9]].
    let db = buf.read_grad(&plan, hologram_transform::TensorId(1));
    assert_eq!(db, &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn backward_planning_emits_kernel_calls_not_runtime_traversal() {
    // The proof here is that `plan.backward` is a `Box<[KernelCall]>` of
    // fixed length, populated at compile time, with concrete spans.
    let plan = compile(&build_matmul_chain()).unwrap();
    assert_eq!(plan.backward_len(), 2);
    // Ensure each backward call has resolved (non-empty) shape metadata.
    for call in plan.backward.iter() {
        match call {
            KernelCall::MatMulGradA(c) => {
                assert!(c.m > 0 && c.k > 0 && c.n > 0);
                assert!(c.da.len > 0);
            }
            KernelCall::MatMulGradB(c) => {
                assert!(c.m > 0 && c.k > 0 && c.n > 0);
                assert!(c.db.len > 0);
            }
            _ => panic!("unexpected backward kernel"),
        }
    }
}

#[test]
fn execute_sub_forward_produces_elementwise_difference() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[3], false);
    let bb = b.add_tensor(&[3], false);
    let c = b.add_tensor(&[3], false);
    b.push_sub_forward_only(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, hologram_transform::TensorId(0), &[10.0, 20.0, 30.0]);
    buf.write_tensor(&plan, hologram_transform::TensorId(1), &[1.0, 2.0, 3.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(
        buf.read_tensor(&plan, hologram_transform::TensorId(2)),
        &[9.0, 18.0, 27.0]
    );
}

#[test]
fn execute_mul_forward_produces_elementwise_product() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[3], false);
    let bb = b.add_tensor(&[3], false);
    let c = b.add_tensor(&[3], false);
    b.push_mul_forward_only(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, hologram_transform::TensorId(0), &[1.0, 2.0, 3.0]);
    buf.write_tensor(&plan, hologram_transform::TensorId(1), &[4.0, 5.0, 6.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(
        buf.read_tensor(&plan, hologram_transform::TensorId(2)),
        &[4.0, 10.0, 18.0]
    );
}

#[test]
fn sub_and_mul_emit_backward_when_grad_enabled() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[2], true);
    let bb = b.add_tensor(&[2], true);
    let c = b.add_tensor(&[2], true);
    b.push_sub(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    b.push_mul(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    let plan = compile(&b.build()).unwrap();
    assert_eq!(plan.forward_len(), 2);
    // 2 ops, each emits one *Grad call → backward len = 2.
    assert_eq!(plan.backward_len(), 2);
}

#[test]
fn sub_and_mul_forward_only_skip_backward() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[2], false);
    let bb = b.add_tensor(&[2], false);
    let c = b.add_tensor(&[2], false);
    b.push_sub_forward_only(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    b.push_mul_forward_only(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    let plan = compile(&b.build()).unwrap();
    assert_eq!(plan.forward_len(), 2);
    assert_eq!(plan.backward_len(), 0);
}

#[test]
fn execute_sub_backward_writes_opposite_signs() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[3], true);
    let bb = b.add_tensor(&[3], true);
    let c = b.add_tensor(&[3], true);
    b.push_sub(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_grad(&plan, hologram_transform::TensorId(2), &[1.0, 2.0, 3.0]);
    Executor::run_backward(&plan, &mut buf).unwrap();
    let da = buf.read_grad(&plan, hologram_transform::TensorId(0));
    let db = buf.read_grad(&plan, hologram_transform::TensorId(1));
    assert_eq!(da, &[1.0, 2.0, 3.0]);
    assert_eq!(db, &[-1.0, -2.0, -3.0]);
}

#[test]
fn execute_mul_backward_uses_forward_inputs() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[3], true);
    let bb = b.add_tensor(&[3], true);
    let c = b.add_tensor(&[3], true);
    b.push_mul(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, hologram_transform::TensorId(0), &[2.0, 3.0, 5.0]);
    buf.write_tensor(&plan, hologram_transform::TensorId(1), &[7.0, 11.0, 13.0]);
    buf.write_grad(&plan, hologram_transform::TensorId(2), &[1.0, 1.0, 1.0]);
    Executor::run_backward(&plan, &mut buf).unwrap();
    // dA = dC * B = [7, 11, 13];  dB = dC * A = [2, 3, 5].
    let da = buf.read_grad(&plan, hologram_transform::TensorId(0));
    let db = buf.read_grad(&plan, hologram_transform::TensorId(1));
    assert_eq!(da, &[7.0, 11.0, 13.0]);
    assert_eq!(db, &[2.0, 3.0, 5.0]);
}

#[test]
fn execute_div_backward_uses_b_squared_term() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[3], true);
    let bb = b.add_tensor(&[3], true);
    let c = b.add_tensor(&[3], true);
    // Build via push_node directly since push_div is forward-only.
    // Use the canonical path: SemanticOp::Div with backward enabled.
    use hologram_ops::SemanticOp;
    use smallvec::SmallVec;
    let nid = hologram_transform::NodeId(0);
    let mut chain = b.build();
    chain.nodes.push(hologram_transform::TransformNode {
        id: nid,
        op: SemanticOp::Div,
        inputs: SmallVec::from_slice(&[AddressRef::of(a), AddressRef::of(bb)]),
        outputs: SmallVec::from_slice(&[AddressRef::of(c)]),
        backward: SemanticOp::Div.backward(),
    });
    let plan = compile(&chain).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, hologram_transform::TensorId(0), &[6.0, 12.0, 20.0]);
    buf.write_tensor(&plan, hologram_transform::TensorId(1), &[2.0, 3.0, 5.0]);
    buf.write_grad(&plan, hologram_transform::TensorId(2), &[1.0, 1.0, 1.0]);
    Executor::run_backward(&plan, &mut buf).unwrap();
    // dA = dC / B = [1/2, 1/3, 1/5]
    // dB = -dC * A / B² = [-6/4, -12/9, -20/25]
    let da = buf.read_grad(&plan, hologram_transform::TensorId(0));
    let db = buf.read_grad(&plan, hologram_transform::TensorId(1));
    for (got, want) in da.iter().zip([0.5_f32, 1.0 / 3.0, 0.2].iter()) {
        assert!((got - want).abs() < 1e-5);
    }
    for (got, want) in db.iter().zip([-1.5_f32, -12.0 / 9.0, -20.0 / 25.0].iter()) {
        assert!((got - want).abs() < 1e-5);
    }
}

#[test]
fn execute_sigmoid_backward_uses_forward_output() {
    // d/dx σ(x) = σ(x) * (1 - σ(x)). Verify via direct dispatch since
    // chain construction for differentiable unaries needs the future
    // `push_unary_with_grad` helper (Phase 3.4 cont.).
    use hologram_ops::{KernelCall, SlotSpan, UnaryGradCall, UnaryGradKind};
    let sigmoid = |x: f32| 1.0 / (1.0 + libm::expf(-x));
    let xs = [-2.0_f32, 0.0, 1.0];
    let n = xs.len();
    // Layout: [forward_out (n), dC (n), dA (n)].
    let mut s = vec![0.0_f32; 3 * n];
    for i in 0..n {
        s[i] = sigmoid(xs[i]);
    }
    s[n..2 * n].copy_from_slice(&[1.0, 1.0, 1.0]); // dC = ones
    let call = KernelCall::UnaryGrad(
        UnaryGradCall {
            source: SlotSpan { offset: 0, len: n },
            dc: SlotSpan { offset: n, len: n },
            da: SlotSpan {
                offset: 2 * n,
                len: n,
            },
        },
        UnaryGradKind::Sigmoid,
    );
    hologram_ops::dispatch(&mut s, &call);
    for (i, &x) in xs.iter().enumerate() {
        let s_x = sigmoid(x);
        let expected = s_x * (1.0 - s_x);
        assert!((s[2 * n + i] - expected).abs() < 1e-5);
    }
}

#[test]
fn execute_neg_backward_flips_sign() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[3], true);
    let out = b.add_tensor(&[3], true);
    b.push_unary(
        SemanticOp::Neg,
        UnaryInputs {
            input: AddressRef::of(inp),
            output: AddressRef::of(out),
        },
    )
    .unwrap();
    // `push_unary` is forward-only by design; manually flip the
    // backward flag on the constructed node by rebuilding via a
    // backward-enabling helper. For now, drive the kernel directly
    // through a forward-only chain and a hand-built backward.
    // (The chain builder gains differentiable unary helpers in a
    // later sprint when more unary ops have backward rules.)
    let _plan = compile(&b.build()).unwrap();
    // Demonstrate the kernel itself via direct dispatch — this also
    // covers `KernelCall::NegGrad` end-to-end.
    use hologram_ops::{KernelCall, NegGradCall, SlotSpan};
    let mut s = [1.0_f32, 2.0, 3.0, 0.0, 0.0, 0.0];
    let call = KernelCall::NegGrad(NegGradCall {
        dc: SlotSpan { offset: 0, len: 3 },
        da: SlotSpan { offset: 3, len: 3 },
    });
    hologram_ops::dispatch(&mut s, &call);
    assert_eq!(&s[3..6], &[-1.0, -2.0, -3.0]);
}

#[test]
fn execute_relu_via_unary_dispatch() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[4], false);
    let out = b.add_tensor(&[4], false);
    b.push_unary(
        SemanticOp::Relu,
        UnaryInputs {
            input: AddressRef::of(inp),
            output: AddressRef::of(out),
        },
    )
    .unwrap();
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(
        &plan,
        hologram_transform::TensorId(0),
        &[-1.0, 2.0, -3.0, 4.0],
    );
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(
        buf.read_tensor(&plan, hologram_transform::TensorId(1)),
        &[0.0, 2.0, 0.0, 4.0]
    );
}

#[test]
fn execute_chain_of_unary_ops_via_unary_dispatch() {
    // -x  →  exp(-x)  →  sigmoid form
    let mut b = TransformChain::builder();
    let t0 = b.add_tensor(&[3], false);
    let t1 = b.add_tensor(&[3], false);
    let t2 = b.add_tensor(&[3], false);
    b.push_unary(
        SemanticOp::Neg,
        UnaryInputs {
            input: AddressRef::of(t0),
            output: AddressRef::of(t1),
        },
    )
    .unwrap();
    b.push_unary(
        SemanticOp::Exp,
        UnaryInputs {
            input: AddressRef::of(t1),
            output: AddressRef::of(t2),
        },
    )
    .unwrap();
    let plan = compile(&b.build()).unwrap();
    assert_eq!(plan.forward_len(), 2);
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, hologram_transform::TensorId(0), &[0.0, 1.0, -1.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    let out = buf.read_tensor(&plan, hologram_transform::TensorId(2));
    let expected: [f32; 3] = [1.0, libm::expf(-1.0), libm::expf(1.0)];
    for (got, want) in out.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5);
    }
}

#[test]
fn execute_div_forward_produces_elementwise_quotient() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[3], false);
    let bb = b.add_tensor(&[3], false);
    let c = b.add_tensor(&[3], false);
    b.push_div(AddInputs {
        a: AddressRef::of(a),
        b: AddressRef::of(bb),
        c: AddressRef::of(c),
    });
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, hologram_transform::TensorId(0), &[10.0, 20.0, 30.0]);
    buf.write_tensor(&plan, hologram_transform::TensorId(1), &[2.0, 4.0, 5.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(
        buf.read_tensor(&plan, hologram_transform::TensorId(2)),
        &[5.0, 5.0, 6.0]
    );
}

#[test]
fn execute_softmax_normalises_rows() {
    // Two rows of size 3.
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[2, 3], false);
    let out = b.add_tensor(&[2, 3], false);
    b.push_softmax(
        3,
        UnaryInputs {
            input: AddressRef::of(inp),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(
        &plan,
        hologram_transform::TensorId(0),
        &[1.0, 2.0, 3.0, 5.0, 5.0, 5.0],
    );
    Executor::run_forward(&plan, &mut buf).unwrap();
    let out = buf.read_tensor(&plan, hologram_transform::TensorId(1));
    let r0_sum: f32 = out[0..3].iter().sum();
    let r1_sum: f32 = out[3..6].iter().sum();
    assert!((r0_sum - 1.0).abs() < 1e-5);
    assert!((r1_sum - 1.0).abs() < 1e-5);
    for v in &out[3..6] {
        assert!((v - (1.0 / 3.0)).abs() < 1e-5);
    }
}

#[test]
fn execute_log_softmax_exp_recovers_softmax() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[3], false);
    let out = b.add_tensor(&[3], false);
    b.push_log_softmax(
        3,
        UnaryInputs {
            input: AddressRef::of(inp),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, hologram_transform::TensorId(0), &[1.0, 2.0, 3.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    let out = buf.read_tensor(&plan, hologram_transform::TensorId(1));
    let sum_exp: f32 = out.iter().map(|x| libm::expf(*x)).sum();
    assert!((sum_exp - 1.0).abs() < 1e-5);
}

#[test]
fn execute_reshape_copies_values() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[6], false);
    let out = b.add_tensor(&[2, 3], false);
    b.push_reshape(UnaryInputs {
        input: AddressRef::of(inp),
        output: AddressRef::of(out),
    });
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(
        &plan,
        hologram_transform::TensorId(0),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(
        buf.read_tensor(&plan, hologram_transform::TensorId(1)),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn reshape_output_aliases_input_span() {
    // Sprint 36 Phase 5: Reshape output should share its input's
    // SlotSpan — the kernel's existing offset-equality shortcircuit
    // then makes it a no-op. Workspace is correspondingly smaller.
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[6], false);
    let out = b.add_tensor(&[2, 3], false);
    b.push_reshape(UnaryInputs {
        input: AddressRef::of(inp),
        output: AddressRef::of(out),
    });
    let plan = compile(&b.build()).unwrap();
    let in_span = plan.address_table.span(hologram_transform::TensorId(0));
    let out_span = plan.address_table.span(hologram_transform::TensorId(1));
    assert_eq!(in_span.offset, out_span.offset);
    assert_eq!(in_span.len, out_span.len);
    // Workspace is just the 6-element input; without aliasing it
    // would be 12 (input + output).
    assert_eq!(plan.workspace_elements(), 6);
}

#[test]
fn chained_reshapes_alias_to_single_root() {
    // Reshape → Reshape collapses both outputs onto the original
    // input's span so the workspace allocates one buffer, not three.
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[12], false);
    let mid = b.add_tensor(&[3, 4], false);
    let out = b.add_tensor(&[2, 6], false);
    b.push_reshape(UnaryInputs {
        input: AddressRef::of(inp),
        output: AddressRef::of(mid),
    });
    b.push_reshape(UnaryInputs {
        input: AddressRef::of(mid),
        output: AddressRef::of(out),
    });
    let plan = compile(&b.build()).unwrap();
    let in_span = plan.address_table.span(hologram_transform::TensorId(0));
    let mid_span = plan.address_table.span(hologram_transform::TensorId(1));
    let out_span = plan.address_table.span(hologram_transform::TensorId(2));
    assert_eq!(in_span.offset, mid_span.offset);
    assert_eq!(in_span.offset, out_span.offset);
    assert_eq!(plan.workspace_elements(), 12);
}

#[test]
fn reshape_aliasing_does_not_change_executed_value() {
    // After aliasing, running the reshape kernel still produces the
    // correct values at the output tensor (because the bytes are
    // already there).
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[6], false);
    let out = b.add_tensor(&[2, 3], false);
    b.push_reshape(UnaryInputs {
        input: AddressRef::of(inp),
        output: AddressRef::of(out),
    });
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(
        &plan,
        hologram_transform::TensorId(0),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    Executor::run_forward(&plan, &mut buf).unwrap();
    // Both reads return the same bytes (aliased span).
    assert_eq!(
        buf.read_tensor(&plan, hologram_transform::TensorId(0)),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        buf.read_tensor(&plan, hologram_transform::TensorId(1)),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn push_unary_rejects_non_unary_op() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[2], false);
    let bb = b.add_tensor(&[2], false);
    let err = b
        .push_unary(
            SemanticOp::Add,
            UnaryInputs {
                input: AddressRef::of(a),
                output: AddressRef::of(bb),
            },
        )
        .unwrap_err();
    assert!(matches!(
        err,
        hologram_transform::PlanError::ArityMismatch { op: "add", .. }
    ));
}

#[test]
fn execute_transpose_2d_via_chain() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[2, 3], false);
    let out = b.add_tensor(&[3, 2], false);
    b.push_transpose(
        TransposeAttrs {
            perm: [1, 0, 0, 0, 0, 0, 0, 0],
            ndim: 2,
        },
        UnaryInputs {
            input: AddressRef::of(inp),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(
        buf.read_tensor(&plan, TensorId(1)),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn execute_slice_last_axis_via_chain() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[2, 4], false);
    let out = b.add_tensor(&[2, 2], false);
    b.push_slice(
        SliceAttrs {
            axis_from_end: 0,
            start: 1,
            end: 3,
            axis_size: 4,
        },
        UnaryInputs {
            input: AddressRef::of(inp),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(
        &plan,
        TensorId(0),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(buf.read_tensor(&plan, TensorId(1)), &[2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn execute_concat_last_axis_via_chain() {
    let mut b = TransformChain::builder();
    let a = b.add_tensor(&[2, 2], false);
    let bb = b.add_tensor(&[2, 1], false);
    let c = b.add_tensor(&[2, 3], false);
    b.push_concat(
        ConcatAttrs {
            size_a: 2,
            size_b: 1,
        },
        AddInputs {
            a: AddressRef::of(a),
            b: AddressRef::of(bb),
            c: AddressRef::of(c),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[1.0, 2.0, 3.0, 4.0]);
    buf.write_tensor(&plan, TensorId(1), &[9.0, 8.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(
        buf.read_tensor(&plan, TensorId(2)),
        &[1.0, 2.0, 9.0, 3.0, 4.0, 8.0]
    );
}

#[test]
fn execute_rms_norm_via_chain() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[2], false);
    let weight = b.add_tensor(&[2], false);
    let out = b.add_tensor(&[2], false);
    b.push_rms_norm(
        NormAttrs {
            size: 2,
            epsilon: 0,
        },
        NormScaleInputs {
            input: AddressRef::of(inp),
            weight: AddressRef::of(weight),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[3.0, 4.0]);
    buf.write_tensor(&plan, TensorId(1), &[1.0, 1.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    let scale = 1.0_f32 / libm::sqrtf(12.5);
    let out = buf.read_tensor(&plan, TensorId(2));
    assert!((out[0] - 3.0 * scale).abs() < 1e-5);
    assert!((out[1] - 4.0 * scale).abs() < 1e-5);
}

#[test]
fn execute_layer_norm_via_chain() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[3], false);
    let weight = b.add_tensor(&[3], false);
    let bias = b.add_tensor(&[3], false);
    let out = b.add_tensor(&[3], false);
    b.push_layer_norm(
        NormAttrs {
            size: 3,
            epsilon: 0,
        },
        NormFullInputs {
            input: AddressRef::of(inp),
            weight: AddressRef::of(weight),
            bias: AddressRef::of(bias),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[1.0, 2.0, 3.0]);
    buf.write_tensor(&plan, TensorId(1), &[1.0, 1.0, 1.0]);
    buf.write_tensor(&plan, TensorId(2), &[0.0, 0.0, 0.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    // mean=2, var=2/3, scale=1/sqrt(2/3)
    let scale = 1.0_f32 / libm::sqrtf(2.0 / 3.0);
    let out = buf.read_tensor(&plan, TensorId(3));
    assert!((out[0] - (-scale)).abs() < 1e-5);
    assert!(out[1].abs() < 1e-5);
    assert!((out[2] - scale).abs() < 1e-5);
}

#[test]
fn execute_instance_norm_via_chain() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[3], false);
    let weight = b.add_tensor(&[3], false);
    let out = b.add_tensor(&[3], false);
    b.push_instance_norm(
        NormAttrs {
            size: 3,
            epsilon: 0,
        },
        NormScaleInputs {
            input: AddressRef::of(inp),
            weight: AddressRef::of(weight),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[1.0, 2.0, 3.0]);
    buf.write_tensor(&plan, TensorId(1), &[1.0, 1.0, 1.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    let scale = 1.0_f32 / libm::sqrtf(2.0 / 3.0);
    let out = buf.read_tensor(&plan, TensorId(2));
    assert!((out[0] - (-scale)).abs() < 1e-5);
    assert!(out[1].abs() < 1e-5);
    assert!((out[2] - scale).abs() < 1e-5);
}

#[test]
fn execute_group_norm_via_chain() {
    let mut b = TransformChain::builder();
    let inp = b.add_tensor(&[6], false);
    let weight = b.add_tensor(&[3], false);
    let bias = b.add_tensor(&[3], false);
    let out = b.add_tensor(&[6], false);
    b.push_group_norm(
        GroupNormAttrs {
            num_groups: 2,
            epsilon: 0,
        },
        NormFullInputs {
            input: AddressRef::of(inp),
            weight: AddressRef::of(weight),
            bias: AddressRef::of(bias),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    buf.write_tensor(&plan, TensorId(1), &[1.0, 1.0, 1.0]);
    buf.write_tensor(&plan, TensorId(2), &[0.0, 0.0, 0.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    let out = buf.read_tensor(&plan, TensorId(3));
    let scale = 1.0_f32 / libm::sqrtf(2.0 / 3.0);
    for r in 0..2 {
        assert!((out[r * 3] - (-scale)).abs() < 1e-4);
        assert!(out[r * 3 + 1].abs() < 1e-4);
        assert!((out[r * 3 + 2] - scale).abs() < 1e-4);
    }
}

#[test]
fn execute_add_rms_norm_via_chain() {
    let mut b = TransformChain::builder();
    let res = b.add_tensor(&[2], false);
    let inp = b.add_tensor(&[2], false);
    let weight = b.add_tensor(&[2], false);
    let out = b.add_tensor(&[2], false);
    b.push_add_rms_norm(
        NormAttrs {
            size: 2,
            epsilon: 0,
        },
        AddRmsNormInputs {
            residual: AddressRef::of(res),
            input: AddressRef::of(inp),
            weight: AddressRef::of(weight),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[1.0, 1.0]);
    buf.write_tensor(&plan, TensorId(1), &[2.0, 3.0]);
    buf.write_tensor(&plan, TensorId(2), &[1.0, 1.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    // sum = [3,4]; mean(sum²)=12.5; scale=1/sqrt(12.5)
    let scale = 1.0_f32 / libm::sqrtf(12.5);
    let out = buf.read_tensor(&plan, TensorId(3));
    assert!((out[0] - 3.0 * scale).abs() < 1e-5);
    assert!((out[1] - 4.0 * scale).abs() < 1e-5);
}

#[test]
fn execute_fused_swiglu_via_chain() {
    let mut b = TransformChain::builder();
    let gate = b.add_tensor(&[3], false);
    let up = b.add_tensor(&[3], false);
    let out = b.add_tensor(&[3], false);
    b.push_fused_swiglu(AddInputs {
        a: AddressRef::of(gate),
        b: AddressRef::of(up),
        c: AddressRef::of(out),
    });
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[0.0, 1.0, -1.0]);
    buf.write_tensor(&plan, TensorId(1), &[2.0, 3.0, 4.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    let silu = |x: f32| x / (1.0 + libm::expf(-x));
    let expected = [silu(0.0) * 2.0, silu(1.0) * 3.0, silu(-1.0) * 4.0];
    let out = buf.read_tensor(&plan, TensorId(2));
    for (got, want) in out.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5);
    }
}

#[test]
fn execute_conv2d_identity_via_chain() {
    let mut b = TransformChain::builder();
    let data = b.add_tensor(&[1, 1, 3, 3], false);
    let weight = b.add_tensor(&[1, 1, 1, 1], false);
    let bias = b.add_tensor(&[1], false);
    let out = b.add_tensor(&[1, 1, 3, 3], false);
    b.push_conv2d(
        Conv2dAttrs {
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            group: 1,
            input_h: 3,
            input_w: 3,
        },
        Conv2dInputs {
            data: AddressRef::of(data),
            weight: AddressRef::of(weight),
            bias: AddressRef::of(bias),
            output: AddressRef::of(out),
        },
    );
    let plan = compile(&b.build()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(
        &plan,
        TensorId(0),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    buf.write_tensor(&plan, TensorId(1), &[1.0]);
    buf.write_tensor(&plan, TensorId(2), &[0.0]);
    Executor::run_forward(&plan, &mut buf).unwrap();
    assert_eq!(
        buf.read_tensor(&plan, TensorId(3)),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn executor_does_not_grow_buffer_set() {
    // Capacity is sized once at plan time and never changes during exec.
    let plan = compile(&build_matmul_chain()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    let cap = buf.capacity();
    Executor::run_forward(&plan, &mut buf).unwrap();
    Executor::run_backward(&plan, &mut buf).unwrap();
    assert_eq!(buf.capacity(), cap);
}

// ── Phase 3.5: backend trait + conformance harness over real plans ──

#[test]
fn cpu_backend_round_trip_via_executor_with() {
    use hologram_transform::CpuBackend;

    let plan = compile(&build_matmul_chain()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    buf.write_tensor(&plan, TensorId(1), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut backend = CpuBackend::new();
    Executor::run_forward_with(&plan, &mut buf, &mut backend).unwrap();
    assert_eq!(buf.read_tensor(&plan, TensorId(2)), &[4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn trace_backend_records_real_planner_emitted_calls() {
    use hologram_transform::{CpuBackend, TraceBackend};

    let plan = compile(&build_matmul_chain()).unwrap();
    let mut buf = BufferSet::for_plan(&plan);
    buf.write_tensor(&plan, TensorId(0), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    buf.write_tensor(&plan, TensorId(1), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    buf.write_grad(&plan, TensorId(2), &[1.0, 1.0, 1.0, 1.0]);

    let mut backend = TraceBackend::new(CpuBackend::new());
    Executor::run_forward_with(&plan, &mut buf, &mut backend).unwrap();
    Executor::run_backward_with(&plan, &mut buf, &mut backend).unwrap();

    let names: Vec<&str> = backend.history().iter().map(|e| e.name).collect();
    // MatMul forward → MatMulGradA, MatMulGradB backward (planner-fixed order).
    assert_eq!(names, ["MatMul", "MatMulGradA", "MatMulGradB"]);
}

#[test]
fn conformance_harness_validates_planner_output_on_cpu_vs_cpu() {
    use hologram_transform::{check_forward_then_backward, CpuBackend, Tolerance};

    let plan = compile(&build_matmul_chain()).unwrap();
    let mut reference = CpuBackend::new();
    let mut candidate = CpuBackend::new();

    let res = check_forward_then_backward(
        &plan,
        &mut reference,
        &mut candidate,
        |buf| {
            buf.write_tensor(&plan, TensorId(0), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            buf.write_tensor(&plan, TensorId(1), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
            buf.write_grad(&plan, TensorId(2), &[1.0, 1.0, 1.0, 1.0]);
        },
        Tolerance::TIGHT,
    )
    .unwrap();

    assert!(res.is_match(), "{:?}", res);
}
