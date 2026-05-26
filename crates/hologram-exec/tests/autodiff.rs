//! **Reverse-mode autodiff — finite-difference grad-check V&V.**
//!
//! Gradients are composed from forward ops (chain rule = composition); their
//! correctness is verified the way autograd libraries verify themselves —
//! against the *definition* of the derivative. For each op we compile the
//! backward graph, run it to get the analytic gradient of `sum(output)` w.r.t.
//! the input (seed = ones), and compare element-wise to the central finite
//! difference `(L(x+ε) − L(x−ε)) / 2ε`. Agreement to tolerance proves the VJP
//! pipeline is mathematically correct end-to-end on the real kernels.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, compile_with_backward, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, NodeId, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const F32: u8 = 8;

fn le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn unle(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Build `x:[n] → op(x) → y`, returning the graph + the output node.
fn unary_graph(op: OpKind, n: u64) -> (Graph, NodeId) {
    let mut g = Graph::new();
    let sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(n));
    let x = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(F32),
        output_shape: sh,
    });
    g.add_input(x);
    let y = g.add_node(Node {
        op: GraphOp::Op(op),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(F32),
        output_shape: sh,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(y)]),
        output_dtype: DTypeId(F32),
        output_shape: sh,
    });
    g.add_output(out);
    (g, y)
}

fn run(archive: &[u8], inputs: &[&[f32]]) -> Vec<Vec<f32>> {
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(archive, CpuBackend::new()).unwrap();
    let bufs: Vec<Vec<u8>> = inputs.iter().map(|x| le(x)).collect();
    let ins: Vec<InputBuffer> = bufs.iter().map(|b| InputBuffer { bytes: b }).collect();
    sess.execute(&ins)
        .unwrap()
        .iter()
        .map(|o| unle(&o.bytes))
        .collect()
}

/// Analytic `d sum(op(x)) / dx` via the compiled backward graph (seed = ones).
fn analytic_unary(op: OpKind, x: &[f32]) -> Vec<f32> {
    let (g, y) = unary_graph(op, x.len() as u64);
    let (out, _grads) = compile_with_backward(g, y, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; x.len()];
    // inputs = [x, seed]; outputs = [y, dL/dx, dL/dseed].
    let res = run(&out.archive, &[x, &ones]);
    res[1].clone()
}

/// Central finite-difference `d sum(op(x)) / dx`.
fn numeric_unary(op: OpKind, x: &[f32], eps: f32) -> Vec<f32> {
    let (g0, _) = unary_graph(op, x.len() as u64);
    let fwd = compile(g0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |xv: &[f32]| -> f32 { run(&fwd.archive, &[xv])[0].iter().sum() };
    (0..x.len())
        .map(|j| {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[j] += eps;
            xm[j] -= eps;
            (sum(&xp) - sum(&xm)) / (2.0 * eps)
        })
        .collect()
}

fn check_unary(op: OpKind, x: &[f32], tol: f32) {
    let a = analytic_unary(op, x);
    let nnum = numeric_unary(op, x, 1e-3);
    for (j, (&av, &nv)) in a.iter().zip(&nnum).enumerate() {
        assert!(
            (av - nv).abs() <= tol + tol * nv.abs(),
            "{op:?} grad[{j}]: analytic {av}, numeric {nv}"
        );
    }
}

#[test]
fn unary_activation_gradients_match_finite_difference() {
    // Inputs chosen finite and away from kinks / domain edges.
    let x = [-1.3f32, -0.4, 0.2, 0.9, 1.7, 2.5];
    check_unary(OpKind::Sigmoid, &x, 2e-2);
    check_unary(OpKind::Tanh, &x, 2e-2);
    check_unary(OpKind::Exp, &x, 2e-2);
    check_unary(OpKind::Neg, &x, 1e-3);
    // Relu away from 0 (the kink is non-differentiable).
    check_unary(OpKind::Relu, &[-2.0, -0.7, 0.5, 1.4, 2.2, 3.1], 1e-2);
    // Positive domain for log / sqrt / reciprocal.
    let xp = [0.3f32, 0.8, 1.2, 1.9, 2.6, 3.4];
    check_unary(OpKind::Log, &xp, 2e-2);
    check_unary(OpKind::Sqrt, &xp, 2e-2);
    check_unary(OpKind::Reciprocal, &xp, 2e-2);
    // Composed activations + transcendentals.
    check_unary(OpKind::Silu, &x, 2e-2);
    check_unary(OpKind::Gelu, &x, 2e-2);
    check_unary(OpKind::Elu, &x, 2e-2);
    check_unary(OpKind::Selu, &x, 2e-2);
    check_unary(OpKind::Erf, &x, 2e-2);
    check_unary(OpKind::Abs, &[-2.0, -0.7, 0.5, 1.4, 2.2, 3.1], 1e-2);
    // Trig away from singularities.
    let xt = [-0.9f32, -0.4, 0.1, 0.5, 0.8, 1.1];
    check_unary(OpKind::Sin, &xt, 2e-2);
    check_unary(OpKind::Cos, &xt, 2e-2);
    check_unary(OpKind::Tan, &xt, 3e-2);
    // Log1p (x > −1) and Atan (all reals).
    check_unary(OpKind::Log1p, &xp, 2e-2);
    check_unary(OpKind::Atan, &xt, 2e-2);
    // Asin/Acos on |x| < 1 (away from the ±1 domain edges).
    let xa = [-0.7f32, -0.3, 0.1, 0.4, 0.6, 0.8];
    check_unary(OpKind::Asin, &xa, 3e-2);
    check_unary(OpKind::Acos, &xa, 3e-2);
    // Ceil/Floor/Round: derivative 0 a.e.; inputs away from integer AND
    // half-integer boundaries (where the step is non-differentiable).
    let xr = [-0.83f32, -0.37, 0.22, 0.71, 1.28, 2.43];
    check_unary(OpKind::Ceil, &xr, 1e-3);
    check_unary(OpKind::Floor, &xr, 1e-3);
    check_unary(OpKind::Round, &xr, 1e-3);
    // CumSum: d sum(cumsum(x))/dx_i = (n − i) — composed as total(g)−cumsum(g)+g.
    check_unary(OpKind::CumSum, &xt, 1e-2);
}

#[test]
fn binary_min_max_pow_gradients_match_finite_difference() {
    // Distinct operands (avoid Min/Max ties, which are non-differentiable).
    let a = [1.0f32, 4.0, 2.0, 5.0];
    let b = [3.0f32, 1.0, 6.0, 2.0];
    check_binary(OpKind::Min, &a, &b, 1e-2);
    check_binary(OpKind::Max, &a, &b, 1e-2);
    // Pow needs a positive base for d/db = z·ln(a).
    let pa = [0.5f32, 1.5, 2.0, 0.8];
    let pb = [2.0f32, 0.5, 3.0, 1.5];
    check_binary(OpKind::Pow, &pa, &pb, 3e-2);
    // Floored modulo y = a − floor(a/b)·b: dy/da = 1, dy/db = −floor(a/b).
    // Choose a, b so a/b is away from integer boundaries (where floor jumps).
    let ma = [1.3f32, 4.7, 2.2, 5.6];
    let mb = [3.0f32, 2.0, 6.0, 4.0];
    check_binary(OpKind::Mod, &ma, &mb, 2e-2);
}

#[test]
fn comparison_gradients_are_zero() {
    // Comparisons are step predicates: derivative 0 a.e. (away from equality).
    // Both inputs grad-check to 0 vs finite differences.
    let a = [0.5f32, -1.2, 2.0, 0.3];
    let b = [1.5f32, 0.7, -0.9, 2.1];
    check_binary(OpKind::Greater, &a, &b, 1e-3);
    check_binary(OpKind::Less, &a, &b, 1e-3);
    check_binary(OpKind::Equal, &a, &b, 1e-3);
    // IsNaN on finite inputs: constant 0 output ⇒ 0 gradient.
    check_unary(OpKind::IsNaN, &a, 1e-3);
}

/// Build `x:[n] → reduce(x) → scalar`; grad-check x.
fn check_reduce(op: OpKind, x: &[f32], tol: f32) {
    let n = x.len() as u64;
    let build = || {
        let mut g = Graph::new();
        let sin = g.shape_registry_mut().intern(ShapeDescriptor::rank1(n));
        let scl = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
        let xn = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sin,
        });
        g.add_input(xn);
        let r = g.add_node(Node {
            op: GraphOp::Op(op),
            inputs: SmallVec::from_iter([InputSource::Node(xn)]),
            output_dtype: DTypeId(F32),
            output_shape: scl,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(r)]),
            output_dtype: DTypeId(F32),
            output_shape: scl,
        });
        g.add_output(out);
        (g, r)
    };
    let (g, r) = build();
    let (out, _) = compile_with_backward(g, r, BackendKind::Cpu, WittLevel::W32).unwrap();
    let da = run(&out.archive, &[x, &[1.0f32]])[1].clone();

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |xv: &[f32]| -> f32 { run(&fwd.archive, &[xv])[0].iter().sum() };
    let eps = 1e-3f32;
    for j in 0..x.len() {
        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
        xp[j] += eps;
        xm[j] -= eps;
        let nd = (sum(&xp) - sum(&xm)) / (2.0 * eps);
        assert!(
            (da[j] - nd).abs() <= tol + tol * nd.abs(),
            "{op:?} grad[{j}]: {} vs {nd}",
            da[j]
        );
    }
}

#[test]
fn reduction_gradients_match_finite_difference() {
    let x = [0.7f32, -1.1, 2.3, 0.4, -0.8, 1.5];
    check_reduce(OpKind::ReduceSum, &x, 1e-3);
    check_reduce(OpKind::ReduceMean, &x, 1e-3);
    // Prod: values near 1, nonzero, so the product is well-conditioned.
    check_reduce(OpKind::ReduceProd, &[0.8f32, 1.2, 0.9, 1.1, 1.3, 0.7], 2e-2);
    // Min/Max: distinct values (unique extremum).
    check_reduce(OpKind::ReduceMin, &x, 1e-3);
    check_reduce(OpKind::ReduceMax, &x, 1e-3);
}

/// Grad-check a normalize-then-weight graph: `op(x) ⊙ w`, with `w` a fixed
/// non-uniform constant so the upstream gradient is non-trivial (a plain
/// `sum(softmax)` is constant and would give a degenerate zero gradient).
fn check_weighted(op: OpKind, x: &[f32], rows: u64, cols: u64, tol: f32) {
    use hologram_graph::constant::ConstantEntry;
    let w: Vec<f32> = (0..x.len()).map(|i| 0.3 + 0.2 * (i % 4) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sh = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(rows, cols));
        let xn = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_input(xn);
        let s = g.add_node(Node {
            op: GraphOp::Op(op),
            inputs: SmallVec::from_iter([InputSource::Node(xn)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: w.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: sh,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(s), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_output(out);
        (g, z)
    };
    let (g, z) = build();
    let (out, _) = compile_with_backward(g, z, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; x.len()];
    let da = run(&out.archive, &[x, &ones])[1].clone();

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |xv: &[f32]| -> f32 { run(&fwd.archive, &[xv])[0].iter().sum() };
    let eps = 1e-3f32;
    for j in 0..x.len() {
        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
        xp[j] += eps;
        xm[j] -= eps;
        let nd = (sum(&xp) - sum(&xm)) / (2.0 * eps);
        assert!(
            (da[j] - nd).abs() <= tol + tol * nd.abs(),
            "{op:?} grad[{j}]: {} vs {nd}",
            da[j]
        );
    }
}

/// Grad-check `norm(x, γ=1, β=0) ⊙ w` (w fixed, non-uniform) for the given
/// norm op. `affine` selects 2-input (RmsNorm) vs 3-input (LayerNorm).
fn check_norm(op: OpKind, x: &[f32], rows: u64, cols: u64, affine3: bool, tol: f32) {
    use hologram_graph::constant::ConstantEntry;
    let f = cols as usize;
    let w: Vec<f32> = (0..x.len()).map(|i| 0.4 + 0.25 * (i % 3) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sh = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(rows, cols));
        let fsh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(cols));
        let xn = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_input(xn);
        let gamma = g.constants_mut().insert(ConstantEntry {
            bytes: vec![1.0f32; f]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            dtype: DTypeId(F32),
            shape: fsh,
        });
        let mut ins = SmallVec::<[InputSource; 4]>::new();
        ins.push(InputSource::Node(xn));
        ins.push(InputSource::Constant(gamma));
        if affine3 {
            let beta = g.constants_mut().insert(ConstantEntry {
                bytes: vec![0.0f32; f]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                dtype: DTypeId(F32),
                shape: fsh,
            });
            ins.push(InputSource::Constant(beta));
        }
        let nrm = g.add_node(Node {
            op: GraphOp::Op(op),
            inputs: ins,
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: w.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: sh,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(nrm), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_output(out);
        (g, z)
    };
    let (g, z) = build();
    let (out, _) = compile_with_backward(g, z, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; x.len()];
    let da = run(&out.archive, &[x, &ones])[1].clone();

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |xv: &[f32]| -> f32 { run(&fwd.archive, &[xv])[0].iter().sum() };
    let eps = 1e-3f32;
    for j in 0..x.len() {
        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
        xp[j] += eps;
        xm[j] -= eps;
        let nd = (sum(&xp) - sum(&xm)) / (2.0 * eps);
        assert!(
            (da[j] - nd).abs() <= tol + tol * nd.abs(),
            "{op:?} grad[{j}]: {} vs {nd}",
            da[j]
        );
    }
}

/// Grad-check a grouped norm (GroupNorm/InstanceNorm) on a real `[N,C,H,W]`
/// tensor with a **non-uniform per-channel γ/β** (so the channel-broadcast and
/// per-group statistics are actually exercised). For GroupNorm, `num_groups` is
/// attached via `NormAttrs`; InstanceNorm derives it (= C) at compile time, so
/// `num_groups` is ignored there.
fn check_group_norm(op: OpKind, x: &[f32], n: u64, c: u64, h: u64, w: u64, num_groups: u32) {
    use hologram_graph::constant::ConstantEntry;
    use hologram_graph::NormAttrs;
    let cc = c as usize;
    let wt: Vec<f32> = (0..x.len()).map(|i| 0.4 + 0.25 * (i % 3) as f32).collect();
    let gamma: Vec<f32> = (0..cc).map(|i| 0.7 + 0.3 * i as f32).collect();
    let beta: Vec<f32> = (0..cc).map(|i| 0.1 * i as f32 - 0.2).collect();
    let build = || {
        let mut g = Graph::new();
        let sh = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(n, c, h, w));
        let csh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(c));
        let xn = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_input(xn);
        let gc = g.constants_mut().insert(ConstantEntry {
            bytes: le(&gamma),
            dtype: DTypeId(F32),
            shape: csh,
        });
        let bc = g.constants_mut().insert(ConstantEntry {
            bytes: le(&beta),
            dtype: DTypeId(F32),
            shape: csh,
        });
        let nrm = g.add_node(Node {
            op: GraphOp::Op(op),
            inputs: SmallVec::from_iter([
                InputSource::Node(xn),
                InputSource::Constant(gc),
                InputSource::Constant(bc),
            ]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        if matches!(op, OpKind::GroupNorm) {
            g.set_norm_attrs(nrm, NormAttrs { num_groups });
        }
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: le(&wt),
            dtype: DTypeId(F32),
            shape: sh,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(nrm), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_output(out);
        (g, z)
    };
    let (g, z) = build();
    let (out, _) = compile_with_backward(g, z, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; x.len()];
    let da = run(&out.archive, &[x, &ones])[1].clone();

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |xv: &[f32]| -> f32 { run(&fwd.archive, &[xv])[0].iter().sum() };
    let eps = 1e-3f32;
    let tol = 3e-2f32;
    for j in 0..x.len() {
        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
        xp[j] += eps;
        xm[j] -= eps;
        let nd = (sum(&xp) - sum(&xm)) / (2.0 * eps);
        assert!(
            (da[j] - nd).abs() <= tol + tol * nd.abs(),
            "{op:?} grad[{j}]: {} vs {nd}",
            da[j]
        );
    }
}

#[test]
fn norm_gradients_match_finite_difference() {
    let x = [0.5f32, -1.0, 2.0, 0.3, 1.2, -0.4];
    check_norm(OpKind::RmsNorm, &x, 2, 3, false, 3e-2);
    check_norm(OpKind::LayerNorm, &x, 2, 3, true, 3e-2);
}

#[test]
fn group_norm_gradients_match_finite_difference() {
    // [N=1, C=4, H=2, W=2]: 16 elements, channels=4, spatial=4.
    let x: Vec<f32> = (0..16).map(|i| 0.3 * i as f32 - 1.7).collect();
    // GroupNorm with 2 groups (each group = 2 channels × 4 spatial = 8 elems).
    check_group_norm(OpKind::GroupNorm, &x, 1, 4, 2, 2, 2);
    // GroupNorm with 1 group (= per-sample LayerNorm over all C×spatial).
    check_group_norm(OpKind::GroupNorm, &x, 1, 4, 2, 2, 1);
    // InstanceNorm: num_groups = C = 4 (each channel's 4 spatial elems alone).
    check_group_norm(OpKind::InstanceNorm, &x, 1, 4, 2, 2, 0);
}

/// Grad-check an axis reduction: `sum( reduce_axes(x) ⊙ w )` w.r.t. `x`, with a
/// fixed non-uniform `w` on the reduced (keepdims) output so the broadcast-back
/// gradient is exercised non-trivially.
fn check_axis_reduce(op: OpKind, in_dims: &[u64], out_dims: &[u64], axes_mask: u32, tol: f32) {
    use hologram_graph::constant::ConstantEntry;
    use hologram_graph::ReduceAttrs;
    let to_sd = |d: &[u64]| match d {
        [a, b, c] => ShapeDescriptor::rank3(*a, *b, *c),
        _ => unreachable!("test uses rank-3 shapes"),
    };
    let out_count: usize = out_dims.iter().product::<u64>() as usize;
    let w: Vec<f32> = (0..out_count).map(|i| 0.5 + 0.3 * (i % 4) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let xsh = g.shape_registry_mut().intern(to_sd(in_dims));
        let osh = g.shape_registry_mut().intern(to_sd(out_dims));
        let xn = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: xsh,
        });
        g.add_input(xn);
        let r = g.add_node(Node {
            op: GraphOp::Op(op),
            inputs: SmallVec::from_iter([InputSource::Node(xn)]),
            output_dtype: DTypeId(F32),
            output_shape: osh,
        });
        g.set_reduce_attrs(
            r,
            ReduceAttrs {
                axes_mask,
                keepdims: true,
            },
        );
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: le(&w),
            dtype: DTypeId(F32),
            shape: osh,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(r), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: osh,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: osh,
        });
        g.add_output(out);
        (g, z)
    };
    let n: usize = in_dims.iter().product::<u64>() as usize;
    let x: Vec<f32> = (0..n).map(|i| 0.2 * i as f32 - 1.3).collect();
    let (g, z) = build();
    let (out, _) = compile_with_backward(g, z, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; n];
    let da = run(&out.archive, &[&x, &ones])[1].clone();
    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |xv: &[f32]| -> f32 { run(&fwd.archive, &[xv])[0].iter().sum() };
    let eps = 1e-3f32;
    for j in 0..n {
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[j] += eps;
        xm[j] -= eps;
        let nd = (sum(&xp) - sum(&xm)) / (2.0 * eps);
        assert!(
            (da[j] - nd).abs() <= tol + tol * nd.abs(),
            "{op:?} mask={axes_mask:#b} grad[{j}]: {} vs {nd}",
            da[j]
        );
    }
}

#[test]
fn axis_reduce_gradients_match_finite_difference() {
    // [2,3,4] reducing single axes and a pair (keepdims), grad-checked.
    check_axis_reduce(OpKind::ReduceSum, &[2, 3, 4], &[2, 1, 4], 0b010, 2e-2);
    check_axis_reduce(OpKind::ReduceSum, &[2, 3, 4], &[1, 3, 4], 0b001, 2e-2);
    check_axis_reduce(OpKind::ReduceMean, &[2, 3, 4], &[2, 3, 1], 0b100, 2e-2);
    check_axis_reduce(OpKind::ReduceMean, &[2, 3, 4], &[2, 1, 1], 0b110, 2e-2);
}

#[test]
fn softmax_gradients_match_finite_difference() {
    // [2,3] — softmax/logsoftmax over the last axis.
    let x = [0.5f32, -1.0, 2.0, 0.3, 1.2, -0.4];
    check_weighted(OpKind::Softmax, &x, 2, 3, 2e-2);
    check_weighted(OpKind::LogSoftmax, &x, 2, 3, 2e-2);
}

fn i64c(g: &mut Graph, vals: &[i64]) -> InputSource {
    use hologram_graph::constant::ConstantEntry;
    let sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(vals.len() as u64));
    let cid = g.constants_mut().insert(ConstantEntry {
        bytes: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        dtype: DTypeId(5),
        shape: sh,
    });
    InputSource::Constant(cid)
}

#[test]
fn slice_pad_concat_gradients_are_correct_scatter() {
    // Slice x:[4,2] → rows[1:3] ; dL/dx scatters the gradient back (rows 1,2=1).
    {
        let mut g = Graph::new();
        let sin = g.shape_registry_mut().intern(ShapeDescriptor::rank2(4, 2));
        let sout = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 2));
        let x = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sin,
        });
        g.add_input(x);
        let (s, e) = (i64c(&mut g, &[1]), i64c(&mut g, &[3]));
        let sl = g.add_node(Node {
            op: GraphOp::Op(OpKind::Slice),
            inputs: SmallVec::from_iter([InputSource::Node(x), s, e]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(sl)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        g.add_output(o);
        let (out, _) = compile_with_backward(g, sl, BackendKind::Cpu, WittLevel::W32).unwrap();
        let da = run(&out.archive, &[&[0.0f32; 8], &[1.0f32; 4]])[1].clone();
        assert_eq!(
            da,
            vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "slice grad scatter"
        );
    }
    // Pad x:[2,2] → [pad 1 before, 1 after] → [4,2] ; dL/dx = g middle rows = 1.
    {
        let mut g = Graph::new();
        let sin = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 2));
        let sout = g.shape_registry_mut().intern(ShapeDescriptor::rank2(4, 2));
        let x = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sin,
        });
        g.add_input(x);
        let pads = i64c(&mut g, &[1, 0, 1, 0]);
        let pd = g.add_node(Node {
            op: GraphOp::Op(OpKind::Pad),
            inputs: SmallVec::from_iter([InputSource::Node(x), pads]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(pd)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        g.add_output(o);
        let (out, _) = compile_with_backward(g, pd, BackendKind::Cpu, WittLevel::W32).unwrap();
        let da = run(&out.archive, &[&[0.0f32; 4], &[1.0f32; 8]])[1].clone();
        assert_eq!(da, vec![1.0; 4], "pad grad = unpadded slice");
    }
    // Concat a:[2,2] ∥ b:[1,2] → [3,2] ; dL/da = g[0:2], dL/db = g[2:3].
    {
        let mut g = Graph::new();
        let sa = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 2));
        let sb = g.shape_registry_mut().intern(ShapeDescriptor::rank2(1, 2));
        let sc = g.shape_registry_mut().intern(ShapeDescriptor::rank2(3, 2));
        let a = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sa,
        });
        g.add_input(a);
        let b = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sb,
        });
        g.add_input(b);
        let cc = g.add_node(Node {
            op: GraphOp::Op(OpKind::Concat),
            inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
            output_dtype: DTypeId(F32),
            output_shape: sc,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(cc)]),
            output_dtype: DTypeId(F32),
            output_shape: sc,
        });
        g.add_output(o);
        let (out, _) = compile_with_backward(g, cc, BackendKind::Cpu, WittLevel::W32).unwrap();
        // inputs [a, b, seed]; outputs [y, dL/da, dL/db].
        let res = run(&out.archive, &[&[0.0f32; 4], &[0.0f32; 2], &[1.0f32; 6]]);
        assert_eq!(res[1], vec![1.0; 4], "concat dA");
        assert_eq!(res[2], vec![1.0; 2], "concat dB");
    }
}

#[test]
fn expand_gradient_matches_finite_difference() {
    // x:[2,1] → Expand → [2,3]; d sum(expand)/dx = #replicas = 3.
    let a = [1.5f32, -0.7];
    let build = || {
        let mut g = Graph::new();
        let sin = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 1));
        let sout = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 3));
        let x = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sin,
        });
        g.add_input(x);
        let e = g.add_node(Node {
            op: GraphOp::Op(OpKind::Expand),
            inputs: SmallVec::from_iter([InputSource::Node(x)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(e)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        g.add_output(out);
        (g, e)
    };
    let (g, e) = build();
    let (out, _) = compile_with_backward(g, e, BackendKind::Cpu, WittLevel::W32).unwrap();
    let da = run(&out.archive, &[&a, &[1.0f32; 6]])[1].clone();
    for (j, &v) in da.iter().enumerate() {
        assert!((v - 3.0).abs() < 1e-4, "expand dA[{j}] = {v}, want 3");
    }
}

#[test]
fn global_avg_pool_gradient_matches_finite_difference() {
    // x:[1,2,2,2] → GlobalAvgPool → [1,2]; dx = 1/(H·W) = 0.25.
    let a: Vec<f32> = (0..8).map(|i| 0.3 * i as f32 - 1.0).collect();
    let build = || {
        let mut g = Graph::new();
        let sin = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(1, 2, 2, 2));
        let sout = g.shape_registry_mut().intern(ShapeDescriptor::rank2(1, 2));
        let x = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sin,
        });
        g.add_input(x);
        let p = g.add_node(Node {
            op: GraphOp::Op(OpKind::GlobalAvgPool),
            inputs: SmallVec::from_iter([InputSource::Node(x)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(p)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        g.add_output(out);
        (g, p)
    };
    let (g, p) = build();
    let (out, _) = compile_with_backward(g, p, BackendKind::Cpu, WittLevel::W32).unwrap();
    let da = run(&out.archive, &[&a, &[1.0f32; 2]])[1].clone();
    for (j, &v) in da.iter().enumerate() {
        assert!(
            (v - 0.25).abs() < 1e-4,
            "globalavgpool dA[{j}] = {v}, want 0.25"
        );
    }
}

/// Grad-check a 2-D pooling op `pool(x) ⊙ w` with non-overlapping windows.
/// `x` is `[1,1,Hᵢ,Wᵢ]`, the window is `kh×kw` (stride = window), output
/// `[1,1,Hᵢ/kh,Wᵢ/kw]`. A fixed non-uniform `w` keeps the upstream gradient
/// non-trivial.
fn check_pool(op: OpKind, x: &[f32], hi: u64, wi: u64, kh: u64, kw: u64, tol: f32) {
    use hologram_graph::constant::ConstantEntry;
    use hologram_graph::node::ConvAttrs;
    let (ho, wo) = (hi / kh, wi / kw);
    let wlen = (ho * wo) as usize;
    let w: Vec<f32> = (0..wlen).map(|i| 0.3 + 0.2 * (i % 4) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sin = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(1, 1, hi, wi));
        let sout = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(1, 1, ho, wo));
        let so2 = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(1, ho * wo));
        let x = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sin,
        });
        g.add_input(x);
        let p = g.add_node(Node {
            op: GraphOp::Op(op),
            inputs: SmallVec::from_iter([InputSource::Node(x)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        // The pool kernel size / stride live in ConvAttrs (stride = window).
        g.set_conv_attrs(
            p,
            ConvAttrs {
                stride_h: kh as u32,
                stride_w: kw as u32,
                ..Default::default()
            },
        );
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: w.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: so2,
        });
        // Reshape pooled [1,1,Ho,Wo] → [1,Ho·Wo] to weight it.
        let pr = g.add_node(Node {
            op: GraphOp::Op(OpKind::Reshape),
            inputs: SmallVec::from_iter([InputSource::Node(p)]),
            output_dtype: DTypeId(F32),
            output_shape: so2,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(pr), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: so2,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: so2,
        });
        g.add_output(out);
        (g, z)
    };
    let (g, z) = build();
    let (out, _) = compile_with_backward(g, z, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; wlen];
    let da = run(&out.archive, &[x, &ones])[1].clone();

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |xv: &[f32]| -> f32 { run(&fwd.archive, &[xv])[0].iter().sum() };
    let eps = 1e-3f32;
    for j in 0..x.len() {
        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
        xp[j] += eps;
        xm[j] -= eps;
        let nd = (sum(&xp) - sum(&xm)) / (2.0 * eps);
        assert!(
            (da[j] - nd).abs() <= tol + tol * nd.abs(),
            "{op:?} grad[{j}]: {} vs {nd}",
            da[j]
        );
    }
}

#[test]
fn pool_gradients_match_finite_difference() {
    // 4×4 input, 2×2 non-overlapping windows → 2×2 output.
    // AvgPool: gradient is uniform 1/(kh·kw) within a window.
    let xa: Vec<f32> = (0..16).map(|i| 0.2 * i as f32 - 1.5).collect();
    check_pool(OpKind::AvgPool2d, &xa, 4, 4, 2, 2, 2e-2);
    // MaxPool: distinct values so each window has a unique maximum (the mask
    // routes the gradient to exactly one element).
    let xm: Vec<f32> = (0..16)
        .map(|i| (((i * 7 + 3) % 16) as f32) * 0.13 - 0.4)
        .collect();
    check_pool(OpKind::MaxPool2d, &xm, 4, 4, 2, 2, 2e-2);
}

/// Generic finite-difference grad-check for a multi-input graph. `build`
/// returns `(graph, z)` where `z` is the (weighted) output node; `ins` are the
/// differentiable inputs in graph-input order. Each input's analytic gradient
/// (from the composed backward graph, seed = ones) is compared to the central
/// finite difference of `sum(z)`.
fn gradcheck(build: impl Fn() -> (Graph, NodeId), ins: &[&[f32]], tol: f32) {
    let (g, z) = build();
    let (out, _) = compile_with_backward(g, z, BackendKind::Cpu, WittLevel::W32).unwrap();
    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let outlen = run(&fwd.archive, ins)[0].len();
    let seed = vec![1.0f32; outlen];
    let mut args: Vec<&[f32]> = ins.to_vec();
    args.push(&seed);
    let res = run(&out.archive, &args);
    let sumfn = |a: &[Vec<f32>]| -> f32 {
        let r: Vec<&[f32]> = a.iter().map(|v| v.as_slice()).collect();
        run(&fwd.archive, &r)[0].iter().sum()
    };
    let eps = 1e-3f32;
    for wi in 0..ins.len() {
        let analytic = &res[1 + wi];
        for j in 0..ins[wi].len() {
            let mut pert: Vec<Vec<f32>> = ins.iter().map(|s| s.to_vec()).collect();
            pert[wi][j] += eps;
            let hi = sumfn(&pert);
            pert[wi][j] -= 2.0 * eps;
            let lo = sumfn(&pert);
            let nd = (hi - lo) / (2.0 * eps);
            assert!(
                (analytic[j] - nd).abs() <= tol + tol * nd.abs(),
                "grad in[{wi}][{j}]: {} vs {nd}",
                analytic[j]
            );
        }
    }
}

fn fnode(g: &mut Graph, sh: hologram_graph::registry::ShapeId) -> NodeId {
    let id = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(F32),
        output_shape: sh,
    });
    g.add_input(id);
    id
}

#[test]
fn gemm_gradient_matches_finite_difference() {
    // Y = A·B + C (α=β=1) ; A:[2,3] B:[3,2] C:[2,2], weighted. Grad-check A,B,C.
    use hologram_graph::constant::ConstantEntry;
    let a = [0.2f32, -0.5, 0.8, 1.1, -0.3, 0.6];
    let b = [0.7f32, -0.2, 0.4, 0.9, -0.6, 0.1];
    let c = [0.3f32, -0.4, 0.5, 0.2];
    let wt: Vec<f32> = (0..4).map(|i| 0.3 + 0.2 * i as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sa = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 3));
        let sb = g.shape_registry_mut().intern(ShapeDescriptor::rank2(3, 2));
        let sc = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 2));
        let (an, bn, cn) = (fnode(&mut g, sa), fnode(&mut g, sb), fnode(&mut g, sc));
        let y = g.add_node(Node {
            op: GraphOp::Op(OpKind::Gemm),
            inputs: SmallVec::from_iter([
                InputSource::Node(an),
                InputSource::Node(bn),
                InputSource::Node(cn),
            ]),
            output_dtype: DTypeId(F32),
            output_shape: sc,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wt.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: sc,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(y), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: sc,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: sc,
        });
        g.add_output(o);
        (g, z)
    };
    gradcheck(build, &[&a, &b, &c], 2e-2);
}

#[test]
fn conv_transpose_2d_gradient_matches_finite_difference() {
    // ConvTranspose2d shares the conv forward kernel, so its VJP = the conv VJP.
    use hologram_graph::constant::ConstantEntry;
    let (b, cin, hin, win, cout, kh, kw) = (1u64, 2u64, 4u64, 4u64, 2u64, 2u64, 2u64);
    let (hout, wout) = (hin - kh + 1, win - kw + 1);
    let xv: Vec<f32> = (0..(b * cin * hin * win) as usize)
        .map(|i| 0.1 * ((i % 7) as f32) - 0.3)
        .collect();
    let wv: Vec<f32> = (0..(cout * cin * kh * kw) as usize)
        .map(|i| 0.12 * ((i % 5) as f32) - 0.2)
        .collect();
    let on_ = (b * cout * hout * wout) as usize;
    let wt: Vec<f32> = (0..on_).map(|i| 0.3 + 0.2 * (i % 4) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sx = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(b, cin, hin, win));
        let sw = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(cout, cin, kh, kw));
        let so = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(b, cout, hout, wout));
        let (xn, wn) = (fnode(&mut g, sx), fnode(&mut g, sw));
        let conv = g.add_node(Node {
            op: GraphOp::Op(OpKind::ConvTranspose2d),
            inputs: SmallVec::from_iter([InputSource::Node(xn), InputSource::Node(wn)]),
            output_dtype: DTypeId(F32),
            output_shape: so,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wt.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: so,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(conv), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: so,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: so,
        });
        g.add_output(o);
        (g, z)
    };
    gradcheck(build, &[&xv, &wv], 2e-2);
}

#[test]
fn add_rms_norm_gradient_matches_finite_difference() {
    // y = rmsnorm(x + residual)·γ (γ=1), weighted. Grad-check x and residual.
    use hologram_graph::constant::ConstantEntry;
    let x = [0.5f32, -1.0, 2.0, 0.3, 1.2, -0.4];
    let r = [0.2f32, 0.7, -0.5, 1.1, -0.8, 0.4];
    let wt: Vec<f32> = (0..6).map(|i| 0.4 + 0.25 * (i % 3) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sh = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 3));
        let fsh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(3));
        let xn = fnode(&mut g, sh);
        let gamma = g.constants_mut().insert(ConstantEntry {
            bytes: [1.0f32; 3].iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: fsh,
        });
        let rn = fnode(&mut g, sh);
        let nrm = g.add_node(Node {
            op: GraphOp::Op(OpKind::AddRmsNorm),
            inputs: SmallVec::from_iter([
                InputSource::Node(xn),
                InputSource::Constant(gamma),
                InputSource::Node(rn),
            ]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wt.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: sh,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(nrm), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_output(o);
        (g, z)
    };
    gradcheck(build, &[&x, &r], 3e-2);
}

#[test]
fn lrn_gradient_matches_finite_difference() {
    use hologram_graph::constant::ConstantEntry;
    use hologram_graph::node::LrnAttrs;
    // x:[1,4,1,1] (4 channels, inner=1), size=3, α=1, β=0.75, bias=1, weighted.
    let x = [0.7f32, -1.1, 1.8, 0.4];
    let wt = [0.5f32, 0.3, 0.7, 0.4];
    let build = || {
        let mut g = Graph::new();
        let s = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(1, 4, 1, 1));
        let xn = fnode(&mut g, s);
        let lrn = g.add_node(Node {
            op: GraphOp::Op(OpKind::Lrn),
            inputs: SmallVec::from_iter([InputSource::Node(xn)]),
            output_dtype: DTypeId(F32),
            output_shape: s,
        });
        g.set_lrn_attrs(
            lrn,
            LrnAttrs {
                size: 3,
                alpha_bits: 1.0f32.to_bits(),
                beta_bits: 0.75f32.to_bits(),
                bias_bits: 1.0f32.to_bits(),
            },
        );
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wt.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: s,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(lrn), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: s,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: s,
        });
        g.add_output(o);
        (g, z)
    };
    gradcheck(build, &[&x], 3e-2);
}

#[test]
fn resize_gradient_matches_finite_difference() {
    // Nearest resize x:[1,1,2,2] → [1,1,4,4], weighted. dx_i = Σ of grads of the
    // outputs that selected input i (each input maps to a 2×2 output block).
    use hologram_graph::constant::ConstantEntry;
    let x = [1.0f32, -2.0, 0.5, 3.0];
    let wt: Vec<f32> = (0..16).map(|i| 0.2 + 0.1 * (i % 5) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sin = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(1, 1, 2, 2));
        let sout = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(1, 1, 4, 4));
        let xn = fnode(&mut g, sin);
        let rz = g.add_node(Node {
            op: GraphOp::Op(OpKind::Resize),
            inputs: SmallVec::from_iter([InputSource::Node(xn)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wt.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: sout,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(rz), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        g.add_output(o);
        (g, z)
    };
    gradcheck(build, &[&x], 1e-2);
}

#[test]
fn rotary_embedding_gradient_matches_finite_difference() {
    // RoPE x:[2,4] (seq=2, head_dim=4) with fixed cos/sin tables, weighted.
    // RoPE is a linear rotation, so dx = RoPE(g, cos, −sin); grad-check x.
    use hologram_graph::constant::ConstantEntry;
    let x = [0.3f32, -0.7, 1.1, 0.5, -0.2, 0.9, 0.4, -1.0];
    // Arbitrary but fixed cos/sin (need not satisfy cos²+sin²=1 for the check).
    let cos = [0.8f32, 0.6, 0.9, 0.5, 0.7, 0.95, 0.85, 0.55];
    let sin = [0.6f32, 0.8, 0.44, 0.87, 0.71, 0.31, 0.53, 0.84];
    let wt: Vec<f32> = (0..8).map(|i| 0.3 + 0.15 * (i % 4) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sh = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 4));
        let xn = fnode(&mut g, sh);
        let mkc = |g: &mut Graph, v: &[f32]| {
            let cid = g.constants_mut().insert(ConstantEntry {
                bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                dtype: DTypeId(F32),
                shape: sh,
            });
            InputSource::Constant(cid)
        };
        let cc = mkc(&mut g, &cos);
        let sc = mkc(&mut g, &sin);
        let rope = g.add_node(Node {
            op: GraphOp::Op(OpKind::RotaryEmbedding),
            inputs: SmallVec::from_iter([InputSource::Node(xn), cc, sc]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wt.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: sh,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(rope), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_output(o);
        (g, z)
    };
    gradcheck(build, &[&x], 2e-2);
}

#[test]
fn where_gradient_matches_finite_difference() {
    // out = (cond≠0) ? a : b ; cond a fixed 0/1 mask. da=g·[cond≠0], db=g·[cond=0].
    use hologram_graph::constant::ConstantEntry;
    let a = [0.5f32, -1.2, 2.0, 0.3, 1.1, -0.6];
    let b = [1.5f32, 0.7, -0.9, 2.1, -0.4, 0.8];
    // cond is a BOOL selector: one byte per element (the Where kernel reads it
    // as raw bytes, not floats).
    let cond: [u8; 6] = [1, 0, 1, 0, 1, 0];
    let wt: Vec<f32> = (0..6).map(|i| 0.3 + 0.2 * (i % 3) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(6));
        let cc = g.constants_mut().insert(ConstantEntry {
            bytes: cond.to_vec(),
            dtype: DTypeId(0), // BOOL
            shape: sh,
        });
        let (an, bn) = (fnode(&mut g, sh), fnode(&mut g, sh));
        let w = g.add_node(Node {
            op: GraphOp::Op(OpKind::Where),
            inputs: SmallVec::from_iter([
                InputSource::Constant(cc),
                InputSource::Node(an),
                InputSource::Node(bn),
            ]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wt.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: sh,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(w), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_output(o);
        (g, z)
    };
    gradcheck(build, &[&a, &b], 1e-2);
}

#[test]
fn conv2d_gradient_matches_finite_difference() {
    // Valid 2-D convolution X:[1,2,4,4] * W:[3,2,2,2] (stride 1) → [1,3,3,3],
    // weighted by a fixed non-uniform constant. Both dX and dW are grad-checked
    // against central finite differences — the whole im2col/matmul/col2im
    // composition end-to-end on the real kernels.
    use hologram_graph::constant::ConstantEntry;
    let (b, cin, hin, win) = (1u64, 2u64, 4u64, 4u64);
    let (cout, kh, kw) = (3u64, 2u64, 2u64);
    let (hout, wout) = (hin - kh + 1, win - kw + 1);
    let xn_ = (b * cin * hin * win) as usize;
    let wn_ = (cout * cin * kh * kw) as usize;
    let on_ = (b * cout * hout * wout) as usize;
    let xv: Vec<f32> = (0..xn_).map(|i| 0.1 * ((i % 9) as f32) - 0.4).collect();
    let wv: Vec<f32> = (0..wn_).map(|i| 0.15 * ((i % 5) as f32) - 0.3).collect();
    let wt: Vec<f32> = (0..on_).map(|i| 0.3 + 0.2 * (i % 4) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let sx = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(b, cin, hin, win));
        let sw = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(cout, cin, kh, kw));
        let so = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(b, cout, hout, wout));
        let xnode = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sx,
        });
        g.add_input(xnode);
        let wnode = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sw,
        });
        g.add_input(wnode);
        let conv = g.add_node(Node {
            op: GraphOp::Op(OpKind::Conv2d),
            inputs: SmallVec::from_iter([InputSource::Node(xnode), InputSource::Node(wnode)]),
            output_dtype: DTypeId(F32),
            output_shape: so,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wt.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: so,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(conv), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: so,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: so,
        });
        g.add_output(out);
        (g, z)
    };
    let (g, z) = build();
    let (out, _) = compile_with_backward(g, z, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; on_];
    // inputs [X, W, seed]; outputs [z, dX, dW, dseed].
    let res = run(&out.archive, &[&xv, &wv, &ones]);
    let (dx, dw) = (res[1].clone(), res[2].clone());

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |xx: &[f32], ww: &[f32]| -> f32 { run(&fwd.archive, &[xx, ww])[0].iter().sum() };
    let eps = 1e-3f32;
    let tol = 2e-2f32;
    for j in 0..xn_ {
        let (mut xp, mut xm) = (xv.clone(), xv.clone());
        xp[j] += eps;
        xm[j] -= eps;
        let nd = (sum(&xp, &wv) - sum(&xm, &wv)) / (2.0 * eps);
        assert!(
            (dx[j] - nd).abs() <= tol + tol * nd.abs(),
            "conv dX[{j}]: {} vs {nd}",
            dx[j]
        );
    }
    for j in 0..wn_ {
        let (mut wp, mut wm) = (wv.clone(), wv.clone());
        wp[j] += eps;
        wm[j] -= eps;
        let nd = (sum(&xv, &wp) - sum(&xv, &wm)) / (2.0 * eps);
        assert!(
            (dw[j] - nd).abs() <= tol + tol * nd.abs(),
            "conv dW[{j}]: {} vs {nd}",
            dw[j]
        );
    }
}

#[test]
fn attention_gradient_matches_finite_difference() {
    // Scaled dot-product attention O = softmax(Q·Kᵀ/√d)·V, weighted by a fixed
    // non-uniform constant so the upstream gradient is non-trivial. Inputs
    // Q,K,V : [B,H,S,D] = [1,1,2,3]. The whole pipeline (incl. the unrolled
    // per-head VJP) is grad-checked against central finite differences.
    use hologram_graph::constant::ConstantEntry;
    let (b, h, s, dpv) = (1u64, 1u64, 2u64, 3u64);
    let n = (b * h * s * dpv) as usize;
    // Concatenate Q‖K‖V as one input so a single finite-diff harness perturbs
    // all three operands.
    let qkv: Vec<f32> = (0..3 * n).map(|i| 0.2 * ((i % 7) as f32) - 0.6).collect();
    let w: Vec<f32> = (0..n).map(|i| 0.3 + 0.15 * (i % 4) as f32).collect();
    let build = || {
        let mut g = Graph::new();
        let s4 = g
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank4(b, h, s, dpv));
        let inp = |g: &mut Graph| {
            let id = g.add_node(Node {
                op: GraphOp::Input,
                inputs: SmallVec::new(),
                output_dtype: DTypeId(F32),
                output_shape: s4,
            });
            g.add_input(id);
            id
        };
        let (qn, kn, vn) = (inp(&mut g), inp(&mut g), inp(&mut g));
        let att = g.add_node(Node {
            op: GraphOp::Op(OpKind::Attention),
            inputs: SmallVec::from_iter([
                InputSource::Node(qn),
                InputSource::Node(kn),
                InputSource::Node(vn),
            ]),
            output_dtype: DTypeId(F32),
            output_shape: s4,
        });
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: w.iter().flat_map(|v| v.to_le_bytes()).collect(),
            dtype: DTypeId(F32),
            shape: s4,
        });
        let z = g.add_node(Node {
            op: GraphOp::Op(OpKind::Mul),
            inputs: SmallVec::from_iter([InputSource::Node(att), InputSource::Constant(wc)]),
            output_dtype: DTypeId(F32),
            output_shape: s4,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(z)]),
            output_dtype: DTypeId(F32),
            output_shape: s4,
        });
        g.add_output(out);
        (g, z)
    };
    let (g, z) = build();
    let (out, _) = compile_with_backward(g, z, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; n];
    let (q0, k0, v0) = (&qkv[..n], &qkv[n..2 * n], &qkv[2 * n..]);
    // inputs [Q,K,V,seed]; outputs [z, dQ, dK, dV, dseed].
    let res = run(&out.archive, &[q0, k0, v0, &ones]);
    let (dq, dk, dv) = (res[1].clone(), res[2].clone(), res[3].clone());

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |qv: &[f32], kv: &[f32], vv: &[f32]| -> f32 {
        run(&fwd.archive, &[qv, kv, vv])[0].iter().sum()
    };
    let eps = 1e-3f32;
    let tol = 3e-2f32;
    for (which, analytic) in [(0usize, &dq), (1, &dk), (2, &dv)] {
        for j in 0..n {
            let mut pert = [q0.to_vec(), k0.to_vec(), v0.to_vec()];
            pert[which][j] += eps;
            let hi = sum(&pert[0], &pert[1], &pert[2]);
            pert[which][j] -= 2.0 * eps;
            let lo = sum(&pert[0], &pert[1], &pert[2]);
            let nd = (hi - lo) / (2.0 * eps);
            assert!(
                (analytic[j] - nd).abs() <= tol + tol * nd.abs(),
                "attention d[{which}][{j}]: {} vs {nd}",
                analytic[j]
            );
        }
    }
}

#[test]
fn transpose_gradient_matches_finite_difference() {
    // A:[2,3] → Transpose → [3,2]; grad-check A.
    let (r, c) = (2u64, 3u64);
    let a = [0.2f32, -0.5, 0.8, 1.1, -0.3, 0.6];
    let build = || {
        let mut g = Graph::new();
        let sin = g.shape_registry_mut().intern(ShapeDescriptor::rank2(r, c));
        let sout = g.shape_registry_mut().intern(ShapeDescriptor::rank2(c, r));
        let x = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sin,
        });
        g.add_input(x);
        let t = g.add_node(Node {
            op: GraphOp::Op(OpKind::Transpose),
            inputs: SmallVec::from_iter([InputSource::Node(x)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(t)]),
            output_dtype: DTypeId(F32),
            output_shape: sout,
        });
        g.add_output(out);
        (g, t)
    };
    let (g, t) = build();
    let (out, _) = compile_with_backward(g, t, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; a.len()];
    let da = run(&out.archive, &[&a, &ones])[1].clone();
    // d sum(transpose(A))/dA = 1 everywhere (transpose just permutes).
    for (j, &v) in da.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-5, "transpose dA[{j}] = {v}, want 1");
    }
}

/// Build `a,b:[n] → op(a,b) → y`; grad-check both inputs.
fn check_binary(op: OpKind, a: &[f32], b: &[f32], tol: f32) {
    let n = a.len() as u64;
    let build = || {
        let mut g = Graph::new();
        let sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(n));
        let an = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_input(an);
        let bn = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_input(bn);
        let y = g.add_node(Node {
            op: GraphOp::Op(op),
            inputs: SmallVec::from_iter([InputSource::Node(an), InputSource::Node(bn)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(y)]),
            output_dtype: DTypeId(F32),
            output_shape: sh,
        });
        g.add_output(out);
        (g, y)
    };
    let (g, y) = build();
    let (out, _) = compile_with_backward(g, y, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; a.len()];
    // inputs [a,b,seed]; outputs [y, dL/da, dL/db, dL/dseed].
    let res = run(&out.archive, &[a, b, &ones]);
    let (da, db) = (res[1].clone(), res[2].clone());

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |av: &[f32], bv: &[f32]| -> f32 { run(&fwd.archive, &[av, bv])[0].iter().sum() };
    let eps = 1e-3f32;
    for j in 0..a.len() {
        let (mut ap, mut am) = (a.to_vec(), a.to_vec());
        ap[j] += eps;
        am[j] -= eps;
        let n_da = (sum(&ap, b) - sum(&am, b)) / (2.0 * eps);
        assert!(
            (da[j] - n_da).abs() <= tol + tol * n_da.abs(),
            "{op:?} d/da[{j}]: {} vs {n_da}",
            da[j]
        );
        let (mut bp, mut bm) = (b.to_vec(), b.to_vec());
        bp[j] += eps;
        bm[j] -= eps;
        let n_db = (sum(a, &bp) - sum(a, &bm)) / (2.0 * eps);
        assert!(
            (db[j] - n_db).abs() <= tol + tol * n_db.abs(),
            "{op:?} d/db[{j}]: {} vs {n_db}",
            db[j]
        );
    }
}

#[test]
fn binary_arithmetic_gradients_match_finite_difference() {
    let a = [0.5f32, -1.2, 2.0, 0.3];
    let b = [1.5f32, 0.7, -0.9, 2.1];
    check_binary(OpKind::Add, &a, &b, 1e-3);
    check_binary(OpKind::Sub, &a, &b, 1e-3);
    check_binary(OpKind::Mul, &a, &b, 1e-2);
    check_binary(OpKind::Div, &a, &b, 2e-2);
}

#[test]
fn matmul_gradient_matches_finite_difference() {
    // A:[2,3] · B:[3,2] = C:[2,2]; grad-check A.
    let (m, k, n) = (2u64, 3u64, 2u64);
    let a = [0.2f32, -0.5, 0.8, 1.1, -0.3, 0.6];
    let b = [0.7f32, -0.2, 0.4, 0.9, -0.6, 0.1];
    let build = || {
        let mut g = Graph::new();
        let sa = g.shape_registry_mut().intern(ShapeDescriptor::rank2(m, k));
        let sb = g.shape_registry_mut().intern(ShapeDescriptor::rank2(k, n));
        let sc = g.shape_registry_mut().intern(ShapeDescriptor::rank2(m, n));
        let an = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sa,
        });
        g.add_input(an);
        let bn = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(F32),
            output_shape: sb,
        });
        g.add_input(bn);
        let c = g.add_node(Node {
            op: GraphOp::Op(OpKind::MatMul),
            inputs: SmallVec::from_iter([InputSource::Node(an), InputSource::Node(bn)]),
            output_dtype: DTypeId(F32),
            output_shape: sc,
        });
        let out = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(c)]),
            output_dtype: DTypeId(F32),
            output_shape: sc,
        });
        g.add_output(out);
        (g, c)
    };
    let (g, c) = build();
    let (out, _) = compile_with_backward(g, c, BackendKind::Cpu, WittLevel::W32).unwrap();
    let ones = vec![1.0f32; (m * n) as usize];
    let res = run(&out.archive, &[&a, &b, &ones]);
    let da = res[1].clone(); // dL/dA, shape [m,k]

    let fwd = compile(build().0, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sum = |av: &[f32]| -> f32 { run(&fwd.archive, &[av, &b])[0].iter().sum() };
    let eps = 1e-3f32;
    for j in 0..a.len() {
        let (mut ap, mut am) = (a.to_vec(), a.to_vec());
        ap[j] += eps;
        am[j] -= eps;
        let n_da = (sum(&ap) - sum(&am)) / (2.0 * eps);
        assert!(
            (da[j] - n_da).abs() <= 1e-2 + 1e-2 * n_da.abs(),
            "matmul dA[{j}]: {} vs {n_da}",
            da[j]
        );
    }
}
