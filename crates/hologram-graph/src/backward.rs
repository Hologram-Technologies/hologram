//! Reverse-mode autodiff by **composition** (spec V.4 / ADR-043).
//!
//! Per the UOR framework, a gradient is not a new primitive — it is a
//! *composition* of the forward primitives the value was decomposed into.
//! The chain rule **is** categorical composition: the vector-Jacobian product
//! (VJP) of each forward op is itself a small pipeline of forward ops, and
//! `append_backward` composes those VJPs in reverse-topological order,
//! summing contributions where a value fans out.
//!
//! Consequences of building backward purely from forward ops:
//!   * No new "backward kernels" exist — gradients run on the already-verified
//!     forward kernels, so there is no second silent-wrong surface.
//!   * Content addressing, elision, and warm-start apply to the backward graph
//!     for free (it is just more nodes).
//!   * Correctness is checked the way autograd libraries check themselves:
//!     finite-difference grad-checking against the definition of the
//!     derivative (see `hologram-exec/tests/autodiff.rs`).
//!
//! An op whose VJP has not yet been composed fails loud
//! ([`BackwardError::NoGradient`]) — it is never silently approximated.

use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::constant::ConstantEntry;
use crate::node::{ConvAttrs, Node};
use crate::registry::{DTypeId, ShapeDescriptor, ShapeId};
use crate::{Graph, GraphOp, InputSource, NodeId};
use hologram_ops::OpKind;

/// f32 dtype tag (the dtype VJP constant fills are emitted in).
const F32: u8 = 8;

/// Errors that can arise during backward emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackwardError {
    /// The named output node is missing from the graph.
    OutputMissing(NodeId),
    /// The op's vector-Jacobian product has not been composed yet. The op is
    /// real and differentiable, but its backward pipeline is not implemented —
    /// reported explicitly rather than emitting an approximate gradient.
    NoGradient(OpKind),
}

/// Per-input VJP contributions: `(forward_input_node, gradient_node)` pairs.
type Contribs = SmallVec<[(NodeId, NodeId); 2]>;

fn meta(graph: &Graph, src: InputSource) -> (DTypeId, ShapeId) {
    match src {
        InputSource::Node(id) => graph
            .get(id)
            .map(|n| (n.output_dtype, n.output_shape))
            .unwrap_or((DTypeId(F32), ShapeId(0))),
        InputSource::Constant(cid) => graph
            .constants()
            .get(cid)
            .map(|e| (e.dtype, e.shape))
            .unwrap_or((DTypeId(F32), ShapeId(0))),
        InputSource::GraphInput(_) => (DTypeId(F32), ShapeId(0)),
    }
}

fn add_op(
    graph: &mut Graph,
    op: OpKind,
    inputs: &[InputSource],
    dt: DTypeId,
    sh: ShapeId,
) -> NodeId {
    graph.add_node(Node {
        op: GraphOp::Op(op),
        inputs: SmallVec::from_iter(inputs.iter().copied()),
        output_dtype: dt,
        output_shape: sh,
    })
}

/// A same-shape constant tensor filled with `v` (f32), as a `Constant` node —
/// used by VJPs that need an identity element (`1−y`, `2·y`, the `x>0` mask).
fn const_fill(graph: &mut Graph, sh: ShapeId, v: f32) -> NodeId {
    let count = graph
        .shape_registry()
        .get(sh)
        .map(|d| d.total_elements())
        .unwrap_or(1)
        .max(1) as usize;
    let mut bytes = Vec::with_capacity(count * 4);
    for _ in 0..count {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let cid = graph.constants_mut().insert(ConstantEntry {
        bytes,
        dtype: DTypeId(F32),
        shape: sh,
    });
    graph.add_node(Node {
        op: GraphOp::Constant(cid),
        inputs: SmallVec::new(),
        output_dtype: DTypeId(F32),
        output_shape: sh,
    })
}

/// Intern the rank-2 transpose of `sh` (swap the two axes), or `None` if `sh`
/// is not rank-2.
fn transpose2(graph: &mut Graph, sh: ShapeId) -> Option<ShapeId> {
    let d = graph.shape_registry().get(sh)?.clone();
    if d.rank != 2 {
        return None;
    }
    let t = ShapeDescriptor::rank2(d.dim(1)?, d.dim(0)?);
    Some(graph.shape_registry_mut().intern(t))
}

/// Intern an all-ones shape of the given rank (the keepdims-form of a full
/// reduction's scalar output), so a scalar gradient can be `Reshape`d to it
/// and then `Expand`ed back to the input shape.
fn ones_shape(graph: &mut Graph, rank: usize) -> ShapeId {
    let mut dims = [0u64; 8];
    for d in dims.iter_mut().take(rank.max(1)) {
        *d = 1;
    }
    graph.shape_registry_mut().intern(ShapeDescriptor {
        rank: rank.max(1) as u8,
        dims,
        dims_overflow: None,
    })
}

/// `Reshape(g)`→ones-rank then `Expand` it to `ash`: broadcast a scalar
/// reduction gradient back over the input shape. Returns the expanded node.
fn broadcast_scalar(graph: &mut Graph, g: InputSource, adt: DTypeId, ash: ShapeId) -> NodeId {
    let rank = graph
        .shape_registry()
        .get(ash)
        .map(|d| d.rank as usize)
        .unwrap_or(1);
    let os = ones_shape(graph, rank);
    let gr = add_op(graph, OpKind::Reshape, &[g], adt, os);
    add_op(graph, OpKind::Expand, &[InputSource::Node(gr)], adt, ash)
}

/// Sum `m` over its **last axis** and broadcast the result back over the full
/// shape `sh` — the per-row reduction softmax/norm backward needs. A per-axis
/// reduction is a linear contraction, so it *is* a matmul with a ones-vector:
/// `Reshape(m)→[B,F] · ones[F,1] → [B,1]`, then `Expand → [B,F]`, reshaped to
/// `sh`. Composed entirely from forward ops (no new reduction primitive).
fn row_reduce_broadcast(
    graph: &mut Graph,
    m: InputSource,
    dt: DTypeId,
    sh: ShapeId,
) -> Option<NodeId> {
    let d = graph.shape_registry().get(sh)?.clone();
    let f = d.dim(d.rank as usize - 1)?;
    let total = d.total_elements();
    if f == 0 {
        return None;
    }
    let b = total / f;
    let bf = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(b, f));
    let b1 = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(b, 1));
    let f1 = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(f, 1));
    let ones = InputSource::Node(const_fill(graph, f1, 1.0));
    let m2 = InputSource::Node(add_op(graph, OpKind::Reshape, &[m], dt, bf));
    let s = InputSource::Node(add_op(graph, OpKind::MatMul, &[m2, ones], dt, b1));
    let sb = InputSource::Node(add_op(graph, OpKind::Expand, &[s], dt, bf));
    Some(add_op(graph, OpKind::Reshape, &[sb], dt, sh))
}

/// Broadcast a per-feature vector `v` (shape `[F]` or `[1,F]`) across the
/// batch to `[B,F]` matching `sh = [B,F]`: `Reshape(v)→[1,F]` then `Expand`.
fn broadcast_feature(
    graph: &mut Graph,
    v: InputSource,
    dt: DTypeId,
    sh: ShapeId,
) -> Option<NodeId> {
    let d = graph.shape_registry().get(sh)?.clone();
    if d.rank != 2 {
        return None;
    }
    let (b, f) = (d.dim(0)?, d.dim(1)?);
    let onef = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, f));
    let bf = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(b, f));
    let vr = InputSource::Node(add_op(graph, OpKind::Reshape, &[v], dt, onef));
    Some(add_op(graph, OpKind::Expand, &[vr], dt, bf))
}

/// Per-row **mean** over the last axis, broadcast back over `sh`
/// (`row_reduce_broadcast` / F).
fn row_mean_broadcast(
    graph: &mut Graph,
    m: InputSource,
    dt: DTypeId,
    sh: ShapeId,
) -> Option<NodeId> {
    let f = graph
        .shape_registry()
        .get(sh)?
        .dim(graph.shape_registry().get(sh)?.rank as usize - 1)?;
    let s = row_reduce_broadcast(graph, m, dt, sh)?;
    let invf = InputSource::Node(const_fill(graph, sh, 1.0 / f as f32));
    Some(add_op(
        graph,
        OpKind::Mul,
        &[InputSource::Node(s), invf],
        dt,
        sh,
    ))
}

/// Build an i64 rank-1 constant (Slice index / Pad width operands), returned
/// as an `InputSource::Constant`.
fn i64_const(graph: &mut Graph, vals: &[i64]) -> InputSource {
    let mut bytes = Vec::with_capacity(vals.len() * 8);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(vals.len() as u64));
    let cid = graph.constants_mut().insert(ConstantEntry {
        bytes,
        dtype: DTypeId(5), // I64
        shape: sh,
    });
    InputSource::Constant(cid)
}

/// Read the leading i64 of a constant operand (a Slice start/end index).
fn read_i64_const(graph: &Graph, src: InputSource) -> Option<i64> {
    let cid = match src {
        InputSource::Constant(c) => c,
        _ => return None,
    };
    let e = graph.constants().get(cid)?;
    e.bytes
        .get(0..8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
}

fn dim0(graph: &Graph, sh: ShapeId) -> Option<i64> {
    graph.shape_registry().get(sh)?.dim(0).map(|d| d as i64)
}
fn rank_of(graph: &Graph, sh: ShapeId) -> usize {
    graph
        .shape_registry()
        .get(sh)
        .map(|d| d.rank as usize)
        .unwrap_or(1)
}
/// Intern a rank-2 `[r, c]` shape (free helper so callers can keep using
/// `graph` directly afterwards — unlike a `&mut graph`-capturing closure).
fn intern2(graph: &mut Graph, r: u64, c: u64) -> ShapeId {
    graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(r, c))
}

/// Upsample a pooled tensor `m` of shape `[N,C,Hₒ,Wₒ]` back to the input grid
/// `[N,C,Hᵢ,Wᵢ]` by replicating each output cell across its `kh×kw` window —
/// the pooling backward scatter, expressed as pure layout ops:
/// `Reshape → [N,C,Hₒ,1,Wₒ,1]`, `Expand → [N,C,Hₒ,kh,Wₒ,kw]`, `Reshape →
/// [N,C,Hᵢ,Wᵢ]`. The final reshape is exact because the flat index
/// `((hₒ·kh+ih)·Wₒ+wₒ)·kw+iw = (hₒ·kh+ih)·Wᵢ + (wₒ·kw+iw)`. Returns `(node, kh,
/// kw)`, or `None` for overlapping / padded pooling (`Hᵢ` not a multiple of
/// `Hₒ`) — those need the general scatter and fail loud rather than approximate.
fn upsample_pool(
    graph: &mut Graph,
    m: InputSource,
    dt: DTypeId,
    pooled: ShapeId,
    input: ShapeId,
) -> Option<(NodeId, u64, u64)> {
    let od = graph.shape_registry().get(pooled)?.clone();
    let idz = graph.shape_registry().get(input)?.clone();
    if od.rank != 4 || idz.rank != 4 {
        return None;
    }
    let (n, c, ho, wo) = (od.dim(0)?, od.dim(1)?, od.dim(2)?, od.dim(3)?);
    let (ni, ci, hi, wi) = (idz.dim(0)?, idz.dim(1)?, idz.dim(2)?, idz.dim(3)?);
    if n != ni || c != ci || ho == 0 || wo == 0 || hi % ho != 0 || wi % wo != 0 {
        return None;
    }
    let (kh, kw) = (hi / ho, wi / wo);
    let split = ShapeDescriptor {
        rank: 6,
        dims: [n, c, ho, 1, wo, 1, 0, 0],
        dims_overflow: None,
    };
    let full = ShapeDescriptor {
        rank: 6,
        dims: [n, c, ho, kh, wo, kw, 0, 0],
        dims_overflow: None,
    };
    let s_split = graph.shape_registry_mut().intern(split);
    let s_full = graph.shape_registry_mut().intern(full);
    let r = InputSource::Node(add_op(graph, OpKind::Reshape, &[m], dt, s_split));
    let e = InputSource::Node(add_op(graph, OpKind::Expand, &[r], dt, s_full));
    Some((add_op(graph, OpKind::Reshape, &[e], dt, input), kh, kw))
}

/// Whether the pool at `id` tiles its input with **non-overlapping** windows
/// (stride = kernel), the only case the `upsample_pool` scatter is exact for.
/// Requires explicit `ConvAttrs` matching the derived kernel — absent or
/// mismatched strides (overlapping / dilated pooling) fail loud upstream.
fn pool_strides_match(graph: &Graph, id: NodeId, kh: u64, kw: u64) -> bool {
    match graph.conv_attrs(id) {
        Some(at) => at.stride_h as u64 == kh && at.stride_w as u64 == kw,
        None => false,
    }
}

/// VJP of scaled dot-product attention `O[b,h] = softmax(Q·Kᵀ/√d)·V`
/// (`cpu::float_kernels::attention_float`: scale `√d`, softmax over keys, no
/// causal mask), composed **entirely from existing rank-2 ops**. Because the
/// batch×head count is static, each head is an independent rank-2 problem: we
/// `Reshape`+`Slice` out `Qₚ,Kₚ,Vₚ,gₚ : [S,D]`, recompute `Pₚ = softmax`, apply
/// the 2-D matmul / softmax chain rule, then `Concat` the per-head gradients
/// back. No batched-matmul primitive — the verified 2-D MatMul/Softmax/
/// Transpose kernels are reused per head, so no new silent-wrong surface.
///
/// Returns `(dQ, dK, dV)`, each shaped like the `[B,H,S,D]` inputs.
fn attention_vjp(
    graph: &mut Graph,
    q: InputSource,
    k: InputSource,
    v: InputSource,
    g: InputSource,
    qsh: ShapeId,
    dt: DTypeId,
) -> Option<(NodeId, NodeId, NodeId)> {
    let d = graph.shape_registry().get(qsh)?.clone();
    if d.rank != 4 {
        return None;
    }
    let (bb, h, s, dd) = (d.dim(0)?, d.dim(1)?, d.dim(2)?, d.dim(3)?);
    let bh = bb * h;
    let sd = s * dd;
    if bh == 0 || sd == 0 || s == 0 || dd == 0 {
        return None;
    }
    let invscale = 1.0 / libm::sqrtf(dd as f32).max(1.0);
    let mut i2 = |r: u64, c: u64| {
        graph
            .shape_registry_mut()
            .intern(ShapeDescriptor::rank2(r, c))
    };
    let r2 = i2(bh, sd);
    let sdsh = i2(s, dd);
    let dssh = i2(dd, s);
    let sssh = i2(s, s);
    let one_sd = i2(1, sd);

    let qr = InputSource::Node(add_op(graph, OpKind::Reshape, &[q], dt, r2));
    let kr = InputSource::Node(add_op(graph, OpKind::Reshape, &[k], dt, r2));
    let vr = InputSource::Node(add_op(graph, OpKind::Reshape, &[v], dt, r2));
    let gr = InputSource::Node(add_op(graph, OpKind::Reshape, &[g], dt, r2));

    // Per-head: extract row p of the [bh, sd] matrices, reshape to [S,D].
    let head = |graph: &mut Graph, m: InputSource, p: i64| -> InputSource {
        let (st, en) = (i64_const(graph, &[p]), i64_const(graph, &[p + 1]));
        let row = InputSource::Node(add_op(graph, OpKind::Slice, &[m, st, en], dt, one_sd));
        InputSource::Node(add_op(graph, OpKind::Reshape, &[row], dt, sdsh))
    };

    let mut dq_acc: Option<(NodeId, u64)> = None;
    let mut dk_acc: Option<(NodeId, u64)> = None;
    let mut dv_acc: Option<(NodeId, u64)> = None;
    // Concat piece `[1,sd]` onto an accumulator of `[count,sd]` → `[count+1,sd]`.
    let append = |graph: &mut Graph, acc: &mut Option<(NodeId, u64)>, piece: NodeId| match *acc {
        None => *acc = Some((piece, 1)),
        Some((prev, cnt)) => {
            let sh = graph
                .shape_registry_mut()
                .intern(ShapeDescriptor::rank2(cnt + 1, sd));
            let c = add_op(
                graph,
                OpKind::Concat,
                &[InputSource::Node(prev), InputSource::Node(piece)],
                dt,
                sh,
            );
            *acc = Some((c, cnt + 1));
        }
    };

    for p in 0..bh as i64 {
        let qp = head(graph, qr, p);
        let kp = head(graph, kr, p);
        let vp = head(graph, vr, p);
        let gp = head(graph, gr, p);
        // Forward recompute Pₚ = softmax(Qₚ·Kₚᵀ·invscale).
        let kpt = InputSource::Node(add_op(graph, OpKind::Transpose, &[kp], dt, dssh));
        let scores = InputSource::Node(add_op(graph, OpKind::MatMul, &[qp, kpt], dt, sssh));
        let sc = InputSource::Node(const_fill(graph, sssh, invscale));
        let scaled = InputSource::Node(add_op(graph, OpKind::Mul, &[scores, sc], dt, sssh));
        let pmat = InputSource::Node(add_op(graph, OpKind::Softmax, &[scaled], dt, sssh));
        // dVₚ = Pₚᵀ·gₚ.
        let pt = InputSource::Node(add_op(graph, OpKind::Transpose, &[pmat], dt, sssh));
        let dvp = add_op(graph, OpKind::MatMul, &[pt, gp], dt, sdsh);
        // dPₚ = gₚ·Vₚᵀ.
        let vpt = InputSource::Node(add_op(graph, OpKind::Transpose, &[vp], dt, dssh));
        let dpmat = InputSource::Node(add_op(graph, OpKind::MatMul, &[gp, vpt], dt, sssh));
        // dScoresₚ = Pₚ ⊙ (dPₚ − rowsum(Pₚ⊙dPₚ)) ; then ·invscale.
        let pdp = InputSource::Node(add_op(graph, OpKind::Mul, &[pmat, dpmat], dt, sssh));
        let rs = InputSource::Node(row_reduce_broadcast(graph, pdp, dt, sssh)?);
        let diff = InputSource::Node(add_op(graph, OpKind::Sub, &[dpmat, rs], dt, sssh));
        let dscore = InputSource::Node(add_op(graph, OpKind::Mul, &[pmat, diff], dt, sssh));
        let sc2 = InputSource::Node(const_fill(graph, sssh, invscale));
        let dscaled = InputSource::Node(add_op(graph, OpKind::Mul, &[dscore, sc2], dt, sssh));
        // dQₚ = dScaledₚ·Kₚ ; dKₚ = dScaledₚᵀ·Qₚ.
        let dqp = add_op(graph, OpKind::MatMul, &[dscaled, kp], dt, sdsh);
        let dst = InputSource::Node(add_op(graph, OpKind::Transpose, &[dscaled], dt, sssh));
        let dkp = add_op(graph, OpKind::MatMul, &[dst, qp], dt, sdsh);
        // Reshape each [S,D] head gradient to [1,sd] and concat.
        let dqp1 = add_op(
            graph,
            OpKind::Reshape,
            &[InputSource::Node(dqp)],
            dt,
            one_sd,
        );
        let dkp1 = add_op(
            graph,
            OpKind::Reshape,
            &[InputSource::Node(dkp)],
            dt,
            one_sd,
        );
        let dvp1 = add_op(
            graph,
            OpKind::Reshape,
            &[InputSource::Node(dvp)],
            dt,
            one_sd,
        );
        append(graph, &mut dq_acc, dqp1);
        append(graph, &mut dk_acc, dkp1);
        append(graph, &mut dv_acc, dvp1);
    }

    let finish = |graph: &mut Graph, acc: Option<(NodeId, u64)>| -> NodeId {
        let (n, _) = acc.expect("bh ≥ 1");
        add_op(graph, OpKind::Reshape, &[InputSource::Node(n)], dt, qsh)
    };
    let dq = finish(graph, dq_acc);
    let dk = finish(graph, dk_acc);
    let dv = finish(graph, dv_acc);
    Some((dq, dk, dv))
}

/// VJP of valid 2-D convolution `O = conv(X, W, stride)` (the runtime's
/// `conv2d_float`: valid — no padding — im2col + per-batch GEMM), composed from
/// the verified MatMul VJP plus the `Im2Col`/`Col2Im` layout primitives. A
/// convolution *is* `W·im2col(X)` per batch, so its gradients are matmul
/// gradients on the patch matrix: `dW = Σ_b gᵦ·im2col(xᵦ)ᵀ` (patch-space weight
/// gradient) and `dX = Σ_b col2im(Wᵀ·gᵦ)` (scatter the input gradient).
/// Batches are unrolled (B is static), reusing the rank-2 kernels — no batched
/// primitive, no backward kernel. Returns `(dX, dW)`.
#[allow(clippy::too_many_arguments)]
fn conv2d_vjp(
    graph: &mut Graph,
    x: InputSource,
    w: InputSource,
    g: InputSource,
    xsh: ShapeId,
    wsh: ShapeId,
    gsh: ShapeId,
    stride_h: u32,
    stride_w: u32,
    dt: DTypeId,
) -> Option<(NodeId, NodeId)> {
    let (xd, wd, gd) = (
        graph.shape_registry().get(xsh)?.clone(),
        graph.shape_registry().get(wsh)?.clone(),
        graph.shape_registry().get(gsh)?.clone(),
    );
    if xd.rank != 4 || wd.rank != 4 || gd.rank != 4 {
        return None;
    }
    let (bb, cin, hin, win) = (xd.dim(0)?, xd.dim(1)?, xd.dim(2)?, xd.dim(3)?);
    let (cout, kh, kw) = (wd.dim(0)?, wd.dim(2)?, wd.dim(3)?);
    let (hout, wout) = (gd.dim(2)?, gd.dim(3)?);
    let nn = hout * wout;
    let kk = cin * kh * kw;
    let chw = cin * hin * win;
    if bb == 0 || nn == 0 || kk == 0 {
        return None;
    }
    let x_flat = intern2(graph, bb, chw);
    let g_flat = intern2(graph, bb, cout * nn);
    let cout_n = intern2(graph, cout, nn);
    let one_cn = intern2(graph, 1, cout * nn);
    let kn = intern2(graph, kk, nn);
    let nk = intern2(graph, nn, kk);
    let k_cout = intern2(graph, kk, cout);
    let cout_k = intern2(graph, cout, kk);
    let img3 = graph.shape_registry_mut().intern(ShapeDescriptor {
        rank: 3,
        dims: [cin, hin, win, 0, 0, 0, 0, 0],
        dims_overflow: None,
    });
    let one_chw = intern2(graph, 1, chw);

    let attrs = ConvAttrs {
        stride_h,
        stride_w,
        pad_h: 0,
        pad_w: 0,
        k_h: kh as u32,
        k_w: kw as u32,
    };
    let patch = |graph: &mut Graph, op: OpKind, inp: InputSource, out_shape: ShapeId| -> NodeId {
        let id = add_op(graph, op, &[inp], dt, out_shape);
        graph.set_conv_attrs(id, attrs);
        id
    };

    let xr = InputSource::Node(add_op(graph, OpKind::Reshape, &[x], dt, x_flat));
    let gr = InputSource::Node(add_op(graph, OpKind::Reshape, &[g], dt, g_flat));
    let wr = InputSource::Node(add_op(graph, OpKind::Reshape, &[w], dt, cout_k));
    let wt = InputSource::Node(add_op(graph, OpKind::Transpose, &[wr], dt, k_cout));

    let mut dw_acc: Option<NodeId> = None;
    let mut dx_acc: Option<(NodeId, u64)> = None;
    for b in 0..bb as i64 {
        let (st, en) = (i64_const(graph, &[b]), i64_const(graph, &[b + 1]));
        let xb_row = InputSource::Node(add_op(graph, OpKind::Slice, &[xr, st, en], dt, one_chw));
        let xb = InputSource::Node(add_op(graph, OpKind::Reshape, &[xb_row], dt, img3));
        let col = InputSource::Node(patch(graph, OpKind::Im2Col, xb, kn));
        let (gst, gen) = (i64_const(graph, &[b]), i64_const(graph, &[b + 1]));
        let gb_row = InputSource::Node(add_op(graph, OpKind::Slice, &[gr, gst, gen], dt, one_cn));
        let gb = InputSource::Node(add_op(graph, OpKind::Reshape, &[gb_row], dt, cout_n));
        // dWᵦ = gᵦ[Cout,N] · colᵀ[N,K].
        let colt = InputSource::Node(add_op(graph, OpKind::Transpose, &[col], dt, nk));
        let dwb = add_op(graph, OpKind::MatMul, &[gb, colt], dt, cout_k);
        dw_acc = Some(match dw_acc {
            None => dwb,
            Some(prev) => add_op(
                graph,
                OpKind::Add,
                &[InputSource::Node(prev), InputSource::Node(dwb)],
                dt,
                cout_k,
            ),
        });
        // dXᵦ = col2im(Wᵀ[K,Cout] · gᵦ[Cout,N]).
        let pre = InputSource::Node(add_op(graph, OpKind::MatMul, &[wt, gb], dt, kn));
        let dxb_img = InputSource::Node(patch(graph, OpKind::Col2Im, pre, img3));
        let dxb_row = add_op(graph, OpKind::Reshape, &[dxb_img], dt, one_chw);
        match dx_acc {
            None => dx_acc = Some((dxb_row, 1)),
            Some((prev, cnt)) => {
                let sh = graph
                    .shape_registry_mut()
                    .intern(ShapeDescriptor::rank2(cnt + 1, chw));
                let c = add_op(
                    graph,
                    OpKind::Concat,
                    &[InputSource::Node(prev), InputSource::Node(dxb_row)],
                    dt,
                    sh,
                );
                dx_acc = Some((c, cnt + 1));
            }
        }
    }
    let dw = add_op(
        graph,
        OpKind::Reshape,
        &[InputSource::Node(dw_acc?)],
        dt,
        wsh,
    );
    let (dx_flat, _) = dx_acc?;
    let dx = add_op(
        graph,
        OpKind::Reshape,
        &[InputSource::Node(dx_flat)],
        dt,
        xsh,
    );
    Some((dx, dw))
}

/// A rank-aware f32 `Constant` node from explicit little-endian bytes (a
/// non-uniform tensor, unlike `const_fill`). Used for the selection / band
/// matrices that express a gather-adjoint (Resize) or windowed-channel sum
/// (Lrn) as a matmul.
fn const_tensor(graph: &mut Graph, sh: ShapeId, bytes: Vec<u8>) -> NodeId {
    let cid = graph.constants_mut().insert(ConstantEntry {
        bytes,
        dtype: DTypeId(F32),
        shape: sh,
    });
    graph.add_node(Node {
        op: GraphOp::Constant(cid),
        inputs: SmallVec::new(),
        output_dtype: DTypeId(F32),
        output_shape: sh,
    })
}

/// Apply a `[C,C]` matrix `m` along the channel axis of a `[batch,C,inner]`
/// tensor `t`: per batch, `m · t_b` where `t_b` is the `[C,inner]` slice. The
/// batch is unrolled (static) into rank-2 matmuls and the results concatenated
/// — the building block for Lrn's windowed-channel sums. Returns `[batch,C,inner]`.
#[allow(clippy::too_many_arguments)]
fn apply_channel_matrix(
    graph: &mut Graph,
    m: InputSource,
    t: InputSource,
    batch: u64,
    ch: u64,
    inner: u64,
    dt: DTypeId,
    tsh: ShapeId,
) -> NodeId {
    let ci = intern2(graph, ch, inner);
    let flat = intern2(graph, batch, ch * inner);
    let one_ci = intern2(graph, 1, ch * inner);
    let tr = InputSource::Node(add_op(graph, OpKind::Reshape, &[t], dt, flat));
    let mut acc: Option<(NodeId, u64)> = None;
    for b in 0..batch as i64 {
        let (st, en) = (i64_const(graph, &[b]), i64_const(graph, &[b + 1]));
        let row = InputSource::Node(add_op(graph, OpKind::Slice, &[tr, st, en], dt, one_ci));
        let mat = InputSource::Node(add_op(graph, OpKind::Reshape, &[row], dt, ci));
        let prod = InputSource::Node(add_op(graph, OpKind::MatMul, &[m, mat], dt, ci));
        let prow = add_op(graph, OpKind::Reshape, &[prod], dt, one_ci);
        match acc {
            None => acc = Some((prow, 1)),
            Some((prev, cnt)) => {
                let sh = intern2(graph, cnt + 1, ch * inner);
                let c = add_op(
                    graph,
                    OpKind::Concat,
                    &[InputSource::Node(prev), InputSource::Node(prow)],
                    dt,
                    sh,
                );
                acc = Some((c, cnt + 1));
            }
        }
    }
    let (flatn, _) = acc.expect("batch ≥ 1");
    add_op(graph, OpKind::Reshape, &[InputSource::Node(flatn)], dt, tsh)
}

/// `dx` of `y = x·invrms·γ`, `invrms = 1/√(mean(x²)+ε)` (ε≈0): the rank-2
/// RmsNorm input gradient `γ·invrms·g − (invrms³·x/F)·Σ_row(g·γ·x)`, composed
/// from forward ops. Shared by `RmsNorm` and `AddRmsNorm` (the latter passes
/// `x := x+residual`).
fn rmsnorm_dx(
    graph: &mut Graph,
    x: InputSource,
    gamma: InputSource,
    g: InputSource,
    dt: DTypeId,
    sh: ShapeId,
) -> Option<NodeId> {
    if graph.shape_registry().get(sh).map(|d| d.rank) != Some(2) {
        return None;
    }
    let gb = InputSource::Node(broadcast_feature(graph, gamma, dt, sh)?);
    let x2 = InputSource::Node(add_op(graph, OpKind::Mul, &[x, x], dt, sh));
    let meansq = InputSource::Node(row_mean_broadcast(graph, x2, dt, sh)?);
    let rms = InputSource::Node(add_op(graph, OpKind::Sqrt, &[meansq], dt, sh));
    let invrms = InputSource::Node(add_op(graph, OpKind::Reciprocal, &[rms], dt, sh));
    let gi = InputSource::Node(add_op(graph, OpKind::Mul, &[gb, invrms], dt, sh));
    let term1 = InputSource::Node(add_op(graph, OpKind::Mul, &[gi, g], dt, sh));
    let ggam = InputSource::Node(add_op(graph, OpKind::Mul, &[g, gb], dt, sh));
    let ggx = InputSource::Node(add_op(graph, OpKind::Mul, &[ggam, x], dt, sh));
    let s = InputSource::Node(row_reduce_broadcast(graph, ggx, dt, sh)?);
    let f = graph
        .shape_registry()
        .get(sh)
        .and_then(|d| d.dim(1))
        .unwrap_or(1);
    let invf = InputSource::Node(const_fill(graph, sh, 1.0 / f as f32));
    let ir2 = InputSource::Node(add_op(graph, OpKind::Mul, &[invrms, invrms], dt, sh));
    let ir3 = InputSource::Node(add_op(graph, OpKind::Mul, &[ir2, invrms], dt, sh));
    let ir3x = InputSource::Node(add_op(graph, OpKind::Mul, &[ir3, x], dt, sh));
    let ir3xs = InputSource::Node(add_op(graph, OpKind::Mul, &[ir3x, s], dt, sh));
    let term2 = InputSource::Node(add_op(graph, OpKind::Mul, &[ir3xs, invf], dt, sh));
    Some(add_op(graph, OpKind::Sub, &[term1, term2], dt, sh))
}

/// Emit the VJP of one forward op: given the upstream gradient node `g`
/// (`dL/d(output)`), produce a gradient node for each `Node` input.
fn emit_vjp(
    graph: &mut Graph,
    kind: OpKind,
    node: &Node,
    fwd_id: NodeId,
    g: NodeId,
) -> Result<Contribs, BackwardError> {
    use OpKind as K;
    let gsrc = InputSource::Node(g);
    let y = InputSource::Node(fwd_id); // forward output value
    let (dt, sh) = (node.output_dtype, node.output_shape);
    let ins = &node.inputs;
    // Convenience: record a contribution only for `Node` inputs (constants and
    // graph-input ports are leaves with no upstream to accumulate into).
    let node_of = |s: InputSource| match s {
        InputSource::Node(id) => Some(id),
        _ => None,
    };
    let mut out: Contribs = SmallVec::new();
    // f32-only guard for VJPs that synthesize constant fills.
    let needs_f32 = || dt.0 == F32;

    match kind {
        K::Add => {
            // dL/da = dL/db = g (passthrough).
            if let Some(a) = node_of(ins[0]) {
                out.push((a, g));
            }
            if ins.len() > 1 {
                if let Some(b) = node_of(ins[1]) {
                    out.push((b, g));
                }
            }
        }
        K::Sub => {
            if let Some(a) = node_of(ins[0]) {
                out.push((a, g));
            }
            if let Some(b) = node_of(ins[1]) {
                let ng = add_op(graph, K::Neg, &[gsrc], dt, sh);
                out.push((b, ng));
            }
        }
        K::Mul => {
            // d/da = g·b ; d/db = g·a.
            if let Some(a) = node_of(ins[0]) {
                let da = add_op(graph, K::Mul, &[gsrc, ins[1]], dt, sh);
                out.push((a, da));
            }
            if let Some(b) = node_of(ins[1]) {
                let db = add_op(graph, K::Mul, &[gsrc, ins[0]], dt, sh);
                out.push((b, db));
            }
        }
        K::Div => {
            // z = a/b ; d/da = g/b ; d/db = −g·a/b².
            if let Some(a) = node_of(ins[0]) {
                let da = add_op(graph, K::Div, &[gsrc, ins[1]], dt, sh);
                out.push((a, da));
            }
            if let Some(b) = node_of(ins[1]) {
                let ga = add_op(graph, K::Mul, &[gsrc, ins[0]], dt, sh);
                let bb = add_op(graph, K::Mul, &[ins[1], ins[1]], dt, sh);
                let q = add_op(
                    graph,
                    K::Div,
                    &[InputSource::Node(ga), InputSource::Node(bb)],
                    dt,
                    sh,
                );
                let db = add_op(graph, K::Neg, &[InputSource::Node(q)], dt, sh);
                out.push((b, db));
            }
        }
        K::Neg => {
            if let Some(a) = node_of(ins[0]) {
                let da = add_op(graph, K::Neg, &[gsrc], dt, sh);
                out.push((a, da));
            }
        }
        K::Relu => {
            // d/dx = g · [x > 0].
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let zero = const_fill(graph, sh, 0.0);
                let mask = add_op(
                    graph,
                    K::Greater,
                    &[ins[0], InputSource::Node(zero)],
                    dt,
                    sh,
                );
                let da = add_op(graph, K::Mul, &[gsrc, InputSource::Node(mask)], dt, sh);
                out.push((a, da));
            }
        }
        K::Sigmoid => {
            // y = σ(x) ; d/dx = g · y · (1 − y).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let one = const_fill(graph, sh, 1.0);
                let omy = add_op(graph, K::Sub, &[InputSource::Node(one), y], dt, sh);
                let yomy = add_op(graph, K::Mul, &[y, InputSource::Node(omy)], dt, sh);
                let da = add_op(graph, K::Mul, &[gsrc, InputSource::Node(yomy)], dt, sh);
                out.push((a, da));
            }
        }
        K::Tanh => {
            // y = tanh(x) ; d/dx = g · (1 − y²).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let one = const_fill(graph, sh, 1.0);
                let yy = add_op(graph, K::Mul, &[y, y], dt, sh);
                let d = add_op(
                    graph,
                    K::Sub,
                    &[InputSource::Node(one), InputSource::Node(yy)],
                    dt,
                    sh,
                );
                let da = add_op(graph, K::Mul, &[gsrc, InputSource::Node(d)], dt, sh);
                out.push((a, da));
            }
        }
        K::Exp => {
            // y = eˣ ; d/dx = g · y.
            if let Some(a) = node_of(ins[0]) {
                let da = add_op(graph, K::Mul, &[gsrc, y], dt, sh);
                out.push((a, da));
            }
        }
        K::Log => {
            // d/dx = g / x.
            if let Some(a) = node_of(ins[0]) {
                let da = add_op(graph, K::Div, &[gsrc, ins[0]], dt, sh);
                out.push((a, da));
            }
        }
        K::Sqrt => {
            // y = √x ; d/dx = g / (2y).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let two = const_fill(graph, sh, 2.0);
                let den = add_op(graph, K::Mul, &[InputSource::Node(two), y], dt, sh);
                let da = add_op(graph, K::Div, &[gsrc, InputSource::Node(den)], dt, sh);
                out.push((a, da));
            }
        }
        K::Reciprocal => {
            // y = 1/x ; d/dx = −g · y².
            if let Some(a) = node_of(ins[0]) {
                let yy = add_op(graph, K::Mul, &[y, y], dt, sh);
                let gm = add_op(graph, K::Mul, &[gsrc, InputSource::Node(yy)], dt, sh);
                let da = add_op(graph, K::Neg, &[InputSource::Node(gm)], dt, sh);
                out.push((a, da));
            }
        }
        K::Reshape => {
            // Reshape is a relabel: route the gradient back to the input shape.
            if let Some(a) = node_of(ins[0]) {
                let (adt, ash) = meta(graph, ins[0]);
                let da = add_op(graph, K::Reshape, &[gsrc], adt, ash);
                out.push((a, da));
            }
        }
        K::Silu => {
            // silu = x·σ(x) ; silu'(x) = σ + x·σ·(1−σ).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let s = InputSource::Node(add_op(graph, K::Sigmoid, &[ins[0]], dt, sh));
                let oms = add_op(graph, K::Sub, &[one, s], dt, sh);
                let xs = add_op(graph, K::Mul, &[ins[0], s], dt, sh);
                let xsoms = add_op(
                    graph,
                    K::Mul,
                    &[InputSource::Node(xs), InputSource::Node(oms)],
                    dt,
                    sh,
                );
                let dsilu = add_op(graph, K::Add, &[s, InputSource::Node(xsoms)], dt, sh);
                let da = add_op(graph, K::Mul, &[gsrc, InputSource::Node(dsilu)], dt, sh);
                out.push((a, da));
            }
        }
        K::Gelu => {
            // tanh-approx gelu = 0.5x(1+tanh(u)), u = c1(x + c2·x³).
            // gelu' = 0.5(1+t) + 0.5x(1−t²)·c1(1+3c2x²).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (c1, c2, c3) = (0.797_884_6_f32, 0.044_715_f32, 0.134_145_f32);
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let half = InputSource::Node(const_fill(graph, sh, 0.5));
                let c1k = InputSource::Node(const_fill(graph, sh, c1));
                let c2k = InputSource::Node(const_fill(graph, sh, c2));
                let c3k = InputSource::Node(const_fill(graph, sh, c3));
                let x2 = InputSource::Node(add_op(graph, K::Mul, &[ins[0], ins[0]], dt, sh));
                let x3 = InputSource::Node(add_op(graph, K::Mul, &[x2, ins[0]], dt, sh));
                let c2x3 = InputSource::Node(add_op(graph, K::Mul, &[c2k, x3], dt, sh));
                let innr = InputSource::Node(add_op(graph, K::Add, &[ins[0], c2x3], dt, sh));
                let u = InputSource::Node(add_op(graph, K::Mul, &[c1k, innr], dt, sh));
                let t = InputSource::Node(add_op(graph, K::Tanh, &[u], dt, sh));
                let onept = InputSource::Node(add_op(graph, K::Add, &[one, t], dt, sh));
                let term1 = InputSource::Node(add_op(graph, K::Mul, &[half, onept], dt, sh));
                let t2 = InputSource::Node(add_op(graph, K::Mul, &[t, t], dt, sh));
                let omt2 = InputSource::Node(add_op(graph, K::Sub, &[one, t2], dt, sh));
                let c3x2 = InputSource::Node(add_op(graph, K::Mul, &[c3k, x2], dt, sh));
                let onep = InputSource::Node(add_op(graph, K::Add, &[one, c3x2], dt, sh));
                let du = InputSource::Node(add_op(graph, K::Mul, &[c1k, onep], dt, sh));
                let hx = InputSource::Node(add_op(graph, K::Mul, &[half, ins[0]], dt, sh));
                let hxo = InputSource::Node(add_op(graph, K::Mul, &[hx, omt2], dt, sh));
                let term2 = InputSource::Node(add_op(graph, K::Mul, &[hxo, du], dt, sh));
                let dg = InputSource::Node(add_op(graph, K::Add, &[term1, term2], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, dg], dt, sh);
                out.push((a, da));
            }
        }
        K::Elu => {
            // elu'(x) = 1 (x≥0) else exp(x) = y+1 ; mask·1 + (1−mask)(y+1).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let zero = InputSource::Node(const_fill(graph, sh, 0.0));
                let m =
                    InputSource::Node(add_op(graph, K::GreaterOrEqual, &[ins[0], zero], dt, sh));
                let yp1 = InputSource::Node(add_op(graph, K::Add, &[y, one], dt, sh));
                let omm = InputSource::Node(add_op(graph, K::Sub, &[one, m], dt, sh));
                let neg = InputSource::Node(add_op(graph, K::Mul, &[omm, yp1], dt, sh));
                let dg = InputSource::Node(add_op(graph, K::Add, &[m, neg], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, dg], dt, sh);
                out.push((a, da));
            }
        }
        K::Selu => {
            // selu' = scale (x≥0) else scale·α·exp(x) = y + scale·α.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (scale, alpha) = (1.050_701_f32, 1.673_263_2_f32);
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let zero = InputSource::Node(const_fill(graph, sh, 0.0));
                let sk = InputSource::Node(const_fill(graph, sh, scale));
                let sak = InputSource::Node(const_fill(graph, sh, scale * alpha));
                let m =
                    InputSource::Node(add_op(graph, K::GreaterOrEqual, &[ins[0], zero], dt, sh));
                let pos = InputSource::Node(add_op(graph, K::Mul, &[m, sk], dt, sh));
                let omm = InputSource::Node(add_op(graph, K::Sub, &[one, m], dt, sh));
                let ysa = InputSource::Node(add_op(graph, K::Add, &[y, sak], dt, sh));
                let neg = InputSource::Node(add_op(graph, K::Mul, &[omm, ysa], dt, sh));
                let dg = InputSource::Node(add_op(graph, K::Add, &[pos, neg], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, dg], dt, sh);
                out.push((a, da));
            }
        }
        K::Sin => {
            if let Some(a) = node_of(ins[0]) {
                let c = InputSource::Node(add_op(graph, K::Cos, &[ins[0]], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, c], dt, sh);
                out.push((a, da));
            }
        }
        K::Cos => {
            if let Some(a) = node_of(ins[0]) {
                let s = InputSource::Node(add_op(graph, K::Sin, &[ins[0]], dt, sh));
                let gs = InputSource::Node(add_op(graph, K::Mul, &[gsrc, s], dt, sh));
                let da = add_op(graph, K::Neg, &[gs], dt, sh);
                out.push((a, da));
            }
        }
        K::Tan => {
            // tan'(x) = 1 + tan²(x) = 1 + y².
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let yy = InputSource::Node(add_op(graph, K::Mul, &[y, y], dt, sh));
                let d = InputSource::Node(add_op(graph, K::Add, &[one, yy], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, d], dt, sh);
                out.push((a, da));
            }
        }
        K::Abs => {
            if let Some(a) = node_of(ins[0]) {
                let s = InputSource::Node(add_op(graph, K::Sign, &[ins[0]], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, s], dt, sh);
                out.push((a, da));
            }
        }
        K::Sign => {
            // d/dx sign = 0 a.e.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let z = const_fill(graph, sh, 0.0);
                out.push((a, z));
            }
        }
        K::Erf => {
            // erf'(x) = 2/√π · exp(−x²).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let coef =
                    InputSource::Node(const_fill(graph, sh, core::f32::consts::FRAC_2_SQRT_PI));
                let xx = InputSource::Node(add_op(graph, K::Mul, &[ins[0], ins[0]], dt, sh));
                let nxx = InputSource::Node(add_op(graph, K::Neg, &[xx], dt, sh));
                let e = InputSource::Node(add_op(graph, K::Exp, &[nxx], dt, sh));
                let ce = InputSource::Node(add_op(graph, K::Mul, &[coef, e], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, ce], dt, sh);
                out.push((a, da));
            }
        }
        K::Min => {
            // d/da = g·[a≤b] ; d/db = g·[a>b].
            if let Some(a) = node_of(ins[0]) {
                let m = InputSource::Node(add_op(graph, K::LessOrEqual, &[ins[0], ins[1]], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, m], dt, sh);
                out.push((a, da));
            }
            if let Some(b) = node_of(ins[1]) {
                let m = InputSource::Node(add_op(graph, K::Greater, &[ins[0], ins[1]], dt, sh));
                let db = add_op(graph, K::Mul, &[gsrc, m], dt, sh);
                out.push((b, db));
            }
        }
        K::Max => {
            // d/da = g·[a≥b] ; d/db = g·[a<b].
            if let Some(a) = node_of(ins[0]) {
                let m =
                    InputSource::Node(add_op(graph, K::GreaterOrEqual, &[ins[0], ins[1]], dt, sh));
                let da = add_op(graph, K::Mul, &[gsrc, m], dt, sh);
                out.push((a, da));
            }
            if let Some(b) = node_of(ins[1]) {
                let m = InputSource::Node(add_op(graph, K::Less, &[ins[0], ins[1]], dt, sh));
                let db = add_op(graph, K::Mul, &[gsrc, m], dt, sh);
                out.push((b, db));
            }
        }
        K::Pow => {
            // z = a^b ; d/da = g·b·a^(b−1) ; d/db = g·z·ln(a)  (a>0).
            if !needs_f32() {
                return Err(BackwardError::NoGradient(kind));
            }
            if let Some(a) = node_of(ins[0]) {
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let bm1 = InputSource::Node(add_op(graph, K::Sub, &[ins[1], one], dt, sh));
                let apow = InputSource::Node(add_op(graph, K::Pow, &[ins[0], bm1], dt, sh));
                let gb = InputSource::Node(add_op(graph, K::Mul, &[gsrc, ins[1]], dt, sh));
                let da = add_op(graph, K::Mul, &[gb, apow], dt, sh);
                out.push((a, da));
            }
            if let Some(b) = node_of(ins[1]) {
                let lna = InputSource::Node(add_op(graph, K::Log, &[ins[0]], dt, sh));
                let gz = InputSource::Node(add_op(graph, K::Mul, &[gsrc, y], dt, sh));
                let db = add_op(graph, K::Mul, &[gz, lna], dt, sh);
                out.push((b, db));
            }
        }
        K::RmsNorm => {
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                match rmsnorm_dx(graph, ins[0], ins[1], gsrc, dt, sh) {
                    Some(da) => out.push((a, da)),
                    None => return Err(BackwardError::NoGradient(kind)),
                }
            }
        }
        K::AddRmsNorm => {
            // y = rmsnorm(x + residual)·γ. The fused add is a passthrough, so
            // dx = dresidual = rmsnorm_dx evaluated at s = x + residual.
            // ins = [x, γ, residual].
            if !needs_f32() || ins.len() < 3 {
                return Err(BackwardError::NoGradient(kind));
            }
            let s = InputSource::Node(add_op(graph, K::Add, &[ins[0], ins[2]], dt, sh));
            let drms = match rmsnorm_dx(graph, s, ins[1], gsrc, dt, sh) {
                Some(n) => n,
                None => return Err(BackwardError::NoGradient(kind)),
            };
            if let Some(xn) = node_of(ins[0]) {
                out.push((xn, drms));
            }
            if let Some(rn) = node_of(ins[2]) {
                out.push((rn, drms));
            }
        }
        K::LayerNorm | K::GroupNorm | K::InstanceNorm => {
            // GroupNorm/InstanceNorm lower to the *same* `layer_norm_float`
            // kernel over the rank-2 [batch, feature] view (no separate
            // grouping is realized — see `cpu/kernels.rs`), so their VJP is
            // identical to LayerNorm's. Differentiating the actual forward
            // authority, not a hypothetical grouped one.
            // x̂ = (x−μ)·invstd ; gg = g·γ.
            // dx = invstd·(gg − mean(gg) − x̂·mean(gg·x̂)).   (ε≈0)
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() || graph.shape_registry().get(sh).map(|d| d.rank) != Some(2) {
                    return Err(BackwardError::NoGradient(kind));
                }
                let x = ins[0];
                let gb = match broadcast_feature(graph, ins[1], dt, sh) {
                    Some(n) => InputSource::Node(n),
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                let mu = InputSource::Node(row_mean_broadcast(graph, x, dt, sh).unwrap());
                let xc = InputSource::Node(add_op(graph, K::Sub, &[x, mu], dt, sh));
                let xc2 = InputSource::Node(add_op(graph, K::Mul, &[xc, xc], dt, sh));
                let var = InputSource::Node(row_mean_broadcast(graph, xc2, dt, sh).unwrap());
                let std = InputSource::Node(add_op(graph, K::Sqrt, &[var], dt, sh));
                let invstd = InputSource::Node(add_op(graph, K::Reciprocal, &[std], dt, sh));
                let xhat = InputSource::Node(add_op(graph, K::Mul, &[xc, invstd], dt, sh));
                let gg = InputSource::Node(add_op(graph, K::Mul, &[gsrc, gb], dt, sh));
                let m1 = InputSource::Node(row_mean_broadcast(graph, gg, dt, sh).unwrap());
                let ggxh = InputSource::Node(add_op(graph, K::Mul, &[gg, xhat], dt, sh));
                let m2 = InputSource::Node(row_mean_broadcast(graph, ggxh, dt, sh).unwrap());
                let xhm2 = InputSource::Node(add_op(graph, K::Mul, &[xhat, m2], dt, sh));
                let inner = InputSource::Node(add_op(graph, K::Sub, &[gg, m1], dt, sh));
                let inner2 = InputSource::Node(add_op(graph, K::Sub, &[inner, xhm2], dt, sh));
                let da = add_op(graph, K::Mul, &[invstd, inner2], dt, sh);
                out.push((a, da));
            }
        }
        K::Softmax => {
            // y = softmax(x) over last axis ; dx = y⊙(g − Σ_row(y⊙g)).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let yg = InputSource::Node(add_op(graph, K::Mul, &[y, gsrc], dt, sh));
                let s = match row_reduce_broadcast(graph, yg, dt, sh) {
                    Some(s) => InputSource::Node(s),
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                let diff = InputSource::Node(add_op(graph, K::Sub, &[gsrc, s], dt, sh));
                let da = add_op(graph, K::Mul, &[y, diff], dt, sh);
                out.push((a, da));
            }
        }
        K::LogSoftmax => {
            // y = logsoftmax(x) ; dx = g − exp(y)·Σ_row(g).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let gs = match row_reduce_broadcast(graph, gsrc, dt, sh) {
                    Some(s) => InputSource::Node(s),
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                let sm = InputSource::Node(add_op(graph, K::Exp, &[y], dt, sh));
                let smgs = InputSource::Node(add_op(graph, K::Mul, &[sm, gs], dt, sh));
                let da = add_op(graph, K::Sub, &[gsrc, smgs], dt, sh);
                out.push((a, da));
            }
        }
        K::Concat => {
            // y = a ∥ b along axis 0 ; dL/da = g[0:a₀] , dL/db = g[a₀:a₀+b₀].
            let (adt, ash) = meta(graph, ins[0]);
            let a0 = match dim0(graph, ash) {
                Some(v) => v,
                None => return Err(BackwardError::NoGradient(kind)),
            };
            if let Some(a) = node_of(ins[0]) {
                let (s, e) = (i64_const(graph, &[0]), i64_const(graph, &[a0]));
                let da = add_op(graph, K::Slice, &[gsrc, s, e], adt, ash);
                out.push((a, da));
            }
            if ins.len() > 1 {
                if let Some(b) = node_of(ins[1]) {
                    let (bdt, bsh) = meta(graph, ins[1]);
                    let b0 = dim0(graph, bsh).unwrap_or(0);
                    let (s, e) = (i64_const(graph, &[a0]), i64_const(graph, &[a0 + b0]));
                    let db = add_op(graph, K::Slice, &[gsrc, s, e], bdt, bsh);
                    out.push((b, db));
                }
            }
        }
        K::Slice => {
            // y = x[start:end] (axis 0) ; dx scatters g back into zeros = Pad.
            if let Some(a) = node_of(ins[0]) {
                let (adt, ash) = meta(graph, ins[0]);
                let (d0, rank) = match dim0(graph, ash) {
                    Some(d0) => (d0, rank_of(graph, ash)),
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                let (start, end) =
                    match (read_i64_const(graph, ins[1]), read_i64_const(graph, ins[2])) {
                        (Some(s), Some(e)) => (s.clamp(0, d0), e.clamp(0, d0)),
                        _ => return Err(BackwardError::NoGradient(kind)),
                    };
                // Pad width format is [begin₀..beginᵣ, end₀..endᵣ]: pad axis-0
                // only — `start` rows before, `d0−end` rows after.
                let mut pads = vec![0i64; 2 * rank];
                pads[0] = start;
                pads[rank] = d0 - end;
                let pc = i64_const(graph, &pads);
                let dx = add_op(graph, K::Pad, &[gsrc, pc], adt, ash);
                out.push((a, dx));
            }
        }
        K::Pad => {
            // y = pad(x) (axis 0, pad_before rows) ; dx = g[pad_before : +x₀].
            if let Some(a) = node_of(ins[0]) {
                let (adt, ash) = meta(graph, ins[0]);
                let x0 = match dim0(graph, ash) {
                    Some(v) => v,
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                let pad_before = read_i64_const(graph, ins[1]).unwrap_or(0);
                let (s, e) = (
                    i64_const(graph, &[pad_before]),
                    i64_const(graph, &[pad_before + x0]),
                );
                let dx = add_op(graph, K::Slice, &[gsrc, s, e], adt, ash);
                out.push((a, dx));
            }
        }
        K::Expand => {
            // y = broadcast(x) ; dx = sum g over the broadcast axes.
            // Rank-2, exactly one broadcast axis: a row/column sum (matmul-ones).
            if let Some(a) = node_of(ins[0]) {
                let (adt, ash) = meta(graph, ins[0]);
                let (id, od) = match (
                    graph.shape_registry().get(ash).cloned(),
                    graph.shape_registry().get(sh).cloned(),
                ) {
                    (Some(i), Some(o)) if i.rank == 2 && o.rank == 2 => (i, o),
                    _ => return Err(BackwardError::NoGradient(kind)),
                };
                let (ib, if_) = (id.dim(0).unwrap_or(0), id.dim(1).unwrap_or(0));
                let (ob, of) = (od.dim(0).unwrap_or(0), od.dim(1).unwrap_or(0));
                if ib == ob && if_ == 1 {
                    // broadcast axis 1: rowsum g[ob,of] · ones[of,1] → [ob,1].
                    let f1 = graph
                        .shape_registry_mut()
                        .intern(ShapeDescriptor::rank2(of, 1));
                    let ones = InputSource::Node(const_fill(graph, f1, 1.0));
                    let dx = add_op(graph, K::MatMul, &[gsrc, ones], adt, ash);
                    out.push((a, dx));
                } else if if_ == of && ib == 1 {
                    // broadcast axis 0: colsum ones[1,ob] · g[ob,of] → [1,of].
                    let o1 = graph
                        .shape_registry_mut()
                        .intern(ShapeDescriptor::rank2(1, ob));
                    let ones = InputSource::Node(const_fill(graph, o1, 1.0));
                    let dx = add_op(graph, K::MatMul, &[ones, gsrc], adt, ash);
                    out.push((a, dx));
                } else {
                    return Err(BackwardError::NoGradient(kind));
                }
            }
        }
        K::GlobalAvgPool => {
            // y[B,C] = mean_{H,W} x[B,C,H,W] ; dx = g/(H·W) broadcast.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (adt, ash) = meta(graph, ins[0]);
                let xd = match graph.shape_registry().get(ash).cloned() {
                    Some(d) if d.rank == 4 => d,
                    _ => return Err(BackwardError::NoGradient(kind)),
                };
                let (b, c, h, w) = (
                    xd.dim(0).unwrap_or(1),
                    xd.dim(1).unwrap_or(1),
                    xd.dim(2).unwrap_or(1),
                    xd.dim(3).unwrap_or(1),
                );
                let hw = (h * w).max(1);
                let bc11 = graph
                    .shape_registry_mut()
                    .intern(ShapeDescriptor::rank4(b, c, 1, 1));
                let gr = InputSource::Node(add_op(graph, K::Reshape, &[gsrc], adt, bc11));
                let e = InputSource::Node(add_op(graph, K::Expand, &[gr], adt, ash));
                let inv = InputSource::Node(const_fill(graph, ash, 1.0 / hw as f32));
                let dx = add_op(graph, K::Mul, &[e, inv], adt, ash);
                out.push((a, dx));
            }
        }
        K::AvgPool2d => {
            // y[N,C,Hₒ,Wₒ] = mean over each kh×kw window ; the window-mean is
            // linear, so dx = (upsample g) · 1/(kh·kw) — replicate each output
            // gradient across its window and scale. Non-overlapping windows.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (adt, ash) = meta(graph, ins[0]);
                let (up, kh, kw) = match upsample_pool(graph, gsrc, dt, sh, ash) {
                    Some(v) => v,
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                if !pool_strides_match(graph, fwd_id, kh, kw) {
                    return Err(BackwardError::NoGradient(kind));
                }
                let inv = InputSource::Node(const_fill(graph, ash, 1.0 / (kh * kw) as f32));
                let dx = add_op(graph, K::Mul, &[InputSource::Node(up), inv], adt, ash);
                out.push((a, dx));
            }
        }
        K::MaxPool2d => {
            // y = max over each kh×kw window ; the gradient routes entirely to
            // the arg-max: dx = (upsample g) ⊙ [x == upsample y]. For a unique
            // window maximum the mask selects exactly that element (ties, a
            // measure-zero set, would split — finite-diff tests use distinct
            // values). Non-overlapping windows.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (adt, ash) = meta(graph, ins[0]);
                let (upg, kh, kw) = match upsample_pool(graph, gsrc, dt, sh, ash) {
                    Some(v) => v,
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                if !pool_strides_match(graph, fwd_id, kh, kw) {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (upy, _, _) = match upsample_pool(graph, y, dt, sh, ash) {
                    Some(v) => v,
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                let mask = add_op(graph, K::Equal, &[ins[0], InputSource::Node(upy)], adt, ash);
                let dx = add_op(
                    graph,
                    K::Mul,
                    &[InputSource::Node(upg), InputSource::Node(mask)],
                    adt,
                    ash,
                );
                out.push((a, dx));
            }
        }
        K::ReduceSum => {
            // Full reduction to a scalar: dx_i = g (broadcast).
            if let Some(a) = node_of(ins[0]) {
                let (adt, ash) = meta(graph, ins[0]);
                let dx = broadcast_scalar(graph, gsrc, adt, ash);
                out.push((a, dx));
            }
        }
        K::ReduceMean => {
            // dx_i = g / N.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (adt, ash) = meta(graph, ins[0]);
                let n = graph
                    .shape_registry()
                    .get(ash)
                    .map(|d| d.total_elements())
                    .unwrap_or(1)
                    .max(1);
                let e = broadcast_scalar(graph, gsrc, adt, ash);
                let invn = InputSource::Node(const_fill(graph, ash, 1.0 / n as f32));
                let dx = add_op(graph, K::Mul, &[InputSource::Node(e), invn], adt, ash);
                out.push((a, dx));
            }
        }
        K::ReduceProd => {
            // y = ∏a ; dx_i = g·y / a_i.
            if let Some(a) = node_of(ins[0]) {
                let (adt, ash) = meta(graph, ins[0]);
                let gy = InputSource::Node(add_op(graph, K::Mul, &[gsrc, y], dt, sh));
                let e = broadcast_scalar(graph, gy, adt, ash);
                let dx = add_op(graph, K::Div, &[InputSource::Node(e), ins[0]], adt, ash);
                out.push((a, dx));
            }
        }
        K::ReduceMin | K::ReduceMax => {
            // dx_i = g where a_i is the selected extremum, else 0.
            if let Some(a) = node_of(ins[0]) {
                let (adt, ash) = meta(graph, ins[0]);
                let yb = InputSource::Node(broadcast_scalar(graph, y, adt, ash));
                let mask = InputSource::Node(add_op(graph, K::Equal, &[ins[0], yb], adt, ash));
                let gb = InputSource::Node(broadcast_scalar(graph, gsrc, adt, ash));
                let dx = add_op(graph, K::Mul, &[gb, mask], adt, ash);
                out.push((a, dx));
            }
        }
        K::Transpose if ins.len() == 1 => {
            // Perm-less (axis-reversing) transpose is its own inverse: route
            // the gradient back through another reversal to the input shape.
            if let Some(a) = node_of(ins[0]) {
                let (adt, ash) = meta(graph, ins[0]);
                let da = add_op(graph, K::Transpose, &[gsrc], adt, ash);
                out.push((a, da));
            }
        }
        K::MatMul => {
            // C = A·B (rank-2): dA = g·Bᵀ ; dB = Aᵀ·g.
            let (adt, ash) = meta(graph, ins[0]);
            let (bdt, bsh) = meta(graph, ins[1]);
            let (bt, at) = match (transpose2(graph, bsh), transpose2(graph, ash)) {
                (Some(bt), Some(at)) => (bt, at),
                _ => return Err(BackwardError::NoGradient(kind)),
            };
            if let Some(a) = node_of(ins[0]) {
                let btn = add_op(graph, K::Transpose, &[ins[1]], bdt, bt);
                let da = add_op(graph, K::MatMul, &[gsrc, InputSource::Node(btn)], adt, ash);
                out.push((a, da));
            }
            if let Some(b) = node_of(ins[1]) {
                let atn = add_op(graph, K::Transpose, &[ins[0]], adt, at);
                let db = add_op(graph, K::MatMul, &[InputSource::Node(atn), gsrc], bdt, bsh);
                out.push((b, db));
            }
        }
        K::Log1p => {
            // d/dx log(1+x) = g/(1+x).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let denom = InputSource::Node(add_op(graph, K::Add, &[ins[0], one], dt, sh));
                let da = add_op(graph, K::Div, &[gsrc, denom], dt, sh);
                out.push((a, da));
            }
        }
        K::Asin | K::Acos => {
            // d/dx asin = g/√(1−x²) ; d/dx acos = −g/√(1−x²).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let x2 = InputSource::Node(add_op(graph, K::Mul, &[ins[0], ins[0]], dt, sh));
                let omx2 = InputSource::Node(add_op(graph, K::Sub, &[one, x2], dt, sh));
                let root = InputSource::Node(add_op(graph, K::Sqrt, &[omx2], dt, sh));
                let q = add_op(graph, K::Div, &[gsrc, root], dt, sh);
                let da = if kind == K::Acos {
                    add_op(graph, K::Neg, &[InputSource::Node(q)], dt, sh)
                } else {
                    q
                };
                out.push((a, da));
            }
        }
        K::Atan => {
            // d/dx atan = g/(1+x²).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let one = InputSource::Node(const_fill(graph, sh, 1.0));
                let x2 = InputSource::Node(add_op(graph, K::Mul, &[ins[0], ins[0]], dt, sh));
                let denom = InputSource::Node(add_op(graph, K::Add, &[one, x2], dt, sh));
                let da = add_op(graph, K::Div, &[gsrc, denom], dt, sh);
                out.push((a, da));
            }
        }
        K::Ceil | K::Floor | K::Round => {
            // Piecewise-constant: derivative 0 almost everywhere.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let z = const_fill(graph, sh, 0.0);
                out.push((a, z));
            }
        }
        K::CumSum => {
            // y_i = Σ_{j≤i} x_j ; dx_i = Σ_{j≥i} g_j = total(g) − cumsum(g) + g.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let scalar = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
                let tot = InputSource::Node(add_op(graph, K::ReduceSum, &[gsrc], dt, scalar));
                let tot_b = InputSource::Node(broadcast_scalar(graph, tot, dt, sh));
                let cs = InputSource::Node(add_op(graph, K::CumSum, &[gsrc], dt, sh));
                let diff = InputSource::Node(add_op(graph, K::Sub, &[tot_b, cs], dt, sh));
                let da = add_op(graph, K::Add, &[diff, gsrc], dt, sh);
                out.push((a, da));
            }
        }
        K::RotaryEmbedding => {
            // RoPE (rotate-half, per-element cos/sin tables):
            //   out[i]   = x[i]·c[i]   − x[i+h]·s[i]
            //   out[i+h] = x[i+h]·c[i+h] + x[i]·s[i+h].
            // Its exact adjoint (correct for *any* tables, not just the
            // orthogonal cos²+sin²=1 case) is
            //   dx = g⊙cos − RoPE(g⊙sin, cos=0, sin=1),
            // since `RoPE(·,0,1)` is the half-swap [a,b]↦[−b,a]. cos/sin are
            // position tables (leaves). Composed from Mul/Sub + RoPE itself.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() || ins.len() < 3 {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (_cdt, csh) = meta(graph, ins[1]);
                let gs = InputSource::Node(add_op(graph, K::Mul, &[gsrc, ins[2]], dt, sh));
                let zeros = InputSource::Node(const_fill(graph, csh, 0.0));
                let ones = InputSource::Node(const_fill(graph, csh, 1.0));
                let rot = InputSource::Node(add_op(
                    graph,
                    K::RotaryEmbedding,
                    &[gs, zeros, ones],
                    dt,
                    sh,
                ));
                let gc = InputSource::Node(add_op(graph, K::Mul, &[gsrc, ins[1]], dt, sh));
                let dx = add_op(graph, K::Sub, &[gc, rot], dt, sh);
                out.push((a, dx));
            }
        }
        K::Gemm => {
            // Z = α·A·B + β·C ; dA = α·g·Bᵀ, dB = α·Aᵀ·g, dC = β·g.
            // α,β are baked into ConvAttrs-style GemmAttrs (default 1).
            let (adt, ash) = meta(graph, ins[0]);
            let (bdt, bsh) = meta(graph, ins[1]);
            let (at, bt) = match (transpose2(graph, ash), transpose2(graph, bsh)) {
                (Some(at), Some(bt)) => (at, bt),
                _ => return Err(BackwardError::NoGradient(kind)),
            };
            let (alpha, beta) = graph
                .gemm_attrs(fwd_id)
                .map(|a| (f32::from_bits(a.alpha_bits), f32::from_bits(a.beta_bits)))
                .unwrap_or((1.0, 1.0));
            if let Some(an) = node_of(ins[0]) {
                let btn = InputSource::Node(add_op(graph, K::Transpose, &[ins[1]], bdt, bt));
                let gb = InputSource::Node(add_op(graph, K::MatMul, &[gsrc, btn], adt, ash));
                let ak = InputSource::Node(const_fill(graph, ash, alpha));
                let da = add_op(graph, K::Mul, &[gb, ak], adt, ash);
                out.push((an, da));
            }
            if let Some(bn) = node_of(ins[1]) {
                let atn = InputSource::Node(add_op(graph, K::Transpose, &[ins[0]], adt, at));
                let ag = InputSource::Node(add_op(graph, K::MatMul, &[atn, gsrc], bdt, bsh));
                let bk = InputSource::Node(const_fill(graph, bsh, alpha));
                let db = add_op(graph, K::Mul, &[ag, bk], bdt, bsh);
                out.push((bn, db));
            }
            if ins.len() > 2 {
                if let Some(cn) = node_of(ins[2]) {
                    let (cdt, csh) = meta(graph, ins[2]);
                    let bk = InputSource::Node(const_fill(graph, csh, beta));
                    let dc = add_op(graph, K::Mul, &[gsrc, bk], cdt, csh);
                    out.push((cn, dc));
                }
            }
        }
        K::Mod => {
            // Floored modulo y = a − floor(a/b)·b (cpu::mod_f). dy/da = 1,
            // dy/db = −floor(a/b) (floor is locally constant a.e.). The blocker
            // was illusory — floored mod needs `floor`, not a `trunc` primitive.
            if !needs_f32() {
                return Err(BackwardError::NoGradient(kind));
            }
            if let Some(a) = node_of(ins[0]) {
                out.push((a, g));
            }
            if let Some(b) = node_of(ins[1]) {
                let q = InputSource::Node(add_op(graph, K::Div, &[ins[0], ins[1]], dt, sh));
                let fl = InputSource::Node(add_op(graph, K::Floor, &[q], dt, sh));
                let gfl = InputSource::Node(add_op(graph, K::Mul, &[gsrc, fl], dt, sh));
                let db = add_op(graph, K::Neg, &[gfl], dt, sh);
                out.push((b, db));
            }
        }
        K::Equal | K::Less | K::LessOrEqual | K::Greater | K::GreaterOrEqual => {
            // A comparison is a step predicate of its float inputs; its
            // classical derivative is 0 almost everywhere (the jump at equality
            // is a measure-zero set). Gradients are detached through it — the
            // standard autograd treatment of predicates.
            if !needs_f32() {
                return Err(BackwardError::NoGradient(kind));
            }
            if let Some(a) = node_of(ins[0]) {
                let z = const_fill(graph, sh, 0.0);
                out.push((a, z));
            }
            if ins.len() > 1 {
                if let Some(b) = node_of(ins[1]) {
                    let z = const_fill(graph, sh, 0.0);
                    out.push((b, z));
                }
            }
        }
        K::IsNaN => {
            // Boolean predicate: 0 gradient (locally constant on finite inputs).
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let z = const_fill(graph, sh, 0.0);
                out.push((a, z));
            }
        }
        K::Where => {
            // out = (cond ≠ 0) ? a : b. The gradient routes g to whichever
            // branch was selected: da = where(cond, g, 0), db = where(cond, 0, g)
            // — reusing the Where kernel itself so the selection matches the
            // forward exactly (cond is a raw-byte/bool selector, not a float).
            // cond is a selector (gradient 0). ins = [cond, a, b].
            if !needs_f32() || ins.len() < 3 {
                return Err(BackwardError::NoGradient(kind));
            }
            let zeros = InputSource::Node(const_fill(graph, sh, 0.0));
            if let Some(an) = node_of(ins[1]) {
                let da = add_op(graph, K::Where, &[ins[0], gsrc, zeros], dt, sh);
                out.push((an, da));
            }
            if let Some(bn) = node_of(ins[2]) {
                let db = add_op(graph, K::Where, &[ins[0], zeros, gsrc], dt, sh);
                out.push((bn, db));
            }
            if let Some(cn) = node_of(ins[0]) {
                let z = const_fill(graph, sh, 0.0);
                out.push((cn, z));
            }
        }
        K::Resize => {
            // Nearest-neighbour resize is a pure gather `out[o] = x[nearest(o)]`
            // (a 0/1 selection), so its VJP scatter-adds: `dx = Sᵀ·g`. Build
            // Sᵀ `[in_total, out_total]` (one 1 per source→output pair, matching
            // the kernel's `floor(coord·in/out)` map) as a constant and matmul.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let (adt, ash) = meta(graph, ins[0]);
                let (id, od) = match (
                    graph.shape_registry().get(ash).cloned(),
                    graph.shape_registry().get(sh).cloned(),
                ) {
                    (Some(i), Some(o)) if i.rank == o.rank && i.rank >= 1 => (i, o),
                    _ => return Err(BackwardError::NoGradient(kind)),
                };
                let rank = id.rank as usize;
                let in_dims: Vec<u64> = (0..rank).map(|i| id.dim(i).unwrap_or(1)).collect();
                let out_dims: Vec<u64> = (0..rank).map(|i| od.dim(i).unwrap_or(1)).collect();
                let in_total: u64 = in_dims.iter().product();
                let out_total: u64 = out_dims.iter().product();
                if in_total == 0 || out_total == 0 {
                    return Err(BackwardError::NoGradient(kind));
                }
                // Sᵀ[in, out] row-major: Sᵀ[in_idx·out_total + o] = 1.
                let mut st = vec![0f32; (in_total * out_total) as usize];
                let mut in_strides = vec![1u64; rank];
                for i in (0..rank).rev().skip(1) {
                    in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
                }
                for o in 0..out_total {
                    let (mut rem, mut in_idx) = (o, 0u64);
                    for i in (0..rank).rev() {
                        let coord = rem % out_dims[i];
                        rem /= out_dims[i];
                        let src = (coord * in_dims[i] / out_dims[i]).min(in_dims[i] - 1);
                        in_idx += src * in_strides[i];
                    }
                    st[(in_idx * out_total + o) as usize] = 1.0;
                }
                let st_bytes: Vec<u8> = st.iter().flat_map(|v| v.to_le_bytes()).collect();
                let st_sh = intern2(graph, in_total, out_total);
                let st_node = InputSource::Node(const_tensor(graph, st_sh, st_bytes));
                let go_sh = intern2(graph, out_total, 1);
                let gflat = InputSource::Node(add_op(graph, K::Reshape, &[gsrc], adt, go_sh));
                let dxf_sh = intern2(graph, in_total, 1);
                let dxf =
                    InputSource::Node(add_op(graph, K::MatMul, &[st_node, gflat], adt, dxf_sh));
                let dx = add_op(graph, K::Reshape, &[dxf], adt, ash);
                out.push((a, dx));
            }
        }
        K::Lrn => {
            // y_c = x_c·d_c^(−β), d_c = bias + (α/size)·Σ_{j∈win(c)} x_j² over a
            // channel window. dx = g·d^(−β) − (2αβ/size)·x·(Bᵀ·[g·x·d^(−β−1)]),
            // where B is the window's band matrix and Bᵀ the adjoint window
            // (i∈win(m)). Windowed sums apply B along the channel axis.
            if let Some(a) = node_of(ins[0]) {
                if !needs_f32() {
                    return Err(BackwardError::NoGradient(kind));
                }
                let attrs = match graph.lrn_attrs(fwd_id) {
                    Some(at) => at,
                    None => return Err(BackwardError::NoGradient(kind)),
                };
                // LRN dims match the compiler's `lrn_dims`: batch=dim0,
                // channels=dim1, inner=∏(dim₂..) — so [N,C,H,W] and [N,C,inner]
                // both work (elementwise intermediates keep the node's shape;
                // only the channel-axis matmul needs the [batch,C,inner] view).
                let d = match graph.shape_registry().get(sh).cloned() {
                    Some(d) if d.rank >= 2 => d,
                    _ => return Err(BackwardError::NoGradient(kind)),
                };
                let rank = d.rank as usize;
                let (batch, ch) = (d.dim(0).unwrap_or(1), d.dim(1).unwrap_or(1));
                let inner: u64 = (2..rank)
                    .map(|i| d.dim(i).unwrap_or(1))
                    .product::<u64>()
                    .max(1);
                let size = attrs.size.max(1) as u64;
                let (alpha, beta, bias) = (
                    f32::from_bits(attrs.alpha_bits),
                    f32::from_bits(attrs.beta_bits),
                    f32::from_bits(attrs.bias_bits),
                );
                let lo_off = (size - 1) / 2;
                // Band matrices B (window) and Bᵀ (adjoint) as [ch,ch] constants.
                let (mut bb, mut bt) = (
                    vec![0f32; (ch * ch) as usize],
                    vec![0f32; (ch * ch) as usize],
                );
                for c in 0..ch {
                    let c0 = c.saturating_sub(lo_off);
                    let c1 = (c + size / 2 + 1).min(ch);
                    for cp in c0..c1 {
                        bb[(c * ch + cp) as usize] = 1.0;
                        bt[(cp * ch + c) as usize] = 1.0;
                    }
                }
                let chsh = intern2(graph, ch, ch);
                let b_node = InputSource::Node(const_tensor(
                    graph,
                    chsh,
                    bb.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ));
                let bt_node = InputSource::Node(const_tensor(
                    graph,
                    chsh,
                    bt.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ));
                let x = ins[0];
                let x2 = InputSource::Node(add_op(graph, K::Mul, &[x, x], dt, sh));
                let bx2 = InputSource::Node(apply_channel_matrix(
                    graph, b_node, x2, batch, ch, inner, dt, sh,
                ));
                let coef = InputSource::Node(const_fill(graph, sh, alpha / size as f32));
                let scaled = InputSource::Node(add_op(graph, K::Mul, &[bx2, coef], dt, sh));
                let biask = InputSource::Node(const_fill(graph, sh, bias));
                let dd = InputSource::Node(add_op(graph, K::Add, &[scaled, biask], dt, sh));
                let negb = InputSource::Node(const_fill(graph, sh, -beta));
                let negb1 = InputSource::Node(const_fill(graph, sh, -beta - 1.0));
                let dnegb = InputSource::Node(add_op(graph, K::Pow, &[dd, negb], dt, sh));
                let dnegb1 = InputSource::Node(add_op(graph, K::Pow, &[dd, negb1], dt, sh));
                let term1 = InputSource::Node(add_op(graph, K::Mul, &[gsrc, dnegb], dt, sh));
                let gx = InputSource::Node(add_op(graph, K::Mul, &[gsrc, x], dt, sh));
                let h = InputSource::Node(add_op(graph, K::Mul, &[gx, dnegb1], dt, sh));
                let bth = InputSource::Node(apply_channel_matrix(
                    graph, bt_node, h, batch, ch, inner, dt, sh,
                ));
                let c2 = InputSource::Node(const_fill(graph, sh, 2.0 * alpha * beta / size as f32));
                let xbth = InputSource::Node(add_op(graph, K::Mul, &[x, bth], dt, sh));
                let term2 = InputSource::Node(add_op(graph, K::Mul, &[xbth, c2], dt, sh));
                let dx = add_op(graph, K::Sub, &[term1, term2], dt, sh);
                out.push((a, dx));
            }
        }
        K::Conv2d | K::ConvTranspose2d => {
            // dX = col2im(Wᵀ·g), dW = Σ_b gᵦ·im2col(xᵦ)ᵀ — conv as W·im2col(x)
            // differentiated through the verified MatMul VJP. Valid conv only
            // (the runtime's conv ignores padding); stride read from ConvAttrs.
            // ConvTranspose2d shares the conv forward kernel, so it shares the
            // VJP (we differentiate the actual forward authority).
            if !needs_f32() || ins.len() < 2 {
                return Err(BackwardError::NoGradient(kind));
            }
            let (_, xsh) = meta(graph, ins[0]);
            let (_, wsh) = meta(graph, ins[1]);
            let attrs = graph.conv_attrs(fwd_id).unwrap_or_default();
            let (dx, dw) = match conv2d_vjp(
                graph,
                ins[0],
                ins[1],
                gsrc,
                xsh,
                wsh,
                sh,
                attrs.stride_h.max(1),
                attrs.stride_w.max(1),
                dt,
            ) {
                Some(v) => v,
                None => return Err(BackwardError::NoGradient(kind)),
            };
            if let Some(xn) = node_of(ins[0]) {
                out.push((xn, dx));
            }
            if let Some(wn) = node_of(ins[1]) {
                out.push((wn, dw));
            }
        }
        K::Attention => {
            // O[b,h] = softmax(Q·Kᵀ/√d)·V per (batch, head). The VJP unrolls
            // over the static batch×head count into rank-2 problems (no batched
            // primitive); contributions for Q, K, V.
            if !needs_f32() || ins.len() < 3 {
                return Err(BackwardError::NoGradient(kind));
            }
            let (_, qsh) = meta(graph, ins[0]);
            let (dq, dk, dv) = match attention_vjp(graph, ins[0], ins[1], ins[2], gsrc, qsh, dt) {
                Some(v) => v,
                None => return Err(BackwardError::NoGradient(kind)),
            };
            if let Some(qn) = node_of(ins[0]) {
                out.push((qn, dq));
            }
            if let Some(kn) = node_of(ins[1]) {
                out.push((kn, dk));
            }
            if let Some(vn) = node_of(ins[2]) {
                out.push((vn, dv));
            }
        }
        other => return Err(BackwardError::NoGradient(other)),
    }
    Ok(out)
}

/// Append a backward subgraph for `output_id`, composed entirely of forward
/// ops. Returns the gradient `NodeId` for each entry of [`Graph::inputs`] (the
/// i-th entry is `dL/d(input_i)`).
///
/// The seed gradient `dL/d(output)` is a fresh `Input` node the caller seeds
/// at runtime (defaulting to all-ones recovers the plain Jacobian-sum). A
/// graph input disconnected from the output gets a genuine zero-constant
/// gradient.
pub fn append_backward(graph: &mut Graph, output_id: NodeId) -> Result<Vec<NodeId>, BackwardError> {
    let n_forward = graph.node_count();
    let output_node = graph
        .get(output_id)
        .ok_or(BackwardError::OutputMissing(output_id))?;
    let output_dtype = output_node.output_dtype;
    let output_shape = output_node.output_shape;

    // The differentiable inputs are exactly those present before we append the
    // seed (which is infrastructure, not a value the model differentiates).
    let original_inputs: Vec<NodeId> = graph.inputs().to_vec();

    // Seed: dL/d(output), supplied by the runtime as the final graph input
    // (all-ones recovers the plain Jacobian-sum / `sum(output)` gradient).
    let seed_id = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype,
        output_shape,
    });
    graph.add_input(seed_id);

    let mut node_grads: Vec<Option<NodeId>> = vec![None; n_forward];
    if (output_id.0 as usize) < n_forward {
        node_grads[output_id.0 as usize] = Some(seed_id);
    }

    // Reverse-topological walk (graph ids are monotone in topo order).
    for i in (0..n_forward).rev() {
        let upstream = match node_grads[i] {
            Some(g) => g,
            None => continue,
        };
        let node = graph.nodes()[i].clone();
        let kind = match node.op {
            GraphOp::Op(k) => k,
            // Leaves: nothing to differentiate through.
            GraphOp::Input | GraphOp::Output | GraphOp::Constant(_) => continue,
        };

        let contribs = emit_vjp(graph, kind, &node, NodeId(i as u32), upstream)?;
        for (input_node, grad_node) in contribs {
            let idx = input_node.0 as usize;
            if idx >= node_grads.len() {
                continue;
            }
            match node_grads[idx] {
                // Fan-out: a value used more than once sums its gradients.
                Some(prev) => {
                    let (dt, sh) = graph
                        .get(input_node)
                        .map(|n| (n.output_dtype, n.output_shape))
                        .unwrap_or((output_dtype, output_shape));
                    let sum = add_op(
                        graph,
                        OpKind::Add,
                        &[InputSource::Node(prev), InputSource::Node(grad_node)],
                        dt,
                        sh,
                    );
                    node_grads[idx] = Some(sum);
                }
                None => node_grads[idx] = Some(grad_node),
            }
        }
    }

    // Per-input gradients. A disconnected input has a genuine zero gradient
    // (shared zero-constant node, created lazily).
    let mut zero_grad_id: Option<NodeId> = None;
    let input_grads: Vec<NodeId> = original_inputs
        .into_iter()
        .map(|nid| {
            let i = nid.0 as usize;
            if let Some(Some(g)) = node_grads.get(i).copied() {
                return g;
            }
            if let Some(g) = zero_grad_id {
                return g;
            }
            // A genuinely-zero gradient, sized to the output shape (f32). An
            // empty constant would have no bytes to materialize.
            let count = graph
                .shape_registry()
                .get(output_shape)
                .map(|d| d.total_elements())
                .unwrap_or(1)
                .max(1) as usize;
            let zero_const_id = graph.constants_mut().insert(ConstantEntry {
                bytes: vec![0u8; count * 4],
                dtype: output_dtype,
                shape: output_shape,
            });
            let id = graph.add_node(Node {
                op: GraphOp::Constant(zero_const_id),
                inputs: SmallVec::new(),
                output_dtype,
                output_shape,
            });
            zero_grad_id = Some(id);
            id
        })
        .collect();

    Ok(input_grads)
}
