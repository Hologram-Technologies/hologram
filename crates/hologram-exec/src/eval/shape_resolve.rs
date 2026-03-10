//! Unified shape resolution for FloatOp nodes.
//!
//! Single source of truth for computing output shapes from input shapes.
//! Called by the pre-propagation pass before each level's data dispatch.
//! Replaces scattered shape logic previously duplicated across
//! `executor.rs`, `shape_propagate.rs`, and `float_dispatch.rs`.

use hologram_core::op::{FloatDType, FloatOp, ShapeDim, ShapeSpec};

/// Context for shape resolution — all borrowed, zero allocation.
pub struct ShapeContext<'a> {
    /// Shapes of each input tensor (from ShapeMap).
    pub input_shapes: &'a [Vec<usize>],
    /// Compiled shape for this node (from .holo archive, may have 0-sentinels).
    pub compiled_shape: Option<&'a Vec<usize>>,
    /// Element count of input[0] (product of input[0].shape).
    pub input_elems: usize,
    /// Raw shape tensor bytes (input[1] for Reshape), if available in arena.
    pub shape_tensor_bytes: Option<&'a [u8]>,
    /// Compiled dtype for this node, used for element size.
    pub compiled_dtype: Option<&'a FloatDType>,
}

/// Resolve the output shape for any FloatOp.
///
/// Returns `None` only if shape truly cannot be determined.
/// This is the single entry point — all shape resolution goes through here.
pub fn resolve_float_shape(op: &FloatOp, ctx: &ShapeContext<'_>) -> Option<Vec<usize>> {
    let spec = op.output_shape_spec();
    match spec {
        ShapeSpec::Custom => resolve_custom(op, ctx),
        _ => resolve_standard(&spec, ctx),
    }
}

// ── Standard ShapeSpec resolution ───────────────────────────────────────────

/// Resolve a non-Custom ShapeSpec from input shapes alone.
fn resolve_standard(spec: &ShapeSpec, ctx: &ShapeContext<'_>) -> Option<Vec<usize>> {
    let shape = match spec {
        ShapeSpec::SameAs(i) => ctx.input_shapes.get(*i as usize)?.clone(),

        ShapeSpec::Broadcast(a, b) => {
            let sa = ctx.input_shapes.get(*a as usize)?;
            let sb = ctx.input_shapes.get(*b as usize)?;
            broadcast_shapes(sa, sb)
        }

        ShapeSpec::DropLastDim(i) => {
            let s = ctx.input_shapes.get(*i as usize)?;
            if s.len() > 1 {
                s[..s.len() - 1].to_vec()
            } else {
                vec![1]
            }
        }

        ShapeSpec::Dims(dims) => resolve_dims(dims, ctx.input_shapes, ctx.input_elems)?,

        ShapeSpec::Custom => return None,
    };

    // Validate against compiled shape if available.
    if let Some(compiled) = ctx.compiled_shape {
        Some(validate_against_compiled(shape, compiled))
    } else {
        Some(shape)
    }
}

/// Resolve a `Dims` spec using input shapes and input element count.
fn resolve_dims(
    dims: &[ShapeDim],
    input_shapes: &[Vec<usize>],
    input_elems: usize,
) -> Option<Vec<usize>> {
    let mut shape = Vec::with_capacity(dims.len());
    let mut known_product = 1usize;
    let mut inferred_idx = None;

    for (i, dim) in dims.iter().enumerate() {
        match dim {
            ShapeDim::Fixed(v) => {
                let v = *v as usize;
                shape.push(v);
                known_product = known_product.saturating_mul(v.max(1));
            }
            ShapeDim::FromInput { input, axis } => {
                let v = input_shapes
                    .get(*input as usize)
                    .and_then(|s| {
                        let idx = if *axis < 0 {
                            s.len().wrapping_add(*axis as usize)
                        } else {
                            *axis as usize
                        };
                        s.get(idx).copied()
                    })
                    .unwrap_or(1);
                shape.push(v);
                known_product = known_product.saturating_mul(v.max(1));
            }
            ShapeDim::Inferred => {
                shape.push(0); // placeholder
                inferred_idx = Some(i);
            }
        }
    }

    if let Some(idx) = inferred_idx {
        if known_product > 0 && input_elems > 0 {
            shape[idx] = input_elems / known_product;
        } else {
            return None;
        }
    }

    if shape.contains(&0) {
        None
    } else {
        Some(shape)
    }
}

/// Broadcast two shapes following numpy-style rules.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        result.push(da.max(db));
    }
    result
}

/// Validate a computed shape against a compiled shape.
/// If compiled has concrete (non-zero) dims where computed has zeros, fill them in.
fn validate_against_compiled(mut computed: Vec<usize>, compiled: &[usize]) -> Vec<usize> {
    if computed.len() == compiled.len() {
        for (i, &cd) in compiled.iter().enumerate() {
            if cd > 0 && computed[i] == 0 {
                computed[i] = cd;
            }
        }
    }
    computed
}

// ── Custom op resolution ────────────────────────────────────────────────────

/// Dispatch to per-op shape resolvers for Custom-spec ops.
fn resolve_custom(op: &FloatOp, ctx: &ShapeContext<'_>) -> Option<Vec<usize>> {
    match op {
        FloatOp::Gather { dim, .. } | FloatOp::Embed { dim } => resolve_gather(ctx, *dim),
        FloatOp::Reshape => resolve_reshape(ctx),
        FloatOp::Transpose { perm, ndim } => resolve_transpose(ctx, perm, *ndim),
        FloatOp::MatMul { k, .. } => resolve_matmul(ctx, *k),
        FloatOp::Gemm { k, .. } => resolve_gemm(ctx, *k),
        FloatOp::Concat { .. } => resolve_concat(ctx),
        FloatOp::Shape { .. } => ctx.input_shapes.first().map(|s| vec![s.len()]),
        FloatOp::Attention { .. } => ctx.input_shapes.first().cloned(),
        FloatOp::GatherND => None, // complex shape, deferred to dispatch
        _ => None,
    }
}

// ── Gather / Embed ──────────────────────────────────────────────────────────

/// Gather/Embed output shape = indices_shape ++ [dim].
/// For indices=[1, 2048] and dim=2048, output is [1, 2048, 2048].
fn resolve_gather(ctx: &ShapeContext<'_>, dim: u32) -> Option<Vec<usize>> {
    let indices_shape = ctx.input_shapes.first()?;
    let mut out = indices_shape.clone();
    out.push(dim as usize);
    Some(out)
}

// ── Reshape ─────────────────────────────────────────────────────────────────

/// Reshape shape resolution — handles both standard reshapes and GQA
/// broadcast expansion where output_elems > input_elems.
fn resolve_reshape(ctx: &ShapeContext<'_>) -> Option<Vec<usize>> {
    if ctx.input_elems == 0 {
        return None;
    }

    // 1. Parse runtime shape tensor if available (Constant bytes in arena).
    let parsed = ctx
        .shape_tensor_bytes
        .and_then(|b| parse_shape_values(b, ctx.input_elems));

    // 2. If shape tensor has no zeros: accept directly.
    //    Handles BOTH standard reshapes AND broadcast expansion (GQA).
    if let Some(ref st) = parsed {
        if !st.contains(&0) {
            let product: usize = st.iter().product();
            if product == ctx.input_elems {
                return Some(st.clone()); // standard reshape
            }
            if product > ctx.input_elems && product.is_multiple_of(ctx.input_elems) {
                return Some(st.clone()); // broadcast expansion (GQA)
            }
        }
    }

    // 3. Compiled shape with 0-sentinel resolution.
    if let Some(compiled) = ctx.compiled_shape {
        // Strategy A: Same-rank merge with runtime shape tensor.
        if let Some(ref rt) = parsed {
            if rt.len() == compiled.len() {
                let merged: Vec<usize> = compiled
                    .iter()
                    .zip(rt.iter())
                    .map(|(&c, &r)| if c > 0 { c } else { r })
                    .collect();
                if is_valid_reshape(&merged, ctx.input_elems) {
                    return Some(merged);
                }
            }
        }

        // Strategy B: Same-rank positional inheritance from input shapes.
        if ctx.input_shapes.first().map(|s| s.len()) == Some(compiled.len()) {
            let resolved = resolve_compiled_sentinels(compiled, ctx.input_elems, ctx.input_shapes);
            if is_valid_reshape(&resolved, ctx.input_elems) {
                return Some(resolved);
            }
        }

        // Strategy C: Single-zero from element count (any rank).
        let zero_count = compiled.iter().filter(|&&d| d == 0).count();
        if zero_count == 1 {
            let known: usize = compiled
                .iter()
                .filter(|&&d| d > 0)
                .product::<usize>()
                .max(1);
            if ctx.input_elems >= known && ctx.input_elems.is_multiple_of(known) {
                let unknown = ctx.input_elems / known;
                let resolved: Vec<usize> = compiled
                    .iter()
                    .map(|&d| if d == 0 { unknown } else { d })
                    .collect();
                if is_valid_reshape(&resolved, ctx.input_elems) {
                    return Some(resolved);
                }
            }
        }

        // Strategy D: Multi-zero — assume batch=1, then single-zero resolve.
        if zero_count >= 2 {
            let mut resolved = compiled.to_vec();
            if resolved[0] == 0 {
                resolved[0] = 1;
            }
            let remaining = resolved.iter().filter(|&&d| d == 0).count();
            if remaining == 1 {
                let known: usize = resolved
                    .iter()
                    .filter(|&&d| d > 0)
                    .product::<usize>()
                    .max(1);
                if ctx.input_elems >= known && ctx.input_elems.is_multiple_of(known) {
                    let unknown = ctx.input_elems / known;
                    for d in &mut resolved {
                        if *d == 0 {
                            *d = unknown;
                            break;
                        }
                    }
                }
            }
            if is_valid_reshape(&resolved, ctx.input_elems) {
                return Some(resolved);
            }
        }
    }

    // 4. Runtime shape tensor with zeros — try filling from compiled.
    if let Some(ref rt) = parsed {
        if let Some(compiled) = ctx.compiled_shape {
            if rt.len() == compiled.len() {
                let merged: Vec<usize> = rt
                    .iter()
                    .zip(compiled.iter())
                    .map(|(&r, &c)| if r > 0 { r } else { c })
                    .collect();
                if is_valid_reshape(&merged, ctx.input_elems) {
                    return Some(merged);
                }
            }
        }
        // Return parsed even if it has zeros — better than nothing.
        if !rt.is_empty() && !rt.iter().all(|&d| d == 0) {
            return Some(rt.clone());
        }
    }

    None
}

/// Check if a resolved reshape shape is valid.
fn is_valid_reshape(shape: &[usize], input_elems: usize) -> bool {
    if shape.contains(&0) {
        return false;
    }
    let product: usize = shape.iter().product();
    // Standard reshape or broadcast expansion.
    product == input_elems
        || (product > input_elems && input_elems > 0 && product.is_multiple_of(input_elems))
}

/// Resolve 0-sentinels in a compiled shape using positional inheritance and element count.
fn resolve_compiled_sentinels(
    compiled: &[usize],
    input_elems: usize,
    input_shapes: &[Vec<usize>],
) -> Vec<usize> {
    let mut resolved = compiled.to_vec();

    // Positional inheritance from input shapes.
    for (i, d) in resolved.iter_mut().enumerate() {
        if *d == 0 {
            for in_shape in input_shapes {
                if let Some(&in_dim) = in_shape.get(i) {
                    if in_dim > 0 {
                        *d = in_dim;
                        break;
                    }
                }
            }
        }
    }

    // Single remaining zero: resolve from element count.
    let remaining = resolved.iter().filter(|&&d| d == 0).count();
    if remaining == 1 {
        let known: usize = resolved
            .iter()
            .filter(|&&d| d > 0)
            .product::<usize>()
            .max(1);
        if input_elems >= known && input_elems.is_multiple_of(known) {
            let unknown = input_elems / known;
            for d in &mut resolved {
                if *d == 0 {
                    *d = unknown;
                    break;
                }
            }
        }
    } else if remaining >= 2 {
        // Assume batch=1 for first dim.
        if resolved[0] == 0 {
            resolved[0] = 1;
        }
        let still = resolved.iter().filter(|&&d| d == 0).count();
        if still == 1 {
            let known: usize = resolved
                .iter()
                .filter(|&&d| d > 0)
                .product::<usize>()
                .max(1);
            if input_elems >= known && input_elems.is_multiple_of(known) {
                let unknown = input_elems / known;
                for d in &mut resolved {
                    if *d == 0 {
                        *d = unknown;
                        break;
                    }
                }
            }
        }
    }

    resolved
}

// ── Shape tensor parsing ────────────────────────────────────────────────────

/// Parse a shape tensor (raw bytes) into a resolved `Vec<usize>`.
///
/// Tries i64 first, then i32. Converts -1 to 0 placeholder.
/// Resolves single -1 dim from `n_elems`.
pub fn parse_shape_values(shape_bytes: &[u8], n_elems: usize) -> Option<Vec<usize>> {
    if shape_bytes.is_empty() {
        return None;
    }

    let shape_vals: Vec<i64> = if shape_bytes.len().is_multiple_of(8) {
        let i64s: &[i64] = bytemuck::try_cast_slice(shape_bytes).ok()?;
        let reasonable = i64s.iter().all(|&v| v >= -1 && v <= n_elems as i64 + 1);
        if reasonable {
            i64s.to_vec()
        } else if shape_bytes.len().is_multiple_of(4) {
            let i32s: &[i32] = bytemuck::try_cast_slice(shape_bytes).ok()?;
            i32s.iter().map(|&v| v as i64).collect()
        } else {
            i64s.to_vec()
        }
    } else if shape_bytes.len().is_multiple_of(4) {
        let i32s: &[i32] = bytemuck::try_cast_slice(shape_bytes).ok()?;
        i32s.iter().map(|&v| v as i64).collect()
    } else {
        return None;
    };

    let shape: Vec<usize> = shape_vals
        .iter()
        .map(|&v| {
            if v == -1 || v == 0 {
                0 // placeholder
            } else if v < 0 {
                1
            } else {
                v as usize
            }
        })
        .collect();

    // Resolve single zero from element count.
    let zero_count = shape.iter().filter(|&&d| d == 0).count();
    if zero_count == 1 {
        let known: usize = shape.iter().filter(|&&d| d > 0).product::<usize>().max(1);
        let unknown = if known > 0 { n_elems / known } else { n_elems };
        Some(
            shape
                .iter()
                .map(|&d| if d == 0 { unknown } else { d })
                .collect(),
        )
    } else {
        Some(shape)
    }
}

// ── Transpose ───────────────────────────────────────────────────────────────

fn resolve_transpose(ctx: &ShapeContext<'_>, perm: &[u8; 8], ndim: u8) -> Option<Vec<usize>> {
    let in_shape = ctx.input_shapes.first()?;
    let nd = ndim as usize;
    if nd == 0 || in_shape.len() < nd {
        return None;
    }
    let p = &perm[..nd];
    if p.iter().any(|&pi| (pi as usize) >= in_shape.len()) {
        return None;
    }
    Some(p.iter().map(|&pi| in_shape[pi as usize]).collect())
}

// ── MatMul ──────────────────────────────────────────────────────────────────

fn resolve_matmul(ctx: &ShapeContext<'_>, k_hint: u32) -> Option<Vec<usize>> {
    if ctx.input_shapes.len() < 2 {
        return None;
    }
    let a_shape = &ctx.input_shapes[0];
    let b_shape = &ctx.input_shapes[1];

    if a_shape.is_empty() || b_shape.is_empty() {
        return None;
    }

    // Batched matmul: both >= 2-D.
    if a_shape.len() >= 2 && b_shape.len() >= 2 {
        let mut out = a_shape[..a_shape.len() - 1].to_vec();
        out.push(*b_shape.last().unwrap());
        if !out.contains(&0) {
            return Some(out);
        }
    }

    // 2D fallback using hints.
    let k = k_hint as usize;
    if k > 0 {
        let a_elems: usize = a_shape.iter().product();
        let b_elems: usize = b_shape.iter().product();
        if a_elems > 0 && b_elems > 0 {
            let m = a_elems / k;
            let n = b_elems / k;
            if m > 0 && n > 0 {
                return Some(vec![m, n]);
            }
        }
    }

    None
}

// ── Gemm ────────────────────────────────────────────────────────────────────

fn resolve_gemm(ctx: &ShapeContext<'_>, k: u32) -> Option<Vec<usize>> {
    if ctx.input_shapes.len() < 2 {
        return None;
    }
    let a_shape = &ctx.input_shapes[0];
    let b_shape = &ctx.input_shapes[1];
    let k = k as usize;
    if k > 0 {
        let a_elems: usize = a_shape.iter().product();
        let b_elems: usize = b_shape.iter().product();
        if a_elems > 0 && b_elems > 0 {
            let m = a_elems / k;
            let n = b_elems / k;
            if m > 0 && n > 0 {
                return Some(vec![m, n]);
            }
        }
    }
    None
}

// ── Concat ──────────────────────────────────────────────────────────────────

fn resolve_concat(ctx: &ShapeContext<'_>) -> Option<Vec<usize>> {
    if ctx.input_shapes.len() < 2 {
        return None;
    }
    let a = &ctx.input_shapes[0];
    let b = &ctx.input_shapes[1];
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut out = a.clone();
    if let Some(last) = out.last_mut() {
        if let Some(b_last) = b.last() {
            *last += b_last;
        }
    }
    Some(out)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_simple(input_shapes: &[Vec<usize>]) -> ShapeContext<'_> {
        let input_elems = input_shapes
            .first()
            .map(|s| s.iter().product::<usize>())
            .unwrap_or(0);
        ShapeContext {
            input_shapes,
            compiled_shape: None,
            input_elems,
            shape_tensor_bytes: None,
            compiled_dtype: None,
        }
    }

    #[test]
    fn test_standard_same_as() {
        let inputs = vec![vec![2, 3, 4]];
        let ctx = ctx_simple(&inputs);
        let result = resolve_standard(&ShapeSpec::SameAs(0), &ctx);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_standard_broadcast() {
        let inputs = vec![vec![1, 3, 4], vec![3, 1]];
        let ctx = ctx_simple(&inputs);
        let result = resolve_standard(&ShapeSpec::Broadcast(0, 1), &ctx);
        assert_eq!(result, Some(vec![1, 3, 4]));
    }

    #[test]
    fn test_standard_drop_last_dim() {
        let inputs = vec![vec![2, 3, 4]];
        let ctx = ctx_simple(&inputs);
        let result = resolve_standard(&ShapeSpec::DropLastDim(0), &ctx);
        assert_eq!(result, Some(vec![2, 3]));
    }

    #[test]
    fn test_standard_dims_inferred() {
        let inputs = vec![vec![6, 64]]; // 384 elements
        let ctx = ctx_simple(&inputs);
        let spec = ShapeSpec::Dims(vec![ShapeDim::Inferred, ShapeDim::Fixed(64)]);
        let result = resolve_standard(&spec, &ctx);
        assert_eq!(result, Some(vec![6, 64]));
    }

    #[test]
    fn test_transpose() {
        let inputs = vec![vec![2, 3, 4]];
        let ctx = ctx_simple(&inputs);
        let perm = [1, 2, 0, 0, 0, 0, 0, 0];
        let result = resolve_transpose(&ctx, &perm, 3);
        assert_eq!(result, Some(vec![3, 4, 2]));
    }

    #[test]
    fn test_matmul_batched() {
        let inputs = vec![vec![2, 3, 4], vec![2, 4, 5]];
        let ctx = ctx_simple(&inputs);
        let result = resolve_matmul(&ctx, 0);
        assert_eq!(result, Some(vec![2, 3, 5]));
    }

    #[test]
    fn test_concat() {
        let inputs = vec![vec![2, 3], vec![2, 5]];
        let ctx = ctx_simple(&inputs);
        let result = resolve_concat(&ctx);
        assert_eq!(result, Some(vec![2, 8]));
    }

    #[test]
    fn test_reshape_standard() {
        // 24 elements, reshape to [2, 3, 4]
        let shape_i64: Vec<i64> = vec![2, 3, 4];
        let shape_bytes: Vec<u8> = bytemuck::cast_slice(&shape_i64).to_vec();
        let inputs = vec![vec![24]];
        let ctx = ShapeContext {
            input_shapes: &inputs,
            compiled_shape: None,
            input_elems: 24,
            shape_tensor_bytes: Some(&shape_bytes),
            compiled_dtype: None,
        };
        let result = resolve_reshape(&ctx);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_reshape_with_neg1() {
        // 24 elements, shape tensor = [2, -1, 4] → [2, 3, 4]
        let shape_i64: Vec<i64> = vec![2, -1, 4];
        let shape_bytes: Vec<u8> = bytemuck::cast_slice(&shape_i64).to_vec();
        let inputs = vec![vec![24]];
        let ctx = ShapeContext {
            input_shapes: &inputs,
            compiled_shape: None,
            input_elems: 24,
            shape_tensor_bytes: Some(&shape_bytes),
            compiled_dtype: None,
        };
        let result = resolve_reshape(&ctx);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_reshape_broadcast_expansion() {
        // GQA: 512 input elements, shape tensor = [1, 2, 4, 8, 1, 2, 64]
        // product = 8192, 8192 / 512 = 16 (broadcast factor)
        let shape_i64: Vec<i64> = vec![1, 2, 4, 8, 1, 2, 64];
        let shape_bytes: Vec<u8> = bytemuck::cast_slice(&shape_i64).to_vec();
        let inputs = vec![vec![512]];
        let ctx = ShapeContext {
            input_shapes: &inputs,
            compiled_shape: None,
            input_elems: 512,
            shape_tensor_bytes: Some(&shape_bytes),
            compiled_dtype: None,
        };
        let result = resolve_reshape(&ctx);
        assert_eq!(result, Some(vec![1, 2, 4, 8, 1, 2, 64]));
    }

    #[test]
    fn test_reshape_compiled_single_zero() {
        // compiled = [0, 3, 4], input_elems = 24, input = [2, 12]
        let compiled = vec![0, 3, 4];
        let inputs = vec![vec![2, 12]];
        let ctx = ShapeContext {
            input_shapes: &inputs,
            compiled_shape: Some(&compiled),
            input_elems: 24,
            shape_tensor_bytes: None,
            compiled_dtype: None,
        };
        let result = resolve_reshape(&ctx);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_reshape_compiled_multi_zero() {
        // compiled = [0, 0, 32, 64], input = [1, 6, 2048]
        // input_elems = 1*6*2048 = 12288
        let compiled = vec![0, 0, 32, 64];
        let inputs = vec![vec![1, 6, 2048]];
        let ctx = ShapeContext {
            input_shapes: &inputs,
            compiled_shape: Some(&compiled),
            input_elems: 12288,
            shape_tensor_bytes: None,
            compiled_dtype: None,
        };
        let result = resolve_reshape(&ctx);
        assert_eq!(result, Some(vec![1, 6, 32, 64]));
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[2, 3], &[3]), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[5], &[1, 5]), vec![1, 5]);
        assert_eq!(broadcast_shapes(&[1, 1, 3], &[2, 1]), vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_shape_values_i64() {
        let vals: Vec<i64> = vec![2, 3, 4];
        let bytes: Vec<u8> = bytemuck::cast_slice(&vals).to_vec();
        let result = parse_shape_values(&bytes, 24);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_parse_shape_values_neg1() {
        let vals: Vec<i64> = vec![2, -1, 4];
        let bytes: Vec<u8> = bytemuck::cast_slice(&vals).to_vec();
        let result = parse_shape_values(&bytes, 24);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }
}
