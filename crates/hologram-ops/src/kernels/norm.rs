//! Canonical normalisation ops (`RmsNorm`, `LayerNorm`, `InstanceNorm`,
//! `GroupNorm`, `AddRmsNorm`) — semantic identity, executable form,
//! and CPU reference kernels.
//!
//! Each kernel iterates the input as rows of `size` (or `num_groups`)
//! elements, computes the relevant statistics, and writes the
//! normalised result. Reference behaviour only.

use crate::attrs::{GroupNormAttrs, NormAttrs};
use crate::span::SlotSpan;
use crate::trait_def::{BackwardRule, Op, OpCategory};

// `RmsNorm` and `LayerNorm` carry backward rules — declared explicitly.
/// Marker struct for the canonical `rms_norm` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNorm(pub NormAttrs);

impl Op for RmsNorm {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "rms_norm"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Normalisation
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::RmsNormBackward)
    }
}

/// Marker struct for the canonical `layer_norm` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerNorm(pub NormAttrs);

impl Op for LayerNorm {
    #[inline]
    fn arity(self) -> u8 {
        3
    }
    #[inline]
    fn name(self) -> &'static str {
        "layer_norm"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Normalisation
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::LayerNormBackward)
    }
}

/// Marker struct for the canonical `instance_norm` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceNorm(pub NormAttrs);

impl Op for InstanceNorm {
    #[inline]
    fn arity(self) -> u8 {
        2
    }
    #[inline]
    fn name(self) -> &'static str {
        "instance_norm"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Normalisation
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::InstanceNormBackward)
    }
}

/// Marker struct for the canonical `add_rms_norm` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AddRmsNorm(pub NormAttrs);

impl Op for AddRmsNorm {
    #[inline]
    fn arity(self) -> u8 {
        3
    }
    #[inline]
    fn name(self) -> &'static str {
        "add_rms_norm"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Normalisation
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::AddRmsNormBackward)
    }
}

/// Marker struct for the canonical `group_norm` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupNorm(pub GroupNormAttrs);

impl Op for GroupNorm {
    #[inline]
    fn arity(self) -> u8 {
        3
    }
    #[inline]
    fn name(self) -> &'static str {
        "group_norm"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Normalisation
    }
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        Some(BackwardRule::GroupNormBackward)
    }
}

/// Pre-resolved arguments for a 2-input weight-only norm (`RmsNorm`,
/// `InstanceNorm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormScaleCall {
    /// Input span.
    pub input: SlotSpan,
    /// Per-axis scale weight.
    pub weight: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Pre-resolved arguments for a 3-input scale+bias norm (`LayerNorm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormFullCall {
    /// Input span.
    pub input: SlotSpan,
    /// Per-axis scale.
    pub weight: SlotSpan,
    /// Per-axis bias.
    pub bias: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Pre-resolved arguments for `GroupNorm`.
///
/// Reference behaviour: treats the input as `num_groups` rows each of
/// length `group_elements = input.len / num_groups`, normalises each
/// row, then scales by `weight` and adds `bias` of length
/// `group_elements`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupNormCall {
    /// Input span.
    pub input: SlotSpan,
    /// Scale weight.
    pub weight: SlotSpan,
    /// Bias.
    pub bias: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Number of groups.
    pub num_groups: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Pre-resolved arguments for `AddRmsNorm` (residual add + RMS norm).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddRmsNormCall {
    /// First addend (residual).
    pub residual: SlotSpan,
    /// Second addend.
    pub input: SlotSpan,
    /// Per-axis scale weight.
    pub weight: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

#[inline]
fn epsilon_f32(bits: u32) -> f32 {
    f32::from_bits(bits)
}

#[inline]
fn rows(span: SlotSpan, size: u32) -> usize {
    debug_assert!(size > 0);
    debug_assert_eq!(span.len % size as usize, 0);
    span.len / size as usize
}

/// Forward: `out[r,i] = x[r,i] / sqrt(mean(x[r,:]²) + eps) * weight[i]`.
#[inline]
pub fn rms_norm(storage: &mut [f32], call: &NormScaleCall) {
    let size = call.size as usize;
    let n_rows = rows(call.input, call.size);
    let eps = epsilon_f32(call.epsilon);
    for r in 0..n_rows {
        let in_off = call.input.offset + r * size;
        let out_off = call.output.offset + r * size;
        let scale = rms_scale(storage, in_off, size, eps);
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            storage[out_off + i] = storage[in_off + i] * scale * w;
        }
    }
}

#[inline]
fn rms_scale(storage: &[f32], in_off: usize, size: usize, eps: f32) -> f32 {
    let mut sq = 0.0_f32;
    for i in 0..size {
        let v = storage[in_off + i];
        sq += v * v;
    }
    let mean_sq = sq / size as f32;
    1.0 / libm::sqrtf(mean_sq + eps)
}

/// Forward: `out[r,i] = (x[r,i] - mean) / sqrt(var + eps) * weight[i] + bias[i]`.
#[inline]
pub fn layer_norm(storage: &mut [f32], call: &NormFullCall) {
    let size = call.size as usize;
    let n_rows = rows(call.input, call.size);
    let eps = epsilon_f32(call.epsilon);
    for r in 0..n_rows {
        let in_off = call.input.offset + r * size;
        let out_off = call.output.offset + r * size;
        let (mean, scale) = mean_var_scale(storage, in_off, size, eps);
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let b = storage[call.bias.offset + i];
            storage[out_off + i] = (storage[in_off + i] - mean) * scale * w + b;
        }
    }
}

/// Forward: like `layer_norm` but without bias (just `weight`).
#[inline]
pub fn instance_norm(storage: &mut [f32], call: &NormScaleCall) {
    let size = call.size as usize;
    let n_rows = rows(call.input, call.size);
    let eps = epsilon_f32(call.epsilon);
    for r in 0..n_rows {
        let in_off = call.input.offset + r * size;
        let out_off = call.output.offset + r * size;
        let (mean, scale) = mean_var_scale(storage, in_off, size, eps);
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            storage[out_off + i] = (storage[in_off + i] - mean) * scale * w;
        }
    }
}

#[inline]
fn mean_var_scale(storage: &[f32], in_off: usize, size: usize, eps: f32) -> (f32, f32) {
    let mut sum = 0.0_f32;
    for i in 0..size {
        sum += storage[in_off + i];
    }
    let mean = sum / size as f32;
    let mut var = 0.0_f32;
    for i in 0..size {
        let d = storage[in_off + i] - mean;
        var += d * d;
    }
    var /= size as f32;
    (mean, 1.0 / libm::sqrtf(var + eps))
}

/// Forward: GroupNorm. Treats the input as `num_groups` blocks of equal
/// length, normalises each block, then applies per-element `weight` and
/// `bias` of length `input.len / num_groups`.
#[inline]
pub fn group_norm(storage: &mut [f32], call: &GroupNormCall) {
    let group_elems = call.input.len / call.num_groups as usize;
    debug_assert!(group_elems > 0);
    debug_assert_eq!(call.input.len % call.num_groups as usize, 0);
    let eps = epsilon_f32(call.epsilon);
    for g in 0..call.num_groups as usize {
        let in_off = call.input.offset + g * group_elems;
        let out_off = call.output.offset + g * group_elems;
        let (mean, scale) = mean_var_scale(storage, in_off, group_elems, eps);
        for i in 0..group_elems {
            let w = storage[call.weight.offset + i];
            let b = storage[call.bias.offset + i];
            storage[out_off + i] = (storage[in_off + i] - mean) * scale * w + b;
        }
    }
}

/// Forward: `out = rms_norm(residual + input, weight)` row-wise.
#[inline]
pub fn add_rms_norm(storage: &mut [f32], call: &AddRmsNormCall) {
    let size = call.size as usize;
    let rows_total = rows(call.input, call.size);
    let eps = epsilon_f32(call.epsilon);
    for r in 0..rows_total {
        let res_off = call.residual.offset + r * size;
        let in_off = call.input.offset + r * size;
        let out_off = call.output.offset + r * size;
        let mut sq = 0.0_f32;
        for i in 0..size {
            let v = storage[res_off + i] + storage[in_off + i];
            storage[out_off + i] = v;
            sq += v * v;
        }
        let mean_sq = sq / size as f32;
        let scale = 1.0 / libm::sqrtf(mean_sq + eps);
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            storage[out_off + i] = storage[out_off + i] * scale * w;
        }
    }
}

/// Pre-resolved arguments for `RmsNorm` backward.
///
/// Computes `dx`, `dw` from upstream `dy`. `dx[r,i] = w[i] * dy[r,i] *
/// rstd - x[r,i] * rstd³ / size * Σ_j(dy[r,j] * w[j] * x[r,j])` and
/// `dw[i] = Σ_r(dy[r,i] * x[r,i] * rstd[r])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormGradCall {
    /// Forward input `x`.
    pub input: SlotSpan,
    /// Forward weight `w` (length = `size`).
    pub weight: SlotSpan,
    /// Upstream gradient `dy` (length = `input.len`).
    pub dy: SlotSpan,
    /// Gradient slot for `x` (length = `input.len`).
    pub dx: SlotSpan,
    /// Gradient slot for `w` (length = `size`).
    pub dw: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Pre-resolved arguments for `LayerNorm` backward.
///
/// Computes `dx`, `dw`, `db`. `dx[r,i] = rstd * (dx̂[i] -
/// mean(dx̂) - x̂[i] * mean(dx̂ * x̂))` where `dx̂[i] = dy[i] * w[i]`
/// and `x̂[i] = (x[i]-μ) * rstd`. `dw[i] = Σ_r(dy[r,i] * x̂[r,i])`,
/// `db[i] = Σ_r(dy[r,i])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerNormGradCall {
    /// Forward input `x`.
    pub input: SlotSpan,
    /// Forward weight `w`.
    pub weight: SlotSpan,
    /// Upstream gradient `dy`.
    pub dy: SlotSpan,
    /// Gradient slot for `x`.
    pub dx: SlotSpan,
    /// Gradient slot for `w` (length = `size`).
    pub dw: SlotSpan,
    /// Gradient slot for `b` (length = `size`).
    pub db: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Backward of `rms_norm`. Accumulates into `dx` and `dw`.
pub fn rms_norm_grad(storage: &mut [f32], call: &RmsNormGradCall) {
    let size = call.size as usize;
    if size == 0 {
        return;
    }
    let n_rows = call.input.len / size;
    let eps = epsilon_f32(call.epsilon);
    let inv_size = 1.0 / size as f32;
    for r in 0..n_rows {
        let in_off = call.input.offset + r * size;
        let dy_off = call.dy.offset + r * size;
        let dx_off = call.dx.offset + r * size;
        let mut sq = 0.0_f32;
        for i in 0..size {
            let v = storage[in_off + i];
            sq += v * v;
        }
        let mean_sq = sq * inv_size;
        let rstd = 1.0 / libm::sqrtf(mean_sq + eps);
        let mut dot = 0.0_f32;
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let x = storage[in_off + i];
            dot += storage[dy_off + i] * w * x;
        }
        let dot_term = rstd * rstd * rstd * inv_size * dot;
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let x = storage[in_off + i];
            let dy = storage[dy_off + i];
            let contrib = w * dy * rstd - x * dot_term;
            if call.dx.len > 0 {
                storage[dx_off + i] += contrib;
            }
            if call.dw.len > 0 {
                storage[call.dw.offset + i] += dy * x * rstd;
            }
        }
    }
}

/// Backward of `layer_norm`. Accumulates into `dx`, `dw`, and `db`.
pub fn layer_norm_grad(storage: &mut [f32], call: &LayerNormGradCall) {
    let size = call.size as usize;
    if size == 0 {
        return;
    }
    let n_rows = call.input.len / size;
    let eps = epsilon_f32(call.epsilon);
    let inv_size = 1.0 / size as f32;
    for r in 0..n_rows {
        let in_off = call.input.offset + r * size;
        let dy_off = call.dy.offset + r * size;
        let dx_off = call.dx.offset + r * size;
        let mut sum = 0.0_f32;
        for i in 0..size {
            sum += storage[in_off + i];
        }
        let mean = sum * inv_size;
        let mut var = 0.0_f32;
        for i in 0..size {
            let d = storage[in_off + i] - mean;
            var += d * d;
        }
        var *= inv_size;
        let rstd = 1.0 / libm::sqrtf(var + eps);
        // x_hat[i] = (x[i]-mean)*rstd; dx_hat[i] = dy[i]*w[i].
        let mut sum_dx_hat = 0.0_f32;
        let mut sum_dx_hat_x_hat = 0.0_f32;
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let x_hat = (storage[in_off + i] - mean) * rstd;
            let dx_hat = storage[dy_off + i] * w;
            sum_dx_hat += dx_hat;
            sum_dx_hat_x_hat += dx_hat * x_hat;
        }
        let mean_dx_hat = sum_dx_hat * inv_size;
        let mean_dx_hat_x_hat = sum_dx_hat_x_hat * inv_size;
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let x_hat = (storage[in_off + i] - mean) * rstd;
            let dy = storage[dy_off + i];
            let dx_hat = dy * w;
            if call.dx.len > 0 {
                storage[dx_off + i] += rstd * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat);
            }
            if call.dw.len > 0 {
                storage[call.dw.offset + i] += dy * x_hat;
            }
            if call.db.len > 0 {
                storage[call.db.offset + i] += dy;
            }
        }
    }
}

/// Pre-resolved arguments for `InstanceNorm` backward.
///
/// Same closed form as `LayerNorm` but without a bias gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceNormGradCall {
    /// Forward input `x`.
    pub input: SlotSpan,
    /// Forward weight `w`.
    pub weight: SlotSpan,
    /// Upstream gradient `dy`.
    pub dy: SlotSpan,
    /// Gradient slot for `x`.
    pub dx: SlotSpan,
    /// Gradient slot for `w`.
    pub dw: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Pre-resolved arguments for `AddRmsNorm` backward.
///
/// Forward: `out = rms_norm(residual + input, weight)`. The partials
/// w.r.t. `residual` and `input` are identical (the sum is symmetric
/// in the two operands), so the kernel writes the same value into
/// both gradient slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddRmsNormGradCall {
    /// Forward residual operand.
    pub residual: SlotSpan,
    /// Forward input operand.
    pub input: SlotSpan,
    /// Forward weight `w`.
    pub weight: SlotSpan,
    /// Upstream gradient `dy`.
    pub dy: SlotSpan,
    /// Gradient slot for the residual operand.
    pub d_residual: SlotSpan,
    /// Gradient slot for the input operand.
    pub d_input: SlotSpan,
    /// Gradient slot for `w`.
    pub dw: SlotSpan,
    /// Length of the normalised (last) axis.
    pub size: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Pre-resolved arguments for `GroupNorm` backward.
///
/// Per-group statistics over `group_elements = input.len /
/// num_groups`; `weight` and `bias` have length `group_elements`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupNormGradCall {
    /// Forward input `x`.
    pub input: SlotSpan,
    /// Forward weight `w` (length = `group_elements`).
    pub weight: SlotSpan,
    /// Upstream gradient `dy` (length = `input.len`).
    pub dy: SlotSpan,
    /// Gradient slot for `x` (length = `input.len`).
    pub dx: SlotSpan,
    /// Gradient slot for `w` (length = `group_elements`).
    pub dw: SlotSpan,
    /// Gradient slot for `b` (length = `group_elements`).
    pub db: SlotSpan,
    /// Number of groups.
    pub num_groups: u32,
    /// Stabilisation epsilon as `f32::to_bits()`.
    pub epsilon: u32,
}

/// Backward of `group_norm`. Closed form (same shape as
/// `layer_norm_grad`) with the reduction taken over each group's
/// `group_elements` window.
pub fn group_norm_grad(storage: &mut [f32], call: &GroupNormGradCall) {
    let groups = call.num_groups as usize;
    if groups == 0 || call.input.len == 0 {
        return;
    }
    let group_elems = call.input.len / groups;
    if group_elems == 0 {
        return;
    }
    debug_assert_eq!(call.input.len % groups, 0);
    let eps = epsilon_f32(call.epsilon);
    let inv_size = 1.0 / group_elems as f32;
    for g in 0..groups {
        let in_off = call.input.offset + g * group_elems;
        let dy_off = call.dy.offset + g * group_elems;
        let dx_off = call.dx.offset + g * group_elems;
        let mut sum = 0.0_f32;
        for i in 0..group_elems {
            sum += storage[in_off + i];
        }
        let mean = sum * inv_size;
        let mut var = 0.0_f32;
        for i in 0..group_elems {
            let d = storage[in_off + i] - mean;
            var += d * d;
        }
        var *= inv_size;
        let rstd = 1.0 / libm::sqrtf(var + eps);
        let mut sum_dx_hat = 0.0_f32;
        let mut sum_dx_hat_x_hat = 0.0_f32;
        for i in 0..group_elems {
            let w = storage[call.weight.offset + i];
            let x_hat = (storage[in_off + i] - mean) * rstd;
            let dx_hat = storage[dy_off + i] * w;
            sum_dx_hat += dx_hat;
            sum_dx_hat_x_hat += dx_hat * x_hat;
        }
        let mean_dx_hat = sum_dx_hat * inv_size;
        let mean_dx_hat_x_hat = sum_dx_hat_x_hat * inv_size;
        for i in 0..group_elems {
            let w = storage[call.weight.offset + i];
            let x_hat = (storage[in_off + i] - mean) * rstd;
            let dy = storage[dy_off + i];
            let dx_hat = dy * w;
            if call.dx.len > 0 {
                storage[dx_off + i] += rstd * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat);
            }
            if call.dw.len > 0 {
                storage[call.dw.offset + i] += dy * x_hat;
            }
            if call.db.len > 0 {
                storage[call.db.offset + i] += dy;
            }
        }
    }
}

/// Backward of `instance_norm`. Mirrors `layer_norm_grad` but skips
/// the bias path.
pub fn instance_norm_grad(storage: &mut [f32], call: &InstanceNormGradCall) {
    let size = call.size as usize;
    if size == 0 {
        return;
    }
    let n_rows = call.input.len / size;
    let eps = epsilon_f32(call.epsilon);
    let inv_size = 1.0 / size as f32;
    for r in 0..n_rows {
        let in_off = call.input.offset + r * size;
        let dy_off = call.dy.offset + r * size;
        let dx_off = call.dx.offset + r * size;
        let mut sum = 0.0_f32;
        for i in 0..size {
            sum += storage[in_off + i];
        }
        let mean = sum * inv_size;
        let mut var = 0.0_f32;
        for i in 0..size {
            let d = storage[in_off + i] - mean;
            var += d * d;
        }
        var *= inv_size;
        let rstd = 1.0 / libm::sqrtf(var + eps);
        let mut sum_dx_hat = 0.0_f32;
        let mut sum_dx_hat_x_hat = 0.0_f32;
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let x_hat = (storage[in_off + i] - mean) * rstd;
            let dx_hat = storage[dy_off + i] * w;
            sum_dx_hat += dx_hat;
            sum_dx_hat_x_hat += dx_hat * x_hat;
        }
        let mean_dx_hat = sum_dx_hat * inv_size;
        let mean_dx_hat_x_hat = sum_dx_hat_x_hat * inv_size;
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let x_hat = (storage[in_off + i] - mean) * rstd;
            let dy = storage[dy_off + i];
            let dx_hat = dy * w;
            if call.dx.len > 0 {
                storage[dx_off + i] += rstd * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat);
            }
            if call.dw.len > 0 {
                storage[call.dw.offset + i] += dy * x_hat;
            }
        }
    }
}

/// Backward of `add_rms_norm`. Recomputes the row sum `s = residual +
/// input` and applies the standard RMS-norm gradient with respect to
/// `s`; both `d_residual` and `d_input` receive the same value.
pub fn add_rms_norm_grad(storage: &mut [f32], call: &AddRmsNormGradCall) {
    let size = call.size as usize;
    if size == 0 {
        return;
    }
    let n_rows = call.input.len / size;
    let eps = epsilon_f32(call.epsilon);
    let inv_size = 1.0 / size as f32;
    for r in 0..n_rows {
        let res_off = call.residual.offset + r * size;
        let in_off = call.input.offset + r * size;
        let dy_off = call.dy.offset + r * size;
        let dres_off = call.d_residual.offset + r * size;
        let din_off = call.d_input.offset + r * size;
        let mut sq = 0.0_f32;
        for i in 0..size {
            let s = storage[res_off + i] + storage[in_off + i];
            sq += s * s;
        }
        let mean_sq = sq * inv_size;
        let rstd = 1.0 / libm::sqrtf(mean_sq + eps);
        let mut dot = 0.0_f32;
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let s = storage[res_off + i] + storage[in_off + i];
            dot += storage[dy_off + i] * w * s;
        }
        let dot_term = rstd * rstd * rstd * inv_size * dot;
        for i in 0..size {
            let w = storage[call.weight.offset + i];
            let s = storage[res_off + i] + storage[in_off + i];
            let dy = storage[dy_off + i];
            let contrib = w * dy * rstd - s * dot_term;
            if call.d_residual.len > 0 {
                storage[dres_off + i] += contrib;
            }
            if call.d_input.len > 0 {
                storage[din_off + i] += contrib;
            }
            if call.dw.len > 0 {
                storage[call.dw.offset + i] += dy * s * rstd;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(off: usize, len: usize) -> SlotSpan {
        SlotSpan { offset: off, len }
    }

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "{} vs {}", a, b);
    }

    #[test]
    fn rms_norm_with_unit_weight_normalises_row_to_unit_rms() {
        let mut s = [3.0_f32, 4.0, 1.0, 1.0, 0.0, 0.0];
        let call = NormScaleCall {
            input: span(0, 2),
            weight: span(2, 2),
            output: span(4, 2),
            size: 2,
            epsilon: 0,
        };
        rms_norm(&mut s, &call);
        let scale = 1.0 / libm::sqrtf(12.5);
        approx_eq(s[4], 3.0 * scale, 1e-5);
        approx_eq(s[5], 4.0 * scale, 1e-5);
    }

    #[test]
    fn layer_norm_zeros_after_mean_subtract() {
        let mut s = [
            1.0_f32, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let call = NormFullCall {
            input: span(0, 3),
            weight: span(3, 3),
            bias: span(6, 3),
            output: span(9, 3),
            size: 3,
            epsilon: 0,
        };
        layer_norm(&mut s, &call);
        let scale = 1.0 / libm::sqrtf(2.0 / 3.0);
        approx_eq(s[9], -scale, 1e-5);
        approx_eq(s[10], 0.0, 1e-5);
        approx_eq(s[11], scale, 1e-5);
    }

    #[test]
    fn instance_norm_matches_layer_norm_without_bias() {
        let mut s = [1.0_f32, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let call = NormScaleCall {
            input: span(0, 3),
            weight: span(3, 3),
            output: span(6, 3),
            size: 3,
            epsilon: 0,
        };
        instance_norm(&mut s, &call);
        let scale = 1.0 / libm::sqrtf(2.0 / 3.0);
        approx_eq(s[6], -scale, 1e-5);
        approx_eq(s[7], 0.0, 1e-5);
        approx_eq(s[8], scale, 1e-5);
    }

    #[test]
    fn group_norm_normalises_each_group() {
        let mut s = [
            1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ];
        let call = GroupNormCall {
            input: span(0, 6),
            weight: span(6, 3),
            bias: span(9, 3),
            output: span(12, 6),
            num_groups: 2,
            epsilon: 0,
        };
        group_norm(&mut s, &call);
        let scale = 1.0 / libm::sqrtf(2.0 / 3.0);
        for r in 0..2 {
            approx_eq(s[12 + r * 3], -scale, 1e-4);
            approx_eq(s[13 + r * 3], 0.0, 1e-4);
            approx_eq(s[14 + r * 3], scale, 1e-4);
        }
    }

    #[test]
    fn add_rms_norm_sums_then_normalises() {
        let mut s = [1.0_f32, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0];
        let call = AddRmsNormCall {
            residual: span(0, 2),
            input: span(2, 2),
            weight: span(4, 2),
            output: span(6, 2),
            size: 2,
            epsilon: 0,
        };
        add_rms_norm(&mut s, &call);
        let scale = 1.0 / libm::sqrtf(12.5);
        approx_eq(s[6], 3.0 * scale, 1e-5);
        approx_eq(s[7], 4.0 * scale, 1e-5);
    }

    // ── Backward kernels: cross-checked against finite differences ───

    fn run_rms_norm_forward(x: &[f32], w: &[f32], size: usize, eps: u32) -> Vec<f32> {
        let n = x.len();
        let mut s = vec![0.0_f32; n + w.len() + n];
        s[..n].copy_from_slice(x);
        s[n..n + w.len()].copy_from_slice(w);
        let call = NormScaleCall {
            input: span(0, n),
            weight: span(n, w.len()),
            output: span(n + w.len(), n),
            size: size as u32,
            epsilon: eps,
        };
        rms_norm(&mut s, &call);
        s[n + w.len()..].to_vec()
    }

    fn run_layer_norm_forward(x: &[f32], w: &[f32], b: &[f32], size: usize, eps: u32) -> Vec<f32> {
        let n = x.len();
        let mut s = vec![0.0_f32; n + w.len() + b.len() + n];
        s[..n].copy_from_slice(x);
        s[n..n + w.len()].copy_from_slice(w);
        s[n + w.len()..n + w.len() + b.len()].copy_from_slice(b);
        let call = NormFullCall {
            input: span(0, n),
            weight: span(n, w.len()),
            bias: span(n + w.len(), b.len()),
            output: span(n + w.len() + b.len(), n),
            size: size as u32,
            epsilon: eps,
        };
        layer_norm(&mut s, &call);
        s[n + w.len() + b.len()..].to_vec()
    }

    fn fd_grad_input<F: Fn(&[f32]) -> Vec<f32>>(f: F, x: &[f32], dy: &[f32], h: f32) -> Vec<f32> {
        let n = x.len();
        let mut g = vec![0.0_f32; n];
        for i in 0..n {
            let mut xp = x.to_vec();
            xp[i] += h;
            let yp = f(&xp);
            let mut xm = x.to_vec();
            xm[i] -= h;
            let ym = f(&xm);
            let mut s = 0.0_f32;
            for k in 0..yp.len() {
                s += (yp[k] - ym[k]) / (2.0 * h) * dy[k];
            }
            g[i] = s;
        }
        g
    }

    #[test]
    fn rms_norm_grad_matches_finite_difference() {
        let x = [0.5_f32, -0.3, 1.2, 0.7, -1.0, 0.4];
        let w = [1.1_f32, 0.9, 1.3];
        let dy = [0.2_f32, -0.4, 0.1, 0.3, 0.5, -0.2];
        let size = 3;
        let eps_bits = 1.0e-5_f32.to_bits();

        // Analytic grad.
        let n = x.len();
        let mut s = vec![0.0_f32; n + w.len() + n + n + w.len()];
        s[..n].copy_from_slice(&x);
        s[n..n + w.len()].copy_from_slice(&w);
        s[n + w.len()..n + w.len() + n].copy_from_slice(&dy);
        let call = RmsNormGradCall {
            input: span(0, n),
            weight: span(n, w.len()),
            dy: span(n + w.len(), n),
            dx: span(n + w.len() + n, n),
            dw: span(n + 2 * n + w.len(), w.len()),
            size: size as u32,
            epsilon: eps_bits,
        };
        rms_norm_grad(&mut s, &call);
        let dx = &s[n + w.len() + n..n + w.len() + n + n];
        let dw = &s[n + 2 * n + w.len()..];

        // FD over x.
        let fd_x = fd_grad_input(
            |xp| run_rms_norm_forward(xp, &w, size, eps_bits),
            &x,
            &dy,
            1e-3,
        );
        for i in 0..n {
            approx_eq(dx[i], fd_x[i], 1e-2);
        }
        // FD over w.
        let fd_w = fd_grad_input(
            |wp| run_rms_norm_forward(&x, wp, size, eps_bits),
            &w,
            &dy,
            1e-3,
        );
        for i in 0..w.len() {
            approx_eq(dw[i], fd_w[i], 1e-2);
        }
    }

    #[test]
    fn layer_norm_grad_matches_finite_difference() {
        let x = [0.5_f32, -0.3, 1.2, 0.7, -1.0, 0.4];
        let w = [1.1_f32, 0.9, 1.3];
        let b = [0.05_f32, -0.1, 0.2];
        let dy = [0.2_f32, -0.4, 0.1, 0.3, 0.5, -0.2];
        let size = 3;
        let eps_bits = 1.0e-5_f32.to_bits();

        let n = x.len();
        let mut s = vec![0.0_f32; n + w.len() + b.len() + n + n + w.len() + b.len()];
        s[..n].copy_from_slice(&x);
        s[n..n + w.len()].copy_from_slice(&w);
        s[n + w.len()..n + w.len() + b.len()].copy_from_slice(&b);
        s[n + w.len() + b.len()..n + w.len() + b.len() + n].copy_from_slice(&dy);
        let dx_off = n + w.len() + b.len() + n;
        let dw_off = dx_off + n;
        let db_off = dw_off + w.len();
        let call = LayerNormGradCall {
            input: span(0, n),
            weight: span(n, w.len()),
            dy: span(n + w.len() + b.len(), n),
            dx: span(dx_off, n),
            dw: span(dw_off, w.len()),
            db: span(db_off, b.len()),
            size: size as u32,
            epsilon: eps_bits,
        };
        layer_norm_grad(&mut s, &call);
        let dx = s[dx_off..dx_off + n].to_vec();
        let dw = s[dw_off..dw_off + w.len()].to_vec();
        let db = s[db_off..db_off + b.len()].to_vec();

        let fd_x = fd_grad_input(
            |xp| run_layer_norm_forward(xp, &w, &b, size, eps_bits),
            &x,
            &dy,
            1e-3,
        );
        for i in 0..n {
            approx_eq(dx[i], fd_x[i], 1e-2);
        }
        let fd_w = fd_grad_input(
            |wp| run_layer_norm_forward(&x, wp, &b, size, eps_bits),
            &w,
            &dy,
            1e-3,
        );
        for i in 0..w.len() {
            approx_eq(dw[i], fd_w[i], 1e-2);
        }
        let fd_b = fd_grad_input(
            |bp| run_layer_norm_forward(&x, &w, bp, size, eps_bits),
            &b,
            &dy,
            1e-3,
        );
        for i in 0..b.len() {
            approx_eq(db[i], fd_b[i], 1e-2);
        }
    }

    fn run_group_norm_forward(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        num_groups: u32,
        eps: u32,
    ) -> Vec<f32> {
        let n = x.len();
        let mut s = vec![0.0_f32; n + w.len() + b.len() + n];
        s[..n].copy_from_slice(x);
        s[n..n + w.len()].copy_from_slice(w);
        s[n + w.len()..n + w.len() + b.len()].copy_from_slice(b);
        let call = GroupNormCall {
            input: span(0, n),
            weight: span(n, w.len()),
            bias: span(n + w.len(), b.len()),
            output: span(n + w.len() + b.len(), n),
            num_groups,
            epsilon: eps,
        };
        group_norm(&mut s, &call);
        s[n + w.len() + b.len()..].to_vec()
    }

    #[test]
    fn group_norm_grad_matches_finite_difference() {
        // 6 input elements, 2 groups → group_elements = 3.
        let x = [0.5_f32, -0.3, 1.2, 0.7, -1.0, 0.4];
        let w = [1.1_f32, 0.9, 1.3];
        let b = [0.05_f32, -0.1, 0.2];
        let dy = [0.2_f32, -0.4, 0.1, 0.3, 0.5, -0.2];
        let num_groups = 2;
        let eps_bits = 1.0e-5_f32.to_bits();

        let n = x.len();
        let mut s = vec![0.0_f32; n + w.len() + n + n + w.len() + b.len()];
        s[..n].copy_from_slice(&x);
        s[n..n + w.len()].copy_from_slice(&w);
        s[n + w.len()..n + w.len() + n].copy_from_slice(&dy);
        let dx_off = n + w.len() + n;
        let dw_off = dx_off + n;
        let db_off = dw_off + w.len();
        let call = GroupNormGradCall {
            input: span(0, n),
            weight: span(n, w.len()),
            dy: span(n + w.len(), n),
            dx: span(dx_off, n),
            dw: span(dw_off, w.len()),
            db: span(db_off, b.len()),
            num_groups,
            epsilon: eps_bits,
        };
        group_norm_grad(&mut s, &call);
        let dx = s[dx_off..dx_off + n].to_vec();
        let dw = s[dw_off..dw_off + w.len()].to_vec();
        let db = s[db_off..db_off + b.len()].to_vec();

        let fd_x = fd_grad_input(
            |xp| run_group_norm_forward(xp, &w, &b, num_groups, eps_bits),
            &x,
            &dy,
            1e-3,
        );
        for i in 0..n {
            approx_eq(dx[i], fd_x[i], 1e-2);
        }
        let fd_w = fd_grad_input(
            |wp| run_group_norm_forward(&x, wp, &b, num_groups, eps_bits),
            &w,
            &dy,
            1e-3,
        );
        for i in 0..w.len() {
            approx_eq(dw[i], fd_w[i], 1e-2);
        }
        let fd_b = fd_grad_input(
            |bp| run_group_norm_forward(&x, &w, bp, num_groups, eps_bits),
            &b,
            &dy,
            1e-3,
        );
        for i in 0..b.len() {
            approx_eq(db[i], fd_b[i], 1e-2);
        }
    }
}
