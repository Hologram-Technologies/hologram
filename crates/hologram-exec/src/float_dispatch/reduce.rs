use super::helpers::*;
use crate::error::{ExecError, ExecResult};

pub(super) fn dispatch_reduce(
    inputs: &[&[u8]],
    size: usize,
    f: impl Fn(&[f32]) -> f32,
) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let out: Vec<f32> = x.chunks(size).map(f).collect();
    Ok(f32_vec_to_bytes(out))
}

pub(super) fn reduce_sum(row: &[f32]) -> f32 {
    row.iter().sum()
}

pub(super) fn reduce_mean(row: &[f32]) -> f32 {
    row.iter().sum::<f32>() / row.len() as f32
}

pub(super) fn reduce_max(row: &[f32]) -> f32 {
    row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

pub(super) fn reduce_min(row: &[f32]) -> f32 {
    row.iter().cloned().fold(f32::INFINITY, f32::min)
}

pub(super) fn reduce_prod(row: &[f32]) -> f32 {
    row.iter().product()
}
