//! Shared source op-attribute parsing semantics.

use alloc::vec::Vec;

use crate::error::CompileError;
use crate::source::SourceAttrs;
use hologram_graph::{
    AttentionAttrs, ConvAttrs, GatherAttrs, GemmAttrs, LrnAttrs, NormAttrs, OpKind, QuantAttrs,
    ReduceAttrs,
};

const REDUCE_ATTRS: &[&str] = &["axes", "axes_mask", "keepdims"];
const CONV_ATTRS: &[&str] = &[
    "stride", "strides", "pad", "pads", "padding", "kernel", "k", "stride_h", "stride_w", "pad_h",
    "pad_w", "k_h", "k_w",
];
const GEMM_ATTRS: &[&str] = &["alpha", "alpha_bits", "beta", "beta_bits"];
const LRN_ATTRS: &[&str] = &[
    "alpha",
    "alpha_bits",
    "beta",
    "beta_bits",
    "bias",
    "bias_bits",
    "size",
];
const GROUP_NORM_ATTRS: &[&str] = &["num_groups"];
const GATHER_ATTRS: &[&str] = &["axis"];
const ATTENTION_ATTRS: &[&str] = &["causal", "scale", "scale_bits"];
const DEQUANTIZE_ATTRS: &[&str] = &["axis", "scale", "scale_bits", "quant_dtype", "zero_point"];
const NO_ATTRS: &[&str] = &[];

/// Return source-level attribute names accepted for a canonical op.
pub fn op_attr_names(op: OpKind) -> &'static [&'static str] {
    match op {
        OpKind::ReduceSum
        | OpKind::ReduceMean
        | OpKind::ReduceProd
        | OpKind::ReduceMin
        | OpKind::ReduceMax => REDUCE_ATTRS,
        OpKind::Conv2d
        | OpKind::ConvTranspose2d
        | OpKind::Im2Col
        | OpKind::Col2Im
        | OpKind::MaxPool2d
        | OpKind::AvgPool2d => CONV_ATTRS,
        OpKind::Gemm => GEMM_ATTRS,
        OpKind::Lrn => LRN_ATTRS,
        OpKind::GroupNorm => GROUP_NORM_ATTRS,
        OpKind::Gather => GATHER_ATTRS,
        OpKind::Attention => ATTENTION_ATTRS,
        OpKind::Dequantize => DEQUANTIZE_ATTRS,
        _ => NO_ATTRS,
    }
}

/// Parsed source-level attribute assignment.
pub(crate) struct ParsedAttr<'a> {
    /// Attribute name.
    pub(crate) name: &'a str,
    /// Attribute value.
    pub(crate) value: AttrValue<'a>,
}

/// Parsed source-level attribute value.
pub(crate) enum AttrValue<'a> {
    /// Boolean attribute value.
    Bool(bool),
    /// Numeric attribute value as source text.
    Number(&'a str),
    /// Numeric list attribute value as source text.
    List(Vec<&'a str>),
}

/// Convert parsed source attributes into sparse graph attributes.
pub(crate) fn attrs_from_assignments(
    op: OpKind,
    assignments: Vec<ParsedAttr<'_>>,
) -> Result<SourceAttrs, CompileError> {
    let mut attrs = SourceAttrs::default();
    for assignment in assignments {
        apply_attr(op, &mut attrs, assignment)?;
    }
    Ok(attrs)
}

/// Apply one parsed source attribute to sparse graph attributes.
pub(crate) fn apply_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    attr: ParsedAttr<'_>,
) -> Result<(), CompileError> {
    match attr.name {
        "axes" => set_reduce_axes(op, attrs, attr.value),
        "axes_mask" => set_reduce_axes_mask(op, attrs, attr.value),
        "keepdims" => set_reduce_keepdims(op, attrs, attr.value),
        "axis" => set_axis_attr(op, attrs, attr.value),
        "num_groups" => set_norm_groups(op, attrs, attr.value),
        "causal" => set_attention_causal(op, attrs, attr.value),
        "scale" => set_scale_attr(op, attrs, attr.value),
        _ => apply_more_attr(op, attrs, attr),
    }
}

fn apply_more_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    attr: ParsedAttr<'_>,
) -> Result<(), CompileError> {
    match attr.name {
        "stride" | "strides" => set_conv_stride(op, attrs, attr.value),
        "pad" | "pads" | "padding" => set_conv_pad(op, attrs, attr.value),
        "kernel" | "k" => set_conv_kernel(op, attrs, attr.value),
        "stride_h" => set_conv_stride_h(op, attrs, attr.value),
        "stride_w" => set_conv_stride_w(op, attrs, attr.value),
        "pad_h" => set_conv_pad_h(op, attrs, attr.value),
        "pad_w" => set_conv_pad_w(op, attrs, attr.value),
        "k_h" => set_conv_k_h(op, attrs, attr.value),
        "k_w" => set_conv_k_w(op, attrs, attr.value),
        _ => apply_tail_attr(op, attrs, attr),
    }
}

fn apply_tail_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    attr: ParsedAttr<'_>,
) -> Result<(), CompileError> {
    match attr.name {
        "alpha" => set_alpha_attr(op, attrs, attr.value),
        "beta" => set_beta_attr(op, attrs, attr.value),
        "bias" => set_lrn_bias(op, attrs, attr.value),
        "size" => set_lrn_size(op, attrs, attr.value),
        "quant_dtype" => set_quant_dtype(op, attrs, attr.value),
        "scale_bits" => set_scale_bits_attr(op, attrs, attr.value),
        "zero_point" => set_quant_zero_point(op, attrs, attr.value),
        "alpha_bits" => set_alpha_bits_attr(op, attrs, attr.value),
        "beta_bits" => set_beta_bits_attr(op, attrs, attr.value),
        "bias_bits" => set_lrn_bias_bits(op, attrs, attr.value),
        _ => Err(CompileError::SourceParse("op: unknown attr")),
    }
}

fn set_reduce_axes(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_reduce_op(op)?;
    reduce_attrs(attrs).axes_mask = axes_mask(value)?;
    Ok(())
}

fn set_reduce_axes_mask(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_reduce_op(op)?;
    reduce_attrs(attrs).axes_mask = value.u32()?;
    Ok(())
}

fn set_reduce_keepdims(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_reduce_op(op)?;
    reduce_attrs(attrs).keepdims = value.bool()?;
    Ok(())
}

fn set_axis_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    match op {
        OpKind::Gather => gather_attrs(attrs).axis = value.i32()?,
        OpKind::Dequantize => quant_attrs(attrs).axis = value.i32()?,
        _ => return Err(CompileError::SourceParse("op: attr not valid")),
    }
    Ok(())
}

fn set_norm_groups(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_norm_group_op(op)?;
    norm_attrs(attrs).num_groups = value.u32()?;
    Ok(())
}

fn set_attention_causal(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_attention_op(op)?;
    attention_attrs(attrs).causal = value.bool()?;
    Ok(())
}

fn set_scale_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    set_scale_bits_value(op, attrs, value.f32_bits()?)
}

fn set_scale_bits_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    set_scale_bits_value(op, attrs, value.u32()?)
}

fn set_scale_bits_value(
    op: OpKind,
    attrs: &mut SourceAttrs,
    bits: u32,
) -> Result<(), CompileError> {
    match op {
        OpKind::Attention => attention_attrs(attrs).scale_bits = bits,
        OpKind::Dequantize => quant_attrs(attrs).scale_bits = bits,
        _ => return Err(CompileError::SourceParse("op: attr not valid")),
    }
    Ok(())
}

fn set_conv_stride(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    let (h, w) = value.u32_pair()?;
    let attrs = conv_attrs(attrs);
    attrs.stride_h = h;
    attrs.stride_w = w;
    Ok(())
}

fn set_conv_pad(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    let (h, w) = pad_pair(value)?;
    let attrs = conv_attrs(attrs);
    attrs.pad_h = h;
    attrs.pad_w = w;
    Ok(())
}

fn set_conv_kernel(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    let (h, w) = value.u32_pair()?;
    let attrs = conv_attrs(attrs);
    attrs.k_h = h;
    attrs.k_w = w;
    Ok(())
}

fn set_conv_stride_h(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    conv_attrs(attrs).stride_h = value.u32()?;
    Ok(())
}

fn set_conv_stride_w(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    conv_attrs(attrs).stride_w = value.u32()?;
    Ok(())
}

fn set_conv_pad_h(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    conv_attrs(attrs).pad_h = value.u32()?;
    Ok(())
}

fn set_conv_pad_w(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    conv_attrs(attrs).pad_w = value.u32()?;
    Ok(())
}

fn set_conv_k_h(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    conv_attrs(attrs).k_h = value.u32()?;
    Ok(())
}

fn set_conv_k_w(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_conv_attr_op(op)?;
    conv_attrs(attrs).k_w = value.u32()?;
    Ok(())
}

fn set_alpha_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    set_alpha_bits_value(op, attrs, value.f32_bits()?)
}

fn set_alpha_bits_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    set_alpha_bits_value(op, attrs, value.u32()?)
}

fn set_alpha_bits_value(
    op: OpKind,
    attrs: &mut SourceAttrs,
    bits: u32,
) -> Result<(), CompileError> {
    match op {
        OpKind::Gemm => gemm_attrs(attrs).alpha_bits = bits,
        OpKind::Lrn => lrn_attrs(attrs).alpha_bits = bits,
        _ => return Err(CompileError::SourceParse("op: attr not valid")),
    }
    Ok(())
}

fn set_beta_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    set_beta_bits_value(op, attrs, value.f32_bits()?)
}

fn set_beta_bits_attr(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    set_beta_bits_value(op, attrs, value.u32()?)
}

fn set_beta_bits_value(op: OpKind, attrs: &mut SourceAttrs, bits: u32) -> Result<(), CompileError> {
    match op {
        OpKind::Gemm => gemm_attrs(attrs).beta_bits = bits,
        OpKind::Lrn => lrn_attrs(attrs).beta_bits = bits,
        _ => return Err(CompileError::SourceParse("op: attr not valid")),
    }
    Ok(())
}

fn set_lrn_bias(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_lrn_op(op)?;
    lrn_attrs(attrs).bias_bits = value.f32_bits()?;
    Ok(())
}

fn set_lrn_bias_bits(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_lrn_op(op)?;
    lrn_attrs(attrs).bias_bits = value.u32()?;
    Ok(())
}

fn set_lrn_size(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_lrn_op(op)?;
    lrn_attrs(attrs).size = value.u32()?;
    Ok(())
}

fn set_quant_dtype(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_quant_op(op)?;
    quant_attrs(attrs).quant_dtype = value.u8()?;
    Ok(())
}

fn set_quant_zero_point(
    op: OpKind,
    attrs: &mut SourceAttrs,
    value: AttrValue<'_>,
) -> Result<(), CompileError> {
    require_quant_op(op)?;
    quant_attrs(attrs).zero_point = value.i32()?;
    Ok(())
}

fn reduce_attrs(attrs: &mut SourceAttrs) -> &mut ReduceAttrs {
    attrs.reduce.get_or_insert_with(ReduceAttrs::default)
}

fn conv_attrs(attrs: &mut SourceAttrs) -> &mut ConvAttrs {
    attrs.conv.get_or_insert_with(ConvAttrs::default)
}

fn lrn_attrs(attrs: &mut SourceAttrs) -> &mut LrnAttrs {
    attrs.lrn.get_or_insert_with(LrnAttrs::default)
}

fn gemm_attrs(attrs: &mut SourceAttrs) -> &mut GemmAttrs {
    attrs.gemm.get_or_insert_with(GemmAttrs::default)
}

fn norm_attrs(attrs: &mut SourceAttrs) -> &mut NormAttrs {
    attrs.norm.get_or_insert_with(NormAttrs::default)
}

fn gather_attrs(attrs: &mut SourceAttrs) -> &mut GatherAttrs {
    attrs.gather.get_or_insert_with(GatherAttrs::default)
}

fn quant_attrs(attrs: &mut SourceAttrs) -> &mut QuantAttrs {
    attrs.quant.get_or_insert_with(QuantAttrs::default)
}

fn attention_attrs(attrs: &mut SourceAttrs) -> &mut AttentionAttrs {
    attrs.attention.get_or_insert_with(AttentionAttrs::default)
}

fn require_reduce_op(op: OpKind) -> Result<(), CompileError> {
    if matches!(
        op,
        OpKind::ReduceSum
            | OpKind::ReduceMean
            | OpKind::ReduceProd
            | OpKind::ReduceMin
            | OpKind::ReduceMax
    ) {
        Ok(())
    } else {
        Err(CompileError::SourceParse("op: attr not valid"))
    }
}

fn require_conv_attr_op(op: OpKind) -> Result<(), CompileError> {
    if matches!(
        op,
        OpKind::Conv2d
            | OpKind::ConvTranspose2d
            | OpKind::Im2Col
            | OpKind::Col2Im
            | OpKind::MaxPool2d
            | OpKind::AvgPool2d
    ) {
        Ok(())
    } else {
        Err(CompileError::SourceParse("op: attr not valid"))
    }
}

fn require_norm_group_op(op: OpKind) -> Result<(), CompileError> {
    require_op(op == OpKind::GroupNorm)
}

fn require_attention_op(op: OpKind) -> Result<(), CompileError> {
    require_op(op == OpKind::Attention)
}

fn require_lrn_op(op: OpKind) -> Result<(), CompileError> {
    require_op(op == OpKind::Lrn)
}

fn require_quant_op(op: OpKind) -> Result<(), CompileError> {
    require_op(op == OpKind::Dequantize)
}

fn require_op(ok: bool) -> Result<(), CompileError> {
    if ok {
        Ok(())
    } else {
        Err(CompileError::SourceParse("op: attr not valid"))
    }
}

fn axes_mask(value: AttrValue<'_>) -> Result<u32, CompileError> {
    let mut mask = 0u32;
    for axis in value.u32_list()? {
        mask |= checked_axis_bit(axis)?;
    }
    Ok(mask)
}

fn checked_axis_bit(axis: u32) -> Result<u32, CompileError> {
    if axis < 32 {
        Ok(1u32 << axis)
    } else {
        Err(CompileError::SourceParse("attr: axis out of range"))
    }
}

fn pad_pair(value: AttrValue<'_>) -> Result<(u32, u32), CompileError> {
    let values = value.u32_list()?;
    match values.as_slice() {
        [h, w] => Ok((*h, *w)),
        [top, left, bottom, right] if top == bottom && left == right => Ok((*top, *left)),
        _ => Err(CompileError::SourceParse("attr: bad list length")),
    }
}

impl<'a> AttrValue<'a> {
    fn bool(self) -> Result<bool, CompileError> {
        match self {
            Self::Bool(value) => Ok(value),
            _ => Err(CompileError::SourceParse("attr: expected bool")),
        }
    }

    fn u8(self) -> Result<u8, CompileError> {
        self.number()?.parse::<u8>().map_err(|_| attr_bad_number())
    }

    fn u32(self) -> Result<u32, CompileError> {
        self.number()?.parse::<u32>().map_err(|_| attr_bad_number())
    }

    fn i32(self) -> Result<i32, CompileError> {
        self.number()?.parse::<i32>().map_err(|_| attr_bad_number())
    }

    fn f32_bits(self) -> Result<u32, CompileError> {
        Ok(self
            .number()?
            .parse::<f32>()
            .map_err(|_| attr_bad_number())?
            .to_bits())
    }

    fn u32_pair(self) -> Result<(u32, u32), CompileError> {
        let values = self.u32_list()?;
        match values.as_slice() {
            [h, w] => Ok((*h, *w)),
            _ => Err(CompileError::SourceParse("attr: bad list length")),
        }
    }

    fn u32_list(self) -> Result<Vec<u32>, CompileError> {
        match self {
            Self::List(values) => values.into_iter().map(parse_u32).collect(),
            _ => Err(CompileError::SourceParse("attr: expected list")),
        }
    }

    fn number(self) -> Result<&'a str, CompileError> {
        match self {
            Self::Number(value) => Ok(value),
            _ => Err(CompileError::SourceParse("attr: expected number")),
        }
    }
}

fn parse_u32(value: &str) -> Result<u32, CompileError> {
    value.parse::<u32>().map_err(|_| attr_bad_number())
}

fn attr_bad_number() -> CompileError {
    CompileError::SourceParse("attr: bad number")
}
