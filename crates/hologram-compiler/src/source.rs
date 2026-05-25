//! Source parser (spec VII.6).
//!
//! Minimal line-oriented grammar:
//!
//!   #-prefixed lines are comments
//!   blank lines are ignored
//!   `input <name>`           — declares a graph input port
//!   `output <name>`          — declares a graph output port
//!   `op <op_name> <input...>` — declares an Op node consuming the named inputs
//!
//! The parser builds a `Graph` whose nodes carry the corresponding `OpKind`.
//! ONNX import is post-v0.5.0; this entry point exists so `compile_from_source`
//! has a non-empty grammar.

use alloc::string::{String, ToString};

use crate::error::CompileError;
use hologram_graph::node::Node;
use hologram_graph::{
    registry::{DTypeId, ShapeId},
    Graph, GraphOp, InputSource, NodeId,
};
use smallvec::SmallVec;

pub fn parse(source: &str) -> Result<Graph, CompileError> {
    let mut graph = Graph::new();
    let mut name_to_id: hashbrown::HashMap<String, NodeId> = hashbrown::HashMap::new();

    for (lineno, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut tokens = trimmed.split_whitespace();
        let head = tokens
            .next()
            .ok_or(CompileError::SourceParse("empty line"))?;
        match head {
            "input" => {
                let name = tokens
                    .next()
                    .ok_or(CompileError::SourceParse("input: missing name"))?;
                let id = graph.add_node(Node {
                    op: GraphOp::Input,
                    inputs: SmallVec::new(),
                    output_dtype: DTypeId(0),
                    output_shape: ShapeId(0),
                });
                graph.add_input(id);
                name_to_id.insert(name.to_string(), id);
            }
            "output" => {
                let name = tokens
                    .next()
                    .ok_or(CompileError::SourceParse("output: missing name"))?;
                let src = name_to_id
                    .get(name)
                    .copied()
                    .ok_or(CompileError::SourceParse("output: unknown source"))?;
                let id = graph.add_node(Node {
                    op: GraphOp::Output,
                    inputs: SmallVec::from_iter([InputSource::Node(src)]),
                    output_dtype: DTypeId(0),
                    output_shape: ShapeId(0),
                });
                graph.add_output(id);
            }
            "op" => {
                let op_name = tokens
                    .next()
                    .ok_or(CompileError::SourceParse("op: missing op name"))?;
                let kind = parse_op_kind(op_name)
                    .ok_or(CompileError::SourceParse("op: unknown op kind"))?;
                let mut inputs: SmallVec<[InputSource; 4]> = SmallVec::new();
                let mut alias: Option<String> = None;
                for tok in tokens {
                    if let Some(rest) = tok.strip_prefix("as=") {
                        alias = Some(rest.to_string());
                    } else {
                        let src = name_to_id
                            .get(tok)
                            .copied()
                            .ok_or(CompileError::SourceParse("op: unresolved input"))?;
                        inputs.push(InputSource::Node(src));
                    }
                }
                let id = graph.add_node(Node {
                    op: GraphOp::Op(kind),
                    inputs,
                    output_dtype: DTypeId(0),
                    output_shape: ShapeId(0),
                });
                if let Some(name) = alias {
                    name_to_id.insert(name, id);
                }
            }
            other => {
                #[cfg(feature = "std")]
                tracing::warn!(lineno, head = other, "unknown source directive");
                #[cfg(not(feature = "std"))]
                let _ = (lineno, other);
                return Err(CompileError::SourceParse("unknown directive"));
            }
        }
    }
    Ok(graph)
}

/// Parse a snake-case op name into an `OpKind`. Drives off
/// `OpKind::name()`, the canonical name table — adding an op kind to the
/// catalog automatically makes it parseable here, with no per-op entry
/// required.
fn parse_op_kind(name: &str) -> Option<hologram_graph::OpKind> {
    use hologram_graph::OpKind as K;
    const ALL: &[K] = &[
        K::Neg,
        K::Bnot,
        K::Succ,
        K::Pred,
        K::Add,
        K::Sub,
        K::Mul,
        K::Xor,
        K::And,
        K::Or,
        K::Relu,
        K::Sigmoid,
        K::Tanh,
        K::Gelu,
        K::Silu,
        K::Elu,
        K::Selu,
        K::Exp,
        K::Log,
        K::Log1p,
        K::Sqrt,
        K::Reciprocal,
        K::Sin,
        K::Cos,
        K::Tan,
        K::Asin,
        K::Acos,
        K::Atan,
        K::Ceil,
        K::Floor,
        K::Round,
        K::Erf,
        K::IsNaN,
        K::Sign,
        K::Abs,
        K::Div,
        K::Pow,
        K::Mod,
        K::Min,
        K::Max,
        K::Equal,
        K::Less,
        K::LessOrEqual,
        K::Greater,
        K::GreaterOrEqual,
        K::MatMul,
        K::Gemm,
        K::Conv2d,
        K::ConvTranspose2d,
        K::LayerNorm,
        K::RmsNorm,
        K::GroupNorm,
        K::InstanceNorm,
        K::AddRmsNorm,
        K::ReduceSum,
        K::ReduceMean,
        K::ReduceProd,
        K::ReduceMin,
        K::ReduceMax,
        K::Reshape,
        K::Transpose,
        K::Concat,
        K::Slice,
        K::Softmax,
        K::LogSoftmax,
        K::MaxPool2d,
        K::AvgPool2d,
        K::GlobalAvgPool,
        K::Attention,
        K::FusedSwiGlu,
        K::Pad,
        K::Expand,
        K::Resize,
        K::CumSum,
        K::RotaryEmbedding,
        K::Clip,
        K::Lrn,
        K::Where,
        K::MatMulGradA,
        K::MatMulGradB,
        K::Conv2dGradX,
        K::Conv2dGradW,
        K::SoftmaxGrad,
        K::LogSoftmaxGrad,
        K::LayerNormGrad,
        K::RmsNormGrad,
        K::GroupNormGrad,
        K::ReduceSumGrad,
        K::ReduceMeanGrad,
        K::ReduceProdGrad,
        K::SubGrad,
        K::MulGrad,
        K::DivGrad,
        K::PowGrad,
        K::MinGrad,
        K::MaxGrad,
        K::ConcatGrad,
        K::SliceGrad,
        K::AvgPool2dGrad,
        K::GlobalAvgPoolGrad,
        K::PadGrad,
        K::AttentionGrad,
        K::FusedSwiGluGrad,
        K::UnaryGrad,
        K::Dequantize,
    ];
    ALL.iter().copied().find(|k| k.name() == name)
}
