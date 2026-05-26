//! Source parser (spec VII.6).
//!
//! Line-oriented grammar (`#` comments, blank lines ignored):
//!
//! ```text
//!   input  <name> [:<shape>]              — graph input port (f32)
//!   const  <name> :<shape> = v0,v1,...    — f32 constant tensor
//!   op     <op_name> <input...> [:<shape>] [as=<alias>]
//!   output <name>                         — graph output port
//! ```
//!
//! A `:<shape>` token is `d0xd1x…` (e.g. `:2x3`, `:1x1x2x2`). Inputs to an op
//! reference any prior `input` / `const` / op-alias name. This lets a graph be
//! expressed end-to-end in text — with the constant operands and shapes the
//! UOR-native ops need (Clip bounds, Slice indices, RoPE cos/sin, …) — rather
//! than only built programmatically.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::error::CompileError;
use hologram_graph::constant::ConstantEntry;
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor, ShapeId};
use hologram_graph::{Graph, GraphOp, InputSource};
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

/// Parse a `:d0xd1x…` shape token into a `ShapeDescriptor` (rank ≤ 8).
fn parse_shape(tok: &str) -> Option<ShapeDescriptor> {
    let body = tok.strip_prefix(':')?;
    let mut dims = [0u64; 8];
    let mut rank = 0usize;
    for part in body.split('x') {
        if rank >= 8 {
            return None;
        }
        dims[rank] = part.parse::<u64>().ok()?;
        rank += 1;
    }
    if rank == 0 {
        return None;
    }
    Some(ShapeDescriptor {
        rank: rank as u8,
        dims,
        dims_overflow: None,
    })
}

pub fn parse(source: &str) -> Result<Graph, CompileError> {
    let mut graph = Graph::new();
    // A name resolves to an operand source: a node (input / op output) or a
    // constant.
    let mut names: hashbrown::HashMap<String, InputSource> = hashbrown::HashMap::new();

    for line in source.lines() {
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
                let shape = match tokens.next() {
                    Some(s) => graph.shape_registry_mut().intern(
                        parse_shape(s).ok_or(CompileError::SourceParse("input: bad shape"))?,
                    ),
                    None => ShapeId(0),
                };
                let id = graph.add_node(Node {
                    op: GraphOp::Input,
                    inputs: SmallVec::new(),
                    output_dtype: DTypeId(DTYPE_F32),
                    output_shape: shape,
                });
                graph.add_input(id);
                names.insert(name.to_string(), InputSource::Node(id));
            }
            "const" => {
                let name = tokens
                    .next()
                    .ok_or(CompileError::SourceParse("const: missing name"))?
                    .to_string();
                let shape_tok = tokens
                    .next()
                    .ok_or(CompileError::SourceParse("const: missing shape"))?;
                let shape_desc =
                    parse_shape(shape_tok).ok_or(CompileError::SourceParse("const: bad shape"))?;
                let shape = graph.shape_registry_mut().intern(shape_desc);
                // `= v0,v1,...`
                let eq = tokens
                    .next()
                    .ok_or(CompileError::SourceParse("const: missing '='"))?;
                if eq != "=" {
                    return Err(CompileError::SourceParse("const: expected '='"));
                }
                let values = tokens
                    .next()
                    .ok_or(CompileError::SourceParse("const: missing values"))?;
                let mut bytes = Vec::new();
                for v in values.split(',') {
                    let f = v
                        .parse::<f32>()
                        .map_err(|_| CompileError::SourceParse("const: bad value"))?;
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                let cid = graph.constants_mut().insert(ConstantEntry {
                    bytes,
                    dtype: DTypeId(DTYPE_F32),
                    shape,
                });
                names.insert(name, InputSource::Constant(cid));
            }
            "output" => {
                let name = tokens
                    .next()
                    .ok_or(CompileError::SourceParse("output: missing name"))?;
                let src = match names.get(name) {
                    Some(InputSource::Node(id)) => *id,
                    _ => return Err(CompileError::SourceParse("output: unknown/!node source")),
                };
                let id = graph.add_node(Node {
                    op: GraphOp::Output,
                    inputs: SmallVec::from_iter([InputSource::Node(src)]),
                    output_dtype: DTypeId(DTYPE_F32),
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
                let mut out_shape = ShapeId(0);
                for tok in tokens {
                    if let Some(rest) = tok.strip_prefix("as=") {
                        alias = Some(rest.to_string());
                    } else if tok.starts_with(':') {
                        out_shape = graph.shape_registry_mut().intern(
                            parse_shape(tok).ok_or(CompileError::SourceParse("op: bad shape"))?,
                        );
                    } else {
                        let src = *names
                            .get(tok)
                            .ok_or(CompileError::SourceParse("op: unresolved input"))?;
                        inputs.push(src);
                    }
                }
                let id = graph.add_node(Node {
                    op: GraphOp::Op(kind),
                    inputs,
                    output_dtype: DTypeId(DTYPE_F32),
                    output_shape: out_shape,
                });
                if let Some(name) = alias {
                    names.insert(name, InputSource::Node(id));
                }
            }
            _ => return Err(CompileError::SourceParse("unknown directive")),
        }
    }
    Ok(graph)
}

/// Parse a snake-case op name into an `OpKind`. Drives off `OpKind::name()`,
/// the canonical name table — adding an op kind to the catalog automatically
/// makes it parseable here.
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
        K::Dequantize,
    ];
    ALL.iter().copied().find(|k| k.name() == name)
}
