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

use hologram_graph::{
    Graph, GraphOp, NodeId, InputSource,
    registry::{DTypeId, ShapeId},
};
use hologram_graph::node::Node;
use smallvec::SmallVec;
use crate::error::CompileError;

pub fn parse(source: &str) -> Result<Graph, CompileError> {
    let mut graph = Graph::new();
    let mut name_to_id: hashbrown::HashMap<String, NodeId> = hashbrown::HashMap::new();

    for (lineno, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut tokens = trimmed.split_whitespace();
        let head = tokens.next().ok_or(CompileError::SourceParse("empty line"))?;
        match head {
            "input" => {
                let name = tokens.next().ok_or(CompileError::SourceParse("input: missing name"))?;
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
                let name = tokens.next().ok_or(CompileError::SourceParse("output: missing name"))?;
                let src = name_to_id.get(name)
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
                let op_name = tokens.next()
                    .ok_or(CompileError::SourceParse("op: missing op name"))?;
                let kind = parse_op_kind(op_name)
                    .ok_or(CompileError::SourceParse("op: unknown op kind"))?;
                let mut inputs: SmallVec<[InputSource; 4]> = SmallVec::new();
                let mut alias: Option<String> = None;
                for tok in tokens {
                    if let Some(rest) = tok.strip_prefix("as=") {
                        alias = Some(rest.to_string());
                    } else {
                        let src = name_to_id.get(tok)
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
                tracing::warn!(lineno, head = other, "unknown source directive");
                return Err(CompileError::SourceParse("unknown directive"));
            }
        }
    }
    Ok(graph)
}

fn parse_op_kind(name: &str) -> Option<hologram_graph::OpKind> {
    use hologram_graph::OpKind as K;
    Some(match name {
        "neg" => K::Neg, "bnot" => K::Bnot, "succ" => K::Succ, "pred" => K::Pred,
        "add" => K::Add, "sub" => K::Sub, "mul" => K::Mul,
        "xor" => K::Xor, "and" => K::And, "or" => K::Or,
        "relu" => K::Relu, "sigmoid" => K::Sigmoid, "tanh" => K::Tanh,
        "gelu" => K::Gelu, "silu" => K::Silu,
        "exp" => K::Exp, "log" => K::Log, "sqrt" => K::Sqrt,
        "matmul" => K::MatMul, "gemm" => K::Gemm,
        "softmax" => K::Softmax, "log_softmax" => K::LogSoftmax,
        "layer_norm" => K::LayerNorm, "rms_norm" => K::RmsNorm,
        "reduce_sum" => K::ReduceSum, "reduce_mean" => K::ReduceMean,
        "reshape" => K::Reshape, "transpose" => K::Transpose,
        "concat" => K::Concat, "slice" => K::Slice,
        "attention" => K::Attention, "fused_swiglu" => K::FusedSwiGlu,
        "max_pool_2d" => K::MaxPool2d, "avg_pool_2d" => K::AvgPool2d,
        "global_avg_pool" => K::GlobalAvgPool,
        _ => return None,
    })
}
