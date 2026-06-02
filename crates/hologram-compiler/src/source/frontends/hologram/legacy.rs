//! Legacy line-oriented Hologram source parser.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::error::CompileError;
use crate::source::ir::{
    SourceBinding, SourceConst, SourceInput, SourceItem, SourceOpCall, SourceOutput, SourceProgram,
    SourceSymbol, SourceTensorLiteral, SourceType,
};
use crate::source::op_table;
use crate::source::{diagnostic, SourceDiagnostic};
use hologram_graph::constant::ConstantEntry;
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor, ShapeId};
use hologram_graph::{Graph, GraphOp, InputSource};
use nom::bytes::complete::take_while1;
use nom::character::complete::space1;
use nom::combinator::all_consuming;
use nom::multi::separated_list1;
use nom::{Err as NomErr, IResult, Parser};
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

/// Parse legacy source directly into the graph IR.
pub fn parse_graph(source: &str) -> Result<Graph, CompileError> {
    let mut parser = GraphParser::new();
    for line in source.lines() {
        parser.parse_line(line)?;
    }
    Ok(parser.finish())
}

/// Parse legacy source into the common source IR with diagnostics.
pub fn parse_program_diagnostic(source: &str) -> Result<SourceProgram, SourceDiagnostic> {
    let mut program = SourceProgram::new();
    for (index, line) in source.lines().enumerate() {
        if let Some(item) = parse_line_diagnostic(index + 1, line, &mut program)? {
            program.push(item);
        }
    }
    Ok(program)
}

fn parse_line_diagnostic(
    line_number: usize,
    line: &str,
    program: &mut SourceProgram,
) -> Result<Option<SourceItem>, SourceDiagnostic> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(None);
    }
    let base_column = line.len() - line.trim_start().len() + 1;
    let tokens = parse_tokens_diagnostic(line_number, base_column, trimmed)?;
    let mut cursor = TokenCursor::new(tokens);
    let head = cursor.next("empty line").map_err(|err| {
        diagnostic::from_line(
            line_number,
            base_column,
            trimmed,
            diagnostic::compile_error_kind(&err),
        )
    })?;
    parse_directive(head, cursor, program)
        .map(Some)
        .map_err(|err| {
            diagnostic::from_line(
                line_number,
                base_column,
                trimmed,
                diagnostic::compile_error_kind(&err),
            )
        })
}

fn parse_tokens_diagnostic(
    line_number: usize,
    base_column: usize,
    input: &str,
) -> Result<Vec<&str>, SourceDiagnostic> {
    token_list(input)
        .map(|(_, tokens)| tokens)
        .map_err(|err| token_diagnostic(line_number, base_column, input, err))
}

fn parse_tokens(input: &str) -> Result<Vec<&str>, CompileError> {
    token_list(input)
        .map(|(_, tokens)| tokens)
        .map_err(|_| CompileError::SourceParse("source: bad tokens"))
}

fn token_list(input: &str) -> IResult<&str, Vec<&str>> {
    all_consuming(separated_list1(space1, token)).parse(input)
}

fn token(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| !c.is_whitespace()).parse(input)
}

fn token_diagnostic(
    line_number: usize,
    base_column: usize,
    input: &str,
    err: NomErr<nom::error::Error<&str>>,
) -> SourceDiagnostic {
    match err {
        NomErr::Error(err) | NomErr::Failure(err) => diagnostic::from_remainder(
            line_number,
            base_column,
            input,
            err.input,
            "source: bad tokens",
        ),
        NomErr::Incomplete(_) => {
            diagnostic::from_line(line_number, base_column, input, "source: bad tokens")
        }
    }
}

struct GraphParser {
    graph: Graph,
    names: hashbrown::HashMap<String, InputSource>,
}

impl GraphParser {
    fn new() -> Self {
        Self {
            graph: Graph::new(),
            names: hashbrown::HashMap::new(),
        }
    }

    fn parse_line(&mut self, line: &str) -> Result<(), CompileError> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return Ok(());
        }
        let tokens = parse_tokens(trimmed)?;
        let mut cursor = TokenCursor::new(tokens);
        let head = cursor.next("empty line")?;
        self.parse_directive(head, cursor)
    }

    fn finish(self) -> Graph {
        self.graph
    }
}

impl GraphParser {
    fn parse_directive(&mut self, head: &str, tokens: TokenCursor<'_>) -> Result<(), CompileError> {
        match head {
            "input" => self.parse_input(tokens),
            "const" => self.parse_const(tokens),
            "output" => self.parse_output(tokens),
            "op" => self.parse_op(tokens),
            _ => Err(CompileError::SourceParse("unknown directive")),
        }
    }

    fn parse_input(&mut self, mut tokens: TokenCursor<'_>) -> Result<(), CompileError> {
        let name = tokens.next("input: missing name")?;
        let shape = self.optional_shape(tokens.optional(), "input: bad shape")?;
        let id = self.graph.add_node(input_node(shape));
        self.graph.add_input(id);
        self.names.insert(name.to_string(), InputSource::Node(id));
        Ok(())
    }

    fn parse_const(&mut self, mut tokens: TokenCursor<'_>) -> Result<(), CompileError> {
        let name = tokens.next("const: missing name")?.to_string();
        let shape = self.next_shape(&mut tokens, "const: missing shape", "const: bad shape")?;
        expect_const_equals(&mut tokens)?;
        let literal = parse_f32_values(tokens.next("const: missing values")?)?;
        let id = self
            .graph
            .constants_mut()
            .insert(const_entry(literal, shape));
        self.names.insert(name, InputSource::Constant(id));
        Ok(())
    }

    fn parse_output(&mut self, mut tokens: TokenCursor<'_>) -> Result<(), CompileError> {
        let src = self.output_source(tokens.next("output: missing name")?)?;
        let id = self.graph.add_node(output_node(src));
        self.graph.add_output(id);
        Ok(())
    }

    fn parse_op(&mut self, mut tokens: TokenCursor<'_>) -> Result<(), CompileError> {
        let op = parse_op_kind(tokens.next("op: missing op name")?)?;
        let tail = self.parse_graph_op_tail(op, tokens)?;
        let id = self.graph.add_node(op_node(op, tail.inputs, tail.shape));
        if let Some(alias) = tail.alias {
            self.names.insert(alias, InputSource::Node(id));
        }
        Ok(())
    }
}

impl GraphParser {
    fn parse_graph_op_tail(
        &mut self,
        op: hologram_graph::OpKind,
        mut tokens: TokenCursor<'_>,
    ) -> Result<GraphOpTail, CompileError> {
        let mut tail = GraphOpTail::new(op);
        while let Some(tok) = tokens.optional() {
            tail.push(tok, self)?;
        }
        Ok(tail)
    }

    fn source(&self, name: &str) -> Result<InputSource, CompileError> {
        self.names
            .get(name)
            .copied()
            .ok_or(CompileError::SourceParse("op: unresolved input"))
    }

    fn output_source(&self, name: &str) -> Result<hologram_graph::NodeId, CompileError> {
        match self.names.get(name) {
            Some(InputSource::Node(id)) => Ok(*id),
            _ => Err(CompileError::SourceParse("output: unknown/!node source")),
        }
    }

    fn optional_shape(
        &mut self,
        tok: Option<&str>,
        err: &'static str,
    ) -> Result<ShapeId, CompileError> {
        match tok {
            Some(tok) => self.shape_id(parse_shape(tok, err)?),
            None => Ok(ShapeId(0)),
        }
    }

    fn next_shape(
        &mut self,
        tokens: &mut TokenCursor<'_>,
        missing: &'static str,
        bad: &'static str,
    ) -> Result<ShapeId, CompileError> {
        let shape = parse_shape(tokens.next(missing)?, bad)?;
        self.shape_id(shape)
    }

    fn shape_id(&mut self, shape: ShapeDescriptor) -> Result<ShapeId, CompileError> {
        Ok(self.graph.shape_registry_mut().intern(shape))
    }
}

struct GraphOpTail {
    inputs: SmallVec<[InputSource; 4]>,
    alias: Option<String>,
    shape: ShapeId,
}

impl GraphOpTail {
    fn new(_op: hologram_graph::OpKind) -> Self {
        Self {
            inputs: SmallVec::new(),
            alias: None,
            shape: ShapeId(0),
        }
    }

    fn push(&mut self, tok: &str, parser: &mut GraphParser) -> Result<(), CompileError> {
        if let Some(rest) = tok.strip_prefix("as=") {
            self.alias = Some(rest.to_string());
        } else if tok.starts_with(':') {
            self.shape = parser.shape_id(parse_shape(tok, "op: bad shape")?)?;
        } else {
            self.inputs.push(parser.source(tok)?);
        }
        Ok(())
    }
}

fn parse_op_kind(op: &str) -> Result<hologram_graph::OpKind, CompileError> {
    op_table::parse(op).ok_or(CompileError::SourceParse("op: unknown op kind"))
}

fn input_node(shape: ShapeId) -> Node {
    Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    }
}

fn output_node(src: hologram_graph::NodeId) -> Node {
    Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(src)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: ShapeId(0),
    }
}

fn op_node(op: hologram_graph::OpKind, inputs: SmallVec<[InputSource; 4]>, shape: ShapeId) -> Node {
    Node {
        op: GraphOp::Op(op),
        inputs,
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    }
}

fn const_entry(literal: SourceTensorLiteral, shape: ShapeId) -> ConstantEntry {
    ConstantEntry {
        bytes: literal.bytes,
        dtype: DTypeId(DTYPE_F32),
        shape,
    }
}

fn parse_directive(
    head: &str,
    tokens: TokenCursor<'_>,
    program: &mut SourceProgram,
) -> Result<SourceItem, CompileError> {
    match head {
        "input" => parse_input(tokens, program),
        "const" => parse_const(tokens, program),
        "output" => parse_output(tokens, program),
        "op" => parse_op(tokens, program),
        _ => Err(CompileError::SourceParse("unknown directive")),
    }
}

fn parse_input(
    mut tokens: TokenCursor<'_>,
    program: &mut SourceProgram,
) -> Result<SourceItem, CompileError> {
    let name = tokens.next("input: missing name")?;
    let shape = optional_shape(tokens.optional(), "input: bad shape")?;
    let ty = SourceType::f32(shape);
    Ok(SourceItem::Input(SourceInput::new(
        program.intern(name),
        ty,
    )))
}

fn parse_const(
    mut tokens: TokenCursor<'_>,
    program: &mut SourceProgram,
) -> Result<SourceItem, CompileError> {
    let name = tokens.next("const: missing name")?;
    let shape = next_shape(&mut tokens, "const: missing shape", "const: bad shape")?;
    expect_const_equals(&mut tokens)?;
    let values = tokens.next("const: missing values")?;
    let literal = parse_f32_values(values)?;
    let ty = SourceType::f32(Some(shape));
    Ok(SourceItem::Const(SourceConst::new(
        program.intern(name),
        ty,
        literal,
    )))
}

fn parse_output(
    mut tokens: TokenCursor<'_>,
    program: &mut SourceProgram,
) -> Result<SourceItem, CompileError> {
    let name = tokens.next("output: missing name")?;
    Ok(SourceItem::Output(SourceOutput::new(program.intern(name))))
}

fn parse_op(
    mut tokens: TokenCursor<'_>,
    program: &mut SourceProgram,
) -> Result<SourceItem, CompileError> {
    let op_name = tokens.next("op: missing op name")?;
    let op = op_table::parse(op_name).ok_or(CompileError::SourceParse("op: unknown op kind"))?;
    let binding = parse_op_tail(op, tokens, program)?;
    Ok(SourceItem::Binding(binding))
}

fn parse_op_tail(
    op: hologram_graph::OpKind,
    mut tokens: TokenCursor<'_>,
    program: &mut SourceProgram,
) -> Result<SourceBinding, CompileError> {
    let mut tail = OpTail::new(op);
    while let Some(tok) = tokens.optional() {
        tail.push(tok, program)?;
    }
    Ok(tail.finish())
}

struct TokenCursor<'a> {
    tokens: Vec<&'a str>,
    pos: usize,
}

impl<'a> TokenCursor<'a> {
    fn new(tokens: Vec<&'a str>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn next(&mut self, err: &'static str) -> Result<&'a str, CompileError> {
        self.optional().ok_or(CompileError::SourceParse(err))
    }

    fn optional(&mut self) -> Option<&'a str> {
        let token = self.tokens.get(self.pos).copied()?;
        self.pos += 1;
        Some(token)
    }
}

struct OpTail {
    op: hologram_graph::OpKind,
    inputs: Vec<SourceSymbol>,
    alias: Option<SourceSymbol>,
    ty: Option<SourceType>,
}

impl OpTail {
    fn new(op: hologram_graph::OpKind) -> Self {
        Self {
            op,
            inputs: Vec::new(),
            alias: None,
            ty: None,
        }
    }

    fn push(&mut self, tok: &str, program: &mut SourceProgram) -> Result<(), CompileError> {
        if let Some(rest) = tok.strip_prefix("as=") {
            self.alias = Some(program.intern(rest));
        } else if tok.starts_with(':') {
            self.ty = Some(SourceType::f32(Some(parse_shape(tok, "op: bad shape")?)));
        } else {
            self.inputs.push(program.intern(tok));
        }
        Ok(())
    }

    fn finish(self) -> SourceBinding {
        let call = SourceOpCall::new(self.op, self.inputs, self.ty);
        SourceBinding::op(self.alias, call)
    }
}

fn expect_const_equals(tokens: &mut TokenCursor<'_>) -> Result<(), CompileError> {
    let found = tokens.next("const: missing '='")?;
    if found == "=" {
        Ok(())
    } else {
        Err(CompileError::SourceParse("const: expected '='"))
    }
}

fn next_shape(
    tokens: &mut TokenCursor<'_>,
    missing: &'static str,
    bad: &'static str,
) -> Result<ShapeDescriptor, CompileError> {
    let tok = tokens.next(missing)?;
    parse_shape(tok, bad)
}

fn optional_shape(
    tok: Option<&str>,
    bad: &'static str,
) -> Result<Option<ShapeDescriptor>, CompileError> {
    match tok {
        Some(tok) => Ok(Some(parse_shape(tok, bad)?)),
        None => Ok(None),
    }
}

fn parse_shape(tok: &str, err: &'static str) -> Result<ShapeDescriptor, CompileError> {
    let body = tok
        .strip_prefix(':')
        .ok_or(CompileError::SourceParse(err))?;
    parse_shape_body(body).ok_or(CompileError::SourceParse(err))
}

fn parse_shape_body(body: &str) -> Option<ShapeDescriptor> {
    let mut dims = [0u64; 8];
    let rank = parse_dims(body, &mut dims)?;
    Some(ShapeDescriptor {
        rank: rank as u8,
        dims,
        dims_overflow: None,
    })
}

fn parse_dims(body: &str, dims: &mut [u64; 8]) -> Option<usize> {
    let mut rank = 0usize;
    for part in body.split('x') {
        if rank >= 8 {
            return None;
        }
        dims[rank] = part.parse::<u64>().ok()?;
        rank += 1;
    }
    nonzero_rank(rank)
}

fn nonzero_rank(rank: usize) -> Option<usize> {
    if rank == 0 {
        None
    } else {
        Some(rank)
    }
}

fn parse_f32_values(values: &str) -> Result<SourceTensorLiteral, CompileError> {
    let mut bytes = Vec::new();
    let mut value_count = 0usize;
    for value in values.split(',') {
        push_f32(value, &mut bytes)?;
        value_count += 1;
    }
    Ok(SourceTensorLiteral::new(bytes, value_count))
}

fn push_f32(value: &str, bytes: &mut Vec<u8>) -> Result<(), CompileError> {
    let f = value
        .parse::<f32>()
        .map_err(|_| CompileError::SourceParse("const: bad value"))?;
    bytes.extend_from_slice(&f.to_le_bytes());
    Ok(())
}
