//! Native Hologram source frontend.

mod legacy;

use alloc::vec::Vec;

use crate::error::CompileError;
use crate::source::attrs::{attrs_from_assignments, AttrValue, ParsedAttr};
use crate::source::frontend::{SourceFrontend, SourceFrontendInfo};
use crate::source::ir::{
    SourceBinding, SourceConst, SourceInput, SourceItem, SourceOpCall, SourceOutput, SourceProgram,
    SourceSymbol, SourceTensorLiteral, SourceType,
};
use crate::source::op_table;
use crate::source::{diagnostic, SourceDiagnostic, SourceDocument, SourceLanguage};
use hologram_graph::registry::ShapeDescriptor;
use hologram_graph::Graph;
use nom::branch::alt;
use nom::bytes::complete::{tag, take_while, take_while1};
use nom::character::complete::{char, digit1, space0, space1};
use nom::combinator::{map_res, recognize};
use nom::multi::{separated_list0, separated_list1};
use nom::number::complete::recognize_float;
use nom::sequence::{delimited, preceded};
use nom::{Err as NomErr, IResult, Parser};

/// Native Hologram source frontend.
#[derive(Debug, Clone, Copy, Default)]
pub struct HologramFrontend;

impl SourceFrontend for HologramFrontend {
    const INFO: SourceFrontendInfo = SourceFrontendInfo::new(
        SourceLanguage::Hologram,
        &["hologram", "holo", "native"],
        &["txt"],
    );

    fn parse_document(&self, source: &str) -> Result<SourceDocument, CompileError> {
        self.parse_document_diagnostic(source)
            .map_err(SourceDiagnostic::into_compile_error)
    }

    fn parse_document_diagnostic(&self, source: &str) -> Result<SourceDocument, SourceDiagnostic> {
        Ok(SourceDocument::single(if looks_like_v2(source) {
            parse_program_diagnostic(source)?
        } else {
            legacy::parse_program_diagnostic(source)?
        }))
    }
}

/// Return whether source appears to use the native v2 DSL.
pub fn looks_like_v2(source: &str) -> bool {
    source.lines().map(str::trim).any(is_v2_line)
}

/// Parse legacy Hologram source directly into graph IR.
pub(crate) fn parse_legacy_graph(source: &str) -> Result<Graph, CompileError> {
    legacy::parse_graph(source)
}

/// Parse native v2 Hologram source into source IR with diagnostics.
pub fn parse_program_diagnostic(source: &str) -> Result<SourceProgram, SourceDiagnostic> {
    let mut program = SourceProgram::new();
    for (index, line) in source.lines().enumerate() {
        if let Some(item) = parse_source_line_diagnostic(index + 1, line, &mut program)? {
            program.push(item);
        }
    }
    Ok(program)
}

fn is_v2_line(line: &str) -> bool {
    line.starts_with("let ")
        || line.starts_with("input ") && line.contains(": f32[")
        || line.starts_with("const ") && line.contains(": f32[")
}

fn parse_source_line_diagnostic(
    line_number: usize,
    line: &str,
    program: &mut SourceProgram,
) -> Result<Option<SourceItem>, SourceDiagnostic> {
    let (line, base_column) = trim_line(line);
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let parsed = parse_line_diagnostic(line_number, base_column, line)?;
    parsed.into_item(program).map(Some).map_err(|err| {
        diagnostic::from_line(
            line_number,
            base_column,
            line,
            diagnostic::compile_error_kind(&err),
        )
    })
}

fn parse_line_diagnostic(
    line_number: usize,
    base_column: usize,
    input: &str,
) -> Result<ParsedLine<'_>, SourceDiagnostic> {
    match line_parser.parse(input) {
        Ok((remainder, line)) if remainder.trim().is_empty() => Ok(line),
        Ok((remainder, _)) => Err(diagnostic::from_remainder(
            line_number,
            base_column,
            input,
            remainder,
            "source: bad v2 syntax",
        )),
        Err(err) => Err(nom_diagnostic(line_number, base_column, input, err)),
    }
}

fn line_parser(input: &str) -> IResult<&str, ParsedLine<'_>> {
    alt((input_line, const_line, let_line, output_line)).parse(input)
}

fn nom_diagnostic(
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
            "source: bad v2 syntax",
        ),
        NomErr::Incomplete(_) => diagnostic::from_line(
            line_number,
            base_column,
            input,
            "source: incomplete v2 syntax",
        ),
    }
}

fn trim_line(line: &str) -> (&str, usize) {
    let trimmed_start = line.trim_start();
    let base_column = line.len() - trimmed_start.len() + 1;
    (trimmed_start.trim_end(), base_column)
}

enum ParsedLine<'a> {
    Input {
        name: &'a str,
        dims: Vec<u64>,
    },
    Const {
        name: &'a str,
        dims: Vec<u64>,
        values: Vec<&'a str>,
    },
    Let {
        name: &'a str,
        dims: Option<Vec<u64>>,
        op: &'a str,
        items: Vec<CallItem<'a>>,
    },
    Output {
        name: &'a str,
    },
}

enum CallItem<'a> {
    Input(&'a str),
    Attr(ParsedAttr<'a>),
}

struct ParsedCall<'a> {
    inputs: Vec<SourceSymbol>,
    attrs: Vec<ParsedAttr<'a>>,
}

impl<'a> ParsedLine<'a> {
    fn into_item(self, program: &mut SourceProgram) -> Result<SourceItem, CompileError> {
        match self {
            Self::Input { name, dims } => input_item(name, dims, program),
            Self::Const { name, dims, values } => const_item(name, dims, values, program),
            Self::Let {
                name,
                dims,
                op,
                items,
            } => let_item(name, dims, op, items, program),
            Self::Output { name } => {
                Ok(SourceItem::Output(SourceOutput::new(program.intern(name))))
            }
        }
    }
}

fn input_item(
    name: &str,
    dims: Vec<u64>,
    program: &mut SourceProgram,
) -> Result<SourceItem, CompileError> {
    let ty = SourceType::f32(Some(shape_from_dims(dims)?));
    Ok(SourceItem::Input(SourceInput::new(
        program.intern(name),
        ty,
    )))
}

fn const_item(
    name: &str,
    dims: Vec<u64>,
    values: Vec<&str>,
    program: &mut SourceProgram,
) -> Result<SourceItem, CompileError> {
    let ty = SourceType::f32(Some(shape_from_dims(dims)?));
    let literal = literal_from_values(values)?;
    Ok(SourceItem::Const(SourceConst::new(
        program.intern(name),
        ty,
        literal,
    )))
}

fn let_item(
    name: &str,
    dims: Option<Vec<u64>>,
    op: &str,
    items: Vec<CallItem<'_>>,
    program: &mut SourceProgram,
) -> Result<SourceItem, CompileError> {
    let call = call_from_parts(dims, op, items, program)?;
    Ok(SourceItem::Binding(SourceBinding::op(
        Some(program.intern(name)),
        call,
    )))
}

fn call_from_parts(
    dims: Option<Vec<u64>>,
    op: &str,
    items: Vec<CallItem<'_>>,
    program: &mut SourceProgram,
) -> Result<SourceOpCall, CompileError> {
    let op = op_table::parse(op).ok_or(CompileError::SourceParse("op: unknown op kind"))?;
    let parts = split_call_items(items, program);
    let ty = optional_type(dims)?;
    let mut call = SourceOpCall::new(op, parts.inputs, ty);
    call.attrs = attrs_from_assignments(op, parts.attrs)?;
    Ok(call)
}

fn split_call_items<'a>(items: Vec<CallItem<'a>>, program: &mut SourceProgram) -> ParsedCall<'a> {
    let mut inputs = Vec::new();
    let mut attrs = Vec::new();
    for item in items {
        match item {
            CallItem::Input(name) => inputs.push(program.intern(name)),
            CallItem::Attr(attr) => attrs.push(attr),
        }
    }
    ParsedCall { inputs, attrs }
}

fn optional_type(dims: Option<Vec<u64>>) -> Result<Option<SourceType>, CompileError> {
    dims.map(shape_from_dims)
        .transpose()
        .map(|shape| shape.map(|shape| SourceType::f32(Some(shape))))
}

fn shape_from_dims(dims: Vec<u64>) -> Result<ShapeDescriptor, CompileError> {
    let mut shape = [0u64; 8];
    if dims.len() > shape.len() {
        return Err(CompileError::SourceParse("source: bad shape"));
    }
    for (slot, dim) in shape.iter_mut().zip(dims.iter()) {
        *slot = *dim;
    }
    Ok(ShapeDescriptor {
        rank: dims.len() as u8,
        dims: shape,
        dims_overflow: None,
    })
}

fn literal_from_values(values: Vec<&str>) -> Result<SourceTensorLiteral, CompileError> {
    let mut bytes = Vec::new();
    let value_count = values.len();
    for value in values {
        push_f32(value, &mut bytes)?;
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

fn input_line(input: &str) -> IResult<&str, ParsedLine<'_>> {
    let (input, _) = tag("input").parse(input)?;
    let (input, _) = space1(input)?;
    let (input, name) = ident(input)?;
    let (input, _) = colon(input)?;
    let (input, dims) = f32_type(input)?;
    Ok((input, ParsedLine::Input { name, dims }))
}

fn const_line(input: &str) -> IResult<&str, ParsedLine<'_>> {
    let (input, _) = tag("const").parse(input)?;
    let (input, _) = space1(input)?;
    let (input, name) = ident(input)?;
    let (input, _) = colon(input)?;
    let (input, dims) = f32_type(input)?;
    let (input, _) = equals(input)?;
    let (input, values) = literal(input)?;
    Ok((input, ParsedLine::Const { name, dims, values }))
}

fn let_line(input: &str) -> IResult<&str, ParsedLine<'_>> {
    let (input, _) = tag("let").parse(input)?;
    let (input, _) = space1(input)?;
    let (input, name) = ident(input)?;
    let (input, dims) = optional_type_annotation(input)?;
    let (input, _) = equals(input)?;
    let (input, (op, items)) = call(input)?;
    Ok((
        input,
        ParsedLine::Let {
            name,
            dims,
            op,
            items,
        },
    ))
}

fn output_line(input: &str) -> IResult<&str, ParsedLine<'_>> {
    let (input, _) = tag("output").parse(input)?;
    let (input, _) = space1(input)?;
    let (input, name) = ident(input)?;
    Ok((input, ParsedLine::Output { name }))
}

fn optional_type_annotation(input: &str) -> IResult<&str, Option<Vec<u64>>> {
    nom::combinator::opt(preceded(colon, f32_type)).parse(input)
}

fn f32_type(input: &str) -> IResult<&str, Vec<u64>> {
    let (input, _) = tag("f32").parse(input)?;
    let (input, _) = space0(input)?;
    delimited(char('['), separated_list1(comma, dim), close_bracket).parse(input)
}

fn literal(input: &str) -> IResult<&str, Vec<&str>> {
    delimited(
        char('['),
        separated_list1(comma, float_token),
        close_bracket,
    )
    .parse(input)
}

fn call(input: &str) -> IResult<&str, (&str, Vec<CallItem<'_>>)> {
    let (input, op) = ident(input)?;
    let (input, items) =
        delimited(open_paren, separated_list0(comma, call_item), close_paren).parse(input)?;
    Ok((input, (op, items)))
}

fn call_item(input: &str) -> IResult<&str, CallItem<'_>> {
    let (input, name) = preceded(space0, ident).parse(input)?;
    let (input, value) = nom::combinator::opt(preceded(equals, attr_value)).parse(input)?;
    Ok((input, call_item_from_parts(name, value)))
}

fn call_item_from_parts<'a>(name: &'a str, value: Option<AttrValue<'a>>) -> CallItem<'a> {
    match value {
        Some(value) => CallItem::Attr(ParsedAttr { name, value }),
        None => CallItem::Input(name),
    }
}

fn attr_value(input: &str) -> IResult<&str, AttrValue<'_>> {
    alt((attr_bool, attr_list, attr_number)).parse(input)
}

fn attr_bool(input: &str) -> IResult<&str, AttrValue<'_>> {
    alt((
        tag("true").map(|_| AttrValue::Bool(true)),
        tag("false").map(|_| AttrValue::Bool(false)),
    ))
    .parse(input)
}

fn attr_list(input: &str) -> IResult<&str, AttrValue<'_>> {
    delimited(
        char('['),
        separated_list1(comma, number_token),
        close_bracket,
    )
    .map(AttrValue::List)
    .parse(input)
}

fn attr_number(input: &str) -> IResult<&str, AttrValue<'_>> {
    number_token.map(AttrValue::Number).parse(input)
}

fn ident(input: &str) -> IResult<&str, &str> {
    recognize((ident_head, ident_tail)).parse(input)
}

fn ident_head(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c == '_' || c.is_ascii_alphabetic()).parse(input)
}

fn ident_tail(input: &str) -> IResult<&str, &str> {
    take_while(|c: char| c == '_' || c.is_ascii_alphanumeric()).parse(input)
}

fn dim(input: &str) -> IResult<&str, u64> {
    preceded(space0, map_res(digit1, str::parse::<u64>)).parse(input)
}

fn float_token(input: &str) -> IResult<&str, &str> {
    preceded(space0, recognize_float).parse(input)
}

fn number_token(input: &str) -> IResult<&str, &str> {
    preceded(space0, recognize_float).parse(input)
}

fn colon(input: &str) -> IResult<&str, char> {
    let (input, _) = space0(input)?;
    let (input, out) = char(':')(input)?;
    let (input, _) = space0(input)?;
    Ok((input, out))
}

fn equals(input: &str) -> IResult<&str, char> {
    let (input, _) = space0(input)?;
    let (input, out) = char('=')(input)?;
    let (input, _) = space0(input)?;
    Ok((input, out))
}

fn comma(input: &str) -> IResult<&str, char> {
    let (input, _) = space0(input)?;
    let (input, out) = char(',')(input)?;
    let (input, _) = space0(input)?;
    Ok((input, out))
}

fn open_paren(input: &str) -> IResult<&str, char> {
    let (input, _) = space0(input)?;
    char('(')(input)
}

fn close_paren(input: &str) -> IResult<&str, char> {
    let (input, _) = space0(input)?;
    char(')')(input)
}

fn close_bracket(input: &str) -> IResult<&str, char> {
    let (input, _) = space0(input)?;
    char(']')(input)
}
