//! Python source frontend.

use crate::error::CompileError;
use crate::source::frontend::{SourceFrontend, SourceFrontendInfo};
use crate::source::{SourceDiagnostic, SourceDocument, SourceLanguage};

/// Restricted Python builder frontend.
#[derive(Debug, Clone, Copy, Default)]
pub struct PythonFrontend;

impl SourceFrontend for PythonFrontend {
    const INFO: SourceFrontendInfo =
        SourceFrontendInfo::new(SourceLanguage::Python, &["python", "py"], &["py", "py3"]);

    fn parse_document(&self, source: &str) -> Result<SourceDocument, CompileError> {
        parse_document(source)
    }

    fn parse_document_diagnostic(&self, source: &str) -> Result<SourceDocument, SourceDiagnostic> {
        parse_document_diagnostic(source)
    }
}

#[cfg(not(feature = "frontend-python"))]
fn parse_document(_source: &str) -> Result<SourceDocument, CompileError> {
    Err(CompileError::SourceParse("source language unsupported"))
}

#[cfg(not(feature = "frontend-python"))]
fn parse_document_diagnostic(_source: &str) -> Result<SourceDocument, SourceDiagnostic> {
    Err(SourceDiagnostic::global("source language unsupported"))
}

#[cfg(feature = "frontend-python")]
mod enabled {
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    use crate::error::CompileError;
    use crate::source::attrs::{apply_attr, AttrValue, ParsedAttr};
    use crate::source::frontends::pyparse as ast;
    use crate::source::frontends::pyparse::{
        Constant, Expr, Parse, ParseError, Ranged, Stmt, TextRange, TextSize, UnaryOp,
    };
    use crate::source::ir::{
        SourceAttrs, SourceBinding, SourceConst, SourceInput, SourceItem, SourceOpCall,
        SourceOutput, SourceProgram, SourceTensorLiteral, SourceType,
    };
    use crate::source::op_table;
    use crate::source::{diagnostic, SourceDiagnostic, SourceDocument, SourceGraph};
    use hologram_graph::registry::ShapeDescriptor;
    use hologram_graph::OpKind;

    pub(super) fn parse_document(source: &str) -> Result<SourceDocument, CompileError> {
        parse_document_diagnostic(source).map_err(SourceDiagnostic::into_compile_error)
    }

    pub(super) fn parse_document_diagnostic(
        source: &str,
    ) -> Result<SourceDocument, SourceDiagnostic> {
        let source = PythonSource::new(source);
        let suite =
            ast::Suite::parse(source.text, "<embedded>").map_err(|err| source.syntax_error(err))?;
        let mut document = SourceDocument::new();
        for stmt in &suite {
            if let Some(graph) = graph_from_stmt(stmt, &source)? {
                document.push(graph);
            }
        }
        Ok(document)
    }

    fn graph_from_stmt(
        stmt: &Stmt,
        source: &PythonSource<'_>,
    ) -> Result<Option<SourceGraph>, SourceDiagnostic> {
        let Stmt::FunctionDef(function) = stmt else {
            return Ok(None);
        };
        let Some(builder) = builder_arg(function) else {
            return Ok(None);
        };
        if !body_uses_builder(&function.body, builder) {
            return Ok(None);
        }
        let program = graph_program(&function.body, builder, source)?;
        Ok(Some(SourceGraph::named(function.name.as_str(), program)))
    }

    fn graph_program(
        body: &[Stmt],
        builder: &str,
        source: &PythonSource<'_>,
    ) -> Result<SourceProgram, SourceDiagnostic> {
        let mut graph = PythonGraph::new(builder, source);
        for stmt in body {
            graph.push_stmt(stmt)?;
        }
        Ok(graph.program)
    }

    struct PythonSource<'a> {
        text: &'a str,
        line_starts: Vec<usize>,
    }

    impl<'a> PythonSource<'a> {
        fn new(text: &'a str) -> Self {
            Self {
                text,
                line_starts: line_starts(text),
            }
        }

        fn syntax_error(&self, err: ParseError) -> SourceDiagnostic {
            self.at_offset(err.offset, "python: bad syntax")
        }

        fn diagnostic<T: Ranged>(&self, node: &T, kind: &'static str) -> SourceDiagnostic {
            let range = node.range();
            let (line, column) = self.position(range.start());
            SourceDiagnostic::new(line, column, kind, self.rejected(range))
        }

        fn at_offset(&self, offset: TextSize, kind: &'static str) -> SourceDiagnostic {
            let (line, column) = self.position(offset);
            SourceDiagnostic::new(line, column, kind, self.rejected_from(offset))
        }

        fn position(&self, offset: TextSize) -> (usize, usize) {
            let offset = usize::from(offset).min(self.text.len());
            let index = match self.line_starts.binary_search(&offset) {
                Ok(index) => index,
                Err(0) => 0,
                Err(index) => index - 1,
            };
            (index + 1, offset - self.line_starts[index] + 1)
        }

        fn rejected(&self, range: TextRange) -> String {
            let start = usize::from(range.start()).min(self.text.len());
            let end = usize::from(range.end()).min(self.text.len());
            self.fragment(start, end)
        }

        fn rejected_from(&self, offset: TextSize) -> String {
            let offset = usize::from(offset).min(self.text.len());
            let line_end = self.text[offset..]
                .find('\n')
                .map(|delta| offset + delta)
                .unwrap_or(self.text.len());
            self.fragment(offset, line_end)
        }

        fn slice(&self, range: TextRange) -> &'a str {
            let start = usize::from(range.start()).min(self.text.len());
            let end = usize::from(range.end()).min(self.text.len());
            self.text.get(start..end).unwrap_or("").trim()
        }

        fn fragment(&self, start: usize, end: usize) -> String {
            let fragment = self.text.get(start..end).unwrap_or("").trim();
            let line = fragment.lines().next().unwrap_or(fragment).trim();
            if line.is_empty() {
                "<eol>".to_string()
            } else {
                line.to_string()
            }
        }
    }

    fn line_starts(text: &str) -> Vec<usize> {
        let mut starts = Vec::from([0]);
        starts.extend(
            text.bytes()
                .enumerate()
                .filter_map(|(index, byte)| (byte == b'\n').then_some(index + 1)),
        );
        starts
    }

    fn builder_arg(function: &ast::StmtFunctionDef) -> Option<&str> {
        function
            .args
            .args
            .first()
            .or_else(|| function.args.posonlyargs.first())
            .map(|arg| arg.def.arg.as_str())
    }

    fn body_uses_builder(body: &[Stmt], builder: &str) -> bool {
        body.iter().any(|stmt| stmt_uses_builder(stmt, builder))
    }

    fn stmt_uses_builder(stmt: &Stmt, builder: &str) -> bool {
        match stmt {
            Stmt::Assign(assign) => expr_uses_builder(&assign.value, builder),
            Stmt::Expr(expr) => expr_uses_builder(&expr.value, builder),
            _ => false,
        }
    }

    fn expr_uses_builder(expr: &Expr, builder: &str) -> bool {
        match expr {
            Expr::Call(call) => {
                call_path(&call.func).is_some_and(|path| path.first() == Some(&builder))
            }
            _ => false,
        }
    }

    struct PythonGraph<'a, 's> {
        builder: &'a str,
        source: &'a PythonSource<'s>,
        program: SourceProgram,
    }

    impl<'a, 's> PythonGraph<'a, 's> {
        fn new(builder: &'a str, source: &'a PythonSource<'s>) -> Self {
            Self {
                builder,
                source,
                program: SourceProgram::new(),
            }
        }

        fn push_stmt(&mut self, stmt: &Stmt) -> Result<(), SourceDiagnostic> {
            match stmt {
                Stmt::Assign(assign) => self.push_assign(assign),
                Stmt::Expr(expr) if is_docstring(&expr.value) => Ok(()),
                Stmt::Expr(expr) => self.push_expr(&expr.value),
                Stmt::Pass(_) => Ok(()),
                _ => Err(self
                    .source
                    .diagnostic(stmt, "python: unsupported graph statement")),
            }
        }

        fn push_assign(&mut self, assign: &ast::StmtAssign) -> Result<(), SourceDiagnostic> {
            let name = assigned_name(assign, self.source)?;
            let call = call_expr(&assign.value, self.source)?;
            match builder_call(call, self.builder, self.source)? {
                BuilderCall::Input => self.push_input(name, call),
                BuilderCall::Const => self.push_const(name, call),
                BuilderCall::Op(op) => self.push_op(name, op, call),
                BuilderCall::Output => Err(self.source.diagnostic(call, "python: bad output")),
            }
        }

        fn push_expr(&mut self, expr: &Expr) -> Result<(), SourceDiagnostic> {
            let call = call_expr(expr, self.source)?;
            match builder_call(call, self.builder, self.source)? {
                BuilderCall::Output => self.push_output(call),
                _ => Err(self
                    .source
                    .diagnostic(expr, "python: unsupported graph expression")),
            }
        }

        fn push_input(
            &mut self,
            target: &str,
            call: &ast::ExprCall,
        ) -> Result<(), SourceDiagnostic> {
            ensure_keywords(call, &["dtype", "name", "shape"], self.source)?;
            let name = source_name(call, target, self.source)?;
            let ty = source_type(call, self.source)?;
            let symbol = self.program.intern(name);
            self.program
                .push(SourceItem::Input(SourceInput::new(symbol, ty)));
            Ok(())
        }

        fn push_const(
            &mut self,
            target: &str,
            call: &ast::ExprCall,
        ) -> Result<(), SourceDiagnostic> {
            ensure_keywords(call, &["dtype", "name", "shape", "values"], self.source)?;
            let name = source_name(call, target, self.source)?;
            let ty = required_shape_type(call, self.source)?;
            let literal =
                tensor_literal(required_keyword(call, "values", self.source)?, self.source)?;
            let symbol = self.program.intern(name);
            self.program
                .push(SourceItem::Const(SourceConst::new(symbol, ty, literal)));
            Ok(())
        }

        fn push_op(
            &mut self,
            target: &str,
            op: &str,
            call: &ast::ExprCall,
        ) -> Result<(), SourceDiagnostic> {
            let kind = op_table::parse(op).ok_or_else(|| {
                self.source
                    .diagnostic(call.func.as_ref(), "python: unknown op kind")
            })?;
            let inputs = source_inputs(call, &mut self.program, self.source)?;
            let ty = optional_source_type(call, self.source)?;
            let attrs = attrs_from_keywords(kind, call, self.source)?;
            let mut op_call = SourceOpCall::new(kind, inputs, ty);
            op_call.attrs = attrs;
            let symbol = self.program.intern(target);
            self.program.push(SourceItem::Binding(SourceBinding::op(
                Some(symbol),
                op_call,
            )));
            Ok(())
        }

        fn push_output(&mut self, call: &ast::ExprCall) -> Result<(), SourceDiagnostic> {
            let name = output_name(call, self.source)?;
            let symbol = self.program.intern(name);
            self.program
                .push(SourceItem::Output(SourceOutput::new(symbol)));
            Ok(())
        }
    }

    enum BuilderCall<'a> {
        Input,
        Const,
        Op(&'a str),
        Output,
    }

    fn builder_call<'a>(
        call: &'a ast::ExprCall,
        builder: &str,
        source: &PythonSource<'_>,
    ) -> Result<BuilderCall<'a>, SourceDiagnostic> {
        let path = call_path(&call.func)
            .ok_or_else(|| source.diagnostic(call.func.as_ref(), "python: bad call"))?;
        match path.as_slice() {
            [root, "input"] if *root == builder => Ok(BuilderCall::Input),
            [root, "const" | "constant"] if *root == builder => Ok(BuilderCall::Const),
            [root, "ops", op] if *root == builder => Ok(BuilderCall::Op(op)),
            [root, "output"] if *root == builder => Ok(BuilderCall::Output),
            _ => Err(source.diagnostic(call.func.as_ref(), "python: unsupported graph call")),
        }
    }

    fn call_path(expr: &Expr) -> Option<Vec<&str>> {
        let mut path = Vec::new();
        collect_path(expr, &mut path).then_some(path)
    }

    fn collect_path<'a>(expr: &'a Expr, path: &mut Vec<&'a str>) -> bool {
        match expr {
            Expr::Name(name) => {
                path.push(name.id.as_str());
                true
            }
            Expr::Attribute(attribute) => {
                collect_path(&attribute.value, path) && {
                    path.push(attribute.attr.as_str());
                    true
                }
            }
            _ => false,
        }
    }

    fn assigned_name<'a>(
        assign: &'a ast::StmtAssign,
        source: &PythonSource<'_>,
    ) -> Result<&'a str, SourceDiagnostic> {
        match assign.targets.as_slice() {
            [Expr::Name(name)] => Ok(name.id.as_str()),
            [target, ..] => Err(source.diagnostic(target, "python: bad assignment")),
            [] => Err(source.diagnostic(assign, "python: bad assignment")),
        }
    }

    fn call_expr<'a>(
        expr: &'a Expr,
        source: &PythonSource<'_>,
    ) -> Result<&'a ast::ExprCall, SourceDiagnostic> {
        match expr {
            Expr::Call(call) => Ok(call),
            _ => Err(source.diagnostic(expr, "python: expected call")),
        }
    }

    fn source_name<'a>(
        call: &'a ast::ExprCall,
        fallback: &'a str,
        source: &PythonSource<'_>,
    ) -> Result<&'a str, SourceDiagnostic> {
        if let Some(first) = call.args.first() {
            return string_constant(first)
                .ok_or_else(|| source.diagnostic(first, "python: bad name"));
        }
        match keyword(call, "name") {
            Some(name) => {
                string_constant(name).ok_or_else(|| source.diagnostic(name, "python: bad name"))
            }
            None => Ok(fallback),
        }
    }

    fn source_type(
        call: &ast::ExprCall,
        source: &PythonSource<'_>,
    ) -> Result<SourceType, SourceDiagnostic> {
        validate_dtype(call, source)?;
        optional_shape(call, source).map(SourceType::f32)
    }

    fn optional_source_type(
        call: &ast::ExprCall,
        source: &PythonSource<'_>,
    ) -> Result<Option<SourceType>, SourceDiagnostic> {
        validate_dtype(call, source)?;
        optional_shape(call, source).map(|shape| shape.map(|shape| SourceType::f32(Some(shape))))
    }

    fn attrs_from_keywords(
        op: OpKind,
        call: &ast::ExprCall,
        source: &PythonSource<'_>,
    ) -> Result<SourceAttrs, SourceDiagnostic> {
        let mut attrs = SourceAttrs::default();
        for keyword in call
            .keywords
            .iter()
            .filter(|keyword| !is_type_keyword(keyword))
        {
            let name = keyword_name(keyword, source)?;
            let value = attr_value(&keyword.value, source)?;
            let attr = ParsedAttr { name, value };
            apply_attr(op, &mut attrs, attr)
                .map_err(|err| source.diagnostic(keyword, diagnostic::compile_error_kind(&err)))?;
        }
        Ok(attrs)
    }

    fn is_type_keyword(keyword: &ast::Keyword) -> bool {
        keyword
            .arg
            .as_ref()
            .is_some_and(|arg| matches!(arg.as_str(), "dtype" | "shape"))
    }

    fn keyword_name<'a>(
        keyword: &'a ast::Keyword,
        source: &PythonSource<'_>,
    ) -> Result<&'a str, SourceDiagnostic> {
        keyword
            .arg
            .as_ref()
            .map(|arg| arg.as_str())
            .ok_or_else(|| source.diagnostic(keyword, "python: unsupported keyword"))
    }

    fn attr_value<'a>(
        expr: &Expr,
        source: &PythonSource<'a>,
    ) -> Result<AttrValue<'a>, SourceDiagnostic> {
        match expr {
            Expr::Constant(constant) => constant_attr_value(&constant.value, expr, source),
            Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
                Ok(AttrValue::Number(source.slice(expr.range())))
            }
            Expr::List(list) => attr_list(&list.elts, source),
            Expr::Tuple(tuple) => attr_list(&tuple.elts, source),
            _ => Err(source.diagnostic(expr, "python: bad attr value")),
        }
    }

    fn constant_attr_value<'a>(
        value: &Constant,
        expr: &Expr,
        source: &PythonSource<'a>,
    ) -> Result<AttrValue<'a>, SourceDiagnostic> {
        match value {
            Constant::Bool(value) => Ok(AttrValue::Bool(*value)),
            Constant::Float(_) | Constant::Int(_) => {
                Ok(AttrValue::Number(source.slice(expr.range())))
            }
            _ => Err(source.diagnostic(expr, "python: bad attr value")),
        }
    }

    fn attr_list<'a>(
        values: &[Expr],
        source: &PythonSource<'a>,
    ) -> Result<AttrValue<'a>, SourceDiagnostic> {
        values
            .iter()
            .map(|value| attr_number(value, source))
            .collect::<Result<Vec<_>, _>>()
            .map(AttrValue::List)
    }

    fn attr_number<'a>(
        expr: &Expr,
        source: &PythonSource<'a>,
    ) -> Result<&'a str, SourceDiagnostic> {
        match expr {
            Expr::Constant(constant)
                if matches!(&constant.value, Constant::Float(_) | Constant::Int(_)) =>
            {
                Ok(source.slice(expr.range()))
            }
            Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
                Ok(source.slice(expr.range()))
            }
            _ => Err(source.diagnostic(expr, "python: bad attr value")),
        }
    }

    fn required_shape_type(
        call: &ast::ExprCall,
        source: &PythonSource<'_>,
    ) -> Result<SourceType, SourceDiagnostic> {
        validate_dtype(call, source)?;
        let shape = required_shape(call, source)?;
        Ok(SourceType::f32(Some(shape)))
    }

    fn validate_dtype(
        call: &ast::ExprCall,
        source: &PythonSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        match keyword(call, "dtype").and_then(string_constant) {
            Some("f32") | None => Ok(()),
            _ => Err(source.diagnostic(
                keyword(call, "dtype").unwrap_or(call.func.as_ref()),
                "python: unsupported dtype",
            )),
        }
    }

    fn optional_shape(
        call: &ast::ExprCall,
        source: &PythonSource<'_>,
    ) -> Result<Option<ShapeDescriptor>, SourceDiagnostic> {
        keyword(call, "shape")
            .map(|expr| shape_literal(expr, source))
            .transpose()
    }

    fn required_shape(
        call: &ast::ExprCall,
        source: &PythonSource<'_>,
    ) -> Result<ShapeDescriptor, SourceDiagnostic> {
        keyword(call, "shape")
            .ok_or_else(|| source.diagnostic(call, "python: missing shape"))
            .and_then(|expr| shape_literal(expr, source))
    }

    fn shape_literal(
        expr: &Expr,
        source: &PythonSource<'_>,
    ) -> Result<ShapeDescriptor, SourceDiagnostic> {
        let dims = integer_list(expr, source)?;
        if dims.is_empty() || dims.len() > 8 {
            return Err(source.diagnostic(expr, "python: bad shape"));
        }
        Ok(shape_from_dims(&dims))
    }

    fn shape_from_dims(dims: &[u64]) -> ShapeDescriptor {
        let mut packed = [0u64; 8];
        packed[..dims.len()].copy_from_slice(dims);
        ShapeDescriptor {
            rank: dims.len() as u8,
            dims: packed,
            dims_overflow: None,
        }
    }

    fn tensor_literal(
        expr: &Expr,
        source: &PythonSource<'_>,
    ) -> Result<SourceTensorLiteral, SourceDiagnostic> {
        let values = numeric_list(expr, source)?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(SourceTensorLiteral::new(bytes, values.len()))
    }

    fn source_inputs(
        call: &ast::ExprCall,
        program: &mut SourceProgram,
        source: &PythonSource<'_>,
    ) -> Result<Vec<crate::source::SourceSymbol>, SourceDiagnostic> {
        call.args
            .iter()
            .map(|expr| symbol_ref(expr, source))
            .map(|name| name.map(|name| program.intern(name)))
            .collect()
    }

    fn symbol_ref<'a>(
        expr: &'a Expr,
        source: &PythonSource<'_>,
    ) -> Result<&'a str, SourceDiagnostic> {
        match expr {
            Expr::Name(name) => Ok(name.id.as_str()),
            _ => Err(source.diagnostic(expr, "python: bad input ref")),
        }
    }

    fn output_name<'a>(
        call: &'a ast::ExprCall,
        source: &PythonSource<'_>,
    ) -> Result<&'a str, SourceDiagnostic> {
        match call.args.as_slice() {
            [Expr::Name(name)] => Ok(name.id.as_str()),
            [alias, Expr::Name(name)] if string_constant(alias).is_some() => Ok(name.id.as_str()),
            [bad, ..] => Err(source.diagnostic(bad, "python: bad output")),
            [] => Err(source.diagnostic(call, "python: bad output")),
        }
    }

    fn required_keyword<'a>(
        call: &'a ast::ExprCall,
        name: &str,
        source: &PythonSource<'_>,
    ) -> Result<&'a Expr, SourceDiagnostic> {
        keyword(call, name).ok_or_else(|| source.diagnostic(call, "python: missing keyword"))
    }

    fn keyword<'a>(call: &'a ast::ExprCall, name: &str) -> Option<&'a Expr> {
        call.keywords
            .iter()
            .find(|keyword| keyword.arg.as_ref().is_some_and(|arg| arg.as_str() == name))
            .map(|keyword| &keyword.value)
    }

    fn ensure_keywords(
        call: &ast::ExprCall,
        allowed: &[&str],
        source: &PythonSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        let unknown = call.keywords.iter().find(|keyword| {
            keyword
                .arg
                .as_ref()
                .is_none_or(|arg| !allowed.contains(&arg.as_str()))
        });
        if let Some(keyword) = unknown {
            return Err(source.diagnostic(keyword, "python: unsupported keyword"));
        }
        Ok(())
    }

    fn string_constant(expr: &Expr) -> Option<&str> {
        match expr {
            Expr::Constant(constant) => match &constant.value {
                Constant::Str(value) => Some(value.as_str()),
                _ => None,
            },
            _ => None,
        }
    }

    fn integer_list(expr: &Expr, source: &PythonSource<'_>) -> Result<Vec<u64>, SourceDiagnostic> {
        expr_list(expr, source)?
            .iter()
            .map(|expr| integer_value(expr, source))
            .collect::<Result<Vec<_>, _>>()
    }

    fn numeric_list(expr: &Expr, source: &PythonSource<'_>) -> Result<Vec<f32>, SourceDiagnostic> {
        expr_list(expr, source)?
            .iter()
            .map(|expr| numeric_value(expr, source))
            .collect::<Result<Vec<_>, _>>()
    }

    fn expr_list<'a>(
        expr: &'a Expr,
        source: &PythonSource<'_>,
    ) -> Result<&'a [Expr], SourceDiagnostic> {
        match expr {
            Expr::List(list) => Ok(&list.elts),
            Expr::Tuple(tuple) => Ok(&tuple.elts),
            _ => Err(source.diagnostic(expr, "python: expected list")),
        }
    }

    fn integer_value(expr: &Expr, source: &PythonSource<'_>) -> Result<u64, SourceDiagnostic> {
        match expr {
            Expr::Constant(constant) => match &constant.value {
                Constant::Int(value) => value
                    .to_string()
                    .parse()
                    .map_err(|_| source.diagnostic(expr, "python: bad integer")),
                _ => Err(source.diagnostic(expr, "python: bad integer")),
            },
            _ => Err(source.diagnostic(expr, "python: bad integer")),
        }
    }

    fn numeric_value(expr: &Expr, source: &PythonSource<'_>) -> Result<f32, SourceDiagnostic> {
        match expr {
            Expr::Constant(constant) => constant_value(&constant.value, expr, source),
            Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => {
                numeric_value(&unary.operand, source).map(|value| -value)
            }
            Expr::UnaryOp(unary) if unary.op == UnaryOp::UAdd => {
                numeric_value(&unary.operand, source)
            }
            _ => Err(source.diagnostic(expr, "python: bad numeric value")),
        }
    }

    fn constant_value(
        value: &Constant,
        expr: &Expr,
        source: &PythonSource<'_>,
    ) -> Result<f32, SourceDiagnostic> {
        match value {
            Constant::Float(value) => Ok(*value as f32),
            Constant::Int(value) => value
                .to_string()
                .parse()
                .map_err(|_| source.diagnostic(expr, "python: bad numeric value")),
            _ => Err(source.diagnostic(expr, "python: bad numeric value")),
        }
    }

    fn is_docstring(expr: &Expr) -> bool {
        matches!(expr, Expr::Constant(constant) if matches!(constant.value, Constant::Str(_)))
    }
}

#[cfg(feature = "frontend-python")]
use enabled::{parse_document, parse_document_diagnostic};
