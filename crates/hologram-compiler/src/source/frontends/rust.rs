//! Rust source frontend.

use crate::error::CompileError;
use crate::source::frontend::{SourceFrontend, SourceFrontendInfo};
use crate::source::{SourceDiagnostic, SourceDocument, SourceLanguage};

/// Restricted Rust builder frontend.
#[derive(Debug, Clone, Copy, Default)]
pub struct RustFrontend;

impl SourceFrontend for RustFrontend {
    const INFO: SourceFrontendInfo =
        SourceFrontendInfo::new(SourceLanguage::Rust, &["rust", "rs"], &["rs"]);

    fn parse_document(&self, source: &str) -> Result<SourceDocument, CompileError> {
        parse_document(source)
    }

    fn parse_document_diagnostic(&self, source: &str) -> Result<SourceDocument, SourceDiagnostic> {
        parse_document_diagnostic(source)
    }
}

#[cfg(not(feature = "frontend-rust"))]
fn parse_document(_source: &str) -> Result<SourceDocument, CompileError> {
    Err(CompileError::SourceParse("source language unsupported"))
}

#[cfg(not(feature = "frontend-rust"))]
fn parse_document_diagnostic(_source: &str) -> Result<SourceDocument, SourceDiagnostic> {
    Err(SourceDiagnostic::global("source language unsupported"))
}

#[cfg(feature = "frontend-rust")]
mod enabled {
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    use crate::error::CompileError;
    use crate::source::attrs::{apply_attr, AttrValue, ParsedAttr};
    use crate::source::diagnostic;
    use crate::source::ir::{
        SourceAttrs, SourceBinding, SourceConst, SourceInput, SourceItem, SourceOpCall,
        SourceOutput, SourceProgram, SourceTensorLiteral, SourceType,
    };
    use crate::source::{op_table, SourceDiagnostic, SourceDocument, SourceGraph, SourceSymbol};
    use hologram_graph::registry::ShapeDescriptor;
    use hologram_graph::OpKind;
    use proc_macro2::{LineColumn, Span};
    use syn::spanned::Spanned;
    use syn::{
        Expr, ExprArray, ExprCall, ExprMethodCall, FnArg, Item, ItemFn, Lit, Pat, Stmt, UnOp,
    };

    pub(super) fn parse_document(source: &str) -> Result<SourceDocument, CompileError> {
        parse_document_diagnostic(source).map_err(SourceDiagnostic::into_compile_error)
    }

    pub(super) fn parse_document_diagnostic(
        source: &str,
    ) -> Result<SourceDocument, SourceDiagnostic> {
        let source = RustSource::new(source);
        let file = syn::parse_file(source.text).map_err(|err| source.syntax_error(&err))?;
        let mut document = SourceDocument::new();
        for item in &file.items {
            if let Some(graph) = graph_from_item(item, &source)? {
                document.push(graph);
            }
        }
        Ok(document)
    }

    fn graph_from_item(
        item: &Item,
        source: &RustSource<'_>,
    ) -> Result<Option<SourceGraph>, SourceDiagnostic> {
        let Item::Fn(function) = item else {
            return Ok(None);
        };
        graph_from_function(function, source)
    }

    fn graph_from_function(
        function: &ItemFn,
        source: &RustSource<'_>,
    ) -> Result<Option<SourceGraph>, SourceDiagnostic> {
        let Some(builder) = builder_arg(function) else {
            return Ok(None);
        };
        if !body_uses_builder(&function.block.stmts, &builder) {
            return Ok(None);
        }
        let program = graph_program(&function.block.stmts, &builder, source)?;
        Ok(Some(SourceGraph::named(
            function.sig.ident.to_string(),
            program,
        )))
    }

    fn graph_program(
        body: &[Stmt],
        builder: &str,
        source: &RustSource<'_>,
    ) -> Result<SourceProgram, SourceDiagnostic> {
        let mut graph = RustGraph::new(builder, source);
        for stmt in body {
            graph.push_stmt(stmt)?;
        }
        Ok(graph.program)
    }

    struct RustSource<'a> {
        text: &'a str,
        line_starts: Vec<usize>,
    }

    impl<'a> RustSource<'a> {
        fn new(text: &'a str) -> Self {
            Self {
                text,
                line_starts: line_starts(text),
            }
        }

        fn syntax_error(&self, err: &syn::Error) -> SourceDiagnostic {
            self.diagnostic_span(err.span(), "rust: bad syntax")
        }

        fn diagnostic<T: Spanned>(&self, node: &T, kind: &'static str) -> SourceDiagnostic {
            self.diagnostic_span(node.span(), kind)
        }

        fn diagnostic_span(&self, span: Span, kind: &'static str) -> SourceDiagnostic {
            let start = span.start();
            let (line, column) = (start.line, start.column + 1);
            SourceDiagnostic::new(line, column, kind, self.rejected(span))
        }

        fn rejected(&self, span: Span) -> String {
            let start = self.offset(span.start());
            let end = self.offset(span.end());
            self.fragment(start, end)
        }

        fn slice(&self, span: Span) -> &'a str {
            let start = self.offset(span.start());
            let end = self.offset(span.end());
            self.text.get(start..end).unwrap_or("").trim()
        }

        fn offset(&self, position: LineColumn) -> usize {
            let line = position.line.saturating_sub(1);
            let start = self.line_starts.get(line).copied().unwrap_or(0);
            let end = self.line_end(line, start);
            start + column_byte_offset(&self.text[start..end], position.column)
        }

        fn line_end(&self, line: usize, start: usize) -> usize {
            self.line_starts
                .get(line + 1)
                .map(|next| next.saturating_sub(1))
                .unwrap_or(self.text.len())
                .max(start)
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

    fn column_byte_offset(line: &str, column: usize) -> usize {
        line.char_indices()
            .map(|(index, _)| index)
            .nth(column)
            .unwrap_or(line.len())
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

    fn builder_arg(function: &ItemFn) -> Option<String> {
        let first = function.sig.inputs.first()?;
        match first {
            FnArg::Typed(arg) => match arg.pat.as_ref() {
                Pat::Ident(ident) => Some(ident.ident.to_string()),
                _ => None,
            },
            FnArg::Receiver(_) => None,
        }
    }

    fn body_uses_builder(body: &[Stmt], builder: &str) -> bool {
        body.iter().any(|stmt| stmt_uses_builder(stmt, builder))
    }

    fn stmt_uses_builder(stmt: &Stmt, builder: &str) -> bool {
        match stmt {
            Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| expr_uses_builder(&init.expr, builder)),
            Stmt::Expr(expr, _) => expr_uses_builder(expr, builder),
            _ => false,
        }
    }

    fn expr_uses_builder(expr: &Expr, builder: &str) -> bool {
        match expr {
            Expr::MethodCall(call) => receiver_root(&call.receiver).as_deref() == Some(builder),
            _ => false,
        }
    }

    struct RustGraph<'a, 's> {
        builder: &'a str,
        source: &'a RustSource<'s>,
        program: SourceProgram,
    }

    impl<'a, 's> RustGraph<'a, 's> {
        fn new(builder: &'a str, source: &'a RustSource<'s>) -> Self {
            Self {
                builder,
                source,
                program: SourceProgram::new(),
            }
        }

        fn push_stmt(&mut self, stmt: &Stmt) -> Result<(), SourceDiagnostic> {
            match stmt {
                Stmt::Local(local) => self.push_local(local),
                Stmt::Expr(expr, _) if is_directive(expr) => Ok(()),
                Stmt::Expr(expr @ Expr::MethodCall(_), _) => self.push_expr(expr),
                Stmt::Expr(_, _) => Err(self
                    .source
                    .diagnostic(stmt, "rust: unsupported graph statement")),
                _ => Err(self
                    .source
                    .diagnostic(stmt, "rust: unsupported graph statement")),
            }
        }

        fn push_local(&mut self, local: &syn::Local) -> Result<(), SourceDiagnostic> {
            let name = assigned_name(local, self.source)?;
            let init = local
                .init
                .as_ref()
                .ok_or_else(|| self.source.diagnostic(local, "rust: bad assignment"))?;
            if init.diverge.is_some() {
                return Err(self.source.diagnostic(&init.expr, "rust: bad assignment"));
            }
            let call = method_call(&init.expr, self.source)?;
            match builder_call(call, self.builder, self.source)? {
                BuilderCall::Input => self.push_input(&name, call),
                BuilderCall::Const => self.push_const(&name, call),
                BuilderCall::Op(op) => self.push_op(&name, &op, call),
                BuilderCall::Output => Err(self.source.diagnostic(call, "rust: bad output")),
            }
        }

        fn push_expr(&mut self, expr: &Expr) -> Result<(), SourceDiagnostic> {
            let call = method_call(expr, self.source)?;
            match builder_call(call, self.builder, self.source)? {
                BuilderCall::Output => self.push_output(call),
                _ => Err(self
                    .source
                    .diagnostic(expr, "rust: unsupported graph expression")),
            }
        }

        fn push_input(
            &mut self,
            target: &str,
            call: &ExprMethodCall,
        ) -> Result<(), SourceDiagnostic> {
            ensure_options(call, &["dtype", "name", "shape"], self.source)?;
            let name = source_name(call, target, self.source)?;
            let ty = source_type(call, self.source)?;
            let symbol = self.program.intern(&name);
            self.program
                .push(SourceItem::Input(SourceInput::new(symbol, ty)));
            Ok(())
        }

        fn push_const(
            &mut self,
            target: &str,
            call: &ExprMethodCall,
        ) -> Result<(), SourceDiagnostic> {
            ensure_options(call, &["dtype", "name", "shape", "values"], self.source)?;
            let name = source_name(call, target, self.source)?;
            let ty = required_shape_type(call, self.source)?;
            let literal =
                tensor_literal(required_option(call, "values", self.source)?, self.source)?;
            let symbol = self.program.intern(&name);
            self.program
                .push(SourceItem::Const(SourceConst::new(symbol, ty, literal)));
            Ok(())
        }

        fn push_op(
            &mut self,
            target: &str,
            op: &str,
            call: &ExprMethodCall,
        ) -> Result<(), SourceDiagnostic> {
            let kind = op_table::parse(op)
                .ok_or_else(|| self.source.diagnostic(call, "rust: unknown op kind"))?;
            let inputs = source_inputs(call, &mut self.program, self.source)?;
            let ty = optional_source_type(call, self.source)?;
            let mut op_call = SourceOpCall::new(kind, inputs, ty);
            op_call.attrs = attrs_from_options(kind, call, self.source)?;
            let symbol = self.program.intern(target);
            self.program.push(SourceItem::Binding(SourceBinding::op(
                Some(symbol),
                op_call,
            )));
            Ok(())
        }

        fn push_output(&mut self, call: &ExprMethodCall) -> Result<(), SourceDiagnostic> {
            let name = output_name(call, self.source)?;
            let symbol = self.program.intern(&name);
            self.program
                .push(SourceItem::Output(SourceOutput::new(symbol)));
            Ok(())
        }
    }

    enum BuilderCall {
        Input,
        Const,
        Op(String),
        Output,
    }

    fn builder_call(
        call: &ExprMethodCall,
        builder: &str,
        source: &RustSource<'_>,
    ) -> Result<BuilderCall, SourceDiagnostic> {
        let method = call.method.to_string();
        if receiver_is_ops_call(&call.receiver, builder) {
            return Ok(BuilderCall::Op(method));
        }
        if receiver_root(&call.receiver).as_deref() == Some(builder) {
            return direct_builder_call(method.as_str(), call, source);
        }
        Err(source.diagnostic(call, "rust: unsupported graph call"))
    }

    fn direct_builder_call(
        method: &str,
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<BuilderCall, SourceDiagnostic> {
        match method {
            "input" => Ok(BuilderCall::Input),
            "constant" | "const_" => Ok(BuilderCall::Const),
            "output" => Ok(BuilderCall::Output),
            _ => Err(source.diagnostic(call, "rust: unsupported graph call")),
        }
    }

    fn receiver_is_ops_call(expr: &Expr, builder: &str) -> bool {
        match expr {
            Expr::MethodCall(call) => {
                call.method == "ops"
                    && call.args.is_empty()
                    && receiver_root(&call.receiver).as_deref() == Some(builder)
            }
            _ => false,
        }
    }

    fn receiver_root(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(path) => single_path_ident(path),
            Expr::MethodCall(call) => receiver_root(&call.receiver),
            Expr::Paren(paren) => receiver_root(&paren.expr),
            Expr::Reference(reference) => receiver_root(&reference.expr),
            _ => None,
        }
    }

    fn assigned_name(
        local: &syn::Local,
        source: &RustSource<'_>,
    ) -> Result<String, SourceDiagnostic> {
        match &local.pat {
            Pat::Ident(ident) => Ok(ident.ident.to_string()),
            _ => Err(source.diagnostic(&local.pat, "rust: bad assignment")),
        }
    }

    fn method_call<'a>(
        expr: &'a Expr,
        source: &RustSource<'_>,
    ) -> Result<&'a ExprMethodCall, SourceDiagnostic> {
        match expr {
            Expr::MethodCall(call) => Ok(call),
            _ => Err(source.diagnostic(expr, "rust: expected builder call")),
        }
    }

    fn source_name(
        call: &ExprMethodCall,
        fallback: &str,
        source: &RustSource<'_>,
    ) -> Result<String, SourceDiagnostic> {
        if let Some(first) = positional_args(call).first() {
            return string_literal(first).ok_or_else(|| source.diagnostic(first, "rust: bad name"));
        }
        match option_value(call, "name", source)? {
            Some(expr) => {
                string_literal(expr).ok_or_else(|| source.diagnostic(expr, "rust: bad name"))
            }
            None => Ok(fallback.to_string()),
        }
    }

    fn source_type(
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<SourceType, SourceDiagnostic> {
        validate_dtype(call, source)?;
        optional_shape(call, source).map(SourceType::f32)
    }

    fn optional_source_type(
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<Option<SourceType>, SourceDiagnostic> {
        validate_dtype(call, source)?;
        optional_shape(call, source).map(|shape| shape.map(|shape| SourceType::f32(Some(shape))))
    }

    fn attrs_from_options(
        op: OpKind,
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<SourceAttrs, SourceDiagnostic> {
        let mut attrs = SourceAttrs::default();
        for option in option_calls(call) {
            apply_option_attr(op, &mut attrs, option, source)?;
        }
        Ok(attrs)
    }

    fn apply_option_attr(
        op: OpKind,
        attrs: &mut SourceAttrs,
        option: &ExprCall,
        source: &RustSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        let name = option_name(option, source)?;
        if matches!(name.as_str(), "dtype" | "shape") {
            return Ok(());
        }
        let attr = ParsedAttr {
            name: name.as_str(),
            value: attr_value(single_option_arg(option, source)?, source)?,
        };
        apply_attr(op, attrs, attr)
            .map_err(|err| source.diagnostic(option, diagnostic::compile_error_kind(&err)))
    }

    fn attr_value<'a>(
        expr: &Expr,
        source: &RustSource<'a>,
    ) -> Result<AttrValue<'a>, SourceDiagnostic> {
        match expr {
            Expr::Lit(lit) => literal_attr_value(&lit.lit, expr, source),
            Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
                Ok(AttrValue::Number(source.slice(expr.span())))
            }
            Expr::Array(array) => attr_list(array, source),
            Expr::Reference(reference) => attr_value(&reference.expr, source),
            _ => Err(source.diagnostic(expr, "rust: bad attr value")),
        }
    }

    fn literal_attr_value<'a>(
        lit: &Lit,
        expr: &Expr,
        source: &RustSource<'a>,
    ) -> Result<AttrValue<'a>, SourceDiagnostic> {
        match lit {
            Lit::Bool(value) => Ok(AttrValue::Bool(value.value)),
            Lit::Float(_) | Lit::Int(_) => Ok(AttrValue::Number(source.slice(expr.span()))),
            _ => Err(source.diagnostic(expr, "rust: bad attr value")),
        }
    }

    fn attr_list<'a>(
        array: &ExprArray,
        source: &RustSource<'a>,
    ) -> Result<AttrValue<'a>, SourceDiagnostic> {
        array
            .elems
            .iter()
            .map(|expr| attr_number(expr, source))
            .collect::<Result<Vec<_>, _>>()
            .map(AttrValue::List)
    }

    fn attr_number<'a>(expr: &Expr, source: &RustSource<'a>) -> Result<&'a str, SourceDiagnostic> {
        match expr {
            Expr::Lit(lit) if matches!(lit.lit, Lit::Float(_) | Lit::Int(_)) => {
                Ok(source.slice(expr.span()))
            }
            Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => Ok(source.slice(expr.span())),
            _ => Err(source.diagnostic(expr, "rust: bad attr value")),
        }
    }

    fn required_shape_type(
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<SourceType, SourceDiagnostic> {
        validate_dtype(call, source)?;
        let shape = required_shape(call, source)?;
        Ok(SourceType::f32(Some(shape)))
    }

    fn validate_dtype(
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        match option_value(call, "dtype", source)? {
            Some(expr) if string_literal(expr).as_deref() == Some("f32") => Ok(()),
            Some(expr) => Err(source.diagnostic(expr, "rust: unsupported dtype")),
            None => Ok(()),
        }
    }

    fn optional_shape(
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<Option<ShapeDescriptor>, SourceDiagnostic> {
        option_value(call, "shape", source)?
            .map(|expr| shape_literal(expr, source))
            .transpose()
    }

    fn required_shape(
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<ShapeDescriptor, SourceDiagnostic> {
        required_option(call, "shape", source).and_then(|expr| shape_literal(expr, source))
    }

    fn shape_literal(
        expr: &Expr,
        source: &RustSource<'_>,
    ) -> Result<ShapeDescriptor, SourceDiagnostic> {
        let dims = integer_list(expr, source)?;
        if dims.is_empty() || dims.len() > 8 {
            return Err(source.diagnostic(expr, "rust: bad shape"));
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
        source: &RustSource<'_>,
    ) -> Result<SourceTensorLiteral, SourceDiagnostic> {
        let values = numeric_list(expr, source)?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(SourceTensorLiteral::new(bytes, values.len()))
    }

    fn source_inputs(
        call: &ExprMethodCall,
        program: &mut SourceProgram,
        source: &RustSource<'_>,
    ) -> Result<Vec<SourceSymbol>, SourceDiagnostic> {
        positional_args(call)
            .iter()
            .map(|expr| symbol_ref(expr, source))
            .map(|name| name.map(|name| program.intern(&name)))
            .collect()
    }

    fn symbol_ref(expr: &Expr, source: &RustSource<'_>) -> Result<String, SourceDiagnostic> {
        match expr {
            Expr::Path(path) => single_path_ident(path)
                .ok_or_else(|| source.diagnostic(expr, "rust: bad input ref")),
            _ => Err(source.diagnostic(expr, "rust: bad input ref")),
        }
    }

    fn output_name(
        call: &ExprMethodCall,
        source: &RustSource<'_>,
    ) -> Result<String, SourceDiagnostic> {
        match call.args.iter().collect::<Vec<_>>().as_slice() {
            [arg] => output_symbol_arg(arg, source),
            [alias, arg] if string_literal(alias).is_some() => output_symbol_arg(arg, source),
            [bad, ..] => Err(source.diagnostic(*bad, "rust: bad output")),
            [] => Err(source.diagnostic(call, "rust: bad output")),
        }
    }

    fn output_symbol_arg(expr: &Expr, source: &RustSource<'_>) -> Result<String, SourceDiagnostic> {
        match expr {
            Expr::Path(path) => {
                single_path_ident(path).ok_or_else(|| source.diagnostic(expr, "rust: bad output"))
            }
            _ => Err(source.diagnostic(expr, "rust: bad output")),
        }
    }

    fn required_option<'a>(
        call: &'a ExprMethodCall,
        name: &str,
        source: &RustSource<'_>,
    ) -> Result<&'a Expr, SourceDiagnostic> {
        option_value(call, name, source)?
            .ok_or_else(|| source.diagnostic(call, "rust: missing option"))
    }

    fn option_value<'a>(
        call: &'a ExprMethodCall,
        name: &str,
        source: &RustSource<'_>,
    ) -> Result<Option<&'a Expr>, SourceDiagnostic> {
        for option in option_calls(call) {
            if option_name(option, source)? == name {
                return single_option_arg(option, source).map(Some);
            }
        }
        Ok(None)
    }

    fn ensure_options(
        call: &ExprMethodCall,
        allowed: &[&str],
        source: &RustSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        for option in option_calls(call) {
            let name = option_name(option, source)?;
            if !allowed.contains(&name.as_str()) {
                return Err(source.diagnostic(option, "rust: unsupported option"));
            }
        }
        Ok(())
    }

    fn option_calls(call: &ExprMethodCall) -> impl Iterator<Item = &ExprCall> {
        call.args.iter().filter_map(option_call)
    }

    fn positional_args(call: &ExprMethodCall) -> Vec<&Expr> {
        call.args
            .iter()
            .filter(|expr| option_call(expr).is_none())
            .collect()
    }

    fn option_call(expr: &Expr) -> Option<&ExprCall> {
        match expr {
            Expr::Call(call) if option_name_unchecked(call).is_some() => Some(call),
            _ => None,
        }
    }

    fn option_name(call: &ExprCall, source: &RustSource<'_>) -> Result<String, SourceDiagnostic> {
        option_name_unchecked(call).ok_or_else(|| source.diagnostic(call, "rust: bad option"))
    }

    fn option_name_unchecked(call: &ExprCall) -> Option<String> {
        match call.func.as_ref() {
            Expr::Path(path) => single_path_ident(path),
            _ => None,
        }
    }

    fn single_option_arg<'a>(
        call: &'a ExprCall,
        source: &RustSource<'_>,
    ) -> Result<&'a Expr, SourceDiagnostic> {
        let mut args = call.args.iter();
        let Some(first) = args.next() else {
            return Err(source.diagnostic(call, "rust: bad option"));
        };
        if args.next().is_some() {
            return Err(source.diagnostic(call, "rust: bad option"));
        }
        Ok(first)
    }

    fn string_literal(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Lit(lit) => match &lit.lit {
                Lit::Str(value) => Some(value.value()),
                _ => None,
            },
            _ => None,
        }
    }

    fn integer_list(expr: &Expr, source: &RustSource<'_>) -> Result<Vec<u64>, SourceDiagnostic> {
        expr_array(expr, source)?
            .elems
            .iter()
            .map(|expr| integer_value(expr, source))
            .collect()
    }

    fn numeric_list(expr: &Expr, source: &RustSource<'_>) -> Result<Vec<f32>, SourceDiagnostic> {
        expr_array(expr, source)?
            .elems
            .iter()
            .map(|expr| numeric_value(expr, source))
            .collect()
    }

    fn expr_array<'a>(
        expr: &'a Expr,
        source: &RustSource<'_>,
    ) -> Result<&'a ExprArray, SourceDiagnostic> {
        match expr {
            Expr::Array(array) => Ok(array),
            Expr::Reference(reference) => expr_array(&reference.expr, source),
            _ => Err(source.diagnostic(expr, "rust: expected array")),
        }
    }

    fn integer_value(expr: &Expr, source: &RustSource<'_>) -> Result<u64, SourceDiagnostic> {
        match expr {
            Expr::Lit(lit) => match &lit.lit {
                Lit::Int(value) => value
                    .base10_parse()
                    .map_err(|_| source.diagnostic(expr, "rust: bad integer")),
                _ => Err(source.diagnostic(expr, "rust: bad integer")),
            },
            _ => Err(source.diagnostic(expr, "rust: bad integer")),
        }
    }

    fn numeric_value(expr: &Expr, source: &RustSource<'_>) -> Result<f32, SourceDiagnostic> {
        match expr {
            Expr::Lit(lit) => literal_number(&lit.lit, expr, source),
            Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
                numeric_value(&unary.expr, source).map(|value| -value)
            }
            _ => Err(source.diagnostic(expr, "rust: bad numeric value")),
        }
    }

    fn literal_number(
        lit: &Lit,
        expr: &Expr,
        source: &RustSource<'_>,
    ) -> Result<f32, SourceDiagnostic> {
        match lit {
            Lit::Float(value) => value
                .base10_parse()
                .map_err(|_| source.diagnostic(expr, "rust: bad numeric value")),
            Lit::Int(value) => value
                .base10_parse()
                .map_err(|_| source.diagnostic(expr, "rust: bad numeric value")),
            _ => Err(source.diagnostic(expr, "rust: bad numeric value")),
        }
    }

    fn single_path_ident(path: &syn::ExprPath) -> Option<String> {
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return None;
        }
        path.path
            .segments
            .iter()
            .next()
            .map(|segment| segment.ident.to_string())
    }

    fn is_directive(expr: &Expr) -> bool {
        matches!(expr, Expr::Lit(lit) if matches!(lit.lit, Lit::Str(_)))
    }
}

#[cfg(feature = "frontend-rust")]
use enabled::{parse_document, parse_document_diagnostic};
