//! TypeScript source frontend.

use crate::error::CompileError;
use crate::source::frontend::{SourceFrontend, SourceFrontendInfo};
use crate::source::{SourceDiagnostic, SourceDocument, SourceLanguage};

/// Restricted TypeScript builder frontend.
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeScriptFrontend;

impl SourceFrontend for TypeScriptFrontend {
    const INFO: SourceFrontendInfo = SourceFrontendInfo::new(
        SourceLanguage::TypeScript,
        &["typescript", "ts"],
        &["ts", "tsx"],
    );

    fn parse_document(&self, source: &str) -> Result<SourceDocument, CompileError> {
        parse_document(source)
    }

    fn parse_document_diagnostic(&self, source: &str) -> Result<SourceDocument, SourceDiagnostic> {
        parse_document_diagnostic(source)
    }
}

#[cfg(not(feature = "frontend-typescript"))]
fn parse_document(_source: &str) -> Result<SourceDocument, CompileError> {
    Err(CompileError::SourceParse("source language unsupported"))
}

#[cfg(not(feature = "frontend-typescript"))]
fn parse_document_diagnostic(_source: &str) -> Result<SourceDocument, SourceDiagnostic> {
    Err(SourceDiagnostic::global("source language unsupported"))
}

#[cfg(feature = "frontend-typescript")]
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
    use swc_common::{sync::Lrc, BytePos, FileName, SourceMap, Span, Spanned};
    use swc_ecma_ast::{
        ArrayLit, Callee, Decl, EsVersion, Expr, ExprOrSpread, FnDecl, Lit, MemberProp, ModuleDecl,
        ModuleItem, ObjectLit, Pat, Prop, PropName, PropOrSpread, Stmt, UnaryOp,
    };
    use swc_ecma_parser::error::Error as SwcError;
    use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};

    pub(super) fn parse_document(source: &str) -> Result<SourceDocument, CompileError> {
        parse_document_diagnostic(source).map_err(SourceDiagnostic::into_compile_error)
    }

    pub(super) fn parse_document_diagnostic(
        source: &str,
    ) -> Result<SourceDocument, SourceDiagnostic> {
        let mut source = TypeScriptSource::new(source);
        let module = parse_module(&mut source)?;
        let mut document = SourceDocument::new();
        for item in &module.body {
            if let Some(graph) = graph_from_item(item, &source)? {
                document.push(graph);
            }
        }
        Ok(document)
    }

    fn parse_module(
        source: &mut TypeScriptSource<'_>,
    ) -> Result<swc_ecma_ast::Module, SourceDiagnostic> {
        let cm: Lrc<SourceMap> = Default::default();
        let fm = cm.new_source_file(
            FileName::Custom("embedded.ts".to_string()).into(),
            source.text.to_string(),
        );
        source.base = fm.start_pos;
        let lexer = Lexer::new(
            Syntax::Typescript(Default::default()),
            EsVersion::Es2022,
            StringInput::from(&*fm),
            None,
        );
        let mut parser = Parser::new_from(lexer);
        let module = parser
            .parse_typescript_module()
            .map_err(|err| source.syntax_error(&err))?;
        if let Some(err) = parser.take_errors().into_iter().next() {
            return Err(source.syntax_error(&err));
        }
        Ok(module)
    }

    fn graph_from_item(
        item: &ModuleItem,
        source: &TypeScriptSource<'_>,
    ) -> Result<Option<SourceGraph>, SourceDiagnostic> {
        let Some(function) = item_fn_decl(item) else {
            return Ok(None);
        };
        graph_from_function(function, source)
    }

    fn graph_from_function(
        function: &FnDecl,
        source: &TypeScriptSource<'_>,
    ) -> Result<Option<SourceGraph>, SourceDiagnostic> {
        let Some(builder) = builder_arg(function) else {
            return Ok(None);
        };
        let Some(body) = function.function.body.as_ref() else {
            return Ok(None);
        };
        if !body_uses_builder(&body.stmts, builder) {
            return Ok(None);
        }
        let program = graph_program(&body.stmts, builder, source)?;
        Ok(Some(SourceGraph::named(
            function.ident.sym.to_string(),
            program,
        )))
    }

    fn item_fn_decl(item: &ModuleItem) -> Option<&FnDecl> {
        match item {
            ModuleItem::Stmt(Stmt::Decl(Decl::Fn(function))) => Some(function),
            ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(export)) => match &export.decl {
                Decl::Fn(function) => Some(function),
                _ => None,
            },
            _ => None,
        }
    }

    fn graph_program(
        body: &[Stmt],
        builder: &str,
        source: &TypeScriptSource<'_>,
    ) -> Result<SourceProgram, SourceDiagnostic> {
        let mut graph = TypeScriptGraph::new(builder, source);
        for stmt in body {
            graph.push_stmt(stmt)?;
        }
        Ok(graph.program)
    }

    struct TypeScriptSource<'a> {
        text: &'a str,
        line_starts: Vec<usize>,
        base: BytePos,
    }

    impl<'a> TypeScriptSource<'a> {
        fn new(text: &'a str) -> Self {
            Self {
                text,
                line_starts: line_starts(text),
                base: BytePos(0),
            }
        }

        fn syntax_error(&self, err: &SwcError) -> SourceDiagnostic {
            self.diagnostic(err, "typescript: bad syntax")
        }

        fn diagnostic<T: Spanned>(&self, node: &T, kind: &'static str) -> SourceDiagnostic {
            let span = node.span();
            let (line, column) = self.position(span.lo);
            SourceDiagnostic::new(line, column, kind, self.rejected(span))
        }

        fn position(&self, pos: BytePos) -> (usize, usize) {
            let offset = self.offset(pos);
            let index = match self.line_starts.binary_search(&offset) {
                Ok(index) => index,
                Err(0) => 0,
                Err(index) => index - 1,
            };
            (index + 1, offset - self.line_starts[index] + 1)
        }

        fn rejected(&self, span: Span) -> String {
            let start = self.offset(span.lo);
            let end = self.offset(span.hi);
            self.fragment(start, end)
        }

        fn slice(&self, span: Span) -> &'a str {
            let start = self.offset(span.lo);
            let end = self.offset(span.hi);
            self.text.get(start..end).unwrap_or("").trim()
        }

        fn offset(&self, pos: BytePos) -> usize {
            pos.0
                .saturating_sub(self.base.0)
                .try_into()
                .unwrap_or(usize::MAX)
                .min(self.text.len())
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

    fn builder_arg(function: &FnDecl) -> Option<&str> {
        let param = function.function.params.first()?;
        match &param.pat {
            Pat::Ident(ident) => Some(ident.id.sym.as_ref()),
            _ => None,
        }
    }

    fn body_uses_builder(body: &[Stmt], builder: &str) -> bool {
        body.iter().any(|stmt| stmt_uses_builder(stmt, builder))
    }

    fn stmt_uses_builder(stmt: &Stmt, builder: &str) -> bool {
        match stmt {
            Stmt::Decl(Decl::Var(var)) => var_decl_uses_builder(var, builder),
            Stmt::Expr(expr) => expr_uses_builder(&expr.expr, builder),
            _ => false,
        }
    }

    fn var_decl_uses_builder(var: &swc_ecma_ast::VarDecl, builder: &str) -> bool {
        var.decls.iter().any(|decl| {
            decl.init
                .as_ref()
                .is_some_and(|expr| expr_uses_builder(expr, builder))
        })
    }

    fn expr_uses_builder(expr: &Expr, builder: &str) -> bool {
        match expr {
            Expr::Call(call) => call_path_from_callee(&call.callee)
                .is_some_and(|path| path.first().is_some_and(|root| root == builder)),
            _ => false,
        }
    }

    struct TypeScriptGraph<'a, 's> {
        builder: &'a str,
        source: &'a TypeScriptSource<'s>,
        program: SourceProgram,
    }

    impl<'a, 's> TypeScriptGraph<'a, 's> {
        fn new(builder: &'a str, source: &'a TypeScriptSource<'s>) -> Self {
            Self {
                builder,
                source,
                program: SourceProgram::new(),
            }
        }

        fn push_stmt(&mut self, stmt: &Stmt) -> Result<(), SourceDiagnostic> {
            match stmt {
                Stmt::Decl(Decl::Var(var)) => self.push_var_decl(var),
                Stmt::Expr(expr) if is_directive(&expr.expr) => Ok(()),
                Stmt::Expr(expr) => self.push_expr(&expr.expr),
                Stmt::Empty(_) => Ok(()),
                _ => Err(self
                    .source
                    .diagnostic(stmt, "typescript: unsupported graph statement")),
            }
        }

        fn push_var_decl(&mut self, var: &swc_ecma_ast::VarDecl) -> Result<(), SourceDiagnostic> {
            for decl in &var.decls {
                self.push_var_declarator(decl)?;
            }
            Ok(())
        }

        fn push_var_declarator(
            &mut self,
            decl: &swc_ecma_ast::VarDeclarator,
        ) -> Result<(), SourceDiagnostic> {
            let name = assigned_name(decl, self.source)?;
            let init = decl
                .init
                .as_ref()
                .ok_or_else(|| self.source.diagnostic(decl, "typescript: bad assignment"))?;
            let call = call_expr(init, self.source)?;
            match builder_call(call, self.builder, self.source)? {
                BuilderCall::Input => self.push_input(name, call),
                BuilderCall::Const => self.push_const(name, call),
                BuilderCall::Op(op) => self.push_op(name, &op, call),
                BuilderCall::Output => Err(self.source.diagnostic(call, "typescript: bad output")),
            }
        }

        fn push_expr(&mut self, expr: &Expr) -> Result<(), SourceDiagnostic> {
            let call = call_expr(expr, self.source)?;
            match builder_call(call, self.builder, self.source)? {
                BuilderCall::Output => self.push_output(call),
                _ => Err(self
                    .source
                    .diagnostic(expr, "typescript: unsupported graph expression")),
            }
        }

        fn push_input(
            &mut self,
            target: &str,
            call: &swc_ecma_ast::CallExpr,
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
            call: &swc_ecma_ast::CallExpr,
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
            call: &swc_ecma_ast::CallExpr,
        ) -> Result<(), SourceDiagnostic> {
            let kind = op_table::parse(op)
                .ok_or_else(|| self.source.diagnostic(call, "typescript: unknown op kind"))?;
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

        fn push_output(&mut self, call: &swc_ecma_ast::CallExpr) -> Result<(), SourceDiagnostic> {
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
        call: &swc_ecma_ast::CallExpr,
        builder: &str,
        source: &TypeScriptSource<'_>,
    ) -> Result<BuilderCall, SourceDiagnostic> {
        let path = call_path_from_callee(&call.callee)
            .ok_or_else(|| source.diagnostic(call, "typescript: bad call"))?;
        match path.as_slice() {
            [root, kind] if root == builder && kind == "input" => Ok(BuilderCall::Input),
            [root, kind] if root == builder && matches!(kind.as_str(), "const" | "constant") => {
                Ok(BuilderCall::Const)
            }
            [root, ops, op] if root == builder && ops == "ops" => Ok(BuilderCall::Op(op.clone())),
            [root, kind] if root == builder && kind == "output" => Ok(BuilderCall::Output),
            _ => Err(source.diagnostic(call, "typescript: unsupported graph call")),
        }
    }

    fn call_path_from_callee(callee: &Callee) -> Option<Vec<String>> {
        match callee {
            Callee::Expr(expr) => call_path(expr),
            _ => None,
        }
    }

    fn call_path(expr: &Expr) -> Option<Vec<String>> {
        let mut path = Vec::new();
        collect_path(expr, &mut path).then_some(path)
    }

    fn collect_path(expr: &Expr, path: &mut Vec<String>) -> bool {
        match expr {
            Expr::Ident(ident) => {
                path.push(ident.sym.to_string());
                true
            }
            Expr::Member(member) => collect_member_path(member, path),
            _ => false,
        }
    }

    fn collect_member_path(member: &swc_ecma_ast::MemberExpr, path: &mut Vec<String>) -> bool {
        collect_path(&member.obj, path) && {
            match &member.prop {
                MemberProp::Ident(ident) => path.push(ident.sym.to_string()),
                _ => return false,
            }
            true
        }
    }

    fn assigned_name<'a>(
        decl: &'a swc_ecma_ast::VarDeclarator,
        source: &TypeScriptSource<'_>,
    ) -> Result<&'a str, SourceDiagnostic> {
        match &decl.name {
            Pat::Ident(ident) => Ok(ident.id.sym.as_ref()),
            _ => Err(source.diagnostic(&decl.name, "typescript: bad assignment")),
        }
    }

    fn call_expr<'a>(
        expr: &'a Expr,
        source: &TypeScriptSource<'_>,
    ) -> Result<&'a swc_ecma_ast::CallExpr, SourceDiagnostic> {
        match expr {
            Expr::Call(call) => Ok(call),
            _ => Err(source.diagnostic(expr, "typescript: expected call")),
        }
    }

    fn source_name(
        call: &swc_ecma_ast::CallExpr,
        fallback: &str,
        source: &TypeScriptSource<'_>,
    ) -> Result<String, SourceDiagnostic> {
        if let Some(first) = positional_args(call).first() {
            let expr = call_arg_expr(first, source)?;
            return string_literal(expr)
                .ok_or_else(|| source.diagnostic(expr, "typescript: bad name"));
        }
        match option_value(call, "name", source)? {
            Some(expr) => {
                string_literal(expr).ok_or_else(|| source.diagnostic(expr, "typescript: bad name"))
            }
            None => Ok(fallback.to_string()),
        }
    }

    fn source_type(
        call: &swc_ecma_ast::CallExpr,
        source: &TypeScriptSource<'_>,
    ) -> Result<SourceType, SourceDiagnostic> {
        validate_dtype(call, source)?;
        optional_shape(call, source).map(SourceType::f32)
    }

    fn optional_source_type(
        call: &swc_ecma_ast::CallExpr,
        source: &TypeScriptSource<'_>,
    ) -> Result<Option<SourceType>, SourceDiagnostic> {
        validate_dtype(call, source)?;
        optional_shape(call, source).map(|shape| shape.map(|shape| SourceType::f32(Some(shape))))
    }

    fn attrs_from_options(
        op: OpKind,
        call: &swc_ecma_ast::CallExpr,
        source: &TypeScriptSource<'_>,
    ) -> Result<SourceAttrs, SourceDiagnostic> {
        let mut attrs = SourceAttrs::default();
        let Some(object) = options_object(call) else {
            return Ok(attrs);
        };
        for prop in &object.props {
            apply_option_attr(op, &mut attrs, prop, source)?;
        }
        Ok(attrs)
    }

    fn apply_option_attr(
        op: OpKind,
        attrs: &mut SourceAttrs,
        prop: &PropOrSpread,
        source: &TypeScriptSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        let kv = key_value_prop(prop, source)?;
        let name = prop_name(&kv.key, source)?;
        if matches!(name, "dtype" | "shape") {
            return Ok(());
        }
        let attr = ParsedAttr {
            name,
            value: attr_value(&kv.value, source)?,
        };
        apply_attr(op, attrs, attr)
            .map_err(|err| source.diagnostic(kv, diagnostic::compile_error_kind(&err)))
    }

    fn attr_value<'a>(
        expr: &Expr,
        source: &TypeScriptSource<'a>,
    ) -> Result<AttrValue<'a>, SourceDiagnostic> {
        match expr {
            Expr::Lit(Lit::Bool(value)) => Ok(AttrValue::Bool(value.value)),
            Expr::Lit(Lit::Num(_)) => Ok(AttrValue::Number(source.slice(expr.span()))),
            Expr::Unary(unary) if is_numeric_unary(unary.op) => {
                Ok(AttrValue::Number(source.slice(expr.span())))
            }
            Expr::Array(array) => attr_list(array, source),
            _ => Err(source.diagnostic(expr, "typescript: bad attr value")),
        }
    }

    fn attr_list<'a>(
        array: &ArrayLit,
        source: &TypeScriptSource<'a>,
    ) -> Result<AttrValue<'a>, SourceDiagnostic> {
        array
            .elems
            .iter()
            .map(|elem| elem_attr_number(elem, source))
            .collect::<Result<Vec<_>, _>>()
            .map(AttrValue::List)
    }

    fn elem_attr_number<'a>(
        elem: &Option<ExprOrSpread>,
        source: &TypeScriptSource<'a>,
    ) -> Result<&'a str, SourceDiagnostic> {
        let expr = array_elem_expr(elem, source)?;
        match expr {
            Expr::Lit(Lit::Num(_)) => Ok(source.slice(expr.span())),
            Expr::Unary(unary) if is_numeric_unary(unary.op) => Ok(source.slice(expr.span())),
            _ => Err(source.diagnostic(expr, "typescript: bad attr value")),
        }
    }

    fn required_shape_type(
        call: &swc_ecma_ast::CallExpr,
        source: &TypeScriptSource<'_>,
    ) -> Result<SourceType, SourceDiagnostic> {
        validate_dtype(call, source)?;
        let shape = required_shape(call, source)?;
        Ok(SourceType::f32(Some(shape)))
    }

    fn validate_dtype(
        call: &swc_ecma_ast::CallExpr,
        source: &TypeScriptSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        match option_value(call, "dtype", source)? {
            Some(expr) if string_literal(expr).as_deref() == Some("f32") => Ok(()),
            Some(expr) => Err(source.diagnostic(expr, "typescript: unsupported dtype")),
            None => Ok(()),
        }
    }

    fn optional_shape(
        call: &swc_ecma_ast::CallExpr,
        source: &TypeScriptSource<'_>,
    ) -> Result<Option<ShapeDescriptor>, SourceDiagnostic> {
        option_value(call, "shape", source)?
            .map(|expr| shape_literal(expr, source))
            .transpose()
    }

    fn required_shape(
        call: &swc_ecma_ast::CallExpr,
        source: &TypeScriptSource<'_>,
    ) -> Result<ShapeDescriptor, SourceDiagnostic> {
        required_option(call, "shape", source).and_then(|expr| shape_literal(expr, source))
    }

    fn shape_literal(
        expr: &Expr,
        source: &TypeScriptSource<'_>,
    ) -> Result<ShapeDescriptor, SourceDiagnostic> {
        let dims = integer_list(expr, source)?;
        if dims.is_empty() || dims.len() > 8 {
            return Err(source.diagnostic(expr, "typescript: bad shape"));
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
        source: &TypeScriptSource<'_>,
    ) -> Result<SourceTensorLiteral, SourceDiagnostic> {
        let values = numeric_list(expr, source)?;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(SourceTensorLiteral::new(bytes, values.len()))
    }

    fn source_inputs(
        call: &swc_ecma_ast::CallExpr,
        program: &mut SourceProgram,
        source: &TypeScriptSource<'_>,
    ) -> Result<Vec<SourceSymbol>, SourceDiagnostic> {
        positional_args(call)
            .iter()
            .map(|arg| symbol_ref(arg, source))
            .map(|name| name.map(|name| program.intern(&name)))
            .collect()
    }

    fn symbol_ref(
        arg: &ExprOrSpread,
        source: &TypeScriptSource<'_>,
    ) -> Result<String, SourceDiagnostic> {
        match call_arg_expr(arg, source)? {
            Expr::Ident(name) => Ok(name.sym.to_string()),
            expr => Err(source.diagnostic(expr, "typescript: bad input ref")),
        }
    }

    fn output_name(
        call: &swc_ecma_ast::CallExpr,
        source: &TypeScriptSource<'_>,
    ) -> Result<String, SourceDiagnostic> {
        match call.args.as_slice() {
            [arg] => output_symbol_arg(arg, source),
            [alias, arg] if string_arg(alias, source)?.is_some() => output_symbol_arg(arg, source),
            [bad, ..] => Err(source.diagnostic(bad, "typescript: bad output")),
            [] => Err(source.diagnostic(call, "typescript: bad output")),
        }
    }

    fn output_symbol_arg(
        arg: &ExprOrSpread,
        source: &TypeScriptSource<'_>,
    ) -> Result<String, SourceDiagnostic> {
        match call_arg_expr(arg, source)? {
            Expr::Ident(name) => Ok(name.sym.to_string()),
            expr => Err(source.diagnostic(expr, "typescript: bad output")),
        }
    }

    fn required_option<'a>(
        call: &'a swc_ecma_ast::CallExpr,
        name: &str,
        source: &TypeScriptSource<'_>,
    ) -> Result<&'a Expr, SourceDiagnostic> {
        option_value(call, name, source)?
            .ok_or_else(|| source.diagnostic(call, "typescript: missing option"))
    }

    fn option_value<'a>(
        call: &'a swc_ecma_ast::CallExpr,
        name: &str,
        source: &TypeScriptSource<'_>,
    ) -> Result<Option<&'a Expr>, SourceDiagnostic> {
        let Some(object) = options_object(call) else {
            return Ok(None);
        };
        object_option(object, name, source)
    }

    fn object_option<'a>(
        object: &'a ObjectLit,
        name: &str,
        source: &TypeScriptSource<'_>,
    ) -> Result<Option<&'a Expr>, SourceDiagnostic> {
        for prop in &object.props {
            let kv = key_value_prop(prop, source)?;
            if prop_name(&kv.key, source)? == name {
                return Ok(Some(&kv.value));
            }
        }
        Ok(None)
    }

    fn ensure_options(
        call: &swc_ecma_ast::CallExpr,
        allowed: &[&str],
        source: &TypeScriptSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        let Some(object) = options_object(call) else {
            return Ok(());
        };
        for prop in &object.props {
            ensure_option_prop(prop, allowed, source)?;
        }
        Ok(())
    }

    fn ensure_option_prop(
        prop: &PropOrSpread,
        allowed: &[&str],
        source: &TypeScriptSource<'_>,
    ) -> Result<(), SourceDiagnostic> {
        let kv = key_value_prop(prop, source)?;
        let name = prop_name(&kv.key, source)?;
        if allowed.contains(&name) {
            Ok(())
        } else {
            Err(source.diagnostic(kv, "typescript: unsupported option"))
        }
    }

    fn key_value_prop<'a>(
        prop: &'a PropOrSpread,
        source: &TypeScriptSource<'_>,
    ) -> Result<&'a swc_ecma_ast::KeyValueProp, SourceDiagnostic> {
        match prop {
            PropOrSpread::Prop(prop) => match &**prop {
                Prop::KeyValue(kv) => Ok(kv),
                _ => Err(source.diagnostic(prop.as_ref(), "typescript: unsupported option")),
            },
            PropOrSpread::Spread(spread) => {
                Err(source.diagnostic(spread, "typescript: unsupported option"))
            }
        }
    }

    fn prop_name<'a>(
        name: &'a PropName,
        source: &TypeScriptSource<'_>,
    ) -> Result<&'a str, SourceDiagnostic> {
        match name {
            PropName::Ident(ident) => Ok(ident.sym.as_ref()),
            PropName::Str(value) => value
                .value
                .as_str()
                .ok_or_else(|| source.diagnostic(name, "typescript: bad option")),
            _ => Err(source.diagnostic(name, "typescript: bad option")),
        }
    }

    fn options_object(call: &swc_ecma_ast::CallExpr) -> Option<&ObjectLit> {
        match call.args.last().map(|arg| &*arg.expr) {
            Some(Expr::Object(object)) => Some(object),
            _ => None,
        }
    }

    fn positional_args(call: &swc_ecma_ast::CallExpr) -> &[ExprOrSpread] {
        match options_object(call) {
            Some(_) => &call.args[..call.args.len().saturating_sub(1)],
            None => &call.args,
        }
    }

    fn call_arg_expr<'a>(
        arg: &'a ExprOrSpread,
        source: &TypeScriptSource<'_>,
    ) -> Result<&'a Expr, SourceDiagnostic> {
        match arg.spread {
            Some(_) => Err(source.diagnostic(arg, "typescript: spread unsupported")),
            None => Ok(&arg.expr),
        }
    }

    fn string_arg(
        arg: &ExprOrSpread,
        source: &TypeScriptSource<'_>,
    ) -> Result<Option<String>, SourceDiagnostic> {
        call_arg_expr(arg, source).map(string_literal)
    }

    fn string_literal(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Lit(Lit::Str(value)) => value.value.as_str().map(str::to_string),
            _ => None,
        }
    }

    fn integer_list(
        expr: &Expr,
        source: &TypeScriptSource<'_>,
    ) -> Result<Vec<u64>, SourceDiagnostic> {
        expr_array(expr, source)?
            .elems
            .iter()
            .map(|elem| integer_value(array_elem_expr(elem, source)?, source))
            .collect()
    }

    fn numeric_list(
        expr: &Expr,
        source: &TypeScriptSource<'_>,
    ) -> Result<Vec<f32>, SourceDiagnostic> {
        expr_array(expr, source)?
            .elems
            .iter()
            .map(|elem| numeric_value(array_elem_expr(elem, source)?, source))
            .collect()
    }

    fn expr_array<'a>(
        expr: &'a Expr,
        source: &TypeScriptSource<'_>,
    ) -> Result<&'a ArrayLit, SourceDiagnostic> {
        match expr {
            Expr::Array(array) => Ok(array),
            _ => Err(source.diagnostic(expr, "typescript: expected array")),
        }
    }

    fn array_elem_expr<'a>(
        elem: &'a Option<ExprOrSpread>,
        source: &TypeScriptSource<'_>,
    ) -> Result<&'a Expr, SourceDiagnostic> {
        match elem {
            Some(arg) => call_arg_expr(arg, source),
            None => Err(SourceDiagnostic::global(
                "typescript: array holes unsupported",
            )),
        }
    }

    fn integer_value(expr: &Expr, source: &TypeScriptSource<'_>) -> Result<u64, SourceDiagnostic> {
        match expr {
            Expr::Lit(Lit::Num(value)) if is_u64_literal(value.value) => Ok(value.value as u64),
            _ => Err(source.diagnostic(expr, "typescript: bad integer")),
        }
    }

    fn numeric_value(expr: &Expr, source: &TypeScriptSource<'_>) -> Result<f32, SourceDiagnostic> {
        match expr {
            Expr::Lit(Lit::Num(value)) => Ok(value.value as f32),
            Expr::Unary(unary) if unary.op == UnaryOp::Minus => {
                numeric_value(&unary.arg, source).map(|value| -value)
            }
            Expr::Unary(unary) if unary.op == UnaryOp::Plus => numeric_value(&unary.arg, source),
            _ => Err(source.diagnostic(expr, "typescript: bad numeric value")),
        }
    }

    fn is_u64_literal(value: f64) -> bool {
        value.is_finite() && value.fract() == 0.0 && (0.0..=u64::MAX as f64).contains(&value)
    }

    fn is_numeric_unary(op: UnaryOp) -> bool {
        matches!(op, UnaryOp::Minus | UnaryOp::Plus)
    }

    fn is_directive(expr: &Expr) -> bool {
        matches!(expr, Expr::Lit(Lit::Str(_)))
    }
}

#[cfg(feature = "frontend-typescript")]
use enabled::{parse_document, parse_document_diagnostic};
